/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the MIT license in thirdparty/faiss/LICENSE. */
#pragma once

#include <faiss/VectorTransform.h>
#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>
#include <faiss/cppcontrib/knowhere/impl/StagedDistanceComputer.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/distances.h>
#include <cmath>
#include <memory>

namespace faiss::cppcontrib::knowhere::rabitq_search {
// Both traversal implementations compare in smaller-is-better units.
// This probability window is not a deterministic lower bound.
inline bool should_refine(float estimate, float f_error, float g_error,
                          float threshold, bool similarity, float scale) {
    const float error = f_error * g_error;
    return similarity ? (estimate + error) * scale > -threshold
                      : std::max(0.0f, estimate - error) < threshold;
}

// Concrete RBQ adapter exposed only to the RBQ evaluator so its per-candidate
// threshold logic can be inlined, without specializing the common searcher.
struct RaBitQStagedDistanceComputer final : StagedDistanceComputer {
    const faiss::VectorTransform& rotation;
    std::unique_ptr<faiss::RaBitQDistanceComputer> dc;
    const float* norms;
    bool similarity;
    float query_inverse_norm = 1;
    std::vector<float> rotated;

    explicit RaBitQStagedDistanceComputer(const IndexHNSWRaBitQ& index,
                                        const faiss::RaBitQSearchParameters* params)
        : rotation(*index.pretransform_index()->chain[0]),
          norms(nullptr), similarity(index.metric_type == METRIC_INNER_PRODUCT),
          rotated(index.d) {
        auto* raw = params ? index.rabitq_index()->get_quantized_distance_computer(params->qb, params->centered)
                           : index.rabitq_index()->get_FlatCodesDistanceComputer();
        auto* typed = dynamic_cast<faiss::RaBitQDistanceComputer*>(raw);
        if (!typed) { delete raw; FAISS_THROW_MSG("RaBitQ distance computer required"); }
        dc.reset(typed);
        if (auto* cosine = dynamic_cast<const IndexHNSWRaBitQCosine*>(&index)) {
            norms = cosine->get_inverse_l2_norms();
        }
    }
    void set_query(const float* q) override {
        estimate_count = refine_count = 0;
        rotation.apply_noalloc(1, q, rotated.data());
        dc->set_query(rotated.data());
        const float norm2 = norms ? faiss::fvec_norm_L2sqr(q, rotation.d_in) : 1;
        query_inverse_norm = norm2 > 0 ? 1 / std::sqrt(norm2) : 1;
    }
    float scale(idx_t id) const { return norms ? norms[id] * query_inverse_norm : 1; }
    float operator()(idx_t id) override {
        float d = (*dc)(id) * scale(id);
        return similarity ? -d : d;
    }
    float symmetric_dis(idx_t, idx_t) override {
        FAISS_THROW_MSG("staged storage is search-only; construct graph with FP32");
    }
    float evaluate(idx_t id, float threshold) override {
        if (dc->nb_bits == 1) return (*this)(id);
        const auto* code = dc->codes + id * dc->code_size;
        const float estimate = dc->distance_to_code_1bit(code);
        ++estimate_count;
        return evaluate_estimate(id, code, estimate, threshold);
    }
    float evaluate_estimate(idx_t id, const uint8_t* code, float estimate, float threshold) {
        const auto* factors = reinterpret_cast<const rabitq_utils::SignBitFactorsWithError*>(
                code + (dc->d + 7) / 8);
        const float s = scale(id);
        float d = estimate;
        // Compare in output-distance units: positive cosine scale preserves
        // ordering, avoiding a division for every visited candidate.
        const bool refine = rabitq_search::should_refine(
                estimate, factors->f_error, dc->g_error, threshold, similarity, s);
        if (refine) {
            d = dc->distance_to_code_full(code);
            ++refine_count;
        }
        return (similarity ? -d : d) * s;
    }
};
} // namespace faiss::cppcontrib::knowhere::rabitq_search
