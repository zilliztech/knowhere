/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>
#include <faiss/cppcontrib/knowhere/impl/RaBitQDistanceEvaluation.h>

#include <faiss/VectorTransform.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/distances.h>
#include <faiss/cppcontrib/knowhere/impl/StagedDistanceComputer.h>

#include <cmath>
#include <memory>

namespace faiss::cppcontrib::knowhere {

namespace {
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
} // namespace

faiss::DistanceComputer* IndexHNSWRaBitQ::get_staged_distance_computer(
        const faiss::RaBitQSearchParameters* params) const {
    return new RaBitQStagedDistanceComputer(*this, params);
}

IndexPreTransformRaBitQCosine::IndexPreTransformRaBitQCosine() = default;

IndexPreTransformRaBitQCosine::IndexPreTransformRaBitQCosine(
        faiss::VectorTransform* transform,
        faiss::Index* index_in)
        : faiss::IndexPreTransform(transform, index_in) {}

void IndexPreTransformRaBitQCosine::add(idx_t n, const float* x) {
    faiss::IndexPreTransform::add(n, x);
    inverse_norms_storage.add(x, n, d);
}

void IndexPreTransformRaBitQCosine::reset() {
    faiss::IndexPreTransform::reset();
    inverse_norms_storage.reset();
}

faiss::DistanceComputer* IndexPreTransformRaBitQCosine::get_distance_computer()
        const {
    FAISS_THROW_IF_NOT_MSG(
            inverse_norms_storage.inverse_l2_norms.size() ==
                    static_cast<size_t>(ntotal),
            "cosine RaBitQ inverse norm count must match ntotal");
    return new WithCosineNormDistanceComputer(
            get_inverse_l2_norms(),
            d,
            std::unique_ptr<faiss::DistanceComputer>(
                    faiss::IndexPreTransform::get_distance_computer()));
}

const float* IndexPreTransformRaBitQCosine::get_inverse_l2_norms() const {
    return inverse_norms_storage.inverse_l2_norms.data();
}

void IndexPreTransformRaBitQCosine::validate_norms() const {
    FAISS_THROW_IF_NOT_MSG(
            inverse_norms_storage.inverse_l2_norms.size() ==
                    static_cast<size_t>(ntotal),
            "cosine RaBitQ inverse norm count must match ntotal");
    for (const float inverse_norm : inverse_norms_storage.inverse_l2_norms) {
        FAISS_THROW_IF_NOT_MSG(
                std::isfinite(inverse_norm) && inverse_norm > 0.0f,
                "cosine RaBitQ inverse norms must be finite and positive");
    }
}

IndexHNSWRaBitQ::IndexHNSWRaBitQ() = default;

void IndexHNSWRaBitQ::add(idx_t, const float*) {
    FAISS_THROW_MSG(
            "IndexHNSWRaBitQ does not support incremental add: build the "
            "HNSW graph with exact storage before attaching RaBitQ storage");
}

const faiss::IndexPreTransform* IndexHNSWRaBitQ::pretransform_index() const {
    return dynamic_cast<const faiss::IndexPreTransform*>(storage);
}

const faiss::IndexRaBitQ* IndexHNSWRaBitQ::rabitq_index() const {
    const auto* pretransform = pretransform_index();
    return pretransform
            ? dynamic_cast<const faiss::IndexRaBitQ*>(pretransform->index)
            : nullptr;
}

void IndexHNSWRaBitQ::validate_storage() const {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT,
            "IndexHNSWRaBitQ only supports L2 and inner product metrics");
    FAISS_THROW_IF_NOT_MSG(
            storage != nullptr, "IndexHNSWRaBitQ requires non-null storage");

    const auto* pretransform = pretransform_index();
    FAISS_THROW_IF_NOT_MSG(
            pretransform != nullptr,
            "IndexHNSWRaBitQ storage must be IndexPreTransform");
    FAISS_THROW_IF_NOT_MSG(
            pretransform->chain.size() == 1,
            "IndexHNSWRaBitQ storage must contain exactly one transform");

    const auto* rotation = dynamic_cast<const faiss::RandomRotationMatrix*>(
            pretransform->chain[0]);
    FAISS_THROW_IF_NOT_MSG(
            rotation != nullptr,
            "IndexHNSWRaBitQ transform must be RandomRotationMatrix");

    const auto* rabitq = rabitq_index();
    FAISS_THROW_IF_NOT_MSG(
            rabitq != nullptr,
            "IndexHNSWRaBitQ pretransform leaf must be IndexRaBitQ");

    FAISS_THROW_IF_NOT_MSG(
            d == pretransform->d && metric_type == pretransform->metric_type &&
                    ntotal == pretransform->ntotal,
            "IndexHNSWRaBitQ outer index and pretransform metadata mismatch");
    FAISS_THROW_IF_NOT_MSG(
            is_trained && pretransform->is_trained && rotation->is_trained &&
                    rabitq->is_trained,
            "IndexHNSWRaBitQ requires fully trained storage");
    FAISS_THROW_IF_NOT_MSG(
            pretransform->index != nullptr &&
                    pretransform->ntotal == rabitq->ntotal &&
                    pretransform->metric_type == rabitq->metric_type,
            "IndexHNSWRaBitQ pretransform and RaBitQ metadata mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rotation->d_in == d && rotation->d_out == rabitq->d &&
                    rotation->d_in == rotation->d_out,
            "IndexHNSWRaBitQ requires a square rotation matching index dimensions");
    FAISS_THROW_IF_NOT_MSG(
            rotation->is_orthonormal && !rotation->have_bias &&
                    rotation->b.empty() &&
                    rotation->A.size() ==
                            static_cast<size_t>(rotation->d_in) *
                                    rotation->d_out,
            "IndexHNSWRaBitQ rotation matrix has invalid storage");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->rabitq.d == static_cast<size_t>(rabitq->d) &&
                    rabitq->rabitq.metric_type == rabitq->metric_type,
            "IndexHNSWRaBitQ RaBitQ quantizer metadata mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->rabitq.nb_bits >= 1 && rabitq->rabitq.nb_bits <= 9,
            "IndexHNSWRaBitQ RaBitQ nb_bits must be in [1, 9]");

    const size_t expected_code_size =
            rabitq->rabitq.compute_code_size(rabitq->d, rabitq->rabitq.nb_bits);
    FAISS_THROW_IF_NOT_MSG(
            rabitq->rabitq.code_size == expected_code_size &&
                    rabitq->code_size == expected_code_size,
            "IndexHNSWRaBitQ RaBitQ code size mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->codes.size() ==
                    static_cast<size_t>(rabitq->ntotal) * expected_code_size,
            "IndexHNSWRaBitQ RaBitQ codes size mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->center.size() == static_cast<size_t>(rabitq->d),
            "IndexHNSWRaBitQ RaBitQ center size mismatch");
    FAISS_THROW_IF_NOT_MSG(
            rabitq->qb <= 8, "IndexHNSWRaBitQ RaBitQ qb must be in [0, 8]");
    FAISS_THROW_IF_NOT_MSG(
            !rabitq->centered, "IndexHNSWRaBitQ V1 requires centered=false");
}

IndexHNSWRaBitQCosine::IndexHNSWRaBitQCosine() = default;

const float* IndexHNSWRaBitQCosine::get_inverse_l2_norms() const {
    const auto* cosine_storage =
            dynamic_cast<const IndexPreTransformRaBitQCosine*>(storage);
    return cosine_storage ? cosine_storage->get_inverse_l2_norms() : nullptr;
}

void IndexHNSWRaBitQCosine::validate_cosine_storage() const {
    validate_storage();
    const auto* cosine_storage =
            dynamic_cast<const IndexPreTransformRaBitQCosine*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            cosine_storage != nullptr,
            "IndexHNSWRaBitQCosine requires cosine-aware pretransform storage");
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_INNER_PRODUCT,
            "IndexHNSWRaBitQCosine requires inner product storage");
    cosine_storage->validate_norms();
}

} // namespace faiss::cppcontrib::knowhere
