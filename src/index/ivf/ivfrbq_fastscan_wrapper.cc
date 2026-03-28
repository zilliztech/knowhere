// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "index/ivf/ivfrbq_fastscan_wrapper.h"

// Safe to include here: this TU does not transitively include
// knowhere's patched simd_result_handlers.h.
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/index_io.h>

#include <memory>
#include <stdexcept>

#include "faiss/impl/io.h"
#include "knowhere/expected.h"

namespace knowhere {

expected<std::unique_ptr<IndexIVFRaBitQFastScanWrapper>>
IndexIVFRaBitQFastScanWrapper::create(const faiss::idx_t d, const size_t nlist, const IvfRaBitQFastScanConfig& cfg,
                                      const faiss::MetricType metric) {
    // NOTE: `metric` is already converted by the caller (TrainInternal):
    //   COSINE → METRIC_INNER_PRODUCT (data is pre-normalized).
    // Build the core Faiss stack first:
    //   IndexFlat -> IndexIVFRaBitQFastScan -> RandomRotation pretransform
    // Optional refine is layered on top afterwards so the whole chain remains
    // serializable through core Faiss IO.
    auto idx_flat = std::make_unique<faiss::IndexFlat>(d, metric);
    auto idx_fastscan = std::make_unique<faiss::IndexIVFRaBitQFastScan>(idx_flat.release(), d, nlist, metric,
                                                                        /*bbs=*/32, /*own_invlists=*/true,
                                                                        /*nb_bits=*/1);
    idx_fastscan->own_fields = true;
    // FastScan requires qb in [1,8]. Use qb=8 (upstream default) for best accuracy.
    idx_fastscan->qb = 8;

    // Keep the same random-rotation structure as the legacy IVFRaBitQ path.
    auto rr = std::make_unique<faiss::RandomRotationMatrix>(d, d);
    auto idx_rr = std::make_unique<faiss::IndexPreTransform>(rr.release(), idx_fastscan.release());
    idx_rr->own_fields = true;

    std::unique_ptr<faiss::Index> idx_final;
    if (cfg.refine.value_or(false)) {
        // v1 only supports refine-flat so the whole stack stays in core Faiss
        // types and can be serialized with faiss::write_index/read_index.
        auto idx_refine = std::make_unique<faiss::IndexRefineFlat>(idx_rr.release());
        idx_refine->own_fields = true;
        idx_refine->own_refine_index = true;
        idx_final = std::move(idx_refine);
    } else {
        idx_final = std::move(idx_rr);
    }

    return std::make_unique<IndexIVFRaBitQFastScanWrapper>(std::move(idx_final));
}

IndexIVFRaBitQFastScanWrapper::IndexIVFRaBitQFastScanWrapper(std::unique_ptr<faiss::Index>&& index_in)
    : Index{index_in->d, index_in->metric_type}, index{std::move(index_in)} {
    ntotal = index->ntotal;
    is_trained = index->is_trained;
    verbose = index->verbose;
    metric_arg = index->metric_arg;
    detect_refine_state();
}

void
IndexIVFRaBitQFastScanWrapper::detect_refine_state() {
    has_refine_ = (dynamic_cast<faiss::IndexRefine*>(index.get()) != nullptr);
}

std::unique_ptr<IndexIVFRaBitQFastScanWrapper>
IndexIVFRaBitQFastScanWrapper::from_deserialized(std::unique_ptr<faiss::Index>&& index_in) {
    // Probe through optional refine to find IndexPreTransform -> IndexIVFRaBitQFastScan.
    faiss::IndexPreTransform* index_pt = dynamic_cast<faiss::IndexPreTransform*>(index_in.get());
    if (index_pt == nullptr) {
        auto* refine = dynamic_cast<faiss::IndexRefine*>(index_in.get());
        if (refine != nullptr) {
            index_pt = dynamic_cast<faiss::IndexPreTransform*>(refine->base_index);
        }
    }
    if (index_pt == nullptr) {
        return nullptr;
    }
    if (dynamic_cast<faiss::IndexIVFRaBitQFastScan*>(index_pt->index) == nullptr) {
        return nullptr;
    }

    return std::make_unique<IndexIVFRaBitQFastScanWrapper>(std::move(index_in));
}

void
IndexIVFRaBitQFastScanWrapper::train(faiss::idx_t n, const float* x) {
    index->train(n, x);
    is_trained = index->is_trained;
}

void
IndexIVFRaBitQFastScanWrapper::add(faiss::idx_t n, const float* x) {
    index->add(n, x);
    this->ntotal = index->ntotal;
}

void
IndexIVFRaBitQFastScanWrapper::search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances,
                                      faiss::idx_t* labels, const faiss::SearchParameters* params) const {
    index->search(n, x, k, distances, labels, params);
}

void
IndexIVFRaBitQFastScanWrapper::range_search(faiss::idx_t n, const float* x, float radius,
                                            faiss::RangeSearchResult* result,
                                            const faiss::SearchParameters* params) const {
    index->range_search(n, x, radius, result, params);
}

void
IndexIVFRaBitQFastScanWrapper::reset() {
    index->reset();
    this->ntotal = 0;
}

void
IndexIVFRaBitQFastScanWrapper::merge_from(faiss::Index& otherIndex, faiss::idx_t add_id) {
    index->merge_from(otherIndex, add_id);
}

faiss::DistanceComputer*
IndexIVFRaBitQFastScanWrapper::get_distance_computer() const {
    return index->get_distance_computer();
}

faiss::IndexIVFRaBitQFastScan*
IndexIVFRaBitQFastScanWrapper::get_fastscan_index() {
    // The public wrapper always owns either:
    //   IndexPreTransform(FastScan), or
    //   IndexRefineFlat(IndexPreTransform(FastScan)).
    // Unwrap those layers here so ivf.cc can configure/train/search against
    // the actual FastScan base index without knowing the wrapper shape.
    faiss::Index* target = index.get();
    auto* refine = dynamic_cast<faiss::IndexRefine*>(target);
    if (refine != nullptr) {
        target = refine->base_index;
    }
    auto* pt = dynamic_cast<faiss::IndexPreTransform*>(target);
    if (pt == nullptr) {
        return nullptr;
    }
    return dynamic_cast<faiss::IndexIVFRaBitQFastScan*>(pt->index);
}

const faiss::IndexIVFRaBitQFastScan*
IndexIVFRaBitQFastScanWrapper::get_fastscan_index() const {
    // Const version of the same unwrapping logic as above.
    const faiss::Index* target = index.get();
    auto* refine = dynamic_cast<const faiss::IndexRefine*>(target);
    if (refine != nullptr) {
        target = refine->base_index;
    }
    auto* pt = dynamic_cast<const faiss::IndexPreTransform*>(target);
    if (pt == nullptr) {
        return nullptr;
    }
    return dynamic_cast<const faiss::IndexIVFRaBitQFastScan*>(pt->index);
}

size_t
IndexIVFRaBitQFastScanWrapper::get_nlist() const {
    const auto* fs = get_fastscan_index();
    return (fs != nullptr) ? fs->nlist : 0;
}

namespace {
struct FaissSizeCounter : faiss::IOWriter {
    size_t total_size = 0;
    size_t
    operator()(const void*, size_t size, size_t nitems) override {
        total_size += size * nitems;
        return nitems;
    }
};
}  // namespace

size_t
IndexIVFRaBitQFastScanWrapper::size() const {
    if (index == nullptr) {
        return 0;
    }
    // Count the serialized size on demand. This is not a hot path and avoids
    // keeping cache invalidation state in the wrapper.
    FaissSizeCounter counter;
    faiss::write_index(index.get(), &counter);
    return counter.total_size;
}

void
IndexIVFRaBitQFastScanWrapper::validate_search(const faiss::IDSelector* sel) {
    if (sel != nullptr) {
        throw std::runtime_error("IVF_RABITQ_FASTSCAN does not support bitset filtering");
    }
}

}  // namespace knowhere
