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

#include "index/ivf/ivfrbq_wrapper.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "faiss/IndexCosine.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/cppcontrib/knowhere/impl/CountSizeIOWriter.h"
#include "faiss/index_io.h"
#include "index/refine/refine_utils.h"

namespace knowhere {

// Cached distance computer for HNSW + IVF_RABITQ optimization
// Pre-computes all centroids and caches FlatCodesDistanceComputer objects
// This wraps the IndexIVFRaBitQ distance computation with PreTransform support
struct CachedHNSWRaBitDistanceComputer : faiss::DistanceComputer {
    const float* q = nullptr;
    const faiss::IndexIVFRaBitQ* parent = nullptr;
    const faiss::IndexPreTransform* pretransform = nullptr;  // For applying rotation

    // Cached data
    std::vector<std::vector<float>> centroids;  // Pre-computed centroids for all lists
    std::vector<std::unique_ptr<faiss::FlatCodesDistanceComputer>> distance_computers;  // One per list
    int cached_qb = -1;                                // Track the qb value used for cached distance computers
    std::unique_ptr<const float[]> transformed_query;  // Store transformed query

    // Lazy initialization: track which distance computers have been set for current query
    mutable std::vector<bool> query_set_flags;  // Flag for each list
    const float* current_query_ptr = nullptr;   // Track current query pointer to detect query changes

    CachedHNSWRaBitDistanceComputer(const faiss::IndexIVFRaBitQ* parent_index, const faiss::IndexPreTransform* pt)
        : parent(parent_index), pretransform(pt) {
        // Pre-compute all centroids (these don't change)
        const size_t nlist = parent->nlist;
        centroids.resize(nlist);
        distance_computers.resize(nlist);
        query_set_flags.resize(nlist, false);  // Initialize lazy flags

        for (size_t list_no = 0; list_no < nlist; ++list_no) {
            // Reconstruct centroid
            centroids[list_no].resize(parent->d);
            parent->quantizer->reconstruct(list_no, centroids[list_no].data());
        }
    }

    void
    set_query(const float* x) override {
        // Apply pretransform (rotation) if available
        if (pretransform != nullptr && !pretransform->chain.empty()) {
            const float* xt = pretransform->apply_chain(1, x);
            if (xt != x) {
                transformed_query.reset(xt);
                q = transformed_query.get();
            } else {
                q = x;
            }
        } else {
            q = x;
        }

        // Check if we need to recreate distance computers due to qb change
        if (cached_qb != parent->qb) {
            // Recreate all distance computers with the new qb value
            for (size_t list_no = 0; list_no < centroids.size(); ++list_no) {
                distance_computers[list_no].reset(
                    parent->rabitq.get_distance_computer(parent->qb, centroids[list_no].data()));
            }
            cached_qb = parent->qb;
            // When qb changes, we need to reset all query flags
            std::fill(query_set_flags.begin(), query_set_flags.end(), false);
        }

        // Check if this is a new query
        if (current_query_ptr != x) {
            current_query_ptr = x;
            // Reset flags to indicate all distance computers need query update
            std::fill(query_set_flags.begin(), query_set_flags.end(), false);
        }
    }

    float
    operator()(faiss::idx_t i) override {
        // Find the appropriate list
        faiss::idx_t lo = parent->direct_map.get(i);
        uint64_t list_no = faiss::lo_listno(lo);
        uint64_t offset = faiss::lo_offset(lo);

        // LAZY INITIALIZATION: Only set query for this specific list if not already set
        if (!query_set_flags[list_no] && distance_computers[list_no]) {
            distance_computers[list_no]->set_query(q);
            query_set_flags[list_no] = true;
        }

        const uint8_t* code = parent->invlists->get_single_code(list_no, offset);

        // Use cached distance computer for this list
        float distance = distance_computers[list_no]->distance_to_code(code);

        // Release code
        parent->invlists->release_codes(list_no, code);

        return distance;
    }

    float
    symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        FAISS_THROW_MSG("Not implemented");
    }
};

expected<std::unique_ptr<IndexIVFRaBitQWrapper>>
IndexIVFRaBitQWrapper::create(const faiss::idx_t d, const size_t nlist, const IvfRaBitQConfig& ivf_rabitq_cfg,
                              const DataFormatEnum raw_data_format, const faiss::MetricType metric) {
    // the index factory string is either `RR(dim),IVFx,RaBitQ,Refine(y)`,
    //   or `RR(dim),IVFx,RaBitQ`, depends on the refine parameters

    // create IndexIVFRaBitQ
    auto qb = ivf_rabitq_cfg.rbq_bits_query.value();

    auto idx_flat = std::make_unique<faiss::IndexFlat>(d, metric, false);
    auto idx_ivfrbq = std::make_unique<faiss::IndexIVFRaBitQ>(idx_flat.release(), d, nlist, metric);
    idx_ivfrbq->own_fields = true;
    idx_ivfrbq->qb = qb;

    // wrap it in an IndexPreTransform
    auto rr = std::make_unique<faiss::RandomRotationMatrix>(d, d);
    auto idx_rr = std::make_unique<faiss::IndexPreTransform>(rr.release(), idx_ivfrbq.release());
    idx_rr->own_fields = true;

    // create a refiner index, if needed
    std::unique_ptr<faiss::Index> idx_final;
    if (ivf_rabitq_cfg.refine.value_or(false) && ivf_rabitq_cfg.refine_type.has_value()) {
        // refine is needed
        const auto base_d = idx_rr->d;
        const auto base_metric_type = idx_rr->metric_type;
        auto final_index_cnd =
            pick_refine_index(raw_data_format, ivf_rabitq_cfg.refine_type, std::move(idx_rr), base_d, base_metric_type);
        if (!final_index_cnd.has_value()) {
            return expected<std::unique_ptr<IndexIVFRaBitQWrapper>>::Err(Status::invalid_args,
                                                                         "Invalid refine parameters");
        }

        idx_final = std::move(final_index_cnd.value());
    } else {
        // refine is not needed
        idx_final = std::move(idx_rr);
    }

    auto result = std::make_unique<IndexIVFRaBitQWrapper>(std::move(idx_final));
    return result;
}

IndexIVFRaBitQWrapper::IndexIVFRaBitQWrapper(std::unique_ptr<faiss::Index>&& index_in)
    : Index{index_in->d, index_in->metric_type}, index{std::move(index_in)} {
    ntotal = index->ntotal;
    is_trained = index->is_trained;
    is_cosine = index->is_cosine;
    verbose = index->verbose;
    metric_arg = index->metric_arg;
}

std::unique_ptr<IndexIVFRaBitQWrapper>
IndexIVFRaBitQWrapper::from_deserialized(std::unique_ptr<faiss::Index>&& index_in) {
    auto index = std::make_unique<IndexIVFRaBitQWrapper>(std::move(index_in));

    // check a provided index type
    auto index_rabitq = index->get_ivfrabitq_index();
    if (index_rabitq == nullptr) {
        return nullptr;
    }

    // construct an index map
    index_rabitq->make_direct_map(true);

    // done
    return index;
}

void
IndexIVFRaBitQWrapper::train(faiss::idx_t n, const float* x) {
    index->train(n, x);
    is_trained = index->is_trained;
}

void
IndexIVFRaBitQWrapper::add(faiss::idx_t n, const float* x) {
    index->add(n, x);
    this->ntotal = index->ntotal;
}

void
IndexIVFRaBitQWrapper::search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
                              const faiss::SearchParameters* params) const {
    index->search(n, x, k, distances, labels, params);
}

void
IndexIVFRaBitQWrapper::range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                                    const faiss::SearchParameters* params) const {
    index->range_search(n, x, radius, result, params);
}

void
IndexIVFRaBitQWrapper::reset() {
    index->reset();
    this->ntotal = 0;
}

void
IndexIVFRaBitQWrapper::merge_from(Index& otherIndex, faiss::idx_t add_id) {
    index->merge_from(otherIndex, add_id);
}

faiss::DistanceComputer*
IndexIVFRaBitQWrapper::get_distance_computer() const {
    return index->get_distance_computer();
}

// IndexHNSWRaBitQWrapper implementation
faiss::DistanceComputer*
IndexHNSWRaBitQWrapper::get_distance_computer() const {
    // Always use cached distance computer when enabled for HNSW optimization
    if (use_cached_distance_computer) {
        // Return cached version for HNSW optimization
        auto* ivfrabitq_idx = get_ivfrabitq_index();
        if (ivfrabitq_idx != nullptr) {
            // Get the PreTransform index to apply rotation
            faiss::IndexRefine* index_refine = dynamic_cast<faiss::IndexRefine*>(index.get());
            faiss::Index* index_for_pt = (index_refine != nullptr) ? index_refine->base_index : index.get();
            faiss::IndexPreTransform* index_pt = dynamic_cast<faiss::IndexPreTransform*>(index_for_pt);

            return new CachedHNSWRaBitDistanceComputer(ivfrabitq_idx, index_pt);
        }
    }
    // Default: use standard distance computer
    return IndexIVFRaBitQWrapper::get_distance_computer();
}

faiss::IndexIVFRaBitQ*
IndexIVFRaBitQWrapper::get_ivfrabitq_index() {
    // try refine
    faiss::IndexRefine* index_refine = dynamic_cast<faiss::IndexRefine*>(index.get());
    faiss::Index* index_for_pt = (index_refine != nullptr) ? index_refine->base_index : index.get();

    // pre-transform
    faiss::IndexPreTransform* index_pt = dynamic_cast<faiss::IndexPreTransform*>(index_for_pt);
    if (index_pt == nullptr) {
        return nullptr;
    }

    return dynamic_cast<faiss::IndexIVFRaBitQ*>(index_pt->index);
}

const faiss::IndexIVFRaBitQ*
IndexIVFRaBitQWrapper::get_ivfrabitq_index() const {
    // try refine
    const faiss::IndexRefine* index_refine = dynamic_cast<const faiss::IndexRefine*>(index.get());
    const faiss::Index* index_for_pt = (index_refine != nullptr) ? index_refine->base_index : index.get();

    // pre-transform
    const faiss::IndexPreTransform* index_pt = dynamic_cast<const faiss::IndexPreTransform*>(index_for_pt);
    if (index_pt == nullptr) {
        return nullptr;
    }

    return dynamic_cast<const faiss::IndexIVFRaBitQ*>(index_pt->index);
}

faiss::IndexRefine*
IndexIVFRaBitQWrapper::get_refine_index() {
    return dynamic_cast<faiss::IndexRefine*>(index.get());
}

const faiss::IndexRefine*
IndexIVFRaBitQWrapper::get_refine_index() const {
    return dynamic_cast<const faiss::IndexRefine*>(index.get());
}

size_t
IndexIVFRaBitQWrapper::size() const {
    if (index == nullptr) {
        return 0;
    }

    if (size_cache_.has_value()) {
        return size_cache_.value();
    }

    // a temporary yet expensive workaround
    faiss::cppcontrib::knowhere::CountSizeIOWriter writer;
    faiss::write_index(index.get(), &writer);

    // cache the size
    size_cache_ = writer.total_size;
    return writer.total_size;
}

std::unique_ptr<faiss::IVFIteratorWorkspace>
IndexIVFRaBitQWrapper::getIteratorWorkspace(const float* query_data,
                                            const faiss::IVFSearchParameters* ivfsearchParams) const {
    // try refine
    const faiss::IndexRefine* index_refine = dynamic_cast<const faiss::IndexRefine*>(index.get());
    faiss::Index* index_for_pt = (index_refine != nullptr) ? index_refine->base_index : index.get();

    const faiss::IndexPreTransform* index_pt = dynamic_cast<const faiss::IndexPreTransform*>(index_for_pt);
    if (index_pt == nullptr) {
        return nullptr;
    }

    const faiss::IndexIVFRaBitQ* index_rbq = dynamic_cast<const faiss::IndexIVFRaBitQ*>(index_pt->index);
    if (index_rbq == nullptr) {
        return nullptr;
    }

    // ok, transform the query
    std::unique_ptr<const float[]> transformed_query(index_pt->apply_chain(1, query_data));
    // create a workspace. This will make a clone of the transformed_query.
    auto workspace = index_rbq->getIteratorWorkspace(transformed_query.get(), ivfsearchParams);

    // check if refine exists
    if (index_refine != nullptr) {
        // create a distance
        // index_rbq == index_refine->base_index

        // a regular use case
        workspace->dis_refine =
            std::unique_ptr<faiss::DistanceComputer>(index_refine->refine_index->get_distance_computer());
        // this points to a previously saved clone
        workspace->dis_refine->set_query(workspace->query_data.data());
    } else {
        // don't use refine
        workspace->dis_refine = nullptr;
    }

    // done
    return workspace;
}

void
IndexIVFRaBitQWrapper::getIteratorNextBatch(faiss::IVFIteratorWorkspace* workspace, size_t current_backup_count) const {
    const auto ivfrbq = this->get_ivfrabitq_index();
    ivfrbq->getIteratorNextBatch(workspace, current_backup_count);
}

}  // namespace knowhere
