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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "faiss/Index.h"
#include "faiss/IndexCosine.h"
#include "faiss/IndexIVF.h"
#include "faiss/IndexIVFRaBitQ.h"
#include "faiss/IndexRefine.h"
#include "index/ivf/ivf_config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"

namespace knowhere {

// This is wrapper is needed, bcz we use faiss::IndexPreTransform
//   for wrapping faiss::IndexIVFRaBitQ, optionally combined with
//   faiss::IndexRefine.
// The problem is that IndexPreTransform is a generic class, suitable
//   for any other use case as well, so this is wrong to reference
//   IndexPreTransform in the ivf.cc file.
struct IndexIVFRaBitQWrapper : faiss::Index {
    // this is one of two:
    // * faiss::IndexPreTransform + faiss::IndexIVFRaBitQ
    // * faiss::IndexPreTransform + faiss::IndexRefine + faiss::IndexIVFRaBitQ
    std::unique_ptr<faiss::Index> index;
    mutable std::optional<size_t> size_cache_ = std::nullopt;

    IndexIVFRaBitQWrapper(std::unique_ptr<faiss::Index>&& index_in);

    static expected<std::unique_ptr<IndexIVFRaBitQWrapper>>
    create(const faiss::idx_t d, const size_t nlist, const IvfRaBitQConfig& ivf_rabitq_cfg,
           // this is the data format of the raw data (if the refine is used)
           const DataFormatEnum raw_data_format, const faiss::MetricType metric = faiss::METRIC_L2);

    // this is for the deserialization.
    // returns nullptr if the provided index type is not the one
    //   as expected.
    static std::unique_ptr<IndexIVFRaBitQWrapper>
    from_deserialized(std::unique_ptr<faiss::Index>&& index_in);

    void
    train(faiss::idx_t n, const float* x) override;

    void
    add(faiss::idx_t n, const float* x) override;

    void
    search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
           const faiss::SearchParameters* params) const override;

    void
    range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                 const faiss::SearchParameters* params) const override;

    void
    reset() override;

    void
    merge_from(Index& otherIndex, faiss::idx_t add_id) override;

    faiss::DistanceComputer*
    get_distance_computer() const override;

    // point to IndexIVFRaBitQ or return nullptr.
    // this may also point to an index, owned by IndexRefine
    faiss::IndexIVFRaBitQ*
    get_ivfrabitq_index();
    const faiss::IndexIVFRaBitQ*
    get_ivfrabitq_index() const;

    // point to IndexRefine or return nullptr.
    faiss::IndexRefine*
    get_refine_index();
    const faiss::IndexRefine*
    get_refine_index() const;

    // return the size of the index
    size_t
    size() const;

    std::unique_ptr<faiss::IVFIteratorWorkspace>
    getIteratorWorkspace(const float* query_data, const faiss::IVFSearchParameters* ivfsearchParams) const;

    void
    getIteratorNextBatch(faiss::IVFIteratorWorkspace* workspace, size_t current_backup_count) const;
};

// Specialized wrapper for HNSW + IVF_RABITQ with cached distance computer optimization
struct IndexHNSWRaBitQWrapper : IndexIVFRaBitQWrapper {
    // Flag to enable/disable cached distance computer for HNSW search optimization
    mutable bool use_cached_distance_computer = false;

    // Query bits for search - mutable to allow setting in const search methods
    // This is used instead of directly modifying the underlying faiss index's qb field
    mutable uint8_t search_qb = 0;

    IndexHNSWRaBitQWrapper(std::unique_ptr<faiss::Index>&& index_in) : IndexIVFRaBitQWrapper(std::move(index_in)) {
    }

    // Override to support cached distance computer
    faiss::DistanceComputer*
    get_distance_computer() const override;

    // Enable/disable cached distance computer for HNSW optimization
    void
    set_use_cached_distance_computer(bool use_cached) const {
        use_cached_distance_computer = use_cached;
    }

    // Set the query bits parameter for search
    void
    set_search_qb(uint8_t qb) const {
        search_qb = qb;
    }

    // Get the query bits parameter
    uint8_t
    get_search_qb() const {
        return search_qb;
    }
};

// Cosine-specialized wrapper for IVF_RABITQ
// Similar to IndexScalarQuantizerCosine - stores inverse L2 norms and returns
// WithCosineNormDistanceComputer for proper cosine distance calculation
struct IndexIVFRaBitQWrapperCosine : IndexIVFRaBitQWrapper, faiss::HasInverseL2Norms {
    faiss::L2NormsStorage inverse_norms_storage;

    IndexIVFRaBitQWrapperCosine(std::unique_ptr<faiss::Index>&& index_in);

    // Create with IP metric for cosine similarity
    static expected<std::unique_ptr<IndexIVFRaBitQWrapperCosine>>
    create(const faiss::idx_t d, const size_t nlist, const IvfRaBitQConfig& ivf_rabitq_cfg,
           const DataFormatEnum raw_data_format);

    // Override to store inverse L2 norms along with data
    void
    add(faiss::idx_t n, const float* x) override;

    void
    reset() override;

    // Return WithCosineNormDistanceComputer for proper cosine distance
    faiss::DistanceComputer*
    get_distance_computer() const override;

    const float*
    get_inverse_l2_norms() const override;
};

// Cosine-specialized HNSW + IVF_RABITQ wrapper with cached distance computer optimization
// Inherits from IndexHNSWRaBitQWrapper for type compatibility, adds cosine norm storage
struct IndexHNSWRaBitQWrapperCosine : IndexHNSWRaBitQWrapper, faiss::HasInverseL2Norms {
    faiss::L2NormsStorage inverse_norms_storage;

    IndexHNSWRaBitQWrapperCosine(std::unique_ptr<faiss::Index>&& index_in);

    // Override to store inverse L2 norms along with data
    void
    add(faiss::idx_t n, const float* x) override;

    void
    reset() override;

    faiss::DistanceComputer*
    get_distance_computer() const override;

    const float*
    get_inverse_l2_norms() const override;
};

}  // namespace knowhere
