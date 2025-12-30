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

#include "faiss/Index.h"
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
};

}  // namespace knowhere
