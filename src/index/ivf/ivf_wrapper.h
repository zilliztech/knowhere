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
#include "faiss/cppcontrib/knowhere/IVFIteratorWorkspace.h"
#include "faiss/cppcontrib/knowhere/IndexFlat.h"
#include "faiss/cppcontrib/knowhere/IndexIVF.h"
#include "faiss/cppcontrib/knowhere/IndexIVFPQ.h"
#include "faiss/cppcontrib/knowhere/IndexRefine.h"
#include "faiss/cppcontrib/knowhere/IndexScalarQuantizer.h"
#include "index/ivf/ivf_config.h"
#include "knowhere/expected.h"

namespace knowhere {

// This is wrapper is needed, bcz we use faiss::IndexIVFPQ/faiss::IndexIVFScalarQuantizer
//   optionally combined with faiss::IndexRefine.
template <typename IndexIVFType>
struct IndexIVFWrapper : faiss::Index {
    // this is one of two:
    // * IndexIVFType
    // * faiss::IndexRefine + IndexIVFType
    std::unique_ptr<faiss::Index> index;
    mutable std::optional<size_t> size_cache_ = std::nullopt;

    IndexIVFWrapper(std::unique_ptr<faiss::Index>&& index_in);

    // this is for the deserialization.
    // returns nullptr if the provided index type is not the one
    //   as expected.
    static std::unique_ptr<IndexIVFWrapper<IndexIVFType>>
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
    merge_from(faiss::Index& otherIndex, faiss::idx_t add_id) override;

    faiss::DistanceComputer*
    get_distance_computer() const override;

    // point to IndexIVFType or return nullptr.
    // this may also point to an index, owned by IndexRefine
    IndexIVFType*
    get_base_ivf_index();
    const IndexIVFType*
    get_base_ivf_index() const;

    // point to IndexRefine or return nullptr.
    faiss::cppcontrib::knowhere::IndexRefine*
    get_refine_index();
    const faiss::cppcontrib::knowhere::IndexRefine*
    get_refine_index() const;

    // return the size of the index
    size_t
    size() const;
};

using IndexIVFPQWrapper = IndexIVFWrapper<faiss::cppcontrib::knowhere::IndexIVFPQ>;
using IndexIVFSQWrapper = IndexIVFWrapper<faiss::cppcontrib::knowhere::IndexIVFScalarQuantizer>;

/// Standalone iterator workspace for IndexIVFWrapper<T>.
/// Navigates the wrapper structure (optional IndexRefine + IndexIVFType)
/// and delegates scanning to IVFBaseIteratorWorkspace.
template <typename IndexIVFTypeT>
struct IVFWrapperIteratorWorkspace : faiss::cppcontrib::knowhere::IVFIteratorWorkspace {
    const IndexIVFWrapper<IndexIVFTypeT>* wrapper;
    std::unique_ptr<faiss::cppcontrib::knowhere::IVFIteratorWorkspace> inner;

    IVFWrapperIteratorWorkspace(const IndexIVFWrapper<IndexIVFTypeT>* wrapper, const float* query_data,
                                const faiss::cppcontrib::knowhere::IVFSearchParameters* params)
        : wrapper(wrapper) {
        const auto* index_refine = wrapper->get_refine_index();
        const auto* index_ivf = wrapper->get_base_ivf_index();

        inner = std::make_unique<faiss::cppcontrib::knowhere::IVFBaseIteratorWorkspace>(index_ivf, query_data, params);

        if (index_refine != nullptr) {
            this->dis_refine =
                std::unique_ptr<faiss::DistanceComputer>(index_refine->refine_index->get_distance_computer());
            this->dis_refine->set_query(inner->query_data.data());
        }
        this->search_params = inner->search_params;
    }

    void
    next_batch(size_t current_backup_count) override {
        inner->next_batch(current_backup_count);
        this->dists = std::move(inner->dists);
    }
};

using IVFSQIteratorWorkspace = IVFWrapperIteratorWorkspace<faiss::cppcontrib::knowhere::IndexIVFScalarQuantizer>;

class IndexIvfFactory {
 public:
    static expected<std::unique_ptr<IndexIVFPQWrapper>>
    create_for_pq(faiss::cppcontrib::knowhere::IndexFlat* qzr_raw_ptr, const faiss::idx_t d, const size_t nlist,
                  const size_t nbits, const IvfPqConfig& ivf_pq_cfg,
                  // this is the data format of the raw data (if the refine is used)
                  const DataFormatEnum raw_data_format, const faiss::MetricType metric = faiss::METRIC_L2);

    static expected<std::unique_ptr<IndexIVFSQWrapper>>
    create_for_sq(faiss::cppcontrib::knowhere::IndexFlat* qzr_raw_ptr, const faiss::idx_t d, const size_t nlist,
                  const IvfSqConfig& ivf_sq_cfg,
                  // this is the data format of the raw data (if the refine is used)
                  const DataFormatEnum raw_data_format, const faiss::MetricType metric = faiss::METRIC_L2);
};

}  // namespace knowhere
