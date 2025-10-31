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
#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexScalarQuantizer.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexRefine.h"
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
    merge_from(Index& otherIndex, faiss::idx_t add_id) override;

    faiss::DistanceComputer*
    get_distance_computer() const override;

    // point to IndexIVFType or return nullptr.
    // this may also point to an index, owned by IndexRefine
    IndexIVFType*
    get_base_ivf_index();
    const IndexIVFType*
    get_base_ivf_index() const;

    // point to IndexRefine or return nullptr.
    faiss::IndexRefine*
    get_refine_index();
    const faiss::IndexRefine*
    get_refine_index() const;

    // return the size of the index
    size_t
    size() const;

    template <typename U = IndexIVFType>
    typename std::enable_if<std::is_same_v<U, faiss::IndexIVFScalarQuantizer>, std::unique_ptr<faiss::IVFIteratorWorkspace>>::type
    getIteratorWorkspace(const float* query_data, const faiss::IVFSearchParameters* ivfsearchParams) const;

    template <typename U = IndexIVFType>
    typename std::enable_if<std::is_same_v<U, faiss::IndexIVFScalarQuantizer>, void>::type
    getIteratorNextBatch(faiss::IVFIteratorWorkspace* workspace, size_t current_backup_count) const;
};

using IndexIVFPQWrapper = IndexIVFWrapper<faiss::IndexIVFPQ>;
using IndexIVFSQWrapper = IndexIVFWrapper<faiss::IndexIVFScalarQuantizer>;

class IndexIvfFactory {
 public:
    static expected<std::unique_ptr<IndexIVFPQWrapper>>
    create_for_pq(faiss::IndexFlat* qzr_raw_ptr,
                  const faiss::idx_t d, const size_t nlist, const size_t nbits, const IvfPqConfig& ivf_pq_cfg,
                  // this is the data format of the raw data (if the refine is used)
                  const DataFormatEnum raw_data_format, const faiss::MetricType metric = faiss::METRIC_L2);

    static expected<std::unique_ptr<IndexIVFSQWrapper>>
    create_for_sq(faiss::IndexFlat* qzr_raw_ptr,
                  const faiss::idx_t d, const size_t nlist, const IvfSqConfig& ivf_sq_cfg,
                  // this is the data format of the raw data (if the refine is used)
                  const DataFormatEnum raw_data_format, const faiss::MetricType metric = faiss::METRIC_L2);
};

}  // namespace knowhere
