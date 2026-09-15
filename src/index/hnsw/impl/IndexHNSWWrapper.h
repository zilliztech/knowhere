// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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

#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/cppcontrib/knowhere/IndexWrapper.h>
#include <faiss/cppcontrib/knowhere/utils/Bitset.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "knowhere/feder/HNSW.h"

namespace knowhere {

// Custom parameters for IndexHNSW.
struct SearchParametersHNSWWrapper : public faiss::cppcontrib::knowhere::SearchParametersHNSW {
    // Stats will be updated if the object pointer is provided.
    faiss::cppcontrib::knowhere::HNSWStats* hnsw_stats = nullptr;
    // feder will be updated if the object pointer is provided.
    knowhere::feder::hnsw::FederResult* feder = nullptr;
    // filtering parameter
    float kAlpha = 1.0f;

    // Request-local storage factory, also used by brute-force fallback.
    virtual faiss::DistanceComputer*
    storage_distance_computer(const faiss::Index* index) const {
        return index->get_distance_computer();
    }

    virtual std::unique_ptr<faiss::Index>
    create_hnsw_wrapper(faiss::cppcontrib::knowhere::IndexHNSW* index) const;

    inline ~SearchParametersHNSWWrapper() {
    }
};

// TODO:
// Please note that this particular searcher is int32_t based, so won't
//   work correctly for 2B+ samples. This can be easily changed, if needed.

// override a search() procedure for IndexHNSW.
struct IndexHNSWWrapper : public faiss::cppcontrib::knowhere::IndexWrapper {
    IndexHNSWWrapper(faiss::cppcontrib::knowhere::IndexHNSW* underlying_index);

    /// entry point for search
    void
    search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
           const faiss::SearchParameters* params) const override;

    /// entry point for range search
    void
    range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                 const faiss::SearchParameters* params) const override;

 protected:
    // Graph-search factory: returns smaller-is-better distances. The index is
    // the owning HNSW index, allowing subclasses to access storage metadata.
    // Unlike the request-parameter factory, this may expose staged evaluation.
    virtual std::unique_ptr<faiss::DistanceComputer>
    storage_distance_computer(const faiss::cppcontrib::knowhere::IndexHNSW* index,
                              const SearchParametersHNSWWrapper* params) const;
    virtual faiss::cppcontrib::knowhere::HNSWStats
    search_query(const faiss::cppcontrib::knowhere::HNSW& graph, faiss::DistanceComputer& dc,
                 faiss::cppcontrib::knowhere::Bitset& visited, faiss::idx_t k, float* distances, faiss::idx_t* labels,
                 const SearchParametersHNSWWrapper* params) const;
};

}  // namespace knowhere
