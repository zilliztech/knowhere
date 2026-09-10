// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "index/hnsw/impl/IndexHNSWWrapper.h"

namespace knowhere {
struct IndexHNSWRaBitQWrapper : IndexHNSWWrapper {
    using IndexHNSWWrapper::IndexHNSWWrapper;
    void
    search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
           const faiss::SearchParameters* params) const override;

 protected:
    std::unique_ptr<faiss::DistanceComputer>
    graph_distance_computer(const faiss::cppcontrib::knowhere::IndexHNSW* index,
                            const SearchParametersHNSWWrapper* params) const override;
    faiss::cppcontrib::knowhere::HNSWStats
    search_query(const faiss::cppcontrib::knowhere::HNSW& graph, faiss::DistanceComputer& dc,
                 faiss::cppcontrib::knowhere::Bitset& visited, faiss::idx_t k, float* distances, faiss::idx_t* labels,
                 const SearchParametersHNSWWrapper* params) const override;
};
}  // namespace knowhere
