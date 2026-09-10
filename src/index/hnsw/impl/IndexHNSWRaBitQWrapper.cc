// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "index/hnsw/impl/IndexHNSWRaBitQWrapper.h"

#include <faiss/cppcontrib/knowhere/impl/RaBitQHnswDistanceEvaluation.h>

#include "index/hnsw/impl/HnswSearchDispatch.h"
#include "index/hnsw/impl/RaBitQSearchParameters.h"

namespace knowhere {
namespace rabitq_search = faiss::cppcontrib::knowhere::rabitq_search;

std::unique_ptr<faiss::DistanceComputer>
IndexHNSWRaBitQWrapper::storage_distance_computer(const faiss::cppcontrib::knowhere::IndexHNSW* index,
                                                  const SearchParametersHNSWWrapper* params) const {
    const auto* rbq = dynamic_cast<const faiss::cppcontrib::knowhere::IndexHNSWRaBitQ*>(index);
    FAISS_THROW_IF_NOT(rbq);
    const auto* rbq_params = dynamic_cast<const SearchParametersHNSWRaBitQWrapper*>(params);
    return std::unique_ptr<faiss::DistanceComputer>(
        rbq->get_staged_distance_computer(rbq_params ? &rbq_params->storage_params : nullptr));
}

faiss::cppcontrib::knowhere::HNSWStats
IndexHNSWRaBitQWrapper::search_query(const faiss::cppcontrib::knowhere::HNSW& graph, faiss::DistanceComputer& dc,
                                     faiss::cppcontrib::knowhere::Bitset& visited, faiss::idx_t k, float* distances,
                                     faiss::idx_t* labels, const SearchParametersHNSWWrapper* params) const {
    return search_hnsw_query<rabitq_search::RaBitQHnswDistanceEvaluation>(graph, dc, visited, k, distances, labels,
                                                                          params);
}
}  // namespace knowhere
