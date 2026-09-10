// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "index/hnsw/impl/IndexHNSWRaBitQWrapper.h"

#include <faiss/cppcontrib/knowhere/MetricType.h>
#include <faiss/cppcontrib/knowhere/impl/RaBitQDistanceEvaluation.h>
#include <faiss/cppcontrib/knowhere/impl/RaBitQSearch.h>

#include "index/hnsw/impl/HnswSearchDispatch.h"
#include "index/hnsw/impl/RaBitQSearchParameters.h"
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {
using idx_t = faiss::idx_t;
namespace rabitq_search = faiss::cppcontrib::knowhere::rabitq_search;

void
IndexHNSWRaBitQWrapper::search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
                               const faiss::SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    const auto* index_hnsw = dynamic_cast<const faiss::cppcontrib::knowhere::IndexHNSWRaBitQ*>(index);
    FAISS_THROW_IF_NOT(index_hnsw && index_hnsw->storage);
    const auto* params = dynamic_cast<const SearchParametersHNSWWrapper*>(params_in);
    FAISS_THROW_IF_NOT_MSG(!params_in || params, "params type invalid");
    if (index_hnsw->hnsw.entry_point == -1) {
        IndexHNSWWrapper::search(n, x, k, distances, labels, params_in);
        return;
    }
    const auto& hnsw = index_hnsw->hnsw;
    const auto* rbq_params = dynamic_cast<const SearchParametersHNSWRaBitQWrapper*>(params);
    // Use the optimized multi-bit path only when its selector/visitor contract
    // is satisfied. RBQ1, filtering and feder use the compatible searcher below.
    const auto* bitset_sel = params ? dynamic_cast<const knowhere::BitsetViewIDSelector*>(params->sel) : nullptr;
    const bool unfiltered = !params || !params->sel || (bitset_sel && bitset_sel->bitset_view.empty());
    if (index_hnsw->rabitq_index()->rabitq.nb_bits > 1 && unfiltered && (!params || !params->feder)) {
        rabitq_search::search(
            *index_hnsw, n, x, k, distances, labels, params ? params->efSearch : hnsw.efSearch,
            params ? params->check_relative_distance : hnsw.check_relative_distance,
            rbq_params ? &rbq_params->storage_params : nullptr, [&](const rabitq_search::SearchStats& counts) {
                const size_t hops = counts.expanded + counts.upper_expanded;
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
                knowhere::knowhere_hnsw_search_hops.Observe(hops);
#endif
                if (params && params->hnsw_stats) {
                    params->hnsw_stats->combine({.n1 = 1,
                                                 .n2 = size_t(counts.exhausted),
                                                 .ndis = counts.estimate + counts.refine + counts.upper_full,
                                                 .nhops = hops});
                }
            });
        if (faiss::cppcontrib::knowhere::is_similarity_metric(index->metric_type)) {
            for (idx_t i = 0; i < k * n; ++i) distances[i] = -distances[i];
        }
        return;
    }

    IndexHNSWWrapper::search(n, x, k, distances, labels, params_in);
}

std::unique_ptr<faiss::DistanceComputer>
IndexHNSWRaBitQWrapper::graph_distance_computer(const faiss::cppcontrib::knowhere::IndexHNSW* index,
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
    return search_hnsw_query<rabitq_search::DistanceEvaluation>(graph, dc, visited, k, distances, labels, params);
}
}  // namespace knowhere
