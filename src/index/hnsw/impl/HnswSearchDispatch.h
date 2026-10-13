// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <faiss/cppcontrib/knowhere/impl/HnswSearcher.h>
#include <faiss/cppcontrib/knowhere/utils/Bitset.h>

#include <type_traits>

#include "index/hnsw/impl/DummyVisitor.h"
#include "index/hnsw/impl/FederVisitor.h"
#include "index/hnsw/impl/IndexHNSWWrapper.h"
#include "knowhere/bitsetview_idselector.h"

namespace knowhere {

// Reuse selector/visitor dispatch without exposing codec types to common HNSW.
template <class DistanceEvaluationT = faiss::cppcontrib::knowhere::DefaultHnswDistanceEvaluation>
faiss::cppcontrib::knowhere::HNSWStats
search_hnsw_query(const faiss::cppcontrib::knowhere::HNSW& graph, faiss::DistanceComputer& distance,
                  faiss::cppcontrib::knowhere::Bitset& visited, faiss::idx_t k, float* distances, faiss::idx_t* labels,
                  const SearchParametersHNSWWrapper* params) {
    auto run = [&](auto& visitor, const auto& selector) {
        using Visitor = std::remove_reference_t<decltype(visitor)>;
        using Selector = std::decay_t<decltype(selector)>;
        faiss::cppcontrib::knowhere::v2_hnsw_searcher<
            faiss::DistanceComputer, Visitor, faiss::cppcontrib::knowhere::Bitset, Selector, DistanceEvaluationT>
            searcher{graph, distance, visitor, visited, selector, params ? params->kAlpha : 0.0f, params};
        return searcher.search(k, distances, labels);
    };
    auto visit = [&](const auto& selector) {
        if (params && params->feder) {
            FederVisitor visitor(params->feder);
            return run(visitor, selector);
        }
        DummyVisitor visitor;
        return run(visitor, selector);
    };
    const auto* selector = params ? dynamic_cast<const BitsetViewIDSelector*>(params->sel) : nullptr;
    if (selector && !selector->bitset_view.empty())
        return visit(*selector);
    faiss::IDSelectorAll all;
    return visit(all);
}
}  // namespace knowhere
