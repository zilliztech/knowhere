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

#include "index/hnsw/impl/IndexHNSWWrapper.h"

#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/cppcontrib/knowhere/impl/Bruteforce.h>
#include <faiss/cppcontrib/knowhere/impl/HnswSearcher.h>
#include <faiss/cppcontrib/knowhere/utils/Bitset.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "index/hnsw/impl/DummyVisitor.h"
#include "index/hnsw/impl/FederVisitor.h"
#include "knowhere/bitsetview.h"
#include "knowhere/bitsetview_idselector.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

/**************************************************************
 * Utilities
 **************************************************************/

namespace {

// cloned from IndexHNSW.cpp
faiss::DistanceComputer*
storage_distance_computer(const faiss::Index* storage) {
    if (faiss::is_similarity_metric(storage->metric_type)) {
        return new faiss::NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

}  // namespace

/**************************************************************
 * IndexHNSWWrapper implementation
 **************************************************************/

using idx_t = faiss::idx_t;

IndexHNSWWrapper::IndexHNSWWrapper(faiss::IndexHNSW* underlying_index)
    : faiss::cppcontrib::knowhere::IndexWrapper(underlying_index) {
}

void
IndexHNSWWrapper::search(idx_t n, const float* __restrict x, idx_t k, float* __restrict distances,
                         idx_t* __restrict labels, const faiss::SearchParameters* __restrict params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    const faiss::IndexHNSW* index_hnsw = dynamic_cast<const faiss::IndexHNSW*>(index);
    FAISS_THROW_IF_NOT(index_hnsw);

    FAISS_THROW_IF_NOT_MSG(index_hnsw->storage, "No storage index");

    // set up
    using C = faiss::HNSW::C;

    // check if the graph is empty
    if (index_hnsw->hnsw.entry_point == -1) {
        for (idx_t i = 0; i < k * n; i++) {
            distances[i] = C::neutral();
            labels[i] = -1;
        }

        return;
    }

    // check parameters
    const SearchParametersHNSWWrapper* params = nullptr;
    const faiss::HNSW& hnsw = index_hnsw->hnsw;

    float kAlpha = 0.0f;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSWWrapper*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");

        kAlpha = params->kAlpha;
    }

    // set up hnsw_stats
    faiss::HNSWStats* __restrict const hnsw_stats = (params == nullptr) ? nullptr : params->hnsw_stats;

    //
    size_t n1 = 0;
    size_t n2 = 0;
    size_t ndis = 0;
    size_t nhops = 0;

    //
    faiss::cppcontrib::knowhere::Bitset bitset_visited_nodes =
        faiss::cppcontrib::knowhere::Bitset::create_uninitialized(index->ntotal);

    // create a distance computer
    std::unique_ptr<faiss::DistanceComputer> dis(storage_distance_computer(index_hnsw->storage));

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        // prepare the query
        dis->set_query(x + i * index->d);

        // prepare the table of visited elements
        bitset_visited_nodes.clear();

        // a visitor
        knowhere::feder::hnsw::FederResult* feder = (params == nullptr) ? nullptr : params->feder;

        // future results
        faiss::HNSWStats local_stats;

        // set up a filter
        faiss::IDSelector* sel = (params == nullptr) ? nullptr : params->sel;

        // try knowhere-specific filter
        const knowhere::BitsetViewIDSelector* __restrict bw_idselector =
            dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);

        if (bw_idselector == nullptr || bw_idselector->bitset_view.empty()) {
            // no filter
            faiss::IDSelectorAll sel_all;

            // feder templating is important, bcz it removes an unneeded 'CALL' instruction.
            if (feder == nullptr) {
                // no feder
                DummyVisitor graph_visitor;

                using searcher_type = faiss::cppcontrib::knowhere::v2_hnsw_searcher<
                    faiss::DistanceComputer, DummyVisitor, faiss::cppcontrib::knowhere::Bitset, faiss::IDSelectorAll>;

                searcher_type searcher{hnsw,    *(dis.get()), graph_visitor, bitset_visited_nodes,
                                       sel_all, kAlpha,       params};

                local_stats = searcher.search(k, distances + i * k, labels + i * k);
            } else {
                // use feder
                FederVisitor graph_visitor(feder);

                using searcher_type = faiss::cppcontrib::knowhere::v2_hnsw_searcher<
                    faiss::DistanceComputer, FederVisitor, faiss::cppcontrib::knowhere::Bitset, faiss::IDSelectorAll>;

                searcher_type searcher{hnsw,    *(dis.get()), graph_visitor, bitset_visited_nodes,
                                       sel_all, kAlpha,       params};

                local_stats = searcher.search(k, distances + i * k, labels + i * k);
            }
        } else {
            // with filter

            // feder templating is important, bcz it removes an unneeded 'CALL' instruction.
            if (feder == nullptr) {
                // no feder
                DummyVisitor graph_visitor;

                using searcher_type =
                    faiss::cppcontrib::knowhere::v2_hnsw_searcher<faiss::DistanceComputer, DummyVisitor,
                                                                  faiss::cppcontrib::knowhere::Bitset,
                                                                  knowhere::BitsetViewIDSelector>;

                searcher_type searcher{hnsw,           *(dis.get()), graph_visitor, bitset_visited_nodes,
                                       *bw_idselector, kAlpha,       params};

                local_stats = searcher.search(k, distances + i * k, labels + i * k);
            } else {
                // use feder
                FederVisitor graph_visitor(feder);

                using searcher_type =
                    faiss::cppcontrib::knowhere::v2_hnsw_searcher<faiss::DistanceComputer, FederVisitor,
                                                                  faiss::cppcontrib::knowhere::Bitset,
                                                                  knowhere::BitsetViewIDSelector>;

                searcher_type searcher{hnsw,           *(dis.get()), graph_visitor, bitset_visited_nodes,
                                       *bw_idselector, kAlpha,       params};

                local_stats = searcher.search(k, distances + i * k, labels + i * k);
            }
        }

        // record some statistics
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        knowhere::knowhere_hnsw_search_hops.Observe(local_stats.nhops);
#endif

        // update stats if possible
        if (hnsw_stats != nullptr) {
            n1 += local_stats.n1;
            n2 += local_stats.n2;
            ndis += local_stats.ndis;
            nhops += local_stats.nhops;
        }
    }

    // update stats if possible
    if (hnsw_stats != nullptr) {
        hnsw_stats->combine({n1, n2, ndis, nhops});
    }

    // done, update the results, if needed
    if (is_similarity_metric(index->metric_type)) {
        // we need to revert the negated distances
        for (idx_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void
IndexHNSWWrapper::range_search(idx_t n, const float* __restrict x, float radius_in,
                               faiss::RangeSearchResult* __restrict result,
                               const faiss::SearchParameters* __restrict params_in) const {
    const faiss::IndexHNSW* index_hnsw = dynamic_cast<const faiss::IndexHNSW*>(index);
    FAISS_THROW_IF_NOT(index_hnsw);

    FAISS_THROW_IF_NOT_MSG(index_hnsw->storage, "No storage index");

    // check if the graph is empty
    if (index_hnsw->hnsw.entry_point == -1) {
        return;
    }

    // check parameters
    const SearchParametersHNSWWrapper* params = nullptr;
    const faiss::HNSW& hnsw = index_hnsw->hnsw;

    float kAlpha = 0.0f;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSWWrapper*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");

        kAlpha = params->kAlpha;
    }

    // set up hnsw_stats
    faiss::HNSWStats* __restrict const hnsw_stats = (params == nullptr) ? nullptr : params->hnsw_stats;

    //
    size_t n1 = 0;
    size_t n2 = 0;
    size_t ndis = 0;
    size_t nhops = 0;

    //
    faiss::cppcontrib::knowhere::Bitset bitset_visited_nodes =
        faiss::cppcontrib::knowhere::Bitset::create_uninitialized(index->ntotal);

    // create a distance computer
    std::unique_ptr<faiss::DistanceComputer> dis(storage_distance_computer(index_hnsw->storage));

    // radius
    float radius = radius_in;
    if (is_similarity_metric(this->metric_type)) {
        radius *= (-1);
    }

    // initialize a ResultHandler
    using RH_min = faiss::RangeSearchBlockResultHandler<faiss::CMax<float, int64_t>>;
    RH_min bres_min(result, radius);

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        //
        typename RH_min::SingleResultHandler res_min(bres_min);
        res_min.begin(i);

        // prepare the query
        dis->set_query(x + i * index->d);

        // prepare the table of visited elements
        bitset_visited_nodes.clear();

        // a visitor
        knowhere::feder::hnsw::FederResult* feder = (params == nullptr) ? nullptr : params->feder;

        // future results
        faiss::HNSWStats local_stats;

        // set up a filter
        faiss::IDSelector* sel = (params == nullptr) ? nullptr : params->sel;

        // try knowhere-specific filter
        const knowhere::BitsetViewIDSelector* __restrict bw_idselector =
            dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);

        if (bw_idselector == nullptr || bw_idselector->bitset_view.empty()) {
            // no filter
            faiss::IDSelectorAll sel_all;

            // feder templating is important, bcz it removes an unneeded 'CALL' instruction.
            if (feder == nullptr) {
                // no feder
                DummyVisitor graph_visitor;

                using searcher_type = faiss::cppcontrib::knowhere::v2_hnsw_searcher<
                    faiss::DistanceComputer, DummyVisitor, faiss::cppcontrib::knowhere::Bitset, faiss::IDSelectorAll>;

                searcher_type searcher{hnsw,    *(dis.get()), graph_visitor, bitset_visited_nodes,
                                       sel_all, kAlpha,       params};

                local_stats = searcher.range_search(radius, &res_min);
            } else {
                // use feder
                FederVisitor graph_visitor(feder);

                using searcher_type = faiss::cppcontrib::knowhere::v2_hnsw_searcher<
                    faiss::DistanceComputer, FederVisitor, faiss::cppcontrib::knowhere::Bitset, faiss::IDSelectorAll>;

                searcher_type searcher{hnsw,    *(dis.get()), graph_visitor, bitset_visited_nodes,
                                       sel_all, kAlpha,       params};

                local_stats = searcher.range_search(radius, &res_min);
            }
        } else {
            // with filter

            // feder templating is important, bcz it removes an unneeded 'CALL' instruction.
            if (feder == nullptr) {
                // no feder
                DummyVisitor graph_visitor;

                using searcher_type =
                    faiss::cppcontrib::knowhere::v2_hnsw_searcher<faiss::DistanceComputer, DummyVisitor,
                                                                  faiss::cppcontrib::knowhere::Bitset,
                                                                  knowhere::BitsetViewIDSelector>;

                searcher_type searcher{hnsw,           *(dis.get()), graph_visitor, bitset_visited_nodes,
                                       *bw_idselector, kAlpha,       params};

                local_stats = searcher.range_search(radius, &res_min);
            } else {
                // use feder
                FederVisitor graph_visitor(feder);

                using searcher_type =
                    faiss::cppcontrib::knowhere::v2_hnsw_searcher<faiss::DistanceComputer, FederVisitor,
                                                                  faiss::cppcontrib::knowhere::Bitset,
                                                                  knowhere::BitsetViewIDSelector>;

                searcher_type searcher{hnsw,           *(dis.get()), graph_visitor, bitset_visited_nodes,
                                       *bw_idselector, kAlpha,       params};

                local_stats = searcher.range_search(radius, &res_min);
            }
        }

        // record some statistics
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        knowhere::knowhere_hnsw_search_hops.Observe(local_stats.nhops);
#endif

        // update stats if possible
        if (hnsw_stats != nullptr) {
            n1 += local_stats.n1;
            n2 += local_stats.n2;
            ndis += local_stats.ndis;
            nhops += local_stats.nhops;
        }

        //
        res_min.end();
    }

    // update stats if possible
    if (hnsw_stats != nullptr) {
        hnsw_stats->combine({n1, n2, ndis, nhops});
    }

    // done, update the results, if needed
    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

}  // namespace knowhere
