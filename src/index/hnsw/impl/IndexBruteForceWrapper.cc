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

#include "index/hnsw/impl/IndexBruteForceWrapper.h"

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/cppcontrib/knowhere/impl/Bruteforce.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>

#include <algorithm>
#include <memory>

#include "knowhere/bitsetview.h"
#include "knowhere/bitsetview_idselector.h"

namespace knowhere {

using idx_t = faiss::idx_t;

// the following structure is a hack, because GCC cannot properly
//   de-virtualize a plain BitsetViewIDSelector.
struct BitsetViewIDSelectorWrapper final {
    const BitsetView bitset_view;

    inline BitsetViewIDSelectorWrapper(BitsetView bitset_view) : bitset_view{bitset_view} {
    }

    [[nodiscard]] inline bool
    is_member(faiss::idx_t id) const {
        // it is by design that bitset_view.empty() is not tested here
        return (!bitset_view.test(id));
    }
};

//
IndexBruteForceWrapper::IndexBruteForceWrapper(faiss::Index* underlying_index)
    : faiss::cppcontrib::knowhere::IndexWrapper{underlying_index} {
}

void
IndexBruteForceWrapper::search(faiss::idx_t n, const float* __restrict x, faiss::idx_t k, float* __restrict distances,
                               faiss::idx_t* __restrict labels,
                               const faiss::SearchParameters* __restrict params) const {
    FAISS_THROW_IF_NOT(k > 0);

    std::unique_ptr<faiss::DistanceComputer> dis(index->get_distance_computer());

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        // prepare the query
        dis->set_query(x + i * index->d);

        // allocate heap
        idx_t* const __restrict local_ids = labels + i * index->d;
        float* const __restrict local_distances = distances + i * index->d;

        // set up a filter
        faiss::IDSelector* sel = (params == nullptr) ? nullptr : params->sel;

        // sel is assumed to be non-null
        if (sel == nullptr) {
            throw;
        }

        // try knowhere-specific filter
        const knowhere::BitsetViewIDSelector* __restrict bw_idselector =
            dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);

        BitsetViewIDSelectorWrapper bw_idselector_w(bw_idselector->bitset_view);

        if (is_similarity_metric(index->metric_type)) {
            using C = faiss::CMin<float, idx_t>;

            if (bw_idselector == nullptr || bw_idselector->bitset_view.empty()) {
                faiss::IDSelectorAll sel_all;
                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, k, local_distances, local_ids);
            } else {
                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer,
                                                                     BitsetViewIDSelectorWrapper>(
                    index->ntotal, *dis, bw_idselector_w, k, local_distances, local_ids);
            }
        } else {
            using C = faiss::CMax<float, idx_t>;

            if (bw_idselector == nullptr || bw_idselector->bitset_view.empty()) {
                faiss::IDSelectorAll sel_all;
                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, k, local_distances, local_ids);
            } else {
                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer,
                                                                     BitsetViewIDSelectorWrapper>(
                    index->ntotal, *dis, bw_idselector_w, k, local_distances, local_ids);
            }
        }
    }
}

void
IndexBruteForceWrapper::range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                                     const faiss::SearchParameters* params) const {
    using RH_min = faiss::RangeSearchBlockResultHandler<faiss::CMax<float, int64_t>>;
    using RH_max = faiss::RangeSearchBlockResultHandler<faiss::CMin<float, int64_t>>;
    RH_min bres_min(result, radius);
    RH_max bres_max(result, radius);

    std::unique_ptr<faiss::DistanceComputer> dis(index->get_distance_computer());

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        // prepare the query
        dis->set_query(x + i * index->d);

        // set up a filter
        faiss::IDSelector* __restrict sel = (params == nullptr) ? nullptr : params->sel;

        if (is_similarity_metric(index->metric_type)) {
            typename RH_max::SingleResultHandler res_max(bres_max);
            res_max.begin(i);

            if (sel == nullptr) {
                // Compiler is expected to de-virtualize virtual method calls
                faiss::IDSelectorAll sel_all;

                faiss::cppcontrib::knowhere::brute_force_range_search_impl<
                    typename RH_max::SingleResultHandler, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, res_max);
            } else {
                faiss::cppcontrib::knowhere::brute_force_range_search_impl<typename RH_max::SingleResultHandler,
                                                                           faiss::DistanceComputer, faiss::IDSelector>(
                    index->ntotal, *dis, *sel, res_max);
            }

            res_max.end();
        } else {
            typename RH_min::SingleResultHandler res_min(bres_min);
            res_min.begin(i);

            if (sel == nullptr) {
                // Compiler is expected to de-virtualize virtual method calls
                faiss::IDSelectorAll sel_all;

                faiss::cppcontrib::knowhere::brute_force_range_search_impl<
                    typename RH_min::SingleResultHandler, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, res_min);
            } else {
                faiss::cppcontrib::knowhere::brute_force_range_search_impl<typename RH_min::SingleResultHandler,
                                                                           faiss::DistanceComputer, faiss::IDSelector>(
                    index->ntotal, *dis, *sel, res_min);
            }

            res_min.end();
        }
    }
}

}  // namespace knowhere
