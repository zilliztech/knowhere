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

#include <faiss/cppcontrib/knowhere/IndexBruteForceWrapper.h>

#include <algorithm>
#include <memory>

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/cppcontrib/knowhere/impl/Bruteforce.h>
#include <faiss/cppcontrib/knowhere/impl/Filters.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

IndexBruteForceWrapper::IndexBruteForceWrapper(Index* underlying_index) :
    IndexWrapper{underlying_index} {}

void IndexBruteForceWrapper::search(
        idx_t n,
        const float* __restrict x,
        idx_t k,
        float* __restrict distances,
        idx_t* __restrict labels,
        const SearchParameters* params
) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t check_period = InterruptCallback::get_period_hint(
            index->d * index->ntotal);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<faiss::DistanceComputer> dis(index->get_distance_computer());

#pragma omp for schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                // prepare the query
                dis->set_query(x + i * index->d);

                // allocate heap
                idx_t* const local_ids = labels + i * index->d;
                float* const local_distances = distances + i * index->d;

                // set up a filter
                IDSelector* sel = (params == nullptr) ? nullptr : params->sel;

                // template, just in case a filter type will be specialized
                //   in order to remove virtual function call overhead.
                using filter_type = DefaultIDSelectorFilter<IDSelector>;
                filter_type filter(sel);

                if (is_similarity_metric(index->metric_type)) {
                    using C = CMin<float, idx_t>;

                    brute_force_search_impl<C, DistanceComputer, filter_type>(
                        index->ntotal,
                        *dis,
                        filter,
                        k,
                        local_distances,
                        local_ids
                    );
                } else {
                    using C = CMax<float, idx_t>;

                    brute_force_search_impl<C, DistanceComputer, filter_type>(
                        index->ntotal,
                        *dis,
                        filter,
                        k,
                        local_distances,
                        local_ids
                    );
                }
            }
        }

        InterruptCallback::check();
    }
}

}
}
}
