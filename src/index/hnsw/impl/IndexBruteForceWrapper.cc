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

#include <algorithm>
#include <memory>

#include "index/hnsw/impl/BitsetFilter.h"
#include "knowhere/bitsetview.h"
#include "knowhere/bitsetview_idselector.h"

namespace knowhere {

using idx_t = faiss::idx_t;

//
IndexBruteForceWrapper::IndexBruteForceWrapper(faiss::Index* underlying_index)
    : faiss::cppcontrib::knowhere::IndexWrapper{underlying_index} {
}

void
IndexBruteForceWrapper::search(faiss::idx_t n, const float* __restrict x, faiss::idx_t k, float* __restrict distances,
                               faiss::idx_t* __restrict labels, const faiss::SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    std::unique_ptr<faiss::DistanceComputer> dis(index->get_distance_computer());

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        // prepare the query
        dis->set_query(x + i * index->d);

        // allocate heap
        idx_t* const local_ids = labels + i * index->d;
        float* const local_distances = distances + i * index->d;

        // set up a filter
        faiss::IDSelector* sel = (params == nullptr) ? nullptr : params->sel;

        // sel is assumed to be non-null
        if (sel == nullptr) {
            throw;
        }

        // try knowhere-specific filter
        const knowhere::BitsetViewIDSelector* bw_idselector = dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);

        knowhere::BitsetFilter filter(bw_idselector->bitset_view);

        if (is_similarity_metric(index->metric_type)) {
            using C = faiss::CMin<float, idx_t>;

            faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer, knowhere::BitsetFilter>(
                index->ntotal, *dis, filter, k, local_distances, local_ids);
        } else {
            using C = faiss::CMax<float, idx_t>;

            faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer, knowhere::BitsetFilter>(
                index->ntotal, *dis, filter, k, local_distances, local_ids);
        }
    }
}

}  // namespace knowhere
