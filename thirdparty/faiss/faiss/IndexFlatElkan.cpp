// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <faiss/IndexFlatElkan.h>

#include <memory>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

namespace faiss {

IndexFlatElkan::IndexFlatElkan(idx_t d, MetricType metric, bool is_cosine, bool use_elkan)
        : IndexFlat(d, metric, is_cosine) {
    this->use_elkan = use_elkan;
    if (this->use_elkan)
        this->tmp_buffer_for_elkan = std::make_unique<float[]>(1024 * (1024 - 1) / 2);
}

void IndexFlatElkan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
        // usually used in IVF k-means algorithm

    FAISS_THROW_IF_NOT_MSG(
        k == 1,
        "this index requires k == 1 in a search() call."
    );
    FAISS_THROW_IF_NOT_MSG(
        params == nullptr,
        "search params not supported for this index"
    );

    float* dis_inner = distances;
    std::unique_ptr<float[]> dis_inner_deleter = nullptr;
    if (distances == nullptr) {
        dis_inner_deleter = std::make_unique<float[]>(n);
        dis_inner = dis_inner_deleter.get();
    }

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
        case METRIC_L2: {
            // ignore the metric_type, both use L2
            if (use_elkan) {
                // use elkan
                elkan_L2_sse(x, get_xb(), d, n, ntotal, labels, dis_inner, tmp_buffer_for_elkan.get());
            }
            else {
                // use L2 search. The same code as in IndexFlat::search() for L2.
                IDSelector* sel = params ? params->sel : nullptr;

                float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
                knn_L2sqr(x, get_xb(), d, n, ntotal, &res, nullptr, sel);
            }

            break;
        }
        default: {
            // binary metrics
            // There may be something wrong, but maintain the original logic
            // now.
            IndexFlat::search(n, x, k, dis_inner, labels, params);
            break;
        }
    }
}

}
