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

#include "range_util.h"

#include <algorithm>
#include <cinttypes>

#include "knowhere/log.h"
namespace knowhere {

///////////////////////////////////////////////////////////////////////////////
// For Faiss index types
size_t
CountValidRangeSearchResult(const faiss::RangeSearchResult& res, const bool is_ip, const int64_t nq, const float radius,
                            const float range_filter, size_t*& lims) {
    lims = new size_t[nq + 1];
    lims[0] = 0;
    for (int64_t i = 0; i < nq; i++) {
        int64_t valid = 0;
        for (size_t j = res.lims[i]; j < res.lims[i + 1]; j++) {
            if (distance_in_range(res.distances[j], radius, range_filter, is_ip)) {
                valid++;
            }
        }
        lims[i + 1] = lims[i] + valid;
    }
    return lims[nq];
}

void
FilterRangeSearchResultForOneNq(const int64_t i_size, const float* i_distances, const int64_t* i_labels,
                                const bool is_ip, const float radius, const float range_filter, const int64_t o_size,
                                float* o_distances, int64_t* o_labels, const BitsetView& bitset) {
    int64_t num = 0;
    for (int64_t i = 0; i < i_size; i++) {
        auto dis = i_distances[i];
        auto id = i_labels[i];
        KNOWHERE_THROW_IF_NOT_MSG(bitset.empty() || !bitset.test(id), "bitset invalid");
        if (distance_in_range(dis, radius, range_filter, is_ip)) {
            o_labels[num] = id;
            o_distances[num] = dis;
            num++;
        }
    }
    KNOWHERE_THROW_IF_NOT_FMT(num == o_size, "%" SCNd64 " not equal %" SCNd64, num, o_size);
}

void
GetRangeSearchResult(const faiss::RangeSearchResult& res, const bool is_ip, const int64_t nq, const float radius,
                     const float range_filter, float*& distances, int64_t*& labels, size_t*& lims,
                     const BitsetView& bitset) {
    auto total_valid = CountValidRangeSearchResult(res, is_ip, nq, radius, range_filter, lims);
    LOG_KNOWHERE_DEBUG_ << "Range search: is_ip " << (is_ip ? "True" : "False") << ", radius " << radius
                        << ", range_filter " << range_filter << ", total result num " << total_valid;

    distances = new float[total_valid];
    labels = new int64_t[total_valid];

#pragma omp parallel for
    for (auto i = 0; i < nq; i++) {
        FilterRangeSearchResultForOneNq(res.lims[i + 1] - res.lims[i], res.distances + res.lims[i],
                                        res.labels + res.lims[i], is_ip, radius, range_filter, lims[i + 1] - lims[i],
                                        distances + lims[i], labels + lims[i], bitset);
    }
}

///////////////////////////////////////////////////////////////////////////////
// for HNSW and DiskANN
void
FilterRangeSearchResultForOneNq(std::vector<float>& distances, std::vector<int64_t>& labels, const bool is_ip,
                                const float radius, const float range_filter) {
    KNOWHERE_THROW_IF_NOT_FMT(distances.size() == labels.size(), "distances' size %ld not equal to labels' size %ld",
                              distances.size(), labels.size());
    auto len = distances.size();
    size_t valid_cnt = 0;
    for (size_t i = 0; i < len; i++) {
        auto dist = distances[i];
        auto id = labels[i];
        if (distance_in_range(dist, radius, range_filter, is_ip)) {
            distances[valid_cnt] = dist;
            labels[valid_cnt] = id;
            valid_cnt++;
        }
    }
    if (valid_cnt != distances.size()) {
        distances.resize(valid_cnt);
        labels.resize(valid_cnt);
    }
}

void
GetRangeSearchResult(const std::vector<std::vector<float>>& result_distances,
                     const std::vector<std::vector<int64_t>>& result_labels, const bool is_ip, const int64_t nq,
                     const float radius, const float range_filter, float*& distances, int64_t*& labels, size_t*& lims) {
    KNOWHERE_THROW_IF_NOT_FMT(result_distances.size() == (size_t)nq, "result distances size %ld not equal to %" SCNd64,
                              result_distances.size(), nq);
    KNOWHERE_THROW_IF_NOT_FMT(result_labels.size() == (size_t)nq, "result labels size %ld not equal to %" SCNd64,
                              result_labels.size(), nq);

    lims = new size_t[nq + 1];
    lims[0] = 0;
    // all distances must be in range scope
    for (int64_t i = 0; i < nq; i++) {
        lims[i + 1] = lims[i] + result_distances[i].size();
    }

    size_t total_valid = lims[nq];
    LOG_KNOWHERE_DEBUG_ << "Range search: is_ip " << (is_ip ? "True" : "False") << ", radius " << radius
                        << ", range_filter " << range_filter << ", total result num " << total_valid;

    distances = new float[total_valid];
    labels = new int64_t[total_valid];

    for (auto i = 0; i < nq; i++) {
        std::copy_n(result_distances[i].data(), lims[i + 1] - lims[i], distances + lims[i]);
        std::copy_n(result_labels[i].data(), lims[i + 1] - lims[i], labels + lims[i]);
    }
}

}  // namespace knowhere
