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

#include "catch2/catch_test_macros.hpp"
#include "faiss/impl/AuxIndexStructures.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/range_util.h"
#include "utils.h"

TEST_CASE("Test distance_in_range", "[range search]") {
    bool is_ip = false;
    float radius = 1.0f;
    float range_filter = 0.0f;
    REQUIRE(knowhere::distance_in_range(0.0, radius, range_filter, is_ip) == true);
    REQUIRE(knowhere::distance_in_range(1.0, radius, range_filter, is_ip) == false);
    REQUIRE(knowhere::distance_in_range(0.314, radius, range_filter, is_ip) == true);
    REQUIRE(knowhere::distance_in_range(-1.0, radius, range_filter, is_ip) == false);
    REQUIRE(knowhere::distance_in_range(1.23, radius, range_filter, is_ip) == false);

    is_ip = true;
    radius = 0.0f;
    range_filter = 1.0f;
    REQUIRE(knowhere::distance_in_range(0.0, radius, range_filter, is_ip) == false);
    REQUIRE(knowhere::distance_in_range(1.0, radius, range_filter, is_ip) == true);
    REQUIRE(knowhere::distance_in_range(0.314, radius, range_filter, is_ip) == true);
    REQUIRE(knowhere::distance_in_range(-1.0, radius, range_filter, is_ip) == false);
    REQUIRE(knowhere::distance_in_range(1.23, radius, range_filter, is_ip) == false);
}

namespace {
void
GenRangeSearchResult(std::vector<std::vector<int64_t>>& labels, std::vector<std::vector<float>>& distances,
                     const int64_t nq, const int64_t label_min, const int64_t label_max, const float distance_min,
                     const float distances_max, int seed = 42) {
    std::mt19937 e(seed);
    std::uniform_int_distribution<> uniform_num(0, 10);
    std::uniform_int_distribution<> uniform_label(label_min, label_max);
    std::uniform_real_distribution<> uniform_distance(distance_min, distances_max);

    labels.resize(nq);
    distances.resize(nq);
    for (int64_t i = 0; i < nq; i++) {
        int64_t num = uniform_num(e);
        for (int64_t j = 0; j < num; j++) {
            auto id = uniform_label(e);
            auto dis = uniform_distance(e);
            labels[i].push_back(id);
            distances[i].push_back(dis);
        }
    }
}

size_t
CountValidRangeSearchResult(const std::vector<std::vector<float>>& distances, const float radius,
                            const float range_filter, const bool is_ip) {
    int64_t valid = 0;
    for (size_t i = 0; i < distances.size(); i++) {
        for (size_t j = 0; j < distances[i].size(); j++) {
            if (knowhere::distance_in_range(distances[i][j], radius, range_filter, is_ip)) {
                valid++;
            }
        }
    }
    return valid;
}
}  // namespace

TEST_CASE("Test GetRangeSearchResult for HNSW/DiskANN", "[range search]") {
    const int64_t nq = 10;
    const int64_t label_min = 0, label_max = 10000;
    const float dist_min = 0.0, dist_max = 100.0;
    std::vector<std::vector<int64_t>> gen_labels;
    std::vector<std::vector<float>> gen_distances;

    GenRangeSearchResult(gen_labels, gen_distances, nq, label_min, label_max, dist_min, dist_max);

    auto GetRangeSearchResultL =
        [](std::vector<std::vector<float>>& result_distances, std::vector<std::vector<int64_t>>& result_labels,
           const bool is_ip, const int64_t nq, const float radius, const float range_filter) -> knowhere::DataSetPtr {
        for (auto i = 0; i < nq; i++) {
            knowhere::FilterRangeSearchResultForOneNq(result_distances[i], result_labels[i], is_ip, radius,
                                                      range_filter);
        }
        auto range_search_result =
            knowhere::GetRangeSearchResult(result_distances, result_labels, is_ip, nq, radius, range_filter);
        return knowhere::GenResultDataSet(nq, std::move(range_search_result));
    };

    std::vector<std::tuple<float, float>> test_sets = {
        std::make_tuple(-10.0, -1.0), std::make_tuple(-10.0, 0.0),   std::make_tuple(-10.0, 50.0),
        std::make_tuple(0.0, 50.0),   std::make_tuple(0.0, 100.0),   std::make_tuple(50.0, 100.0),
        std::make_tuple(50.0, 200.0), std::make_tuple(100.0, 200.0),
    };

    for (auto& item : test_sets) {
        for (bool is_ip : {true, false}) {
            float radius = is_ip ? std::get<0>(item) : std::get<1>(item);
            float range_filter = is_ip ? std::get<1>(item) : std::get<0>(item);
            std::vector<std::vector<int64_t>> bak_labels = gen_labels;
            std::vector<std::vector<float>> bak_distances = gen_distances;
            auto result = GetRangeSearchResultL(bak_distances, bak_labels, is_ip, nq, radius, range_filter);
            REQUIRE(result->GetLims()[nq] == CountValidRangeSearchResult(gen_distances, radius, range_filter, is_ip));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
#if 0
namespace {
void
GenRangeSearchResult(faiss::RangeSearchResult& res, const int64_t nq, const int64_t label_min, const int64_t label_max,
                     const float distance_min, const float distances_max, int seed = 42) {
    std::mt19937 e(seed);
    std::uniform_int_distribution<> uniform_num(0, 10);
    std::uniform_int_distribution<> uniform_label(label_min, label_max);
    std::uniform_real_distribution<> uniform_distance(distance_min, distances_max);

    assert(res.nq == (size_t)nq);
    for (int64_t i = 0; i < nq; i++) {
        int64_t num = uniform_num(e);
        res.lims[i + 1] = res.lims[i] + num;
    }

    res.labels = new int64_t[res.lims[nq]];
    res.distances = new float[res.lims[nq]];
    for (int64_t i = 0; i < nq; i++) {
        for (size_t j = res.lims[i]; j < res.lims[i + 1]; j++) {
            auto id = uniform_label(e);
            auto dis = uniform_distance(e);
            res.labels[j] = id;
            res.distances[j] = dis;
        }
    }
}

size_t
CountValidRangeSearchResult(const float* distances, const size_t* lims, const int64_t nq, const float radius,
                            const float range_filter, const bool is_ip) {
    int64_t valid = 0;
    for (int64_t i = 0; i < nq; i++) {
        for (size_t j = lims[i]; j < lims[i + 1]; j++) {
            if (knowhere::distance_in_range(distances[j], radius, range_filter, is_ip)) {
                valid++;
            }
        }
    }
    return valid;
}
}  // namespace
#endif
