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

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/utils/Heap.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/utils.h"
#include "simd/hook.h"
#include "utils.h"

template <typename T>
void
check_search(const knowhere::DataSetPtr train_ds, const knowhere::DataSetPtr query_ds, const int64_t k,
             const knowhere::MetricType metric, const knowhere::Json& conf) {
    auto base = knowhere::ConvertToDataTypeIfNeeded<T>(train_ds);
    auto query = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds);

    auto res = knowhere::BruteForce::Search<T>(base, query, conf, nullptr);
    auto nq = query_ds->GetRows();
    REQUIRE(res.has_value());
    auto ids = res.value()->GetIds();
    auto dist = res.value()->GetDistance();
    for (int64_t i = 0; i < nq; i++) {
        REQUIRE(ids[i * k] == i);
        if (metric == knowhere::metric::L2) {
            REQUIRE(dist[i * k] == 0);
        } else {
            REQUIRE(std::abs(dist[i * k] - 1.0) < 0.00001);
        }
    }
}

template <typename T>
void
check_search_with_buf(const knowhere::DataSetPtr train_ds, const knowhere::DataSetPtr query_ds, const int64_t k,
                      const knowhere::MetricType metric, const knowhere::Json& conf) {
    auto nq = query_ds->GetRows();
    auto ids = new int64_t[nq * k];
    auto dist = new float[nq * k];

    auto base = knowhere::ConvertToDataTypeIfNeeded<T>(train_ds);
    auto query = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds);

    auto res = knowhere::BruteForce::SearchWithBuf<T>(base, query, ids, dist, conf, nullptr);
    REQUIRE(res == knowhere::Status::success);
    for (int64_t i = 0; i < nq; i++) {
        REQUIRE(ids[i * k] == i);
        if (metric == knowhere::metric::L2) {
            REQUIRE(dist[i * k] == 0);
        } else {
            REQUIRE(std::abs(dist[i * k] - 1.0) < 0.00001);
        }
    }
    delete[] ids;
    delete[] dist;
}

template <typename T>
void
check_range_search(const knowhere::DataSetPtr train_ds, const knowhere::DataSetPtr query_ds, const int64_t k,
                   const knowhere::MetricType metric, const knowhere::Json& conf) {
    auto base = knowhere::ConvertToDataTypeIfNeeded<T>(train_ds);
    auto query = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds);

    auto res = knowhere::BruteForce::RangeSearch<T>(base, query, conf, nullptr);
    REQUIRE(res.has_value());
    auto ids = res.value()->GetIds();
    auto dist = res.value()->GetDistance();
    auto lims = res.value()->GetLims();
    auto nq = query_ds->GetRows();
    for (int64_t i = 0; i < nq; i++) {
        REQUIRE(lims[i] == (size_t)i);
        REQUIRE(ids[i] == i);
        if (metric == knowhere::metric::L2) {
            REQUIRE(dist[i] == 0);
        } else {
            REQUIRE(std::abs(dist[i] - 1.0) < 0.00001);
        }
    }
}

template <typename T>
void
check_search_with_out_ids(const uint64_t nb, const uint64_t nq, const uint64_t dim, const int64_t k,
                          const knowhere::MetricType metric, const knowhere::Json& conf) {
    auto total_train_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(nb, dim));
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(nq, dim));
    std::vector<int64_t> block_prefix = {0, 111, 333, 500, 555, 666, 888, 1000};

    // generate filter id and data
    auto filter_bits = GenerateBitsetWithRandomTbitsSet(nb, 100);
    knowhere::BitsetView bitset(filter_bits.data(), nb);

    std::vector<float> dis(nq * k, std::numeric_limits<float>::quiet_NaN());
    std::vector<int64_t> ids(nq * k, -1);
    if (metric == knowhere::metric::L2) {
        faiss::float_maxheap_array_t heaps{nq, (size_t)k, ids.data(), dis.data()};
        heaps.heapify();
        for (size_t i = 0; i < block_prefix.size() - 1; i++) {
            auto begin_id = block_prefix[i];
            auto end_id = block_prefix[i + 1];
            auto blk_rows = end_id - begin_id;
            auto tensor = (const T*)total_train_ds->GetTensor() + dim * begin_id;
            auto blk_train_ds = knowhere::GenDataSet(blk_rows, dim, tensor, begin_id);
            auto partial_v = knowhere::BruteForce::Search<T>(blk_train_ds, query_ds, conf, bitset);
            REQUIRE(partial_v.has_value());
            auto partial_res = partial_v.value();
            heaps.addn_with_ids(k, partial_res->GetDistance(), partial_res->GetIds(), k, 0, nq);
        }
        heaps.reorder();
    } else {
        faiss::float_minheap_array_t heaps{nq, (size_t)k, ids.data(), dis.data()};
        heaps.heapify();
        for (size_t i = 0; i < block_prefix.size() - 1; i++) {
            auto begin_id = block_prefix[i];
            auto end_id = block_prefix[i + 1];
            auto blk_rows = end_id - begin_id;
            auto tensor = (const T*)total_train_ds->GetTensor() + dim * begin_id;
            auto blk_train_ds = knowhere::GenDataSet(blk_rows, dim, tensor, begin_id);
            auto partial_v = knowhere::BruteForce::Search<T>(blk_train_ds, query_ds, conf, bitset);
            REQUIRE(partial_v.has_value());
            auto partial_res = partial_v.value();
            heaps.addn_with_ids(k, partial_res->GetDistance(), partial_res->GetIds(), k, 0, nq);
        }
        heaps.reorder();
    }

    auto gt = knowhere::BruteForce::Search<T>(total_train_ds, query_ds, conf, bitset);
    auto gt_ids = gt.value()->GetIds();
    const float* gt_dis = gt.value()->GetDistance();
    for (size_t i = 0; i < nq * k; i++) {
        REQUIRE(gt_ids[i] == ids[i]);
        REQUIRE(GetRelativeLoss(gt_dis[i], dis[i]) < 0.00001);
    }
}

TEST_CASE("Test Brute Force", "[float vector]") {
    using Catch::Approx;

    const int64_t nb = 1000;
    const int64_t nq = 10;
    const int64_t dim = 128;
    const int64_t k = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = CopyDataSet(train_ds, nq);

    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
        {knowhere::meta::RADIUS, knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99},
    };

    SECTION("Test Search") {
        check_search<knowhere::fp32>(train_ds, query_ds, k, metric, conf);
        check_search<knowhere::fp16>(train_ds, query_ds, k, metric, conf);
        check_search<knowhere::bf16>(train_ds, query_ds, k, metric, conf);
        check_search<knowhere::int8>(train_ds, query_ds, k, metric, conf);
    }

    SECTION("Test Search With Buf") {
        check_search_with_buf<knowhere::fp32>(train_ds, query_ds, k, metric, conf);
        check_search_with_buf<knowhere::fp16>(train_ds, query_ds, k, metric, conf);
        check_search_with_buf<knowhere::bf16>(train_ds, query_ds, k, metric, conf);
        check_search_with_buf<knowhere::int8>(train_ds, query_ds, k, metric, conf);
    }

    SECTION("Test Range Search") {
        check_range_search<knowhere::fp32>(train_ds, query_ds, k, metric, conf);
        check_range_search<knowhere::fp16>(train_ds, query_ds, k, metric, conf);
        check_range_search<knowhere::bf16>(train_ds, query_ds, k, metric, conf);
        check_range_search<knowhere::int8>(train_ds, query_ds, k, metric, conf);
    }
}

TEST_CASE("Test Brute Force", "[binary vector]") {
    using Catch::Approx;

    const int64_t nb = 1000;
    const int64_t nq = 10;
    const int64_t dim = 1024;
    const int64_t k = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::HAMMING, knowhere::metric::JACCARD,
                           knowhere::metric::SUPERSTRUCTURE, knowhere::metric::SUBSTRUCTURE);

    const auto train_ds = GenBinDataSet(nb, dim);
    const auto query_ds = CopyBinDataSet(train_ds, nq);

    std::unordered_map<std::string, float> radius_map = {
        {knowhere::metric::HAMMING, 1.0},
        {knowhere::metric::JACCARD, 0.1},
    };
    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
    };

    SECTION("Test Search") {
        auto res = knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, conf, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(ids[i * k] == i);
            REQUIRE(dist[i * k] == 0);
        }
    }

    SECTION("Test Search With Buf") {
        auto ids = new int64_t[nq * k];
        auto dist = new float[nq * k];
        auto res = knowhere::BruteForce::SearchWithBuf<knowhere::bin1>(train_ds, query_ds, ids, dist, conf, nullptr);
        REQUIRE(res == knowhere::Status::success);
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(ids[i * k] == i);
            REQUIRE(dist[i * k] == 0);
        }
        delete[] ids;
        delete[] dist;
    }

    SECTION("Test Range Search") {
        if (metric == knowhere::metric::SUPERSTRUCTURE || metric == knowhere::metric::SUBSTRUCTURE) {
            return;
        }

        // set radius for different metric type
        auto cfg = conf;
        cfg[knowhere::meta::RADIUS] = radius_map[metric];

        auto res = knowhere::BruteForce::RangeSearch<knowhere::bin1>(train_ds, query_ds, cfg, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();
        auto lims = res.value()->GetLims();
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(lims[i] == (size_t)i);
            REQUIRE(ids[i] == i);
            REQUIRE(dist[i] == 0);
        }
    }
}

TEST_CASE("Test Brute Force with input ids", "[float vector]") {
    using Catch::Approx;
    const int64_t nb = 1000;
    const int64_t nq = 10;
    const int64_t dim = 128;
    const int64_t k = 10;
    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    const knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
    };
    check_search_with_out_ids<knowhere::fp32>(nb, nq, dim, k, metric, conf);
    check_search_with_out_ids<knowhere::fp16>(nb, nq, dim, k, metric, conf);
    check_search_with_out_ids<knowhere::bf16>(nb, nq, dim, k, metric, conf);
    check_search_with_out_ids<knowhere::int8>(nb, nq, dim, k, metric, conf);
}
