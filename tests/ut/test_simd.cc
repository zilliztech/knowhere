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
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "utils.h"

TEST_CASE("Test BruteForce Search SIMD", "[bf]") {
    using Catch::Approx;

    const int64_t nb = 1000;
    const int64_t nq = 10;
    const int64_t dim = 127;
    const int64_t k = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = CopyDataSet(train_ds, nq);

    knowhere::Json conf = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, k},
    };

    auto test_search_with_simd = [&](knowhere::KnowhereConfig::SimdType simd_type) {
        knowhere::KnowhereConfig::SetSimdType(simd_type);
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
        REQUIRE(gt.has_value());
        auto gt_ids = gt.value()->GetIds();
        auto gt_dist = gt.value()->GetDistance();

        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(gt_ids[i * k] == i);
            if (metric == knowhere::metric::L2) {
                REQUIRE(gt_dist[i * k] == 0);
            } else {
                REQUIRE(std::abs(gt_dist[i * k] - 1.0) < 0.00001);
            }
        }
    };

    for (auto simd_type : {knowhere::KnowhereConfig::SimdType::AVX512, knowhere::KnowhereConfig::SimdType::AVX2,
                           knowhere::KnowhereConfig::SimdType::SSE4_2, knowhere::KnowhereConfig::SimdType::GENERIC,
                           knowhere::KnowhereConfig::SimdType::AUTO}) {
        test_search_with_simd(simd_type);
    }
}

TEST_CASE("Test PQ Search SIMD", "[pq]") {
    using Catch::Approx;

    const int64_t nb = 1000;
    const int64_t nq = 10;
    const int64_t dim = 128;
    const int64_t k = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = CopyDataSet(train_ds, nq);

    knowhere::Json conf = {
        {knowhere::meta::DIM, dim},        {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, k},
        {knowhere::indexparam::NLIST, 16}, {knowhere::indexparam::NPROBE, 8},     {knowhere::indexparam::NBITS, 8},
    };

    auto test_search_with_simd = [&](const int64_t m, knowhere::KnowhereConfig::SimdType simd_type) {
        conf[knowhere::indexparam::M] = m;

        knowhere::KnowhereConfig::SetSimdType(simd_type);
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
        REQUIRE(gt.has_value());
        auto gt_ids = gt.value()->GetIds();
        auto gt_dist = gt.value()->GetDistance();

        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(gt_ids[i * k] == i);
            if (metric == knowhere::metric::L2) {
                REQUIRE(gt_dist[i * k] == 0);
            } else {
                REQUIRE(std::abs(gt_dist[i * k] - 1.0) < 0.00001);
            }
        }

        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, version)
                       .value();
        REQUIRE(idx.Build(train_ds, conf) == knowhere::Status::success);
        auto res = idx.Search(query_ds, conf, nullptr);
        REQUIRE(res.has_value());
        float recall = GetKNNRecall(*gt.value(), *res.value());
        REQUIRE(recall > 0.2);
    };

    for (auto simd_type : {knowhere::KnowhereConfig::SimdType::GENERIC, knowhere::KnowhereConfig::SimdType::AUTO}) {
        for (int64_t m : {8, 16, 32, 64, 128}) {
            test_search_with_simd(m, simd_type);
        }
    }
}
