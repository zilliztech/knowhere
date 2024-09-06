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
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "simd/distances_ref.h"
#include "simd/hook.h"
#if defined(__x86_64__)
#include "simd/distances_avx.h"
#include "simd/distances_avx512.h"
#include "simd/distances_sse.h"
#endif

#if defined(__ARM_NEON)
#include "simd/distances_neon.h"
#endif

#include "utils.h"
template <typename DataType>
std::unique_ptr<DataType[]>
GenRandomVector(int dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib(0.0, 100.0);
    auto x = std::make_unique<DataType[]>(dim);
    for (int i = 0; i < dim; ++i) x[i] = (DataType)distrib(rng);
    return x;
}

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

TEST_CASE("Test fp16 distance", "[fp16]") {
    using Catch::Approx;
    auto dim = GENERATE(as<size_t>{}, 1, 2, 10, 69, 128, 141, 510, 1024);

    auto x = GenRandomVector<knowhere::fp16>(dim, 11);
    auto y = GenRandomVector<knowhere::fp16>(dim, 22);
    auto ref_l2_dist = faiss::fp16_vec_L2sqr_ref(x.get(), y.get(), dim);
    auto ref_ip_dist = faiss::fp16_vec_inner_product_ref(x.get(), y.get(), dim);
    auto ref_norm_l2_dist = faiss::fp16_vec_norm_L2sqr_ref(x.get(), dim);
#if defined(__ARM_NEON)
    // neon
    REQUIRE_THAT(faiss::fp16_vec_L2sqr_neon(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_l2_dist, 0.001f));
    REQUIRE_THAT(faiss::fp16_vec_inner_product_neon(x.get(), y.get(), dim),
                 Catch::Matchers::WithinRel(ref_ip_dist, 0.001f));
    REQUIRE_THAT(faiss::fp16_vec_norm_L2sqr_neon(x.get(), dim), Catch::Matchers::WithinRel(ref_norm_l2_dist, 0.001f));
#endif
#if defined(__x86_64__)
    if (faiss::cpu_support_avx2()) {
        REQUIRE_THAT(faiss::fp16_vec_L2sqr_avx(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_l2_dist, 0.001f));
        REQUIRE_THAT(faiss::fp16_vec_inner_product_avx(x.get(), y.get(), dim),
                     Catch::Matchers::WithinRel(ref_ip_dist, 0.001f));
        REQUIRE_THAT(faiss::fp16_vec_norm_L2sqr_avx(x.get(), dim),
                     Catch::Matchers::WithinRel(ref_norm_l2_dist, 0.001f));
    }
    if (faiss::cpu_support_avx512()) {
        REQUIRE_THAT(faiss::fp16_vec_L2sqr_avx512(x.get(), y.get(), dim),
                     Catch::Matchers::WithinRel(ref_l2_dist, 0.001f));
        REQUIRE_THAT(faiss::fp16_vec_inner_product_avx512(x.get(), y.get(), dim),
                     Catch::Matchers::WithinRel(ref_ip_dist, 0.001f));
        REQUIRE_THAT(faiss::fp16_vec_norm_L2sqr_avx512(x.get(), dim),
                     Catch::Matchers::WithinRel(ref_norm_l2_dist, 0.001f));
    }
#endif
}

TEST_CASE("Test bf16 distance", "[bf16]") {
    using Catch::Approx;

    auto dim = GENERATE(as<size_t>{}, 1, 2, 10, 69, 128, 141, 510, 1024);

    auto x = GenRandomVector<knowhere::bf16>(dim, 11);
    auto y = GenRandomVector<knowhere::bf16>(dim, 22);
    auto ref_l2_dist = faiss::bf16_vec_L2sqr_ref(x.get(), y.get(), dim);
    auto ref_ip_dist = faiss::bf16_vec_inner_product_ref(x.get(), y.get(), dim);
    auto ref_norm_l2_dist = faiss::bf16_vec_norm_L2sqr_ref(x.get(), dim);
#if defined(__ARM_NEON)
    // neon
    REQUIRE_THAT(faiss::bf16_vec_L2sqr_neon(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_l2_dist, 0.001f));
    REQUIRE_THAT(faiss::bf16_vec_inner_product_neon(x.get(), y.get(), dim),
                 Catch::Matchers::WithinRel(ref_ip_dist, 0.001f));
    REQUIRE_THAT(faiss::bf16_vec_norm_L2sqr_neon(x.get(), dim), Catch::Matchers::WithinRel(ref_norm_l2_dist, 0.001f));
#endif
#if defined(__x86_64__)
    if (faiss::cpu_support_sse4_2()) {
        REQUIRE_THAT(faiss::bf16_vec_L2sqr_sse(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_l2_dist, 0.001f));
        REQUIRE_THAT(faiss::bf16_vec_inner_product_sse(x.get(), y.get(), dim),
                     Catch::Matchers::WithinRel(ref_ip_dist, 0.001f));
        REQUIRE_THAT(faiss::bf16_vec_norm_L2sqr_sse(x.get(), dim),
                     Catch::Matchers::WithinRel(ref_norm_l2_dist, 0.001f));
    }
    if (faiss::cpu_support_avx2()) {
        REQUIRE_THAT(faiss::bf16_vec_L2sqr_avx(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_l2_dist, 0.001f));
        REQUIRE_THAT(faiss::bf16_vec_inner_product_avx(x.get(), y.get(), dim),
                     Catch::Matchers::WithinRel(ref_ip_dist, 0.001f));
        REQUIRE_THAT(faiss::bf16_vec_norm_L2sqr_avx(x.get(), dim),
                     Catch::Matchers::WithinRel(ref_norm_l2_dist, 0.001f));
    }
    if (faiss::cpu_support_avx512()) {
        REQUIRE_THAT(faiss::bf16_vec_L2sqr_avx512(x.get(), y.get(), dim),
                     Catch::Matchers::WithinRel(ref_l2_dist, 0.001f));
        REQUIRE_THAT(faiss::bf16_vec_inner_product_avx512(x.get(), y.get(), dim),
                     Catch::Matchers::WithinRel(ref_ip_dist, 0.001f));
        REQUIRE_THAT(faiss::bf16_vec_norm_L2sqr_avx512(x.get(), dim),
                     Catch::Matchers::WithinRel(ref_norm_l2_dist, 0.001f));
    }
#endif
}
