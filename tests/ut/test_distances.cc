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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <random>

#include "simd/distances_ref.h"
#include "simd/hook.h"

namespace {
constexpr float kRelTolerance = 0.01f;
constexpr float kAbsTolerance = 1e-5f;

std::vector<float>
GenRandomFloatVector(size_t dim, std::mt19937& rng, float min_val = -100.0f, float max_val = 100.0f) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}
}  // namespace

TEST_CASE("Test Distance Compute", "[distance]") {
    std::mt19937 rng;
    std::uniform_int_distribution<> distrib(1, 100000);
    std::uniform_real_distribution<float> fill_distrib(1, 1000000);
    std::string ins;
    faiss::cppcontrib::knowhere::fvec_hook(ins);

    using std::make_tuple;
    SECTION("Test Binary Distance Compute") {
        typedef float (*FUNC)(const float*, const float*, size_t);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::cppcontrib::knowhere::fvec_L1, faiss::cppcontrib::knowhere::fvec_L1_ref),
            make_tuple(faiss::cppcontrib::knowhere::fvec_L2sqr, faiss::cppcontrib::knowhere::fvec_L2sqr_ref),
            make_tuple(faiss::cppcontrib::knowhere::fvec_Linf, faiss::cppcontrib::knowhere::fvec_Linf_ref),
            make_tuple(faiss::cppcontrib::knowhere::fvec_inner_product,
                       faiss::cppcontrib::knowhere::fvec_inner_product_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }
            REQUIRE_THAT(real_func(a.data(), b.data(), len),
                         Catch::Matchers::WithinRel(gold_func(a.data(), b.data(), len), 0.001f));
        }
    }

    SECTION("Test Normal Compute") {
        typedef float (*FUNC)(const float*, size_t);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::cppcontrib::knowhere::fvec_norm_L2sqr, faiss::cppcontrib::knowhere::fvec_norm_L2sqr_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
            }
            REQUIRE_THAT(real_func(a.data(), len), Catch::Matchers::WithinRel(gold_func(a.data(), len), 0.001f));
        }
    }

    SECTION("Test Madd and Argmin") {
        typedef int (*FUNC)(size_t, const float*, float, const float*, float*);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::cppcontrib::knowhere::fvec_madd_and_argmin,
                       faiss::cppcontrib::knowhere::fvec_madd_and_argmin_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }

            std::vector<float> c(len);
            std::vector<float> c_gold(len);
            float pf = fill_distrib(rng);
            REQUIRE(real_func(len, a.data(), pf, b.data(), c.data()) ==
                    gold_func(len, a.data(), pf, b.data(), c_gold.data()));

            for (int i = 0; i < len; ++i) {
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(c_gold[i], 0.001f));
            }
        }
    }

    SECTION("Test Madd") {
        typedef void (*FUNC)(size_t, const float*, float, const float*, float*);
        auto [real_func, gold_func] = GENERATE(table<FUNC, FUNC>({
            make_tuple(faiss::cppcontrib::knowhere::fvec_madd, faiss::cppcontrib::knowhere::fvec_madd_ref),
        }));

        for (int i = 0; i < 1000; ++i) {
            CAPTURE(i);
            auto len = distrib(rng);
            std::vector<float> a(len);
            std::vector<float> b(len);
            for (int i = 0; i < len; ++i) {
                a[i] = fill_distrib(rng);
                b[i] = fill_distrib(rng);
            }

            std::vector<float> c(len);
            std::vector<float> c_gold(len);
            float pf = fill_distrib(rng);
            real_func(len, a.data(), pf, b.data(), c.data());
            gold_func(len, a.data(), pf, b.data(), c_gold.data());

            for (int i = 0; i < len; ++i) {
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(c_gold[i], 0.001f));
            }
        }
    }
}

TEST_CASE("Test Distance Known Values", "[distance][known_values]") {
    std::string ins;
    faiss::fvec_hook(ins);

    SECTION("L2 squared distance") {
        // x = [1, 2, 3], y = [4, 5, 6]
        // L2^2 = (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 5.0f, 6.0f};
        REQUIRE_THAT(faiss::fvec_L2sqr(x.data(), y.data(), 3), Catch::Matchers::WithinRel(27.0f, kRelTolerance));
    }

    SECTION("Inner product") {
        // x = [1, 2, 3], y = [4, 5, 6]
        // IP = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 5.0f, 6.0f};
        REQUIRE_THAT(faiss::fvec_inner_product(x.data(), y.data(), 3), Catch::Matchers::WithinRel(32.0f, kRelTolerance));
    }

    SECTION("L1 distance") {
        // x = [1, 2, 3], y = [4, 1, 7]
        // L1 = |1-4| + |2-1| + |3-7| = 3 + 1 + 4 = 8
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 1.0f, 7.0f};
        REQUIRE_THAT(faiss::fvec_L1(x.data(), y.data(), 3), Catch::Matchers::WithinRel(8.0f, kRelTolerance));
    }

    SECTION("Linf distance") {
        // x = [1, 2, 3], y = [4, 1, 10]
        // Linf = max(|1-4|, |2-1|, |3-10|) = max(3, 1, 7) = 7
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 1.0f, 10.0f};
        REQUIRE_THAT(faiss::fvec_Linf(x.data(), y.data(), 3), Catch::Matchers::WithinRel(7.0f, kRelTolerance));
    }

    SECTION("Norm L2 squared") {
        // x = [3, 4], norm^2 = 9 + 16 = 25
        std::vector<float> x = {3.0f, 4.0f};
        REQUIRE_THAT(faiss::fvec_norm_L2sqr(x.data(), 2), Catch::Matchers::WithinRel(25.0f, kRelTolerance));
    }
}

TEST_CASE("Test Distance Edge Cases", "[distance][edge]") {
    std::string ins;
    faiss::fvec_hook(ins);

    SECTION("Dimension = 1") {
        std::vector<float> x = {5.0f};
        std::vector<float> y = {3.0f};
        REQUIRE_THAT(faiss::fvec_L2sqr(x.data(), y.data(), 1), Catch::Matchers::WithinRel(4.0f, kRelTolerance));
        REQUIRE_THAT(faiss::fvec_inner_product(x.data(), y.data(), 1), Catch::Matchers::WithinRel(15.0f, kRelTolerance));
        REQUIRE_THAT(faiss::fvec_L1(x.data(), y.data(), 1), Catch::Matchers::WithinRel(2.0f, kRelTolerance));
    }

    SECTION("Zero distance (identical vectors)") {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        REQUIRE(faiss::fvec_L2sqr(x.data(), x.data(), 5) < kAbsTolerance);
    }

    SECTION("Orthogonal vectors") {
        std::vector<float> x = {1.0f, 0.0f, 0.0f};
        std::vector<float> y = {0.0f, 1.0f, 0.0f};
        REQUIRE(std::abs(faiss::fvec_inner_product(x.data(), y.data(), 3)) < kAbsTolerance);
    }

    SECTION("Zero vectors") {
        std::vector<float> x(128, 0.0f);
        std::vector<float> y(128, 0.0f);
        REQUIRE(faiss::fvec_L2sqr(x.data(), y.data(), 128) < kAbsTolerance);
        REQUIRE(std::abs(faiss::fvec_inner_product(x.data(), y.data(), 128)) < kAbsTolerance);
    }

    SECTION("Large values") {
        std::vector<float> x = {1e6f, 1e6f, 1e6f};
        std::vector<float> y = {1e6f, 1e6f, 1e6f};
        REQUIRE(faiss::fvec_L2sqr(x.data(), y.data(), 3) < kAbsTolerance);
        REQUIRE_THAT(faiss::fvec_inner_product(x.data(), y.data(), 3), Catch::Matchers::WithinRel(3e12f, kRelTolerance));
    }

    SECTION("Mixed positive and negative values") {
        std::vector<float> x = {-1.0f, 2.0f, -3.0f, 4.0f};
        std::vector<float> y = {1.0f, -2.0f, 3.0f, -4.0f};
        // L2^2 = 4 + 16 + 36 + 64 = 120
        REQUIRE_THAT(faiss::fvec_L2sqr(x.data(), y.data(), 4), Catch::Matchers::WithinRel(120.0f, kRelTolerance));
        // IP = -1 - 4 - 9 - 16 = -30
        REQUIRE_THAT(faiss::fvec_inner_product(x.data(), y.data(), 4), Catch::Matchers::WithinRel(-30.0f, kRelTolerance));
    }
}

TEST_CASE("Test SIMD Boundary Dimensions", "[distance][simd]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    // Test dimensions around typical SIMD register sizes
    auto dim = GENERATE(1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65);

    SECTION("L2sqr with SIMD boundary dimensions") {
        for (int trial = 0; trial < 20; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);
            float optimized = faiss::fvec_L2sqr(x.data(), y.data(), dim);
            float reference = faiss::fvec_L2sqr_ref(x.data(), y.data(), dim);
            REQUIRE_THAT(optimized, Catch::Matchers::WithinRel(reference, kRelTolerance));
        }
    }

    SECTION("Inner product with SIMD boundary dimensions") {
        for (int trial = 0; trial < 20; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);
            float optimized = faiss::fvec_inner_product(x.data(), y.data(), dim);
            float reference = faiss::fvec_inner_product_ref(x.data(), y.data(), dim);
            REQUIRE_THAT(optimized, Catch::Matchers::WithinRel(reference, kRelTolerance));
        }
    }
}
