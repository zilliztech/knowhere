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

/**
 * @file test_distance_datatypes.cc
 * @brief Unit tests for distance computations with different data types (fp16, bf16, int8)
 *
 * Tests cover:
 * - fp16 (float16) distance computations
 * - bf16 (bfloat16) distance computations
 * - int8 distance computations
 * - Cross-type consistency validation
 * - Precision loss expectations
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "knowhere/operands.h"
#include "simd/distances_ref.h"
#include "simd/hook.h"

namespace {

// Relative tolerance - fp16/bf16 have lower precision than fp32
constexpr float kFp16RelTolerance = 0.01f;   // 1% for fp16
constexpr float kBf16RelTolerance = 0.01f;   // 1% for bf16
constexpr float kInt8RelTolerance = 0.001f;  // int8 should be exact

// Helper to generate random float vector and convert to fp16
std::pair<std::vector<float>, std::vector<knowhere::fp16>>
GenRandomFp16Vector(size_t dim, std::mt19937& rng, float min_val = -10.0f, float max_val = 10.0f) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> fp32_vec(dim);
    std::vector<knowhere::fp16> fp16_vec(dim);

    for (size_t i = 0; i < dim; ++i) {
        fp32_vec[i] = dist(rng);
        fp16_vec[i] = knowhere::fp16(fp32_vec[i]);
    }
    return {fp32_vec, fp16_vec};
}

// Helper to generate random float vector and convert to bf16
std::pair<std::vector<float>, std::vector<knowhere::bf16>>
GenRandomBf16Vector(size_t dim, std::mt19937& rng, float min_val = -10.0f, float max_val = 10.0f) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> fp32_vec(dim);
    std::vector<knowhere::bf16> bf16_vec(dim);

    for (size_t i = 0; i < dim; ++i) {
        fp32_vec[i] = dist(rng);
        bf16_vec[i] = knowhere::bf16(fp32_vec[i]);
    }
    return {fp32_vec, bf16_vec};
}

// Helper to generate random int8 vector
std::vector<int8_t>
GenRandomInt8Vector(size_t dim, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(-128, 127);
    std::vector<int8_t> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = static_cast<int8_t>(dist(rng));
    }
    return vec;
}

// Compute fp32 reference distance from fp16 values (for comparison)
float
Fp16ToFp32InnerProduct(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        sum += static_cast<double>(static_cast<float>(x[i])) * static_cast<double>(static_cast<float>(y[i]));
    }
    return static_cast<float>(sum);
}

float
Fp16ToFp32L2Sqr(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double diff = static_cast<double>(static_cast<float>(x[i])) - static_cast<double>(static_cast<float>(y[i]));
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

float
Bf16ToFp32InnerProduct(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        sum += static_cast<double>(static_cast<float>(x[i])) * static_cast<double>(static_cast<float>(y[i]));
    }
    return static_cast<float>(sum);
}

float
Bf16ToFp32L2Sqr(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double diff = static_cast<double>(static_cast<float>(x[i])) - static_cast<double>(static_cast<float>(y[i]));
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

}  // namespace

// ==================== FP16 Distance Tests ====================

TEST_CASE("Test FP16 Inner Product", "[distance][fp16][IP]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128, 256);

    SECTION("Compare fp16_vec_inner_product_ref with manual computation") {
        for (int trial = 0; trial < 50; ++trial) {
            auto [fp32_x, fp16_x] = GenRandomFp16Vector(dim, rng);
            auto [fp32_y, fp16_y] = GenRandomFp16Vector(dim, rng);

            float result = faiss::fp16_vec_inner_product_ref(fp16_x.data(), fp16_y.data(), dim);
            float expected = Fp16ToFp32InnerProduct(fp16_x.data(), fp16_y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(expected, kFp16RelTolerance));
        }
    }

    SECTION("Known values") {
        // Simple case: [1, 2, 3] dot [4, 5, 6] = 32
        std::vector<knowhere::fp16> x = {knowhere::fp16(1.0f), knowhere::fp16(2.0f), knowhere::fp16(3.0f)};
        std::vector<knowhere::fp16> y = {knowhere::fp16(4.0f), knowhere::fp16(5.0f), knowhere::fp16(6.0f)};

        float result = faiss::fp16_vec_inner_product_ref(x.data(), y.data(), 3);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(32.0f, kFp16RelTolerance));
    }
}

TEST_CASE("Test FP16 L2 Squared Distance", "[distance][fp16][L2]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128, 256);

    SECTION("Compare fp16_vec_L2sqr_ref with manual computation") {
        for (int trial = 0; trial < 50; ++trial) {
            auto [fp32_x, fp16_x] = GenRandomFp16Vector(dim, rng);
            auto [fp32_y, fp16_y] = GenRandomFp16Vector(dim, rng);

            float result = faiss::fp16_vec_L2sqr_ref(fp16_x.data(), fp16_y.data(), dim);
            float expected = Fp16ToFp32L2Sqr(fp16_x.data(), fp16_y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(expected, kFp16RelTolerance));
        }
    }

    SECTION("Zero distance for identical vectors") {
        auto [fp32, fp16] = GenRandomFp16Vector(dim, rng);
        float result = faiss::fp16_vec_L2sqr_ref(fp16.data(), fp16.data(), dim);
        REQUIRE(result < 1e-5f);
    }
}

TEST_CASE("Test FP16 Norm L2 Squared", "[distance][fp16][norm]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128);

    SECTION("Compare with manual computation") {
        for (int trial = 0; trial < 50; ++trial) {
            auto [fp32, fp16] = GenRandomFp16Vector(dim, rng);

            float result = faiss::fp16_vec_norm_L2sqr_ref(fp16.data(), dim);

            // Manual norm computation
            double expected = 0.0;
            for (size_t i = 0; i < dim; ++i) {
                float val = static_cast<float>(fp16[i]);
                expected += static_cast<double>(val) * static_cast<double>(val);
            }

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(static_cast<float>(expected), kFp16RelTolerance));
        }
    }

    SECTION("Known value: [3, 4] -> norm^2 = 25") {
        std::vector<knowhere::fp16> x = {knowhere::fp16(3.0f), knowhere::fp16(4.0f)};
        float result = faiss::fp16_vec_norm_L2sqr_ref(x.data(), 2);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(25.0f, kFp16RelTolerance));
    }
}

TEST_CASE("Test FP16 Batch 4 Functions", "[distance][fp16][batch4]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128);

    auto [_, x] = GenRandomFp16Vector(dim, rng);
    auto [__, y0] = GenRandomFp16Vector(dim, rng);
    auto [___, y1] = GenRandomFp16Vector(dim, rng);
    auto [____, y2] = GenRandomFp16Vector(dim, rng);
    auto [_____, y3] = GenRandomFp16Vector(dim, rng);

    SECTION("L2sqr batch 4") {
        float dis0, dis1, dis2, dis3;
        faiss::fp16_vec_L2sqr_batch_4_ref(x.data(), y0.data(), y1.data(), y2.data(), y3.data(), dim, dis0, dis1, dis2,
                                          dis3);

        REQUIRE_THAT(
            dis0, Catch::Matchers::WithinRel(faiss::fp16_vec_L2sqr_ref(x.data(), y0.data(), dim), kFp16RelTolerance));
        REQUIRE_THAT(
            dis1, Catch::Matchers::WithinRel(faiss::fp16_vec_L2sqr_ref(x.data(), y1.data(), dim), kFp16RelTolerance));
        REQUIRE_THAT(
            dis2, Catch::Matchers::WithinRel(faiss::fp16_vec_L2sqr_ref(x.data(), y2.data(), dim), kFp16RelTolerance));
        REQUIRE_THAT(
            dis3, Catch::Matchers::WithinRel(faiss::fp16_vec_L2sqr_ref(x.data(), y3.data(), dim), kFp16RelTolerance));
    }

    SECTION("Inner product batch 4") {
        float dis0, dis1, dis2, dis3;
        faiss::fp16_vec_inner_product_batch_4_ref(x.data(), y0.data(), y1.data(), y2.data(), y3.data(), dim, dis0, dis1,
                                                  dis2, dis3);

        REQUIRE_THAT(dis0, Catch::Matchers::WithinRel(faiss::fp16_vec_inner_product_ref(x.data(), y0.data(), dim),
                                                      kFp16RelTolerance));
        REQUIRE_THAT(dis1, Catch::Matchers::WithinRel(faiss::fp16_vec_inner_product_ref(x.data(), y1.data(), dim),
                                                      kFp16RelTolerance));
        REQUIRE_THAT(dis2, Catch::Matchers::WithinRel(faiss::fp16_vec_inner_product_ref(x.data(), y2.data(), dim),
                                                      kFp16RelTolerance));
        REQUIRE_THAT(dis3, Catch::Matchers::WithinRel(faiss::fp16_vec_inner_product_ref(x.data(), y3.data(), dim),
                                                      kFp16RelTolerance));
    }
}

// ==================== BF16 Distance Tests ====================

TEST_CASE("Test BF16 Inner Product", "[distance][bf16][IP]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128, 256);

    SECTION("Compare bf16_vec_inner_product_ref with manual computation") {
        for (int trial = 0; trial < 50; ++trial) {
            auto [fp32_x, bf16_x] = GenRandomBf16Vector(dim, rng);
            auto [fp32_y, bf16_y] = GenRandomBf16Vector(dim, rng);

            float result = faiss::bf16_vec_inner_product_ref(bf16_x.data(), bf16_y.data(), dim);
            float expected = Bf16ToFp32InnerProduct(bf16_x.data(), bf16_y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(expected, kBf16RelTolerance));
        }
    }

    SECTION("Known values") {
        std::vector<knowhere::bf16> x = {knowhere::bf16(1.0f), knowhere::bf16(2.0f), knowhere::bf16(3.0f)};
        std::vector<knowhere::bf16> y = {knowhere::bf16(4.0f), knowhere::bf16(5.0f), knowhere::bf16(6.0f)};

        float result = faiss::bf16_vec_inner_product_ref(x.data(), y.data(), 3);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(32.0f, kBf16RelTolerance));
    }
}

TEST_CASE("Test BF16 L2 Squared Distance", "[distance][bf16][L2]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128, 256);

    SECTION("Compare bf16_vec_L2sqr_ref with manual computation") {
        for (int trial = 0; trial < 50; ++trial) {
            auto [fp32_x, bf16_x] = GenRandomBf16Vector(dim, rng);
            auto [fp32_y, bf16_y] = GenRandomBf16Vector(dim, rng);

            float result = faiss::bf16_vec_L2sqr_ref(bf16_x.data(), bf16_y.data(), dim);
            float expected = Bf16ToFp32L2Sqr(bf16_x.data(), bf16_y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(expected, kBf16RelTolerance));
        }
    }

    SECTION("Zero distance for identical vectors") {
        auto [fp32, bf16] = GenRandomBf16Vector(dim, rng);
        float result = faiss::bf16_vec_L2sqr_ref(bf16.data(), bf16.data(), dim);
        REQUIRE(result < 1e-5f);
    }
}

TEST_CASE("Test BF16 Norm L2 Squared", "[distance][bf16][norm]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128);

    SECTION("Compare with manual computation") {
        for (int trial = 0; trial < 50; ++trial) {
            auto [fp32, bf16] = GenRandomBf16Vector(dim, rng);

            float result = faiss::bf16_vec_norm_L2sqr_ref(bf16.data(), dim);

            double expected = 0.0;
            for (size_t i = 0; i < dim; ++i) {
                float val = static_cast<float>(bf16[i]);
                expected += static_cast<double>(val) * static_cast<double>(val);
            }

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(static_cast<float>(expected), kBf16RelTolerance));
        }
    }
}

TEST_CASE("Test BF16 Batch 4 Functions", "[distance][bf16][batch4]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128);

    auto [_, x] = GenRandomBf16Vector(dim, rng);
    auto [__, y0] = GenRandomBf16Vector(dim, rng);
    auto [___, y1] = GenRandomBf16Vector(dim, rng);
    auto [____, y2] = GenRandomBf16Vector(dim, rng);
    auto [_____, y3] = GenRandomBf16Vector(dim, rng);

    SECTION("L2sqr batch 4") {
        float dis0, dis1, dis2, dis3;
        faiss::bf16_vec_L2sqr_batch_4_ref(x.data(), y0.data(), y1.data(), y2.data(), y3.data(), dim, dis0, dis1, dis2,
                                          dis3);

        REQUIRE_THAT(
            dis0, Catch::Matchers::WithinRel(faiss::bf16_vec_L2sqr_ref(x.data(), y0.data(), dim), kBf16RelTolerance));
        REQUIRE_THAT(
            dis1, Catch::Matchers::WithinRel(faiss::bf16_vec_L2sqr_ref(x.data(), y1.data(), dim), kBf16RelTolerance));
        REQUIRE_THAT(
            dis2, Catch::Matchers::WithinRel(faiss::bf16_vec_L2sqr_ref(x.data(), y2.data(), dim), kBf16RelTolerance));
        REQUIRE_THAT(
            dis3, Catch::Matchers::WithinRel(faiss::bf16_vec_L2sqr_ref(x.data(), y3.data(), dim), kBf16RelTolerance));
    }

    SECTION("Inner product batch 4") {
        float dis0, dis1, dis2, dis3;
        faiss::bf16_vec_inner_product_batch_4_ref(x.data(), y0.data(), y1.data(), y2.data(), y3.data(), dim, dis0, dis1,
                                                  dis2, dis3);

        REQUIRE_THAT(dis0, Catch::Matchers::WithinRel(faiss::bf16_vec_inner_product_ref(x.data(), y0.data(), dim),
                                                      kBf16RelTolerance));
        REQUIRE_THAT(dis1, Catch::Matchers::WithinRel(faiss::bf16_vec_inner_product_ref(x.data(), y1.data(), dim),
                                                      kBf16RelTolerance));
        REQUIRE_THAT(dis2, Catch::Matchers::WithinRel(faiss::bf16_vec_inner_product_ref(x.data(), y2.data(), dim),
                                                      kBf16RelTolerance));
        REQUIRE_THAT(dis3, Catch::Matchers::WithinRel(faiss::bf16_vec_inner_product_ref(x.data(), y3.data(), dim),
                                                      kBf16RelTolerance));
    }
}

// ==================== Int8 Distance Tests ====================

TEST_CASE("Test Int8 Inner Product - Comprehensive", "[distance][int8][IP]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128, 256);

    SECTION("Compare with manual computation") {
        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomInt8Vector(dim, rng);
            auto y = GenRandomInt8Vector(dim, rng);

            float result = faiss::int8_vec_inner_product_ref(x.data(), y.data(), dim);

            // Manual computation with int64 to avoid overflow
            int64_t expected = 0;
            for (size_t i = 0; i < dim; ++i) {
                expected += static_cast<int64_t>(x[i]) * static_cast<int64_t>(y[i]);
            }

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(static_cast<float>(expected), kInt8RelTolerance));
        }
    }

    SECTION("Known values") {
        std::vector<int8_t> x = {1, 2, 3, 4};
        std::vector<int8_t> y = {5, 6, 7, 8};
        // IP = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70

        float result = faiss::int8_vec_inner_product_ref(x.data(), y.data(), 4);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(70.0f, kInt8RelTolerance));
    }

    SECTION("Negative values") {
        std::vector<int8_t> x = {-1, -2, -3, -4};
        std::vector<int8_t> y = {1, 2, 3, 4};
        // IP = -1 - 4 - 9 - 16 = -30

        float result = faiss::int8_vec_inner_product_ref(x.data(), y.data(), 4);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(-30.0f, kInt8RelTolerance));
    }
}

TEST_CASE("Test Int8 L2 Squared - Comprehensive", "[distance][int8][L2]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128, 256);

    SECTION("Compare with manual computation") {
        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomInt8Vector(dim, rng);
            auto y = GenRandomInt8Vector(dim, rng);

            float result = faiss::int8_vec_L2sqr_ref(x.data(), y.data(), dim);

            int64_t expected = 0;
            for (size_t i = 0; i < dim; ++i) {
                int64_t diff = static_cast<int64_t>(x[i]) - static_cast<int64_t>(y[i]);
                expected += diff * diff;
            }

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(static_cast<float>(expected), kInt8RelTolerance));
        }
    }

    SECTION("Zero distance for identical vectors") {
        auto x = GenRandomInt8Vector(dim, rng);
        float result = faiss::int8_vec_L2sqr_ref(x.data(), x.data(), dim);
        REQUIRE(result == 0.0f);
    }

    SECTION("Known values") {
        std::vector<int8_t> x = {1, 2, 3};
        std::vector<int8_t> y = {4, 5, 6};
        // L2^2 = 9 + 9 + 9 = 27

        float result = faiss::int8_vec_L2sqr_ref(x.data(), y.data(), 3);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(27.0f, kInt8RelTolerance));
    }
}

TEST_CASE("Test Int8 Norm L2 Squared", "[distance][int8][norm]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128);

    SECTION("Compare with manual computation") {
        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomInt8Vector(dim, rng);

            float result = faiss::int8_vec_norm_L2sqr_ref(x.data(), dim);

            int64_t expected = 0;
            for (size_t i = 0; i < dim; ++i) {
                expected += static_cast<int64_t>(x[i]) * static_cast<int64_t>(x[i]);
            }

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(static_cast<float>(expected), kInt8RelTolerance));
        }
    }

    SECTION("Known value: [3, 4] -> norm^2 = 25") {
        std::vector<int8_t> x = {3, 4};
        float result = faiss::int8_vec_norm_L2sqr_ref(x.data(), 2);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(25.0f, kInt8RelTolerance));
    }
}

TEST_CASE("Test Int8 Batch 4 Functions", "[distance][int8][batch4]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128);

    auto x = GenRandomInt8Vector(dim, rng);
    auto y0 = GenRandomInt8Vector(dim, rng);
    auto y1 = GenRandomInt8Vector(dim, rng);
    auto y2 = GenRandomInt8Vector(dim, rng);
    auto y3 = GenRandomInt8Vector(dim, rng);

    SECTION("L2sqr batch 4") {
        float dis0, dis1, dis2, dis3;
        faiss::int8_vec_L2sqr_batch_4_ref(x.data(), y0.data(), y1.data(), y2.data(), y3.data(), dim, dis0, dis1, dis2,
                                          dis3);

        REQUIRE_THAT(
            dis0, Catch::Matchers::WithinRel(faiss::int8_vec_L2sqr_ref(x.data(), y0.data(), dim), kInt8RelTolerance));
        REQUIRE_THAT(
            dis1, Catch::Matchers::WithinRel(faiss::int8_vec_L2sqr_ref(x.data(), y1.data(), dim), kInt8RelTolerance));
        REQUIRE_THAT(
            dis2, Catch::Matchers::WithinRel(faiss::int8_vec_L2sqr_ref(x.data(), y2.data(), dim), kInt8RelTolerance));
        REQUIRE_THAT(
            dis3, Catch::Matchers::WithinRel(faiss::int8_vec_L2sqr_ref(x.data(), y3.data(), dim), kInt8RelTolerance));
    }

    SECTION("Inner product batch 4") {
        float dis0, dis1, dis2, dis3;
        faiss::int8_vec_inner_product_batch_4_ref(x.data(), y0.data(), y1.data(), y2.data(), y3.data(), dim, dis0, dis1,
                                                  dis2, dis3);

        REQUIRE_THAT(dis0, Catch::Matchers::WithinRel(faiss::int8_vec_inner_product_ref(x.data(), y0.data(), dim),
                                                      kInt8RelTolerance));
        REQUIRE_THAT(dis1, Catch::Matchers::WithinRel(faiss::int8_vec_inner_product_ref(x.data(), y1.data(), dim),
                                                      kInt8RelTolerance));
        REQUIRE_THAT(dis2, Catch::Matchers::WithinRel(faiss::int8_vec_inner_product_ref(x.data(), y2.data(), dim),
                                                      kInt8RelTolerance));
        REQUIRE_THAT(dis3, Catch::Matchers::WithinRel(faiss::int8_vec_inner_product_ref(x.data(), y3.data(), dim),
                                                      kInt8RelTolerance));
    }
}

// ==================== Cross-Type Consistency Tests ====================

TEST_CASE("Test Cross-Type Distance Consistency", "[distance][cross_type]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    auto dim = GENERATE(64, 128);

    SECTION("fp32 vs fp16 vs bf16 should produce similar results") {
        // Generate fp32 vectors with small values to minimize precision loss
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

        for (int trial = 0; trial < 20; ++trial) {
            std::vector<float> fp32_x(dim), fp32_y(dim);
            std::vector<knowhere::fp16> fp16_x(dim), fp16_y(dim);
            std::vector<knowhere::bf16> bf16_x(dim), bf16_y(dim);

            for (size_t i = 0; i < dim; ++i) {
                fp32_x[i] = dist(rng);
                fp32_y[i] = dist(rng);
                fp16_x[i] = knowhere::fp16(fp32_x[i]);
                fp16_y[i] = knowhere::fp16(fp32_y[i]);
                bf16_x[i] = knowhere::bf16(fp32_x[i]);
                bf16_y[i] = knowhere::bf16(fp32_y[i]);
            }

            float ip_fp32 = faiss::fvec_inner_product_ref(fp32_x.data(), fp32_y.data(), dim);
            float ip_fp16 = faiss::fp16_vec_inner_product_ref(fp16_x.data(), fp16_y.data(), dim);
            float ip_bf16 = faiss::bf16_vec_inner_product_ref(bf16_x.data(), bf16_y.data(), dim);

            // fp16 and bf16 should be within 10% of fp32 for small values
            // bf16 has less mantissa precision so needs more tolerance
            if (std::abs(ip_fp32) > 1.0f) {
                REQUIRE_THAT(ip_fp16, Catch::Matchers::WithinRel(ip_fp32, 0.10f));
                REQUIRE_THAT(ip_bf16, Catch::Matchers::WithinRel(ip_fp32, 0.10f));
            }
        }
    }
}
