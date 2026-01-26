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
 * @file test_distance_unit.cc
 * @brief Unit tests for distance computation functions
 *
 * This file contains comprehensive unit tests for:
 * - Basic distance computations (L2, IP, L1, Linf)
 * - Batch distance computations
 * - Multi-data type distance computations (fp16, bf16, int8)
 * - Edge cases and boundary conditions
 * - SIMD alignment validation
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "knowhere/operands.h"
#include "simd/distances_ref.h"
#include "simd/hook.h"

namespace {

// Relative tolerance for floating point comparisons
// Note: SIMD implementations may have slight precision differences from scalar
constexpr float kRelTolerance = 0.01f;  // 1% tolerance for SIMD vs reference
// Absolute tolerance for near-zero comparisons
constexpr float kAbsTolerance = 1e-5f;

// Helper to generate random float vector
std::vector<float>
GenRandomFloatVector(size_t dim, std::mt19937& rng, float min_val = -100.0f, float max_val = 100.0f) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
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

// Manual computation of L2 squared distance
float
ManualL2Sqr(const float* x, const float* y, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        double diff = static_cast<double>(x[i]) - static_cast<double>(y[i]);
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

// Manual computation of inner product
float
ManualIP(const float* x, const float* y, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        sum += static_cast<double>(x[i]) * static_cast<double>(y[i]);
    }
    return static_cast<float>(sum);
}

// Manual computation of L1 distance
float
ManualL1(const float* x, const float* y, size_t d) {
    double sum = 0.0;
    for (size_t i = 0; i < d; ++i) {
        sum += std::abs(static_cast<double>(x[i]) - static_cast<double>(y[i]));
    }
    return static_cast<float>(sum);
}

// Manual computation of Linf distance
float
ManualLinf(const float* x, const float* y, size_t d) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = std::abs(x[i] - y[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

}  // namespace

// ==================== Basic Float Distance Tests ====================

TEST_CASE("Test L2 Squared Distance - Basic", "[distance][L2]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    SECTION("Known values") {
        // x = [1, 2, 3], y = [4, 5, 6]
        // L2^2 = (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 5.0f, 6.0f};

        float result = faiss::fvec_L2sqr(x.data(), y.data(), 3);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(27.0f, kRelTolerance));
    }

    SECTION("Zero distance (identical vectors)") {
        std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float result = faiss::fvec_L2sqr(x.data(), x.data(), 5);
        REQUIRE(result < kAbsTolerance);
    }

    SECTION("Random vectors - compare with manual") {
        auto dim = GENERATE(1, 4, 7, 16, 33, 128, 256, 1000);

        for (int trial = 0; trial < 10; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);

            float result = faiss::fvec_L2sqr(x.data(), y.data(), dim);
            float expected = ManualL2Sqr(x.data(), y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(expected, kRelTolerance));
        }
    }

    SECTION("Compare optimized with reference") {
        auto dim = GENERATE(1, 4, 7, 16, 33, 128, 256);

        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);

            float result = faiss::fvec_L2sqr(x.data(), y.data(), dim);
            float ref_result = faiss::fvec_L2sqr_ref(x.data(), y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(ref_result, kRelTolerance));
        }
    }
}

TEST_CASE("Test Inner Product - Basic", "[distance][IP]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    SECTION("Known values") {
        // x = [1, 2, 3], y = [4, 5, 6]
        // IP = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 5.0f, 6.0f};

        float result = faiss::fvec_inner_product(x.data(), y.data(), 3);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(32.0f, kRelTolerance));
    }

    SECTION("Orthogonal vectors") {
        // [1, 0, 0] dot [0, 1, 0] = 0
        std::vector<float> x = {1.0f, 0.0f, 0.0f};
        std::vector<float> y = {0.0f, 1.0f, 0.0f};

        float result = faiss::fvec_inner_product(x.data(), y.data(), 3);
        REQUIRE(std::abs(result) < kAbsTolerance);
    }

    SECTION("Random vectors - compare with manual") {
        auto dim = GENERATE(1, 4, 7, 16, 33, 128, 256);

        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);

            float result = faiss::fvec_inner_product(x.data(), y.data(), dim);
            float expected = ManualIP(x.data(), y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(expected, kRelTolerance));
        }
    }

    SECTION("Compare optimized with reference") {
        auto dim = GENERATE(1, 4, 7, 16, 33, 128, 256);

        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);

            float result = faiss::fvec_inner_product(x.data(), y.data(), dim);
            float ref_result = faiss::fvec_inner_product_ref(x.data(), y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(ref_result, kRelTolerance));
        }
    }
}

TEST_CASE("Test L1 Distance", "[distance][L1]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    SECTION("Known values") {
        // x = [1, 2, 3], y = [4, 1, 7]
        // L1 = |1-4| + |2-1| + |3-7| = 3 + 1 + 4 = 8
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 1.0f, 7.0f};

        float result = faiss::fvec_L1(x.data(), y.data(), 3);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(8.0f, kRelTolerance));
    }

    SECTION("Compare optimized with reference") {
        auto dim = GENERATE(1, 4, 7, 16, 33, 128);

        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);

            float result = faiss::fvec_L1(x.data(), y.data(), dim);
            float ref_result = faiss::fvec_L1_ref(x.data(), y.data(), dim);
            float manual = ManualL1(x.data(), y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(ref_result, kRelTolerance));
            REQUIRE_THAT(result, Catch::Matchers::WithinRel(manual, kRelTolerance));
        }
    }
}

TEST_CASE("Test Linf Distance", "[distance][Linf]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    SECTION("Known values") {
        // x = [1, 2, 3], y = [4, 1, 10]
        // Linf = max(|1-4|, |2-1|, |3-10|) = max(3, 1, 7) = 7
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 1.0f, 10.0f};

        float result = faiss::fvec_Linf(x.data(), y.data(), 3);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(7.0f, kRelTolerance));
    }

    SECTION("Compare optimized with reference") {
        auto dim = GENERATE(1, 4, 7, 16, 33, 128);

        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);

            float result = faiss::fvec_Linf(x.data(), y.data(), dim);
            float ref_result = faiss::fvec_Linf_ref(x.data(), y.data(), dim);
            float manual = ManualLinf(x.data(), y.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(ref_result, kRelTolerance));
            REQUIRE_THAT(result, Catch::Matchers::WithinRel(manual, kRelTolerance));
        }
    }
}

TEST_CASE("Test Vector Norm L2 Squared", "[distance][norm]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    SECTION("Known values") {
        // x = [3, 4], norm^2 = 9 + 16 = 25
        std::vector<float> x = {3.0f, 4.0f};

        float result = faiss::fvec_norm_L2sqr(x.data(), 2);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(25.0f, kRelTolerance));
    }

    SECTION("Unit vector") {
        std::vector<float> x = {1.0f, 0.0f, 0.0f};
        float result = faiss::fvec_norm_L2sqr(x.data(), 3);
        REQUIRE_THAT(result, Catch::Matchers::WithinRel(1.0f, kRelTolerance));
    }

    SECTION("Compare optimized with reference") {
        auto dim = GENERATE(1, 4, 7, 16, 33, 128, 256);

        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);

            float result = faiss::fvec_norm_L2sqr(x.data(), dim);
            float ref_result = faiss::fvec_norm_L2sqr_ref(x.data(), dim);

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(ref_result, kRelTolerance));
        }
    }
}

// ==================== Batch Distance Tests ====================

TEST_CASE("Test Batch L2 Squared Distance (ny vectors)", "[distance][batch]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    auto dim = GENERATE(16, 64, 128);
    auto ny = GENERATE(10, 50, 100);

    SECTION("Compare batch with individual computations") {
        auto x = GenRandomFloatVector(dim, rng);
        std::vector<float> y(dim * ny);
        for (size_t i = 0; i < ny * dim; ++i) {
            y[i] = GenRandomFloatVector(1, rng)[0];
        }

        std::vector<float> batch_results(ny);
        faiss::fvec_L2sqr_ny_ref(batch_results.data(), x.data(), y.data(), dim, ny);

        // Compare with individual computations
        for (size_t i = 0; i < ny; ++i) {
            float individual = faiss::fvec_L2sqr_ref(x.data(), y.data() + i * dim, dim);
            REQUIRE_THAT(batch_results[i], Catch::Matchers::WithinRel(individual, kRelTolerance));
        }
    }
}

TEST_CASE("Test Batch 4 Distance Functions", "[distance][batch4]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    auto dim = GENERATE(16, 64, 128);

    auto x = GenRandomFloatVector(dim, rng);
    auto y0 = GenRandomFloatVector(dim, rng);
    auto y1 = GenRandomFloatVector(dim, rng);
    auto y2 = GenRandomFloatVector(dim, rng);
    auto y3 = GenRandomFloatVector(dim, rng);

    SECTION("L2sqr batch 4") {
        float dis0, dis1, dis2, dis3;
        faiss::fvec_L2sqr_batch_4_ref(x.data(), y0.data(), y1.data(), y2.data(), y3.data(), dim, dis0, dis1, dis2,
                                      dis3);

        REQUIRE_THAT(dis0, Catch::Matchers::WithinRel(faiss::fvec_L2sqr_ref(x.data(), y0.data(), dim), kRelTolerance));
        REQUIRE_THAT(dis1, Catch::Matchers::WithinRel(faiss::fvec_L2sqr_ref(x.data(), y1.data(), dim), kRelTolerance));
        REQUIRE_THAT(dis2, Catch::Matchers::WithinRel(faiss::fvec_L2sqr_ref(x.data(), y2.data(), dim), kRelTolerance));
        REQUIRE_THAT(dis3, Catch::Matchers::WithinRel(faiss::fvec_L2sqr_ref(x.data(), y3.data(), dim), kRelTolerance));
    }

    SECTION("Inner product batch 4") {
        float dis0, dis1, dis2, dis3;
        faiss::fvec_inner_product_batch_4_ref(x.data(), y0.data(), y1.data(), y2.data(), y3.data(), dim, dis0, dis1,
                                              dis2, dis3);

        REQUIRE_THAT(
            dis0, Catch::Matchers::WithinRel(faiss::fvec_inner_product_ref(x.data(), y0.data(), dim), kRelTolerance));
        REQUIRE_THAT(
            dis1, Catch::Matchers::WithinRel(faiss::fvec_inner_product_ref(x.data(), y1.data(), dim), kRelTolerance));
        REQUIRE_THAT(
            dis2, Catch::Matchers::WithinRel(faiss::fvec_inner_product_ref(x.data(), y2.data(), dim), kRelTolerance));
        REQUIRE_THAT(
            dis3, Catch::Matchers::WithinRel(faiss::fvec_inner_product_ref(x.data(), y3.data(), dim), kRelTolerance));
    }
}

// ==================== Int8 Distance Tests ====================

TEST_CASE("Test Int8 Distance Functions", "[distance][int8]") {
    std::mt19937 rng(42);

    auto dim = GENERATE(16, 64, 128);

    SECTION("Int8 Inner Product") {
        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomInt8Vector(dim, rng);
            auto y = GenRandomInt8Vector(dim, rng);

            float result = faiss::int8_vec_inner_product_ref(x.data(), y.data(), dim);

            // Manual computation
            int64_t expected = 0;
            for (size_t i = 0; i < dim; ++i) {
                expected += static_cast<int64_t>(x[i]) * static_cast<int64_t>(y[i]);
            }

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(static_cast<float>(expected), kRelTolerance));
        }
    }

    SECTION("Int8 L2 Squared") {
        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomInt8Vector(dim, rng);
            auto y = GenRandomInt8Vector(dim, rng);

            float result = faiss::int8_vec_L2sqr_ref(x.data(), y.data(), dim);

            // Manual computation
            int64_t expected = 0;
            for (size_t i = 0; i < dim; ++i) {
                int64_t diff = static_cast<int64_t>(x[i]) - static_cast<int64_t>(y[i]);
                expected += diff * diff;
            }

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(static_cast<float>(expected), kRelTolerance));
        }
    }

    SECTION("Int8 Norm L2 Squared") {
        for (int trial = 0; trial < 100; ++trial) {
            auto x = GenRandomInt8Vector(dim, rng);

            float result = faiss::int8_vec_norm_L2sqr_ref(x.data(), dim);

            // Manual computation
            int64_t expected = 0;
            for (size_t i = 0; i < dim; ++i) {
                expected += static_cast<int64_t>(x[i]) * static_cast<int64_t>(x[i]);
            }

            REQUIRE_THAT(result, Catch::Matchers::WithinRel(static_cast<float>(expected), kRelTolerance));
        }
    }
}

// ==================== Edge Cases ====================

TEST_CASE("Test Distance Edge Cases", "[distance][edge]") {
    std::string ins;
    faiss::fvec_hook(ins);

    SECTION("Dimension = 1") {
        std::vector<float> x = {5.0f};
        std::vector<float> y = {3.0f};

        REQUIRE_THAT(faiss::fvec_L2sqr(x.data(), y.data(), 1), Catch::Matchers::WithinRel(4.0f, kRelTolerance));
        REQUIRE_THAT(faiss::fvec_inner_product(x.data(), y.data(), 1),
                     Catch::Matchers::WithinRel(15.0f, kRelTolerance));
        REQUIRE_THAT(faiss::fvec_L1(x.data(), y.data(), 1), Catch::Matchers::WithinRel(2.0f, kRelTolerance));
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

        float l2_result = faiss::fvec_L2sqr(x.data(), y.data(), 3);
        REQUIRE(l2_result < kAbsTolerance);  // Same vectors

        float ip_result = faiss::fvec_inner_product(x.data(), y.data(), 3);
        REQUIRE_THAT(ip_result, Catch::Matchers::WithinRel(3e12f, kRelTolerance));
    }

    SECTION("Mixed positive and negative values") {
        std::vector<float> x = {-1.0f, 2.0f, -3.0f, 4.0f};
        std::vector<float> y = {1.0f, -2.0f, 3.0f, -4.0f};

        // L2^2 = 4 + 16 + 36 + 64 = 120
        REQUIRE_THAT(faiss::fvec_L2sqr(x.data(), y.data(), 4), Catch::Matchers::WithinRel(120.0f, kRelTolerance));

        // IP = -1 - 4 - 9 - 16 = -30
        REQUIRE_THAT(faiss::fvec_inner_product(x.data(), y.data(), 4),
                     Catch::Matchers::WithinRel(-30.0f, kRelTolerance));
    }
}

// ==================== SIMD Boundary Conditions ====================

TEST_CASE("Test SIMD Boundary Dimensions", "[distance][simd]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    // Test dimensions around typical SIMD register sizes
    // AVX: 8 floats (256 bits), AVX-512: 16 floats (512 bits)
    auto dim = GENERATE(1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65);

    SECTION("L2sqr with SIMD boundary dimensions") {
        for (int trial = 0; trial < 50; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);

            float optimized = faiss::fvec_L2sqr(x.data(), y.data(), dim);
            float reference = faiss::fvec_L2sqr_ref(x.data(), y.data(), dim);

            REQUIRE_THAT(optimized, Catch::Matchers::WithinRel(reference, kRelTolerance));
        }
    }

    SECTION("Inner product with SIMD boundary dimensions") {
        for (int trial = 0; trial < 50; ++trial) {
            auto x = GenRandomFloatVector(dim, rng);
            auto y = GenRandomFloatVector(dim, rng);

            float optimized = faiss::fvec_inner_product(x.data(), y.data(), dim);
            float reference = faiss::fvec_inner_product_ref(x.data(), y.data(), dim);

            REQUIRE_THAT(optimized, Catch::Matchers::WithinRel(reference, kRelTolerance));
        }
    }
}

// ==================== Madd Functions ====================

TEST_CASE("Test Madd Functions", "[distance][madd]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    auto dim = GENERATE(16, 64, 128);

    SECTION("fvec_madd") {
        for (int trial = 0; trial < 50; ++trial) {
            auto a = GenRandomFloatVector(dim, rng);
            auto b = GenRandomFloatVector(dim, rng);
            std::vector<float> c(dim);
            std::vector<float> c_ref(dim);
            float bf = GenRandomFloatVector(1, rng)[0];

            faiss::fvec_madd(dim, a.data(), bf, b.data(), c.data());
            faiss::fvec_madd_ref(dim, a.data(), bf, b.data(), c_ref.data());

            for (size_t i = 0; i < dim; ++i) {
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(c_ref[i], kRelTolerance));
                // Also verify against manual: c = a + bf * b
                float expected = a[i] + bf * b[i];
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(expected, kRelTolerance));
            }
        }
    }

    SECTION("fvec_madd_and_argmin") {
        for (int trial = 0; trial < 50; ++trial) {
            auto a = GenRandomFloatVector(dim, rng);
            auto b = GenRandomFloatVector(dim, rng);
            std::vector<float> c(dim);
            std::vector<float> c_ref(dim);
            float bf = GenRandomFloatVector(1, rng)[0];

            int argmin = faiss::fvec_madd_and_argmin(dim, a.data(), bf, b.data(), c.data());
            int argmin_ref = faiss::fvec_madd_and_argmin_ref(dim, a.data(), bf, b.data(), c_ref.data());

            REQUIRE(argmin == argmin_ref);

            for (size_t i = 0; i < dim; ++i) {
                REQUIRE_THAT(c[i], Catch::Matchers::WithinRel(c_ref[i], kRelTolerance));
            }

            // Verify argmin is correct
            float min_val = c[0];
            int expected_argmin = 0;
            for (size_t i = 1; i < dim; ++i) {
                if (c[i] < min_val) {
                    min_val = c[i];
                    expected_argmin = i;
                }
            }
            REQUIRE(argmin == expected_argmin);
        }
    }
}
