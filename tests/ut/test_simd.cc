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

#include <cmath>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "knowhere/comp/knowhere_config.h"
#include "simd/distances_ref.h"
#include "simd/hook.h"
#include "utils.h"

template <typename DataType>
std::unique_ptr<DataType[]>
GenRandomVector(int dim, int rows, int seed, int range = 128) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib(-1.0 * range, 1.0 * (range - 1));
    auto x = std::make_unique<DataType[]>(rows * dim);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < dim; j++) {
            x[i * dim + j] = (DataType)distrib(rng);
        }
    }
    return x;
}

template <typename DataType>
std::unique_ptr<DataType[]>
ConvertVector(float* data, int dim, int rows) {
    auto x = std::make_unique<DataType[]>(rows * dim);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < dim; j++) {
            x[i * dim + j] = (DataType)data[i * dim + j];
        }
    }
    return x;
}
TEST_CASE("Test minhash function") {
    auto simd_type = knowhere::KnowhereConfig::SimdType::AVX512;
    knowhere::KnowhereConfig::SetSimdType(simd_type);
    constexpr size_t seed = 111;
    SECTION("test binary search function with comprehensive corner cases") {
        auto size = GENERATE(as<size_t>{}, 1, 2, 4, 7, 8, 12, 14, 16, 21, 28, 32, 35, 42, 49, 56, 64, 111, 128, 256,
                             1000, 2221, 5555);
        auto iteration = GENERATE(0, 1, 2, 3, 4);
        auto seed = 12345 + iteration * 1000;
        auto x = GenRandomVector<uint64_t>(size, 1, seed, 100000);
        std::sort(x.get(), x.get() + size);
        auto key = x[1000 % size];
        CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, key), faiss::u64_binary_search_eq_ref(x.get(), size, key));
        CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, key), faiss::u64_binary_search_ge_ref(x.get(), size, key));

        uint64_t first_key = x[0];
        CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, first_key),
                 faiss::u64_binary_search_eq_ref(x.get(), size, first_key));
        CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, first_key),
                 faiss::u64_binary_search_ge_ref(x.get(), size, first_key));

        uint64_t last_key = x[size - 1];
        CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, last_key),
                 faiss::u64_binary_search_eq_ref(x.get(), size, last_key));
        CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, last_key),
                 faiss::u64_binary_search_ge_ref(x.get(), size, last_key));

        uint64_t mid_key = x[size / 2];
        CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, mid_key),
                 faiss::u64_binary_search_eq_ref(x.get(), size, mid_key));
        CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, mid_key),
                 faiss::u64_binary_search_ge_ref(x.get(), size, mid_key));

        if (x[0] > 0) {
            uint64_t smaller_key = x[0] - 1;
            CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, smaller_key),
                     faiss::u64_binary_search_eq_ref(x.get(), size, smaller_key));
            CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, smaller_key),
                     faiss::u64_binary_search_ge_ref(x.get(), size, smaller_key));
        }

        uint64_t larger_key = x[size - 1] + 1;
        CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, larger_key),
                 faiss::u64_binary_search_eq_ref(x.get(), size, larger_key));
        CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, larger_key),
                 faiss::u64_binary_search_ge_ref(x.get(), size, larger_key));

        if (size > 1) {
            for (size_t i = 0; i < size - 1; ++i) {
                if (x[i + 1] > x[i] + 1) {
                    uint64_t gap_key = x[i] + 1;
                    CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, gap_key),
                             faiss::u64_binary_search_eq_ref(x.get(), size, gap_key));
                    CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, gap_key),
                             faiss::u64_binary_search_ge_ref(x.get(), size, gap_key));
                    break;
                }
            }
        }
        if (size >= 8) {
            uint64_t simd_boundary_key = x[7];
            CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, simd_boundary_key),
                     faiss::u64_binary_search_eq_ref(x.get(), size, simd_boundary_key));
            CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, simd_boundary_key),
                     faiss::u64_binary_search_ge_ref(x.get(), size, simd_boundary_key));
        }

        if (size >= 16) {
            uint64_t double_simd_key = x[15];
            CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, double_simd_key),
                     faiss::u64_binary_search_eq_ref(x.get(), size, double_simd_key));
            CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, double_simd_key),
                     faiss::u64_binary_search_ge_ref(x.get(), size, double_simd_key));
        }

        uint64_t zero_key = 0;
        CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, zero_key),
                 faiss::u64_binary_search_eq_ref(x.get(), size, zero_key));
        CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, zero_key),
                 faiss::u64_binary_search_ge_ref(x.get(), size, zero_key));

        uint64_t max_key = UINT64_MAX;
        CHECK_EQ(faiss::u64_binary_search_eq(x.get(), size, max_key),
                 faiss::u64_binary_search_eq_ref(x.get(), size, max_key));
        CHECK_EQ(faiss::u64_binary_search_ge(x.get(), size, max_key),
                 faiss::u64_binary_search_ge_ref(x.get(), size, max_key));
    }

    SECTION("test binary search edge cases") {
        uint64_t* empty_arr = nullptr;
        CHECK_EQ(faiss::u64_binary_search_eq(empty_arr, 0, 42), faiss::u64_binary_search_eq_ref(empty_arr, 0, 42));
        CHECK_EQ(faiss::u64_binary_search_ge(empty_arr, 0, 42), faiss::u64_binary_search_ge_ref(empty_arr, 0, 42));

        uint64_t single_arr[] = {100};
        CHECK_EQ(faiss::u64_binary_search_eq(single_arr, 1, 100), faiss::u64_binary_search_eq_ref(single_arr, 1, 100));
        CHECK_EQ(faiss::u64_binary_search_eq(single_arr, 1, 99), faiss::u64_binary_search_eq_ref(single_arr, 1, 99));
        CHECK_EQ(faiss::u64_binary_search_eq(single_arr, 1, 101), faiss::u64_binary_search_eq_ref(single_arr, 1, 101));

        CHECK_EQ(faiss::u64_binary_search_ge(single_arr, 1, 100), faiss::u64_binary_search_ge_ref(single_arr, 1, 100));
        CHECK_EQ(faiss::u64_binary_search_ge(single_arr, 1, 99), faiss::u64_binary_search_ge_ref(single_arr, 1, 99));
        CHECK_EQ(faiss::u64_binary_search_ge(single_arr, 1, 101), faiss::u64_binary_search_ge_ref(single_arr, 1, 101));

        uint64_t identical_arr[] = {50, 50, 50, 50, 50, 50, 50, 50, 50, 50};
        size_t identical_size = sizeof(identical_arr) / sizeof(identical_arr[0]);

        CHECK_EQ(faiss::u64_binary_search_eq(identical_arr, identical_size, 50),
                 faiss::u64_binary_search_eq_ref(identical_arr, identical_size, 50));
        CHECK_EQ(faiss::u64_binary_search_eq(identical_arr, identical_size, 49),
                 faiss::u64_binary_search_eq_ref(identical_arr, identical_size, 49));
        CHECK_EQ(faiss::u64_binary_search_eq(identical_arr, identical_size, 51),
                 faiss::u64_binary_search_eq_ref(identical_arr, identical_size, 51));

        CHECK_EQ(faiss::u64_binary_search_ge(identical_arr, identical_size, 50),
                 faiss::u64_binary_search_ge_ref(identical_arr, identical_size, 50));
        CHECK_EQ(faiss::u64_binary_search_ge(identical_arr, identical_size, 49),
                 faiss::u64_binary_search_ge_ref(identical_arr, identical_size, 49));
        CHECK_EQ(faiss::u64_binary_search_ge(identical_arr, identical_size, 51),
                 faiss::u64_binary_search_ge_ref(identical_arr, identical_size, 51));

        uint64_t strict_inc[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25};
        size_t strict_size = sizeof(strict_inc) / sizeof(strict_inc[0]);

        CHECK_EQ(faiss::u64_binary_search_eq(strict_inc, strict_size, 1),
                 faiss::u64_binary_search_eq_ref(strict_inc, strict_size, 1));
        CHECK_EQ(faiss::u64_binary_search_eq(strict_inc, strict_size, 25),
                 faiss::u64_binary_search_eq_ref(strict_inc, strict_size, 25));
        CHECK_EQ(faiss::u64_binary_search_eq(strict_inc, strict_size, 13),
                 faiss::u64_binary_search_eq_ref(strict_inc, strict_size, 13));

        CHECK_EQ(faiss::u64_binary_search_eq(strict_inc, strict_size, 2),
                 faiss::u64_binary_search_eq_ref(strict_inc, strict_size, 2));
        CHECK_EQ(faiss::u64_binary_search_eq(strict_inc, strict_size, 14),
                 faiss::u64_binary_search_eq_ref(strict_inc, strict_size, 14));

        CHECK_EQ(faiss::u64_binary_search_ge(strict_inc, strict_size, 2),
                 faiss::u64_binary_search_ge_ref(strict_inc, strict_size, 2));
        CHECK_EQ(faiss::u64_binary_search_ge(strict_inc, strict_size, 14),
                 faiss::u64_binary_search_ge_ref(strict_inc, strict_size, 14));

        uint64_t many_dups[] = {1, 1, 1, 5, 5, 5, 5, 5, 10, 10, 15, 15, 15};
        size_t dups_size = sizeof(many_dups) / sizeof(many_dups[0]);

        CHECK_EQ(faiss::u64_binary_search_eq(many_dups, dups_size, 1),
                 faiss::u64_binary_search_eq_ref(many_dups, dups_size, 1));
        CHECK_EQ(faiss::u64_binary_search_eq(many_dups, dups_size, 5),
                 faiss::u64_binary_search_eq_ref(many_dups, dups_size, 5));
        CHECK_EQ(faiss::u64_binary_search_eq(many_dups, dups_size, 10),
                 faiss::u64_binary_search_eq_ref(many_dups, dups_size, 10));
        CHECK_EQ(faiss::u64_binary_search_eq(many_dups, dups_size, 15),
                 faiss::u64_binary_search_eq_ref(many_dups, dups_size, 15));

        CHECK_EQ(faiss::u64_binary_search_ge(many_dups, dups_size, 1),
                 faiss::u64_binary_search_ge_ref(many_dups, dups_size, 1));
        CHECK_EQ(faiss::u64_binary_search_ge(many_dups, dups_size, 5),
                 faiss::u64_binary_search_ge_ref(many_dups, dups_size, 5));
        CHECK_EQ(faiss::u64_binary_search_ge(many_dups, dups_size, 3),
                 faiss::u64_binary_search_ge_ref(many_dups, dups_size, 3));
        CHECK_EQ(faiss::u64_binary_search_ge(many_dups, dups_size, 12),
                 faiss::u64_binary_search_ge_ref(many_dups, dups_size, 12));
    }
    SECTION("test minhash distance") {
        auto dim = GENERATE(as<size_t>{}, 1, 2, 4, 7, 8, 12, 14, 16, 21, 28, 32, 35, 42, 49, 56, 64, 128, 256);
        auto u64_x = GenRandomVector<uint64_t>(dim, 4, seed);
        auto u64_y = GenRandomVector<uint64_t>(dim, 1, seed + 222);
        auto u32_x = GenRandomVector<uint32_t>(dim, 4, seed);
        auto u32_y = GenRandomVector<uint32_t>(dim, 1, seed + 222);
        float res_dis[4], gt_ids[4];
        CHECK_EQ(faiss::u64_jaccard_distance((const char*)u64_x.get(), (const char*)u64_x.get(), dim, 8), 1.0);
        CHECK_EQ(faiss::u64_jaccard_distance((const char*)u64_x.get(), (const char*)u64_y.get(), dim, 8),
                 faiss::u64_jaccard_distance_ref((const char*)u64_x.get(), (const char*)u64_y.get(), dim, 8));
        CHECK_EQ(faiss::u32_jaccard_distance((const char*)u32_x.get(), (const char*)u32_x.get(), dim, 4), 1.0);
        CHECK_EQ(faiss::u32_jaccard_distance((const char*)u32_x.get(), (const char*)u32_y.get(), dim, 4),
                 faiss::u32_jaccard_distance_ref((const char*)u32_x.get(), (const char*)u32_y.get(), dim, 4));
        faiss::u32_jaccard_distance_batch_4(
            (const char*)(&u32_y[0]), (const char*)(&u32_x[0]), (const char*)(&u32_x[1]), (const char*)(&u32_x[2]),
            (const char*)(&u32_x[3]), dim, 4, res_dis[0], res_dis[1], res_dis[2], res_dis[3]);
        faiss::u32_jaccard_distance_batch_4_ref(
            (const char*)(&u32_y[0]), (const char*)(&u32_x[0]), (const char*)(&u32_x[1]), (const char*)(&u32_x[2]),
            (const char*)(&u32_x[3]), dim, 4, gt_ids[0], gt_ids[1], gt_ids[2], gt_ids[3]);
        CHECK_EQ(res_dis[0], gt_ids[0]);
        CHECK_EQ(res_dis[1], gt_ids[1]);
        CHECK_EQ(res_dis[2], gt_ids[2]);
        CHECK_EQ(res_dis[3], gt_ids[3]);
        faiss::u64_jaccard_distance_batch_4(
            (const char*)(&u64_y[0]), (const char*)(&u64_x[0]), (const char*)(&u64_x[1]), (const char*)(&u64_x[2]),
            (const char*)(&u64_x[3]), dim, 8, res_dis[0], res_dis[1], res_dis[2], res_dis[3]);
        faiss::u64_jaccard_distance_batch_4_ref(
            (const char*)(&u64_y[0]), (const char*)(&u64_x[0]), (const char*)(&u64_x[1]), (const char*)(&u64_x[2]),
            (const char*)(&u64_x[3]), dim, 8, gt_ids[0], gt_ids[1], gt_ids[2], gt_ids[3]);
        CHECK_EQ(res_dis[0], gt_ids[0]);
        CHECK_EQ(res_dis[1], gt_ids[1]);
        CHECK_EQ(res_dis[2], gt_ids[2]);
        CHECK_EQ(res_dis[3], gt_ids[3]);
    }
}
TEST_CASE("Test distance") {
    using Catch::Approx;
    auto simd_type = GENERATE(as<knowhere::KnowhereConfig::SimdType>{}, knowhere::KnowhereConfig::SimdType::AVX512,
                              knowhere::KnowhereConfig::SimdType::AVX2, knowhere::KnowhereConfig::SimdType::SSE4_2,
                              knowhere::KnowhereConfig::SimdType::GENERIC, knowhere::KnowhereConfig::SimdType::AUTO);
    auto dim = GENERATE(as<size_t>{}, 1, 2, 4, 7, 8, 12, 14, 16, 21, 28, 32, 35, 42, 49, 56, 64, 128, 256);

    LOG_KNOWHERE_INFO_ << "simd type: " << simd_type << ", dim: " << dim;
    knowhere::KnowhereConfig::SetSimdType(simd_type);

    const size_t nx = 1, ny = 4;

    // fp32's accuracy is 0.000001, consider the accumulation of precision loss
    const float tolerance = 0.000005f;
    const auto x = GenRandomVector<float>(dim, nx, 314);
    const auto y = GenRandomVector<float>(dim, ny, 271);

    // fp16's accuracy is 0.001, consider the accumulation of precision loss
    const float fp16_tolerance = 0.004f;
    const auto x_fp16 = ConvertVector<knowhere::fp16>(x.get(), nx, dim);
    const auto y_fp16 = ConvertVector<knowhere::fp16>(y.get(), ny, dim);

    // bf16's accuracy is 0.01, consider the accumulation of precision loss
    const float bf16_tolerance = 0.03f;
    const auto x_bf16 = ConvertVector<knowhere::bf16>(x.get(), nx, dim);
    const auto y_bf16 = ConvertVector<knowhere::bf16>(y.get(), ny, dim);

    // int8 should have no precision loss
    const float int8_tolerance = 0.000001f;
    const auto x_int8 = ConvertVector<knowhere::int8>(x.get(), nx, dim);
    const auto y_int8 = ConvertVector<knowhere::int8>(y.get(), ny, dim);

    // int8, for hnsw sq, obsolete
    const auto xi = ConvertVector<int8_t>(x.get(), nx, dim);
    const auto yi = ConvertVector<int8_t>(y.get(), ny, dim);

    SECTION("test single distance calculation") {
        // calculate the float result ref
        std::vector<float> ref_ip, ref_L2sqr, ref_L1, ref_Linf, ref_norm_L2sqr;
        for (size_t i = 0; i < ny; i++) {
            const float* x_data = x.get();
            const float* y_data = y.get() + dim;
            ref_ip.push_back(faiss::fvec_inner_product_ref(x_data, y_data, dim));
            ref_L2sqr.push_back(faiss::fvec_L2sqr_ref(x_data, y_data, dim));
            ref_L1.push_back(faiss::fvec_L1_ref(x_data, y_data, dim));
            ref_Linf.push_back(faiss::fvec_Linf_ref(x_data, y_data, dim));
            ref_norm_L2sqr.push_back(faiss::fvec_norm_L2sqr_ref(y_data, dim));
        }

        // float
        for (size_t i = 0; i < ny; i++) {
            const float* x_data = x.get();
            const float* y_data = y.get() + dim;
            REQUIRE_THAT(faiss::fvec_inner_product(x_data, y_data, dim),
                         Catch::Matchers::WithinRel(ref_ip[i], tolerance));
            REQUIRE_THAT(faiss::fvec_L2sqr(x_data, y_data, dim), Catch::Matchers::WithinRel(ref_L2sqr[i], tolerance));
            REQUIRE_THAT(faiss::fvec_L1(x_data, y_data, dim), Catch::Matchers::WithinRel(ref_L1[i], tolerance));
            REQUIRE_THAT(faiss::fvec_Linf(x_data, y_data, dim), Catch::Matchers::WithinRel(ref_Linf[i], tolerance));
            REQUIRE_THAT(faiss::fvec_norm_L2sqr(y_data, dim), Catch::Matchers::WithinRel(ref_norm_L2sqr[i], tolerance));
        }

        // fp16
        for (size_t i = 0; i < ny; i++) {
            const knowhere::fp16* x_data = x_fp16.get();
            const knowhere::fp16* y_data = y_fp16.get() + dim;
            REQUIRE_THAT(faiss::fp16_vec_inner_product(x_data, y_data, dim),
                         Catch::Matchers::WithinRel(ref_ip[i], fp16_tolerance));
            REQUIRE_THAT(faiss::fp16_vec_L2sqr(x_data, y_data, dim),
                         Catch::Matchers::WithinRel(ref_L2sqr[i], fp16_tolerance));
            REQUIRE_THAT(faiss::fp16_vec_norm_L2sqr(y_data, dim),
                         Catch::Matchers::WithinRel(ref_norm_L2sqr[i], fp16_tolerance));
        }

        // bf16
        for (size_t i = 0; i < ny; i++) {
            const knowhere::bf16* x_data = x_bf16.get();
            const knowhere::bf16* y_data = y_bf16.get() + dim;
            REQUIRE_THAT(faiss::bf16_vec_inner_product(x_data, y_data, dim),
                         Catch::Matchers::WithinRel(ref_ip[i], bf16_tolerance));
            REQUIRE_THAT(faiss::bf16_vec_L2sqr(x_data, y_data, dim),
                         Catch::Matchers::WithinRel(ref_L2sqr[i], bf16_tolerance));
            REQUIRE_THAT(faiss::bf16_vec_norm_L2sqr(y_data, dim),
                         Catch::Matchers::WithinRel(ref_norm_L2sqr[i], bf16_tolerance));
        }
    }

    SECTION("test single distance calculation for int8") {
        // calculate the float result ref
        std::vector<float> ref_ip, ref_L2sqr, ref_norm_L2sqr;
        for (size_t i = 0; i < ny; i++) {
            const knowhere::int8* x_data = x_int8.get();
            const knowhere::int8* y_data = y_int8.get() + dim;
            ref_ip.push_back(faiss::int8_vec_inner_product_ref(x_data, y_data, dim));
            ref_L2sqr.push_back(faiss::int8_vec_L2sqr_ref(x_data, y_data, dim));
            ref_norm_L2sqr.push_back(faiss::int8_vec_norm_L2sqr_ref(y_data, dim));
        }

        // int8
        for (size_t i = 0; i < ny; i++) {
            const knowhere::int8* x_data = x_int8.get();
            const knowhere::int8* y_data = y_int8.get() + dim;
            REQUIRE_THAT(faiss::int8_vec_inner_product(x_data, y_data, dim),
                         Catch::Matchers::WithinRel(ref_ip[i], int8_tolerance));
            REQUIRE_THAT(faiss::int8_vec_L2sqr(x_data, y_data, dim),
                         Catch::Matchers::WithinRel(ref_L2sqr[i], int8_tolerance));
            REQUIRE_THAT(faiss::int8_vec_norm_L2sqr(y_data, dim),
                         Catch::Matchers::WithinRel(ref_norm_L2sqr[i], int8_tolerance));
        }
    }

    // obsolete
    SECTION("test single distance calculation for hnsw sq") {
        // calculate the int32 result ref
        std::vector<int32_t> ref_ip, ref_L2sqr;
        for (size_t i = 0; i < ny; i++) {
            const int8_t* x_data = xi.get();
            const int8_t* y_data = yi.get() + dim;
            ref_ip.push_back(faiss::ivec_inner_product_ref(x_data, y_data, dim));
            ref_L2sqr.push_back(faiss::ivec_L2sqr_ref(x_data, y_data, dim));
        }

        // int8
        for (size_t i = 0; i < ny; i++) {
            const int8_t* x_data = xi.get();
            const int8_t* y_data = yi.get() + dim;
            CHECK_EQ(faiss::ivec_inner_product(x_data, y_data, dim), ref_ip[i]);
            CHECK_EQ(faiss::ivec_L2sqr(x_data, y_data, dim), ref_L2sqr[i]);
        }
    }

    SECTION("test batch_4 distance calculation") {
        // float
        {
            const float* x_data = x.get();
            std::vector<const float*> y_data{y.get(), y.get() + dim, y.get() + 2 * dim, y.get() + 3 * dim};

            // calculate the float result ref
            std::vector<float> ref_l2_batch_4(4), ref_ip_batch_4(4);
            faiss::fvec_inner_product_batch_4_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                  ref_ip_batch_4[0], ref_ip_batch_4[1], ref_ip_batch_4[2],
                                                  ref_ip_batch_4[3]);
            faiss::fvec_L2sqr_batch_4_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim, ref_l2_batch_4[0],
                                          ref_l2_batch_4[1], ref_l2_batch_4[2], ref_l2_batch_4[3]);

            // float
            std::vector<float> l2_batch_4(4), ip_batch_4(4);
            faiss::fvec_inner_product_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim, ip_batch_4[0],
                                              ip_batch_4[1], ip_batch_4[2], ip_batch_4[3]);
            faiss::fvec_L2sqr_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim, l2_batch_4[0],
                                      l2_batch_4[1], l2_batch_4[2], l2_batch_4[3]);

            REQUIRE_THAT(ip_batch_4[0], Catch::Matchers::WithinRel(ref_ip_batch_4[0], tolerance));
            REQUIRE_THAT(ip_batch_4[1], Catch::Matchers::WithinRel(ref_ip_batch_4[1], tolerance));
            REQUIRE_THAT(ip_batch_4[2], Catch::Matchers::WithinRel(ref_ip_batch_4[2], tolerance));
            REQUIRE_THAT(ip_batch_4[3], Catch::Matchers::WithinRel(ref_ip_batch_4[3], tolerance));

            REQUIRE_THAT(l2_batch_4[0], Catch::Matchers::WithinRel(ref_l2_batch_4[0], tolerance));
            REQUIRE_THAT(l2_batch_4[1], Catch::Matchers::WithinRel(ref_l2_batch_4[1], tolerance));
            REQUIRE_THAT(l2_batch_4[2], Catch::Matchers::WithinRel(ref_l2_batch_4[2], tolerance));
            REQUIRE_THAT(l2_batch_4[3], Catch::Matchers::WithinRel(ref_l2_batch_4[3], tolerance));
        }

        // fp16
        {
            const knowhere::fp16* x_data = x_fp16.get();
            std::vector<const knowhere::fp16*> y_data{y_fp16.get(), y_fp16.get() + dim, y_fp16.get() + 2 * dim,
                                                      y_fp16.get() + 3 * dim};

            // calculate the fp16 result ref
            std::vector<float> ref_l2_batch_4(4), ref_ip_batch_4(4);
            faiss::fp16_vec_inner_product_batch_4_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                      ref_ip_batch_4[0], ref_ip_batch_4[1], ref_ip_batch_4[2],
                                                      ref_ip_batch_4[3]);
            faiss::fp16_vec_L2sqr_batch_4_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                              ref_l2_batch_4[0], ref_l2_batch_4[1], ref_l2_batch_4[2],
                                              ref_l2_batch_4[3]);

            std::vector<float> l2_batch_4(4), ip_batch_4(4);
            faiss::fp16_vec_inner_product_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                  ip_batch_4[0], ip_batch_4[1], ip_batch_4[2], ip_batch_4[3]);
            faiss::fp16_vec_L2sqr_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim, l2_batch_4[0],
                                          l2_batch_4[1], l2_batch_4[2], l2_batch_4[3]);

            REQUIRE_THAT(ip_batch_4[0], Catch::Matchers::WithinRel(ref_ip_batch_4[0], tolerance));
            REQUIRE_THAT(ip_batch_4[1], Catch::Matchers::WithinRel(ref_ip_batch_4[1], tolerance));
            REQUIRE_THAT(ip_batch_4[2], Catch::Matchers::WithinRel(ref_ip_batch_4[2], tolerance));
            REQUIRE_THAT(ip_batch_4[3], Catch::Matchers::WithinRel(ref_ip_batch_4[3], tolerance));

            REQUIRE_THAT(l2_batch_4[0], Catch::Matchers::WithinRel(ref_l2_batch_4[0], tolerance));
            REQUIRE_THAT(l2_batch_4[1], Catch::Matchers::WithinRel(ref_l2_batch_4[1], tolerance));
            REQUIRE_THAT(l2_batch_4[2], Catch::Matchers::WithinRel(ref_l2_batch_4[2], tolerance));
            REQUIRE_THAT(l2_batch_4[3], Catch::Matchers::WithinRel(ref_l2_batch_4[3], tolerance));
        }

        // bf16
        {
            const knowhere::bf16* x_data = x_bf16.get();
            std::vector<const knowhere::bf16*> y_data{y_bf16.get(), y_bf16.get() + dim, y_bf16.get() + 2 * dim,
                                                      y_bf16.get() + 3 * dim};

            // calculate the bf16 result ref
            std::vector<float> ref_l2_batch_4(4), ref_ip_batch_4(4);
            faiss::bf16_vec_inner_product_batch_4_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                      ref_ip_batch_4[0], ref_ip_batch_4[1], ref_ip_batch_4[2],
                                                      ref_ip_batch_4[3]);
            faiss::bf16_vec_L2sqr_batch_4_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                              ref_l2_batch_4[0], ref_l2_batch_4[1], ref_l2_batch_4[2],
                                              ref_l2_batch_4[3]);

            std::vector<float> l2_batch_4(4), ip_batch_4(4);
            faiss::bf16_vec_inner_product_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                  ip_batch_4[0], ip_batch_4[1], ip_batch_4[2], ip_batch_4[3]);
            faiss::bf16_vec_L2sqr_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim, l2_batch_4[0],
                                          l2_batch_4[1], l2_batch_4[2], l2_batch_4[3]);

            REQUIRE_THAT(ip_batch_4[0], Catch::Matchers::WithinRel(ref_ip_batch_4[0], tolerance));
            REQUIRE_THAT(ip_batch_4[1], Catch::Matchers::WithinRel(ref_ip_batch_4[1], tolerance));
            REQUIRE_THAT(ip_batch_4[2], Catch::Matchers::WithinRel(ref_ip_batch_4[2], tolerance));
            REQUIRE_THAT(ip_batch_4[3], Catch::Matchers::WithinRel(ref_ip_batch_4[3], tolerance));

            REQUIRE_THAT(l2_batch_4[0], Catch::Matchers::WithinRel(ref_l2_batch_4[0], tolerance));
            REQUIRE_THAT(l2_batch_4[1], Catch::Matchers::WithinRel(ref_l2_batch_4[1], tolerance));
            REQUIRE_THAT(l2_batch_4[2], Catch::Matchers::WithinRel(ref_l2_batch_4[2], tolerance));
            REQUIRE_THAT(l2_batch_4[3], Catch::Matchers::WithinRel(ref_l2_batch_4[3], tolerance));
        }

        // int8
        {
            const knowhere::int8* x_data = x_int8.get();
            std::vector<const knowhere::int8*> y_data{y_int8.get(), y_int8.get() + dim, y_int8.get() + 2 * dim,
                                                      y_int8.get() + 3 * dim};

            // calculate the int8 result ref
            std::vector<float> ref_l2_batch_4(4), ref_ip_batch_4(4);
            faiss::int8_vec_inner_product_batch_4_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                      ref_ip_batch_4[0], ref_ip_batch_4[1], ref_ip_batch_4[2],
                                                      ref_ip_batch_4[3]);
            faiss::int8_vec_L2sqr_batch_4_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                              ref_l2_batch_4[0], ref_l2_batch_4[1], ref_l2_batch_4[2],
                                              ref_l2_batch_4[3]);

            // int8
            std::vector<float> l2_batch_4(4), ip_batch_4(4);
            faiss::int8_vec_inner_product_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                  ip_batch_4[0], ip_batch_4[1], ip_batch_4[2], ip_batch_4[3]);
            faiss::int8_vec_L2sqr_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim, l2_batch_4[0],
                                          l2_batch_4[1], l2_batch_4[2], l2_batch_4[3]);

            REQUIRE_THAT(ip_batch_4[0], Catch::Matchers::WithinRel(ref_ip_batch_4[0], int8_tolerance));
            REQUIRE_THAT(ip_batch_4[1], Catch::Matchers::WithinRel(ref_ip_batch_4[1], int8_tolerance));
            REQUIRE_THAT(ip_batch_4[2], Catch::Matchers::WithinRel(ref_ip_batch_4[2], int8_tolerance));
            REQUIRE_THAT(ip_batch_4[3], Catch::Matchers::WithinRel(ref_ip_batch_4[3], int8_tolerance));

            REQUIRE_THAT(l2_batch_4[0], Catch::Matchers::WithinRel(ref_l2_batch_4[0], int8_tolerance));
            REQUIRE_THAT(l2_batch_4[1], Catch::Matchers::WithinRel(ref_l2_batch_4[1], int8_tolerance));
            REQUIRE_THAT(l2_batch_4[2], Catch::Matchers::WithinRel(ref_l2_batch_4[2], int8_tolerance));
            REQUIRE_THAT(l2_batch_4[3], Catch::Matchers::WithinRel(ref_l2_batch_4[3], int8_tolerance));
        }
    }

    SECTION("test ny distance calculation") {
        // calculate the float result ref
        auto ref_ip = std::make_unique<float[]>(ny);
        faiss::fvec_inner_products_ny_ref(ref_ip.get(), x.get(), y.get(), dim, ny);
        auto ref_l2 = std::make_unique<float[]>(ny);
        faiss::fvec_L2sqr_ny_ref(ref_l2.get(), x.get(), y.get(), dim, ny);

        auto ip_dis = std::make_unique<float[]>(ny);
        auto l2_dis = std::make_unique<float[]>(ny);

        faiss::fvec_inner_products_ny(ip_dis.get(), x.get(), y.get(), dim, ny);
        faiss::fvec_L2sqr_ny(l2_dis.get(), x.get(), y.get(), dim, ny);
        for (size_t i = 0; i < ny; i++) {
            REQUIRE_THAT(ip_dis[i], Catch::Matchers::WithinRel(ref_ip[i], tolerance));
            REQUIRE_THAT(l2_dis[i], Catch::Matchers::WithinRel(ref_l2[i], tolerance));
        }
    }

    SECTION("test madd distance calculation") {
        const float bf = 3.14159;

        // calculate the float result ref
        auto ref_madd = std::make_unique<float[]>(dim);
        faiss::fvec_madd_ref(dim, x.get(), bf, y.get(), ref_madd.get());
        auto ref_madd_and_argmin = std::make_unique<float[]>(dim);
        faiss::fvec_madd_and_argmin_ref(dim, x.get(), bf, y.get(), ref_madd_and_argmin.get());

        auto madd_dis = std::make_unique<float[]>(dim);
        auto madd_and_argmin_dis = std::make_unique<float[]>(dim);

        faiss::fvec_madd(dim, x.get(), bf, y.get(), madd_dis.get());
        faiss::fvec_madd_and_argmin(dim, x.get(), bf, y.get(), madd_and_argmin_dis.get());
        for (size_t i = 0; i < dim; i++) {
            REQUIRE_THAT(madd_dis[i], Catch::Matchers::WithinRel(ref_madd[i], tolerance));
            REQUIRE_THAT(madd_and_argmin_dis[i], Catch::Matchers::WithinRel(ref_madd_and_argmin[i], tolerance));
        }
    }

    SECTION("test bf16_patch distance calculation") {
        const float* x_data = x.get();
        std::vector<const float*> y_data{y.get(), y.get() + dim, y.get() + 2 * dim, y.get() + 3 * dim};

        const auto ref_ip = faiss::fvec_inner_product_ref(x.get(), y.get(), dim);
        const auto ref_L2sqr = faiss::fvec_L2sqr_ref(x.get(), y.get(), dim);

        // calculate the bf16 patch result ref
        std::vector<float> ref_l2_batch_4(4), ref_ip_batch_4(4);
        faiss::fvec_inner_product_batch_4_bf16_patch_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                         ref_ip_batch_4[0], ref_ip_batch_4[1], ref_ip_batch_4[2],
                                                         ref_ip_batch_4[3]);
        faiss::fvec_L2sqr_batch_4_bf16_patch_ref(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim,
                                                 ref_l2_batch_4[0], ref_l2_batch_4[1], ref_l2_batch_4[2],
                                                 ref_l2_batch_4[3]);

        auto run_test = [&]() {
            REQUIRE_THAT(faiss::fvec_inner_product(x.get(), y.get(), dim),
                         Catch::Matchers::WithinRel(ref_ip, bf16_tolerance));

            REQUIRE_THAT(faiss::fvec_L2sqr(x.get(), y.get(), dim),
                         Catch::Matchers::WithinRel(ref_L2sqr, bf16_tolerance));

            std::vector<float> ip_batch_4(4);
            faiss::fvec_inner_product_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim, ip_batch_4[0],
                                              ip_batch_4[1], ip_batch_4[2], ip_batch_4[3]);
            REQUIRE_THAT(ip_batch_4[0], Catch::Matchers::WithinRel(ref_ip_batch_4[0], bf16_tolerance));
            REQUIRE_THAT(ip_batch_4[1], Catch::Matchers::WithinRel(ref_ip_batch_4[1], bf16_tolerance));
            REQUIRE_THAT(ip_batch_4[2], Catch::Matchers::WithinRel(ref_ip_batch_4[2], bf16_tolerance));
            REQUIRE_THAT(ip_batch_4[3], Catch::Matchers::WithinRel(ref_ip_batch_4[3], bf16_tolerance));

            std::vector<float> l2_batch_4(4);
            faiss::fvec_L2sqr_batch_4(x_data, y_data[0], y_data[1], y_data[2], y_data[3], dim, l2_batch_4[0],
                                      l2_batch_4[1], l2_batch_4[2], l2_batch_4[3]);
            REQUIRE_THAT(l2_batch_4[0], Catch::Matchers::WithinRel(ref_l2_batch_4[0], bf16_tolerance));
            REQUIRE_THAT(l2_batch_4[1], Catch::Matchers::WithinRel(ref_l2_batch_4[1], bf16_tolerance));
            REQUIRE_THAT(l2_batch_4[2], Catch::Matchers::WithinRel(ref_l2_batch_4[2], bf16_tolerance));
            REQUIRE_THAT(l2_batch_4[3], Catch::Matchers::WithinRel(ref_l2_batch_4[3], bf16_tolerance));
        };

        knowhere::KnowhereConfig::EnablePatchForComputeFP32AsBF16();
        run_test();

        knowhere::KnowhereConfig::DisablePatchForComputeFP32AsBF16();
        run_test();
    }
}

TEST_CASE("Test Distance Known Values - Multi Types", "[distance][known_values]") {
    constexpr float kTolerance = 0.01f;

    SECTION("fp16 known values") {
        // [1, 2, 3] dot [4, 5, 6] = 32
        std::vector<knowhere::fp16> x = {knowhere::fp16(1.0f), knowhere::fp16(2.0f), knowhere::fp16(3.0f)};
        std::vector<knowhere::fp16> y = {knowhere::fp16(4.0f), knowhere::fp16(5.0f), knowhere::fp16(6.0f)};
        REQUIRE_THAT(faiss::fp16_vec_inner_product_ref(x.data(), y.data(), 3),
                     Catch::Matchers::WithinRel(32.0f, kTolerance));

        // norm^2 of [3, 4] = 25
        std::vector<knowhere::fp16> n = {knowhere::fp16(3.0f), knowhere::fp16(4.0f)};
        REQUIRE_THAT(faiss::fp16_vec_norm_L2sqr_ref(n.data(), 2), Catch::Matchers::WithinRel(25.0f, kTolerance));
    }

    SECTION("bf16 known values") {
        std::vector<knowhere::bf16> x = {knowhere::bf16(1.0f), knowhere::bf16(2.0f), knowhere::bf16(3.0f)};
        std::vector<knowhere::bf16> y = {knowhere::bf16(4.0f), knowhere::bf16(5.0f), knowhere::bf16(6.0f)};
        REQUIRE_THAT(faiss::bf16_vec_inner_product_ref(x.data(), y.data(), 3),
                     Catch::Matchers::WithinRel(32.0f, kTolerance));
    }

    SECTION("int8 known values") {
        std::vector<int8_t> x = {1, 2, 3, 4};
        std::vector<int8_t> y = {5, 6, 7, 8};
        // IP = 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        REQUIRE_THAT(faiss::int8_vec_inner_product_ref(x.data(), y.data(), 4),
                     Catch::Matchers::WithinRel(70.0f, 0.001f));

        // L2^2 of [1,2,3] vs [4,5,6] = 9+9+9 = 27
        std::vector<int8_t> a = {1, 2, 3};
        std::vector<int8_t> b = {4, 5, 6};
        REQUIRE_THAT(faiss::int8_vec_L2sqr_ref(a.data(), b.data(), 3), Catch::Matchers::WithinRel(27.0f, 0.001f));
    }
}

TEST_CASE("Test Cross-Type Distance Consistency", "[distance][cross_type]") {
    std::mt19937 rng(42);
    std::string ins;
    faiss::fvec_hook(ins);

    auto dim = GENERATE(64, 128);
    constexpr float kCrossTypeTolerance = 0.10f;

    SECTION("fp32 vs fp16 vs bf16 should produce similar results") {
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

        for (int trial = 0; trial < 10; ++trial) {
            std::vector<float> fp32_x(dim), fp32_y(dim);
            std::vector<knowhere::fp16> fp16_x(dim), fp16_y(dim);
            std::vector<knowhere::bf16> bf16_x(dim), bf16_y(dim);

            for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
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

            if (std::abs(ip_fp32) > 1.0f) {
                REQUIRE_THAT(ip_fp16, Catch::Matchers::WithinRel(ip_fp32, kCrossTypeTolerance));
                REQUIRE_THAT(ip_bf16, Catch::Matchers::WithinRel(ip_fp32, kCrossTypeTolerance));
            }
        }
    }
}
