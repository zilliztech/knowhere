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
#include "knowhere/comp/knowhere_config.h"
#include "simd/distances_ref.h"
#include "simd/hook.h"
#include "utils.h"

template <typename DataType>
std::unique_ptr<DataType[]>
GenRandomVector(int dim, int rows, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib(-128.0, 127.0);
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
