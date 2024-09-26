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
    std::uniform_real_distribution<> distrib(-10.0, 10.0);
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
    auto dim = GENERATE(as<size_t>{}, 1, 7, 14, 21, 28, 35, 42, 49, 56, 64, 128, 256, 512);

    LOG_KNOWHERE_INFO_ << "simd type: " << simd_type << ", dim: " << dim;
    knowhere::KnowhereConfig::SetSimdType(simd_type);

    SECTION("test single distance calculation") {
        const size_t nx = 1, ny = 1;

        const float tolerance = 0.000001f;
        const auto x = GenRandomVector<float>(dim, nx, 314);
        const auto y = GenRandomVector<float>(dim, ny, 271);

        // fp16's accuracy is 0.001, during calculation we let the tolerance be 0.002
        const float fp16_tolerance = 0.002f;
        const auto x_fp16 = ConvertVector<knowhere::fp16>(x.get(), nx, dim);
        const auto y_fp16 = ConvertVector<knowhere::fp16>(y.get(), ny, dim);

        // bf16's accuracy is 0.01, during calculation we let the tolerance be 0.02
        const float bf16_tolerance = 0.02f;
        const auto x_bf16 = ConvertVector<knowhere::bf16>(x.get(), nx, dim);
        const auto y_bf16 = ConvertVector<knowhere::bf16>(y.get(), ny, dim);

        // int8
        const auto xi = ConvertVector<int8_t>(x.get(), nx, dim);
        const auto yi = ConvertVector<int8_t>(y.get(), ny, dim);

        const auto ref_ip = faiss::fvec_inner_product_ref(x.get(), y.get(), dim);
        const auto ref_L2sqr = faiss::fvec_L2sqr_ref(x.get(), y.get(), dim);
        const auto ref_L1 = faiss::fvec_L1_ref(x.get(), y.get(), dim);
        const auto ref_Linf = faiss::fvec_Linf_ref(x.get(), y.get(), dim);
        const auto ref_norm_L2sqr = faiss::fvec_norm_L2sqr_ref(x.get(), dim);

        const auto ref_i_ip = faiss::ivec_inner_product_ref(xi.get(), yi.get(), dim);
        const auto ref_i_L2sqr = faiss::ivec_L2sqr_ref(xi.get(), yi.get(), dim);

        // float
        REQUIRE_THAT(faiss::fvec_inner_product(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_ip, tolerance));
        REQUIRE_THAT(faiss::fvec_L2sqr(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_L2sqr, tolerance));
        REQUIRE_THAT(faiss::fvec_L1(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_L1, tolerance));
        REQUIRE_THAT(faiss::fvec_Linf(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_Linf, tolerance));
        REQUIRE_THAT(faiss::fvec_norm_L2sqr(x.get(), dim), Catch::Matchers::WithinRel(ref_norm_L2sqr, tolerance));

        // fp16
        REQUIRE_THAT(faiss::fp16_vec_inner_product(x_fp16.get(), y_fp16.get(), dim),
                     Catch::Matchers::WithinRel(ref_ip, fp16_tolerance));
        REQUIRE_THAT(faiss::fp16_vec_L2sqr(x_fp16.get(), y_fp16.get(), dim),
                     Catch::Matchers::WithinRel(ref_L2sqr, fp16_tolerance));
        REQUIRE_THAT(faiss::fp16_vec_norm_L2sqr(x_fp16.get(), dim),
                     Catch::Matchers::WithinRel(ref_norm_L2sqr, fp16_tolerance));

        // bf16
        REQUIRE_THAT(faiss::bf16_vec_inner_product(x_bf16.get(), y_bf16.get(), dim),
                     Catch::Matchers::WithinRel(ref_ip, bf16_tolerance));
        REQUIRE_THAT(faiss::bf16_vec_L2sqr(x_bf16.get(), y_bf16.get(), dim),
                     Catch::Matchers::WithinRel(ref_L2sqr, bf16_tolerance));
        REQUIRE_THAT(faiss::bf16_vec_norm_L2sqr(x_bf16.get(), dim),
                     Catch::Matchers::WithinRel(ref_norm_L2sqr, bf16_tolerance));

        // int8
        CHECK_EQ(faiss::ivec_inner_product(xi.get(), yi.get(), dim), ref_i_ip);
        CHECK_EQ(faiss::ivec_L2sqr(xi.get(), yi.get(), dim), ref_i_L2sqr);
    }

    SECTION("test ny distance calculation") {
        const size_t nx = 1, ny = 10;

        const float tolerance = 0.0001f;
        const auto x = GenRandomVector<float>(dim, nx, 314);
        const auto y = GenRandomVector<float>(dim, ny, 271);

        auto ref_ip = std::make_unique<float[]>(ny);
        faiss::fvec_inner_products_ny_ref(ref_ip.get(), x.get(), y.get(), dim, ny);
        auto ref_l2 = std::make_unique<float[]>(ny);
        faiss::fvec_L2sqr_ny_ref(ref_l2.get(), x.get(), y.get(), dim, ny);

        auto dis = std::make_unique<float[]>(ny);

        faiss::fvec_inner_products_ny(dis.get(), x.get(), y.get(), dim, ny);
        for (size_t i = 0; i < ny; i++) {
            REQUIRE_THAT(dis[i], Catch::Matchers::WithinRel(ref_ip[i], tolerance));
        }

        faiss::fvec_L2sqr_ny(dis.get(), x.get(), y.get(), dim, ny);
        for (size_t i = 0; i < ny; i++) {
            REQUIRE_THAT(dis[i], Catch::Matchers::WithinRel(ref_l2[i], tolerance));
        }
    }

    SECTION("test madd distance calculation") {
        const size_t n = 1;

        const float tolerance = 0.0001f;
        const auto a = GenRandomVector<float>(dim, n, 314);
        const auto b = GenRandomVector<float>(dim, n, 271);
        const float bf = 3.14159;

        auto ref_madd = std::make_unique<float[]>(dim);
        faiss::fvec_madd_ref(dim, a.get(), bf, b.get(), ref_madd.get());
        auto ref_madd_and_argmin = std::make_unique<float[]>(dim);
        faiss::fvec_madd_and_argmin_ref(dim, a.get(), bf, b.get(), ref_madd_and_argmin.get());

        auto dis = std::make_unique<float[]>(dim);

        faiss::fvec_madd(dim, a.get(), bf, b.get(), dis.get());
        for (size_t i = 0; i < dim; i++) {
            REQUIRE_THAT(dis[i], Catch::Matchers::WithinRel(ref_madd[i], tolerance));
        }

        faiss::fvec_madd_and_argmin(dim, a.get(), bf, b.get(), dis.get());
        for (size_t i = 0; i < dim; i++) {
            REQUIRE_THAT(dis[i], Catch::Matchers::WithinRel(ref_madd_and_argmin[i], tolerance));
        }
    }

    SECTION("test batch_4 distance calculation") {
        const size_t nx = 1, ny = 1;

        float tolerance = 0.00001f;
        const auto x = GenRandomVector<float>(dim, nx, 314);
        const auto y = GenRandomVector<float>(dim, ny, 271);

        const auto ref_ip = faiss::fvec_inner_product_ref(x.get(), y.get(), dim);
        const auto ref_L2sqr = faiss::fvec_L2sqr_ref(x.get(), y.get(), dim);

        float batch_tolerance = 0.0002f;
        const auto y0 = GenRandomVector<float>(dim, ny, 271);
        const auto y1 = GenRandomVector<float>(dim, ny, 272);
        const auto y2 = GenRandomVector<float>(dim, ny, 273);
        const auto y3 = GenRandomVector<float>(dim, ny, 274);

        float ref_ip_0, ref_ip_1, ref_ip_2, ref_ip_3;
        faiss::fvec_inner_product_batch_4_ref(x.get(), y0.get(), y1.get(), y2.get(), y3.get(), dim, ref_ip_0, ref_ip_1,
                                              ref_ip_2, ref_ip_3);
        float ref_l2_0, ref_l2_1, ref_l2_2, ref_l2_3;
        faiss::fvec_L2sqr_batch_4_ref(x.get(), y0.get(), y1.get(), y2.get(), y3.get(), dim, ref_l2_0, ref_l2_1,
                                      ref_l2_2, ref_l2_3);

        auto run_test = [&]() {
            // float
            REQUIRE_THAT(faiss::fvec_inner_product(x.get(), y.get(), dim),
                         Catch::Matchers::WithinRel(ref_ip, tolerance));
            REQUIRE_THAT(faiss::fvec_L2sqr(x.get(), y.get(), dim), Catch::Matchers::WithinRel(ref_L2sqr, tolerance));

            // batch
            float dis0, dis1, dis2, dis3;

            faiss::fvec_inner_product_batch_4(x.get(), y0.get(), y1.get(), y2.get(), y3.get(), dim, dis0, dis1, dis2,
                                              dis3);
            REQUIRE_THAT(dis0, Catch::Matchers::WithinRel(ref_ip_0, batch_tolerance));
            REQUIRE_THAT(dis1, Catch::Matchers::WithinRel(ref_ip_1, batch_tolerance));
            REQUIRE_THAT(dis2, Catch::Matchers::WithinRel(ref_ip_2, batch_tolerance));
            REQUIRE_THAT(dis3, Catch::Matchers::WithinRel(ref_ip_3, batch_tolerance));

            faiss::fvec_L2sqr_batch_4(x.get(), y0.get(), y1.get(), y2.get(), y3.get(), dim, dis0, dis1, dis2, dis3);
            REQUIRE_THAT(dis0, Catch::Matchers::WithinRel(ref_l2_0, batch_tolerance));
            REQUIRE_THAT(dis1, Catch::Matchers::WithinRel(ref_l2_1, batch_tolerance));
            REQUIRE_THAT(dis2, Catch::Matchers::WithinRel(ref_l2_2, batch_tolerance));
            REQUIRE_THAT(dis3, Catch::Matchers::WithinRel(ref_l2_3, batch_tolerance));
        };

        tolerance = 0.02f;
        batch_tolerance = 0.05f;
        knowhere::KnowhereConfig::EnablePatchForComputeFP32AsBF16();
        // TODO caiyd: need enable this test
        // run_test();

        tolerance = 0.00001f;
        batch_tolerance = 0.0002f;
        knowhere::KnowhereConfig::DisablePatchForComputeFP32AsBF16();
        run_test();
    }
}
