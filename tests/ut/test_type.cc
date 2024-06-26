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
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/utils.h"
#include "simd/distances_ref.h"
#include "simd/hook.h"
#include "utils.h"

TEST_CASE("Test bf16 patch", "[bf16 patch]") {
    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    auto train_tensor = reinterpret_cast<const float*>(train_ds->GetTensor());

    std::vector<float> l2_dist(nb * nq);
    std::vector<float> l2_dist_patch(nb * nq);
    std::vector<float> l2_dist_patch_ref(nb * nq);

    std::vector<float> ip_dist(nb * nq);
    std::vector<float> ip_dist_patch(nb * nq);
    std::vector<float> ip_dist_patch_ref(nb * nq);

    auto query_tensor = reinterpret_cast<const float*>(query_ds->GetTensor());

    auto compute_dist = [&](float* l2_dist_ret, float* ip_dist_ret) {
        for (int64_t i = 0; i < nq; i++) {
            for (int64_t j = 0; j < nb; j++) {
                l2_dist_ret[i * nq + j] = faiss::fvec_L2sqr(query_tensor + i * dim, train_tensor + j * dim, dim);
                ip_dist_ret[i * nq + j] =
                    faiss::fvec_inner_product(query_tensor + i * dim, train_tensor + j * dim, dim);
            }
        }
    };

    auto compute_ref_dist = [&](float* l2_dist_ret, float* ip_dist_ret) {
        for (int64_t i = 0; i < nq; i++) {
            for (int64_t j = 0; j < nb; j++) {
                l2_dist_ret[i * nq + j] = faiss::fvec_L2sqr_ref(query_tensor + i * dim, train_tensor + j * dim, dim);
                ip_dist_ret[i * nq + j] =
                    faiss::fvec_inner_product_ref(query_tensor + i * dim, train_tensor + j * dim, dim);
            }
        }
    };

    compute_dist(l2_dist.data(), ip_dist.data());

    knowhere::KnowhereConfig::EnablePatchForComputeFP32AsBF16();

    compute_dist(l2_dist_patch.data(), ip_dist_patch.data());
    compute_ref_dist(l2_dist_patch_ref.data(), ip_dist_patch_ref.data());

    double l2_error_sum = 0.0;
    double l2_ref_error_sum = 0.0;
    double l2_total_sum = 0.0;

    double ip_error_sum = 0.0;
    double ip_ref_error_sum = 0.0;
    double ip_total_sum = 0.0;
    for (int64_t i = 0; i < nq * nb; i++) {
        l2_error_sum += abs(l2_dist[i] - l2_dist_patch[i]);
        l2_ref_error_sum += abs(l2_dist[i] - l2_dist_patch_ref[i]);
        l2_total_sum += abs(l2_dist[i]);

        ip_error_sum += abs(ip_dist[i] - ip_dist_patch[i]);
        ip_ref_error_sum += abs(ip_dist[i] - ip_dist_patch_ref[i]);
        ip_total_sum += abs(ip_dist[i]);
    }

    double l2_relative_error = l2_error_sum / l2_total_sum;
    double l2_ref_relative_error = l2_ref_error_sum / l2_total_sum;
    double ip_relative_error = ip_error_sum / ip_total_sum;
    double ip_ref_relative_error = ip_ref_error_sum / ip_total_sum;

    REQUIRE(l2_relative_error < pow(2, -11.0));
    REQUIRE(l2_ref_relative_error < pow(2, -11.0));
    REQUIRE(ip_relative_error < pow(2, -11.0));
    REQUIRE(ip_ref_relative_error < pow(2, -11.0));

    knowhere::KnowhereConfig::DisablePatchForComputeFP32AsBF16();

    std::vector<float> l2_dist_new(nb * nq);
    std::vector<float> ip_dist_new(nb * nq);
    compute_dist(l2_dist_new.data(), ip_dist_new.data());
    for (int64_t i = 0; i < nq * nb; i++) {
        REQUIRE(l2_dist[i] == l2_dist_new[i]);
        REQUIRE(ip_dist[i] == ip_dist_new[i]);
    }
}

template <typename T>
void
check_data_type_accuracy(float accuracy) {
    const int64_t nb = 100;
    const int64_t dim = 16;

    auto fp32_base_ds = GenDataSet(nb, dim);

    auto type_base_ds = knowhere::ConvertToDataTypeIfNeeded<T>(fp32_base_ds);
    auto fp32_base_ds_2 = knowhere::ConvertFromDataTypeIfNeeded<T>(type_base_ds);

    auto bv1 = static_cast<const float*>(fp32_base_ds->GetTensor());
    auto bv2 = static_cast<const float*>(fp32_base_ds_2->GetTensor());

    for (int64_t i = 0; i < nb * dim; i++) {
        REQUIRE(std::abs(bv2[i] / bv1[i] - 1.0) < accuracy);
    }
}

TEST_CASE("Test Float16", "[fp16]") {
    check_data_type_accuracy<knowhere::fp16>(0.001);
    check_data_type_accuracy<knowhere::bf16>(0.01);
}
