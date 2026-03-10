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
#include "knowhere/utils.h"
#include "simd/distances_ref.h"
#include "utils.h"

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
