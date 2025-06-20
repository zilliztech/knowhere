// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#pragma once

#include <cstdint>
#include <cstdio>

#include "knowhere/operands.h"

namespace faiss {

float
fvec_inner_product_rvv(const float* x, const float* y, size_t d);

float
fvec_L2sqr_rvv(const float* x, const float* y, size_t d);

float
fvec_L1_rvv(const float* x, const float* y, size_t d);

float
fvec_Linf_rvv(const float* x, const float* y, size_t d);

float
fvec_norm_L2sqr_rvv(const float* x, size_t d);

void
fvec_L2sqr_ny_rvv(float* dis, const float* x, const float* y, size_t d, size_t ny);

void
fvec_inner_products_ny_rvv(float* ip, const float* x, const float* y, size_t d, size_t ny);

void
fvec_madd_rvv(size_t n, const float* a, float bf, const float* b, float* c);

}  // namespace faiss
