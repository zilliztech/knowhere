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

#include <arm_sve.h>

#include <cstdint>
#include <cstdio>

#include "knowhere/operands.h"

#if defined(__ARM_FEATURE_SVE)
namespace faiss {

float
fvec_L2sqr_sve(const float* x, const float* y, size_t d);

float
fvec_inner_product_sve(const float* x, const float* y, size_t d);

float
fp16_vec_L2sqr_sve(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_inner_product_sve(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fvec_L1_sve(const float* x, const float* y, size_t d);

float
fvec_Linf_sve(const float* x, const float* y, size_t d);

float
fvec_norm_L2sqr_sve(const float* x, size_t d);

float
fp16_vec_norm_L2sqr_sve(const knowhere::fp16* x, size_t d);

void
fvec_madd_sve(size_t n, const float* a, float bf, const float* b, float* c);

int
fvec_madd_and_argmin_sve(size_t n, const float* a, float bf, const float* b, float* c);

int32_t
ivec_L2sqr_sve(const int8_t* x, const int8_t* y, size_t d);

void
fvec_inner_product_batch_4_sve(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                               const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

void
fvec_L2sqr_batch_4_sve(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                       const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

void
fvec_L2sqr_ny_sve(float* dis, const float* x, const float* y, size_t d, size_t ny);

/// compute ny square L2 distance between x and a set of contiguous y vectors
/// and return the index of the nearest vector.
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_sve(float* distances_tmp_buffer, const float* x, const float* y, size_t d, size_t ny);

void
fvec_inner_products_ny_sve(float* ip, const float* x, const float* y, size_t d, size_t ny);

float
int8_vec_L2sqr_sve(const int8_t* x, const int8_t* y, size_t d);

float
int8_vec_norm_L2sqr_sve(const int8_t* x, size_t d);

void
int8_vec_L2sqr_batch_4_sve(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                           const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3);

float
int8_vec_inner_product_sve(const int8_t* x, const int8_t* y, size_t d);

void
int8_vec_inner_product_batch_4_sve(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                   const int8_t* y3, const size_t dim, float& dis0, float& dis1, float& dis2,
                                   float& dis3);

float
bf16_vec_L2sqr_sve(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_norm_L2sqr_sve(const knowhere::bf16* x, size_t d);

void
bf16_vec_L2sqr_batch_4_sve(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                           const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3);

float
bf16_vec_inner_product_sve(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

void
bf16_vec_inner_product_batch_4_sve(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                   const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3);

}  // namespace faiss
#endif
