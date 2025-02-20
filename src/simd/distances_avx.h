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

#ifndef DISTANCES_AVX_H
#define DISTANCES_AVX_H

#include <cstddef>
#include <cstdint>

#include "knowhere/operands.h"

namespace faiss {

/// Squared L2 distance between two vectors
float
fvec_L2sqr_avx(const float* x, const float* y, size_t d);

/// inner product
float
fvec_inner_product_avx(const float* x, const float* y, size_t d);

/// L1 distance
float
fvec_L1_avx(const float* x, const float* y, size_t d);

/// infinity distance
float
fvec_Linf_avx(const float* x, const float* y, size_t d);

void
fvec_madd_avx(size_t n, const float* a, float bf, const float* b, float* c);

void
fvec_inner_product_batch_4_avx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                               const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

void
fvec_L2sqr_batch_4_avx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                       const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

float
fvec_norm_L2sqr_avx(const float* x, size_t d);

void
fvec_L2sqr_ny_avx(float* dis, const float* x, const float* y, size_t d, size_t ny);

size_t
fvec_L2sqr_ny_nearest_avx(float* distances_tmp_buffer, const float* x, const float* y, size_t d, size_t ny);

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete

int32_t
ivec_inner_product_avx(const int8_t* x, const int8_t* y, size_t d);

int32_t
ivec_L2sqr_avx(const int8_t* x, const int8_t* y, size_t d);

///////////////////////////////////////////////////////////////////////////////
// fp16

float
fp16_vec_inner_product_avx(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_L2sqr_avx(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_norm_L2sqr_avx(const knowhere::fp16* x, size_t d);

void
fp16_vec_inner_product_batch_4_avx(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                   const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3);
void
fp16_vec_L2sqr_batch_4_avx(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                           const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3);

///////////////////////////////////////////////////////////////////////////////
// bf16

float
bf16_vec_inner_product_avx(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_L2sqr_avx(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_norm_L2sqr_avx(const knowhere::bf16* x, size_t d);

void
bf16_vec_inner_product_batch_4_avx(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                   const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3);

void
bf16_vec_L2sqr_batch_4_avx(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                           const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3);

///////////////////////////////////////////////////////////////////////////////
// for cardinal

float
fvec_inner_product_bf16_patch_avx(const float* x, const float* y, size_t d);

float
fvec_L2sqr_bf16_patch_avx(const float* x, const float* y, size_t d);

void
fvec_inner_product_batch_4_bf16_patch_avx(const float* x, const float* y0, const float* y1, const float* y2,
                                          const float* y3, const size_t d, float& dis0, float& dis1, float& dis2,
                                          float& dis3);

void
fvec_L2sqr_batch_4_bf16_patch_avx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                  const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

}  // namespace faiss

#endif /* DISTANCES_AVX_H */
