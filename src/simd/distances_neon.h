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

#ifndef DISTANCES_NEON_H
#define DISTANCES_NEON_H

#include <cstdint>
#include <cstdio>

#include "knowhere/operands.h"

namespace faiss {

/// Squared L2 distance between two vectors
float
fvec_L2sqr_neon(const float* x, const float* y, size_t d);
float
fvec_L2sqr_neon_bf16_patch(const float* x, const float* y, size_t d);

float
fp16_vec_L2sqr_neon(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
bf16_vec_L2sqr_neon(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

/// inner product
float
fvec_inner_product_neon(const float* x, const float* y, size_t d);
float
fvec_inner_product_neon_bf16_patch(const float* x, const float* y, size_t d);

float
fp16_vec_inner_product_neon(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
bf16_vec_inner_product_neon(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

/// L1 distance
float
fvec_L1_neon(const float* x, const float* y, size_t d);

/// infinity distance
float
fvec_Linf_neon(const float* x, const float* y, size_t d);

/// squared norm of a vector
float
fvec_norm_L2sqr_neon(const float* x, size_t d);

float
fp16_vec_norm_L2sqr_neon(const knowhere::fp16* x, size_t d);

float
bf16_vec_norm_L2sqr_neon(const knowhere::bf16* x, size_t d);

/// compute ny square L2 distance between x and a set of contiguous y vectors
void
fvec_L2sqr_ny_neon(float* dis, const float* x, const float* y, size_t d, size_t ny);

/// compute the inner product between nx vectors x and one y
void
fvec_inner_products_ny_neon(float* ip, const float* x, const float* y, size_t d, size_t ny);

void
fvec_madd_neon(size_t n, const float* a, float bf, const float* b, float* c);

int
fvec_madd_and_argmin_neon(size_t n, const float* a, float bf, const float* b, float* c);

int32_t
ivec_inner_product_neon(const int8_t* x, const int8_t* y, size_t d);

int32_t
ivec_L2sqr_neon(const int8_t* x, const int8_t* y, size_t d);

/// Special version of inner product that computes 4 distances
/// between x and yi, which is performance oriented.
void
fvec_inner_product_batch_4_neon(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3);

void
fvec_inner_product_batch_4_neon_bf16_patch(const float* x, const float* y0, const float* y1, const float* y2,
                                           const float* y3, const size_t dim, float& dis0, float& dis1, float& dis2,
                                           float& dis3);

void
fp16_vec_inner_product_batch_4_neon(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                    const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                                    float& dis1, float& dis2, float& dis3);

void
bf16_vec_inner_product_batch_4_neon(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                    const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                    float& dis1, float& dis2, float& dis3);

/// Special version of L2sqr that computes 4 distances
/// between x and yi, which is performance oriented.
void
fvec_L2sqr_batch_4_neon(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                        const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3);

void
fvec_L2sqr_batch_4_neon_bf16_patch(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                   const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3);

void
fp16_vec_L2sqr_batch_4_neon(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                            const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                            float& dis1, float& dis2, float& dis3);

void
bf16_vec_L2sqr_batch_4_neon(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                            const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                            float& dis1, float& dis2, float& dis3);

}  // namespace faiss

#endif /* DISTANCES_NEON_H */
