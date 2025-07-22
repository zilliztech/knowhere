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

#pragma once

#include <cstddef>
#include <cstdint>

#include "knowhere/operands.h"

namespace faiss {

float
fvec_L2sqr_avx512(const float* x, const float* y, size_t d);

/// inner product
float
fvec_inner_product_avx512(const float* x, const float* y, size_t d);

/// L1 distance
float
fvec_L1_avx512(const float* x, const float* y, size_t d);

/// infinity distance
float
fvec_Linf_avx512(const float* x, const float* y, size_t d);

void
fvec_madd_avx512(size_t n, const float* a, float bf, const float* b, float* c);

void
fvec_inner_product_batch_4_avx512(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                  const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

void
fvec_L2sqr_batch_4_avx512(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                          const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

float
fvec_norm_L2sqr_avx512(const float* x, size_t d);

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete

int32_t
ivec_inner_product_avx512(const int8_t* x, const int8_t* y, size_t d);

int32_t
ivec_L2sqr_avx512(const int8_t* x, const int8_t* y, size_t d);

///////////////////////////////////////////////////////////////////////////////
// fp16

float
fp16_vec_inner_product_avx512(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_L2sqr_avx512(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_norm_L2sqr_avx512(const knowhere::fp16* x, size_t d);

void
fp16_vec_inner_product_batch_4_avx512(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                      const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                                      float& dis1, float& dis2, float& dis3);

void
fp16_vec_L2sqr_batch_4_avx512(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                              const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                              float& dis1, float& dis2, float& dis3);

///////////////////////////////////////////////////////////////////////////////
// bf16

float
bf16_vec_inner_product_avx512(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_L2sqr_avx512(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_norm_L2sqr_avx512(const knowhere::bf16* x, size_t d);

void
bf16_vec_inner_product_batch_4_avx512(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                      const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                      float& dis1, float& dis2, float& dis3);

void
bf16_vec_L2sqr_batch_4_avx512(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                              const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                              float& dis1, float& dis2, float& dis3);

///////////////////////////////////////////////////////////////////////////////
// int8

float
int8_vec_inner_product_avx512(const int8_t* x, const int8_t* y, size_t d);

float
int8_vec_L2sqr_avx512(const int8_t* x, const int8_t* y, size_t d);

float
int8_vec_norm_L2sqr_avx512(const int8_t* x, size_t d);

void
int8_vec_inner_product_batch_4_avx512(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                      const int8_t* y3, const size_t d, float& dis0, float& dis1, float& dis2,
                                      float& dis3);

void
int8_vec_L2sqr_batch_4_avx512(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                              const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

///////////////////////////////////////////////////////////////////////////////
// for cardinal

float
fvec_inner_product_bf16_patch_avx512(const float* x, const float* y, size_t d);

float
fvec_L2sqr_bf16_patch_avx512(const float* x, const float* y, size_t d);

void
fvec_inner_product_batch_4_bf16_patch_avx512(const float* x, const float* y0, const float* y1, const float* y2,
                                             const float* y3, const size_t d, float& dis0, float& dis1, float& dis2,
                                             float& dis3);

void
fvec_L2sqr_batch_4_bf16_patch_avx512(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                     const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

///////////////////////////////////////////////////////////////////////////////
// rabitq
float
fvec_masked_sum_avx512(const float* q, const uint8_t* x, const size_t d);
int
rabitq_dp_popcnt_avx512(const uint8_t* q, const uint8_t* x, const size_t d, const size_t nb);

///////////////////////////////////////////////////////////////////////////////
// minhash
int
u64_binary_search_eq_avx512(const uint64_t*, const size_t n, const uint64_t key);
int
u64_binary_search_ge_avx512(const uint64_t*, const size_t n, const uint64_t key);

uint64_t
calculate_hash_avx512(const char* data, size_t size);
float
u32_jaccard_distance_avx512(const char*, const char*, size_t size, size_t);
void
u32_jaccard_distance_batch_4_avx512(const char*, const char*, const char*, const char*, const char*, size_t, size_t,
                                    float&, float&, float&, float&);
float
u64_jaccard_distance_avx512(const char*, const char*, size_t size, size_t);
void
u64_jaccard_distance_batch_4_avx512(const char*, const char*, const char*, const char*, const char*, size_t, size_t,
                                    float&, float&, float&, float&);
}  // namespace faiss
