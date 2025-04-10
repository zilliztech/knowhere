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

#include <cstdint>
#include <cstdio>

#include "knowhere/operands.h"

namespace faiss {

/// Squared L2 distance between two vectors
float
fvec_L2sqr_sse(const float* x, const float* y, size_t d);

/// inner product
float
fvec_inner_product_sse(const float* x, const float* y, size_t d);

/// L1 distance
float
fvec_L1_sse(const float* x, const float* y, size_t d);

/// infinity distance
float
fvec_Linf_sse(const float* x, const float* y, size_t d);

float
fvec_norm_L2sqr_sse(const float* x, size_t d);

void
fvec_L2sqr_ny_sse(float* dis, const float* x, const float* y, size_t d, size_t ny);

void
fvec_inner_products_ny_sse(float* ip, const float* x, const float* y, size_t d, size_t ny);

void
fvec_madd_sse(size_t n, const float* a, float bf, const float* b, float* c);

int
fvec_madd_and_argmin_sse(size_t n, const float* a, float bf, const float* b, float* c);

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete

int32_t
ivec_inner_product_sse(const int8_t* x, const int8_t* y, size_t d);

int32_t
ivec_L2sqr_sse(const int8_t* x, const int8_t* y, size_t d);

///////////////////////////////////////////////////////////////////////////////
// bf16

float
bf16_vec_inner_product_sse(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_L2sqr_sse(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_norm_L2sqr_sse(const knowhere::bf16* x, size_t d);

///////////////////////////////////////////////////////////////////////////////
// int8

float
int8_vec_inner_product_sse(const int8_t* x, const int8_t* y, size_t d);

float
int8_vec_L2sqr_sse(const int8_t* x, const int8_t* y, size_t d);

float
int8_vec_norm_L2sqr_sse(const int8_t* x, size_t d);

///////////////////////////////////////////////////////////////////////////////
// rabitq
float
fvec_masked_sum_sse(const float* q, const uint8_t* x, const size_t d);
int
rabitq_dp_popcnt_sse(const uint8_t* q, const uint8_t* x, const size_t d, const size_t nb);

}  // namespace faiss
