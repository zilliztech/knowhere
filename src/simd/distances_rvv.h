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

#include "hook.h"
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

int
fvec_madd_and_argmin_rvv(size_t n, const float* a, float bf, const float* b, float* c);

void
fvec_inner_product_batch_4_rvv(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                               size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

void
fvec_L2sqr_batch_4_rvv(const float* x, const float* y0, const float* y1, const float* y2, const float* y3, size_t d,
                       float& dis0, float& dis1, float& dis2, float& dis3);

int32_t
ivec_inner_product_rvv(const int8_t* x, const int8_t* y, size_t d);

int32_t
ivec_L2sqr_rvv(const int8_t* x, const int8_t* y, size_t d);

float
int8_vec_inner_product_rvv(const int8_t* x, const int8_t* y, size_t d);

float
int8_vec_L2sqr_rvv(const int8_t* x, const int8_t* y, size_t d);

float
int8_vec_norm_L2sqr_rvv(const int8_t* x, size_t d);

void
int8_vec_inner_product_batch_4_rvv(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                   const int8_t* y3, size_t d, float& dis0, float& dis1, float& dis2, float& dis3);
void
int8_vec_L2sqr_batch_4_rvv(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                           size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

float
fp16_vec_inner_product_rvv(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_L2sqr_rvv(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_norm_L2sqr_rvv(const knowhere::fp16* x, size_t d);

float
bf16_vec_inner_product_rvv(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_L2sqr_rvv(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_norm_L2sqr_rvv(const knowhere::bf16* x, size_t d);

void
bf16_vec_inner_product_batch_4_rvv(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                   const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3);
void
bf16_vec_L2sqr_batch_4_rvv(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                           const knowhere::bf16* y2, const knowhere::bf16* y3, size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3);

float
fvec_inner_product_bf16_patch_rvv(const float* x, const float* y, size_t d);

float
fvec_L2sqr_bf16_patch_rvv(const float* x, const float* y, size_t d);

}  // namespace faiss
