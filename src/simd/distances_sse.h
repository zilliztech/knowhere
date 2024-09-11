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

#ifndef DISTANCES_SSE_H
#define DISTANCES_SSE_H

#include <cstdio>
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

}  // namespace faiss

#endif /* DISTANCES_SSE_H */
