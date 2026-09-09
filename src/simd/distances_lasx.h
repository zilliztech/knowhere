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

#include <cstddef>

namespace faiss::cppcontrib::knowhere {

float
fvec_inner_product_lasx(const float* x, const float* y, size_t d);

float
fvec_L2sqr_lasx(const float* x, const float* y, size_t d);

float
fvec_L1_lasx(const float* x, const float* y, size_t d);

float
fvec_Linf_lasx(const float* x, const float* y, size_t d);

float
fvec_norm_L2sqr_lasx(const float* x, size_t d);

void
fvec_L2sqr_ny_lasx(float* dis, const float* x, const float* y, size_t d, size_t ny);

void
fvec_inner_products_ny_lasx(float* ip, const float* x, const float* y, size_t d, size_t ny);

void
fvec_L2sqr_ny_transposed_lasx(float* dis, const float* x, const float* y, const float* y_sqlen, size_t d,
                              size_t d_offset, size_t ny);

size_t
fvec_L2sqr_ny_nearest_lasx(float* distances_tmp_buffer, const float* x, const float* y, size_t d, size_t ny);

size_t
fvec_L2sqr_ny_nearest_y_transposed_lasx(float* distances_tmp_buffer, const float* x, const float* y,
                                        const float* y_sqlen, size_t d, size_t d_offset, size_t ny);

void
fvec_madd_lasx(size_t n, const float* a, float bf, const float* b, float* c);

int
fvec_madd_and_argmin_lasx(size_t n, const float* a, float bf, const float* b, float* c);

void
fvec_inner_product_batch_4_lasx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

void
fvec_L2sqr_batch_4_lasx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3, size_t d,
                        float& dis0, float& dis1, float& dis2, float& dis3);

}  // namespace faiss::cppcontrib::knowhere
