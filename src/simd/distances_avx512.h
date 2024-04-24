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

#ifndef DISTANCES_AVX512_H
#define DISTANCES_AVX512_H

#include <cstddef>
#include <cstdint>

namespace faiss {

extern bool enable_avx512_patch_fp32_bf16;

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

}  // namespace faiss

#endif /* DISTANCES_AVX512_H */
