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

#ifndef HOOK_H
#define HOOK_H

#include <string>
namespace faiss {

/// inner product
extern float (*fvec_inner_product)(const float*, const float*, size_t);

/// Squared L2 distance between two vectors
extern float (*fvec_L2sqr)(const float*, const float*, size_t);

/// L1 distance
extern float (*fvec_L1)(const float*, const float*, size_t);

/// infinity distance
extern float (*fvec_Linf)(const float*, const float*, size_t);

/// squared norm of a vector
extern float (*fvec_norm_L2sqr)(const float*, size_t);

/// compute ny square L2 distance between x and a set of contiguous y vectors
extern void (*fvec_L2sqr_ny)(float*, const float*, const float*, size_t, size_t);

/// compute the inner product between nx vectors x and one y
extern void (*fvec_inner_products_ny)(float*, const float*, const float*, size_t, size_t);

/// compute ny square L2 distance between x and a set of transposed contiguous
/// y vectors. squared lengths of y should be provided as well
/// todo aguzhva: bring non-ref versions
extern void (*fvec_L2sqr_ny_transposed)(float*, const float*, const float*, const float*, size_t, size_t, size_t);

/// compute ny square L2 distance between x and a set of contiguous y vectors
/// and return the index of the nearest vector.
/// return 0 if ny == 0.
/// todo aguzhva: bring non-ref versions
extern size_t (*fvec_L2sqr_ny_nearest)(float*, const float*, const float*, size_t, size_t);

/// compute ny square L2 distance between x and a set of transposed contiguous
/// y vectors and return the index of the nearest vector.
/// squared lengths of y should be provided as well
/// return 0 if ny == 0.
/// todo aguzhva: bring non-ref versions
extern size_t (*fvec_L2sqr_ny_nearest_y_transposed)(float*, const float*, const float*, const float*, size_t, size_t,
                                                    size_t);

extern void (*fvec_madd)(size_t, const float*, float, const float*, float*);
extern int (*fvec_madd_and_argmin)(size_t, const float*, float, const float*, float*);

/// Special version of inner product that computes 4 distances
/// between x and yi, which is performance oriented.
/// todo aguzhva: bring non-ref versions
extern void (*fvec_inner_product_batch_4)(const float*, const float*, const float*, const float*, const float*,
                                          const size_t, float&, float&, float&, float&);

/// Special version of L2sqr that computes 4 distances
/// between x and yi, which is performance oriented.
/// todo aguzhva: bring non-ref versions
extern void (*fvec_L2sqr_batch_4)(const float*, const float*, const float*, const float*, const float*, const size_t,
                                  float&, float&, float&, float&);

extern int32_t (*ivec_inner_product)(const int8_t*, const int8_t*, size_t);

extern int32_t (*ivec_L2sqr)(const int8_t*, const int8_t*, size_t);

extern void (*fvec_inner_product_batch)(uint32_t xrow, uint32_t xcol, uint32_t yrow, uint32_t ycol,
                                        float* in_f32_1, float* in_f32_2, float** out_f32);

#if defined(__x86_64__)
extern bool use_avx512;
extern bool use_avx2;
extern bool use_sse4_2;
#endif

extern bool support_pq_fast_scan;

#if defined(__x86_64__)
bool
cpu_support_avx512();
bool
cpu_support_avx2();
bool
cpu_support_sse4_2();
#endif

void
enable_patch_for_fp32_bf16();

void
disable_patch_for_fp32_bf16();

void
fvec_hook(std::string&);

}  // namespace faiss

#endif /* HOOK_H */
