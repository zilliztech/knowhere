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

extern float (*fvec_inner_product)(const float*, const float*, size_t);
extern float (*fvec_L2sqr)(const float*, const float*, size_t);
extern float (*fvec_L1)(const float*, const float*, size_t);
extern float (*fvec_Linf)(const float*, const float*, size_t);
extern float (*fvec_norm_L2sqr)(const float*, size_t);
extern void (*fvec_L2sqr_ny)(float*, const float*, const float*, size_t, size_t);
extern void (*fvec_inner_products_ny)(float*, const float*, const float*, size_t, size_t);
extern void (*fvec_madd)(size_t, const float*, float, const float*, float*);
extern int (*fvec_madd_and_argmin)(size_t, const float*, float, const float*, float*);

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
