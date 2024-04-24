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

// -*- c++ -*-

#include "hook.h"

#include <iostream>
#include <mutex>

#include "faiss/FaissHook.h"

#if defined(__ARM_NEON)
#include "distances_neon.h"
#endif

#if defined(__x86_64__)
#include "distances_avx.h"
#include "distances_avx512.h"
#include "distances_sse.h"
#include "instruction_set.h"
#endif

#include "distances_ref.h"
#include "knowhere/log.h"
namespace faiss {

#if defined(__x86_64__)
bool use_avx512 = true;
bool use_avx2 = true;
bool use_sse4_2 = true;
#endif

bool support_pq_fast_scan = true;

decltype(fvec_inner_product) fvec_inner_product = fvec_inner_product_ref;
decltype(fvec_L2sqr) fvec_L2sqr = fvec_L2sqr_ref;
decltype(fvec_L1) fvec_L1 = fvec_L1_ref;
decltype(fvec_Linf) fvec_Linf = fvec_Linf_ref;
decltype(fvec_norm_L2sqr) fvec_norm_L2sqr = fvec_norm_L2sqr_ref;
decltype(fvec_L2sqr_ny) fvec_L2sqr_ny = fvec_L2sqr_ny_ref;
decltype(fvec_inner_products_ny) fvec_inner_products_ny = fvec_inner_products_ny_ref;
decltype(fvec_madd) fvec_madd = fvec_madd_ref;
decltype(fvec_madd_and_argmin) fvec_madd_and_argmin = fvec_madd_and_argmin_ref;

decltype(fvec_L2sqr_ny_nearest) fvec_L2sqr_ny_nearest = fvec_L2sqr_ny_nearest_ref;
decltype(fvec_L2sqr_ny_nearest_y_transposed) fvec_L2sqr_ny_nearest_y_transposed =
    fvec_L2sqr_ny_nearest_y_transposed_ref;
decltype(fvec_L2sqr_ny_transposed) fvec_L2sqr_ny_transposed = fvec_L2sqr_ny_transposed_ref;

decltype(fvec_inner_product_batch_4) fvec_inner_product_batch_4 = fvec_inner_product_batch_4_ref;
decltype(fvec_L2sqr_batch_4) fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_ref;

decltype(ivec_inner_product) ivec_inner_product = ivec_inner_product_ref;
decltype(ivec_L2sqr) ivec_L2sqr = ivec_L2sqr_ref;

#if defined(__x86_64__)
bool
cpu_support_avx512() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.AVX512F() && instruction_set_inst.AVX512DQ() && instruction_set_inst.AVX512BW());
}

bool
cpu_support_avx2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.AVX2());
}

bool
cpu_support_sse4_2() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.SSE42());
}
#endif

void
fvec_hook(std::string& simd_type) {
    static std::mutex hook_mutex;
    std::lock_guard<std::mutex> lock(hook_mutex);
#if defined(__x86_64__)
    if (use_avx512 && cpu_support_avx512()) {
        fvec_inner_product = fvec_inner_product_avx512;
        fvec_L2sqr = fvec_L2sqr_avx512;
        fvec_L1 = fvec_L1_avx512;
        fvec_Linf = fvec_Linf_avx512;

        fvec_norm_L2sqr = fvec_norm_L2sqr_sse;
        fvec_L2sqr_ny = fvec_L2sqr_ny_sse;
        fvec_inner_products_ny = fvec_inner_products_ny_sse;
        fvec_madd = fvec_madd_avx512;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sse;

        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_avx512;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_avx512;

        ivec_inner_product = ivec_inner_product_avx512;
        ivec_L2sqr = ivec_L2sqr_avx512;

        simd_type = "AVX512";
        support_pq_fast_scan = true;
    } else if (use_avx2 && cpu_support_avx2()) {
        fvec_inner_product = fvec_inner_product_avx;
        fvec_L2sqr = fvec_L2sqr_avx;
        fvec_L1 = fvec_L1_avx;
        fvec_Linf = fvec_Linf_avx;

        fvec_norm_L2sqr = fvec_norm_L2sqr_sse;
        fvec_L2sqr_ny = fvec_L2sqr_ny_sse;
        fvec_inner_products_ny = fvec_inner_products_ny_sse;
        fvec_madd = fvec_madd_avx;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sse;

        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_avx;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_avx;

        ivec_inner_product = ivec_inner_product_avx;
        ivec_L2sqr = ivec_L2sqr_avx;

        simd_type = "AVX2";
        support_pq_fast_scan = true;
    } else if (use_sse4_2 && cpu_support_sse4_2()) {
        fvec_inner_product = fvec_inner_product_sse;
        fvec_L2sqr = fvec_L2sqr_sse;
        fvec_L1 = fvec_L1_sse;
        fvec_Linf = fvec_Linf_sse;

        fvec_norm_L2sqr = fvec_norm_L2sqr_sse;
        fvec_L2sqr_ny = fvec_L2sqr_ny_sse;
        fvec_inner_products_ny = fvec_inner_products_ny_sse;
        fvec_madd = fvec_madd_sse;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sse;

        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_ref;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_ref;

        ivec_inner_product = ivec_inner_product_sse;
        ivec_L2sqr = ivec_L2sqr_sse;

        simd_type = "SSE4_2";
        support_pq_fast_scan = false;
    } else {
        fvec_inner_product = fvec_inner_product_ref;
        fvec_L2sqr = fvec_L2sqr_ref;
        fvec_L1 = fvec_L1_ref;
        fvec_Linf = fvec_Linf_ref;

        fvec_norm_L2sqr = fvec_norm_L2sqr_ref;
        fvec_L2sqr_ny = fvec_L2sqr_ny_ref;
        fvec_inner_products_ny = fvec_inner_products_ny_ref;
        fvec_madd = fvec_madd_ref;
        fvec_madd_and_argmin = fvec_madd_and_argmin_ref;

        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_ref;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_ref;

        ivec_inner_product = ivec_inner_product_ref;
        ivec_L2sqr = ivec_L2sqr_ref;

        simd_type = "GENERIC";
        support_pq_fast_scan = false;
    }
#endif

#if defined(__ARM_NEON)
    fvec_inner_product = fvec_inner_product_neon;
    fvec_L2sqr = fvec_L2sqr_neon;
    fvec_L1 = fvec_L1_neon;
    fvec_Linf = fvec_Linf_neon;

    fvec_norm_L2sqr = fvec_norm_L2sqr_neon;
    fvec_L2sqr_ny = fvec_L2sqr_ny_neon;
    fvec_inner_products_ny = fvec_inner_products_ny_neon;
    fvec_madd = fvec_madd_neon;
    fvec_madd_and_argmin = fvec_madd_and_argmin_neon;

    ivec_inner_product = ivec_inner_product_neon;
    ivec_L2sqr = ivec_L2sqr_neon;

    simd_type = "NEON";
    support_pq_fast_scan = true;

#endif

// ToDo MG: include VSX intrinsics via distances_vsx once _ref tests succeed
#if defined(__powerpc64__) || defined(__loongarch64)
    fvec_inner_product = fvec_inner_product_ref;
    fvec_L2sqr = fvec_L2sqr_ref;
    fvec_L1 = fvec_L1_ref;
    fvec_Linf = fvec_Linf_ref;

    fvec_norm_L2sqr = fvec_norm_L2sqr_ref;
    fvec_L2sqr_ny = fvec_L2sqr_ny_ref;
    fvec_inner_products_ny = fvec_inner_products_ny_ref;
    fvec_madd = fvec_madd_ref;
    fvec_madd_and_argmin = fvec_madd_and_argmin_ref;

    ivec_inner_product = ivec_inner_product_ref;
    ivec_L2sqr = ivec_L2sqr_ref;

    simd_type = "GENERIC";
    support_pq_fast_scan = false;
#endif
}

static int init_hook_ = []() {
    std::string simd_type;
    fvec_hook(simd_type);
    faiss::sq_hook();
    return 0;
}();

}  // namespace faiss
