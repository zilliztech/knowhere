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

#include "hook.h"

#include <mutex>

#include "faiss/FaissHook.h"

#if defined(__x86_64__)
#include "distances_avx.h"
#include "distances_avx512.h"
#include "distances_avx512icx.h"
#include "distances_sse.h"
#include "instruction_set.h"
#endif

#if defined(__ARM_NEON)
#include "distances_neon.h"
#endif

#if defined(__riscv_vector)
#include "distances_rvv.h"
#endif

#if defined(__ARM_FEATURE_SVE)
#include "distances_sve.h"
#endif

#if defined(__powerpc64__)
#include "distances_powerpc.h"
#endif

#if defined(__aarch64__) && !defined(__APPLE__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#include "distances_ref.h"

namespace faiss {

#if defined(__x86_64__)
bool use_avx512 = true;
bool use_avx2 = true;
bool use_sse4_2 = true;
#endif

bool support_pq_fast_scan = true;

///////////////////////////////////////////////////////////////////////////////
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

// for hnsw sq, obsolete
decltype(ivec_inner_product) ivec_inner_product = ivec_inner_product_ref;
decltype(ivec_L2sqr) ivec_L2sqr = ivec_L2sqr_ref;

// fp16
decltype(fp16_vec_L2sqr) fp16_vec_L2sqr = fp16_vec_L2sqr_ref;
decltype(fp16_vec_inner_product) fp16_vec_inner_product = fp16_vec_inner_product_ref;
decltype(fp16_vec_norm_L2sqr) fp16_vec_norm_L2sqr = fp16_vec_norm_L2sqr_ref;

decltype(fp16_vec_inner_product_batch_4) fp16_vec_inner_product_batch_4 = fp16_vec_inner_product_batch_4_ref;
decltype(fp16_vec_L2sqr_batch_4) fp16_vec_L2sqr_batch_4 = fp16_vec_L2sqr_batch_4_ref;

// bf16
decltype(bf16_vec_L2sqr) bf16_vec_L2sqr = bf16_vec_L2sqr_ref;
decltype(bf16_vec_inner_product) bf16_vec_inner_product = bf16_vec_inner_product_ref;
decltype(bf16_vec_norm_L2sqr) bf16_vec_norm_L2sqr = bf16_vec_norm_L2sqr_ref;

decltype(bf16_vec_inner_product_batch_4) bf16_vec_inner_product_batch_4 = bf16_vec_inner_product_batch_4_ref;
decltype(bf16_vec_L2sqr_batch_4) bf16_vec_L2sqr_batch_4 = bf16_vec_L2sqr_batch_4_ref;

// int8
decltype(int8_vec_L2sqr) int8_vec_L2sqr = int8_vec_L2sqr_ref;
decltype(int8_vec_inner_product) int8_vec_inner_product = int8_vec_inner_product_ref;
decltype(int8_vec_norm_L2sqr) int8_vec_norm_L2sqr = int8_vec_norm_L2sqr_ref;

decltype(int8_vec_inner_product_batch_4) int8_vec_inner_product_batch_4 = int8_vec_inner_product_batch_4_ref;
decltype(int8_vec_L2sqr_batch_4) int8_vec_L2sqr_batch_4 = int8_vec_L2sqr_batch_4_ref;

// rabitq
decltype(fvec_masked_sum) fvec_masked_sum = fvec_masked_sum_ref;
decltype(rabitq_dp_popcnt) rabitq_dp_popcnt = rabitq_dp_popcnt_ref;

// minhash
decltype(u64_binary_search_eq) u64_binary_search_eq = u64_binary_search_eq_ref;
decltype(u64_binary_search_ge) u64_binary_search_ge = u64_binary_search_ge_ref;
decltype(calculate_hash) calculate_hash = calculate_hash_ref;
decltype(u32_jaccard_distance) u32_jaccard_distance = u32_jaccard_distance_ref;
decltype(u32_jaccard_distance_batch_4) u32_jaccard_distance_batch_4 = u32_jaccard_distance_batch_4_ref;
decltype(u64_jaccard_distance) u64_jaccard_distance = u64_jaccard_distance_ref;
decltype(u64_jaccard_distance_batch_4) u64_jaccard_distance_batch_4 = u64_jaccard_distance_batch_4_ref;
///////////////////////////////////////////////////////////////////////////////
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

bool
cpu_support_f16c() {
    InstructionSet& instruction_set_inst = InstructionSet::GetInstance();
    return (instruction_set_inst.F16C());
}
#endif

#if defined(__aarch64__)
#if defined(__APPLE__)
bool
supports_sve() {
    return false;
}
#else
bool
supports_sve() {
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & HWCAP_SVE) != 0;
}
#endif
#endif

static std::mutex patch_bf16_mutex;

void
enable_patch_for_fp32_bf16() {
    std::lock_guard<std::mutex> lock(patch_bf16_mutex);
#if defined(__x86_64__)
    if (use_avx512 && cpu_support_avx512()) {
        // Cloud branch
        fvec_inner_product = fvec_inner_product_bf16_patch_avx512;
        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_bf16_patch_avx512;

        fvec_L2sqr = fvec_L2sqr_bf16_patch_avx512;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_bf16_patch_avx512;
    } else if (use_avx2 && cpu_support_avx2()) {
        fvec_inner_product = fvec_inner_product_bf16_patch_avx;
        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_bf16_patch_avx;

        fvec_L2sqr = fvec_L2sqr_bf16_patch_avx;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_bf16_patch_avx;
    } else if (use_sse4_2 && cpu_support_sse4_2()) {
        // The branch that can't be reached
    } else {
        fvec_inner_product = fvec_inner_product_bf16_patch_ref;
        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_bf16_patch_ref;

        fvec_L2sqr = fvec_L2sqr_bf16_patch_ref;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_bf16_patch_ref;
    }
#endif

#if defined(__aarch64__)

#if defined(__ARM_NEON) && !defined(__ARM_FEATURE_SVE)

    fvec_inner_product = fvec_inner_product_bf16_patch_neon;
    fvec_inner_product_batch_4 = fvec_inner_product_batch_4_bf16_patch_neon;

    fvec_L2sqr = fvec_L2sqr_bf16_patch_neon;
    fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_bf16_patch_neon;

#endif

#endif
}

void
disable_patch_for_fp32_bf16() {
    std::lock_guard<std::mutex> lock(patch_bf16_mutex);
#if defined(__x86_64__)
    if (use_avx512 && cpu_support_avx512()) {
        // Cloud branch
        fvec_inner_product = fvec_inner_product_avx512;
        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_avx512;

        fvec_L2sqr = fvec_L2sqr_avx512;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_avx512;
    } else if (use_avx2 && cpu_support_avx2()) {
        fvec_inner_product = fvec_inner_product_avx;
        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_avx;

        fvec_L2sqr = fvec_L2sqr_avx;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_avx;
    } else if (use_sse4_2 && cpu_support_sse4_2()) {
        // The branch that can't be reached
    } else {
        fvec_inner_product = fvec_inner_product_ref;
        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_ref;

        fvec_L2sqr = fvec_L2sqr_ref;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_ref;
    }
#endif
}

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

        fvec_norm_L2sqr = fvec_norm_L2sqr_avx512;
        fvec_L2sqr_ny = fvec_L2sqr_ny_avx;
        fvec_inner_products_ny = fvec_inner_products_ny_sse;
        fvec_madd = fvec_madd_avx512;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sse;

        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_avx512;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_avx512;
        fvec_L2sqr_ny_nearest = fvec_L2sqr_ny_nearest_avx;  // avx2 compute small dim faster than avx512

        // for hnsw sq, obsolete
        ivec_inner_product = ivec_inner_product_avx512;
        ivec_L2sqr = ivec_L2sqr_avx512;

        // fp16
        fp16_vec_inner_product = fp16_vec_inner_product_avx512;
        fp16_vec_L2sqr = fp16_vec_L2sqr_avx512;
        fp16_vec_norm_L2sqr = fp16_vec_norm_L2sqr_avx512;

        fp16_vec_inner_product_batch_4 = fp16_vec_inner_product_batch_4_avx512;
        fp16_vec_L2sqr_batch_4 = fp16_vec_L2sqr_batch_4_avx512;

        // bf16
        bf16_vec_inner_product = bf16_vec_inner_product_avx512;
        bf16_vec_L2sqr = bf16_vec_L2sqr_avx512;
        bf16_vec_norm_L2sqr = bf16_vec_norm_L2sqr_avx512;

        bf16_vec_inner_product_batch_4 = bf16_vec_inner_product_batch_4_avx512;
        bf16_vec_L2sqr_batch_4 = bf16_vec_L2sqr_batch_4_avx512;

        // int8
        int8_vec_inner_product = int8_vec_inner_product_avx512;
        int8_vec_L2sqr = int8_vec_L2sqr_avx512;
        int8_vec_norm_L2sqr = int8_vec_norm_L2sqr_avx512;

        int8_vec_inner_product_batch_4 = int8_vec_inner_product_batch_4_avx512;
        int8_vec_L2sqr_batch_4 = int8_vec_L2sqr_batch_4_avx512;

        // rabitq
        fvec_masked_sum = fvec_masked_sum_avx512;
        if (InstructionSet::GetInstance().AVX512VPOPCNTDQ()) {
            rabitq_dp_popcnt = rabitq_dp_popcnt_avx512icx;
        } else {
            rabitq_dp_popcnt = rabitq_dp_popcnt_avx512;
        }
        // minhash
        u64_binary_search_eq = u64_binary_search_eq_avx512;
        u64_binary_search_ge = u64_binary_search_ge_avx512;
        calculate_hash = calculate_hash_avx512;
        u32_jaccard_distance = u32_jaccard_distance_ref;
        u32_jaccard_distance_batch_4 = u32_jaccard_distance_batch_4_ref;
        u64_jaccard_distance = u64_jaccard_distance_ref;
        u64_jaccard_distance_batch_4 = u64_jaccard_distance_batch_4_ref;
        //
        simd_type = "AVX512";
        support_pq_fast_scan = true;
    } else if (use_avx2 && cpu_support_avx2()) {
        fvec_inner_product = fvec_inner_product_avx;
        fvec_L2sqr = fvec_L2sqr_avx;
        fvec_L1 = fvec_L1_avx;
        fvec_Linf = fvec_Linf_avx;

        fvec_norm_L2sqr = fvec_norm_L2sqr_avx;
        fvec_L2sqr_ny = fvec_L2sqr_ny_avx;
        fvec_inner_products_ny = fvec_inner_products_ny_sse;
        fvec_madd = fvec_madd_avx;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sse;

        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_avx;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_avx;
        fvec_L2sqr_ny_nearest = fvec_L2sqr_ny_nearest_avx;

        // for hnsw sq, obsolete
        ivec_inner_product = ivec_inner_product_avx;
        ivec_L2sqr = ivec_L2sqr_avx;

        // fp16
        fp16_vec_inner_product = fp16_vec_inner_product_avx;
        fp16_vec_L2sqr = fp16_vec_L2sqr_avx;
        fp16_vec_norm_L2sqr = fp16_vec_norm_L2sqr_avx;

        fp16_vec_inner_product_batch_4 = fp16_vec_inner_product_batch_4_avx;
        fp16_vec_L2sqr_batch_4 = fp16_vec_L2sqr_batch_4_avx;

        // bf16
        bf16_vec_inner_product = bf16_vec_inner_product_avx;
        bf16_vec_L2sqr = bf16_vec_L2sqr_avx;
        bf16_vec_norm_L2sqr = bf16_vec_norm_L2sqr_avx;

        bf16_vec_inner_product_batch_4 = bf16_vec_inner_product_batch_4_avx;
        bf16_vec_L2sqr_batch_4 = bf16_vec_L2sqr_batch_4_avx;

        // int8
        int8_vec_inner_product = int8_vec_inner_product_avx;
        int8_vec_L2sqr = int8_vec_L2sqr_avx;
        int8_vec_norm_L2sqr = int8_vec_norm_L2sqr_avx;

        int8_vec_inner_product_batch_4 = int8_vec_inner_product_batch_4_avx;
        int8_vec_L2sqr_batch_4 = int8_vec_L2sqr_batch_4_avx;

        // rabitq
        fvec_masked_sum = fvec_masked_sum_avx;
        rabitq_dp_popcnt = rabitq_dp_popcnt_avx;

        //
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

        // for hnsw sq, obsolete
        ivec_inner_product = ivec_inner_product_sse;
        ivec_L2sqr = ivec_L2sqr_sse;

        // fp16
        fp16_vec_inner_product = fp16_vec_inner_product_ref;
        fp16_vec_L2sqr = fp16_vec_L2sqr_ref;
        fp16_vec_norm_L2sqr = fp16_vec_norm_L2sqr_ref;

        fp16_vec_inner_product_batch_4 = fp16_vec_inner_product_batch_4_ref;
        fp16_vec_L2sqr_batch_4 = fp16_vec_L2sqr_batch_4_ref;

        // bf16
        bf16_vec_inner_product = bf16_vec_inner_product_sse;
        bf16_vec_L2sqr = bf16_vec_L2sqr_sse;
        bf16_vec_norm_L2sqr = bf16_vec_norm_L2sqr_sse;

        bf16_vec_inner_product_batch_4 = bf16_vec_inner_product_batch_4_ref;
        bf16_vec_L2sqr_batch_4 = bf16_vec_L2sqr_batch_4_ref;

        // int8
        int8_vec_inner_product = int8_vec_inner_product_sse;
        int8_vec_L2sqr = int8_vec_L2sqr_sse;
        int8_vec_norm_L2sqr = int8_vec_norm_L2sqr_sse;

        int8_vec_inner_product_batch_4 = int8_vec_inner_product_batch_4_ref;
        int8_vec_L2sqr_batch_4 = int8_vec_L2sqr_batch_4_ref;

        // rabitq
        fvec_masked_sum = fvec_masked_sum_sse;
        rabitq_dp_popcnt = rabitq_dp_popcnt_sse;

        //
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

        // for hnsw sq, obsolete
        ivec_inner_product = ivec_inner_product_ref;
        ivec_L2sqr = ivec_L2sqr_ref;

        // fp16
        fp16_vec_inner_product = fp16_vec_inner_product_ref;
        fp16_vec_L2sqr = fp16_vec_L2sqr_ref;
        fp16_vec_norm_L2sqr = fp16_vec_norm_L2sqr_ref;

        fp16_vec_inner_product_batch_4 = fp16_vec_inner_product_batch_4_ref;
        fp16_vec_L2sqr_batch_4 = fp16_vec_L2sqr_batch_4_ref;

        // bf16
        bf16_vec_inner_product = bf16_vec_inner_product_ref;
        bf16_vec_L2sqr = bf16_vec_L2sqr_ref;
        bf16_vec_norm_L2sqr = bf16_vec_norm_L2sqr_ref;

        bf16_vec_inner_product_batch_4 = bf16_vec_inner_product_batch_4_ref;
        bf16_vec_L2sqr_batch_4 = bf16_vec_L2sqr_batch_4_ref;

        // int8
        int8_vec_inner_product = int8_vec_inner_product_ref;
        int8_vec_L2sqr = int8_vec_L2sqr_ref;
        int8_vec_norm_L2sqr = int8_vec_norm_L2sqr_ref;

        int8_vec_inner_product_batch_4 = int8_vec_inner_product_batch_4_ref;
        int8_vec_L2sqr_batch_4 = int8_vec_L2sqr_batch_4_ref;

        // rabitq
        fvec_masked_sum = fvec_masked_sum_ref;
        rabitq_dp_popcnt = rabitq_dp_popcnt_ref;

        //
        simd_type = "GENERIC";
        support_pq_fast_scan = false;
    }
#endif

#if defined(__aarch64__)
    if (supports_sve()) {
#if defined(__ARM_FEATURE_SVE)
        // ToDo: Enable remaining functions on SVE
        fvec_L2sqr = fvec_L2sqr_sve;
        fvec_L1 = fvec_L1_sve;
        fvec_Linf = fvec_Linf_sve;
        fvec_norm_L2sqr = fvec_norm_L2sqr_sve;
        fvec_madd = fvec_madd_sve;
        fvec_madd_and_argmin = fvec_madd_and_argmin_sve;

        fvec_inner_product = fvec_inner_product_sve;
        fvec_L2sqr_ny = fvec_L2sqr_ny_sve;
        fvec_inner_products_ny = fvec_inner_products_ny_sve;

        ivec_inner_product = ivec_inner_product_neon;
        ivec_L2sqr = ivec_L2sqr_neon;

        // fp16
        fp16_vec_inner_product = fp16_vec_inner_product_sve;
        fp16_vec_L2sqr = fp16_vec_L2sqr_sve;
        fp16_vec_norm_L2sqr = fp16_vec_norm_L2sqr_sve;

        // bf16
        bf16_vec_inner_product = bf16_vec_inner_product_sve;
        bf16_vec_L2sqr = bf16_vec_L2sqr_sve;
        bf16_vec_norm_L2sqr = bf16_vec_norm_L2sqr_sve;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_sve;
        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_sve;

        bf16_vec_L2sqr_batch_4 = bf16_vec_L2sqr_batch_4_sve;
        bf16_vec_inner_product_batch_4 = bf16_vec_inner_product_batch_4_sve;

        // int8
        int8_vec_L2sqr = int8_vec_L2sqr_sve;
        int8_vec_norm_L2sqr = int8_vec_norm_L2sqr_sve;
        int8_vec_L2sqr_batch_4 = int8_vec_L2sqr_batch_4_sve;
        int8_vec_inner_product = int8_vec_inner_product_sve;
        int8_vec_inner_product_batch_4 = int8_vec_inner_product_batch_4_sve;

        simd_type = "SVE";
        support_pq_fast_scan = true;
#endif
    } else {
#if defined(__ARM_NEON)
        // NEON functions
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

        fvec_inner_product_batch_4 = fvec_inner_product_batch_4_neon;
        fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_neon;

        ivec_inner_product = ivec_inner_product_neon;
        ivec_L2sqr = ivec_L2sqr_neon;

        // fp16
        fp16_vec_inner_product = fp16_vec_inner_product_neon;
        fp16_vec_L2sqr = fp16_vec_L2sqr_neon;
        fp16_vec_norm_L2sqr = fp16_vec_norm_L2sqr_neon;

        fp16_vec_inner_product_batch_4 = fp16_vec_inner_product_batch_4_neon;
        fp16_vec_L2sqr_batch_4 = fp16_vec_L2sqr_batch_4_neon;

        // bf16
        bf16_vec_inner_product = bf16_vec_inner_product_neon;
        bf16_vec_L2sqr = bf16_vec_L2sqr_neon;
        bf16_vec_norm_L2sqr = bf16_vec_norm_L2sqr_neon;

        bf16_vec_inner_product_batch_4 = bf16_vec_inner_product_batch_4_neon;
        bf16_vec_L2sqr_batch_4 = bf16_vec_L2sqr_batch_4_neon;

        //
        simd_type = "NEON";
        support_pq_fast_scan = true;
#endif
    }
#endif

#if defined(__riscv_vector)
    fvec_inner_product = fvec_inner_product_rvv;
    fvec_L1 = fvec_L1_rvv;
    fvec_Linf = fvec_Linf_rvv;

    fvec_L2sqr = fvec_L2sqr_rvv;
    fvec_inner_products_ny = fvec_inner_products_ny_rvv;
    fvec_inner_product_batch_4 = fvec_inner_product_batch_4_rvv;
    fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_rvv;

    fvec_norm_L2sqr = fvec_norm_L2sqr_rvv;
    fvec_L2sqr_ny = fvec_L2sqr_ny_rvv;
    fvec_madd = fvec_madd_rvv;
    fvec_madd_and_argmin = fvec_madd_and_argmin_rvv;

    fvec_inner_product_bf16_patch = fvec_inner_product_bf16_patch_rvv;
    fvec_L2sqr_bf16_patch = fvec_L2sqr_bf16_patch_rvv;

    ivec_inner_product = ivec_inner_product_rvv;
    ivec_L2sqr = ivec_L2sqr_rvv;

    int8_vec_inner_product = int8_vec_inner_product_rvv;
    int8_vec_L2sqr = int8_vec_L2sqr_rvv;
    int8_vec_norm_L2sqr = int8_vec_norm_L2sqr_rvv;
    int8_vec_inner_product_batch_4 = int8_vec_inner_product_batch_4_rvv;
    int8_vec_L2sqr_batch_4 = int8_vec_L2sqr_batch_4_rvv;

    fp16_vec_inner_product = fp16_vec_inner_product_rvv;
    fp16_vec_L2sqr = fp16_vec_L2sqr_rvv;
    fp16_vec_norm_L2sqr = fp16_vec_norm_L2sqr_rvv;

    bf16_vec_inner_product = bf16_vec_inner_product_rvv;
    bf16_vec_L2sqr = bf16_vec_L2sqr_rvv;
    bf16_vec_norm_L2sqr = bf16_vec_norm_L2sqr_rvv;
    bf16_vec_inner_product_batch_4 = bf16_vec_inner_product_batch_4_rvv;
    bf16_vec_L2sqr_batch_4 = bf16_vec_L2sqr_batch_4_rvv;

    simd_type = "RVV";
    support_pq_fast_scan = false;
#endif

// ToDo MG: include VSX intrinsics via distances_vsx once _ref tests succeed
#if defined(__powerpc64__)
    fvec_inner_product = fvec_inner_product_ppc;
    fvec_L1 = fvec_L1_ppc;
    fvec_Linf = fvec_Linf_ppc;

    fvec_L2sqr = fvec_L2sqr_ppc;
    fvec_L2sqr_ny_nearest = fvec_L2sqr_ny_nearest_ppc;
    fvec_L2sqr_ny_transposed = fvec_L2sqr_ny_transposed_ppc;
    fvec_inner_products_ny = fvec_inner_products_ny_ppc;
    fvec_inner_product_batch_4 = fvec_inner_product_batch_4_ppc;
    fvec_L2sqr_batch_4 = fvec_L2sqr_batch_4_ppc;

    fvec_norm_L2sqr = fvec_norm_L2sqr_ppc;
    fvec_L2sqr_ny = fvec_L2sqr_ny_ppc;
    fvec_inner_products_ny = fvec_inner_products_ny_ppc;
    fvec_madd = fvec_madd_ppc;
    fvec_madd_and_argmin = fvec_madd_and_argmin_ppc;

    // for hnsw sq, obsolete
    ivec_inner_product = ivec_inner_product_ppc;
    ivec_L2sqr = ivec_L2sqr_ppc;

    //
    simd_type = "PPC";
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
