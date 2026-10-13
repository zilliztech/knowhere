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

#include <lsxintrin.h>
#include <stdint.h>

// The sparse codecs retain SSE intrinsics from their upstream implementations.
// Map only the operations they use so LoongArch builds execute native LSX.
static inline __m128i
knowhere_lsx_loadu_si128(const __m128i* ptr) {
    return __lsx_vld((void*)ptr, 0);
}

static inline void
knowhere_lsx_storeu_si128(__m128i* ptr, __m128i value) {
    __lsx_vst(value, ptr, 0);
}

static inline void
knowhere_lsx_storel_epi64(__m128i* ptr, __m128i value) {
    __lsx_vstelm_d(value, ptr, 0, 0);
}

static inline int
knowhere_lsx_movemask_epi8(__m128i value) {
    return (int)__lsx_vpickve2gr_hu(__lsx_vmskltz_b(value), 0);
}

static inline __m128i
knowhere_lsx_cvtepi8_epi32(__m128i value) {
    typedef int8_t i8x4 __attribute__((vector_size(4)));
    const i8x4 lower = __builtin_shufflevector((v16i8)value, (v16i8)value, 0, 1, 2, 3);
    return (__m128i) __builtin_convertvector(lower, v4i32);
}

static inline __m128i
knowhere_lsx_shuffle_epi8(__m128i value, __m128i mask) {
    const __m128i index = __lsx_vor_v(__lsx_vandi_b(mask, 0x0f), __lsx_vandi_b(__lsx_vslti_b(mask, 0), 0x10));
    return __lsx_vshuf_b(__lsx_vldi(0), value, index);
}

static inline __m128i
knowhere_lsx_setr_epi16(int16_t e0, int16_t e1, int16_t e2, int16_t e3, int16_t e4, int16_t e5, int16_t e6,
                        int16_t e7) {
    return (__m128i)(v8i16){e0, e1, e2, e3, e4, e5, e6, e7};
}

static inline __m128i
knowhere_lsx_setr_epi8(int8_t e0, int8_t e1, int8_t e2, int8_t e3, int8_t e4, int8_t e5, int8_t e6, int8_t e7,
                       int8_t e8, int8_t e9, int8_t e10, int8_t e11, int8_t e12, int8_t e13, int8_t e14, int8_t e15) {
    return (__m128i)(v16i8){e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15};
}

#define _mm_and_si128(a, b) __lsx_vand_v((a), (b))
#define _mm_add_epi32(a, b) __lsx_vadd_w((a), (b))
#define _mm_cvtsi128_si32(a) __lsx_vpickve2gr_w((a), 0)
#define _mm_cvtepi8_epi32(a) knowhere_lsx_cvtepi8_epi32(a)
#define _mm_lddqu_si128(ptr) knowhere_lsx_loadu_si128(ptr)
#define _mm_loadu_si128(ptr) knowhere_lsx_loadu_si128(ptr)
#define _mm_movemask_epi8(a) knowhere_lsx_movemask_epi8(a)
#define _mm_mullo_epi16(a, b) __lsx_vmul_h((a), (b))
#define _mm_or_si128(a, b) __lsx_vor_v((a), (b))
#define _mm_set1_epi16(a) __lsx_vreplgr2vr_h(a)
#define _mm_set1_epi32(a) __lsx_vreplgr2vr_w(a)
#define _mm_set1_epi8(a) __lsx_vreplgr2vr_b(a)
#define _mm_setr_epi16(...) knowhere_lsx_setr_epi16(__VA_ARGS__)
#define _mm_setr_epi8(...) knowhere_lsx_setr_epi8(__VA_ARGS__)
#define _mm_shuffle_epi8(a, b) knowhere_lsx_shuffle_epi8((a), (b))
#define _mm_shuffle_epi32(a, imm) __lsx_vshuf4i_w((a), (imm))
#define _mm_slli_epi32(a, imm) __lsx_vslli_w((a), (imm))
#define _mm_slli_epi64(a, imm) __lsx_vslli_d((a), (imm))
#define _mm_slli_si128(a, imm) __lsx_vbsll_v((a), (imm))
#define _mm_srli_epi16(a, imm) __lsx_vsrli_h((a), (imm))
#define _mm_srli_epi32(a, imm) __lsx_vsrli_w((a), (imm))
#define _mm_srli_epi64(a, imm) __lsx_vsrli_d((a), (imm))
#define _mm_srli_si128(a, imm) __lsx_vbsrl_v((a), (imm))
#define _mm_storel_epi64(ptr, a) knowhere_lsx_storel_epi64((ptr), (a))
#define _mm_storeu_si128(ptr, a) knowhere_lsx_storeu_si128((ptr), (a))
