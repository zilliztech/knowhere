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

#if defined(__x86_64__)

#include "distances_avx512.h"

#include <immintrin.h>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>

#include "faiss/impl/platform_macros.h"
#include "xxhash.h"

namespace faiss::cppcontrib::knowhere {

namespace {
// reads 0 <= d < 4 floats as __m128
inline __m128
masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps(buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}

inline __m512
_mm512_bf16_to_fp32(const __m256i& x) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(x), 16));
}
}  // namespace

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_avx512(const float* x, const float* y, size_t d) {
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        res += x[i] * y[i];
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_L2sqr_avx512(const float* x, const float* y, size_t d) {
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

float
fvec_L1_avx512(const float* x, const float* y, size_t d) {
    __m512 msum0 = _mm512_setzero_ps();
    __m512 signmask0 = __m512(_mm512_set1_epi32(0x7fffffffUL));

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps(x);
        x += 16;
        __m512 my = _mm512_loadu_ps(y);
        y += 16;
        const __m512 a_m_b = mx - my;
        msum0 += _mm512_and_ps(signmask0, a_m_b);
        d -= 16;
    }

    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 += _mm512_extractf32x8_ps(msum0, 0);
    __m256 signmask1 = __m256(_mm256_set1_epi32(0x7fffffffUL));

    if (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b = mx - my;
        msum1 += _mm256_and_ps(signmask1, a_m_b);
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 += _mm256_extractf128_ps(msum1, 0);
    __m128 signmask2 = __m128(_mm_set1_epi32(0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b = mx - my;
        msum2 += _mm_and_ps(signmask2, a_m_b);
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b = mx - my;
        msum2 += _mm_and_ps(signmask2, a_m_b);
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

float
fvec_Linf_avx512(const float* x, const float* y, size_t d) {
    __m512 msum0 = _mm512_setzero_ps();
    __m512 signmask0 = __m512(_mm512_set1_epi32(0x7fffffffUL));

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps(x);
        x += 16;
        __m512 my = _mm512_loadu_ps(y);
        y += 16;
        const __m512 a_m_b = mx - my;
        msum0 = _mm512_max_ps(msum0, _mm512_and_ps(signmask0, a_m_b));
        d -= 16;
    }

    __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
    msum1 = _mm256_max_ps(msum1, _mm512_extractf32x8_ps(msum0, 0));
    __m256 signmask1 = __m256(_mm256_set1_epi32(0x7fffffffUL));

    if (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b = mx - my;
        msum1 = _mm256_max_ps(msum1, _mm256_and_ps(signmask1, a_m_b));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_max_ps(msum2, _mm256_extractf128_ps(msum1, 0));
    __m128 signmask2 = __m128(_mm_set1_epi32(0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b = mx - my;
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b = mx - my;
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
    }

    msum2 = _mm_max_ps(_mm_movehl_ps(msum2, msum2), msum2);
    msum2 = _mm_max_ps(msum2, _mm_shuffle_ps(msum2, msum2, 1));
    return _mm_cvtss_f32(msum2);
}

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_madd_avx512(size_t n, const float* a, float bf, const float* b, float* c) {
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_inner_product_batch_4_avx512(const float* __restrict x, const float* __restrict y0, const float* __restrict y1,
                                  const float* __restrict y2, const float* __restrict y3, const size_t d, float& dis0,
                                  float& dis1, float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        d0 += x[i] * y0[i];
        d1 += x[i] * y1[i];
        d2 += x[i] * y2[i];
        d3 += x[i] * y3[i];
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_L2sqr_batch_4_avx512(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                          const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - y0[i];
        const float q1 = x[i] - y1[i];
        const float q2 = x[i] - y2[i];
        const float q3 = x[i] - y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

float
fvec_norm_L2sqr_avx512(const float* x, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_loadu_ps(x);
        auto mx_1 = _mm512_loadu_ps(x + 16);
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_loadu_ps(x);
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_maskz_loadu_ps(mask, x);
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete

int32_t
ivec_inner_product_avx512(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

int32_t
ivec_L2sqr_avx512(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return res;
}

///////////////////////////////////////////////////////////////////////////////
// fp16

float
fp16_vec_inner_product_avx512(const ::knowhere::fp16* x, const ::knowhere::fp16* y, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y));
        auto mx_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(x + 16)));
        auto my_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(y + 16)));
        m512_res = _mm512_fmadd_ps(mx_0, my_0, m512_res);
        m512_res = _mm512_fmadd_ps(mx_1, my_1, m512_res);
        x += 32;
        y += 32;
        d -= 32;
    }
    if (d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y));
        m512_res = _mm512_fmadd_ps(mx, my, m512_res);
        x += 16;
        y += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        auto my = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y));
        m512_res = _mm512_fmadd_ps(mx, my, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

float
fp16_vec_L2sqr_avx512(const ::knowhere::fp16* x, const ::knowhere::fp16* y, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y));
        auto mx_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(x + 16)));
        auto my_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(y + 16)));
        mx_0 = mx_0 - my_0;
        mx_1 = mx_1 - my_1;
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        y += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y));
        mx = mx - my;
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        y += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        auto my = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y));
        mx = _mm512_sub_ps(mx, my);
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

float
fp16_vec_norm_L2sqr_avx512(const ::knowhere::fp16* x, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto mx_1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(x + 16)));
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

void
fp16_vec_inner_product_batch_4_avx512(const ::knowhere::fp16* x, const ::knowhere::fp16* y0, const ::knowhere::fp16* y1,
                                      const ::knowhere::fp16* y2, const ::knowhere::fp16* y3, const size_t d,
                                      float& dis0, float& dis1, float& dis2, float& dis3) {
    __m512 m512_res_0 = _mm512_setzero_ps();
    __m512 m512_res_1 = _mm512_setzero_ps();
    __m512 m512_res_2 = _mm512_setzero_ps();
    __m512 m512_res_3 = _mm512_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y0));
        auto my1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y1));
        auto my2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y2));
        auto my3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y3));
        m512_res_0 = _mm512_fmadd_ps(mx, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(mx, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(mx, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(mx, my3, m512_res_3);
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
        cur_d -= 16;
    }
    if (cur_d > 0) {
        const __mmask16 mask = (1U << cur_d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        auto my0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y0));
        auto my1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y1));
        auto my2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y2));
        auto my3 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y3));
        m512_res_0 = _mm512_fmadd_ps(mx, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(mx, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(mx, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(mx, my3, m512_res_3);
    }
    dis0 = _mm512_reduce_add_ps(m512_res_0);
    dis1 = _mm512_reduce_add_ps(m512_res_1);
    dis2 = _mm512_reduce_add_ps(m512_res_2);
    dis3 = _mm512_reduce_add_ps(m512_res_3);
}

void
fp16_vec_L2sqr_batch_4_avx512(const ::knowhere::fp16* x, const ::knowhere::fp16* y0, const ::knowhere::fp16* y1,
                              const ::knowhere::fp16* y2, const ::knowhere::fp16* y3, const size_t d, float& dis0,
                              float& dis1, float& dis2, float& dis3) {
    __m512 m512_res_0 = _mm512_setzero_ps();
    __m512 m512_res_1 = _mm512_setzero_ps();
    __m512 m512_res_2 = _mm512_setzero_ps();
    __m512 m512_res_3 = _mm512_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 16) {
        auto mx = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)x));
        auto my0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y0));
        auto my1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y1));
        auto my2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y2));
        auto my3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)y3));
        my0 = _mm512_sub_ps(mx, my0);
        my1 = _mm512_sub_ps(mx, my1);
        my2 = _mm512_sub_ps(mx, my2);
        my3 = _mm512_sub_ps(mx, my3);
        m512_res_0 = _mm512_fmadd_ps(my0, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(my1, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(my2, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(my3, my3, m512_res_3);
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
        cur_d -= 16;
    }
    if (cur_d > 0) {
        const __mmask16 mask = (1U << cur_d) - 1U;
        auto mx = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, x));
        auto my0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y0));
        auto my1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y1));
        auto my2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y2));
        auto my3 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, y3));
        my0 = _mm512_sub_ps(mx, my0);
        my1 = _mm512_sub_ps(mx, my1);
        my2 = _mm512_sub_ps(mx, my2);
        my3 = _mm512_sub_ps(mx, my3);
        m512_res_0 = _mm512_fmadd_ps(my0, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(my1, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(my2, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(my3, my3, m512_res_3);
    }
    dis0 = _mm512_reduce_add_ps(m512_res_0);
    dis1 = _mm512_reduce_add_ps(m512_res_1);
    dis2 = _mm512_reduce_add_ps(m512_res_2);
    dis3 = _mm512_reduce_add_ps(m512_res_3);
}

///////////////////////////////////////////////////////////////////////////////
// bf16

float
bf16_vec_inner_product_avx512(const ::knowhere::bf16* x, const ::knowhere::bf16* y, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y));
        auto mx_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(x + 16)));
        auto my_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(y + 16)));
        m512_res = _mm512_fmadd_ps(mx_0, my_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, my_1, m512_res_0);
        x += 32;
        y += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y));
        m512_res = _mm512_fmadd_ps(mx, my, m512_res);
        x += 16;
        y += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        auto my = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y));
        m512_res = _mm512_fmadd_ps(mx, my, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

float
bf16_vec_L2sqr_avx512(const ::knowhere::bf16* x, const ::knowhere::bf16* y, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y));
        auto mx_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(x + 16)));
        auto my_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(y + 16)));
        mx_0 = mx_0 - my_0;
        mx_1 = mx_1 - my_1;
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        y += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y));
        mx = mx - my;
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        y += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        auto my = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y));
        mx = _mm512_sub_ps(mx, my);
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

float
bf16_vec_norm_L2sqr_avx512(const ::knowhere::bf16* x, size_t d) {
    __m512 m512_res = _mm512_setzero_ps();
    __m512 m512_res_0 = _mm512_setzero_ps();
    while (d >= 32) {
        auto mx_0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto mx_1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)(x + 16)));
        m512_res = _mm512_fmadd_ps(mx_0, mx_0, m512_res);
        m512_res_0 = _mm512_fmadd_ps(mx_1, mx_1, m512_res_0);
        x += 32;
        d -= 32;
    }
    m512_res = m512_res + m512_res_0;
    if (d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
        x += 16;
        d -= 16;
    }
    if (d > 0) {
        const __mmask16 mask = (1U << d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        m512_res = _mm512_fmadd_ps(mx, mx, m512_res);
    }
    return _mm512_reduce_add_ps(m512_res);
}

void
bf16_vec_inner_product_batch_4_avx512(const ::knowhere::bf16* x, const ::knowhere::bf16* y0, const ::knowhere::bf16* y1,
                                      const ::knowhere::bf16* y2, const ::knowhere::bf16* y3, const size_t d,
                                      float& dis0, float& dis1, float& dis2, float& dis3) {
    __m512 m512_res_0 = _mm512_setzero_ps();
    __m512 m512_res_1 = _mm512_setzero_ps();
    __m512 m512_res_2 = _mm512_setzero_ps();
    __m512 m512_res_3 = _mm512_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y0));
        auto my1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y1));
        auto my2 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y2));
        auto my3 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y3));
        m512_res_0 = _mm512_fmadd_ps(mx, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(mx, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(mx, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(mx, my3, m512_res_3);
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
        cur_d -= 16;
    }
    if (cur_d > 0) {
        const __mmask16 mask = (1U << cur_d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        auto my0 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y0));
        auto my1 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y1));
        auto my2 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y2));
        auto my3 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y3));
        m512_res_0 = _mm512_fmadd_ps(mx, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(mx, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(mx, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(mx, my3, m512_res_3);
    }
    dis0 = _mm512_reduce_add_ps(m512_res_0);
    dis1 = _mm512_reduce_add_ps(m512_res_1);
    dis2 = _mm512_reduce_add_ps(m512_res_2);
    dis3 = _mm512_reduce_add_ps(m512_res_3);
}

void
bf16_vec_L2sqr_batch_4_avx512(const ::knowhere::bf16* x, const ::knowhere::bf16* y0, const ::knowhere::bf16* y1,
                              const ::knowhere::bf16* y2, const ::knowhere::bf16* y3, const size_t d, float& dis0,
                              float& dis1, float& dis2, float& dis3) {
    __m512 m512_res_0 = _mm512_setzero_ps();
    __m512 m512_res_1 = _mm512_setzero_ps();
    __m512 m512_res_2 = _mm512_setzero_ps();
    __m512 m512_res_3 = _mm512_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 16) {
        auto mx = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)x));
        auto my0 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y0));
        auto my1 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y1));
        auto my2 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y2));
        auto my3 = _mm512_bf16_to_fp32(_mm256_loadu_si256((__m256i*)y3));
        my0 = mx - my0;
        my1 = mx - my1;
        my2 = mx - my2;
        my3 = mx - my3;
        m512_res_0 = _mm512_fmadd_ps(my0, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(my1, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(my2, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(my3, my3, m512_res_3);
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
        cur_d -= 16;
    }
    if (cur_d > 0) {
        const __mmask16 mask = (1U << cur_d) - 1U;
        auto mx = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, x));
        auto my0 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y0));
        auto my1 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y1));
        auto my2 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y2));
        auto my3 = _mm512_bf16_to_fp32(_mm256_maskz_loadu_epi16(mask, y3));
        my0 = _mm512_sub_ps(mx, my0);
        my1 = _mm512_sub_ps(mx, my1);
        my2 = _mm512_sub_ps(mx, my2);
        my3 = _mm512_sub_ps(mx, my3);
        m512_res_0 = _mm512_fmadd_ps(my0, my0, m512_res_0);
        m512_res_1 = _mm512_fmadd_ps(my1, my1, m512_res_1);
        m512_res_2 = _mm512_fmadd_ps(my2, my2, m512_res_2);
        m512_res_3 = _mm512_fmadd_ps(my3, my3, m512_res_3);
    }
    dis0 = _mm512_reduce_add_ps(m512_res_0);
    dis1 = _mm512_reduce_add_ps(m512_res_1);
    dis2 = _mm512_reduce_add_ps(m512_res_2);
    dis3 = _mm512_reduce_add_ps(m512_res_3);
}

///////////////////////////////////////////////////////////////////////////////
// int8

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
int8_vec_inner_product_avx512(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * (int32_t)y[i];
    }
    return (float)res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
int8_vec_L2sqr_avx512(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return (float)res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
int8_vec_norm_L2sqr_avx512(const int8_t* x, size_t d) {
    int32_t res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * (int32_t)x[i];
    }
    return (float)res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
int8_vec_inner_product_batch_4_avx512(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                      const int8_t* y3, const size_t d, float& dis0, float& dis1, float& dis2,
                                      float& dis3) {
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        auto x_i = (int32_t)x[i];
        d0 += x_i * (int32_t)y0[i];
        d1 += x_i * (int32_t)y1[i];
        d2 += x_i * (int32_t)y2[i];
        d3 += x_i * (int32_t)y3[i];
    }

    dis0 = (float)d0;
    dis1 = (float)d1;
    dis2 = (float)d2;
    dis3 = (float)d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
int8_vec_L2sqr_batch_4_avx512(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                              const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        auto x_i = (int32_t)x[i];
        const int32_t q0 = x_i - (int32_t)y0[i];
        const int32_t q1 = x_i - (int32_t)y1[i];
        const int32_t q2 = x_i - (int32_t)y2[i];
        const int32_t q3 = x_i - (int32_t)y3[i];
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = (float)d0;
    dis1 = (float)d1;
    dis2 = (float)d2;
    dis3 = (float)d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

///////////////////////////////////////////////////////////////////////////////
// for cardinal

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_bf16_patch_avx512(const float* x, const float* y, size_t d) {
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        res += x[i] * bf16_float(y[i]);
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_L2sqr_bf16_patch_avx512(const float* x, const float* y, size_t d) {
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        const float tmp = x[i] - bf16_float(y[i]);
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_inner_product_batch_4_bf16_patch_avx512(const float* __restrict x, const float* __restrict y0,
                                             const float* __restrict y1, const float* __restrict y2,
                                             const float* __restrict y3, const size_t d, float& dis0, float& dis1,
                                             float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        d0 += x[i] * bf16_float(y0[i]);
        d1 += x[i] * bf16_float(y1[i]);
        d2 += x[i] * bf16_float(y2[i]);
        d3 += x[i] * bf16_float(y3[i]);
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_L2sqr_batch_4_bf16_patch_avx512(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                     const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; ++i) {
        const float q0 = x[i] - bf16_float(y0[i]);
        const float q1 = x[i] - bf16_float(y1[i]);
        const float q2 = x[i] - bf16_float(y2[i]);
        const float q3 = x[i] - bf16_float(y3[i]);
        d0 += q0 * q0;
        d1 += q1 * q1;
        d2 += q2 * q2;
        d3 += q3 * q3;
    }

    dis0 = d0;
    dis1 = d1;
    dis2 = d2;
    dis3 = d3;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

///////////////////////////////////////////////////////////////////////////////
// rabitq
float
fvec_masked_sum_avx512(const float* q, const uint8_t* x, const size_t d) {
    __m512 sum = _mm512_setzero_ps();

    const size_t d_16 = (d / 16) * 16;

    for (size_t i = 0; i < d_16; i += 16) {
        __mmask16 mask = *((const uint16_t*)(x + i / 8));
        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps(mask, q + i));
    }

    if (d != d_16) {
        const size_t leftovers = d - d_16;
        __mmask16 len_mask = (1U << leftovers) - 1;

        __mmask16 mask = 0;
        if (leftovers > 8) {
            mask = *((const uint16_t*)(x + d_16 / 8));
        } else {
            mask = *(x + d_16 / 8);
        }

        sum = _mm512_add_ps(sum, _mm512_maskz_loadu_ps(mask & len_mask, q + d_16));
    }

    return _mm512_reduce_add_ps(sum);
}

int
rabitq_dp_popcnt_avx512(const uint8_t* q, const uint8_t* x, const size_t d, const size_t nb) {
    // this is the scheme for popcount
    const size_t di_8b = (d + 7) / 8;
    const size_t di_64b = (di_8b / 8) * 8;

    int dot = 0;
    for (size_t j = 0; j < nb; j++) {
        const uint8_t* q_j = q + j * di_8b;

        // process 64-bit popcounts
        int count_dot = 0;
        for (size_t i = 0; i < di_64b; i += 8) {
            const auto qv = *(const uint64_t*)(q_j + i);
            const auto xv = *(const uint64_t*)(x + i);
            count_dot += __builtin_popcountll(qv & xv);
        }

        // process leftovers
        for (size_t i = di_64b; i < di_8b; i++) {
            const auto qv = *(q_j + i);
            const auto xv = *(x + i);
            count_dot += __builtin_popcount(qv & xv);
        }

        dot += (count_dot << j);
    }

    return dot;
}

///////////////////////////////////////////////////////////////////////////////
// minhash
int
u64_binary_search_eq_avx512(const uint64_t* arr, const size_t size, const uint64_t key) {
    if (size == 0) {
        return -1;
    }
    constexpr int CHUNK_SIZE = 8;
    const __m512i vtarget = _mm512_set1_epi64(key);

    if (size <= 32) {
        for (size_t i = 0; i + CHUNK_SIZE <= size; i += CHUNK_SIZE) {
            __m512i vdata = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&arr[i]));
            __mmask8 eq_mask = _mm512_cmpeq_epu64_mask(vdata, vtarget);
            if (eq_mask != 0) {
                return static_cast<int>(i + __builtin_ctz(eq_mask));
            }
        }

        for (size_t i = (size / CHUNK_SIZE) * CHUNK_SIZE; i < size; ++i) {
            if (arr[i] == key) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    intptr_t left = 0;
    intptr_t right = static_cast<intptr_t>(size) - 1;

    if (size > 128) {
        while (right - left + 1 >= CHUNK_SIZE * 2) {
            intptr_t mid = left + (right - left) / 2;

            intptr_t chunk_start = std::max(left, mid - CHUNK_SIZE / 2);
            chunk_start = (chunk_start / CHUNK_SIZE) * CHUNK_SIZE;

            if (chunk_start + CHUNK_SIZE <= static_cast<intptr_t>(size) && chunk_start + CHUNK_SIZE - 1 <= right) {
                __m512i vdata = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&arr[chunk_start]));

                __mmask8 eq_mask = _mm512_cmpeq_epu64_mask(vdata, vtarget);
                if (eq_mask != 0) {
                    return static_cast<int>(chunk_start + __builtin_ctz(eq_mask));
                }

                uint64_t chunk_mid_value = arr[chunk_start + CHUNK_SIZE / 2];

                if (chunk_mid_value < key) {
                    left = chunk_start + CHUNK_SIZE;
                } else {
                    right = chunk_start - 1;
                }
            } else {
                if (arr[mid] == key) {
                    return static_cast<int>(mid);
                } else if (arr[mid] < key) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
    }

    while (left <= right) {
        intptr_t mid = left + (right - left) / 2;
        if (arr[mid] == key) {
            return static_cast<int>(mid);
        } else if (arr[mid] < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}
int
u64_binary_search_ge_avx512(const uint64_t* data, const size_t size, const uint64_t target) {
    if (size == 0) {
        return -1;
    }

    constexpr int CHUNK_SIZE = 8;
    const __m512i v_target = _mm512_set1_epi64(target);
    int left = 0;
    int right = static_cast<int>(size) - 1;
    int result = -1;

    while (right - left + 1 >= CHUNK_SIZE * 4) {
        int mid = left + (right - left) / 2;

        int aligned_mid = (mid / CHUNK_SIZE) * CHUNK_SIZE;

        if (aligned_mid + CHUNK_SIZE - 1 >= static_cast<int>(size)) {
            aligned_mid = static_cast<int>(size) - CHUNK_SIZE;
        }
        if (aligned_mid < left) {
            aligned_mid = left;
        }

        if (aligned_mid + CHUNK_SIZE <= static_cast<int>(size)) {
            __m512i v_data = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&data[aligned_mid]));
            __mmask8 ge_mask = _mm512_cmpge_epu64_mask(v_data, v_target);

            if (ge_mask != 0) {
                right = aligned_mid + CHUNK_SIZE - 1;
            } else {
                left = aligned_mid + CHUNK_SIZE;
            }
        } else {
            if (data[mid] >= target) {
                result = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
    }

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (data[mid] >= target) {
            result = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return result;
}

uint64_t
calculate_hash_avx512(const char* data, size_t size) {
    return XXH3_64bits(data, size);
}

float
u32_jaccard_distance_avx512(const char* x, const char* y, size_t element_length, size_t element_size) {
    const uint32_t* u32_x = reinterpret_cast<const uint32_t*>(x);
    const uint32_t* u32_y = reinterpret_cast<const uint32_t*>(y);
    uint64_t equal_sum = 0;
    int64_t count = element_length;
    while (count - 16 > 0) {
        __m512i vec_x = _mm512_loadu_si512(u32_x);
        __m512i vec_y = _mm512_loadu_si512(u32_y);
        __mmask16 cmp_result = _mm512_cmpeq_epu32_mask(vec_x, vec_y);
        equal_sum += __builtin_popcount(static_cast<unsigned int>(cmp_result));
        count -= 16;
        u32_x += 16;
        u32_y += 16;
    }
    if (count > 0) {
        const __mmask16 mask = (1U << count) - 1U;
        auto mx = _mm512_maskz_loadu_epi32(mask, u32_x);
        auto my = _mm512_maskz_loadu_epi32(mask, u32_y);
        __mmask16 cmp_result = _mm512_cmpeq_epu32_mask(mx, my);
        equal_sum += __builtin_popcount(static_cast<unsigned int>(cmp_result & mask));
    }
    return float(equal_sum) / float(element_length);
}
void
u32_jaccard_distance_batch_4_avx512(const char* x, const char* y0, const char* y1, const char* y2, const char* y3,
                                    size_t element_length, size_t element_size, float& dis0, float& dis1, float& dis2,
                                    float& dis3) {
    const uint32_t* u32_x = reinterpret_cast<const uint32_t*>(x);
    const uint32_t* u32_y0 = reinterpret_cast<const uint32_t*>(y0);
    const uint32_t* u32_y1 = reinterpret_cast<const uint32_t*>(y1);
    const uint32_t* u32_y2 = reinterpret_cast<const uint32_t*>(y2);
    const uint32_t* u32_y3 = reinterpret_cast<const uint32_t*>(y3);
    int64_t count = element_length;
    uint32_t d0, d1, d2, d3;
    d0 = d1 = d2 = d3 = 0;
    while (count - 16 > 0) {
        __m512i vec_x = _mm512_loadu_si512(u32_x);
        __m512i vec_y0 = _mm512_loadu_si512(u32_y0);
        __m512i vec_y1 = _mm512_loadu_si512(u32_y1);
        __m512i vec_y2 = _mm512_loadu_si512(u32_y2);
        __m512i vec_y3 = _mm512_loadu_si512(u32_y3);
        __mmask16 cmp_result0 = _mm512_cmpeq_epu32_mask(vec_x, vec_y0);
        __mmask16 cmp_result1 = _mm512_cmpeq_epu32_mask(vec_x, vec_y1);
        __mmask16 cmp_result2 = _mm512_cmpeq_epu32_mask(vec_x, vec_y2);
        __mmask16 cmp_result3 = _mm512_cmpeq_epu32_mask(vec_x, vec_y3);
        d0 += __builtin_popcount(static_cast<unsigned int>(cmp_result0));
        d1 += __builtin_popcount(static_cast<unsigned int>(cmp_result1));
        d2 += __builtin_popcount(static_cast<unsigned int>(cmp_result2));
        d3 += __builtin_popcount(static_cast<unsigned int>(cmp_result3));
        count -= 16;
        u32_x += 16;
        u32_y0 += 16;
        u32_y1 += 16;
        u32_y2 += 16;
        u32_y3 += 16;
    }
    if (count > 0) {
        const __mmask16 mask = (1U << count) - 1U;
        auto mx = _mm512_maskz_loadu_epi32(mask, u32_x);
        auto my0 = _mm512_maskz_loadu_epi32(mask, u32_y0);
        auto my1 = _mm512_maskz_loadu_epi32(mask, u32_y1);
        auto my2 = _mm512_maskz_loadu_epi32(mask, u32_y2);
        auto my3 = _mm512_maskz_loadu_epi32(mask, u32_y3);
        __mmask16 cmp_result0 = _mm512_cmpeq_epu32_mask(mx, my0);
        __mmask16 cmp_result1 = _mm512_cmpeq_epu32_mask(mx, my1);
        __mmask16 cmp_result2 = _mm512_cmpeq_epu32_mask(mx, my2);
        __mmask16 cmp_result3 = _mm512_cmpeq_epu32_mask(mx, my3);
        d0 += __builtin_popcount(static_cast<unsigned int>(cmp_result0 & mask));
        d1 += __builtin_popcount(static_cast<unsigned int>(cmp_result1 & mask));
        d2 += __builtin_popcount(static_cast<unsigned int>(cmp_result2 & mask));
        d3 += __builtin_popcount(static_cast<unsigned int>(cmp_result3 & mask));
    }
    dis0 = float(d0) / float(element_length);
    dis1 = float(d1) / float(element_length);
    dis2 = float(d2) / float(element_length);
    dis3 = float(d3) / float(element_length);
}

float
u64_jaccard_distance_avx512(const char* x, const char* y, size_t element_length, size_t element_size) {
    const uint64_t* u64_x = reinterpret_cast<const uint64_t*>(x);
    const uint64_t* u64_y = reinterpret_cast<const uint64_t*>(y);
    uint64_t equal_sum = 0;
    int64_t count = element_length;
    while (count - 8 > 0) {
        __m512i vec_x = _mm512_loadu_si512(u64_x);
        __m512i vec_y = _mm512_loadu_si512(u64_y);
        __mmask8 cmp_result = _mm512_cmpeq_epu64_mask(vec_x, vec_y);
        equal_sum += __builtin_popcount(static_cast<unsigned int>(cmp_result));

        count -= 8;
        u64_x += 8;
        u64_y += 8;
    }
    if (count > 0) {
        const __mmask8 mask = (1U << count) - 1U;
        auto mx = _mm512_maskz_loadu_epi64(mask, u64_x);
        auto my = _mm512_maskz_loadu_epi64(mask, u64_y);
        __mmask8 cmp_result = _mm512_cmpeq_epu64_mask(mx, my);
        equal_sum += __builtin_popcount(static_cast<unsigned int>(cmp_result & mask));
    }
    return float(equal_sum) / element_length;
}
void
u64_jaccard_distance_batch_4_avx512(const char* x, const char* y0, const char* y1, const char* y2, const char* y3,
                                    size_t element_length, size_t element_size, float& dis0, float& dis1, float& dis2,
                                    float& dis3) {
    const uint64_t* u64_x = reinterpret_cast<const uint64_t*>(x);
    const uint64_t* u64_y0 = reinterpret_cast<const uint64_t*>(y0);
    const uint64_t* u64_y1 = reinterpret_cast<const uint64_t*>(y1);
    const uint64_t* u64_y2 = reinterpret_cast<const uint64_t*>(y2);
    const uint64_t* u64_y3 = reinterpret_cast<const uint64_t*>(y3);
    int64_t count = element_length;
    uint64_t d0, d1, d2, d3;
    d0 = d1 = d2 = d3 = 0;
    while (count - 8 > 0) {
        __m512i vec_x = _mm512_loadu_si512(u64_x);
        __m512i vec_y0 = _mm512_loadu_si512(u64_y0);
        __m512i vec_y1 = _mm512_loadu_si512(u64_y1);
        __m512i vec_y2 = _mm512_loadu_si512(u64_y2);
        __m512i vec_y3 = _mm512_loadu_si512(u64_y3);
        __mmask8 cmp_result0 = _mm512_cmpeq_epu64_mask(vec_x, vec_y0);
        __mmask8 cmp_result1 = _mm512_cmpeq_epu64_mask(vec_x, vec_y1);
        __mmask8 cmp_result2 = _mm512_cmpeq_epu64_mask(vec_x, vec_y2);
        __mmask8 cmp_result3 = _mm512_cmpeq_epu64_mask(vec_x, vec_y3);
        d0 += __builtin_popcount(static_cast<unsigned int>(cmp_result0));
        d1 += __builtin_popcount(static_cast<unsigned int>(cmp_result1));
        d2 += __builtin_popcount(static_cast<unsigned int>(cmp_result2));
        d3 += __builtin_popcount(static_cast<unsigned int>(cmp_result3));
        count -= 8;
        u64_x += 8;
        u64_y0 += 8;
        u64_y1 += 8;
        u64_y2 += 8;
        u64_y3 += 8;
    }
    if (count > 0) {
        const __mmask8 mask = (1U << count) - 1U;
        auto mx = _mm512_maskz_loadu_epi64(mask, u64_x);
        auto my0 = _mm512_maskz_loadu_epi64(mask, u64_y0);
        auto my1 = _mm512_maskz_loadu_epi64(mask, u64_y1);
        auto my2 = _mm512_maskz_loadu_epi64(mask, u64_y2);
        auto my3 = _mm512_maskz_loadu_epi64(mask, u64_y3);
        __mmask8 cmp_result0 = _mm512_cmpeq_epu64_mask(mx, my0);
        __mmask8 cmp_result1 = _mm512_cmpeq_epu64_mask(mx, my1);
        __mmask8 cmp_result2 = _mm512_cmpeq_epu64_mask(mx, my2);
        __mmask8 cmp_result3 = _mm512_cmpeq_epu64_mask(mx, my3);
        d0 += __builtin_popcount(static_cast<unsigned int>(cmp_result0 & mask));
        d1 += __builtin_popcount(static_cast<unsigned int>(cmp_result1 & mask));
        d2 += __builtin_popcount(static_cast<unsigned int>(cmp_result2 & mask));
        d3 += __builtin_popcount(static_cast<unsigned int>(cmp_result3 & mask));
    }
    dis0 = float(d0) / element_length;
    dis1 = float(d1) / element_length;
    dis2 = float(d2) / element_length;
    dis3 = float(d3) / element_length;
}

}  // namespace faiss::cppcontrib::knowhere

#endif
