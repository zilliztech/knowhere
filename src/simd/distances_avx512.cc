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
#include <string>

#include "faiss/impl/platform_macros.h"

namespace faiss {

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
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        res += x[i] * y[i];
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

float
fp16_vec_L2sqr_avx512(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
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
bf16_vec_L2sqr_avx512(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
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

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_avx512_bf16_patch(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        res += x[i] * bf16_float(y[i]);
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_L2sqr_avx512(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

float
fp16_vec_inner_product_avx512(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
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
bf16_vec_inner_product_avx512(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
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

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_L2sqr_avx512_bf16_patch(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - bf16_float(y[i]);
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
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
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

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_inner_product_batch_4_avx512_bf16_patch(const float* __restrict x, const float* __restrict y0,
                                             const float* __restrict y1, const float* __restrict y2,
                                             const float* __restrict y3, const size_t d, float& dis0, float& dis1,
                                             float& dis2, float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
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

void
fp16_vec_inner_product_batch_4_avx512(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                      const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
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
    return;
}

void
bf16_vec_inner_product_batch_4_avx512(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                      const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
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
    return;
}

void
fp16_vec_L2sqr_batch_4_avx512(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                              const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
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
    return;
}

void
bf16_vec_L2sqr_batch_4_avx512(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                              const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
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
    return;
}
// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_L2sqr_batch_4_avx512(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                          const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
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

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_L2sqr_batch_4_avx512_bf16_patch(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                     const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;
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

// trust the compiler to unroll this properly
int32_t
ivec_inner_product_avx512(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;
    for (i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

// trust the compiler to unroll this properly
int32_t
ivec_L2sqr_avx512(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;
    for (i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return res;
}

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

float
fp16_vec_norm_L2sqr_avx512(const knowhere::fp16* x, size_t d) {
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

float
bf16_vec_norm_L2sqr_avx512(const knowhere::bf16* x, size_t d) {
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

}  // namespace faiss

#endif
