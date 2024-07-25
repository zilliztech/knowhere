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

#include "distances_avx.h"

#include <immintrin.h>

#include <cassert>

#include "faiss/impl/platform_macros.h"
#include "simd_util.h"

namespace faiss {

#define ALIGNED(x) __attribute__((aligned(x)))

// reads 0 <= d < 4 floats as __m128
static inline __m128
masked_read(int d, const float* x) {
    assert(0 <= d && d < 4);
    ALIGNED(16) float buf[4] = {0, 0, 0, 0};
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

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_avx(const float* x, const float* y, size_t d) {
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
fp16_vec_inner_product_avx(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx_0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)x));
        auto my_0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y));
        auto mx_1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + 8)));
        auto my_1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + 8)));
        msum_0 = _mm256_fmadd_ps(mx_0, my_0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx_1, my_1, msum_1);
        x += 16;
        y += 16;
        d -= 16;
    }
    msum_0 = msum_0 + msum_1;
    while (d >= 8) {
        auto mx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)x));
        auto my = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y));
        msum_0 = _mm256_fmadd_ps(mx, my, msum_0);
        x += 8;
        y += 8;
        d -= 8;
    }
    if (d > 0) {
        auto mx = _mm256_cvtph_ps(mm_masked_read_short(d, (uint16_t*)x));
        auto my = _mm256_cvtph_ps(mm_masked_read_short(d, (uint16_t*)y));
        msum_0 = _mm256_fmadd_ps(mx, my, msum_0);
    }
    auto res = _mm256_reduce_add_ps(msum_0);
    return res;
}

float
bf16_vec_inner_product_avx(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx_0 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)x));
        auto my_0 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y));
        auto mx_1 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)(x + 8)));
        auto my_1 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)(y + 8)));
        msum_0 = _mm256_fmadd_ps(mx_0, my_0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx_1, my_1, msum_1);
        x += 16;
        y += 16;
        d -= 16;
    }
    msum_0 = msum_0 + msum_1;
    while (d >= 8) {
        auto mx = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)x));
        auto my = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y));
        msum_0 = _mm256_fmadd_ps(mx, my, msum_0);
        x += 8;
        y += 8;
        d -= 8;
    }
    if (d > 0) {
        auto mx = _mm256_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)x));
        auto my = _mm256_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)y));
        msum_0 = _mm256_fmadd_ps(mx, my, msum_0);
    }
    auto res = _mm256_reduce_add_ps(msum_0);
    return res;
}

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_avx_bf16_patch(const float* x, const float* y, size_t d) {
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
fvec_L2sqr_avx(const float* x, const float* y, size_t d) {
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
fp16_vec_L2sqr_avx(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx_0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)x));
        auto my_0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y));
        auto mx_1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + 8)));
        auto my_1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(y + 8)));
        mx_0 = _mm256_sub_ps(mx_0, my_0);
        mx_1 = _mm256_sub_ps(mx_1, my_1);
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx_1, mx_1, msum_1);
        x += 16;
        y += 16;
        d -= 16;
    }
    msum_0 = msum_0 + msum_1;
    while (d >= 8) {
        auto mx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)x));
        auto my = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y));
        mx = _mm256_sub_ps(mx, my);
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
        x += 8;
        y += 8;
        d -= 8;
    }
    if (d > 0) {
        auto mx = _mm256_cvtph_ps(mm_masked_read_short(d, (uint16_t*)x));
        auto my = _mm256_cvtph_ps(mm_masked_read_short(d, (uint16_t*)y));
        mx = _mm256_sub_ps(mx, my);
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
    }
    auto res = _mm256_reduce_add_ps(msum_0);
    return res;
}

float
bf16_vec_L2sqr_avx(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx_0 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)x));
        auto my_0 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y));
        auto mx_1 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)(x + 8)));
        auto my_1 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)(y + 8)));
        mx_0 = _mm256_sub_ps(mx_0, my_0);
        mx_1 = _mm256_sub_ps(mx_1, my_1);
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx_1, mx_1, msum_1);
        x += 16;
        y += 16;
        d -= 16;
    }
    msum_0 = msum_0 + msum_1;
    while (d >= 8) {
        auto mx = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)x));
        auto my = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y));
        mx = _mm256_sub_ps(mx, my);
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
        x += 8;
        y += 8;
        d -= 8;
    }
    if (d > 0) {
        auto mx = _mm256_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)x));
        auto my = _mm256_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)y));
        mx = _mm256_sub_ps(mx, my);
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
    }
    auto res = _mm256_reduce_add_ps(msum_0);
    return res;
}

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_L2sqr_avx_bf16_patch(const float* x, const float* y, size_t d) {
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
fvec_L1_avx(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();
    __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffUL));

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b = _mm256_sub_ps(mx, my);
        msum1 = _mm256_add_ps(msum1, _mm256_and_ps(signmask, a_m_b));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_add_ps(msum2, _mm256_extractf128_ps(msum1, 0));
    __m128 signmask2 = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_and_ps(signmask2, a_m_b));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b = _mm_sub_ps(mx, my);
        msum2 = _mm_add_ps(msum2, _mm_and_ps(signmask2, a_m_b));
    }

    msum2 = _mm_hadd_ps(msum2, msum2);
    msum2 = _mm_hadd_ps(msum2, msum2);
    return _mm_cvtss_f32(msum2);
}

float
fvec_Linf_avx(const float* x, const float* y, size_t d) {
    __m256 msum1 = _mm256_setzero_ps();
    __m256 signmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffffUL));

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps(x);
        x += 8;
        __m256 my = _mm256_loadu_ps(y);
        y += 8;
        const __m256 a_m_b = _mm256_sub_ps(mx, my);
        msum1 = _mm256_max_ps(msum1, _mm256_and_ps(signmask, a_m_b));
        d -= 8;
    }

    __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
    msum2 = _mm_max_ps(msum2, _mm256_extractf128_ps(msum1, 0));
    __m128 signmask2 = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffffUL));

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b = _mm_sub_ps(mx, my);
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b = _mm_sub_ps(mx, my);
        msum2 = _mm_max_ps(msum2, _mm_and_ps(signmask2, a_m_b));
    }

    msum2 = _mm_max_ps(_mm_movehl_ps(msum2, msum2), msum2);
    msum2 = _mm_max_ps(msum2, _mm_shuffle_ps(msum2, msum2, 1));
    return _mm_cvtss_f32(msum2);
}

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_madd_avx(size_t n, const float* a, float bf, const float* b, float* c) {
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_inner_product_batch_4_avx(const float* __restrict x, const float* __restrict y0, const float* __restrict y1,
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
fvec_inner_product_batch_4_avx_bf16_patch(const float* __restrict x, const float* __restrict y0,
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

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_L2sqr_batch_4_avx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
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
fvec_L2sqr_batch_4_avx_bf16_patch(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
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
ivec_inner_product_avx(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;
    for (i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

// trust the compiler to unroll this properly
int32_t
ivec_L2sqr_avx(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;
    for (i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return res;
}

float
fp16_vec_norm_L2sqr_avx(const knowhere::fp16* x, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx_0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)x));
        auto mx_1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(x + 8)));
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx_1, mx_1, msum_1);
        x += 16;
        d -= 16;
    }
    msum_0 = msum_0 + msum_1;
    while (d >= 8) {
        auto mx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)x));
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
        x += 8;
        d -= 8;
    }
    if (d > 0) {
        auto mx = _mm256_cvtph_ps(mm_masked_read_short(d, (uint16_t*)x));
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
    }
    auto res = _mm256_reduce_add_ps(msum_0);
    return res;
}

float
bf16_vec_norm_L2sqr_avx(const knowhere::bf16* x, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx_0 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)x));
        auto mx_1 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)(x + 8)));
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx_1, mx_1, msum_1);
        x += 16;
        d -= 16;
    }
    msum_0 = msum_0 + msum_1;
    while (d >= 8) {
        auto mx = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)x));
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
        x += 8;
        d -= 8;
    }
    if (d > 0) {
        auto mx = _mm256_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)x));
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
    }
    auto res = _mm256_reduce_add_ps(msum_0);
    return res;
}
}  // namespace faiss
#endif
