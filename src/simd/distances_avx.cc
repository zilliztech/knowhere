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

namespace faiss {

#define ALIGNED(x) __attribute__((aligned(x)))

namespace {
// reads 0 <= d < 4 floats as __m128
inline __m128
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

inline __m128i
mm_masked_read_short(int d, const uint16_t* x) {
    assert(0 <= d && d < 8);
    ALIGNED(16) uint16_t buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    switch (d) {
        case 7:
            buf[6] = x[6];
        case 6:
            buf[5] = x[5];
        case 5:
            buf[4] = x[4];
        case 4:
            buf[3] = x[3];
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_loadu_si128((__m128i*)buf);
}

inline __m256
_mm256_bf16_to_fp32(const __m128i& a) {
    __m256i o = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a), 16);
    return _mm256_castsi256_ps(o);
}

inline float
_mm256_reduce_add_ps(const __m256 res) {
    const __m128 sum = _mm_add_ps(_mm256_castps256_ps128(res), _mm256_extractf128_ps(res, 1));
    const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
    const __m128 v1 = _mm_add_ps(sum, v0);
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    const __m128 v3 = _mm_add_ps(v1, v2);
    return _mm_cvtss_f32(v3);
}
}  // namespace

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_avx(const float* x, const float* y, size_t d) {
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
fvec_L2sqr_avx(const float* x, const float* y, size_t d) {
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

// trust the compiler to unroll this properly
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
void
fvec_L2sqr_batch_4_avx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
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
fvec_norm_L2sqr_avx(const float* x, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx_0 = _mm256_loadu_ps(x);
        auto mx_1 = _mm256_loadu_ps(x + 8);
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        auto msum_1 = _mm256_mul_ps(mx_1, mx_1);
        msum_0 = msum_0 + msum_1;
        x += 16;
        d -= 16;
    }
    if (d >= 8) {
        auto mx = _mm256_loadu_ps(x);
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
        x += 8;
        d -= 8;
    }
    if (d > 0) {
        __m128 rest_0 = _mm_setzero_ps();
        __m128 rest_1 = _mm_setzero_ps();
        if (d >= 4) {
            rest_0 = _mm_loadu_ps(x);
            x += 4;
            d -= 4;
        }
        if (d >= 0) {
            rest_1 = masked_read(d, x);
        }
        auto mx = _mm256_set_m128(rest_0, rest_1);
        msum_0 = _mm256_fmadd_ps(mx, mx, msum_0);
    }
    auto res = _mm256_reduce_add_ps(msum_0);
    return res;
}

namespace {
FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
inline void
fvec_L2sqr_ny_avx_impl(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    size_t i = 0;
    for (; i + 3 < ny; i += 4) {
        const float* __restrict y1 = y + d * i;
        const float* __restrict y2 = y + d * (i + 1);
        const float* __restrict y3 = y + d * (i + 2);
        const float* __restrict y4 = y + d * (i + 3);
        fvec_L2sqr_batch_4_avx(x, y1, y2, y3, y4, d, dis[i], dis[i + 1], dis[i + 2], dis[i + 3]);
    }
    while (i < ny) {
        const float* __restrict y_i = y + d * i;
        dis[i] = fvec_L2sqr_avx(x, y_i, d);
        i++;
    }
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

inline void
fvec_L2sqr_ny_avx_d2_impl(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    size_t y_i = ny;
    auto mx = _mm256_setr_ps(x[0], x[1], x[0], x[1], x[0], x[1], x[0], x[1]);
    const float* y_i_addr = y;
    while (y_i >= 16) {
        auto my1 = _mm256_loadu_ps(y_i_addr);
        auto my2 = _mm256_loadu_ps(y_i_addr + 8);   // 4-th
        auto my3 = _mm256_loadu_ps(y_i_addr + 16);  // 8-th
        auto my4 = _mm256_loadu_ps(y_i_addr + 24);  // 12-th
        my1 = _mm256_sub_ps(my1, mx);
        my1 = _mm256_mul_ps(my1, my1);
        my2 = _mm256_sub_ps(my2, mx);
        my2 = _mm256_mul_ps(my2, my2);
        my3 = _mm256_sub_ps(my3, mx);
        my3 = _mm256_mul_ps(my3, my3);
        my4 = _mm256_sub_ps(my4, mx);
        my4 = _mm256_mul_ps(my4, my4);
        my1 = _mm256_hadd_ps(my1, my2);
        my3 = _mm256_hadd_ps(my3, my4);
        my1 = _mm256_permutevar8x32_ps(my1, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        my3 = _mm256_permutevar8x32_ps(my3, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        _mm256_storeu_ps(dis + (ny - y_i), my1);
        _mm256_storeu_ps(dis + (ny - y_i) + 8, my3);
        y_i_addr += 16 * d;
        y_i -= 16;
    }
    while (y_i >= 4) {
        auto my1 = _mm256_loadu_ps(y_i_addr);
        my1 = _mm256_sub_ps(my1, mx);
        my1 = _mm256_mul_ps(my1, my1);
        my1 = _mm256_hadd_ps(my1, my1);
        my1 = _mm256_permutevar8x32_ps(my1, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        __m128 high = _mm256_extractf128_ps(my1, 0);
        _mm_storeu_ps(dis + (ny - y_i), high);
        y_i_addr += 4 * d;
        y_i -= 4;
    }
    while (y_i > 0) {
        float dis1;
        dis1 = (x[0] - y_i_addr[0]) * (x[0] - y_i_addr[0]);
        dis1 += (x[1] - y_i_addr[1]) * (x[1] - y_i_addr[1]);
        dis[ny - y_i] = dis1;
        y_i_addr += d;
        y_i -= 1;
    }
}

inline void
fvec_L2sqr_ny_avx_d4_impl(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    size_t y_i = ny;
    auto mx_t = _mm_loadu_ps(x);
    auto mx = _mm256_set_m128(mx_t, mx_t);
    const float* y_i_addr = y;
    while (y_i >= 8) {
        auto my1 = _mm256_loadu_ps(y_i_addr);
        auto my2 = _mm256_loadu_ps(y_i_addr + 8);
        auto my3 = _mm256_loadu_ps(y_i_addr + 16);
        auto my4 = _mm256_loadu_ps(y_i_addr + 24);
        my1 = _mm256_sub_ps(my1, mx);
        my1 = _mm256_mul_ps(my1, my1);
        my2 = _mm256_sub_ps(my2, mx);
        my2 = _mm256_mul_ps(my2, my2);
        my3 = _mm256_sub_ps(my3, mx);
        my3 = _mm256_mul_ps(my3, my3);
        my4 = _mm256_sub_ps(my4, mx);
        my4 = _mm256_mul_ps(my4, my4);
        my1 = _mm256_hadd_ps(my1, my2);
        my3 = _mm256_hadd_ps(my3, my4);
        my1 = _mm256_hadd_ps(my1, my3);
        my1 = _mm256_permutevar8x32_ps(my1, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_ps(dis + (ny - y_i), my1);
        y_i_addr += 8 * d;
        y_i -= 8;
    }
    if (y_i >= 4) {
        auto my1 = _mm256_loadu_ps(y_i_addr);
        auto my2 = _mm256_loadu_ps(y_i_addr + 8);
        my1 = _mm256_sub_ps(my1, mx);
        my1 = _mm256_mul_ps(my1, my1);
        my2 = _mm256_sub_ps(my2, mx);
        my2 = _mm256_mul_ps(my2, my2);
        my1 = _mm256_hadd_ps(my1, my2);
        my1 = _mm256_hadd_ps(my1, my1);
        my1 = _mm256_permutevar8x32_ps(my1, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        __m128 res = _mm256_extractf128_ps(my1, 0);
        _mm_storeu_ps(dis + (ny - y_i), res);
        y_i_addr = y_i_addr + 4 * d;
        y_i -= 4;
    }
    if (y_i >= 2) {
        auto my1 = _mm256_loadu_ps(y_i_addr);
        my1 = _mm256_sub_ps(my1, mx);
        my1 = _mm256_mul_ps(my1, my1);
        __m128 high = _mm256_extractf128_ps(my1, 1);
        __m128 low = _mm256_extractf128_ps(my1, 0);
        __m128 sum_low = _mm_hadd_ps(low, low);
        sum_low = _mm_hadd_ps(sum_low, sum_low);

        __m128 sum_high = _mm_hadd_ps(high, high);
        sum_high = _mm_hadd_ps(sum_high, sum_high);

        dis[ny - y_i] = _mm_cvtss_f32(sum_low);
        dis[ny - y_i + 1] = _mm_cvtss_f32(sum_high);
        y_i_addr = y_i_addr + 2 * d;
        y_i -= 2;
    }
    if (y_i > 0) {
        float dis1, dis2;
        dis1 = (x[0] - y_i_addr[0]) * (x[0] - y_i_addr[0]);
        dis2 = (x[1] - y_i_addr[1]) * (x[1] - y_i_addr[1]);
        dis1 += (x[2] - y_i_addr[2]) * (x[2] - y_i_addr[2]);
        dis2 += (x[3] - y_i_addr[3]) * (x[3] - y_i_addr[3]);
        dis[ny - y_i] = dis1 + dis2;
    }
}
}  // namespace

void
fvec_L2sqr_ny_avx(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    // todo: add more small dim support
    if (d == 2) {
        return fvec_L2sqr_ny_avx_d2_impl(dis, x, y, d, ny);
    } else if (d == 4) {
        return fvec_L2sqr_ny_avx_d4_impl(dis, x, y, d, ny);
    } else {
        return fvec_L2sqr_ny_avx_impl(dis, x, y, d, ny);
    }
}

size_t
fvec_L2sqr_ny_nearest_avx(float* __restrict distances_tmp_buffer, const float* __restrict x, const float* __restrict y,
                          size_t d, size_t ny) {
    fvec_L2sqr_ny_avx(distances_tmp_buffer, x, y, d, ny);

    size_t nearest_idx = 0;
    float min_dis = std::numeric_limits<float>::max();

    for (size_t i = 0; i < ny; i++) {
        if (distances_tmp_buffer[i] < min_dis) {
            min_dis = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }
    return nearest_idx;
}

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete

// trust the compiler to unroll this properly
int32_t
ivec_inner_product_avx(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

// trust the compiler to unroll this properly
int32_t
ivec_L2sqr_avx(const int8_t* x, const int8_t* y, size_t d) {
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
fp16_vec_inner_product_avx(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx = _mm256_loadu_si256((__m256i*)x);
        auto mx_0 = _mm256_cvtph_ps(_mm256_extracti128_si256(mx, 0));
        auto mx_1 = _mm256_cvtph_ps(_mm256_extracti128_si256(mx, 1));

        auto my = _mm256_loadu_si256((__m256i*)y);
        auto my_0 = _mm256_cvtph_ps(_mm256_extracti128_si256(my, 0));
        auto my_1 = _mm256_cvtph_ps(_mm256_extracti128_si256(my, 1));

        msum_0 = _mm256_fmadd_ps(mx_0, my_0, msum_0);
        auto msum_1 = _mm256_mul_ps(mx_1, my_1);
        msum_0 = msum_0 + msum_1;
        x += 16;
        y += 16;
        d -= 16;
    }
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
fp16_vec_L2sqr_avx(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx = _mm256_loadu_si256((__m256i*)x);
        auto mx_0 = _mm256_cvtph_ps(_mm256_extracti128_si256(mx, 0));
        auto mx_1 = _mm256_cvtph_ps(_mm256_extracti128_si256(mx, 1));

        auto my = _mm256_loadu_si256((__m256i*)y);
        auto my_0 = _mm256_cvtph_ps(_mm256_extracti128_si256(my, 0));
        auto my_1 = _mm256_cvtph_ps(_mm256_extracti128_si256(my, 1));

        mx_0 = _mm256_sub_ps(mx_0, my_0);
        mx_1 = _mm256_sub_ps(mx_1, my_1);
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        msum_0 = _mm256_fmadd_ps(mx_1, mx_1, msum_0);

        x += 16;
        y += 16;
        d -= 16;
    }
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
fp16_vec_norm_L2sqr_avx(const knowhere::fp16* x, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx = _mm256_loadu_si256((__m256i*)x);
        auto mx_0 = _mm256_cvtph_ps(_mm256_extracti128_si256(mx, 0));
        auto mx_1 = _mm256_cvtph_ps(_mm256_extracti128_si256(mx, 1));
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        auto msum_1 = _mm256_mul_ps(mx_1, mx_1);
        msum_0 = msum_0 + msum_1;
        x += 16;
        d -= 16;
    }
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

void
fp16_vec_inner_product_batch_4_avx(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                   const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    __m256 msum_2 = _mm256_setzero_ps();
    __m256 msum_3 = _mm256_setzero_ps();

    size_t cur_d = d;
    while (cur_d >= 8) {
        auto mx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)x));
        auto my0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y0));
        auto my1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y1));
        auto my2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y2));
        auto my3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y3));
        msum_0 = _mm256_fmadd_ps(mx, my0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx, my1, msum_1);
        msum_2 = _mm256_fmadd_ps(mx, my2, msum_2);
        msum_3 = _mm256_fmadd_ps(mx, my3, msum_3);
        x += 8;
        y0 += 8;
        y1 += 8;
        y2 += 8;
        y3 += 8;
        cur_d -= 8;
    }
    if (cur_d > 0) {
        auto mx = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)x));
        auto my0 = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)y0));
        auto my1 = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)y1));
        auto my2 = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)y2));
        auto my3 = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)y3));
        msum_0 = _mm256_fmadd_ps(mx, my0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx, my1, msum_1);
        msum_2 = _mm256_fmadd_ps(mx, my2, msum_2);
        msum_3 = _mm256_fmadd_ps(mx, my3, msum_3);
    }
    dis0 = _mm256_reduce_add_ps(msum_0);
    dis1 = _mm256_reduce_add_ps(msum_1);
    dis2 = _mm256_reduce_add_ps(msum_2);
    dis3 = _mm256_reduce_add_ps(msum_3);
}

void
fp16_vec_L2sqr_batch_4_avx(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                           const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    __m256 msum_2 = _mm256_setzero_ps();
    __m256 msum_3 = _mm256_setzero_ps();
    auto cur_d = d;
    while (cur_d >= 8) {
        auto mx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)x));
        auto my0 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y0));
        auto my1 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y1));
        auto my2 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y2));
        auto my3 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)y3));
        my0 = _mm256_sub_ps(mx, my0);
        msum_0 = _mm256_fmadd_ps(my0, my0, msum_0);
        my1 = _mm256_sub_ps(mx, my1);
        msum_1 = _mm256_fmadd_ps(my1, my1, msum_1);
        my2 = _mm256_sub_ps(mx, my2);
        msum_2 = _mm256_fmadd_ps(my2, my2, msum_2);
        my3 = _mm256_sub_ps(mx, my3);
        msum_3 = _mm256_fmadd_ps(my3, my3, msum_3);
        x += 8;
        y0 += 8;
        y1 += 8;
        y2 += 8;
        y3 += 8;
        cur_d -= 8;
    }
    if (cur_d > 0) {
        auto mx = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)x));
        auto my0 = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)y0));
        auto my1 = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)y1));
        auto my2 = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)y2));
        auto my3 = _mm256_cvtph_ps(mm_masked_read_short(cur_d, (uint16_t*)y3));
        my0 = _mm256_sub_ps(mx, my0);
        my1 = _mm256_sub_ps(mx, my1);
        my2 = _mm256_sub_ps(mx, my2);
        my3 = _mm256_sub_ps(mx, my3);
        msum_0 = _mm256_fmadd_ps(my0, my0, msum_0);
        msum_1 = _mm256_fmadd_ps(my1, my1, msum_1);
        msum_2 = _mm256_fmadd_ps(my2, my2, msum_2);
        msum_3 = _mm256_fmadd_ps(my3, my3, msum_3);
    }
    dis0 = _mm256_reduce_add_ps(msum_0);
    dis1 = _mm256_reduce_add_ps(msum_1);
    dis2 = _mm256_reduce_add_ps(msum_2);
    dis3 = _mm256_reduce_add_ps(msum_3);
}

///////////////////////////////////////////////////////////////////////////////
// bf16

float
bf16_vec_inner_product_avx(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx = _mm256_loadu_si256((__m256i*)x);
        auto mx_0 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(mx, 0));
        auto mx_1 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(mx, 1));

        auto my = _mm256_loadu_si256((__m256i*)y);
        auto my_0 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(my, 0));
        auto my_1 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(my, 1));

        msum_0 = _mm256_fmadd_ps(mx_0, my_0, msum_0);
        auto msum_1 = _mm256_mul_ps(mx_1, my_1);
        msum_0 = msum_0 + msum_1;
        x += 16;
        y += 16;
        d -= 16;
    }
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

float
bf16_vec_L2sqr_avx(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx = _mm256_loadu_si256((__m256i*)x);
        auto mx_0 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(mx, 0));
        auto mx_1 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(mx, 1));

        auto my = _mm256_loadu_si256((__m256i*)y);
        auto my_0 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(my, 0));
        auto my_1 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(my, 1));
        mx_0 = _mm256_sub_ps(mx_0, my_0);
        mx_1 = _mm256_sub_ps(mx_1, my_1);
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        msum_0 = _mm256_fmadd_ps(mx_1, mx_1, msum_0);
        x += 16;
        y += 16;
        d -= 16;
    }
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

float
bf16_vec_norm_L2sqr_avx(const knowhere::bf16* x, size_t d) {
    __m256 msum_0 = _mm256_setzero_ps();
    while (d >= 16) {
        auto mx = _mm256_loadu_si256((__m256i*)x);
        auto mx_0 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(mx, 0));
        auto mx_1 = _mm256_bf16_to_fp32(_mm256_extracti128_si256(mx, 1));
        msum_0 = _mm256_fmadd_ps(mx_0, mx_0, msum_0);
        auto msum_1 = _mm256_mul_ps(mx_1, mx_1);
        msum_0 = msum_0 + msum_1;
        x += 16;
        d -= 16;
    }
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

void
bf16_vec_inner_product_batch_4_avx(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                   const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    __m256 msum_2 = _mm256_setzero_ps();
    __m256 msum_3 = _mm256_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 8) {
        auto mx = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)x));
        auto my0 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y0));
        auto my1 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y1));
        auto my2 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y2));
        auto my3 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y3));
        msum_0 = _mm256_fmadd_ps(mx, my0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx, my1, msum_1);
        msum_2 = _mm256_fmadd_ps(mx, my2, msum_2);
        msum_3 = _mm256_fmadd_ps(mx, my3, msum_3);
        x += 8;
        y0 += 8;
        y1 += 8;
        y2 += 8;
        y3 += 8;
        cur_d -= 8;
    }
    if (cur_d > 0) {
        auto mx = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)x));
        auto my0 = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)y0));
        auto my1 = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)y1));
        auto my2 = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)y2));
        auto my3 = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)y3));
        msum_0 = _mm256_fmadd_ps(mx, my0, msum_0);
        msum_1 = _mm256_fmadd_ps(mx, my1, msum_1);
        msum_2 = _mm256_fmadd_ps(mx, my2, msum_2);
        msum_3 = _mm256_fmadd_ps(mx, my3, msum_3);
    }
    dis0 = _mm256_reduce_add_ps(msum_0);
    dis1 = _mm256_reduce_add_ps(msum_1);
    dis2 = _mm256_reduce_add_ps(msum_2);
    dis3 = _mm256_reduce_add_ps(msum_3);
}

void
bf16_vec_L2sqr_batch_4_avx(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                           const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3) {
    __m256 msum_0 = _mm256_setzero_ps();
    __m256 msum_1 = _mm256_setzero_ps();
    __m256 msum_2 = _mm256_setzero_ps();
    __m256 msum_3 = _mm256_setzero_ps();
    size_t cur_d = d;
    while (cur_d >= 8) {
        auto mx = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)x));
        auto my0 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y0));
        auto my1 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y1));
        auto my2 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y2));
        auto my3 = _mm256_bf16_to_fp32(_mm_loadu_si128((__m128i*)y3));
        my0 = _mm256_sub_ps(mx, my0);
        my1 = _mm256_sub_ps(mx, my1);
        my2 = _mm256_sub_ps(mx, my2);
        my3 = _mm256_sub_ps(mx, my3);
        msum_0 = _mm256_fmadd_ps(my0, my0, msum_0);
        msum_1 = _mm256_fmadd_ps(my1, my1, msum_1);
        msum_2 = _mm256_fmadd_ps(my2, my2, msum_2);
        msum_3 = _mm256_fmadd_ps(my3, my3, msum_3);
        x += 8;
        y0 += 8;
        y1 += 8;
        y2 += 8;
        y3 += 8;
        cur_d -= 8;
    }
    if (cur_d > 0) {
        auto mx = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)x));
        auto my0 = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)y0));
        auto my1 = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)y1));
        auto my2 = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)y2));
        auto my3 = _mm256_bf16_to_fp32(mm_masked_read_short(cur_d, (uint16_t*)y3));
        my0 = _mm256_sub_ps(mx, my0);
        my1 = _mm256_sub_ps(mx, my1);
        my2 = _mm256_sub_ps(mx, my2);
        my3 = _mm256_sub_ps(mx, my3);
        msum_0 = _mm256_fmadd_ps(my0, my0, msum_0);
        msum_1 = _mm256_fmadd_ps(my1, my1, msum_1);
        msum_2 = _mm256_fmadd_ps(my2, my2, msum_2);
        msum_3 = _mm256_fmadd_ps(my3, my3, msum_3);
    }
    dis0 = _mm256_reduce_add_ps(msum_0);
    dis1 = _mm256_reduce_add_ps(msum_1);
    dis2 = _mm256_reduce_add_ps(msum_2);
    dis3 = _mm256_reduce_add_ps(msum_3);
}

///////////////////////////////////////////////////////////////////////////////
// for cardinal

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
fvec_inner_product_bf16_patch_avx(const float* x, const float* y, size_t d) {
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
fvec_L2sqr_bf16_patch_avx(const float* x, const float* y, size_t d) {
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
fvec_inner_product_batch_4_bf16_patch_avx(const float* __restrict x, const float* __restrict y0,
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
fvec_L2sqr_batch_4_bf16_patch_avx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
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

}  // namespace faiss
#endif
