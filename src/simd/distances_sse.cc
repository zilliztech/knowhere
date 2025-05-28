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

#include "distances_sse.h"

#include <immintrin.h>

#include <cassert>
#include <cstdint>

#include "distances_ref.h"
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

inline __m128
_mm_bf16_to_fp32(const __m128i& a) {
    auto o = _mm_slli_epi32(_mm_cvtepu16_epi32(a), 16);
    return _mm_castsi128_ps(o);
}
}  // namespace

float
fvec_inner_product_sse(const float* x, const float* y, size_t d) {
    __m128 mx, my;
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        mx = _mm_loadu_ps(x);
        x += 4;
        my = _mm_loadu_ps(y);
        y += 4;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, my));
        d -= 4;
    }

    // add the last 1, 2, or 3 values
    mx = masked_read(d, x);
    my = masked_read(d, y);
    __m128 prod = _mm_mul_ps(mx, my);

    msum1 = _mm_add_ps(msum1, prod);

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

float
fvec_L2sqr_sse(const float* x, const float* y, size_t d) {
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        __m128 mx = _mm_loadu_ps(x);
        x += 4;
        __m128 my = _mm_loadu_ps(y);
        y += 4;
        const __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
        d -= 4;
    }

    if (d > 0) {
        // add the last 1, 2 or 3 values
        __m128 mx = masked_read(d, x);
        __m128 my = masked_read(d, y);
        __m128 a_m_b1 = _mm_sub_ps(mx, my);
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(a_m_b1, a_m_b1));
    }

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

float
fvec_L1_sse(const float* x, const float* y, size_t d) {
    return fvec_L1_ref(x, y, d);
}

float
fvec_Linf_sse(const float* x, const float* y, size_t d) {
    return fvec_Linf_ref(x, y, d);
}

void
fvec_madd_sse(size_t n, const float* a, float bf, const float* b, float* c) {
    if ((n & 3) != 0 || ((((int64_t)a) | ((int64_t)b) | ((int64_t)c)) & 15) != 0) {
        fvec_madd_ref(n, a, bf, b, c);
        return;
    }

    n >>= 2;
    __m128 bf4 = _mm_set_ps1(bf);
    __m128* a4 = (__m128*)a;
    __m128* b4 = (__m128*)b;
    __m128* c4 = (__m128*)c;

    while (n--) {
        *c4 = _mm_add_ps(*a4, _mm_mul_ps(bf4, *b4));
        b4++;
        a4++;
        c4++;
    }
}

int
fvec_madd_and_argmin_sse(size_t n, const float* a, float bf, const float* b, float* c) {
    if ((n & 3) != 0 || ((((int64_t)a) | ((int64_t)b) | ((int64_t)c)) & 15) != 0) {
        return fvec_madd_and_argmin_ref(n, a, bf, b, c);
    }

    n >>= 2;
    __m128 bf4 = _mm_set_ps1(bf);
    __m128 vmin4 = _mm_set_ps1(1e20);
    __m128i imin4 = _mm_set1_epi32(-1);
    __m128i idx4 = _mm_set_epi32(3, 2, 1, 0);
    __m128i inc4 = _mm_set1_epi32(4);
    __m128* a4 = (__m128*)a;
    __m128* b4 = (__m128*)b;
    __m128* c4 = (__m128*)c;

    while (n--) {
        __m128 vc4 = _mm_add_ps(*a4, _mm_mul_ps(bf4, *b4));
        *c4 = vc4;
        __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(vmin4, vc4));
        // imin4 = _mm_blendv_epi8 (imin4, idx4, mask); // slower!

        imin4 = _mm_or_si128(_mm_and_si128(mask, idx4), _mm_andnot_si128(mask, imin4));
        vmin4 = _mm_min_ps(vmin4, vc4);
        b4++;
        a4++;
        c4++;
        idx4 = _mm_add_epi32(idx4, inc4);
    }

    // 4 values -> 2
    {
        idx4 = _mm_shuffle_epi32(imin4, 3 << 2 | 2);
        __m128 vc4 = _mm_shuffle_ps(vmin4, vmin4, 3 << 2 | 2);
        __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(vmin4, vc4));
        imin4 = _mm_or_si128(_mm_and_si128(mask, idx4), _mm_andnot_si128(mask, imin4));
        vmin4 = _mm_min_ps(vmin4, vc4);
    }
    // 2 values -> 1
    {
        idx4 = _mm_shuffle_epi32(imin4, 1);
        __m128 vc4 = _mm_shuffle_ps(vmin4, vmin4, 1);
        __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(vmin4, vc4));
        imin4 = _mm_or_si128(_mm_and_si128(mask, idx4), _mm_andnot_si128(mask, imin4));
        // vmin4 = _mm_min_ps (vmin4, vc4);
    }
    return _mm_cvtsi128_si32(imin4);
}

float
fvec_norm_L2sqr_sse(const float* x, size_t d) {
    __m128 mx;
    __m128 msum1 = _mm_setzero_ps();

    while (d >= 4) {
        mx = _mm_loadu_ps(x);
        x += 4;
        msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, mx));
        d -= 4;
    }

    mx = masked_read(d, x);
    msum1 = _mm_add_ps(msum1, _mm_mul_ps(mx, mx));

    msum1 = _mm_hadd_ps(msum1, msum1);
    msum1 = _mm_hadd_ps(msum1, msum1);
    return _mm_cvtss_f32(msum1);
}

namespace {

/// Function that does a component-wise operation between x and y
/// to compute L2 distances. ElementOp can then be used in the fvec_op_ny
/// functions below
struct ElementOpL2 {
    static float
    op(float x, float y) {
        float tmp = x - y;
        return tmp * tmp;
    }

    static __m128
    op(__m128 x, __m128 y) {
        __m128 tmp = _mm_sub_ps(x, y);
        return _mm_mul_ps(tmp, tmp);
    }
};

/// Function that does a component-wise operation between x and y
/// to compute inner products
struct ElementOpIP {
    static float
    op(float x, float y) {
        return x * y;
    }

    static __m128
    op(__m128 x, __m128 y) {
        return _mm_mul_ps(x, y);
    }
};

template <class ElementOp>
void
fvec_op_ny_D1(float* dis, const float* x, const float* y, size_t ny) {
    float x0s = x[0];
    __m128 x0 = _mm_set_ps(x0s, x0s, x0s, x0s);

    size_t i;
    for (i = 0; i + 3 < ny; i += 4) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        dis[i] = _mm_cvtss_f32(accu);
        __m128 tmp = _mm_shuffle_ps(accu, accu, 1);
        dis[i + 1] = _mm_cvtss_f32(tmp);
        tmp = _mm_shuffle_ps(accu, accu, 2);
        dis[i + 2] = _mm_cvtss_f32(tmp);
        tmp = _mm_shuffle_ps(accu, accu, 3);
        dis[i + 3] = _mm_cvtss_f32(tmp);
    }
    while (i < ny) {  // handle non-multiple-of-4 case
        dis[i++] = ElementOp::op(x0s, *y++);
    }
}

template <class ElementOp>
void
fvec_op_ny_D2(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_set_ps(x[1], x[0], x[1], x[0]);

    size_t i;
    for (i = 0; i + 1 < ny; i += 2) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
        accu = _mm_shuffle_ps(accu, accu, 3);
        dis[i + 1] = _mm_cvtss_f32(accu);
    }
    if (i < ny) {  // handle odd case
        dis[i] = ElementOp::op(x[0], y[0]) + ElementOp::op(x[1], y[1]);
    }
}

template <class ElementOp>
void
fvec_op_ny_D4(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

template <class ElementOp>
void
fvec_op_ny_D8(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x1, _mm_loadu_ps(y)));
        y += 4;
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

template <class ElementOp>
void
fvec_op_ny_D12(float* dis, const float* x, const float* y, size_t ny) {
    __m128 x0 = _mm_loadu_ps(x);
    __m128 x1 = _mm_loadu_ps(x + 4);
    __m128 x2 = _mm_loadu_ps(x + 8);

    for (size_t i = 0; i < ny; i++) {
        __m128 accu = ElementOp::op(x0, _mm_loadu_ps(y));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x1, _mm_loadu_ps(y)));
        y += 4;
        accu = _mm_add_ps(accu, ElementOp::op(x2, _mm_loadu_ps(y)));
        y += 4;
        accu = _mm_hadd_ps(accu, accu);
        accu = _mm_hadd_ps(accu, accu);
        dis[i] = _mm_cvtss_f32(accu);
    }
}

}  // anonymous namespace

/***************************************************************************
 * heavily optimized table computations
 ***************************************************************************/

void
fvec_inner_products_ny_sse(float* dis, const float* x, const float* y, size_t d, size_t ny) {
#define DISPATCH(dval)                                  \
    case dval:                                          \
        fvec_op_ny_D##dval<ElementOpIP>(dis, x, y, ny); \
        return;

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
        default:
            for (size_t i = 0; i < ny; i++) {
                dis[i] = fvec_inner_product_sse(x, y, d);
                y += d;
            }
            return;
    }
#undef DISPATCH
}

void
fvec_L2sqr_ny_sse(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    // optimized for a few special cases

#define DISPATCH(dval)                                  \
    case dval:                                          \
        fvec_op_ny_D##dval<ElementOpL2>(dis, x, y, ny); \
        return;

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
        default:
            for (size_t i = 0; i < ny; i++) {
                dis[i] = fvec_L2sqr_sse(x, y, d);
                y += d;
            }
            return;
    }
#undef DISPATCH
}

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete

int32_t
ivec_inner_product_sse(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

int32_t
ivec_L2sqr_sse(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return res;
}

///////////////////////////////////////////////////////////////////////////////
// bf16

float
bf16_vec_inner_product_sse(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    __m128 m_res = _mm_setzero_ps();
    while (d >= 4) {
        __m128 m_x = _mm_bf16_to_fp32(_mm_loadl_epi64((const __m128i*)x));
        __m128 m_y = _mm_bf16_to_fp32(_mm_loadl_epi64((const __m128i*)y));
        m_res = _mm_add_ps(m_res, _mm_mul_ps(m_x, m_y));
        x += 4;
        y += 4;
        d -= 4;
    }
    if (d > 0) {
        __m128 m_x = _mm_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)x));
        __m128 m_y = _mm_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)y));
        m_res = _mm_add_ps(m_res, _mm_mul_ps(m_x, m_y));
    }
    m_res = _mm_hadd_ps(m_res, m_res);
    m_res = _mm_hadd_ps(m_res, m_res);
    return _mm_cvtss_f32(m_res);
}

float
bf16_vec_L2sqr_sse(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    __m128 m_res = _mm_setzero_ps();
    while (d >= 4) {
        __m128 m_x = _mm_bf16_to_fp32(_mm_loadl_epi64((const __m128i*)x));
        __m128 m_y = _mm_bf16_to_fp32(_mm_loadl_epi64((const __m128i*)y));
        m_x = _mm_sub_ps(m_x, m_y);
        m_res = _mm_add_ps(m_res, _mm_mul_ps(m_x, m_x));
        x += 4;
        y += 4;
        d -= 4;
    }
    if (d > 0) {
        __m128 m_x = _mm_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)x));
        __m128 m_y = _mm_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)y));
        m_x = _mm_sub_ps(m_x, m_y);
        m_res = _mm_add_ps(m_res, _mm_mul_ps(m_x, m_x));
    }
    m_res = _mm_hadd_ps(m_res, m_res);
    m_res = _mm_hadd_ps(m_res, m_res);
    return _mm_cvtss_f32(m_res);
}

float
bf16_vec_norm_L2sqr_sse(const knowhere::bf16* x, size_t d) {
    __m128 m_res = _mm_setzero_ps();
    while (d >= 4) {
        __m128 m_x = _mm_bf16_to_fp32(_mm_loadl_epi64((const __m128i*)x));
        m_res = _mm_add_ps(m_res, _mm_mul_ps(m_x, m_x));
        x += 4;
        d -= 4;
    }
    if (d > 0) {
        __m128 m_x = _mm_bf16_to_fp32(mm_masked_read_short(d, (uint16_t*)x));
        m_res = _mm_add_ps(m_res, _mm_mul_ps(m_x, m_x));
    }
    m_res = _mm_hadd_ps(m_res, m_res);
    m_res = _mm_hadd_ps(m_res, m_res);
    return _mm_cvtss_f32(m_res);
}

///////////////////////////////////////////////////////////////////////////////
// int8

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
float
int8_vec_inner_product_sse(const int8_t* x, const int8_t* y, size_t d) {
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
int8_vec_L2sqr_sse(const int8_t* x, const int8_t* y, size_t d) {
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
int8_vec_norm_L2sqr_sse(const int8_t* x, size_t d) {
    int32_t res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * (int32_t)x[i];
    }
    return (float)res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

///////////////////////////////////////////////////////////////////////////////
// rabitq
float
fvec_masked_sum_sse(const float* q, const uint8_t* x, const size_t d) {
    float sum = 0;

    for (size_t i = 0; i < d; i++) {
        // extract i-th bit
        const uint8_t masker = (1 << (i % 8));
        const bool b_bit = ((x[i / 8] & masker) == masker);

        // accumulate dp
        sum += b_bit ? q[i] : 0;
    }

    return sum;
}

int
rabitq_dp_popcnt_sse(const uint8_t* q, const uint8_t* x, const size_t d, const size_t nb) {
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

}  // namespace faiss
#endif
