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

#include "distances_sve.h"

#include <cmath>

#include "faiss/impl/platform_macros.h"
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
namespace faiss {
namespace cppcontrib {
namespace knowhere {

namespace {
inline size_t
find_min_index_sve(const float* data, size_t ny) {
    if (ny <= 0)
        return 0;
    uint64_t vl = svcntw();
    svfloat32_t min_val = svdup_n_f32(data[0]);
    svuint32_t min_idx = svdup_n_u32(0);

    for (size_t i = 0; i < ny; i += vl) {
        svbool_t pg = svwhilelt_b32(i, ny);
        svfloat32_t vec = svld1_f32(pg, data + i);
        svbool_t cmp = svcmplt_f32(pg, vec, min_val);
        auto current_idx = svadd_u32_x(pg, svindex_u32(0, 1), svdup_n_u32(i));
        min_val = svsel_f32(cmp, vec, min_val);
        min_idx = svsel_u32(cmp, current_idx, min_idx);
    }

    svbool_t res_pg = svptrue_b32();
    float min_scalar = svminv_f32(res_pg, min_val);

    svbool_t min_mask = svcmpeq_f32(res_pg, min_val, svdup_n_f32(min_scalar));
    size_t min_index = svlastb_u32(svbrka_b_z(min_mask, min_mask), min_idx);
    return min_index;
}

inline void
fvec_L2sqr_ny_dimN_sve(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    const size_t prefetch_distance = 2 * svcntw();
    size_t i = 0;

    svprfd(svptrue_b32(), y + 0 * d, SV_PLDL1STRM);
    svprfd(svptrue_b32(), y + 1 * d, SV_PLDL1STRM);
    svprfd(svptrue_b32(), y + 2 * d, SV_PLDL1STRM);
    svprfd(svptrue_b32(), y + 3 * d, SV_PLDL1STRM);

    for (; i + 4 <= ny; i += 4) {
        if (i + 4 + prefetch_distance <= ny) {
            svprfd(svptrue_b32(), y + 4 * d + 0 * d, SV_PLDL1STRM);
            svprfd(svptrue_b32(), y + 4 * d + 1 * d, SV_PLDL1STRM);
            svprfd(svptrue_b32(), y + 4 * d + 2 * d, SV_PLDL1STRM);
            svprfd(svptrue_b32(), y + 4 * d + 3 * d, SV_PLDL1STRM);
        }

        fvec_L2sqr_batch_4_sve(x, y, y + d, y + 2 * d, y + 3 * d, d, dis[i], dis[i + 1], dis[i + 2], dis[i + 3]);
        y += 4 * d;
    }

    for (; i < ny; ++i) {
        svprfd(svptrue_b32(), y, SV_PLDL1STRM);
        dis[i] = fvec_L2sqr_sve(x, y, d);
        y += d;
    }
}

inline void
fvec_L2sqr_ny_dim1_sve(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    const size_t vl = svcntw();
    if (vl % d != 0) {
        return fvec_L2sqr_ny_dimN_sve(dis, x, y, d, ny);
    }
    const size_t step = vl;

    svfloat32_t x0 = svdup_n_f32(x[0]);
    size_t i = 0;
    for (; i < ny; i += step) {
        svbool_t active = svwhilelt_b32(i, ny);

        svfloat32_t y0 = svld1(active, y + i);
        svfloat32_t diff0 = svsub_f32_z(active, x0, y0);
        svfloat32_t sq_diff0 = svmul_f32_z(active, diff0, diff0);
        svst1(active, &dis[i], sq_diff0);
    }
}
inline void
fvec_L2sqr_ny_dim2_sve(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    const size_t vl = svcntw();
    if (vl % d != 0) {
        return fvec_L2sqr_ny_dimN_sve(dis, x, y, d, ny);
    }
    const size_t step = vl / 2;  // caculate step y vectors each round
    svfloat32_t x0 = svdup_n_f32(x[0]);
    svfloat32_t x1 = svdup_n_f32(x[1]);
    size_t i = 0;
    for (; i < ny; i += step) {
        svbool_t active = svwhilelt_b32(i * d, ny * d);
        svfloat32_t y0 = svld1_f32(active, y + d * i + 0);
        svfloat32_t y1 = svld1_f32(active, y + d * i + 1);
        auto part0 = svuzp1(y0, y1);
        auto part1 = svuzp2(y0, y1);

        svfloat32_t diff0 = svsub_f32_z(active, x0, part0);
        svfloat32_t diff1 = svsub_f32_z(active, x1, part1);

        svfloat32_t sq_diff0 = svmul_f32_z(active, diff0, diff0);
        svfloat32_t sq_diff1 = svmul_f32_z(active, diff1, diff1);

        svfloat32_t sum = svadd_f32_z(active, sq_diff0, sq_diff1);
        svbool_t store = svwhilelt_b32(size_t(0), std::min(step, ny - i));
        svst1(store, &dis[i], sum);
    }
}

inline void
fvec_L2sqr_ny_dim4_sve(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    const size_t vl = svcntw();
    const size_t step = vl * 2 / d;
    if (vl % d != 0 && step < 1) {
        return fvec_L2sqr_ny_dimN_sve(dis, x, y, d, ny);
    }

    svbool_t pg = svptrue_b32();
    svfloat32_t x_vec = svld1rq_f32(pg, x);
    svuint32_t indices = svand_n_u32_z(pg, svindex_u32(0, 1), 0x3);

    auto x_n = svtbl_f32(x_vec, indices);  // [x0, x1, x2, x3, x0, x1, x2, x3]
    size_t i = 0;
    for (; i + step <= ny; i += step) {
        svbool_t active0 = svwhilelt_b32(i * 4, ny * 4);
        svbool_t active1 = svwhilelt_b32(i * 4 + vl, ny * 4);
        svfloat32_t y0 = svld1_f32(active0, y + 4 * i + 0);
        svfloat32_t y1 = svld1_f32(active1, y + 4 * i + vl);

        y0 = svsub_f32_z(pg, x_n, y0);
        y1 = svsub_f32_z(pg, x_n, y1);
        y0 = svmul_f32_z(pg, y0, y0);
        y1 = svmul_f32_z(pg, y1, y1);

        auto part0 = svuzp1(y0, y1);
        auto part1 = svuzp2(y0, y1);
        part0 = svadd_f32_z(pg, part0, part1);
        svfloat32_t merged_evens = svuzp1(part0, svdup_n_f32(0.0f));
        svfloat32_t merged_odds = svuzp2(part0, svdup_n_f32(0.0f));
        merged_evens = svadd_f32_z(pg, merged_evens, merged_odds);

        svbool_t store = svwhilelt_b32(size_t(0), std::min(step, ny - i));
        svst1(store, &dis[i], merged_evens);
    }
    while (i < ny) {
        dis[i] = fvec_L2sqr_sve(x, y + i * d, d);
        i++;
    }
}
}  // namespace

float
fvec_L2sqr_sve(const float* x, const float* y, size_t d) {
    const size_t vl = svcntw();
    const svbool_t pg = svptrue_b32();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * vl <= d; i += 4 * vl) {
        svfloat32_t d0 = svsub_f32_x(pg, svld1_f32(pg, x + i), svld1_f32(pg, y + i));
        svfloat32_t d1 = svsub_f32_x(pg, svld1_f32(pg, x + i + vl), svld1_f32(pg, y + i + vl));
        svfloat32_t d2 = svsub_f32_x(pg, svld1_f32(pg, x + i + 2 * vl), svld1_f32(pg, y + i + 2 * vl));
        svfloat32_t d3 = svsub_f32_x(pg, svld1_f32(pg, x + i + 3 * vl), svld1_f32(pg, y + i + 3 * vl));
        sum0 = svmla_f32_x(pg, sum0, d0, d0);
        sum1 = svmla_f32_x(pg, sum1, d1, d1);
        sum2 = svmla_f32_x(pg, sum2, d2, d2);
        sum3 = svmla_f32_x(pg, sum3, d3, d3);
    }

    for (; i + vl <= d; i += vl) {
        svfloat32_t d0 = svsub_f32_x(pg, svld1_f32(pg, x + i), svld1_f32(pg, y + i));
        sum0 = svmla_f32_x(pg, sum0, d0, d0);
    }

    if (i < d) {
        svbool_t pg_tail = svwhilelt_b32(i, d);
        svfloat32_t d0 = svsub_f32_z(pg_tail, svld1_f32(pg_tail, x + i), svld1_f32(pg_tail, y + i));
        sum0 = svmla_f32_m(pg_tail, sum0, d0, d0);
    }

    sum0 = svadd_f32_x(pg, sum0, sum1);
    sum2 = svadd_f32_x(pg, sum2, sum3);
    sum0 = svadd_f32_x(pg, sum0, sum2);
    return svaddv_f32(pg, sum0);
}

float
fvec_inner_product_sve(const float* x, const float* y, size_t d) {
    const size_t vl = svcntw();
    const svbool_t pg = svptrue_b32();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * vl <= d; i += 4 * vl) {
        sum0 = svmla_f32_x(pg, sum0, svld1_f32(pg, x + i), svld1_f32(pg, y + i));
        sum1 = svmla_f32_x(pg, sum1, svld1_f32(pg, x + i + vl), svld1_f32(pg, y + i + vl));
        sum2 = svmla_f32_x(pg, sum2, svld1_f32(pg, x + i + 2 * vl), svld1_f32(pg, y + i + 2 * vl));
        sum3 = svmla_f32_x(pg, sum3, svld1_f32(pg, x + i + 3 * vl), svld1_f32(pg, y + i + 3 * vl));
    }

    for (; i + vl <= d; i += vl) {
        sum0 = svmla_f32_x(pg, sum0, svld1_f32(pg, x + i), svld1_f32(pg, y + i));
    }

    if (i < d) {
        svbool_t pg_tail = svwhilelt_b32(i, d);
        sum0 = svmla_f32_m(pg_tail, sum0, svld1_f32(pg_tail, x + i), svld1_f32(pg_tail, y + i));
    }

    sum0 = svadd_f32_x(pg, sum0, sum1);
    sum2 = svadd_f32_x(pg, sum2, sum3);
    sum0 = svadd_f32_x(pg, sum0, sum2);
    return svaddv_f32(pg, sum0);
}

float
fp16_vec_L2sqr_sve(const ::knowhere::fp16* x, const ::knowhere::fp16* y, size_t d) {
    const size_t vl16 = svcnth();
    const svbool_t pg_16 = svptrue_b16();
    const svbool_t pg_32 = svptrue_b32();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    const auto* px = reinterpret_cast<const __fp16*>(x);
    const auto* py = reinterpret_cast<const __fp16*>(y);

    size_t i = 0;
    for (; i + 2 * vl16 <= d; i += 2 * vl16) {
        svfloat16_t a0 = svld1_f16(pg_16, px + i);
        svfloat16_t b0 = svld1_f16(pg_16, py + i);
        svfloat32_t d0_lo = svsub_f32_x(pg_32, svcvt_f32_f16_x(pg_32, a0), svcvt_f32_f16_x(pg_32, b0));
        svfloat32_t d0_hi =
            svsub_f32_x(pg_32, svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0)), svcvt_f32_f16_x(pg_32, svtrn2_f16(b0, b0)));
        sum0 = svmla_f32_x(pg_32, sum0, d0_lo, d0_lo);
        sum1 = svmla_f32_x(pg_32, sum1, d0_hi, d0_hi);

        svfloat16_t a1 = svld1_f16(pg_16, px + i + vl16);
        svfloat16_t b1 = svld1_f16(pg_16, py + i + vl16);
        svfloat32_t d1_lo = svsub_f32_x(pg_32, svcvt_f32_f16_x(pg_32, a1), svcvt_f32_f16_x(pg_32, b1));
        svfloat32_t d1_hi =
            svsub_f32_x(pg_32, svcvt_f32_f16_x(pg_32, svtrn2_f16(a1, a1)), svcvt_f32_f16_x(pg_32, svtrn2_f16(b1, b1)));
        sum2 = svmla_f32_x(pg_32, sum2, d1_lo, d1_lo);
        sum3 = svmla_f32_x(pg_32, sum3, d1_hi, d1_hi);
    }

    for (; i + vl16 <= d; i += vl16) {
        svfloat16_t a0 = svld1_f16(pg_16, px + i);
        svfloat16_t b0 = svld1_f16(pg_16, py + i);
        svfloat32_t d0_lo = svsub_f32_x(pg_32, svcvt_f32_f16_x(pg_32, a0), svcvt_f32_f16_x(pg_32, b0));
        svfloat32_t d0_hi =
            svsub_f32_x(pg_32, svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0)), svcvt_f32_f16_x(pg_32, svtrn2_f16(b0, b0)));
        sum0 = svmla_f32_x(pg_32, sum0, d0_lo, d0_lo);
        sum1 = svmla_f32_x(pg_32, sum1, d0_hi, d0_hi);
    }

    if (i < d) {
        svbool_t pg_tail = svwhilelt_b16(i, d);
        svfloat16_t a0 = svld1_f16(pg_tail, px + i);
        svfloat16_t b0 = svld1_f16(pg_tail, py + i);
        svfloat32_t d0_lo = svsub_f32_x(pg_32, svcvt_f32_f16_x(pg_32, a0), svcvt_f32_f16_x(pg_32, b0));
        svfloat32_t d0_hi =
            svsub_f32_x(pg_32, svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0)), svcvt_f32_f16_x(pg_32, svtrn2_f16(b0, b0)));
        sum0 = svmla_f32_x(pg_32, sum0, d0_lo, d0_lo);
        sum1 = svmla_f32_x(pg_32, sum1, d0_hi, d0_hi);
    }

    sum0 = svadd_f32_x(pg_32, sum0, sum1);
    sum2 = svadd_f32_x(pg_32, sum2, sum3);
    sum0 = svadd_f32_x(pg_32, sum0, sum2);
    return svaddv_f32(pg_32, sum0);
}

float
fp16_vec_inner_product_sve(const ::knowhere::fp16* x, const ::knowhere::fp16* y, size_t d) {
    const size_t vl16 = svcnth();
    const svbool_t pg_16 = svptrue_b16();
    const svbool_t pg_32 = svptrue_b32();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    const auto* px = reinterpret_cast<const __fp16*>(x);
    const auto* py = reinterpret_cast<const __fp16*>(y);

    size_t i = 0;
    for (; i + 2 * vl16 <= d; i += 2 * vl16) {
        svfloat16_t a0 = svld1_f16(pg_16, px + i);
        svfloat16_t b0 = svld1_f16(pg_16, py + i);
        sum0 = svmla_f32_x(pg_32, sum0, svcvt_f32_f16_x(pg_32, a0), svcvt_f32_f16_x(pg_32, b0));
        sum1 = svmla_f32_x(pg_32, sum1, svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0)),
                           svcvt_f32_f16_x(pg_32, svtrn2_f16(b0, b0)));

        svfloat16_t a1 = svld1_f16(pg_16, px + i + vl16);
        svfloat16_t b1 = svld1_f16(pg_16, py + i + vl16);
        sum2 = svmla_f32_x(pg_32, sum2, svcvt_f32_f16_x(pg_32, a1), svcvt_f32_f16_x(pg_32, b1));
        sum3 = svmla_f32_x(pg_32, sum3, svcvt_f32_f16_x(pg_32, svtrn2_f16(a1, a1)),
                           svcvt_f32_f16_x(pg_32, svtrn2_f16(b1, b1)));
    }

    for (; i + vl16 <= d; i += vl16) {
        svfloat16_t a0 = svld1_f16(pg_16, px + i);
        svfloat16_t b0 = svld1_f16(pg_16, py + i);
        sum0 = svmla_f32_x(pg_32, sum0, svcvt_f32_f16_x(pg_32, a0), svcvt_f32_f16_x(pg_32, b0));
        sum1 = svmla_f32_x(pg_32, sum1, svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0)),
                           svcvt_f32_f16_x(pg_32, svtrn2_f16(b0, b0)));
    }

    if (i < d) {
        svbool_t pg_tail = svwhilelt_b16(i, d);
        svfloat16_t a0 = svld1_f16(pg_tail, px + i);
        svfloat16_t b0 = svld1_f16(pg_tail, py + i);
        sum0 = svmla_f32_x(pg_32, sum0, svcvt_f32_f16_x(pg_32, a0), svcvt_f32_f16_x(pg_32, b0));
        sum1 = svmla_f32_x(pg_32, sum1, svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0)),
                           svcvt_f32_f16_x(pg_32, svtrn2_f16(b0, b0)));
    }

    sum0 = svadd_f32_x(pg_32, sum0, sum1);
    sum2 = svadd_f32_x(pg_32, sum2, sum3);
    sum0 = svadd_f32_x(pg_32, sum0, sum2);
    return svaddv_f32(pg_32, sum0);
}

float
fvec_L1_sve(const float* x, const float* y, size_t d) {
    svfloat32_t sum = svdup_f32(0.0f);
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    while (i < d) {
        if (d - i < svcntw())
            pg = svwhilelt_b32(i, d);

        svfloat32_t a = svld1_f32(pg, x + i);
        svfloat32_t b = svld1_f32(pg, y + i);
        svfloat32_t diff = svabs_f32_x(pg, svsub_f32_m(pg, a, b));
        sum = svadd_f32_m(pg, sum, diff);
        i += svcntw();
    }

    return svaddv_f32(svptrue_b32(), sum);
}

float
fvec_Linf_sve(const float* x, const float* y, size_t d) {
    svfloat32_t max_val = svdup_f32(0.0f);
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    while (i < d) {
        if (d - i < svcntw())
            pg = svwhilelt_b32(i, d);

        svfloat32_t a = svld1_f32(pg, x + i);
        svfloat32_t b = svld1_f32(pg, y + i);
        svfloat32_t diff = svabs_f32_x(pg, svsub_f32_m(pg, a, b));
        max_val = svmax_f32_m(pg, max_val, diff);
        i += svcntw();
    }

    return svmaxv_f32(svptrue_b32(), max_val);
}

float
fvec_norm_L2sqr_sve(const float* x, size_t d) {
    const size_t vl = svcntw();
    const svbool_t pg = svptrue_b32();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * vl <= d; i += 4 * vl) {
        svfloat32_t x0 = svld1_f32(pg, x + i);
        svfloat32_t x1 = svld1_f32(pg, x + i + vl);
        svfloat32_t x2 = svld1_f32(pg, x + i + 2 * vl);
        svfloat32_t x3 = svld1_f32(pg, x + i + 3 * vl);
        sum0 = svmla_f32_x(pg, sum0, x0, x0);
        sum1 = svmla_f32_x(pg, sum1, x1, x1);
        sum2 = svmla_f32_x(pg, sum2, x2, x2);
        sum3 = svmla_f32_x(pg, sum3, x3, x3);
    }

    for (; i + vl <= d; i += vl) {
        svfloat32_t x0 = svld1_f32(pg, x + i);
        sum0 = svmla_f32_x(pg, sum0, x0, x0);
    }

    if (i < d) {
        svbool_t pg_tail = svwhilelt_b32(i, d);
        svfloat32_t x0 = svld1_f32(pg_tail, x + i);
        sum0 = svmla_f32_m(pg_tail, sum0, x0, x0);
    }

    sum0 = svadd_f32_x(pg, sum0, sum1);
    sum2 = svadd_f32_x(pg, sum2, sum3);
    sum0 = svadd_f32_x(pg, sum0, sum2);
    return svaddv_f32(pg, sum0);
}

float
fp16_vec_norm_L2sqr_sve(const ::knowhere::fp16* x, size_t d) {
    const size_t vl16 = svcnth();
    const svbool_t pg_16 = svptrue_b16();
    const svbool_t pg_32 = svptrue_b32();

    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    const auto* px = reinterpret_cast<const __fp16*>(x);

    size_t i = 0;
    for (; i + 2 * vl16 <= d; i += 2 * vl16) {
        svfloat16_t a0 = svld1_f16(pg_16, px + i);
        svfloat32_t a0_lo = svcvt_f32_f16_x(pg_32, a0);
        svfloat32_t a0_hi = svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0));
        sum0 = svmla_f32_x(pg_32, sum0, a0_lo, a0_lo);
        sum1 = svmla_f32_x(pg_32, sum1, a0_hi, a0_hi);

        svfloat16_t a1 = svld1_f16(pg_16, px + i + vl16);
        svfloat32_t a1_lo = svcvt_f32_f16_x(pg_32, a1);
        svfloat32_t a1_hi = svcvt_f32_f16_x(pg_32, svtrn2_f16(a1, a1));
        sum2 = svmla_f32_x(pg_32, sum2, a1_lo, a1_lo);
        sum3 = svmla_f32_x(pg_32, sum3, a1_hi, a1_hi);
    }

    for (; i + vl16 <= d; i += vl16) {
        svfloat16_t a0 = svld1_f16(pg_16, px + i);
        svfloat32_t a0_lo = svcvt_f32_f16_x(pg_32, a0);
        svfloat32_t a0_hi = svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0));
        sum0 = svmla_f32_x(pg_32, sum0, a0_lo, a0_lo);
        sum1 = svmla_f32_x(pg_32, sum1, a0_hi, a0_hi);
    }

    if (i < d) {
        svbool_t pg_tail = svwhilelt_b16(i, d);
        svfloat16_t a0 = svld1_f16(pg_tail, px + i);
        svfloat32_t a0_lo = svcvt_f32_f16_x(pg_32, a0);
        svfloat32_t a0_hi = svcvt_f32_f16_x(pg_32, svtrn2_f16(a0, a0));
        sum0 = svmla_f32_x(pg_32, sum0, a0_lo, a0_lo);
        sum1 = svmla_f32_x(pg_32, sum1, a0_hi, a0_hi);
    }

    sum0 = svadd_f32_x(pg_32, sum0, sum1);
    sum2 = svadd_f32_x(pg_32, sum2, sum3);
    sum0 = svadd_f32_x(pg_32, sum0, sum2);
    return svaddv_f32(pg_32, sum0);
}

void
fvec_madd_sve(size_t n, const float* a, float bf, const float* b, float* c) {
    size_t i = 0;
    svfloat32_t bf_vec = svdup_f32(bf);

    svbool_t pg = svptrue_b32();

    while (i < n) {
        if (n - i < svcntw())
            pg = svwhilelt_b32(i, n);

        svfloat32_t a_vec = svld1_f32(pg, a + i);
        svfloat32_t b_vec = svld1_f32(pg, b + i);
        svfloat32_t c_vec = svmla_f32_m(pg, a_vec, b_vec, bf_vec);
        svst1_f32(pg, c + i, c_vec);
        i += svcntw();
    }
}

int
fvec_madd_and_argmin_sve(size_t n, const float* a, float bf, const float* b, float* c) {
    size_t i = 0;
    svfloat32_t min_val = svdup_f32(INFINITY);
    svuint32_t min_idx = svdup_u32(0);
    svuint32_t idx_base = svindex_u32(0, 1);

    svfloat32_t bf_vec = svdup_f32(bf);
    svbool_t pg = svptrue_b32();

    while (i < n) {
        if (n - i < svcntw())
            pg = svwhilelt_b32(i, n);

        svuint32_t idx = svadd_u32_z(pg, idx_base, svdup_u32(i));
        svfloat32_t a_vec = svld1_f32(pg, a + i);
        svfloat32_t b_vec = svld1_f32(pg, b + i);
        svfloat32_t c_vec = svmla_f32_m(pg, a_vec, b_vec, bf_vec);
        svst1_f32(pg, c + i, c_vec);

        svbool_t cmp = svcmplt(pg, c_vec, min_val);
        min_val = svsel_f32(cmp, c_vec, min_val);
        min_idx = svsel_u32(cmp, idx, min_idx);

        i += svcntw();
    }

    float min_value = svminv_f32(svptrue_b32(), min_val);
    svbool_t pg_min = svcmpeq(svptrue_b32(), min_val, svdup_f32(min_value));
    uint32_t min_index = svlastb_u32(pg_min, min_idx);

    return static_cast<int>(min_index);
}

void
fvec_inner_product_batch_4_sve(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                               const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    const size_t vl = svcntw();
    const svbool_t pg = svptrue_b32();

    svfloat32_t acc0_0 = svdup_f32(0.0f), acc0_1 = svdup_f32(0.0f), acc0_2 = svdup_f32(0.0f), acc0_3 = svdup_f32(0.0f);
    svfloat32_t acc1_0 = svdup_f32(0.0f), acc1_1 = svdup_f32(0.0f), acc1_2 = svdup_f32(0.0f), acc1_3 = svdup_f32(0.0f);
    svfloat32_t acc2_0 = svdup_f32(0.0f), acc2_1 = svdup_f32(0.0f), acc2_2 = svdup_f32(0.0f), acc2_3 = svdup_f32(0.0f);
    svfloat32_t acc3_0 = svdup_f32(0.0f), acc3_1 = svdup_f32(0.0f), acc3_2 = svdup_f32(0.0f), acc3_3 = svdup_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * vl <= d; i += 4 * vl) {
        svfloat32_t vx0 = svld1_f32(pg, x + i);
        acc0_0 = svmla_f32_x(pg, acc0_0, vx0, svld1_f32(pg, y0 + i));
        acc1_0 = svmla_f32_x(pg, acc1_0, vx0, svld1_f32(pg, y1 + i));
        acc2_0 = svmla_f32_x(pg, acc2_0, vx0, svld1_f32(pg, y2 + i));
        acc3_0 = svmla_f32_x(pg, acc3_0, vx0, svld1_f32(pg, y3 + i));

        svfloat32_t vx1 = svld1_f32(pg, x + i + vl);
        acc0_1 = svmla_f32_x(pg, acc0_1, vx1, svld1_f32(pg, y0 + i + vl));
        acc1_1 = svmla_f32_x(pg, acc1_1, vx1, svld1_f32(pg, y1 + i + vl));
        acc2_1 = svmla_f32_x(pg, acc2_1, vx1, svld1_f32(pg, y2 + i + vl));
        acc3_1 = svmla_f32_x(pg, acc3_1, vx1, svld1_f32(pg, y3 + i + vl));

        svfloat32_t vx2 = svld1_f32(pg, x + i + 2 * vl);
        acc0_2 = svmla_f32_x(pg, acc0_2, vx2, svld1_f32(pg, y0 + i + 2 * vl));
        acc1_2 = svmla_f32_x(pg, acc1_2, vx2, svld1_f32(pg, y1 + i + 2 * vl));
        acc2_2 = svmla_f32_x(pg, acc2_2, vx2, svld1_f32(pg, y2 + i + 2 * vl));
        acc3_2 = svmla_f32_x(pg, acc3_2, vx2, svld1_f32(pg, y3 + i + 2 * vl));

        svfloat32_t vx3 = svld1_f32(pg, x + i + 3 * vl);
        acc0_3 = svmla_f32_x(pg, acc0_3, vx3, svld1_f32(pg, y0 + i + 3 * vl));
        acc1_3 = svmla_f32_x(pg, acc1_3, vx3, svld1_f32(pg, y1 + i + 3 * vl));
        acc2_3 = svmla_f32_x(pg, acc2_3, vx3, svld1_f32(pg, y2 + i + 3 * vl));
        acc3_3 = svmla_f32_x(pg, acc3_3, vx3, svld1_f32(pg, y3 + i + 3 * vl));
    }

    for (; i + vl <= d; i += vl) {
        svfloat32_t vx0 = svld1_f32(pg, x + i);
        acc0_0 = svmla_f32_x(pg, acc0_0, vx0, svld1_f32(pg, y0 + i));
        acc1_0 = svmla_f32_x(pg, acc1_0, vx0, svld1_f32(pg, y1 + i));
        acc2_0 = svmla_f32_x(pg, acc2_0, vx0, svld1_f32(pg, y2 + i));
        acc3_0 = svmla_f32_x(pg, acc3_0, vx0, svld1_f32(pg, y3 + i));
    }

    if (i < d) {
        svbool_t pg_tail = svwhilelt_b32(i, d);
        svfloat32_t vx = svld1_f32(pg_tail, x + i);
        acc0_0 = svmla_f32_m(pg_tail, acc0_0, vx, svld1_f32(pg_tail, y0 + i));
        acc1_0 = svmla_f32_m(pg_tail, acc1_0, vx, svld1_f32(pg_tail, y1 + i));
        acc2_0 = svmla_f32_m(pg_tail, acc2_0, vx, svld1_f32(pg_tail, y2 + i));
        acc3_0 = svmla_f32_m(pg_tail, acc3_0, vx, svld1_f32(pg_tail, y3 + i));
    }

    acc0_0 = svadd_f32_x(pg, svadd_f32_x(pg, acc0_0, acc0_1), svadd_f32_x(pg, acc0_2, acc0_3));
    acc1_0 = svadd_f32_x(pg, svadd_f32_x(pg, acc1_0, acc1_1), svadd_f32_x(pg, acc1_2, acc1_3));
    acc2_0 = svadd_f32_x(pg, svadd_f32_x(pg, acc2_0, acc2_1), svadd_f32_x(pg, acc2_2, acc2_3));
    acc3_0 = svadd_f32_x(pg, svadd_f32_x(pg, acc3_0, acc3_1), svadd_f32_x(pg, acc3_2, acc3_3));

    dis0 = svaddv_f32(pg, acc0_0);
    dis1 = svaddv_f32(pg, acc1_0);
    dis2 = svaddv_f32(pg, acc2_0);
    dis3 = svaddv_f32(pg, acc3_0);
}

void
fvec_L2sqr_batch_4_sve(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                       const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    const size_t vl = svcntw();
    const svbool_t pg = svptrue_b32();

    svfloat32_t acc0_0 = svdup_f32(0.0f), acc0_1 = svdup_f32(0.0f), acc0_2 = svdup_f32(0.0f), acc0_3 = svdup_f32(0.0f);
    svfloat32_t acc1_0 = svdup_f32(0.0f), acc1_1 = svdup_f32(0.0f), acc1_2 = svdup_f32(0.0f), acc1_3 = svdup_f32(0.0f);
    svfloat32_t acc2_0 = svdup_f32(0.0f), acc2_1 = svdup_f32(0.0f), acc2_2 = svdup_f32(0.0f), acc2_3 = svdup_f32(0.0f);
    svfloat32_t acc3_0 = svdup_f32(0.0f), acc3_1 = svdup_f32(0.0f), acc3_2 = svdup_f32(0.0f), acc3_3 = svdup_f32(0.0f);

    size_t i = 0;
    for (; i + 4 * vl <= d; i += 4 * vl) {
        svfloat32_t vx0 = svld1_f32(pg, x + i);
        svfloat32_t d00 = svsub_f32_x(pg, vx0, svld1_f32(pg, y0 + i));
        svfloat32_t d10 = svsub_f32_x(pg, vx0, svld1_f32(pg, y1 + i));
        svfloat32_t d20 = svsub_f32_x(pg, vx0, svld1_f32(pg, y2 + i));
        svfloat32_t d30 = svsub_f32_x(pg, vx0, svld1_f32(pg, y3 + i));
        acc0_0 = svmla_f32_x(pg, acc0_0, d00, d00);
        acc1_0 = svmla_f32_x(pg, acc1_0, d10, d10);
        acc2_0 = svmla_f32_x(pg, acc2_0, d20, d20);
        acc3_0 = svmla_f32_x(pg, acc3_0, d30, d30);

        svfloat32_t vx1 = svld1_f32(pg, x + i + vl);
        svfloat32_t d01 = svsub_f32_x(pg, vx1, svld1_f32(pg, y0 + i + vl));
        svfloat32_t d11 = svsub_f32_x(pg, vx1, svld1_f32(pg, y1 + i + vl));
        svfloat32_t d21 = svsub_f32_x(pg, vx1, svld1_f32(pg, y2 + i + vl));
        svfloat32_t d31 = svsub_f32_x(pg, vx1, svld1_f32(pg, y3 + i + vl));
        acc0_1 = svmla_f32_x(pg, acc0_1, d01, d01);
        acc1_1 = svmla_f32_x(pg, acc1_1, d11, d11);
        acc2_1 = svmla_f32_x(pg, acc2_1, d21, d21);
        acc3_1 = svmla_f32_x(pg, acc3_1, d31, d31);

        svfloat32_t vx2 = svld1_f32(pg, x + i + 2 * vl);
        svfloat32_t d02 = svsub_f32_x(pg, vx2, svld1_f32(pg, y0 + i + 2 * vl));
        svfloat32_t d12 = svsub_f32_x(pg, vx2, svld1_f32(pg, y1 + i + 2 * vl));
        svfloat32_t d22 = svsub_f32_x(pg, vx2, svld1_f32(pg, y2 + i + 2 * vl));
        svfloat32_t d32 = svsub_f32_x(pg, vx2, svld1_f32(pg, y3 + i + 2 * vl));
        acc0_2 = svmla_f32_x(pg, acc0_2, d02, d02);
        acc1_2 = svmla_f32_x(pg, acc1_2, d12, d12);
        acc2_2 = svmla_f32_x(pg, acc2_2, d22, d22);
        acc3_2 = svmla_f32_x(pg, acc3_2, d32, d32);

        svfloat32_t vx3 = svld1_f32(pg, x + i + 3 * vl);
        svfloat32_t d03 = svsub_f32_x(pg, vx3, svld1_f32(pg, y0 + i + 3 * vl));
        svfloat32_t d13 = svsub_f32_x(pg, vx3, svld1_f32(pg, y1 + i + 3 * vl));
        svfloat32_t d23 = svsub_f32_x(pg, vx3, svld1_f32(pg, y2 + i + 3 * vl));
        svfloat32_t d33 = svsub_f32_x(pg, vx3, svld1_f32(pg, y3 + i + 3 * vl));
        acc0_3 = svmla_f32_x(pg, acc0_3, d03, d03);
        acc1_3 = svmla_f32_x(pg, acc1_3, d13, d13);
        acc2_3 = svmla_f32_x(pg, acc2_3, d23, d23);
        acc3_3 = svmla_f32_x(pg, acc3_3, d33, d33);
    }

    for (; i + vl <= d; i += vl) {
        svfloat32_t vx0 = svld1_f32(pg, x + i);
        svfloat32_t d00 = svsub_f32_x(pg, vx0, svld1_f32(pg, y0 + i));
        svfloat32_t d10 = svsub_f32_x(pg, vx0, svld1_f32(pg, y1 + i));
        svfloat32_t d20 = svsub_f32_x(pg, vx0, svld1_f32(pg, y2 + i));
        svfloat32_t d30 = svsub_f32_x(pg, vx0, svld1_f32(pg, y3 + i));
        acc0_0 = svmla_f32_x(pg, acc0_0, d00, d00);
        acc1_0 = svmla_f32_x(pg, acc1_0, d10, d10);
        acc2_0 = svmla_f32_x(pg, acc2_0, d20, d20);
        acc3_0 = svmla_f32_x(pg, acc3_0, d30, d30);
    }

    if (i < d) {
        svbool_t pg_tail = svwhilelt_b32(i, d);
        svfloat32_t vx = svld1_f32(pg_tail, x + i);
        svfloat32_t d0 = svsub_f32_z(pg_tail, vx, svld1_f32(pg_tail, y0 + i));
        svfloat32_t d1 = svsub_f32_z(pg_tail, vx, svld1_f32(pg_tail, y1 + i));
        svfloat32_t d2 = svsub_f32_z(pg_tail, vx, svld1_f32(pg_tail, y2 + i));
        svfloat32_t d3 = svsub_f32_z(pg_tail, vx, svld1_f32(pg_tail, y3 + i));
        acc0_0 = svmla_f32_m(pg_tail, acc0_0, d0, d0);
        acc1_0 = svmla_f32_m(pg_tail, acc1_0, d1, d1);
        acc2_0 = svmla_f32_m(pg_tail, acc2_0, d2, d2);
        acc3_0 = svmla_f32_m(pg_tail, acc3_0, d3, d3);
    }

    acc0_0 = svadd_f32_x(pg, svadd_f32_x(pg, acc0_0, acc0_1), svadd_f32_x(pg, acc0_2, acc0_3));
    acc1_0 = svadd_f32_x(pg, svadd_f32_x(pg, acc1_0, acc1_1), svadd_f32_x(pg, acc1_2, acc1_3));
    acc2_0 = svadd_f32_x(pg, svadd_f32_x(pg, acc2_0, acc2_1), svadd_f32_x(pg, acc2_2, acc2_3));
    acc3_0 = svadd_f32_x(pg, svadd_f32_x(pg, acc3_0, acc3_1), svadd_f32_x(pg, acc3_2, acc3_3));

    dis0 = svaddv_f32(pg, acc0_0);
    dis1 = svaddv_f32(pg, acc1_0);
    dis2 = svaddv_f32(pg, acc2_0);
    dis3 = svaddv_f32(pg, acc3_0);
}

void
fvec_L2sqr_ny_sve(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    switch (d) {
        case 1:
            fvec_L2sqr_ny_dim1_sve(dis, x, y, d, ny);
            break;
        case 2:
            fvec_L2sqr_ny_dim2_sve(dis, x, y, d, ny);
            break;
        case 4:
            fvec_L2sqr_ny_dim4_sve(dis, x, y, d, ny);
            break;
        default:
            fvec_L2sqr_ny_dimN_sve(dis, x, y, d, ny);
            break;
    }
}

/// compute ny square L2 distance between x and a set of contiguous y vectors
/// and return the index of the nearest vector.
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_sve(float* __restrict distances_tmp_buffer, const float* __restrict x, const float* __restrict y,
                          size_t d, size_t ny) {
    fvec_L2sqr_ny_sve(distances_tmp_buffer, x, y, d, ny);

    if (ny >= 16) {
        return find_min_index_sve(distances_tmp_buffer, ny);
    } else {
        size_t nearest_idx = 0;
        float min_dis = HUGE_VALF;

        for (size_t i = 0; i < ny; i++) {
            if (distances_tmp_buffer[i] < min_dis) {
                min_dis = distances_tmp_buffer[i];
                nearest_idx = i;
            }
        }
        return nearest_idx;
    }
}

void
fvec_inner_products_ny_sve(float* ip, const float* x, const float* y, size_t d, size_t ny) {
    size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        fvec_inner_product_batch_4_sve(x, y, y + d, y + 2 * d, y + 3 * d, d, ip[i], ip[i + 1], ip[i + 2], ip[i + 3]);
        y += 4 * d;
    }
    for (; i < ny; ++i) {
        ip[i] = fvec_inner_product_sve(x, y, d);
        y += d;
    }
}

float
int8_vec_L2sqr_sve(const int8_t* x, const int8_t* y, size_t d) {
    int32_t scalar_sum = 0;

    if (d < 16) {
        for (size_t i = 0; i < d; ++i) {
            int32_t diff = x[i] - y[i];
            scalar_sum += diff * diff;
        }
        return static_cast<float>(scalar_sum);
    }

    svint32_t sum_a2 = svdup_n_s32(0);
    svint32_t sum_ab = svdup_n_s32(0);
    svint32_t sum_b2 = svdup_n_s32(0);

    size_t vl = svcntb();
    size_t i = 0;
    svbool_t pg = svptrue_b8();

    for (; i + vl <= d; i += vl) {
        svint8_t vx = svld1_s8(pg, x + i);
        svint8_t vy = svld1_s8(pg, y + i);
        sum_a2 = svdot_s32(sum_a2, vx, vx);
        sum_ab = svdot_s32(sum_ab, vx, vy);
        sum_b2 = svdot_s32(sum_b2, vy, vy);
    }

    if (i < d) {
        pg = svwhilelt_b8(i, d);
        svint8_t vx = svld1_s8(pg, x + i);
        svint8_t vy = svld1_s8(pg, y + i);
        sum_a2 = svdot_s32(sum_a2, vx, vx);
        sum_ab = svdot_s32(sum_ab, vx, vy);
        sum_b2 = svdot_s32(sum_b2, vy, vy);
    }

    int32_t total_a2 = svaddv_s32(svptrue_b32(), sum_a2);
    int32_t total_ab = svaddv_s32(svptrue_b32(), sum_ab);
    int32_t total_b2 = svaddv_s32(svptrue_b32(), sum_b2);

    return static_cast<float>(total_a2 - 2 * total_ab + total_b2);
}

float
int8_vec_norm_L2sqr_sve(const int8_t* x, size_t d) {
    const size_t vl = svcntb();
    svint32_t sum = svdup_n_s32(0);

    size_t i = 0;
    svbool_t pg = svwhilelt_b8((unsigned long)i, (unsigned long)d);
    while (svptest_any(svptrue_b8(), pg)) {
        svint8_t a = svld1_s8(pg, x + i);

        sum = svdot_s32(sum, a, a);

        i += vl;
        pg = svwhilelt_b8((unsigned long)i, (unsigned long)d);
    }

    int32_t result = svaddv_s32(svptrue_b32(), sum);
    return static_cast<float>(result);
}

void
int8_vec_L2sqr_batch_4_sve(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                           const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3) {
    int32_t sum_a2 = 0;
    int32_t sum_ab0 = 0, sum_ab1 = 0, sum_ab2 = 0, sum_ab3 = 0;
    int32_t sum_b20 = 0, sum_b21 = 0, sum_b22 = 0, sum_b23 = 0;
    if (dim == 0) {
        dis0 = dis1 = dis2 = dis3 = 0.0f;
        return;
    }
    if (dim < 16) {
        for (size_t i = 0; i < dim; i++) {
            int16_t xi = x[i];
            int16_t y0i = y0[i];
            int16_t y1i = y1[i];
            int16_t y2i = y2[i];
            int16_t y3i = y3[i];

            sum_a2 += xi * xi;
            sum_ab0 += xi * y0i;
            sum_ab1 += xi * y1i;
            sum_ab2 += xi * y2i;
            sum_ab3 += xi * y3i;
            sum_b20 += y0i * y0i;
            sum_b21 += y1i * y1i;
            sum_b22 += y2i * y2i;
            sum_b23 += y3i * y3i;
        }

        dis0 = static_cast<float>(sum_a2 - 2 * sum_ab0 + sum_b20);
        dis1 = static_cast<float>(sum_a2 - 2 * sum_ab1 + sum_b21);
        dis2 = static_cast<float>(sum_a2 - 2 * sum_ab2 + sum_b22);
        dis3 = static_cast<float>(sum_a2 - 2 * sum_ab3 + sum_b23);
        return;
    }
    svint32_t vec_a2 = svdup_n_s32(0);
    svint32_t vec_ab0 = svdup_n_s32(0);
    svint32_t vec_ab1 = svdup_n_s32(0);
    svint32_t vec_ab2 = svdup_n_s32(0);
    svint32_t vec_ab3 = svdup_n_s32(0);
    svint32_t vec_b20 = svdup_n_s32(0);
    svint32_t vec_b21 = svdup_n_s32(0);
    svint32_t vec_b22 = svdup_n_s32(0);
    svint32_t vec_b23 = svdup_n_s32(0);
    const size_t vl = svcntb();
    const svbool_t pg = svptrue_b8();
    size_t d = 0;
    for (; d + vl <= dim; d += vl) {
        svint8_t a = svld1_s8(pg, x + d);
        svint8_t b0 = svld1_s8(pg, y0 + d);
        svint8_t b1 = svld1_s8(pg, y1 + d);
        svint8_t b2 = svld1_s8(pg, y2 + d);
        svint8_t b3 = svld1_s8(pg, y3 + d);
        vec_a2 = svdot_s32(vec_a2, a, a);
        vec_ab0 = svdot_s32(vec_ab0, a, b0);
        vec_ab1 = svdot_s32(vec_ab1, a, b1);
        vec_ab2 = svdot_s32(vec_ab2, a, b2);
        vec_ab3 = svdot_s32(vec_ab3, a, b3);
        vec_b20 = svdot_s32(vec_b20, b0, b0);
        vec_b21 = svdot_s32(vec_b21, b1, b1);
        vec_b22 = svdot_s32(vec_b22, b2, b2);
        vec_b23 = svdot_s32(vec_b23, b3, b3);
    }
    for (; d < dim; d++) {
        int16_t xi = x[d];
        int16_t y0i = y0[d];
        int16_t y1i = y1[d];
        int16_t y2i = y2[d];
        int16_t y3i = y3[d];

        sum_a2 += xi * xi;
        sum_ab0 += xi * y0i;
        sum_ab1 += xi * y1i;
        sum_ab2 += xi * y2i;
        sum_ab3 += xi * y3i;
        sum_b20 += y0i * y0i;
        sum_b21 += y1i * y1i;
        sum_b22 += y2i * y2i;
        sum_b23 += y3i * y3i;
    }
    sum_a2 += svaddv_s32(svptrue_b32(), vec_a2);
    sum_ab0 += svaddv_s32(svptrue_b32(), vec_ab0);
    sum_ab1 += svaddv_s32(svptrue_b32(), vec_ab1);
    sum_ab2 += svaddv_s32(svptrue_b32(), vec_ab2);
    sum_ab3 += svaddv_s32(svptrue_b32(), vec_ab3);
    sum_b20 += svaddv_s32(svptrue_b32(), vec_b20);
    sum_b21 += svaddv_s32(svptrue_b32(), vec_b21);
    sum_b22 += svaddv_s32(svptrue_b32(), vec_b22);
    sum_b23 += svaddv_s32(svptrue_b32(), vec_b23);
    dis0 = static_cast<float>(sum_a2 - 2 * sum_ab0 + sum_b20);
    dis1 = static_cast<float>(sum_a2 - 2 * sum_ab1 + sum_b21);
    dis2 = static_cast<float>(sum_a2 - 2 * sum_ab2 + sum_b22);
    dis3 = static_cast<float>(sum_a2 - 2 * sum_ab3 + sum_b23);
}

float
int8_vec_inner_product_sve(const int8_t* x, const int8_t* y, size_t d) {
    svint32_t sum0 = svdup_s32(0);
    svint32_t sum1 = svdup_s32(0);
    svint32_t sum2 = svdup_s32(0);
    svint32_t sum3 = svdup_s32(0);

    size_t vl = svcntb();
    size_t step = 4 * vl;

    while (d >= step) {
        svbool_t pg = svptrue_b8();

        svint8_t a0 = svld1_s8(pg, x);
        svint8_t b0 = svld1_s8(pg, y);
        sum0 = svdot_s32(sum0, a0, b0);

        svint8_t a1 = svld1_s8(pg, x + vl);
        svint8_t b1 = svld1_s8(pg, y + vl);
        sum1 = svdot_s32(sum1, a1, b1);

        svint8_t a2 = svld1_s8(pg, x + 2 * vl);
        svint8_t b2 = svld1_s8(pg, y + 2 * vl);
        sum2 = svdot_s32(sum2, a2, b2);

        svint8_t a3 = svld1_s8(pg, x + 3 * vl);
        svint8_t b3 = svld1_s8(pg, y + 3 * vl);
        sum3 = svdot_s32(sum3, a3, b3);

        x += step;
        y += step;
        d -= step;
    }

    svint32_t sum = svadd_s32_x(svptrue_b32(), sum0, sum1);
    sum = svadd_s32_x(svptrue_b32(), sum, sum2);
    sum = svadd_s32_x(svptrue_b32(), sum, sum3);

    while (d >= vl) {
        svbool_t pg = svptrue_b8();
        svint8_t a = svld1_s8(pg, x);
        svint8_t b = svld1_s8(pg, y);
        sum = svdot_s32(sum, a, b);
        x += vl;
        y += vl;
        d -= vl;
    }

    if (d > 0) {
        svbool_t pg = svwhilelt_b8_s32(0, d);
        svint8_t a = svld1_s8(pg, x);
        svint8_t b = svld1_s8(pg, y);
        sum = svdot_s32(sum, a, b);
    }

    int32_t total = svaddv_s32(svptrue_b32(), sum);

    return static_cast<float>(total);
}
void
int8_vec_inner_product_batch_4_sve(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                   const int8_t* y3, const size_t dim, float& dis0, float& dis1, float& dis2,
                                   float& dis3) {
    svint32_t sum0 = svdup_n_s32(0);
    svint32_t sum1 = svdup_n_s32(0);
    svint32_t sum2 = svdup_n_s32(0);
    svint32_t sum3 = svdup_n_s32(0);
    size_t d = 0;
    const size_t vl = svcntb() / sizeof(int8_t);
    svbool_t pg = svptrue_b8();
    while (d + vl <= dim) {
        svint8_t a = svld1_s8(pg, x + d);
        svint8_t b0 = svld1_s8(pg, y0 + d);
        svint8_t b1 = svld1_s8(pg, y1 + d);
        svint8_t b2 = svld1_s8(pg, y2 + d);
        svint8_t b3 = svld1_s8(pg, y3 + d);
        sum0 = svdot_s32(sum0, a, b0);
        sum1 = svdot_s32(sum1, a, b1);
        sum2 = svdot_s32(sum2, a, b2);
        sum3 = svdot_s32(sum3, a, b3);
        d += vl;
    }
    if (d < dim) {
        pg = svwhilelt_b8(d, dim);
        svint8_t a = svld1_s8(pg, x + d);
        svint8_t b0 = svld1_s8(pg, y0 + d);
        svint8_t b1 = svld1_s8(pg, y1 + d);
        svint8_t b2 = svld1_s8(pg, y2 + d);
        svint8_t b3 = svld1_s8(pg, y3 + d);
        sum0 = svdot_s32(sum0, a, b0);
        sum1 = svdot_s32(sum1, a, b1);
        sum2 = svdot_s32(sum2, a, b2);
        sum3 = svdot_s32(sum3, a, b3);
    }
    dis0 = static_cast<float>(svaddv_s32(svptrue_b32(), sum0));
    dis1 = static_cast<float>(svaddv_s32(svptrue_b32(), sum1));
    dis2 = static_cast<float>(svaddv_s32(svptrue_b32(), sum2));
    dis3 = static_cast<float>(svaddv_s32(svptrue_b32(), sum3));
}

float
bf16_vec_L2sqr_sve(const ::knowhere::bf16* x, const ::knowhere::bf16* y, size_t d) {
    svfloat32_t acc_squared_norm_x = svdup_f32(0.0f);
    svfloat32_t acc_squared_norm_y = svdup_f32(0.0f);
    svfloat32_t acc_dot_product = svdup_f32(0.0f);

    size_t i = 0;
    svbool_t pg = svptrue_b16();

    while (i < d) {
        if (d - i < svcnth())
            pg = svwhilelt_b16(i, d);

        svbfloat16_t bf16_x = svld1_bf16(pg, reinterpret_cast<const __bf16*>(x + i));
        svbfloat16_t bf16_y = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y + i));

        acc_squared_norm_x = svbfdot_f32(acc_squared_norm_x, bf16_x, bf16_x);
        acc_squared_norm_y = svbfdot_f32(acc_squared_norm_y, bf16_y, bf16_y);
        acc_dot_product = svbfdot_f32(acc_dot_product, bf16_x, bf16_y);

        i += svcnth();
    }

    float norm_x_sq = svaddv_f32(svptrue_b32(), acc_squared_norm_x);
    float norm_y_sq = svaddv_f32(svptrue_b32(), acc_squared_norm_y);
    float dot_xy = svaddv_f32(svptrue_b32(), acc_dot_product);

    return norm_x_sq + norm_y_sq - 2.0f * dot_xy;
}

float
bf16_vec_norm_L2sqr_sve(const ::knowhere::bf16* x, size_t d) {
    svfloat32_t acc = svdup_f32(0.0f);

    size_t i = 0;
    svbool_t pg = svptrue_b16();

    while (i < d) {
        if (d - i < svcnth())
            pg = svwhilelt_b16(i, d);

        svbfloat16_t x_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(x + i));

        acc = svbfdot_f32(acc, x_vec, x_vec);

        i += svcnth();
    }

    return svaddv_f32(svptrue_b32(), acc);
}

void
bf16_vec_L2sqr_batch_4_sve(const ::knowhere::bf16* x, const ::knowhere::bf16* y0, const ::knowhere::bf16* y1,
                           const ::knowhere::bf16* y2, const ::knowhere::bf16* y3, size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3) {
    svfloat32_t acc_x = svdup_f32(0.0f);
    svfloat32_t acc_y0 = svdup_f32(0.0f);
    svfloat32_t acc_y1 = svdup_f32(0.0f);
    svfloat32_t acc_y2 = svdup_f32(0.0f);
    svfloat32_t acc_y3 = svdup_f32(0.0f);

    svfloat32_t acc_ip0 = svdup_f32(0.0f);
    svfloat32_t acc_ip1 = svdup_f32(0.0f);
    svfloat32_t acc_ip2 = svdup_f32(0.0f);
    svfloat32_t acc_ip3 = svdup_f32(0.0f);

    size_t i = 0;
    svbool_t pg = svptrue_b16();

    while (i < d) {
        if (d - i < svcnth())
            pg = svwhilelt_b16(i, d);

        svbfloat16_t x_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(x + i));
        svbfloat16_t y0_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y0 + i));
        svbfloat16_t y1_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y1 + i));
        svbfloat16_t y2_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y2 + i));
        svbfloat16_t y3_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y3 + i));

        acc_x = svbfdot_f32(acc_x, x_vec, x_vec);
        acc_y0 = svbfdot_f32(acc_y0, y0_vec, y0_vec);
        acc_y1 = svbfdot_f32(acc_y1, y1_vec, y1_vec);
        acc_y2 = svbfdot_f32(acc_y2, y2_vec, y2_vec);
        acc_y3 = svbfdot_f32(acc_y3, y3_vec, y3_vec);

        acc_ip0 = svbfdot_f32(acc_ip0, x_vec, y0_vec);
        acc_ip1 = svbfdot_f32(acc_ip1, x_vec, y1_vec);
        acc_ip2 = svbfdot_f32(acc_ip2, x_vec, y2_vec);
        acc_ip3 = svbfdot_f32(acc_ip3, x_vec, y3_vec);

        i += svcnth();
    }

    float norm_x = svaddv_f32(svptrue_b32(), acc_x);
    float norm_y0 = svaddv_f32(svptrue_b32(), acc_y0);
    float norm_y1 = svaddv_f32(svptrue_b32(), acc_y1);
    float norm_y2 = svaddv_f32(svptrue_b32(), acc_y2);
    float norm_y3 = svaddv_f32(svptrue_b32(), acc_y3);

    float ip0 = svaddv_f32(svptrue_b32(), acc_ip0);
    float ip1 = svaddv_f32(svptrue_b32(), acc_ip1);
    float ip2 = svaddv_f32(svptrue_b32(), acc_ip2);
    float ip3 = svaddv_f32(svptrue_b32(), acc_ip3);

    dis0 = norm_x + norm_y0 - 2.0f * ip0;
    dis1 = norm_x + norm_y1 - 2.0f * ip1;
    dis2 = norm_x + norm_y2 - 2.0f * ip2;
    dis3 = norm_x + norm_y3 - 2.0f * ip3;
}

float
bf16_vec_inner_product_sve(const ::knowhere::bf16* x, const ::knowhere::bf16* y, size_t d) {
    svfloat32_t acc = svdup_f32(0.0f);

    size_t i = 0;
    svbool_t pg = svptrue_b16();

    while (i < d) {
        if (d - i < svcnth())
            pg = svwhilelt_b16(i, d);

        svbfloat16_t x_vec = svld1_bf16(pg, reinterpret_cast<const bfloat16_t*>(x + i));
        svbfloat16_t y_vec = svld1_bf16(pg, reinterpret_cast<const bfloat16_t*>(y + i));

        acc = svbfdot_f32(acc, x_vec, y_vec);

        i += svcnth();
    }

    return svaddv_f32(svptrue_b32(), acc);
}

void
bf16_vec_inner_product_batch_4_sve(const ::knowhere::bf16* x, const ::knowhere::bf16* y0, const ::knowhere::bf16* y1,
                                   const ::knowhere::bf16* y2, const ::knowhere::bf16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3) {
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);

    size_t i = 0;
    svbool_t pg = svptrue_b16();

    while (i < d) {
        if (d - i < svcnth())
            pg = svwhilelt_b16(i, d);

        svbfloat16_t x_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(x + i));
        svbfloat16_t y0_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y0 + i));
        svbfloat16_t y1_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y1 + i));
        svbfloat16_t y2_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y2 + i));
        svbfloat16_t y3_vec = svld1_bf16(pg, reinterpret_cast<const __bf16*>(y3 + i));

        acc0 = svbfdot_f32(acc0, x_vec, y0_vec);
        acc1 = svbfdot_f32(acc1, x_vec, y1_vec);
        acc2 = svbfdot_f32(acc2, x_vec, y2_vec);
        acc3 = svbfdot_f32(acc3, x_vec, y3_vec);

        i += svcnth();
    }

    dis0 = svaddv_f32(svptrue_b32(), acc0);
    dis1 = svaddv_f32(svptrue_b32(), acc1);
    dis2 = svaddv_f32(svptrue_b32(), acc2);
    dis3 = svaddv_f32(svptrue_b32(), acc3);
}

}  // namespace knowhere
}  // namespace cppcontrib
}  // namespace faiss

#endif
