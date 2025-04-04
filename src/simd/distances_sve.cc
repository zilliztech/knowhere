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

#include <arm_sve.h>

#include <cmath>

#include "faiss/impl/platform_macros.h"
#if defined(__ARM_FEATURE_SVE)
namespace faiss {

float
fvec_L2sqr_sve(const float* x, const float* y, size_t d) {
    svfloat32_t sum = svdup_f32(0.0f);
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    while (i < d) {
        if (d - i < svcntw())
            pg = svwhilelt_b32(i, d);

        svfloat32_t a = svld1_f32(pg, x + i);
        svfloat32_t b = svld1_f32(pg, y + i);
        svfloat32_t diff = svsub_f32_m(pg, a, b);
        sum = svmla_f32_m(pg, sum, diff, diff);
        i += svcntw();
    }

    return svaddv_f32(svptrue_b32(), sum);
}

float
fp16_vec_L2sqr_sve(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    size_t i = 0;

    svbool_t pg_16 = svptrue_b16();
    svbool_t pg_32 = svptrue_b32();

    while (i < d) {
        if (d - i < svcnth())
            pg_16 = svwhilelt_b16(i, d);

        svfloat16_t a_fp16 = svld1_f16(pg_16, reinterpret_cast<const __fp16*>(x + i));
        svfloat16_t b_fp16 = svld1_f16(pg_16, reinterpret_cast<const __fp16*>(y + i));

        svfloat32_t a_fp32_low = svcvt_f32_f16_z(pg_32, svtrn1_f16(a_fp16, a_fp16));
        svfloat32_t a_fp32_high = svcvt_f32_f16_z(pg_32, svtrn2_f16(a_fp16, a_fp16));
        svfloat32_t b_fp32_low = svcvt_f32_f16_z(pg_32, svtrn1_f16(b_fp16, b_fp16));
        svfloat32_t b_fp32_high = svcvt_f32_f16_z(pg_32, svtrn2_f16(b_fp16, b_fp16));

        svfloat32_t diff_fp32_low = svsub_f32_m(pg_32, a_fp32_low, b_fp32_low);
        svfloat32_t diff_fp32_high = svsub_f32_m(pg_32, a_fp32_high, b_fp32_high);

        sum1 = svmla_f32_m(pg_32, sum1, diff_fp32_low, diff_fp32_low);
        sum2 = svmla_f32_m(pg_32, sum2, diff_fp32_high, diff_fp32_high);

        i += svcnth();
    }

    svfloat32_t total_sum = svadd_f32_m(pg_32, sum1, sum2);
    return svaddv_f32(pg_32, total_sum);
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
    svfloat32_t sum = svdup_f32(0.0f);
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    while (i < d) {
        if (d - i < svcntw())
            pg = svwhilelt_b32(i, d);

        svfloat32_t a = svld1_f32(pg, x + i);
        sum = svmla_f32_m(pg, sum, a, a);
        i += svcntw();
    }

    return svaddv_f32(svptrue_b32(), sum);
}

float
fp16_vec_norm_L2sqr_sve(const knowhere::fp16* x, size_t d) {
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    size_t i = 0;

    svbool_t pg_16 = svptrue_b16();
    svbool_t pg_32 = svptrue_b32();

    while (i < d) {
        if (d - i < svcnth())
            pg_16 = svwhilelt_b16(i, d);

        svfloat16_t a_fp16 = svld1_f16(pg_16, reinterpret_cast<const __fp16*>(x + i));

        svfloat32_t a_fp32_low = svcvt_f32_f16_z(pg_32, svtrn1_f16(a_fp16, a_fp16));
        svfloat32_t a_fp32_high = svcvt_f32_f16_z(pg_32, svtrn2_f16(a_fp16, a_fp16));

        svfloat32_t square_fp32_low = svmul_f32_m(pg_32, a_fp32_low, a_fp32_low);
        svfloat32_t square_fp32_high = svmul_f32_m(pg_32, a_fp32_high, a_fp32_high);

        sum1 = svadd_f32_m(pg_32, sum1, square_fp32_low);
        sum2 = svadd_f32_m(pg_32, sum2, square_fp32_high);

        i += svcnth();
    }

    return svaddv_f32(pg_32, sum1) + svaddv_f32(pg_32, sum2);
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
    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);

    size_t i = 0;
    svbool_t pg = svptrue_b32();

    while (i < d) {
        if (d - i < svcntw())
            pg = svwhilelt_b32(i, d);

        svfloat32_t vx = svld1(pg, &x[i]);
        svfloat32_t vy0 = svld1(pg, &y0[i]);
        svfloat32_t vy1 = svld1(pg, &y1[i]);
        svfloat32_t vy2 = svld1(pg, &y2[i]);
        svfloat32_t vy3 = svld1(pg, &y3[i]);

        acc0 = svmla_f32_m(pg, acc0, vx, vy0);
        acc1 = svmla_f32_m(pg, acc1, vx, vy1);
        acc2 = svmla_f32_m(pg, acc2, vx, vy2);
        acc3 = svmla_f32_m(pg, acc3, vx, vy3);

        i += svcntw();
    }

    dis0 = svaddv_f32(svptrue_b32(), acc0);
    dis1 = svaddv_f32(svptrue_b32(), acc1);
    dis2 = svaddv_f32(svptrue_b32(), acc2);
    dis3 = svaddv_f32(svptrue_b32(), acc3);
}

void
fvec_L2sqr_batch_4_sve(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                       const size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    float d0 = 0;
    float d1 = 0;
    float d2 = 0;
    float d3 = 0;

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

void
fvec_L2sqr_ny_sve(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; ++i) {
        dis[i] = fvec_L2sqr_sve(x, y, d);
        y += d;
    }
}

}  // namespace faiss

#endif
