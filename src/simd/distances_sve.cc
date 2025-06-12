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
fvec_inner_product_sve(const float* x, const float* y, size_t d) {
    svfloat32_t sum = svdup_f32(0.0f);
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    while (i < d) {
        if (d - i < svcntw())
            pg = svwhilelt_b32(i, d);

        svfloat32_t a = svld1_f32(pg, x + i);
        svfloat32_t b = svld1_f32(pg, y + i);
        sum = svmla_f32_m(pg, sum, a, b);
        i += svcntw();
    }

    float result = svaddv_f32(svptrue_b32(), sum);

    return result;
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
fp16_vec_inner_product_sve(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
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

        sum1 = svmla_f32_m(pg_32, sum1, a_fp32_low, b_fp32_low);
        sum2 = svmla_f32_m(pg_32, sum2, a_fp32_high, b_fp32_high);

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

void
fvec_inner_products_ny_sve(float* ip, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; ++i) {
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
bf16_vec_L2sqr_sve(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
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
bf16_vec_norm_L2sqr_sve(const knowhere::bf16* x, size_t d) {
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
bf16_vec_L2sqr_batch_4_sve(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                           const knowhere::bf16* y2, const knowhere::bf16* y3, size_t d, float& dis0, float& dis1,
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

}  // namespace faiss

#endif
