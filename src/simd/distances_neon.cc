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
#if defined(__ARM_NEON)
#pragma GCC optimize("O3,fast-math,inline")
#include "distances_neon.h"

#include <arm_neon.h>
#include <math.h>

#include "simd_util.h"
namespace faiss {
float
fvec_inner_product_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum_ = {0.0f, 0.0f, 0.0f, 0.0f};
    auto dim = d;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        float32x4x4_t b = vld1q_f32_x4(y + dim - d);
        float32x4x4_t c;
        c.val[0] = vmulq_f32(a.val[0], b.val[0]);
        c.val[1] = vmulq_f32(a.val[1], b.val[1]);
        c.val[2] = vmulq_f32(a.val[2], b.val[2]);
        c.val[3] = vmulq_f32(a.val[3], b.val[3]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        c.val[2] = vaddq_f32(c.val[2], c.val[3]);
        c.val[0] = vaddq_f32(c.val[0], c.val[2]);

        sum_ = vaddq_f32(sum_, c.val[0]);

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);
        float32x4x2_t b = vld1q_f32_x2(y + dim - d);
        float32x4x2_t c;
        c.val[0] = vmulq_f32(a.val[0], b.val[0]);
        c.val[1] = vmulq_f32(a.val[1], b.val[1]);
        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        sum_ = vaddq_f32(sum_, c.val[0]);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        float32x4_t b = vld1q_f32(y + dim - d);
        float32x4_t c;
        c = vmulq_f32(a, b);
        sum_ = vaddq_f32(sum_, c);
        d -= 4;
    }

    float32x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
    float32x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 2);
        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 1);
        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 0);
        d -= 1;
    }

    sum_ = vaddq_f32(sum_, vmulq_f32(res_x, res_y));

    return vaddvq_f32(sum_);
}

float
fp16_vec_inner_product_neon(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    while (d >= 16) {
        float32x4x4_t a = vcvt4_f32_f16(vld4_f16((const __fp16*)x));
        float32x4x4_t b = vcvt4_f32_f16(vld4_f16((const __fp16*)y));

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], b.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], b.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[3], b.val[3]);
        d -= 16;
        x += 16;
        y += 16;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    res.val[2] = vaddq_f32(res.val[2], res.val[3]);
    if (d >= 8) {
        float32x4x2_t a = vcvt2_f32_f16(vld2_f16((const __fp16*)x));
        float32x4x2_t b = vcvt2_f32_f16(vld2_f16((const __fp16*)y));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], b.val[1]);
        d -= 8;
        x += 8;
        y += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (d >= 4) {
        float32x4_t a = vcvt_f32_f16(vld1_f16((const __fp16*)x));
        float32x4_t b = vcvt_f32_f16(vld1_f16((const __fp16*)y));
        res.val[0] = vmlaq_f32(res.val[0], a, b);
        d -= 4;
        x += 4;
        y += 4;
    }
    if (d >= 0) {
        float16x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
        switch (d) {
            case 3:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 2);
                res_y = vld1_lane_f16((const __fp16*)y, res_y, 2);
                x++;
                y++;
                d--;
            case 2:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 1);
                res_y = vld1_lane_f16((const __fp16*)y, res_y, 1);
                x++;
                y++;
                d--;
            case 1:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 0);
                res_y = vld1_lane_f16((const __fp16*)y, res_y, 0);
                x++;
                y++;
                d--;
        }
        res.val[0] = vmlaq_f32(res.val[0], vcvt_f32_f16(res_x), vcvt_f32_f16(res_y));
    }
    return vaddvq_f32(res.val[0]);
}

float
bf16_vec_inner_product_neon(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    while (d >= 16) {
        float32x4x4_t a = vcvt4_f32_half(vld4_u16((const uint16_t*)x));
        float32x4x4_t b = vcvt4_f32_half(vld4_u16((const uint16_t*)y));

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], b.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], b.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[3], b.val[3]);
        d -= 16;
        x += 16;
        y += 16;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    res.val[2] = vaddq_f32(res.val[2], res.val[3]);
    if (d >= 8) {
        float32x4x2_t a = vcvt2_f32_half(vld2_u16((const uint16_t*)x));
        float32x4x2_t b = vcvt2_f32_half(vld2_u16((const uint16_t*)y));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], b.val[1]);
        d -= 8;
        x += 8;
        y += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (d >= 4) {
        float32x4_t a = vcvt_f32_half(vld1_u16((const uint16_t*)x));
        float32x4_t b = vcvt_f32_half(vld1_u16((const uint16_t*)y));
        res.val[0] = vmlaq_f32(res.val[0], a, b);
        d -= 4;
        x += 4;
        y += 4;
    }
    if (d >= 0) {
        uint16x4_t res_x = {0, 0, 0, 0};
        uint16x4_t res_y = {0, 0, 0, 0};
        switch (d) {
            case 3:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 2);
                res_y = vld1_lane_u16((const uint16_t*)y, res_y, 2);
                x++;
                y++;
                d--;
            case 2:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 1);
                res_y = vld1_lane_u16((const uint16_t*)y, res_y, 1);
                x++;
                y++;
                d--;
            case 1:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 0);
                res_y = vld1_lane_u16((const uint16_t*)y, res_y, 0);
                x++;
                y++;
                d--;
        }
        res.val[0] = vmlaq_f32(res.val[0], vcvt_f32_half(res_x), vcvt_f32_half(res_y));
    }
    return vaddvq_f32(res.val[0]);
}

float
fvec_L2sqr_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum_ = {0.0f, 0.0f, 0.0f, 0.0f};

    auto dim = d;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        float32x4x4_t b = vld1q_f32_x4(y + dim - d);
        float32x4x4_t c;

        c.val[0] = vsubq_f32(a.val[0], b.val[0]);
        c.val[1] = vsubq_f32(a.val[1], b.val[1]);
        c.val[2] = vsubq_f32(a.val[2], b.val[2]);
        c.val[3] = vsubq_f32(a.val[3], b.val[3]);

        c.val[0] = vmulq_f32(c.val[0], c.val[0]);
        c.val[1] = vmulq_f32(c.val[1], c.val[1]);
        c.val[2] = vmulq_f32(c.val[2], c.val[2]);
        c.val[3] = vmulq_f32(c.val[3], c.val[3]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        c.val[2] = vaddq_f32(c.val[2], c.val[3]);
        c.val[0] = vaddq_f32(c.val[0], c.val[2]);

        sum_ = vaddq_f32(sum_, c.val[0]);

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);
        float32x4x2_t b = vld1q_f32_x2(y + dim - d);
        float32x4x2_t c;
        c.val[0] = vsubq_f32(a.val[0], b.val[0]);
        c.val[1] = vsubq_f32(a.val[1], b.val[1]);

        c.val[0] = vmulq_f32(c.val[0], c.val[0]);
        c.val[1] = vmulq_f32(c.val[1], c.val[1]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        sum_ = vaddq_f32(sum_, c.val[0]);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        float32x4_t b = vld1q_f32(y + dim - d);
        float32x4_t c;
        c = vsubq_f32(a, b);
        c = vmulq_f32(c, c);

        sum_ = vaddq_f32(sum_, c);
        d -= 4;
    }

    float32x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
    float32x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 2);
        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 1);
        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 0);
        d -= 1;
    }

    sum_ = vaddq_f32(sum_, vmulq_f32(vsubq_f32(res_x, res_y), vsubq_f32(res_x, res_y)));

    return vaddvq_f32(sum_);
}

float
fp16_vec_L2sqr_neon(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    while (d >= 16) {
        float32x4x4_t a = vcvt4_f32_f16(vld4_f16((const __fp16*)x));
        float32x4x4_t b = vcvt4_f32_f16(vld4_f16((const __fp16*)y));
        a.val[0] = vsubq_f32(a.val[0], b.val[0]);
        a.val[1] = vsubq_f32(a.val[1], b.val[1]);
        a.val[2] = vsubq_f32(a.val[2], b.val[2]);
        a.val[3] = vsubq_f32(a.val[3], b.val[3]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], a.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], a.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[3], a.val[3]);
        d -= 16;
        x += 16;
        y += 16;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    res.val[2] = vaddq_f32(res.val[2], res.val[3]);
    if (d >= 8) {
        float32x4x2_t a = vcvt2_f32_f16(vld2_f16((const __fp16*)x));
        float32x4x2_t b = vcvt2_f32_f16(vld2_f16((const __fp16*)y));
        a.val[0] = vsubq_f32(a.val[0], b.val[0]);
        a.val[1] = vsubq_f32(a.val[1], b.val[1]);
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], a.val[1]);
        d -= 8;
        x += 8;
        y += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (d >= 4) {
        float32x4_t a = vcvt_f32_f16(vld1_f16((const __fp16*)x));
        float32x4_t b = vcvt_f32_f16(vld1_f16((const __fp16*)y));
        a = vsubq_f32(a, b);
        res.val[0] = vmlaq_f32(res.val[0], a, a);
        d -= 4;
        x += 4;
        y += 4;
    }
    if (d >= 0) {
        float16x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
        switch (d) {
            case 3:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 2);
                res_y = vld1_lane_f16((const __fp16*)y, res_y, 2);
                x++;
                y++;
                d--;
            case 2:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 1);
                res_y = vld1_lane_f16((const __fp16*)y, res_y, 1);
                x++;
                y++;
                d--;
            case 1:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 0);
                res_y = vld1_lane_f16((const __fp16*)y, res_y, 0);
                x++;
                y++;
                d--;
        }
        float32x4_t diff = vsubq_f32(vcvt_f32_f16(res_x), vcvt_f32_f16(res_y));

        res.val[0] = vmlaq_f32(res.val[0], diff, diff);
    }
    return vaddvq_f32(res.val[0]);
}

float
bf16_vec_L2sqr_neon(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    while (d >= 16) {
        float32x4x4_t a = vcvt4_f32_half(vld4_u16((const uint16_t*)x));
        float32x4x4_t b = vcvt4_f32_half(vld4_u16((const uint16_t*)y));
        a.val[0] = vsubq_f32(a.val[0], b.val[0]);
        a.val[1] = vsubq_f32(a.val[1], b.val[1]);
        a.val[2] = vsubq_f32(a.val[2], b.val[2]);
        a.val[3] = vsubq_f32(a.val[3], b.val[3]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], a.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], a.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[3], a.val[3]);
        d -= 16;
        x += 16;
        y += 16;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    res.val[2] = vaddq_f32(res.val[2], res.val[3]);
    if (d >= 8) {
        float32x4x2_t a = vcvt2_f32_half(vld2_u16((const uint16_t*)x));
        float32x4x2_t b = vcvt2_f32_half(vld2_u16((const uint16_t*)y));
        a.val[0] = vsubq_f32(a.val[0], b.val[0]);
        a.val[1] = vsubq_f32(a.val[1], b.val[1]);
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], a.val[1]);
        d -= 8;
        x += 8;
        y += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (d >= 4) {
        float32x4_t a = vcvt_f32_half(vld1_u16((const uint16_t*)x));
        float32x4_t b = vcvt_f32_half(vld1_u16((const uint16_t*)y));
        a = vsubq_f32(a, b);
        res.val[0] = vmlaq_f32(res.val[0], a, a);
        d -= 4;
        x += 4;
        y += 4;
    }
    if (d >= 0) {
        uint16x4_t res_x = {0, 0, 0, 0};
        uint16x4_t res_y = {0, 0, 0, 0};
        switch (d) {
            case 3:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 2);
                res_y = vld1_lane_u16((const uint16_t*)y, res_y, 2);
                x++;
                y++;
                d--;
            case 2:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 1);
                res_y = vld1_lane_u16((const uint16_t*)y, res_y, 1);
                x++;
                y++;
                d--;
            case 1:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 0);
                res_y = vld1_lane_u16((const uint16_t*)y, res_y, 0);
                x++;
                y++;
                d--;
        }

        float32x4_t diff = vsubq_f32(vcvt_f32_half(res_x), vcvt_f32_half(res_y));

        res.val[0] = vmlaq_f32(res.val[0], diff, diff);
    }
    return vaddvq_f32(res.val[0]);
}

float
fvec_L1_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum_ = {0.f};

    auto dim = d;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        float32x4x4_t b = vld1q_f32_x4(y + dim - d);
        float32x4x4_t c;

        c.val[0] = vsubq_f32(a.val[0], b.val[0]);
        c.val[1] = vsubq_f32(a.val[1], b.val[1]);
        c.val[2] = vsubq_f32(a.val[2], b.val[2]);
        c.val[3] = vsubq_f32(a.val[3], b.val[3]);

        c.val[0] = vabsq_f32(c.val[0]);
        c.val[1] = vabsq_f32(c.val[1]);
        c.val[2] = vabsq_f32(c.val[2]);
        c.val[3] = vabsq_f32(c.val[3]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        c.val[2] = vaddq_f32(c.val[2], c.val[3]);
        c.val[0] = vaddq_f32(c.val[0], c.val[2]);

        sum_ = vaddq_f32(sum_, c.val[0]);

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);
        float32x4x2_t b = vld1q_f32_x2(y + dim - d);
        float32x4x2_t c;
        c.val[0] = vsubq_f32(a.val[0], b.val[0]);
        c.val[1] = vsubq_f32(a.val[1], b.val[1]);

        c.val[0] = vabsq_f32(c.val[0]);
        c.val[1] = vabsq_f32(c.val[1]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        sum_ = vaddq_f32(sum_, c.val[0]);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        float32x4_t b = vld1q_f32(y + dim - d);
        float32x4_t c;
        c = vsubq_f32(a, b);
        c = vabsq_f32(c);

        sum_ = vaddq_f32(sum_, c);
        d -= 4;
    }

    float32x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
    float32x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 2);
        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 1);
        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 0);
        d -= 1;
    }

    sum_ = vaddq_f32(sum_, vabsq_f32(vsubq_f32(res_x, res_y)));

    return vaddvq_f32(sum_);
}

float
fvec_Linf_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum_ = {0.0f, 0.0f, 0.0f, 0.0f};

    auto dim = d;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        float32x4x4_t b = vld1q_f32_x4(y + dim - d);
        float32x4x4_t c;

        c.val[0] = vsubq_f32(a.val[0], b.val[0]);
        c.val[1] = vsubq_f32(a.val[1], b.val[1]);
        c.val[2] = vsubq_f32(a.val[2], b.val[2]);
        c.val[3] = vsubq_f32(a.val[3], b.val[3]);

        c.val[0] = vabsq_f32(c.val[0]);
        c.val[1] = vabsq_f32(c.val[1]);
        c.val[2] = vabsq_f32(c.val[2]);
        c.val[3] = vabsq_f32(c.val[3]);

        c.val[0] = vmaxq_f32(c.val[0], c.val[1]);
        c.val[2] = vmaxq_f32(c.val[2], c.val[3]);
        c.val[0] = vmaxq_f32(c.val[0], c.val[2]);

        sum_ = vmaxq_f32(sum_, c.val[0]);

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);
        float32x4x2_t b = vld1q_f32_x2(y + dim - d);
        float32x4x2_t c;
        c.val[0] = vsubq_f32(a.val[0], b.val[0]);
        c.val[1] = vsubq_f32(a.val[1], b.val[1]);

        c.val[0] = vabsq_f32(c.val[0]);
        c.val[1] = vabsq_f32(c.val[1]);

        c.val[0] = vmaxq_f32(c.val[0], c.val[1]);
        sum_ = vmaxq_f32(sum_, c.val[0]);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        float32x4_t b = vld1q_f32(y + dim - d);
        float32x4_t c;
        c = vsubq_f32(a, b);
        c = vabsq_f32(c);

        sum_ = vmaxq_f32(sum_, c);
        d -= 4;
    }

    float32x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
    float32x4_t res_y = {0.0f, 0.0f, 0.0f, 0.0f};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 2);
        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 1);
        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        res_y = vld1q_lane_f32(y + dim - d, res_y, 0);
        d -= 1;
    }

    sum_ = vmaxq_f32(sum_, vabsq_f32(vsubq_f32(res_x, res_y)));

    return vmaxvq_f32(sum_);
}

float
fvec_norm_L2sqr_neon(const float* x, size_t d) {
    float32x4_t sum_ = {0.0f, 0.0f, 0.0f, 0.0f};
    auto dim = d;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        float32x4x4_t c;
        c.val[0] = vmulq_f32(a.val[0], a.val[0]);
        c.val[1] = vmulq_f32(a.val[1], a.val[1]);
        c.val[2] = vmulq_f32(a.val[2], a.val[2]);
        c.val[3] = vmulq_f32(a.val[3], a.val[3]);

        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        c.val[2] = vaddq_f32(c.val[2], c.val[3]);
        c.val[0] = vaddq_f32(c.val[0], c.val[2]);

        sum_ = vaddq_f32(sum_, c.val[0]);

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);
        float32x4x2_t c;
        c.val[0] = vmulq_f32(a.val[0], a.val[0]);
        c.val[1] = vmulq_f32(a.val[1], a.val[1]);
        c.val[0] = vaddq_f32(c.val[0], c.val[1]);
        sum_ = vaddq_f32(sum_, c.val[0]);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        float32x4_t c;
        c = vmulq_f32(a, a);
        sum_ = vaddq_f32(sum_, c);
        d -= 4;
    }

    float32x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        d -= 1;
    }

    sum_ = vaddq_f32(sum_, vmulq_f32(res_x, res_x));

    return vaddvq_f32(sum_);
}

float
fp16_vec_norm_L2sqr_neon(const knowhere::fp16* x, size_t d) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    while (d >= 16) {
        float32x4x4_t a = vcvt4_f32_f16(vld4_f16((const __fp16*)x));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], a.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], a.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[3], a.val[3]);
        d -= 16;
        x += 16;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    res.val[2] = vaddq_f32(res.val[2], res.val[3]);
    if (d >= 8) {
        float32x4x2_t a = vcvt2_f32_f16(vld2_f16((const __fp16*)x));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], a.val[1]);
        d -= 8;
        x += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (d >= 4) {
        float32x4_t a = vcvt_f32_f16(vld1_f16((const __fp16*)x));
        res.val[0] = vmlaq_f32(res.val[0], a, a);
        d -= 4;
        x += 4;
    }
    if (d >= 0) {
        float16x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
        switch (d) {
            case 3:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 2);
                x++;
                d--;
            case 2:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 1);
                x++;
                d--;
            case 1:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 0);
                x++;
                d--;
        }
        float32x4_t x_f32 = vcvt_f32_f16(res_x);
        res.val[0] = vmlaq_f32(res.val[0], x_f32, x_f32);
    }
    return vaddvq_f32(res.val[0]);
}

float
bf16_vec_norm_L2sqr_neon(const knowhere::bf16* x, size_t d) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    while (d >= 16) {
        float32x4x4_t a = vcvt4_f32_half(vld4_u16((const uint16_t*)x));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], a.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], a.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[3], a.val[3]);
        d -= 16;
        x += 16;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[1]);
    res.val[2] = vaddq_f32(res.val[2], res.val[3]);
    if (d >= 8) {
        float32x4x2_t a = vcvt2_f32_half(vld2_u16((const uint16_t*)x));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], a.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], a.val[1]);
        d -= 8;
        x += 8;
    }
    res.val[0] = vaddq_f32(res.val[0], res.val[2]);
    if (d >= 4) {
        float32x4_t a = vcvt_f32_half(vld1_u16((const uint16_t*)x));
        res.val[0] = vmlaq_f32(res.val[0], a, a);
        d -= 4;
        x += 4;
    }
    if (d >= 0) {
        uint16x4_t res_x = {0, 0, 0, 0};
        switch (d) {
            case 3:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 2);
                x++;
                d--;
            case 2:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 1);
                x++;
                d--;
            case 1:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 0);
                x++;
                d--;
        }

        float32x4_t x_fp32 = vcvt_f32_half(res_x);

        res.val[0] = vmlaq_f32(res.val[0], x_fp32, x_fp32);
    }
    return vaddvq_f32(res.val[0]);
}

void
fvec_L2sqr_ny_neon(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr_neon(x, y, d);
        y += d;
    }
}

void
fvec_inner_products_ny_neon(float* ip, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product_neon(x, y, d);
        y += d;
    }
}

void
fvec_madd_neon(size_t n, const float* a, float bf, const float* b, float* c) {
    size_t len = n;
    while (n >= 16) {
        auto a_ = vld1q_f32_x4(a + len - n);
        auto b_ = vld1q_f32_x4(b + len - n);
        b_.val[0] = vmulq_n_f32(b_.val[0], bf);
        b_.val[1] = vmulq_n_f32(b_.val[1], bf);
        b_.val[2] = vmulq_n_f32(b_.val[2], bf);
        b_.val[3] = vmulq_n_f32(b_.val[3], bf);
        float32x4x4_t c_;
        c_.val[0] = vaddq_f32(b_.val[0], a_.val[0]);
        c_.val[1] = vaddq_f32(b_.val[1], a_.val[1]);
        c_.val[2] = vaddq_f32(b_.val[2], a_.val[2]);
        c_.val[3] = vaddq_f32(b_.val[3], a_.val[3]);
        vst1q_f32_x4(c + len - n, c_);
        n -= 16;
    }

    if (n >= 8) {
        auto a_ = vld1q_f32_x2(a + len - n);
        auto b_ = vld1q_f32_x2(b + len - n);
        b_.val[0] = vmulq_n_f32(b_.val[0], bf);
        b_.val[1] = vmulq_n_f32(b_.val[1], bf);
        float32x4x2_t c_;
        c_.val[0] = vaddq_f32(b_.val[0], a_.val[0]);
        c_.val[1] = vaddq_f32(b_.val[1], a_.val[1]);
        vst1q_f32_x2(c + len - n, c_);
        n -= 8;
    }

    if (n >= 4) {
        auto a_ = vld1q_f32(a + len - n);
        auto b_ = vld1q_f32(b + len - n);
        b_ = vmulq_n_f32(b_, bf);
        float32x4_t c_ = vaddq_f32(b_, a_);
        vst1q_f32(c + len - n, c_);
        n -= 4;
    }

    if (n == 3) {
        float32x4_t a_ = {0.0f, 0.0f, 0.0f, 0.0f};
        float32x4_t b_ = {0.0f, 0.0f, 0.0f, 0.0f};

        a_ = vld1q_lane_f32(a + len - n + 2, a_, 2);
        a_ = vld1q_lane_f32(a + len - n + 1, a_, 1);
        a_ = vld1q_lane_f32(a + len - n, a_, 0);
        b_ = vld1q_lane_f32(b + len - n + 2, b_, 2);
        b_ = vld1q_lane_f32(b + len - n + 1, b_, 1);
        b_ = vld1q_lane_f32(b + len - n, b_, 0);
        b_ = vmulq_n_f32(b_, bf);
        float32x4_t c_ = vaddq_f32(b_, a_);
        vst1q_lane_f32(c + len - n + 2, c_, 2);
        vst1q_lane_f32(c + len - n + 1, c_, 1);
        vst1q_lane_f32(c + len - n, c_, 0);
    }
    if (n == 2) {
        float32x4_t a_ = {0.0f, 0.0f, 0.0f, 0.0f};
        float32x4_t b_ = {0.0f, 0.0f, 0.0f, 0.0f};

        a_ = vld1q_lane_f32(a + len - n + 1, a_, 1);
        a_ = vld1q_lane_f32(a + len - n, a_, 0);
        b_ = vld1q_lane_f32(b + len - n + 1, b_, 1);
        b_ = vld1q_lane_f32(b + len - n, b_, 0);
        b_ = vmulq_n_f32(b_, bf);
        float32x4_t c_ = vaddq_f32(b_, a_);
        vst1q_lane_f32(c + len - n + 1, c_, 1);
        vst1q_lane_f32(c + len - n, c_, 0);
    }
    if (n == 1) {
        float32x4_t a_ = {0.0f, 0.0f, 0.0f, 0.0f};
        float32x4_t b_ = {0.0f, 0.0f, 0.0f, 0.0f};

        a_ = vld1q_lane_f32(a + len - n, a_, 0);
        b_ = vld1q_lane_f32(b + len - n, b_, 0);
        b_ = vmulq_n_f32(b_, bf);
        float32x4_t c_ = vaddq_f32(b_, a_);
        vst1q_lane_f32(c + len - n, c_, 0);
    }
}

int
fvec_madd_and_argmin_neon(size_t n, const float* a, float bf, const float* b, float* c) {
    size_t len = n;
    uint32x4_t ids = {0, 0, 0, 0};
    float32x4_t val = {
        INFINITY,
        INFINITY,
        INFINITY,
        INFINITY,
    };
    while (n >= 16) {
        auto a_ = vld1q_f32_x4(a + len - n);
        auto b_ = vld1q_f32_x4(b + len - n);
        b_.val[0] = vmulq_n_f32(b_.val[0], bf);
        b_.val[1] = vmulq_n_f32(b_.val[1], bf);
        b_.val[2] = vmulq_n_f32(b_.val[2], bf);
        b_.val[3] = vmulq_n_f32(b_.val[3], bf);
        float32x4x4_t c_;
        c_.val[0] = vaddq_f32(b_.val[0], a_.val[0]);
        c_.val[1] = vaddq_f32(b_.val[1], a_.val[1]);
        c_.val[2] = vaddq_f32(b_.val[2], a_.val[2]);
        c_.val[3] = vaddq_f32(b_.val[3], a_.val[3]);

        vst1q_f32_x4(c + len - n, c_);

        uint32_t loc = len - n;
        auto cmp = vcleq_f32(c_.val[0], val);

        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{0, 1, 2, 3}, vld1q_dup_u32(&loc)), ids);

        val = vminq_f32(c_.val[0], val);

        cmp = vcleq_f32(c_.val[1], val);

        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{4, 5, 6, 7}, vld1q_dup_u32(&loc)), ids);

        val = vminq_f32(val, c_.val[1]);

        cmp = vcleq_f32(c_.val[2], val);

        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{8, 9, 10, 11}, vld1q_dup_u32(&loc)), ids);

        val = vminq_f32(val, c_.val[2]);

        cmp = vcleq_f32(c_.val[3], val);

        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{12, 13, 14, 15}, vld1q_dup_u32(&loc)), ids);

        val = vminq_f32(val, c_.val[3]);

        n -= 16;
    }

    if (n >= 8) {
        auto a_ = vld1q_f32_x2(a + len - n);
        auto b_ = vld1q_f32_x2(b + len - n);
        b_.val[0] = vmulq_n_f32(b_.val[0], bf);
        b_.val[1] = vmulq_n_f32(b_.val[1], bf);
        float32x4x2_t c_;
        c_.val[0] = vaddq_f32(b_.val[0], a_.val[0]);
        c_.val[1] = vaddq_f32(b_.val[1], a_.val[1]);
        vst1q_f32_x2(c + len - n, c_);

        uint32_t loc = len - n;

        auto cmp = vcleq_f32(c_.val[0], val);
        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{0, 1, 2, 3}, vld1q_dup_u32(&loc)), ids);
        val = vminq_f32(val, c_.val[0]);
        cmp = vcleq_f32(c_.val[1], val);
        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{4, 5, 6, 7}, vld1q_dup_u32(&loc)), ids);
        val = vminq_f32(val, c_.val[1]);
        n -= 8;
    }

    if (n >= 4) {
        auto a_ = vld1q_f32(a + len - n);
        auto b_ = vld1q_f32(b + len - n);
        b_ = vmulq_n_f32(b_, bf);
        float32x4_t c_ = vaddq_f32(b_, a_);
        vst1q_f32(c + len - n, c_);

        uint32_t loc = len - n;

        auto cmp = vcleq_f32(c_, val);

        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{0, 1, 2, 3}, vld1q_dup_u32(&loc)), ids);

        val = vminq_f32(val, c_);
        n -= 4;
    }

    if (n == 3) {
        float32x4_t a_ = {0.0f, 0.0f, 0.0f, 0.0f};
        float32x4_t b_ = {0.0f, 0.0f, 0.0f, 0.0f};

        a_ = vld1q_lane_f32(a + len - n + 2, a_, 2);
        a_ = vld1q_lane_f32(a + len - n + 1, a_, 1);
        a_ = vld1q_lane_f32(a + len - n, a_, 0);
        b_ = vld1q_lane_f32(b + len - n + 2, b_, 2);
        b_ = vld1q_lane_f32(b + len - n + 1, b_, 1);
        b_ = vld1q_lane_f32(b + len - n, b_, 0);
        b_ = vmulq_n_f32(b_, bf);
        float32x4_t c_ = vaddq_f32(b_, a_);
        vst1q_lane_f32(c + len - n + 2, c_, 2);
        vst1q_lane_f32(c + len - n + 1, c_, 1);
        vst1q_lane_f32(c + len - n, c_, 0);
        uint32_t loc = len - n;
        c_ = vsetq_lane_f32(INFINITY, c_, 3);
        auto cmp = vcleq_f32(c_, val);
        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{0, 1, 2, 3}, vld1q_dup_u32(&loc)), ids);
    }
    if (n == 2) {
        float32x4_t a_ = {0.0f, 0.0f, 0.0f, 0.0f};
        float32x4_t b_ = {0.0f, 0.0f, 0.0f, 0.0f};

        a_ = vld1q_lane_f32(a + len - n + 1, a_, 1);
        a_ = vld1q_lane_f32(a + len - n, a_, 0);
        b_ = vld1q_lane_f32(b + len - n + 1, b_, 1);
        b_ = vld1q_lane_f32(b + len - n, b_, 0);
        b_ = vmulq_n_f32(b_, bf);
        float32x4_t c_ = vaddq_f32(b_, a_);
        vst1q_lane_f32(c + len - n + 1, c_, 1);
        vst1q_lane_f32(c + len - n, c_, 0);
        uint32_t loc = len - n;
        c_ = vsetq_lane_f32(INFINITY, c_, 2);
        c_ = vsetq_lane_f32(INFINITY, c_, 3);
        auto cmp = vcleq_f32(c_, val);
        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{0, 1, 2, 3}, vld1q_dup_u32(&loc)), ids);
    }
    if (n == 1) {
        float32x4_t a_ = {0.0f, 0.0f, 0.0f, 0.0f};
        float32x4_t b_ = {0.0f, 0.0f, 0.0f, 0.0f};

        a_ = vld1q_lane_f32(a + len - n, a_, 0);
        b_ = vld1q_lane_f32(b + len - n, b_, 0);
        b_ = vmulq_n_f32(b_, bf);
        float32x4_t c_ = vaddq_f32(b_, a_);
        vst1q_lane_f32(c + len - n, c_, 0);
        uint32_t loc = len - n;
        c_ = vsetq_lane_f32(INFINITY, c_, 1);
        c_ = vsetq_lane_f32(INFINITY, c_, 2);
        c_ = vsetq_lane_f32(INFINITY, c_, 3);
        auto cmp = vcleq_f32(c_, val);

        ids = vbslq_u32(cmp, vaddq_u32(uint32x4_t{0, 1, 2, 3}, vld1q_dup_u32(&loc)), ids);
    }

    uint32_t ids_[4];
    vst1q_u32(ids_, ids);
    float32_t min_ = INFINITY;
    uint32_t ans_ = 0;

    for (int i = 0; i < 4; ++i) {
        if (c[ids_[i]] < min_) {
            ans_ = ids_[i];
            min_ = c[ids_[i]];
        }
    }
    return ans_;
}

// trust the compiler to unroll this properly
int32_t
ivec_inner_product_neon(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;
    for (i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

// trust the compiler to unroll this properly
int32_t
ivec_L2sqr_neon(const int8_t* x, const int8_t* y, size_t d) {
    size_t i;
    int32_t res = 0;
    for (i = 0; i < d; i++) {
        const int32_t tmp = (int32_t)x[i] - (int32_t)y[i];
        res += tmp * tmp;
    }
    return res;
}

}  // namespace faiss
#endif
