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

namespace faiss {
namespace cppcontrib {
namespace knowhere {

namespace {
inline float32x4x4_t
vcvt4_f32_half(const uint16x4x4_t x) {
    float32x4x4_t c;
    c.val[0] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[0]), 16));
    c.val[1] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[1]), 16));
    c.val[2] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[2]), 16));
    c.val[3] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[3]), 16));
    return c;
}

inline float32x4x2_t
vcvt2_f32_half(const uint16x4x2_t x) {
    float32x4x2_t c;
    c.val[0] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[0]), 16));
    c.val[1] = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x.val[1]), 16));
    return c;
}

inline float32x4_t
vcvt_f32_half(const uint16x4_t x) {
    return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(x), 16));
}

void
fvec_L2sqr_ny_dimN_neon(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr_neon(x, y, d);
        y += d;
    }
}

void
fvec_L2sqr_ny_dim1_neon(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    const float x_val = x[0];
    float32x4_t x_vec = vdupq_n_f32(x_val);
    size_t i = 0;
    while (ny - i >= 16) {
        float32x4x4_t y_vec4x4 = vld4q_f32(y + i);
        y_vec4x4.val[0] = vsubq_f32(x_vec, y_vec4x4.val[0]);
        y_vec4x4.val[1] = vsubq_f32(x_vec, y_vec4x4.val[1]);
        y_vec4x4.val[2] = vsubq_f32(x_vec, y_vec4x4.val[2]);
        y_vec4x4.val[3] = vsubq_f32(x_vec, y_vec4x4.val[3]);
        float32x4x4_t res;
        res.val[0] = vmulq_f32(y_vec4x4.val[0], y_vec4x4.val[0]);
        res.val[1] = vmulq_f32(y_vec4x4.val[1], y_vec4x4.val[1]);
        res.val[2] = vmulq_f32(y_vec4x4.val[2], y_vec4x4.val[2]);
        res.val[3] = vmulq_f32(y_vec4x4.val[3], y_vec4x4.val[3]);
        vst4q_f32(dis + i, res);
        i += 16;
    }

    if (ny - i >= 8) {
        float32x4x2_t y_vec4x2 = vld1q_f32_x2(y + i);
        y_vec4x2.val[0] = vsubq_f32(x_vec, y_vec4x2.val[0]);
        y_vec4x2.val[1] = vsubq_f32(x_vec, y_vec4x2.val[1]);
        float32x4x2_t res;
        res.val[0] = vmulq_f32(y_vec4x2.val[0], y_vec4x2.val[0]);
        res.val[1] = vmulq_f32(y_vec4x2.val[1], y_vec4x2.val[1]);
        vst1q_f32(dis + i, res.val[0]);
        vst1q_f32(dis + i + 4, res.val[1]);
        i += 8;
    }

    if (ny - i >= 4) {
        float32x4_t y0 = vld1q_f32(y + i);
        y0 = vsubq_f32(x_vec, y0);
        float32x4_t res;
        res = vmulq_f32(y0, y0);
        vst1q_f32(dis + i, res);
        i += 4;
    }

    if (ny - i > 0) {
        switch (ny - i) {
            case 3:
                dis[i + 2] = (x[0] - y[i + 2]) * (x[0] - y[i + 2]);
            case 2:
                dis[i + 1] = (x[0] - y[i + 1]) * (x[0] - y[i + 1]);
            case 1:
                dis[i] = (x[0] - y[i]) * (x[0] - y[i]);
            default:
                break;
        }
    }
}

void
fvec_L2sqr_ny_dim2_neon(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    float32x4_t x0 = {x[0], x[1], x[0], x[1]};
    float32x4_t x1 = x0;
    size_t i = 0;
    while (ny - i >= 8) {
        float32x4x2_t y0 = vld1q_f32_x2(y + i * d);
        float32x4x2_t y1 = vld1q_f32_x2(y + (i + 4) * d);
        float32x4x2_t res0 = {vsubq_f32(x0, y0.val[0]), vsubq_f32(x0, y0.val[1])};
        float32x4x2_t res1 = {vsubq_f32(x1, y1.val[0]), vsubq_f32(x1, y1.val[1])};
        res0 = {vmulq_f32(res0.val[0], res0.val[0]), vmulq_f32(res0.val[1], res0.val[1])};
        res1 = {vmulq_f32(res1.val[0], res1.val[0]), vmulq_f32(res1.val[1], res1.val[1])};

        float32x4_t even = vuzp1q_f32(res0.val[0], res0.val[1]);
        float32x4_t odd = vuzp2q_f32(res0.val[0], res0.val[1]);
        vst1q_f32(dis + i, vaddq_f32(even, odd));
        even = vuzp1q_f32(res1.val[0], res1.val[1]);
        odd = vuzp2q_f32(res1.val[0], res1.val[1]);
        vst1q_f32(dis + i + 4, vaddq_f32(even, odd));
        i += 8;
    }
    if (ny - i >= 4) {
        float32x4_t y0 = vld1q_f32(y + i * d);
        float32x4_t y1 = vld1q_f32(y + (i + 2) * d);
        y0 = vsubq_f32(x0, y0);
        y1 = vsubq_f32(x1, y1);
        y0 = vmulq_f32(y0, y0);
        y1 = vmulq_f32(y1, y1);
        float32x4_t low = vuzp1q_f32(y0, y1);
        float32x4_t high = vuzp2q_f32(y0, y1);
        float32x4_t res = vaddq_f32(low, high);
        vst1q_f32(dis + i, res);
        i += 4;
    }
    if (ny - i > 0) {
        for (; i < ny; i++) {
            dis[i] = (x[0] - y[i * d]) * (x[0] - y[i * d]) + (x[1] - y[i * d + 1]) * (x[1] - y[i * d + 1]);
        }
    }
}

void
fvec_L2sqr_ny_dim4_neon(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    float32x4_t x_vec = vld1q_f32(x);
    size_t i = 0;
    for (; i + 4 <= ny; i += 4) {
        float32x4x4_t y_vec = vld1q_f32_x4(y + i * d);
        y_vec.val[0] = vsubq_f32(x_vec, y_vec.val[0]);
        y_vec.val[1] = vsubq_f32(x_vec, y_vec.val[1]);
        y_vec.val[2] = vsubq_f32(x_vec, y_vec.val[2]);
        y_vec.val[3] = vsubq_f32(x_vec, y_vec.val[3]);

        float32x4x4_t res;
        res.val[0] = vmulq_f32(y_vec.val[0], y_vec.val[0]);
        res.val[1] = vmulq_f32(y_vec.val[1], y_vec.val[1]);
        res.val[2] = vmulq_f32(y_vec.val[2], y_vec.val[2]);
        res.val[3] = vmulq_f32(y_vec.val[3], y_vec.val[3]);

        dis[i] = vaddvq_f32(res.val[0]);
        dis[i + 1] = vaddvq_f32(res.val[1]);
        dis[i + 2] = vaddvq_f32(res.val[2]);
        dis[i + 3] = vaddvq_f32(res.val[3]);
    }
    for (; i < ny; i++) {
        float32x4_t y_vec = vld1q_f32(y + i * d);
        y_vec = vsubq_f32(x_vec, y_vec);
        float32x4_t res = vmulq_f32(y_vec, y_vec);
        dis[i] = vaddvq_f32(res);
    }
}
}  // namespace

// The main goal is to reduce the original precision of floats to maintain consistency with the distance result
// precision of the cardinal index.
__attribute__((always_inline)) inline float32x4_t
bf16_float_neon(float32x4_t f) {
    // Convert float to integer bits
    uint32x4_t bits = vreinterpretq_u32_f32(f);

    // Add rounding constant
    uint32x4_t rounded_bits = vaddq_u32(bits, vdupq_n_u32(0x8000));

    // Mask to retain only the upper 16 bits (for BF16 representation)
    rounded_bits = vandq_u32(rounded_bits, vdupq_n_u32(0xFFFF0000));

    // Convert back to float
    return vreinterpretq_f32_u32(rounded_bits);
}

float
fvec_inner_product_neon(const float* x, const float* y, size_t d) {
    float32x4_t suma = vdupq_n_f32(0.0f);
    float32x4_t sumb = vdupq_n_f32(0.0f);
    while (d >= 16) {
        float32x4_t a0 = vld1q_f32(x);
        float32x4_t a1 = vld1q_f32(x + 4);
        float32x4_t a2 = vld1q_f32(x + 8);
        float32x4_t a3 = vld1q_f32(x + 12);
        x += 16;

        float32x4_t b0 = vld1q_f32(y);
        float32x4_t b1 = vld1q_f32(y + 4);
        float32x4_t b2 = vld1q_f32(y + 8);
        float32x4_t b3 = vld1q_f32(y + 12);
        y += 16;

        suma = vfmaq_f32(suma, a0, b0);
        sumb = vfmaq_f32(sumb, a1, b1);
        suma = vfmaq_f32(suma, a2, b2);
        sumb = vfmaq_f32(sumb, a3, b3);
        d -= 16;
    }

    suma = vaddq_f32(suma, sumb);

    if (d >= 8) {
        float32x4_t a0 = vld1q_f32(x);
        float32x4_t a1 = vld1q_f32(x + 4);
        x += 8;

        float32x4_t b0 = vld1q_f32(y);
        float32x4_t b1 = vld1q_f32(y + 4);
        y += 8;

        suma = vfmaq_f32(suma, a0, b0);
        suma = vfmaq_f32(suma, a1, b1);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a0 = vld1q_f32(x);
        x += 4;
        suma = vfmaq_f32(suma, a0, vld1q_f32(y));
        y += 4;
        d -= 4;
    }

    float s0 = vaddvq_f32(suma);
    while (d > 0) {
        s0 = fmaf(*x++, *y++, s0);
        --d;
    }
    return s0;
}

float
fvec_L2sqr_neon(const float* x, const float* y, size_t d) {
    float32x4_t suma = vdupq_n_f32(0.0f);
    float32x4_t sumb = vdupq_n_f32(0.0f);
    while (d >= 16) {
        float32x4_t a0 = vld1q_f32(x);
        float32x4_t a1 = vld1q_f32(x + 4);
        float32x4_t a2 = vld1q_f32(x + 8);
        float32x4_t a3 = vld1q_f32(x + 12);
        x += 16;

        float32x4_t d0 = vsubq_f32(a0, vld1q_f32(y));
        float32x4_t d1 = vsubq_f32(a1, vld1q_f32(y + 4));
        float32x4_t d2 = vsubq_f32(a2, vld1q_f32(y + 8));
        float32x4_t d3 = vsubq_f32(a3, vld1q_f32(y + 12));
        y += 16;

        suma = vfmaq_f32(suma, d0, d0);
        sumb = vfmaq_f32(sumb, d1, d1);
        suma = vfmaq_f32(suma, d2, d2);
        sumb = vfmaq_f32(sumb, d3, d3);
        d -= 16;
    }

    suma = vaddq_f32(suma, sumb);

    if (d >= 8) {
        float32x4_t a0 = vld1q_f32(x);
        float32x4_t a1 = vld1q_f32(x + 4);
        x += 8;

        float32x4_t d0 = vsubq_f32(a0, vld1q_f32(y));
        float32x4_t d1 = vsubq_f32(a1, vld1q_f32(y + 4));
        y += 8;

        suma = vfmaq_f32(suma, d0, d0);
        suma = vfmaq_f32(suma, d1, d1);
        d -= 8;
    }
    if (d >= 4) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(x), vld1q_f32(y));
        x += 4;
        y += 4;
        suma = vfmaq_f32(suma, d0, d0);
        d -= 4;
    }

    float s0 = vaddvq_f32(suma);
    while (d > 0) {
        float diff = (*x++) - (*y++);
        s0 = fmaf(diff, diff, s0);
        --d;
    }
    return s0;
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

    float32x4_t res_x = vdupq_n_f32(0.0f);
    float32x4_t res_y = vdupq_n_f32(0.0f);
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
    float32x4_t sum_ = vdupq_n_f32(0.0f);

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

    float32x4_t res_x = vdupq_n_f32(0.0f);
    float32x4_t res_y = vdupq_n_f32(0.0f);
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
    float32x4_t sum_ = vdupq_n_f32(0.0f);
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

    float32x4_t res_x = vdupq_n_f32(0.0f);
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

void
fvec_L2sqr_ny_neon(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    switch (d) {
        case 1:
            fvec_L2sqr_ny_dim1_neon(dis, x, y, d, ny);
            break;
        case 2:
            fvec_L2sqr_ny_dim2_neon(dis, x, y, d, ny);
            break;
        case 4:
            fvec_L2sqr_ny_dim4_neon(dis, x, y, d, ny);
            break;
        default:
            fvec_L2sqr_ny_dimN_neon(dis, x, y, d, ny);
            break;
    }
}

/// compute ny square L2 distance between x and a set of contiguous y vectors
/// and return the index of the nearest vector.
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_neon(float* __restrict distances_tmp_buffer, const float* __restrict x, const float* __restrict y,
                           size_t d, size_t ny) {
    fvec_L2sqr_ny_neon(distances_tmp_buffer, x, y, d, ny);

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
        float32x4_t a_ = vdupq_n_f32(0.0f);
        float32x4_t b_ = vdupq_n_f32(0.0f);

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
        float32x4_t a_ = vdupq_n_f32(0.0f);
        float32x4_t b_ = vdupq_n_f32(0.0f);

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
        float32x4_t a_ = vdupq_n_f32(0.0f);
        float32x4_t b_ = vdupq_n_f32(0.0f);

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
    uint32x4_t ids = vdupq_n_u32(0);
    float32x4_t val = vdupq_n_f32(INFINITY);
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
        float32x4_t a_ = vdupq_n_f32(0.0f);
        float32x4_t b_ = vdupq_n_f32(0.0f);

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
        float32x4_t a_ = vdupq_n_f32(0.0f);
        float32x4_t b_ = vdupq_n_f32(0.0f);

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
        float32x4_t a_ = vdupq_n_f32(0.0f);
        float32x4_t b_ = vdupq_n_f32(0.0f);

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

void
fvec_inner_product_batch_4_neon(const float* __restrict x, const float* __restrict y0, const float* __restrict y1,
                                const float* __restrict y2, const float* __restrict y3, const size_t dim, float& dis0,
                                float& dis1, float& dis2, float& dis3) {
    float32x4_t sum0a = vdupq_n_f32(0.0f);
    float32x4_t sum0b = vdupq_n_f32(0.0f);
    float32x4_t sum1a = vdupq_n_f32(0.0f);
    float32x4_t sum1b = vdupq_n_f32(0.0f);
    float32x4_t sum2a = vdupq_n_f32(0.0f);
    float32x4_t sum2b = vdupq_n_f32(0.0f);
    float32x4_t sum3a = vdupq_n_f32(0.0f);
    float32x4_t sum3b = vdupq_n_f32(0.0f);

    size_t d = dim;
    while (d >= 16) {
        float32x4_t a0 = vld1q_f32(x);
        float32x4_t a1 = vld1q_f32(x + 4);
        float32x4_t a2 = vld1q_f32(x + 8);
        float32x4_t a3 = vld1q_f32(x + 12);
        x += 16;

        float32x4_t b00 = vld1q_f32(y0);
        float32x4_t b01 = vld1q_f32(y0 + 4);
        float32x4_t b02 = vld1q_f32(y0 + 8);
        float32x4_t b03 = vld1q_f32(y0 + 12);
        y0 += 16;

        float32x4_t b10 = vld1q_f32(y1);
        float32x4_t b11 = vld1q_f32(y1 + 4);
        float32x4_t b12 = vld1q_f32(y1 + 8);
        float32x4_t b13 = vld1q_f32(y1 + 12);
        y1 += 16;

        sum0a = vfmaq_f32(sum0a, a0, b00);
        sum1a = vfmaq_f32(sum1a, a0, b10);
        sum0b = vfmaq_f32(sum0b, a1, b01);
        sum1b = vfmaq_f32(sum1b, a1, b11);
        sum0a = vfmaq_f32(sum0a, a2, b02);
        sum1a = vfmaq_f32(sum1a, a2, b12);
        sum0b = vfmaq_f32(sum0b, a3, b03);
        sum1b = vfmaq_f32(sum1b, a3, b13);

        float32x4_t b20 = vld1q_f32(y2);
        float32x4_t b21 = vld1q_f32(y2 + 4);
        float32x4_t b22 = vld1q_f32(y2 + 8);
        float32x4_t b23 = vld1q_f32(y2 + 12);
        y2 += 16;

        float32x4_t b30 = vld1q_f32(y3);
        float32x4_t b31 = vld1q_f32(y3 + 4);
        float32x4_t b32 = vld1q_f32(y3 + 8);
        float32x4_t b33 = vld1q_f32(y3 + 12);
        y3 += 16;

        sum2a = vfmaq_f32(sum2a, a0, b20);
        sum3a = vfmaq_f32(sum3a, a0, b30);
        sum2b = vfmaq_f32(sum2b, a1, b21);
        sum3b = vfmaq_f32(sum3b, a1, b31);
        sum2a = vfmaq_f32(sum2a, a2, b22);
        sum3a = vfmaq_f32(sum3a, a2, b32);
        sum2b = vfmaq_f32(sum2b, a3, b23);
        sum3b = vfmaq_f32(sum3b, a3, b33);

        d -= 16;
    }

    sum0a = vaddq_f32(sum0a, sum0b);
    sum1a = vaddq_f32(sum1a, sum1b);
    sum2a = vaddq_f32(sum2a, sum2b);
    sum3a = vaddq_f32(sum3a, sum3b);

    if (d >= 8) {
        float32x4_t a0 = vld1q_f32(x);
        float32x4_t a1 = vld1q_f32(x + 4);
        x += 8;

        float32x4_t b00 = vld1q_f32(y0);
        float32x4_t b01 = vld1q_f32(y0 + 4);
        y0 += 8;
        float32x4_t b10 = vld1q_f32(y1);
        float32x4_t b11 = vld1q_f32(y1 + 4);
        y1 += 8;
        sum0a = vfmaq_f32(sum0a, a0, b00);
        sum1a = vfmaq_f32(sum1a, a0, b10);
        sum0a = vfmaq_f32(sum0a, a1, b01);
        sum1a = vfmaq_f32(sum1a, a1, b11);

        float32x4_t b20 = vld1q_f32(y2);
        float32x4_t b21 = vld1q_f32(y2 + 4);
        y2 += 8;
        float32x4_t b30 = vld1q_f32(y3);
        float32x4_t b31 = vld1q_f32(y3 + 4);
        y3 += 8;
        sum2a = vfmaq_f32(sum2a, a0, b20);
        sum3a = vfmaq_f32(sum3a, a0, b30);
        sum2a = vfmaq_f32(sum2a, a1, b21);
        sum3a = vfmaq_f32(sum3a, a1, b31);

        d -= 8;
    }

    if (d >= 4) {
        float32x4_t a0 = vld1q_f32(x);
        x += 4;
        sum0a = vfmaq_f32(sum0a, a0, vld1q_f32(y0));
        y0 += 4;
        sum1a = vfmaq_f32(sum1a, a0, vld1q_f32(y1));
        y1 += 4;
        sum2a = vfmaq_f32(sum2a, a0, vld1q_f32(y2));
        y2 += 4;
        sum3a = vfmaq_f32(sum3a, a0, vld1q_f32(y3));
        y3 += 4;
        d -= 4;
    }

    float s0 = vaddvq_f32(sum0a);
    float s1 = vaddvq_f32(sum1a);
    float s2 = vaddvq_f32(sum2a);
    float s3 = vaddvq_f32(sum3a);

    while (d > 0) {
        float vx = *x++;
        s0 = fmaf(vx, *y0++, s0);
        s1 = fmaf(vx, *y1++, s1);
        s2 = fmaf(vx, *y2++, s2);
        s3 = fmaf(vx, *y3++, s3);
        --d;
    }

    dis0 = s0;
    dis1 = s1;
    dis2 = s2;
    dis3 = s3;
}

void
fvec_L2sqr_batch_4_neon(const float* __restrict x, const float* __restrict y0, const float* __restrict y1,
                        const float* __restrict y2, const float* __restrict y3, const size_t dim, float& dis0,
                        float& dis1, float& dis2, float& dis3) {
    float32x4_t sum0a = vdupq_n_f32(0.0f);
    float32x4_t sum0b = vdupq_n_f32(0.0f);
    float32x4_t sum1a = vdupq_n_f32(0.0f);
    float32x4_t sum1b = vdupq_n_f32(0.0f);
    float32x4_t sum2a = vdupq_n_f32(0.0f);
    float32x4_t sum2b = vdupq_n_f32(0.0f);
    float32x4_t sum3a = vdupq_n_f32(0.0f);
    float32x4_t sum3b = vdupq_n_f32(0.0f);

    size_t d = dim;
    while (d >= 16) {
        float32x4_t a0 = vld1q_f32(x);
        float32x4_t a1 = vld1q_f32(x + 4);
        float32x4_t a2 = vld1q_f32(x + 8);
        float32x4_t a3 = vld1q_f32(x + 12);
        x += 16;

        float32x4_t d00 = vsubq_f32(a0, vld1q_f32(y0));
        float32x4_t d01 = vsubq_f32(a1, vld1q_f32(y0 + 4));
        float32x4_t d02 = vsubq_f32(a2, vld1q_f32(y0 + 8));
        float32x4_t d03 = vsubq_f32(a3, vld1q_f32(y0 + 12));
        y0 += 16;

        float32x4_t d10 = vsubq_f32(a0, vld1q_f32(y1));
        float32x4_t d11 = vsubq_f32(a1, vld1q_f32(y1 + 4));
        float32x4_t d12 = vsubq_f32(a2, vld1q_f32(y1 + 8));
        float32x4_t d13 = vsubq_f32(a3, vld1q_f32(y1 + 12));
        y1 += 16;

        sum0a = vfmaq_f32(sum0a, d00, d00);
        sum1a = vfmaq_f32(sum1a, d10, d10);
        sum0b = vfmaq_f32(sum0b, d01, d01);
        sum1b = vfmaq_f32(sum1b, d11, d11);
        sum0a = vfmaq_f32(sum0a, d02, d02);
        sum1a = vfmaq_f32(sum1a, d12, d12);
        sum0b = vfmaq_f32(sum0b, d03, d03);
        sum1b = vfmaq_f32(sum1b, d13, d13);

        float32x4_t d20 = vsubq_f32(a0, vld1q_f32(y2));
        float32x4_t d21 = vsubq_f32(a1, vld1q_f32(y2 + 4));
        float32x4_t d22 = vsubq_f32(a2, vld1q_f32(y2 + 8));
        float32x4_t d23 = vsubq_f32(a3, vld1q_f32(y2 + 12));
        y2 += 16;

        float32x4_t d30 = vsubq_f32(a0, vld1q_f32(y3));
        float32x4_t d31 = vsubq_f32(a1, vld1q_f32(y3 + 4));
        float32x4_t d32 = vsubq_f32(a2, vld1q_f32(y3 + 8));
        float32x4_t d33 = vsubq_f32(a3, vld1q_f32(y3 + 12));
        y3 += 16;

        sum2a = vfmaq_f32(sum2a, d20, d20);
        sum3a = vfmaq_f32(sum3a, d30, d30);
        sum2b = vfmaq_f32(sum2b, d21, d21);
        sum3b = vfmaq_f32(sum3b, d31, d31);
        sum2a = vfmaq_f32(sum2a, d22, d22);
        sum3a = vfmaq_f32(sum3a, d32, d32);
        sum2b = vfmaq_f32(sum2b, d23, d23);
        sum3b = vfmaq_f32(sum3b, d33, d33);

        d -= 16;
    }

    sum0a = vaddq_f32(sum0a, sum0b);
    sum1a = vaddq_f32(sum1a, sum1b);
    sum2a = vaddq_f32(sum2a, sum2b);
    sum3a = vaddq_f32(sum3a, sum3b);

    if (d >= 8) {
        float32x4_t a0 = vld1q_f32(x);
        float32x4_t a1 = vld1q_f32(x + 4);
        x += 8;

        float32x4_t d00 = vsubq_f32(a0, vld1q_f32(y0));
        float32x4_t d01 = vsubq_f32(a1, vld1q_f32(y0 + 4));
        y0 += 8;
        float32x4_t d10 = vsubq_f32(a0, vld1q_f32(y1));
        float32x4_t d11 = vsubq_f32(a1, vld1q_f32(y1 + 4));
        y1 += 8;
        sum0a = vfmaq_f32(sum0a, d00, d00);
        sum1a = vfmaq_f32(sum1a, d10, d10);
        sum0a = vfmaq_f32(sum0a, d01, d01);
        sum1a = vfmaq_f32(sum1a, d11, d11);

        float32x4_t d20 = vsubq_f32(a0, vld1q_f32(y2));
        float32x4_t d21 = vsubq_f32(a1, vld1q_f32(y2 + 4));
        y2 += 8;
        float32x4_t d30 = vsubq_f32(a0, vld1q_f32(y3));
        float32x4_t d31 = vsubq_f32(a1, vld1q_f32(y3 + 4));
        y3 += 8;
        sum2a = vfmaq_f32(sum2a, d20, d20);
        sum3a = vfmaq_f32(sum3a, d30, d30);
        sum2a = vfmaq_f32(sum2a, d21, d21);
        sum3a = vfmaq_f32(sum3a, d31, d31);

        d -= 8;
    }

    if (d >= 4) {
        float32x4_t a0 = vld1q_f32(x);
        x += 4;
        float32x4_t d0 = vsubq_f32(a0, vld1q_f32(y0));
        y0 += 4;
        float32x4_t d1 = vsubq_f32(a0, vld1q_f32(y1));
        y1 += 4;
        float32x4_t d2 = vsubq_f32(a0, vld1q_f32(y2));
        y2 += 4;
        float32x4_t d3 = vsubq_f32(a0, vld1q_f32(y3));
        y3 += 4;

        sum0a = vfmaq_f32(sum0a, d0, d0);
        sum1a = vfmaq_f32(sum1a, d1, d1);
        sum2a = vfmaq_f32(sum2a, d2, d2);
        sum3a = vfmaq_f32(sum3a, d3, d3);

        d -= 4;
    }

    float s0 = vaddvq_f32(sum0a);
    float s1 = vaddvq_f32(sum1a);
    float s2 = vaddvq_f32(sum2a);
    float s3 = vaddvq_f32(sum3a);

    while (d > 0) {
        float vx = *x++;
        float df0 = vx - (*y0++);
        float df1 = vx - (*y1++);
        float df2 = vx - (*y2++);
        float df3 = vx - (*y3++);
        s0 = fmaf(df0, df0, s0);
        s1 = fmaf(df1, df1, s1);
        s2 = fmaf(df2, df2, s2);
        s3 = fmaf(df3, df3, s3);
        --d;
    }

    dis0 = s0;
    dis1 = s1;
    dis2 = s2;
    dis3 = s3;
}

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete

int32_t
ivec_inner_product_neon(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    for (size_t i = 0; i < d; i++) {
        res += (int32_t)x[i] * y[i];
    }
    return res;
}

int32_t
ivec_L2sqr_neon(const int8_t* x, const int8_t* y, size_t d) {
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
fp16_vec_inner_product_neon(const ::knowhere::fp16* x, const ::knowhere::fp16* y, size_t d) {
    const auto* x_f16 = reinterpret_cast<const __fp16*>(x);
    const auto* y_f16 = reinterpret_cast<const __fp16*>(y);

    float32x4_t suma = vdupq_n_f32(0.0f);
    float32x4_t sumb = vdupq_n_f32(0.0f);
    while (d >= 16) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        float16x8_t x_h1 = vld1q_f16(x_f16 + 8);
        x_f16 += 16;

        float16x8_t y_h0 = vld1q_f16(y_f16);
        float16x8_t y_h1 = vld1q_f16(y_f16 + 8);
        y_f16 += 16;

        suma = vfmaq_f32(suma, vcvt_f32_f16(vget_low_f16(x_h0)), vcvt_f32_f16(vget_low_f16(y_h0)));
        sumb = vfmaq_f32(sumb, vcvt_high_f32_f16(x_h0), vcvt_high_f32_f16(y_h0));
        suma = vfmaq_f32(suma, vcvt_f32_f16(vget_low_f16(x_h1)), vcvt_f32_f16(vget_low_f16(y_h1)));
        sumb = vfmaq_f32(sumb, vcvt_high_f32_f16(x_h1), vcvt_high_f32_f16(y_h1));
        d -= 16;
    }

    if (d >= 8) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        x_f16 += 8;
        float16x8_t y_h0 = vld1q_f16(y_f16);
        y_f16 += 8;

        suma = vfmaq_f32(suma, vcvt_f32_f16(vget_low_f16(x_h0)), vcvt_f32_f16(vget_low_f16(y_h0)));
        sumb = vfmaq_f32(sumb, vcvt_high_f32_f16(x_h0), vcvt_high_f32_f16(y_h0));
        d -= 8;
    }

    suma = vaddq_f32(suma, sumb);

    if (d >= 4) {
        float32x4_t a0 = vcvt_f32_f16(vld1_f16(x_f16));
        x_f16 += 4;
        suma = vfmaq_f32(suma, a0, vcvt_f32_f16(vld1_f16(y_f16)));
        y_f16 += 4;
        d -= 4;
    }

    float s0 = vaddvq_f32(suma);
    while (d > 0) {
        s0 = fmaf(static_cast<float>(*x_f16++), static_cast<float>(*y_f16++), s0);
        --d;
    }
    return s0;
}

float
fp16_vec_L2sqr_neon(const ::knowhere::fp16* x, const ::knowhere::fp16* y, size_t d) {
    const auto* x_f16 = reinterpret_cast<const __fp16*>(x);
    const auto* y_f16 = reinterpret_cast<const __fp16*>(y);

    float32x4_t suma = vdupq_n_f32(0.0f);
    float32x4_t sumb = vdupq_n_f32(0.0f);
    while (d >= 16) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        float16x8_t x_h1 = vld1q_f16(x_f16 + 8);
        x_f16 += 16;

        float16x8_t y_h0 = vld1q_f16(y_f16);
        float16x8_t y_h1 = vld1q_f16(y_f16 + 8);
        y_f16 += 16;

        float32x4_t d0 = vsubq_f32(vcvt_f32_f16(vget_low_f16(x_h0)), vcvt_f32_f16(vget_low_f16(y_h0)));
        float32x4_t d1 = vsubq_f32(vcvt_high_f32_f16(x_h0), vcvt_high_f32_f16(y_h0));
        float32x4_t d2 = vsubq_f32(vcvt_f32_f16(vget_low_f16(x_h1)), vcvt_f32_f16(vget_low_f16(y_h1)));
        float32x4_t d3 = vsubq_f32(vcvt_high_f32_f16(x_h1), vcvt_high_f32_f16(y_h1));

        suma = vfmaq_f32(suma, d0, d0);
        sumb = vfmaq_f32(sumb, d1, d1);
        suma = vfmaq_f32(suma, d2, d2);
        sumb = vfmaq_f32(sumb, d3, d3);
        d -= 16;
    }

    if (d >= 8) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        x_f16 += 8;
        float16x8_t y_h0 = vld1q_f16(y_f16);
        y_f16 += 8;

        float32x4_t d0 = vsubq_f32(vcvt_f32_f16(vget_low_f16(x_h0)), vcvt_f32_f16(vget_low_f16(y_h0)));
        float32x4_t d1 = vsubq_f32(vcvt_high_f32_f16(x_h0), vcvt_high_f32_f16(y_h0));

        suma = vfmaq_f32(suma, d0, d0);
        sumb = vfmaq_f32(sumb, d1, d1);
        d -= 8;
    }

    suma = vaddq_f32(suma, sumb);

    if (d >= 4) {
        float32x4_t d0 = vsubq_f32(vcvt_f32_f16(vld1_f16(x_f16)), vcvt_f32_f16(vld1_f16(y_f16)));
        x_f16 += 4;
        y_f16 += 4;
        suma = vfmaq_f32(suma, d0, d0);
        d -= 4;
    }

    float s0 = vaddvq_f32(suma);
    while (d > 0) {
        float diff = static_cast<float>(*x_f16++) - static_cast<float>(*y_f16++);
        s0 = fmaf(diff, diff, s0);
        --d;
    }
    return s0;
}

float
fp16_vec_norm_L2sqr_neon(const ::knowhere::fp16* x, size_t d) {
    const auto* x_f16 = reinterpret_cast<const __fp16*>(x);

    float32x4_t suma = vdupq_n_f32(0.0f);
    float32x4_t sumb = vdupq_n_f32(0.0f);
    while (d >= 16) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        float16x8_t x_h1 = vld1q_f16(x_f16 + 8);
        x_f16 += 16;

        float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x_h0));
        float32x4_t a1 = vcvt_high_f32_f16(x_h0);
        float32x4_t a2 = vcvt_f32_f16(vget_low_f16(x_h1));
        float32x4_t a3 = vcvt_high_f32_f16(x_h1);

        suma = vfmaq_f32(suma, a0, a0);
        sumb = vfmaq_f32(sumb, a1, a1);
        suma = vfmaq_f32(suma, a2, a2);
        sumb = vfmaq_f32(sumb, a3, a3);
        d -= 16;
    }

    if (d >= 8) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        x_f16 += 8;
        float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x_h0));
        float32x4_t a1 = vcvt_high_f32_f16(x_h0);
        suma = vfmaq_f32(suma, a0, a0);
        sumb = vfmaq_f32(sumb, a1, a1);
        d -= 8;
    }

    suma = vaddq_f32(suma, sumb);

    if (d >= 4) {
        float32x4_t a0 = vcvt_f32_f16(vld1_f16(x_f16));
        x_f16 += 4;
        suma = vfmaq_f32(suma, a0, a0);
        d -= 4;
    }

    float s0 = vaddvq_f32(suma);
    while (d > 0) {
        float vx = static_cast<float>(*x_f16++);
        s0 = fmaf(vx, vx, s0);
        --d;
    }
    return s0;
}

void
fp16_vec_inner_product_batch_4_neon(const ::knowhere::fp16* __restrict x, const ::knowhere::fp16* __restrict y0,
                                    const ::knowhere::fp16* __restrict y1, const ::knowhere::fp16* __restrict y2,
                                    const ::knowhere::fp16* __restrict y3, const size_t dim, float& dis0, float& dis1,
                                    float& dis2, float& dis3) {
    const auto* __restrict x_f16 = reinterpret_cast<const __fp16*>(x);
    const auto* __restrict y0_f16 = reinterpret_cast<const __fp16*>(y0);
    const auto* __restrict y1_f16 = reinterpret_cast<const __fp16*>(y1);
    const auto* __restrict y2_f16 = reinterpret_cast<const __fp16*>(y2);
    const auto* __restrict y3_f16 = reinterpret_cast<const __fp16*>(y3);

    float32x4_t sum0a = vdupq_n_f32(0.0f);
    float32x4_t sum0b = vdupq_n_f32(0.0f);
    float32x4_t sum1a = vdupq_n_f32(0.0f);
    float32x4_t sum1b = vdupq_n_f32(0.0f);
    float32x4_t sum2a = vdupq_n_f32(0.0f);
    float32x4_t sum2b = vdupq_n_f32(0.0f);
    float32x4_t sum3a = vdupq_n_f32(0.0f);
    float32x4_t sum3b = vdupq_n_f32(0.0f);

    size_t d = dim;
    while (d >= 32) {
        for (int step = 0; step < 2; ++step) {
            float16x8_t x_h0 = vld1q_f16(x_f16);
            float16x8_t x_h1 = vld1q_f16(x_f16 + 8);
            x_f16 += 16;

            float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x_h0));
            float32x4_t a1 = vcvt_high_f32_f16(x_h0);
            float32x4_t a2 = vcvt_f32_f16(vget_low_f16(x_h1));
            float32x4_t a3 = vcvt_high_f32_f16(x_h1);

            float16x8_t b0_h0 = vld1q_f16(y0_f16);
            float16x8_t b0_h1 = vld1q_f16(y0_f16 + 8);
            y0_f16 += 16;

            float16x8_t b1_h0 = vld1q_f16(y1_f16);
            float16x8_t b1_h1 = vld1q_f16(y1_f16 + 8);
            y1_f16 += 16;

            sum0a = vfmaq_f32(sum0a, a0, vcvt_f32_f16(vget_low_f16(b0_h0)));
            sum1a = vfmaq_f32(sum1a, a0, vcvt_f32_f16(vget_low_f16(b1_h0)));
            sum0b = vfmaq_f32(sum0b, a1, vcvt_high_f32_f16(b0_h0));
            sum1b = vfmaq_f32(sum1b, a1, vcvt_high_f32_f16(b1_h0));
            sum0a = vfmaq_f32(sum0a, a2, vcvt_f32_f16(vget_low_f16(b0_h1)));
            sum1a = vfmaq_f32(sum1a, a2, vcvt_f32_f16(vget_low_f16(b1_h1)));
            sum0b = vfmaq_f32(sum0b, a3, vcvt_high_f32_f16(b0_h1));
            sum1b = vfmaq_f32(sum1b, a3, vcvt_high_f32_f16(b1_h1));

            float16x8_t b2_h0 = vld1q_f16(y2_f16);
            float16x8_t b2_h1 = vld1q_f16(y2_f16 + 8);
            y2_f16 += 16;

            float16x8_t b3_h0 = vld1q_f16(y3_f16);
            float16x8_t b3_h1 = vld1q_f16(y3_f16 + 8);
            y3_f16 += 16;

            sum2a = vfmaq_f32(sum2a, a0, vcvt_f32_f16(vget_low_f16(b2_h0)));
            sum3a = vfmaq_f32(sum3a, a0, vcvt_f32_f16(vget_low_f16(b3_h0)));
            sum2b = vfmaq_f32(sum2b, a1, vcvt_high_f32_f16(b2_h0));
            sum3b = vfmaq_f32(sum3b, a1, vcvt_high_f32_f16(b3_h0));
            sum2a = vfmaq_f32(sum2a, a2, vcvt_f32_f16(vget_low_f16(b2_h1)));
            sum3a = vfmaq_f32(sum3a, a2, vcvt_f32_f16(vget_low_f16(b3_h1)));
            sum2b = vfmaq_f32(sum2b, a3, vcvt_high_f32_f16(b2_h1));
            sum3b = vfmaq_f32(sum3b, a3, vcvt_high_f32_f16(b3_h1));
        }
        d -= 32;
    }

    if (d >= 16) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        float16x8_t x_h1 = vld1q_f16(x_f16 + 8);
        x_f16 += 16;

        float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x_h0));
        float32x4_t a1 = vcvt_high_f32_f16(x_h0);
        float32x4_t a2 = vcvt_f32_f16(vget_low_f16(x_h1));
        float32x4_t a3 = vcvt_high_f32_f16(x_h1);

        float16x8_t b0_h0 = vld1q_f16(y0_f16);
        float16x8_t b0_h1 = vld1q_f16(y0_f16 + 8);
        y0_f16 += 16;
        float16x8_t b1_h0 = vld1q_f16(y1_f16);
        float16x8_t b1_h1 = vld1q_f16(y1_f16 + 8);
        y1_f16 += 16;

        sum0a = vfmaq_f32(sum0a, a0, vcvt_f32_f16(vget_low_f16(b0_h0)));
        sum1a = vfmaq_f32(sum1a, a0, vcvt_f32_f16(vget_low_f16(b1_h0)));
        sum0b = vfmaq_f32(sum0b, a1, vcvt_high_f32_f16(b0_h0));
        sum1b = vfmaq_f32(sum1b, a1, vcvt_high_f32_f16(b1_h0));
        sum0a = vfmaq_f32(sum0a, a2, vcvt_f32_f16(vget_low_f16(b0_h1)));
        sum1a = vfmaq_f32(sum1a, a2, vcvt_f32_f16(vget_low_f16(b1_h1)));
        sum0b = vfmaq_f32(sum0b, a3, vcvt_high_f32_f16(b0_h1));
        sum1b = vfmaq_f32(sum1b, a3, vcvt_high_f32_f16(b1_h1));

        float16x8_t b2_h0 = vld1q_f16(y2_f16);
        float16x8_t b2_h1 = vld1q_f16(y2_f16 + 8);
        y2_f16 += 16;
        float16x8_t b3_h0 = vld1q_f16(y3_f16);
        float16x8_t b3_h1 = vld1q_f16(y3_f16 + 8);
        y3_f16 += 16;

        sum2a = vfmaq_f32(sum2a, a0, vcvt_f32_f16(vget_low_f16(b2_h0)));
        sum3a = vfmaq_f32(sum3a, a0, vcvt_f32_f16(vget_low_f16(b3_h0)));
        sum2b = vfmaq_f32(sum2b, a1, vcvt_high_f32_f16(b2_h0));
        sum3b = vfmaq_f32(sum3b, a1, vcvt_high_f32_f16(b3_h0));
        sum2a = vfmaq_f32(sum2a, a2, vcvt_f32_f16(vget_low_f16(b2_h1)));
        sum3a = vfmaq_f32(sum3a, a2, vcvt_f32_f16(vget_low_f16(b3_h1)));
        sum2b = vfmaq_f32(sum2b, a3, vcvt_high_f32_f16(b2_h1));
        sum3b = vfmaq_f32(sum3b, a3, vcvt_high_f32_f16(b3_h1));

        d -= 16;
    }

    if (d >= 8) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        x_f16 += 8;
        float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x_h0));
        float32x4_t a1 = vcvt_high_f32_f16(x_h0);

        float16x8_t b0_h0 = vld1q_f16(y0_f16);
        y0_f16 += 8;
        float16x8_t b1_h0 = vld1q_f16(y1_f16);
        y1_f16 += 8;
        sum0a = vfmaq_f32(sum0a, a0, vcvt_f32_f16(vget_low_f16(b0_h0)));
        sum1a = vfmaq_f32(sum1a, a0, vcvt_f32_f16(vget_low_f16(b1_h0)));
        sum0b = vfmaq_f32(sum0b, a1, vcvt_high_f32_f16(b0_h0));
        sum1b = vfmaq_f32(sum1b, a1, vcvt_high_f32_f16(b1_h0));

        float16x8_t b2_h0 = vld1q_f16(y2_f16);
        y2_f16 += 8;
        float16x8_t b3_h0 = vld1q_f16(y3_f16);
        y3_f16 += 8;
        sum2a = vfmaq_f32(sum2a, a0, vcvt_f32_f16(vget_low_f16(b2_h0)));
        sum3a = vfmaq_f32(sum3a, a0, vcvt_f32_f16(vget_low_f16(b3_h0)));
        sum2b = vfmaq_f32(sum2b, a1, vcvt_high_f32_f16(b2_h0));
        sum3b = vfmaq_f32(sum3b, a1, vcvt_high_f32_f16(b3_h0));

        d -= 8;
    }

    sum0a = vaddq_f32(sum0a, sum0b);
    sum1a = vaddq_f32(sum1a, sum1b);
    sum2a = vaddq_f32(sum2a, sum2b);
    sum3a = vaddq_f32(sum3a, sum3b);

    if (d >= 4) {
        float32x4_t a0 = vcvt_f32_f16(vld1_f16(x_f16));
        x_f16 += 4;
        sum0a = vfmaq_f32(sum0a, a0, vcvt_f32_f16(vld1_f16(y0_f16)));
        y0_f16 += 4;
        sum1a = vfmaq_f32(sum1a, a0, vcvt_f32_f16(vld1_f16(y1_f16)));
        y1_f16 += 4;
        sum2a = vfmaq_f32(sum2a, a0, vcvt_f32_f16(vld1_f16(y2_f16)));
        y2_f16 += 4;
        sum3a = vfmaq_f32(sum3a, a0, vcvt_f32_f16(vld1_f16(y3_f16)));
        y3_f16 += 4;
        d -= 4;
    }

    float32x4_t pair01 = vpaddq_f32(sum0a, sum1a);
    float32x4_t pair23 = vpaddq_f32(sum2a, sum3a);
    float32x4_t totals = vpaddq_f32(pair01, pair23);
    float s0 = vgetq_lane_f32(totals, 0);
    float s1 = vgetq_lane_f32(totals, 1);
    float s2 = vgetq_lane_f32(totals, 2);
    float s3 = vgetq_lane_f32(totals, 3);

    while (d > 0) {
        float vx = static_cast<float>(*x_f16++);
        s0 = fmaf(vx, static_cast<float>(*y0_f16++), s0);
        s1 = fmaf(vx, static_cast<float>(*y1_f16++), s1);
        s2 = fmaf(vx, static_cast<float>(*y2_f16++), s2);
        s3 = fmaf(vx, static_cast<float>(*y3_f16++), s3);
        --d;
    }

    dis0 = s0;
    dis1 = s1;
    dis2 = s2;
    dis3 = s3;
}

void
fp16_vec_L2sqr_batch_4_neon(const ::knowhere::fp16* __restrict x, const ::knowhere::fp16* __restrict y0,
                            const ::knowhere::fp16* __restrict y1, const ::knowhere::fp16* __restrict y2,
                            const ::knowhere::fp16* __restrict y3, const size_t dim, float& dis0, float& dis1,
                            float& dis2, float& dis3) {
    const auto* __restrict x_f16 = reinterpret_cast<const __fp16*>(x);
    const auto* __restrict y0_f16 = reinterpret_cast<const __fp16*>(y0);
    const auto* __restrict y1_f16 = reinterpret_cast<const __fp16*>(y1);
    const auto* __restrict y2_f16 = reinterpret_cast<const __fp16*>(y2);
    const auto* __restrict y3_f16 = reinterpret_cast<const __fp16*>(y3);

    float32x4_t sum0a = vdupq_n_f32(0.0f);
    float32x4_t sum0b = vdupq_n_f32(0.0f);
    float32x4_t sum1a = vdupq_n_f32(0.0f);
    float32x4_t sum1b = vdupq_n_f32(0.0f);
    float32x4_t sum2a = vdupq_n_f32(0.0f);
    float32x4_t sum2b = vdupq_n_f32(0.0f);
    float32x4_t sum3a = vdupq_n_f32(0.0f);
    float32x4_t sum3b = vdupq_n_f32(0.0f);

    size_t d = dim;
    while (d >= 32) {
        for (int step = 0; step < 2; ++step) {
            float16x8_t x_h0 = vld1q_f16(x_f16);
            float16x8_t x_h1 = vld1q_f16(x_f16 + 8);
            x_f16 += 16;

            float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x_h0));
            float32x4_t a1 = vcvt_high_f32_f16(x_h0);
            float32x4_t a2 = vcvt_f32_f16(vget_low_f16(x_h1));
            float32x4_t a3 = vcvt_high_f32_f16(x_h1);

            float16x8_t b0_h0 = vld1q_f16(y0_f16);
            float16x8_t b0_h1 = vld1q_f16(y0_f16 + 8);
            y0_f16 += 16;

            float16x8_t b1_h0 = vld1q_f16(y1_f16);
            float16x8_t b1_h1 = vld1q_f16(y1_f16 + 8);
            y1_f16 += 16;

            float32x4_t d00 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b0_h0)));
            float32x4_t d10 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b1_h0)));
            float32x4_t d01 = vsubq_f32(a1, vcvt_high_f32_f16(b0_h0));
            float32x4_t d11 = vsubq_f32(a1, vcvt_high_f32_f16(b1_h0));
            float32x4_t d02 = vsubq_f32(a2, vcvt_f32_f16(vget_low_f16(b0_h1)));
            float32x4_t d12 = vsubq_f32(a2, vcvt_f32_f16(vget_low_f16(b1_h1)));
            float32x4_t d03 = vsubq_f32(a3, vcvt_high_f32_f16(b0_h1));
            float32x4_t d13 = vsubq_f32(a3, vcvt_high_f32_f16(b1_h1));

            sum0a = vfmaq_f32(sum0a, d00, d00);
            sum1a = vfmaq_f32(sum1a, d10, d10);
            sum0b = vfmaq_f32(sum0b, d01, d01);
            sum1b = vfmaq_f32(sum1b, d11, d11);
            sum0a = vfmaq_f32(sum0a, d02, d02);
            sum1a = vfmaq_f32(sum1a, d12, d12);
            sum0b = vfmaq_f32(sum0b, d03, d03);
            sum1b = vfmaq_f32(sum1b, d13, d13);

            float16x8_t b2_h0 = vld1q_f16(y2_f16);
            float16x8_t b2_h1 = vld1q_f16(y2_f16 + 8);
            y2_f16 += 16;

            float16x8_t b3_h0 = vld1q_f16(y3_f16);
            float16x8_t b3_h1 = vld1q_f16(y3_f16 + 8);
            y3_f16 += 16;

            float32x4_t d20 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b2_h0)));
            float32x4_t d30 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b3_h0)));
            float32x4_t d21 = vsubq_f32(a1, vcvt_high_f32_f16(b2_h0));
            float32x4_t d31 = vsubq_f32(a1, vcvt_high_f32_f16(b3_h0));
            float32x4_t d22 = vsubq_f32(a2, vcvt_f32_f16(vget_low_f16(b2_h1)));
            float32x4_t d32 = vsubq_f32(a2, vcvt_f32_f16(vget_low_f16(b3_h1)));
            float32x4_t d23 = vsubq_f32(a3, vcvt_high_f32_f16(b2_h1));
            float32x4_t d33 = vsubq_f32(a3, vcvt_high_f32_f16(b3_h1));

            sum2a = vfmaq_f32(sum2a, d20, d20);
            sum3a = vfmaq_f32(sum3a, d30, d30);
            sum2b = vfmaq_f32(sum2b, d21, d21);
            sum3b = vfmaq_f32(sum3b, d31, d31);
            sum2a = vfmaq_f32(sum2a, d22, d22);
            sum3a = vfmaq_f32(sum3a, d32, d32);
            sum2b = vfmaq_f32(sum2b, d23, d23);
            sum3b = vfmaq_f32(sum3b, d33, d33);
        }
        d -= 32;
    }

    if (d >= 16) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        float16x8_t x_h1 = vld1q_f16(x_f16 + 8);
        x_f16 += 16;

        float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x_h0));
        float32x4_t a1 = vcvt_high_f32_f16(x_h0);
        float32x4_t a2 = vcvt_f32_f16(vget_low_f16(x_h1));
        float32x4_t a3 = vcvt_high_f32_f16(x_h1);

        float16x8_t b0_h0 = vld1q_f16(y0_f16);
        float16x8_t b0_h1 = vld1q_f16(y0_f16 + 8);
        y0_f16 += 16;
        float16x8_t b1_h0 = vld1q_f16(y1_f16);
        float16x8_t b1_h1 = vld1q_f16(y1_f16 + 8);
        y1_f16 += 16;

        float32x4_t d00 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b0_h0)));
        float32x4_t d10 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b1_h0)));
        float32x4_t d01 = vsubq_f32(a1, vcvt_high_f32_f16(b0_h0));
        float32x4_t d11 = vsubq_f32(a1, vcvt_high_f32_f16(b1_h0));
        float32x4_t d02 = vsubq_f32(a2, vcvt_f32_f16(vget_low_f16(b0_h1)));
        float32x4_t d12 = vsubq_f32(a2, vcvt_f32_f16(vget_low_f16(b1_h1)));
        float32x4_t d03 = vsubq_f32(a3, vcvt_high_f32_f16(b0_h1));
        float32x4_t d13 = vsubq_f32(a3, vcvt_high_f32_f16(b1_h1));

        sum0a = vfmaq_f32(sum0a, d00, d00);
        sum1a = vfmaq_f32(sum1a, d10, d10);
        sum0b = vfmaq_f32(sum0b, d01, d01);
        sum1b = vfmaq_f32(sum1b, d11, d11);
        sum0a = vfmaq_f32(sum0a, d02, d02);
        sum1a = vfmaq_f32(sum1a, d12, d12);
        sum0b = vfmaq_f32(sum0b, d03, d03);
        sum1b = vfmaq_f32(sum1b, d13, d13);

        float16x8_t b2_h0 = vld1q_f16(y2_f16);
        float16x8_t b2_h1 = vld1q_f16(y2_f16 + 8);
        y2_f16 += 16;
        float16x8_t b3_h0 = vld1q_f16(y3_f16);
        float16x8_t b3_h1 = vld1q_f16(y3_f16 + 8);
        y3_f16 += 16;

        float32x4_t d20 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b2_h0)));
        float32x4_t d30 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b3_h0)));
        float32x4_t d21 = vsubq_f32(a1, vcvt_high_f32_f16(b2_h0));
        float32x4_t d31 = vsubq_f32(a1, vcvt_high_f32_f16(b3_h0));
        float32x4_t d22 = vsubq_f32(a2, vcvt_f32_f16(vget_low_f16(b2_h1)));
        float32x4_t d32 = vsubq_f32(a2, vcvt_f32_f16(vget_low_f16(b3_h1)));
        float32x4_t d23 = vsubq_f32(a3, vcvt_high_f32_f16(b2_h1));
        float32x4_t d33 = vsubq_f32(a3, vcvt_high_f32_f16(b3_h1));

        sum2a = vfmaq_f32(sum2a, d20, d20);
        sum3a = vfmaq_f32(sum3a, d30, d30);
        sum2b = vfmaq_f32(sum2b, d21, d21);
        sum3b = vfmaq_f32(sum3b, d31, d31);
        sum2a = vfmaq_f32(sum2a, d22, d22);
        sum3a = vfmaq_f32(sum3a, d32, d32);
        sum2b = vfmaq_f32(sum2b, d23, d23);
        sum3b = vfmaq_f32(sum3b, d33, d33);

        d -= 16;
    }

    if (d >= 8) {
        float16x8_t x_h0 = vld1q_f16(x_f16);
        x_f16 += 8;
        float32x4_t a0 = vcvt_f32_f16(vget_low_f16(x_h0));
        float32x4_t a1 = vcvt_high_f32_f16(x_h0);

        float16x8_t b0_h0 = vld1q_f16(y0_f16);
        y0_f16 += 8;
        float16x8_t b1_h0 = vld1q_f16(y1_f16);
        y1_f16 += 8;
        float32x4_t d00 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b0_h0)));
        float32x4_t d10 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b1_h0)));
        float32x4_t d01 = vsubq_f32(a1, vcvt_high_f32_f16(b0_h0));
        float32x4_t d11 = vsubq_f32(a1, vcvt_high_f32_f16(b1_h0));
        sum0a = vfmaq_f32(sum0a, d00, d00);
        sum1a = vfmaq_f32(sum1a, d10, d10);
        sum0b = vfmaq_f32(sum0b, d01, d01);
        sum1b = vfmaq_f32(sum1b, d11, d11);

        float16x8_t b2_h0 = vld1q_f16(y2_f16);
        y2_f16 += 8;
        float16x8_t b3_h0 = vld1q_f16(y3_f16);
        y3_f16 += 8;
        float32x4_t d20 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b2_h0)));
        float32x4_t d30 = vsubq_f32(a0, vcvt_f32_f16(vget_low_f16(b3_h0)));
        float32x4_t d21 = vsubq_f32(a1, vcvt_high_f32_f16(b2_h0));
        float32x4_t d31 = vsubq_f32(a1, vcvt_high_f32_f16(b3_h0));
        sum2a = vfmaq_f32(sum2a, d20, d20);
        sum3a = vfmaq_f32(sum3a, d30, d30);
        sum2b = vfmaq_f32(sum2b, d21, d21);
        sum3b = vfmaq_f32(sum3b, d31, d31);

        d -= 8;
    }

    sum0a = vaddq_f32(sum0a, sum0b);
    sum1a = vaddq_f32(sum1a, sum1b);
    sum2a = vaddq_f32(sum2a, sum2b);
    sum3a = vaddq_f32(sum3a, sum3b);

    if (d >= 4) {
        float32x4_t a0 = vcvt_f32_f16(vld1_f16(x_f16));
        x_f16 += 4;
        float32x4_t d0 = vsubq_f32(a0, vcvt_f32_f16(vld1_f16(y0_f16)));
        y0_f16 += 4;
        float32x4_t d1 = vsubq_f32(a0, vcvt_f32_f16(vld1_f16(y1_f16)));
        y1_f16 += 4;
        float32x4_t d2 = vsubq_f32(a0, vcvt_f32_f16(vld1_f16(y2_f16)));
        y2_f16 += 4;
        float32x4_t d3 = vsubq_f32(a0, vcvt_f32_f16(vld1_f16(y3_f16)));
        y3_f16 += 4;

        sum0a = vfmaq_f32(sum0a, d0, d0);
        sum1a = vfmaq_f32(sum1a, d1, d1);
        sum2a = vfmaq_f32(sum2a, d2, d2);
        sum3a = vfmaq_f32(sum3a, d3, d3);
        d -= 4;
    }

    float32x4_t pair01 = vpaddq_f32(sum0a, sum1a);
    float32x4_t pair23 = vpaddq_f32(sum2a, sum3a);
    float32x4_t totals = vpaddq_f32(pair01, pair23);
    float s0 = vgetq_lane_f32(totals, 0);
    float s1 = vgetq_lane_f32(totals, 1);
    float s2 = vgetq_lane_f32(totals, 2);
    float s3 = vgetq_lane_f32(totals, 3);

    while (d > 0) {
        float vx = static_cast<float>(*x_f16++);
        float df0 = vx - static_cast<float>(*y0_f16++);
        float df1 = vx - static_cast<float>(*y1_f16++);
        float df2 = vx - static_cast<float>(*y2_f16++);
        float df3 = vx - static_cast<float>(*y3_f16++);
        s0 = fmaf(df0, df0, s0);
        s1 = fmaf(df1, df1, s1);
        s2 = fmaf(df2, df2, s2);
        s3 = fmaf(df3, df3, s3);
        --d;
    }

    dis0 = s0;
    dis1 = s1;
    dis2 = s2;
    dis3 = s3;
}

///////////////////////////////////////////////////////////////////////////////
// bf16

float
bf16_vec_inner_product_neon(const ::knowhere::bf16* x, const ::knowhere::bf16* y, size_t d) {
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
bf16_vec_L2sqr_neon(const ::knowhere::bf16* x, const ::knowhere::bf16* y, size_t d) {
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
        uint16x4_t res_x = vdup_n_u16(0);
        uint16x4_t res_y = vdup_n_u16(0);
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
bf16_vec_norm_L2sqr_neon(const ::knowhere::bf16* x, size_t d) {
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
        uint16x4_t res_x = vdup_n_u16(0);
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
bf16_vec_inner_product_batch_4_neon(const ::knowhere::bf16* x, const ::knowhere::bf16* y0, const ::knowhere::bf16* y1,
                                    const ::knowhere::bf16* y2, const ::knowhere::bf16* y3, const size_t d, float& dis0,
                                    float& dis1, float& dis2, float& dis3) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    auto cur_d = d;
    while (cur_d >= 16) {
        float32x4x4_t a = vcvt4_f32_half(vld4_u16((const uint16_t*)x));
        float32x4x4_t b0 = vcvt4_f32_half(vld4_u16((const uint16_t*)y0));
        float32x4x4_t b1 = vcvt4_f32_half(vld4_u16((const uint16_t*)y1));
        float32x4x4_t b2 = vcvt4_f32_half(vld4_u16((const uint16_t*)y2));
        float32x4x4_t b3 = vcvt4_f32_half(vld4_u16((const uint16_t*)y3));

        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b0.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[0], b1.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[0], b2.val[0]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[0], b3.val[0]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[1], b0.val[1]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], b1.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], b2.val[1]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[1], b3.val[1]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[2], b0.val[2]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[2], b1.val[2]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[2], b2.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[2], b3.val[2]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[3], b0.val[3]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[3], b1.val[3]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[3], b2.val[3]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[3], b3.val[3]);

        cur_d -= 16;
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
    }
    if (cur_d >= 8) {
        float32x4x2_t a = vcvt2_f32_half(vld2_u16((const uint16_t*)x));
        float32x4x2_t b0 = vcvt2_f32_half(vld2_u16((const uint16_t*)y0));
        float32x4x2_t b1 = vcvt2_f32_half(vld2_u16((const uint16_t*)y1));
        float32x4x2_t b2 = vcvt2_f32_half(vld2_u16((const uint16_t*)y2));
        float32x4x2_t b3 = vcvt2_f32_half(vld2_u16((const uint16_t*)y3));
        res.val[0] = vmlaq_f32(res.val[0], a.val[0], b0.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[0], b1.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[0], b2.val[0]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[0], b3.val[0]);

        res.val[0] = vmlaq_f32(res.val[0], a.val[1], b0.val[1]);
        res.val[1] = vmlaq_f32(res.val[1], a.val[1], b1.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], a.val[1], b2.val[1]);
        res.val[3] = vmlaq_f32(res.val[3], a.val[1], b3.val[1]);

        cur_d -= 8;
        x += 8;
        y0 += 8;
        y1 += 8;
        y2 += 8;
        y3 += 8;
    }
    if (cur_d >= 4) {
        float32x4_t a = vcvt_f32_half(vld1_u16((const uint16_t*)x));
        float32x4_t b0 = vcvt_f32_half(vld1_u16((const uint16_t*)y0));
        float32x4_t b1 = vcvt_f32_half(vld1_u16((const uint16_t*)y1));
        float32x4_t b2 = vcvt_f32_half(vld1_u16((const uint16_t*)y2));
        float32x4_t b3 = vcvt_f32_half(vld1_u16((const uint16_t*)y3));
        res.val[0] = vmlaq_f32(res.val[0], a, b0);
        res.val[1] = vmlaq_f32(res.val[1], a, b1);
        res.val[2] = vmlaq_f32(res.val[2], a, b2);
        res.val[3] = vmlaq_f32(res.val[3], a, b3);
        cur_d -= 4;
        x += 4;
        y0 += 4;
        y1 += 4;
        y2 += 4;
        y3 += 4;
    }
    if (cur_d >= 0) {
        uint16x4_t res_x = vdup_n_u16(0);
        uint16x4_t res_y0 = vdup_n_u16(0);
        uint16x4_t res_y1 = vdup_n_u16(0);
        uint16x4_t res_y2 = vdup_n_u16(0);
        uint16x4_t res_y3 = vdup_n_u16(0);
        switch (cur_d) {
            case 3:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 2);
                res_y0 = vld1_lane_u16((const uint16_t*)y0, res_y0, 2);
                res_y1 = vld1_lane_u16((const uint16_t*)y1, res_y1, 2);
                res_y2 = vld1_lane_u16((const uint16_t*)y2, res_y2, 2);
                res_y3 = vld1_lane_u16((const uint16_t*)y3, res_y3, 2);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
            case 2:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 1);
                res_y0 = vld1_lane_u16((const uint16_t*)y0, res_y0, 1);
                res_y1 = vld1_lane_u16((const uint16_t*)y1, res_y1, 1);
                res_y2 = vld1_lane_u16((const uint16_t*)y2, res_y2, 1);
                res_y3 = vld1_lane_u16((const uint16_t*)y3, res_y3, 1);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
            case 1:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 0);
                res_y0 = vld1_lane_u16((const uint16_t*)y0, res_y0, 0);
                res_y1 = vld1_lane_u16((const uint16_t*)y1, res_y1, 0);
                res_y2 = vld1_lane_u16((const uint16_t*)y2, res_y2, 0);
                res_y3 = vld1_lane_u16((const uint16_t*)y3, res_y3, 0);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
        }
        res.val[0] = vmlaq_f32(res.val[0], vcvt_f32_half(res_x), vcvt_f32_half(res_y0));
        res.val[1] = vmlaq_f32(res.val[1], vcvt_f32_half(res_x), vcvt_f32_half(res_y1));
        res.val[2] = vmlaq_f32(res.val[2], vcvt_f32_half(res_x), vcvt_f32_half(res_y2));
        res.val[3] = vmlaq_f32(res.val[3], vcvt_f32_half(res_x), vcvt_f32_half(res_y3));
    }
    dis0 = vaddvq_f32(res.val[0]);
    dis1 = vaddvq_f32(res.val[1]);
    dis2 = vaddvq_f32(res.val[2]);
    dis3 = vaddvq_f32(res.val[3]);
}

void
bf16_vec_L2sqr_batch_4_neon(const ::knowhere::bf16* x, const ::knowhere::bf16* y0, const ::knowhere::bf16* y1,
                            const ::knowhere::bf16* y2, const ::knowhere::bf16* y3, const size_t d, float& dis0,
                            float& dis1, float& dis2, float& dis3) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    auto cur_d = d;
    while (cur_d >= 16) {
        float32x4x4_t a = vcvt4_f32_half(vld4_u16((const uint16_t*)x));
        float32x4x4_t b0 = vcvt4_f32_half(vld4_u16((const uint16_t*)y0));
        float32x4x4_t b1 = vcvt4_f32_half(vld4_u16((const uint16_t*)y1));
        float32x4x4_t b2 = vcvt4_f32_half(vld4_u16((const uint16_t*)y2));
        float32x4x4_t b3 = vcvt4_f32_half(vld4_u16((const uint16_t*)y3));

        b0.val[0] = vsubq_f32(a.val[0], b0.val[0]);
        b0.val[1] = vsubq_f32(a.val[1], b0.val[1]);
        b0.val[2] = vsubq_f32(a.val[2], b0.val[2]);
        b0.val[3] = vsubq_f32(a.val[3], b0.val[3]);

        b1.val[0] = vsubq_f32(a.val[0], b1.val[0]);
        b1.val[1] = vsubq_f32(a.val[1], b1.val[1]);
        b1.val[2] = vsubq_f32(a.val[2], b1.val[2]);
        b1.val[3] = vsubq_f32(a.val[3], b1.val[3]);

        b2.val[0] = vsubq_f32(a.val[0], b2.val[0]);
        b2.val[1] = vsubq_f32(a.val[1], b2.val[1]);
        b2.val[2] = vsubq_f32(a.val[2], b2.val[2]);
        b2.val[3] = vsubq_f32(a.val[3], b2.val[3]);

        b3.val[0] = vsubq_f32(a.val[0], b3.val[0]);
        b3.val[1] = vsubq_f32(a.val[1], b3.val[1]);
        b3.val[2] = vsubq_f32(a.val[2], b3.val[2]);
        b3.val[3] = vsubq_f32(a.val[3], b3.val[3]);

        res.val[0] = vmlaq_f32(res.val[0], b0.val[0], b0.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], b1.val[0], b1.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], b2.val[0], b2.val[0]);
        res.val[3] = vmlaq_f32(res.val[3], b3.val[0], b3.val[0]);

        res.val[0] = vmlaq_f32(res.val[0], b0.val[1], b0.val[1]);
        res.val[1] = vmlaq_f32(res.val[1], b1.val[1], b1.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], b2.val[1], b2.val[1]);
        res.val[3] = vmlaq_f32(res.val[3], b3.val[1], b3.val[1]);

        res.val[0] = vmlaq_f32(res.val[0], b0.val[2], b0.val[2]);
        res.val[1] = vmlaq_f32(res.val[1], b1.val[2], b1.val[2]);
        res.val[2] = vmlaq_f32(res.val[2], b2.val[2], b2.val[2]);
        res.val[3] = vmlaq_f32(res.val[3], b3.val[2], b3.val[2]);

        res.val[0] = vmlaq_f32(res.val[0], b0.val[3], b0.val[3]);
        res.val[1] = vmlaq_f32(res.val[1], b1.val[3], b1.val[3]);
        res.val[2] = vmlaq_f32(res.val[2], b2.val[3], b2.val[3]);
        res.val[3] = vmlaq_f32(res.val[3], b3.val[3], b3.val[3]);

        cur_d -= 16;
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
    }
    if (cur_d >= 8) {
        float32x4x2_t a = vcvt2_f32_half(vld2_u16((const uint16_t*)x));
        float32x4x2_t b0 = vcvt2_f32_half(vld2_u16((const uint16_t*)y0));
        float32x4x2_t b1 = vcvt2_f32_half(vld2_u16((const uint16_t*)y1));
        float32x4x2_t b2 = vcvt2_f32_half(vld2_u16((const uint16_t*)y2));
        float32x4x2_t b3 = vcvt2_f32_half(vld2_u16((const uint16_t*)y3));

        b0.val[0] = vsubq_f32(a.val[0], b0.val[0]);
        b0.val[1] = vsubq_f32(a.val[1], b0.val[1]);

        b1.val[0] = vsubq_f32(a.val[0], b1.val[0]);
        b1.val[1] = vsubq_f32(a.val[1], b1.val[1]);

        b2.val[0] = vsubq_f32(a.val[0], b2.val[0]);
        b2.val[1] = vsubq_f32(a.val[1], b2.val[1]);

        b3.val[0] = vsubq_f32(a.val[0], b3.val[0]);
        b3.val[1] = vsubq_f32(a.val[1], b3.val[1]);

        res.val[0] = vmlaq_f32(res.val[0], b0.val[0], b0.val[0]);
        res.val[1] = vmlaq_f32(res.val[1], b1.val[0], b1.val[0]);
        res.val[2] = vmlaq_f32(res.val[2], b2.val[0], b2.val[0]);
        res.val[3] = vmlaq_f32(res.val[3], b3.val[0], b3.val[0]);

        res.val[0] = vmlaq_f32(res.val[0], b0.val[1], b0.val[1]);
        res.val[1] = vmlaq_f32(res.val[1], b1.val[1], b1.val[1]);
        res.val[2] = vmlaq_f32(res.val[2], b2.val[1], b2.val[1]);
        res.val[3] = vmlaq_f32(res.val[3], b3.val[1], b3.val[1]);

        cur_d -= 8;
        x += 8;
        y0 += 8;
        y1 += 8;
        y2 += 8;
        y3 += 8;
    }
    if (cur_d >= 4) {
        float32x4_t a = vcvt_f32_half(vld1_u16((const uint16_t*)x));
        float32x4_t b0 = vcvt_f32_half(vld1_u16((const uint16_t*)y0));
        float32x4_t b1 = vcvt_f32_half(vld1_u16((const uint16_t*)y1));
        float32x4_t b2 = vcvt_f32_half(vld1_u16((const uint16_t*)y2));
        float32x4_t b3 = vcvt_f32_half(vld1_u16((const uint16_t*)y3));
        b0 = vsubq_f32(a, b0);
        b1 = vsubq_f32(a, b1);
        b2 = vsubq_f32(a, b2);
        b3 = vsubq_f32(a, b3);

        res.val[0] = vmlaq_f32(res.val[0], b0, b0);
        res.val[1] = vmlaq_f32(res.val[1], b1, b1);
        res.val[2] = vmlaq_f32(res.val[2], b2, b2);
        res.val[3] = vmlaq_f32(res.val[3], b3, b3);
        cur_d -= 4;
        x += 4;
        y0 += 4;
        y1 += 4;
        y2 += 4;
        y3 += 4;
    }
    if (cur_d >= 0) {
        uint16x4_t res_x = vdup_n_u16(0);
        uint16x4_t res_y0 = vdup_n_u16(0);
        uint16x4_t res_y1 = vdup_n_u16(0);
        uint16x4_t res_y2 = vdup_n_u16(0);
        uint16x4_t res_y3 = vdup_n_u16(0);
        switch (cur_d) {
            case 3:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 2);
                res_y0 = vld1_lane_u16((const uint16_t*)y0, res_y0, 2);
                res_y1 = vld1_lane_u16((const uint16_t*)y1, res_y1, 2);
                res_y2 = vld1_lane_u16((const uint16_t*)y2, res_y2, 2);
                res_y3 = vld1_lane_u16((const uint16_t*)y3, res_y3, 2);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
            case 2:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 1);
                res_y0 = vld1_lane_u16((const uint16_t*)y0, res_y0, 1);
                res_y1 = vld1_lane_u16((const uint16_t*)y1, res_y1, 1);
                res_y2 = vld1_lane_u16((const uint16_t*)y2, res_y2, 1);
                res_y3 = vld1_lane_u16((const uint16_t*)y3, res_y3, 1);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
            case 1:
                res_x = vld1_lane_u16((const uint16_t*)x, res_x, 0);
                res_y0 = vld1_lane_u16((const uint16_t*)y0, res_y0, 0);
                res_y1 = vld1_lane_u16((const uint16_t*)y1, res_y1, 0);
                res_y2 = vld1_lane_u16((const uint16_t*)y2, res_y2, 0);
                res_y3 = vld1_lane_u16((const uint16_t*)y3, res_y3, 0);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
        }
        float32x4_t diff0 = vsubq_f32(vcvt_f32_half(res_x), vcvt_f32_half(res_y0));
        float32x4_t diff1 = vsubq_f32(vcvt_f32_half(res_x), vcvt_f32_half(res_y1));
        float32x4_t diff2 = vsubq_f32(vcvt_f32_half(res_x), vcvt_f32_half(res_y2));
        float32x4_t diff3 = vsubq_f32(vcvt_f32_half(res_x), vcvt_f32_half(res_y3));
        res.val[0] = vmlaq_f32(res.val[0], diff0, diff0);
        res.val[1] = vmlaq_f32(res.val[1], diff1, diff1);
        res.val[2] = vmlaq_f32(res.val[2], diff2, diff2);
        res.val[3] = vmlaq_f32(res.val[3], diff3, diff3);
    }
    dis0 = vaddvq_f32(res.val[0]);
    dis1 = vaddvq_f32(res.val[1]);
    dis2 = vaddvq_f32(res.val[2]);
    dis3 = vaddvq_f32(res.val[3]);
}

///////////////////////////////////////////////////////////////////////////////
// int8

float
int8_vec_inner_product_neon(const int8_t* x, const int8_t* y, size_t d) {
    // initialize the accumulator
    int32x4_t sum_ = vdupq_n_s32(0);

    // main loop: process 16 int8_t elements each time
    while (d >= 16) {
        // load 16 int8_t element into NEON register
        int8x16_t a = vld1q_s8(x);
        int8x16_t b = vld1q_s8(y);

        // extend int8_t to int16_t
        int16x8_t a_low = vmovl_s8(vget_low_s8(a));
        int16x8_t a_high = vmovl_s8(vget_high_s8(a));
        int16x8_t b_low = vmovl_s8(vget_low_s8(b));
        int16x8_t b_high = vmovl_s8(vget_high_s8(b));

        // extend int16_t to int32_t
        int32x4_t a_low_low = vmovl_s16(vget_low_s16(a_low));
        int32x4_t a_low_high = vmovl_s16(vget_high_s16(a_low));
        int32x4_t a_high_low = vmovl_s16(vget_low_s16(a_high));
        int32x4_t a_high_high = vmovl_s16(vget_high_s16(a_high));

        int32x4_t b_low_low = vmovl_s16(vget_low_s16(b_low));
        int32x4_t b_low_high = vmovl_s16(vget_high_s16(b_low));
        int32x4_t b_high_low = vmovl_s16(vget_low_s16(b_high));
        int32x4_t b_high_high = vmovl_s16(vget_high_s16(b_high));

        // accumulate partial sum
        sum_ = vaddq_s32(sum_, vmulq_s32(a_low_low, b_low_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(a_low_high, b_low_high));
        sum_ = vaddq_s32(sum_, vmulq_s32(a_high_low, b_high_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(a_high_high, b_high_high));

        // update the pointer and the count of remaining elements
        x += 16;
        y += 16;
        d -= 16;
    }

    // process 8 int8_t elements each time
    if (d >= 8) {
        int8x8_t a = vld1_s8(x);
        int8x8_t b = vld1_s8(y);

        int16x8_t a_ext = vmovl_s8(a);
        int16x8_t b_ext = vmovl_s8(b);

        int32x4_t a_low = vmovl_s16(vget_low_s16(a_ext));
        int32x4_t a_high = vmovl_s16(vget_high_s16(a_ext));
        int32x4_t b_low = vmovl_s16(vget_low_s16(b_ext));
        int32x4_t b_high = vmovl_s16(vget_high_s16(b_ext));

        sum_ = vaddq_s32(sum_, vmulq_s32(a_low, b_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(a_high, b_high));

        x += 8;
        y += 8;
        d -= 8;
    }

    // process 4 int8_t elements each time
    if (d >= 4) {
        int8x8_t a = vld1_s8(x);
        int8x8_t b = vld1_s8(y);

        int16x8_t a_ext = vmovl_s8(a);
        int16x8_t b_ext = vmovl_s8(b);

        int32x4_t a_low = vmovl_s16(vget_low_s16(a_ext));
        int32x4_t b_low = vmovl_s16(vget_low_s16(b_ext));

        sum_ = vaddq_s32(sum_, vmulq_s32(a_low, b_low));

        x += 4;
        y += 4;
        d -= 4;
    }

    // process left elements
    int32_t rem_sum = 0;
    for (size_t i = 0; i < d; ++i) {
        rem_sum += static_cast<int32_t>(x[i]) * static_cast<int32_t>(y[i]);
    }

    // accumulate the total sum
    return static_cast<float>(vaddvq_s32(sum_) + rem_sum);
}

float
int8_vec_L2sqr_neon(const int8_t* x, const int8_t* y, size_t d) {
    // initialize the accumulator
    int32x4_t sum_ = vdupq_n_s32(0);

    // main loop: process 16 int8_t elements each time
    while (d >= 16) {
        // load 16 int8_t element into NEON register
        int8x16_t a = vld1q_s8(x);
        int8x16_t b = vld1q_s8(y);

        // extend int8_t to int16_t
        int16x8_t a_low = vmovl_s8(vget_low_s8(a));
        int16x8_t a_high = vmovl_s8(vget_high_s8(a));
        int16x8_t b_low = vmovl_s8(vget_low_s8(b));
        int16x8_t b_high = vmovl_s8(vget_high_s8(b));

        // calculate the diff and extend it to int32_t
        int32x4_t diff_low_low = vsubl_s16(vget_low_s16(a_low), vget_low_s16(b_low));
        int32x4_t diff_low_high = vsubl_s16(vget_high_s16(a_low), vget_high_s16(b_low));
        int32x4_t diff_high_low = vsubl_s16(vget_low_s16(a_high), vget_low_s16(b_high));
        int32x4_t diff_high_high = vsubl_s16(vget_high_s16(a_high), vget_high_s16(b_high));

        // accumulate partial sum
        sum_ = vaddq_s32(sum_, vmulq_s32(diff_low_low, diff_low_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(diff_low_high, diff_low_high));
        sum_ = vaddq_s32(sum_, vmulq_s32(diff_high_low, diff_high_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(diff_high_high, diff_high_high));

        // update the pointer and the count of remaining elements
        x += 16;
        y += 16;
        d -= 16;
    }

    // process 8 int8_t elements each time
    if (d >= 8) {
        int8x8_t a = vld1_s8(x);
        int8x8_t b = vld1_s8(y);

        int16x8_t a_ext = vmovl_s8(a);
        int16x8_t b_ext = vmovl_s8(b);

        int32x4_t diff_low = vsubl_s16(vget_low_s16(a_ext), vget_low_s16(b_ext));
        int32x4_t diff_high = vsubl_s16(vget_high_s16(a_ext), vget_high_s16(b_ext));

        sum_ = vaddq_s32(sum_, vmulq_s32(diff_low, diff_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(diff_high, diff_high));

        x += 8;
        y += 8;
        d -= 8;
    }

    // process 4 int8_t elements each time
    if (d >= 4) {
        int8x8_t a = vld1_s8(x);
        int8x8_t b = vld1_s8(y);

        int16x8_t a_ext = vmovl_s8(a);
        int16x8_t b_ext = vmovl_s8(b);

        int32x4_t diff_low = vsubl_s16(vget_low_s16(a_ext), vget_low_s16(b_ext));

        sum_ = vaddq_s32(sum_, vmulq_s32(diff_low, diff_low));

        x += 4;
        y += 4;
        d -= 4;
    }

    // process left elements
    int32_t rem_sum = 0;
    for (size_t i = 0; i < d; ++i) {
        int32_t diff = static_cast<int32_t>(x[i]) - static_cast<int32_t>(y[i]);
        rem_sum += diff * diff;
    }

    // accumulate the total sum
    return static_cast<float>(vaddvq_s32(sum_) + rem_sum);
}

float
int8_vec_norm_L2sqr_neon(const int8_t* x, size_t d) {
    // initialize the accumulator
    int32x4_t sum_ = vdupq_n_s32(0);

    // main loop: process 16 int8_t elements each time
    while (d >= 16) {
        // load 16 int8_t element into NEON register
        int8x16_t a = vld1q_s8(x);

        // extend int8_t to int16_t
        int16x8_t a_low = vmovl_s8(vget_low_s8(a));
        int16x8_t a_high = vmovl_s8(vget_high_s8(a));

        // extend int16_t to int32_t
        int32x4_t a_low_low = vmovl_s16(vget_low_s16(a_low));
        int32x4_t a_low_high = vmovl_s16(vget_high_s16(a_low));
        int32x4_t a_high_low = vmovl_s16(vget_low_s16(a_high));
        int32x4_t a_high_high = vmovl_s16(vget_high_s16(a_high));

        // accumulate partial sum
        sum_ = vaddq_s32(sum_, vmulq_s32(a_low_low, a_low_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(a_low_high, a_low_high));
        sum_ = vaddq_s32(sum_, vmulq_s32(a_high_low, a_high_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(a_high_high, a_high_high));

        // update the pointer and the count of remaining elements
        x += 16;
        d -= 16;
    }

    // process 8 int8_t elements each time
    if (d >= 8) {
        int8x8_t a = vld1_s8(x);

        int16x8_t a_ext = vmovl_s8(a);

        int32x4_t a_low = vmovl_s16(vget_low_s16(a_ext));
        int32x4_t a_high = vmovl_s16(vget_high_s16(a_ext));

        sum_ = vaddq_s32(sum_, vmulq_s32(a_low, a_low));
        sum_ = vaddq_s32(sum_, vmulq_s32(a_high, a_high));

        x += 8;
        d -= 8;
    }

    // process 4 int8_t elements each time
    if (d >= 4) {
        int8x8_t a = vld1_s8(x);

        int16x8_t a_ext = vmovl_s8(a);

        int32x4_t a_low = vmovl_s16(vget_low_s16(a_ext));

        sum_ = vaddq_s32(sum_, vmulq_s32(a_low, a_low));

        x += 4;
        d -= 4;
    }

    // process left elements
    int32_t remaining_sum = 0;
    for (size_t i = 0; i < d; ++i) {
        int32_t val = static_cast<int32_t>(x[i]);
        remaining_sum += val * val;
    }

    // accumulate the total sum
    return static_cast<float>(vaddvq_s32(sum_) + remaining_sum);
}

void
int8_vec_inner_product_batch_4_neon(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                    const int8_t* y3, const size_t dim, float& dis0, float& dis1, float& dis2,
                                    float& dis3) {
    // initialize the accumulator
    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);

    size_t d = dim;

    // main loop: process 16 int8_t elements each time
    while (d >= 16) {
        // load 16 int8_t element into NEON register
        int8x16_t a = vld1q_s8(x + dim - d);
        int8x16_t b0 = vld1q_s8(y0 + dim - d);
        int8x16_t b1 = vld1q_s8(y1 + dim - d);
        int8x16_t b2 = vld1q_s8(y2 + dim - d);
        int8x16_t b3 = vld1q_s8(y3 + dim - d);

        // extend int8_t to int16_t
        int16x8_t a_low = vmovl_s8(vget_low_s8(a));
        int16x8_t a_high = vmovl_s8(vget_high_s8(a));
        int16x8_t b0_low = vmovl_s8(vget_low_s8(b0));
        int16x8_t b0_high = vmovl_s8(vget_high_s8(b0));
        int16x8_t b1_low = vmovl_s8(vget_low_s8(b1));
        int16x8_t b1_high = vmovl_s8(vget_high_s8(b1));
        int16x8_t b2_low = vmovl_s8(vget_low_s8(b2));
        int16x8_t b2_high = vmovl_s8(vget_high_s8(b2));
        int16x8_t b3_low = vmovl_s8(vget_low_s8(b3));
        int16x8_t b3_high = vmovl_s8(vget_high_s8(b3));

        // extend int16_t to int32_t
        int32x4_t a_low_low = vmovl_s16(vget_low_s16(a_low));
        int32x4_t a_low_high = vmovl_s16(vget_high_s16(a_low));
        int32x4_t a_high_low = vmovl_s16(vget_low_s16(a_high));
        int32x4_t a_high_high = vmovl_s16(vget_high_s16(a_high));

        int32x4_t b0_low_low = vmovl_s16(vget_low_s16(b0_low));
        int32x4_t b0_low_high = vmovl_s16(vget_high_s16(b0_low));
        int32x4_t b0_high_low = vmovl_s16(vget_low_s16(b0_high));
        int32x4_t b0_high_high = vmovl_s16(vget_high_s16(b0_high));

        int32x4_t b1_low_low = vmovl_s16(vget_low_s16(b1_low));
        int32x4_t b1_low_high = vmovl_s16(vget_high_s16(b1_low));
        int32x4_t b1_high_low = vmovl_s16(vget_low_s16(b1_high));
        int32x4_t b1_high_high = vmovl_s16(vget_high_s16(b1_high));

        int32x4_t b2_low_low = vmovl_s16(vget_low_s16(b2_low));
        int32x4_t b2_low_high = vmovl_s16(vget_high_s16(b2_low));
        int32x4_t b2_high_low = vmovl_s16(vget_low_s16(b2_high));
        int32x4_t b2_high_high = vmovl_s16(vget_high_s16(b2_high));

        int32x4_t b3_low_low = vmovl_s16(vget_low_s16(b3_low));
        int32x4_t b3_low_high = vmovl_s16(vget_high_s16(b3_low));
        int32x4_t b3_high_low = vmovl_s16(vget_low_s16(b3_high));
        int32x4_t b3_high_high = vmovl_s16(vget_high_s16(b3_high));

        // accumulate partial sum
        sum0 = vaddq_s32(sum0, vmulq_s32(a_low_low, b0_low_low));
        sum0 = vaddq_s32(sum0, vmulq_s32(a_low_high, b0_low_high));
        sum0 = vaddq_s32(sum0, vmulq_s32(a_high_low, b0_high_low));
        sum0 = vaddq_s32(sum0, vmulq_s32(a_high_high, b0_high_high));

        sum1 = vaddq_s32(sum1, vmulq_s32(a_low_low, b1_low_low));
        sum1 = vaddq_s32(sum1, vmulq_s32(a_low_high, b1_low_high));
        sum1 = vaddq_s32(sum1, vmulq_s32(a_high_low, b1_high_low));
        sum1 = vaddq_s32(sum1, vmulq_s32(a_high_high, b1_high_high));

        sum2 = vaddq_s32(sum2, vmulq_s32(a_low_low, b2_low_low));
        sum2 = vaddq_s32(sum2, vmulq_s32(a_low_high, b2_low_high));
        sum2 = vaddq_s32(sum2, vmulq_s32(a_high_low, b2_high_low));
        sum2 = vaddq_s32(sum2, vmulq_s32(a_high_high, b2_high_high));

        sum3 = vaddq_s32(sum3, vmulq_s32(a_low_low, b3_low_low));
        sum3 = vaddq_s32(sum3, vmulq_s32(a_low_high, b3_low_high));
        sum3 = vaddq_s32(sum3, vmulq_s32(a_high_low, b3_high_low));
        sum3 = vaddq_s32(sum3, vmulq_s32(a_high_high, b3_high_high));

        d -= 16;
    }

    // process 8 int8_t elements each time
    if (d >= 8) {
        int8x8_t a = vld1_s8(x + dim - d);
        int8x8_t b0 = vld1_s8(y0 + dim - d);
        int8x8_t b1 = vld1_s8(y1 + dim - d);
        int8x8_t b2 = vld1_s8(y2 + dim - d);
        int8x8_t b3 = vld1_s8(y3 + dim - d);

        // extend int8_t to int16_t
        int16x8_t a_ext = vmovl_s8(a);
        int16x8_t b0_ext = vmovl_s8(b0);
        int16x8_t b1_ext = vmovl_s8(b1);
        int16x8_t b2_ext = vmovl_s8(b2);
        int16x8_t b3_ext = vmovl_s8(b3);

        // extend int16_t to int32_t
        int32x4_t a_low = vmovl_s16(vget_low_s16(a_ext));
        int32x4_t a_high = vmovl_s16(vget_high_s16(a_ext));

        int32x4_t b0_low = vmovl_s16(vget_low_s16(b0_ext));
        int32x4_t b0_high = vmovl_s16(vget_high_s16(b0_ext));

        int32x4_t b1_low = vmovl_s16(vget_low_s16(b1_ext));
        int32x4_t b1_high = vmovl_s16(vget_high_s16(b1_ext));

        int32x4_t b2_low = vmovl_s16(vget_low_s16(b2_ext));
        int32x4_t b2_high = vmovl_s16(vget_high_s16(b2_ext));

        int32x4_t b3_low = vmovl_s16(vget_low_s16(b3_ext));
        int32x4_t b3_high = vmovl_s16(vget_high_s16(b3_ext));

        // update partial sum
        sum0 = vaddq_s32(sum0, vmulq_s32(a_low, b0_low));
        sum0 = vaddq_s32(sum0, vmulq_s32(a_high, b0_high));

        sum1 = vaddq_s32(sum1, vmulq_s32(a_low, b1_low));
        sum1 = vaddq_s32(sum1, vmulq_s32(a_high, b1_high));

        sum2 = vaddq_s32(sum2, vmulq_s32(a_low, b2_low));
        sum2 = vaddq_s32(sum2, vmulq_s32(a_high, b2_high));

        sum3 = vaddq_s32(sum3, vmulq_s32(a_low, b3_low));
        sum3 = vaddq_s32(sum3, vmulq_s32(a_high, b3_high));

        d -= 8;
    }

    // process 4 int8_t elements each time
    if (d >= 4) {
        int8x8_t a = vld1_s8(x + dim - d);
        int8x8_t b0 = vld1_s8(y0 + dim - d);
        int8x8_t b1 = vld1_s8(y1 + dim - d);
        int8x8_t b2 = vld1_s8(y2 + dim - d);
        int8x8_t b3 = vld1_s8(y3 + dim - d);

        // extend int8_t to int16_t
        int16x8_t a_ext = vmovl_s8(a);
        int16x8_t b0_ext = vmovl_s8(b0);
        int16x8_t b1_ext = vmovl_s8(b1);
        int16x8_t b2_ext = vmovl_s8(b2);
        int16x8_t b3_ext = vmovl_s8(b3);

        // extend int16_t to int32_t
        int32x4_t a_low = vmovl_s16(vget_low_s16(a_ext));

        int32x4_t b0_low = vmovl_s16(vget_low_s16(b0_ext));
        int32x4_t b1_low = vmovl_s16(vget_low_s16(b1_ext));
        int32x4_t b2_low = vmovl_s16(vget_low_s16(b2_ext));
        int32x4_t b3_low = vmovl_s16(vget_low_s16(b3_ext));

        // update partial sum
        sum0 = vaddq_s32(sum0, vmulq_s32(a_low, b0_low));
        sum1 = vaddq_s32(sum1, vmulq_s32(a_low, b1_low));
        sum2 = vaddq_s32(sum2, vmulq_s32(a_low, b2_low));
        sum3 = vaddq_s32(sum3, vmulq_s32(a_low, b3_low));

        d -= 4;
    }

    // process left elements
    int32_t rem_sum0 = 0;
    int32_t rem_sum1 = 0;
    int32_t rem_sum2 = 0;
    int32_t rem_sum3 = 0;
    for (size_t i = 0; i < d; ++i) {
        int32_t val_x = static_cast<int32_t>(x[dim - d + i]);
        rem_sum0 += val_x * static_cast<int32_t>(y0[dim - d + i]);
        rem_sum1 += val_x * static_cast<int32_t>(y1[dim - d + i]);
        rem_sum2 += val_x * static_cast<int32_t>(y2[dim - d + i]);
        rem_sum3 += val_x * static_cast<int32_t>(y3[dim - d + i]);
    }

    // accumulate the total sum
    dis0 = static_cast<float>(vaddvq_s32(sum0) + rem_sum0);
    dis1 = static_cast<float>(vaddvq_s32(sum1) + rem_sum1);
    dis2 = static_cast<float>(vaddvq_s32(sum2) + rem_sum2);
    dis3 = static_cast<float>(vaddvq_s32(sum3) + rem_sum3);
}

void
int8_vec_L2sqr_batch_4_neon(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                            const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3) {
    // initialize the accumulator
    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);

    size_t d = dim;

    // main loop: process 16 int8_t elements each time
    while (d >= 16) {
        // load 16 int8_t element into NEON register
        int8x16_t a = vld1q_s8(x + dim - d);
        int8x16_t b0 = vld1q_s8(y0 + dim - d);
        int8x16_t b1 = vld1q_s8(y1 + dim - d);
        int8x16_t b2 = vld1q_s8(y2 + dim - d);
        int8x16_t b3 = vld1q_s8(y3 + dim - d);

        // extend int8_t to int16_t
        int16x8_t a_low = vmovl_s8(vget_low_s8(a));
        int16x8_t a_high = vmovl_s8(vget_high_s8(a));
        int16x8_t b0_low = vmovl_s8(vget_low_s8(b0));
        int16x8_t b0_high = vmovl_s8(vget_high_s8(b0));
        int16x8_t b1_low = vmovl_s8(vget_low_s8(b1));
        int16x8_t b1_high = vmovl_s8(vget_high_s8(b1));
        int16x8_t b2_low = vmovl_s8(vget_low_s8(b2));
        int16x8_t b2_high = vmovl_s8(vget_high_s8(b2));
        int16x8_t b3_low = vmovl_s8(vget_low_s8(b3));
        int16x8_t b3_high = vmovl_s8(vget_high_s8(b3));

        // extend int16_t to int32_t
        int32x4_t a_low_low = vmovl_s16(vget_low_s16(a_low));
        int32x4_t a_low_high = vmovl_s16(vget_high_s16(a_low));
        int32x4_t a_high_low = vmovl_s16(vget_low_s16(a_high));
        int32x4_t a_high_high = vmovl_s16(vget_high_s16(a_high));

        int32x4_t b0_low_low = vmovl_s16(vget_low_s16(b0_low));
        int32x4_t b0_low_high = vmovl_s16(vget_high_s16(b0_low));
        int32x4_t b0_high_low = vmovl_s16(vget_low_s16(b0_high));
        int32x4_t b0_high_high = vmovl_s16(vget_high_s16(b0_high));

        int32x4_t b1_low_low = vmovl_s16(vget_low_s16(b1_low));
        int32x4_t b1_low_high = vmovl_s16(vget_high_s16(b1_low));
        int32x4_t b1_high_low = vmovl_s16(vget_low_s16(b1_high));
        int32x4_t b1_high_high = vmovl_s16(vget_high_s16(b1_high));

        int32x4_t b2_low_low = vmovl_s16(vget_low_s16(b2_low));
        int32x4_t b2_low_high = vmovl_s16(vget_high_s16(b2_low));
        int32x4_t b2_high_low = vmovl_s16(vget_low_s16(b2_high));
        int32x4_t b2_high_high = vmovl_s16(vget_high_s16(b2_high));

        int32x4_t b3_low_low = vmovl_s16(vget_low_s16(b3_low));
        int32x4_t b3_low_high = vmovl_s16(vget_high_s16(b3_low));
        int32x4_t b3_high_low = vmovl_s16(vget_low_s16(b3_high));
        int32x4_t b3_high_high = vmovl_s16(vget_high_s16(b3_high));

        // calculate the diff
        int32x4_t diff0_low_low = vsubq_s32(a_low_low, b0_low_low);
        int32x4_t diff0_low_high = vsubq_s32(a_low_high, b0_low_high);
        int32x4_t diff0_high_low = vsubq_s32(a_high_low, b0_high_low);
        int32x4_t diff0_high_high = vsubq_s32(a_high_high, b0_high_high);

        int32x4_t diff1_low_low = vsubq_s32(a_low_low, b1_low_low);
        int32x4_t diff1_low_high = vsubq_s32(a_low_high, b1_low_high);
        int32x4_t diff1_high_low = vsubq_s32(a_high_low, b1_high_low);
        int32x4_t diff1_high_high = vsubq_s32(a_high_high, b1_high_high);

        int32x4_t diff2_low_low = vsubq_s32(a_low_low, b2_low_low);
        int32x4_t diff2_low_high = vsubq_s32(a_low_high, b2_low_high);
        int32x4_t diff2_high_low = vsubq_s32(a_high_low, b2_high_low);
        int32x4_t diff2_high_high = vsubq_s32(a_high_high, b2_high_high);

        int32x4_t diff3_low_low = vsubq_s32(a_low_low, b3_low_low);
        int32x4_t diff3_low_high = vsubq_s32(a_low_high, b3_low_high);
        int32x4_t diff3_high_low = vsubq_s32(a_high_low, b3_high_low);
        int32x4_t diff3_high_high = vsubq_s32(a_high_high, b3_high_high);

        // accumulate partial sum
        sum0 = vaddq_s32(sum0, vmulq_s32(diff0_low_low, diff0_low_low));
        sum0 = vaddq_s32(sum0, vmulq_s32(diff0_low_high, diff0_low_high));
        sum0 = vaddq_s32(sum0, vmulq_s32(diff0_high_low, diff0_high_low));
        sum0 = vaddq_s32(sum0, vmulq_s32(diff0_high_high, diff0_high_high));

        sum1 = vaddq_s32(sum1, vmulq_s32(diff1_low_low, diff1_low_low));
        sum1 = vaddq_s32(sum1, vmulq_s32(diff1_low_high, diff1_low_high));
        sum1 = vaddq_s32(sum1, vmulq_s32(diff1_high_low, diff1_high_low));
        sum1 = vaddq_s32(sum1, vmulq_s32(diff1_high_high, diff1_high_high));

        sum2 = vaddq_s32(sum2, vmulq_s32(diff2_low_low, diff2_low_low));
        sum2 = vaddq_s32(sum2, vmulq_s32(diff2_low_high, diff2_low_high));
        sum2 = vaddq_s32(sum2, vmulq_s32(diff2_high_low, diff2_high_low));
        sum2 = vaddq_s32(sum2, vmulq_s32(diff2_high_high, diff2_high_high));

        sum3 = vaddq_s32(sum3, vmulq_s32(diff3_low_low, diff3_low_low));
        sum3 = vaddq_s32(sum3, vmulq_s32(diff3_low_high, diff3_low_high));
        sum3 = vaddq_s32(sum3, vmulq_s32(diff3_high_low, diff3_high_low));
        sum3 = vaddq_s32(sum3, vmulq_s32(diff3_high_high, diff3_high_high));

        d -= 16;
    }

    // process 8 int8_t elements each time
    if (d >= 8) {
        int8x8_t a = vld1_s8(x + dim - d);
        int8x8_t b0 = vld1_s8(y0 + dim - d);
        int8x8_t b1 = vld1_s8(y1 + dim - d);
        int8x8_t b2 = vld1_s8(y2 + dim - d);
        int8x8_t b3 = vld1_s8(y3 + dim - d);

        // extend int8_t to int16_t
        int16x8_t a_ext = vmovl_s8(a);
        int16x8_t b0_ext = vmovl_s8(b0);
        int16x8_t b1_ext = vmovl_s8(b1);
        int16x8_t b2_ext = vmovl_s8(b2);
        int16x8_t b3_ext = vmovl_s8(b3);

        // extend int16_t to int32_t
        int32x4_t a_low = vmovl_s16(vget_low_s16(a_ext));
        int32x4_t a_high = vmovl_s16(vget_high_s16(a_ext));

        int32x4_t b0_low = vmovl_s16(vget_low_s16(b0_ext));
        int32x4_t b0_high = vmovl_s16(vget_high_s16(b0_ext));

        int32x4_t b1_low = vmovl_s16(vget_low_s16(b1_ext));
        int32x4_t b1_high = vmovl_s16(vget_high_s16(b1_ext));

        int32x4_t b2_low = vmovl_s16(vget_low_s16(b2_ext));
        int32x4_t b2_high = vmovl_s16(vget_high_s16(b2_ext));

        int32x4_t b3_low = vmovl_s16(vget_low_s16(b3_ext));
        int32x4_t b3_high = vmovl_s16(vget_high_s16(b3_ext));

        // calculate the diff
        int32x4_t diff0_low = vsubq_s32(a_low, b0_low);
        int32x4_t diff0_high = vsubq_s32(a_high, b0_high);

        int32x4_t diff1_low = vsubq_s32(a_low, b1_low);
        int32x4_t diff1_high = vsubq_s32(a_high, b1_high);

        int32x4_t diff2_low = vsubq_s32(a_low, b2_low);
        int32x4_t diff2_high = vsubq_s32(a_high, b2_high);

        int32x4_t diff3_low = vsubq_s32(a_low, b3_low);
        int32x4_t diff3_high = vsubq_s32(a_high, b3_high);

        // accumulate partial sum
        sum0 = vaddq_s32(sum0, vmulq_s32(diff0_low, diff0_low));
        sum0 = vaddq_s32(sum0, vmulq_s32(diff0_high, diff0_high));

        sum1 = vaddq_s32(sum1, vmulq_s32(diff1_low, diff1_low));
        sum1 = vaddq_s32(sum1, vmulq_s32(diff1_high, diff1_high));

        sum2 = vaddq_s32(sum2, vmulq_s32(diff2_low, diff2_low));
        sum2 = vaddq_s32(sum2, vmulq_s32(diff2_high, diff2_high));

        sum3 = vaddq_s32(sum3, vmulq_s32(diff3_low, diff3_low));
        sum3 = vaddq_s32(sum3, vmulq_s32(diff3_high, diff3_high));

        d -= 8;
    }

    // process 4 int8_t elements each time
    if (d >= 4) {
        int8x8_t a = vld1_s8(x + dim - d);
        int8x8_t b0 = vld1_s8(y0 + dim - d);
        int8x8_t b1 = vld1_s8(y1 + dim - d);
        int8x8_t b2 = vld1_s8(y2 + dim - d);
        int8x8_t b3 = vld1_s8(y3 + dim - d);

        // extend int8_t to int16_t
        int16x8_t a_ext = vmovl_s8(a);
        int16x8_t b0_ext = vmovl_s8(b0);
        int16x8_t b1_ext = vmovl_s8(b1);
        int16x8_t b2_ext = vmovl_s8(b2);
        int16x8_t b3_ext = vmovl_s8(b3);

        // extend int16_t to int32_t
        int32x4_t a_low = vmovl_s16(vget_low_s16(a_ext));

        int32x4_t b0_low = vmovl_s16(vget_low_s16(b0_ext));
        int32x4_t b1_low = vmovl_s16(vget_low_s16(b1_ext));
        int32x4_t b2_low = vmovl_s16(vget_low_s16(b2_ext));
        int32x4_t b3_low = vmovl_s16(vget_low_s16(b3_ext));

        // calculate the diff
        int32x4_t diff0_low = vsubq_s32(a_low, b0_low);
        int32x4_t diff1_low = vsubq_s32(a_low, b1_low);
        int32x4_t diff2_low = vsubq_s32(a_low, b2_low);
        int32x4_t diff3_low = vsubq_s32(a_low, b3_low);

        // accumulate partial sum
        sum0 = vaddq_s32(sum0, vmulq_s32(diff0_low, diff0_low));
        sum1 = vaddq_s32(sum1, vmulq_s32(diff1_low, diff1_low));
        sum2 = vaddq_s32(sum2, vmulq_s32(diff2_low, diff2_low));
        sum3 = vaddq_s32(sum3, vmulq_s32(diff3_low, diff3_low));

        d -= 4;
    }

    // process left elements
    int32_t rem_sum0 = 0;
    int32_t rem_sum1 = 0;
    int32_t rem_sum2 = 0;
    int32_t rem_sum3 = 0;
    for (size_t i = 0; i < d; ++i) {
        int32_t val_x = static_cast<int32_t>(x[dim - d + i]);
        rem_sum0 += (val_x - static_cast<int32_t>(y0[dim - d + i])) * (val_x - static_cast<int32_t>(y0[dim - d + i]));
        rem_sum1 += (val_x - static_cast<int32_t>(y1[dim - d + i])) * (val_x - static_cast<int32_t>(y1[dim - d + i]));
        rem_sum2 += (val_x - static_cast<int32_t>(y2[dim - d + i])) * (val_x - static_cast<int32_t>(y2[dim - d + i]));
        rem_sum3 += (val_x - static_cast<int32_t>(y3[dim - d + i])) * (val_x - static_cast<int32_t>(y3[dim - d + i]));
    }

    // accumulate the total sum
    dis0 = static_cast<float>(vaddvq_s32(sum0) + rem_sum0);
    dis1 = static_cast<float>(vaddvq_s32(sum1) + rem_sum1);
    dis2 = static_cast<float>(vaddvq_s32(sum2) + rem_sum2);
    dis3 = static_cast<float>(vaddvq_s32(sum3) + rem_sum3);
}

}  // namespace knowhere
}  // namespace cppcontrib
}  // namespace faiss
#endif
