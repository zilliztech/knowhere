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

namespace {
inline float32x4x4_t
vcvt4_f32_f16(const float16x4x4_t a) {
    float32x4x4_t c;
    c.val[0] = vcvt_f32_f16(a.val[0]);
    c.val[1] = vcvt_f32_f16(a.val[1]);
    c.val[2] = vcvt_f32_f16(a.val[2]);
    c.val[3] = vcvt_f32_f16(a.val[3]);
    return c;
}

inline float32x4x2_t
vcvt2_f32_f16(const float16x4x2_t a) {
    float32x4x2_t c;
    c.val[0] = vcvt_f32_f16(a.val[0]);
    c.val[1] = vcvt_f32_f16(a.val[1]);
    return c;
}

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
    float32x4_t sum_ = vdupq_n_f32(0.0f);
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

    sum_ = vaddq_f32(sum_, vmulq_f32(res_x, res_y));
    return vaddvq_f32(sum_);
}

float
fvec_L2sqr_neon(const float* x, const float* y, size_t d) {
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

    sum_ = vaddq_f32(sum_, vmulq_f32(vsubq_f32(res_x, res_y), vsubq_f32(res_x, res_y)));
    return vaddvq_f32(sum_);
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
fvec_inner_product_batch_4_neon(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3) {
    float32x4x4_t sum_ = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    auto d = dim;

    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        {
            float32x4x4_t b = vld1q_f32_x4(y0 + dim - d);
            float32x4x4_t c;
            c.val[0] = vmulq_f32(a.val[0], b.val[0]);
            c.val[1] = vmulq_f32(a.val[1], b.val[1]);
            c.val[2] = vmulq_f32(a.val[2], b.val[2]);
            c.val[3] = vmulq_f32(a.val[3], b.val[3]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[0] = vaddq_f32(sum_.val[0], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y1 + dim - d);
            float32x4x4_t c;
            c.val[0] = vmulq_f32(a.val[0], b.val[0]);
            c.val[1] = vmulq_f32(a.val[1], b.val[1]);
            c.val[2] = vmulq_f32(a.val[2], b.val[2]);
            c.val[3] = vmulq_f32(a.val[3], b.val[3]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[1] = vaddq_f32(sum_.val[1], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y2 + dim - d);
            float32x4x4_t c;
            c.val[0] = vmulq_f32(a.val[0], b.val[0]);
            c.val[1] = vmulq_f32(a.val[1], b.val[1]);
            c.val[2] = vmulq_f32(a.val[2], b.val[2]);
            c.val[3] = vmulq_f32(a.val[3], b.val[3]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[2] = vaddq_f32(sum_.val[2], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y3 + dim - d);
            float32x4x4_t c;
            c.val[0] = vmulq_f32(a.val[0], b.val[0]);
            c.val[1] = vmulq_f32(a.val[1], b.val[1]);
            c.val[2] = vmulq_f32(a.val[2], b.val[2]);
            c.val[3] = vmulq_f32(a.val[3], b.val[3]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[3] = vaddq_f32(sum_.val[3], c.val[0]);
        }

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);

        {
            float32x4x2_t b = vld1q_f32_x2(y0 + dim - d);
            float32x4x2_t c;
            c.val[0] = vmulq_f32(a.val[0], b.val[0]);
            c.val[1] = vmulq_f32(a.val[1], b.val[1]);
            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[0] = vaddq_f32(sum_.val[0], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y1 + dim - d);
            float32x4x2_t c;
            c.val[0] = vmulq_f32(a.val[0], b.val[0]);
            c.val[1] = vmulq_f32(a.val[1], b.val[1]);
            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[1] = vaddq_f32(sum_.val[1], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y2 + dim - d);
            float32x4x2_t c;
            c.val[0] = vmulq_f32(a.val[0], b.val[0]);
            c.val[1] = vmulq_f32(a.val[1], b.val[1]);
            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[2] = vaddq_f32(sum_.val[2], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y3 + dim - d);
            float32x4x2_t c;
            c.val[0] = vmulq_f32(a.val[0], b.val[0]);
            c.val[1] = vmulq_f32(a.val[1], b.val[1]);
            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[3] = vaddq_f32(sum_.val[3], c.val[0]);
        }

        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        {
            float32x4_t b = vld1q_f32(y0 + dim - d);
            float32x4_t c;
            c = vmulq_f32(a, b);
            sum_.val[0] = vaddq_f32(sum_.val[0], c);
        }

        {
            float32x4_t b = vld1q_f32(y1 + dim - d);
            float32x4_t c;
            c = vmulq_f32(a, b);
            sum_.val[1] = vaddq_f32(sum_.val[1], c);
        }

        {
            float32x4_t b = vld1q_f32(y2 + dim - d);
            float32x4_t c;
            c = vmulq_f32(a, b);
            sum_.val[2] = vaddq_f32(sum_.val[2], c);
        }
        {
            float32x4_t b = vld1q_f32(y3 + dim - d);
            float32x4_t c;
            c = vmulq_f32(a, b);
            sum_.val[3] = vaddq_f32(sum_.val[3], c);
        }

        d -= 4;
    }

    float32x4_t res_x = vdupq_n_f32(0.0f);
    float32x4x4_t res_y = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 2);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 2);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 2);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 2);

        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 1);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 1);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 1);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 1);

        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 0);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 0);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 0);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 0);

        d -= 1;
    }

    sum_.val[0] = vaddq_f32(sum_.val[0], vmulq_f32(res_x, res_y.val[0]));
    sum_.val[1] = vaddq_f32(sum_.val[1], vmulq_f32(res_x, res_y.val[1]));
    sum_.val[2] = vaddq_f32(sum_.val[2], vmulq_f32(res_x, res_y.val[2]));
    sum_.val[3] = vaddq_f32(sum_.val[3], vmulq_f32(res_x, res_y.val[3]));

    dis0 = vaddvq_f32(sum_.val[0]);
    dis1 = vaddvq_f32(sum_.val[1]);
    dis2 = vaddvq_f32(sum_.val[2]);
    dis3 = vaddvq_f32(sum_.val[3]);
}

void
fvec_L2sqr_batch_4_neon(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                        const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3) {
    float32x4x4_t sum_ = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    auto d = dim;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        {
            float32x4x4_t b = vld1q_f32_x4(y0 + dim - d);
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

            sum_.val[0] = vaddq_f32(sum_.val[0], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y1 + dim - d);
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

            sum_.val[1] = vaddq_f32(sum_.val[1], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y2 + dim - d);
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

            sum_.val[2] = vaddq_f32(sum_.val[2], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y3 + dim - d);
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

            sum_.val[3] = vaddq_f32(sum_.val[3], c.val[0]);
        }

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);

        {
            float32x4x2_t b = vld1q_f32_x2(y0 + dim - d);
            float32x4x2_t c;

            c.val[0] = vsubq_f32(a.val[0], b.val[0]);
            c.val[1] = vsubq_f32(a.val[1], b.val[1]);

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[0] = vaddq_f32(sum_.val[0], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y1 + dim - d);
            float32x4x2_t c;

            c.val[0] = vsubq_f32(a.val[0], b.val[0]);
            c.val[1] = vsubq_f32(a.val[1], b.val[1]);

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[1] = vaddq_f32(sum_.val[1], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y2 + dim - d);
            float32x4x2_t c;

            c.val[0] = vsubq_f32(a.val[0], b.val[0]);
            c.val[1] = vsubq_f32(a.val[1], b.val[1]);

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[2] = vaddq_f32(sum_.val[2], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y3 + dim - d);
            float32x4x2_t c;

            c.val[0] = vsubq_f32(a.val[0], b.val[0]);
            c.val[1] = vsubq_f32(a.val[1], b.val[1]);

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[3] = vaddq_f32(sum_.val[3], c.val[0]);
        }

        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        {
            float32x4_t b = vld1q_f32(y0 + dim - d);
            float32x4_t c;
            c = vsubq_f32(a, b);
            c = vmulq_f32(c, c);
            sum_.val[0] = vaddq_f32(sum_.val[0], c);
        }

        {
            float32x4_t b = vld1q_f32(y1 + dim - d);
            float32x4_t c;
            c = vsubq_f32(a, b);
            c = vmulq_f32(c, c);
            sum_.val[1] = vaddq_f32(sum_.val[1], c);
        }

        {
            float32x4_t b = vld1q_f32(y2 + dim - d);
            float32x4_t c;
            c = vsubq_f32(a, b);
            c = vmulq_f32(c, c);
            sum_.val[2] = vaddq_f32(sum_.val[2], c);
        }
        {
            float32x4_t b = vld1q_f32(y3 + dim - d);
            float32x4_t c;
            c = vsubq_f32(a, b);
            c = vmulq_f32(c, c);
            sum_.val[3] = vaddq_f32(sum_.val[3], c);
        }

        d -= 4;
    }

    float32x4_t res_x = vdupq_n_f32(0.0f);
    float32x4x4_t res_y = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 2);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 2);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 2);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 2);

        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 1);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 1);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 1);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 1);

        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 0);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 0);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 0);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 0);

        d -= 1;
    }

    sum_.val[0] = vaddq_f32(sum_.val[0], vmulq_f32(vsubq_f32(res_x, res_y.val[0]), vsubq_f32(res_x, res_y.val[0])));
    sum_.val[1] = vaddq_f32(sum_.val[1], vmulq_f32(vsubq_f32(res_x, res_y.val[1]), vsubq_f32(res_x, res_y.val[1])));
    sum_.val[2] = vaddq_f32(sum_.val[2], vmulq_f32(vsubq_f32(res_x, res_y.val[2]), vsubq_f32(res_x, res_y.val[2])));
    sum_.val[3] = vaddq_f32(sum_.val[3], vmulq_f32(vsubq_f32(res_x, res_y.val[3]), vsubq_f32(res_x, res_y.val[3])));

    dis0 = vaddvq_f32(sum_.val[0]);
    dis1 = vaddvq_f32(sum_.val[1]);
    dis2 = vaddvq_f32(sum_.val[2]);
    dis3 = vaddvq_f32(sum_.val[3]);
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
        float16x4_t res_x = vdup_n_f16(0.0f);
        float16x4_t res_y = vdup_n_f16(0.0f);
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
        float16x4_t res_x = vdup_n_f16(0.0f);
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

void
fp16_vec_inner_product_batch_4_neon(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                    const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                                    float& dis1, float& dis2, float& dis3) {
    // res store sub result of {4*dis0, 4*dis1, d4*is2, 4*dis3}
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    auto cur_d = d;
    while (cur_d >= 16) {
        float32x4x4_t a = vcvt4_f32_f16(vld4_f16((const __fp16*)x));
        float32x4x4_t b0 = vcvt4_f32_f16(vld4_f16((const __fp16*)y0));
        float32x4x4_t b1 = vcvt4_f32_f16(vld4_f16((const __fp16*)y1));
        float32x4x4_t b2 = vcvt4_f32_f16(vld4_f16((const __fp16*)y2));
        float32x4x4_t b3 = vcvt4_f32_f16(vld4_f16((const __fp16*)y3));

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
        float32x4x2_t a = vcvt2_f32_f16(vld2_f16((const __fp16*)x));
        float32x4x2_t b0 = vcvt2_f32_f16(vld2_f16((const __fp16*)y0));
        float32x4x2_t b1 = vcvt2_f32_f16(vld2_f16((const __fp16*)y1));
        float32x4x2_t b2 = vcvt2_f32_f16(vld2_f16((const __fp16*)y2));
        float32x4x2_t b3 = vcvt2_f32_f16(vld2_f16((const __fp16*)y3));
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
        float32x4_t a = vcvt_f32_f16(vld1_f16((const __fp16*)x));
        float32x4_t b0 = vcvt_f32_f16(vld1_f16((const __fp16*)y0));
        float32x4_t b1 = vcvt_f32_f16(vld1_f16((const __fp16*)y1));
        float32x4_t b2 = vcvt_f32_f16(vld1_f16((const __fp16*)y2));
        float32x4_t b3 = vcvt_f32_f16(vld1_f16((const __fp16*)y3));
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
        float16x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y0 = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y1 = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y2 = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y3 = {0.0f, 0.0f, 0.0f, 0.0f};
        switch (cur_d) {
            case 3:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 2);
                res_y0 = vld1_lane_f16((const __fp16*)y0, res_y0, 2);
                res_y1 = vld1_lane_f16((const __fp16*)y1, res_y1, 2);
                res_y2 = vld1_lane_f16((const __fp16*)y2, res_y2, 2);
                res_y3 = vld1_lane_f16((const __fp16*)y3, res_y3, 2);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
            case 2:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 1);
                res_y0 = vld1_lane_f16((const __fp16*)y0, res_y0, 1);
                res_y1 = vld1_lane_f16((const __fp16*)y1, res_y1, 1);
                res_y2 = vld1_lane_f16((const __fp16*)y2, res_y2, 1);
                res_y3 = vld1_lane_f16((const __fp16*)y3, res_y3, 1);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
            case 1:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 0);
                res_y0 = vld1_lane_f16((const __fp16*)y0, res_y0, 0);
                res_y1 = vld1_lane_f16((const __fp16*)y1, res_y1, 0);
                res_y2 = vld1_lane_f16((const __fp16*)y2, res_y2, 0);
                res_y3 = vld1_lane_f16((const __fp16*)y3, res_y3, 0);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
        }
        res.val[0] = vmlaq_f32(res.val[0], vcvt_f32_f16(res_x), vcvt_f32_f16(res_y0));
        res.val[1] = vmlaq_f32(res.val[1], vcvt_f32_f16(res_x), vcvt_f32_f16(res_y1));
        res.val[2] = vmlaq_f32(res.val[2], vcvt_f32_f16(res_x), vcvt_f32_f16(res_y2));
        res.val[3] = vmlaq_f32(res.val[3], vcvt_f32_f16(res_x), vcvt_f32_f16(res_y3));
    }
    dis0 = vaddvq_f32(res.val[0]);
    dis1 = vaddvq_f32(res.val[1]);
    dis2 = vaddvq_f32(res.val[2]);
    dis3 = vaddvq_f32(res.val[3]);
}

void
fp16_vec_L2sqr_batch_4_neon(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                            const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                            float& dis1, float& dis2, float& dis3) {
    float32x4x4_t res = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    auto cur_d = d;
    while (cur_d >= 16) {
        float32x4x4_t a = vcvt4_f32_f16(vld4_f16((const __fp16*)x));
        float32x4x4_t b0 = vcvt4_f32_f16(vld4_f16((const __fp16*)y0));
        float32x4x4_t b1 = vcvt4_f32_f16(vld4_f16((const __fp16*)y1));
        float32x4x4_t b2 = vcvt4_f32_f16(vld4_f16((const __fp16*)y2));
        float32x4x4_t b3 = vcvt4_f32_f16(vld4_f16((const __fp16*)y3));

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

        res.val[0] = vaddq_f32(vmulq_f32(b0.val[0], b0.val[0]), res.val[0]);
        res.val[1] = vaddq_f32(vmulq_f32(b1.val[0], b1.val[0]), res.val[1]);
        res.val[2] = vaddq_f32(vmulq_f32(b2.val[0], b2.val[0]), res.val[2]);
        res.val[3] = vaddq_f32(vmulq_f32(b3.val[0], b3.val[0]), res.val[3]);

        res.val[0] = vaddq_f32(vmulq_f32(b0.val[1], b0.val[1]), res.val[0]);
        res.val[1] = vaddq_f32(vmulq_f32(b1.val[1], b1.val[1]), res.val[1]);
        res.val[2] = vaddq_f32(vmulq_f32(b2.val[1], b2.val[1]), res.val[2]);
        res.val[3] = vaddq_f32(vmulq_f32(b3.val[1], b3.val[1]), res.val[3]);

        res.val[0] = vaddq_f32(vmulq_f32(b0.val[2], b0.val[2]), res.val[0]);
        res.val[1] = vaddq_f32(vmulq_f32(b1.val[2], b1.val[2]), res.val[1]);
        res.val[2] = vaddq_f32(vmulq_f32(b2.val[2], b2.val[2]), res.val[2]);
        res.val[3] = vaddq_f32(vmulq_f32(b3.val[2], b3.val[2]), res.val[3]);

        res.val[0] = vaddq_f32(vmulq_f32(b0.val[3], b0.val[3]), res.val[0]);
        res.val[1] = vaddq_f32(vmulq_f32(b1.val[3], b1.val[3]), res.val[1]);
        res.val[2] = vaddq_f32(vmulq_f32(b2.val[3], b2.val[3]), res.val[2]);
        res.val[3] = vaddq_f32(vmulq_f32(b3.val[3], b3.val[3]), res.val[3]);

        cur_d -= 16;
        x += 16;
        y0 += 16;
        y1 += 16;
        y2 += 16;
        y3 += 16;
    }
    if (cur_d >= 8) {
        float32x4x2_t a = vcvt2_f32_f16(vld2_f16((const __fp16*)x));
        float32x4x2_t b0 = vcvt2_f32_f16(vld2_f16((const __fp16*)y0));
        float32x4x2_t b1 = vcvt2_f32_f16(vld2_f16((const __fp16*)y1));
        float32x4x2_t b2 = vcvt2_f32_f16(vld2_f16((const __fp16*)y2));
        float32x4x2_t b3 = vcvt2_f32_f16(vld2_f16((const __fp16*)y3));
        b0.val[0] = vsubq_f32(a.val[0], b0.val[0]);
        b0.val[1] = vsubq_f32(a.val[1], b0.val[1]);

        b1.val[0] = vsubq_f32(a.val[0], b1.val[0]);
        b1.val[1] = vsubq_f32(a.val[1], b1.val[1]);

        b2.val[0] = vsubq_f32(a.val[0], b2.val[0]);
        b2.val[1] = vsubq_f32(a.val[1], b2.val[1]);

        b3.val[0] = vsubq_f32(a.val[0], b3.val[0]);
        b3.val[1] = vsubq_f32(a.val[1], b3.val[1]);

        res.val[0] = vaddq_f32(vmulq_f32(b0.val[0], b0.val[0]), res.val[0]);
        res.val[1] = vaddq_f32(vmulq_f32(b1.val[0], b1.val[0]), res.val[1]);
        res.val[2] = vaddq_f32(vmulq_f32(b2.val[0], b2.val[0]), res.val[2]);
        res.val[3] = vaddq_f32(vmulq_f32(b3.val[0], b3.val[0]), res.val[3]);

        res.val[0] = vaddq_f32(vmulq_f32(b0.val[1], b0.val[1]), res.val[0]);
        res.val[1] = vaddq_f32(vmulq_f32(b1.val[1], b1.val[1]), res.val[1]);
        res.val[2] = vaddq_f32(vmulq_f32(b2.val[1], b2.val[1]), res.val[2]);
        res.val[3] = vaddq_f32(vmulq_f32(b3.val[1], b3.val[1]), res.val[3]);

        cur_d -= 8;
        x += 8;
        y0 += 8;
        y1 += 8;
        y2 += 8;
        y3 += 8;
    }
    if (cur_d >= 4) {
        float32x4_t a = vcvt_f32_f16(vld1_f16((const __fp16*)x));
        float32x4_t b0 = vcvt_f32_f16(vld1_f16((const __fp16*)y0));
        float32x4_t b1 = vcvt_f32_f16(vld1_f16((const __fp16*)y1));
        float32x4_t b2 = vcvt_f32_f16(vld1_f16((const __fp16*)y2));
        float32x4_t b3 = vcvt_f32_f16(vld1_f16((const __fp16*)y3));

        b0 = vsubq_f32(a, b0);
        b1 = vsubq_f32(a, b1);
        b2 = vsubq_f32(a, b2);
        b3 = vsubq_f32(a, b3);

        res.val[0] = vaddq_f32(vmulq_f32(b0, b0), res.val[0]);
        res.val[1] = vaddq_f32(vmulq_f32(b1, b1), res.val[1]);
        res.val[2] = vaddq_f32(vmulq_f32(b2, b2), res.val[2]);
        res.val[3] = vaddq_f32(vmulq_f32(b3, b3), res.val[3]);
        cur_d -= 4;
        x += 4;
        y0 += 4;
        y1 += 4;
        y2 += 4;
        y3 += 4;
    }
    if (cur_d >= 0) {
        float16x4_t res_x = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y0 = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y1 = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y2 = {0.0f, 0.0f, 0.0f, 0.0f};
        float16x4_t res_y3 = {0.0f, 0.0f, 0.0f, 0.0f};
        switch (cur_d) {
            case 3:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 2);
                res_y0 = vld1_lane_f16((const __fp16*)y0, res_y0, 2);
                res_y1 = vld1_lane_f16((const __fp16*)y1, res_y1, 2);
                res_y2 = vld1_lane_f16((const __fp16*)y2, res_y2, 2);
                res_y3 = vld1_lane_f16((const __fp16*)y3, res_y3, 2);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
            case 2:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 1);
                res_y0 = vld1_lane_f16((const __fp16*)y0, res_y0, 1);
                res_y1 = vld1_lane_f16((const __fp16*)y1, res_y1, 1);
                res_y2 = vld1_lane_f16((const __fp16*)y2, res_y2, 1);
                res_y3 = vld1_lane_f16((const __fp16*)y3, res_y3, 1);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
            case 1:
                res_x = vld1_lane_f16((const __fp16*)x, res_x, 0);
                res_y0 = vld1_lane_f16((const __fp16*)y0, res_y0, 0);
                res_y1 = vld1_lane_f16((const __fp16*)y1, res_y1, 0);
                res_y2 = vld1_lane_f16((const __fp16*)y2, res_y2, 0);
                res_y3 = vld1_lane_f16((const __fp16*)y3, res_y3, 0);
                x++;
                y0++;
                y1++;
                y2++;
                y3++;
                cur_d--;
        }
        float32x4_t diff0 = vsubq_f32(vcvt_f32_f16(res_x), vcvt_f32_f16(res_y0));
        float32x4_t diff1 = vsubq_f32(vcvt_f32_f16(res_x), vcvt_f32_f16(res_y1));
        float32x4_t diff2 = vsubq_f32(vcvt_f32_f16(res_x), vcvt_f32_f16(res_y2));
        float32x4_t diff3 = vsubq_f32(vcvt_f32_f16(res_x), vcvt_f32_f16(res_y3));
        res.val[0] = vaddq_f32(vmulq_f32(diff0, diff0), res.val[0]);
        res.val[1] = vaddq_f32(vmulq_f32(diff1, diff1), res.val[1]);
        res.val[2] = vaddq_f32(vmulq_f32(diff2, diff2), res.val[2]);
        res.val[3] = vaddq_f32(vmulq_f32(diff3, diff3), res.val[3]);
    }
    dis0 = vaddvq_f32(res.val[0]);
    dis1 = vaddvq_f32(res.val[1]);
    dis2 = vaddvq_f32(res.val[2]);
    dis3 = vaddvq_f32(res.val[3]);
}

///////////////////////////////////////////////////////////////////////////////
// bf16

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
bf16_vec_inner_product_batch_4_neon(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                    const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
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
bf16_vec_L2sqr_batch_4_neon(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                            const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
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

        // use vdotq_s32 to calculate dot product and accumulate
        sum_ = vdotq_s32(sum_, a, b);

        x += 16;
        y += 16;
        d -= 16;
    }

    // process 8 int8_t elements each time
    if (d >= 8) {
        int8x8_t a = vld1_s8(x);
        int8x8_t b = vld1_s8(y);

        int8x16_t a_ext = vcombine_s8(a, vdup_n_s8(0));
        int8x16_t b_ext = vcombine_s8(b, vdup_n_s8(0));

        sum_ = vdotq_s32(sum_, a_ext, b_ext);

        x += 8;
        y += 8;
        d -= 8;
    }

    // process 4 int8_t elements each time
    if (d >= 4) {
        int8x8_t a = vld1_s8(x);
        int8x8_t b = vld1_s8(y);

        int8x16_t a_ext = vcombine_s8(a, vdup_n_s8(0));
        int8x16_t b_ext = vcombine_s8(b, vdup_n_s8(0));

        sum_ = vdotq_s32(sum_, a_ext, b_ext);

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

///////////////////////////////////////////////////////////////////////////////
// for cardinal

float
fvec_inner_product_bf16_patch_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum_ = vdupq_n_f32(0.0f);
    auto dim = d;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        float32x4x4_t b = vld1q_f32_x4(y + dim - d);

        a.val[0] = bf16_float_neon(a.val[0]);
        a.val[1] = bf16_float_neon(a.val[1]);
        a.val[2] = bf16_float_neon(a.val[2]);
        a.val[3] = bf16_float_neon(a.val[3]);

        b.val[0] = bf16_float_neon(b.val[0]);
        b.val[1] = bf16_float_neon(b.val[1]);
        b.val[2] = bf16_float_neon(b.val[2]);
        b.val[3] = bf16_float_neon(b.val[3]);
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

        a.val[0] = bf16_float_neon(a.val[0]);
        a.val[1] = bf16_float_neon(a.val[1]);

        b.val[0] = bf16_float_neon(b.val[0]);
        b.val[1] = bf16_float_neon(b.val[1]);

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
        a = bf16_float_neon(a);
        b = bf16_float_neon(b);
        float32x4_t c;
        c = vmulq_f32(a, b);
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
    res_x = bf16_float_neon(res_x);
    res_y = bf16_float_neon(res_y);

    sum_ = vaddq_f32(sum_, vmulq_f32(res_x, res_y));
    return vaddvq_f32(sum_);
}

float
fvec_L2sqr_bf16_patch_neon(const float* x, const float* y, size_t d) {
    float32x4_t sum_ = vdupq_n_f32(0.0f);
    auto dim = d;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        float32x4x4_t b = vld1q_f32_x4(y + dim - d);
        a.val[0] = bf16_float_neon(a.val[0]);
        a.val[1] = bf16_float_neon(a.val[1]);
        a.val[2] = bf16_float_neon(a.val[2]);
        a.val[3] = bf16_float_neon(a.val[3]);

        b.val[0] = bf16_float_neon(b.val[0]);
        b.val[1] = bf16_float_neon(b.val[1]);
        b.val[2] = bf16_float_neon(b.val[2]);
        b.val[3] = bf16_float_neon(b.val[3]);

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

        a.val[0] = bf16_float_neon(a.val[0]);
        a.val[1] = bf16_float_neon(a.val[1]);

        b.val[0] = bf16_float_neon(b.val[0]);
        b.val[1] = bf16_float_neon(b.val[1]);

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
        a = bf16_float_neon(a);
        b = bf16_float_neon(b);
        float32x4_t c;
        c = vsubq_f32(a, b);
        c = vmulq_f32(c, c);

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

    res_x = bf16_float_neon(res_x);
    res_y = bf16_float_neon(res_y);

    sum_ = vaddq_f32(sum_, vmulq_f32(vsubq_f32(res_x, res_y), vsubq_f32(res_x, res_y)));
    return vaddvq_f32(sum_);
}

void
fvec_inner_product_batch_4_bf16_patch_neon(const float* x, const float* y0, const float* y1, const float* y2,
                                           const float* y3, const size_t dim, float& dis0, float& dis1, float& dis2,
                                           float& dis3) {
    float32x4x4_t sum_ = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    auto d = dim;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);

        a.val[0] = bf16_float_neon(a.val[0]);
        a.val[1] = bf16_float_neon(a.val[1]);
        a.val[2] = bf16_float_neon(a.val[2]);
        a.val[3] = bf16_float_neon(a.val[3]);

        {
            float32x4x4_t b = vld1q_f32_x4(y0 + dim - d);
            float32x4x4_t c;
            c.val[0] = vmulq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vmulq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[2] = vmulq_f32(a.val[2], bf16_float_neon(b.val[2]));
            c.val[3] = vmulq_f32(a.val[3], bf16_float_neon(b.val[3]));

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[0] = vaddq_f32(sum_.val[0], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y1 + dim - d);
            float32x4x4_t c;
            c.val[0] = vmulq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vmulq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[2] = vmulq_f32(a.val[2], bf16_float_neon(b.val[2]));
            c.val[3] = vmulq_f32(a.val[3], bf16_float_neon(b.val[3]));

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[1] = vaddq_f32(sum_.val[1], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y2 + dim - d);
            float32x4x4_t c;
            c.val[0] = vmulq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vmulq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[2] = vmulq_f32(a.val[2], bf16_float_neon(b.val[2]));
            c.val[3] = vmulq_f32(a.val[3], bf16_float_neon(b.val[3]));

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[2] = vaddq_f32(sum_.val[2], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y3 + dim - d);
            float32x4x4_t c;
            c.val[0] = vmulq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vmulq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[2] = vmulq_f32(a.val[2], bf16_float_neon(b.val[2]));
            c.val[3] = vmulq_f32(a.val[3], bf16_float_neon(b.val[3]));

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[3] = vaddq_f32(sum_.val[3], c.val[0]);
        }

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);
        a.val[0] = bf16_float_neon(a.val[0]);
        a.val[1] = bf16_float_neon(a.val[1]);

        {
            float32x4x2_t b = vld1q_f32_x2(y0 + dim - d);
            float32x4x2_t c;
            c.val[0] = vmulq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vmulq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[0] = vaddq_f32(sum_.val[0], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y1 + dim - d);
            float32x4x2_t c;
            c.val[0] = vmulq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vmulq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[1] = vaddq_f32(sum_.val[1], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y2 + dim - d);
            float32x4x2_t c;
            c.val[0] = vmulq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vmulq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[2] = vaddq_f32(sum_.val[2], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y3 + dim - d);
            float32x4x2_t c;
            c.val[0] = vmulq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vmulq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[3] = vaddq_f32(sum_.val[3], c.val[0]);
        }

        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        a = bf16_float_neon(a);
        {
            float32x4_t b = vld1q_f32(y0 + dim - d);
            float32x4_t c;
            c = vmulq_f32(a, bf16_float_neon(b));
            sum_.val[0] = vaddq_f32(sum_.val[0], c);
        }

        {
            float32x4_t b = vld1q_f32(y1 + dim - d);
            float32x4_t c;
            c = vmulq_f32(a, bf16_float_neon(b));
            sum_.val[1] = vaddq_f32(sum_.val[1], c);
        }

        {
            float32x4_t b = vld1q_f32(y2 + dim - d);
            float32x4_t c;
            c = vmulq_f32(a, bf16_float_neon(b));
            sum_.val[2] = vaddq_f32(sum_.val[2], c);
        }
        {
            float32x4_t b = vld1q_f32(y3 + dim - d);
            float32x4_t c;
            c = vmulq_f32(a, bf16_float_neon(b));
            sum_.val[3] = vaddq_f32(sum_.val[3], c);
        }

        d -= 4;
    }

    float32x4_t res_x = vdupq_n_f32(0.0f);
    float32x4x4_t res_y = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 2);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 2);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 2);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 2);

        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 1);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 1);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 1);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 1);

        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 0);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 0);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 0);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 0);

        d -= 1;
    }

    res_x = bf16_float_neon(res_x);
    res_y.val[0] = bf16_float_neon(res_y.val[0]);
    res_y.val[1] = bf16_float_neon(res_y.val[1]);
    res_y.val[2] = bf16_float_neon(res_y.val[2]);
    res_y.val[3] = bf16_float_neon(res_y.val[3]);

    sum_.val[0] = vaddq_f32(sum_.val[0], vmulq_f32(res_x, res_y.val[0]));
    sum_.val[1] = vaddq_f32(sum_.val[1], vmulq_f32(res_x, res_y.val[1]));
    sum_.val[2] = vaddq_f32(sum_.val[2], vmulq_f32(res_x, res_y.val[2]));
    sum_.val[3] = vaddq_f32(sum_.val[3], vmulq_f32(res_x, res_y.val[3]));

    dis0 = vaddvq_f32(sum_.val[0]);
    dis1 = vaddvq_f32(sum_.val[1]);
    dis2 = vaddvq_f32(sum_.val[2]);
    dis3 = vaddvq_f32(sum_.val[3]);
}

void
fvec_L2sqr_batch_4_bf16_patch_neon(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                   const size_t dim, float& dis0, float& dis1, float& dis2, float& dis3) {
    float32x4x4_t sum_ = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    auto d = dim;
    while (d >= 16) {
        float32x4x4_t a = vld1q_f32_x4(x + dim - d);
        a.val[0] = bf16_float_neon(a.val[0]);
        a.val[1] = bf16_float_neon(a.val[1]);
        a.val[2] = bf16_float_neon(a.val[2]);
        a.val[3] = bf16_float_neon(a.val[3]);

        {
            float32x4x4_t b = vld1q_f32_x4(y0 + dim - d);
            float32x4x4_t c;

            c.val[0] = vsubq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vsubq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[2] = vsubq_f32(a.val[2], bf16_float_neon(b.val[2]));
            c.val[3] = vsubq_f32(a.val[3], bf16_float_neon(b.val[3]));

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);
            c.val[2] = vmulq_f32(c.val[2], c.val[2]);
            c.val[3] = vmulq_f32(c.val[3], c.val[3]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[0] = vaddq_f32(sum_.val[0], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y1 + dim - d);
            float32x4x4_t c;

            c.val[0] = vsubq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vsubq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[2] = vsubq_f32(a.val[2], bf16_float_neon(b.val[2]));
            c.val[3] = vsubq_f32(a.val[3], bf16_float_neon(b.val[3]));

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);
            c.val[2] = vmulq_f32(c.val[2], c.val[2]);
            c.val[3] = vmulq_f32(c.val[3], c.val[3]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[1] = vaddq_f32(sum_.val[1], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y2 + dim - d);
            float32x4x4_t c;

            c.val[0] = vsubq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vsubq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[2] = vsubq_f32(a.val[2], bf16_float_neon(b.val[2]));
            c.val[3] = vsubq_f32(a.val[3], bf16_float_neon(b.val[3]));

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);
            c.val[2] = vmulq_f32(c.val[2], c.val[2]);
            c.val[3] = vmulq_f32(c.val[3], c.val[3]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[2] = vaddq_f32(sum_.val[2], c.val[0]);
        }

        {
            float32x4x4_t b = vld1q_f32_x4(y3 + dim - d);
            float32x4x4_t c;

            c.val[0] = vsubq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vsubq_f32(a.val[1], bf16_float_neon(b.val[1]));
            c.val[2] = vsubq_f32(a.val[2], bf16_float_neon(b.val[2]));
            c.val[3] = vsubq_f32(a.val[3], bf16_float_neon(b.val[3]));

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);
            c.val[2] = vmulq_f32(c.val[2], c.val[2]);
            c.val[3] = vmulq_f32(c.val[3], c.val[3]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            c.val[2] = vaddq_f32(c.val[2], c.val[3]);
            c.val[0] = vaddq_f32(c.val[0], c.val[2]);

            sum_.val[3] = vaddq_f32(sum_.val[3], c.val[0]);
        }

        d -= 16;
    }

    if (d >= 8) {
        float32x4x2_t a = vld1q_f32_x2(x + dim - d);
        a.val[0] = bf16_float_neon(a.val[0]);
        a.val[1] = bf16_float_neon(a.val[1]);
        {
            float32x4x2_t b = vld1q_f32_x2(y0 + dim - d);
            float32x4x2_t c;

            c.val[0] = vsubq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vsubq_f32(a.val[1], bf16_float_neon(b.val[1]));

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[0] = vaddq_f32(sum_.val[0], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y1 + dim - d);
            float32x4x2_t c;

            c.val[0] = vsubq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vsubq_f32(a.val[1], bf16_float_neon(b.val[1]));

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[1] = vaddq_f32(sum_.val[1], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y2 + dim - d);
            float32x4x2_t c;

            c.val[0] = vsubq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vsubq_f32(a.val[1], bf16_float_neon(b.val[1]));

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[2] = vaddq_f32(sum_.val[2], c.val[0]);
        }
        {
            float32x4x2_t b = vld1q_f32_x2(y3 + dim - d);
            float32x4x2_t c;

            c.val[0] = vsubq_f32(a.val[0], bf16_float_neon(b.val[0]));
            c.val[1] = vsubq_f32(a.val[1], bf16_float_neon(b.val[1]));

            c.val[0] = vmulq_f32(c.val[0], c.val[0]);
            c.val[1] = vmulq_f32(c.val[1], c.val[1]);

            c.val[0] = vaddq_f32(c.val[0], c.val[1]);
            sum_.val[3] = vaddq_f32(sum_.val[3], c.val[0]);
        }

        d -= 8;
    }
    if (d >= 4) {
        float32x4_t a = vld1q_f32(x + dim - d);
        a = bf16_float_neon(a);
        {
            float32x4_t b = vld1q_f32(y0 + dim - d);
            float32x4_t c;
            c = vsubq_f32(a, bf16_float_neon(b));
            c = vmulq_f32(c, c);
            sum_.val[0] = vaddq_f32(sum_.val[0], c);
        }

        {
            float32x4_t b = vld1q_f32(y1 + dim - d);
            float32x4_t c;
            c = vsubq_f32(a, bf16_float_neon(b));
            c = vmulq_f32(c, c);
            sum_.val[1] = vaddq_f32(sum_.val[1], c);
        }

        {
            float32x4_t b = vld1q_f32(y2 + dim - d);
            float32x4_t c;
            c = vsubq_f32(a, bf16_float_neon(b));
            c = vmulq_f32(c, c);
            sum_.val[2] = vaddq_f32(sum_.val[2], c);
        }
        {
            float32x4_t b = vld1q_f32(y3 + dim - d);
            float32x4_t c;
            c = vsubq_f32(a, bf16_float_neon(b));
            c = vmulq_f32(c, c);
            sum_.val[3] = vaddq_f32(sum_.val[3], c);
        }

        d -= 4;
    }

    float32x4_t res_x = vdupq_n_f32(0.0f);
    float32x4x4_t res_y = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
    if (d >= 3) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 2);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 2);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 2);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 2);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 2);

        d -= 1;
    }

    if (d >= 2) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 1);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 1);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 1);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 1);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 1);

        d -= 1;
    }

    if (d >= 1) {
        res_x = vld1q_lane_f32(x + dim - d, res_x, 0);
        res_y.val[0] = vld1q_lane_f32(y0 + dim - d, res_y.val[0], 0);
        res_y.val[1] = vld1q_lane_f32(y1 + dim - d, res_y.val[1], 0);
        res_y.val[2] = vld1q_lane_f32(y2 + dim - d, res_y.val[2], 0);
        res_y.val[3] = vld1q_lane_f32(y3 + dim - d, res_y.val[3], 0);

        d -= 1;
    }

    res_x = bf16_float_neon(res_x);
    res_y.val[0] = bf16_float_neon(res_y.val[0]);
    res_y.val[1] = bf16_float_neon(res_y.val[1]);
    res_y.val[2] = bf16_float_neon(res_y.val[2]);
    res_y.val[3] = bf16_float_neon(res_y.val[3]);

    sum_.val[0] = vaddq_f32(sum_.val[0], vmulq_f32(vsubq_f32(res_x, res_y.val[0]), vsubq_f32(res_x, res_y.val[0])));
    sum_.val[1] = vaddq_f32(sum_.val[1], vmulq_f32(vsubq_f32(res_x, res_y.val[1]), vsubq_f32(res_x, res_y.val[1])));
    sum_.val[2] = vaddq_f32(sum_.val[2], vmulq_f32(vsubq_f32(res_x, res_y.val[2]), vsubq_f32(res_x, res_y.val[2])));
    sum_.val[3] = vaddq_f32(sum_.val[3], vmulq_f32(vsubq_f32(res_x, res_y.val[3]), vsubq_f32(res_x, res_y.val[3])));

    dis0 = vaddvq_f32(sum_.val[0]);
    dis1 = vaddvq_f32(sum_.val[1]);
    dis2 = vaddvq_f32(sum_.val[2]);
    dis3 = vaddvq_f32(sum_.val[3]);
}

}  // namespace faiss
#endif
