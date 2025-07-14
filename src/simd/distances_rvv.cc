// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#if defined(__riscv_vector)

#include "distances_rvv.h"

#include <math.h>
#include <riscv_vector.h>

namespace faiss {

// bf16 conversion helper function
__attribute__((always_inline)) inline vfloat32m2_t
bf16_float_rvv(vfloat32m2_t f, size_t vl) {
    // Convert float to integer bits
    vuint32m2_t bits = __riscv_vreinterpret_v_f32m2_u32m2(f);

    // Add rounding constant
    vuint32m2_t rounded_bits = __riscv_vadd_vx_u32m2(bits, 0x8000, vl);

    // Mask to retain only the upper 16 bits (for BF16 representation)
    rounded_bits = __riscv_vand_vx_u32m2(rounded_bits, 0xFFFF0000, vl);

    // Convert back to float
    return __riscv_vreinterpret_v_u32m2_f32m2(rounded_bits);
}

__attribute__((always_inline)) inline vfloat32m1_t
bf16_float_rvv(vfloat32m1_t f, size_t vl) {
    // Convert float to integer bits
    vuint32m1_t bits = __riscv_vreinterpret_v_f32m1_u32m1(f);

    // Add rounding constant
    vuint32m1_t rounded_bits = __riscv_vadd_vx_u32m1(bits, 0x8000, vl);

    // Mask to retain only the upper 16 bits (for BF16 representation)
    rounded_bits = __riscv_vand_vx_u32m1(rounded_bits, 0xFFFF0000, vl);

    // Convert back to float
    return __riscv_vreinterpret_v_u32m1_f32m1(rounded_bits);
}

// =================== float distances ===================
float
fvec_inner_product_rvv(const float* x, const float* y, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();  // Use m2 to support 4-way parallelism

    // 4 accumulators
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

    size_t offset = 0;

    // 4-way unrolled loop
    while (d >= 4 * vlmax) {
        size_t vl = vlmax;

        vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy0 = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x + offset + vl, vl);
        vfloat32m2_t vy1 = __riscv_vle32_v_f32m2(y + offset + vl, vl);
        vfloat32m2_t vx2 = __riscv_vle32_v_f32m2(x + offset + 2 * vl, vl);
        vfloat32m2_t vy2 = __riscv_vle32_v_f32m2(y + offset + 2 * vl, vl);
        vfloat32m2_t vx3 = __riscv_vle32_v_f32m2(x + offset + 3 * vl, vl);
        vfloat32m2_t vy3 = __riscv_vle32_v_f32m2(y + offset + 3 * vl, vl);

        // Parallel FMACC operations
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vx0, vy0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m2_tu(vacc1, vx1, vy1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m2_tu(vacc2, vx2, vy2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m2_tu(vacc3, vx3, vy3, vl);

        offset += 4 * vl;
        d -= 4 * vl;
    }

    // Merge accumulators
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc1, vlmax);
    vacc2 = __riscv_vfadd_vv_f32m2(vacc2, vacc3, vlmax);
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc2, vlmax);

    // Handle remaining elements
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m2(d);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy = __riscv_vle32_v_f32m2(y + offset, vl);
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vx, vy, vl);

        offset += vl;
        d -= vl;
    }

    // Final reduction
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);

    return __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

float
fvec_L2sqr_rvv(const float* x, const float* y, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    size_t offset = 0;
    while (d >= 4 * vlmax) {
        size_t vl = vlmax;
        vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy0 = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x + offset + vl, vl);
        vfloat32m2_t vy1 = __riscv_vle32_v_f32m2(y + offset + vl, vl);
        vfloat32m2_t vx2 = __riscv_vle32_v_f32m2(x + offset + 2 * vl, vl);
        vfloat32m2_t vy2 = __riscv_vle32_v_f32m2(y + offset + 2 * vl, vl);
        vfloat32m2_t vx3 = __riscv_vle32_v_f32m2(x + offset + 3 * vl, vl);
        vfloat32m2_t vy3 = __riscv_vle32_v_f32m2(y + offset + 3 * vl, vl);
        vfloat32m2_t vtmp0 = __riscv_vfsub_vv_f32m2(vx0, vy0, vl);
        vfloat32m2_t vtmp1 = __riscv_vfsub_vv_f32m2(vx1, vy1, vl);
        vfloat32m2_t vtmp2 = __riscv_vfsub_vv_f32m2(vx2, vy2, vl);
        vfloat32m2_t vtmp3 = __riscv_vfsub_vv_f32m2(vx3, vy3, vl);
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vtmp0, vtmp0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m2_tu(vacc1, vtmp1, vtmp1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m2_tu(vacc2, vtmp2, vtmp2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m2_tu(vacc3, vtmp3, vtmp3, vl);
        offset += 4 * vl;
        d -= 4 * vl;
    }
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc1, vlmax);
    vacc2 = __riscv_vfadd_vv_f32m2(vacc2, vacc3, vlmax);
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc2, vlmax);
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m2(d);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vtmp = __riscv_vfsub_vv_f32m2(vx, vy, vl);
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vtmp, vtmp, vl);
        offset += vl;
        d -= vl;
    }
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

float
fvec_L1_rvv(const float* x, const float* y, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    size_t offset = 0;
    while (d >= 4 * vlmax) {
        size_t vl = vlmax;
        vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy0 = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x + offset + vl, vl);
        vfloat32m2_t vy1 = __riscv_vle32_v_f32m2(y + offset + vl, vl);
        vfloat32m2_t vx2 = __riscv_vle32_v_f32m2(x + offset + 2 * vl, vl);
        vfloat32m2_t vy2 = __riscv_vle32_v_f32m2(y + offset + 2 * vl, vl);
        vfloat32m2_t vx3 = __riscv_vle32_v_f32m2(x + offset + 3 * vl, vl);
        vfloat32m2_t vy3 = __riscv_vle32_v_f32m2(y + offset + 3 * vl, vl);
        vfloat32m2_t vtmp0 = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx0, vy0, vl), vl);
        vfloat32m2_t vtmp1 = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx1, vy1, vl), vl);
        vfloat32m2_t vtmp2 = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx2, vy2, vl), vl);
        vfloat32m2_t vtmp3 = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx3, vy3, vl), vl);
        vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vtmp0, vl);
        vacc1 = __riscv_vfadd_vv_f32m2(vacc1, vtmp1, vl);
        vacc2 = __riscv_vfadd_vv_f32m2(vacc2, vtmp2, vl);
        vacc3 = __riscv_vfadd_vv_f32m2(vacc3, vtmp3, vl);
        offset += 4 * vl;
        d -= 4 * vl;
    }
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc1, vlmax);
    vacc2 = __riscv_vfadd_vv_f32m2(vacc2, vacc3, vlmax);
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc2, vlmax);
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m2(d);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vtmp = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx, vy, vl), vl);
        vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vtmp, vl);
        offset += vl;
        d -= vl;
    }
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

float
fvec_Linf_rvv(const float* x, const float* y, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vmax0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vmax1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vmax2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vmax3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    size_t offset = 0;
    while (d >= 4 * vlmax) {
        size_t vl = vlmax;
        vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy0 = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x + offset + vl, vl);
        vfloat32m2_t vy1 = __riscv_vle32_v_f32m2(y + offset + vl, vl);
        vfloat32m2_t vx2 = __riscv_vle32_v_f32m2(x + offset + 2 * vl, vl);
        vfloat32m2_t vy2 = __riscv_vle32_v_f32m2(y + offset + 2 * vl, vl);
        vfloat32m2_t vx3 = __riscv_vle32_v_f32m2(x + offset + 3 * vl, vl);
        vfloat32m2_t vy3 = __riscv_vle32_v_f32m2(y + offset + 3 * vl, vl);
        vfloat32m2_t vtmp0 = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx0, vy0, vl), vl);
        vfloat32m2_t vtmp1 = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx1, vy1, vl), vl);
        vfloat32m2_t vtmp2 = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx2, vy2, vl), vl);
        vfloat32m2_t vtmp3 = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx3, vy3, vl), vl);
        vmax0 = __riscv_vfmax_vv_f32m2(vmax0, vtmp0, vl);
        vmax1 = __riscv_vfmax_vv_f32m2(vmax1, vtmp1, vl);
        vmax2 = __riscv_vfmax_vv_f32m2(vmax2, vtmp2, vl);
        vmax3 = __riscv_vfmax_vv_f32m2(vmax3, vtmp3, vl);
        offset += 4 * vl;
        d -= 4 * vl;
    }
    vmax0 = __riscv_vfmax_vv_f32m2(vmax0, vmax1, vlmax);
    vmax2 = __riscv_vfmax_vv_f32m2(vmax2, vmax3, vlmax);
    vmax0 = __riscv_vfmax_vv_f32m2(vmax0, vmax2, vlmax);
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m2(d);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vtmp = __riscv_vfabs_v_f32m2(__riscv_vfsub_vv_f32m2(vx, vy, vl), vl);
        vmax0 = __riscv_vfmax_vv_f32m2(vmax0, vtmp, vl);
        offset += vl;
        d -= vl;
    }
    vfloat32m1_t max_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    max_scalar = __riscv_vfredmax_vs_f32m2_f32m1(vmax0, max_scalar, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(max_scalar);
}

float
fvec_norm_L2sqr_rvv(const float* x, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    size_t offset = 0;
    while (d >= 4 * vlmax) {
        size_t vl = vlmax;
        vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x + offset + vl, vl);
        vfloat32m2_t vx2 = __riscv_vle32_v_f32m2(x + offset + 2 * vl, vl);
        vfloat32m2_t vx3 = __riscv_vle32_v_f32m2(x + offset + 3 * vl, vl);
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vx0, vx0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m2_tu(vacc1, vx1, vx1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m2_tu(vacc2, vx2, vx2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m2_tu(vacc3, vx3, vx3, vl);
        offset += 4 * vl;
        d -= 4 * vl;
    }
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc1, vlmax);
    vacc2 = __riscv_vfadd_vv_f32m2(vacc2, vacc3, vlmax);
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc2, vlmax);
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m2(d);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + offset, vl);
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vx, vx, vl);
        offset += vl;
        d -= vl;
    }
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

void
fvec_L2sqr_ny_rvv(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; ++i) {
        dis[i] = fvec_L2sqr_rvv(x, y, d);
        y += d;
    }
}

void
fvec_inner_products_ny_rvv(float* ip, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; ++i) {
        ip[i] = fvec_inner_product_rvv(x, y, d);
        y += d;
    }
}

void
fvec_madd_rvv(size_t n, const float* a, float bf, const float* b, float* c) {
    size_t offset = 0;
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vbf = __riscv_vfmv_v_f_f32m2(bf, vlmax);
    while (n >= vlmax) {
        size_t vl = vlmax;
        vfloat32m2_t va = __riscv_vle32_v_f32m2(a + offset, vl);
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + offset, vl);
        vfloat32m2_t vc = __riscv_vfmacc_vv_f32m2(va, vbf, vb, vl);
        __riscv_vse32_v_f32m2(c + offset, vc, vl);
        offset += vl;
        n -= vl;
    }
    if (n > 0) {
        size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t va = __riscv_vle32_v_f32m2(a + offset, vl);
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + offset, vl);
        vfloat32m2_t vbf_tail = __riscv_vfmv_v_f_f32m2(bf, vl);
        vfloat32m2_t vc = __riscv_vfmacc_vv_f32m2(va, vbf_tail, vb, vl);
        __riscv_vse32_v_f32m2(c + offset, vc, vl);
    }
}

int
fvec_madd_and_argmin_rvv(size_t n, const float* a, float bf, const float* b, float* c) {
    size_t offset = 0;
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    float min_val = 1e20f;
    int min_idx = -1;
    int idx_base = 0;
    vfloat32m2_t vbf = __riscv_vfmv_v_f_f32m2(bf, vlmax);
    while (n >= vlmax) {
        size_t vl = vlmax;
        vfloat32m2_t va = __riscv_vle32_v_f32m2(a + offset, vl);
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + offset, vl);
        vfloat32m2_t vc = __riscv_vfmacc_vv_f32m2(va, vbf, vb, vl);
        __riscv_vse32_v_f32m2(c + offset, vc, vl);

        // Reduction to find minimum value
        vfloat32m1_t vmin = __riscv_vfmv_s_f_f32m1(1e20f, 1);
        vmin = __riscv_vfredmin_vs_f32m2_f32m1(vc, vmin, vl);
        float local_min = __riscv_vfmv_f_s_f32m1_f32(vmin);
        if (local_min < min_val) {
            // Find the index of minimum value
            for (size_t i = 0; i < vl; ++i) {
                float val = c[offset + i];
                if (val < min_val) {
                    min_val = val;
                    min_idx = idx_base + i;
                }
            }
        }
        offset += vl;
        n -= vl;
        idx_base += vl;
    }
    if (n > 0) {
        size_t vl = __riscv_vsetvl_e32m2(n);
        vfloat32m2_t va = __riscv_vle32_v_f32m2(a + offset, vl);
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + offset, vl);
        vfloat32m2_t vbf_tail = __riscv_vfmv_v_f_f32m2(bf, vl);
        vfloat32m2_t vc = __riscv_vfmacc_vv_f32m2(va, vbf_tail, vb, vl);
        __riscv_vse32_v_f32m2(c + offset, vc, vl);
        vfloat32m1_t vmin = __riscv_vfmv_s_f_f32m1(1e20f, 1);
        vmin = __riscv_vfredmin_vs_f32m2_f32m1(vc, vmin, vl);
        float local_min = __riscv_vfmv_f_s_f32m1_f32(vmin);
        if (local_min < min_val) {
            for (size_t i = 0; i < vl; ++i) {
                float val = c[offset + i];
                if (val < min_val) {
                    min_val = val;
                    min_idx = idx_base + i;
                }
            }
        }
    }
    return min_idx;
}

void
fvec_inner_product_batch_4_rvv(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                               size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    // Use smaller vector length to reduce memory pressure
    size_t vlmax = __riscv_vsetvlmax_e32m1();  // Use m1 instead of m2

    // 4 accumulators
    vfloat32m1_t vacc0 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vacc1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vacc2 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vacc3 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

    size_t offset = 0;

    // Use smaller vector length to reduce memory pressure
    while (d >= vlmax) {
        size_t vl = vlmax;

        // Load data
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + offset, vl);
        vfloat32m1_t vy0 = __riscv_vle32_v_f32m1(y0 + offset, vl);
        vfloat32m1_t vy1 = __riscv_vle32_v_f32m1(y1 + offset, vl);
        vfloat32m1_t vy2 = __riscv_vle32_v_f32m1(y2 + offset, vl);
        vfloat32m1_t vy3 = __riscv_vle32_v_f32m1(y3 + offset, vl);

        // Parallel FMACC operations
        vacc0 = __riscv_vfmacc_vv_f32m1_tu(vacc0, vx, vy0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m1_tu(vacc1, vx, vy1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m1_tu(vacc2, vx, vy2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m1_tu(vacc3, vx, vy3, vl);

        offset += vl;
        d -= vl;
    }

    // Handle remaining elements
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m1(d);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + offset, vl);
        vfloat32m1_t vy0 = __riscv_vle32_v_f32m1(y0 + offset, vl);
        vfloat32m1_t vy1 = __riscv_vle32_v_f32m1(y1 + offset, vl);
        vfloat32m1_t vy2 = __riscv_vle32_v_f32m1(y2 + offset, vl);
        vfloat32m1_t vy3 = __riscv_vle32_v_f32m1(y3 + offset, vl);

        vacc0 = __riscv_vfmacc_vv_f32m1_tu(vacc0, vx, vy0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m1_tu(vacc1, vx, vy1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m1_tu(vacc2, vx, vy2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m1_tu(vacc3, vx, vy3, vl);

        offset += vl;
        d -= vl;
    }

    // Final reduction
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m1_f32m1(vacc0, sum_scalar, vlmax);
    dis0 = __riscv_vfmv_f_s_f32m1_f32(sum_scalar);

    sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m1_f32m1(vacc1, sum_scalar, vlmax);
    dis1 = __riscv_vfmv_f_s_f32m1_f32(sum_scalar);

    sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m1_f32m1(vacc2, sum_scalar, vlmax);
    dis2 = __riscv_vfmv_f_s_f32m1_f32(sum_scalar);

    sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m1_f32m1(vacc3, sum_scalar, vlmax);
    dis3 = __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

void
fvec_L2sqr_batch_4_rvv(const float* x, const float* y0, const float* y1, const float* y2, const float* y3, size_t d,
                       float& dis0, float& dis1, float& dis2, float& dis3) {
    // Use smaller vector length to reduce memory pressure
    size_t vlmax = __riscv_vsetvlmax_e32m1();  // Use m1 instead of m2

    // 4 accumulators
    vfloat32m1_t vacc0 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vacc1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vacc2 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
    vfloat32m1_t vacc3 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

    size_t offset = 0;

    // Use smaller vector length to reduce memory pressure
    while (d >= vlmax) {
        size_t vl = vlmax;

        // Load data
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + offset, vl);
        vfloat32m1_t vy0 = __riscv_vle32_v_f32m1(y0 + offset, vl);
        vfloat32m1_t vy1 = __riscv_vle32_v_f32m1(y1 + offset, vl);
        vfloat32m1_t vy2 = __riscv_vle32_v_f32m1(y2 + offset, vl);
        vfloat32m1_t vy3 = __riscv_vle32_v_f32m1(y3 + offset, vl);

        // Calculate difference and square
        vfloat32m1_t vtmp0 = __riscv_vfsub_vv_f32m1(vx, vy0, vl);
        vfloat32m1_t vtmp1 = __riscv_vfsub_vv_f32m1(vx, vy1, vl);
        vfloat32m1_t vtmp2 = __riscv_vfsub_vv_f32m1(vx, vy2, vl);
        vfloat32m1_t vtmp3 = __riscv_vfsub_vv_f32m1(vx, vy3, vl);

        // Parallel FMACC operations
        vacc0 = __riscv_vfmacc_vv_f32m1_tu(vacc0, vtmp0, vtmp0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m1_tu(vacc1, vtmp1, vtmp1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m1_tu(vacc2, vtmp2, vtmp2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m1_tu(vacc3, vtmp3, vtmp3, vl);

        offset += vl;
        d -= vl;
    }

    // Handle remaining elements
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m1(d);
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + offset, vl);
        vfloat32m1_t vy0 = __riscv_vle32_v_f32m1(y0 + offset, vl);
        vfloat32m1_t vy1 = __riscv_vle32_v_f32m1(y1 + offset, vl);
        vfloat32m1_t vy2 = __riscv_vle32_v_f32m1(y2 + offset, vl);
        vfloat32m1_t vy3 = __riscv_vle32_v_f32m1(y3 + offset, vl);

        vfloat32m1_t vtmp0 = __riscv_vfsub_vv_f32m1(vx, vy0, vl);
        vfloat32m1_t vtmp1 = __riscv_vfsub_vv_f32m1(vx, vy1, vl);
        vfloat32m1_t vtmp2 = __riscv_vfsub_vv_f32m1(vx, vy2, vl);
        vfloat32m1_t vtmp3 = __riscv_vfsub_vv_f32m1(vx, vy3, vl);

        vacc0 = __riscv_vfmacc_vv_f32m1_tu(vacc0, vtmp0, vtmp0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m1_tu(vacc1, vtmp1, vtmp1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m1_tu(vacc2, vtmp2, vtmp2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m1_tu(vacc3, vtmp3, vtmp3, vl);

        offset += vl;
        d -= vl;
    }

    // Final reduction
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m1_f32m1(vacc0, sum_scalar, vlmax);
    dis0 = __riscv_vfmv_f_s_f32m1_f32(sum_scalar);

    sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m1_f32m1(vacc1, sum_scalar, vlmax);
    dis1 = __riscv_vfmv_f_s_f32m1_f32(sum_scalar);

    sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m1_f32m1(vacc2, sum_scalar, vlmax);
    dis2 = __riscv_vfmv_f_s_f32m1_f32(sum_scalar);

    sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m1_f32m1(vacc3, sum_scalar, vlmax);
    dis3 = __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

int32_t
ivec_inner_product_rvv(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    size_t vl;

    for (size_t i = 0; i < d;) {
        vl = __riscv_vsetvl_e8m1(d - i);

        // Load int8 vectors
        vint8m1_t vx = __riscv_vle8_v_i8m1(x + i, vl);
        vint8m1_t vy = __riscv_vle8_v_i8m1(y + i, vl);

        // Use widening multiplication directly (int8 * int8 -> int16)
        vint16m2_t vmul = __riscv_vwmul_vv_i16m2(vx, vy, vl);

        // Extend to int32 and accumulate
        vint32m4_t vmul_ext = __riscv_vsext_vf2_i32m4(vmul, vl);
        vint32m1_t vsum = __riscv_vredsum_vs_i32m4_i32m1(vmul_ext, __riscv_vmv_s_x_i32m1(0, 1), vl);

        res += __riscv_vmv_x_s_i32m1_i32(vsum);
        i += vl;
    }

    return res;
}

int32_t
ivec_L2sqr_rvv(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    size_t vl;

    for (size_t i = 0; i < d;) {
        vl = __riscv_vsetvl_e8m1(d - i);

        // Load int8 vectors
        vint8m1_t vx = __riscv_vle8_v_i8m1(x + i, vl);
        vint8m1_t vy = __riscv_vle8_v_i8m1(y + i, vl);

        // Use widening subtraction (int8 - int8 -> int16)
        vint16m2_t vdiff = __riscv_vwsub_vv_i16m2(vx, vy, vl);

        // Use widening multiplication directly from int16 to int32
        vint32m4_t vsqr = __riscv_vwmul_vv_i32m4(vdiff, vdiff, vl);

        // Vector reduction sum
        vint32m1_t vsum = __riscv_vredsum_vs_i32m4_i32m1(vsqr, __riscv_vmv_s_x_i32m1(0, 1), vl);

        res += __riscv_vmv_x_s_i32m1_i32(vsum);
        i += vl;
    }

    return res;
}

float
int8_vec_inner_product_rvv(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    size_t vl;
    for (size_t i = 0; i < d;) {
        vl = __riscv_vsetvl_e8m1(d - i);
        vint8m1_t vx = __riscv_vle8_v_i8m1(x + i, vl);
        vint8m1_t vy = __riscv_vle8_v_i8m1(y + i, vl);
        vint16m2_t vmul = __riscv_vwmul_vv_i16m2(vx, vy, vl);
        vint32m4_t vmul_ext = __riscv_vsext_vf2_i32m4(vmul, vl);
        vint32m1_t vsum = __riscv_vredsum_vs_i32m4_i32m1(vmul_ext, __riscv_vmv_s_x_i32m1(0, 1), vl);
        res += __riscv_vmv_x_s_i32m1_i32(vsum);
        i += vl;
    }
    return (float)res;
}

float
int8_vec_L2sqr_rvv(const int8_t* x, const int8_t* y, size_t d) {
    int32_t res = 0;
    size_t vl;
    for (size_t i = 0; i < d;) {
        vl = __riscv_vsetvl_e8m1(d - i);
        vint8m1_t vx = __riscv_vle8_v_i8m1(x + i, vl);
        vint8m1_t vy = __riscv_vle8_v_i8m1(y + i, vl);
        vint16m2_t vdiff = __riscv_vwsub_vv_i16m2(vx, vy, vl);
        vint32m4_t vsqr = __riscv_vwmul_vv_i32m4(vdiff, vdiff, vl);
        vint32m1_t vsum = __riscv_vredsum_vs_i32m4_i32m1(vsqr, __riscv_vmv_s_x_i32m1(0, 1), vl);
        res += __riscv_vmv_x_s_i32m1_i32(vsum);
        i += vl;
    }
    return (float)res;
}

float
int8_vec_norm_L2sqr_rvv(const int8_t* x, size_t d) {
    int32_t res = 0;
    size_t vl;
    for (size_t i = 0; i < d;) {
        vl = __riscv_vsetvl_e8m1(d - i);
        vint8m1_t vx = __riscv_vle8_v_i8m1(x + i, vl);
        vint16m2_t vx_ext = __riscv_vwadd_vx_i16m2(vx, 0, vl);  // sign-extend int8->int16
        vint32m4_t vsqr = __riscv_vwmul_vv_i32m4(vx_ext, vx_ext, vl);
        vint32m1_t vsum = __riscv_vredsum_vs_i32m4_i32m1(vsqr, __riscv_vmv_s_x_i32m1(0, 1), vl);
        res += __riscv_vmv_x_s_i32m1_i32(vsum);
        i += vl;
    }
    return (float)res;
}

void
int8_vec_inner_product_batch_4_rvv(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                   const int8_t* y3, size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    size_t vl;
    for (size_t i = 0; i < d;) {
        vl = __riscv_vsetvl_e8m1(d - i);
        vint8m1_t vx = __riscv_vle8_v_i8m1(x + i, vl);
        vint8m1_t vy0 = __riscv_vle8_v_i8m1(y0 + i, vl);
        vint8m1_t vy1 = __riscv_vle8_v_i8m1(y1 + i, vl);
        vint8m1_t vy2 = __riscv_vle8_v_i8m1(y2 + i, vl);
        vint8m1_t vy3 = __riscv_vle8_v_i8m1(y3 + i, vl);
        vint16m2_t vmul0 = __riscv_vwmul_vv_i16m2(vx, vy0, vl);
        vint16m2_t vmul1 = __riscv_vwmul_vv_i16m2(vx, vy1, vl);
        vint16m2_t vmul2 = __riscv_vwmul_vv_i16m2(vx, vy2, vl);
        vint16m2_t vmul3 = __riscv_vwmul_vv_i16m2(vx, vy3, vl);
        vint32m4_t vmul0_ext = __riscv_vsext_vf2_i32m4(vmul0, vl);
        vint32m4_t vmul1_ext = __riscv_vsext_vf2_i32m4(vmul1, vl);
        vint32m4_t vmul2_ext = __riscv_vsext_vf2_i32m4(vmul2, vl);
        vint32m4_t vmul3_ext = __riscv_vsext_vf2_i32m4(vmul3, vl);
        vint32m1_t vsum0 = __riscv_vredsum_vs_i32m4_i32m1(vmul0_ext, __riscv_vmv_s_x_i32m1(0, 1), vl);
        vint32m1_t vsum1 = __riscv_vredsum_vs_i32m4_i32m1(vmul1_ext, __riscv_vmv_s_x_i32m1(0, 1), vl);
        vint32m1_t vsum2 = __riscv_vredsum_vs_i32m4_i32m1(vmul2_ext, __riscv_vmv_s_x_i32m1(0, 1), vl);
        vint32m1_t vsum3 = __riscv_vredsum_vs_i32m4_i32m1(vmul3_ext, __riscv_vmv_s_x_i32m1(0, 1), vl);
        acc0 += __riscv_vmv_x_s_i32m1_i32(vsum0);
        acc1 += __riscv_vmv_x_s_i32m1_i32(vsum1);
        acc2 += __riscv_vmv_x_s_i32m1_i32(vsum2);
        acc3 += __riscv_vmv_x_s_i32m1_i32(vsum3);
        i += vl;
    }
    dis0 = (float)acc0;
    dis1 = (float)acc1;
    dis2 = (float)acc2;
    dis3 = (float)acc3;
}

void
int8_vec_L2sqr_batch_4_rvv(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                           size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    size_t vl;
    for (size_t i = 0; i < d;) {
        vl = __riscv_vsetvl_e8m1(d - i);
        vint8m1_t vx = __riscv_vle8_v_i8m1(x + i, vl);
        vint8m1_t vy0 = __riscv_vle8_v_i8m1(y0 + i, vl);
        vint8m1_t vy1 = __riscv_vle8_v_i8m1(y1 + i, vl);
        vint8m1_t vy2 = __riscv_vle8_v_i8m1(y2 + i, vl);
        vint8m1_t vy3 = __riscv_vle8_v_i8m1(y3 + i, vl);
        vint16m2_t vdiff0 = __riscv_vwsub_vv_i16m2(vx, vy0, vl);
        vint16m2_t vdiff1 = __riscv_vwsub_vv_i16m2(vx, vy1, vl);
        vint16m2_t vdiff2 = __riscv_vwsub_vv_i16m2(vx, vy2, vl);
        vint16m2_t vdiff3 = __riscv_vwsub_vv_i16m2(vx, vy3, vl);
        vint32m4_t vsqr0 = __riscv_vwmul_vv_i32m4(vdiff0, vdiff0, vl);
        vint32m4_t vsqr1 = __riscv_vwmul_vv_i32m4(vdiff1, vdiff1, vl);
        vint32m4_t vsqr2 = __riscv_vwmul_vv_i32m4(vdiff2, vdiff2, vl);
        vint32m4_t vsqr3 = __riscv_vwmul_vv_i32m4(vdiff3, vdiff3, vl);
        vint32m1_t vsum0 = __riscv_vredsum_vs_i32m4_i32m1(vsqr0, __riscv_vmv_s_x_i32m1(0, 1), vl);
        vint32m1_t vsum1 = __riscv_vredsum_vs_i32m4_i32m1(vsqr1, __riscv_vmv_s_x_i32m1(0, 1), vl);
        vint32m1_t vsum2 = __riscv_vredsum_vs_i32m4_i32m1(vsqr2, __riscv_vmv_s_x_i32m1(0, 1), vl);
        vint32m1_t vsum3 = __riscv_vredsum_vs_i32m4_i32m1(vsqr3, __riscv_vmv_s_x_i32m1(0, 1), vl);
        acc0 += __riscv_vmv_x_s_i32m1_i32(vsum0);
        acc1 += __riscv_vmv_x_s_i32m1_i32(vsum1);
        acc2 += __riscv_vmv_x_s_i32m1_i32(vsum2);
        acc3 += __riscv_vmv_x_s_i32m1_i32(vsum3);
        i += vl;
    }
    dis0 = (float)acc0;
    dis1 = (float)acc1;
    dis2 = (float)acc2;
    dis3 = (float)acc3;
}

///////////////////////////////////////////////////////////////////////////////
// fp16

float
fp16_vec_inner_product_rvv(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    const size_t original_d = d;
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

    const _Float16* x_ptr = reinterpret_cast<const _Float16*>(x);
    const _Float16* y_ptr = reinterpret_cast<const _Float16*>(y);

    for (size_t vl; (vl = __riscv_vsetvl_e16m4(d)) > 0; d -= vl) {
        vfloat16m4_t vx = __riscv_vle16_v_f16m4(x_ptr, vl);
        vfloat16m4_t vy = __riscv_vle16_v_f16m4(y_ptr, vl);

        vfloat32m8_t vfx = __riscv_vfwcvt_f_f_v_f32m8(vx, vl);
        vfloat32m8_t vfy = __riscv_vfwcvt_f_f_v_f32m8(vy, vl);

        acc = __riscv_vfmacc_vv_f32m8(acc, vfx, vfy, vl);

        x_ptr += vl;
        y_ptr += vl;
    }

    vfloat32m1_t scalar_sum_vec = __riscv_vfredusum_vs_f32m8_f32m1(acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), original_d);

    return __riscv_vfmv_f_s_f32m1_f32(scalar_sum_vec);
}

float
fp16_vec_L2sqr_rvv(const knowhere::fp16* x, const knowhere::fp16* y, size_t d) {
    const size_t original_d = d;
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
    const _Float16* x_ptr = reinterpret_cast<const _Float16*>(x);
    const _Float16* y_ptr = reinterpret_cast<const _Float16*>(y);

    for (size_t vl; (vl = __riscv_vsetvl_e16m2(d)) > 0; d -= vl) {
        vfloat16m2_t vx = __riscv_vle16_v_f16m2(x_ptr, vl);
        vfloat16m2_t vy = __riscv_vle16_v_f16m2(y_ptr, vl);

        vfloat32m4_t vfx = __riscv_vfwcvt_f_f_v_f32m4(vx, vl);
        vfloat32m4_t vfy = __riscv_vfwcvt_f_f_v_f32m4(vy, vl);

        vfloat32m4_t v_diff = __riscv_vfsub_vv_f32m4(vfx, vfy, vl);

        acc = __riscv_vfmacc_vv_f32m4(acc, v_diff, v_diff, vl);
        x_ptr += vl;
        y_ptr += vl;
    }

    vfloat32m1_t scalar_sum_vec = __riscv_vfredusum_vs_f32m4_f32m1(acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), original_d);

    return __riscv_vfmv_f_s_f32m1_f32(scalar_sum_vec);
}

float
fp16_vec_norm_L2sqr_rvv(const knowhere::fp16* x, size_t d) {
    const size_t original_d = d;
    vfloat32m8_t acc = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());
    const _Float16* x_ptr = reinterpret_cast<const _Float16*>(x);

    for (size_t vl; (vl = __riscv_vsetvl_e16m4(d)) > 0; d -= vl) {
        vfloat16m4_t vx = __riscv_vle16_v_f16m4(x_ptr, vl);

        vfloat32m8_t vfx = __riscv_vfwcvt_f_f_v_f32m8(vx, vl);

        acc = __riscv_vfmacc_vv_f32m8(acc, vfx, vfx, vl);

        x_ptr += vl;
    }

    vfloat32m1_t scalar_sum_vec = __riscv_vfredusum_vs_f32m8_f32m1(acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), original_d);

    return __riscv_vfmv_f_s_f32m1_f32(scalar_sum_vec);
}

///////////////////////////////////////////////////////////////////////////////
// bf16
float
bf16_vec_inner_product_rvv(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    const size_t original_d = d;
    const uint16_t* x_ptr = reinterpret_cast<const uint16_t*>(x);
    const uint16_t* y_ptr = reinterpret_cast<const uint16_t*>(y);

    size_t vlmax_m8 = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax_m8);

    for (size_t vl; (vl = __riscv_vsetvl_e16m4(d)) > 0; d -= vl) {
        vuint16m4_t v_x_u16 = __riscv_vle16_v_u16m4(x_ptr, vl);
        vuint16m4_t v_y_u16 = __riscv_vle16_v_u16m4(y_ptr, vl);

        vuint32m8_t v_x_u32 = __riscv_vwaddu_vx_u32m8(v_x_u16, 0, vl);
        vuint32m8_t v_y_u32 = __riscv_vwaddu_vx_u32m8(v_y_u16, 0, vl);

        vuint32m8_t v_x_shifted = __riscv_vsll_vx_u32m8(v_x_u32, 16, vl);
        vuint32m8_t v_y_shifted = __riscv_vsll_vx_u32m8(v_y_u32, 16, vl);

        vfloat32m8_t v_x_w = __riscv_vreinterpret_v_u32m8_f32m8(v_x_shifted);
        vfloat32m8_t v_y_w = __riscv_vreinterpret_v_u32m8_f32m8(v_y_shifted);

        v_acc = __riscv_vfmacc_vv_f32m8(v_acc, v_x_w, v_y_w, vl);

        x_ptr += vl;
        y_ptr += vl;
    }
    vfloat32m1_t v_scalar_sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    size_t vl_for_reduction = __riscv_vsetvl_e32m8(original_d);
    v_scalar_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_acc, v_scalar_sum, vl_for_reduction);
    return __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);
}

float
bf16_vec_L2sqr_rvv(const knowhere::bf16* x, const knowhere::bf16* y, size_t d) {
    const size_t original_d = d;
    const uint16_t* x_ptr = reinterpret_cast<const uint16_t*>(x);
    const uint16_t* y_ptr = reinterpret_cast<const uint16_t*>(y);

    size_t vlmax_m4 = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t v_acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax_m4);

    for (size_t vl; (vl = __riscv_vsetvl_e16m2(d)) > 0; d -= vl) {
        vuint16m2_t v_x_u16 = __riscv_vle16_v_u16m2(x_ptr, vl);
        vuint16m2_t v_y_u16 = __riscv_vle16_v_u16m2(y_ptr, vl);

        vuint32m4_t v_x_u32 = __riscv_vwaddu_vx_u32m4(v_x_u16, 0, vl);
        vuint32m4_t v_y_u32 = __riscv_vwaddu_vx_u32m4(v_y_u16, 0, vl);

        vuint32m4_t v_x_shifted = __riscv_vsll_vx_u32m4(v_x_u32, 16, vl);
        vuint32m4_t v_y_shifted = __riscv_vsll_vx_u32m4(v_y_u32, 16, vl);

        vfloat32m4_t v_x_w = __riscv_vreinterpret_v_u32m4_f32m4(v_x_shifted);
        vfloat32m4_t v_y_w = __riscv_vreinterpret_v_u32m4_f32m4(v_y_shifted);

        vfloat32m4_t v_diff = __riscv_vfsub_vv_f32m4(v_x_w, v_y_w, vl);

        v_acc = __riscv_vfmacc_vv_f32m4(v_acc, v_diff, v_diff, vl);

        x_ptr += vl;
        y_ptr += vl;
    }

    vfloat32m1_t v_scalar_sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    size_t vl_for_reduction = __riscv_vsetvl_e32m4(original_d);
    v_scalar_sum = __riscv_vfredusum_vs_f32m4_f32m1(v_acc, v_scalar_sum, vl_for_reduction);
    return __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);
}

float
bf16_vec_norm_L2sqr_rvv(const knowhere::bf16* x, size_t d) {
    const size_t original_d = d;
    const uint16_t* x_ptr = reinterpret_cast<const uint16_t*>(x);

    size_t vlmax_m8 = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax_m8);

    for (size_t vl; (vl = __riscv_vsetvl_e16m4(d)) > 0; d -= vl) {
        vuint16m4_t v_x_u16 = __riscv_vle16_v_u16m4(x_ptr, vl);

        vuint32m8_t v_x_u32 = __riscv_vwaddu_vx_u32m8(v_x_u16, 0, vl);
        vuint32m8_t v_x_shifted = __riscv_vsll_vx_u32m8(v_x_u32, 16, vl);
        vfloat32m8_t v_x_fp32 = __riscv_vreinterpret_v_u32m8_f32m8(v_x_shifted);

        v_acc = __riscv_vfmacc_vv_f32m8(v_acc, v_x_fp32, v_x_fp32, vl);
        x_ptr += vl;
    }
    vfloat32m1_t v_scalar_sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    size_t vl_for_reduction = __riscv_vsetvl_e32m8(original_d);
    v_scalar_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_acc, v_scalar_sum, vl_for_reduction);
    return __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);
}

void
bf16_vec_inner_product_batch_4_rvv(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                   const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3) {
    size_t temp_d = d;
    const uint16_t* x_ptr = reinterpret_cast<const uint16_t*>(x);
    const uint16_t* y0_ptr = reinterpret_cast<const uint16_t*>(y0);
    const uint16_t* y1_ptr = reinterpret_cast<const uint16_t*>(y1);
    const uint16_t* y2_ptr = reinterpret_cast<const uint16_t*>(y2);
    const uint16_t* y3_ptr = reinterpret_cast<const uint16_t*>(y3);

    vfloat32m4_t vacc0 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vacc1 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vacc2 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());
    vfloat32m4_t vacc3 = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvlmax_e32m4());

    for (size_t vl; (vl = __riscv_vsetvl_e16m2(temp_d)) > 0; temp_d -= vl) {
        vuint16m2_t vx_u16 = __riscv_vle16_v_u16m2(x_ptr, vl);
        vfloat32m4_t vx_fp32 =
            __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vsll_vx_u32m4(__riscv_vwaddu_vx_u32m4(vx_u16, 0, vl), 16, vl));

        // --- Process y0 ---
        vuint16m2_t vy0_u16 = __riscv_vle16_v_u16m2(y0_ptr, vl);
        vacc0 = __riscv_vfmacc_vv_f32m4(
            vacc0, vx_fp32,
            __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vsll_vx_u32m4(__riscv_vwaddu_vx_u32m4(vy0_u16, 0, vl), 16, vl)),
            vl);

        // --- Process y1 ---
        vuint16m2_t vy1_u16 = __riscv_vle16_v_u16m2(y1_ptr, vl);
        vacc1 = __riscv_vfmacc_vv_f32m4(
            vacc1, vx_fp32,
            __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vsll_vx_u32m4(__riscv_vwaddu_vx_u32m4(vy1_u16, 0, vl), 16, vl)),
            vl);

        // --- Process y2 ---
        vuint16m2_t vy2_u16 = __riscv_vle16_v_u16m2(y2_ptr, vl);
        vacc2 = __riscv_vfmacc_vv_f32m4(
            vacc2, vx_fp32,
            __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vsll_vx_u32m4(__riscv_vwaddu_vx_u32m4(vy2_u16, 0, vl), 16, vl)),
            vl);

        // --- Process y3 ---
        vuint16m2_t vy3_u16 = __riscv_vle16_v_u16m2(y3_ptr, vl);
        vacc3 = __riscv_vfmacc_vv_f32m4(
            vacc3, vx_fp32,
            __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vsll_vx_u32m4(__riscv_vwaddu_vx_u32m4(vy3_u16, 0, vl), 16, vl)),
            vl);

        x_ptr += vl;
        y0_ptr += vl;
        y1_ptr += vl;
        y2_ptr += vl;
        y3_ptr += vl;
    }

    vfloat32m1_t v_scalar_sum_init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    dis0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vacc0, v_scalar_sum_init, d));
    dis1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vacc1, v_scalar_sum_init, d));
    dis2 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vacc2, v_scalar_sum_init, d));
    dis3 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vacc3, v_scalar_sum_init, d));
}

void
bf16_vec_L2sqr_batch_4_rvv(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                           const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3) {
    size_t temp_d = d;
    const uint16_t* x_ptr = reinterpret_cast<const uint16_t*>(x);
    const uint16_t* y0_ptr = reinterpret_cast<const uint16_t*>(y0);
    const uint16_t* y1_ptr = reinterpret_cast<const uint16_t*>(y1);
    const uint16_t* y2_ptr = reinterpret_cast<const uint16_t*>(y2);
    const uint16_t* y3_ptr = reinterpret_cast<const uint16_t*>(y3);

    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, __riscv_vsetvlmax_e32m2());
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, __riscv_vsetvlmax_e32m2());
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, __riscv_vsetvlmax_e32m2());
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, __riscv_vsetvlmax_e32m2());

    for (size_t vl; (vl = __riscv_vsetvl_e16m1(temp_d)) > 0; temp_d -= vl) {
        vfloat32m2_t vx_fp32 = __riscv_vreinterpret_v_u32m2_f32m2(
            __riscv_vsll_vx_u32m2(__riscv_vwaddu_vx_u32m2(__riscv_vle16_v_u16m1(x_ptr, vl), 0, vl), 16, vl));

        vuint16m1_t vy0_u16 = __riscv_vle16_v_u16m1(y0_ptr, vl);
        vfloat32m2_t vdiff0 = __riscv_vfsub_vv_f32m2(
            vx_fp32,
            __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(__riscv_vwaddu_vx_u32m2(vy0_u16, 0, vl), 16, vl)),
            vl);
        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, vdiff0, vdiff0, vl);

        vuint16m1_t vy1_u16 = __riscv_vle16_v_u16m1(y1_ptr, vl);
        vfloat32m2_t vdiff1 = __riscv_vfsub_vv_f32m2(
            vx_fp32,
            __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(__riscv_vwaddu_vx_u32m2(vy1_u16, 0, vl), 16, vl)),
            vl);
        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, vdiff1, vdiff1, vl);

        vuint16m1_t vy2_u16 = __riscv_vle16_v_u16m1(y2_ptr, vl);
        vfloat32m2_t vdiff2 = __riscv_vfsub_vv_f32m2(
            vx_fp32,
            __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(__riscv_vwaddu_vx_u32m2(vy2_u16, 0, vl), 16, vl)),
            vl);
        vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, vdiff2, vdiff2, vl);

        vuint16m1_t vy3_u16 = __riscv_vle16_v_u16m1(y3_ptr, vl);
        vfloat32m2_t vdiff3 = __riscv_vfsub_vv_f32m2(
            vx_fp32,
            __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsll_vx_u32m2(__riscv_vwaddu_vx_u32m2(vy3_u16, 0, vl), 16, vl)),
            vl);
        vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, vdiff3, vdiff3, vl);

        x_ptr += vl;
        y0_ptr += vl;
        y1_ptr += vl;
        y2_ptr += vl;
        y3_ptr += vl;
    }

    vfloat32m1_t v_scalar_sum_init = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    dis0 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vacc0, v_scalar_sum_init, d));
    dis1 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vacc1, v_scalar_sum_init, d));
    dis2 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vacc2, v_scalar_sum_init, d));
    dis3 = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(vacc3, v_scalar_sum_init, d));
}

// bf16 patch functions
float
fvec_inner_product_bf16_patch_rvv(const float* x, const float* y, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    size_t offset = 0;

    // 4-way unrolled loop
    while (d >= 4 * vlmax) {
        size_t vl = vlmax;

        // Load data
        vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy0 = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x + offset + vl, vl);
        vfloat32m2_t vy1 = __riscv_vle32_v_f32m2(y + offset + vl, vl);
        vfloat32m2_t vx2 = __riscv_vle32_v_f32m2(x + offset + 2 * vl, vl);
        vfloat32m2_t vy2 = __riscv_vle32_v_f32m2(y + offset + 2 * vl, vl);
        vfloat32m2_t vx3 = __riscv_vle32_v_f32m2(x + offset + 3 * vl, vl);
        vfloat32m2_t vy3 = __riscv_vle32_v_f32m2(y + offset + 3 * vl, vl);

        // Convert y vectors to bf16
        vy0 = bf16_float_rvv(vy0, vl);
        vy1 = bf16_float_rvv(vy1, vl);
        vy2 = bf16_float_rvv(vy2, vl);
        vy3 = bf16_float_rvv(vy3, vl);

        // Parallel FMACC operations
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vx0, vy0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m2_tu(vacc1, vx1, vy1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m2_tu(vacc2, vx2, vy2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m2_tu(vacc3, vx3, vy3, vl);

        offset += 4 * vl;
        d -= 4 * vl;
    }

    // Merge accumulators
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc1, vlmax);
    vacc2 = __riscv_vfadd_vv_f32m2(vacc2, vacc3, vlmax);
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc2, vlmax);

    // Handle remaining elements
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m2(d);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy = __riscv_vle32_v_f32m2(y + offset, vl);
        vy = bf16_float_rvv(vy, vl);
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vx, vy, vl);

        offset += vl;
        d -= vl;
    }

    // Final reduction
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);

    return __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

float
fvec_L2sqr_bf16_patch_rvv(const float* x, const float* y, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    size_t offset = 0;

    // 4-way unrolled loop
    while (d >= 4 * vlmax) {
        size_t vl = vlmax;

        // Load data
        vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy0 = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x + offset + vl, vl);
        vfloat32m2_t vy1 = __riscv_vle32_v_f32m2(y + offset + vl, vl);
        vfloat32m2_t vx2 = __riscv_vle32_v_f32m2(x + offset + 2 * vl, vl);
        vfloat32m2_t vy2 = __riscv_vle32_v_f32m2(y + offset + 2 * vl, vl);
        vfloat32m2_t vx3 = __riscv_vle32_v_f32m2(x + offset + 3 * vl, vl);
        vfloat32m2_t vy3 = __riscv_vle32_v_f32m2(y + offset + 3 * vl, vl);

        // Convert y vectors to bf16
        vy0 = bf16_float_rvv(vy0, vl);
        vy1 = bf16_float_rvv(vy1, vl);
        vy2 = bf16_float_rvv(vy2, vl);
        vy3 = bf16_float_rvv(vy3, vl);

        // Calculate difference and square
        vfloat32m2_t vtmp0 = __riscv_vfsub_vv_f32m2(vx0, vy0, vl);
        vfloat32m2_t vtmp1 = __riscv_vfsub_vv_f32m2(vx1, vy1, vl);
        vfloat32m2_t vtmp2 = __riscv_vfsub_vv_f32m2(vx2, vy2, vl);
        vfloat32m2_t vtmp3 = __riscv_vfsub_vv_f32m2(vx3, vy3, vl);

        // Parallel FMACC operations
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vtmp0, vtmp0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m2_tu(vacc1, vtmp1, vtmp1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m2_tu(vacc2, vtmp2, vtmp2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m2_tu(vacc3, vtmp3, vtmp3, vl);

        offset += 4 * vl;
        d -= 4 * vl;
    }

    // Merge accumulators
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc1, vlmax);
    vacc2 = __riscv_vfadd_vv_f32m2(vacc2, vacc3, vlmax);
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc2, vlmax);

    // Handle remaining elements
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m2(d);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy = __riscv_vle32_v_f32m2(y + offset, vl);
        vy = bf16_float_rvv(vy, vl);
        vfloat32m2_t vtmp = __riscv_vfsub_vv_f32m2(vx, vy, vl);
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vtmp, vtmp, vl);

        offset += vl;
        d -= vl;
    }

    // Final reduction
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);

    return __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

}  // namespace faiss

#endif
