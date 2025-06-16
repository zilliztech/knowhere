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
#pragma GCC optimize("O3,fast-math,inline")
#include "distances_rvv.h"

#include <math.h>
#include <riscv_vector.h>

namespace faiss {

// =================== float distances ===================
float
fvec_inner_product_rvv(const float* x, const float* y, size_t d) {
    size_t vlmax = __riscv_vsetvlmax_e32m2();  // 使用m2以支持4路并行

    // 4个累积器
    vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

    size_t offset = 0;

    // 4路展开循环
    while (d >= 4 * vlmax) {
        size_t vl = vlmax;

        // 预取数据 (如果支持)
        // __builtin_prefetch(x + offset + 4 * vl, 0, 3);
        // __builtin_prefetch(y + offset + 4 * vl, 0, 3);

        vfloat32m2_t vx0 = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy0 = __riscv_vle32_v_f32m2(y + offset, vl);
        vfloat32m2_t vx1 = __riscv_vle32_v_f32m2(x + offset + vl, vl);
        vfloat32m2_t vy1 = __riscv_vle32_v_f32m2(y + offset + vl, vl);
        vfloat32m2_t vx2 = __riscv_vle32_v_f32m2(x + offset + 2 * vl, vl);
        vfloat32m2_t vy2 = __riscv_vle32_v_f32m2(y + offset + 2 * vl, vl);
        vfloat32m2_t vx3 = __riscv_vle32_v_f32m2(x + offset + 3 * vl, vl);
        vfloat32m2_t vy3 = __riscv_vle32_v_f32m2(y + offset + 3 * vl, vl);

        // 并行FMACC操作
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vx0, vy0, vl);
        vacc1 = __riscv_vfmacc_vv_f32m2_tu(vacc1, vx1, vy1, vl);
        vacc2 = __riscv_vfmacc_vv_f32m2_tu(vacc2, vx2, vy2, vl);
        vacc3 = __riscv_vfmacc_vv_f32m2_tu(vacc3, vx3, vy3, vl);

        offset += 4 * vl;
        d -= 4 * vl;
    }

    // 合并累积器
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc1, vlmax);
    vacc2 = __riscv_vfadd_vv_f32m2(vacc2, vacc3, vlmax);
    vacc0 = __riscv_vfadd_vv_f32m2(vacc0, vacc2, vlmax);

    // 处理剩余元素
    while (d > 0) {
        size_t vl = __riscv_vsetvl_e32m2(d);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + offset, vl);
        vfloat32m2_t vy = __riscv_vle32_v_f32m2(y + offset, vl);
        vacc0 = __riscv_vfmacc_vv_f32m2_tu(vacc0, vx, vy, vl);

        offset += vl;
        d -= vl;
    }

    // 最终归约
    vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum_scalar = __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);

    return __riscv_vfmv_f_s_f32m1_f32(sum_scalar);
}

}  // namespace faiss

#endif
