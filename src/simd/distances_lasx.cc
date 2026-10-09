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

#include "distances_lasx.h"

#include <lasxintrin.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace faiss::cppcontrib::knowhere {
namespace {

constexpr size_t kLanes = 8;

inline __m256
load_f32(const float* ptr) {
    return std::bit_cast<__m256>(__lasx_xvld(const_cast<float*>(ptr), 0));
}

inline void
store_f32(float* ptr, __m256 value) {
    __lasx_xvst(std::bit_cast<__m256i>(value), ptr, 0);
}

inline __m256
zero_f32() {
    return std::bit_cast<__m256>(__lasx_xvldi(0));
}

inline __m256
broadcast_f32(float value) {
    return std::bit_cast<__m256>(__lasx_xvreplgr2vr_w(std::bit_cast<int32_t>(value)));
}

inline __m256
abs_f32(__m256 value) {
    return std::bit_cast<__m256>(__lasx_xvbitclri_w(std::bit_cast<__m256i>(value), 31));
}

inline float
reduce_sum(__m256 value) {
    alignas(32) float lanes[kLanes];
    store_f32(lanes, value);
    float result = 0.0f;
    for (float lane : lanes) {
        result += lane;
    }
    return result;
}

inline float
reduce_max(__m256 value) {
    alignas(32) float lanes[kLanes];
    store_f32(lanes, value);
    float result = 0.0f;
    for (float lane : lanes) {
        result = std::fmax(result, lane);
    }
    return result;
}

}  // namespace

float
fvec_inner_product_lasx(const float* x, const float* y, size_t d) {
    __m256 sum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        sum = __lasx_xvfmadd_s(load_f32(x + i), load_f32(y + i), sum);
    }

    float result = reduce_sum(sum);
    for (; i < d; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

float
fvec_L2sqr_lasx(const float* x, const float* y, size_t d) {
    __m256 sum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m256 diff = __lasx_xvfsub_s(load_f32(x + i), load_f32(y + i));
        sum = __lasx_xvfmadd_s(diff, diff, sum);
    }

    float result = reduce_sum(sum);
    for (; i < d; ++i) {
        const float diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

float
fvec_L1_lasx(const float* x, const float* y, size_t d) {
    __m256 sum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m256 diff = __lasx_xvfsub_s(load_f32(x + i), load_f32(y + i));
        sum = __lasx_xvfadd_s(sum, abs_f32(diff));
    }

    float result = reduce_sum(sum);
    for (; i < d; ++i) {
        result += std::fabs(x[i] - y[i]);
    }
    return result;
}

float
fvec_Linf_lasx(const float* x, const float* y, size_t d) {
    __m256 maximum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m256 diff = __lasx_xvfsub_s(load_f32(x + i), load_f32(y + i));
        maximum = __lasx_xvfmax_s(maximum, abs_f32(diff));
    }

    float result = reduce_max(maximum);
    for (; i < d; ++i) {
        result = std::fmax(result, std::fabs(x[i] - y[i]));
    }
    return result;
}

float
fvec_norm_L2sqr_lasx(const float* x, size_t d) {
    __m256 sum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m256 value = load_f32(x + i);
        sum = __lasx_xvfmadd_s(value, value, sum);
    }

    float result = reduce_sum(sum);
    for (; i < d; ++i) {
        result += x[i] * x[i];
    }
    return result;
}

void
fvec_L2sqr_ny_lasx(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; ++i) {
        dis[i] = fvec_L2sqr_lasx(x, y + i * d, d);
    }
}

void
fvec_inner_products_ny_lasx(float* ip, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; ++i) {
        ip[i] = fvec_inner_product_lasx(x, y + i * d, d);
    }
}

void
fvec_L2sqr_ny_transposed_lasx(float* dis, const float* x, const float* y, const float* y_sqlen, size_t d,
                              size_t d_offset, size_t ny) {
    const float x_sqlen = fvec_norm_L2sqr_lasx(x, d);
    const __m256 x_sqlen_v = broadcast_f32(x_sqlen);
    size_t i = 0;
    for (; i + kLanes <= ny; i += kLanes) {
        __m256 dot_product = zero_f32();
        for (size_t j = 0; j < d; ++j) {
            dot_product = __lasx_xvfmadd_s(broadcast_f32(x[j]), load_f32(y + i + j * d_offset), dot_product);
        }
        const __m256 squared_lengths = __lasx_xvfadd_s(x_sqlen_v, load_f32(y_sqlen + i));
        store_f32(dis + i, __lasx_xvfsub_s(squared_lengths, __lasx_xvfadd_s(dot_product, dot_product)));
    }

    for (; i < ny; ++i) {
        float dot_product = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            dot_product += x[j] * y[i + j * d_offset];
        }
        dis[i] = x_sqlen + y_sqlen[i] - 2.0f * dot_product;
    }
}

size_t
fvec_L2sqr_ny_nearest_lasx(float* distances_tmp_buffer, const float* x, const float* y, size_t d, size_t ny) {
    fvec_L2sqr_ny_lasx(distances_tmp_buffer, x, y, d, ny);
    size_t nearest_idx = 0;
    float min_distance = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < ny; ++i) {
        if (distances_tmp_buffer[i] < min_distance) {
            min_distance = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }
    return nearest_idx;
}

size_t
fvec_L2sqr_ny_nearest_y_transposed_lasx(float* distances_tmp_buffer, const float* x, const float* y,
                                        const float* y_sqlen, size_t d, size_t d_offset, size_t ny) {
    fvec_L2sqr_ny_transposed_lasx(distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
    size_t nearest_idx = 0;
    float min_distance = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < ny; ++i) {
        if (distances_tmp_buffer[i] < min_distance) {
            min_distance = distances_tmp_buffer[i];
            nearest_idx = i;
        }
    }
    return nearest_idx;
}

void
fvec_madd_lasx(size_t n, const float* a, float bf, const float* b, float* c) {
    const __m256 bf_v = broadcast_f32(bf);
    size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        store_f32(c + i, __lasx_xvfmadd_s(bf_v, load_f32(b + i), load_f32(a + i)));
    }
    for (; i < n; ++i) {
        c[i] = a[i] + bf * b[i];
    }
}

int
fvec_madd_and_argmin_lasx(size_t n, const float* a, float bf, const float* b, float* c) {
    fvec_madd_lasx(n, a, bf, b, c);
    float minimum = 1e20f;
    int minimum_index = -1;
    for (size_t i = 0; i < n; ++i) {
        if (c[i] < minimum) {
            minimum = c[i];
            minimum_index = static_cast<int>(i);
        }
    }
    return minimum_index;
}

void
fvec_inner_product_batch_4_lasx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    __m256 sum0 = zero_f32();
    __m256 sum1 = zero_f32();
    __m256 sum2 = zero_f32();
    __m256 sum3 = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m256 x_v = load_f32(x + i);
        sum0 = __lasx_xvfmadd_s(x_v, load_f32(y0 + i), sum0);
        sum1 = __lasx_xvfmadd_s(x_v, load_f32(y1 + i), sum1);
        sum2 = __lasx_xvfmadd_s(x_v, load_f32(y2 + i), sum2);
        sum3 = __lasx_xvfmadd_s(x_v, load_f32(y3 + i), sum3);
    }

    dis0 = reduce_sum(sum0);
    dis1 = reduce_sum(sum1);
    dis2 = reduce_sum(sum2);
    dis3 = reduce_sum(sum3);
    for (; i < d; ++i) {
        dis0 += x[i] * y0[i];
        dis1 += x[i] * y1[i];
        dis2 += x[i] * y2[i];
        dis3 += x[i] * y3[i];
    }
}

void
fvec_L2sqr_batch_4_lasx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3, size_t d,
                        float& dis0, float& dis1, float& dis2, float& dis3) {
    __m256 sum0 = zero_f32();
    __m256 sum1 = zero_f32();
    __m256 sum2 = zero_f32();
    __m256 sum3 = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m256 x_v = load_f32(x + i);
        const __m256 diff0 = __lasx_xvfsub_s(x_v, load_f32(y0 + i));
        const __m256 diff1 = __lasx_xvfsub_s(x_v, load_f32(y1 + i));
        const __m256 diff2 = __lasx_xvfsub_s(x_v, load_f32(y2 + i));
        const __m256 diff3 = __lasx_xvfsub_s(x_v, load_f32(y3 + i));
        sum0 = __lasx_xvfmadd_s(diff0, diff0, sum0);
        sum1 = __lasx_xvfmadd_s(diff1, diff1, sum1);
        sum2 = __lasx_xvfmadd_s(diff2, diff2, sum2);
        sum3 = __lasx_xvfmadd_s(diff3, diff3, sum3);
    }

    dis0 = reduce_sum(sum0);
    dis1 = reduce_sum(sum1);
    dis2 = reduce_sum(sum2);
    dis3 = reduce_sum(sum3);
    for (; i < d; ++i) {
        const float diff0 = x[i] - y0[i];
        const float diff1 = x[i] - y1[i];
        const float diff2 = x[i] - y2[i];
        const float diff3 = x[i] - y3[i];
        dis0 += diff0 * diff0;
        dis1 += diff1 * diff1;
        dis2 += diff2 * diff2;
        dis3 += diff3 * diff3;
    }
}

}  // namespace faiss::cppcontrib::knowhere
