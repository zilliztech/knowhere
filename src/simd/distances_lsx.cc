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

#include "distances_lsx.h"

#include <lsxintrin.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace faiss::cppcontrib::knowhere {
namespace {

constexpr size_t kLanes = 4;

inline __m128
load_f32(const float* ptr) {
    return std::bit_cast<__m128>(__lsx_vld(const_cast<float*>(ptr), 0));
}

inline void
store_f32(float* ptr, __m128 value) {
    __lsx_vst(std::bit_cast<__m128i>(value), ptr, 0);
}

inline __m128
zero_f32() {
    return std::bit_cast<__m128>(__lsx_vldi(0));
}

inline __m128
broadcast_f32(float value) {
    return std::bit_cast<__m128>(__lsx_vreplgr2vr_w(std::bit_cast<int32_t>(value)));
}

inline __m128
abs_f32(__m128 value) {
    return std::bit_cast<__m128>(__lsx_vbitclri_w(std::bit_cast<__m128i>(value), 31));
}

inline float
reduce_sum(__m128 value) {
    alignas(16) float lanes[kLanes];
    store_f32(lanes, value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

inline float
reduce_max(__m128 value) {
    alignas(16) float lanes[kLanes];
    store_f32(lanes, value);
    float result = 0.0f;
    for (float lane : lanes) {
        result = std::fmax(result, lane);
    }
    return result;
}

template <size_t Parts>
struct FloatVectors {
    __m128 lanes[Parts];
};

inline FloatVectors<2>
load_fp16(const ::knowhere::fp16* ptr) {
    const __m128i packed = __lsx_vld(const_cast<::knowhere::fp16*>(ptr), 0);
    return {{__lsx_vfcvtl_s_h(packed), __lsx_vfcvth_s_h(packed)}};
}

inline FloatVectors<2>
load_bf16(const ::knowhere::bf16* ptr) {
    const __m128i packed = __lsx_vld(const_cast<::knowhere::bf16*>(ptr), 0);
    const __m128i zero = __lsx_vldi(0);
    const __m128i low = __lsx_vslli_w(__lsx_vilvl_h(zero, packed), 16);
    const __m128i high = __lsx_vslli_w(__lsx_vexth_wu_hu(packed), 16);
    return {{std::bit_cast<__m128>(low), std::bit_cast<__m128>(high)}};
}

using i8x16 = int8_t __attribute__((vector_size(16)));
using i8x4 = int8_t __attribute__((vector_size(4)));
using i32x4 = int32_t __attribute__((vector_size(16)));

template <int Offset>
inline __m128i
expand_i8_to_i32(__m128i packed) {
    const i8x16 bytes = std::bit_cast<i8x16>(packed);
    const i8x4 selected = __builtin_shufflevector(bytes, bytes, Offset, Offset + 1, Offset + 2, Offset + 3);
    return std::bit_cast<__m128i>(__builtin_convertvector(selected, i32x4));
}

inline FloatVectors<4>
load_int8(const int8_t* ptr) {
    const __m128i packed = __lsx_vld(const_cast<int8_t*>(ptr), 0);
    return {{std::bit_cast<__m128>(__lsx_vffint_s_w(expand_i8_to_i32<0>(packed))),
             std::bit_cast<__m128>(__lsx_vffint_s_w(expand_i8_to_i32<4>(packed))),
             std::bit_cast<__m128>(__lsx_vffint_s_w(expand_i8_to_i32<8>(packed))),
             std::bit_cast<__m128>(__lsx_vffint_s_w(expand_i8_to_i32<12>(packed)))}};
}

enum class ConvertedMetric {
    InnerProduct,
    L2,
};

template <ConvertedMetric Metric>
inline __m128
accumulate(__m128 x, __m128 y, __m128 sum) {
    if constexpr (Metric == ConvertedMetric::InnerProduct) {
        return __lsx_vfmadd_s(x, y, sum);
    } else {
        const __m128 diff = __lsx_vfsub_s(x, y);
        return __lsx_vfmadd_s(diff, diff, sum);
    }
}

template <ConvertedMetric Metric, typename T, size_t Parts>
float
converted_distance(const T* x, const T* y, size_t d, FloatVectors<Parts> (*load)(const T*)) {
    __m128 sums[Parts];
    for (size_t part = 0; part < Parts; ++part) {
        sums[part] = zero_f32();
    }

    constexpr size_t width = Parts * kLanes;
    size_t i = 0;
    for (; i + width <= d; i += width) {
        const auto x_vectors = load(x + i);
        const auto y_vectors = load(y + i);
        for (size_t part = 0; part < Parts; ++part) {
            sums[part] = accumulate<Metric>(x_vectors.lanes[part], y_vectors.lanes[part], sums[part]);
        }
    }

    float result = 0.0f;
    for (size_t part = 0; part < Parts; ++part) {
        result += reduce_sum(sums[part]);
    }
    for (; i < d; ++i) {
        const float x_value = static_cast<float>(x[i]);
        const float y_value = static_cast<float>(y[i]);
        if constexpr (Metric == ConvertedMetric::InnerProduct) {
            result += x_value * y_value;
        } else {
            const float diff = x_value - y_value;
            result += diff * diff;
        }
    }
    return result;
}

template <ConvertedMetric Metric, typename T, size_t Parts>
void
converted_batch_4(const T* x, const T* y0, const T* y1, const T* y2, const T* y3, size_t d, float& dis0, float& dis1,
                  float& dis2, float& dis3, FloatVectors<Parts> (*load)(const T*)) {
    __m128 sums[4][Parts];
    for (size_t output = 0; output < 4; ++output) {
        for (size_t part = 0; part < Parts; ++part) {
            sums[output][part] = zero_f32();
        }
    }

    constexpr size_t width = Parts * kLanes;
    size_t i = 0;
    for (; i + width <= d; i += width) {
        const auto x_vectors = load(x + i);
        const FloatVectors<Parts> y_vectors[4] = {load(y0 + i), load(y1 + i), load(y2 + i), load(y3 + i)};
        for (size_t part = 0; part < Parts; ++part) {
            for (size_t output = 0; output < 4; ++output) {
                sums[output][part] =
                    accumulate<Metric>(x_vectors.lanes[part], y_vectors[output].lanes[part], sums[output][part]);
            }
        }
    }

    float* outputs[4] = {&dis0, &dis1, &dis2, &dis3};
    const T* y[4] = {y0, y1, y2, y3};
    for (size_t output = 0; output < 4; ++output) {
        *outputs[output] = 0.0f;
        for (size_t part = 0; part < Parts; ++part) {
            *outputs[output] += reduce_sum(sums[output][part]);
        }
    }
    for (; i < d; ++i) {
        const float x_value = static_cast<float>(x[i]);
        for (size_t output = 0; output < 4; ++output) {
            const float y_value = static_cast<float>(y[output][i]);
            if constexpr (Metric == ConvertedMetric::InnerProduct) {
                *outputs[output] += x_value * y_value;
            } else {
                const float diff = x_value - y_value;
                *outputs[output] += diff * diff;
            }
        }
    }
}

}  // namespace

float
fvec_inner_product_lsx(const float* x, const float* y, size_t d) {
    __m128 sum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        sum = __lsx_vfmadd_s(load_f32(x + i), load_f32(y + i), sum);
    }

    float result = reduce_sum(sum);
    for (; i < d; ++i) {
        result += x[i] * y[i];
    }
    return result;
}

float
fvec_L2sqr_lsx(const float* x, const float* y, size_t d) {
    __m128 sum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m128 diff = __lsx_vfsub_s(load_f32(x + i), load_f32(y + i));
        sum = __lsx_vfmadd_s(diff, diff, sum);
    }

    float result = reduce_sum(sum);
    for (; i < d; ++i) {
        const float diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

float
fvec_L1_lsx(const float* x, const float* y, size_t d) {
    __m128 sum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m128 diff = __lsx_vfsub_s(load_f32(x + i), load_f32(y + i));
        sum = __lsx_vfadd_s(sum, abs_f32(diff));
    }

    float result = reduce_sum(sum);
    for (; i < d; ++i) {
        result += std::fabs(x[i] - y[i]);
    }
    return result;
}

float
fvec_Linf_lsx(const float* x, const float* y, size_t d) {
    __m128 maximum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m128 diff = __lsx_vfsub_s(load_f32(x + i), load_f32(y + i));
        maximum = __lsx_vfmax_s(maximum, abs_f32(diff));
    }

    float result = reduce_max(maximum);
    for (; i < d; ++i) {
        result = std::fmax(result, std::fabs(x[i] - y[i]));
    }
    return result;
}

float
fvec_norm_L2sqr_lsx(const float* x, size_t d) {
    __m128 sum = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m128 value = load_f32(x + i);
        sum = __lsx_vfmadd_s(value, value, sum);
    }

    float result = reduce_sum(sum);
    for (; i < d; ++i) {
        result += x[i] * x[i];
    }
    return result;
}

void
fvec_L2sqr_ny_lsx(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; ++i) {
        dis[i] = fvec_L2sqr_lsx(x, y + i * d, d);
    }
}

void
fvec_inner_products_ny_lsx(float* ip, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; ++i) {
        ip[i] = fvec_inner_product_lsx(x, y + i * d, d);
    }
}

void
fvec_L2sqr_ny_transposed_lsx(float* dis, const float* x, const float* y, const float* y_sqlen, size_t d,
                             size_t d_offset, size_t ny) {
    const float x_sqlen = fvec_norm_L2sqr_lsx(x, d);
    const __m128 x_sqlen_v = broadcast_f32(x_sqlen);
    size_t i = 0;
    for (; i + kLanes <= ny; i += kLanes) {
        __m128 dot_product = zero_f32();
        for (size_t j = 0; j < d; ++j) {
            dot_product = __lsx_vfmadd_s(broadcast_f32(x[j]), load_f32(y + i + j * d_offset), dot_product);
        }
        const __m128 squared_lengths = __lsx_vfadd_s(x_sqlen_v, load_f32(y_sqlen + i));
        store_f32(dis + i, __lsx_vfsub_s(squared_lengths, __lsx_vfadd_s(dot_product, dot_product)));
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
fvec_L2sqr_ny_nearest_lsx(float* distances_tmp_buffer, const float* x, const float* y, size_t d, size_t ny) {
    fvec_L2sqr_ny_lsx(distances_tmp_buffer, x, y, d, ny);
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
fvec_L2sqr_ny_nearest_y_transposed_lsx(float* distances_tmp_buffer, const float* x, const float* y,
                                       const float* y_sqlen, size_t d, size_t d_offset, size_t ny) {
    fvec_L2sqr_ny_transposed_lsx(distances_tmp_buffer, x, y, y_sqlen, d, d_offset, ny);
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
fvec_madd_lsx(size_t n, const float* a, float bf, const float* b, float* c) {
    const __m128 bf_v = broadcast_f32(bf);
    size_t i = 0;
    for (; i + kLanes <= n; i += kLanes) {
        store_f32(c + i, __lsx_vfmadd_s(bf_v, load_f32(b + i), load_f32(a + i)));
    }
    for (; i < n; ++i) {
        c[i] = a[i] + bf * b[i];
    }
}

int
fvec_madd_and_argmin_lsx(size_t n, const float* a, float bf, const float* b, float* c) {
    fvec_madd_lsx(n, a, bf, b, c);
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
fvec_inner_product_batch_4_lsx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                               size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    __m128 sum0 = zero_f32();
    __m128 sum1 = zero_f32();
    __m128 sum2 = zero_f32();
    __m128 sum3 = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m128 x_v = load_f32(x + i);
        sum0 = __lsx_vfmadd_s(x_v, load_f32(y0 + i), sum0);
        sum1 = __lsx_vfmadd_s(x_v, load_f32(y1 + i), sum1);
        sum2 = __lsx_vfmadd_s(x_v, load_f32(y2 + i), sum2);
        sum3 = __lsx_vfmadd_s(x_v, load_f32(y3 + i), sum3);
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
fvec_L2sqr_batch_4_lsx(const float* x, const float* y0, const float* y1, const float* y2, const float* y3, size_t d,
                       float& dis0, float& dis1, float& dis2, float& dis3) {
    __m128 sum0 = zero_f32();
    __m128 sum1 = zero_f32();
    __m128 sum2 = zero_f32();
    __m128 sum3 = zero_f32();
    size_t i = 0;
    for (; i + kLanes <= d; i += kLanes) {
        const __m128 x_v = load_f32(x + i);
        const __m128 diff0 = __lsx_vfsub_s(x_v, load_f32(y0 + i));
        const __m128 diff1 = __lsx_vfsub_s(x_v, load_f32(y1 + i));
        const __m128 diff2 = __lsx_vfsub_s(x_v, load_f32(y2 + i));
        const __m128 diff3 = __lsx_vfsub_s(x_v, load_f32(y3 + i));
        sum0 = __lsx_vfmadd_s(diff0, diff0, sum0);
        sum1 = __lsx_vfmadd_s(diff1, diff1, sum1);
        sum2 = __lsx_vfmadd_s(diff2, diff2, sum2);
        sum3 = __lsx_vfmadd_s(diff3, diff3, sum3);
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

int32_t
ivec_inner_product_lsx(const int8_t* x, const int8_t* y, size_t d) {
    __m128i sum = __lsx_vldi(0);
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m128i x_values = __lsx_vld(const_cast<int8_t*>(x + i), 0);
        const __m128i y_values = __lsx_vld(const_cast<int8_t*>(y + i), 0);
        sum = __lsx_vadd_w(sum, __lsx_vmul_w(expand_i8_to_i32<0>(x_values), expand_i8_to_i32<0>(y_values)));
        sum = __lsx_vadd_w(sum, __lsx_vmul_w(expand_i8_to_i32<4>(x_values), expand_i8_to_i32<4>(y_values)));
        sum = __lsx_vadd_w(sum, __lsx_vmul_w(expand_i8_to_i32<8>(x_values), expand_i8_to_i32<8>(y_values)));
        sum = __lsx_vadd_w(sum, __lsx_vmul_w(expand_i8_to_i32<12>(x_values), expand_i8_to_i32<12>(y_values)));
    }

    alignas(16) int32_t lanes[kLanes];
    __lsx_vst(sum, lanes, 0);
    int32_t result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < d; ++i) {
        result += static_cast<int32_t>(x[i]) * static_cast<int32_t>(y[i]);
    }
    return result;
}

int32_t
ivec_L2sqr_lsx(const int8_t* x, const int8_t* y, size_t d) {
    __m128i sum = __lsx_vldi(0);
    size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        const __m128i x_values = __lsx_vld(const_cast<int8_t*>(x + i), 0);
        const __m128i y_values = __lsx_vld(const_cast<int8_t*>(y + i), 0);
        const __m128i diff0 = __lsx_vsub_w(expand_i8_to_i32<0>(x_values), expand_i8_to_i32<0>(y_values));
        const __m128i diff1 = __lsx_vsub_w(expand_i8_to_i32<4>(x_values), expand_i8_to_i32<4>(y_values));
        const __m128i diff2 = __lsx_vsub_w(expand_i8_to_i32<8>(x_values), expand_i8_to_i32<8>(y_values));
        const __m128i diff3 = __lsx_vsub_w(expand_i8_to_i32<12>(x_values), expand_i8_to_i32<12>(y_values));
        sum = __lsx_vadd_w(sum, __lsx_vmul_w(diff0, diff0));
        sum = __lsx_vadd_w(sum, __lsx_vmul_w(diff1, diff1));
        sum = __lsx_vadd_w(sum, __lsx_vmul_w(diff2, diff2));
        sum = __lsx_vadd_w(sum, __lsx_vmul_w(diff3, diff3));
    }

    alignas(16) int32_t lanes[kLanes];
    __lsx_vst(sum, lanes, 0);
    int32_t result = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < d; ++i) {
        const int32_t diff = static_cast<int32_t>(x[i]) - static_cast<int32_t>(y[i]);
        result += diff * diff;
    }
    return result;
}

float
fp16_vec_inner_product_lsx(const ::knowhere::fp16* x, const ::knowhere::fp16* y, size_t d) {
    return converted_distance<ConvertedMetric::InnerProduct>(x, y, d, load_fp16);
}

float
fp16_vec_L2sqr_lsx(const ::knowhere::fp16* x, const ::knowhere::fp16* y, size_t d) {
    return converted_distance<ConvertedMetric::L2>(x, y, d, load_fp16);
}

float
fp16_vec_norm_L2sqr_lsx(const ::knowhere::fp16* x, size_t d) {
    return converted_distance<ConvertedMetric::InnerProduct>(x, x, d, load_fp16);
}

void
fp16_vec_inner_product_batch_4_lsx(const ::knowhere::fp16* x, const ::knowhere::fp16* y0, const ::knowhere::fp16* y1,
                                   const ::knowhere::fp16* y2, const ::knowhere::fp16* y3, size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3) {
    converted_batch_4<ConvertedMetric::InnerProduct>(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3, load_fp16);
}

void
fp16_vec_L2sqr_batch_4_lsx(const ::knowhere::fp16* x, const ::knowhere::fp16* y0, const ::knowhere::fp16* y1,
                           const ::knowhere::fp16* y2, const ::knowhere::fp16* y3, size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3) {
    converted_batch_4<ConvertedMetric::L2>(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3, load_fp16);
}

float
bf16_vec_inner_product_lsx(const ::knowhere::bf16* x, const ::knowhere::bf16* y, size_t d) {
    return converted_distance<ConvertedMetric::InnerProduct>(x, y, d, load_bf16);
}

float
bf16_vec_L2sqr_lsx(const ::knowhere::bf16* x, const ::knowhere::bf16* y, size_t d) {
    return converted_distance<ConvertedMetric::L2>(x, y, d, load_bf16);
}

float
bf16_vec_norm_L2sqr_lsx(const ::knowhere::bf16* x, size_t d) {
    return converted_distance<ConvertedMetric::InnerProduct>(x, x, d, load_bf16);
}

void
bf16_vec_inner_product_batch_4_lsx(const ::knowhere::bf16* x, const ::knowhere::bf16* y0, const ::knowhere::bf16* y1,
                                   const ::knowhere::bf16* y2, const ::knowhere::bf16* y3, size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3) {
    converted_batch_4<ConvertedMetric::InnerProduct>(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3, load_bf16);
}

void
bf16_vec_L2sqr_batch_4_lsx(const ::knowhere::bf16* x, const ::knowhere::bf16* y0, const ::knowhere::bf16* y1,
                           const ::knowhere::bf16* y2, const ::knowhere::bf16* y3, size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3) {
    converted_batch_4<ConvertedMetric::L2>(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3, load_bf16);
}

float
int8_vec_inner_product_lsx(const int8_t* x, const int8_t* y, size_t d) {
    return converted_distance<ConvertedMetric::InnerProduct>(x, y, d, load_int8);
}

float
int8_vec_L2sqr_lsx(const int8_t* x, const int8_t* y, size_t d) {
    return converted_distance<ConvertedMetric::L2>(x, y, d, load_int8);
}

float
int8_vec_norm_L2sqr_lsx(const int8_t* x, size_t d) {
    return converted_distance<ConvertedMetric::InnerProduct>(x, x, d, load_int8);
}

void
int8_vec_inner_product_batch_4_lsx(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2,
                                   const int8_t* y3, size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    converted_batch_4<ConvertedMetric::InnerProduct>(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3, load_int8);
}

void
int8_vec_L2sqr_batch_4_lsx(const int8_t* x, const int8_t* y0, const int8_t* y1, const int8_t* y2, const int8_t* y3,
                           size_t d, float& dis0, float& dis1, float& dis2, float& dis3) {
    converted_batch_4<ConvertedMetric::L2>(x, y0, y1, y2, y3, d, dis0, dis1, dis2, dis3, load_int8);
}

}  // namespace faiss::cppcontrib::knowhere
