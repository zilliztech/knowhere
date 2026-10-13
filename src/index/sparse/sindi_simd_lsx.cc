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

#include "index/sparse/sindi_simd.h"

#if defined(__loongarch_sx)

#include <lsxintrin.h>

#include <bit>
#include <cmath>

namespace knowhere::sparse::inverted::sindi {
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
broadcast_f32(float value) {
    return std::bit_cast<__m128>(__lsx_vreplgr2vr_w(std::bit_cast<int32_t>(value)));
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

}  // namespace

float
ip_accumulate_lsx_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    const __m128 qval_v = broadcast_f32(qval);
    __m128 maximum = std::bit_cast<__m128>(__lsx_vldi(0));
    alignas(16) float gathered[kLanes];
    alignas(16) float updated[kLanes];
    int32_t i = 0;

    // LSX has no indexed gather/scatter. Process eight FP16 values with native
    // conversion and vector arithmetic, keeping only the indexed accesses scalar.
    for (; i + 8 <= num; i += 8) {
        const __m128i half = __lsx_vld(const_cast<knowhere::fp16*>(vals + i), 0);
        const __m128 values[2] = {__lsx_vfcvtl_s_h(half), __lsx_vfcvth_s_h(half)};
        for (int32_t half_index = 0; half_index < 2; ++half_index) {
            const int32_t offset = i + half_index * static_cast<int32_t>(kLanes);
            for (size_t lane = 0; lane < kLanes; ++lane) {
                gathered[lane] = out[ids[offset + static_cast<int32_t>(lane)]];
            }
            const __m128 sum = __lsx_vfmadd_s(values[half_index], qval_v, load_f32(gathered));
            store_f32(updated, sum);
            for (size_t lane = 0; lane < kLanes; ++lane) {
                out[ids[offset + static_cast<int32_t>(lane)]] = updated[lane];
            }
            maximum = __lsx_vfmax_s(maximum, sum);
        }
    }

    float max_val = reduce_max(maximum);
    for (; i < num; ++i) {
        const float new_val = (out[ids[i]] += qval * static_cast<float>(vals[i]));
        max_val = std::fmax(max_val, new_val);
    }
    return max_val;
}

float
bm25_accumulate_lsx_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                        float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;
    const __m128 numerator_scale = broadcast_f32(qval * p1);
    const __m128 p2_v = broadcast_f32(p2);
    const __m128 p3_v = broadcast_f32(p3);
    __m128 maximum = std::bit_cast<__m128>(__lsx_vldi(0));
    alignas(16) float doc_lengths[kLanes];
    alignas(16) float gathered[kLanes];
    alignas(16) float updated[kLanes];
    int32_t i = 0;

    for (; i + 8 <= num; i += 8) {
        const __m128i packed = __lsx_vld(const_cast<uint16_t*>(vals + i), 0);
        const __m128 tf[2] = {__lsx_vffint_s_wu(__lsx_vilvl_h(__lsx_vldi(0), packed)),
                              __lsx_vffint_s_wu(__lsx_vexth_wu_hu(packed))};
        for (int32_t half_index = 0; half_index < 2; ++half_index) {
            const int32_t offset = i + half_index * static_cast<int32_t>(kLanes);
            for (size_t lane = 0; lane < kLanes; ++lane) {
                const uint16_t id = ids[offset + static_cast<int32_t>(lane)];
                doc_lengths[lane] = row_sums[id];
                gathered[lane] = out[id];
            }

            const __m128 numerator = __lsx_vfmul_s(tf[half_index], numerator_scale);
            const __m128 denominator = __lsx_vfadd_s(tf[half_index], __lsx_vfmadd_s(load_f32(doc_lengths), p3_v, p2_v));
            const __m128 sum = __lsx_vfadd_s(load_f32(gathered), __lsx_vfdiv_s(numerator, denominator));
            store_f32(updated, sum);
            for (size_t lane = 0; lane < kLanes; ++lane) {
                out[ids[offset + static_cast<int32_t>(lane)]] = updated[lane];
            }
            maximum = __lsx_vfmax_s(maximum, sum);
        }
    }

    float max_val = reduce_max(maximum);
    for (; i < num; ++i) {
        const float tf = static_cast<float>(vals[i]);
        const uint16_t docid = ids[i];
        const float score = qval * p1 * tf / (tf + p2 + p3 * row_sums[docid]);
        const float new_val = (out[docid] += score);
        max_val = std::fmax(max_val, new_val);
    }
    return max_val;
}

void
batch_insert_lsx(const float* scores, size_t docid_start, size_t count,
                 knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset) {
    alignas(16) int32_t selected[kLanes];
    size_t i = 0;
    for (; i + kLanes <= count; i += kLanes) {
        const __m128 threshold_v = broadcast_f32(threshold);
        const __m128 values = load_f32(scores + i);
        __lsx_vst(__lsx_vfcmp_clt_s(threshold_v, values), selected, 0);
        for (size_t lane = 0; lane < kLanes; ++lane) {
            if (selected[lane] == 0) {
                continue;
            }
            const size_t index = i + lane;
            if (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + index))) {
                continue;
            }
            if (topk_q.Push(scores[index], static_cast<uint32_t>(docid_start + index)) && topk_q.Full()) {
                threshold = topk_q.Threshold();
            }
        }
    }

    for (; i < count; ++i) {
        const float score = scores[i];
        if (score <= threshold || (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + i)))) {
            continue;
        }
        if (topk_q.Push(score, static_cast<uint32_t>(docid_start + i)) && topk_q.Full()) {
            threshold = topk_q.Threshold();
        }
    }
}

}  // namespace knowhere::sparse::inverted::sindi

#endif
