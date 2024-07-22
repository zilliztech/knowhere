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

#ifndef BFLOAT16_CONVERT_H
#define BFLOAT16_CONVERT_H
#include <knowhere/operands.h>
#include <cassert>
#if defined(__ARM_NEON)

#endif

#if defined(__x86_64__)
#include <immintrin.h>
#endif
namespace faiss {
#if defined(__x86_64__)
#define ALIGNED(x) __attribute__((aligned(x)))

static inline __m128
_mm_bf16_to_fp32(const __m128i& a) {
    auto o = _mm_slli_epi32(_mm_cvtepu16_epi32(a), 16);
    return _mm_castsi128_ps(o);
}

static inline __m256
_mm256_bf16_to_fp32(const __m128i& a) {
    __m256i o = _mm256_slli_epi32(_mm256_cvtepu16_epi32(a), 16);
    return _mm256_castsi256_ps(o);
}

static inline __m512
_mm512_bf16_to_fp32(const __m256i& x) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(x), 16));
}

static inline __m128i
mm_masked_read_short(int d, const uint16_t* x) {
    assert(0 <= d && d < 8);
    ALIGNED(16) uint16_t buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    switch (d) {
        case 7:
            buf[6] = x[6];
        case 6:
            buf[5] = x[5];
        case 5:
            buf[4] = x[4];
        case 4:
            buf[3] = x[3];
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_si128((__m128i*)buf);
}

static inline float
_mm256_reduce_add_ps(const __m256 res) {
    const __m128 sum = _mm_add_ps(_mm256_castps256_ps128(res), _mm256_extractf128_ps(res, 1));
    const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
    const __m128 v1 = _mm_add_ps(sum, v0);
    __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
    const __m128 v3 = _mm_add_ps(v1, v2);
    return _mm_cvtss_f32(v3);
}
#endif

#if defined(__ARM_NEON)

#endif
}  // namespace faiss
#endif /* BFLOAT16_CONVERT_H */
