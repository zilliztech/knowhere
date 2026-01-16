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

#if defined(__x86_64__)

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "distances_avx512.h"

namespace faiss::cppcontrib::knowhere {

namespace {

template <size_t qb>
static int
rabitq_dp_q(const uint8_t* q, const uint8_t* x, const size_t d) {
    __m512i sum_512 = _mm512_setzero_si512();

    const size_t di_8b = (d + 7) / 8;

    const size_t d_512 = (d / 512) * 512;
    const size_t d_256 = (d / 256) * 256;
    const size_t d_128 = (d / 128) * 128;

    for (size_t i = 0; i < d_512; i += 512) {
        __m512i v_x = _mm512_loadu_si512(x + i / 8);
        for (size_t j = 0; j < qb; j++) {
            __m512i v_q = _mm512_loadu_si512(q + j * di_8b + i / 8);
            __m512i v_and = _mm512_and_si512(v_q, v_x);
            __m512i v_popcnt = _mm512_popcnt_epi32(v_and);
            sum_512 = _mm512_add_epi32(sum_512, _mm512_slli_epi32(v_popcnt, j));
        }
    }

    __m256i sum_256 = _mm256_add_epi32(_mm512_extracti32x8_epi32(sum_512, 0), _mm512_extracti32x8_epi32(sum_512, 1));

    if (d_256 != d_512) {
        __m256i v_x = _mm256_loadu_si256((const __m256i*)(x + d_512 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m256i v_q = _mm256_loadu_si256((const __m256i*)(q + j * di_8b + d_512 / 8));
            __m256i v_and = _mm256_and_si256(v_q, v_x);
            __m256i v_popcnt = _mm256_popcnt_epi32(v_and);
            sum_256 = _mm256_add_epi32(sum_256, _mm256_slli_epi32(v_popcnt, j));
        }
    }

    __m128i sum_128 = _mm_add_epi32(_mm256_extracti32x4_epi32(sum_256, 0), _mm256_extracti32x4_epi32(sum_256, 1));

    if (d_128 != d_256) {
        __m128i v_x = _mm_loadu_si128((const __m128i*)(x + d_256 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m128i v_q = _mm_loadu_si128((const __m128i*)(q + j * di_8b + d_256 / 8));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            __m128i v_popcnt = _mm_popcnt_epi32(v_and);
            sum_128 = _mm_add_epi32(sum_128, _mm_slli_epi32(v_popcnt, j));
        }
    }

    if (d != d_128) {
        const size_t leftovers = d - d_128;
        const __mmask16 mask = (1 << ((leftovers + 7) / 8)) - 1;

        __m128i v_x = _mm_maskz_loadu_epi8(mask, (const __m128i*)(x + d_128 / 8));
        for (size_t j = 0; j < qb; j++) {
            __m128i v_q = _mm_maskz_loadu_epi8(mask, (const __m128i*)(q + j * di_8b + d_128 / 8));
            __m128i v_and = _mm_and_si128(v_q, v_x);
            __m128i v_popcnt = _mm_popcnt_epi32(v_and);
            sum_128 = _mm_add_epi32(sum_128, _mm_slli_epi32(v_popcnt, j));
        }
    }

    int sum_64le = 0;
    sum_64le += _mm_extract_epi32(sum_128, 0);
    sum_64le += _mm_extract_epi32(sum_128, 1);
    sum_64le += _mm_extract_epi32(sum_128, 2);
    sum_64le += _mm_extract_epi32(sum_128, 3);

    return sum_64le;
}

}  // namespace

int
rabitq_dp_popcnt_avx512icx(const uint8_t* q, const uint8_t* x, const size_t d, const size_t nb) {
    switch (nb) {
        case 1:
            return rabitq_dp_q<1>(q, x, d);
        case 2:
            return rabitq_dp_q<2>(q, x, d);
        case 3:
            return rabitq_dp_q<3>(q, x, d);
        case 4:
            return rabitq_dp_q<4>(q, x, d);
        case 5:
            return rabitq_dp_q<5>(q, x, d);
        case 6:
            return rabitq_dp_q<6>(q, x, d);
        case 7:
            return rabitq_dp_q<7>(q, x, d);
        case 8:
            return rabitq_dp_q<8>(q, x, d);
    }

    return 0;
}

}  // namespace faiss::cppcontrib::knowhere

#endif
