// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
// on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
// for the specific language governing permissions and limitations under the License.

#include "index/sparse/codec/simd_bitpacking_kernel.h"

#include <string.h>

// These are pruned generated kernels derived from fast-pack/simdcomp; see simdcomp/UPSTREAM.md and LICENSE.
// Every retained entry point is static and is consumed only by the block-oriented wrappers below.
#include "simdcomp/src/simdbitpacking.c"
#include "simdcomp/src/simdintegratedbitpacking.c"

enum { KNOWHERE_SIMD_BLOCK_SIZE = 128 };

#define KNOWHERE_SIMD_PACK_CASE(BITS)                                 \
    case BITS:                                                        \
        do {                                                          \
            __SIMD_fastpackwithoutmask##BITS##_32(in, (__m128i*)out); \
            in += KNOWHERE_SIMD_BLOCK_SIZE;                           \
            out += (size_t)(BITS) * sizeof(__m128i);                  \
        } while (--block_count != 0);                                 \
        return

void
knowhere_simd_pack_128_blocks(const uint32_t* in, uint8_t* out, size_t block_count, uint32_t bits) {
    if (block_count == 0) {
        return;
    }
    if (bits == 32) {
        memcpy(out, in, block_count * KNOWHERE_SIMD_BLOCK_SIZE * sizeof(*in));
        return;
    }
    switch (bits) {
        KNOWHERE_SIMD_PACK_CASE(1);
        KNOWHERE_SIMD_PACK_CASE(2);
        KNOWHERE_SIMD_PACK_CASE(3);
        KNOWHERE_SIMD_PACK_CASE(4);
        KNOWHERE_SIMD_PACK_CASE(5);
        KNOWHERE_SIMD_PACK_CASE(6);
        KNOWHERE_SIMD_PACK_CASE(7);
        KNOWHERE_SIMD_PACK_CASE(8);
        KNOWHERE_SIMD_PACK_CASE(9);
        KNOWHERE_SIMD_PACK_CASE(10);
        KNOWHERE_SIMD_PACK_CASE(11);
        KNOWHERE_SIMD_PACK_CASE(12);
        KNOWHERE_SIMD_PACK_CASE(13);
        KNOWHERE_SIMD_PACK_CASE(14);
        KNOWHERE_SIMD_PACK_CASE(15);
        KNOWHERE_SIMD_PACK_CASE(16);
        KNOWHERE_SIMD_PACK_CASE(17);
        KNOWHERE_SIMD_PACK_CASE(18);
        KNOWHERE_SIMD_PACK_CASE(19);
        KNOWHERE_SIMD_PACK_CASE(20);
        KNOWHERE_SIMD_PACK_CASE(21);
        KNOWHERE_SIMD_PACK_CASE(22);
        KNOWHERE_SIMD_PACK_CASE(23);
        KNOWHERE_SIMD_PACK_CASE(24);
        KNOWHERE_SIMD_PACK_CASE(25);
        KNOWHERE_SIMD_PACK_CASE(26);
        KNOWHERE_SIMD_PACK_CASE(27);
        KNOWHERE_SIMD_PACK_CASE(28);
        KNOWHERE_SIMD_PACK_CASE(29);
        KNOWHERE_SIMD_PACK_CASE(30);
        KNOWHERE_SIMD_PACK_CASE(31);
        default:
            return;
    }
}

#undef KNOWHERE_SIMD_PACK_CASE

#define KNOWHERE_SIMD_UNPACK_CASE(BITS)                            \
    case BITS:                                                     \
        do {                                                       \
            __SIMD_fastunpack##BITS##_32((const __m128i*)in, out); \
            in += (size_t)(BITS) * sizeof(__m128i);                \
            out += KNOWHERE_SIMD_BLOCK_SIZE;                       \
        } while (--block_count != 0);                              \
        return

void
knowhere_simd_unpack_128_blocks(const uint8_t* in, uint32_t* out, size_t block_count, uint32_t bits) {
    if (block_count == 0) {
        return;
    }
    if (bits == 32) {
        memcpy(out, in, block_count * KNOWHERE_SIMD_BLOCK_SIZE * sizeof(*out));
        return;
    }
    switch (bits) {
        KNOWHERE_SIMD_UNPACK_CASE(1);
        KNOWHERE_SIMD_UNPACK_CASE(2);
        KNOWHERE_SIMD_UNPACK_CASE(3);
        KNOWHERE_SIMD_UNPACK_CASE(4);
        KNOWHERE_SIMD_UNPACK_CASE(5);
        KNOWHERE_SIMD_UNPACK_CASE(6);
        KNOWHERE_SIMD_UNPACK_CASE(7);
        KNOWHERE_SIMD_UNPACK_CASE(8);
        KNOWHERE_SIMD_UNPACK_CASE(9);
        KNOWHERE_SIMD_UNPACK_CASE(10);
        KNOWHERE_SIMD_UNPACK_CASE(11);
        KNOWHERE_SIMD_UNPACK_CASE(12);
        KNOWHERE_SIMD_UNPACK_CASE(13);
        KNOWHERE_SIMD_UNPACK_CASE(14);
        KNOWHERE_SIMD_UNPACK_CASE(15);
        KNOWHERE_SIMD_UNPACK_CASE(16);
        KNOWHERE_SIMD_UNPACK_CASE(17);
        KNOWHERE_SIMD_UNPACK_CASE(18);
        KNOWHERE_SIMD_UNPACK_CASE(19);
        KNOWHERE_SIMD_UNPACK_CASE(20);
        KNOWHERE_SIMD_UNPACK_CASE(21);
        KNOWHERE_SIMD_UNPACK_CASE(22);
        KNOWHERE_SIMD_UNPACK_CASE(23);
        KNOWHERE_SIMD_UNPACK_CASE(24);
        KNOWHERE_SIMD_UNPACK_CASE(25);
        KNOWHERE_SIMD_UNPACK_CASE(26);
        KNOWHERE_SIMD_UNPACK_CASE(27);
        KNOWHERE_SIMD_UNPACK_CASE(28);
        KNOWHERE_SIMD_UNPACK_CASE(29);
        KNOWHERE_SIMD_UNPACK_CASE(30);
        KNOWHERE_SIMD_UNPACK_CASE(31);
        default:
            return;
    }
}

#undef KNOWHERE_SIMD_UNPACK_CASE

void
knowhere_simd_unpack_d1_128_blocks(const uint8_t* in, uint32_t* out, size_t block_count, uint32_t bits,
                                   uint32_t previous_value) {
    if (block_count == 0 || bits == 0 || bits >= 32) {
        return;
    }
    do {
        simdunpackd1(previous_value, (const __m128i*)in, out, bits);
        previous_value = out[KNOWHERE_SIMD_BLOCK_SIZE - 1];
        in += (size_t)bits * sizeof(__m128i);
        out += KNOWHERE_SIMD_BLOCK_SIZE;
    } while (--block_count != 0);
}

void
knowhere_simd_integrate_doc_id_gaps(uint32_t* values, size_t count, uint32_t previous_value) {
    const __m128i one = _mm_set1_epi32(1);
    __m128i previous = _mm_set1_epi32(previous_value);
    const size_t vector_end = count & ~(size_t)3;
    size_t i = 0;
    for (; i < vector_end; i += 4) {
        const __m128i gaps = _mm_loadu_si128((const __m128i*)(values + i));
        const __m128i deltas = _mm_add_epi32(gaps, one);
        const __m128i pairs = _mm_add_epi32(_mm_slli_si128(deltas, 8), deltas);
        const __m128i prefix = _mm_add_epi32(_mm_slli_si128(pairs, 4), pairs);
        previous = _mm_add_epi32(prefix, _mm_shuffle_epi32(previous, 0xff));
        _mm_storeu_si128((__m128i*)(values + i), previous);
    }

    if (vector_end != 0) {
        previous_value = (uint32_t)_mm_cvtsi128_si32(_mm_shuffle_epi32(previous, 0xff));
    }
    for (; i < count; ++i) {
        previous_value += values[i] + 1;
        values[i] = previous_value;
    }
}
