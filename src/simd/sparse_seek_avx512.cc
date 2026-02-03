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

// This file is compiled with -mavx512f -mavx512dq flags to enable AVX512 intrinsics
// Runtime CPU detection ensures it's only called on CPUs with AVX512 support

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace knowhere::sparse {

// AVX512 SIMD seek: find first position where id >= target
// Returns position or size if not found
// Uses 16-wide vectorization (AVX512 processes 16 x 32-bit integers)
size_t
simd_seek_avx512_impl(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
    constexpr size_t AVX512_WIDTH = 16;
    constexpr size_t SIMD_ALIGNMENT = 64;
    size_t pos = start_pos;

    // Scalar until aligned to 64-byte boundary
    while (pos < size && (reinterpret_cast<uintptr_t>(&ids[pos]) % SIMD_ALIGNMENT) != 0) {
        if (ids[pos] >= target) {
            return pos;
        }
        pos++;
    }

    // Broadcast target to all 16 lanes
    __m512i target_vec = _mm512_set1_epi32(static_cast<int32_t>(target));

    // SIMD loop: process 16 elements at a time
    while (pos + AVX512_WIDTH <= size) {
        __m512i id_vec = _mm512_load_si512(reinterpret_cast<const __m512i*>(&ids[pos]));

        // AVX512 has native >= comparison
        // _mm512_cmpge_epi32_mask returns a 16-bit mask where bit i is set if id[i] >= target
        __mmask16 ge_mask = _mm512_cmpge_epi32_mask(id_vec, target_vec);

        if (ge_mask != 0) {
            // Found at least one element where id >= target
            // Return first position using count trailing zeros
            return pos + __builtin_ctz(ge_mask);
        }
        pos += AVX512_WIDTH;
    }

    // Scalar tail: handle remaining elements
    while (pos < size) {
        if (ids[pos] >= target) {
            return pos;
        }
        pos++;
    }

    return size;  // Not found
}

}  // namespace knowhere::sparse
