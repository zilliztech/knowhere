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

#pragma once

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "index/sparse/codec/simd_prefix_sum.h"

namespace knowhere::sparse::inverted::simd_bitpacking {

// Complete 128-value chunks at widths 1..31 use a four-lane vertical SIMD kernel; width 32 is copied directly. A
// full 256-value compressed payload invokes the kernel twice, while only the remaining fewer than 128 values use
// the compact scalar path. The payload is exactly ceil(n * bit_width / 8) bytes, including for partial blocks.
inline constexpr size_t kChunkSize = 128;

static_assert(std::endian::native == std::endian::little,
              "the SIMD bit-packing disk layout requires a little-endian target");

namespace detail {

inline uint8_t*
pack_tail(uint32_t const* in, size_t n, uint8_t bits, uint8_t* out) noexcept {
    uint64_t accumulator = 0;
    uint32_t buffered_bits = 0;
    for (size_t i = 0; i < n; ++i) {
        accumulator |= static_cast<uint64_t>(in[i]) << buffered_bits;
        buffered_bits += bits;
        while (buffered_bits >= 8) {
            *out++ = static_cast<uint8_t>(accumulator);
            accumulator >>= 8;
            buffered_bits -= 8;
        }
    }
    if (buffered_bits != 0) {
        *out++ = static_cast<uint8_t>(accumulator);
    }
    return out;
}

inline uint8_t const*
unpack_tail(uint8_t const* in, uint32_t* out, size_t n, uint8_t bits) noexcept {
    const uint64_t mask = bits == 32 ? UINT32_MAX : (uint64_t{1} << bits) - 1;
    uint64_t accumulator = 0;
    uint32_t buffered_bits = 0;
    for (size_t i = 0; i < n; ++i) {
        while (buffered_bits < bits) {
            accumulator |= static_cast<uint64_t>(*in++) << buffered_bits;
            buffered_bits += 8;
        }
        out[i] = static_cast<uint32_t>(accumulator & mask);
        accumulator >>= bits;
        buffered_bits -= bits;
    }
    return in;
}

inline void
validate_arguments(void const* in, void const* out, uint8_t bits) {
    assert(in != nullptr);
    assert(out != nullptr);
    if (bits == 0 || bits > 32) {
        throw std::invalid_argument("SIMD bit width must be in [1, 32]");
    }
}

}  // namespace detail

[[nodiscard]] inline constexpr size_t
packed_size(size_t n, uint8_t bits) noexcept {
    return (n * bits + 7) / 8;
}

inline void
pack(uint32_t const* in, size_t n, uint8_t bits, uint8_t* out) {
    detail::validate_arguments(in, out, bits);

    const size_t chunks = n / kChunkSize;
    if (chunks != 0) {
        knowhere_simd_pack_128_blocks(in, out, chunks, bits);
        const size_t chunk_values = chunks * kChunkSize;
        in += chunk_values;
        out += chunks * static_cast<size_t>(bits) * 16;
        n -= chunk_values;
    }
    static_cast<void>(detail::pack_tail(in, n, bits, out));
}

[[nodiscard]] inline uint8_t const*
unpack(uint8_t const* in, uint32_t* out, size_t n, uint8_t bits) {
    detail::validate_arguments(in, out, bits);

    const size_t chunks = n / kChunkSize;
    if (chunks != 0) {
        knowhere_simd_unpack_128_blocks(in, out, chunks, bits);
        const size_t chunk_values = chunks * kChunkSize;
        in += chunks * static_cast<size_t>(bits) * 16;
        out += chunk_values;
        n -= chunk_values;
    }
    return detail::unpack_tail(in, out, n, bits);
}

// Doc-ID payloads store (current - previous - 1). Complete 128-value chunks use a simdcomp simdunpackd1 kernel whose
// prefix sum folds in the omitted one and directly reconstructs absolute document IDs. A compact tail is unpacked
// normally and then integrated by the shared SIMD prefix helper. Width 32 uses the regular unpack path (a direct
// copy for complete chunks) because upstream d1 treats it as raw absolute values rather than packed deltas.
[[nodiscard]] inline uint8_t const*
unpack_doc_ids(uint8_t const* in, uint32_t* out, size_t n, uint8_t bits, uint32_t previous_value) {
    detail::validate_arguments(in, out, bits);

    if (bits == 32) {
        uint8_t const* next = unpack(in, out, n, bits);
        simd_prefix_sum::integrate_doc_id_gaps(out, n, previous_value);
        return next;
    }

    const size_t chunks = n / kChunkSize;
    if (chunks != 0) {
        knowhere_simd_unpack_d1_128_blocks(in, out, chunks, bits, previous_value);
        const size_t chunk_values = chunks * kChunkSize;
        in += chunks * static_cast<size_t>(bits) * 16;
        out += chunk_values;
        n -= chunk_values;
        previous_value = out[-1];
    }

    uint8_t const* next = detail::unpack_tail(in, out, n, bits);
    simd_prefix_sum::integrate_doc_id_gaps(out, n, previous_value);
    return next;
}

}  // namespace knowhere::sparse::inverted::simd_bitpacking
