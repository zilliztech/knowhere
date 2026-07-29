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

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "index/sparse/codec/block_codec.h"
#include "index/sparse/codec/simd_bitpacking.h"
#include "index/sparse/codec/streamvbyte.h"

namespace knowhere::sparse::inverted {

/**
 * Adaptive compression for blocks of up to 256 unsigned integers.
 *
 * Document-ID gaps, in both complete and final partial blocks, choose among fixed-width bit packing, the custom
 * StreamVByte-0124 format, and an all-equal representation. Complete term-frequency blocks use an all-equal
 * representation when possible and otherwise use a Lucene-inspired patched-bitpacking format; this format is not
 * wire-compatible with Lucene. A complete block that cannot reduce a 32-bit base within the exception limit uses
 * plain 32-bit packing. Final partial term-frequency blocks choose the smallest payload among StreamVByte-0124,
 * fixed-width bit packing, all-equal, and patched bit packing. Partial blocks encode exactly their logical value
 * count and are never padded to 256 values.
 *
 * Complete 128-value chunks at widths 1..31 use vertical SIMD kernels derived from fast-pack/simdcomp; width 32 is a
 * direct copy. Document-ID decoding uses the integrated d1 unpacker to reconstruct absolute IDs; TF decoding remains
 * ordinary bit unpacking. A logical block still carries one outer encoding tag.
 *
 * Block layout:
 *   byte 0      : encoding tag
 *   remaining   : encoding-specific payload
 *
 * Tags 1..32 directly encode the bit width. Other tags are listed below.
 */
class AdaptiveBlockCodec final : public BlockCodec {
 public:
    // Maximum number of values in one logical adaptive block.
    static constexpr size_t kBlockSize = 256;
    static constexpr std::string_view name = "block_adaptive";

    void
    encode(uint32_t const* in, size_t n, std::vector<uint8_t>& out) const override {
        assert(in != nullptr);
        assert(n > 0 && n <= kBlockSize);

        encode_block(in, n, out, /*doc_ids=*/false);
    }

    void
    encode_doc_ids(uint32_t const* in, size_t n, std::vector<uint8_t>& out) const override {
        assert(in != nullptr);
        assert(n > 0 && n <= kBlockSize);

        encode_block(in, n, out, /*doc_ids=*/true);
    }

    uint8_t const*
    decode(uint8_t const* in, uint32_t* out, size_t n) const override {
        assert(in != nullptr);
        assert(out != nullptr);
        assert(n > 0 && n <= kBlockSize);
        return decode_block(in, out, n);
    }

    uint8_t const*
    decode_doc_ids(uint8_t const* in, uint32_t* out, size_t n, uint32_t previous_value) const override {
        assert(in != nullptr);
        assert(out != nullptr);
        assert(n > 0 && n <= kBlockSize);

        const uint8_t type = *in;
        if (is_bitpacked(type)) {
            return simd_bitpacking::unpack_doc_ids(in + 1, out, n, type, previous_value);
        }
        return BlockCodec::decode_doc_ids(in, out, n, previous_value);
    }

    [[nodiscard]] auto
    block_size() const noexcept -> size_t override {
        return kBlockSize;
    }

    [[nodiscard]] auto
    get_name() const noexcept -> std::string_view override {
        return name;
    }

    [[nodiscard]] bool
    supports_singleton_short_form() const noexcept override {
        return true;
    }

 private:
    void
    encode_block(uint32_t const* in, size_t n, std::vector<uint8_t>& out, bool doc_ids) const {
        assert(n > 0 && n <= kBlockSize);

        const Selection selection = select_encoding(in, n, doc_ids);
        out.push_back(selection.type);

        if (selection.type == kAllZero) {
            return;
        }
        if (is_bitpacked(selection.type)) {
            const size_t begin = out.size();
            out.resize(begin + selection.payload_size);
            pack_bits(in, n, selection.type, out.data() + begin);
            return;
        }

        switch (selection.type) {
            case kAllEqual8:
                out.push_back(static_cast<uint8_t>(in[0]));
                return;
            case kAllEqual16:
                append_u16(in[0], out);
                return;
            case kAllEqual32:
                append_u32(in[0], out);
                return;
            case kStreamVByte: {
                alignas(16) thread_local std::array<uint8_t, kMaxStreamVByteBytes> buffer{};
                const size_t encoded_size = streamvbyte_encode_0124(in, static_cast<uint32_t>(n), buffer.data());
                assert(encoded_size == selection.payload_size);
                out.insert(out.end(), buffer.data(), buffer.data() + encoded_size);
                return;
            }
            case kPatchedBitpack: {
                const size_t begin = out.size();
                encode_patched_bitpack(in, n, selection, out);
                assert(out.size() - begin == selection.payload_size);
                static_cast<void>(begin);
                return;
            }
            default:
                assert(false && "invalid adaptive block encoding");
                return;
        }
    }

    uint8_t const*
    decode_block(uint8_t const* in, uint32_t* out, size_t n) const {
        assert(n > 0 && n <= kBlockSize);

        const uint8_t type = *in++;
        if (type == kAllZero) {
            std::fill_n(out, n, 0U);
            return in;
        }
        if (is_bitpacked(type)) {
            return unpack_bits(in, out, n, type);
        }

        switch (type) {
            case kAllEqual8:
                std::fill_n(out, n, static_cast<uint32_t>(*in));
                return in + 1;
            case kAllEqual16: {
                const uint32_t value = read_u16(in);
                std::fill_n(out, n, value);
                return in + sizeof(uint16_t);
            }
            case kAllEqual32: {
                const uint32_t value = read_u32(in);
                std::fill_n(out, n, value);
                return in + sizeof(uint32_t);
            }
            case kStreamVByte:
                return in + streamvbyte_decode_0124(in, out, static_cast<uint32_t>(n));
            case kPatchedBitpack:
                return decode_patched_bitpack(in, out, n);
            default:
                throw std::runtime_error("invalid adaptive block encoding tag");
        }
    }
    // Current wire tags. Values 1..32 directly represent fixed bit widths.
    static constexpr uint8_t kAllZero = 0;
    static constexpr uint8_t kAllEqual8 = 33;
    static constexpr uint8_t kAllEqual16 = 34;
    static constexpr uint8_t kAllEqual32 = 35;
    static constexpr uint8_t kStreamVByte = 36;
    static constexpr uint8_t kPatchedBitpack = 37;
    static constexpr uint8_t kMaxPForExceptions = 7;
    static constexpr size_t kMaxStreamVByteBytes = streamvbyte_max_compressedbytes(kBlockSize);

    struct Selection {
        uint8_t type;
        size_t payload_size;
        uint8_t patched_bits{0};
        uint8_t exceptions{0};
        bool constant_base{false};
        uint32_t base_value{0};
    };

    [[nodiscard]] static constexpr bool
    is_bitpacked(uint8_t type) noexcept {
        return type >= 1 && type <= 32;
    }

    [[nodiscard]] static constexpr size_t
    packed_size(size_t n, uint8_t bits) noexcept {
        return (n * bits + 7) / 8;
    }

    [[nodiscard]] static constexpr size_t
    byte_size_0124(uint32_t value) noexcept {
        if (value == 0) {
            return 0;
        }
        if (value <= UINT8_MAX) {
            return 1;
        }
        if (value <= UINT16_MAX) {
            return 2;
        }
        return 4;
    }

    [[nodiscard]] static constexpr size_t
    vint_size(uint32_t value) noexcept {
        size_t size = 1;
        while (value >= 128) {
            value >>= 7;
            ++size;
        }
        return size;
    }

    [[nodiscard]] static Selection
    select_full_patched_bitpack(uint32_t const* in, const std::array<uint16_t, 33>& histogram, uint8_t max_bits) {
        assert(max_bits > 0 && max_bits <= 32);

        const int min_bits = std::max(0, static_cast<int>(max_bits) - 8);
        size_t cumulative_exceptions = 0;
        uint8_t patched_bits = max_bits;
        uint8_t exceptions = 0;
        for (int bits = max_bits; bits >= min_bits; --bits) {
            if (cumulative_exceptions > kMaxPForExceptions) {
                break;
            }
            patched_bits = static_cast<uint8_t>(bits);
            exceptions = static_cast<uint8_t>(cumulative_exceptions);
            cumulative_exceptions += histogram[bits];
        }

        // The token has five bits for the base width. Up to seven 32-bit values can be exceptions over a base of at
        // most 31 bits; only a plan whose base itself still needs 32 bits falls back to plain packing.
        if (patched_bits == 32) {
            return {32, packed_size(kBlockSize, 32), 0};
        }

        const uint32_t mask = patched_bits == 0 ? 0U : (uint32_t{1} << patched_bits) - 1;
        const uint32_t base_value = in[0] & mask;
        bool constant_base = max_bits <= 8;
        for (size_t i = 1; constant_base && i < kBlockSize; ++i) {
            constant_base = (in[i] & mask) == base_value;
        }

        const size_t base_size = constant_base ? vint_size(base_value) : packed_size(kBlockSize, patched_bits);
        const size_t payload_size = 1 + base_size + 2 * static_cast<size_t>(exceptions);
        return {kPatchedBitpack, payload_size, patched_bits, exceptions, constant_base, base_value};
    }

    [[nodiscard]] static Selection
    select_tail_patched_bitpack(uint32_t const* in, size_t n, const std::array<uint16_t, 33>& histogram,
                                uint8_t max_bits) {
        assert(n > 0 && n < kBlockSize);
        assert(max_bits > 0 && max_bits <= 32);

        // Plain bit packing is both the fallback and the upper bound a PFor plan must beat. Enumerate wider bases
        // first and use strict comparisons so equal-size plans keep fewer exceptions and a shorter scalar patch path.
        Selection best{max_bits, packed_size(n, max_bits), 0};
        size_t exceptions = 0;
        int bits = max_bits;
        if (bits == 32) {
            exceptions = histogram[32];
            --bits;
        }

        const int min_bits = std::max(1, static_cast<int>(max_bits) - 8);
        for (; bits >= min_bits && exceptions <= kMaxPForExceptions; --bits) {
            const auto base_bits = static_cast<uint8_t>(bits);
            const auto exception_count = static_cast<uint8_t>(exceptions);
            const size_t exception_bytes = 2 * exceptions;

            Selection candidate{kPatchedBitpack, 1 + packed_size(n, base_bits) + exception_bytes, base_bits,
                                exception_count};
            if (candidate.payload_size < best.payload_size) {
                best = candidate;
            }

            // A zero token width is reserved for a constant low-bit base. Keep the planned width at least one so
            // histogram exception counts agree with the encoder for zero values.
            if (max_bits <= 8) {
                const uint32_t mask = (uint32_t{1} << base_bits) - 1;
                const uint32_t base_value = in[0] & mask;
                bool constant_base = true;
                for (size_t i = 1; constant_base && i < n; ++i) {
                    constant_base = (in[i] & mask) == base_value;
                }
                if (constant_base) {
                    candidate = {
                        kPatchedBitpack, 1 + vint_size(base_value) + exception_bytes, base_bits, exception_count, true,
                        base_value};
                    if (candidate.payload_size < best.payload_size) {
                        best = candidate;
                    }
                }
            }

            exceptions += histogram[base_bits];
        }
        return best;
    }

    [[nodiscard]] static Selection
    select_encoding(uint32_t const* in, size_t n, bool doc_ids) {
        uint32_t max_value = 0;
        bool all_equal = true;
        const bool tail = n < kBlockSize;
        const bool streamvbyte_candidate = tail || doc_ids;
        size_t streamvbyte_size = streamvbyte_candidate ? (n + 3) / 4 : 0;
        const bool full_tf = !doc_ids && n == kBlockSize;
        std::array<uint16_t, 33> bit_width_histogram{};

        for (size_t i = 0; i < n; ++i) {
            const uint32_t value = in[i];
            max_value = std::max(max_value, value);
            all_equal = all_equal && value == in[0];
            if (streamvbyte_candidate) {
                streamvbyte_size += byte_size_0124(value);
            }
            if (!doc_ids) {
                // Match Lucene PackedInts.bitsRequired: zero still occupies one base bit. A token width of zero is
                // reserved for the constant-base optimization after exception planning.
                ++bit_width_histogram[std::max(1, static_cast<int>(std::bit_width(value)))];
            }
        }

        if (full_tf && !all_equal) {
            // Full TF blocks intentionally keep the PFor layout even when the base width cannot be reduced, trading
            // its one-byte token for a uniform full-block TF decode path. A 32-bit base is the only plain fallback
            // because the five-bit PFor token cannot represent width 32.
            const auto max_bits = static_cast<uint8_t>(std::bit_width(max_value));
            return select_full_patched_bitpack(in, bit_width_histogram, max_bits);
        }

        Selection best{kStreamVByte, streamvbyte_size, 0};
        if (!tail) {
            // Start complete blocks from fixed-width packing. A complete document-ID block changes to
            // StreamVByte-0124 only when its payload is strictly smaller, so ties retain bit packing or all-equal.
            if (max_value == 0) {
                best = {kAllZero, 0, 0};
            } else {
                const auto bits = static_cast<uint8_t>(std::bit_width(max_value));
                best = {bits, packed_size(n, bits), 0};
            }
        } else if (max_value != 0) {
            const auto bits = static_cast<uint8_t>(std::bit_width(max_value));
            const size_t size = packed_size(n, bits);
            if (size < best.payload_size) {
                best = {bits, size, 0};
            }
        }

        if (all_equal) {
            Selection all_equal_encoding{};
            if (max_value == 0) {
                all_equal_encoding = {kAllZero, 0, 0};
            } else if (max_value <= UINT8_MAX) {
                all_equal_encoding = {kAllEqual8, 1, 0};
            } else if (max_value <= UINT16_MAX) {
                all_equal_encoding = {kAllEqual16, 2, 0};
            } else {
                all_equal_encoding = {kAllEqual32, 4, 0};
            }
            // Prefer all-equal on ties because it avoids the general StreamVByte or bit-unpack path.
            if (all_equal_encoding.payload_size <= best.payload_size) {
                best = all_equal_encoding;
            }
        }

        if (doc_ids && !tail && streamvbyte_size < best.payload_size) {
            best = {kStreamVByte, streamvbyte_size, 0};
        }

        if (!doc_ids && tail && !all_equal) {
            const auto max_bits = static_cast<uint8_t>(std::bit_width(max_value));
            const Selection pfor = select_tail_patched_bitpack(in, n, bit_width_histogram, max_bits);
            // PFor must be strictly smaller; a tie retains the current StreamVByte or bit-packed plan and avoids the
            // PFor token/exception decode path.
            if (pfor.type == kPatchedBitpack && pfor.payload_size < best.payload_size) {
                best = pfor;
            }
        }

        return best;
    }

    static void
    pack_bits(uint32_t const* in, size_t n, uint8_t bits, uint8_t* out) {
        simd_bitpacking::pack(in, n, bits, out);
    }

    [[nodiscard]] static uint8_t const*
    unpack_bits(uint8_t const* in, uint32_t* out, size_t n, uint8_t bits) {
        return simd_bitpacking::unpack(in, out, n, bits);
    }

    static void
    append_vint(uint32_t value, std::vector<uint8_t>& out) {
        while (value >= 128) {
            out.push_back(static_cast<uint8_t>(value | 0x80U));
            value >>= 7;
        }
        out.push_back(static_cast<uint8_t>(value));
    }

    [[nodiscard]] static uint8_t const*
    read_vint(uint8_t const* in, uint32_t& value) {
        value = 0;
        uint32_t shift = 0;
        for (;;) {
            const uint8_t byte = *in++;
            value |= static_cast<uint32_t>(byte & 0x7fU) << shift;
            if ((byte & 0x80U) == 0) {
                return in;
            }
            shift += 7;
        }
    }

    static void
    encode_patched_bitpack(uint32_t const* in, size_t n, const Selection& selection, std::vector<uint8_t>& out) {
        assert(n > 0 && n <= kBlockSize);
        assert(selection.patched_bits <= 31);
        assert(selection.exceptions <= kMaxPForExceptions);

        const uint8_t token_bits = selection.constant_base ? 0 : selection.patched_bits;
        out.push_back(static_cast<uint8_t>((selection.exceptions << 5) | token_bits));

        if (!selection.constant_base && selection.exceptions == 0) {
            assert(selection.patched_bits != 0);
            const size_t begin = out.size();
            out.resize(begin + packed_size(n, selection.patched_bits));
            pack_bits(in, n, selection.patched_bits, out.data() + begin);
            return;
        }

        const uint32_t mask = selection.patched_bits == 0 ? 0U : (uint32_t{1} << selection.patched_bits) - 1;
        alignas(64) thread_local std::array<uint32_t, kBlockSize> low_values{};
        std::array<uint8_t, 2 * kMaxPForExceptions> exception_data{};
        uint8_t exception_count = 0;
        for (size_t i = 0; i < n; ++i) {
            const uint32_t value = in[i];
            if (!selection.constant_base) {
                low_values[i] = value & mask;
            }
            if (value > mask) {
                assert(exception_count < selection.exceptions);
                const uint32_t patch = selection.constant_base ? value & ~mask : value >> selection.patched_bits;
                assert(patch > 0 && patch <= UINT8_MAX);
                exception_data[2 * exception_count] = static_cast<uint8_t>(i);
                exception_data[2 * exception_count + 1] = static_cast<uint8_t>(patch);
                ++exception_count;
            }
        }
        assert(exception_count == selection.exceptions);

        if (selection.constant_base) {
            append_vint(selection.base_value, out);
        } else {
            assert(selection.patched_bits != 0);
            const size_t begin = out.size();
            out.resize(begin + packed_size(n, selection.patched_bits));
            pack_bits(low_values.data(), n, selection.patched_bits, out.data() + begin);
        }
        out.insert(out.end(), exception_data.data(), exception_data.data() + 2 * exception_count);
    }

    [[nodiscard]] static uint8_t const*
    decode_patched_bitpack(uint8_t const* in, uint32_t* out, size_t n) {
        assert(n > 0 && n <= kBlockSize);

        const uint8_t token = *in++;
        const uint8_t bits = token & 0x1fU;
        const uint8_t exceptions = token >> 5;
        if (bits == 0) {
            uint32_t base_value = 0;
            in = read_vint(in, base_value);
            std::fill_n(out, n, base_value);
        } else {
            in = unpack_bits(in, out, n, bits);
        }

        for (uint8_t exception = 0; exception < exceptions; ++exception) {
            const uint8_t position = *in++;
            const uint8_t patch = *in++;
            out[position] |= static_cast<uint32_t>(patch) << bits;
        }
        return in;
    }

    static void
    append_u16(uint32_t value, std::vector<uint8_t>& out) {
        out.push_back(static_cast<uint8_t>(value));
        out.push_back(static_cast<uint8_t>(value >> 8));
    }

    static void
    append_u32(uint32_t value, std::vector<uint8_t>& out) {
        out.push_back(static_cast<uint8_t>(value));
        out.push_back(static_cast<uint8_t>(value >> 8));
        out.push_back(static_cast<uint8_t>(value >> 16));
        out.push_back(static_cast<uint8_t>(value >> 24));
    }

    [[nodiscard]] static uint32_t
    read_u16(uint8_t const* in) {
        return static_cast<uint32_t>(in[0]) | (static_cast<uint32_t>(in[1]) << 8);
    }

    [[nodiscard]] static uint32_t
    read_u32(uint8_t const* in) {
        return static_cast<uint32_t>(in[0]) | (static_cast<uint32_t>(in[1]) << 8) |
               (static_cast<uint32_t>(in[2]) << 16) | (static_cast<uint32_t>(in[3]) << 24);
    }
};

}  // namespace knowhere::sparse::inverted
