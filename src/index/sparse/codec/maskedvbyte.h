// MaskedVByte codec wrapper.
// Derived from the MaskedVByte library by Daniel Lemire, Nathan Kurz, and Christoph Rupp.
//   Paper: "Vectorized VByte Decoding" (Jeff Plaisance, Nathan Kurz, Daniel Lemire)
//          https://arxiv.org/abs/1503.07387
//   Repository: https://github.com/lemire/MaskedVByte
//   License: Apache License 2.0

#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <string_view>
#include <vector>

#include "index/sparse/codec/block_codec.h"

namespace knowhere::sparse::inverted {

extern "C" {
// VariableByte functions
size_t
masked_vbyte_read_loop(const uint8_t* in, uint32_t* out, int len_signed);
size_t
masked_vbyte_read_loop_fromcompressedsize(const uint8_t* in, uint32_t* out, size_t inputsize);
size_t
altmasked_vbyte_read_loop(const uint8_t* in, uint32_t* out, int len_signed);
size_t
altmasked_vbyte_read_loop_fromcompressedsize(const uint8_t* in, uint32_t* out, size_t inputsize);
}

/**
 * MaskedVByte coding.
 *
 * Uses MaskedVByte's vbyte_encode/masked_vbyte_decode functions for compression/decompression.
 */
class MaskedVByteBlockCodec : public BlockCodec {
    static constexpr uint64_t m_block_size = 256;
    static constexpr size_t m_max_compressed_bytes = 4 * m_block_size * sizeof(uint32_t);

 public:
    constexpr static std::string_view name = "block_maskedvbyte";

    virtual ~MaskedVByteBlockCodec() = default;

    size_t
    encode_block(const uint32_t* in, const size_t length, uint8_t* out) const {
        const uint8_t* const initbout = out;
        uint8_t* bout = out;
        for (size_t k = 0; k < length; ++k) {
            const uint32_t val(in[k]);
            /**
             * Code below could be shorter. Whether it could be faster
             * depends on your compiler and machine.
             */
            if (val < (1U << 7)) {
                *bout = val & 0x7F;
                ++bout;
            } else if (val < (1U << 14)) {
                *bout = static_cast<uint8_t>((val & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(val >> 7);
                ++bout;
            } else if (val < (1U << 21)) {
                *bout = static_cast<uint8_t>((val & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(((val >> 7) & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(val >> 14);
                ++bout;
            } else if (val < (1U << 28)) {
                *bout = static_cast<uint8_t>((val & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(((val >> 7) & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(((val >> 14) & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(val >> 21);
                ++bout;
            } else {
                *bout = static_cast<uint8_t>((val & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(((val >> 7) & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(((val >> 14) & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(((val >> 21) & 0x7F) | (1U << 7));
                ++bout;
                *bout = static_cast<uint8_t>(val >> 28);
                ++bout;
            }
        }
        return bout - initbout;
    }

    void
    encode(uint32_t const* in, size_t n, std::vector<uint8_t>& out) const override {
        assert(n <= m_block_size);
        auto* src = const_cast<uint32_t*>(in);
        thread_local std::array<uint8_t, m_max_compressed_bytes> buf{};
        size_t out_len = encode_block(src, n, buf.data());
        assert(out_len <= m_max_compressed_bytes);
        out.insert(out.end(), buf.data(), buf.data() + out_len);
    }

    uint8_t const*
    decode(uint8_t const* in, uint32_t* out, size_t n) const override {
        assert(n <= m_block_size);
        size_t nvalue = masked_vbyte_read_loop(in, out, n);
        return in + nvalue;
    }

    [[nodiscard]] auto
    block_size() const noexcept -> size_t override {
        return m_block_size;
    }

    [[nodiscard]] auto
    get_name() const noexcept -> std::string_view override {
        return name;
    }
};
}  // namespace knowhere::sparse::inverted
