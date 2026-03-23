// StreamVByte codec wrapper.
// Derived from the StreamVByte library by Daniel Lemire et al.
//   Paper: "Stream VByte: Faster Byte-Oriented Integer Compression"
//          https://arxiv.org/abs/1709.08990
//   Repository: https://github.com/lemire/streamvbyte
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
size_t
streamvbyte_encode_0124(const uint32_t* in, uint32_t count, uint8_t* out);
size_t
streamvbyte_decode_0124(const uint8_t* in, uint32_t* out, uint32_t count);
}

// This is a constexpr version of the function in the streamvbyte library.
constexpr size_t
streamvbyte_max_compressedbytes(uint32_t length) {
    // number of control bytes:
    size_t cb = (length + 3) / 4;
    // maximum number of data bytes:
    size_t db = (size_t)length * sizeof(uint32_t);
    return cb + db;
}

/**
 * StreamVByte coding.
 *
 * Daniel Lemire, Nathan Kurz, Christoph Rupp: Stream VByte: Faster byte-oriented integer
 * compression. Inf. Process. Lett. 130: 1-6 (2018). DOI: https://doi.org/10.1016/j.ipl.2017.09.011
 */
class StreamVByteBlockCodec : public BlockCodec {
    static constexpr uint64_t m_streamvbyte_padding = 16;
    static constexpr uint64_t m_block_size = 256;
    static constexpr size_t m_max_compressed_bytes = streamvbyte_max_compressedbytes(m_block_size);

 public:
    constexpr static std::string_view name = "block_streamvbyte";

    virtual ~StreamVByteBlockCodec() = default;

    void
    encode(uint32_t const* in, size_t n, std::vector<uint8_t>& out) const override {
        assert(n <= m_block_size);
        auto* src = const_cast<uint32_t*>(in);
        thread_local std::array<uint8_t, m_max_compressed_bytes> buf{};
        size_t out_len = streamvbyte_encode_0124(src, n, buf.data());
        out.insert(out.end(), buf.data(), buf.data() + out_len);
    }

    uint8_t const*
    decode(uint8_t const* in, uint32_t* out, size_t n) const override {
        assert(n <= m_block_size);
        auto read = streamvbyte_decode_0124(in, out, n);
        return in + read;
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
