#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace knowhere::sparse::inverted {

/**
 * Block codecs encode and decode a list of integers. This is in opposition to a streaming codec,
 * which can encode and decode values one by one.
 */
class BlockCodec {
 public:
    virtual ~BlockCodec() = default;

    /**
     * Encodes a list of `n` unsigned integers and appends them to the output buffer.
     */
    virtual void
    encode(uint32_t const* in, size_t n, std::vector<uint8_t>& out) const = 0;

    /**
     * Decodes a list of `n` unsigned integers from a binary buffer and writes them to pre-allocated
     * memory.
     */
    virtual uint8_t const*
    decode(uint8_t const* in, uint32_t* out, size_t n) const = 0;

    /**
     * Returns the block size of the encoding.
     *
     * Block codecs write blocks of fixed size, e.g., 128 integers. Thus, it is only possible to
     * encode at most `block_size()` elements with a single `encode` call.
     */
    [[nodiscard]] virtual auto
    block_size() const noexcept -> size_t = 0;

    /**
     * Returns the name of the codec.
     */
    [[nodiscard]] virtual auto
    get_name() const noexcept -> std::string_view = 0;
};

using BlockCodecPtr = std::shared_ptr<BlockCodec>;

}  // namespace knowhere::sparse::inverted
