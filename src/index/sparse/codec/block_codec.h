#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "index/sparse/codec/simd_prefix_sum.h"

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
     * Encodes document-ID gaps. Codecs may override this method to use a document-ID-specific candidate set or
     * tie-breaking policy. The default preserves the regular encoding behavior.
     */
    virtual void
    encode_doc_ids(uint32_t const* in, size_t n, std::vector<uint8_t>& out) const {
        encode(in, n, out);
    }

    /**
     * Decodes a list of `n` unsigned integers from a binary buffer and writes them to pre-allocated
     * memory.
     */
    virtual uint8_t const*
    decode(uint8_t const* in, uint32_t* out, size_t n) const = 0;

    /**
     * Decodes document-ID gaps and reconstructs absolute IDs with the shared SIMD prefix sum. `previous_value` is
     * the document ID immediately before this block; UINT32_MAX represents the implicit value before document zero.
     * Codecs may override this to fuse unpacking and prefix sums.
     */
    virtual uint8_t const*
    decode_doc_ids(uint8_t const* in, uint32_t* out, size_t n, uint32_t previous_value) const {
        uint8_t const* next = decode(in, out, n);
        simd_prefix_sum::integrate_doc_id_gaps(out, n, previous_value);
        return next;
    }

    /**
     * Returns the maximum logical block length. Complete blocks contain `block_size()` values; the final block may
     * be shorter, and one encode/decode call handles at most this many values.
     */
    [[nodiscard]] virtual auto
    block_size() const noexcept -> size_t = 0;

    /**
     * Returns the name of the codec.
     */
    [[nodiscard]] virtual auto
    get_name() const noexcept -> std::string_view = 0;

    /**
     * Returns whether BlockInvertedIndex may use its singleton short form: omit the document-ID payload, store the
     * sole document ID in the posting block max-ID slot, encode BM25's TF - 1 as an untagged internal varint, and
     * leave an IP value in its regular raw representation.
     */
    [[nodiscard]] virtual bool
    supports_singleton_short_form() const noexcept {
        return false;
    }
};

using BlockCodecPtr = std::shared_ptr<BlockCodec>;

}  // namespace knowhere::sparse::inverted
