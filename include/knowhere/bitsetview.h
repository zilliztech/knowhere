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

#ifndef BITSET_H
#define BITSET_H

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace knowhere {
class BitsetView {
 public:
    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const uint8_t* data, size_t num_bits, size_t num_filtered_out_bits = 0, size_t id_offset = 0)
        : bits_(data), num_bits_(num_bits), num_filtered_out_bits_(num_filtered_out_bits), id_offset_(id_offset) {
    }

    BitsetView(const std::nullptr_t) : BitsetView() {
    }

    bool
    empty() const {
        return num_bits_ == 0;
    }

    // return the number of the bits. if with id mapping, return the number of the internal ids.
    size_t
    size() const {
        if (out_ids_ != nullptr) {
            return num_internal_ids_;
        }
        return num_bits_;
    }

    // return the number of filtered out bits. if with id mapping, return the number of filtered out ids.
    size_t
    count() const {
        if (out_ids_ != nullptr) {
            return num_filtered_out_ids_;
        }
        return num_filtered_out_bits_;
    }

    size_t
    byte_size() const {
        return (num_bits_ + 8 - 1) >> 3;
    }

    const uint8_t*
    data() const {
        return bits_;
    }

    bool
    has_out_ids() const {
        return out_ids_ != nullptr;
    }

    void
    set_out_ids(const uint32_t* out_ids, size_t num_internal_ids,
                std::optional<size_t> num_filtered_out_ids = std::nullopt) {
        out_ids_ = out_ids;
        num_internal_ids_ = num_internal_ids;
        if (num_filtered_out_ids.has_value()) {
            num_filtered_out_ids_ = num_filtered_out_ids.value();
        } else {
            // auto calculate num_filtered_out_ids if not provided
            num_filtered_out_ids_ = get_filtered_out_num_();
        }
    }

    const uint32_t*
    out_ids_data() const {
        if (out_ids_ == nullptr) {
            return nullptr;
        }
        return out_ids_;
    }

    void
    set_id_offset(size_t id_offset) {
        id_offset_ = id_offset;
    }

    // if the test succeeds, then the index should be skipped during search; otherwise, it should be included.
    bool
    test(int64_t index) const {
        int64_t out_id = index + id_offset_;
        if (out_ids_ != nullptr) {
            out_id = out_ids_[out_id];
        }
        // when index is larger than the max_offset, ignore it
        return (out_id >= static_cast<int64_t>(num_bits_)) || (bits_[out_id >> 3] & (0x1 << (out_id & 0x7)));
    }
    // return the filtered ratio. if with id mapping, calculated by internal_ids rather than bits.
    float
    filter_ratio() const {
        return empty() ? 0.0f : ((float)count() / size());
    }

    // Return whether every ID in the half-open range [begin, end) is filtered.
    bool
    range_all_filtered(size_t begin, size_t end) const {
        assert(begin <= end);
        assert(end <= size());
        if (begin == end) {
            return true;
        }

        // With id mapping the bit lookup requires per-index indirection, so a word-level scan is not possible.
        if (out_ids_ != nullptr) {
            for (size_t index = begin; index < end; ++index) {
                if (!test(index)) {
                    return false;
                }
            }
            return true;
        }

        // Without id mapping, internal index i maps to bit (i + id_offset_). Mapped ids >= num_bits_ are
        // treated as filtered by test(), so only the clamped range [lo, hi) needs an explicit bit check.
        const size_t lo = begin + id_offset_;
        const size_t hi = std::min(end + id_offset_, num_bits_);
        if (hi <= lo) {
            return true;
        }
        return all_bits_set(lo, hi);
    }

    // Return the last valid ID below upper_bound.
    std::optional<size_t>
    previous_valid_index(size_t upper_bound) const {
        if (upper_bound == 0) {
            return std::nullopt;
        }
        if (empty()) {
            return upper_bound - 1;
        }

        // With id mapping the bit lookup requires per-index indirection, so a word-level scan is not possible.
        if (out_ids_ != nullptr) {
            size_t index = std::min(upper_bound, num_internal_ids_);
            while (index > 0) {
                --index;
                if (!test(index)) {
                    return index;
                }
            }
            return std::nullopt;
        }

        // Without id mapping, internal index i maps to bit (i + id_offset_). Scan the bit range
        // [id_offset_, hi_bit) word by word, where bits >= num_bits_ are treated as filtered out.
        const size_t low_bit = id_offset_;
        const size_t hi_bit = std::min(id_offset_ + upper_bound, num_bits_);
        if (hi_bit <= low_bit) {
            return std::nullopt;
        }
        const size_t low_word = low_bit >> 6;
        size_t word_index = (hi_bit - 1) >> 6;
        uint64_t valid = ~load_word(word_index) & lower_bits_mask(((hi_bit - 1) & 63) + 1);
        while (true) {
            if (word_index == low_word) {
                valid &= ~lower_bits_mask(low_bit & 63);
            }
            if (valid != 0) {
                return (word_index << 6) + 63 - __builtin_clzll(valid) - id_offset_;
            }
            if (word_index == low_word) {
                return std::nullopt;
            }
            --word_index;
            valid = ~load_word(word_index);
        }
    }

    size_t
    get_filtered_out_num_() const {
        if (empty()) {
            return 0;
        }
        if (out_ids_ != nullptr) {
            // if with id mapping, there is no optimization for the traversal.
            size_t count = 0;
            for (size_t i = 0; i < num_internal_ids_; i++) {
                if (test(i)) {
                    count++;
                }
            }
            return count;
        }
        // if without id mapping, use a better algorithm to calculate the number of filtered out bits.
        size_t ret = 0;
        auto len_uint8 = byte_size();
        auto len_uint64 = len_uint8 >> 3;

        auto popcount8 = [&](uint8_t x) -> int {
            x = (x & 0x55) + ((x >> 1) & 0x55);
            x = (x & 0x33) + ((x >> 2) & 0x33);
            x = (x & 0x0F) + ((x >> 4) & 0x0F);
            return x;
        };

        uint64_t* p_uint64 = (uint64_t*)bits_;
        for (size_t i = 0; i < len_uint64; i++) {
            ret += __builtin_popcountll(*p_uint64);
            p_uint64++;
        }

        // calculate remainder
        uint8_t* p_uint8 = (uint8_t*)bits_ + (len_uint64 << 3);
        for (size_t i = (len_uint64 << 3); i < len_uint8; i++) {
            ret += popcount8(*p_uint8);
            p_uint8++;
        }

        return ret;
    }

    // return the first valid idx. if with id mapping, return the first valid internal_id.
    size_t
    get_first_valid_index() const {
        if (out_ids_ != nullptr) {
            // if with id mapping, there is no optimization for the traversal.
            for (size_t i = 0; i < num_internal_ids_; i++) {
                if (!test(i)) {
                    return i;
                }
            }
            return num_internal_ids_;
        }
        // if without id mapping, use a better algorithm to find the first valid index.
        size_t ret = 0;
        auto len_uint8 = byte_size();
        auto len_uint64 = len_uint8 >> 3;

        uint64_t* p_uint64 = (uint64_t*)bits_;
        for (size_t i = 0; i < len_uint64; i++) {
            uint64_t value = (~(*p_uint64));
            if (value == 0) {
                p_uint64++;
                continue;
            }
            ret = __builtin_ctzll(value);
            return i * 64 + ret;
        }

        // calculate remainder
        uint8_t* p_uint8 = (uint8_t*)bits_ + (len_uint64 << 3);
        for (size_t i = 0; i < len_uint8 - (len_uint64 << 3); i++) {
            uint8_t value = (~(*p_uint8));
            if (value == 0) {
                p_uint8++;
                continue;
            }
            ret = __builtin_ctz(value);
            return len_uint64 * 64 + i * 8 + ret;
        }

        return num_bits_;
    }

    std::string
    to_string(size_t from, size_t to) const {
        if (empty()) {
            return "";
        }
        std::stringbuf buf;
        to = std::min<size_t>(to, num_bits_);
        for (size_t i = from; i < to; i++) {
            buf.sputc(test(i) ? '1' : '0');
        }
        return buf.str();
    }

 private:
    static uint64_t
    lower_bits_mask(size_t bits) {
        assert(bits <= 64);
        return bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
    }

    static uint64_t
    load_full_word(const uint8_t* data) {
        return static_cast<uint64_t>(data[0]) | (static_cast<uint64_t>(data[1]) << 8) |
               (static_cast<uint64_t>(data[2]) << 16) | (static_cast<uint64_t>(data[3]) << 24) |
               (static_cast<uint64_t>(data[4]) << 32) | (static_cast<uint64_t>(data[5]) << 40) |
               (static_cast<uint64_t>(data[6]) << 48) | (static_cast<uint64_t>(data[7]) << 56);
    }

    uint64_t
    load_word(size_t word_index) const {
        const size_t bytes = byte_size();
        if (bytes == 0 || word_index > (bytes - 1) / sizeof(uint64_t)) {
            return 0;
        }

        const size_t byte_offset = word_index * sizeof(uint64_t);
        const auto* data = bits_ + byte_offset;
        const size_t remaining_bytes = bytes - byte_offset;
        if (remaining_bytes >= sizeof(uint64_t)) {
            return load_full_word(data);
        }

        uint64_t word = 0;
        for (size_t byte = 0; byte < remaining_bytes; ++byte) {
            word |= static_cast<uint64_t>(data[byte]) << (byte * 8);
        }
        return word;
    }

    // Return whether every bit in the half-open bit range [bit_begin, bit_end) is set.
    // Requires bit_begin < bit_end <= num_bits_.
    bool
    all_bits_set(size_t bit_begin, size_t bit_end) const {
        // Only the first word carries a low boundary and only the last word carries a high boundary; the
        // interior words must be fully set, so they reduce to a plain "all bits set" check.
        const size_t first_word = bit_begin >> 6;
        const size_t last_word = (bit_end - 1) >> 6;
        const uint64_t first_mask = ~lower_bits_mask(bit_begin & 63);
        const uint64_t last_mask = lower_bits_mask(((bit_end - 1) & 63) + 1);

        if (first_word == last_word) {
            const uint64_t mask = first_mask & last_mask;
            return (load_word(first_word) & mask) == mask;
        }

        if ((load_word(first_word) & first_mask) != first_mask) {
            return false;
        }
        // Interior words are fully in-bounds (highest bit < bit_end <= num_bits_), so read them directly.
        for (size_t word_index = first_word + 1; word_index < last_word; ++word_index) {
            if (load_full_word(bits_ + (word_index << 3)) != ~uint64_t{0}) {
                return false;
            }
        }
        return (load_word(last_word) & last_mask) == last_mask;
    }

    const uint8_t* bits_ = nullptr;
    size_t num_bits_ = 0;
    size_t num_filtered_out_bits_ = 0;

    // optional. many indexes will share one bitset, requiring offset to distinguish between them.
    //  like multi-chunk brute-force in /src/common/comp/brute_force.cc, or mv-only in /src/index/hnsw/faiss_hnsw.cc
    size_t id_offset_ = 0;  // offset of the internal ids

    // optional. bitset supports id mapping.
    // Even allows multiple ids to map to the same bit, so the number of internal ids and bits may be not equal.
    const uint32_t* out_ids_ = nullptr;
    size_t num_internal_ids_ = 0;
    size_t num_filtered_out_ids_ = 0;
};

}  // namespace knowhere

#endif /* BITSET_H */
