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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "knowhere/array_store.h"

namespace knowhere {

// Non-owning filter view.
//
// bits_ is always addressed by public ids. Backend selectors pass dense
// internal/local ids to test(); id_offset_ handles contiguous windows and
// out_ids_ handles compaction or backend relayout. Count fields are expressed
// in the backend vector domain used by the selector.
class BitsetView {
 public:
    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const uint8_t* data, size_t num_bits, std::optional<size_t> filtered_count = std::nullopt)
        : bits_(data), num_bits_(num_bits), vector_count_(num_bits), filtered_count_(filtered_count) {
    }

    BitsetView(const std::nullptr_t) : BitsetView() {
    }

    bool
    empty() const {
        if (num_bits_ == 0) {
            return true;
        }
        if (!filtered_count_.has_value() || filtered_count_.value() != 0) {
            return false;
        }
        return !has_id_boundary_filter_();
    }

    size_t
    size() const {
        return vector_count_;
    }

    bool
    has_known_count() const {
        return num_bits_ == 0 || filtered_count_.has_value();
    }

    size_t
    count() const {
        if (num_bits_ == 0) {
            return 0;
        }
        if (!filtered_count_.has_value()) {
            throw std::logic_error("BitsetView filtered count is unknown");
        }
        return filtered_count_.value();
    }

    size_t
    byte_size() const {
        return (num_bits_ + 8 - 1) >> 3;
    }

    size_t
    num_bits() const {
        return num_bits_;
    }

    const uint8_t*
    data() const {
        return bits_;
    }

    // Recomputes filter counters for a backend id range.
    void
    count_filtered_bits(size_t bit_offset, size_t bit_count, const uint8_t* valid_bitmap = nullptr) {
        count_filtered_bits_impl_(
            bit_offset, bit_count, valid_bitmap != nullptr,
            [valid_bitmap](size_t byte_idx) { return valid_bitmap[byte_idx]; },
            [valid_bitmap](size_t byte_idx) { return load_u64_unaligned_(valid_bitmap + byte_idx); });
    }

    void
    count_filtered_bits(size_t bit_offset, size_t bit_count, const BitmapArray& valid_bitmap) {
        count_filtered_bits_impl_(
            bit_offset, bit_count, !valid_bitmap.empty(),
            [&valid_bitmap](size_t byte_idx) { return valid_bitmap[byte_idx]; },
            [&valid_bitmap](size_t byte_idx) {
                uint64_t value = 0;
                for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                    value |= static_cast<uint64_t>(valid_bitmap[byte_idx + i]) << (i * 8);
                }
                return value;
            });
    }

    void
    set_vector_count(size_t vector_count) {
        vector_count_ = vector_count;
    }

    void
    set_filter_count(size_t filter_count) {
        filtered_count_ = filter_count;
    }

    bool
    has_out_ids() const {
        return out_ids_count_ != 0;
    }

    void
    set_out_ids(const IdArray& out_ids, size_t out_ids_count) {
        if (out_ids_count > out_ids.size()) {
            throw std::invalid_argument("out ids count exceeds out ids size");
        }
        out_ids_ = out_ids;
        out_ids_count_ = out_ids_count;
    }

    const IdArray&
    get_out_ids() const {
        return out_ids_;
    }

    size_t
    out_ids_count() const {
        return out_ids_count_;
    }

    void
    set_id_offset(size_t id_offset) {
        id_offset_ = id_offset;
    }

    size_t
    id_offset() const {
        return id_offset_;
    }

    // Returns true when a backend id should be skipped.
    bool
    test(int64_t index) const {
        if (index < 0) {
            return true;
        }
        const auto internal_id = static_cast<size_t>(index);
        auto out_id = internal_id + id_offset_;
        if (has_out_ids()) {
            if (out_id >= out_ids_count_) {
                return true;
            }
            const auto mapped_id = out_ids_[out_id];
            if (mapped_id < 0) {
                return true;
            }
            out_id = static_cast<size_t>(mapped_id);
        }
        return out_id >= num_bits_ || (bits_[out_id >> 3] & (0x1 << (out_id & 0x7)));
    }

    float
    filter_ratio() const {
        auto current_size = size();
        return current_size == 0 ? 0.0f : ((float)count() / current_size);
    }

    // Return whether every backend id in [begin, end) is filtered.
    bool
    range_all_filtered(size_t begin, size_t end) const {
        assert(begin <= end);
        assert(end <= size());
        if (begin == end) {
            return true;
        }

        // Mapped ids require per-id tests.
        if (has_out_ids()) {
            for (size_t index = begin; index < end; ++index) {
                if (!test(index)) {
                    return false;
                }
            }
            return true;
        }

        // Contiguous ids can scan the translated public-bit range.
        const auto offset = static_cast<size_t>(id_offset_);
        const size_t lo = begin + offset;
        const size_t hi = std::min(end + offset, num_bits_);
        if (hi <= lo) {
            return true;
        }
        return all_bits_set(lo, hi);
    }

    // Return the last unfiltered backend id below upper_bound.
    std::optional<size_t>
    previous_valid_index(size_t upper_bound) const {
        if (upper_bound == 0) {
            return std::nullopt;
        }
        if (empty()) {
            return upper_bound - 1;
        }

        // Mapped ids require per-id tests.
        if (has_out_ids()) {
            size_t index = std::min(upper_bound, size());
            while (index > 0) {
                --index;
                if (!test(index)) {
                    return index;
                }
            }
            return std::nullopt;
        }

        // Contiguous ids can scan the translated public-bit range.
        const auto offset = static_cast<size_t>(id_offset_);
        const size_t low_bit = offset;
        const size_t hi_bit = std::min(offset + upper_bound, num_bits_);
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
                return (word_index << 6) + 63 - __builtin_clzll(valid) - offset;
            }
            if (word_index == low_word) {
                return std::nullopt;
            }
            --word_index;
            valid = ~load_word(word_index);
        }
    }

    // Return the first unfiltered backend id.
    size_t
    get_first_valid_index() const {
        if (has_out_ids()) {
            for (size_t i = 0; i < size(); i++) {
                if (!test(i)) {
                    return i;
                }
            }
            return size();
        }

        const size_t bit_begin = std::min(id_offset_, num_bits_);
        const size_t bit_count = std::min(size(), num_bits_ - bit_begin);
        if (bit_count == 0) {
            return size();
        }

        const size_t bit_end = bit_begin + bit_count;
        const size_t last_word = (bit_end - 1) >> 6;
        for (size_t word_index = bit_begin >> 6; word_index <= last_word; ++word_index) {
            uint64_t value = ~load_word(word_index);
            if (word_index == (bit_begin >> 6)) {
                value &= ~lower_bits_mask(bit_begin & 63);
            }
            if (word_index == last_word) {
                value &= lower_bits_mask(((bit_end - 1) & 63) + 1);
            }
            if (value != 0) {
                return (word_index << 6) + __builtin_ctzll(value) - id_offset_;
            }
        }

        return size();
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
    template <typename ValidByteAt, typename ValidWordAt>
    void
    count_filtered_bits_impl_(size_t bit_offset, size_t bit_count, bool has_valid_bitmap, ValidByteAt valid_byte_at,
                              ValidWordAt valid_word_at) {
        if (bits_ == nullptr || num_bits_ == 0 || bit_count == 0 || bit_offset >= num_bits_) {
            set_vector_count(0);
            set_filter_count(0);
            return;
        }

        const auto count_bits = std::min(bit_count, num_bits_ - bit_offset);
        const auto end_bit = bit_offset + count_bits;
        size_t bit_pos = bit_offset;
        size_t filtered_count = 0;
        size_t vector_count = 0;

        if ((bit_pos & 0x7) != 0) {
            const auto byte_idx = bit_pos >> 3;
            const auto bits_in_byte = std::min<size_t>(8 - (bit_pos & 0x7), end_bit - bit_pos);
            const auto mask = static_cast<uint8_t>(((1U << bits_in_byte) - 1) << (bit_pos & 0x7));
            auto bits = bits_[byte_idx];
            auto valid_bits = mask;
            if (has_valid_bitmap) {
                valid_bits &= valid_byte_at(byte_idx);
                bits &= valid_bits;
            } else {
                bits &= valid_bits;
            }
            vector_count += __builtin_popcount(static_cast<unsigned>(valid_bits));
            filtered_count += __builtin_popcount(static_cast<unsigned>(bits));
            bit_pos += bits_in_byte;
        }

        const auto full_bytes = (end_bit - bit_pos) >> 3;
        const auto byte_begin = bit_pos >> 3;
        const auto len_uint64 = full_bytes >> 3;
        for (size_t i = 0; i < len_uint64; ++i) {
            auto bits = load_u64_unaligned_(bits_ + byte_begin + i * sizeof(uint64_t));
            if (has_valid_bitmap) {
                auto valid_bits = valid_word_at(byte_begin + i * sizeof(uint64_t));
                vector_count += __builtin_popcountll(valid_bits);
                bits &= valid_bits;
            } else {
                vector_count += sizeof(uint64_t) * 8;
            }
            filtered_count += __builtin_popcountll(bits);
        }

        auto byte_pos = byte_begin + (len_uint64 << 3);
        const auto byte_end = byte_begin + full_bytes;
        while (byte_pos < byte_end) {
            auto bits = bits_[byte_pos];
            if (has_valid_bitmap) {
                auto valid_bits = valid_byte_at(byte_pos);
                vector_count += __builtin_popcount(static_cast<unsigned>(valid_bits));
                bits &= valid_bits;
            } else {
                vector_count += 8;
            }
            filtered_count += __builtin_popcount(static_cast<unsigned>(bits));
            ++byte_pos;
        }
        bit_pos += full_bytes << 3;

        if (bit_pos < end_bit) {
            const auto byte_idx = bit_pos >> 3;
            const auto tail_bits = end_bit - bit_pos;
            const auto mask = static_cast<uint8_t>((1U << tail_bits) - 1);
            auto bits = bits_[byte_idx];
            auto valid_bits = mask;
            if (has_valid_bitmap) {
                valid_bits &= valid_byte_at(byte_idx);
                bits &= valid_bits;
            } else {
                bits &= valid_bits;
            }
            vector_count += __builtin_popcount(static_cast<unsigned>(valid_bits));
            filtered_count += __builtin_popcount(static_cast<unsigned>(bits));
        }

        set_vector_count(vector_count);
        set_filter_count(filtered_count);
    }

    static uint64_t
    lower_bits_mask(size_t bits) {
        assert(bits <= 64);
        return bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
    }

    static uint64_t
    load_u64_unaligned_(const uint8_t* data) {
        uint64_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
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
            return load_u64_unaligned_(data);
        }

        uint64_t word = 0;
        for (size_t byte = 0; byte < remaining_bytes; ++byte) {
            word |= static_cast<uint64_t>(data[byte]) << (byte * 8);
        }
        return word;
    }

    bool
    all_bits_set(size_t bit_begin, size_t bit_end) const {
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
        for (size_t word_index = first_word + 1; word_index < last_word; ++word_index) {
            if (load_u64_unaligned_(bits_ + (word_index << 3)) != ~uint64_t{0}) {
                return false;
            }
        }
        return (load_word(last_word) & last_mask) == last_mask;
    }

    bool
    has_id_boundary_filter_() const {
        if (vector_count_ == 0) {
            return false;
        }
        if (has_out_ids()) {
            return out_ids_count_ < vector_count_;
        }
        return id_offset_ >= num_bits_ || vector_count_ > num_bits_ - id_offset_;
    }

    const uint8_t* bits_ = nullptr;
    size_t num_bits_ = 0;
    // Backend-vector count used as the filter-ratio denominator.
    size_t vector_count_ = 0;
    // Backend-vector count filtered out by bits_.
    // std::nullopt means unknown; 0 means known empty filtering.
    std::optional<size_t> filtered_count_ = std::nullopt;

    // Contiguous backend id window into the public bitset.
    size_t id_offset_ = 0;

    // Backend/local id -> public id map. Owning index maps and BF-local
    // pointer windows are both represented by IdArray without copying data.
    IdArray out_ids_;
    size_t out_ids_count_ = 0;
};

}  // namespace knowhere

#endif /* BITSET_H */
