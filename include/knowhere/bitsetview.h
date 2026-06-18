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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace knowhere {

class BitsetView {
 public:
    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const uint8_t* data, size_t num_bits) : bits_(data), num_bits_(num_bits) {
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
        return has_internal_id_count_ ? num_internal_ids_ : num_bits_;
    }

    // return the number of filtered out bits. if with id mapping, return the number of filtered out ids.
    size_t
    count() const {
        return has_internal_id_count_ ? num_filtered_out_ids_ : num_filtered_out_bits_;
    }

    bool
    need_filter() const {
        return !empty() && count() != 0;
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

    void
    count_filtered_bits(size_t bit_offset, size_t bit_count, const uint8_t* valid_bitmap = nullptr) {
        if (bits_ == nullptr || num_bits_ == 0 || bit_count == 0 || bit_offset >= num_bits_) {
            return;
        }

        const auto count_bits = std::min(bit_count, num_bits_ - bit_offset);
        const auto end_bit = bit_offset + count_bits;
        size_t bit_pos = bit_offset;
        size_t count = 0;

        if ((bit_pos & 0x7) != 0) {
            const auto byte_idx = bit_pos >> 3;
            const auto bits_in_byte = std::min<size_t>(8 - (bit_pos & 0x7), end_bit - bit_pos);
            const auto mask = static_cast<uint8_t>(((1U << bits_in_byte) - 1) << (bit_pos & 0x7));
            auto bits = bits_[byte_idx];
            if (valid_bitmap != nullptr) {
                bits &= valid_bitmap[byte_idx];
            }
            count += __builtin_popcount(static_cast<unsigned>(bits & mask));
            bit_pos += bits_in_byte;
        }

        const auto full_bytes = (end_bit - bit_pos) >> 3;
        const auto byte_begin = bit_pos >> 3;
        const auto len_uint64 = full_bytes >> 3;
        for (size_t i = 0; i < len_uint64; ++i) {
            auto bits = load_u64_unaligned_(bits_ + byte_begin + i * sizeof(uint64_t));
            if (valid_bitmap != nullptr) {
                bits &= load_u64_unaligned_(valid_bitmap + byte_begin + i * sizeof(uint64_t));
            }
            count += __builtin_popcountll(bits);
        }

        auto byte_pos = byte_begin + (len_uint64 << 3);
        const auto byte_end = byte_begin + full_bytes;
        while (byte_pos < byte_end) {
            auto bits = bits_[byte_pos];
            if (valid_bitmap != nullptr) {
                bits &= valid_bitmap[byte_pos];
            }
            count += __builtin_popcount(static_cast<unsigned>(bits));
            ++byte_pos;
        }
        bit_pos += full_bytes << 3;

        if (bit_pos < end_bit) {
            const auto byte_idx = bit_pos >> 3;
            const auto tail_bits = end_bit - bit_pos;
            const auto mask = static_cast<uint8_t>((1U << tail_bits) - 1);
            auto bits = bits_[byte_idx];
            if (valid_bitmap != nullptr) {
                bits &= valid_bitmap[byte_idx];
            }
            count += __builtin_popcount(static_cast<unsigned>(bits & mask));
        }

        set_filter_count(count);
    }

    void
    set_filter_count(size_t filter_count) {
        if (has_internal_id_count_) {
            num_filtered_out_ids_ = filter_count;
        } else {
            num_filtered_out_bits_ = filter_count;
        }
    }

    bool
    has_out_ids() const {
        return out_ids_ != nullptr;
    }

    void
    set_out_ids(const void* out_ids, size_t num_internal_ids) {
        out_ids_ = out_ids;
        num_internal_ids_ = num_internal_ids;
        has_internal_id_count_ = true;
        num_filtered_out_ids_ = num_filtered_out_bits_;
    }

    const void*
    out_ids_data() const {
        return out_ids_;
    }

    void
    set_id_offset(int64_t id_offset) {
        id_offset_ = id_offset;
    }

    // if the test succeeds, then the index should be skipped during search; otherwise, it should be included.
    bool
    test(int64_t index) const {
        int64_t out_id = index + id_offset_;
        if (has_out_ids()) {
            out_id = static_cast<const uint32_t*>(out_ids_)[out_id];
        }
        // when index is larger than the max_offset, ignore it
        return (out_id >= static_cast<int64_t>(num_bits_)) || (bits_[out_id >> 3] & (0x1 << (out_id & 0x7)));
    }
    // return the filtered ratio. if with id mapping, calculated by internal_ids rather than bits.
    float
    filter_ratio() const {
        auto current_size = size();
        return current_size == 0 ? 0.0f : ((float)count() / current_size);
    }

    // return the first valid idx. if with id mapping, return the first valid internal_id.
    size_t
    get_first_valid_index() const {
        if (has_out_ids()) {
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

        for (size_t i = 0; i < len_uint64; i++) {
            uint64_t value = ~load_u64_unaligned_(bits_ + i * sizeof(uint64_t));
            if (value == 0) {
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
    load_u64_unaligned_(const uint8_t* data) {
        uint64_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    const uint8_t* bits_ = nullptr;
    size_t num_bits_ = 0;
    size_t num_filtered_out_bits_ = 0;

    // optional. many indexes will share one bitset, requiring offset to distinguish between them.
    //  like multi-chunk brute-force in /src/common/comp/brute_force.cc, or mv-only in /src/index/hnsw/faiss_hnsw.cc
    int64_t id_offset_ = 0;  // offset of the internal ids

    // optional. bitset supports id mapping.
    // Even allows multiple ids to map to the same bit, so the number of internal ids and bits may be not equal.
    const void* out_ids_ = nullptr;
    size_t num_internal_ids_ = 0;
    size_t num_filtered_out_ids_ = 0;
    bool has_internal_id_count_ = false;
};
}  // namespace knowhere

#endif /* BITSET_H */
