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
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace knowhere {

class BitsetView {
 public:
    BitsetView() = default;
    ~BitsetView() = default;

    BitsetView(const uint8_t* data, size_t num_bits, int64_t filter_count = -1)
        : bits_(data), num_bits_(num_bits), vector_count_(num_bits), filtered_count_(filter_count) {
    }

    BitsetView(const std::nullptr_t) : BitsetView() {
    }

    bool
    empty() const {
        return num_bits_ == 0 || filtered_count_ == 0;
    }

    size_t
    size() const {
        return vector_count_;
    }

    int64_t
    count() const {
        if (num_bits_ == 0) {
            return 0;
        }
        assert(filtered_count_ >= 0);
        return filtered_count_;
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
            set_vector_count(0);
            set_filter_count(0);
            return;
        }

        const auto count_bits = std::min(bit_count, num_bits_ - bit_offset);
        const auto end_bit = bit_offset + count_bits;
        size_t bit_pos = bit_offset;
        size_t count = 0;
        size_t vector_count = 0;

        if ((bit_pos & 0x7) != 0) {
            const auto byte_idx = bit_pos >> 3;
            const auto bits_in_byte = std::min<size_t>(8 - (bit_pos & 0x7), end_bit - bit_pos);
            const auto mask = static_cast<uint8_t>(((1U << bits_in_byte) - 1) << (bit_pos & 0x7));
            auto bits = bits_[byte_idx];
            auto valid_bits = mask;
            if (valid_bitmap != nullptr) {
                valid_bits &= valid_bitmap[byte_idx];
                bits &= valid_bits;
            } else {
                bits &= valid_bits;
            }
            vector_count += __builtin_popcount(static_cast<unsigned>(valid_bits));
            count += __builtin_popcount(static_cast<unsigned>(bits));
            bit_pos += bits_in_byte;
        }

        const auto full_bytes = (end_bit - bit_pos) >> 3;
        const auto byte_begin = bit_pos >> 3;
        const auto len_uint64 = full_bytes >> 3;
        for (size_t i = 0; i < len_uint64; ++i) {
            auto bits = load_u64_unaligned_(bits_ + byte_begin + i * sizeof(uint64_t));
            if (valid_bitmap != nullptr) {
                auto valid_bits = load_u64_unaligned_(valid_bitmap + byte_begin + i * sizeof(uint64_t));
                vector_count += __builtin_popcountll(valid_bits);
                bits &= valid_bits;
            } else {
                vector_count += sizeof(uint64_t) * 8;
            }
            count += __builtin_popcountll(bits);
        }

        auto byte_pos = byte_begin + (len_uint64 << 3);
        const auto byte_end = byte_begin + full_bytes;
        while (byte_pos < byte_end) {
            auto bits = bits_[byte_pos];
            if (valid_bitmap != nullptr) {
                auto valid_bits = valid_bitmap[byte_pos];
                vector_count += __builtin_popcount(static_cast<unsigned>(valid_bits));
                bits &= valid_bits;
            } else {
                vector_count += 8;
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
            auto valid_bits = mask;
            if (valid_bitmap != nullptr) {
                valid_bits &= valid_bitmap[byte_idx];
                bits &= valid_bits;
            } else {
                bits &= valid_bits;
            }
            vector_count += __builtin_popcount(static_cast<unsigned>(valid_bits));
            count += __builtin_popcount(static_cast<unsigned>(bits));
        }

        set_vector_count(vector_count);
        set_filter_count(count);
    }

    void
    set_vector_count(size_t vector_count) {
        vector_count_ = vector_count;
    }

    void
    set_filter_count(size_t filter_count) {
        assert(filter_count <= static_cast<size_t>(std::numeric_limits<int64_t>::max()));
        filtered_count_ = static_cast<int64_t>(filter_count);
    }

    bool
    has_out_ids() const {
        return out_ids_ != nullptr;
    }

    void
    set_out_ids(const void* out_ids, size_t out_ids_count) {
        out_ids_ = out_ids;
        out_ids_count_ = out_ids_count;
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
            if (out_id < 0 || out_id >= static_cast<int64_t>(out_ids_count_)) {
                return true;
            }
            out_id = static_cast<const int32_t*>(out_ids_)[out_id];
        }
        // when index is larger than the max_offset, ignore it
        return (out_id < 0) || (out_id >= static_cast<int64_t>(num_bits_)) ||
               (bits_[out_id >> 3] & (0x1 << (out_id & 0x7)));
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
            for (size_t i = 0; i < out_ids_count_; i++) {
                if (!test(i)) {
                    return i;
                }
            }
            return out_ids_count_;
        }
        // if without id mapping, use a better algorithm to find the first valid index.
        size_t ret = 0;
        auto len_uint8 = byte_size();
        auto len_uint64 = len_uint8 >> 3;

        for (size_t i = 0; i < len_uint64; i++) {
            uint64_t value = ~load_u64_unaligned_(bits_ + i * sizeof(uint64_t));
            const auto bit_begin = i * 64;
            const auto remain = num_bits_ - bit_begin;
            if (remain < 64) {
                value &= (uint64_t(1) << remain) - 1;
            }
            if (value == 0) {
                continue;
            }
            ret = __builtin_ctzll(value);
            return bit_begin + ret;
        }

        // calculate remainder
        uint8_t* p_uint8 = (uint8_t*)bits_ + (len_uint64 << 3);
        for (size_t i = 0; i < len_uint8 - (len_uint64 << 3); i++) {
            const auto bit_begin = len_uint64 * 64 + i * 8;
            const auto remain = num_bits_ - bit_begin;
            auto value = static_cast<uint8_t>(~(*p_uint8));
            if (remain < 8) {
                value &= static_cast<uint8_t>((1U << remain) - 1);
            }
            if (value == 0) {
                p_uint8++;
                continue;
            }
            ret = __builtin_ctz(value);
            return bit_begin + ret;
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
    size_t vector_count_ = 0;
    int64_t filtered_count_ = -1;

    // optional. many indexes will share one bitset, requiring offset to distinguish between them.
    //  like multi-chunk brute-force in /src/common/comp/brute_force.cc, or mv-only in /src/index/hnsw/faiss_hnsw.cc
    int64_t id_offset_ = 0;  // offset of the internal ids

    // optional. bitset supports id mapping.
    // Even allows multiple ids to map to the same bit, so the number of internal ids and bits may be not equal.
    const void* out_ids_ = nullptr;
    size_t out_ids_count_ = 0;
};
}  // namespace knowhere

#endif /* BITSET_H */
