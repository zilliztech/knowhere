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

#include <cassert>
#include <cstdint>
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
    set_out_ids(const uint32_t* out_ids, size_t num_internal_ids, size_t num_filtered_out_ids = 0) {
        out_ids_ = out_ids;
        num_internal_ids_ = num_internal_ids;
        if (num_filtered_out_ids > 0) {
            num_filtered_out_ids_ = num_filtered_out_ids;
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
