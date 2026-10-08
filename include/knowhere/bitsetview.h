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
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "knowhere/array_store.h"
#include "knowhere/candidate_evaluator.h"
#include "knowhere/candidate_evaluator_execution.h"

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

    // Separate, non-owning search predicate attachment. Bitmap operations and
    // counts describe mandatory exclusions only; they never evaluate this
    // predicate. Bind once at each independent CPU search-task boundary.
    void
    set_candidate_evaluator(const CandidateEvaluatorViewV1* evaluator) {
        if (evaluator != nullptr && !evaluator->valid()) {
            throw std::invalid_argument("invalid candidate evaluator V1");
        }
        candidate_evaluator_ = evaluator;
        execution_.reset();
    }

    // Copies the immutable filter description, creating fresh execution state
    // for ONE independent search task. Subsequent copies belong to that same
    // task and retain its state; never share a bound filter across query tasks.
    BitsetView
    bind() const {
        auto result = *this;
        result.execution_ = candidate_evaluator_ == nullptr
            ? nullptr : std::make_shared<CandidateEvaluatorExecution>(*candidate_evaluator_);
        return result;
    }

    bool
    bound() const {
        return candidate_evaluator_ == nullptr || execution_ != nullptr;
    }

    size_t
    callback_batches() const {
        return execution_ ? execution_->batch_calls : 0;
    }

    size_t
    callback_rows() const {
        return execution_ ? execution_->rows : 0;
    }

    // Representation, not a strategy decision. Raw bitmap scans/counts are
    // complete only for this form; other forms must use test().
    bool
    is_bitmap() const {
        return candidate_evaluator_ == nullptr;
    }

    const CandidateEvaluatorViewV1*
    candidate_evaluator() const {
        return candidate_evaluator_;
    }

    bool
    empty() const {
        if (candidate_evaluator_ != nullptr) {
            return false;
        }
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
        if (candidate_evaluator_ != nullptr) {
            throw std::logic_error("dynamic filter cannot be consumed as a complete raw bitmap");
        }
        return bits_;
    }

    // Transport for an adapter that ALSO preserves candidate_evaluator(). This
    // is not the complete predicate bitmap and must not replace test().
    const uint8_t*
    mandatory_data() const {
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
        if (candidate_evaluator_ != nullptr) {
            if (index < 0 || index > std::numeric_limits<int32_t>::max()) {
                return true;
            }
            const int32_t id = static_cast<int32_t>(index);
            return test(&id, 1) != 0;
        }
        return test_mandatory(index);
    }

 private:
    // Internal prefilter only: true means excluded by mandatory visibility.
    // Does not evaluate the callback and is NOT a complete predicate test.
    bool
    test_mandatory(int64_t index) const {
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
        if (num_bits_ == 0) {
            return vector_count_ != 0 && internal_id >= vector_count_;
        }
        return out_id >= num_bits_ || (bits_[out_id >> 3] & (0x1 << (out_id & 0x7)));
    }

 public:
    float
    filter_ratio() const {
        auto current_size = size();
        return current_size == 0 ? 0.0f : ((float)count() / current_size);
    }

    // Batch form of test: bit i is set when backend ID ids[i] is excluded.
    // The bound filter owns its private execution state. Bitmap counts remain
    // mandatory-only; an all-visible bitmap still runs the callback.
    uint64_t
    test(const int32_t* ids, uint32_t count) const {
        return test_rows(ids, count, !has_out_ids() && id_offset_ == 0, [&](int32_t id) -> int64_t {
            if (id < 0 || (num_bits_ == 0 && vector_count_ != 0 && static_cast<size_t>(id) >= vector_count_)) {
                return -1;
            }
            size_t row = static_cast<size_t>(id) + id_offset_;
            if (has_out_ids()) {
                return row < out_ids_count_ ? out_ids_[row] : -1;
            }
            return row;
        });
    }

    // Adapter entry for backends with their own reorder map. This map replaces
    // the view's ID projection: its output is already in the public bitmap /
    // segment-row domain. The non-owning map is read once per active input;
    // no allocation, intermediate ID pass or extra indirect call is needed.
    uint64_t
    test_mapped(const int32_t* ids, uint32_t count, const int32_t* row_map, size_t id_count) const {
        if (row_map == nullptr) {
            return test_rows(ids, count, true, [=](int32_t id) -> int32_t {
                return id < 0 || static_cast<size_t>(id) >= id_count ? -1 : id;
            });
        }
        return test_rows(ids, count, false, [=](int32_t id) -> int32_t {
            if (id < 0 || static_cast<size_t>(id) >= id_count) {
                return -1;
            }
            return row_map[id];
        });
    }

 private:
    template <typename RowAt>
    uint64_t
    test_rows(const int32_t* ids, uint32_t count, bool borrow_ids, RowAt row_at) const {
        const auto lanes = CandidateEvaluatorExecution::LaneMask(count);
        if ((count != 0 && ids == nullptr) || !bound()) {
            throw std::invalid_argument("ann_fusing: invalid batch IDs or unbound filter");
        }
        // Choose once per batch, not once per lane. An absent mandatory bitmap
        // is common for all-visible sealed segments, but a zero backend count
        // is NOT enough to omit a public-domain bitmap (see below).
        if (num_bits_ == 0) {
            return test_rows_impl<false, false>(ids, count, lanes, borrow_ids, row_at);
        }
        // With an identity projection and a count covering the entire public
        // domain, an exact zero proves that the bitmap contains no exclusions.
        // A window or compacted/mapped count cannot establish this property.
        // Still check public row bounds, including an adapter's reorder map.
        if (vector_count_ == num_bits_ && id_offset_ == 0 && !has_out_ids() && filtered_count_.value_or(1) == 0) {
            return test_rows_impl<true, false>(ids, count, lanes, borrow_ids, row_at);
        }
        return test_rows_impl<true, true>(ids, count, lanes, borrow_ids, row_at);
    }

    template <bool CheckBoundary, bool CheckBitmap, typename RowAt>
    uint64_t
    test_rows_impl(const int32_t* ids, uint32_t count, uint64_t lanes, bool borrow_ids, RowAt row_at) const {
        // Most candidate IDs survive mandatory visibility. Clear rejected
        // lanes only, avoiding a shift/OR for every normally active candidate.
        uint64_t active = lanes;
        std::array<int32_t, 64> rows;
        for (uint32_t lane = 0; lane < count; ++lane) {
            const auto row = row_at(ids[lane]);
            if (row < 0 || (CheckBoundary && static_cast<size_t>(row) >= num_bits_)) {
                active &= ~(uint64_t{1} << lane);
                continue;
            }
            if constexpr (CheckBitmap) {
                // A zero backend-domain count need not mean all public rows
                // are visible (e.g. a sparse reorder map or a public-ID scan).
                if (bits_[row >> 3] & (uint8_t{1} << (row & 7))) {
                    active &= ~(uint64_t{1} << lane);
                    continue;
                }
            }
            if (!is_bitmap() && row > std::numeric_limits<int32_t>::max()) {
                throw std::out_of_range("ann_fusing: segment row exceeds callback ID domain");
            }
            if (!borrow_ids) {
                rows[lane] = static_cast<int32_t>(row);
            }
        }
        return is_bitmap() ? lanes & ~active : execution_->test(borrow_ids ? ids : rows.data(), count, active);
    }

 public:
    // Return whether every backend id in [begin, end) is filtered.
    bool
    range_all_filtered(size_t begin, size_t end) const {
        assert(begin <= end);
        assert(end <= size());
        if (begin == end) {
            return true;
        }

        // Mapped ids require per-id tests.
        if (has_out_ids() || candidate_evaluator_ != nullptr) {
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
        if (has_out_ids() || candidate_evaluator_ != nullptr) {
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
        if (has_out_ids() || candidate_evaluator_ != nullptr) {
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

    const CandidateEvaluatorViewV1* candidate_evaluator_ = nullptr;
    std::shared_ptr<CandidateEvaluatorExecution> execution_;
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
