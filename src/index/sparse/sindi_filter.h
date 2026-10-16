// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted::sindi {

template <typename Value>
struct PostingView {
    const uint16_t* ids;
    const Value* values;
    size_t count;

    bool
    empty() const noexcept {
        return count == 0;
    }
};

// Query-local scratch shared by all terms/windows. Posting IDs remain window-local;
// compaction never changes the posting cursors or the U8 overflow sidecar offsets.
template <typename Value>
class PostingFilter {
 public:
    bool
    prepare(const BitsetView& bitset, size_t docid_start, size_t window_size, size_t max_eligible) {
        eligible_ids_.clear();
        if (bitset.empty()) {
            return false;
        }
        docid_start_ = docid_start;
        direct_ = bitset.supports_direct_test(docid_start + window_size);

        // Use local density: tenant/time filters can have very different densities
        // in adjacent windows even when the query-wide filter count is known.
        if (direct_) {
            const size_t end = docid_start + window_size;
            size_t eligible = 0;
            size_t pos = docid_start;
            // Stop counting as soon as the fixed survivor limit is exceeded.
            // Dense/all-pass masks usually stop within
            // a few words; no query-wide density assumption is needed.
            while (pos < end) {
                if ((pos & 7) == 0 && end - pos >= 64) {
                    uint64_t word;
                    std::memcpy(&word, bitset.data() + pos / 8, sizeof(word));
                    eligible += __builtin_popcountll(~word);
                    pos += 64;
                } else {
                    eligible += !bitset.test_direct(pos++);
                }
                if (eligible > max_eligible) {
                    return false;
                }
            }
            eligible_ids_.resize(eligible);
            size_t selected = 0;
            pos = docid_start;
            while (pos < end) {
                if ((pos & 7) == 0 && end - pos >= 64) {
                    uint64_t word;
                    std::memcpy(&word, bitset.data() + pos / 8, sizeof(word));
                    if constexpr (std::endian::native == std::endian::big) {
                        word = __builtin_bswap64(word);
                    }
                    uint64_t bits = ~word;
                    while (bits != 0) {
                        eligible_ids_[selected++] = static_cast<uint16_t>(pos + __builtin_ctzll(bits) - docid_start);
                        bits &= bits - 1;
                    }
                    pos += 64;
                } else {
                    if (!bitset.test_direct(pos)) {
                        eligible_ids_[selected++] = static_cast<uint16_t>(pos - docid_start);
                    }
                    ++pos;
                }
            }
        } else {
            // Preserve offsets, mapped IDs, and out-of-range exclusions. Stop as
            // soon as we know this window should use the unfiltered SIMD kernel.
            for (size_t id = 0; id < window_size; ++id) {
                if (!bitset.test(static_cast<int64_t>(docid_start + id))) {
                    eligible_ids_.push_back(static_cast<uint16_t>(id));
                    if (eligible_ids_.size() > max_eligible) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    // Disabled selection borrows the original postings without copying. A
    // filtered view borrows scratch until the next enabled select() call.
    PostingView<Value>
    select(const BitsetView& bitset, PostingView<Value> postings, bool enabled) {
        if (!enabled) {
            return postings;
        }
        const auto count = compact(bitset, postings.ids, postings.values, postings.count);
        return {ids_.data(), values_.data(), count};
    }

 private:
    size_t
    compact(const BitsetView& bitset, const uint16_t* ids, const Value* values, size_t count) {
        const size_t capacity = std::min(count, eligible_ids_.size());
        ids_.resize(capacity);
        values_.resize(capacity);
        size_t selected = 0;
        if (capacity == 0) {
            return 0;
        }
        // Serialized posting sections can start at odd byte offsets. Avoid
        // typed loads (including std::lower_bound's dereference) on those views.
        const auto* id_bytes = reinterpret_cast<const uint8_t*>(ids);
        const auto* value_bytes = reinterpret_cast<const uint8_t*>(values);
        const auto load_id = [&](size_t offset) {
            uint16_t id;
            std::memcpy(&id, id_bytes + offset * sizeof(uint16_t), sizeof(id));
            return id;
        };
        const auto append = [&](uint16_t id, size_t offset) {
            ids_[selected] = id;
            std::memcpy(&values_[selected++], value_bytes + offset * sizeof(Value), sizeof(Value));
        };

        if (count >= 16 * eligible_ids_.size()) {
            // Very sparse survivors and a long sorted posting list: seek to each
            // eligible ID instead of testing every rejected posting.
            size_t next = 0;
            for (const auto id : eligible_ids_) {
                size_t remaining = count - next;
                while (remaining != 0) {
                    const size_t half = remaining / 2;
                    const size_t mid = next + half;
                    if (load_id(mid) < id) {
                        next = mid + 1;
                        remaining -= half + 1;
                    } else {
                        remaining = half;
                    }
                }
                if (next == count) {
                    break;
                }
                if (load_id(next) == id) {
                    append(id, next++);
                }
            }
        } else {
            const auto scan = [&]<bool Direct>() {
                for (size_t i = 0; i < count; ++i) {
                    const auto local_id = load_id(i);
                    const auto id = docid_start_ + local_id;
                    const bool excluded = Direct ? bitset.test_direct(id) : bitset.test(static_cast<int64_t>(id));
                    if (!excluded) {
                        append(local_id, i);
                    }
                }
            };
            if (direct_) {
                scan.template operator()<true>();
            } else {
                scan.template operator()<false>();
            }
        }
        return selected;
    }

    size_t docid_start_ = 0;
    bool direct_ = false;
    std::vector<uint16_t> eligible_ids_;
    std::vector<uint16_t> ids_;
    std::vector<Value> values_;
};

}  // namespace knowhere::sparse::inverted::sindi
