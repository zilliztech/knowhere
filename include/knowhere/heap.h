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

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

namespace knowhere {

/// Top-k result heap with efficient sift-down and threshold support.
///
/// Compare is the comparator passed to std::push_heap:
///   - std::less<DisT> (default): max-heap (largest on top) → keeps k smallest (dense: min distance)
///   - std::greater<DisT>: min-heap (smallest on top) → keeps k largest (sparse: max score)
template <typename DisT, typename IdT, typename Compare = std::less<DisT>>
class ResultHeap {
 public:
    using entry_type = std::pair<DisT, IdT>;

    explicit ResultHeap(std::size_t k) : k_(k) {
        q_.reserve(k_ + 1);
    }

    /// Attempts to insert an entry. Returns true if inserted, false if below threshold.
    bool
    Push(DisT val, IdT id) {
        if (!WouldEnter(val)) {
            return false;
        }
        q_.emplace_back(val, id);
        if (q_.size() <= k_) [[unlikely]] {
            std::push_heap(q_.begin(), q_.end(), heap_order);
            if (q_.size() == k_) [[unlikely]] {
                threshold_ = q_.front().first;
                threshold_valid_ = true;
            }
        } else {
            std::iter_swap(q_.begin(), std::prev(q_.end()));
            q_.pop_back();
            sift_down(q_.begin(), q_.end());
            threshold_ = q_.front().first;
        }
        return true;
    }

    /// Checks if an entry with the given value would be inserted.
    /// Returns true if val is better than the current worst in the heap.
    [[nodiscard]] bool
    WouldEnter(DisT val) const {
        return !threshold_valid_ || comp_(val, threshold_);
    }

    /// Returns the current threshold (the worst value among k results).
    /// Only meaningful when the heap is full.
    [[nodiscard]] DisT
    Threshold() const noexcept {
        return threshold_;
    }

    /// Returns true if the heap contains k elements.
    [[nodiscard]] bool
    Full() const noexcept {
        return q_.size() >= k_;
    }

    /// Pops the worst element (heap top). Compatible with old ResultMaxHeap API.
    std::optional<entry_type>
    Pop() {
        if (q_.empty()) {
            return std::nullopt;
        }
        std::pop_heap(q_.begin(), q_.end(), heap_order);
        auto result = q_.back();
        q_.pop_back();
        threshold_valid_ = false;
        return result;
    }

    /// Sorts results: best first (ascending distance for dense, descending score for sparse).
    void
    Finalize() {
        std::sort_heap(q_.begin(), q_.end(), heap_order);
    }

    /// Returns a reference to the internal result buffer.
    /// Call Finalize() first to get sorted results.
    [[nodiscard]] const std::vector<entry_type>&
    Results() const noexcept {
        return q_;
    }

    [[nodiscard]] std::size_t
    Size() const noexcept {
        return q_.size();
    }

    [[nodiscard]] std::size_t
    Capacity() const noexcept {
        return k_;
    }

    void
    Clear() noexcept {
        q_.clear();
        threshold_valid_ = false;
    }

 private:
    [[nodiscard]] static bool
    heap_order(const entry_type& lhs, const entry_type& rhs) noexcept {
        return Compare()(lhs.first, rhs.first);
    }

    using iter_type = typename std::vector<entry_type>::iterator;

    /// Custom sift-down for better performance than STL.
    /// See: https://github.com/pisa-engine/pisa/issues/504
    static void
    sift_down(iter_type first, iter_type last) {
        auto cmp = [first](std::size_t lhs, std::size_t rhs) {
            return Compare()((first + lhs)->first, (first + rhs)->first);
        };
        auto swap = [first](std::size_t lhs, std::size_t rhs) { std::iter_swap(first + lhs, first + rhs); };

        std::size_t len = std::distance(first, last);
        std::size_t idx = 0;
        std::size_t left = 0;
        std::size_t right = 0;

        while ((right = 2 * (idx + 1)) < len) {
            left = right - 1;
            auto next = idx;
            if (cmp(next, left)) {
                next = left;
            }
            if (cmp(next, right)) {
                next = right;
            }
            if (next == idx) {
                return;
            }
            swap(idx, next);
            idx = next;
        }
        if (left = 2 * idx + 1; left < len && cmp(idx, left)) {
            swap(idx, left);
        }
    }

    std::size_t k_;
    std::vector<entry_type> q_;
    DisT threshold_{};
    bool threshold_valid_ = false;
    Compare comp_;
};

/// Dense search: keeps k smallest distances (max-heap, largest on top).
template <typename DisT, typename IdT>
using ResultMaxHeap = ResultHeap<DisT, IdT, std::less<DisT>>;

/// Sparse search: keeps k largest scores (min-heap, smallest on top).
template <typename DisT, typename IdT>
using ResultMinHeap = ResultHeap<DisT, IdT, std::greater<DisT>>;

}  // namespace knowhere
