// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

struct Neighbor {
    static constexpr int kChecked = 0;
    static constexpr int kValid = 1;
    static constexpr int kInvalid = 2;

    unsigned id;
    float distance;
    int status;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, int status) : id{id}, distance{distance}, status(status) {}

    inline bool
    operator<(const Neighbor& other) const {
        return distance < other.distance;
    }

    inline bool
    operator>(const Neighbor& other) const {
        return distance > other.distance;
    }
};

using IteratorMinHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>;

template <bool need_save>
class NeighborSetPopList {
 private:
    inline void
    insert_helper(const Neighbor& nbr, size_t pos) {
        const size_t active_end = offset_ + size_;
        // move
        std::memmove(&data_[pos + 1], &data_[pos], (active_end - pos) * sizeof(Neighbor));
        if (size_ < capacity_) {
            size_++;
        }

        // insert
        data_[pos] = nbr;
    }

 public:
    explicit NeighborSetPopList(size_t capacity)
            : capacity_(capacity), data_(need_save ? capacity + 1 : capacity * 2 + 1) {}

    inline bool
    insert(const Neighbor nbr, IteratorMinHeap* disqualified = nullptr) {
        if constexpr (!need_save) {
            if (offset_ + size_ >= data_.size()) {
                std::memmove(&data_[0], &data_[offset_], size_ * sizeof(Neighbor));
                offset_ = 0;
            }
        }
        const size_t begin = offset_;
        size_t pos = std::upper_bound(
                             &data_[begin], &data_[begin] + size_, nbr) -
                &data_[0];
        if (pos - begin >= capacity_) {
            if (disqualified) {
                disqualified->push(nbr);
            }
            return false;
        }
        if (size_ == capacity_ && disqualified) {
            disqualified->push(data_[begin + size_ - 1]);
        }
        insert_helper(nbr, pos);
        if constexpr (need_save) {
            if (pos < cur_) {
                cur_ = pos;
            }
        }
        return true;
    }

    inline auto
    pop() -> Neighbor {
        auto ret = data_[need_save ? cur_ : offset_];
        if constexpr (need_save) {
            data_[cur_].status = Neighbor::kChecked;
            cur_++;
            while (cur_ < size_ && data_[cur_].status == Neighbor::kChecked) {
                cur_++;
            }
        } else {
            ++offset_;
            size_--;
            if (size_ == 0) {
                offset_ = 0;
            }
        }
        return ret;
    }

    inline auto
    has_next() const -> bool {
        if constexpr (need_save) {
            return cur_ < size_;
        } else {
            return size_ > 0;
        }
    }

    inline auto
    size() const -> size_t {
        return size_;
    }

    inline auto
    cur() const -> const Neighbor& {
        if constexpr (need_save) {
            return data_[cur_];
        } else {
            return data_[offset_];
        }
    }

    inline auto
    at_search_back_dist() const -> float {
        if (size_ < capacity_) {
            return std::numeric_limits<float>::max();
        }
        return data_[offset_ + capacity_ - 1].distance;
    }

    void
    clear() {
        size_ = 0;
        cur_ = 0;
        offset_ = 0;
    }

    inline const Neighbor&
    operator[](size_t i) {
        return data_[offset_ + i];
    }

 private:
    size_t capacity_ = 0, size_ = 0, cur_ = 0, offset_ = 0;
    std::vector<Neighbor> data_;
};

class NeighborSetDoublePopList {
 public:
    explicit NeighborSetDoublePopList(size_t capacity = 0)
            : valid_ns_(capacity), invalid_ns_(capacity) {}

    // will push any neighbor that does not fit into NeighborSet to disqualified.
    // When searching for iterator, those points removed from NeighborSet may be
    // qualified candidates as the iterator iterates, thus we need to retain
    // instead of disposing them.
    bool
    insert(const Neighbor& nbr, IteratorMinHeap* disqualified = nullptr) {
        if (nbr.status == Neighbor::kValid) {
            return valid_ns_.insert(nbr, disqualified);
        } else {
            if (nbr.distance < valid_ns_.at_search_back_dist()) {
                return invalid_ns_.insert(nbr, disqualified);
            } else if (disqualified) {
                disqualified->push(nbr);
            }
        }
        return false;
    }

    auto
    pop() -> Neighbor {
        return pop_based_on_distance();
    }

    auto
    has_next() const -> bool {
        return valid_ns_.has_next() ||
               (invalid_ns_.has_next() && invalid_ns_.cur().distance < valid_ns_.at_search_back_dist());
    }

    inline const Neighbor&
    operator[](size_t i) {
        return valid_ns_[i];
    }

    inline size_t
    size() const {
        return valid_ns_.size();
    }

 private:
    auto
    pop_based_on_distance() -> Neighbor {
        bool hasCandNext = invalid_ns_.has_next();
        bool hasResNext = valid_ns_.has_next();

        if (hasCandNext && hasResNext) {
            return invalid_ns_.cur().distance < valid_ns_.cur().distance ? invalid_ns_.pop() : valid_ns_.pop();
        }
        if (hasCandNext != hasResNext) {
            return hasCandNext ? invalid_ns_.pop() : valid_ns_.pop();
        }
        return {0, 0, Neighbor::kValid};
    }

    NeighborSetPopList<true> valid_ns_;
    NeighborSetPopList<false> invalid_ns_;
};

static inline int
InsertIntoPool(Neighbor* addr, intptr_t size, Neighbor nn) {
    intptr_t p = std::lower_bound(addr, addr + size, nn) - addr;
    std::memmove(addr + p + 1, addr + p, (size - p) * sizeof(Neighbor));
    addr[p] = nn;
    return p;
}

}
}
}
