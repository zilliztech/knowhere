#pragma once

#include <algorithm>
#include <cstring>

namespace hnswlib {

struct Neighbor {
    static constexpr int kChecked = 0;
    static constexpr int kValid = 1;
    static constexpr int kInvalid = 2;

    unsigned id;
    float distance;
    int status;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, int status) : id{id}, distance{distance}, status(status) {
    }

    inline bool
    operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
    inline bool
    operator>(const Neighbor& other) const {
        return distance > other.distance;
    }
};
typedef std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> IteratorMinHeap;
class NeighborSet {
 public:
    explicit NeighborSet(size_t capacity = 0) : capacity_(capacity), data_(capacity_ + 1) {
    }

    // will push any neighbor that does not fit into NeighborSet to disqualified.
    // When searching for iterator, those points removed from NeighborSet may be
    // qualified candidates as the iterator iterates, thus we need to retain
    // instead of disposing them.
    bool
    insert(Neighbor nbr, IteratorMinHeap* disqualified = nullptr) {
        if (size_ == capacity_ && nbr.distance >= data_[size_ - 1].distance) {
            if (disqualified) {
                disqualified->push(nbr);
            }
            return false;
        }
        if (size_ == capacity_ && disqualified) {
            disqualified->push(data_[size_ - 1]);
        }
        int lo = 0, hi = size_;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (data_[mid].distance > nbr.distance) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor));
        data_[lo] = nbr;
        if (size_ < capacity_) {
            size_++;
        }
        if (lo < cur_) {
            cur_ = lo;
        }
        return true;
    }

    Neighbor
    pop() {
        auto ret = data_[cur_];
        if (data_[cur_].status == Neighbor::kValid) {
            data_[cur_].status = Neighbor::kChecked;
        } else if (data_[cur_].status == Neighbor::kInvalid) {
            std::memmove(&data_[cur_], &data_[cur_ + 1], (size_ - cur_ - 1) * sizeof(Neighbor));
            size_--;
        }
        while (cur_ < size_ && data_[cur_].status == Neighbor::kChecked) {
            cur_++;
        }
        return ret;
    }

    bool
    has_next() const {
        return cur_ < size_;
    }

    size_t
    size() const {
        return size_;
    }
    size_t
    capacity() const {
        return capacity_;
    }

    Neighbor&
    operator[](size_t i) {
        return data_[i];
    }

    const Neighbor&
    operator[](size_t i) const {
        return data_[i];
    }

    void
    clear() {
        size_ = 0;
        cur_ = 0;
    }

 private:
    size_t size_ = 0;
    size_t capacity_;
    size_t cur_ = 0;
    std::vector<Neighbor> data_;
};

static inline int
InsertIntoPool(Neighbor* addr, int size, Neighbor nn) {
    int p = std::lower_bound(addr, addr + size, nn) - addr;
    std::memmove(addr + p + 1, addr + p, (size - p) * sizeof(Neighbor));
    addr[p] = nn;
    return p;
}

}  // namespace hnswlib
