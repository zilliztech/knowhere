// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// valributed under the License is valributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "knowhere/operands.h"

namespace knowhere::sparse {

// integer type in SparseRow
using table_t = uint32_t;
// type used to represent the id of a vector in the index interface.
// this is same as other index types.
using label_t = int64_t;

template <typename T>
struct IdVal {
    table_t id;
    T val;

    IdVal() = default;
    IdVal(table_t id, T val) : id(id), val(val) {
    }

    inline friend bool
    operator<(const IdVal& lhs, const IdVal& rhs) {
        return lhs.val < rhs.val || (lhs.val == rhs.val && lhs.id < rhs.id);
    }
    inline friend bool
    operator>(const IdVal& lhs, const IdVal& rhs) {
        return !(lhs < rhs);
    }

    inline friend bool
    operator==(const IdVal& lhs, const IdVal& rhs) {
        return lhs.id == rhs.id && lhs.val == rhs.val;
    }
};

template <typename T>
class SparseRow {
    static_assert(std::is_same_v<T, fp32>, "SparseRow supports float only");

 public:
    // construct an SparseRow with memory allocated to hold `count` elements.
    SparseRow(size_t count = 0)
        : data_(count ? new uint8_t[count * element_size()] : nullptr), count_(count), own_data_(true) {
    }

    SparseRow(size_t count, uint8_t* data, bool own_data) : data_(data), count_(count), own_data_(own_data) {
    }

    // copy constructor and copy assignment operator perform deep copy
    SparseRow(const SparseRow<T>& other) : SparseRow(other.count_) {
        std::memcpy(data_, other.data_, data_byte_size());
    }

    SparseRow(SparseRow<T>&& other) noexcept : SparseRow() {
        swap(*this, other);
    }

    SparseRow&
    operator=(const SparseRow<T>& other) {
        if (this != &other) {
            SparseRow<T> tmp(other);
            swap(*this, tmp);
        }
        return *this;
    }

    SparseRow&
    operator=(SparseRow<T>&& other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~SparseRow() {
        if (own_data_ && data_ != nullptr) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    size_t
    size() const {
        return count_;
    }

    size_t
    memory_usage() const {
        return data_byte_size() + sizeof(*this);
    }

    // return the number of bytes used by the underlying data array.
    size_t
    data_byte_size() const {
        return count_ * element_size();
    }

    void*
    data() {
        return data_;
    }

    const void*
    data() const {
        return data_;
    }

    // dim of a sparse vector is the max index + 1, or 0 for an empty vector.
    int64_t
    dim() const {
        if (count_ == 0) {
            return 0;
        }
        auto* elem = reinterpret_cast<const ElementProxy*>(data_) + count_ - 1;
        return elem->index + 1;
    }

    IdVal<T>
    operator[](size_t i) const {
        auto* elem = reinterpret_cast<const ElementProxy*>(data_) + i;
        return {elem->index, elem->value};
    }

    void
    set_at(size_t i, table_t index, T value) {
        auto* elem = reinterpret_cast<ElementProxy*>(data_) + i;
        elem->index = index;
        elem->value = value;
    }

    float
    dot(const SparseRow<T>& other) const {
        float product_sum = 0.0f;
        size_t i = 0, j = 0;
        // TODO: improve with _mm_cmpistrm or the AVX512 alternative.
        while (i < count_ && j < other.count_) {
            auto* left = reinterpret_cast<const ElementProxy*>(data_) + i;
            auto* right = reinterpret_cast<const ElementProxy*>(other.data_) + j;

            if (left->index < right->index) {
                ++i;
            } else if (left->index > right->index) {
                ++j;
            } else {
                product_sum += left->value * right->value;
                ++i;
                ++j;
            }
        }
        return product_sum;
    }

    friend void
    swap(SparseRow<T>& left, SparseRow<T>& right) {
        using std::swap;
        swap(left.count_, right.count_);
        swap(left.data_, right.data_);
        swap(left.own_data_, right.own_data_);
    }

    static inline size_t
    element_size() {
        return sizeof(table_t) + sizeof(T);
    }

 private:
    // ElementProxy is used to access elements in the data_ array and should
    // never be actually constructed.
    struct __attribute__((packed)) ElementProxy {
        table_t index;
        T value;
        ElementProxy() = delete;
        ElementProxy(const ElementProxy&) = delete;
    };
    // data_ must be sorted by column id. use raw pointer for easy mmap and zero
    // copy.
    uint8_t* data_;
    size_t count_;
    bool own_data_;
};

// When pushing new elements into a MaxMinHeap, only `capacity` elements with the
// largest val are kept. pop()/top() returns the smallest element out of them.
template <typename T>
class MaxMinHeap {
 public:
    explicit MaxMinHeap(int capacity) : capacity_(capacity), pool_(capacity) {
    }
    void
    push(table_t id, T val) {
        if (size_ < capacity_) {
            pool_[size_] = {id, val};
            size_ += 1;
            std::push_heap(pool_.begin(), pool_.begin() + size_, std::greater<IdVal<T>>());
        } else if (val > pool_[0].val) {
            sift_down(id, val);
        }
    }
    table_t
    pop() {
        std::pop_heap(pool_.begin(), pool_.begin() + size_, std::greater<IdVal<T>>());
        size_ -= 1;
        return pool_[size_].id;
    }
    [[nodiscard]] size_t
    size() const {
        return size_;
    }
    [[nodiscard]] bool
    empty() const {
        return size() == 0;
    }
    IdVal<T>
    top() const {
        return pool_[0];
    }
    [[nodiscard]] bool
    full() const {
        return size_ == capacity_;
    }

 private:
    void
    sift_down(table_t id, T val) {
        size_t i = 0;
        for (; 2 * i + 1 < size_;) {
            size_t j = i;
            size_t l = 2 * i + 1, r = 2 * i + 2;
            if (pool_[l].val < val) {
                j = l;
            }
            if (r < size_ && pool_[r].val < std::min(pool_[l].val, val)) {
                j = r;
            }
            if (i == j) {
                break;
            }
            pool_[i] = pool_[j];
            i = j;
        }
        pool_[i] = {id, val};
    }

    size_t size_ = 0, capacity_;
    std::vector<IdVal<T>> pool_;
};  // class MaxMinHeap

}  // namespace knowhere::sparse
