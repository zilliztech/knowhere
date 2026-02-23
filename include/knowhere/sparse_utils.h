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
#include <atomic>
#include <boost/iterator/iterator_facade.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

#include "knowhere/expected.h"
#include "knowhere/object.h"
#include "knowhere/operands.h"

namespace knowhere::sparse {

// Seek distance instrumentation (compile with -DSEEK_INSTRUMENTATION to enable)
#ifdef SEEK_INSTRUMENTATION
struct SeekStats {
    std::atomic<uint64_t> bucket_0{0};         // delta = 0
    std::atomic<uint64_t> bucket_1_3{0};       // delta 1-3
    std::atomic<uint64_t> bucket_4_15{0};      // delta 4-15
    std::atomic<uint64_t> bucket_16_63{0};     // delta 16-63
    std::atomic<uint64_t> bucket_64_255{0};    // delta 64-255
    std::atomic<uint64_t> bucket_256_plus{0};  // delta 256+
    std::atomic<uint64_t> seek_hits{0};        // seek found target doc_id
    std::atomic<uint64_t> seek_misses{0};      // seek did NOT find target doc_id

    void
    record(size_t delta) {
        if (delta == 0)
            bucket_0++;
        else if (delta <= 3)
            bucket_1_3++;
        else if (delta <= 15)
            bucket_4_15++;
        else if (delta <= 63)
            bucket_16_63++;
        else if (delta <= 255)
            bucket_64_255++;
        else
            bucket_256_plus++;
    }

    void
    record_hit() {
        seek_hits++;
    }
    void
    record_miss() {
        seek_misses++;
    }

    void
    print(const char* label = nullptr) const {
        if (label)
            printf("\n[Seek Stats: %s]\n", label);
        else
            printf("\n[Seek Distance Distribution]\n");
        uint64_t total = bucket_0 + bucket_1_3 + bucket_4_15 + bucket_16_63 + bucket_64_255 + bucket_256_plus;
        printf("  delta=0:      %lu (%.1f%%)\n", bucket_0.load(), total ? 100.0 * bucket_0 / total : 0);
        printf("  delta 1-3:    %lu (%.1f%%)\n", bucket_1_3.load(), total ? 100.0 * bucket_1_3 / total : 0);
        printf("  delta 4-15:   %lu (%.1f%%)\n", bucket_4_15.load(), total ? 100.0 * bucket_4_15 / total : 0);
        printf("  delta 16-63:  %lu (%.1f%%)\n", bucket_16_63.load(), total ? 100.0 * bucket_16_63 / total : 0);
        printf("  delta 64-255: %lu (%.1f%%)\n", bucket_64_255.load(), total ? 100.0 * bucket_64_255 / total : 0);
        printf("  delta 256+:   %lu (%.1f%%)\n", bucket_256_plus.load(), total ? 100.0 * bucket_256_plus / total : 0);
        printf("  total seeks:  %lu\n", total);
        uint64_t h = seek_hits.load(), m = seek_misses.load();
        uint64_t hm = h + m;
        printf("  seek hits:    %lu (%.1f%%)\n", h, hm ? 100.0 * h / hm : 0);
        printf("  seek misses:  %lu (%.1f%%)\n", m, hm ? 100.0 * m / hm : 0);
    }

    void
    reset() {
        bucket_0 = bucket_1_3 = bucket_4_15 = 0;
        bucket_16_63 = bucket_64_255 = bucket_256_plus = 0;
        seek_hits = seek_misses = 0;
    }
};

inline SeekStats g_seek_stats;

struct DspStats {
    std::atomic<uint64_t> total_blocks{0};      // total non-zero blocks
    std::atomic<uint64_t> blocks_processed{0};  // blocks actually scored
    std::atomic<uint64_t> blocks_pruned{0};     // blocks pruned by threshold
    std::atomic<uint64_t> entries_scored{0};    // posting list entries iterated
    std::atomic<uint64_t> docs_pushed{0};       // docs pushed to heap
    std::atomic<uint64_t> queries{0};           // number of queries

    void
    print(const char* label = nullptr) const {
        if (label)
            printf("\n[DSP Block Stats: %s]\n", label);
        else
            printf("\n[DSP Block Stats]\n");
        uint64_t q = queries.load();
        uint64_t tb = total_blocks.load(), bp = blocks_processed.load();
        uint64_t pruned = tb > bp ? tb - bp : 0;
        printf("  queries:           %lu\n", q);
        printf("  total non-zero:    %lu (avg %.1f/q)\n", tb, q ? (double)tb / q : 0);
        printf("  blocks processed:  %lu (avg %.1f/q)\n", bp, q ? (double)bp / q : 0);
        printf("  blocks pruned:     %lu (%.1f%%)\n", pruned, tb ? 100.0 * pruned / tb : 0);
        printf("  entries scored:    %lu (avg %.1f/q)\n", entries_scored.load(), q ? (double)entries_scored / q : 0);
        printf("  docs pushed:       %lu (avg %.1f/q)\n", docs_pushed.load(), q ? (double)docs_pushed / q : 0);
        if (bp > 0) {
            printf("  entries/block:     %.1f\n", (double)entries_scored / bp);
        }
    }

    void
    reset() {
        total_blocks = blocks_processed = blocks_pruned = 0;
        entries_scored = docs_pushed = queries = 0;
    }
};

inline DspStats g_dsp_stats;
#endif

enum class SparseMetricType {
    METRIC_IP = 1,
    METRIC_BM25 = 2,
};

// integer type in SparseRow
using table_t = uint32_t;
// type used to represent the id of a vector in the index interface.
// this is same as other index types.
using label_t = int64_t;

template <typename T>
using SparseIdVal = IdVal<table_t, T>;

// DocValueComputer takes a value of a doc vector and returns the a computed
// value that can be used to multiply directly with the corresponding query
// value. The second parameter is the document length of the database vector,
// which is used in BM25.
template <typename T>
using DocValueComputer = std::function<float(const T&, const float)>;

template <typename T>
auto
GetDocValueOriginalComputer() {
    static DocValueComputer<T> lambda = [](const T& right, const float) -> float { return right; };
    return lambda;
}

template <typename T>
auto
GetDocValueBM25Computer(float k1, float b, float avgdl) {
    return [k1, b, avgdl](const T& tf, const float doc_len) -> float {
        return tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avgdl)));
    };
}

// A docid filter that tests whether a given id is in the list of docids, which is regarded as another form of BitSet.
// Note that all ids to be tested must be tested exactly once and in order.
class DocIdFilterByVector {
 public:
    DocIdFilterByVector(std::vector<table_t>&& docids) : docids_(std::move(docids)) {
        std::sort(docids_.begin(), docids_.end());
    }

    [[nodiscard]] bool
    test(const table_t id) {
        // find the first id that is greater than or equal to the specific id
        while (pos_ < docids_.size() && docids_[pos_] < id) {
            ++pos_;
        }
        return !(pos_ < docids_.size() && docids_[pos_] == id);
    }

    [[nodiscard]] bool
    empty() const {
        return docids_.empty();
    }

 private:
    std::vector<table_t> docids_;
    size_t pos_ = 0;
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

    SparseRow(const std::vector<std::pair<table_t, T>>& data) : count_(data.size()), own_data_(true) {
        data_ = new uint8_t[count_ * element_size()];
        for (size_t i = 0; i < count_; ++i) {
            auto* elem = reinterpret_cast<ElementProxy*>(data_) + i;
            elem->index = data[i].first;
            elem->value = data[i].second;
        }
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

    SparseIdVal<T>
    operator[](size_t i) const {
        auto* elem = reinterpret_cast<const ElementProxy*>(data_) + i;
        return {elem->index, elem->value};
    }

    void
    set_at(size_t i, table_t index, T value) {
        if (i >= count_) {
            throw std::out_of_range("set_at on a SparseRow with invalid index");
        }
        auto* elem = reinterpret_cast<ElementProxy*>(data_) + i;
        elem->index = index;
        elem->value = value;
    }

    // In the case of asymetric distance functions, this should be the query
    // and the other should be the database vector. For example using BM25, we
    // should call query_vec.dot(doc_vec) instead of doc_vec.dot(query_vec).
    template <typename Computer = DocValueComputer<T>>
    float
    dot(const SparseRow<T>& other, Computer computer = GetDocValueOriginalComputer<T>(), const T other_sum = 0) const {
        float product_sum = 0.0f;
        size_t i = 0;
        size_t j = 0;
        // TODO: improve with _mm_cmpistrm or the AVX512 alternative.
        while (i < count_ && j < other.count_) {
            auto* left = reinterpret_cast<const ElementProxy*>(data_) + i;
            auto* right = reinterpret_cast<const ElementProxy*>(other.data_) + j;

            if (left->index < right->index) {
                ++i;
            } else if (left->index > right->index) {
                ++j;
            } else {
                product_sum += left->value * computer(right->value, other_sum);
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
            std::push_heap(pool_.begin(), pool_.begin() + size_, std::greater<SparseIdVal<T>>());
        } else if (val > pool_[0].val) {
            sift_down(id, val);
        }
    }
    table_t
    pop() {
        std::pop_heap(pool_.begin(), pool_.begin() + size_, std::greater<SparseIdVal<T>>());
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
    SparseIdVal<T>
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
    std::vector<SparseIdVal<T>> pool_;
};  // class MaxMinHeap

// A std::vector like container but uses fixed size free memory(typically from
// mmap) as backing store and can only be appended at the end.
//
// Must be initialized with a valid pointer to memory when used. The memory must be
// valid during the lifetime of this object. After initialization, GrowableVectorView will
// have space for mmap_element_count_ elements, none of which are initialized.
//
// Currently only used in sparse InvertedIndex. Move to other places if needed.
template <typename T>
class GrowableVectorView {
 public:
    using value_type = T;
    using size_type = size_t;

    GrowableVectorView() = default;

    void
    initialize(void* data, size_type byte_size) {
        if (byte_size % sizeof(T) != 0) {
            throw std::invalid_argument("GrowableVectorView byte_size must be a multiple of element size");
        }
        mmap_data_ = data;
        mmap_byte_size_ = byte_size;
        mmap_element_count_ = 0;
    }

    [[nodiscard]] size_type
    capacity() const {
        return mmap_byte_size_ / sizeof(T);
    }

    [[nodiscard]] size_type
    size() const {
        return mmap_element_count_;
    }

    template <typename... Args>
    T&
    emplace_back(Args&&... args) {
        if (size() == capacity()) {
            throw std::out_of_range("emplace_back on a full GrowableVectorView");
        }
        auto* elem = reinterpret_cast<T*>(mmap_data_) + mmap_element_count_++;
        return *new (elem) T(std::forward<Args>(args)...);
    }

    T&
    operator[](size_type i) {
        return reinterpret_cast<T*>(mmap_data_)[i];
    }

    const T&
    operator[](size_type i) const {
        return reinterpret_cast<const T*>(mmap_data_)[i];
    }

    T&
    at(size_type i) {
        if (i >= mmap_element_count_) {
            throw std::out_of_range("GrowableVectorView index out of range");
        }
        return reinterpret_cast<T*>(mmap_data_)[i];
    }

    const T*
    data() const {
        return reinterpret_cast<const T*>(mmap_data_);
    }

    const T&
    at(size_type i) const {
        if (i >= mmap_element_count_) {
            throw std::out_of_range("GrowableVectorView index out of range");
        }
        return reinterpret_cast<const T*>(mmap_data_)[i];
    }

    class iterator : public boost::iterator_facade<iterator, T, boost::random_access_traversal_tag, T&> {
     public:
        iterator() = default;
        explicit iterator(T* ptr) : ptr_(ptr) {
        }

        friend class GrowableVectorView;
        friend class boost::iterator_core_access;

        T&
        dereference() const {
            return *ptr_;
        }

        void
        increment() {
            ++ptr_;
        }

        void
        decrement() {
            --ptr_;
        }

        void
        advance(std::ptrdiff_t n) {
            ptr_ += n;
        }

        std::ptrdiff_t
        distance_to(const iterator& other) const {
            return other.ptr_ - ptr_;
        }

        bool
        equal(const iterator& other) const {
            return ptr_ == other.ptr_;
        }

     private:
        T* ptr_ = nullptr;
    };

    iterator
    begin() const {
        return iterator(reinterpret_cast<T*>(mmap_data_));
    }

    iterator
    end() const {
        return iterator(reinterpret_cast<T*>(mmap_data_) + mmap_element_count_);
    }

 private:
    void* mmap_data_ = nullptr;
    size_type mmap_byte_size_ = 0;
    size_type mmap_element_count_ = 0;
};

}  // namespace knowhere::sparse
