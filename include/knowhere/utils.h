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

#include <strings.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

#include "knowhere/binaryset.h"
#include "knowhere/dataset.h"

namespace knowhere {

extern const float FloatAccuracy;

inline bool
IsMetricType(const std::string& str, const knowhere::MetricType& metric_type) {
    return !strcasecmp(str.data(), metric_type.c_str());
}

inline bool
IsFlatIndex(const knowhere::IndexType& index_type) {
    static std::vector<knowhere::IndexType> flat_index_list = {IndexEnum::INDEX_FAISS_IDMAP,
                                                               IndexEnum::INDEX_FAISS_GPU_IDMAP};
    return std::find(flat_index_list.begin(), flat_index_list.end(), index_type) != flat_index_list.end();
}

extern float
NormalizeVec(float* x, int32_t d);

extern std::vector<float>
NormalizeVecs(float* x, size_t rows, int32_t dim);

extern void
Normalize(const DataSet& dataset);

extern std::unique_ptr<float[]>
CopyAndNormalizeVecs(const float* x, size_t rows, int32_t dim);

constexpr inline uint64_t seed = 0xc70f6907UL;

inline uint64_t
hash_vec(const float* x, size_t d) {
    uint64_t h = seed;
    for (size_t i = 0; i < d; ++i) {
        h = h * 13331 + *(uint32_t*)(x + i);
    }
    return h;
}

inline uint64_t
hash_binary_vec(const uint8_t* x, size_t d) {
    size_t len = (d + 7) / 8;
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h = h * 13331 + x[i];
    }
    return h;
}

template <typename T>
inline T
round_down(const T value, const T align) {
    return value / align * align;
}

extern void
ConvertIVFFlat(const BinarySet& binset, const MetricType metric_type, const uint8_t* raw_data, const size_t raw_size);

bool
UseDiskLoad(const std::string& index_type, const int32_t& /*version*/);

template <typename T, typename W>
static void
writeBinaryPOD(W& out, const T& podRef) {
    out.write((char*)&podRef, sizeof(T));
}

template <typename T, typename R>
static void
readBinaryPOD(R& in, T& podRef) {
    in.read((char*)&podRef, sizeof(T));
}

// utilities for sparse index
namespace sparse {

// type used to represent the id of a vector in the index interface.
// this is same as other index types.
using label_t = int64_t;
// type used to represent the id of a vector inside the index.
using table_t = uint32_t;

/**
CSR(Compressed Sparse Row) Matrix Format:

   +------------+-----------+------------+------------+--------------+-------------+-------------+
   |            |    rows   |    cols    |    nnz     |   indptr     |   indices   |   data      |
   +------------+-----------+------------+------------+--------------+-------------+-------------+
   |    Type    |   ShapeT  |   ShapeT   |   ShapeT   |    IntPtrT   |   IndicesT  |   ValueT    |
   +------------+-----------+------------+------------+--------------+-------------+-------------+
   | elem count |     1     |     1      |     1      |    rows + 1  |     nnz     |     nnz     |
   +------------+-----------+------------+------------+--------------+-------------+-------------+

*/

// indptr, indices and data references the original data, so they should not be freed by the caller.
// csr_matrix must outlive them.
template <typename ValueT, typename IndPtrT = int64_t, typename IndicesT = int32_t, typename ShapeT = int64_t>
void
parse_csr_matrix(const void* csr_matrix, size_t& rows, size_t& cols, size_t& nnz, const IndPtrT*& indptr,
                 const IndicesT*& indices, const ValueT*& data) {
    const ShapeT* header = static_cast<const ShapeT*>(csr_matrix);
    rows = header[0];
    cols = header[1];
    nnz = header[2];

    std::size_t offset = 3 * sizeof(ShapeT);

    indptr = reinterpret_cast<const IndPtrT*>(static_cast<const char*>(csr_matrix) + offset);
    offset += (rows + 1) * sizeof(IndPtrT);

    indices = reinterpret_cast<const IndicesT*>(static_cast<const char*>(csr_matrix) + offset);
    offset += nnz * sizeof(IndicesT);

    data = reinterpret_cast<const ValueT*>(static_cast<const char*>(csr_matrix) + offset);
}

// indices and data references the original data, so they should not be freed by the caller.
// csr_matrix must outlive them.
template <typename ValueT, typename IndPtrT = int64_t, typename IndicesT = int32_t, typename ShapeT = int64_t>
void
get_row(const void* csr_matrix, size_t idx, size_t& len, const IndicesT*& indices, const ValueT*& data) {
    const ShapeT* header = reinterpret_cast<const ShapeT*>(csr_matrix);
    size_t n_rows = header[0];
    if (idx >= n_rows) {
        len = 0;
        indices = nullptr;
        data = nullptr;
        return;
    }
    const IndPtrT* indptr = reinterpret_cast<const IndPtrT*>(header + 3);
    const IndicesT* csr_indices = reinterpret_cast<const IndicesT*>(indptr + n_rows + 1);
    const ValueT* csr_data = reinterpret_cast<const ValueT*>(csr_indices + header[2]);

    len = static_cast<size_t>(indptr[idx + 1] - indptr[idx]);
    indices = const_cast<IndicesT*>(&csr_indices[indptr[idx]]);
    data = const_cast<ValueT*>(&csr_data[indptr[idx]]);
}

template <typename dist_t = float>
struct Neighbor {
    table_t id;
    dist_t distance;

    Neighbor() = default;
    Neighbor(table_t id, dist_t distance) : id(id), distance(distance) {
    }

    inline friend bool
    operator<(const Neighbor& lhs, const Neighbor& rhs) {
        return lhs.distance < rhs.distance || (lhs.distance == rhs.distance && lhs.id < rhs.id);
    }
    inline friend bool
    operator>(const Neighbor& lhs, const Neighbor& rhs) {
        return !(lhs < rhs);
    }
};

// when pushing new elements into a MinMaxHeap, only the `capacity` smallest elements are kept.
// pop()/top() returns the largest element out of those `capacity` smallest elements.
template <typename T = float>
class MinMaxHeap {
 public:
    explicit MinMaxHeap(int capacity) : capacity_(capacity), pool_(capacity) {
    }
    void
    push(table_t id, T dist) {
        if (size_ < capacity_) {
            pool_[size_] = {id, dist};
            std::push_heap(pool_.begin(), pool_.begin() + ++size_);
        } else if (dist < pool_[0].distance) {
            sift_down(id, dist);
        }
    }
    table_t
    pop() {
        std::pop_heap(pool_.begin(), pool_.begin() + size_--);
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
    Neighbor<T>
    top() const {
        return pool_[0];
    }
    [[nodiscard]] bool
    full() const {
        return size_ == capacity_;
    }

 private:
    void
    sift_down(table_t id, T dist) {
        size_t i = 0;
        for (; 2 * i + 1 < size_;) {
            size_t j = i;
            size_t l = 2 * i + 1, r = 2 * i + 2;
            if (pool_[l].distance > dist) {
                j = l;
            }
            if (r < size_ && pool_[r].distance > std::max(pool_[l].distance, dist)) {
                j = r;
            }
            if (i == j) {
                break;
            }
            pool_[i] = pool_[j];
            i = j;
        }
        pool_[i] = {id, dist};
    }

    size_t size_ = 0, capacity_;
    std::vector<Neighbor<T>> pool_;
};  // class MinMaxHeap

}  // namespace sparse

}  // namespace knowhere
