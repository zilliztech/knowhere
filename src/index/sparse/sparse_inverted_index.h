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

#ifndef SPARSE_INVERTED_INDEX_H
#define SPARSE_INVERTED_INDEX_H

#include <queue>
#include <unordered_map>
#include <vector>

#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/expected.h"
#include "knowhere/utils.h"

namespace knowhere::sparse {

// Not thread safe, concurrent access must be protected. Concurrent read operations are allowed.
// TODO: make class thread safe so we can perform concurrent add/search.
template <typename T, typename IndPtrT = int64_t, typename IndicesT = int32_t, typename ShapeT = int64_t>
class InvertedIndex {
 public:
    explicit InvertedIndex() {
    }

    void
    SetUseWand(bool use_wand) {
        use_wand_ = use_wand;
    }

    void
    SetDropRatioBuild(float drop_ratio_build) {
        drop_ratio_build_ = drop_ratio_build;
    }

    Status
    Save(MemoryIOWriter& writer) {
        writeBinaryPOD(writer, use_wand_);
        writeBinaryPOD(writer, drop_ratio_build_);
        writeBinaryPOD(writer, n_rows_);
        writeBinaryPOD(writer, n_cols_);
        writeBinaryPOD(writer, nnz_);
        for (size_t i = 0; i <= n_rows_; ++i) {
            writeBinaryPOD(writer, indptr_[i]);
        }
        for (size_t i = 0; i < nnz_; ++i) {
            writeBinaryPOD(writer, indices_[i]);
            writeBinaryPOD(writer, data_[i]);
        }
        for (size_t i = 0; i < n_cols_; ++i) {
            auto lut = inverted_lut_[i];
            writeBinaryPOD(writer, lut.size());
            for (auto [idx, val] : lut) {
                writeBinaryPOD(writer, idx);
                writeBinaryPOD(writer, val);
            }
        }
        if (use_wand_) {
            for (size_t i = 0; i < n_cols_; ++i) {
                writeBinaryPOD(writer, max_in_dim_[i]);
            }
        }
        return Status::success;
    }

    Status
    Load(MemoryIOReader& reader) {
        readBinaryPOD(reader, use_wand_);
        readBinaryPOD(reader, drop_ratio_build_);
        readBinaryPOD(reader, n_rows_);
        readBinaryPOD(reader, n_cols_);
        readBinaryPOD(reader, nnz_);
        indptr_.resize(n_rows_ + 1);
        for (size_t i = 0; i <= n_rows_; ++i) {
            readBinaryPOD(reader, indptr_[i]);
        }
        indices_.resize(nnz_);
        data_.resize(nnz_);
        for (size_t i = 0; i < nnz_; ++i) {
            readBinaryPOD(reader, indices_[i]);
            readBinaryPOD(reader, data_[i]);
        }
        inverted_lut_.resize(n_cols_);
        for (size_t i = 0; i < n_cols_; ++i) {
            size_t lut_size;
            readBinaryPOD(reader, lut_size);
            inverted_lut_[i].resize(lut_size);
            for (size_t j = 0; j < lut_size; ++j) {
                readBinaryPOD(reader, inverted_lut_[i][j].id);
                readBinaryPOD(reader, inverted_lut_[i][j].distance);
            }
        }
        if (use_wand_) {
            max_in_dim_.resize(n_cols_);
            for (size_t i = 0; i < n_cols_; ++i) {
                readBinaryPOD(reader, max_in_dim_[i]);
            }
        }
        return Status::success;
    }

    Status
    Add(const void* csr_matrix) {
        size_t rows, cols, nnz;
        const IndPtrT* indptr;
        const IndicesT* indices;
        const T* data;
        parse_csr_matrix(csr_matrix, rows, cols, nnz, indptr, indices, data);

        for (size_t i = 0; i < rows + 1; ++i) {
            indptr_.push_back(nnz_ + indptr[i]);
        }

        // TODO: benchmark performance: for growing segments with lots of small
        // csr_matrix to add, it may be better to rely on the vector's internal
        // memory management to avoid frequent reallocations caused by reserve.
        indices_.reserve(nnz_ + nnz);
        indices_.insert(indices_.end(), indices, indices + nnz);
        data_.reserve(nnz_ + nnz);
        data_.insert(data_.end(), data, data + nnz);

        if (n_cols_ < cols) {
            n_cols_ = cols;
            inverted_lut_.resize(n_cols_);
            if (use_wand_) {
                max_in_dim_.resize(n_cols_);
            }
        }

        for (size_t i = n_rows_; i < n_rows_ + rows; ++i) {
            for (IndPtrT j = indptr_[i]; j < indptr_[i + 1]; ++j) {
                inverted_lut_[indices_[j]].emplace_back(i, data_[j]);
                if (use_wand_) {
                    max_in_dim_[indices_[j]] = std::max(max_in_dim_[indices_[j]], data_[j]);
                }
            }
        }

        n_rows_ += rows;
        nnz_ += nnz;
        return Status::success;
    }

    void
    Search(const void* query_csr_matrix, int64_t q_id, size_t k, float drop_ratio_search, float* distances,
           label_t* labels, size_t refine_factor, const BitsetView& bitset) const {
        size_t len;
        const IndicesT* indices;
        const T* data;
        get_row<T>(query_csr_matrix, q_id, len, indices, data);

        // initially set result distances to NaN and labels to -1
        std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
        std::fill(labels, labels + k, -1);
        if (len == 0) {
            return;
        }

        // if no data was dropped during both build and search, no refinement is needed.
        if (drop_ratio_build_ == 0 && drop_ratio_search == 0) {
            refine_factor = 1;
        }

        std::vector<std::pair<IndicesT, T>> q_vec(len);
        for (size_t i = 0; i < len; ++i) {
            q_vec[i] = std::make_pair(indices[i], data[i]);
        }
        std::sort(q_vec.begin(), q_vec.end(),
                  [](const auto& lhs, const auto& rhs) { return std::abs(lhs.second) > std::abs(rhs.second); });
        while (!q_vec.empty() && q_vec[0].second * drop_ratio_search > q_vec.back().second) {
            q_vec.pop_back();
        }

        MaxMinHeap<T> heap(k * refine_factor);
        if (!use_wand_) {
            search_brute_force(q_vec, heap, bitset);
        } else {
            search_wand(q_vec, heap, bitset);
        }

        // no refinement needed
        if (refine_factor == 1) {
            collect_result(heap, distances, labels);
        } else {
            // TODO tweak the map buckets number for best performance
            std::unordered_map<IndicesT, T> q_map(4 * len);
            for (size_t i = 0; i < len; ++i) {
                q_map[indices[i]] = data[i];
            }
            refine_and_collect(q_map, heap, k, distances, labels);
        }
    }

    [[nodiscard]] size_t
    size() const {
        size_t res = 0;
        res += sizeof(*this);
        res += sizeof(T) * data_.capacity();
        res += sizeof(IndicesT) * indices_.capacity();
        res += sizeof(IndPtrT) * indptr_.capacity();
        res += sizeof(std::vector<Neighbor<T>>) * inverted_lut_.capacity();
        for (auto& lut : inverted_lut_) {
            res += sizeof(Neighbor<T>) * lut.capacity();
        }
        if (use_wand_) {
            res += sizeof(T) * max_in_dim_.capacity();
        }
        return res;
    }

    [[nodiscard]] size_t
    n_rows() const {
        return n_rows_;
    }

    [[nodiscard]] size_t
    n_cols() const {
        return n_cols_;
    }

 private:
    [[nodiscard]] float
    dot_product(const std::unordered_map<IndicesT, T>& q_map, table_t u) const {
        float res = 0.0f;
        for (IndPtrT i = indptr_[u]; i < indptr_[u + 1]; ++i) {
            auto idx = indices_[i];
            float val = float(data_[i]);
            auto it = q_map.find(idx);
            if (it != q_map.end()) {
                res += val * it->second;
            }
        }
        return res;
    }

    // find the top-k candidates using brute force search, k as specified by the capacity of the heap.
    void
    search_brute_force(const std::vector<std::pair<IndicesT, T>>& q_vec, MaxMinHeap<T>& heap,
                       const BitsetView& bitset) const {
        std::vector<float> scores(n_rows_, 0.0f);
        for (auto [i, v] : q_vec) {
            for (size_t j = 0; j < inverted_lut_[i].size(); j++) {
                auto [idx, val] = inverted_lut_[i][j];
                scores[idx] += v * float(val);
            }
        }
        for (size_t i = 0; i < n_rows_; ++i) {
            if ((bitset.empty() || !bitset.test(i)) && scores[i] != 0) {
                heap.push(i, scores[i]);
            }
        }
    }

    class Cursor {
     public:
        Cursor(const std::vector<Neighbor<T>>& lut, size_t num_vec, float max_score, float q_value,
               const BitsetView bitset)
            : lut_(lut), num_vec_(num_vec), max_score_(max_score), q_value_(q_value), bitset_(bitset) {
            while (loc_ < lut_.size() && !bitset_.empty() && bitset_.test(cur_vec_id())) {
                loc_++;
            }
        }
        Cursor(const Cursor& rhs) = delete;

        void
        next() {
            loc_++;
            while (loc_ < lut_.size() && !bitset_.empty() && bitset_.test(cur_vec_id())) {
                loc_++;
            }
        }
        // advance loc until cur_vec_id() >= vec_id
        void
        seek(table_t vec_id) {
            while (loc_ < lut_.size() && cur_vec_id() < vec_id) {
                next();
            }
        }
        [[nodiscard]] table_t
        cur_vec_id() const {
            if (is_end()) {
                return num_vec_;
            }
            return lut_[loc_].id;
        }
        T
        cur_distance() const {
            return lut_[loc_].distance;
        }
        [[nodiscard]] bool
        is_end() const {
            return loc_ >= size();
        }
        [[nodiscard]] float
        q_value() const {
            return q_value_;
        }
        [[nodiscard]] size_t
        size() const {
            return lut_.size();
        }
        [[nodiscard]] float
        max_score() const {
            return max_score_;
        }

     private:
        const std::vector<Neighbor<T>>& lut_;
        size_t loc_ = 0;
        size_t num_vec_ = 0;
        float max_score_ = 0.0f;
        float q_value_ = 0.0f;
        const BitsetView bitset_;
    };  // class Cursor

    void
    search_wand(std::vector<std::pair<IndicesT, T>>& q_vec, MaxMinHeap<T>& heap, const BitsetView& bitset) const {
        auto q_dim = q_vec.size();
        std::vector<std::shared_ptr<Cursor>> cursors(q_dim);
        for (size_t i = 0; i < q_dim; ++i) {
            auto [idx, val] = q_vec[i];
            cursors[i] = std::make_shared<Cursor>(inverted_lut_[idx], n_rows_, max_in_dim_[idx] * val, val, bitset);
        }
        auto sort_cursors = [&cursors] {
            std::sort(cursors.begin(), cursors.end(),
                      [](auto& x, auto& y) { return x->cur_vec_id() < y->cur_vec_id(); });
        };
        sort_cursors();
        auto score_above_threshold = [&heap](float x) { return !heap.full() || x > heap.top().distance; };
        while (true) {
            float upper_bound = 0;
            size_t pivot;
            bool found_pivot = false;
            for (pivot = 0; pivot < cursors.size(); ++pivot) {
                if (cursors[pivot]->is_end()) {
                    break;
                }
                upper_bound += cursors[pivot]->max_score();
                if (score_above_threshold(upper_bound)) {
                    found_pivot = true;
                    break;
                }
            }
            if (!found_pivot) {
                break;
            }
            table_t pivot_id = cursors[pivot]->cur_vec_id();
            if (pivot_id == cursors[0]->cur_vec_id()) {
                float score = 0;
                for (auto& cursor : cursors) {
                    if (cursor->cur_vec_id() != pivot_id) {
                        break;
                    }
                    score += cursor->cur_distance() * cursor->q_value();
                    cursor->next();
                }
                heap.push(pivot_id, score);
                sort_cursors();
            } else {
                uint64_t next_list = pivot;
                for (; cursors[next_list]->cur_vec_id() == pivot_id; --next_list) {
                }
                cursors[next_list]->seek(pivot_id);
                for (size_t i = next_list + 1; i < cursors.size(); ++i) {
                    if (cursors[i]->cur_vec_id() >= cursors[i - 1]->cur_vec_id()) {
                        break;
                    }
                    std::swap(cursors[i], cursors[i - 1]);
                }
            }
        }
    }

    void
    refine_and_collect(const std::unordered_map<IndicesT, T>& q_map, MaxMinHeap<T>& inaccurate, size_t k,
                       float* distances, label_t* labels) const {
        std::priority_queue<Neighbor<T>, std::vector<Neighbor<T>>, std::greater<Neighbor<T>>> heap;

        while (!inaccurate.empty()) {
            auto [u, d] = inaccurate.top();
            inaccurate.pop();

            auto dist_acc = dot_product(q_map, u);
            if (heap.size() < k) {
                heap.emplace(u, dist_acc);
            } else if (heap.top().distance < dist_acc) {
                heap.pop();
                heap.emplace(u, dist_acc);
            }
        }
        collect_result(heap, distances, labels);
    }

    template <typename HeapType>
    void
    collect_result(HeapType& heap, float* distances, label_t* labels) const {
        int cnt = heap.size();
        for (auto i = cnt - 1; i >= 0; --i) {
            labels[i] = heap.top().id;
            distances[i] = heap.top().distance;
            heap.pop();
        }
    }

    size_t n_rows_ = 0;
    size_t n_cols_ = 0;
    size_t nnz_ = 0;
    std::vector<std::vector<Neighbor<T>>> inverted_lut_;

    std::vector<T> data_;
    std::vector<IndicesT> indices_;
    std::vector<IndPtrT> indptr_;

    bool use_wand_ = false;
    float drop_ratio_build_ = 0;
    std::vector<T> max_in_dim_;

};  // class InvertedIndex

}  // namespace knowhere::sparse

#endif  // SPARSE_INVERTED_INDEX_H
