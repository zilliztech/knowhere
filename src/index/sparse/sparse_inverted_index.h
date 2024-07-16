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

#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

#include "index/sparse/sparse_inverted_index_config.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere::sparse {
template <typename T>
class BaseInvertedIndex {
 public:
    virtual ~BaseInvertedIndex() = default;

    virtual Status
    Save(MemoryIOWriter& writer) = 0;

    virtual Status
    Load(MemoryIOReader& reader, bool is_mmap) = 0;

    virtual Status
    Train(const SparseRow<T>* data, size_t rows, float drop_ratio_build) = 0;

    virtual Status
    Add(const SparseRow<T>* data, size_t rows, int64_t dim) = 0;

    virtual void
    Search(const SparseRow<T>& query, size_t k, float drop_ratio_search, float* distances, label_t* labels,
           size_t refine_factor, const BitsetView& bitset, const DocValueComputer<T>& computer) const = 0;

    virtual std::vector<float>
    GetAllDistances(const SparseRow<T>& query, float drop_ratio_search, const BitsetView& bitset,
                    const DocValueComputer<T>& computer) const = 0;

    virtual void
    GetVectorById(const label_t id, SparseRow<T>& output) const = 0;

    virtual expected<DocValueComputer<T>>
    GetDocValueComputer(const SparseInvertedIndexConfig& cfg) const = 0;

    [[nodiscard]] virtual size_t
    size() const = 0;

    [[nodiscard]] virtual size_t
    n_rows() const = 0;

    [[nodiscard]] virtual size_t
    n_cols() const = 0;
};

template <typename T, bool use_wand = false, bool bm25 = false>
class InvertedIndex : public BaseInvertedIndex<T> {
 public:
    explicit InvertedIndex() {
    }

    void
    SetBM25Params(float k1, float b, float avgdl, float max_score_ratio) {
        bm25_params_ = std::make_unique<BM25Params>(k1, b, avgdl, max_score_ratio);
    }

    expected<DocValueComputer<T>>
    GetDocValueComputer(const SparseInvertedIndexConfig& cfg) const override {
        // Now milvus doesn't require the user to set metric type in search request,
        // and cfg.metric_type will default to L2 in that case. We should not throw
        // error in that case. Milvus will check if the metric type(if provided in
        // search request) matches the index metric type so here we assume the
        // metric type will be valid.
        if constexpr (!bm25) {
            return GetDocValueOriginalComputer<T>();
        }
        // avgdl must be supplied during search
        if (!cfg.bm25_avgdl.has_value()) {
            return expected<DocValueComputer<T>>::Err(Status::invalid_args, "avgdl must be supplied during searching");
        }
        auto avgdl = cfg.bm25_avgdl.value();
        if constexpr (use_wand) {
            // wand: search time k1/b must equal load time config.
            if ((cfg.bm25_k1.has_value() && cfg.bm25_k1.value() != bm25_params_->k1) ||
                ((cfg.bm25_b.has_value() && cfg.bm25_b.value() != bm25_params_->b))) {
                return expected<DocValueComputer<T>>::Err(
                    Status::invalid_args, "search time k1/b must equal load time config for WAND index.");
            }
            return GetDocValueBM25Computer<T>(bm25_params_->k1, bm25_params_->b, avgdl);
        } else {
            // inverted index: search time k1/b may override load time config.
            auto k1 = cfg.bm25_k1.has_value() ? cfg.bm25_k1.value() : bm25_params_->k1;
            auto b = cfg.bm25_b.has_value() ? cfg.bm25_b.value() : bm25_params_->b;
            return GetDocValueBM25Computer<T>(k1, b, avgdl);
        }
    }

    Status
    Save(MemoryIOWriter& writer) override {
        /**
         * zero copy is not yet implemented, now serializing in a zero copy
         * compatible way while still copying during deserialization.
         *
         * Layout:
         *
         * 1. int32_t rows, sign indicates whether to use wand
         * 2. int32_t cols
         * 3. for each row:
         *     1. int32_t len
         *     2. for each non-zero value:
         *        1. table_t idx
         *        2. T val
         *     With zero copy deserization, each SparseRow object should
         *     reference(not owning) the memory address of the first element.
         *
         * inverted_lut_ and max_score_in_dim_ not serialized, they will be
         * constructed dynamically during deserialization.
         *
         * Data are densly packed in serialized bytes and no padding is added.
         */
        std::shared_lock<std::shared_mutex> lock(mu_);
        writeBinaryPOD(writer, n_rows_internal());
        writeBinaryPOD(writer, n_cols_internal());
        writeBinaryPOD(writer, value_threshold_);
        for (size_t i = 0; i < n_rows_internal(); ++i) {
            auto& row = raw_data_[i];
            writeBinaryPOD(writer, row.size());
            if (row.size() == 0) {
                continue;
            }
            writer.write(row.data(), row.size() * SparseRow<T>::element_size());
        }
        return Status::success;
    }

    Status
    Load(MemoryIOReader& reader, bool is_mmap) override {
        std::unique_lock<std::shared_mutex> lock(mu_);
        int64_t rows;
        readBinaryPOD(reader, rows);
        // previous versions used the signness of rows to indicate whether to
        // use wand. now we use a template parameter to control this thus simply
        // take the absolute value of rows.
        rows = std::abs(rows);
        readBinaryPOD(reader, max_dim_);
        readBinaryPOD(reader, value_threshold_);

        raw_data_.reserve(rows);

        for (int64_t i = 0; i < rows; ++i) {
            size_t count;
            readBinaryPOD(reader, count);
            if (is_mmap) {
                raw_data_.emplace_back(count, reader.data() + reader.tellg(), false);
                reader.advance(count * SparseRow<T>::element_size());
            } else {
                raw_data_.emplace_back(count);
                if (count == 0) {
                    continue;
                }
                reader.read(raw_data_[i].data(), count * SparseRow<T>::element_size());
            }
            add_row_to_index(raw_data_[i], i);
        }

        return Status::success;
    }

    // Non zero drop ratio is only supported for static index, i.e. data should
    // include all rows that'll be added to the index.
    Status
    Train(const SparseRow<T>* data, size_t rows, float drop_ratio_build) override {
        if (drop_ratio_build == 0.0f) {
            return Status::success;
        }
        // TODO: maybe i += 10 to down sample to speed up.
        size_t amount = 0;
        for (size_t i = 0; i < rows; ++i) {
            amount += data[i].size();
        }
        std::vector<T> vals(amount);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                vals.push_back(fabs(data[i][j].val));
            }
        }
        auto pos = vals.begin() + static_cast<size_t>(drop_ratio_build * vals.size());
        std::nth_element(vals.begin(), pos, vals.end());

        std::unique_lock<std::shared_mutex> lock(mu_);
        value_threshold_ = *pos;
        drop_during_build_ = true;
        return Status::success;
    }

    Status
    Add(const SparseRow<T>* data, size_t rows, int64_t dim) override {
        std::unique_lock<std::shared_mutex> lock(mu_);
        auto current_rows = n_rows_internal();
        if (current_rows > 0 && drop_during_build_) {
            LOG_KNOWHERE_ERROR_ << "Not allowed to add data to a built index with drop_ratio_build > 0.";
            return Status::invalid_args;
        }
        if ((size_t)dim > max_dim_) {
            max_dim_ = dim;
        }

        raw_data_.insert(raw_data_.end(), data, data + rows);
        for (size_t i = 0; i < rows; ++i) {
            add_row_to_index(data[i], current_rows + i);
        }
        return Status::success;
    }

    void
    Search(const SparseRow<T>& query, size_t k, float drop_ratio_search, float* distances, label_t* labels,
           size_t refine_factor, const BitsetView& bitset, const DocValueComputer<T>& computer) const override {
        // initially set result distances to NaN and labels to -1
        std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
        std::fill(labels, labels + k, -1);
        if (query.size() == 0) {
            return;
        }

        std::vector<T> values(query.size());
        for (size_t i = 0; i < query.size(); ++i) {
            values[i] = std::abs(query[i].val);
        }
        auto pos = values.begin() + static_cast<size_t>(drop_ratio_search * values.size());
        std::nth_element(values.begin(), pos, values.end());
        auto q_threshold = *pos;

        std::shared_lock<std::shared_mutex> lock(mu_);
        // if no data was dropped during both build and search, no refinement is
        // needed.
        if (!drop_during_build_ && drop_ratio_search == 0) {
            refine_factor = 1;
        }
        MaxMinHeap<T> heap(k * refine_factor);
        if constexpr (!use_wand) {
            search_brute_force(query, q_threshold, heap, bitset, computer);
        } else {
            search_wand(query, q_threshold, heap, bitset, computer);
        }

        if (refine_factor == 1) {
            collect_result(heap, distances, labels);
        } else {
            refine_and_collect(query, heap, k, distances, labels, computer);
        }
    }

    std::vector<float>
    GetAllDistances(const SparseRow<T>& query, float drop_ratio_search, const BitsetView& bitset,
                    const DocValueComputer<T>& computer) const override {
        if (query.size() == 0) {
            return {};
        }
        std::vector<T> values(query.size());
        for (size_t i = 0; i < query.size(); ++i) {
            values[i] = std::abs(query[i].val);
        }
        auto pos = values.begin() + static_cast<size_t>(drop_ratio_search * values.size());
        std::nth_element(values.begin(), pos, values.end());
        auto q_threshold = *pos;
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto distances = compute_all_distances(query, q_threshold, computer);
        for (size_t i = 0; i < distances.size(); ++i) {
            if (bitset.empty() || !bitset.test(i)) {
                continue;
            }
            distances[i] = 0.0f;
        }
        return distances;
    }

    void
    GetVectorById(const label_t id, SparseRow<T>& output) const override {
        output = raw_data_[id];
    }

    [[nodiscard]] size_t
    size() const override {
        // TODO:
        std::shared_lock<std::shared_mutex> lock(mu_);
        size_t res = sizeof(*this);
        res += sizeof(SparseRow<T>) * n_rows_internal();
        for (auto& row : raw_data_) {
            res += row.memory_usage();
        }

        res += (sizeof(table_t) + sizeof(std::vector<SparseIdVal<T>>)) * inverted_lut_.size();
        for (const auto& [idx, lut] : inverted_lut_) {
            res += sizeof(SparseIdVal<T>) * lut.capacity();
        }
        if constexpr (use_wand) {
            res += (sizeof(table_t) + sizeof(T)) * max_score_in_dim_.size();
        }
        return res;
    }

    [[nodiscard]] size_t
    n_rows() const override {
        std::shared_lock<std::shared_mutex> lock(mu_);
        return n_rows_internal();
    }

    [[nodiscard]] size_t
    n_cols() const override {
        std::shared_lock<std::shared_mutex> lock(mu_);
        return n_cols_internal();
    }

 private:
    size_t
    n_rows_internal() const {
        return raw_data_.size();
    }

    size_t
    n_cols_internal() const {
        return max_dim_;
    }

    std::vector<float>
    compute_all_distances(const SparseRow<T>& q_vec, T q_threshold, const DocValueComputer<T>& computer) const {
        std::vector<float> scores(n_rows_internal(), 0.0f);
        for (size_t idx = 0; idx < q_vec.size(); ++idx) {
            auto [i, v] = q_vec[idx];
            if (v < q_threshold || i >= n_cols_internal()) {
                continue;
            }
            auto lut_it = inverted_lut_.find(i);
            if (lut_it == inverted_lut_.end()) {
                continue;
            }
            // TODO: improve with SIMD
            auto& lut = lut_it->second;
            for (size_t j = 0; j < lut.size(); j++) {
                auto [idx, val] = lut[j];
                T val_sum = bm25 ? bm25_params_->row_sums.at(idx) : 0;
                scores[idx] += v * computer(val, val_sum);
            }
        }
        return scores;
    }

    // find the top-k candidates using brute force search, k as specified by the capacity of the heap.
    // any value in q_vec that is smaller than q_threshold and any value with dimension >= n_cols() will be ignored.
    // TODO: may switch to row-wise brute force if filter rate is high. Benchmark needed.
    void
    search_brute_force(const SparseRow<T>& q_vec, T q_threshold, MaxMinHeap<T>& heap, const BitsetView& bitset,
                       const DocValueComputer<T>& computer) const {
        auto scores = compute_all_distances(q_vec, q_threshold, computer);
        for (size_t i = 0; i < n_rows_internal(); ++i) {
            if ((bitset.empty() || !bitset.test(i)) && scores[i] != 0) {
                heap.push(i, scores[i]);
            }
        }
    }

    // LUT supports size() and operator[] which returns an SparseIdVal.
    template <typename LUT>
    class Cursor {
     public:
        Cursor(const LUT& lut, size_t num_vec, float max_score, float q_value, const BitsetView bitset)
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
        cur_vec_val() const {
            return lut_[loc_].val;
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
        const LUT& lut_;
        size_t loc_ = 0;
        size_t num_vec_ = 0;
        float max_score_ = 0.0f;
        float q_value_ = 0.0f;
        const BitsetView bitset_;
    };  // class Cursor

    // any value in q_vec that is smaller than q_threshold will be ignored.
    void
    search_wand(const SparseRow<T>& q_vec, T q_threshold, MaxMinHeap<T>& heap, const BitsetView& bitset,
                const DocValueComputer<T>& computer) const {
        auto q_dim = q_vec.size();
        std::vector<std::shared_ptr<Cursor<std::vector<SparseIdVal<T>>>>> cursors(q_dim);
        auto valid_q_dim = 0;
        for (size_t i = 0; i < q_dim; ++i) {
            auto [idx, val] = q_vec[i];
            if (std::abs(val) < q_threshold || idx >= n_cols_internal()) {
                continue;
            }
            auto lut_it = inverted_lut_.find(idx);
            if (lut_it == inverted_lut_.end()) {
                continue;
            }
            auto& lut = lut_it->second;
            cursors[valid_q_dim++] = std::make_shared<Cursor<std::vector<SparseIdVal<T>>>>(
                lut, n_rows_internal(), max_score_in_dim_.find(idx)->second * val, val, bitset);
        }
        if (valid_q_dim == 0) {
            return;
        }
        cursors.resize(valid_q_dim);
        auto sort_cursors = [&cursors] {
            std::sort(cursors.begin(), cursors.end(),
                      [](auto& x, auto& y) { return x->cur_vec_id() < y->cur_vec_id(); });
        };
        sort_cursors();
        auto score_above_threshold = [&heap](float x) { return !heap.full() || x > heap.top().val; };
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
                    T cur_vec_sum = bm25 ? bm25_params_->row_sums.at(cursor->cur_vec_id()) : 0;
                    score += cursor->q_value() * computer(cursor->cur_vec_val(), cur_vec_sum);
                    cursor->next();
                }
                heap.push(pivot_id, score);
                sort_cursors();
            } else {
                size_t next_list = pivot;
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
    refine_and_collect(const SparseRow<T>& q_vec, MaxMinHeap<T>& inaccurate, size_t k, float* distances,
                       label_t* labels, const DocValueComputer<T>& computer) const {
        std::priority_queue<SparseIdVal<T>, std::vector<SparseIdVal<T>>, std::greater<SparseIdVal<T>>> heap;

        while (!inaccurate.empty()) {
            auto [u, d] = inaccurate.top();
            inaccurate.pop();

            T u_sum = bm25 ? bm25_params_->row_sums.at(u) : 0;

            auto dist_acc = q_vec.dot(raw_data_[u], computer, u_sum);
            if (heap.size() < k) {
                heap.emplace(u, dist_acc);
            } else if (heap.top().val < dist_acc) {
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
            distances[i] = heap.top().val;
            heap.pop();
        }
    }

    inline void
    add_row_to_index(const SparseRow<T>& row, table_t id) {
        [[maybe_unused]] T row_sum = 0;
        for (size_t j = 0; j < row.size(); ++j) {
            auto [idx, val] = row[j];
            if constexpr (bm25) {
                row_sum += val;
            }
            // Skip values equals to or close enough to zero(which contributes
            // little to the total IP score).
            if (val == 0 || (drop_during_build_ && fabs(val) < value_threshold_)) {
                continue;
            }
            if (inverted_lut_.find(idx) == inverted_lut_.end()) {
                inverted_lut_[idx];
                if constexpr (use_wand) {
                    max_score_in_dim_[idx] = 0;
                }
            }
            inverted_lut_[idx].emplace_back(id, val);
            if constexpr (use_wand) {
                auto score = val;
                if constexpr (bm25) {
                    score = bm25_params_->max_score_ratio * bm25_params_->wand_max_score_computer(val, row_sum);
                }
                max_score_in_dim_[idx] = std::max(max_score_in_dim_[idx], score);
            }
        }
        if constexpr (bm25) {
            bm25_params_->row_sums[id] = row_sum;
        }
    }

    std::vector<SparseRow<T>> raw_data_;
    mutable std::shared_mutex mu_;

    std::unordered_map<table_t, std::vector<SparseIdVal<T>>> inverted_lut_;
    // If we want to drop small values during build, we must first train the
    // index with all the data to compute value_threshold_.
    bool drop_during_build_ = false;
    // when drop_during_build_ is true, any value smaller than value_threshold_
    // will not be added to inverted_lut_. value_threshold_ is set to the
    // drop_ratio_build-th percentile of all absolute values in the index.
    T value_threshold_ = 0.0f;
    std::unordered_map<table_t, T> max_score_in_dim_;
    size_t max_dim_ = 0;

    struct BM25Params {
        float k1;
        float b;
        // row_sums is used to cache the sum of values of each row, which
        // corresponds to the document length of each doc in the BM25 formula.
        std::unordered_map<size_t, T> row_sums;

        // below are used only for WAND index.
        // BM25Params::avgdl is segment level average document length, used only
        // by WAND to compute max score. Should not be used for actual score
        // computing.
        float avgdl;
        float max_score_ratio;
        DocValueComputer<T> wand_max_score_computer;

        BM25Params(float k1, float b, float avgdl, float max_score_ratio)
            : k1(k1),
              b(b),
              avgdl(avgdl),
              max_score_ratio(max_score_ratio),
              wand_max_score_computer(GetDocValueBM25Computer<T>(k1, b, avgdl)) {
        }
    };  // struct BM25Params

    std::unique_ptr<BM25Params> bm25_params_;

};  // class InvertedIndex

}  // namespace knowhere::sparse

#endif  // SPARSE_INVERTED_INDEX_H
