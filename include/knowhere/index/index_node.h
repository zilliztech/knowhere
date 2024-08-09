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

#ifndef INDEX_NODE_H
#define INDEX_NODE_H

#include <functional>
#include <queue>
#include <utility>
#include <vector>

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/object.h"
#include "knowhere/operands.h"
#include "knowhere/range_util.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/comp/thread_pool.h"
#endif

namespace knowhere {

class IndexNode : public Object {
 public:
    IndexNode(const int32_t ver) : version_(ver) {
    }

    IndexNode() : version_(Version::GetDefaultVersion()) {
    }

    IndexNode(const IndexNode& other) : version_(other.version_) {
    }

    IndexNode(const IndexNode&& other) : version_(other.version_) {
    }

    virtual Status
    Build(const DataSetPtr dataset, const Config& cfg) {
        RETURN_IF_ERROR(Train(dataset, cfg));
        return Add(dataset, cfg);
    }

    virtual Status
    Train(const DataSetPtr dataset, const Config& cfg) = 0;

    virtual Status
    Add(const DataSetPtr dataset, const Config& cfg) = 0;

    virtual expected<DataSetPtr>
    Search(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const = 0;

    // not thread safe.
    class iterator {
     public:
        virtual std::pair<int64_t, float>
        Next() = 0;
        [[nodiscard]] virtual bool
        HasNext() = 0;
        virtual ~iterator() {
        }
    };
    using IteratorPtr = std::shared_ptr<iterator>;

    virtual expected<std::vector<IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const {
        return expected<std::vector<std::shared_ptr<iterator>>>::Err(
            Status::not_implemented, "annIterator not supported for current index type");
    }

    // Default range search implementation based on iterator. Assumes the iterator will buffer an expanded range and
    // return the closest elements on each Next() call.
    //
    // TODO: test with mock AnnIterator after we introduced mock framework into knowhere. Currently this is tested in
    // test_sparse.cc with real sparse vector index.
    virtual expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const {
        auto its_or = AnnIterator(dataset, cfg, bitset);
        if (!its_or.has_value()) {
            return expected<DataSetPtr>::Err(its_or.error(),
                                             "RangeSearch failed due to AnnIterator failure: " + its_or.what());
        }
        const auto base_cfg = static_cast<const BaseConfig&>(cfg);
        const float closer_bound = base_cfg.range_filter.value();
        const bool has_closer_bound = closer_bound != defaultRangeFilter;
        // The `range_search_k` is used to early terminate the iterator-search.
        //  When the number of range-valid results exceeds range_search_k, the `further_bound` will be updated.
        //  "-1" means no early termination, `further_bound` will not be updated.
        // Note that the number of results is not guaranteed to be exactly range_search_k, it may be more or less.
        float further_bound = base_cfg.radius.value();
        const int32_t range_search_k = base_cfg.range_search_k.value();
        LOG_KNOWHERE_DEBUG_ << "range_search_k: " << range_search_k;

        const bool the_larger_the_closer = IsMetricType(base_cfg.metric_type.value(), metric::IP) ||
                                           IsMetricType(base_cfg.metric_type.value(), metric::COSINE) ||
                                           IsMetricType(base_cfg.metric_type.value(), metric::BM25);
        auto is_first_closer = [&the_larger_the_closer](const float dist_1, const float dist_2) {
            return the_larger_the_closer ? dist_1 > dist_2 : dist_1 < dist_2;
        };
        auto too_close = [&is_first_closer, &closer_bound](float dist) { return is_first_closer(dist, closer_bound); };
        auto same_or_too_far = [&is_first_closer, &further_bound](float dist) {
            return !is_first_closer(dist, further_bound);
        };

        const auto its = its_or.value();
        const auto nq = its.size();
        std::vector<std::vector<int64_t>> result_id_array(nq);
        std::vector<std::vector<float>> result_dist_array(nq);

        constexpr size_t k_min_num_consecutive_over_further_bound = 16;
        const auto range_search_level = base_cfg.range_search_level.value();  // from 0 to 0.5
        LOG_KNOWHERE_DEBUG_ << "range_search_level: " << range_search_level;

        auto task = [&](size_t idx) {
            // max-heap, use top (the current kth-furthest dist) as the further_bound if size == range_search_k
            std::priority_queue<float, std::vector<float>, decltype(is_first_closer)> early_stop_further_bounds(
                is_first_closer);
            auto it = its[idx];
            size_t num_next = 0;
            size_t num_consecutive_over_further_bound = 0;
            while (it->HasNext()) {
                auto [id, dist] = it->Next();
                num_next++;
                if (has_closer_bound && too_close(dist)) {
                    continue;
                }
                if (same_or_too_far(dist)) {
                    num_consecutive_over_further_bound++;
                    if (num_consecutive_over_further_bound >
                        std::max(k_min_num_consecutive_over_further_bound, (size_t)(num_next * range_search_level))) {
                        break;
                    }
                    continue;
                }
                if (range_search_k > 0) {
                    if (static_cast<int32_t>(early_stop_further_bounds.size()) < range_search_k) {
                        early_stop_further_bounds.emplace(dist);
                    } else {
                        early_stop_further_bounds.pop();
                        early_stop_further_bounds.emplace(dist);
                        further_bound = early_stop_further_bounds.top();
                    }
                }
                num_consecutive_over_further_bound = 0;
                result_id_array[idx].push_back(id);
                result_dist_array[idx].push_back(dist);
            }
        };
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        for (size_t i = 0; i < nq; i++) {
            futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&, idx = i]() { task(idx); }));
        }
        WaitAllSuccess(futs);
#else
        for (size_t i = 0; i < nq; i++) {
            task(i);
        }
#endif

        auto range_search_result = GetRangeSearchResult(result_dist_array, result_id_array, the_larger_the_closer, nq,
                                                        further_bound, closer_bound);
        return GenResultDataSet(nq, std::move(range_search_result));
    }

    virtual expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const = 0;

    virtual bool
    HasRawData(const std::string& metric_type) const = 0;

    virtual bool
    IsAdditionalScalarSupported() const {
        return false;
    }

    virtual expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const = 0;

    virtual Status
    Serialize(BinarySet& binset) const = 0;

    virtual Status
    Deserialize(const BinarySet& binset, const Config& config) = 0;

    virtual Status
    DeserializeFromFile(const std::string& filename, const Config& config) = 0;

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const = 0;

    virtual int64_t
    Dim() const = 0;

    virtual int64_t
    Size() const = 0;

    virtual int64_t
    Count() const = 0;

    virtual std::string
    Type() const = 0;

    virtual ~IndexNode() {
    }

 protected:
    Version version_;
};

// Common superclass for iterators that expand search range as needed. Subclasses need
// to override `next_batch` which will add expanded vectors to the results. For indexes
// with quantization, override `raw_distance`.
class IndexIterator : public IndexNode::iterator {
 public:
    IndexIterator(bool larger_is_closer, float refine_ratio = 0.0f, bool retain_iterator_order = false)
        : refine_ratio_(refine_ratio),
          refine_(refine_ratio != 0.0f),
          retain_iterator_order_(retain_iterator_order),
          sign_(larger_is_closer ? -1 : 1) {
    }

    std::pair<int64_t, float>
    Next() override {
        if (!initialized_) {
            throw std::runtime_error("Next should not be called before initialization");
        }
        auto& q = refined_res_.empty() ? res_ : refined_res_;
        if (q.empty()) {
            throw std::runtime_error("No more elements");
        }
        auto ret = q.top();
        q.pop();
        UpdateNext();
        if (retain_iterator_order_) {
            while (HasNext()) {
                auto& q = refined_res_.empty() ? res_ : refined_res_;
                auto next_ret = q.top();
                // with the help of `sign_`, both `res_` and `refine_res` are min-heap.
                //   such as `COSINE`, `-dist` will be inserted to `res_` or `refine_res`.
                // just make sure that the next value is greater than or equal to the current value.
                if (next_ret.val >= ret.val) {
                    break;
                }
                q.pop();
                UpdateNext();
            }
        }
        return std::make_pair(ret.id, ret.val * sign_);
    }

    [[nodiscard]] bool
    HasNext() override {
        if (!initialized_) {
            throw std::runtime_error("HasNext should not be called before initialization");
        }
        return !res_.empty() || !refined_res_.empty();
    }

    virtual void
    initialize() {
        if (initialized_) {
            throw std::runtime_error("initialize should not be called twice");
        }
        UpdateNext();
        initialized_ = true;
    }

 protected:
    virtual void
    next_batch(std::function<void(const std::vector<DistId>&)> batch_handler) = 0;
    // will be called only if refine_ratio_ is not 0.
    virtual float
    raw_distance(int64_t id) {
        if (!refine_) {
            throw std::runtime_error("raw_distance should not be called for indexes without quantization");
        }
        throw std::runtime_error("raw_distance not implemented");
    }

    const float refine_ratio_;
    const bool refine_;

    std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> res_;
    // unused if refine_ is false
    std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> refined_res_;

 private:
    inline size_t
    min_refine_size() const {
        // TODO: maybe make this configurable
        return std::max((size_t)20, (size_t)(res_.size() * refine_ratio_));
    }

    void
    UpdateNext() {
        auto batch_handler = [this](const std::vector<DistId>& batch) {
            if (batch.empty()) {
                return;
            }
            for (const auto& dist_id : batch) {
                res_.emplace(dist_id.id, dist_id.val * sign_);
            }
            if (refine_) {
                while (!res_.empty() && (refined_res_.empty() || refined_res_.size() < min_refine_size())) {
                    auto pair = res_.top();
                    res_.pop();
                    refined_res_.emplace(pair.id, raw_distance(pair.id) * sign_);
                }
            }
        };
        next_batch(batch_handler);
    }

    bool initialized_ = false;
    bool retain_iterator_order_ = false;
    const int64_t sign_;
};

// An iterator implementation that accepts a list of distances and ids and returns them in order.
class PrecomputedDistanceIterator : public IndexNode::iterator {
 public:
    PrecomputedDistanceIterator(std::vector<DistId>&& distances_ids, bool larger_is_closer)
        : larger_is_closer_(larger_is_closer), results_(std::move(distances_ids)) {
        sort_size_ = get_sort_size(results_.size());
        sort_next();
    }

    // Construct an iterator from a list of distances with index being id, filtering out zero distances.
    PrecomputedDistanceIterator(const std::vector<float>& distances, bool larger_is_closer)
        : larger_is_closer_(larger_is_closer) {
        // 30% is a ratio guesstimate of non-zero distances: probability of 2 random sparse splade vectors(100 non zero
        // dims out of 30000 total dims) sharing at least 1 common non-zero dimension.
        results_.reserve(distances.size() * 0.3);
        for (size_t i = 0; i < distances.size(); i++) {
            if (distances[i] != 0) {
                results_.emplace_back((int64_t)i, distances[i]);
            }
        }
        sort_size_ = get_sort_size(results_.size());
        sort_next();
    }

    std::pair<int64_t, float>
    Next() override {
        sort_next();
        auto& result = results_[next_++];
        return std::make_pair(result.id, result.val);
    }

    [[nodiscard]] bool
    HasNext() override {
        return next_ < results_.size() && results_[next_].id != -1;
    }

 private:
    static inline size_t
    get_sort_size(size_t rows) {
        return std::max((size_t)50000, rows / 10);
    }

    // sort the next sort_size_ elements
    inline void
    sort_next() {
        if (next_ < sorted_) {
            return;
        }
        size_t current_end = std::min(results_.size(), sorted_ + sort_size_);
        if (larger_is_closer_) {
            std::partial_sort(results_.begin() + sorted_, results_.begin() + current_end, results_.end(),
                              std::greater<DistId>());
        } else {
            std::partial_sort(results_.begin() + sorted_, results_.begin() + current_end, results_.end(),
                              std::less<DistId>());
        }

        sorted_ = current_end;
    }
    const bool larger_is_closer_;

    std::vector<DistId> results_;
    size_t next_ = 0;
    size_t sorted_ = 0;
    size_t sort_size_ = 0;
};

}  // namespace knowhere

#endif /* INDEX_NODE_H */
