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

#include "knowhere/index/index_node.h"

#include <cmath>
#include <queue>
#include <unordered_set>

#include "knowhere/context.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/comp/task.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

// NOLINTBEGIN(google-default-arguments)
expected<DataSetPtr>
IndexNode::RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                       milvus::OpContext* op_context) const {
    const auto base_cfg = static_cast<const BaseConfig&>(*cfg);
    const float closer_bound = base_cfg.range_filter.value();
    const bool has_closer_bound = closer_bound != defaultRangeFilter;
    float further_bound = base_cfg.radius.value();

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

    /** The `range_search_k` is used to early terminate the iterator-search.
     * - `range_search_k < 0` means no early termination.
     * - `range_search_k == 0` will return empty results.
     * - Note that the number of results is not guaranteed to be exactly range_search_k, it may be more or less.
     * */
    const int32_t range_search_k = base_cfg.range_search_k.value();
    LOG_KNOWHERE_DEBUG_ << "range_search_k: " << range_search_k;
    if (range_search_k == 0) {
        auto nq = dataset->GetRows();
        std::vector<std::vector<int64_t>> result_id_array(nq);
        std::vector<std::vector<float>> result_dist_array(nq);
        auto range_search_result = GetRangeSearchResult(result_dist_array, result_id_array, the_larger_the_closer, nq,
                                                        further_bound, closer_bound);
        return GenResultDataSet(nq, std::move(range_search_result));
    }

    // The range_search function has utilized the search_pool to concurrently handle various queries.
    // To prevent potential deadlocks, the iterator for a single query no longer requires additional thread
    //   control over the next() call.
    auto its_or = AnnIterator(dataset, std::move(cfg), bitset, false);
    if (!its_or.has_value()) {
        return expected<DataSetPtr>::Err(its_or.error(),
                                         "RangeSearch failed due to AnnIterator failure: " + its_or.what());
    }

    const auto its = its_or.value();
    const auto nq = its.size();
    std::vector<std::vector<int64_t>> result_id_array(nq);
    std::vector<std::vector<float>> result_dist_array(nq);

    const bool retain_iterator_order = base_cfg.retain_iterator_order.value();
    LOG_KNOWHERE_DEBUG_ << "retain_iterator_order: " << retain_iterator_order;

    /**
     * use ordered iterator (retain_iterator_order == true)
     * - terminate iterator if next distance exceeds `further_bound`.
     * - terminate iterator if get enough results. (`range_search_k`)
     * */
    auto task_with_ordered_iterator = [&](size_t idx) {
#if defined(NOT_COMPILE_FOR_SWIG)
        checkCancellation(op_context);
#endif
        auto it = its[idx];
        while (it->HasNext()) {
#if defined(NOT_COMPILE_FOR_SWIG)
            checkCancellation(op_context);
#endif
            auto [id, dist] = it->Next();
            if (has_closer_bound && too_close(dist)) {
                continue;
            }
            if (same_or_too_far(dist)) {
                break;
            }
            result_id_array[idx].push_back(id);
            result_dist_array[idx].push_back(dist);
            if (range_search_k >= 0 && static_cast<int32_t>(result_id_array[idx].size()) >= range_search_k) {
                break;
            }
        }
    };

    /**
     * use default unordered iterator (retain_iterator_order == false)
     * - terminate iterator if next distance [consecutively] exceeds `further_bound` several times.
     * - if get enough results (`range_search_k`), update a `tighter_further_bound`, to early terminate iterator.
     * */
    const auto range_search_level = base_cfg.range_search_level.value();  // from 0 to 0.5
    LOG_KNOWHERE_DEBUG_ << "range_search_level: " << range_search_level;
    auto task_with_unordered_iterator = [&](size_t idx) {
#if defined(NOT_COMPILE_FOR_SWIG)
        checkCancellation(op_context);
#endif
        // max-heap, use top (the current kth-furthest dist) as the further_bound if size == range_search_k
        std::priority_queue<float, std::vector<float>, decltype(is_first_closer)> early_stop_further_bounds(
            is_first_closer);
        auto it = its[idx];
        size_t num_next = 0;
        size_t num_consecutive_over_further_bound = 0;
        float tighter_further_bound = base_cfg.radius.value();
        auto same_or_too_far = [&is_first_closer, &tighter_further_bound](float dist) {
            return !is_first_closer(dist, tighter_further_bound);
        };
        while (it->HasNext()) {
#if defined(NOT_COMPILE_FOR_SWIG)
            checkCancellation(op_context);
#endif
            auto [id, dist] = it->Next();
            num_next++;
            if (has_closer_bound && too_close(dist)) {
                continue;
            }
            if (same_or_too_far(dist)) {
                num_consecutive_over_further_bound++;
                if (num_consecutive_over_further_bound >
                    static_cast<size_t>(std::ceil(num_next * range_search_level))) {
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
                    tighter_further_bound = early_stop_further_bounds.top();
                }
            }
            num_consecutive_over_further_bound = 0;
            result_id_array[idx].push_back(id);
            result_dist_array[idx].push_back(dist);
        }
    };
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    try {
        std::vector<folly::Future<folly::Unit>> futs;
        futs.reserve(nq);
        if (retain_iterator_order) {
            for (size_t i = 0; i < nq; i++) {
                futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&, idx = i]() {
                    ThreadPool::ScopedSearchOmpSetter setter(1);
                    task_with_ordered_iterator(idx);
                }));
            }
        } else {
            for (size_t i = 0; i < nq; i++) {
                futs.emplace_back(ThreadPool::GetGlobalSearchThreadPool()->push([&, idx = i]() {
                    ThreadPool::ScopedSearchOmpSetter setter(1);
                    task_with_unordered_iterator(idx);
                }));
            }
        }
        WaitAllSuccess(futs);
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "range search error: " << e.what();
        return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
    }
#else
    if (retain_iterator_order) {
        for (size_t i = 0; i < nq; i++) {
            task_with_ordered_iterator(i);
        }
    } else {
        for (size_t i = 0; i < nq; i++) {
            task_with_unordered_iterator(i);
        }
    }
#endif

    auto range_search_result = GetRangeSearchResult(result_dist_array, result_id_array, the_larger_the_closer, nq,
                                                    further_bound, closer_bound);
    return GenResultDataSet(nq, std::move(range_search_result));
}

expected<DataSetPtr>
IndexNode::SearchEmbList(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                         milvus::OpContext* op_context) const {
    if (!emb_list_strategy_) {
        LOG_KNOWHERE_ERROR_ << "EmbList strategy not initialized";
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "strategy not initialized");
    }

    // 1. Parse query offset
    const size_t* lims = dataset->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
    if (lims == nullptr) {
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "missing emb_list offset, could not search");
    }
    auto num_q_vecs = static_cast<size_t>(dataset->GetRows());
    EmbListOffset query_offset(lims, num_q_vecs);

    // 2. Parse config and metric types
    auto& config = static_cast<BaseConfig&>(*cfg);
    auto target_k = config.k.value();

    // Parse sub metric type for underlying ANN search
    auto metric_info_or = ParseEmbListMetric(config);
    if (!metric_info_or.has_value()) {
        return expected<DataSetPtr>::Err(metric_info_or.error(), metric_info_or.what());
    }
    auto sub_metric_type = metric_info_or.value().sub_metric_type;

    // 3. Build search context with callbacks
    EmbListSearchContext ctx;

    // Store config - use shared_ptr wrapper to allow moving to callback
    // Note: ann_search and ann_iterator are mutually exclusive (only one will be called)
    auto cfg_wrapper = std::make_shared<std::unique_ptr<Config>>(std::move(cfg));

    // ANN search callback (used by MUVERA strategy)
    ctx.ann_search = [this, cfg_wrapper, sub_metric_type, &bitset, &op_context](const DataSetPtr query,
                                                                                int32_t k) -> expected<DataSetPtr> {
        if (!*cfg_wrapper) {
            LOG_KNOWHERE_ERROR_ << "ann_search: config already consumed";
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "config already consumed");
        }
        auto& search_cfg = static_cast<BaseConfig&>(**cfg_wrapper);
        search_cfg.k = k;
        search_cfg.metric_type = sub_metric_type;
        return this->Search(query, std::move(*cfg_wrapper), bitset, op_context);
    };

    // ANN iterator callback (used by Direct strategy) - for incremental result fetching
    ctx.ann_iterator = [this, cfg_wrapper, sub_metric_type, &bitset,
                        &op_context](const DataSetPtr query) -> expected<std::vector<AnnResultIteratorPtr>> {
        if (!*cfg_wrapper) {
            LOG_KNOWHERE_ERROR_ << "ann_iterator: config already consumed";
            return expected<std::vector<AnnResultIteratorPtr>>::Err(Status::emb_list_inner_error,
                                                                    "config already consumed");
        }
        auto& iter_cfg = static_cast<BaseConfig&>(**cfg_wrapper);
        iter_cfg.metric_type = sub_metric_type;
        auto iterators_result = this->AnnIterator(query, std::move(*cfg_wrapper), bitset, true, op_context);
        if (!iterators_result.has_value()) {
            return expected<std::vector<AnnResultIteratorPtr>>::Err(iterators_result.error(), iterators_result.what());
        }

        // Wrap IndexNode::iterator to AnnResultIterator
        std::vector<AnnResultIteratorPtr> result;
        result.reserve(iterators_result.value().size());
        for (auto& iter : iterators_result.value()) {
            // Create wrapper that adapts IndexNode::iterator to AnnResultIterator
            class IteratorWrapper : public AnnResultIterator {
             public:
                explicit IteratorWrapper(std::shared_ptr<IndexNode::iterator> iter) : iter_(std::move(iter)) {
                }
                std::pair<int64_t, float>
                Next() override {
                    return iter_->Next();
                }
                bool
                HasNext() override {
                    return iter_->HasNext();
                }

             private:
                std::shared_ptr<IndexNode::iterator> iter_;
            };
            result.push_back(std::make_shared<IteratorWrapper>(std::move(iter)));
        }
        return result;
    };

    // Calculate distance by IDs callback
    ctx.calc_distance_by_ids = [this, &bitset, &op_context](const DataSetPtr query, const int64_t* ids, size_t ids_len,
                                                            bool is_cosine) -> expected<DataSetPtr> {
        return this->CalcDistByIDs(query, bitset, ids, ids_len, is_cosine, op_context);
    };

    // Get vectors by IDs callback
    ctx.get_vectors_by_ids = [this](const int64_t* ids, size_t ids_len) -> expected<DataSetPtr> {
        return this->GetVectorByIds(GenIdsDataSet(ids_len, ids));
    };

    // Get index count callback
    ctx.get_index_count = [this]() -> int64_t { return this->Count(); };

    // Get query code size callback
    ctx.get_query_code_size = [this](const DataSetPtr ds) -> expected<size_t> {
        auto result = this->GetQueryCodeSize(ds);
        if (result.has_value()) {
            return result.value();
        }
        return expected<size_t>::Err(Status::emb_list_inner_error, "failed to get query code size");
    };

    // Bitset for filtering
    ctx.bitset = bitset;

    // 4. Delegate search to strategy (strategy has full control over search flow)
    return emb_list_strategy_->Search(dataset, query_offset, target_k, config, ctx);
}

expected<DataSetPtr>
IndexNode::SearchEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset,
                               milvus::OpContext* op_context) const {
    auto cfg = static_cast<const knowhere::BaseConfig&>(*config);
    auto el_metric_type_or = get_el_metric_type(cfg.metric_type.value());
    auto metric_is_emb_list = el_metric_type_or.has_value();
    bool query_is_emb_list = dataset->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET) != nullptr;
    if (metric_is_emb_list && !query_is_emb_list) {
        LOG_KNOWHERE_WARNING_ << "Not found emb_list offset in query dataset, but metric type is of emb_list";
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error,
                                         "Not found emb_list offset in query dataset, but metric type is of emb_list");
    }
    if (!metric_is_emb_list && query_is_emb_list) {
        LOG_KNOWHERE_WARNING_ << "Invalid emb_list metric type, but found emb_list offset in query dataset: "
                              << cfg.metric_type.value();
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error,
                                         "Invalid emb_list metric type, but found emb_list offset in query dataset.");
    }
    if (!metric_is_emb_list && !query_is_emb_list) {
        // if both metric and query dataset are not emb_list, use the default search method
        return Search(dataset, std::move(config), bitset, op_context);
    }

    return SearchEmbList(dataset, std::move(config), bitset, op_context);
}

expected<DataSetPtr>
IndexNode::RangeSearchEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                                    milvus::OpContext* op_context) const {
    auto config = static_cast<const knowhere::BaseConfig&>(*cfg);
    auto el_metric_type_or = get_el_metric_type(config.metric_type.value());
    auto metric_is_emb_list = el_metric_type_or.has_value();
    if (metric_is_emb_list) {
        LOG_KNOWHERE_WARNING_ << "Range search is not supported for emb_list";
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "range search is not supported for emb_list");
    }
    return RangeSearch(dataset, std::move(cfg), bitset, op_context);
}

expected<std::vector<IndexNode::IteratorPtr>>
IndexNode::AnnIteratorEmbListIfNeed(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                                    bool use_knowhere_search_pool, milvus::OpContext* op_context) const {
    auto config = static_cast<const knowhere::BaseConfig&>(*cfg);
    auto el_metric_type_or = get_el_metric_type(config.metric_type.value());
    auto metric_is_emb_list = el_metric_type_or.has_value();
    if (metric_is_emb_list) {
        LOG_KNOWHERE_WARNING_ << "Ann iterator is not supported for emb_list";
        return expected<std::vector<IteratorPtr>>::Err(Status::emb_list_inner_error,
                                                       "ann iterator is not supported for emb_list");
    }
    return AnnIterator(dataset, std::move(cfg), bitset, use_knowhere_search_pool, op_context);
}
// NOLINTEND(google-default-arguments)

}  // namespace knowhere
