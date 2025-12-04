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
    auto dim = dataset->GetDim();
    const size_t* lims = dataset->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
    if (lims == nullptr) {
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "missing emb_list offset, could not search");
    }
    auto num_q_vecs = static_cast<size_t>(dataset->GetRows());
    EmbListOffset query_emb_list_offset(lims, num_q_vecs);
    auto num_q_el = query_emb_list_offset.num_el();
    auto& config = static_cast<BaseConfig&>(*cfg);
    auto metric_type = config.metric_type.value();
    auto el_metric_type_or = get_el_metric_type(metric_type);
    if (!el_metric_type_or.has_value()) {
        LOG_KNOWHERE_WARNING_ << "Invalid metric type: " << metric_type;
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
    }
    auto el_metric_type = el_metric_type_or.value();
    LOG_KNOWHERE_DEBUG_ << "search emb_list with el metric_type: " << el_metric_type;
    auto el_agg_func_or = get_emb_list_agg_func(el_metric_type);
    if (!el_agg_func_or.has_value()) {
        LOG_KNOWHERE_ERROR_ << "Invalid emb list aggeration function for metric type: " << el_metric_type;
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error,
                                         "invalid emb list aggeration function for metric type: " + el_metric_type);
    }
    auto el_agg_func = el_agg_func_or.value();

    auto sub_metric_type_or = get_sub_metric_type(metric_type);
    if (!sub_metric_type_or.has_value()) {
        LOG_KNOWHERE_WARNING_ << "Invalid metric type: " << metric_type;
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
    }
    auto sub_metric_type = sub_metric_type_or.value();
    bool larger_is_closer = true;
    if (sub_metric_type == metric::L2 || sub_metric_type == metric::HAMMING || sub_metric_type == metric::JACCARD) {
        larger_is_closer = false;
    }
    bool is_cosine = sub_metric_type == metric::COSINE ? true : false;
    LOG_KNOWHERE_DEBUG_ << "search emb_list with sub metric_type: " << sub_metric_type;
    auto el_k = config.k.value();

    // Allocate result arrays
    auto ids = std::make_unique<int64_t[]>(num_q_el * el_k);
    auto dists = std::make_unique<float[]>(num_q_el * el_k);

    // Stage 1: base-index search - retrieve top k' vectors
    //  top k' = k * retrieval_ann_ratio
    config.metric_type = sub_metric_type;
    auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
    if (retrieval_ann_ratio <= 0.0f) {
        auto err_msg = "retrieval_ann_ratio could not be less than or equal to 0";
        LOG_KNOWHERE_WARNING_ << err_msg;
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, err_msg);
    }
    int32_t vec_topk = std::min(std::max((int32_t)(el_k * retrieval_ann_ratio), 1), (int32_t)Count());
    config.k = vec_topk;
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    knowhere_search_emb_list_retrieval_ann_ratio.Observe(retrieval_ann_ratio);
    TimeRecorder rc("Emb List Search - 1st round ann search");
#endif
    auto ann_search_res = Search(dataset, std::move(cfg), bitset, op_context).value();
    const auto stage1_ids = ann_search_res->GetIds();

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    auto time = rc.ElapseFromBegin("done");
    time *= 0.001;  // convert to ms
    knowhere_search_emb_list_1st_ann_latency.Observe(time);
#endif
    // Stage 2: For each query emb_list, perform brute-force distance calculation and aggregate scores
    auto query_code_size_or = GetQueryCodeSize(dataset);
    if (!query_code_size_or.has_value()) {
        LOG_KNOWHERE_ERROR_ << "could not get query code size";
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "could not get query code size");
    }
    auto query_code_size = query_code_size_or.value();
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    TimeRecorder rc2("Emb List Search - 2nd round bf and agg");
#endif
    for (size_t i = 0; i < num_q_el; i++) {
        auto start_offset = query_emb_list_offset.offset[i];
        auto end_offset = query_emb_list_offset.offset[i + 1];
        auto nq = end_offset - start_offset;

        // Collect unique emb_list IDs hit in stage 1
        std::unordered_set<size_t> el_ids_set;
        for (size_t j = start_offset * vec_topk; j < end_offset * vec_topk; j++) {
            if (stage1_ids[j] < 0) {
                continue;
            }
            el_ids_set.emplace(emb_list_offset_->get_el_id((size_t)stage1_ids[j]));
        }

        // For each emb_list, perform brute-force calculation and aggregate scores
        std::priority_queue<DistId, std::vector<DistId>, std::greater<>> minheap;
        std::priority_queue<DistId, std::vector<DistId>, std::less<>> maxheap;
        for (const auto& el_id : el_ids_set) {
            if (el_id >= emb_list_offset_->num_el()) {
                LOG_KNOWHERE_ERROR_ << "Invalid el_id: " << el_id;
                return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid emb_list id");
            }
            auto vids = emb_list_offset_->get_vids(el_id);
            // Generate query dataset for the current query_emb_list
            auto tensor = (const char*)dataset->GetTensor();
            size_t tensor_offset = start_offset * query_code_size;

            // Brute-force compute distances between all vectors in the query emb_list and all vectors in the
            // candidate emb_list
            auto bf_query_dataset = GenDataSet(end_offset - start_offset, dim, tensor + tensor_offset);
            auto bf_search_res = CalcDistByIDs(bf_query_dataset, bitset, vids.data(), vids.size(), is_cosine);
            if (!bf_search_res.has_value()) {
                LOG_KNOWHERE_ERROR_ << "bf search error: " << bf_search_res.what();
                return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "bf search error");
            }
            const auto bf_dists = bf_search_res.value()->GetDistance();

            // Aggregate score for the emb_list (e.g., sum of max similarities)
            auto score_or = el_agg_func(bf_dists, nq, vids.size(), larger_is_closer);
            if (!score_or.has_value()) {
                LOG_KNOWHERE_WARNING_ << "get_sum_max_sim failed, nq: " << nq << ", vids.size(): " << vids.size();
                return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "get_sum_max_sim failed");
            }
            auto score = score_or.value();
            if (larger_is_closer) {
                if (minheap.size() < (size_t)el_k) {
                    minheap.emplace((int64_t)el_id, score);
                } else {
                    if (score > minheap.top().val) {
                        minheap.pop();
                        minheap.emplace((int64_t)el_id, score);
                    }
                }
            } else {
                if (maxheap.size() < (size_t)el_k) {
                    maxheap.emplace((int64_t)el_id, score);
                } else {
                    if (score < maxheap.top().val) {
                        maxheap.pop();
                        maxheap.emplace((int64_t)el_id, score);
                    }
                }
            }
        }
        // Write results and fill remaining slots if not enough results
        size_t real_el_k = 0;
        if (larger_is_closer) {
            real_el_k = minheap.size();
            for (size_t j = 0; j < real_el_k; j++) {
                auto& a = minheap.top();
                ids[i * el_k + real_el_k - j - 1] = a.id;
                dists[i * el_k + real_el_k - j - 1] = a.val;
                minheap.pop();
            }
        } else {
            real_el_k = maxheap.size();
            for (size_t j = 0; j < real_el_k; j++) {
                auto& a = maxheap.top();
                ids[i * el_k + real_el_k - j - 1] = a.id;
                dists[i * el_k + real_el_k - j - 1] = a.val;
                maxheap.pop();
            }
        }
        std::fill(ids.get() + i * el_k + real_el_k, ids.get() + i * el_k + el_k, -1);
        std::fill(dists.get() + i * el_k + real_el_k, dists.get() + i * el_k + el_k,
                  larger_is_closer ? std::numeric_limits<float>::min() : std::numeric_limits<float>::max());
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    time = rc2.ElapseFromBegin("done");
    time *= 0.001;  // convert to ms
    knowhere_search_emb_list_2nd_bf_agg_latency.Observe(time);
#endif
    return GenResultDataSet((int64_t)num_q_el, (int64_t)el_k, std::move(ids), std::move(dists));
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
