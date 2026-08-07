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
#include <cstdint>
#include <fstream>
#include <optional>
#include <queue>
#include <unordered_set>
#include <utility>

#include "faiss/cppcontrib/knowhere/index_io.h"
#include "index/emb_list/emb_list_raw_storage.h"
#include "io/memory_io.h"
#include "knowhere/context.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/utils.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/comp/task.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

struct EmbListSeparateAnnIndexHolder {
    explicit EmbListSeparateAnnIndexHolder(Index<IndexNode>&& index_in) : index(std::move(index_in)) {
    }

    Index<IndexNode> index;
};

IndexNode*
IndexNode::AnnIndexNode() {
    if (emb_list_separate_ann_index_) {
        return emb_list_separate_ann_index_->index.Node();
    }
    return this;
}

const IndexNode*
IndexNode::AnnIndexNode() const {
    if (emb_list_separate_ann_index_) {
        return emb_list_separate_ann_index_->index.Node();
    }
    return this;
}

namespace {

inline const char*
AnnIndexTargetName(EmbListAnnIndexTarget target) {
    switch (target) {
        case EmbListAnnIndexTarget::BaseIndex:
            return "BaseIndex";
        case EmbListAnnIndexTarget::SeparateIndex:
            return "SeparateIndex";
    }
    return "Unknown";
}

inline const char*
AnnIndexDataTypeName(EmbListAnnIndexDataType data_type) {
    switch (data_type) {
        case EmbListAnnIndexDataType::SameAsBaseIndex:
            return "SameAsBaseIndex";
        case EmbListAnnIndexDataType::Fp32:
            return "Fp32";
        case EmbListAnnIndexDataType::Fp16:
            return "Fp16";
        case EmbListAnnIndexDataType::Bf16:
            return "Bf16";
        case EmbListAnnIndexDataType::Int8:
            return "Int8";
        case EmbListAnnIndexDataType::Bin1:
            return "Bin1";
    }
    return "Unknown";
}

expected<Index<IndexNode>>
CreateEmbListSeparateAnnIndex(EmbListAnnIndexDataType data_type, const std::string& index_type, int32_t version) {
    switch (data_type) {
        case EmbListAnnIndexDataType::Fp32:
            return IndexFactory::Instance().Create<fp32>(index_type, version);
        case EmbListAnnIndexDataType::SameAsBaseIndex:
            return expected<Index<IndexNode>>::Err(
                Status::not_implemented,
                "emb_list separate ANN index does not support SameAsBaseIndex; use BaseIndex target instead");
        case EmbListAnnIndexDataType::Fp16:
            return expected<Index<IndexNode>>::Err(Status::not_implemented,
                                                   "emb_list separate fp16 ANN index is not implemented");
        case EmbListAnnIndexDataType::Bf16:
            return expected<Index<IndexNode>>::Err(Status::not_implemented,
                                                   "emb_list separate bf16 ANN index is not implemented");
        case EmbListAnnIndexDataType::Int8:
            return expected<Index<IndexNode>>::Err(Status::not_implemented,
                                                   "emb_list separate int8 ANN index is not implemented");
        case EmbListAnnIndexDataType::Bin1:
            return expected<Index<IndexNode>>::Err(Status::not_implemented,
                                                   "emb_list separate binary ANN index is not implemented");
    }
    return expected<Index<IndexNode>>::Err(Status::not_implemented, "unknown emb_list separate ANN index type");
}

class ScopedMetricTypeOverride {
 public:
    ScopedMetricTypeOverride(BaseConfig& config, std::string metric_type)
        : config_(config), original_metric_type_(config.metric_type) {
        config_.metric_type = std::move(metric_type);
    }

    ~ScopedMetricTypeOverride() {
        config_.metric_type = std::move(original_metric_type_);
    }

 private:
    BaseConfig& config_;
    std::optional<std::string> original_metric_type_;
};

}  // namespace

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
    std::vector<Status> task_status(nq, Status::success);
    std::vector<std::string> task_msg(nq);

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
        while (true) {
#if defined(NOT_COMPILE_FOR_SWIG)
            checkCancellation(op_context);
#endif
            auto has_next = it->HasNext();
            if (!has_next.has_value()) {
                task_status[idx] = has_next.error();
                task_msg[idx] = has_next.what();
                return;
            }
            if (!has_next.value()) {
                break;
            }
            auto next = it->Next();
            if (!next.has_value()) {
                task_status[idx] = next.error();
                task_msg[idx] = next.what();
                return;
            }
            auto [id, dist] = next.value();
            if (has_closer_bound && too_close(dist)) {
                continue;
            }
            if (same_or_too_far(dist)) {
                break;
            }
            result_id_array[idx].push_back(id);
            result_dist_array[idx].push_back(dist);

            if (range_search_k >= 0 && result_id_array[idx].size() >= static_cast<size_t>(range_search_k)) {
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
        while (true) {
#if defined(NOT_COMPILE_FOR_SWIG)
            checkCancellation(op_context);
#endif
            auto has_next = it->HasNext();
            if (!has_next.has_value()) {
                task_status[idx] = has_next.error();
                task_msg[idx] = has_next.what();
                return;
            }
            if (!has_next.value()) {
                break;
            }
            auto next = it->Next();
            if (!next.has_value()) {
                task_status[idx] = next.error();
                task_msg[idx] = next.what();
                return;
            }
            auto [id, dist] = next.value();
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
                if (early_stop_further_bounds.size() < static_cast<size_t>(range_search_k)) {
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

    for (size_t i = 0; i < nq; i++) {
        if (task_status[i] != Status::success) {
            LOG_KNOWHERE_WARNING_ << "range search iterator error: " << task_msg[i];
            return expected<DataSetPtr>::Err(task_status[i], task_msg[i]);
        }
    }

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

    // 2. Delegate search to strategy
    return emb_list_strategy_->Search(dataset, query_offset, this, std::move(cfg), bitset, op_context);
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
IndexNode::SearchEmbListAnnIndex(const DataSetPtr dataset, std::unique_ptr<Config> config, const BitsetView& bitset,
                                 milvus::OpContext* op_context) const {
    return AnnIndexNode()->Search(dataset, std::move(config), bitset, op_context);
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
expected<DataSetPtr>
IndexNode::GetEmbListByIds(const DataSetPtr dataset, const std::string& metric_type,
                           milvus::OpContext* op_context) const {
    if (emb_list_offset_ == nullptr) {
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error,
                                         "GetEmbListByIds requires emb_list_offset, but it is not available");
    }
    auto sub_metric = get_sub_metric_type(metric_type);
    if (!sub_metric.has_value()) {
        return expected<DataSetPtr>::Err(Status::not_implemented,
                                         "GetEmbListByIds: invalid metric type " + metric_type);
    }

    // Raw data can come from emb_list_raw_storage_ (MUVERA/LEMUR) or base index (TokenANN).
    bool use_raw_storage = (emb_list_raw_storage_ != nullptr);
    if (!use_raw_storage && !HasRawData(sub_metric.value())) {
        return expected<DataSetPtr>::Err(
            Status::not_implemented,
            "GetEmbListByIds requires raw data support, but the index does not store raw vectors");
    }

    auto num_el_ids = dataset->GetRows();
    auto el_ids = dataset->GetIds();
    auto dim = use_raw_storage ? emb_list_raw_storage_->Dim() : Dim();

    // Build the output offset array
    std::vector<size_t> out_offsets(num_el_ids + 1);
    out_offsets[0] = 0;
    for (int64_t i = 0; i < num_el_ids; i++) {
        auto el_id = el_ids[i];
        if (el_id < 0 || static_cast<size_t>(el_id) >= emb_list_offset_->num_el()) {
            return expected<DataSetPtr>::Err(Status::invalid_args,
                                             "GetEmbListByIds: el_id " + std::to_string(el_id) + " out of range [0, " +
                                                 std::to_string(emb_list_offset_->num_el()) + ")");
        }
        out_offsets[i + 1] = out_offsets[i] + emb_list_offset_->get_el_len(el_id);
    }

    auto total_vecs = out_offsets[num_el_ids];

    if (total_vecs == 0) {
        // all emblist are empty list
        auto result = GenResultDataSet(num_el_ids, dim, nullptr);
        auto* offsets_ptr = new size_t[out_offsets.size()];
        std::memcpy(offsets_ptr, out_offsets.data(), out_offsets.size() * sizeof(size_t));
        result->Set(meta::EMB_LIST_OFFSET, static_cast<const size_t*>(offsets_ptr));
        return result;
    }

    const void* tensor = nullptr;

    if (use_raw_storage) {
        // MUVERA/LEMUR raw vectors are contiguous per el in emb_list_raw_storage_, use reconstruct_n.
        const size_t code_size = emb_list_raw_storage_->CodeSize();
        auto data = std::make_unique<char[]>(total_vecs * code_size);
        auto* ptr = reinterpret_cast<uint8_t*>(data.get());
        for (int64_t i = 0; i < num_el_ids; i++) {
            auto start = static_cast<int64_t>(emb_list_offset_->offset[el_ids[i]]);
            auto len = static_cast<int64_t>(out_offsets[i + 1] - out_offsets[i]);
            if (len > 0) {
                const auto status = emb_list_raw_storage_->ReconstructN(start, len, ptr);
                if (status != Status::success) {
                    return expected<DataSetPtr>::Err(status, "failed to reconstruct vectors from emb_list raw storage");
                }
                ptr += len * code_size;
            }
        }
        tensor = data.release();
    } else {
        // TokenANN: collect vec_ids and use base index GetVectorByIds
        //
        // TODO(perf): Vectors within each embedding list are contiguous in the index. However, the current
        // implementation collects all these contiguous IDs into a flat array and passes them to GetVectorByIds,
        // which internally calls reconstruct(id, ...) one vector at a time. This could be optimized by using
        // reconstruct_n(start, len, ...) or direct memcpy from raw data storage, avoiding both the redundant
        // ID array allocation and per-vector overhead. We don't do this yet because it would require
        // index-type-specific implementations (HNSW, IVF, FLAT, etc. each store raw data differently),
        // whereas the current approach works generically across all index types via the GetVectorByIds interface.
        std::vector<int64_t> vec_ids;
        vec_ids.reserve(total_vecs);
        for (int64_t i = 0; i < num_el_ids; i++) {
            size_t start = emb_list_offset_->offset[el_ids[i]];
            size_t len = out_offsets[i + 1] - out_offsets[i];
            for (size_t j = 0; j < len; j++) {
                vec_ids.push_back(static_cast<int64_t>(start + j));
            }
        }

        // Build result: transfer tensor ownership from GetVectorByIds result to new dataset
        auto vec_dataset = GenIdsDataSet(vec_ids.size(), vec_ids.data());
        auto res = GetVectorByIds(vec_dataset, op_context);
        if (!res.has_value()) {
            return res;
        }

        auto vec_result = res.value();
        tensor = vec_result->GetTensor();
        vec_result->SetIsOwner(false);
    }

    auto result = GenResultDataSet(num_el_ids, dim, tensor);
    auto* offsets_ptr = new size_t[out_offsets.size()];
    std::memcpy(offsets_ptr, out_offsets.data(), out_offsets.size() * sizeof(size_t));
    result->Set(meta::EMB_LIST_OFFSET, static_cast<const size_t*>(offsets_ptr));
    return result;
}

// NOLINTEND(google-default-arguments)

Status
IndexNode::BuildEmbList(const DataSetPtr dataset, std::shared_ptr<Config> cfg, const size_t* lims, size_t num_rows,
                        bool use_knowhere_build_pool) {
    auto& config = static_cast<BaseConfig&>(*cfg);

    // 1. Parse metric types
    auto metric_info_or = ParseEmbListMetric(config);
    if (!metric_info_or.has_value()) {
        return metric_info_or.error();
    }
    el_metric_type_ = metric_info_or.value().el_metric_type;
    auto sub_metric_type = metric_info_or.value().sub_metric_type;
    emb_list_raw_metric_type_ = sub_metric_type;

    // 2. Create document offset structure
    EmbListOffset doc_offset(lims, num_rows);

    // 3. Create emb_list strategy
    auto strategy_type = config.emb_list_strategy.value_or(meta::EMB_LIST_STRATEGY_TOKENANN);
    auto strategy_or = CreateEmbListStrategy(strategy_type, config);
    if (!strategy_or.has_value()) {
        LOG_KNOWHERE_WARNING_ << "Failed to create emb_list strategy: " << strategy_type;
        return strategy_or.error();
    }
    emb_list_strategy_ = std::move(strategy_or.value());

    // 4. Prepare the ANN dataset and ask the strategy how it should be indexed.
    auto build_data_or = emb_list_strategy_->PrepareDataForBuild(dataset, doc_offset, config);
    if (!build_data_or.has_value()) {
        LOG_KNOWHERE_WARNING_ << "Failed to prepare emb_list ANN build data";
        return build_data_or.error();
    }
    // Some strategies build an ANN index over a representation whose data type differs from the
    // raw vectors, e.g. binary LEMUR keeps raw data as bin1 but builds an fp32 learned ANN index.
    // Ask the strategy where that ANN dataset should be indexed, which data type it uses, and
    // whether it needs a metric different from the outer emb-list raw sub metric.
    const auto ann_index_spec = emb_list_strategy_->AnnIndexSpec(config);

    const auto ann_index_target = ann_index_spec.target;
    const auto ann_index_data_type = ann_index_spec.data_type;
    const auto ann_metric_type = ann_index_spec.ann_metric_type.value_or(sub_metric_type);

    // 5. Build underlying index (if strategy provides data)
    LOG_KNOWHERE_INFO_ << "Build EmbList-Index with strategy: " << strategy_type << ", metric type: " << el_metric_type_
                       << ", sub metric type: " << sub_metric_type << ", ann metric type: " << ann_metric_type
                       << ", ann index target: " << AnnIndexTargetName(ann_index_target)
                       << ", ann index data type: " << AnnIndexDataTypeName(ann_index_data_type);
    if (build_data_or.value().has_value()) {
        // The underlying ANN index consumes a plain metric, not the outer emb-list metric.
        // Keep this override scoped because cfg is shared with later emb-list build steps.
        ScopedMetricTypeOverride metric_guard(config, ann_metric_type);
        switch (ann_index_target) {
            case EmbListAnnIndexTarget::BaseIndex:
                // BaseIndex means the ANN dataset is compatible with this IndexNode's own data type.
                // No child index is allocated; the current node builds its normal ANN index directly.
                RETURN_IF_ERROR(Build(build_data_or.value().value(), cfg, use_knowhere_build_pool));
                break;
            case EmbListAnnIndexTarget::SeparateIndex: {
                // SeparateIndex is used when the strategy emits a representation whose data type differs from
                // this IndexNode, for example binary LEMUR producing fp32 learned vectors.
                auto separate_ann_index_or =
                    CreateEmbListSeparateAnnIndex(ann_index_data_type, Type(), version_.VersionNumber());
                if (!separate_ann_index_or.has_value()) {
                    LOG_KNOWHERE_WARNING_
                        << "Failed to create separate ANN index for emb_list strategy: " << strategy_type
                        << ", index type: " << Type()
                        << ", ann index data type: " << AnnIndexDataTypeName(ann_index_data_type);
                    return separate_ann_index_or.error();
                }
                auto separate_ann_index =
                    std::make_shared<EmbListSeparateAnnIndexHolder>(std::move(separate_ann_index_or.value()));
                RETURN_IF_ERROR(separate_ann_index->index.Node()->Build(build_data_or.value().value(), cfg,
                                                                        use_knowhere_build_pool));
                emb_list_separate_ann_index_ = std::move(separate_ann_index);
                break;
            }
        }
    }

    // 6. Create raw vector storage if strategy needs it
    if (emb_list_strategy_->NeedsRawVectorStorage()) {
        auto raw_storage_or = CreateEmbListRawStorageForBuild(dataset, sub_metric_type);
        if (!raw_storage_or.has_value()) {
            return raw_storage_or.error();
        }
        emb_list_raw_storage_ = std::move(raw_storage_or.value());
        LOG_KNOWHERE_INFO_ << "Created raw vector storage: " << emb_list_raw_storage_->Count()
                           << " vectors, dim=" << emb_list_raw_storage_->Dim() << ", metric=" << sub_metric_type;
    }

    // 7. Strategy post-build hook
    RETURN_IF_ERROR(emb_list_strategy_->OnBuildComplete(dataset, doc_offset, config));

    // 8. Set ID mapping if strategy requires it (Direct needs vector->doc mapping)
    emb_list_offset_ = emb_list_strategy_->GetEmbListOffset();
    if (emb_list_strategy_->NeedsBaseIndexIDMap()) {
        return SetBaseIndexIDMap();
    }

    return Status::success;
}

IndexNode::EmbListMetaHeader
IndexNode::ParseEmbListMetaHeader(const uint8_t* data, int64_t size) {
    MemoryIOReader reader(const_cast<uint8_t*>(data), size);
    int64_t magic = 0;
    readBinaryPOD(reader, magic);
    if (magic == kEmbListMetaMagic) {
        size_t type_len = 0;
        readBinaryPOD(reader, type_len);
        std::string strategy_type(reinterpret_cast<const char*>(data + reader.tellg()), type_len);
        reader.advance(type_len);
        return {.strategy_type = std::move(strategy_type),
                .strategy_blob = data + reader.tellg(),
                .strategy_blob_size = static_cast<int64_t>(reader.remaining())};
    }
    return {.strategy_type = meta::EMB_LIST_STRATEGY_TOKENANN, .strategy_blob = data, .strategy_blob_size = size};
}

Status
IndexNode::SerializeEmbList(BinarySet& binset) const {
    LOG_KNOWHERE_INFO_ << "Serialize emb_list with strategy: " << emb_list_strategy_->Type();
    try {
        // 1. Get strategy blob
        std::shared_ptr<uint8_t[]> strategy_data;
        int64_t strategy_size = 0;
        RETURN_IF_ERROR(emb_list_strategy_->Serialize(strategy_data, strategy_size));

        // 2. Build EMB_LIST_META = [magic][type_len][type][strategy_blob]
        auto strategy_type = emb_list_strategy_->Type();
        size_t type_len = strategy_type.size();

        MemoryIOWriter writer;
        int64_t magic = kEmbListMetaMagic;
        writeBinaryPOD(writer, magic);
        writeBinaryPOD(writer, type_len);
        writer(strategy_type.data(), type_len, 1);
        writer(strategy_data.get(), strategy_size, 1);

        std::shared_ptr<uint8_t[]> meta_data(writer.data());
        binset.Append(meta::EMB_LIST_META, meta_data, writer.tellg());

        // 3. Raw vector index as separate key (large, needs mmap in file path)
        if (emb_list_raw_storage_) {
            RETURN_IF_ERROR(emb_list_raw_storage_->Serialize(binset));
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "serialize emb_list error: " << e.what();
        return Status::emb_list_inner_error;
    }

    // 4. Serialize ANN index. BaseIndex is this node; SeparateIndex is the optional child index.
    if (emb_list_separate_ann_index_) {
        return emb_list_separate_ann_index_->index.Node()->Serialize(binset);
    }
    return Serialize(binset);
}

Status
IndexNode::DeserializeEmbListFromBinarySet(const BinarySet& binset, std::shared_ptr<Config> config) {
    auto& cfg = static_cast<knowhere::BaseConfig&>(*config);

    // 1. Parse metric type
    auto metric_info_or = ParseEmbListMetric(cfg);
    if (!metric_info_or.has_value()) {
        return metric_info_or.error();
    }
    el_metric_type_ = metric_info_or.value().el_metric_type;
    emb_list_raw_metric_type_ = metric_info_or.value().sub_metric_type;

    try {
        // 2. Read EMB_LIST_META and parse strategy type + strategy blob
        auto meta_bin = binset.GetByName(meta::EMB_LIST_META);
        if (!meta_bin) {
            LOG_KNOWHERE_WARNING_ << "EMB_LIST_META not found in binary set";
            return Status::emb_list_inner_error;
        }

        auto [strategy_type, strategy_blob, strategy_blob_size] =
            ParseEmbListMetaHeader(meta_bin->data.get(), meta_bin->size);

        // 3. Create strategy and deserialize strategy-specific data
        auto strategy_or = CreateEmbListStrategy(strategy_type, cfg);
        if (!strategy_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Failed to create emb_list strategy: " << strategy_type;
            return strategy_or.error();
        }
        emb_list_strategy_ = std::move(strategy_or.value());

        LOG_KNOWHERE_INFO_ << "Deserialize emb_list with strategy: " << strategy_type;
        RETURN_IF_ERROR(emb_list_strategy_->Deserialize(strategy_blob, strategy_blob_size, cfg));

        // 4. Deserialize ANN index using the same strategy spec selected at build time.
        const auto ann_index_spec = emb_list_strategy_->AnnIndexSpec(cfg);
        const auto ann_metric_type = ann_index_spec.ann_metric_type.value_or(emb_list_raw_metric_type_);
        {
            ScopedMetricTypeOverride metric_guard(cfg, ann_metric_type);
            switch (ann_index_spec.target) {
                case EmbListAnnIndexTarget::BaseIndex:
                    // BaseIndex was serialized as this IndexNode's own ANN index.
                    RETURN_IF_ERROR(Deserialize(binset, config));
                    break;
                case EmbListAnnIndexTarget::SeparateIndex: {
                    auto separate_ann_index_or =
                        CreateEmbListSeparateAnnIndex(ann_index_spec.data_type, Type(), version_.VersionNumber());
                    if (!separate_ann_index_or.has_value()) {
                        LOG_KNOWHERE_WARNING_
                            << "Failed to create separate ANN index for emb_list strategy: " << strategy_type
                            << ", index type: " << Type()
                            << ", ann index data type: " << AnnIndexDataTypeName(ann_index_spec.data_type);
                        return separate_ann_index_or.error();
                    }
                    auto separate_ann_index =
                        std::make_shared<EmbListSeparateAnnIndexHolder>(std::move(separate_ann_index_or.value()));
                    RETURN_IF_ERROR(separate_ann_index->index.Node()->Deserialize(binset, config));
                    emb_list_separate_ann_index_ = std::move(separate_ann_index);
                    break;
                }
            }
        }

        // 5. Deserialize raw vector index from BinarySet (if present)
        auto raw_index_bin = binset.GetByName(meta::EMB_LIST_RAW_INDEX);
        if (raw_index_bin) {
            auto raw_storage_or = ReadEmbListRawStorageFromBinary(raw_index_bin, emb_list_raw_metric_type_);
            if (!raw_storage_or.has_value()) {
                return raw_storage_or.error();
            }
            emb_list_raw_storage_ = std::move(raw_storage_or.value());
        } else if (emb_list_strategy_->NeedsRawVectorStorage()) {
            LOG_KNOWHERE_WARNING_ << "Strategy requires raw vector storage but EMB_LIST_RAW_INDEX not found";
            return Status::emb_list_inner_error;
        }

        // 6. Set ID mapping if needed
        emb_list_offset_ = emb_list_strategy_->GetEmbListOffset();
        if (emb_list_strategy_->NeedsBaseIndexIDMap()) {
            return SetBaseIndexIDMap();
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "deserialize emb_list error: " << e.what();
        return Status::emb_list_inner_error;
    }

    return Status::success;
}

Status
IndexNode::DeserializeEmbListFromFile(const std::string& filename, std::shared_ptr<Config> config) {
    auto& cfg = static_cast<knowhere::BaseConfig&>(*config);

    // 1. Parse metric type
    auto metric_info_or = ParseEmbListMetric(cfg);
    if (!metric_info_or.has_value()) {
        return metric_info_or.error();
    }
    el_metric_type_ = metric_info_or.value().el_metric_type;
    emb_list_raw_metric_type_ = metric_info_or.value().sub_metric_type;

    // 2. Read meta file and parse strategy type + strategy blob
    if (!cfg.emb_list_meta_file_path.has_value() || cfg.emb_list_meta_file_path.value().empty()) {
        LOG_KNOWHERE_WARNING_ << "emb_list_meta_file is empty, but metric type is emb_list";
        return Status::emb_list_inner_error;
    }
    auto emb_list_meta_file_path = cfg.emb_list_meta_file_path.value();

    // Read entire meta file into memory
    std::shared_ptr<uint8_t[]> file_data;
    int64_t file_size = 0;
    {
        std::ifstream in(emb_list_meta_file_path, std::ios::binary | std::ios::ate);
        if (!in) {
            LOG_KNOWHERE_WARNING_ << "Failed to open emb_list meta file: " << emb_list_meta_file_path;
            return Status::emb_list_inner_error;
        }
        file_size = static_cast<int64_t>(in.tellg());
        in.seekg(0);
        file_data = std::shared_ptr<uint8_t[]>(new uint8_t[file_size]);
        in.read(reinterpret_cast<char*>(file_data.get()), file_size);
        if (!in) {
            LOG_KNOWHERE_WARNING_ << "Failed to read emb_list meta file: " << emb_list_meta_file_path;
            return Status::emb_list_inner_error;
        }
    }

    if (file_size < sizeof(int32_t)) {
        LOG_KNOWHERE_WARNING_ << "emb_list meta file too small: " << file_size;
        return Status::emb_list_inner_error;
    }

    auto [strategy_type, strategy_blob, strategy_blob_size] = ParseEmbListMetaHeader(file_data.get(), file_size);

    try {
        // 3. Create strategy and deserialize strategy-specific data
        auto strategy_or = CreateEmbListStrategy(strategy_type, cfg);
        if (!strategy_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Failed to create emb_list strategy: " << strategy_type;
            return strategy_or.error();
        }
        emb_list_strategy_ = std::move(strategy_or.value());

        LOG_KNOWHERE_INFO_ << "Deserialize emb_list from file with strategy: " << strategy_type;
        RETURN_IF_ERROR(emb_list_strategy_->Deserialize(strategy_blob, strategy_blob_size, cfg));

        // 4. Deserialize ANN index from file using the same strategy spec selected at build time.
        const auto ann_index_spec = emb_list_strategy_->AnnIndexSpec(cfg);
        const auto ann_metric_type = ann_index_spec.ann_metric_type.value_or(emb_list_raw_metric_type_);
        {
            ScopedMetricTypeOverride metric_guard(cfg, ann_metric_type);
            switch (ann_index_spec.target) {
                case EmbListAnnIndexTarget::BaseIndex:
                    // BaseIndex was serialized as this IndexNode's own ANN index.
                    RETURN_IF_ERROR(DeserializeFromFile(filename, config));
                    break;
                case EmbListAnnIndexTarget::SeparateIndex: {
                    auto separate_ann_index_or =
                        CreateEmbListSeparateAnnIndex(ann_index_spec.data_type, Type(), version_.VersionNumber());
                    if (!separate_ann_index_or.has_value()) {
                        LOG_KNOWHERE_WARNING_
                            << "Failed to create separate ANN index for emb_list strategy: " << strategy_type
                            << ", index type: " << Type()
                            << ", ann index data type: " << AnnIndexDataTypeName(ann_index_spec.data_type);
                        return separate_ann_index_or.error();
                    }
                    auto separate_ann_index =
                        std::make_shared<EmbListSeparateAnnIndexHolder>(std::move(separate_ann_index_or.value()));
                    RETURN_IF_ERROR(separate_ann_index->index.Node()->DeserializeFromFile(filename, config));
                    emb_list_separate_ann_index_ = std::move(separate_ann_index);
                    break;
                }
            }
        }

        // 5. Load raw vector index from separate file (if strategy needs it)
        if (emb_list_strategy_->NeedsRawVectorStorage()) {
            if (!cfg.emb_list_raw_index_file_path.has_value() || cfg.emb_list_raw_index_file_path.value().empty()) {
                LOG_KNOWHERE_WARNING_ << "Strategy requires raw vector storage but "
                                      << "emb_list_raw_index_file_path is empty";
                return Status::emb_list_inner_error;
            }

            int io_flags = 0;
            if (cfg.enable_mmap.value()) {
                io_flags |= faiss::cppcontrib::knowhere::IO_FLAG_MMAP_IFC;
            }

            auto raw_storage_or = ReadEmbListRawStorageFromFile(cfg.emb_list_raw_index_file_path.value(), io_flags,
                                                                emb_list_raw_metric_type_);
            if (!raw_storage_or.has_value()) {
                return raw_storage_or.error();
            }
            emb_list_raw_storage_ = std::move(raw_storage_or.value());
            LOG_KNOWHERE_INFO_ << "Loaded raw vector storage from file, mmap=" << cfg.enable_mmap.value();
        }

        // 6. Set ID mapping if needed
        emb_list_offset_ = emb_list_strategy_->GetEmbListOffset();
        if (emb_list_strategy_->NeedsBaseIndexIDMap()) {
            return SetBaseIndexIDMap();
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "deserialize emb_list from file error: " << e.what();
        return Status::emb_list_inner_error;
    }

    return Status::success;
}

expected<DataSetPtr>
IndexNode::CalcDistByRawIndex(const DataSetPtr dataset, const int64_t* labels, size_t labels_len, bool is_cosine,
                              std::shared_ptr<ThreadPool> pool, milvus::OpContext* op_context) const {
    if (!emb_list_raw_storage_) {
        return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "emb_list raw storage not initialized");
    }
    return emb_list_raw_storage_->CalcDistance(dataset, labels, labels_len, emb_list_raw_metric_type_, is_cosine,
                                               std::move(pool), op_context);
}

}  // namespace knowhere
