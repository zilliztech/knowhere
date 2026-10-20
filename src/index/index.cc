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

#include "knowhere/index/index.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "folly/ScopeGuard.h"
#include "folly/futures/Future.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/thread_pool.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "common/Tracer.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/tracer.h"
#endif

namespace knowhere {

namespace {

Status
ValidateMappingMmapOptions(const MappingMmapOptions& options, bool growing = false) {
    if ((options.search_ids || options.retrieval_ids) && (growing || options.directory.empty())) {
        return Status::invalid_args;
    }
    return Status::success;
}

std::optional<ValidityData>
CopyValidity(const IdMap& map) {
    // Export may run after a file-only Build, before its backend is loaded.
    // It must not query backend Count/Dim just to inspect nullable metadata.
    const auto bitmap = map.ValidBitmap();
    if (bitmap.empty()) {
        return std::nullopt;
    }
    ValidityData result;
    result.logical_row_count = bitmap.size();
    result.packed_bits.assign(result.logical_row_count / 8 + (result.logical_row_count % 8 != 0), 0xff);
    bitmap.AndRange(0, result.logical_row_count, result.packed_bits.data());
    if (result.logical_row_count % 8 != 0) {
        result.packed_bits.back() &= (1U << (result.logical_row_count % 8)) - 1;
    }
    return result;
}

// Normalize the bit offset before passing it to kernels. Checking bytes avoids
// overflowing capacity*8 or the caller's bit_offset+count.
Status
NormalizeValidityTarget(size_t count, MutableBitmapView& target) {
    if (count == 0) {
        return Status::success;
    }
    const auto byte_begin = target.bit_offset / 8;
    const auto intra = target.bit_offset % 8;
    if (target.data == nullptr || byte_begin > target.capacity_bytes ||
        count > std::numeric_limits<size_t>::max() - intra) {
        return Status::invalid_args;
    }
    const auto span = intra + count;
    const auto need = span / 8 + (span % 8 != 0);
    if (need > target.capacity_bytes - byte_begin) {
        return Status::invalid_args;
    }
    target.data += byte_begin;
    target.capacity_bytes -= byte_begin;
    target.bit_offset = intra;
    return Status::success;
}

Status
ValidateValidityInput(const LoadOptions& input) {
    if (!input.validity.has_value()) {
        return Status::success;
    }
    const auto& validity = *input.validity;
    if (validity.logical_row_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1) {
        return Status::invalid_args;
    }
    const auto required = validity.logical_row_count / 8 + (validity.logical_row_count % 8 != 0);
    if (required > validity.size_bytes || (required != 0 && validity.data == nullptr)) {
        return Status::invalid_args;
    }
    return Status::success;
}

Status
PrepareNullableLoad(IndexNode* node, const LoadOptions& input) {
    if (input.validity.has_value()) {
        auto& map = node->GetIdMap();
        node->SetIdMapType(IdMap::Type::SEALED);
        map.ConfigureMmap({.enable_in_to_out_ids = input.mmap.search_ids,
                           .enable_out_to_in_ids = input.mmap.retrieval_ids,
                           .mmap_dir_path = input.mmap.directory});
        const auto& validity = *input.validity;
        map.AddFromData(IdMapData::FromValidBitmap(validity.data, validity.logical_row_count));
    }
    return Status::success;
}

Status
PrepareBuildMapping(IndexNode* node, const DataSetPtr& dataset, const BuildOptions& options) {
    if (dataset != nullptr && dataset->HasIdMapData()) {
        RETURN_IF_ERROR(ValidateMappingMmapOptions(options.mmap, options.growing));
        if (!node->GetIdMap().IsEnabled()) {
            node->SetIdMapType(options.growing ? IdMap::Type::GROWING : IdMap::Type::SEALED);
        }
        node->GetIdMap().ConfigureMmap({.enable_in_to_out_ids = options.mmap.search_ids,
                                        .enable_out_to_in_ids = options.mmap.retrieval_ids,
                                        .mmap_dir_path = options.mmap.directory});
    }
    return Status::success;
}

}  // namespace

inline Status
LoadConfig(BaseConfig* cfg, const Json& json, knowhere::PARAM_TYPE param_type, const std::string& method,
           std::string* const msg = nullptr) {
    Json json_(json);
    auto res = Config::FormatAndCheck(*cfg, json_, msg);
    LOG_KNOWHERE_DEBUG_ << method << " config dump: " << json_.dump();
    RETURN_IF_ERROR(res);
    cfg->CaptureRawJson(json_);
    return Config::Load(*cfg, json_, param_type, msg);
}

inline void
ThrowInvalidIdMapData(const std::string& msg) {
    throw StatusException(Status::invalid_args, msg);
}

inline bool
UsesElementIdSpace(const DataSetPtr& dataset, const BaseConfig& config) {
    const auto has_emb_list_layout =
        (dataset != nullptr && dataset->Get<const size_t*>(meta::EMB_LIST_OFFSET) != nullptr) ||
        config.emb_list_offset_file_path.has_value();
    return has_emb_list_layout && !get_el_metric_type(config.metric_type.value()).has_value();
}

inline bool
AddDatasetIdMapData(const DataSetPtr& dataset, const BaseConfig& config, IdMap& id_map, int64_t indexed_count) {
    const auto has_id_map_data = dataset != nullptr && dataset->HasIdMapData();
    if (!has_id_map_data) {
        if (id_map.IsEnabled()) {
            ThrowInvalidIdMapData("mapped id map requires id map data for every build/add batch");
        }
        return false;
    }
    if (!id_map.IsEnabled()) {
        ThrowInvalidIdMapData("id map data requires enabled id map storage");
    }
    if (indexed_count > 0 && !id_map.HasVectorMap()) {
        ThrowInvalidIdMapData("cannot append id map data to a plain index");
    }

    try {
        if (UsesElementIdSpace(dataset, config)) {
            // An ordinary metric over EmbList input builds a flat vector
            // index.  The attached nullable metadata describes outer lists,
            // but null/empty lists contribute no vectors, so flattened vector
            // ids remain the contiguous public element ids.  Materialize that
            // vector-domain identity map to preserve the normal IdMap
            // lifecycle for the element index.
            const auto vector_count = dataset->GetRows();
            if (vector_count < 0 || static_cast<uint64_t>(vector_count) > std::numeric_limits<int32_t>::max()) {
                ThrowInvalidIdMapData("element id map count overflows int32");
            }
            std::vector<int32_t> vector_ids(static_cast<size_t>(vector_count));
            std::iota(vector_ids.begin(), vector_ids.end(), 0);
            id_map.AddFromData(IdMapData::FromIds(vector_ids.data(), vector_ids.size(), vector_ids.size()));
        } else {
            // Consume nullable id-map input into index-owned state.
            id_map.AddFromData(dataset->GetIdMapData());
        }
    } catch (const std::exception& e) {
        ThrowInvalidIdMapData(e.what());
    }
    return true;
}

// Owns prepared filter state for lazy iterator calls.
class PreparedBitsetIterator : public IndexNode::iterator {
 public:
    PreparedBitsetIterator(IndexNode::IteratorPtr iterator, std::shared_ptr<const IndexNode::PreparedBitset> bitset)
        : iterator_(std::move(iterator)), bitset_(std::move(bitset)) {
    }

    expected<std::pair<int64_t, float>>
    Next() override {
        return iterator_->Next();
    }

    [[nodiscard]] expected<bool>
    HasNext() override {
        return iterator_->HasNext();
    }

 private:
    IndexNode::IteratorPtr iterator_;
    std::shared_ptr<const IndexNode::PreparedBitset> bitset_;
};

inline void
WrapIteratorsWithPreparedBitset(std::vector<IndexNode::IteratorPtr>& iterators,
                                const std::shared_ptr<const IndexNode::PreparedBitset>& bitset) {
    if (bitset == nullptr || !bitset->bitset.has_out_ids()) {
        return;
    }
    for (auto& iterator : iterators) {
        iterator = std::make_shared<PreparedBitsetIterator>(std::move(iterator), bitset);
    }
}

template <typename T>
expected<bool>
Index<T>::HasValidity() const noexcept {
    return GuardedCall([&]() -> expected<bool> {
        if (node == nullptr) {
            return expected<bool>::Err(Status::empty_index, "index is empty");
        }
        const auto& map = std::as_const(*node).GetIdMap();
        return !map.ValidBitmap().empty();
    });
}

template <typename T>
expected<size_t>
Index<T>::ValidCount() const noexcept {
    return GuardedCall([&]() -> expected<size_t> {
        if (node == nullptr) {
            return expected<size_t>::Err(Status::empty_index, "index is empty");
        }
        const auto& map = std::as_const(*node).GetIdMap();
        return map.InCount();
    });
}

template <typename T>
expected<size_t>
Index<T>::LogicalCount() const noexcept {
    return GuardedCall([&]() -> expected<size_t> {
        if (node == nullptr) {
            return expected<size_t>::Err(Status::empty_index, "index is empty");
        }
        const auto& map = std::as_const(*node).GetIdMap();
        return map.OutCount();
    });
}

template <typename T>
expected<bool>
Index<T>::IsRowValid(int64_t logical_offset) const noexcept {
    return GuardedCall([&]() -> expected<bool> {
        if (node == nullptr) {
            return expected<bool>::Err(Status::empty_index, "index is empty");
        }
        return std::as_const(*node).GetIdMap().IsValidOutId(logical_offset);
    });
}

template <typename T>
Status
Index<T>::IntersectValidity(int64_t logical_offset, size_t count, MutableBitmapView target) const noexcept {
    return GuardedCall([&]() -> Status {
        if (node == nullptr) {
            return Status::empty_index;
        }
        RETURN_IF_ERROR(NormalizeValidityTarget(count, target));
        const auto& map = std::as_const(*node).GetIdMap();
        map.AndValidityByRange(logical_offset, count, target.data, target.bit_offset);
        return Status::success;
    });
}

template <typename T>
Status
Index<T>::IntersectValidityAt(const int64_t* logical_offsets, size_t count, MutableBitmapView target) const noexcept {
    return GuardedCall([&]() -> Status {
        if (node == nullptr) {
            return Status::empty_index;
        }
        if ((count != 0 && logical_offsets == nullptr) ||
            count > std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
            return Status::invalid_args;
        }
        RETURN_IF_ERROR(NormalizeValidityTarget(count, target));
        const auto& map = std::as_const(*node).GetIdMap();
        map.ApplyValidityByOffsets(logical_offsets, count, [&](size_t i, bool valid) {
            if (!valid) {
                const auto bit = target.bit_offset + i;
                target.data[bit / 8] &= static_cast<uint8_t>(~(1U << (bit % 8)));
            }
        });
        return Status::success;
    });
}

template <typename T>
expected<std::optional<ValidityData>>
Index<T>::ExportValidity() const noexcept {
    return GuardedCall([&]() -> expected<std::optional<ValidityData>> {
        if (node == nullptr) {
            return expected<std::optional<ValidityData>>::Err(Status::empty_index, "index is empty");
        }
        return CopyValidity(std::as_const(*node).GetIdMap());
    });
}

template <typename T>
Status
Index<T>::ImportValidity(const ValidityView& input, const MappingMmapOptions& mmap) noexcept {
    return GuardedCall([&]() -> Status {
        if (node == nullptr) {
            return Status::empty_index;
        }
        const LoadOptions options{.validity = input, .mmap = mmap};
        RETURN_IF_ERROR(ValidateValidityInput(options));
        RETURN_IF_ERROR(ValidateMappingMmapOptions(mmap));
        RETURN_IF_ERROR(PrepareNullableLoad(node, options));
        return this->node->FinalizeIdMap();
    });
}

template <typename T>
inline expected<bool>
Index<T>::TryReserveBuild(const DataSetPtr dataset, const Json& json) noexcept {
    return GuardedCall([&]() -> expected<bool> {
        auto cfg = this->node->CreateConfig();
        std::string msg;
        const auto status = LoadConfig(cfg.get(), json, knowhere::TRAIN, "TryReserveBuild", &msg);
        if (status != Status::success) {
            return expected<bool>::Err(status, msg);
        }
        return this->node->TryReserveBuild(dataset, std::move(cfg));
    });
}

#ifdef KNOWHERE_WITH_CARDINAL
template <typename T>
inline const std::shared_ptr<Interrupt>
Index<T>::BuildAsync(const DataSetPtr dataset, const Json& json, const std::chrono::seconds timeout,
                     const BuildOptions& options, std::shared_ptr<milvus::OpContext> op_context) noexcept {
    return GuardedCall([&]() -> std::shared_ptr<Interrupt> {
        auto release_on_failure = folly::makeGuard([&] { this->node->ReleaseBuildReservation(); });
        auto pool = ThreadPool::GetGlobalBuildThreadPool();
        auto interrupt = std::make_shared<Interrupt>(timeout);
        interrupt->Set(pool->push([this, dataset, json, interrupt, options, op_context]() {
            return GuardedCall([&]() -> Status {
                auto release_reservation = folly::makeGuard([&] { this->node->ReleaseBuildReservation(); });
                auto cfg = this->node->CreateConfig();
                RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Build"));
                RETURN_IF_ERROR(PrepareBuildMapping(node, dataset, options));
                auto& id_map = this->node->GetIdMap();
                const auto has_id_map = AddDatasetIdMapData(dataset, *cfg, id_map, 0);
                if (has_id_map && dataset != nullptr && dataset->GetRows() == 0) {
                    // All-null nullable build produces id-map metadata only.
                    return this->node->FinalizeIdMap();
                }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
                TimeRecorder rc("BuildAsync index ", 2);
                auto res =
                    this->node->BulidAsyncEmbListIfNeed(dataset, std::move(cfg), interrupt.get(), op_context.get());
                if (res == Status::success) {
                    res = this->node->FinalizeIdMap();
                }
                auto time = rc.ElapseFromBegin("done");
                time *= 0.000001;  // convert to s
                this->node->GetBuildLatencyMetric().Observe(time);
#else
                auto res =
                    this->node->BulidAsyncEmbListIfNeed(dataset, std::move(cfg), interrupt.get(), op_context.get());
                if (res == Status::success) {
                    res = this->node->FinalizeIdMap();
                }
#endif
                return res;
            });
        }));
        release_on_failure.dismiss();
        return interrupt;
    });
}
#else
template <typename T>
inline const std::shared_ptr<Interrupt>
Index<T>::BuildAsync(const DataSetPtr dataset, const Json& json, bool use_knowhere_build_pool,
                     const BuildOptions& options, std::shared_ptr<milvus::OpContext> op_context) noexcept {
    return GuardedCall([&]() -> std::shared_ptr<Interrupt> {
        auto release_on_failure = folly::makeGuard([&] { this->node->ReleaseBuildReservation(); });
        auto pool = ThreadPool::GetGlobalBuildThreadPool();
        auto interrupt = std::make_shared<Interrupt>();
        interrupt->Set(pool->push([this, dataset, json, use_knowhere_build_pool, options, op_context]() {
            return this->Build(dataset, json, use_knowhere_build_pool, options, op_context.get());
        }));
        release_on_failure.dismiss();
        return interrupt;
    });
}
#endif

template <typename T>
inline Status
Index<T>::Build(const DataSetPtr dataset, const Json& json, bool use_knowhere_build_pool, const BuildOptions& options,
                milvus::OpContext* op_context) noexcept {
    return GuardedCall([&]() -> Status {
        auto release_reservation = folly::makeGuard([&] { this->node->ReleaseBuildReservation(); });
        auto cfg = this->node->CreateConfig();
        RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Build"));
        RETURN_IF_ERROR(PrepareBuildMapping(node, dataset, options));
        auto& id_map = this->node->GetIdMap();
        const auto has_id_map = AddDatasetIdMapData(dataset, *cfg, id_map, 0);
        if (has_id_map && dataset != nullptr && dataset->GetRows() == 0) {
            // All-null nullable build produces id-map metadata only.
            return this->node->FinalizeIdMap();
        }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        TimeRecorder rc("Build index", 2);
        auto res = this->node->BuildEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool, op_context);
        if (res == Status::success) {
            res = this->node->FinalizeIdMap();
        }
        auto time = rc.ElapseFromBegin("done");
        time *= 0.000001;  // convert to s
        this->node->GetBuildLatencyMetric().Observe(time);
#else
        auto res = this->node->BuildEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool, op_context);
        if (res == Status::success) {
            res = this->node->FinalizeIdMap();
        }
#endif
        return res;
    });
}

template <typename T>
inline Status
Index<T>::Train(const DataSetPtr dataset, const Json& json, bool use_knowhere_build_pool) noexcept {
    return GuardedCall([&]() -> Status {
        bool is_emb_list = dataset->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET) != nullptr;
        if (is_emb_list) {
            // should use Index::Build instead.
            LOG_KNOWHERE_WARNING_ << "EmbList should use Index::Build instead.";
            return Status::emb_list_inner_error;
        }
        auto cfg = this->node->CreateConfig();
        std::string msg;
        RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Train", &msg));
        return this->node->Train(dataset, std::move(cfg), use_knowhere_build_pool);
    });
}

template <typename T>
inline Status
Index<T>::Add(const DataSetPtr dataset, const Json& json, bool use_knowhere_build_pool) noexcept {
    return GuardedCall([&]() -> Status {
        auto cfg = this->node->CreateConfig();
        std::string msg;
        RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Add", &msg));
        auto& id_map = this->node->GetIdMap();
        if (dataset != nullptr && dataset->HasIdMapData() && !id_map.IsEnabled() && node->Count() == 0) {
            node->SetIdMapType(IdMap::Type::GROWING);
        }
        AddDatasetIdMapData(dataset, *cfg, id_map, this->node->Count());
        const auto is_emb_list =
            dataset != nullptr && dataset->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET) != nullptr;
        if (dataset != nullptr && dataset->GetRows() == 0 && !is_emb_list) {
            return Status::success;
        }
        return this->node->AddEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool);
    });
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::Search(const DataSetPtr dataset, const Json& json, const BitsetView& bitset_,
                 milvus::OpContext* op_context) const noexcept {
    return GuardedCall([&]() -> expected<DataSetPtr> {
        auto cfg = this->node->CreateConfig();
        std::string msg;
        const Status load_status = LoadConfig(cfg.get(), json, knowhere::SEARCH, "Search", &msg);
        if (load_status != Status::success) {
            return expected<DataSetPtr>::Err(load_status, msg);
        }
        auto bitset_or = this->node->PrepareBitset(bitset_);
        if (!bitset_or.has_value()) {
            return expected<DataSetPtr>::Err(bitset_or.error(), bitset_or.what());
        }
        auto prepared_bitset = std::move(bitset_or.value());

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        const BaseConfig& b_cfg = static_cast<const BaseConfig&>(*cfg);
        // LCOV_EXCL_START
        auto span = knowhere::tracer::StartMilvusSpanFromOpContextOrConfig("knowhere search", op_context, b_cfg);
        auto span_ptr = span != nullptr ? span->GetSpan() : milvus::tracer::SpanPtr{};
        if (span_ptr != nullptr) {
            span_ptr->SetAttribute(meta::METRIC_TYPE, b_cfg.metric_type.value());
            span_ptr->SetAttribute(meta::TOPK, b_cfg.k.value());
            span_ptr->SetAttribute(meta::ROWS, Count());
            span_ptr->SetAttribute(meta::DIM, Dim());
            span_ptr->SetAttribute(meta::NQ, dataset->GetRows());
        }
        auto op_context_trace_span_guard =
            milvus::tracer::NestedSpanGuard(op_context ? &op_context->trace_span : nullptr, span_ptr);
        // LCOV_EXCL_STOP

        TimeRecorder rc("Search");
        auto k = cfg->k.value();
        auto res = this->node->SearchEmbListIfNeed(dataset, std::move(cfg), prepared_bitset.bitset, op_context);
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;  // convert to ms
        this->node->GetSearchLatencyMetric().Observe(time);
        knowhere_search_topk.Observe(k);
#else
        auto res = this->node->SearchEmbListIfNeed(dataset, std::move(cfg), prepared_bitset.bitset, op_context);
#endif
        return res;
    });
}

template <typename T>
inline expected<std::vector<std::shared_ptr<IndexNode::iterator>>>
Index<T>::AnnIterator(const DataSetPtr dataset, const Json& json, const BitsetView& bitset_,
                      bool use_knowhere_search_pool, milvus::OpContext* op_context) const noexcept {
    return GuardedCall([&]() -> expected<std::vector<std::shared_ptr<IndexNode::iterator>>> {
        auto cfg = this->node->CreateConfig();
        std::string msg;
        Status status = LoadConfig(cfg.get(), json, knowhere::ITERATOR, "Iterator", &msg);
        if (status != Status::success) {
            return expected<std::vector<std::shared_ptr<IndexNode::iterator>>>::Err(status, msg);
        }
        auto bitset_or = this->node->PrepareBitset(bitset_);
        if (!bitset_or.has_value()) {
            return expected<std::vector<std::shared_ptr<IndexNode::iterator>>>::Err(bitset_or.error(),
                                                                                    bitset_or.what());
        }
        auto prepared_bitset = std::make_shared<IndexNode::PreparedBitset>(std::move(bitset_or.value()));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        // note that this time includes only the initial search phase of iterator.
        TimeRecorder rc("AnnIterator");
        auto res = this->node->AnnIteratorEmbListIfNeed(dataset, std::move(cfg), prepared_bitset->bitset,
                                                        use_knowhere_search_pool, op_context);
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;  // convert to ms
        this->node->GetSearchLatencyMetric().Observe(time);
#else
        auto res = this->node->AnnIteratorEmbListIfNeed(dataset, std::move(cfg), prepared_bitset->bitset,
                                                        use_knowhere_search_pool, op_context);
#endif
        if (res.has_value()) {
            WrapIteratorsWithPreparedBitset(res.value(), prepared_bitset);
        }
        return res;
    });
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::RangeSearch(const DataSetPtr dataset, const Json& json, const BitsetView& bitset_,
                      milvus::OpContext* op_context) const noexcept {
    return GuardedCall([&]() -> expected<DataSetPtr> {
        auto cfg = this->node->CreateConfig();
        std::string msg;
        auto status = LoadConfig(cfg.get(), json, knowhere::RANGE_SEARCH, "RangeSearch", &msg);
        if (status != Status::success) {
            return expected<DataSetPtr>::Err(status, std::move(msg));
        }
        auto bitset_or = this->node->PrepareBitset(bitset_);
        if (!bitset_or.has_value()) {
            return expected<DataSetPtr>::Err(bitset_or.error(), bitset_or.what());
        }
        auto prepared_bitset = std::move(bitset_or.value());

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        const BaseConfig& b_cfg = static_cast<const BaseConfig&>(*cfg);
        // LCOV_EXCL_START
        auto span = knowhere::tracer::StartMilvusSpanFromOpContextOrConfig("knowhere range search", op_context, b_cfg);
        auto span_ptr = span != nullptr ? span->GetSpan() : milvus::tracer::SpanPtr{};
        if (span_ptr != nullptr) {
            span_ptr->SetAttribute(meta::METRIC_TYPE, b_cfg.metric_type.value());
            span_ptr->SetAttribute(meta::RADIUS, b_cfg.radius.value());
            if (b_cfg.range_filter.value() != defaultRangeFilter) {
                span_ptr->SetAttribute(meta::RANGE_FILTER, b_cfg.range_filter.value());
            }
            span_ptr->SetAttribute(meta::ROWS, Count());
            span_ptr->SetAttribute(meta::DIM, Dim());
            span_ptr->SetAttribute(meta::NQ, dataset->GetRows());
        }
        auto op_context_trace_span_guard =
            milvus::tracer::NestedSpanGuard(op_context ? &op_context->trace_span : nullptr, span_ptr);
        // LCOV_EXCL_STOP

        TimeRecorder rc("Range Search");
        auto res = this->node->RangeSearchEmbListIfNeed(dataset, std::move(cfg), prepared_bitset.bitset, op_context);
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;  // convert to ms
        this->node->GetRangeSearchLatencyMetric().Observe(time);
#else
        auto res = this->node->RangeSearchEmbListIfNeed(dataset, std::move(cfg), prepared_bitset.bitset, op_context);
#endif
        return res;
    });
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const noexcept {
    return GuardedCall([&]() { return this->node->GetVectorByIds(dataset, op_context); });
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::GetEmbListByIds(const DataSetPtr dataset, const std::string& metric_type,
                          milvus::OpContext* op_context) const noexcept {
    return GuardedCall([&]() { return this->node->GetEmbListByIds(dataset, metric_type, op_context); });
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::CalcDistByIDs(const DataSetPtr dataset, const BitsetView& bitset, const int64_t* labels,
                        const size_t labels_len, const bool is_cosine, milvus::OpContext* op_context) const noexcept {
    return GuardedCall([&]() -> expected<DataSetPtr> {
        auto bitset_or = this->node->PrepareBitset(bitset);
        if (!bitset_or.has_value()) {
            return expected<DataSetPtr>::Err(bitset_or.error(), bitset_or.what());
        }
        auto prepared_bitset = std::move(bitset_or.value());
        return this->node->CalcDistByIDs(dataset, prepared_bitset.bitset, labels, labels_len, is_cosine, op_context);
    });
}

template <typename T>
inline bool
Index<T>::HasRawData(const std::string& metric_type) const noexcept {
    return GuardedCall([&]() { return this->node->HasRawData(metric_type); });
}

template <typename T>
inline bool
Index<T>::IsAdditionalScalarSupported(bool is_mv_only) const noexcept {
    return GuardedCall([&]() { return this->node->IsAdditionalScalarSupported(is_mv_only); });
}

template <typename T>
inline bool
Index<T>::IsIndexRefineEnabled() const noexcept {
    return GuardedCall([&]() { return this->node->IsIndexRefineEnabled(); });
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::GetIndexMeta(const Json& json) const noexcept {
    return GuardedCall([&]() -> expected<DataSetPtr> {
        auto cfg = this->node->CreateConfig();
        std::string msg;
        auto status = LoadConfig(cfg.get(), json, knowhere::FEDER, "GetIndexMeta", &msg);
        if (status != Status::success) {
            return expected<DataSetPtr>::Err(status, msg);
        }
        return this->node->GetIndexMeta(std::move(cfg));
    });
}

template <typename T>
inline Status
Index<T>::Serialize(BinarySet& binset) const noexcept {
    return GuardedCall([&]() { return this->node->SerializeEmbListIfNeed(binset); });
}

template <typename T>
inline Status
Index<T>::Deserialize(const BinarySet& binset, const Json& json, const LoadOptions& options) noexcept {
    return GuardedCall([&]() -> Status {
        RETURN_IF_ERROR(ValidateValidityInput(options));
        RETURN_IF_ERROR(ValidateMappingMmapOptions(options.mmap));
        RETURN_IF_ERROR(PrepareNullableLoad(node, options));
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        {
            auto res = Config::FormatAndCheck(*cfg, json_);
            LOG_KNOWHERE_DEBUG_ << "Deserialize config dump: " << json_.dump();
            if (res != Status::success) {
                return res;
            }
        }
        auto res = Config::Load(*cfg, json_, knowhere::DESERIALIZE);
        if (res != Status::success) {
            return res;
        }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        TimeRecorder rc("Load index", 2);
        res = this->node->DeserializeEmbListIfNeed(binset, std::move(cfg));
        if (res == Status::success) {
            res = this->node->FinalizeIdMap();
        }
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;  // convert to ms
        this->node->GetLoadLatencyMetric().Observe(time);
#else
        res = this->node->DeserializeEmbListIfNeed(binset, std::move(cfg));
        if (res == Status::success) {
            res = this->node->FinalizeIdMap();
        }
#endif
        return res;
    });
}

template <typename T>
inline Status
Index<T>::DeserializeFromFile(const std::string& filename, const Json& json, const LoadOptions& options) noexcept {
    return GuardedCall([&]() -> Status {
        RETURN_IF_ERROR(ValidateValidityInput(options));
        RETURN_IF_ERROR(ValidateMappingMmapOptions(options.mmap));
        RETURN_IF_ERROR(PrepareNullableLoad(node, options));
        Json json_(json);
        auto cfg = this->node->CreateConfig();
        {
            auto res = Config::FormatAndCheck(*cfg, json_);
            LOG_KNOWHERE_DEBUG_ << "DeserializeFromFile config dump: " << json_.dump();
            if (res != Status::success) {
                return res;
            }
        }
        auto res = Config::Load(*cfg, json_, knowhere::DESERIALIZE_FROM_FILE);
        if (res != Status::success) {
            return res;
        }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        TimeRecorder rc("Load index from file", 2);
        res = this->node->DeserializeFromFileIfNeed(filename, std::move(cfg));
        if (res == Status::success) {
            res = this->node->FinalizeIdMap();
        }
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;  // convert to ms
        this->node->GetLoadLatencyMetric().Observe(time);
#else
        res = this->node->DeserializeFromFileIfNeed(filename, std::move(cfg));
        if (res == Status::success) {
            res = this->node->FinalizeIdMap();
        }
#endif
        return res;
    });
}

template <typename T>
inline int64_t
Index<T>::Dim() const noexcept {
    return GuardedCall([&]() { return this->node->Dim(); });
}

template <typename T>
inline int64_t
Index<T>::Size() const noexcept {
    return GuardedCall([&]() { return this->node->Size(); });
}

template <typename T>
inline int64_t
Index<T>::Count() const noexcept {
    return GuardedCall([&]() { return this->node->Count(); });
}

template <typename T>
inline std::string
Index<T>::Type() const noexcept {
    return GuardedCall([&]() { return this->node->Type(); });
}

template <typename T>
inline bool
Index<T>::LoadIndexWithStream() const noexcept {
    return GuardedCall([&]() { return this->node->LoadIndexWithStream(); });
}

template class Index<IndexNode>;

}  // namespace knowhere
