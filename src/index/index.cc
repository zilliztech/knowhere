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

#include "folly/futures/Future.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/thread_pool.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/prometheus_client.h"
#include "knowhere/tracer.h"
#endif

namespace knowhere {

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

class PreparedBitsetIterator : public IndexNode::iterator {
 public:
    PreparedBitsetIterator(IndexNode::IteratorPtr iterator, std::shared_ptr<const IndexNode::PreparedBitset> bitset)
        : iterator_(std::move(iterator)), bitset_(std::move(bitset)) {
    }

    std::pair<int64_t, float>
    Next() override {
        return iterator_->Next();
    }

    bool
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

#ifdef KNOWHERE_WITH_CARDINAL
template <typename T>
inline const std::shared_ptr<Interrupt>
Index<T>::BuildAsync(const DataSetPtr dataset, const Json& json, const std::chrono::seconds timeout) noexcept {
    return GuardedCall([&]() -> std::shared_ptr<Interrupt> {
        auto pool = ThreadPool::GetGlobalBuildThreadPool();
        auto interrupt = std::make_shared<Interrupt>(timeout);
        interrupt->Set(pool->push([this, dataset, json, interrupt]() {
            return GuardedCall([&]() -> Status {
                auto cfg = this->node->CreateConfig();
                RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Build"));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
                TimeRecorder rc("BuildAsync index ", 2);
                auto res = this->node->BulidAsyncEmbListIfNeed(dataset, std::move(cfg), interrupt.get());
                if (res == Status::success) {
                    res = this->node->FinalizeIdMap();
                }
                auto time = rc.ElapseFromBegin("done");
                time *= 0.000001;  // convert to s
                this->node->GetBuildLatencyMetric().Observe(time);
#else
                auto res = this->node->BulidAsyncEmbListIfNeed(dataset, std::move(cfg), interrupt.get());
                if (res == Status::success) {
                    res = this->node->FinalizeIdMap();
                }
#endif
                return res;
            });
        }));
        return interrupt;
    });
}
#else
template <typename T>
inline const std::shared_ptr<Interrupt>
Index<T>::BuildAsync(const DataSetPtr dataset, const Json& json, bool use_knowhere_build_pool) noexcept {
    return GuardedCall([&]() -> std::shared_ptr<Interrupt> {
        auto pool = ThreadPool::GetGlobalBuildThreadPool();
        auto interrupt = std::make_shared<Interrupt>();
        interrupt->Set(pool->push([this, dataset, json, use_knowhere_build_pool]() {
            return this->Build(dataset, json, use_knowhere_build_pool);
        }));
        return interrupt;
    });
}
#endif

template <typename T>
inline Status
Index<T>::Build(const DataSetPtr dataset, const Json& json, bool use_knowhere_build_pool) noexcept {
    return GuardedCall([&]() -> Status {
        auto cfg = this->node->CreateConfig();
        RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Build"));

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        TimeRecorder rc("Build index", 2);
        auto res = this->node->BuildEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool);
        if (res == Status::success) {
            res = this->node->FinalizeIdMap();
        }
        auto time = rc.ElapseFromBegin("done");
        time *= 0.000001;  // convert to s
        this->node->GetBuildLatencyMetric().Observe(time);
#else
        auto res = this->node->BuildEmbListIfNeed(dataset, std::move(cfg), use_knowhere_build_pool);
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
        std::shared_ptr<tracer::trace::Span> span = nullptr;
        if (b_cfg.trace_id.has_value()) {
            auto trace_id_str = tracer::GetIDFromHexStr(b_cfg.trace_id.value());
            auto span_id_str = tracer::GetIDFromHexStr(b_cfg.span_id.value());
            auto ctx = tracer::TraceContext{.traceID = reinterpret_cast<const uint8_t*>(trace_id_str.c_str()),
                                            .spanID = reinterpret_cast<const uint8_t*>(span_id_str.c_str()),
                                            .traceFlags = static_cast<uint8_t>(b_cfg.trace_flags.value())};
            span = tracer::StartSpan("knowhere search", &ctx);
            span->SetAttribute(meta::METRIC_TYPE, b_cfg.metric_type.value());
            span->SetAttribute(meta::TOPK, b_cfg.k.value());
            span->SetAttribute(meta::ROWS, Count());
            span->SetAttribute(meta::DIM, Dim());
            span->SetAttribute(meta::NQ, dataset->GetRows());
        }
        // LCOV_EXCL_STOP

        TimeRecorder rc("Search");
        bool has_trace_id = b_cfg.trace_id.has_value();
        auto k = cfg->k.value();
        auto res = this->node->SearchEmbListIfNeed(dataset, std::move(cfg), prepared_bitset.bitset, op_context);
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;  // convert to ms
        this->node->GetSearchLatencyMetric().Observe(time);
        knowhere_search_topk.Observe(k);

        // LCOV_EXCL_START
        if (has_trace_id) {
            span->End();
        }
        // LCOV_EXCL_STOP
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
        std::shared_ptr<tracer::trace::Span> span = nullptr;
        if (b_cfg.trace_id.has_value()) {
            auto trace_id_str = tracer::GetIDFromHexStr(b_cfg.trace_id.value());
            auto span_id_str = tracer::GetIDFromHexStr(b_cfg.span_id.value());
            auto ctx = tracer::TraceContext{.traceID = reinterpret_cast<const uint8_t*>(trace_id_str.c_str()),
                                            .spanID = reinterpret_cast<const uint8_t*>(span_id_str.c_str()),
                                            .traceFlags = static_cast<uint8_t>(b_cfg.trace_flags.value())};
            span = tracer::StartSpan("knowhere range search", &ctx);
            span->SetAttribute(meta::METRIC_TYPE, b_cfg.metric_type.value());
            span->SetAttribute(meta::RADIUS, b_cfg.radius.value());
            if (b_cfg.range_filter.value() != defaultRangeFilter) {
                span->SetAttribute(meta::RANGE_FILTER, b_cfg.range_filter.value());
            }
            span->SetAttribute(meta::ROWS, Count());
            span->SetAttribute(meta::DIM, Dim());
            span->SetAttribute(meta::NQ, dataset->GetRows());
        }
        // LCOV_EXCL_STOP

        TimeRecorder rc("Range Search");
        bool has_trace_id = b_cfg.trace_id.has_value();
        auto res = this->node->RangeSearchEmbListIfNeed(dataset, std::move(cfg), prepared_bitset.bitset, op_context);
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;  // convert to ms
        this->node->GetRangeSearchLatencyMetric().Observe(time);

        // LCOV_EXCL_START
        if (has_trace_id) {
            span->End();
        }
        // LCOV_EXCL_STOP
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
Index<T>::Deserialize(const BinarySet& binset, const Json& json) noexcept {
    return GuardedCall([&]() -> Status {
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
Index<T>::DeserializeFromFile(const std::string& filename, const Json& json) noexcept {
    return GuardedCall([&]() -> Status {
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
