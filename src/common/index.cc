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

#include "knowhere/index.h"

#include "knowhere/comp/time_recorder.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"

#ifdef NOT_COMPILE_FOR_SWIG
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

inline Status
LoadConfig(BaseConfig* cfg, const Json& json, knowhere::PARAM_TYPE param_type, const std::string& method,
           std::string* const msg = nullptr) {
    Json json_(json);
    auto res = Config::FormatAndCheck(*cfg, json_, msg);
    LOG_KNOWHERE_DEBUG_ << method << " config dump: " << json_.dump();
    RETURN_IF_ERROR(res);
    return Config::Load(*cfg, json_, param_type, msg);
}

template <typename T>
inline Status
Index<T>::Build(const DataSet& dataset, const Json& json) {
    auto cfg = this->node->CreateConfig();
    RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Build"));

#ifdef NOT_COMPILE_FOR_SWIG
    TimeRecorder rc("Build index", 2);
    auto res = this->node->Build(dataset, *cfg);
    auto span = rc.ElapseFromBegin("done");
    span *= 0.000001;  // convert to s
    knowhere_build_latency.Observe(span);
#else
    auto res = this->node->Build(dataset, *cfg);
#endif
    return res;
}

template <typename T>
inline Status
Index<T>::Train(const DataSet& dataset, const Json& json) {
    auto cfg = this->node->CreateConfig();
    RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Train"));
    return this->node->Train(dataset, *cfg);
}

template <typename T>
inline Status
Index<T>::Add(const DataSet& dataset, const Json& json) {
    auto cfg = this->node->CreateConfig();
    RETURN_IF_ERROR(LoadConfig(cfg.get(), json, knowhere::TRAIN, "Add"));
    return this->node->Add(dataset, *cfg);
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::Search(const DataSet& dataset, const Json& json, const BitsetView& bitset_) const {
    auto cfg = this->node->CreateConfig();
    std::string msg;
    const Status load_status = LoadConfig(cfg.get(), json, knowhere::SEARCH, "Search", &msg);
    if (load_status != Status::success) {
        return expected<DataSetPtr>::Err(load_status, msg);
    }
    const auto bitset = BitsetView(bitset_.data(), bitset_.size(), bitset_.get_filtered_out_num_());

#ifdef NOT_COMPILE_FOR_SWIG
    TimeRecorder rc("Search");
    auto res = this->node->Search(dataset, *cfg, bitset);
    auto span = rc.ElapseFromBegin("done");
    span *= 0.001;  // convert to ms
    knowhere_search_latency.Observe(span);
    knowhere_search_topk.Observe(cfg->k.value());
#else
    auto res = this->node->Search(dataset, *cfg, bitset);
#endif
    return res;
}

template <typename T>
inline expected<std::vector<std::shared_ptr<IndexNode::iterator>>>
Index<T>::AnnIterator(const DataSet& dataset, const Json& json, const BitsetView& bitset_) const {
    auto cfg = this->node->CreateConfig();
    std::string msg;
    Status status = LoadConfig(cfg.get(), json, knowhere::ITERATOR, "Iterator", &msg);
    if (status != Status::success) {
        return expected<std::vector<std::shared_ptr<IndexNode::iterator>>>::Err(status, msg);
    }
    const auto bitset = BitsetView(bitset_.data(), bitset_.size(), bitset_.get_filtered_out_num_());

#ifdef NOT_COMPILE_FOR_SWIG
    // note that this time includes only the initial search phase of iterator.
    TimeRecorder rc("AnnIterator");
    auto res = this->node->AnnIterator(dataset, *cfg, bitset);
    auto span = rc.ElapseFromBegin("done");
    span *= 0.001;  // convert to ms
    knowhere_search_latency.Observe(span);
#else
    auto res = this->node->AnnIterator(dataset, *cfg, bitset);
#endif
    return res;
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::RangeSearch(const DataSet& dataset, const Json& json, const BitsetView& bitset_) const {
    auto cfg = this->node->CreateConfig();
    std::string msg;
    auto status = LoadConfig(cfg.get(), json, knowhere::RANGE_SEARCH, "RangeSearch", &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, std::move(msg));
    }
    const auto bitset = BitsetView(bitset_.data(), bitset_.size(), bitset_.get_filtered_out_num_());

#ifdef NOT_COMPILE_FOR_SWIG
    TimeRecorder rc("Range Search");
    auto res = this->node->RangeSearch(dataset, *cfg, bitset);
    auto span = rc.ElapseFromBegin("done");
    span *= 0.001;  // convert to ms
    knowhere_range_search_latency.Observe(span);
#else
    auto res = this->node->RangeSearch(dataset, *cfg, bitset);
#endif
    return res;
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::GetVectorByIds(const DataSet& dataset) const {
    return this->node->GetVectorByIds(dataset);
}

template <typename T>
inline bool
Index<T>::HasRawData(const std::string& metric_type) const {
    return this->node->HasRawData(metric_type);
}

template <typename T>
inline expected<DataSetPtr>
Index<T>::GetIndexMeta(const Json& json) const {
    auto cfg = this->node->CreateConfig();
    std::string msg;
    auto status = LoadConfig(cfg.get(), json, knowhere::FEDER, "GetIndexMeta", &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, msg);
    }
    return this->node->GetIndexMeta(*cfg);
}

template <typename T>
inline Status
Index<T>::Serialize(BinarySet& binset) const {
    return this->node->Serialize(binset);
}

template <typename T>
inline Status
Index<T>::Deserialize(const BinarySet& binset, const Json& json) {
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

#ifdef NOT_COMPILE_FOR_SWIG
    TimeRecorder rc("Load index", 2);
    res = this->node->Deserialize(binset, *cfg);
    auto span = rc.ElapseFromBegin("done");
    span *= 0.001;  // convert to ms
    knowhere_load_latency.Observe(span);
#else
    res = this->node->Deserialize(binset, *cfg);
#endif
    return res;
}

template <typename T>
inline Status
Index<T>::DeserializeFromFile(const std::string& filename, const Json& json) {
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

#ifdef NOT_COMPILE_FOR_SWIG
    TimeRecorder rc("Load index from file", 2);
    res = this->node->DeserializeFromFile(filename, *cfg);
    auto span = rc.ElapseFromBegin("done");
    span *= 0.001;  // convert to ms
    knowhere_load_latency.Observe(span);
#else
    res = this->node->DeserializeFromFile(filename, *cfg);
#endif
    return res;
}

template <typename T>
inline int64_t
Index<T>::Dim() const {
    return this->node->Dim();
}

template <typename T>
inline int64_t
Index<T>::Size() const {
    return this->node->Size();
}

template <typename T>
inline int64_t
Index<T>::Count() const {
    return this->node->Count();
}

template <typename T>
inline std::string
Index<T>::Type() const {
    return this->node->Type();
}

template class Index<IndexNode>;

}  // namespace knowhere
