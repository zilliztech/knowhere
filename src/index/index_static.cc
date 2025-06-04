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

#include "knowhere/index/index_static.h"

#include <set>

#include "knowhere/operands.h"

namespace knowhere {

inline Status
LoadStaticConfig(BaseConfig* cfg, const Json& json, knowhere::PARAM_TYPE param_type, const std::string& method,
                 std::string* const msg = nullptr) {
    Json json_(json);
    auto res = Config::FormatAndCheck(*cfg, json_, msg);
    LOG_KNOWHERE_DEBUG_ << method << " config dump: " << json_.dump();
    RETURN_IF_ERROR(res);
    return Config::Load(*cfg, json_, param_type, msg);
}

template <typename DataType>
IndexStaticFaced<DataType>&
IndexStaticFaced<DataType>::Instance() {
    static IndexStaticFaced<DataType> instance;
    return instance;
}

template <typename DataType>
std::unique_ptr<BaseConfig>
IndexStaticFaced<DataType>::CreateConfig(const IndexType& indexType, const IndexVersion& version) {
    if (Instance().staticCreateConfigMap.find(indexType) != Instance().staticCreateConfigMap.end()) {
        return Instance().staticCreateConfigMap[indexType]();
    }
    LOG_KNOWHERE_WARNING_ << "unhandled create config for indexType: " << indexType;
    return std::make_unique<BaseConfig>();
}

template <typename DataType>
knowhere::Status
IndexStaticFaced<DataType>::ConfigCheck(const IndexType& indexType, const IndexVersion& version, const Json& params,
                                        std::string& msg) {
    auto cfg = IndexStaticFaced<DataType>::CreateConfig(indexType, version);

    const Status status = LoadStaticConfig(cfg.get(), params, knowhere::PARAM_TYPE::TRAIN, "ConfigCheck", &msg);
    if (status != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Load Config failed, msg = " << msg;
        return status;
    }

    if (Instance().staticConfigCheckMap.find(indexType) != Instance().staticConfigCheckMap.end()) {
        return Instance().staticConfigCheckMap[indexType](*cfg, version, msg);
    }

    return knowhere::Status::success;
}

template <typename DataType>
expected<Resource>
IndexStaticFaced<DataType>::EstimateLoadResource(const knowhere::IndexType& indexType,
                                                 const knowhere::IndexVersion& version, const float file_size,
                                                 const knowhere::Json& params) {
    auto cfg = IndexStaticFaced<DataType>::CreateConfig(indexType, version);

    std::string msg;
    const Status status = LoadStaticConfig(cfg.get(), params, knowhere::STATIC, "EstimateLoadResource", &msg);
    if (status != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Load Config failed, msg = " << msg;
        return expected<Resource>::Err(status, msg);
    }

    if (Instance().staticEstimateLoadResourceMap.find(indexType) != Instance().staticEstimateLoadResourceMap.end()) {
        return Instance().staticEstimateLoadResourceMap[indexType](file_size, *cfg, version);
    }

    return InternalEstimateLoadResource(file_size, *cfg, version);
}

template <typename DataType>
expected<Resource>
IndexStaticFaced<DataType>::InternalEstimateLoadResource(const float file_size, const BaseConfig& config,
                                                         const IndexVersion& version) {
    Resource resource;
    if (config.enable_mmap.has_value() && config.enable_mmap.value()) {
        resource.diskCost = file_size;
        resource.memoryCost = 0.0f;
    } else {
        resource.diskCost = 0;
        resource.memoryCost = 1.0f * file_size;
    }
    return resource;
}

template <typename DataType>
bool
IndexStaticFaced<DataType>::HasRawData(const IndexType& indexType, const IndexVersion& version, const Json& params) {
    auto cfg = IndexStaticFaced<DataType>::CreateConfig(indexType, version);
    std::string msg;
    const Status status = LoadStaticConfig(cfg.get(), params, knowhere::STATIC, "HasRawData", &msg);

    if (status != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Load Config failed, msg = " << msg;
        return false;
    }

    if (Instance().staticHasRawDataMap.find(indexType) != Instance().staticHasRawDataMap.end()) {
        return Instance().staticHasRawDataMap[indexType](*cfg, version);
    }

    static std::set<knowhere::IndexType> has_raw_data_index_set = {
        IndexEnum::INDEX_FAISS_BIN_IDMAP, IndexEnum::INDEX_FAISS_BIN_IVFFLAT, IndexEnum::INDEX_FAISS_IVFFLAT,
        IndexEnum::INDEX_FAISS_IVFFLAT_CC};

    static std::set<knowhere::IndexType> has_raw_data_index_alias_set = {"IVFBIN", "BINFLAT", "IVFFLAT", "IVFFLATCC"};

    if (has_raw_data_index_set.find(indexType) != has_raw_data_index_set.end() ||
        has_raw_data_index_alias_set.find(indexType) != has_raw_data_index_alias_set.end()) {
        return true;
    }

    return InternalStaticHasRawData(*cfg, version);
}

template <typename DataType>
bool
IndexStaticFaced<DataType>::InternalStaticHasRawData(const BaseConfig& /*config*/, const IndexVersion& version) {
    return false;
}

template <typename DataType>
std::unique_ptr<BaseConfig>
IndexStaticFaced<DataType>::InternalStaticCreateConfig() {
    return std::make_unique<BaseConfig>();
}

template <typename DataType>
knowhere::Status
IndexStaticFaced<DataType>::InternalConfigCheck(const BaseConfig& config, const IndexVersion& version,
                                                std::string& msg) {
    return knowhere::Status::success;
}

template class IndexStaticFaced<knowhere::fp32>;
template class IndexStaticFaced<knowhere::fp16>;
template class IndexStaticFaced<knowhere::bf16>;
template class IndexStaticFaced<knowhere::bin1>;
template class IndexStaticFaced<knowhere::int8>;

}  // namespace knowhere
