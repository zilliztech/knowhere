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

#ifndef COMP_KNOWHERE_CHECKER_H
#define COMP_KNOWHERE_CHECKER_H

#include <string>

#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#ifdef KNOWHERE_WITH_CARDINAL
#include "cardinal/cardinal_utils.h"
#endif

namespace knowhere {
namespace KnowhereCheck {
static bool
IndexTypeAndDataTypeCheck(const std::string& index_name, VecType data_type) {
    auto& static_index_table = std::get<0>(IndexFactory::StaticIndexTableInstance());
    auto key = std::pair<std::string, VecType>(index_name, data_type);
    if (static_index_table.find(key) != static_index_table.end()) {
        return true;
    } else {
        return false;
    }
}

static bool
SuppportMmapIndexTypeCheck(const std::string& index_name) {
    auto& mmap_index_table = std::get<1>(IndexFactory::StaticIndexTableInstance());
    if (mmap_index_table.find(index_name) != mmap_index_table.end()) {
        return true;
    } else {
        return false;
    }
}

inline bool
CheckBooleanInJson(const knowhere::Json& json, std::string key, bool target) {
    if (json.find(key) == json.end()) {
        return false;
    }
    if (json[key].is_boolean()) {
        return json[key] == target;
    }
    if (json[key].is_string()) {
        if (target) {
            return json[key] == "true";
        } else {
            return json[key] == "false";
        }
    }
    return false;
}

template <typename DataType>
bool
IndexHasRawData(const knowhere::IndexType& indexType, const knowhere::MetricType& metricType,
                const knowhere::IndexVersion& version, const knowhere::Json& params) {
    static std::set<knowhere::IndexType> has_raw_data_index_set = {
        IndexEnum::INDEX_FAISS_BIN_IDMAP,  IndexEnum::INDEX_FAISS_BIN_IVFFLAT, IndexEnum::INDEX_FAISS_IVFFLAT,
        IndexEnum::INDEX_FAISS_IVFFLAT_CC, IndexEnum::INDEX_HNSW_SQ8_REFINE,   IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
        IndexEnum::INDEX_SPARSE_WAND};
    static std::set<knowhere::IndexType> has_raw_data_index_alias_set = {"IVFBIN", "BINFLAT", "IVFFLAT", "IVFFLATCC"};

    static std::set<knowhere::IndexType> no_raw_data_index_set = {
        IndexEnum::INDEX_FAISS_IVFPQ,     IndexEnum::INDEX_FAISS_IVFSQ8,      IndexEnum::INDEX_HNSW_SQ8,
        IndexEnum::INDEX_FAISS_GPU_IDMAP, IndexEnum::INDEX_FAISS_GPU_IVFFLAT, IndexEnum::INDEX_FAISS_GPU_IVFSQ8,
        IndexEnum::INDEX_FAISS_GPU_IVFPQ, IndexEnum::INDEX_RAFT_CAGRA,        IndexEnum::INDEX_GPU_CAGRA,
        IndexEnum::INDEX_RAFT_IVFFLAT,    IndexEnum::INDEX_GPU_IVFFLAT,       IndexEnum::INDEX_RAFT_IVFPQ,
        IndexEnum::INDEX_GPU_IVFPQ,
    };

    static std::set<knowhere::IndexType> no_raw_data_index_alias_set = {"IVFPQ", "IVFSQ"};

    static std::set<knowhere::IndexType> conditional_hold_raw_data_index_set = {
        IndexEnum::INDEX_HNSW,           IndexEnum::INDEX_DISKANN,     IndexEnum::INDEX_FAISS_SCANN,
        IndexEnum::INDEX_FAISS_IVFSQ_CC, IndexEnum::INDEX_FAISS_IDMAP,
    };

    if (has_raw_data_index_set.find(indexType) != has_raw_data_index_set.end() ||
        has_raw_data_index_alias_set.find(indexType) != has_raw_data_index_alias_set.end()) {
        return true;
    }

    if (no_raw_data_index_set.find(indexType) != no_raw_data_index_set.end() ||
        no_raw_data_index_alias_set.find(indexType) != no_raw_data_index_alias_set.end()) {
        return false;
    }

    if (conditional_hold_raw_data_index_set.find(indexType) != conditional_hold_raw_data_index_set.end()) {
        if (indexType == IndexEnum::INDEX_HNSW) {
#ifdef KNOWHERE_WITH_CARDINAL
            return IndexHoldRawData<DataType>(indexType, metricType, version, params);
#else
            return true;
#endif
        } else if (indexType == IndexEnum::INDEX_DISKANN) {
#ifdef KNOWHERE_WITH_CARDINAL
            return IndexHoldRawData<DataType>(indexType, metricType, version, params);
#else
            return IsMetricType(metricType, metric::L2) || IsMetricType(metricType, metric::COSINE);
#endif
        } else if (indexType == IndexEnum::INDEX_FAISS_SCANN) {
            return !CheckBooleanInJson(params, indexparam::WITH_RAW_DATA, false);
            // INDEX_FAISS_IVFSQ_CC is not online yet
        } else if (indexType == IndexEnum::INDEX_FAISS_IVFSQ_CC) {
            return params.find(indexparam::RAW_DATA_STORE_PREFIX) != params.end();
        } else if (indexType == IndexEnum::INDEX_FAISS_IDMAP) {
            if (knowhere::Version(version) <= Version::GetMinimalVersion()) {
                return !IsMetricType(metricType, metric::COSINE);
            } else {
                return true;
            }
        } else {
            LOG_KNOWHERE_ERROR_ << "unhandled index type : " << indexType;
        }
    } else {
        LOG_KNOWHERE_ERROR_ << "unknown index type : " << indexType;
    }
    return false;
}

}  // namespace KnowhereCheck
}  // namespace knowhere

#endif /* COMP_KNOWHERE_CHECKER_H */
