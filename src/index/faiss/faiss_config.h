// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under
// the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific language
// governing permissions and limitations under the License.

#pragma once

#include <unordered_set>

#include "knowhere/config.h"

namespace knowhere {

inline bool
IsFaissRawParamBlacklisted(const std::string& key) {
    static const std::unordered_set<std::string> kParams = {
        "analyzer_extra_info",
        "build_dram_budget_gb",
        "build_id",
        "cluster_id",
        "collection_id",
        "current_index_version",
        "current_scalar_index_version",
        "data_type",
        "element_type",
        "field_id",
        "index.nonEncoding",
        "index_build_id",
        "index_engine_version",
        "index_file_prefix",
        "index_id",
        "index_num_rows",
        "index_store_path_version",
        "index_type",
        "index_version",
        "insert_files",
        "lack_binlog_rows",
        "manifest",
        "num_rows",
        "num_build_thread_ratio",
        "opt_fields",
        "partition_id",
        "partition_key_isolation",
        "scalar_index_engine_version",
        "segment_id",
        "segment_insert_files",
        "segment_manifest",
        "stats_base_path",
        "storage_version",
        "tantivy_index_version",
    };
    return kParams.count(key) > 0;
}

class FaissConfig : public BaseConfig {
 public:
    // Required. faiss DSL understood by faiss::index_factory (fp32) or
    // faiss::index_binary_factory (bin1). Examples: "Flat", "IVF1024,PQ16x8",
    // "HNSW32,Flat", "BIVF256,Hamming".
    CFG_STRING faiss_index_name;

    // Captured subset of the incoming JSON: only keys that this config's __DICT__
    // does NOT declare (i.e. not owned by Knowhere's native config layer). Those are
    // the keys the vanilla faiss adapter forwards to faiss::ParameterSpace
    // (build) and per-family SearchParametersXxx (search). Declared keys (k,
    // metric_type, trace_id, faiss_index_name, ...) are consumed by Config::Load
    // into typed fields and therefore filtered out of raw_params at capture time.
    Json raw_params;

    KNOWHERE_DECLARE_CONFIG(FaissConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(faiss_index_name)
            .description("faiss factory string, e.g. \"IVF1024,PQ16x8\"")
            .allow_empty_without_default()
            .for_train()
            .for_deserialize()
            .for_deserialize_from_file();
    }

    void
    CaptureRawJson(const Json& json) override {
        raw_params = Json::object();
        for (auto it = json.begin(); it != json.end(); ++it) {
            // Skip keys consumed by Knowhere's typed config layer and framework
            // metadata injected by callers such as Milvus. Everything left is a
            // faiss-bound knob and will be validated by faiss_dispatch.
            if (__DICT__.count(it.key()) == 0 && !IsFaissRawParamBlacklisted(it.key())) {
                raw_params[it.key()] = it.value();
            }
        }
    }
};

}  // namespace knowhere
