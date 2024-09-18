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

#include "knowhere/config.h"

#include "index/diskann/diskann_config.h"
#include "index/flat/flat_config.h"
#include "index/gpu_raft/gpu_raft_brute_force_config.h"
#include "index/gpu_raft/gpu_raft_cagra_config.h"
#include "index/gpu_raft/gpu_raft_ivf_flat_config.h"
#include "index/gpu_raft/gpu_raft_ivf_pq_config.h"
#include "index/hnsw/hnsw_config.h"
#include "index/ivf/ivf_config.h"
#include "index/sparse/sparse_inverted_index_config.h"
#include "knowhere/log.h"

namespace knowhere {

static const std::unordered_set<std::string> ext_legal_json_keys = {"metric_type",
                                                                    "dim",
                                                                    "nlist",           // IVF param
                                                                    "nprobe",          // IVF param
                                                                    "use_elkan",       // IVF param
                                                                    "ssize",           // IVF_FLAT_CC param
                                                                    "nbits",           // IVF_PQ param
                                                                    "m",               // IVF_PQ param
                                                                    "M",               // HNSW param
                                                                    "efConstruction",  // HNSW param
                                                                    "ef",              // HNSW param
                                                                    "level",
                                                                    "index_type",
                                                                    "index_mode",
                                                                    "collection_id",
                                                                    "partition_id",
                                                                    "segment_id",
                                                                    "field_id",
                                                                    "index_build_id",
                                                                    "index_id",
                                                                    "index_version",
                                                                    "pq_code_budget_gb_ratio",
                                                                    "num_build_thread_ratio",
                                                                    "search_cache_budget_gb_ratio",
                                                                    "num_load_thread_ratio",
                                                                    "beamwidth_ratio",
                                                                    "search_list",
                                                                    "num_build_thread",
                                                                    "num_load_thread",
                                                                    "index_files",
                                                                    "gpu_id",
                                                                    "num_threads",
                                                                    "round_decimal",
                                                                    "offset",
                                                                    "for_tuning",
                                                                    "index_engine_version",
                                                                    "reorder_k"};

Status
Config::FormatAndCheck(const Config& cfg, Json& json, std::string* const err_msg) {
    // Deprecated invalid json key check for now
    // try {
    //     for (auto& it : json.items()) {
    //         // valid only if it.key() exists in one of cfg.__DICT__ and ext_legal_json_keys
    //         if (cfg.__DICT__.find(it.key()) == cfg.__DICT__.end() &&
    //             ext_legal_json_keys.find(it.key()) == ext_legal_json_keys.end()) {
    //             throw KnowhereException(std::string("invalid json key ") + it.key());
    //         }
    //     }
    // } catch (std::exception& e) {
    //     LOG_KNOWHERE_ERROR_ << e.what();
    //     if (err_msg) {
    //         *err_msg = e.what();
    //     }
    //     return Status::invalid_param_in_json;
    // }

    try {
        for (const auto& it : cfg.__DICT__) {
            const auto& var = it.second;
            if (json.find(it.first) != json.end() && json[it.first].is_string()) {
                if (std::get_if<Entry<CFG_INT>>(&var)) {
                    std::string::size_type sz;
                    auto value_str = json[it.first].get<std::string>();
                    CFG_INT::value_type v = std::stoi(value_str.c_str(), &sz);
                    if (sz < value_str.length()) {
                        KNOWHERE_THROW_MSG(std::string("wrong data type in json ") + value_str);
                    }
                    json[it.first] = v;
                }
                if (std::get_if<Entry<CFG_FLOAT>>(&var)) {
                    CFG_FLOAT::value_type v = std::stof(json[it.first].get<std::string>().c_str());
                    json[it.first] = v;
                }

                if (std::get_if<Entry<CFG_BOOL>>(&var)) {
                    if (json[it.first] == "true") {
                        json[it.first] = true;
                    }
                    if (json[it.first] == "false") {
                        json[it.first] = false;
                    }
                }
            }
        }
    } catch (std::exception& e) {
        LOG_KNOWHERE_ERROR_ << e.what();
        if (err_msg) {
            *err_msg = e.what();
        }
        return Status::invalid_value_in_json;
    }
    return Status::success;
}

}  // namespace knowhere

extern "C" __attribute__((visibility("default"))) int
CheckConfig(int index_type, char const* str, int n, int param_type);

int
CheckConfig(int index_type, const char* str, int n, int param_type) {
    if (!str || n <= 0) {
        return int(knowhere::Status::invalid_args);
    }
    knowhere::Json json = knowhere::Json::parse(str, str + n);
    std::unique_ptr<knowhere::Config> cfg;

    switch (index_type) {
        case 0:
            cfg = std::make_unique<knowhere::FlatConfig>();
            break;
        case 1:
            cfg = std::make_unique<knowhere::DiskANNConfig>();
            break;
        case 2:
            cfg = std::make_unique<knowhere::HnswConfig>();
            break;
        case 3:
            cfg = std::make_unique<knowhere::IvfFlatConfig>();
            break;
        case 4:
            cfg = std::make_unique<knowhere::IvfPqConfig>();
            break;
        case 5:
            cfg = std::make_unique<knowhere::GpuRaftCagraConfig>();
            break;
        case 6:
            cfg = std::make_unique<knowhere::GpuRaftIvfPqConfig>();
            break;
        case 7:
            cfg = std::make_unique<knowhere::GpuRaftIvfFlatConfig>();
            break;
        case 8:
            cfg = std::make_unique<knowhere::GpuRaftBruteForceConfig>();
            break;
        default:
            return int(knowhere::Status::invalid_args);
    }

    auto res = knowhere::Config::FormatAndCheck(*cfg, json, nullptr);
    if (res != knowhere::Status::success) {
        return int(res);
    }
    return int(knowhere::Config::Load(*cfg, json, knowhere::PARAM_TYPE(param_type), nullptr));
}
