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
#include "knowhere/index/index_factory.h"
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
                    auto key_str = it.first;
                    auto value_str = json[key_str].get<std::string>();
                    try {
                        int64_t v = std::stoll(value_str, &sz);
                        if (sz < value_str.length()) {
                            KNOWHERE_THROW_MSG("wrong data type in json, key: '" + key_str + "', value: '" + value_str +
                                               "'");
                        }
                        if (v < std::numeric_limits<CFG_INT::value_type>::min() ||
                            v > std::numeric_limits<CFG_INT::value_type>::max()) {
                            if (err_msg) {
                                *err_msg =
                                    "integer value out of range, key: '" + key_str + "', value: '" + value_str + "'";
                            }
                            return knowhere::Status::invalid_value_in_json;
                        }
                        json[key_str] = static_cast<CFG_INT::value_type>(v);
                    } catch (const std::out_of_range&) {
                        if (err_msg) {
                            *err_msg = "integer value out of range, key: '" + key_str + "', value: '" + value_str + "'";
                        }
                        return knowhere::Status::invalid_value_in_json;
                    } catch (const std::invalid_argument&) {
                        KNOWHERE_THROW_MSG("invalid integer value, key: '" + key_str + "', value: '" + value_str + "'");
                    }
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
