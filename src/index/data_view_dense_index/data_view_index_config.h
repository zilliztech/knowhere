
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

#ifndef DATA_VIEW_INDEX_CONFIG_H
#define DATA_VIEW_INDEX_CONFIG_H

#include "index/ivf/ivf_config.h"
#include "simd/hook.h"
namespace knowhere {
class IndexWithDataViewRefinerConfig : public ScannConfig {
 public:
    CFG_INT reorder_k;
    KNOHWERE_DECLARE_CONFIG(IndexWithDataViewRefinerConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(reorder_k)
            .description("reorder k used for refining")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_search();
    }
    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        if (param_type == PARAM_TYPE::TRAIN) {
            auto topk = k.value();
            if (!reorder_k.has_value()) {
                reorder_k = topk;
            } else if (reorder_k.value() < topk) {
                if (!err_msg) {
                    err_msg = new std::string();
                }
                std::string msg = "reorder_k(" + std::to_string(reorder_k.value()) + ") should be larger than k(" +
                                  std::to_string(k.value()) + ")";
                return HandleError(err_msg, msg, Status::out_of_range_in_json);
            }
        }
        return Status::success;
    }
};
class ScannWithDataViewRefinerConfig : public ScannConfig {
 public:
    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        if (param_type == PARAM_TYPE::TRAIN) {
            auto topk = k.value();
            if (!reorder_k.has_value()) {
                reorder_k = topk;
            } else if (reorder_k.value() < topk) {
                if (!err_msg) {
                    err_msg = new std::string();
                }
                std::string msg = "reorder_k(" + std::to_string(reorder_k.value()) + ") should be larger than k(" +
                                  std::to_string(k.value()) + ")";
                return HandleError(err_msg, msg, Status::out_of_range_in_json);
            }
        }
        if (!faiss::support_pq_fast_scan) {
            LOG_KNOWHERE_ERROR_ << "SCANN index is not supported on the current CPU model, avx2 support is "
                                   "needed for x86 arch.";
            return Status::invalid_instruction_set;
        } else {
            return Status::success;
        }
        return Status::success;
    }
};

static void
AdaptToBaseIndexConfig(Config* cfg, PARAM_TYPE param_type, size_t dim) {
    // config can't do copy, change the base config in place.
    if (cfg == nullptr)
        return;
    if (auto base_cfg = dynamic_cast<ScannWithDataViewRefinerConfig*>(cfg)) {
        if (base_cfg->metric_type.value() == metric::COSINE) {
            base_cfg->metric_type.value() = metric::IP;
        }
        switch (param_type) {
            case PARAM_TYPE::TRAIN: {
                base_cfg->with_raw_data = false;
                int sub_dim = base_cfg->sub_dim.value();
                if (dim % sub_dim != 0) {
                    dim = ROUND_UP(dim, sub_dim);
                    base_cfg->dim = dim;
                } else {
                    base_cfg->dim = dim;
                }
                break;
            }
            case PARAM_TYPE::SEARCH: {
                if (base_cfg->reorder_k.has_value()) {
                    base_cfg->k = base_cfg->reorder_k.value();
                }
                break;
            }
            case PARAM_TYPE::RANGE_SEARCH: {
                base_cfg->range_filter = defaultRangeFilter;
                break;
            }
            case PARAM_TYPE::ITERATOR: {
                if (base_cfg->iterator_refine_ratio != 0.0) {
                    base_cfg->retain_iterator_order = false;
                }
                break;
            }
            default:
                break;
        }
    } else if (auto base_cfg = dynamic_cast<IndexWithDataViewRefinerConfig*>(cfg)) {
        if (base_cfg->metric_type.value() == metric::COSINE) {
            base_cfg->metric_type.value() = metric::IP;
        }
        switch (param_type) {
            case PARAM_TYPE::SEARCH: {
                base_cfg->k = base_cfg->reorder_k.value();
                break;
            }
            case PARAM_TYPE::RANGE_SEARCH: {
                base_cfg->range_filter = defaultRangeFilter;
                break;
            }
            case PARAM_TYPE::ITERATOR: {
                if (base_cfg->iterator_refine_ratio != 0.0) {
                    base_cfg->retain_iterator_order = false;
                }
                break;
            }
            default:
                break;
        }
    } else {
        throw std::runtime_error("Not a valid config for DV(Data View) refiner index.");
    }
}
}  // namespace knowhere
#endif
