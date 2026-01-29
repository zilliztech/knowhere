
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
#include "knowhere/utils.h"
#include "simd/hook.h"
namespace knowhere {
/**
 * parameters:
 * refine_type, refine_ratio and refine_with_quant only used by data view index
 * - refine_type, train parameter, has several config:
 *    DATA_VIEW, not alloc extra memory in refiner
 *    FLOAT16_QUANT, keep data as float16 vector in memory in refiner
 *    BFLOAT16_QUANT, keep data as bfloat16 vector in memory in refiner
 *    UINT8_QUANT, keep data as uint8 vector in memory in refiner
 * - refine_with_quant, search parameter, whether to use quantized data to refine, faster but lost a little
 * -  refine_ratio, search parameter, the ratio of data view refiner index search out of total k,
 * precision
 */

#define DECLARE_DATA_VIEW_REFINER_MEMBERS() \
    CFG_INT refine_type;                    \
    CFG_BOOL refine_with_quant;             \
    CFG_FLOAT refine_ratio;

#define REGISTER_DATA_VIEW_REFINER_CONFIG()                                     \
    KNOWHERE_CONFIG_DECLARE_FIELD(refine_type)                                  \
        .description("refiner type , no memory by default")                     \
        .set_default(RefineType::DATA_VIEW)                                     \
        .for_train();                                                           \
    KNOWHERE_CONFIG_DECLARE_FIELD(refine_with_quant)                            \
        .description("search parameters, whether use quantized data to refine") \
        .set_default(false)                                                     \
        .for_search()                                                           \
        .for_range_search()                                                     \
        .for_iterator();                                                        \
    KNOWHERE_CONFIG_DECLARE_FIELD(refine_ratio)                                 \
        .description("refine_ratio used for refining")                          \
        .set_default(1.0f)                                                      \
        .for_search();

class IndexWithDataViewRefinerBaseConfig : public BaseConfig {
 public:
    DECLARE_DATA_VIEW_REFINER_MEMBERS()
    KNOHWERE_DECLARE_CONFIG(IndexWithDataViewRefinerBaseConfig) {
        REGISTER_DATA_VIEW_REFINER_CONFIG()
    }
};
class ScannWithDataViewRefinerConfig : public ScannConfig {
 public:
    DECLARE_DATA_VIEW_REFINER_MEMBERS()
    KNOHWERE_DECLARE_CONFIG(ScannWithDataViewRefinerConfig){REGISTER_DATA_VIEW_REFINER_CONFIG()}

    Status CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        if (!faiss::support_pq_fast_scan) {
            LOG_KNOWHERE_ERROR_ << "SCANN index is not supported on the current CPU model, avx2 support is "
                                   "needed for x86 arch.";
            return Status::invalid_instruction_set;
        }
        return Status::success;
    }
};

// Helper functions to safely get data view refiner field values from BaseConfig*
inline CFG_INT
GetRefineType(const BaseConfig* cfg) {
    if (auto* dv_cfg = dynamic_cast<const ScannWithDataViewRefinerConfig*>(cfg)) {
        return dv_cfg->refine_type;
    } else if (auto* dv_cfg = dynamic_cast<const IndexWithDataViewRefinerBaseConfig*>(cfg)) {
        return dv_cfg->refine_type;
    }
    return std::nullopt;
}

inline CFG_BOOL
GetRefineWithQuant(const BaseConfig* cfg) {
    if (auto* dv_cfg = dynamic_cast<const ScannWithDataViewRefinerConfig*>(cfg)) {
        return dv_cfg->refine_with_quant;
    } else if (auto* dv_cfg = dynamic_cast<const IndexWithDataViewRefinerBaseConfig*>(cfg)) {
        return dv_cfg->refine_with_quant;
    }
    return std::nullopt;
}

[[maybe_unused]] static void
AdaptToBaseIndexConfig(Config* cfg, PARAM_TYPE param_type, size_t dim) {
    // config can't do copy, change the base config in place.
    if (cfg == nullptr)
        return;
    if (auto base_cfg = dynamic_cast<ScannWithDataViewRefinerConfig*>(cfg)) {
        if (base_cfg->metric_type.value() == metric::COSINE) {
            base_cfg->metric_type.value() = metric::L2;
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
                base_cfg->use_elkan = false;
                break;
            }
            case PARAM_TYPE::SEARCH: {
                auto reorder_k = int(base_cfg->k.value() * base_cfg->refine_ratio.value());
                base_cfg->k = reorder_k;
                base_cfg->reorder_k = reorder_k;
                base_cfg->ensure_topk_full = true;
                break;
            }
            case PARAM_TYPE::RANGE_SEARCH: {
                base_cfg->range_filter = defaultRangeFilter;
                break;
            }
            case PARAM_TYPE::ITERATOR: {
                if (base_cfg->iterator_refine_ratio != 0.0f) {
                    base_cfg->retain_iterator_order = false;
                }
                break;
            }
            default:
                break;
        }
    } else if (auto base_cfg = dynamic_cast<IndexWithDataViewRefinerBaseConfig*>(cfg)) {
        if (base_cfg->metric_type.value() == metric::COSINE) {
            base_cfg->metric_type.value() = metric::IP;
        }
        // how to handle refine_ratio depends on different index type
        switch (param_type) {
            case PARAM_TYPE::RANGE_SEARCH: {
                base_cfg->range_filter = defaultRangeFilter;
                break;
            }
            case PARAM_TYPE::ITERATOR: {
                if (base_cfg->iterator_refine_ratio != 0.0f) {
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
