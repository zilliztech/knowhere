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

#pragma once

#include "knowhere/config.h"
#include "knowhere/operands.h"
namespace knowhere {

inline Status
MinhashConfigCheck(const size_t dim, const DataFormatEnum data_type, const uint32_t fun_type, const BaseConfig* cfg,
                   const BitsetView* bitset) {
    if (data_type != DataFormatEnum::fp32) {
        LOG_KNOWHERE_ERROR_ << "Metric MH_JACCARD only support fp32.";
        return Status::not_implemented;
    }
    uint32_t invalid_type = ~(PARAM_TYPE::TRAIN | PARAM_TYPE::SEARCH);
    if ((fun_type & invalid_type) != 0) {
        LOG_KNOWHERE_ERROR_ << "Metric MH_JACCARD only support search and train.";
        return Status::not_implemented;
    }
    if (fun_type & PARAM_TYPE::TRAIN) {
        size_t mh_d = cfg->band.value();
        if (dim % mh_d != 0) {
            LOG_KNOWHERE_ERROR_ << "Metric MH_JACCARD not supported for dim % band != 0.";
            return Status::not_implemented;
        }
    }
    if (fun_type & PARAM_TYPE::SEARCH) {
        if (bitset != nullptr && !bitset->empty()) {
            LOG_KNOWHERE_ERROR_ << "Metric MH_JACCARD not supported for bitset not empty case.";
            return Status::not_implemented;
        }
        size_t topk = cfg->k.value();
        if (topk != 1) {
            LOG_KNOWHERE_ERROR_ << "Metric MH_JACCARD not supported for topk != 1 case.";
            return Status::not_implemented;
        }
    }
    return Status::success;
}
}  // namespace knowhere
