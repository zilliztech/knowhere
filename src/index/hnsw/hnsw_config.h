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

#ifndef HNSW_CONFIG_H
#define HNSW_CONFIG_H

#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"

namespace knowhere {

namespace {

constexpr const CFG_INT::value_type kIteratorSeedEf = 40;
constexpr const CFG_INT::value_type kEfMinValue = 16;
constexpr const CFG_INT::value_type kDefaultRangeSearchEf = 16;

}  // namespace

class HnswConfig : public BaseConfig {
 public:
    CFG_INT M;
    CFG_INT efConstruction;
    CFG_INT ef;
    CFG_INT overview_levels;
    KNOHWERE_DECLARE_CONFIG(HnswConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(M).description("hnsw M").set_default(30).set_range(2, 2048).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(efConstruction)
            .description("hnsw efConstruction")
            .set_default(360)
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(ef)
            .description("hnsw ef")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(overview_levels)
            .description("hnsw overview levels for feder")
            .set_default(3)
            .set_range(1, 5)
            .for_feder();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        switch (param_type) {
            case PARAM_TYPE::SEARCH: {
                if (!ef.has_value()) {
                    ef = std::max(k.value(), kEfMinValue);
                } else if (k.value() > ef.value()) {
                    *err_msg = "ef(" + std::to_string(ef.value()) + ") should be larger than k(" +
                               std::to_string(k.value()) + ")";
                    LOG_KNOWHERE_ERROR_ << *err_msg;
                    return Status::out_of_range_in_json;
                }
                break;
            }
            case PARAM_TYPE::RANGE_SEARCH: {
                if (!ef.has_value()) {
                    // if ef is not set by user, set it to default
                    ef = kDefaultRangeSearchEf;
                }
                break;
            }
            default:
                break;
        }
        return Status::success;
    }
};

}  // namespace knowhere

#endif /* HNSW_CONFIG_H */
