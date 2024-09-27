// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef FAISS_HNSW_CONFIG_H
#define FAISS_HNSW_CONFIG_H

#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/tolower.h"

namespace knowhere {

namespace {

constexpr const CFG_INT::value_type kIteratorSeedEf = 40;
constexpr const CFG_INT::value_type kEfMinValue = 16;
constexpr const CFG_INT::value_type kDefaultRangeSearchEf = 512;

}  // namespace

class FaissHnswConfig : public BaseConfig {
 public:
    CFG_INT M;
    CFG_INT efConstruction;
    CFG_INT ef;
    CFG_INT seed_ef;
    CFG_INT overview_levels;

    // whether an index is built with a refine support
    CFG_BOOL refine;
    // undefined value leads to a search without a refine
    CFG_FLOAT refine_k;
    // type of refine
    CFG_STRING refine_type;

    KNOHWERE_DECLARE_CONFIG(FaissHnswConfig) {
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
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(seed_ef)
            .description("hnsw seed_ef when using iterator")
            .set_default(kIteratorSeedEf)
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(overview_levels)
            .description("hnsw overview levels for feder")
            .set_default(3)
            .set_range(1, 5)
            .for_feder();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine)
            .description("whether the refine is used during the train")
            .set_default(false)
            .for_train()
            .for_static();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_k)
            .description("refine k")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_type)
            .description("the type of a refine index")
            .allow_empty_without_default()
            .for_train()
            .for_static();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        switch (param_type) {
            case PARAM_TYPE::SEARCH: {
                // validate ef
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

 protected:
    bool
    WhetherAcceptableRefineType(const std::string& refine_type) {
        // 'flat' is identical to 'fp32'
        std::vector<std::string> allowed_list = {"sq6", "sq8", "fp16", "bf16", "fp32", "flat"};
        std::string refine_type_tolower = str_to_lower(refine_type);

        for (const auto& allowed : allowed_list) {
            if (refine_type_tolower == allowed) {
                return true;
            }
        }

        return false;
    }
};

class FaissHnswFlatConfig : public FaissHnswConfig {
 public:
    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        // check the base class
        const auto base_status = FaissHnswConfig::CheckAndAdjust(param_type, err_msg);
        if (base_status != Status::success) {
            return base_status;
        }

        // check our parameters
        if (param_type == PARAM_TYPE::TRAIN) {
            // prohibit refine
            if (refine.value_or(false) || refine_type.has_value() || refine_k.has_value()) {
                *err_msg = "refine is not supported for this index";
                LOG_KNOWHERE_ERROR_ << *err_msg;
                return Status::invalid_value_in_json;
            }
        }
        return Status::success;
    }
};

class FaissHnswSqConfig : public FaissHnswConfig {
 public:
    // user can use quant_type to control quantizer type.
    // we have fp16, bf16, etc, so '8', '4' and '6' is insufficient
    CFG_STRING sq_type;
    KNOHWERE_DECLARE_CONFIG(FaissHnswSqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(sq_type)
            .set_default("SQ8")
            .description("scalar quantizer type")
            .for_train()
            .for_static();
    };

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        // check the base class
        const auto base_status = FaissHnswConfig::CheckAndAdjust(param_type, err_msg);
        if (base_status != Status::success) {
            return base_status;
        }

        // check our parameters
        if (param_type == PARAM_TYPE::TRAIN) {
            auto sq_type_v = sq_type.value();
            if (!WhetherAcceptableQuantType(sq_type_v)) {
                *err_msg = "invalid scalar quantizer type";
                LOG_KNOWHERE_ERROR_ << *err_msg;
                return Status::invalid_value_in_json;
            }

            // check refine
            if (refine_type.has_value()) {
                if (!WhetherAcceptableRefineType(refine_type.value())) {
                    *err_msg = "invalid refine type type";
                    LOG_KNOWHERE_ERROR_ << *err_msg;
                    return Status::invalid_value_in_json;
                }
            }
        }
        return Status::success;
    }

 private:
    bool
    WhetherAcceptableQuantType(const std::string& sq_type) {
        // todo: add more
        std::vector<std::string> allowed_list = {"sq6", "sq8", "fp16", "bf16"};
        std::string sq_type_tolower = str_to_lower(sq_type);

        for (const auto& allowed : allowed_list) {
            if (sq_type_tolower == allowed) {
                return true;
            }
        }

        return false;
    }
};

class FaissHnswPqConfig : public FaissHnswConfig {
 public:
    // number of subquantizers
    CFG_INT m;
    // number of bits per subquantizer
    CFG_INT nbits;

    KNOHWERE_DECLARE_CONFIG(FaissHnswPqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(m).description("m").set_default(32).for_train().set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(nbits).description("nbits").set_default(8).for_train().set_range(1, 16);
    }
};

class FaissHnswPrqConfig : public FaissHnswConfig {
 public:
    // number of subquantizer splits
    CFG_INT m;
    // number of residual quantizers
    CFG_INT nrq;
    // number of bits per subquantizer
    CFG_INT nbits;
    KNOHWERE_DECLARE_CONFIG(FaissHnswPrqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(m).description("Number of splits").set_default(2).for_train().set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(nrq).description("Number of residual subquantizers").for_train().set_range(1, 64);
        KNOWHERE_CONFIG_DECLARE_FIELD(nbits).description("nbits").set_default(8).for_train().set_range(1, 64);
    }
};

}  // namespace knowhere

#endif /* FAISS_HNSW_CONFIG_H */
