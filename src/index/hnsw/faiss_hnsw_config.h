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

#include "index/hnsw/base_hnsw_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/tolower.h"

namespace knowhere {

class FaissHnswConfig : public BaseHnswConfig {
 public:
    CFG_INT seed_ef;

    // whether an index is built with a refine support
    CFG_BOOL refine;
    // undefined value leads to a search without a refine
    CFG_FLOAT refine_k;
    // type of refine
    CFG_STRING refine_type;

    KNOHWERE_DECLARE_CONFIG(FaissHnswConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(seed_ef)
            .description("hnsw seed_ef when using iterator")
            .set_default(kIteratorSeedEf)
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine)
            .description("whether the refine is used during the train")
            .set_default(false)
            .for_train()
            .for_static();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_k)
            .description("refine k")
            .set_default(1)
            .set_range(1, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_type)
            .description("the type of a refine index")
            .allow_empty_without_default()
            .for_train()
            .for_static();
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
            if (refine.value_or(false) || refine_type.has_value()) {
                if (err_msg) {
                    *err_msg = "refine is not supported for this index";
                    LOG_KNOWHERE_ERROR_ << *err_msg;
                }
                return Status::invalid_args;
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
                if (err_msg) {
                    *err_msg = "invalid scalar quantizer type";
                    LOG_KNOWHERE_ERROR_ << *err_msg;
                }
                return Status::invalid_args;
            }

            // check refine
            if (refine_type.has_value()) {
                if (!WhetherAcceptableRefineType(refine_type.value())) {
                    if (err_msg) {
                        *err_msg = "invalid refine type type";
                        LOG_KNOWHERE_ERROR_ << *err_msg;
                    }
                    return Status::invalid_args;
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
        // FAISS rejects nbits > 24, because it is not practical
        KNOWHERE_CONFIG_DECLARE_FIELD(nbits).description("nbits").set_default(8).for_train().set_range(1, 24);
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        switch (param_type) {
            case PARAM_TYPE::TRAIN: {
                if (dim.has_value() && m.has_value()) {
                    int vec_dim = dim.value();
                    int param_m = m.value();
                    if (vec_dim % param_m != 0) {
                        if (err_msg != nullptr) {
                            *err_msg =
                                "The dimension of the vector (dim) should be a multiple of the number of subquantizers "
                                "(m). Dimension: " +
                                std::to_string(vec_dim) + ", m: " + std::to_string(param_m);
                        }
                        return Status::invalid_args;
                    }
                }

                // check refine
                if (refine_type.has_value()) {
                    if (!WhetherAcceptableRefineType(refine_type.value())) {
                        if (err_msg) {
                            *err_msg = "invalid refine type type";
                            LOG_KNOWHERE_ERROR_ << *err_msg;
                        }
                        return Status::invalid_args;
                    }
                }
            }
            default:
                break;
        }
        return Status::success;
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
        // I'm not sure whether nrq > 16 is practical
        KNOWHERE_CONFIG_DECLARE_FIELD(nrq)
            .description("Number of residual subquantizers")
            .set_default(2)
            .for_train()
            .set_range(1, 16);
        // FAISS rejects nbits > 24, because it is not practical
        KNOWHERE_CONFIG_DECLARE_FIELD(nbits).description("nbits").set_default(8).for_train().set_range(1, 24);
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        switch (param_type) {
            case PARAM_TYPE::TRAIN: {
                if (dim.has_value() && m.has_value()) {
                    int vec_dim = dim.value();
                    int param_m = m.value();
                    if (vec_dim % param_m != 0) {
                        if (err_msg != nullptr) {
                            *err_msg =
                                "The dimension of a vector (dim) should be a multiple of the number of subquantizers "
                                "(m). Dimension: " +
                                std::to_string(vec_dim) + ", m: " + std::to_string(param_m);
                        }
                        return Status::invalid_args;
                    }
                }
                // check refine
                if (refine_type.has_value()) {
                    if (!WhetherAcceptableRefineType(refine_type.value())) {
                        if (err_msg) {
                            *err_msg = "invalid refine type type";
                            LOG_KNOWHERE_ERROR_ << *err_msg;
                        }
                        return Status::invalid_args;
                    }
                }
            }
            default:
                break;
        }
        return Status::success;
    }
};

}  // namespace knowhere

#endif /* FAISS_HNSW_CONFIG_H */
