// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "index/diskann/diskann_config.h"
namespace knowhere {
// PQ-specific extension, also used by AiSAQ, which only supports PQ navigation.
class PQDiskANNConfig : public DiskANNConfig {
 public:
    // The ratio of the size reserved for the pq code to the size of the raw data (defined with vec_field_size_gb)
    // This parameter will replace pq_code_budget_gb to avoid calculating the actual size on the Milvus side.
    // The index can indirectly obtain pq_code_budget_gb by vec_field_size_gb * pq_code_budget_gb_ratio
    CFG_FLOAT pq_code_budget_gb_ratio;
    // Limit the size of the PQ code after the raw vector has been PQ-encoded. PQ code is a (pq_code_budget_gb * 1024 *
    // 1024 * 1024) / row_num)-dimensional uint8 vector. If pq_code_budget_gb is too large, it will be adjusted to the
    // size of dim*row_num.
    CFG_FLOAT pq_code_budget_gb;
    KNOWHERE_DECLARE_CONFIG(PQDiskANNConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb_ratio)
            .description("the size of PQ compared with vector field data")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb)
            .description("the ratio of the size reserved for the pq code to the size of the raw data.")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
    }
    Status
    CheckAndAdjust(PARAM_TYPE phase, std::string* error) override {
        const auto status = DiskANNConfig::CheckAndAdjust(phase, error);
        if (status != Status::success)
            return status;
        if (phase == PARAM_TYPE::TRAIN)
            pq_code_budget_gb =
                std::max(pq_code_budget_gb.value(), pq_code_budget_gb_ratio.value() * vec_field_size_gb.value());
        return Status::success;
    }
};
// Metadata only describes supplied build settings; budgets are not persisted.
// Do not invoke TRAIN defaults/validation for a loaded-model metadata request.
class PQNavigationMetadataConfig : public Config {
 public:
    CFG_FLOAT pq_code_budget_gb;
    CFG_FLOAT pq_code_budget_gb_ratio;
    CFG_FLOAT vec_field_size_gb;
    KNOWHERE_DECLARE_CONFIG(PQNavigationMetadataConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb)
            .allow_empty_without_default()
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_feder();
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb_ratio)
            .allow_empty_without_default()
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_feder();
        KNOWHERE_CONFIG_DECLARE_FIELD(vec_field_size_gb)
            .allow_empty_without_default()
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_feder();
    }
    std::optional<double>
    Budget() const {
        if (pq_code_budget_gb_ratio.value_or(0) != 0 && !vec_field_size_gb.has_value())
            return std::nullopt;
        if (!pq_code_budget_gb.has_value() && !(pq_code_budget_gb_ratio.has_value() && vec_field_size_gb.has_value()))
            return std::nullopt;
        return std::max(pq_code_budget_gb.value_or(0),
                        pq_code_budget_gb_ratio.value_or(0) * vec_field_size_gb.value_or(0));
    }
};

}  // namespace knowhere
