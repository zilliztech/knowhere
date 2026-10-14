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
}  // namespace knowhere
