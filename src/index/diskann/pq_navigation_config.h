// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>

#include "diskann/pq_navigation.h"
#include "navigation_config.h"

namespace knowhere {
class PQNavigationConfig : public NavigationConfig {
 public:
    // The ratio of the size reserved for the pq code to the size of the raw data (defined with vec_field_size_gb)
    // This parameter will replace pq_code_budget_gb to avoid calculating the actual size on the Milvus side.
    // The index can indirectly obtain pq_code_budget_gb by vec_field_size_gb * pq_code_budget_gb_ratio
    CFG_FLOAT pq_code_budget_gb_ratio;
    // Limit the size of the PQ code after the raw vector has been PQ-encoded. PQ code is a (pq_code_budget_gb * 1024 *
    // 1024 * 1024) / row_num)-dimensional uint8 vector. If pq_code_budget_gb is too large, it will be adjusted to the
    // size of dim*row_num.
    CFG_FLOAT pq_code_budget_gb;

    PQNavigationConfig() {
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb_ratio)
            .description("the size of PQ compared with vector field data")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_code_budget_gb)
            .description("the ratio of the size reserved for the pq code to the size of the raw data.")
            .allow_empty_without_default()
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train();
    }
    Status
    Load(const Json& json, PARAM_TYPE type, float vec_field_size_gb, std::string* error) override {
        auto status = NavigationConfig::Load(json, type, vec_field_size_gb, error);
        if (status == Status::success && type == PARAM_TYPE::TRAIN) {
            // A positive ratio remains valid before the vector field size is known.
            if ((json.contains("pq_code_budget_gb") || json.contains("pq_code_budget_gb_ratio")) &&
                pq_code_budget_gb.value_or(0) <= 0 && pq_code_budget_gb_ratio.value() <= 0) {
                return HandleError(error,
                                   "PQ navigation requires positive pq_code_budget_gb or pq_code_budget_gb_ratio",
                                   Status::invalid_args);
            }
            pq_code_budget_gb =
                std::max(pq_code_budget_gb.value_or(0), pq_code_budget_gb_ratio.value() * vec_field_size_gb);
            if (!std::isfinite(pq_code_budget_gb.value())) {
                return HandleError(error, "Derived pq_code_budget_gb must be finite", Status::invalid_args);
            }
        }
        return status;
    }
    std::unique_ptr<diskann::NavigationStore>
    CreateForBuild() const override {
        return std::make_unique<diskann::PQNavigationStore>(true, false, BuildOptions());
    }
    std::optional<uint64_t>
    EstimateLoadMemory(uint64_t file_size) const override {
        return file_size / 4;
    }
    diskann::PQNavigationBuildOptions
    BuildOptions() const {
        return {pq_code_budget_gb.value()};
    }
    void
    AppendBuildMetadata(Json& build_params) const override {
        // A loaded index does not recover its original configured budget.
        if (pq_code_budget_gb.has_value()) {
            build_params["pq_code_budget_gb"] = *pq_code_budget_gb;
        }
    }
};
}  // namespace knowhere
