// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "diskann/pq_navigation.h"
#include "knowhere/config.h"
#include "rabitq_navigation_config.h"

namespace knowhere {
// Register codec fields in the existing Config dictionary. The public JSON
// names stay unchanged; the graph configuration does not interpret them.
class PQNavigationConfig {
 public:
    // The ratio of the size reserved for the pq code to the size of the raw data (defined with vec_field_size_gb)
    // This parameter will replace pq_code_budget_gb to avoid calculating the actual size on the Milvus side.
    // The index can indirectly obtain pq_code_budget_gb by vec_field_size_gb * pq_code_budget_gb_ratio
    CFG_FLOAT pq_code_budget_gb_ratio;
    // Limit the size of the PQ code after the raw vector has been PQ-encoded. PQ code is a (pq_code_budget_gb * 1024 *
    // 1024 * 1024) / row_num)-dimensional uint8 vector. If pq_code_budget_gb is too large, it will be adjusted to the
    // size of dim*row_num.
    CFG_FLOAT pq_code_budget_gb;

    void
    Register(Config& config) {
        auto& __DICT__ = config.__DICT__;
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
    void
    Adjust(float vec_field_size_gb) {
        pq_code_budget_gb = std::max(pq_code_budget_gb.value(), pq_code_budget_gb_ratio.value() * vec_field_size_gb);
    }
    diskann::PQNavigationBuildOptions
    BuildOptions() const {
        return {pq_code_budget_gb.value()};
    }
    void
    AppendBuildMetadata(Json& build_params) const {
        // A loaded index does not recover its original configured budget.
        if (pq_code_budget_gb.has_value())
            build_params["pq_code_budget_gb"] = *pq_code_budget_gb;
    }
};

// Compatibility adapter for DISKANN/AiSAQ's flat JSON configuration. Codec
// selection and codec-specific validation stay here, outside graph building.
class DiskANNNavigationConfig {
 public:
    // Omission uses legacy PQ behavior; keep it distinct from an explicit choice.
    CFG_STRING navigation_codec;
    PQNavigationConfig pq;
    RaBitQNavigationConfig rabitq;

    void
    Register(Config& config) {
        auto& __DICT__ = config.__DICT__;
        KNOWHERE_CONFIG_DECLARE_FIELD(navigation_codec)
            .description("navigation codec: build defaults to PQ; load reads the stored type, optionally constrained")
            .allow_empty_without_default()
            .for_train()
            .for_deserialize()
            .for_static();
        pq.Register(config);
        rabitq.Register(config);
    }
    std::optional<std::string>
    CheckAndAdjust(PARAM_TYPE type, float vec_field_size_gb) {
        if (navigation_codec && (navigation_codec->empty() || (type == PARAM_TYPE::TRAIN && *navigation_codec != "PQ" &&
                                                               *navigation_codec != "RABITQ"))) {
            return "Unsupported DiskANN navigation codec: " + navigation_codec.value();
        }
        if (type == PARAM_TYPE::TRAIN && navigation_codec.value_or("PQ") == "PQ")
            pq.Adjust(vec_field_size_gb);
        return std::nullopt;
    }
    std::unique_ptr<diskann::NavigationStore>
    CreateForBuild() const {
        const auto codec = navigation_codec.value_or("PQ");
        if (codec == "PQ")
            return std::make_unique<diskann::PQNavigationStore>(true, false, std::string{}, pq.BuildOptions());
        if (codec == "RABITQ")
            return std::make_unique<diskann::RaBitQNavigationStore>(rabitq.BuildOptions());
        throw diskann::NavigationTypeError("Unsupported DiskANN navigation codec: " + codec);
    }
    std::optional<uint64_t>
    EstimateLoadMemory(uint64_t file_size) const {
        if (navigation_codec == "PQ")
            return file_size / 4;
        // Before reading metadata, the serialized payload is a conservative
        // model allowance. Do not assume PQ or RaBitQ's build-time bit default.
        if (file_size > std::numeric_limits<uint64_t>::max() - file_size / 4)
            return std::nullopt;
        return file_size + file_size / 4;
    }
    diskann::NavigationQuerySetup
    QuerySetup(const std::string& codec) const {
        return codec == "RABITQ" ? rabitq.QuerySetup() : diskann::NavigationQuerySetup{};
    }
    void
    AppendBuildMetadata(const std::string& codec, Json& build_params) const {
        if (codec == "PQ")
            pq.AppendBuildMetadata(build_params);
        else if (codec == "RABITQ")
            rabitq.AppendBuildMetadata(build_params);
        else
            throw diskann::NavigationTypeError("Unsupported DiskANN navigation codec: " + codec);
    }
};
}  // namespace knowhere
