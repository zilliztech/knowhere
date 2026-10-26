// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "diskann/rabitq_navigation.h"
#include "navigation_config.h"

namespace knowhere {
class RaBitQNavigationConfig : public NavigationConfig {
 public:
    CFG_INT rbq_bits;
    CFG_INT rbq_bits_query;

    RaBitQNavigationConfig() {
        KNOWHERE_CONFIG_DECLARE_FIELD(rbq_bits)
            .description("RaBitQ database bits per dimension (build defaults to 1)")
            .allow_empty_without_default()
            .set_range(1, 9)
            .for_train()
            .for_static();
        KNOWHERE_CONFIG_DECLARE_FIELD(rbq_bits_query)
            .description("RaBitQ query bits; 0 uses an unquantized query")
            .set_default(4)
            .set_range(0, 8)
            .for_search()
            .for_range_search()
            .for_iterator();
    }
    std::unique_ptr<diskann::NavigationStore>
    CreateForBuild() const override {
        return std::make_unique<diskann::RaBitQNavigationStore>(BuildOptions());
    }
    std::optional<uint64_t>
    EstimateLoadMemory(uint64_t file_size) const override {
        if (file_size > std::numeric_limits<uint64_t>::max() - file_size / 4) {
            return std::nullopt;
        }
        return file_size + file_size / 4;
    }
    diskann::RaBitQNavigationBuildOptions
    BuildOptions() const {
        return {static_cast<uint8_t>(rbq_bits.value_or(1))};
    }
    diskann::NavigationQuerySetup
    QuerySetup() const override {
        return [bits = static_cast<uint8_t>(rbq_bits_query.value_or(4))](auto& computer) {
            diskann::RaBitQNavigationStore::ConfigureQuery(computer, bits);
        };
    }
    void
    AppendBuildMetadata(Json& params) const override {
        if (rbq_bits) {
            params["rbq_bits"] = *rbq_bits;
        }
    }
};
}  // namespace knowhere
