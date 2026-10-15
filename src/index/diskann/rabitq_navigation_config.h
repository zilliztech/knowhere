// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "diskann/rabitq_navigation.h"
#include "knowhere/config.h"

namespace knowhere {
class RaBitQNavigationConfig {
 public:
    CFG_INT rbq_bits;
    CFG_INT rbq_bits_query;

    void
    Register(Config& config) {
        auto& __DICT__ = config.__DICT__;
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
    diskann::RaBitQNavigationBuildOptions
    BuildOptions() const {
        return {static_cast<uint8_t>(rbq_bits.value_or(1))};
    }
    diskann::NavigationQuerySetup
    QuerySetup() const {
        return [bits = static_cast<uint8_t>(rbq_bits_query.value_or(4))](auto& computer) {
            diskann::RaBitQNavigationStore::ConfigureQuery(computer, bits);
        };
    }
    void
    AppendBuildMetadata(Json& params) const {
        if (rbq_bits)
            params["rbq_bits"] = *rbq_bits;
    }
};
}  // namespace knowhere
