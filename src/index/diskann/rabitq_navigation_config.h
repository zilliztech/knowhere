// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "knowhere/config.h"
namespace knowhere {
class RaBitQBuildConfig : public Config {
 public:
    CFG_INT rbq_bits;
    KNOWHERE_DECLARE_CONFIG(RaBitQBuildConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(rbq_bits)
            .description("RaBitQ database bits; build defaults to 1")
            .allow_empty_without_default()
            .set_range(1, 9)
            .for_train()
            .for_static();
    }
};
class RaBitQQueryConfig : public Config {
 public:
    CFG_INT rbq_bits_query;
    KNOWHERE_DECLARE_CONFIG(RaBitQQueryConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(rbq_bits_query)
            .description("RaBitQ query bits; 0 uses FP32")
            .set_default(4)
            .set_range(0, 8)
            .for_search()
            .for_iterator();
    }
};
}  // namespace knowhere
