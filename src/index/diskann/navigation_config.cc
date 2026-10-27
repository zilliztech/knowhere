// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "pq_navigation_config.h"
#include "rabitq_navigation_config.h"

namespace knowhere {
std::unique_ptr<NavigationConfig>
CreateNavigationConfig(const std::string& codec) {
    if (codec == "PQ") {
        return std::make_unique<PQNavigationConfig>();
    }
    if (codec == "RABITQ") {
        return std::make_unique<RaBitQNavigationConfig>();
    }
    throw diskann::NavigationTypeError("Unsupported DiskANN navigation codec: " + codec);
}
}  // namespace knowhere
