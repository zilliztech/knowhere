// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "index/diskann/navigation_store.h"

#include <stdexcept>

#include "index/diskann/diskann_config.h"
#include "index/diskann/rabitq_store.h"

namespace knowhere {
namespace {
const DiskANNNavigationConfig*
ExternalConfig(const DiskANNConfig& config) {
    const auto* navigation = dynamic_cast<const DiskANNNavigationConfig*>(&config);
    if (!navigation || navigation->navigation_codec.value_or("PQ") == "PQ") {
        return nullptr;
    }
    if (navigation->navigation_codec.value() != "RABITQ") {
        throw std::invalid_argument("unsupported DiskANN navigation codec");
    }
    return navigation;
}
}  // namespace

bool
UsesExternalNavigation(const DiskANNConfig& config) {
    return ExternalConfig(config) != nullptr;
}

std::vector<std::string>
NavigationFiles(const DiskANNConfig& config, const std::string& prefix) {
    return ExternalConfig(config) ? std::vector<std::string>{RaBitQStore::SidecarFilename(prefix)}
                                  : std::vector<std::string>{};
}

void
BuildNavigationStore(const DiskANNConfig& config, const std::string& source, const std::string& prefix) {
    if (const auto* navigation = ExternalConfig(config)) {
        RaBitQStore::BuildFromFloatBin(source, RaBitQStore::SidecarFilename(prefix),
                                       static_cast<uint8_t>(navigation->rbq_bits.value_or(1)));
    }
}

std::unique_ptr<NavigationStore>
LoadNavigationStore(const DiskANNConfig& config, const std::string& prefix) {
    if (ExternalConfig(config)) {
        return std::make_unique<RaBitQStore>(RaBitQStore::SidecarFilename(prefix));
    }
    return nullptr;
}
}  // namespace knowhere
