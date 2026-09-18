// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "index/diskann/navigation_store.h"

#include <limits>
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

uint64_t
EstimateNavigationMemory(const DiskANNConfig& config, int64_t rows, int64_t dim) {
    const auto* navigation = ExternalConfig(config);
    if (!navigation) {
        return 0;
    }
    if (dim <= 0 || dim >= std::numeric_limits<int>::max()) {
        throw std::invalid_argument("invalid DiskANN navigation dimension");
    }
    const auto prepared_dim = dim + (config.metric_type.value_or(metric::L2) == metric::IP ? 1 : 0);
    return RaBitQStore::EstimateMemorySize(rows, prepared_dim, static_cast<uint8_t>(navigation->rbq_bits.value_or(1)));
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
