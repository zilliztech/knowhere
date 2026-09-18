// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "diskann/pq_flash_index.h"

namespace knowhere {
class DiskANNConfig;

// Immutable after load. Each request owns its scorer and transformed-query
// scratch; this store must outlive those scorers. PQ remains the native engine
// path (a null external store), without a virtual call per candidate.
class NavigationStore {
 public:
    virtual ~NavigationStore() = default;
    virtual int64_t
    Count() const = 0;
    virtual int64_t
    Dimension() const = 0;
    virtual size_t
    MemorySize() const = 0;
    virtual std::unique_ptr<diskann::NavigationDistanceComputer>
    CreateDistanceComputer(const DiskANNConfig& config) const = 0;
};

// Factory boundary: graph/IO code never inspects a concrete quantizer or its
// search parameters. New codecs are added here, not to cached_beam_search.
bool
UsesExternalNavigation(const DiskANNConfig& config);
uint64_t
EstimateNavigationMemory(const DiskANNConfig& config, int64_t rows, int64_t dim);
std::vector<std::string>
NavigationFiles(const DiskANNConfig& config, const std::string& prefix);
void
BuildNavigationStore(const DiskANNConfig& config, const std::string& prepared_source, const std::string& prefix);
std::unique_ptr<NavigationStore>
LoadNavigationStore(const DiskANNConfig& config, const std::string& prefix);
}  // namespace knowhere
