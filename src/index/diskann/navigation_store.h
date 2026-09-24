// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "diskann/navigation_build.h"
#include "diskann/pq_flash_index.h"
#include "filemanager/FileManager.h"
#include "knowhere/expected.h"

namespace knowhere {
class DiskANNConfig;
class DiskANNNavigationConfig;

expected<std::string>
DetectNavigationCodec(const DiskANNNavigationConfig& config, const std::string& prefix, milvus::FileManager& manager);

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
// Missing codec/model parameters require a file-size-based static estimate;
// build defaults must not be mistaken for persisted model metadata.
std::optional<uint64_t>
EstimateNavigationMemory(const DiskANNConfig& config, int64_t rows, int64_t dim);
struct NavigationFileSet {
    std::vector<std::string> required;
    std::vector<std::string> optional;
};
NavigationFileSet
NavigationFiles(const DiskANNConfig& config, const std::string& prefix);
std::vector<std::string>
AllNavigationFiles(const std::string& prefix);
std::unique_ptr<diskann::NavigationBuilder>
CreateNavigationBuilder(const DiskANNConfig& config, std::unique_ptr<diskann::NavigationBuilder> native_pq);
std::unique_ptr<NavigationStore>
LoadNavigationStore(const DiskANNConfig& config, const std::string& prefix);
}  // namespace knowhere
