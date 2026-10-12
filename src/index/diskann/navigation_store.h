// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "diskann/navigation_build.h"
#include "diskann/navigation_runtime.h"
#include "filemanager/FileManager.h"
#include "knowhere/expected.h"

namespace knowhere {
class DiskANNConfig;
class DiskANNNavigationConfig;

expected<std::string>
DetectNavigationCodec(const DiskANNNavigationConfig& config, const std::string& prefix, milvus::FileManager& manager);

using NavigationStore = diskann::NavigationStore;
struct NavigationFeatures {
    bool fp32_only;
    bool iterator;
    bool emb_list;
    bool sample_cache;
};
NavigationFeatures
GetNavigationFeatures(const DiskANNConfig& config);
diskann::NavigationQueryOptions
GetNavigationQueryOptions(const DiskANNConfig& config);
enum class NavigationInputType { FP32, FP16, BF16 };
// Additional admission allowance beyond the legacy file-size/4 heuristic.
// PQ is already covered by that heuristic; its actual model/workspace memory
// is reported separately by NavigationStore::MemorySize().
// Missing codec/model parameters require a file-size-based static estimate;
// build defaults must not be mistaken for persisted model metadata.
std::optional<uint64_t>
EstimateAdditionalNavigationMemory(const DiskANNConfig& config, int64_t rows, int64_t dim);
struct NavigationFileSet {
    std::vector<std::string> required;
    std::vector<std::string> optional;
};
NavigationFileSet
NavigationFiles(const DiskANNConfig& config, const std::string& prefix);
std::vector<std::string>
AllNavigationFiles(const std::string& prefix);
std::unique_ptr<diskann::NavigationBuilder>
CreateNavigationBuilder(const DiskANNConfig& config, NavigationInputType input_type);
std::unique_ptr<NavigationStore>
LoadNavigationStore(const DiskANNConfig& config, const std::string& prefix, size_t workspaces = 0);
}  // namespace knowhere
