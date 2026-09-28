// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <unordered_set>

#include "filemanager/FileManager.h"

namespace knowhere {
// Publication may have partial effects that FileManager cannot roll back.
// The caller must retain completed local outputs before attempting uploads.
class DiskANNBuildRegistration {
 public:
    explicit DiskANNBuildRegistration(milvus::FileManager& manager) : manager_(manager) {
    }
    DiskANNBuildRegistration(const DiskANNBuildRegistration&) = delete;
    DiskANNBuildRegistration&
    operator=(const DiskANNBuildRegistration&) = delete;
    bool
    Reserve(const std::string& path) {
        // Check before creating local outputs: FileManager implementations may
        // consult the local filesystem as well as their registered objects.
        const auto exists = manager_.IsExisted(path);
        if (!exists.has_value() || exists.value()) {
            return false;
        }
        reserved_.insert(path);
        return true;
    }
    bool
    Add(const std::string& path) {
        if (!reserved_.count(path))
            return false;
        return manager_.AddFile(path);
    }

 private:
    milvus::FileManager& manager_;
    std::unordered_set<std::string> reserved_;
};
}  // namespace knowhere
