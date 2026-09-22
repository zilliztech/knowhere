// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "filemanager/FileManager.h"
#include "knowhere/log.h"

namespace knowhere {
// Publication is separate from ownership of local build files. Failed uploads
// can have partial effects, so roll back attempted registrations as well.
class DiskANNBuildRegistration {
 public:
    explicit DiskANNBuildRegistration(milvus::FileManager& manager) : manager_(manager) {
    }
    DiskANNBuildRegistration(const DiskANNBuildRegistration&) = delete;
    DiskANNBuildRegistration&
    operator=(const DiskANNBuildRegistration&) = delete;
    ~DiskANNBuildRegistration() {
        if (!committed_) {
            for (auto it = attempted_.rbegin(); it != attempted_.rend(); ++it) {
                try {
                    if (!manager_.RemoveFile(*it)) {
                        LOG_KNOWHERE_WARNING_ << "Failed to roll back DiskANN registration: " << *it;
                    }
                } catch (const std::exception& e) {
                    LOG_KNOWHERE_WARNING_ << "Failed to roll back DiskANN registration: " << e.what();
                }
            }
        }
    }
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
        attempted_.push_back(path);
        return manager_.AddFile(path);
    }
    void
    Commit() noexcept {
        committed_ = true;
    }

 private:
    milvus::FileManager& manager_;
    std::unordered_set<std::string> reserved_;
    std::vector<std::string> attempted_;
    bool committed_ = false;
};
}  // namespace knowhere
