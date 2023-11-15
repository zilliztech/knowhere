// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <unordered_set>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif

#include "knowhere/file_manager.h"
namespace knowhere {
/**
 * @brief LocalFileManager is used for placeholder purpose. It will not do anything to the file on disk.
 *
 * This class is not thread-safe.
 */
class LocalFileManager : public FileManager {
 public:
    auto
    LoadFile(const std::string& filename) noexcept -> bool override {
        return true;
    }

    auto
    AddFile(const std::string& filename) noexcept -> bool override {
        files.insert(filename);
        return true;
    }

    auto
    IsExisted(const std::string& filename) noexcept -> std::optional<bool> override {
        return std::make_optional<bool>(files.find(filename) != files.end());
    }

    auto
    RemoveFile(const std::string& filename) noexcept -> bool override {
        files.erase(filename);
        return true;
    }

 private:
    std::unordered_set<std::string> files;
};

}  // namespace knowhere
