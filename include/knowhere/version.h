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

#include <regex>
#include <string>

#include "log.h"

namespace knowhere {
namespace {
static const std::regex version_regex(R"(^knowhere-v(\d+)$)");
static constexpr const char* default_version = "knowhere-v0";
static constexpr const char* minimal_vesion = "knowhere-v0";
static constexpr const char* current_version = "knowhere-v0";
}  // namespace

class Version {
 public:
    explicit Version(const std::string& version_code_) : version_code(version_code_) {
        try {
            std::smatch matches;
            if (std::regex_match(version_code_, matches, version_regex)) {
                version_ = std::stoi(matches[1]);
            } else {
                LOG_KNOWHERE_ERROR_ << "unexpected version code : " << version_code_;
            }
        } catch (std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "version code " << version_code_ << " parse failed : " << e.what();
        }
    }

    bool
    Valid() {
        return version_ != unexpected_version_num;
    };

    const std::string&
    VersionCode() const {
        return version_code;
    }

    static bool
    VersionCheck(const std::string& version) {
        try {
            return std::regex_match(version.c_str(), version_regex);
        } catch (std::regex_error& e) {
            LOG_KNOWHERE_ERROR_ << "unexpected index version : " << version;
        }
        return false;
    }

    // used when version is not set
    static inline Version
    GetDefaultVersion() {
        return Version(default_version);
    }

    // the current version (newest version support)
    static inline Version
    GetCurrentVersion() {
        return Version(current_version);
    }

    // the minimal version (oldest version support)
    static inline Version
    GetMinimalSupport() {
        return Version(minimal_vesion);
    }

    static inline bool
    VersionSupport(const Version& version) {
        return VersionCheck(version.version_code) && GetMinimalSupport() <= version && version <= GetCurrentVersion();
    }

    friend bool
    operator<=(const Version& lhs, const Version& rhs) {
        return lhs.version_ <= rhs.version_;
    }

 private:
    static constexpr int32_t unexpected_version_num = -1;
    const std::string version_code;
    int32_t version_ = unexpected_version_num;
};

}  // namespace knowhere
