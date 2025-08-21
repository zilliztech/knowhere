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

#include "comp/index_param.h"
#include "log.h"

namespace knowhere {
namespace {
static constexpr int32_t default_version = 0;
static constexpr int32_t minimal_version = 0;
static constexpr int32_t current_version = 9;
static constexpr int32_t maximum_version = 9;
}  // namespace

class Version {
 public:
    explicit Version(const IndexVersion version) : version_(version) {
    }

    // used when version is not set
    static inline Version
    GetDefaultVersion() {
        return Version(default_version);
    }

    // the recommended version
    static inline Version
    GetCurrentVersion() {
        return Version(current_version);
    }

    // the maximum version (beta version)
    static inline Version
    GetMaximumVersion() {
        return Version(maximum_version);
    }

    // the minimal version (oldest version support)
    static inline Version
    GetMinimalVersion() {
        return Version(minimal_version);
    }

    static inline bool
    VersionSupport(const Version& version) {
        return GetMinimalVersion() <= version && version <= GetMaximumVersion();
    }

    static inline std::pair<Version, Version>
    GetSupportRange() {
        return std::make_pair(Version(minimal_version), Version(maximum_version));
    }

    // the version number
    IndexVersion
    VersionNumber() const {
        return version_;
    }

    friend bool
    operator<=(const Version& lhs, const Version& rhs) {
        return lhs.version_ <= rhs.version_;
    }

 private:
    IndexVersion version_ = default_version;
};

}  // namespace knowhere
