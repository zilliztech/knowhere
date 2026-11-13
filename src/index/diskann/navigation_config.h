// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "diskann/navigation_runtime.h"
#include "knowhere/config.h"

namespace knowhere {
class NavigationConfig : public Config {
 public:
    virtual Status
    Load(const Json& json, PARAM_TYPE type, float, std::string* error) {
        Json parameters(json);
        auto status = Config::FormatAndCheck(*this, parameters, error);
        return status == Status::success ? Config::Load(*this, parameters, type, error) : status;
    }
    // Called by Build once the raw data size is known, before CreateForBuild.
    virtual void
    ResolveBuildBudget(double /*raw_data_gb*/) {
    }
    virtual std::unique_ptr<diskann::NavigationStore>
    CreateForBuild() const = 0;
    virtual std::optional<uint64_t>
    EstimateLoadMemory(uint64_t file_size) const = 0;
    virtual diskann::NavigationQuerySetup
    QuerySetup() const {
        return {};
    }
    virtual void
    AppendBuildMetadata(Json& build_params) const = 0;
};

std::unique_ptr<NavigationConfig>
CreateNavigationConfig(const std::string& codec);
}  // namespace knowhere
