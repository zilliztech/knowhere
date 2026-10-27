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

#ifndef DISKANN_CONFIG_H
#define DISKANN_CONFIG_H

#include "diskann_ssd_config.h"
#include "navigation_config.h"

namespace knowhere {
class DiskANNConfig : public DiskANNSSDConfig {
 public:
    CFG_STRING quant_type;
    std::unique_ptr<NavigationConfig> navigation;

    KNOWHERE_DECLARE_CONFIG(DiskANNConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(quant_type)
            .description("navigation codec: build defaults to PQ; load reads the stored type, optionally constrained")
            .allow_empty_without_default()
            .for_train()
            .for_deserialize()
            .for_static();
    }

    Status
    LoadSubConfig(const Json& json, PARAM_TYPE type, std::string* error) override {
        if (quant_type && (quant_type->empty() || ((type == PARAM_TYPE::TRAIN || type == PARAM_TYPE::STATIC) &&
                                                   !diskann::IsSupportedNavigationCodec(*quant_type))))
            return HandleError(error, "Unsupported DiskANN navigation codec: " + *quant_type, Status::invalid_args);
        if (type == PARAM_TYPE::TRAIN || type == PARAM_TYPE::STATIC)
            navigation = CreateNavigationConfig(quant_type.value_or("PQ"));
        return navigation ? navigation->Load(json, type, vec_field_size_gb.value_or(0), error) : Status::success;
    }
};
}  // namespace knowhere
#endif /* DISKANN_CONFIG_H */
