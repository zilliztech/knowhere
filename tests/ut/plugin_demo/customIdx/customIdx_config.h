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

#ifndef FLAT_CONFIG_H
#define FLAT_CONFIG_H

#include "knowhere/config.h"

namespace demo {
class CustomIdxConfig : public knowhere::BaseConfig {
 public:
    // See more data types in knowhere/config.h
    CFG_BOOL early_terminate;
    KNOHWERE_DECLARE_CONFIG(CustomIdxConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(early_terminate)
            // Default value settings. Value will be used when user does not specify it in user request
            .set_default(false)
            .description("Return as soon as k results are found, rather than finding the optimal k results")
            // Configuration scope, only meaningful for search stage. See more scope in config.h
            .for_search();
    }
};
}  // namespace demo
// namespace knowhere
#endif /* FLAT_CONFIG_H */
