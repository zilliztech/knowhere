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

#ifndef FAISS_CONFIG_H
#define FAISS_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

class FaissConfig : public BaseConfig {
public:
    CFG_STRING factory_string;
    KNOHWERE_DECLARE_CONFIG(FaissConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(factory_string)
            .set_default("Flat")
            .description("FAISS factory string.")
            .for_train();
    }
};

}  // namespace knowhere

#endif /* FLAT_CONFIG_H */
