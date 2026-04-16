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

#include "index/ivf/ivf_config.h"

namespace knowhere {

class GpuIvfFlatConfig : public IvfFlatConfig {
 public:
    CFG_INT gpu_id;
    KNOWHERE_DECLARE_CONFIG(GpuIvfFlatConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_id).description("gpu device id").set_default(0).for_train();
    }
};

class GpuIvfPqConfig : public IvfPqConfig {
 public:
    CFG_INT gpu_id;
    KNOWHERE_DECLARE_CONFIG(GpuIvfPqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_id).description("gpu device id").set_default(0).for_train();
    }
};

class GpuIvfSqConfig : public IvfSqConfig {
 public:
    CFG_INT gpu_id;
    KNOWHERE_DECLARE_CONFIG(GpuIvfSqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_id).description("gpu device id").set_default(0).for_train();
    }
};

}  // namespace knowhere
