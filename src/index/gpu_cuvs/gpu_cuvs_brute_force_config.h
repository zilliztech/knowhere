/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GPU_CUVS_BRUTE_FORCE_CONFIG_H
#define GPU_CUVS_BRUTE_FORCE_CONFIG_H

#include "common/cuvs/integration/cuvs_knowhere_config.hpp"
#include "common/cuvs/proto/cuvs_index_kind.hpp"
#include "knowhere/config.h"

namespace knowhere {

struct GpuCuvsBruteForceConfig : public BaseConfig {
    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        if (param_type == PARAM_TYPE::TRAIN) {
            constexpr std::array<std::string_view, 3> legal_metric_list{"L2", "IP", "COSINE"};
            std::string metric = metric_type.value();
            if (std::find(legal_metric_list.begin(), legal_metric_list.end(), metric) == legal_metric_list.end()) {
                std::string msg = "metric type " + metric + " not found or not supported, supported: [L2 IP]";
                return HandleError(err_msg, msg, Status::invalid_metric_type);
            }
        }
        return Status::success;
    }
};

[[nodiscard]] inline auto
to_cuvs_knowhere_config(GpuCuvsBruteForceConfig const& cfg) {
    auto result = cuvs_knowhere::cuvs_knowhere_config{cuvs_proto::cuvs_index_kind::brute_force};

    result.metric_type = cfg.metric_type.value();
    result.k = cfg.k.value();

    return result;
}

}  // namespace knowhere

#endif /*GPU_CUVS_BRUTE_FORCE_CONFIG_H*/
