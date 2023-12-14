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
#ifndef GPU_RAFT_BRUTE_FORCE_CONFIG_H
#define GPU_RAFT_BRUTE_FORCE_CONFIG_H

#include "common/raft/integration/raft_knowhere_config.hpp"
#include "common/raft/proto/raft_index_kind.hpp"
#include "knowhere/config.h"

namespace knowhere {

struct GpuRaftBruteForceConfig : public BaseConfig {};

[[nodiscard]] inline auto
to_raft_knowhere_config(GpuRaftBruteForceConfig const& cfg) {
    auto result = raft_knowhere::raft_knowhere_config{raft_proto::raft_index_kind::brute_force};

    result.metric_type = cfg.metric_type.value();
    result.k = cfg.k.value();

    return result;
}

}  // namespace knowhere

#endif /*GPU_RAFT_BRUTE_FORCE_CONFIG_H*/
