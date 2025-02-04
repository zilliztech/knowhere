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
#include <cstddef>
#include <optional>
namespace cuvs_knowhere {
struct raft_configuration {
    std::size_t streams_per_device = std::size_t{16};
    std::size_t stream_pools_per_device = std::size_t{};
    std::optional<std::size_t> stream_pool_size = std::nullopt;
    std::optional<std::size_t> init_mem_pool_size_mb = std::nullopt;
    std::optional<std::size_t> max_mem_pool_size_mb = std::nullopt;
    std::optional<std::size_t> max_workspace_size_mb = std::nullopt;
};

void
initialize_raft(raft_configuration const& config);
}  // namespace cuvs_knowhere
