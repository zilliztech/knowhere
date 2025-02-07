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
#include "common/cuvs/integration/raft_initialization.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
namespace cuvs_knowhere {

void
initialize_raft(raft_configuration const& config) {
    auto static initialization_flag = std::once_flag{};
    std::call_once(initialization_flag, [&config]() {
        raft::device_resources_manager::set_streams_per_device(config.streams_per_device);
        if (config.stream_pool_size) {
            raft::device_resources_manager::set_stream_pools_per_device(config.stream_pools_per_device,
                                                                        *(config.stream_pool_size));
        } else {
            raft::device_resources_manager::set_stream_pools_per_device(config.stream_pools_per_device);
        }
        if (config.init_mem_pool_size_mb) {
            raft::device_resources_manager::set_init_mem_pool_size(*(config.init_mem_pool_size_mb) << 20);
        }
        if (config.max_mem_pool_size_mb) {
            if (*config.max_mem_pool_size_mb > 0) {
                raft::device_resources_manager::set_max_mem_pool_size(*(config.max_mem_pool_size_mb) << 20);
            }
        } else {
            raft::device_resources_manager::set_max_mem_pool_size(std::nullopt);
        }
        if (config.max_workspace_size_mb) {
            raft::device_resources_manager::set_workspace_allocation_limit(*(config.max_workspace_size_mb) << 20);
        }
        auto device_count = []() {
            auto result = 0;
            RAFT_CUDA_TRY(cudaGetDeviceCount(&result));
            RAFT_EXPECTS(result != 0, "No CUDA devices found");
            return result;
        }();

        for (auto device_id = 0; device_id < device_count; ++device_id) {
            auto scoped_device = raft::device_setter{device_id};
            auto workspace_size = std::size_t{};
            if (config.max_workspace_size_mb) {
                workspace_size = *(config.max_workspace_size_mb) << 20;
            } else {
                auto free_mem = std::size_t{};
                auto total_mem = std::size_t{};
                RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free_mem, &total_mem));
                // Heuristic: If workspace size is not explicitly specified, use half of free memory or a quarter of
                // total memory, whichever is larger
                workspace_size = std::max(free_mem / std::size_t{2}, total_mem / std::size_t{4});
            }
            if (workspace_size > std::size_t{}) {
                raft::device_resources_manager::set_workspace_memory_resource(
                    raft::resource::workspace_resource_factory::default_pool_resource(workspace_size), device_id);
            }
        }
    });
}

}  // namespace cuvs_knowhere
