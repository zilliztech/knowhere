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
#pragma once
#include <optional>
#include <string>

#include "common/raft/proto/raft_index_kind.hpp"

namespace raft_knowhere {
// This struct includes all parameters that may be passed to underlying RAFT
// indexes. It is designed to not expose ANY RAFT types in order to cleanly
// separate RAFT from knowhere headers.
struct raft_knowhere_config {
    raft_proto::raft_index_kind index_type;
    int k = 10;

    // Common Parameters
    std::string metric_type = std::string{"L2"};
    float metric_arg = 2.0f;
    bool add_data_on_build = true;
    bool cache_dataset_on_device = false;
    float refine_ratio = 1.0f;

    // Shared IVF Parameters
    std::optional<int> nlist = std::nullopt;
    std::optional<int> nprobe = std::nullopt;
    std::optional<int> kmeans_n_iters = std::nullopt;
    std::optional<float> kmeans_trainset_fraction = std::nullopt;

    // IVF Flat only Parameters
    std::optional<bool> adaptive_centers = std::nullopt;

    // IVFPQ only Parameters
    std::optional<int> m = std::nullopt;
    std::optional<int> nbits = std::nullopt;
    std::optional<std::string> codebook_kind = std::nullopt;
    std::optional<bool> force_random_rotation = std::nullopt;
    std::optional<bool> conservative_memory_allocation = std::nullopt;
    std::optional<std::string> lookup_table_dtype = std::nullopt;
    std::optional<std::string> internal_distance_dtype = std::nullopt;
    std::optional<float> preferred_shmem_carveout = std::nullopt;

    // CAGRA Parameters
    std::optional<int> intermediate_graph_degree = std::nullopt;
    std::optional<int> graph_degree = std::nullopt;
    std::optional<int> itopk_size = std::nullopt;
    std::optional<int> max_queries = std::nullopt;
    std::optional<std::string> build_algo = std::nullopt;
    std::optional<std::string> search_algo = std::nullopt;
    std::optional<int> team_size = std::nullopt;
    std::optional<int> search_width = std::nullopt;
    std::optional<int> min_iterations = std::nullopt;
    std::optional<int> max_iterations = std::nullopt;
    std::optional<int> thread_block_size = std::nullopt;
    std::optional<std::string> hashmap_mode = std::nullopt;
    std::optional<int> hashmap_min_bitlen = std::nullopt;
    std::optional<float> hashmap_max_fill_rate = std::nullopt;
    std::optional<int> nn_descent_niter = std::nullopt;
};

// The following function provides a single source of truth for default values
// of RAFT index configurations.
[[nodiscard]] inline auto
validate_raft_knowhere_config(raft_knowhere_config config) {
    if (config.index_type == raft_proto::raft_index_kind::brute_force) {
        config.add_data_on_build = false;
        config.cache_dataset_on_device = true;
    }
    if (config.index_type == raft_proto::raft_index_kind::ivf_flat ||
        config.index_type == raft_proto::raft_index_kind::ivf_pq) {
        config.add_data_on_build = true;
        config.nlist = config.nlist.value_or(128);
        config.nprobe = config.nprobe.value_or(8);
        config.kmeans_n_iters = config.kmeans_n_iters.value_or(20);
        config.kmeans_trainset_fraction = config.kmeans_trainset_fraction.value_or(0.5f);
    }
    if (config.index_type == raft_proto::raft_index_kind::ivf_flat) {
        config.adaptive_centers = config.adaptive_centers.value_or(false);
    }
    if (config.index_type == raft_proto::raft_index_kind::ivf_pq) {
        config.m = config.m.value_or(0);
        config.nbits = config.nbits.value_or(8);
        config.codebook_kind = config.codebook_kind.value_or("PER_SUBSPACE");
        config.force_random_rotation = config.force_random_rotation.value_or(false);
        config.conservative_memory_allocation = config.conservative_memory_allocation.value_or(false);
        config.lookup_table_dtype = config.lookup_table_dtype.value_or("CUDA_R_32F");
        config.internal_distance_dtype = config.internal_distance_dtype.value_or("CUDA_R_32F");
        config.preferred_shmem_carveout = config.preferred_shmem_carveout.value_or(1.0f);
    }
    if (config.index_type == raft_proto::raft_index_kind::cagra) {
        config.add_data_on_build = true;
        config.intermediate_graph_degree = config.intermediate_graph_degree.value_or(128);
        config.graph_degree = config.graph_degree.value_or(64);
        config.itopk_size = config.itopk_size.value_or(64);
        config.max_queries = config.max_queries.value_or(0);
        config.build_algo = config.build_algo.value_or("IVF_PQ");
        config.search_algo = config.search_algo.value_or("AUTO");
        config.team_size = config.team_size.value_or(0);
        config.search_width = config.search_width.value_or(1);
        config.min_iterations = config.min_iterations.value_or(0);
        config.max_iterations = config.max_iterations.value_or(0);
        config.thread_block_size = config.thread_block_size.value_or(0);
        config.hashmap_mode = config.hashmap_mode.value_or("AUTO");
        config.hashmap_min_bitlen = config.hashmap_min_bitlen.value_or(0);
        config.hashmap_max_fill_rate = config.hashmap_max_fill_rate.value_or(0.5f);
        config.nn_descent_niter = config.nn_descent_niter.value_or(20);
    }
    return config;
}

}  // namespace raft_knowhere
