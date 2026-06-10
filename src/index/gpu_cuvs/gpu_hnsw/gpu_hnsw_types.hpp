/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, 6sense Insights Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace knowhere::detail::gpu_hnsw {

struct search_params {
    int ef               = 200;
    int search_width     = 4;
    int max_iterations   = 0;  // 0 = auto
    int thread_block_size = 0; // 0 = auto (128)
};

struct device_upper_layer {
    uint32_t* d_node_ids  = nullptr;
    uint32_t* d_neighbors = nullptr;
    uint32_t  num_nodes   = 0;
    uint32_t  max_degree  = 0;
};

struct gpu_hnsw_index {
    float*    d_dataset      = nullptr;  // [n_rows x dim], row-major float32
    uint32_t* d_layer0_graph = nullptr;  // [n_rows x max_degree0]
    std::vector<device_upper_layer> upper_layers;

    int64_t  n_rows      = 0;
    int64_t  dim         = 0;
    uint32_t entry_point = 0;
    int      num_layers  = 0;
    int      M           = 0;
    int      max_degree0 = 0;
    bool     use_ip      = false;

    ~gpu_hnsw_index() {
        if (d_dataset)      cudaFree(d_dataset);
        if (d_layer0_graph) cudaFree(d_layer0_graph);
        for (auto& ul : upper_layers) {
            if (ul.d_node_ids)  cudaFree(ul.d_node_ids);
            if (ul.d_neighbors) cudaFree(ul.d_neighbors);
        }
    }
};

}  // namespace knowhere::detail::gpu_hnsw
