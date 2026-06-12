/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, 6sense Insights Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gpu_hnsw_types.hpp"
#include "gpu_hnsw_search_kernel.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#define GPU_HNSW_CUDA_CHECK(expr)                                                   \
    do {                                                                            \
        cudaError_t _e = (expr);                                                    \
        if (_e != cudaSuccess) {                                                    \
            throw std::runtime_error(std::string("CUDA error: ") +                 \
                                     cudaGetErrorString(_e) + " at " +             \
                                     __FILE__ + ":" + std::to_string(__LINE__));   \
        }                                                                           \
    } while (0)

namespace knowhere::detail::gpu_hnsw {

namespace kernel = cuvs::neighbors::gpu_hnsw::detail;

inline void search(cudaStream_t stream,
                   const search_params& params,
                   const gpu_hnsw_index& idx,
                   const float* d_queries,
                   int          num_queries,
                   uint64_t*    d_neighbors,
                   float*       d_distances,
                   int          k)
{
    int ef       = params.ef;
    int sw       = params.search_width;
    int max_iter = params.max_iterations > 0 ? params.max_iterations
                                              : 2 * ef / sw + 10;
    int dim      = static_cast<int>(idx.dim);

    uint32_t* d_entry_points = nullptr;
    GPU_HNSW_CUDA_CHECK(cudaMalloc(&d_entry_points, num_queries * sizeof(uint32_t)));

    int num_upper_layers = static_cast<int>(idx.upper_layers.size());

    if (num_upper_layers > 0) {
        std::vector<kernel::upper_layer_ptrs> h_layer_ptrs(num_upper_layers);
        for (int i = 0; i < num_upper_layers; i++) {
            const auto& ul  = idx.upper_layers[i];
            h_layer_ptrs[i] = {ul.d_node_ids, ul.d_neighbors, ul.num_nodes, ul.max_degree};
        }
        kernel::upper_layer_ptrs* d_layer_ptrs = nullptr;
        GPU_HNSW_CUDA_CHECK(cudaMalloc(&d_layer_ptrs,
                                        num_upper_layers * sizeof(kernel::upper_layer_ptrs)));
        GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(d_layer_ptrs, h_layer_ptrs.data(),
                                             num_upper_layers * sizeof(kernel::upper_layer_ptrs),
                                             cudaMemcpyHostToDevice, stream));

        int warps_per_block   = 4;
        int threads_per_block = warps_per_block * 32;
        int num_blocks        = (num_queries + warps_per_block - 1) / warps_per_block;

        kernel::upper_layer_search_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            d_queries, idx.d_dataset, d_layer_ptrs, d_entry_points,
            idx.entry_point, num_queries, dim, num_upper_layers, idx.use_ip);
        GPU_HNSW_CUDA_CHECK(cudaGetLastError());

#ifdef GPU_HNSW_DIAGNOSTICS
        {
            static bool ep_diag_logged = false;
            if (!ep_diag_logged) {
                ep_diag_logged = true;
                GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<uint32_t> h_eps(num_queries);
                GPU_HNSW_CUDA_CHECK(cudaMemcpy(h_eps.data(), d_entry_points,
                                                num_queries * sizeof(uint32_t),
                                                cudaMemcpyDeviceToHost));
                std::set<uint32_t> unique_eps(h_eps.begin(), h_eps.end());
                fprintf(stderr,
                        "[gpu_hnsw_diag] search: %d queries, %zu unique entry points "
                        "(global_ep=%u, first_ep=%u, last_ep=%u)\n",
                        num_queries, unique_eps.size(), idx.entry_point,
                        h_eps.front(), h_eps.back());
            }
        }
#endif

        GPU_HNSW_CUDA_CHECK(cudaFree(d_layer_ptrs));
    } else {
        std::vector<uint32_t> h_eps(num_queries, idx.entry_point);
        GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(d_entry_points, h_eps.data(),
                                             num_queries * sizeof(uint32_t),
                                             cudaMemcpyHostToDevice, stream));
    }

    int    block_size   = params.thread_block_size > 0 ? params.thread_block_size : 128;

    // Clamp ef to avoid shared memory overflow (48 KB limit)
    // smem = ef*12 + sw*max_degree0*8 + sw*4 + 12
    // Solve: ef*12 <= 49152 - (sw*max_degree0*8 + sw*4 + 12)
    {
        int smem_overhead = sw * idx.max_degree0 * 8 + sw * 4 + 12;
        int max_ef = (49152 - smem_overhead) / 12;
        if (ef > max_ef) {
            fprintf(stderr, "[gpu_hnsw] clamping ef %d -> %d (smem limit)\n", ef, max_ef);
            ef = max_ef;
        }
    }

    size_t smem_size    = kernel::calc_layer0_smem_size(ef, sw, idx.max_degree0);
    int    N_int        = static_cast<int>(idx.n_rows);
    size_t bitmap_bytes = kernel::calc_visited_bitmap_size(num_queries, N_int);

#ifdef GPU_HNSW_DIAGNOSTICS
    {
        static bool search_diag_logged = false;
        if (!search_diag_logged) {
            search_diag_logged = true;
            fprintf(stderr, "[gpu_hnsw_diag] search params: ef=%d sw=%d max_iter=%d k=%d "
                    "block_size=%d smem=%zu bitmap=%zu N=%d dim=%d use_ip=%d\n",
                    ef, sw, max_iter, k, block_size, smem_size, bitmap_bytes,
                    N_int, dim, (int)idx.use_ip);
        }
    }
#endif

    uint32_t* d_visited_bitmaps = nullptr;
    GPU_HNSW_CUDA_CHECK(cudaMalloc(&d_visited_bitmaps, bitmap_bytes));
    GPU_HNSW_CUDA_CHECK(cudaMemsetAsync(d_visited_bitmaps, 0, bitmap_bytes, stream));

    kernel::layer0_beam_search_kernel<<<num_queries, block_size, smem_size, stream>>>(
        d_queries, idx.d_dataset, idx.d_layer0_graph, d_entry_points,
        d_visited_bitmaps,
        d_neighbors, d_distances,
        num_queries, N_int, dim, idx.max_degree0,
        k, ef, sw, max_iter, idx.use_ip);
    GPU_HNSW_CUDA_CHECK(cudaGetLastError());

    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));
    GPU_HNSW_CUDA_CHECK(cudaFree(d_entry_points));
    GPU_HNSW_CUDA_CHECK(cudaFree(d_visited_bitmaps));
}

}  // namespace knowhere::detail::gpu_hnsw
