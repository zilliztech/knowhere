/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, 6sense Insights Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_hnsw_search_kernel.cuh"
#include "gpu_hnsw_types.hpp"

#define GPU_HNSW_CUDA_CHECK(expr)                                                                                     \
    do {                                                                                                              \
        cudaError_t _e = (expr);                                                                                      \
        if (_e != cudaSuccess) {                                                                                      \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e) + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                                       \
        }                                                                                                             \
    } while (0)

namespace knowhere::detail::gpu_hnsw {

namespace kernel = cuvs::neighbors::gpu_hnsw::detail;

// Search using pre-allocated scratch buffers from the index.
// Caller must hold idx.scratch_mutex.
inline void
search(cudaStream_t stream, const search_params& params, const gpu_hnsw_index& idx, int num_queries, int k) {
    auto& sc = idx.scratch;

    int ef = params.ef;
    int sw = params.search_width;
    int overflow_ef = params.overflow_factor * ef;
    int max_iter = params.max_iterations > 0 ? params.max_iterations : (ef + overflow_ef) / sw + 20;
    int dim = static_cast<int>(idx.dim);

    int num_upper_layers = idx.num_upper_layers_built;

    // Helper lambda: launch upper layer + beam search kernels for a given DataT.
    auto launch_kernels = [&]<typename DataT>(const DataT* d_data, const float* d_inv_norms) {
        if (num_upper_layers > 0) {
            auto* d_layer_ptrs = static_cast<kernel::upper_layer_ptrs*>(idx.d_upper_layer_ptrs);

            int warps_per_block = 4;
            int threads_per_block = warps_per_block * 32;
            int num_blocks = (num_queries + warps_per_block - 1) / warps_per_block;

            kernel::upper_layer_search_kernel<DataT><<<num_blocks, threads_per_block, 0, stream>>>(
                sc.d_queries, d_data, d_inv_norms, d_layer_ptrs, sc.d_entry_points, idx.entry_point, num_queries, dim,
                num_upper_layers, idx.use_ip);
            GPU_HNSW_CUDA_CHECK(cudaGetLastError());

#ifdef GPU_HNSW_DIAGNOSTICS
            {
                static bool ep_diag_logged = false;
                if (!ep_diag_logged) {
                    ep_diag_logged = true;
                    GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));
                    std::vector<uint32_t> h_eps(num_queries);
                    GPU_HNSW_CUDA_CHECK(cudaMemcpy(h_eps.data(), sc.d_entry_points, num_queries * sizeof(uint32_t),
                                                   cudaMemcpyDeviceToHost));
                    std::set<uint32_t> unique_eps(h_eps.begin(), h_eps.end());
                    fprintf(stderr,
                            "[gpu_hnsw_diag] search: %d queries, %zu unique entry points "
                            "(global_ep=%u, first_ep=%u, last_ep=%u)\n",
                            num_queries, unique_eps.size(), idx.entry_point, h_eps.front(), h_eps.back());
                }
            }
#endif
        } else {
            std::vector<uint32_t> h_eps(num_queries, idx.entry_point);
            GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(sc.d_entry_points, h_eps.data(), num_queries * sizeof(uint32_t),
                                                cudaMemcpyHostToDevice, stream));
        }

        int block_size = params.thread_block_size > 0 ? params.thread_block_size : 128;

        {
            int smem_overhead = sw * idx.max_degree0 * 8 + sw * 4 + 12;
            int max_ef = (49152 - smem_overhead) / 12;
            if (ef > max_ef) {
                fprintf(stderr, "[gpu_hnsw] clamping ef %d -> %d (smem limit)\n", ef, max_ef);
                ef = max_ef;
            }
        }

        size_t smem_size = kernel::calc_layer0_smem_size(ef, sw, idx.max_degree0);
        int N_int = static_cast<int>(idx.n_rows);
        size_t bitmap_bytes = kernel::calc_visited_bitmap_size(num_queries, N_int);

#ifdef GPU_HNSW_DIAGNOSTICS
        {
            static bool search_diag_logged = false;
            if (!search_diag_logged) {
                search_diag_logged = true;
                fprintf(stderr,
                        "[gpu_hnsw_diag] search params: ef=%d sw=%d max_iter=%d k=%d "
                        "block_size=%d smem=%zu bitmap=%zu overflow_ef=%d N=%d dim=%d use_ip=%d int8=%d\n",
                        ef, sw, max_iter, k, block_size, smem_size, bitmap_bytes, overflow_ef, N_int, dim,
                        (int)idx.use_ip, (int)idx.dataset_int8);
            }
        }
#endif

        GPU_HNSW_CUDA_CHECK(cudaMemsetAsync(sc.d_visited_bitmaps, 0, bitmap_bytes, stream));
        // Zero overflow count (entries are never read beyond count, so only count needs zeroing)
        GPU_HNSW_CUDA_CHECK(
            cudaMemsetAsync(sc.d_overflow_count, 0, static_cast<size_t>(num_queries) * sizeof(int), stream));

        kernel::layer0_beam_search_kernel<DataT><<<num_queries, block_size, smem_size, stream>>>(
            sc.d_queries, d_data, d_inv_norms, idx.d_layer0_graph, sc.d_entry_points, sc.d_visited_bitmaps,
            sc.d_neighbors, sc.d_distances, num_queries, N_int, dim, idx.max_degree0, k, ef, sw, max_iter, idx.use_ip,
            overflow_ef, sc.d_overflow_ids, sc.d_overflow_dists, sc.d_overflow_expanded, sc.d_overflow_count);
        GPU_HNSW_CUDA_CHECK(cudaGetLastError());
    };

    if (idx.dataset_int8) {
        launch_kernels(static_cast<const int8_t*>(idx.d_dataset), idx.d_inv_norms);
    } else {
        launch_kernels(static_cast<const float*>(idx.d_dataset), idx.d_inv_norms);
    }
}

}  // namespace knowhere::detail::gpu_hnsw
