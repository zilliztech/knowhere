/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, 6sense Insights Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gpu_hnsw_interface.hpp"
#include "gpu_hnsw_faiss_build.hpp"
#include "gpu_hnsw_impl.cuh"
#include "gpu_hnsw_types.hpp"

#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace knowhere::detail::gpu_hnsw {

void* build_gpu_index(const ::faiss::cppcontrib::knowhere::IndexHNSW* faiss_idx, bool use_ip, bool is_cosine) {
    if (!faiss_idx) return nullptr;
    try {
        // Try quantized storage (SQ8, FP16, BF16, etc.) first.
        if (dynamic_cast<const faiss::IndexScalarQuantizer*>(faiss_idx->storage)) {
            auto idx = from_faiss_hnsw_sq(*faiss_idx, use_ip, is_cosine);
            return static_cast<void*>(idx.release());
        }
        // Fall back to plain float32 storage (IndexHNSWFlat / IndexFlat).
        auto idx = from_faiss_hnsw_flat(*faiss_idx, use_ip, is_cosine);
        return static_cast<void*>(idx.release());
    } catch (const std::exception& e) {
        fprintf(stderr, "[gpu_hnsw] build_gpu_index failed: %s\n", e.what());
        return nullptr;
    } catch (...) {
        fprintf(stderr, "[gpu_hnsw] build_gpu_index failed: unknown exception\n");
        return nullptr;
    }
}

void destroy_gpu_index(void* handle) {
    delete static_cast<gpu_hnsw_index*>(handle);
}

int search_gpu(void* handle,
               const float* h_queries,
               int nq,
               int k,
               int ef,
               int64_t* out_ids,
               float*   out_dists)
{
    if (!handle) return -1;
    auto* idx = static_cast<gpu_hnsw_index*>(handle);

    int dim = static_cast<int>(idx->dim);

    float*    d_queries   = nullptr;
    uint64_t* d_neighbors = nullptr;
    float*    d_distances = nullptr;

    try {
        GPU_HNSW_CUDA_CHECK(cudaMalloc(&d_queries,   nq * dim * sizeof(float)));
        GPU_HNSW_CUDA_CHECK(cudaMalloc(&d_neighbors, nq * k   * sizeof(uint64_t)));
        GPU_HNSW_CUDA_CHECK(cudaMalloc(&d_distances, nq * k   * sizeof(float)));

        GPU_HNSW_CUDA_CHECK(cudaMemcpy(d_queries, h_queries,
                                        nq * dim * sizeof(float),
                                        cudaMemcpyHostToDevice));

        search_params sp;
        sp.ef = ef;

        cudaStream_t stream = nullptr;
        search(stream, sp, *idx, d_queries, nq, d_neighbors, d_distances, k);

        // Copy neighbors as uint64 then convert to int64
        auto tmp = std::make_unique<uint64_t[]>(nq * k);
        GPU_HNSW_CUDA_CHECK(cudaMemcpy(tmp.get(), d_neighbors,
                                        nq * k * sizeof(uint64_t),
                                        cudaMemcpyDeviceToHost));
        GPU_HNSW_CUDA_CHECK(cudaMemcpy(out_dists, d_distances,
                                        nq * k * sizeof(float),
                                        cudaMemcpyDeviceToHost));

        for (int i = 0; i < nq * k; i++) {
            out_ids[i] = (tmp[i] == UINT64_MAX) ? -1 : static_cast<int64_t>(tmp[i]);
        }

        cudaFree(d_queries);
        cudaFree(d_neighbors);
        cudaFree(d_distances);
        return 0;

    } catch (const std::exception& e) {
        fprintf(stderr, "[gpu_hnsw] search_gpu failed: %s\n", e.what());
        if (d_queries)   cudaFree(d_queries);
        if (d_neighbors) cudaFree(d_neighbors);
        if (d_distances) cudaFree(d_distances);
        return -1;
    } catch (...) {
        fprintf(stderr, "[gpu_hnsw] search_gpu failed: unknown exception\n");
        if (d_queries)   cudaFree(d_queries);
        if (d_neighbors) cudaFree(d_neighbors);
        if (d_distances) cudaFree(d_distances);
        return -1;
    }
}

}  // namespace knowhere::detail::gpu_hnsw
