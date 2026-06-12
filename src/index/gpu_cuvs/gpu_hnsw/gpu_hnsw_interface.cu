/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, 6sense Insights Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "gpu_hnsw_faiss_build.hpp"
#include "gpu_hnsw_impl.cuh"
#include "gpu_hnsw_interface.hpp"
#include "gpu_hnsw_types.hpp"

namespace knowhere::detail::gpu_hnsw {

void*
build_gpu_index(const ::faiss::cppcontrib::knowhere::IndexHNSW* faiss_idx, bool use_ip, bool is_cosine) {
    if (!faiss_idx)
        return nullptr;
    try {
#ifdef GPU_HNSW_DIAGNOSTICS
        fprintf(stderr, "[gpu_hnsw_diag] build: use_ip=%d is_cosine=%d metric_type=%d ntotal=%ld d=%d\n", (int)use_ip,
                (int)is_cosine, (int)faiss_idx->metric_type, (long)faiss_idx->ntotal, (int)faiss_idx->d);
#endif
        // Try quantized storage (SQ8, FP16, BF16, etc.) first.
        if (dynamic_cast<const faiss::IndexScalarQuantizer*>(faiss_idx->storage)) {
#ifdef GPU_HNSW_DIAGNOSTICS
            const auto* sq = dynamic_cast<const faiss::IndexScalarQuantizer*>(faiss_idx->storage);
            fprintf(stderr, "[gpu_hnsw_diag] build: storage=SQ qtype=%d storage_metric=%d\n", (int)sq->sq.qtype,
                    (int)sq->metric_type);
#endif
            auto idx = from_faiss_hnsw_sq(*faiss_idx, use_ip, is_cosine);
            return static_cast<void*>(idx.release());
        }
        // Fall back to plain float32 storage (IndexHNSWFlat / IndexFlat).
#ifdef GPU_HNSW_DIAGNOSTICS
        fprintf(stderr, "[gpu_hnsw_diag] build: storage=Flat\n");
#endif
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

void
destroy_gpu_index(void* handle) {
    delete static_cast<gpu_hnsw_index*>(handle);
}

int
search_gpu(void* handle, const float* h_queries, int nq, int k, int ef, int64_t* out_ids, float* out_dists) {
    if (!handle)
        return -1;
    auto* idx = static_cast<gpu_hnsw_index*>(handle);

    int dim = static_cast<int>(idx->dim);
    cudaStream_t stream = idx->search_stream;

    try {
        // Acquire scratch mutex — serializes concurrent searches on same index.
        // GPU is the bottleneck anyway, so this adds negligible overhead.
        std::lock_guard<std::mutex> lock(idx->scratch_mutex);
        auto& sc = idx->scratch;

        // Ensure scratch buffers have sufficient capacity (no-op in steady state).
        sc.ensure(nq, k, dim, static_cast<int>(idx->n_rows));

#ifdef GPU_HNSW_DIAGNOSTICS
        cudaEvent_t ev_start, ev_h2d, ev_kernel, ev_sync;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_h2d);
        cudaEventCreate(&ev_kernel);
        cudaEventCreate(&ev_sync);
        cudaEventRecord(ev_start, stream);
#endif

        // Async H2D: query vectors
        GPU_HNSW_CUDA_CHECK(cudaMemcpyAsync(sc.d_queries, h_queries, static_cast<size_t>(nq) * dim * sizeof(float),
                                            cudaMemcpyHostToDevice, stream));

#ifdef GPU_HNSW_DIAGNOSTICS
        cudaEventRecord(ev_h2d, stream);
#endif

        search_params sp;
        sp.ef = ef;

        search(stream, sp, *idx, nq, k);

#ifdef GPU_HNSW_DIAGNOSTICS
        cudaEventRecord(ev_kernel, stream);
#endif

        // Sync stream: wait for kernel to finish before D2H copy.
        GPU_HNSW_CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef GPU_HNSW_DIAGNOSTICS
        cudaEventRecord(ev_sync, stream);
        cudaEventSynchronize(ev_sync);
        {
            static int diag_count = 0;
            if (diag_count < 5) {
                float ms_h2d = 0, ms_kernel = 0, ms_total = 0;
                cudaEventElapsedTime(&ms_h2d, ev_start, ev_h2d);
                cudaEventElapsedTime(&ms_kernel, ev_h2d, ev_kernel);
                cudaEventElapsedTime(&ms_total, ev_start, ev_sync);
                fprintf(stderr,
                        "[gpu_hnsw_diag] search_gpu[%d]: nq=%d k=%d ef=%d "
                        "H2D=%.3fms kernel=%.3fms total=%.3fms\n",
                        diag_count, nq, k, ef, ms_h2d, ms_kernel, ms_total);
                diag_count++;
            }
        }
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_h2d);
        cudaEventDestroy(ev_kernel);
        cudaEventDestroy(ev_sync);
#endif

        // D2H: copy neighbors (uint64 -> int64 conversion on host)
        auto tmp = std::make_unique<uint64_t[]>(nq * k);
        GPU_HNSW_CUDA_CHECK(cudaMemcpy(tmp.get(), sc.d_neighbors, static_cast<size_t>(nq) * k * sizeof(uint64_t),
                                       cudaMemcpyDeviceToHost));
        GPU_HNSW_CUDA_CHECK(
            cudaMemcpy(out_dists, sc.d_distances, static_cast<size_t>(nq) * k * sizeof(float), cudaMemcpyDeviceToHost));

        for (int i = 0; i < nq * k; i++) {
            out_ids[i] = (tmp[i] == UINT64_MAX) ? -1 : static_cast<int64_t>(tmp[i]);
        }

        return 0;

    } catch (const std::exception& e) {
        fprintf(stderr, "[gpu_hnsw] search_gpu failed: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "[gpu_hnsw] search_gpu failed: unknown exception\n");
        return -1;
    }
}

}  // namespace knowhere::detail::gpu_hnsw
