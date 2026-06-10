/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, 6sense Insights Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Plain C++ interface (no CUDA headers) for GPU HNSW search.
 * Implemented in gpu_hnsw_interface.cu.
 */

#pragma once

#include <cstdint>

// Forward declaration — avoids pulling in faiss/CUDA headers here.
namespace faiss { struct IndexHNSW; }

namespace knowhere::detail::gpu_hnsw {

/// Build a GPU HNSW index from a faiss IndexHNSW whose storage is
/// IndexScalarQuantizer (QT_8bit).  Dequantizes int8 → float32 before upload.
/// @param use_ip     true for IP or COSINE metrics
/// @param is_cosine  true for COSINE metric (stored vectors will be normalized)
/// Returns an opaque handle (heap-allocated gpu_hnsw_index*) or nullptr on error.
void* build_gpu_index(const faiss::IndexHNSW* faiss_idx, bool use_ip, bool is_cosine);

/// Free a GPU index created by build_gpu_index.
void destroy_gpu_index(void* handle);

/// Run GPU HNSW search.
/// @param handle    opaque gpu_hnsw_index* from build_gpu_index
/// @param queries   row-major float32 host array [nq x dim]
/// @param nq        number of queries
/// @param k         number of neighbors
/// @param ef        search ef parameter
/// @param out_ids   output int64_t host array [nq x k]; -1 for not-found
/// @param out_dists output float host array [nq x k]
/// Returns 0 on success, non-zero on error.
int search_gpu(void* handle,
               const float* queries,
               int nq,
               int k,
               int ef,
               int64_t* out_ids,
               float*   out_dists);

}  // namespace knowhere::detail::gpu_hnsw
