/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, 6sense Insights Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

// Converts a faiss::IndexHNSW (CSR graph format) + dequantized float32 vectors
// into a knowhere::detail::gpu_hnsw::gpu_hnsw_index on device memory.

#pragma once

#include "gpu_hnsw_types.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>
#include <faiss/cppcontrib/knowhere/impl/HNSW.h>

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#define GPU_HNSW_BUILD_CUDA_CHECK(expr)                                             \
    do {                                                                            \
        cudaError_t _e = (expr);                                                    \
        if (_e != cudaSuccess) {                                                    \
            throw std::runtime_error(std::string("CUDA error: ") +                 \
                                     cudaGetErrorString(_e) + " at " +             \
                                     __FILE__ + ":" + std::to_string(__LINE__));   \
        }                                                                           \
    } while (0)

namespace knowhere::detail::gpu_hnsw {

// Extract the multi-layer graph from a faiss::HNSW (CSR format) into the
// GPU-friendly dense per-layer format used by gpu_hnsw_index.
//
// faiss CSR layout:
//   offsets[i] .. offsets[i+1] spans all layers for node i.
//   For node i at layer L, the neighbor range is:
//     [offsets[i] + cum_nb_neighbors(L), offsets[i] + cum_nb_neighbors(L+1))
//   nb_neighbors(0) = 2*M  (maxM0), nb_neighbors(L>0) = M
inline void extract_faiss_hnsw_layers(const faiss::cppcontrib::knowhere::HNSW& hnsw,
                                       int64_t n_rows,
                                       std::vector<device_upper_layer>& h_upper_layers,
                                       std::vector<uint32_t>& h_layer0_flat,
                                       uint32_t& entry_point,
                                       int& M,
                                       int& max_degree0,
                                       int& num_layers)
{
    const int maxM0  = hnsw.nb_neighbors(0);  // 2*M
    const int maxM   = hnsw.nb_neighbors(1);  // M  (for layers > 0)
    const int max_lv = hnsw.max_level;        // number of upper layers

    entry_point = static_cast<uint32_t>(hnsw.entry_point);
    M           = maxM;
    max_degree0 = maxM0;
    num_layers  = max_lv + 1;

#ifdef GPU_HNSW_DIAGNOSTICS
    fprintf(stderr, "[gpu_hnsw_diag] n_rows=%ld entry_point=%u max_level=%d maxM0=%d maxM=%d\n",
            (long)n_rows, entry_point, max_lv, maxM0, maxM);
#endif

    // --- Layer 0: dense [n_rows x maxM0] ---
    h_layer0_flat.assign(n_rows * maxM0, UINT32_MAX);
    for (int64_t i = 0; i < n_rows; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        uint32_t count = static_cast<uint32_t>(end - begin);
        for (uint32_t j = 0; j < count; j++) {
            auto nb = hnsw.neighbors[begin + j];
            if (nb >= 0)
                h_layer0_flat[i * maxM0 + j] = static_cast<uint32_t>(nb);
        }
    }

#ifdef GPU_HNSW_DIAGNOSTICS
    {
        uint32_t check_ids[] = {entry_point, 0, 1, 2};
        int num_check = (n_rows >= 3) ? 4 : static_cast<int>(n_rows) + 1;
        for (int ci = 0; ci < num_check; ci++) {
            uint32_t nid = check_ids[ci];
            if (nid >= static_cast<uint32_t>(n_rows)) continue;
            int valid = 0;
            for (int j = 0; j < maxM0; j++) {
                if (h_layer0_flat[nid * maxM0 + j] != UINT32_MAX) valid++;
            }
            fprintf(stderr,
                    "[gpu_hnsw_diag] layer0 node %u: %d/%d valid neighbors",
                    nid, valid, maxM0);
            fprintf(stderr, " [");
            for (int j = 0; j < std::min(5, maxM0); j++) {
                uint32_t nb = h_layer0_flat[nid * maxM0 + j];
                if (nb != UINT32_MAX) fprintf(stderr, "%u ", nb);
                else fprintf(stderr, "- ");
            }
            fprintf(stderr, "...]\n");
        }
    }
#endif

    // --- Upper layers (1 .. max_level): sparse [num_nodes_at_L x maxM] ---
    h_upper_layers.resize(max_lv);
    for (int layer = 1; layer <= max_lv; layer++) {
        auto& ul = h_upper_layers[layer - 1];
        ul.max_degree = static_cast<uint32_t>(maxM);

        // Count nodes at this layer
        std::vector<uint32_t> node_ids;
        for (int64_t i = 0; i < n_rows; i++) {
            if (hnsw.levels[i] > layer)  // levels[i] = max_layer + 1
                node_ids.push_back(static_cast<uint32_t>(i));
        }
        ul.num_nodes = static_cast<uint32_t>(node_ids.size());

        // Build neighbor array [num_nodes x maxM]
        std::vector<uint32_t> h_neighbors(ul.num_nodes * maxM, UINT32_MAX);
        std::vector<uint32_t> h_node_ids = node_ids;

        for (uint32_t idx = 0; idx < ul.num_nodes; idx++) {
            int64_t i = node_ids[idx];
            size_t begin, end;
            hnsw.neighbor_range(i, layer, &begin, &end);
            uint32_t count = static_cast<uint32_t>(end - begin);
            for (uint32_t j = 0; j < count; j++) {
                auto nb = hnsw.neighbors[begin + j];
                if (nb >= 0)
                    h_neighbors[idx * maxM + j] = static_cast<uint32_t>(nb);
            }
        }

#ifdef GPU_HNSW_DIAGNOSTICS
        {
            uint32_t check_nodes = std::min(ul.num_nodes, (uint32_t)10);
            uint32_t total_slots = check_nodes * maxM;
            uint32_t empty_slots = 0;
            for (uint32_t idx2 = 0; idx2 < check_nodes; idx2++) {
                for (int j = 0; j < maxM; j++) {
                    if (h_neighbors[idx2 * maxM + j] == UINT32_MAX) empty_slots++;
                }
            }
            bool ep_in_layer = false;
            for (uint32_t idx2 = 0; idx2 < ul.num_nodes; idx2++) {
                if (node_ids[idx2] == entry_point) { ep_in_layer = true; break; }
            }
            fprintf(stderr,
                    "[gpu_hnsw_diag]   layer %d: %u nodes, ep_in_layer=%s, "
                    "first-%u-nodes empty_neighbor_slots=%u/%u\n",
                    layer, ul.num_nodes, ep_in_layer ? "YES" : "NO",
                    check_nodes, empty_slots, total_slots);
        }
#endif

        // Upload to device
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&ul.d_node_ids,
                                              ul.num_nodes * sizeof(uint32_t)));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(ul.d_node_ids, h_node_ids.data(),
                                              ul.num_nodes * sizeof(uint32_t),
                                              cudaMemcpyHostToDevice));

        GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&ul.d_neighbors,
                                              ul.num_nodes * maxM * sizeof(uint32_t)));
        GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(ul.d_neighbors, h_neighbors.data(),
                                              ul.num_nodes * maxM * sizeof(uint32_t),
                                              cudaMemcpyHostToDevice));
    }
}

// Normalize stored vectors in-place to unit L2 length (for COSINE metric).
inline void normalize_vectors(std::vector<float>& h_vectors, int64_t n_rows, int64_t dim)
{
    for (int64_t i = 0; i < n_rows; i++) {
        float* v = h_vectors.data() + i * dim;
        float sq_norm = 0.0f;
        for (int64_t d = 0; d < dim; d++) sq_norm += v[d] * v[d];
        if (sq_norm > 0.0f) {
            float inv = 1.0f / std::sqrt(sq_norm);
            for (int64_t d = 0; d < dim; d++) v[d] *= inv;
        }
    }
}

// Upload float32 vectors and HNSW graph to GPU, populating idx in-place.
// h_vectors may be modified (e.g. for cosine normalization) before calling.
inline void upload_to_gpu(gpu_hnsw_index& idx,
                           std::vector<float>& h_vectors,
                           const faiss::cppcontrib::knowhere::HNSW& hnsw,
                           int64_t n_rows,
                           bool is_cosine)
{
    int64_t dim = idx.dim;

    // For COSINE metric: pre-normalize stored vectors to unit length.
    // Our GPU kernel computes plain inner product when use_ip=true, so
    // IP(q/|q|, v/|v|) = cosine(q,v) after normalization.
    if (is_cosine) {
        normalize_vectors(h_vectors, n_rows, dim);
    }

    // Upload float32 vectors
    size_t dataset_bytes = static_cast<size_t>(n_rows) * dim * sizeof(float);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_dataset, dataset_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(idx.d_dataset, h_vectors.data(),
                                          dataset_bytes, cudaMemcpyHostToDevice));

    // Extract and upload graph
    std::vector<uint32_t> h_layer0_flat;
    extract_faiss_hnsw_layers(hnsw, n_rows,
                               idx.upper_layers, h_layer0_flat,
                               idx.entry_point, idx.M, idx.max_degree0, idx.num_layers);

    size_t graph0_bytes = static_cast<size_t>(n_rows) * idx.max_degree0 * sizeof(uint32_t);
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMalloc(&idx.d_layer0_graph, graph0_bytes));
    GPU_HNSW_BUILD_CUDA_CHECK(cudaMemcpy(idx.d_layer0_graph, h_layer0_flat.data(),
                                          graph0_bytes, cudaMemcpyHostToDevice));
}

// Build a gpu_hnsw_index from a faiss::IndexHNSW whose storage is
// IndexScalarQuantizer (QT_8bit / QT_fp16 / QT_bf16 etc.).
// The quantized codes are dequantized to float32 before uploading to the GPU.
// @param use_ip       true for IP/COSINE metrics (GPU kernel uses inner product)
// @param is_cosine    true for COSINE metric (stored vectors must be normalized)
inline std::unique_ptr<gpu_hnsw_index> from_faiss_hnsw_sq(
    const faiss::cppcontrib::knowhere::IndexHNSW& hnsw_index,
    bool use_ip,
    bool is_cosine = false)
{
    const auto* sq_storage = dynamic_cast<const faiss::IndexScalarQuantizer*>(
        hnsw_index.storage);
    if (!sq_storage)
        throw std::runtime_error("gpu_hnsw: storage is not IndexScalarQuantizer");

    int64_t n_rows = hnsw_index.ntotal;
    int64_t dim    = hnsw_index.d;

    // Dequantize all vectors: quantized codes → float32
    std::vector<float> h_vectors(n_rows * dim);
    sq_storage->sa_decode(n_rows, sq_storage->codes.data(), h_vectors.data());

    auto idx = std::make_unique<gpu_hnsw_index>();
    idx->n_rows  = n_rows;
    idx->dim     = dim;
    idx->use_ip  = use_ip;

    upload_to_gpu(*idx, h_vectors, hnsw_index.hnsw, n_rows, is_cosine);
    return idx;
}

// Build a gpu_hnsw_index from a faiss::IndexHNSW whose storage is
// IndexFlat (plain float32 — i.e. the standard HNSW / IndexHNSWFlat).
// @param use_ip       true for IP/COSINE metrics (GPU kernel uses inner product)
// @param is_cosine    true for COSINE metric (stored vectors must be normalized)
inline std::unique_ptr<gpu_hnsw_index> from_faiss_hnsw_flat(
    const faiss::cppcontrib::knowhere::IndexHNSW& hnsw_index,
    bool use_ip,
    bool is_cosine = false)
{
    const auto* flat_storage = dynamic_cast<const faiss::IndexFlat*>(
        hnsw_index.storage);
    if (!flat_storage)
        throw std::runtime_error("gpu_hnsw: storage is not IndexFlat");

    int64_t n_rows = hnsw_index.ntotal;
    int64_t dim    = hnsw_index.d;

    // Copy raw float32 vectors from IndexFlat (stored as uint8 codes, accessed via get_xb())
    const float* xb = flat_storage->get_xb();
    std::vector<float> h_vectors(xb, xb + n_rows * dim);

    auto idx = std::make_unique<gpu_hnsw_index>();
    idx->n_rows  = n_rows;
    idx->dim     = dim;
    idx->use_ip  = use_ip;

    upload_to_gpu(*idx, h_vectors, hnsw_index.hnsw, n_rows, is_cosine);
    return idx;
}

}  // namespace knowhere::detail::gpu_hnsw
