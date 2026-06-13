/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, 6sense Insights Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <vector>

namespace knowhere::detail::gpu_hnsw {

struct search_params {
    int ef = 200;
    int search_width = 8;
    int max_iterations = 0;     // 0 = auto
    int thread_block_size = 0;  // 0 = auto (128)
    int overflow_factor = 2;    // overflow_ef = overflow_factor * ef (secondary candidate queue)
};

struct device_upper_layer {
    uint32_t* d_node_ids = nullptr;
    uint32_t* d_neighbors = nullptr;
    uint32_t num_nodes = 0;
    uint32_t max_degree = 0;
};

// Pre-allocated GPU scratch buffers for search, reused across calls.
// Avoids per-search cudaMalloc/cudaFree overhead (~1-2ms per call).
struct search_scratch {
    float* d_queries = nullptr;
    uint64_t* d_neighbors = nullptr;
    float* d_distances = nullptr;
    uint32_t* d_entry_points = nullptr;
    uint32_t* d_visited_bitmaps = nullptr;

    // Overflow candidate queue (OCQ) — secondary expansion buffer in global memory.
    // Holds candidates rejected from the result buffer that are still worth expanding.
    uint32_t* d_overflow_ids = nullptr;       // [nq * overflow_ef]
    float* d_overflow_dists = nullptr;        // [nq * overflow_ef]
    uint32_t* d_overflow_expanded = nullptr;  // [nq * overflow_ef]
    int* d_overflow_count = nullptr;          // [nq] current valid entries per query

    size_t queries_bytes = 0;    // nq * dim * sizeof(float)
    size_t neighbors_bytes = 0;  // nq * k * sizeof(uint64_t)
    size_t distances_bytes = 0;  // nq * k * sizeof(float)
    int entry_cap = 0;           // max nq for entry_points
    size_t bitmap_bytes = 0;     // nq * bitmap_words * sizeof(uint32_t)
    size_t overflow_bytes = 0;   // total allocated for overflow buffers

    void
    ensure(int nq, int k, int dim, int N, int overflow_ef) {
        size_t need_q = static_cast<size_t>(nq) * dim * sizeof(float);
        if (need_q > queries_bytes) {
            if (d_queries)
                cudaFree(d_queries);
            cudaMalloc(&d_queries, need_q);
            queries_bytes = need_q;
        }
        size_t need_n = static_cast<size_t>(nq) * k * sizeof(uint64_t);
        if (need_n > neighbors_bytes) {
            if (d_neighbors)
                cudaFree(d_neighbors);
            cudaMalloc(&d_neighbors, need_n);
            neighbors_bytes = need_n;
        }
        size_t need_d = static_cast<size_t>(nq) * k * sizeof(float);
        if (need_d > distances_bytes) {
            if (d_distances)
                cudaFree(d_distances);
            cudaMalloc(&d_distances, need_d);
            distances_bytes = need_d;
        }
        if (nq > entry_cap) {
            if (d_entry_points)
                cudaFree(d_entry_points);
            cudaMalloc(&d_entry_points, static_cast<size_t>(nq) * sizeof(uint32_t));
            entry_cap = nq;
        }
        int bitmap_words = (N + 31) / 32;
        size_t need_bm = static_cast<size_t>(nq) * bitmap_words * sizeof(uint32_t);
        if (need_bm > bitmap_bytes) {
            if (d_visited_bitmaps)
                cudaFree(d_visited_bitmaps);
            cudaMalloc(&d_visited_bitmaps, need_bm);
            bitmap_bytes = need_bm;
        }
        // Overflow candidate queue buffers
        size_t ovf_entries = static_cast<size_t>(nq) * overflow_ef;
        size_t need_ovf =
            ovf_entries * (sizeof(uint32_t) + sizeof(float) + sizeof(uint32_t)) + static_cast<size_t>(nq) * sizeof(int);
        if (need_ovf > overflow_bytes) {
            if (d_overflow_ids)
                cudaFree(d_overflow_ids);
            if (d_overflow_dists)
                cudaFree(d_overflow_dists);
            if (d_overflow_expanded)
                cudaFree(d_overflow_expanded);
            if (d_overflow_count)
                cudaFree(d_overflow_count);
            cudaMalloc(&d_overflow_ids, ovf_entries * sizeof(uint32_t));
            cudaMalloc(&d_overflow_dists, ovf_entries * sizeof(float));
            cudaMalloc(&d_overflow_expanded, ovf_entries * sizeof(uint32_t));
            cudaMalloc(&d_overflow_count, static_cast<size_t>(nq) * sizeof(int));
            overflow_bytes = need_ovf;
        }
    }

    ~search_scratch() {
        if (d_queries)
            cudaFree(d_queries);
        if (d_neighbors)
            cudaFree(d_neighbors);
        if (d_distances)
            cudaFree(d_distances);
        if (d_entry_points)
            cudaFree(d_entry_points);
        if (d_visited_bitmaps)
            cudaFree(d_visited_bitmaps);
        if (d_overflow_ids)
            cudaFree(d_overflow_ids);
        if (d_overflow_dists)
            cudaFree(d_overflow_dists);
        if (d_overflow_expanded)
            cudaFree(d_overflow_expanded);
        if (d_overflow_count)
            cudaFree(d_overflow_count);
    }

    search_scratch() = default;
    search_scratch(const search_scratch&) = delete;
    search_scratch&
    operator=(const search_scratch&) = delete;
};

struct gpu_hnsw_index {
    void* d_dataset = nullptr;           // [n_rows x dim], row-major. int8 if dataset_int8, else fp32
    bool dataset_int8 = false;           // true when stored as int8_t (1 byte/elem, direct SQ8)
    float* d_inv_norms = nullptr;        // [n_rows] reciprocal L2 norms (COSINE + INT8 only)
    uint32_t* d_layer0_graph = nullptr;  // [n_rows x max_degree0]
    std::vector<device_upper_layer> upper_layers;

    int64_t n_rows = 0;
    int64_t dim = 0;
    uint32_t entry_point = 0;
    int num_layers = 0;
    int M = 0;
    int max_degree0 = 0;
    bool use_ip = false;

    // Pre-built upper layer device pointers (uploaded once at build time)
    void* d_upper_layer_ptrs = nullptr;
    int num_upper_layers_built = 0;

    // Dedicated CUDA stream for async search operations
    cudaStream_t search_stream = nullptr;

    // Pre-allocated scratch buffers (protected by mutex for concurrent access)
    mutable std::mutex scratch_mutex;
    mutable search_scratch scratch;

    ~gpu_hnsw_index() {
        if (d_dataset)
            cudaFree(d_dataset);
        if (d_inv_norms)
            cudaFree(d_inv_norms);
        if (d_layer0_graph)
            cudaFree(d_layer0_graph);
        for (auto& ul : upper_layers) {
            if (ul.d_node_ids)
                cudaFree(ul.d_node_ids);
            if (ul.d_neighbors)
                cudaFree(ul.d_neighbors);
        }
        if (d_upper_layer_ptrs)
            cudaFree(d_upper_layer_ptrs);
        if (search_stream)
            cudaStreamDestroy(search_stream);
    }
};

}  // namespace knowhere::detail::gpu_hnsw
