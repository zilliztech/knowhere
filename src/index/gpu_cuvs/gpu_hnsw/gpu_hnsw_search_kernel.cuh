/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#ifdef GPU_HNSW_DIAGNOSTICS
#include <cstdio>
#endif

namespace cuvs::neighbors::gpu_hnsw::detail {

// ============================================================================
// Distance computation helpers
// ============================================================================

/**
 * Warp-cooperative L2 squared distance.
 * All 32 lanes compute partial sums over strided dimensions, then reduce via shuffle.
 * Only lane 0 holds the final result.
 */
__device__ __forceinline__ float warp_l2_distance(const float* __restrict__ a,
                                                  const float* __restrict__ b,
                                                  int dim)
{
  float partial = 0.0f;
  int lane      = threadIdx.x % 32;
  for (int d = lane; d < dim; d += 32) {
    float diff = a[d] - b[d];
    partial += diff * diff;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffff, partial, offset);
  }
  return partial;
}

/**
 * Warp-cooperative inner product distance (negated for min-heap compatibility).
 */
__device__ __forceinline__ float warp_ip_distance(const float* __restrict__ a,
                                                  const float* __restrict__ b,
                                                  int dim)
{
  float partial = 0.0f;
  int lane      = threadIdx.x % 32;
  for (int d = lane; d < dim; d += 32) {
    partial += a[d] * b[d];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffff, partial, offset);
  }
  return -partial;
}

/**
 * Single-thread distance computation (for use in block-parallel patterns
 * where each thread handles one candidate independently).
 */
__device__ __forceinline__ float thread_l2_distance(const float* __restrict__ a,
                                                    const float* __restrict__ b,
                                                    int dim)
{
  float sum = 0.0f;
  for (int d = 0; d < dim; d++) {
    float diff = a[d] - b[d];
    sum += diff * diff;
  }
  return sum;
}

__device__ __forceinline__ float thread_ip_distance(const float* __restrict__ a,
                                                    const float* __restrict__ b,
                                                    int dim)
{
  float sum = 0.0f;
  for (int d = 0; d < dim; d++) {
    sum += a[d] * b[d];
  }
  return -sum;
}

// ============================================================================
// Phase 1: Upper-layer greedy search
// ============================================================================

struct upper_layer_ptrs {
  const uint32_t* d_node_ids;   // [num_nodes] sorted global IDs at this layer
  const uint32_t* d_neighbors;  // [num_nodes x max_degree]
  uint32_t num_nodes;
  uint32_t max_degree;
};

__device__ __forceinline__ uint32_t binary_search_node(const uint32_t* d_node_ids,
                                                       uint32_t n,
                                                       uint32_t global_id)
{
  uint32_t lo = 0, hi = n;
  while (lo < hi) {
    uint32_t mid = (lo + hi) / 2;
    if (d_node_ids[mid] < global_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < n && d_node_ids[lo] == global_id) return lo;
  return UINT32_MAX;
}

/**
 * Phase 1: one warp per query, greedy walk from top layer down to layer 1.
 * Outputs the entry point for layer-0 beam search.
 *
 * Each lane independently computes per-neighbor distances using thread_*_distance
 * (no warp shuffle needed in the inner loop), then the warp reduces to find the
 * global best.
 */
__global__ void upper_layer_search_kernel(
  const float* __restrict__ d_queries,
  const float* __restrict__ d_dataset,
  const upper_layer_ptrs* __restrict__ d_layer_ptrs,
  uint32_t* __restrict__ d_entry_points,
  uint32_t global_entry_point,
  int num_queries,
  int dim,
  int num_upper_layers,
  bool use_inner_product)
{
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane    = threadIdx.x % 32;
  if (warp_id >= num_queries) return;

  const float* query = d_queries + static_cast<int64_t>(warp_id) * dim;
  uint32_t current   = global_entry_point;

  float best_dist;
  if (lane == 0) {
    if (use_inner_product) {
      best_dist = thread_ip_distance(query, d_dataset + static_cast<int64_t>(current) * dim, dim);
    } else {
      best_dist = thread_l2_distance(query, d_dataset + static_cast<int64_t>(current) * dim, dim);
    }
  }
  best_dist = __shfl_sync(0xffffffff, best_dist, 0);

  // Traverse from top layer down to layer 1
  for (int li = num_upper_layers - 1; li >= 0; li--) {
    const upper_layer_ptrs& lp = d_layer_ptrs[li];
    bool improved = true;
    while (improved) {
      improved = false;
      uint32_t local_idx = binary_search_node(lp.d_node_ids, lp.num_nodes, current);
      if (local_idx == UINT32_MAX) break;

      // Each lane independently checks a different neighbor.
      // We use thread_*_distance per lane + warp reduction for the minimum.
      // warp_*_distance cannot be used here because all 32 lanes must cooperate
      // on the SAME vector pair, but the outer loop assigns different neighbors
      // to different lanes.
      uint32_t best_nbr   = UINT32_MAX;
      float best_nbr_dist = best_dist;

      for (uint32_t j = lane; j < lp.max_degree; j += 32) {
        uint32_t nbr = lp.d_neighbors[static_cast<int64_t>(local_idx) * lp.max_degree + j];
        float dist = FLT_MAX;
        if (nbr != UINT32_MAX) {
          const float* nbr_vec = d_dataset + static_cast<int64_t>(nbr) * dim;
          if (use_inner_product) {
            dist = thread_ip_distance(query, nbr_vec, dim);
          } else {
            dist = thread_l2_distance(query, nbr_vec, dim);
          }
        }
        if (dist < best_nbr_dist) {
          best_nbr_dist = dist;
          best_nbr      = nbr;
        }
      }

      // Warp-level reduction: find the best neighbor across all lanes
      for (int offset = 16; offset > 0; offset >>= 1) {
        float other_dist  = __shfl_down_sync(0xffffffff, best_nbr_dist, offset);
        uint32_t other_id = __shfl_down_sync(0xffffffff, best_nbr, offset);
        if (other_dist < best_nbr_dist) {
          best_nbr_dist = other_dist;
          best_nbr      = other_id;
        }
      }
      best_nbr_dist = __shfl_sync(0xffffffff, best_nbr_dist, 0);
      best_nbr      = __shfl_sync(0xffffffff, best_nbr, 0);

      if (best_nbr != UINT32_MAX && best_nbr_dist < best_dist) {
        best_dist = best_nbr_dist;
        current   = best_nbr;
        improved  = true;
      }
    }
  }

  if (lane == 0) {
    d_entry_points[warp_id] = current;
#ifdef GPU_HNSW_DIAGNOSTICS
    if (warp_id == 0) {
      printf("[beam_diag] q0 upper_layer: ep=%u dist=%.6f (global_ep=%u)\n",
             current, best_dist, global_entry_point);
    }
#endif
  }
}

// ============================================================================
// Phase 2: Layer-0 beam search kernel
// ============================================================================

/**
 * Visited-set using a bitmap in shared memory.
 * For node_id in [0, N), atomically sets bit node_id and returns true if it was
 * newly set (i.e., node not previously visited), false if already visited.
 *
 * Requires bitmap[] to be zero-initialized before use.
 */
__device__ __forceinline__ bool bitmap_visit(uint32_t* bitmap,
                                             uint32_t node_id)
{
  uint32_t word = node_id >> 5;         // node_id / 32
  uint32_t bit  = 1u << (node_id & 31); // node_id % 32
  uint32_t old  = atomicOr(&bitmap[word], bit);
  return (old & bit) == 0;  // true if bit was newly set
}

/**
 * Phase 2: Layer-0 beam search with guided entry points.
 *
 * One thread block per query. Shared memory holds:
 *   - Result buffer: sorted (id, dist) pairs of the best `ef` candidates
 *   - is_expanded: per-slot flags tracking expansion state per result slot
 *   - Staging buffer: newly computed (id, dist) candidates from current iteration
 *   - Parent buffer + metadata
 *
 * The visited bitmap lives in global memory (one bitmap per query block) to
 * support large N (e.g. N=2M → 250KB bitmap, exceeds 48KB shared memory limit).
 * At runtime, L2 cache (48MB+ on modern GPUs) keeps the active bitmap hot,
 * so atomicOr on global memory is fast for typical access patterns.
 *
 * Correctness note on is_expanded[]:
 *   The result buffer is sorted ascending by distance. Insertions at position lo
 *   shift existing entries right. Per-slot is_expanded[] flags are shifted
 *   together with (id, dist), keeping expansion state correct regardless of
 *   insertion position.
 */
__global__ void layer0_beam_search_kernel(
  const float* __restrict__ d_queries,
  const float* __restrict__ d_dataset,
  const uint32_t* __restrict__ d_layer0_graph,
  const uint32_t* __restrict__ d_entry_points,
  uint32_t* __restrict__ d_visited_bitmaps,  // [num_queries x bitmap_words], pre-zeroed
  uint64_t* __restrict__ d_neighbors,
  float* __restrict__ d_distances,
  int num_queries,
  int N,
  int dim,
  int max_degree0,
  int k,
  int ef,
  int search_width,
  int max_iterations,
  bool use_inner_product)
{
  int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const float* query = d_queries + static_cast<int64_t>(query_idx) * dim;

  // --- Shared memory layout ---
  extern __shared__ char smem[];

  int max_staging   = search_width * max_degree0;
  int bitmap_words  = (N + 31) / 32;  // ceil(N/32)

  // Shared memory partitioning (all 4-byte aligned):
  // Visited bitmap lives in global memory (too large for smem at N=2M+).
  uint32_t* result_ids    = reinterpret_cast<uint32_t*>(smem);
  float*    result_dists  = reinterpret_cast<float*>(result_ids + ef);
  uint32_t* is_expanded   = reinterpret_cast<uint32_t*>(result_dists + ef);
  uint32_t* staging_ids   = is_expanded + ef;
  float*    staging_dists = reinterpret_cast<float*>(staging_ids + max_staging);
  uint32_t* parent_ids    = reinterpret_cast<uint32_t*>(staging_dists + max_staging);
  int*      meta          = reinterpret_cast<int*>(parent_ids + search_width);
  // meta[0]=result_count, meta[1]=staging_count, meta[2]=num_parents

  // Per-query visited bitmap in global memory (pre-zeroed by host)
  uint32_t* visited_bmap = d_visited_bitmaps + static_cast<int64_t>(query_idx) * bitmap_words;

  // Initialize result buffer and expansion flags
  for (int i = threadIdx.x; i < ef; i += blockDim.x) {
    result_ids[i]   = UINT32_MAX;
    result_dists[i] = FLT_MAX;
    is_expanded[i]  = 0;
  }
  if (threadIdx.x == 0) {
    meta[0] = 0;
    meta[1] = 0;
    meta[2] = 0;
  }
  __syncthreads();

  // --- Seed with entry point ---
  uint32_t ep = d_entry_points[query_idx];
  if (threadIdx.x == 0) {
    float ep_dist;
    if (use_inner_product) {
      ep_dist = thread_ip_distance(query, d_dataset + static_cast<int64_t>(ep) * dim, dim);
    } else {
      ep_dist = thread_l2_distance(query, d_dataset + static_cast<int64_t>(ep) * dim, dim);
    }
    result_ids[0]   = ep;
    result_dists[0] = ep_dist;
    is_expanded[0]  = 0;
    meta[0] = 1;
    bitmap_visit(visited_bmap, ep);
#ifdef GPU_HNSW_DIAGNOSTICS
    if (query_idx == 0) {
      printf("[beam_diag] q0 seed: ep=%u ep_dist=%.6f ef=%d sw=%d max_iter=%d N=%d\n",
             ep, ep_dist, ef, search_width, max_iterations, N);
    }
#endif
  }
  __syncthreads();

  // --- Seed with entry point's neighbors ---
  if (threadIdx.x == 0) meta[1] = 0;
  __syncthreads();

  for (int j = threadIdx.x; j < max_degree0; j += blockDim.x) {
    uint32_t nbr = d_layer0_graph[static_cast<int64_t>(ep) * max_degree0 + j];
    if (nbr == UINT32_MAX || nbr >= static_cast<uint32_t>(N)) continue;
    if (!bitmap_visit(visited_bmap, nbr)) continue;

    float dist;
    if (use_inner_product) {
      dist = thread_ip_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
    } else {
      dist = thread_l2_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
    }

    int slot = atomicAdd(&meta[1], 1);
    if (slot < max_staging) {
      staging_ids[slot]   = nbr;
      staging_dists[slot] = dist;
    }
  }
  __syncthreads();

  // Thread 0 merges ep's neighbors into result buffer, then marks ep expanded
  if (threadIdx.x == 0) {
    int staging_count = min(meta[1], max_staging);
    int rc = meta[0];
    for (int s = 0; s < staging_count; s++) {
      uint32_t sid  = staging_ids[s];
      float sdist   = staging_dists[s];
      if (rc >= ef && sdist >= result_dists[rc - 1]) continue;

      int lo = 0, hi = rc;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (result_dists[mid] < sdist) lo = mid + 1;
        else hi = mid;
      }

      int insert_end = rc < ef ? rc : ef - 1;
      for (int i = insert_end; i > lo; i--) {
        result_ids[i]   = result_ids[i - 1];
        result_dists[i] = result_dists[i - 1];
        is_expanded[i]  = is_expanded[i - 1];
      }
      result_ids[lo]   = sid;
      result_dists[lo] = sdist;
      is_expanded[lo]  = 0;
      if (rc < ef) rc++;
    }

    // Mark ep expanded (it may have shifted in the result buffer)
    for (int i = 0; i < rc; i++) {
      if (result_ids[i] == ep) {
        is_expanded[i] = 1;
        break;
      }
    }
    meta[0] = rc;
  }
  __syncthreads();

  // --- Main beam search loop ---
  for (int iter = 0; iter < max_iterations; iter++) {
    // Step 1: Thread 0 selects next search_width unexpanded candidates as parents
    if (threadIdx.x == 0) {
      int num_parents = 0;
      int rc = meta[0];
      for (int i = 0; i < rc && num_parents < search_width; i++) {
        if (!is_expanded[i]) {
          parent_ids[num_parents++] = result_ids[i];
          is_expanded[i] = 1;
        }
      }
      meta[2] = num_parents;
    }
    __syncthreads();

    int num_parents = meta[2];
    if (num_parents == 0) {
#ifdef GPU_HNSW_DIAGNOSTICS
      if (query_idx == 0 && threadIdx.x == 0) {
        printf("[beam_diag] q0 converged at iter=%d rc=%d best=%.6f worst=%.6f\n",
               iter, meta[0], result_dists[0],
               meta[0] > 0 ? result_dists[meta[0] - 1] : 999.0f);
      }
#endif
      break;
    }

    // Step 2: Expand parents' neighbors in parallel
    if (threadIdx.x == 0) meta[1] = 0;
    __syncthreads();

    int total_work = num_parents * max_degree0;
    for (int wi = threadIdx.x; wi < total_work; wi += blockDim.x) {
      int parent_idx = wi / max_degree0;
      int nbr_slot   = wi % max_degree0;

      uint32_t parent = parent_ids[parent_idx];
      uint32_t nbr = d_layer0_graph[static_cast<int64_t>(parent) * max_degree0 + nbr_slot];
      if (nbr == UINT32_MAX || nbr >= static_cast<uint32_t>(N)) continue;
      if (!bitmap_visit(visited_bmap, nbr)) continue;

      float dist;
      if (use_inner_product) {
        dist = thread_ip_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
      } else {
        dist = thread_l2_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
      }

      int slot = atomicAdd(&meta[1], 1);
      if (slot < max_staging) {
        staging_ids[slot]   = nbr;
        staging_dists[slot] = dist;
      }
    }
    __syncthreads();

    // Step 3: Thread 0 merges staging buffer into result buffer
    if (threadIdx.x == 0) {
      int staging_count = min(meta[1], max_staging);
      int rc = meta[0];
      for (int s = 0; s < staging_count; s++) {
        uint32_t sid = staging_ids[s];
        float sdist  = staging_dists[s];
        if (rc >= ef && sdist >= result_dists[rc - 1]) continue;

        int lo = 0, hi = rc;
        while (lo < hi) {
          int mid = (lo + hi) / 2;
          if (result_dists[mid] < sdist) lo = mid + 1;
          else hi = mid;
        }
        int insert_end = rc < ef ? rc : ef - 1;
        for (int i = insert_end; i > lo; i--) {
          result_ids[i]   = result_ids[i - 1];
          result_dists[i] = result_dists[i - 1];
          is_expanded[i]  = is_expanded[i - 1];
        }
        result_ids[lo]   = sid;
        result_dists[lo] = sdist;
        is_expanded[lo]  = 0;
        if (rc < ef) rc++;
      }
      meta[0] = rc;
#ifdef GPU_HNSW_DIAGNOSTICS
      if (query_idx == 0 && (iter < 5 || iter % 20 == 0)) {
        printf("[beam_diag] q0 iter=%d parents=%d staged=%d rc=%d best=%.6f worst=%.6f\n",
               iter, num_parents, staging_count, rc,
               result_dists[0], rc > 0 ? result_dists[rc - 1] : 999.0f);
      }
#endif
    }
    __syncthreads();
  }

#ifdef GPU_HNSW_DIAGNOSTICS
  if (query_idx == 0 && threadIdx.x == 0) {
    int final_rc = meta[0];
    printf("[beam_diag] q0 FINAL: rc=%d\n", final_rc);
    for (int i = 0; i < min(10, final_rc); i++) {
      printf("[beam_diag] q0 result[%d]: id=%u dist=%.6f expanded=%u\n",
             i, result_ids[i], result_dists[i], is_expanded[i]);
    }
    int sample_step = N > 10 ? N / 10 : 1;
    float best_sample_dist = FLT_MAX;
    uint32_t best_sample_id = UINT32_MAX;
    for (int si = 0; si < N && si < 10 * sample_step; si += sample_step) {
      float d;
      if (use_inner_product) {
        d = thread_ip_distance(query, d_dataset + static_cast<int64_t>(si) * dim, dim);
      } else {
        d = thread_l2_distance(query, d_dataset + static_cast<int64_t>(si) * dim, dim);
      }
      if (d < best_sample_dist) {
        best_sample_dist = d;
        best_sample_id = si;
      }
    }
    printf("[beam_diag] q0 brute_sample(10 of %d): best_id=%u best_dist=%.6f\n",
           N, best_sample_id, best_sample_dist);
  }
#endif

  // --- Copy top-k results to global memory ---
  int rc = meta[0];
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    if (i < rc) {
      d_neighbors[static_cast<int64_t>(query_idx) * k + i] = static_cast<uint64_t>(result_ids[i]);
      d_distances[static_cast<int64_t>(query_idx) * k + i] = result_dists[i];
    } else {
      d_neighbors[static_cast<int64_t>(query_idx) * k + i] = UINT64_MAX;
      d_distances[static_cast<int64_t>(query_idx) * k + i] = FLT_MAX;
    }
  }
}

/**
 * Calculate shared memory size needed for layer0_beam_search_kernel.
 * N is the number of vectors; visited bitmap size scales with N.
 */
/**
 * Calculate shared memory size needed for layer0_beam_search_kernel.
 * The visited bitmap is in global memory (not counted here).
 */
inline size_t calc_layer0_smem_size(int ef, int search_width, int max_degree0)
{
  int max_staging  = search_width * max_degree0;

  size_t size = 0;
  size += ef * sizeof(uint32_t);            // result_ids
  size += ef * sizeof(float);               // result_dists
  size += ef * sizeof(uint32_t);            // is_expanded
  size += max_staging * sizeof(uint32_t);   // staging_ids
  size += max_staging * sizeof(float);      // staging_dists
  size += search_width * sizeof(uint32_t);  // parent_ids
  size += 3 * sizeof(int);                  // meta
  return size;
}

/**
 * Calculate global memory size needed for visited bitmaps (one per query).
 */
inline size_t calc_visited_bitmap_size(int num_queries, int N)
{
  int bitmap_words = (N + 31) / 32;
  return static_cast<size_t>(num_queries) * bitmap_words * sizeof(uint32_t);
}

}  // namespace cuvs::neighbors::gpu_hnsw::detail
