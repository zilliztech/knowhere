// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// This file is compiled with -mavx512f flag to enable AVX512 intrinsics
// Runtime CPU detection ensures it's only called on CPUs with AVX512 support

#include <immintrin.h>

#include "sparse_simd.h"

namespace knowhere::sparse {

// ============================================================================
// AVX512 SIMD Implementation (16-wide vectorization with hardware scatter)
// ============================================================================
// Accumulates contributions from a single posting list for IP metric
// scores[doc_ids[i]] += q_weight * doc_vals[i] for all i in [0, list_size)
//
// TODO: Future optimization - pipelined gathers
// Moving gathers earlier (gather0, gather1, then compute0, scatter0, compute1, scatter1)
// could hide ~15-20 cycle gather latency and provide 1.3-1.5x speedup.
// However, this is only safe when doc_ids are unique within the 32-element window.
// For single-term posting lists this is guaranteed (each doc appears once per term),
// but would need conflict detection (AVX512CD) for multi-term fusion scenarios.
void
accumulate_posting_list_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                                  float* scores) {
    constexpr size_t SIMD_WIDTH = 16;  // AVX512 processes 16 floats
    size_t j = 0;

    // Broadcast q_weight to all 16 lanes once before the loops
    __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    // 2x unrolled SIMD loop to hide gather/scatter latency
    for (; j + 2 * SIMD_WIDTH <= list_size; j += 2 * SIMD_WIDTH) {
        // Chunk 0: elements [j, j+16)
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));

        // Chunk 1: elements [j+16, j+32)
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);
        __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));

        // Process chunk 0: new_score = current_score + val * q_weight (FMA)
        __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores, sizeof(float));
        __m512 new_scores0 = _mm512_fmadd_ps(vals0, q_weight_vec, current_scores0);
        _mm512_i32scatter_ps(scores, doc_ids0, new_scores0, sizeof(float));

        // Process chunk 1: new_score = current_score + val * q_weight (FMA)
        __m512 current_scores1 = _mm512_i32gather_ps(doc_ids1, scores, sizeof(float));
        __m512 new_scores1 = _mm512_fmadd_ps(vals1, q_weight_vec, current_scores1);
        _mm512_i32scatter_ps(scores, doc_ids1, new_scores1, sizeof(float));
    }

    // Handle remaining 16-31 elements
    for (; j + SIMD_WIDTH <= list_size; j += SIMD_WIDTH) {
        __m512 vals = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 current_scores = _mm512_i32gather_ps(doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_i32scatter_ps(scores, doc_ids_vec, new_scores, sizeof(float));
    }

    // Masked tail: handle remaining 0-15 elements in SIMD mode instead of scalar
    if (j < list_size) {
        __mmask16 mask = (1u << (list_size - j)) - 1;  // Enable only valid lanes
        __m512 vals = _mm512_maskz_loadu_ps(mask, &doc_vals[j]);
        __m512i doc_ids_vec = _mm512_maskz_loadu_epi32(mask, &doc_ids[j]);
        __m512 current_scores = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, doc_ids_vec, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current_scores);
        _mm512_mask_i32scatter_ps(scores, mask, doc_ids_vec, new_scores, sizeof(float));
    }
}

// ============================================================================
// AVX512 Windowed Accumulation for Batched MaxScore
// ============================================================================
// Accumulates contributions from a posting list segment to a window-relative buffer
// scores[doc_ids[i] - window_start] += q_weight * doc_vals[i]
// Only processes elements where doc_ids[i] is in [window_start, window_end)
//
// REQUIRES: doc_ids must contain unique values within the processed range.
// This is guaranteed by the posting list construction in add_row_to_index()
// (see sparse_inverted_index.h) which ensures each document appears at most
// once per posting list. AVX512 scatter has undefined behavior with duplicate
// indices - only one lane's value would be written, losing other contributions.
//
// Parameters:
//   doc_ids: posting list doc IDs (must be sorted and unique per posting list)
//   doc_vals: posting list values
//   list_start: start index in posting list (elements before this are < window_start)
//   list_end: end index in posting list (elements from this point are >= window_end)
//   q_weight: query term weight
//   scores: window-relative score buffer (size = window_end - window_start)
//   window_start: first doc ID in window (subtracted from doc_ids to get buffer index)
void
accumulate_window_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_start, size_t list_end,
                            float q_weight, float* scores, uint32_t window_start) {
    constexpr size_t SIMD_WIDTH = 16;

    // Broadcast constants
    __m512 q_weight_vec = _mm512_set1_ps(q_weight);
    __m512i window_start_vec = _mm512_set1_epi32(static_cast<int32_t>(window_start));

    size_t j = list_start;

    // 2x unrolled SIMD loop
    for (; j + 2 * SIMD_WIDTH <= list_end; j += 2 * SIMD_WIDTH) {
        // Load doc IDs and values for chunk 0
        __m512i ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);

        // Load doc IDs and values for chunk 1
        __m512i ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);

        // Convert to window-relative indices
        __m512i local_ids0 = _mm512_sub_epi32(ids0, window_start_vec);
        __m512i local_ids1 = _mm512_sub_epi32(ids1, window_start_vec);

        // Gather-FMA-Scatter for chunk 0
        __m512 current0 = _mm512_i32gather_ps(local_ids0, scores, sizeof(float));
        __m512 new0 = _mm512_fmadd_ps(vals0, q_weight_vec, current0);
        _mm512_i32scatter_ps(scores, local_ids0, new0, sizeof(float));

        // Gather-FMA-Scatter for chunk 1
        __m512 current1 = _mm512_i32gather_ps(local_ids1, scores, sizeof(float));
        __m512 new1 = _mm512_fmadd_ps(vals1, q_weight_vec, current1);
        _mm512_i32scatter_ps(scores, local_ids1, new1, sizeof(float));
    }

    // Handle remaining 16-31 elements
    for (; j + SIMD_WIDTH <= list_end; j += SIMD_WIDTH) {
        __m512i ids = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals = _mm512_loadu_ps(&doc_vals[j]);
        __m512i local_ids = _mm512_sub_epi32(ids, window_start_vec);

        __m512 current = _mm512_i32gather_ps(local_ids, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current);
        _mm512_i32scatter_ps(scores, local_ids, new_scores, sizeof(float));
    }

    // Masked tail
    if (j < list_end) {
        __mmask16 mask = (1u << (list_end - j)) - 1;
        __m512i ids = _mm512_maskz_loadu_epi32(mask, &doc_ids[j]);
        __m512 vals = _mm512_maskz_loadu_ps(mask, &doc_vals[j]);
        __m512i local_ids = _mm512_sub_epi32(ids, window_start_vec);

        __m512 current = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, local_ids, scores, sizeof(float));
        __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current);
        _mm512_mask_i32scatter_ps(scores, mask, local_ids, new_scores, sizeof(float));
    }
}

}  // namespace knowhere::sparse
