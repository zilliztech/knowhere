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

// This file is compiled with -mavx512f -mavx512cd flags to enable AVX512 intrinsics
// Runtime CPU detection ensures it's only called on CPUs with AVX512 support

#include <immintrin.h>

#include "sparse_simd.h"

namespace knowhere::sparse {

// ============================================================================
// Conflict Detection Configuration
// ============================================================================
// By default, the code assumes there are no duplicate indices within a SIMD batch,
// and it does not work correctly by design if there are any duplicates.
// Set check_conflicts to true to enable runtime conflict detection with scalar fallback.
// This is useful for debugging or when input data quality cannot be guaranteed.
constexpr bool check_conflicts = false;

// ============================================================================
// Conflict Detection Helper (AVX512CD)
// ============================================================================
// Uses _mm512_conflict_epi32 to detect duplicate indices within a 16-element SIMD batch.
// Returns true if all indices are unique (safe for scatter).
// _mm512_conflict_epi32 returns a vector where each lane i has a bitmask of
// lanes j < i that have the same value. If all lanes are unique, all masks are 0.
inline bool
has_no_conflicts(__m512i indices) {
    __m512i conflicts = _mm512_conflict_epi32(indices);
    __mmask16 has_dups = _mm512_cmpneq_epi32_mask(conflicts, _mm512_setzero_si512());
    return has_dups == 0;  // true = no duplicates, safe for scatter
}

// Scalar fallback for when conflicts are detected
inline void
scalar_accumulate_window(const uint32_t* doc_ids, const float* doc_vals, size_t start, size_t end, float q_weight,
                         float* scores, uint32_t window_start) {
    for (size_t i = start; i < end; ++i) {
        uint32_t local_id = doc_ids[i] - window_start;
        scores[local_id] += q_weight * doc_vals[i];
    }
}

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
// If check_conflicts is enabled, uses AVX512CD to detect duplicates at runtime
// and falls back to scalar accumulation when conflicts are found.
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

        // Process chunk 0: check for conflicts if enabled
        if constexpr (check_conflicts) {
            if (!has_no_conflicts(local_ids0)) {
                scalar_accumulate_window(doc_ids, doc_vals, j, j + SIMD_WIDTH, q_weight, scores, window_start);
            } else {
                __m512 current0 = _mm512_i32gather_ps(local_ids0, scores, sizeof(float));
                __m512 new0 = _mm512_fmadd_ps(vals0, q_weight_vec, current0);
                _mm512_i32scatter_ps(scores, local_ids0, new0, sizeof(float));
            }
        } else {
            __m512 current0 = _mm512_i32gather_ps(local_ids0, scores, sizeof(float));
            __m512 new0 = _mm512_fmadd_ps(vals0, q_weight_vec, current0);
            _mm512_i32scatter_ps(scores, local_ids0, new0, sizeof(float));
        }

        // Process chunk 1: check for conflicts if enabled
        if constexpr (check_conflicts) {
            if (!has_no_conflicts(local_ids1)) {
                scalar_accumulate_window(doc_ids, doc_vals, j + SIMD_WIDTH, j + 2 * SIMD_WIDTH, q_weight, scores,
                                         window_start);
            } else {
                __m512 current1 = _mm512_i32gather_ps(local_ids1, scores, sizeof(float));
                __m512 new1 = _mm512_fmadd_ps(vals1, q_weight_vec, current1);
                _mm512_i32scatter_ps(scores, local_ids1, new1, sizeof(float));
            }
        } else {
            __m512 current1 = _mm512_i32gather_ps(local_ids1, scores, sizeof(float));
            __m512 new1 = _mm512_fmadd_ps(vals1, q_weight_vec, current1);
            _mm512_i32scatter_ps(scores, local_ids1, new1, sizeof(float));
        }
    }

    // Handle remaining 16-31 elements
    for (; j + SIMD_WIDTH <= list_end; j += SIMD_WIDTH) {
        __m512i ids = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals = _mm512_loadu_ps(&doc_vals[j]);
        __m512i local_ids = _mm512_sub_epi32(ids, window_start_vec);

        if constexpr (check_conflicts) {
            if (!has_no_conflicts(local_ids)) {
                scalar_accumulate_window(doc_ids, doc_vals, j, j + SIMD_WIDTH, q_weight, scores, window_start);
            } else {
                __m512 current = _mm512_i32gather_ps(local_ids, scores, sizeof(float));
                __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current);
                _mm512_i32scatter_ps(scores, local_ids, new_scores, sizeof(float));
            }
        } else {
            __m512 current = _mm512_i32gather_ps(local_ids, scores, sizeof(float));
            __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current);
            _mm512_i32scatter_ps(scores, local_ids, new_scores, sizeof(float));
        }
    }

    // Masked tail - use scalar for simplicity (small number of elements)
    if (j < list_end) {
        scalar_accumulate_window(doc_ids, doc_vals, j, list_end, q_weight, scores, window_start);
    }
}

// ============================================================================
// AVX512 SIMD Candidate Extraction
// ============================================================================
// Finds all indices where scores[i] > threshold using SIMD comparison
// Returns candidate indices via compress-store for efficient sparse output
//
// Performance: ~4x faster than scalar for sparse candidates (threshold filters most)
// Uses compare-mask + compress-store pattern to avoid branch mispredictions
size_t
extract_candidates_avx512(const float* scores, size_t window_size, float threshold, uint32_t* candidates) {
    constexpr size_t SIMD_WIDTH = 16;
    size_t num_candidates = 0;

    // Broadcast threshold for comparison
    __m512 threshold_vec = _mm512_set1_ps(threshold);

    // Base indices template [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    const __m512i base_indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    size_t i = 0;

    // Main SIMD loop - process 16 scores at a time
    for (; i + SIMD_WIDTH <= window_size; i += SIMD_WIDTH) {
        __m512 scores_vec = _mm512_loadu_ps(&scores[i]);

        // Compare: mask bit set where score > threshold
        __mmask16 mask = _mm512_cmp_ps_mask(scores_vec, threshold_vec, _CMP_GT_OQ);

        if (mask != 0) {
            // Create indices for this chunk: [i, i+1, i+2, ..., i+15]
            __m512i chunk_indices = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(i)), base_indices);

            // Compress-store: write only the indices where mask bit is set
            _mm512_mask_compressstoreu_epi32(&candidates[num_candidates], mask, chunk_indices);
            num_candidates += _mm_popcnt_u32(mask);
        }
    }

    // Handle tail elements (0-15 remaining)
    if (i < window_size) {
        __mmask16 valid_mask = (1u << (window_size - i)) - 1;
        __m512 scores_vec = _mm512_maskz_loadu_ps(valid_mask, &scores[i]);

        // Compare only valid elements
        __mmask16 mask = _mm512_mask_cmp_ps_mask(valid_mask, scores_vec, threshold_vec, _CMP_GT_OQ);

        if (mask != 0) {
            __m512i chunk_indices = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(i)), base_indices);
            _mm512_mask_compressstoreu_epi32(&candidates[num_candidates], mask, chunk_indices);
            num_candidates += _mm_popcnt_u32(mask);
        }
    }

    return num_candidates;
}

}  // namespace knowhere::sparse
