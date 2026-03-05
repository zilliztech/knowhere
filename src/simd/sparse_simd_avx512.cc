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

// This file is compiled with -mavx512f -mavx512dq -mavx512cd flags to enable AVX512 intrinsics
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
inline bool
has_no_conflicts(__m512i indices) {
    __m512i conflicts = _mm512_conflict_epi32(indices);
    __mmask16 has_dups = _mm512_cmpneq_epi32_mask(conflicts, _mm512_setzero_si512());
    return has_dups == 0;
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
void
accumulate_posting_list_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                                  float* scores) {
    constexpr size_t SIMD_WIDTH = 16;
    size_t j = 0;

    __m512 q_weight_vec = _mm512_set1_ps(q_weight);

    // 2x unrolled SIMD loop to hide gather/scatter latency
    for (; j + 2 * SIMD_WIDTH <= list_size; j += 2 * SIMD_WIDTH) {
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);
        __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));

        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);
        __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));

        __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores, sizeof(float));
        __m512 new_scores0 = _mm512_fmadd_ps(vals0, q_weight_vec, current_scores0);
        _mm512_i32scatter_ps(scores, doc_ids0, new_scores0, sizeof(float));

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

    // Masked tail: handle remaining 0-15 elements
    if (j < list_size) {
        __mmask16 mask = (1u << (list_size - j)) - 1;
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
//
// REQUIRES: doc_ids must contain unique values within the processed range.
// This is guaranteed by the posting list construction in add_row_to_index()
// which ensures each document appears at most once per posting list.
void
accumulate_window_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_start, size_t list_end,
                            float q_weight, float* scores, uint32_t window_start) {
    constexpr size_t SIMD_WIDTH = 16;

    __m512 q_weight_vec = _mm512_set1_ps(q_weight);
    __m512i window_start_vec = _mm512_set1_epi32(static_cast<int32_t>(window_start));

    size_t j = list_start;

    // 2x unrolled SIMD loop
    for (; j + 2 * SIMD_WIDTH <= list_end; j += 2 * SIMD_WIDTH) {
        __m512i ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j]));
        __m512 vals0 = _mm512_loadu_ps(&doc_vals[j]);

        __m512i ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&doc_ids[j + SIMD_WIDTH]));
        __m512 vals1 = _mm512_loadu_ps(&doc_vals[j + SIMD_WIDTH]);

        __m512i local_ids0 = _mm512_sub_epi32(ids0, window_start_vec);
        __m512i local_ids1 = _mm512_sub_epi32(ids1, window_start_vec);

        if (check_conflicts && !has_no_conflicts(local_ids0)) {
            scalar_accumulate_window(doc_ids, doc_vals, j, j + SIMD_WIDTH, q_weight, scores, window_start);
        } else {
            __m512 current0 = _mm512_i32gather_ps(local_ids0, scores, sizeof(float));
            __m512 new0 = _mm512_fmadd_ps(vals0, q_weight_vec, current0);
            _mm512_i32scatter_ps(scores, local_ids0, new0, sizeof(float));
        }

        if (check_conflicts && !has_no_conflicts(local_ids1)) {
            scalar_accumulate_window(doc_ids, doc_vals, j + SIMD_WIDTH, j + 2 * SIMD_WIDTH, q_weight, scores,
                                     window_start);
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

        if (check_conflicts && !has_no_conflicts(local_ids)) {
            scalar_accumulate_window(doc_ids, doc_vals, j, j + SIMD_WIDTH, q_weight, scores, window_start);
        } else {
            __m512 current = _mm512_i32gather_ps(local_ids, scores, sizeof(float));
            __m512 new_scores = _mm512_fmadd_ps(vals, q_weight_vec, current);
            _mm512_i32scatter_ps(scores, local_ids, new_scores, sizeof(float));
        }
    }

    // Masked tail: handle remaining 0-15 elements
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

// ============================================================================
// AVX512 SIMD Candidate Extraction
// ============================================================================
// Finds all indices where scores[i] > threshold using SIMD comparison
// Returns candidate indices via compress-store for efficient sparse output
size_t
extract_candidates_avx512(const float* scores, size_t window_size, float threshold, uint32_t* candidates) {
    constexpr size_t SIMD_WIDTH = 16;
    size_t num_candidates = 0;

    __m512 threshold_vec = _mm512_set1_ps(threshold);
    const __m512i base_indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    size_t i = 0;

    for (; i + SIMD_WIDTH <= window_size; i += SIMD_WIDTH) {
        __m512 scores_vec = _mm512_loadu_ps(&scores[i]);
        __mmask16 mask = _mm512_cmp_ps_mask(scores_vec, threshold_vec, _CMP_GT_OQ);
        __m512i chunk_indices = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(i)), base_indices);
        _mm512_mask_compressstoreu_epi32(&candidates[num_candidates], mask, chunk_indices);
        num_candidates += _mm_popcnt_u32(mask);
    }

    // Handle tail elements
    if (i < window_size) {
        __mmask16 valid_mask = (1u << (window_size - i)) - 1;
        __m512 scores_vec = _mm512_maskz_loadu_ps(valid_mask, &scores[i]);
        __mmask16 mask = _mm512_mask_cmp_ps_mask(valid_mask, scores_vec, threshold_vec, _CMP_GT_OQ);
        __m512i chunk_indices = _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(i)), base_indices);
        _mm512_mask_compressstoreu_epi32(&candidates[num_candidates], mask, chunk_indices);
        num_candidates += _mm_popcnt_u32(mask);
    }

    return num_candidates;
}

}  // namespace knowhere::sparse
