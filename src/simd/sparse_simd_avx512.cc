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

// This file is compiled with -mavx512f -mavx512bw flags to enable AVX512 intrinsics
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
// AVX512 BW: Block Max UB Accumulation
// ============================================================================
// Accumulates u8 block max scores into u16 upper bound array with saturating add
// ub[i] = sat_add_u16(ub[i], query_weight * block_max[i])
// n must be a multiple of 32 (caller pads arrays)
//
// Uses AVX-512 BW instructions:
// - _mm256_loadu_si256: load 32 u8 block max values
// - _mm512_cvtepu8_epi16: zero-extend 32 u8 → 32 u16
// - _mm512_mullo_epi16: multiply 32 u16 pairs
// - _mm512_adds_epu16: saturating add to u16 accumulators
void
accumulate_block_ub_avx512(uint16_t* ub, const uint8_t* block_max, uint16_t query_weight, uint32_t n) {
    const __m512i qw = _mm512_set1_epi16(static_cast<int16_t>(query_weight));

    for (uint32_t i = 0; i < n; i += 32) {
        // Load 32 u8 block max values
        __m256i bm8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_max + i));
        // Zero-extend to 32 u16
        __m512i bm16 = _mm512_cvtepu8_epi16(bm8);
        // Multiply: product = query_weight * block_max (u16 × u16 → u16 low bits)
        __m512i prod = _mm512_mullo_epi16(bm16, qw);
        // Load current UB values
        __m512i cur = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ub + i));
        // Saturating add
        cur = _mm512_adds_epu16(cur, prod);
        // Store back
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(ub + i), cur);
    }
}

}  // namespace knowhere::sparse
