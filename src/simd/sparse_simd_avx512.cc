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
template <typename QType>
std::vector<float>
compute_all_distances_avx512(size_t n_rows_internal, const std::vector<std::pair<size_t, float>>& q_vec,
                             const std::vector<boost::span<const table_t>>& inverted_index_ids_spans,
                             const std::vector<boost::span<const QType>>& inverted_index_vals_spans,
                             const DocValueComputer<float>& computer, SparseMetricType metric_type,
                             const boost::span<const float>* doc_len_ratios_spans_ptr) {
    // Static asserts for type safety
    static_assert(sizeof(table_t) == 4, "SIMD gather requires 32-bit doc IDs");
    static_assert(std::is_same_v<QType, float>, "SIMD operations require float values");

    // Note: This function is only called for IP metric with float values
    // (BM25 uses scalar path due to DocValueComputer overhead)
    (void)metric_type;               // Unused, kept for API consistency
    (void)computer;                  // Unused for IP metric
    (void)doc_len_ratios_spans_ptr;  // Unused for IP metric

    std::vector<float> scores(n_rows_internal, 0.0f);
    constexpr size_t SIMD_WIDTH = 16;  // AVX512 processes 16 floats

    // IP metric - simple multiplication without DocValueComputer
    for (size_t i = 0; i < q_vec.size(); ++i) {
        const auto& plist_ids = inverted_index_ids_spans[q_vec[i].first];
        const auto& plist_vals = inverted_index_vals_spans[q_vec[i].first];
        const float q_weight = q_vec[i].second;

        size_t j = 0;

        // 2x unrolled SIMD loop to hide gather latency
        for (; j + 2 * SIMD_WIDTH <= plist_ids.size(); j += 2 * SIMD_WIDTH) {
            // No manual prefetch - random access patterns don't benefit and can pollute cache
            // Hardware prefetchers + AVX512 gather units handle this better

            // Chunk 0: elements [j, j+16)
            __m512 vals0 = _mm512_loadu_ps(reinterpret_cast<const float*>(&plist_vals[j]));
            __m512i doc_ids0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&plist_ids[j]));

            // Chunk 1: elements [j+16, j+32)
            __m512 vals1 = _mm512_loadu_ps(reinterpret_cast<const float*>(&plist_vals[j + SIMD_WIDTH]));
            __m512i doc_ids1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&plist_ids[j + SIMD_WIDTH]));

            __m512 q_weight_vec = _mm512_set1_ps(q_weight);

            // Process chunk 0
            __m512 contribution0 = _mm512_mul_ps(vals0, q_weight_vec);
            __m512 current_scores0 = _mm512_i32gather_ps(doc_ids0, scores.data(), sizeof(float));
            __m512 new_scores0 = _mm512_add_ps(current_scores0, contribution0);
            _mm512_i32scatter_ps(scores.data(), doc_ids0, new_scores0, sizeof(float));

            // Process chunk 1
            __m512 contribution1 = _mm512_mul_ps(vals1, q_weight_vec);
            __m512 current_scores1 = _mm512_i32gather_ps(doc_ids1, scores.data(), sizeof(float));
            __m512 new_scores1 = _mm512_add_ps(current_scores1, contribution1);
            _mm512_i32scatter_ps(scores.data(), doc_ids1, new_scores1, sizeof(float));
        }

        // Handle remaining 16-31 elements
        for (; j + SIMD_WIDTH <= plist_ids.size(); j += SIMD_WIDTH) {
            __m512 vals = _mm512_loadu_ps(reinterpret_cast<const float*>(&plist_vals[j]));
            __m512i doc_ids = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&plist_ids[j]));
            __m512 q_weight_vec = _mm512_set1_ps(q_weight);
            __m512 contribution = _mm512_mul_ps(vals, q_weight_vec);
            __m512 current_scores = _mm512_i32gather_ps(doc_ids, scores.data(), sizeof(float));
            __m512 new_scores = _mm512_add_ps(current_scores, contribution);
            _mm512_i32scatter_ps(scores.data(), doc_ids, new_scores, sizeof(float));
        }

        // Scalar tail (remaining 0-15 elements)
        for (; j < plist_ids.size(); ++j) {
            scores[plist_ids[j]] += q_weight * plist_vals[j];
        }
    }

    return scores;
}

// Explicit template instantiation for float
template std::vector<float>
compute_all_distances_avx512<float>(size_t n_rows_internal, const std::vector<std::pair<size_t, float>>& q_vec,
                                    const std::vector<boost::span<const table_t>>& inverted_index_ids_spans,
                                    const std::vector<boost::span<const float>>& inverted_index_vals_spans,
                                    const DocValueComputer<float>& computer, SparseMetricType metric_type,
                                    const boost::span<const float>* doc_len_ratios_spans_ptr);

}  // namespace knowhere::sparse
