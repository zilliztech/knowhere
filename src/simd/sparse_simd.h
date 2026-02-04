#ifndef KNOWHERE_SIMD_SPARSE_SIMD_H
#define KNOWHERE_SIMD_SPARSE_SIMD_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "knowhere/sparse_utils.h"
#include "simd/instruction_set.h"

namespace knowhere::sparse {

#if defined(__x86_64__) || defined(_M_X64)
void
accumulate_posting_list_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                                  float* scores);

// AVX512 windowed accumulation for batched MaxScore
// scores[doc_ids[i] - window_start] += q_weight * doc_vals[i] for i in [list_start, list_end)
void
accumulate_window_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_start, size_t list_end,
                            float q_weight, float* scores, uint32_t window_start);

// AVX512 SIMD seek: find first position where id >= target (16-wide)
size_t
simd_seek_avx512_impl(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target);

// AVX512 SIMD candidate extraction: find indices where scores[i] > threshold
// Returns number of candidates found, writes candidate indices to candidates array
// candidates array must have space for at least window_size elements
size_t
extract_candidates_avx512(const float* scores, size_t window_size, float threshold, uint32_t* candidates);
#endif

// Scalar seek implementation (fallback)
inline size_t
scalar_seek_impl(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
    for (size_t pos = start_pos; pos < size; pos++) {
        if (ids[pos] >= target) {
            return pos;
        }
    }
    return size;
}

// Minimum remaining elements to use SIMD seek (threshold)
// For small seeks, scalar is faster due to SIMD setup overhead
constexpr size_t SIMD_SEEK_THRESHOLD = 64;

// Dispatch function for SIMD seek with runtime CPU detection
// Only uses SIMD when there are enough remaining elements to amortize setup cost
inline size_t
simd_seek_dispatch(const uint32_t* __restrict__ ids, size_t size, size_t start_pos, uint32_t target) {
    size_t remaining = size - start_pos;

#if defined(__x86_64__) || defined(_M_X64)
    // Only use SIMD for larger seeks where setup overhead is amortized
    if (remaining >= SIMD_SEEK_THRESHOLD) {
        if (faiss::InstructionSet::GetInstance().AVX512F()) {
            return simd_seek_avx512_impl(ids, size, start_pos, target);
        }
    }
#endif
    return scalar_seek_impl(ids, size, start_pos, target);
}

template <typename QType>
inline void
accumulate_posting_list_contribution_ip_dispatch(const uint32_t* doc_ids, const QType* doc_vals, size_t list_size,
                                                 float q_weight, float* scores) {
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<QType, float>) {
        if (faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512F()) {
            accumulate_posting_list_ip_avx512(doc_ids, doc_vals, list_size, q_weight, scores);
            return;
        }
    }
#endif

    // Scalar fallback for IP computation
    for (size_t i = 0; i < list_size; ++i) {
        const auto doc_id = doc_ids[i];
        scores[doc_id] += q_weight * static_cast<float>(doc_vals[i]);
    }
}

// Dispatch function for windowed accumulation (used in batched MaxScore)
// Accumulates scores[doc_ids[i] - window_start] += q_weight * doc_vals[i] for i in [list_start, list_end)
template <typename QType>
inline void
accumulate_window_ip_dispatch(const uint32_t* doc_ids, const QType* doc_vals, size_t list_start, size_t list_end,
                              float q_weight, float* scores, uint32_t window_start) {
#if defined(__x86_64__) || defined(_M_X64)
    if constexpr (std::is_same_v<QType, float>) {
        if (faiss::InstructionSet::GetInstance().AVX512F()) {
            accumulate_window_ip_avx512(doc_ids, doc_vals, list_start, list_end, q_weight, scores, window_start);
            return;
        }
    }
#endif

    // Scalar fallback
    for (size_t i = list_start; i < list_end; ++i) {
        const uint32_t local_id = doc_ids[i] - window_start;
        scores[local_id] += q_weight * static_cast<float>(doc_vals[i]);
    }
}

// Scalar candidate extraction fallback
inline size_t
extract_candidates_scalar(const float* scores, size_t window_size, float threshold, uint32_t* candidates) {
    size_t num_candidates = 0;
    for (size_t i = 0; i < window_size; ++i) {
        if (scores[i] > threshold) {
            candidates[num_candidates++] = static_cast<uint32_t>(i);
        }
    }
    return num_candidates;
}

// Dispatch function for candidate extraction with runtime CPU detection
inline size_t
extract_candidates_dispatch(const float* scores, size_t window_size, float threshold, uint32_t* candidates) {
#if defined(__x86_64__) || defined(_M_X64)
    if (faiss::InstructionSet::GetInstance().AVX512F()) {
        return extract_candidates_avx512(scores, window_size, threshold, candidates);
    }
#endif
    return extract_candidates_scalar(scores, window_size, threshold, candidates);
}

}  // namespace knowhere::sparse

#endif  // KNOWHERE_SIMD_SPARSE_SIMD_H
