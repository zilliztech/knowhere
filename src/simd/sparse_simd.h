#ifndef KNOWHERE_SIMD_SPARSE_SIMD_H
#define KNOWHERE_SIMD_SPARSE_SIMD_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "knowhere/sparse_utils.h"
#include "simd/instruction_set.h"

namespace knowhere::sparse {

#if defined(__x86_64__) || defined(_M_X64)
// ---- AVX512 BW: Block UB threshold scan ----
// Stride-specific specializations (no loop counter overhead)
bool
scan_block_ub_any_above_avx512_32(const uint16_t* block_ub, uint16_t threshold);
bool
scan_block_ub_any_above_avx512_64(const uint16_t* block_ub, uint16_t threshold);
// Generic loop fallback for non-standard sizes (n must be a multiple of 32).
// In the current DSP code path, callers use kStride = 64, so this precondition holds.
bool
scan_block_ub_any_above_avx512_generic(const uint16_t* block_ub, uint16_t threshold, uint32_t n);
// Legacy entry point — dispatches internally to stride-specific or generic
bool
scan_block_ub_any_above_avx512(const uint16_t* block_ub, uint16_t threshold, uint32_t n);

// ---- AVX512 BW: Block max UB accumulation ----
// Stride-specific specializations (no loop counter overhead)
void
accumulate_block_ub_avx512_32(uint16_t* __restrict ub, const uint8_t* __restrict block_max, uint16_t query_weight);
void
accumulate_block_ub_avx512_64(uint16_t* __restrict ub, const uint8_t* __restrict block_max, uint16_t query_weight);
// Generic loop fallback for non-standard sizes (n must be a multiple of 32).
// In the current DSP code path, callers use kStride = 64, so this precondition holds.
void
accumulate_block_ub_avx512_generic(uint16_t* __restrict ub, const uint8_t* __restrict block_max, uint16_t query_weight,
                                   uint32_t n);
// Legacy entry point — dispatches internally to stride-specific or generic
void
accumulate_block_ub_avx512(uint16_t* ub, const uint8_t* block_max, uint16_t query_weight, uint32_t n);

// Accumulate all dense query-term rows one superblock at a time, keeping the 64 u16 accumulators resident across
// terms. block_max_rows point to full per-term arrays indexed by subblock ID.
void
accumulate_dense_block_ubs_avx512(uint16_t* block_ub, uint64_t* spb_candidate_mask, uint16_t threshold,
                                  const uint8_t* const* block_max_rows, const uint16_t* query_weights, uint32_t n_terms,
                                  const uint32_t* superblock_ids, uint32_t n_superblocks, uint32_t stride);

// Intersect a short sorted query with one sorted block term list. The kernel gallops over
// 16-term chunk maxima and uses one vector equality comparison in the selected chunk.
uint32_t
find_terms_hybrid_avx512(const uint32_t* terms, uint32_t count, const uint32_t* query_dims, uint32_t query_count,
                         uint32_t* positions);

// ---- AVX512: Posting list IP accumulation ----
void
accumulate_posting_list_ip_avx512(const uint32_t* doc_ids, const float* doc_vals, size_t list_size, float q_weight,
                                  float* scores);
#endif

inline constexpr uint32_t kHybridMergeMaxQueryTerms = 16;
inline constexpr uint32_t kHybridMergeMinBlockTerms = 64;

inline bool
find_terms_hybrid_dispatch(const uint32_t* terms, uint32_t count, const uint32_t* query_dims, uint32_t query_count,
                           uint32_t* positions, uint32_t* probes = nullptr) {
#if defined(__x86_64__) || defined(_M_X64)
    if (query_count <= kHybridMergeMaxQueryTerms && count >= kHybridMergeMinBlockTerms &&
        faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512BW()) {
        const uint32_t local_probes = find_terms_hybrid_avx512(terms, count, query_dims, query_count, positions);
        if (probes != nullptr) {
            *probes = local_probes;
        }
        return true;
    }
#endif
    if (probes != nullptr) {
        *probes = 0;
    }
    return false;
}

// Scalar fallback for SIMD block UB scan: check if any of n u16 values > threshold
inline bool
scan_block_ub_any_above_scalar(const uint16_t* block_ub, uint16_t threshold, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        if (block_ub[i] > threshold) {
            return true;
        }
    }
    return false;
}

// Dispatch for block UB scan with runtime CPU detection.
// Routes to stride-specific AVX512 kernels for n=32/64 (the DSP hot path).
inline bool
scan_block_ub_any_above_dispatch(const uint16_t* block_ub, uint16_t threshold, uint32_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512BW()) {
        if (n == 64) {
            return scan_block_ub_any_above_avx512_64(block_ub, threshold);
        }
        if (n == 32) {
            return scan_block_ub_any_above_avx512_32(block_ub, threshold);
        }
        return scan_block_ub_any_above_avx512_generic(block_ub, threshold, n);
    }
#endif
    return scan_block_ub_any_above_scalar(block_ub, threshold, n);
}

// Scalar fallback for u8 block max to u16 UB accumulation
inline void
accumulate_block_ub_scalar(uint16_t* ub, const uint8_t* block_max, uint16_t query_weight, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t prod = static_cast<uint32_t>(query_weight) * block_max[i];
        uint32_t sum = static_cast<uint32_t>(ub[i]) + prod;
        ub[i] = static_cast<uint16_t>(sum < 65535u ? sum : 65535u);
    }
}

// Dispatch for u8 block max to u16 UB accumulation.
// Routes to stride-specific AVX512 kernels for n=32/64 (the DSP hot path).
inline void
accumulate_block_ub_dispatch(uint16_t* __restrict ub, const uint8_t* __restrict block_max, uint16_t query_weight,
                             uint32_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512BW()) {
        if (n == 64) {
            accumulate_block_ub_avx512_64(ub, block_max, query_weight);
            return;
        }
        if (n == 32) {
            accumulate_block_ub_avx512_32(ub, block_max, query_weight);
            return;
        }
        accumulate_block_ub_avx512_generic(ub, block_max, query_weight, n);
        return;
    }
#endif
    accumulate_block_ub_scalar(ub, block_max, query_weight, n);
}

inline void
accumulate_dense_block_ubs_scalar(uint16_t* block_ub, uint64_t* spb_candidate_mask, uint16_t threshold,
                                  const uint8_t* const* block_max_rows, const uint16_t* query_weights, uint32_t n_terms,
                                  const uint32_t* superblock_ids, uint32_t n_superblocks, uint32_t stride) {
    assert(stride == 64 && "accumulate_dense_block_ubs_scalar expects 64 subblocks per superblock");
    for (uint32_t spb_index = 0; spb_index < n_superblocks; ++spb_index) {
        const uint32_t offset = superblock_ids[spb_index] * stride;
        uint16_t accumulators[64];
        std::copy_n(block_ub + offset, stride, accumulators);
        for (uint32_t term = 0; term < n_terms; ++term) {
            const uint8_t* block_max = block_max_rows[term] + offset;
            const uint32_t query_weight = query_weights[term];
            for (uint32_t lane = 0; lane < stride; ++lane) {
                const uint32_t sum = static_cast<uint32_t>(accumulators[lane]) + query_weight * block_max[lane];
                accumulators[lane] = static_cast<uint16_t>(sum < 65535u ? sum : 65535u);
            }
        }
        uint64_t candidate_mask = 0;
        for (uint32_t lane = 0; lane < stride; ++lane) {
            candidate_mask |= static_cast<uint64_t>(accumulators[lane] > threshold) << lane;
        }
        spb_candidate_mask[superblock_ids[spb_index]] = candidate_mask;
        std::copy_n(accumulators, stride, block_ub + offset);
    }
}

inline void
accumulate_dense_block_ubs_dispatch(uint16_t* block_ub, uint64_t* spb_candidate_mask, uint16_t threshold,
                                    const uint8_t* const* block_max_rows, const uint16_t* query_weights,
                                    uint32_t n_terms, const uint32_t* superblock_ids, uint32_t n_superblocks,
                                    uint32_t stride) {
#if defined(__x86_64__) || defined(_M_X64)
    if (faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512BW()) {
        accumulate_dense_block_ubs_avx512(block_ub, spb_candidate_mask, threshold, block_max_rows, query_weights,
                                          n_terms, superblock_ids, n_superblocks, stride);
        return;
    }
#endif
    accumulate_dense_block_ubs_scalar(block_ub, spb_candidate_mask, threshold, block_max_rows, query_weights, n_terms,
                                      superblock_ids, n_superblocks, stride);
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

}  // namespace knowhere::sparse

#endif  // KNOWHERE_SIMD_SPARSE_SIMD_H
