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

// AVX512 BW: accumulate u8 block max values into u16 UB array with saturating add
// ub[i] = sat_add_u16(ub[i], query_weight * block_max[i])
// n must be a multiple of 32 (caller pads arrays)
void
accumulate_block_ub_avx512(uint16_t* ub, const uint8_t* block_max, uint16_t query_weight, uint32_t n);
#endif

// Scalar fallback for u8 block max to u16 UB accumulation
inline void
accumulate_block_ub_scalar(uint16_t* ub, const uint8_t* block_max, uint16_t query_weight, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t prod = static_cast<uint32_t>(query_weight) * block_max[i];
        uint32_t sum = static_cast<uint32_t>(ub[i]) + prod;
        ub[i] = static_cast<uint16_t>(sum < 65535u ? sum : 65535u);
    }
}

// Dispatch for u8 block max to u16 UB accumulation
inline void
accumulate_block_ub_dispatch(uint16_t* ub, const uint8_t* block_max, uint16_t query_weight, uint32_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (faiss::cppcontrib::knowhere::InstructionSet::GetInstance().AVX512BW()) {
        accumulate_block_ub_avx512(ub, block_max, query_weight, n);
        return;
    }
#endif
    accumulate_block_ub_scalar(ub, block_max, query_weight, n);
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
