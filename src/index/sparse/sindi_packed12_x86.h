#pragma once

#include <immintrin.h>

#include <cstring>

namespace knowhere::sparse::inverted::sindi {

// Eight contiguous U12 codes occupy exactly 12 bytes. Two bounded loads avoid
// reading beyond an exact-sized stream. Shuffle and extraction stay in registers.
static inline __m128i
unpack12_eight_x86(const uint8_t* p) {
    uint32_t last;
    std::memcpy(&last, p + 8, sizeof(last));

    const auto raw = _mm_insert_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)), last, 2);
    const auto pairs = _mm_shuffle_epi8(raw, _mm_setr_epi8(0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11));
    const auto aligned = _mm_blend_epi16(pairs, _mm_srli_epi16(pairs, 4), 0xaa);
    return _mm_and_si128(aligned, _mm_set1_epi16(4095));
}

}  // namespace knowhere::sparse::inverted::sindi
