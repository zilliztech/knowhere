/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/******************************************************************************
 * Knowhere-local prelude over baseline sq-avx2.cpp.
 *
 * See sq-avx512-fastpath.cpp for the full design note on how this pattern
 * works (full DCTemplate specialization declared here; baseline .cpp
 * included below; template lookup picks our specialization).
 *
 * This file ports the AVX2 variant of the fork's DistanceComputerSQ4UByte
 * for QT_4bit_uniform + L2.
 *****************************************************************************/

#ifdef COMPILE_SIMD_AVX2

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

#include <immintrin.h>
#include <cmath>
#include <vector>

namespace faiss {

namespace scalar_quantizer {

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec4bit<SIMDLevel::AVX2>,
                QuantizerTemplateScaling::UNIFORM,
                SIMDLevel::AVX2>,
        SimilarityL2<SIMDLevel::AVX2>,
        SIMDLevel::AVX2> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::AVX2>;

    size_t d;
    float vmin;
    float vdiff;
    float final_scale_sq;
    std::vector<uint8_t> q_lo;
    std::vector<uint8_t> q_hi;

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              vmin(trained[0]),
              vdiff(trained[1]),
              // Over-allocate by 32 bytes so full 256-bit loads past the
              // logical length read a safe zero-filled tail.
              q_lo((d_in + 1) / 2 + 32, 0),
              q_hi((d_in + 1) / 2 + 32, 0) {
        const float final_scale = vdiff / 15.0f;
        final_scale_sq = final_scale * final_scale;
    }

    void set_query(const float* x) final {
        this->q = x;
        const float inv_scale = (vdiff == 0.0f) ? 0.0f : 15.0f / vdiff;
        for (size_t i = 0; i < d; i++) {
            float val = (x[i] - vmin) * inv_scale;
            int code = static_cast<int>(std::floor(val + 0.5f));
            if (code < 0) {
                code = 0;
            }
            if (code > 15) {
                code = 15;
            }
            if (i % 2 == 0) {
                q_lo[i / 2] = static_cast<uint8_t>(code);
            } else {
                q_hi[i / 2] = static_cast<uint8_t>(code);
            }
        }
    }

    float query_to_code(const uint8_t* code) const final {
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        __m256i acc = _mm256_setzero_si256();
        const __m256i mask_f = _mm256_set1_epi8(0xF);
        const __m256i one = _mm256_set1_epi16(1);
        const __m256i zero = _mm256_setzero_si256();

        size_t i = 0;
        // 64 dims per iteration (32 bytes of packed 4-bit codes).
        for (; i + 64 <= d; i += 64) {
            __m256i c256 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(code + i / 2));

            __m256i nibbles_lo = _mm256_and_si256(c256, mask_f);
            __m256i nibbles_hi =
                    _mm256_and_si256(_mm256_srli_epi16(c256, 4), mask_f);

            __m256i q_lo_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q_lo_ptr + i / 2));
            __m256i q_hi_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q_hi_ptr + i / 2));

            __m256i diff_lo = _mm256_sub_epi8(q_lo_vec, nibbles_lo);
            __m256i diff_hi = _mm256_sub_epi8(q_hi_vec, nibbles_hi);

            // AVX2 has no _mm256_abs_epi8; emulate via max(x, -x).
            diff_lo = _mm256_max_epi8(diff_lo, _mm256_sub_epi8(zero, diff_lo));
            diff_hi = _mm256_max_epi8(diff_hi, _mm256_sub_epi8(zero, diff_hi));

            __m256i sq_lo = _mm256_maddubs_epi16(diff_lo, diff_lo);
            __m256i sq_hi = _mm256_maddubs_epi16(diff_hi, diff_hi);

            __m256i sum_lo = _mm256_madd_epi16(sq_lo, one);
            __m256i sum_hi = _mm256_madd_epi16(sq_hi, one);

            acc = _mm256_add_epi32(acc, sum_lo);
            acc = _mm256_add_epi32(acc, sum_hi);
        }

        // Horizontal reduction.
        __m128i acc_lo = _mm256_castsi256_si128(acc);
        __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
        acc_lo = _mm_add_epi32(acc_lo, acc_hi);
        acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
        acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
        int32_t sum = _mm_cvtsi128_si32(acc_lo);

        // Scalar tail.
        for (; i < d; i++) {
            uint8_t c = code[i / 2];
            uint8_t nibble = (i % 2 == 0)
                    ? static_cast<uint8_t>(c & 0x0F)
                    : static_cast<uint8_t>(c >> 4);
            int q_code = (i % 2 == 0) ? q_lo[i / 2] : q_hi[i / 2];
            int diff = q_code - int(nibble);
            sum += diff * diff;
        }

        return static_cast<float>(sum) * final_scale_sq;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        const uint8_t* c1 = codes + i * code_size;
        const uint8_t* c2 = codes + j * code_size;
        int64_t acc = 0;
        for (size_t k = 0; k < d; k++) {
            uint8_t a = (k % 2 == 0)
                    ? static_cast<uint8_t>(c1[k / 2] & 0x0F)
                    : static_cast<uint8_t>(c1[k / 2] >> 4);
            uint8_t b = (k % 2 == 0)
                    ? static_cast<uint8_t>(c2[k / 2] & 0x0F)
                    : static_cast<uint8_t>(c2[k / 2] >> 4);
            int diff = int(a) - int(b);
            acc += diff * diff;
        }
        return static_cast<float>(acc) * final_scale_sq;
    }

    /// Batch-4: 128 dims per outer iter, two 64-dim chunks sharing q_lo/q_hi
    /// loads across four input codes. Ported verbatim from the fork's AVX2
    /// DistanceComputerSQ4UByte_avx. AVX2 has no abs_epi8 so |diff| is
    /// emulated via max(x, -x).
    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        const __m256i mask_f = _mm256_set1_epi8(0x0F);
        const __m256i one = _mm256_set1_epi16(1);
        const __m256i zero = _mm256_setzero_si256();

        size_t i = 0;
        // 128 dims per outer iter.
        for (; i + 128 <= d; i += 128) {
            __m256i q_lo_0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q_lo_ptr + i / 2));
            __m256i q_hi_0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q_hi_ptr + i / 2));

            auto process_chunk_64 = [&](const uint8_t* code,
                                        __m256i& acc,
                                        __m256i q_lo_v,
                                        __m256i q_hi_v,
                                        int offset) {
                __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        code + i / 2 + offset));
                __m256i nibbles_lo = _mm256_and_si256(c, mask_f);
                __m256i nibbles_hi =
                        _mm256_and_si256(_mm256_srli_epi16(c, 4), mask_f);

                __m256i diff_lo = _mm256_sub_epi8(q_lo_v, nibbles_lo);
                __m256i diff_hi = _mm256_sub_epi8(q_hi_v, nibbles_hi);

                diff_lo = _mm256_max_epi8(
                        diff_lo, _mm256_sub_epi8(zero, diff_lo));
                diff_hi = _mm256_max_epi8(
                        diff_hi, _mm256_sub_epi8(zero, diff_hi));

                __m256i sq_lo = _mm256_maddubs_epi16(diff_lo, diff_lo);
                __m256i sq_hi = _mm256_maddubs_epi16(diff_hi, diff_hi);

                __m256i sum_lo = _mm256_madd_epi16(sq_lo, one);
                __m256i sum_hi = _mm256_madd_epi16(sq_hi, one);

                acc = _mm256_add_epi32(acc, sum_lo);
                acc = _mm256_add_epi32(acc, sum_hi);
            };

            process_chunk_64(code_0, acc0, q_lo_0, q_hi_0, 0);
            process_chunk_64(code_1, acc1, q_lo_0, q_hi_0, 0);
            process_chunk_64(code_2, acc2, q_lo_0, q_hi_0, 0);
            process_chunk_64(code_3, acc3, q_lo_0, q_hi_0, 0);

            __m256i q_lo_1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q_lo_ptr + i / 2 + 32));
            __m256i q_hi_1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q_hi_ptr + i / 2 + 32));

            process_chunk_64(code_0, acc0, q_lo_1, q_hi_1, 32);
            process_chunk_64(code_1, acc1, q_lo_1, q_hi_1, 32);
            process_chunk_64(code_2, acc2, q_lo_1, q_hi_1, 32);
            process_chunk_64(code_3, acc3, q_lo_1, q_hi_1, 32);
        }

        // 64-dim remainder chunk.
        if (i + 64 <= d) {
            __m256i q_lo_0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q_lo_ptr + i / 2));
            __m256i q_hi_0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(q_hi_ptr + i / 2));

            auto process = [&](const uint8_t* code, __m256i& acc) {
                __m256i c = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(code + i / 2));
                __m256i nibbles_lo = _mm256_and_si256(c, mask_f);
                __m256i nibbles_hi =
                        _mm256_and_si256(_mm256_srli_epi16(c, 4), mask_f);

                __m256i diff_lo = _mm256_sub_epi8(q_lo_0, nibbles_lo);
                __m256i diff_hi = _mm256_sub_epi8(q_hi_0, nibbles_hi);

                diff_lo = _mm256_max_epi8(
                        diff_lo, _mm256_sub_epi8(zero, diff_lo));
                diff_hi = _mm256_max_epi8(
                        diff_hi, _mm256_sub_epi8(zero, diff_hi));

                __m256i sq_lo = _mm256_maddubs_epi16(diff_lo, diff_lo);
                __m256i sq_hi = _mm256_maddubs_epi16(diff_hi, diff_hi);

                __m256i sum_lo = _mm256_madd_epi16(sq_lo, one);
                __m256i sum_hi = _mm256_madd_epi16(sq_hi, one);

                acc = _mm256_add_epi32(acc, sum_lo);
                acc = _mm256_add_epi32(acc, sum_hi);
            };

            process(code_0, acc0);
            process(code_1, acc1);
            process(code_2, acc2);
            process(code_3, acc3);

            i += 64;
        }

        auto reduce = [](const __m256i& acc) -> int32_t {
            __m128i acc_lo = _mm256_castsi256_si128(acc);
            __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
            acc_lo = _mm_add_epi32(acc_lo, acc_hi);
            acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
            acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
            return _mm_cvtsi128_si32(acc_lo);
        };

        dis0 = static_cast<float>(reduce(acc0));
        dis1 = static_cast<float>(reduce(acc1));
        dis2 = static_cast<float>(reduce(acc2));
        dis3 = static_cast<float>(reduce(acc3));

        // Scalar tail.
        for (; i < d; i++) {
            uint8_t nibble_lo = q_lo[i / 2];
            uint8_t nibble_hi = q_hi[i / 2];

            auto process_scalar = [&](const uint8_t* code, float& dis) {
                uint8_t c = code[i / 2];
                uint8_t nibble = (i % 2 == 0)
                        ? static_cast<uint8_t>(c & 0x0F)
                        : static_cast<uint8_t>(c >> 4);
                int q_code = (i % 2 == 0) ? nibble_lo : nibble_hi;
                int diff = q_code - int(nibble);
                dis += static_cast<float>(diff * diff);
            };

            process_scalar(code_0, dis0);
            process_scalar(code_1, dis1);
            process_scalar(code_2, dis2);
            process_scalar(code_3, dis3);
        }

        dis0 *= final_scale_sq;
        dis1 *= final_scale_sq;
        dis2 *= final_scale_sq;
        dis3 *= final_scale_sq;
    }
};

} // namespace scalar_quantizer
} // namespace faiss

#include "../../../impl/scalar_quantizer/sq-avx2.cpp"

#endif // COMPILE_SIMD_AVX2
