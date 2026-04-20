/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/******************************************************************************
 * Knowhere-local prelude over baseline sq-neon.cpp.
 *
 * See sq-avx512-fastpath.cpp for the full design note on how this pattern
 * works. This file ports the NEON variant of the fork's
 * DistanceComputerSQ4UByte for QT_4bit_uniform + L2.
 *****************************************************************************/

#ifdef COMPILE_SIMD_ARM_NEON

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

#include <arm_neon.h>
#include <cmath>
#include <vector>

namespace faiss {

namespace scalar_quantizer {

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec4bit<SIMDLevel::ARM_NEON>,
                QuantizerTemplateScaling::UNIFORM,
                SIMDLevel::ARM_NEON>,
        SimilarityL2<SIMDLevel::ARM_NEON>,
        SIMDLevel::ARM_NEON> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::ARM_NEON>;

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
              // Over-allocate by 16 bytes for safe 128-bit vld1q_u8 past
              // the logical length (readers ignore out-of-range lanes).
              q_lo((d_in + 1) / 2 + 16, 0),
              q_hi((d_in + 1) / 2 + 16, 0) {
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

        uint32x4_t acc = vdupq_n_u32(0);
        const uint8x16_t mask_f = vdupq_n_u8(0x0F);

        size_t i = 0;
        // 32 dims per iteration (16 bytes of packed 4-bit codes).
        for (; i + 32 <= d; i += 32) {
            uint8x16_t c = vld1q_u8(code + i / 2);

            uint8x16_t nibbles_lo = vandq_u8(c, mask_f);
            uint8x16_t nibbles_hi = vandq_u8(vshrq_n_u8(c, 4), mask_f);

            uint8x16_t q_lo_vec = vld1q_u8(q_lo_ptr + i / 2);
            uint8x16_t q_hi_vec = vld1q_u8(q_hi_ptr + i / 2);

            uint8x16_t diff_lo = vabdq_u8(q_lo_vec, nibbles_lo);
            uint8x16_t diff_hi = vabdq_u8(q_hi_vec, nibbles_hi);

            // Widen + square — each byte in [0, 15] so squared fits in u16.
            uint16x8_t sq_lo_1 =
                    vmull_u8(vget_low_u8(diff_lo), vget_low_u8(diff_lo));
            uint16x8_t sq_lo_2 =
                    vmull_u8(vget_high_u8(diff_lo), vget_high_u8(diff_lo));
            uint16x8_t sq_hi_1 =
                    vmull_u8(vget_low_u8(diff_hi), vget_low_u8(diff_hi));
            uint16x8_t sq_hi_2 =
                    vmull_u8(vget_high_u8(diff_hi), vget_high_u8(diff_hi));

            acc = vpadalq_u16(acc, sq_lo_1);
            acc = vpadalq_u16(acc, sq_lo_2);
            acc = vpadalq_u16(acc, sq_hi_1);
            acc = vpadalq_u16(acc, sq_hi_2);
        }

        uint32_t sum = vaddvq_u32(acc);

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

    /// Batch-4: 32 dims per outer iter with four parallel u32 accumulators,
    /// amortizing the q_lo / q_hi load across four input codes. Ported
    /// verbatim from the fork's NEON DistanceComputerSQ4UByte_neon.
    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        uint32x4_t acc0 = vdupq_n_u32(0);
        uint32x4_t acc1 = vdupq_n_u32(0);
        uint32x4_t acc2 = vdupq_n_u32(0);
        uint32x4_t acc3 = vdupq_n_u32(0);

        const uint8x16_t mask_f = vdupq_n_u8(0x0F);
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        size_t i = 0;
        for (; i + 32 <= d; i += 32) {
            uint8x16_t q_lo_vec = vld1q_u8(q_lo_ptr + i / 2);
            uint8x16_t q_hi_vec = vld1q_u8(q_hi_ptr + i / 2);

            auto process = [&](const uint8_t* code, uint32x4_t& acc) {
                uint8x16_t c = vld1q_u8(code + i / 2);
                uint8x16_t nibbles_lo = vandq_u8(c, mask_f);
                uint8x16_t nibbles_hi = vandq_u8(vshrq_n_u8(c, 4), mask_f);

                uint8x16_t diff_lo = vabdq_u8(q_lo_vec, nibbles_lo);
                uint8x16_t diff_hi = vabdq_u8(q_hi_vec, nibbles_hi);

                uint16x8_t sq_lo_1 =
                        vmull_u8(vget_low_u8(diff_lo), vget_low_u8(diff_lo));
                uint16x8_t sq_lo_2 =
                        vmull_u8(vget_high_u8(diff_lo), vget_high_u8(diff_lo));
                uint16x8_t sq_hi_1 =
                        vmull_u8(vget_low_u8(diff_hi), vget_low_u8(diff_hi));
                uint16x8_t sq_hi_2 =
                        vmull_u8(vget_high_u8(diff_hi), vget_high_u8(diff_hi));

                acc = vpadalq_u16(acc, sq_lo_1);
                acc = vpadalq_u16(acc, sq_lo_2);
                acc = vpadalq_u16(acc, sq_hi_1);
                acc = vpadalq_u16(acc, sq_hi_2);
            };

            process(code_0, acc0);
            process(code_1, acc1);
            process(code_2, acc2);
            process(code_3, acc3);
        }

        dis0 = static_cast<float>(vaddvq_u32(acc0));
        dis1 = static_cast<float>(vaddvq_u32(acc1));
        dis2 = static_cast<float>(vaddvq_u32(acc2));
        dis3 = static_cast<float>(vaddvq_u32(acc3));

        // Scalar tail.
        if (i < d) {
            size_t rem = d - i;
            for (size_t j = 0; j < rem; j++) {
                size_t idx = i + j;
                uint8_t nibble_lo = q_lo[idx / 2];
                uint8_t nibble_hi = q_hi[idx / 2];

                auto process_scalar = [&](const uint8_t* code, float& dis) {
                    uint8_t c = code[idx / 2];
                    uint8_t nibble = (idx % 2 == 0)
                            ? static_cast<uint8_t>(c & 0x0F)
                            : static_cast<uint8_t>(c >> 4);
                    int q_code = (idx % 2 == 0) ? nibble_lo : nibble_hi;
                    int diff = q_code - int(nibble);
                    dis += static_cast<float>(diff * diff);
                };

                process_scalar(code_0, dis0);
                process_scalar(code_1, dis1);
                process_scalar(code_2, dis2);
                process_scalar(code_3, dis3);
            }
        }

        dis0 *= final_scale_sq;
        dis1 *= final_scale_sq;
        dis2 *= final_scale_sq;
        dis3 *= final_scale_sq;
    }
};

} // namespace scalar_quantizer
} // namespace faiss

#include "../../../impl/scalar_quantizer/sq-neon.cpp"

#endif // COMPILE_SIMD_ARM_NEON
