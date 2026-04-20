/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/******************************************************************************
 * Knowhere-local prelude over baseline sq-avx512.cpp.
 *
 * What this file does, and why:
 *   - Declares a FULL template specialization of
 *     faiss::scalar_quantizer::DCTemplate<Q, SimilarityL2<AVX512>, AVX512>
 *     for Q = QuantizerTemplate<Codec4bit<AVX512>, UNIFORM, AVX512>.
 *   - Then textually `#include`s the baseline
 *     thirdparty/faiss/faiss/impl/scalar_quantizer/sq-avx512.cpp.
 *   - Knowhere's CMake swaps this file in place of that baseline .cpp when
 *     building the faiss_avx512 object library.
 *
 * Effect: baseline's sq-dispatch.h dispatcher (included at the bottom of the
 * baseline .cpp we pull in below) instantiates DCTemplate<...> for the
 * template args the dispatcher writes for QT_4bit_uniform. C++ template
 * lookup picks our full specialization because it is strictly more
 * specialized than baseline's partial specialization for the AVX512 level.
 * Non-matching combinations (other qtypes, IP metric) still resolve to
 * baseline's partial specialization — nothing else changes.
 *
 * IMPORTANT constraint: the full specialization body must NOT contain a
 * member of type Quantizer<...>. Inside baseline's sq-avx512.cpp, the AVX512
 * partial specializations of Codec4bit / QuantizerTemplate are declared
 * BELOW the point at which we include that file here. At the point of our
 * full specialization, those types are incomplete (primary template only),
 * so we cannot have a member of that type. Workaround: read `trained[0]`
 * and `trained[1]` directly in the constructor.
 *
 * For a detailed design note see the project plan's §1.2.A.
 *****************************************************************************/

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

#include <immintrin.h>
#include <cmath>
#include <vector>

namespace faiss {

namespace scalar_quantizer {

/*************************************************************************
 * QT_4bit_uniform + L2 fast path, AVX512.
 *
 * Math recap: for UNIFORM 4-bit scaling,
 *     recon(c) = vmin + vdiff * (c + 0.5) / 15 = final_scale * c + bias
 *     final_scale = vdiff / 15
 *     L2(recon(q), recon(c)) = final_scale^2 * (q_c - c_c)^2
 *
 * We pre-nibble the query floats into q_lo / q_hi (even / odd lanes) once
 * at set_query time and then compute everything in the int domain, paying
 * one float multiply at the end.
 ************************************************************************/

template <>
struct DCTemplate<
        QuantizerTemplate<
                Codec4bit<SIMDLevel::AVX512>,
                QuantizerTemplateScaling::UNIFORM,
                SIMDLevel::AVX512>,
        SimilarityL2<SIMDLevel::AVX512>,
        SIMDLevel::AVX512> : SQDistanceComputer {
    using Sim = SimilarityL2<SIMDLevel::AVX512>;

    size_t d;
    float vmin;
    float vdiff;
    float final_scale_sq;
    std::vector<uint8_t> q_lo;
    std::vector<uint8_t> q_hi;
    bool has_vnni;

    DCTemplate(size_t d_in, const std::vector<float>& trained)
            : d(d_in),
              vmin(trained[0]),
              vdiff(trained[1]),
              // Over-allocate by 64 bytes so full 512-bit loads past the
              // logical length are safe (readers mask off unused lanes).
              q_lo((d_in + 1) / 2 + 64, 0),
              q_hi((d_in + 1) / 2 + 64, 0),
              has_vnni(__builtin_cpu_supports("avx512vnni")) {
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
        __m512i acc = _mm512_setzero_si512();
        const __m512i mask_f = _mm512_set1_epi8(0xF);
        const __m512i one = _mm512_set1_epi16(1);
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        size_t i = 0;
        // 128 dims per iteration (64 bytes of packed 4-bit codes).
        for (; i + 128 <= d; i += 128) {
            __m512i c512 = _mm512_loadu_si512(
                    reinterpret_cast<const __m512i*>(code + i / 2));

            __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
            __m512i nibbles_hi =
                    _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);

            __m512i q_lo_vec = _mm512_loadu_si512(q_lo_ptr + i / 2);
            __m512i q_hi_vec = _mm512_loadu_si512(q_hi_ptr + i / 2);

            __m512i diff_lo = _mm512_sub_epi8(q_lo_vec, nibbles_lo);
            __m512i diff_hi = _mm512_sub_epi8(q_hi_vec, nibbles_hi);

            diff_lo = _mm512_abs_epi8(diff_lo);
            diff_hi = _mm512_abs_epi8(diff_hi);

            __m512i sq_lo = _mm512_maddubs_epi16(diff_lo, diff_lo);
            __m512i sq_hi = _mm512_maddubs_epi16(diff_hi, diff_hi);

            __m512i sq_sum = _mm512_add_epi16(sq_lo, sq_hi);
            __m512i sum_32 = _mm512_madd_epi16(sq_sum, one);

            acc = _mm512_add_epi32(acc, sum_32);
        }

        // Tail. q_lo / q_hi are over-allocated so masked loads past the
        // logical length read zeros; code is also loaded with mask_even and
        // nibbles_hi is masked separately so odd-lane overread is zeroed.
        if (i < d) {
            size_t rem = d - i;
            uint64_t mask_even = (rem + 1) / 2 >= 64
                    ? ~0ULL
                    : (1ULL << ((rem + 1) / 2)) - 1;
            uint64_t mask_odd =
                    rem / 2 >= 64 ? ~0ULL : (1ULL << (rem / 2)) - 1;

            __m512i c512 = _mm512_maskz_loadu_epi8(mask_even, code + i / 2);

            __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
            __m512i nibbles_hi =
                    _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);

            __m512i q_lo_vec =
                    _mm512_maskz_loadu_epi8(mask_even, q_lo_ptr + i / 2);
            __m512i q_hi_vec =
                    _mm512_maskz_loadu_epi8(mask_odd, q_hi_ptr + i / 2);

            __m512i mask_odd_vec = _mm512_movm_epi8(mask_odd);
            nibbles_hi = _mm512_and_si512(nibbles_hi, mask_odd_vec);

            __m512i diff_lo = _mm512_sub_epi8(q_lo_vec, nibbles_lo);
            __m512i diff_hi = _mm512_sub_epi8(q_hi_vec, nibbles_hi);

            diff_lo = _mm512_abs_epi8(diff_lo);
            diff_hi = _mm512_abs_epi8(diff_hi);

            __m512i sq_lo = _mm512_maddubs_epi16(diff_lo, diff_lo);
            __m512i sq_hi = _mm512_maddubs_epi16(diff_hi, diff_hi);

            __m512i sq_sum = _mm512_add_epi16(sq_lo, sq_hi);
            __m512i sum_32 = _mm512_madd_epi16(sq_sum, one);

            acc = _mm512_add_epi32(acc, sum_32);
        }

        const int32_t sum = _mm512_reduce_add_epi32(acc);
        return static_cast<float>(sum) * final_scale_sq;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        // Not on the critical query path; scalar version suffices.
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

    /// Batch-4 entry point: dispatches to VNNI or non-VNNI path based on
    /// runtime CPU capability. Both paths process 256 dims per outer loop
    /// iteration by amortizing two q_lo / q_hi chunks across the four
    /// input codes.
    void query_to_codes_batch_4(
            const uint8_t* code_0,
            const uint8_t* code_1,
            const uint8_t* code_2,
            const uint8_t* code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const final {
        if (has_vnni) {
            query_to_codes_batch_4_vnni(
                    code_0, code_1, code_2, code_3, dis0, dis1, dis2, dis3);
        } else {
            query_to_codes_batch_4_avx512(
                    code_0, code_1, code_2, code_3, dis0, dis1, dis2, dis3);
        }
    }

    /// VNNI path: uses _mm512_dpbusd_epi32 to fuse square-and-accumulate.
    /// Still valid because for 4-bit codes the differences are in [-15, 15]
    /// and |diff|^2 fits in u8 × u8 → i32 without overflow.
    __attribute__((target("avx512vnni"))) void query_to_codes_batch_4_vnni(
            const uint8_t* __restrict code_0,
            const uint8_t* __restrict code_1,
            const uint8_t* __restrict code_2,
            const uint8_t* __restrict code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();

        const __m512i mask_f = _mm512_set1_epi8(0x0F);
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        size_t i = 0;
        // 256 dims per iteration — two 128-dim chunks sharing two q loads.
        for (; i + 256 <= d; i += 256) {
            __m512i q_lo_0 = _mm512_loadu_si512(q_lo_ptr + i / 2);
            __m512i q_hi_0 = _mm512_loadu_si512(q_hi_ptr + i / 2);
            __m512i q_lo_1 = _mm512_loadu_si512(q_lo_ptr + i / 2 + 64);
            __m512i q_hi_1 = _mm512_loadu_si512(q_hi_ptr + i / 2 + 64);

            auto process_chunk = [&](const uint8_t* code,
                                     __m512i& acc,
                                     __m512i q_lo_v,
                                     __m512i q_hi_v,
                                     int offset)
                    __attribute__((target("avx512vnni"))) {
                __m512i c512 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(code + i / 2 + offset));
                __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
                __m512i nibbles_hi =
                        _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);

                __m512i diff_lo = _mm512_sub_epi8(q_lo_v, nibbles_lo);
                __m512i diff_hi = _mm512_sub_epi8(q_hi_v, nibbles_hi);

                diff_lo = _mm512_abs_epi8(diff_lo);
                diff_hi = _mm512_abs_epi8(diff_hi);

                acc = _mm512_dpbusd_epi32(acc, diff_lo, diff_lo);
                acc = _mm512_dpbusd_epi32(acc, diff_hi, diff_hi);
            };

            process_chunk(code_0, acc0, q_lo_0, q_hi_0, 0);
            process_chunk(code_1, acc1, q_lo_0, q_hi_0, 0);
            process_chunk(code_2, acc2, q_lo_0, q_hi_0, 0);
            process_chunk(code_3, acc3, q_lo_0, q_hi_0, 0);

            process_chunk(code_0, acc0, q_lo_1, q_hi_1, 64);
            process_chunk(code_1, acc1, q_lo_1, q_hi_1, 64);
            process_chunk(code_2, acc2, q_lo_1, q_hi_1, 64);
            process_chunk(code_3, acc3, q_lo_1, q_hi_1, 64);
        }

        // 128-dim remainder (one q chunk).
        if (i + 128 <= d) {
            __m512i q_lo_0 = _mm512_loadu_si512(q_lo_ptr + i / 2);
            __m512i q_hi_0 = _mm512_loadu_si512(q_hi_ptr + i / 2);

            auto process_chunk = [&](const uint8_t* code, __m512i& acc)
                    __attribute__((target("avx512vnni"))) {
                __m512i c512 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(code + i / 2));
                __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
                __m512i nibbles_hi =
                        _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);

                __m512i diff_lo = _mm512_sub_epi8(q_lo_0, nibbles_lo);
                __m512i diff_hi = _mm512_sub_epi8(q_hi_0, nibbles_hi);

                diff_lo = _mm512_abs_epi8(diff_lo);
                diff_hi = _mm512_abs_epi8(diff_hi);

                acc = _mm512_dpbusd_epi32(acc, diff_lo, diff_lo);
                acc = _mm512_dpbusd_epi32(acc, diff_hi, diff_hi);
            };

            process_chunk(code_0, acc0);
            process_chunk(code_1, acc1);
            process_chunk(code_2, acc2);
            process_chunk(code_3, acc3);

            i += 128;
        }

        // Sub-128-dim tail with masked loads.
        if (i < d) {
            size_t rem = d - i;
            uint64_t mask_even = (rem + 1) / 2 >= 64
                    ? ~0ULL
                    : (1ULL << ((rem + 1) / 2)) - 1;
            uint64_t mask_odd =
                    rem / 2 >= 64 ? ~0ULL : (1ULL << (rem / 2)) - 1;

            __m512i q_lo_vec =
                    _mm512_maskz_loadu_epi8(mask_even, q_lo_ptr + i / 2);
            __m512i q_hi_vec =
                    _mm512_maskz_loadu_epi8(mask_odd, q_hi_ptr + i / 2);
            __m512i mask_odd_vec = _mm512_movm_epi8(mask_odd);

            auto process = [&](const uint8_t* code, __m512i& acc)
                    __attribute__((target("avx512vnni"))) {
                __m512i c512 =
                        _mm512_maskz_loadu_epi8(mask_even, code + i / 2);
                __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
                __m512i nibbles_hi =
                        _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);
                nibbles_hi = _mm512_and_si512(nibbles_hi, mask_odd_vec);

                __m512i diff_lo = _mm512_sub_epi8(q_lo_vec, nibbles_lo);
                __m512i diff_hi = _mm512_sub_epi8(q_hi_vec, nibbles_hi);

                diff_lo = _mm512_abs_epi8(diff_lo);
                diff_hi = _mm512_abs_epi8(diff_hi);

                acc = _mm512_dpbusd_epi32(acc, diff_lo, diff_lo);
                acc = _mm512_dpbusd_epi32(acc, diff_hi, diff_hi);
            };

            process(code_0, acc0);
            process(code_1, acc1);
            process(code_2, acc2);
            process(code_3, acc3);
        }

        dis0 = static_cast<float>(_mm512_reduce_add_epi32(acc0)) *
                final_scale_sq;
        dis1 = static_cast<float>(_mm512_reduce_add_epi32(acc1)) *
                final_scale_sq;
        dis2 = static_cast<float>(_mm512_reduce_add_epi32(acc2)) *
                final_scale_sq;
        dis3 = static_cast<float>(_mm512_reduce_add_epi32(acc3)) *
                final_scale_sq;
    }

    /// Non-VNNI path: squares via _mm512_maddubs_epi16 (u8×u8 → i16) and
    /// accumulates to i32 with _mm512_madd_epi16. Same 256-dim outer loop.
    void query_to_codes_batch_4_avx512(
            const uint8_t* __restrict code_0,
            const uint8_t* __restrict code_1,
            const uint8_t* __restrict code_2,
            const uint8_t* __restrict code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {
        __m512i acc0 = _mm512_setzero_si512();
        __m512i acc1 = _mm512_setzero_si512();
        __m512i acc2 = _mm512_setzero_si512();
        __m512i acc3 = _mm512_setzero_si512();

        const __m512i mask_f = _mm512_set1_epi8(0x0F);
        const __m512i one = _mm512_set1_epi16(1);
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        size_t i = 0;
        for (; i + 256 <= d; i += 256) {
            __m512i q_lo_0 = _mm512_loadu_si512(q_lo_ptr + i / 2);
            __m512i q_hi_0 = _mm512_loadu_si512(q_hi_ptr + i / 2);
            __m512i q_lo_1 = _mm512_loadu_si512(q_lo_ptr + i / 2 + 64);
            __m512i q_hi_1 = _mm512_loadu_si512(q_hi_ptr + i / 2 + 64);

            auto process_chunk = [&](const uint8_t* code,
                                     __m512i& acc,
                                     __m512i q_lo_v,
                                     __m512i q_hi_v,
                                     int offset) {
                __m512i c512 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(code + i / 2 + offset));
                __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
                __m512i nibbles_hi =
                        _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);

                __m512i diff_lo = _mm512_sub_epi8(q_lo_v, nibbles_lo);
                __m512i diff_hi = _mm512_sub_epi8(q_hi_v, nibbles_hi);

                diff_lo = _mm512_abs_epi8(diff_lo);
                diff_hi = _mm512_abs_epi8(diff_hi);

                __m512i sq_lo = _mm512_maddubs_epi16(diff_lo, diff_lo);
                __m512i sq_hi = _mm512_maddubs_epi16(diff_hi, diff_hi);

                __m512i sum_lo = _mm512_madd_epi16(sq_lo, one);
                __m512i sum_hi = _mm512_madd_epi16(sq_hi, one);

                acc = _mm512_add_epi32(acc, sum_lo);
                acc = _mm512_add_epi32(acc, sum_hi);
            };

            process_chunk(code_0, acc0, q_lo_0, q_hi_0, 0);
            process_chunk(code_1, acc1, q_lo_0, q_hi_0, 0);
            process_chunk(code_2, acc2, q_lo_0, q_hi_0, 0);
            process_chunk(code_3, acc3, q_lo_0, q_hi_0, 0);

            process_chunk(code_0, acc0, q_lo_1, q_hi_1, 64);
            process_chunk(code_1, acc1, q_lo_1, q_hi_1, 64);
            process_chunk(code_2, acc2, q_lo_1, q_hi_1, 64);
            process_chunk(code_3, acc3, q_lo_1, q_hi_1, 64);
        }

        if (i + 128 <= d) {
            __m512i q_lo_0 = _mm512_loadu_si512(q_lo_ptr + i / 2);
            __m512i q_hi_0 = _mm512_loadu_si512(q_hi_ptr + i / 2);

            auto process_chunk = [&](const uint8_t* code, __m512i& acc) {
                __m512i c512 = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(code + i / 2));
                __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
                __m512i nibbles_hi =
                        _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);

                __m512i diff_lo = _mm512_sub_epi8(q_lo_0, nibbles_lo);
                __m512i diff_hi = _mm512_sub_epi8(q_hi_0, nibbles_hi);

                diff_lo = _mm512_abs_epi8(diff_lo);
                diff_hi = _mm512_abs_epi8(diff_hi);

                __m512i sq_lo = _mm512_maddubs_epi16(diff_lo, diff_lo);
                __m512i sq_hi = _mm512_maddubs_epi16(diff_hi, diff_hi);

                __m512i sum_lo = _mm512_madd_epi16(sq_lo, one);
                __m512i sum_hi = _mm512_madd_epi16(sq_hi, one);

                acc = _mm512_add_epi32(acc, sum_lo);
                acc = _mm512_add_epi32(acc, sum_hi);
            };

            process_chunk(code_0, acc0);
            process_chunk(code_1, acc1);
            process_chunk(code_2, acc2);
            process_chunk(code_3, acc3);

            i += 128;
        }

        if (i < d) {
            size_t rem = d - i;
            uint64_t mask_even = (rem + 1) / 2 >= 64
                    ? ~0ULL
                    : (1ULL << ((rem + 1) / 2)) - 1;
            uint64_t mask_odd =
                    rem / 2 >= 64 ? ~0ULL : (1ULL << (rem / 2)) - 1;

            __m512i q_lo_vec =
                    _mm512_maskz_loadu_epi8(mask_even, q_lo_ptr + i / 2);
            __m512i q_hi_vec =
                    _mm512_maskz_loadu_epi8(mask_odd, q_hi_ptr + i / 2);
            __m512i mask_odd_vec = _mm512_movm_epi8(mask_odd);

            auto process = [&](const uint8_t* code, __m512i& acc) {
                __m512i c512 =
                        _mm512_maskz_loadu_epi8(mask_even, code + i / 2);
                __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
                __m512i nibbles_hi =
                        _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);
                nibbles_hi = _mm512_and_si512(nibbles_hi, mask_odd_vec);

                __m512i diff_lo = _mm512_sub_epi8(q_lo_vec, nibbles_lo);
                __m512i diff_hi = _mm512_sub_epi8(q_hi_vec, nibbles_hi);

                diff_lo = _mm512_abs_epi8(diff_lo);
                diff_hi = _mm512_abs_epi8(diff_hi);

                __m512i sq_lo = _mm512_maddubs_epi16(diff_lo, diff_lo);
                __m512i sq_hi = _mm512_maddubs_epi16(diff_hi, diff_hi);

                __m512i sum_lo = _mm512_madd_epi16(sq_lo, one);
                __m512i sum_hi = _mm512_madd_epi16(sq_hi, one);

                acc = _mm512_add_epi32(acc, sum_lo);
                acc = _mm512_add_epi32(acc, sum_hi);
            };

            process(code_0, acc0);
            process(code_1, acc1);
            process(code_2, acc2);
            process(code_3, acc3);
        }

        dis0 = static_cast<float>(_mm512_reduce_add_epi32(acc0)) *
                final_scale_sq;
        dis1 = static_cast<float>(_mm512_reduce_add_epi32(acc1)) *
                final_scale_sq;
        dis2 = static_cast<float>(_mm512_reduce_add_epi32(acc2)) *
                final_scale_sq;
        dis3 = static_cast<float>(_mm512_reduce_add_epi32(acc3)) *
                final_scale_sq;
    }
};

} // namespace scalar_quantizer
} // namespace faiss

// Pull in baseline's sq-avx512.cpp. Its AVX512 partial specializations of
// Codec / QuantizerTemplate / Similarity / DCTemplate, its Similarity
// structs, and its dispatcher instantiation all come online after this
// point. Our full specialization above is already visible, so at the
// instantiation moment inside sq-dispatch.h, C++ template lookup selects
// it over the partial one.
#include "../../../impl/scalar_quantizer/sq-avx512.cpp"

#endif // COMPILE_SIMD_AVX512
