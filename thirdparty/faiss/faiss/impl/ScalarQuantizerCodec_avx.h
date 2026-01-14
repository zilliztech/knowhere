/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ScalarQuantizerCodec.h>
#include <faiss/utils/utils.h>

namespace faiss {

using QuantizerType = ScalarQuantizer::QuantizerType;
using RangeStat = ScalarQuantizer::RangeStat;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;
using SQuantizer = ScalarQuantizer::SQuantizer;

/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

struct Codec8bit_avx : public Codec8bit {
    static FAISS_ALWAYS_INLINE __m256
    decode_8_components(const uint8_t* code, int i) {
        const uint64_t c8 = *(uint64_t*)(code + i);

        const __m128i i8 = _mm_set1_epi64x(c8);
        const __m256i i32 = _mm256_cvtepu8_epi32(i8);
        const __m256 f8 = _mm256_cvtepi32_ps(i32);
        const __m256 half_one_255 = _mm256_set1_ps(0.5f / 255.f);
        const __m256 one_255 = _mm256_set1_ps(1.f / 255.f);
        return _mm256_fmadd_ps(f8, one_255, half_one_255);
    }
};

struct Codec4bit_avx : public Codec4bit {
    static FAISS_ALWAYS_INLINE __m256
    decode_8_components(const uint8_t* code, int i) {
        uint32_t c4 = *(uint32_t*)(code + (i >> 1));
        uint32_t mask = 0x0f0f0f0f;
        uint32_t c4ev = c4 & mask;
        uint32_t c4od = (c4 >> 4) & mask;

        // the 8 lower bytes of c8 contain the values
        __m128i c8 =
                _mm_unpacklo_epi8(_mm_set1_epi32(c4ev), _mm_set1_epi32(c4od));
        __m128i c4lo = _mm_cvtepu8_epi32(c8);
        __m128i c4hi = _mm_cvtepu8_epi32(_mm_srli_si128(c8, 4));
        __m256i i8 = _mm256_castsi128_si256(c4lo);
        i8 = _mm256_insertf128_si256(i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps(i8);
        __m256 half = _mm256_set1_ps(0.5f);
        f8 = _mm256_add_ps(f8, half);
        __m256 one_255 = _mm256_set1_ps(1.f / 15.f);
        return _mm256_mul_ps(f8, one_255);
    }

    static FAISS_ALWAYS_INLINE __m256i
    decode_8_components_int(const uint8_t* code, int i) {
        // Load 4 bytes containing 8 nibbles
        uint32_t c4 = *(uint32_t*)(code + (i >> 1));
        uint32_t mask = 0x0f0f0f0f;
        uint32_t c4ev = c4 & mask;        // Even nibbles
        uint32_t c4od = (c4 >> 4) & mask; // Odd nibbles

        // Interleave even and odd nibbles
        __m128i c8 =
                _mm_unpacklo_epi8(_mm_set1_epi32(c4ev), _mm_set1_epi32(c4od));

        // Convert to 8x32-bit integers
        __m128i c4lo = _mm_cvtepu8_epi32(c8);
        __m128i c4hi = _mm_cvtepu8_epi32(_mm_srli_si128(c8, 4));
        __m256i result = _mm256_castsi128_si256(c4lo);
        result = _mm256_insertf128_si256(result, c4hi, 1);

        return result;
    }
};

struct Codec6bit_avx : public Codec6bit {
    /* Load 6 bytes that represent 8 6-bit values, return them as a
     * 8*32 bit vector register */
    static FAISS_ALWAYS_INLINE __m256i load6(const uint16_t* code16) {
        const __m128i perm = _mm_set_epi8(
                -1, 5, 5, 4, 4, 3, -1, 3, -1, 2, 2, 1, 1, 0, -1, 0);
        const __m256i shifts = _mm256_set_epi32(2, 4, 6, 0, 2, 4, 6, 0);

        // load 6 bytes
        __m128i c1 =
                _mm_set_epi16(0, 0, 0, 0, 0, code16[2], code16[1], code16[0]);

        // put in 8 * 32 bits
        __m128i c2 = _mm_shuffle_epi8(c1, perm);
        __m256i c3 = _mm256_cvtepi16_epi32(c2);

        // shift and mask out useless bits
        __m256i c4 = _mm256_srlv_epi32(c3, shifts);
        __m256i c5 = _mm256_and_si256(_mm256_set1_epi32(63), c4);
        return c5;
    }

    static FAISS_ALWAYS_INLINE __m256
    decode_8_components(const uint8_t* code, int i) {
        // // Faster code for Intel CPUs or AMD Zen3+, just keeping it here
        // // for the reference, maybe, it becomes used oned day.
        // const uint16_t* data16 = (const uint16_t*)(code + (i >> 2) * 3);
        // const uint32_t* data32 = (const uint32_t*)data16;
        // const uint64_t val = *data32 + ((uint64_t)data16[2] << 32);
        // const uint64_t vext = _pdep_u64(val, 0x3F3F3F3F3F3F3F3FULL);
        // const __m128i i8 = _mm_set1_epi64x(vext);
        // const __m256i i32 = _mm256_cvtepi8_epi32(i8);
        // const __m256 f8 = _mm256_cvtepi32_ps(i32);
        // const __m256 half_one_255 = _mm256_set1_ps(0.5f / 63.f);
        // const __m256 one_255 = _mm256_set1_ps(1.f / 63.f);
        // return _mm256_fmadd_ps(f8, one_255, half_one_255);

        __m256i i8 = load6((const uint16_t*)(code + (i >> 2) * 3));
        __m256 f8 = _mm256_cvtepi32_ps(i8);
        // this could also be done with bit manipulations but it is
        // not obviously faster
        const __m256 half_one_255 = _mm256_set1_ps(0.5f / 63.f);
        const __m256 one_255 = _mm256_set1_ps(1.f / 63.f);
        return _mm256_fmadd_ps(f8, one_255, half_one_255);
    }
};

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

template <class Codec, QuantizerTemplateScaling SCALING, int SIMD>
struct QuantizerTemplate_avx {};

template <class Codec>
struct QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::UNIFORM, 1>
        : public QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1> {
    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::UNIFORM, 8>
        : public QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1> {
    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1>(d, trained) {}

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i);
        return _mm256_fmadd_ps(
                xi, _mm256_set1_ps(this->vdiff), _mm256_set1_ps(this->vmin));
    }
};

template <class Codec>
struct QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1>
        : public QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1> {
    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::NON_UNIFORM, 8>
        : public QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1> {
    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1>(d, trained) {}

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i);
        return _mm256_fmadd_ps(
                xi,
                _mm256_loadu_ps(this->vdiff + i),
                _mm256_loadu_ps(this->vmin + i));
    }
};

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerFP16_avx {};

template <>
struct QuantizerFP16_avx<1> : public QuantizerFP16<1> {
    QuantizerFP16_avx(size_t d, const std::vector<float>& unused)
            : QuantizerFP16<1>(d, unused) {}
};

template <>
struct QuantizerFP16_avx<8> : public QuantizerFP16<1> {
    QuantizerFP16_avx(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i codei = _mm_loadu_si128((const __m128i*)(code + 2 * i));
        return _mm256_cvtph_ps(codei);
    }
};

/*******************************************************************
 * BF16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerBF16_avx {};

template <>
struct QuantizerBF16_avx<1> : public QuantizerBF16<1> {
    QuantizerBF16_avx(size_t d, const std::vector<float>& unused)
            : QuantizerBF16<1>(d, unused) {}
};

template <>
struct QuantizerBF16_avx<8> : public QuantizerBF16<1> {
    QuantizerBF16_avx(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i code_128i = _mm_loadu_si128((const __m128i*)(code + 2 * i));
        __m256i code_256i = _mm256_cvtepu16_epi32(code_128i);
        code_256i = _mm256_slli_epi32(code_256i, 16);
        return _mm256_castsi256_ps(code_256i);
    }
};

/*******************************************************************
 * Specialized QuantizerTemplate for SQ4U
 *******************************************************************/

template <>
struct QuantizerTemplate_avx<
        Codec4bit_avx,
        QuantizerTemplateScaling::UNIFORM,
        8>
        : public QuantizerTemplate<
                  Codec4bit_avx,
                  QuantizerTemplateScaling::UNIFORM,
                  1> {
    float final_scale;
    float final_bias;

    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec4bit_avx,
                      QuantizerTemplateScaling::UNIFORM,
                      1>(d, trained) {
        final_scale = this->vdiff / 15.0f;
        final_bias = this->vmin + this->vdiff * 0.5f / 15.0f;
    }

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m256i nibbles = Codec4bit_avx::decode_8_components_int(code, i);
        __m256 nibbles_f = _mm256_cvtepi32_ps(nibbles);

        return _mm256_fmadd_ps(
                nibbles_f,
                _mm256_set1_ps(final_scale),
                _mm256_set1_ps(final_bias));
    }
};

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirect_avx {};

template <>
struct Quantizer8bitDirect_avx<1> : public Quantizer8bitDirect<1> {
    Quantizer8bitDirect_avx(size_t d, const std::vector<float>& unused)
            : Quantizer8bitDirect(d, unused) {}
};

template <>
struct Quantizer8bitDirect_avx<8> : public Quantizer8bitDirect<1> {
    Quantizer8bitDirect_avx(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<1>(d, trained) {}

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32(x8);              // 8 * int32
        return _mm256_cvtepi32_ps(y8);                      // 8 * float32
    }
};

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirectSigned_avx {};

template <>
struct Quantizer8bitDirectSigned_avx<1> : public Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned_avx(size_t d, const std::vector<float>& unused)
            : Quantizer8bitDirectSigned(d, unused) {}
};

template <>
struct Quantizer8bitDirectSigned_avx<8> : public Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned_avx(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<1>(d, trained) {}

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32(x8);              // 8 * int32
        __m256i c8 = _mm256_set1_epi32(128);
        __m256i z8 = _mm256_sub_epi32(y8, c8); // subtract 128 from all lanes
        return _mm256_cvtepi32_ps(z8);         // 8 * float32
    }
};

template <int SIMDWIDTH>
SQuantizer* select_quantizer_1_avx(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case QuantizerType::QT_8bit:
            return new QuantizerTemplate_avx<Codec8bit_avx, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_6bit:
            return new QuantizerTemplate_avx<Codec6bit_avx, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_4bit:
            return new QuantizerTemplate_avx<Codec4bit_avx, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_8bit_uniform:
            return new QuantizerTemplate_avx<Codec8bit_avx, QuantizerTemplateScaling::UNIFORM, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_4bit_uniform:
            return new QuantizerTemplate_avx<Codec4bit_avx, QuantizerTemplateScaling::UNIFORM, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_fp16:
            return new QuantizerFP16_avx<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_bf16:
            return new QuantizerBF16_avx<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct:
            return new Quantizer8bitDirect_avx<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned_avx<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_1bit_direct:
            // todo: add more SIMDWIDTH support for avx if needed
            return new Quantizer1bitDirect(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
}

/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object. The data fields just encapsulate
 * an accumulator.
 */

template <int SIMDWIDTH>
struct SimilarityL2_avx {};

template <>
struct SimilarityL2_avx<1> : public SimilarityL2<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    explicit SimilarityL2_avx(const float* y) : SimilarityL2<1>(y) {}
};

template <>
struct SimilarityL2_avx<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2_avx(const float* y) : y(y) {}
    __m256 accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(__m256 x) {
        __m256 yiv = _mm256_loadu_ps(yi);
        yi += 8;
        __m256 tmp = _mm256_sub_ps(yiv, x);
        accu8 = _mm256_fmadd_ps(tmp, tmp, accu8);
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(__m256 x, __m256 y_2) {
        __m256 tmp = _mm256_sub_ps(y_2, x);
        accu8 = _mm256_fmadd_ps(tmp, tmp, accu8);
    }

    FAISS_ALWAYS_INLINE float result_8() {
        const __m128 sum = _mm_add_ps(
                _mm256_castps256_ps128(accu8), _mm256_extractf128_ps(accu8, 1));
        const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
        const __m128 v1 = _mm_add_ps(sum, v0);
        __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
        const __m128 v3 = _mm_add_ps(v1, v2);
        return _mm_cvtss_f32(v3);
    }
};

template <int SIMDWIDTH>
struct SimilarityIP_avx {};

template <>
struct SimilarityIP_avx<1> : public SimilarityIP<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    explicit SimilarityIP_avx(const float* y) : SimilarityIP<1>(y) {}
};

template <>
struct SimilarityIP_avx<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP_avx(const float* y) : y(y) {}

    __m256 accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(__m256 x) {
        __m256 yiv = _mm256_loadu_ps(yi);
        yi += 8;
        accu8 = _mm256_fmadd_ps(yiv, x, accu8);
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(__m256 x1, __m256 x2) {
        accu8 = _mm256_fmadd_ps(x1, x2, accu8);
    }

    FAISS_ALWAYS_INLINE float result_8() {
        const __m128 sum = _mm_add_ps(
                _mm256_castps256_ps128(accu8), _mm256_extractf128_ps(accu8, 1));
        const __m128 v0 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(0, 0, 3, 2));
        const __m128 v1 = _mm_add_ps(sum, v0);
        __m128 v2 = _mm_shuffle_ps(v1, v1, _MM_SHUFFLE(0, 0, 0, 1));
        const __m128 v3 = _mm_add_ps(v1, v2);
        return _mm_cvtss_f32(v3);
    }
};

/*******************************************************************
 * SQ4U specialized distance computer (AVX2 version)
 *******************************************************************/

template <class Similarity>
struct DistanceComputerSQ4UByte_avx : SQDistanceComputer {
    using Quantizer = QuantizerTemplate_avx<
            Codec4bit_avx,
            QuantizerTemplateScaling::UNIFORM,
            8>;
    using Sim = Similarity;

    Quantizer quant;
    std::vector<uint8_t> q_lo;
    std::vector<uint8_t> q_hi;
    float final_scale_sq;

    DistanceComputerSQ4UByte_avx(size_t d, const std::vector<float>& trained)
            : quant(d, trained),
              q_lo((d + 1) / 2 + 32, 0),
              q_hi((d + 1) / 2 + 32, 0) {
        final_scale_sq = quant.final_scale * quant.final_scale;
    }

    void set_query(const float* x) final {
        float inv_scale = 1.0f / quant.final_scale;
        float offset = quant.vmin;

        for (size_t i = 0; i < quant.d; i++) {
            float val = (x[i] - offset) * inv_scale;
            int code = (int)std::floor(val);
            if (code < 0)
                code = 0;
            if (code > 15)
                code = 15;

            if (i % 2 == 0) {
                q_lo[i / 2] = (uint8_t)code;
            } else {
                q_hi[i / 2] = (uint8_t)code;
            }
        }
    }

    // Only computes L2 distance
    float compute_distance(const float* x, const uint8_t* code) const {
        return compute_distance_l2(code);
    }

    float compute_distance_l2(const uint8_t* code) const {
        const size_t d = quant.d;
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        __m256i acc = _mm256_setzero_si256();
        const __m256i mask_f = _mm256_set1_epi8(0xF);
        const __m256i one = _mm256_set1_epi16(1);

        size_t i = 0;
        // Process 64 dimensions per iteration (32 bytes = 64 nibbles)
        for (; i + 64 <= d; i += 64) {
            __m256i c256 = _mm256_loadu_si256((const __m256i*)(code + i / 2));

            __m256i nibbles_lo = _mm256_and_si256(c256, mask_f);
            __m256i nibbles_hi =
                    _mm256_and_si256(_mm256_srli_epi16(c256, 4), mask_f);

            __m256i q_lo_vec =
                    _mm256_loadu_si256((const __m256i*)(q_lo_ptr + i / 2));
            __m256i q_hi_vec =
                    _mm256_loadu_si256((const __m256i*)(q_hi_ptr + i / 2));

            // Compute absolute differences
            __m256i diff_lo = _mm256_sub_epi8(q_lo_vec, nibbles_lo);
            __m256i diff_hi = _mm256_sub_epi8(q_hi_vec, nibbles_hi);

            // AVX2 doesn't have _mm256_abs_epi8, so we use max(x, -x)
            diff_lo = _mm256_max_epi8(
                    diff_lo, _mm256_sub_epi8(_mm256_setzero_si256(), diff_lo));
            diff_hi = _mm256_max_epi8(
                    diff_hi, _mm256_sub_epi8(_mm256_setzero_si256(), diff_hi));

            // Square using maddubs: treats input as unsigned bytes
            __m256i sq_lo = _mm256_maddubs_epi16(diff_lo, diff_lo);
            __m256i sq_hi = _mm256_maddubs_epi16(diff_hi, diff_hi);

            // Accumulate to 32-bit
            __m256i sum_lo = _mm256_madd_epi16(sq_lo, one);
            __m256i sum_hi = _mm256_madd_epi16(sq_hi, one);

            acc = _mm256_add_epi32(acc, sum_lo);
            acc = _mm256_add_epi32(acc, sum_hi);
        }

        // Horizontal reduction of acc
        __m128i acc_lo = _mm256_castsi256_si128(acc);
        __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
        acc_lo = _mm_add_epi32(acc_lo, acc_hi);
        acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
        acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
        int32_t sum = _mm_cvtsi128_si32(acc_lo);

        // Handle remaining dimensions scalar
        for (; i < d; i++) {
            uint8_t c = code[i / 2];
            uint8_t nibble;
            if (i % 2 == 0) {
                nibble = c & 0xF;
            } else {
                nibble = (c >> 4) & 0xF;
            }

            int diff;
            if (i % 2 == 0) {
                diff = (int)q_lo[i / 2] - (int)nibble;
            } else {
                diff = (int)q_hi[i / 2] - (int)nibble;
            }
            sum += diff * diff;
        }

        return sum * final_scale_sq;
    }

    float compute_code_distance_l2(const uint8_t* code1, const uint8_t* code2)
            const {
        const size_t d = quant.d;
        __m256i acc = _mm256_setzero_si256();
        const __m256i mask_f = _mm256_set1_epi8(0xF);
        const __m256i one = _mm256_set1_epi16(1);

        size_t i = 0;
        for (; i + 64 <= d; i += 64) {
            __m256i c1_256 =
                    _mm256_loadu_si256((const __m256i*)(code1 + i / 2));
            __m256i c2_256 =
                    _mm256_loadu_si256((const __m256i*)(code2 + i / 2));

            __m256i c1_nibbles_lo = _mm256_and_si256(c1_256, mask_f);
            __m256i c1_nibbles_hi =
                    _mm256_and_si256(_mm256_srli_epi16(c1_256, 4), mask_f);

            __m256i c2_nibbles_lo = _mm256_and_si256(c2_256, mask_f);
            __m256i c2_nibbles_hi =
                    _mm256_and_si256(_mm256_srli_epi16(c2_256, 4), mask_f);

            __m256i diff_lo = _mm256_sub_epi8(c1_nibbles_lo, c2_nibbles_lo);
            __m256i diff_hi = _mm256_sub_epi8(c1_nibbles_hi, c2_nibbles_hi);

            diff_lo = _mm256_max_epi8(
                    diff_lo, _mm256_sub_epi8(_mm256_setzero_si256(), diff_lo));
            diff_hi = _mm256_max_epi8(
                    diff_hi, _mm256_sub_epi8(_mm256_setzero_si256(), diff_hi));

            __m256i sq_lo = _mm256_maddubs_epi16(diff_lo, diff_lo);
            __m256i sq_hi = _mm256_maddubs_epi16(diff_hi, diff_hi);

            __m256i sum_lo = _mm256_madd_epi16(sq_lo, one);
            __m256i sum_hi = _mm256_madd_epi16(sq_hi, one);

            acc = _mm256_add_epi32(acc, sum_lo);
            acc = _mm256_add_epi32(acc, sum_hi);
        }

        __m128i acc_lo = _mm256_castsi256_si128(acc);
        __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
        acc_lo = _mm_add_epi32(acc_lo, acc_hi);
        acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
        acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
        int32_t sum = _mm_cvtsi128_si32(acc_lo);

        for (; i < d; i++) {
            uint8_t c1 = code1[i / 2];
            uint8_t c2 = code2[i / 2];
            uint8_t n1, n2;
            if (i % 2 == 0) {
                n1 = c1 & 0xF;
                n2 = c2 & 0xF;
            } else {
                n1 = (c1 >> 4) & 0xF;
                n2 = (c2 >> 4) & 0xF;
            }
            int diff = (int)n1 - (int)n2;
            sum += diff * diff;
        }

        return sum * final_scale_sq;
    }

    float operator()(idx_t i) final {
        return compute_distance(nullptr, codes + i * code_size);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance_l2(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_distance(nullptr, code);
    }

    void query_to_codes_batch_4(
            const uint8_t* __restrict code_0,
            const uint8_t* __restrict code_1,
            const uint8_t* __restrict code_2,
            const uint8_t* __restrict code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {
        const size_t d = quant.d;
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = _mm256_setzero_si256();
        __m256i acc2 = _mm256_setzero_si256();
        __m256i acc3 = _mm256_setzero_si256();

        const __m256i mask_f = _mm256_set1_epi8(0xF);
        const __m256i one = _mm256_set1_epi16(1);
        const __m256i zero = _mm256_setzero_si256();

        size_t i = 0;
        // Process 128 dimensions per outer iteration
        for (; i + 128 <= d; i += 128) {
            // Chunk 0: first 64 dimensions
            __m256i q_lo_0 =
                    _mm256_loadu_si256((const __m256i*)(q_lo_ptr + i / 2));
            __m256i q_hi_0 =
                    _mm256_loadu_si256((const __m256i*)(q_hi_ptr + i / 2));

            auto process_chunk_64 = [&](const uint8_t* code,
                                        __m256i& acc,
                                        __m256i q_lo,
                                        __m256i q_hi,
                                        int offset) {
                __m256i c = _mm256_loadu_si256(
                        (const __m256i*)(code + i / 2 + offset));
                __m256i nibbles_lo = _mm256_and_si256(c, mask_f);
                __m256i nibbles_hi =
                        _mm256_and_si256(_mm256_srli_epi16(c, 4), mask_f);

                __m256i diff_lo = _mm256_sub_epi8(q_lo, nibbles_lo);
                __m256i diff_hi = _mm256_sub_epi8(q_hi, nibbles_hi);

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

            // Chunk 1: next 64 dimensions
            __m256i q_lo_1 =
                    _mm256_loadu_si256((const __m256i*)(q_lo_ptr + i / 2 + 32));
            __m256i q_hi_1 =
                    _mm256_loadu_si256((const __m256i*)(q_hi_ptr + i / 2 + 32));

            process_chunk_64(code_0, acc0, q_lo_1, q_hi_1, 32);
            process_chunk_64(code_1, acc1, q_lo_1, q_hi_1, 32);
            process_chunk_64(code_2, acc2, q_lo_1, q_hi_1, 32);
            process_chunk_64(code_3, acc3, q_lo_1, q_hi_1, 32);
        }

        // Handle remaining 64-dimensional chunk
        if (i + 64 <= d) {
            __m256i q_lo_0 =
                    _mm256_loadu_si256((const __m256i*)(q_lo_ptr + i / 2));
            __m256i q_hi_0 =
                    _mm256_loadu_si256((const __m256i*)(q_hi_ptr + i / 2));

            auto process = [&](const uint8_t* code, __m256i& acc) {
                __m256i c = _mm256_loadu_si256((const __m256i*)(code + i / 2));
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

        // Horizontal reductions
        auto reduce = [](const __m256i& acc) -> int32_t {
            __m128i acc_lo = _mm256_castsi256_si128(acc);
            __m128i acc_hi = _mm256_extracti128_si256(acc, 1);
            acc_lo = _mm_add_epi32(acc_lo, acc_hi);
            acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
            acc_lo = _mm_hadd_epi32(acc_lo, acc_lo);
            return _mm_cvtsi128_si32(acc_lo);
        };

        dis0 = reduce(acc0);
        dis1 = reduce(acc1);
        dis2 = reduce(acc2);
        dis3 = reduce(acc3);

        // Handle remaining dimensions scalar
        for (; i < d; i++) {
            uint8_t nibble_lo = q_lo[i / 2];
            uint8_t nibble_hi = q_hi[i / 2];

            auto process_scalar = [&](const uint8_t* code, float& dis) {
                uint8_t c = code[i / 2];
                uint8_t nibble;
                if (i % 2 == 0) {
                    nibble = c & 0xF;
                } else {
                    nibble = (c >> 4) & 0xF;
                }
                int diff;
                if (i % 2 == 0) {
                    diff = (int)nibble_lo - (int)nibble;
                } else {
                    diff = (int)nibble_hi - (int)nibble;
                }
                dis += diff * diff;
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

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        query_to_codes_batch_4(
                codes + idx0 * code_size,
                codes + idx1 * code_size,
                codes + idx2 * code_size,
                codes + idx3 * code_size,
                dis0,
                dis1,
                dis2,
                dis3);
    }
};

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, int SIMDWIDTH>
struct DCTemplate_avx : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate_avx<Quantizer, Similarity, 1>
        : public DCTemplate<Quantizer, Similarity, 1> {
    DCTemplate_avx(size_t d, const std::vector<float>& trained)
            : DCTemplate<Quantizer, Similarity, 1>(d, trained) {}
};

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <class Quantizer, class Similarity>
struct DCTemplate_avx<Quantizer, Similarity, 8> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate_avx(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 xi = quant.reconstruct_8_components(code, i);
            sim.add_8_components(xi);
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 x1 = quant.reconstruct_8_components(code1, i);
            __m256 x2 = quant.reconstruct_8_components(code2, i);
            sim.add_8_components_2(x1, x2);
        }
        return sim.result_8();
    }

    void set_query(const float* x) final {
        q = x;
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) final {
        return query_to_code(codes + i * code_size);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_distance(q, code);
    }

    void query_to_codes_batch_4(
            const uint8_t* __restrict code_0,
            const uint8_t* __restrict code_1,
            const uint8_t* __restrict code_2,
            const uint8_t* __restrict code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const override final {
        Similarity sim0(q);
        Similarity sim1(q);
        Similarity sim2(q);
        Similarity sim3(q);

        sim0.begin_8();
        sim1.begin_8();
        sim2.begin_8();
        sim3.begin_8();

        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 xi0 = quant.reconstruct_8_components(code_0, i);
            __m256 xi1 = quant.reconstruct_8_components(code_1, i);
            __m256 xi2 = quant.reconstruct_8_components(code_2, i);
            __m256 xi3 = quant.reconstruct_8_components(code_3, i);
            sim0.add_8_components(xi0);
            sim1.add_8_components(xi1);
            sim2.add_8_components(xi2);
            sim3.add_8_components(xi3);
        }

        dis0 = sim0.result_8();
        dis1 = sim1.result_8();
        dis2 = sim2.result_8();
        dis3 = sim3.result_8();
    }
};
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, int SIMDWIDTH>
struct DistanceComputerByte_avx : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte_avx<Similarity, 1>
        : public DistanceComputerByte<Similarity, 1> {
    DistanceComputerByte_avx(int d, const std::vector<float>& unused)
            : DistanceComputerByte<Similarity, 1>(d, unused) {}
};

template <class Similarity>
struct DistanceComputerByte_avx<Similarity, 8> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte_avx(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        // __m256i accu = _mm256_setzero_ps ();
        __m256i accu = _mm256_setzero_si256();
        for (int i = 0; i < d; i += 16) {
            // load 16 bytes, convert to 16 uint16_t
            __m256i c1 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code1 + i)));
            __m256i c2 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code2 + i)));
            __m256i prod32;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                prod32 = _mm256_madd_epi16(c1, c2);
            } else {
                __m256i diff = _mm256_sub_epi16(c1, c2);
                prod32 = _mm256_madd_epi16(diff, diff);
            }
            accu = _mm256_add_epi32(accu, prod32);
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32(sum, _mm256_extractf128_si256(accu, 1));
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(sum);
    }

    void set_query(const float* x) final {
        /*
        for (int i = 0; i < d; i += 8) {
            __m256 xi = _mm256_loadu_ps (x + i);
            __m256i ci = _mm256_cvtps_epi32(xi);
        */
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) final {
        return query_to_code(codes + i * code_size);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_code_distance(tmp.data(), code);
    }
};

/*******************************************************************
 * select_distance_computer: runtime selection of template
 * specialization
 *******************************************************************/

template <class Sim>
SQDistanceComputer* select_distance_computer_avx(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr int SIMDWIDTH = Sim::simdwidth;
    switch (qtype) {
        case QuantizerType::QT_8bit_uniform:
            return new DCTemplate_avx<
                    QuantizerTemplate_avx<Codec8bit_avx, QuantizerTemplateScaling::UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit_uniform:
            return new DistanceComputerSQ4UByte_avx<Sim>(d, trained);

        case QuantizerType::QT_8bit:
            return new DCTemplate_avx<
                    QuantizerTemplate_avx<Codec8bit_avx, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_6bit:
            return new DCTemplate_avx<
                    QuantizerTemplate_avx<Codec6bit_avx, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit:
            return new DCTemplate_avx<
                    QuantizerTemplate_avx<Codec4bit_avx, QuantizerTemplateScaling::NON_UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_fp16:
            return new DCTemplate_avx<
                    QuantizerFP16_avx<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_bf16:
            return new DCTemplate_avx<
                    QuantizerBF16_avx<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_8bit_direct:
            if (d % 16 == 0) {
                return new DistanceComputerByte_avx<Sim, SIMDWIDTH>(d, trained);
            } else {
                return new DCTemplate_avx<
                        Quantizer8bitDirect_avx<SIMDWIDTH>,
                        Sim,
                        SIMDWIDTH>(d, trained);
            }

        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate_avx<
                    Quantizer8bitDirectSigned_avx<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <class DCClass>
InvertedListScanner* sel2_InvertedListScanner_avx(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity, class Codec, QuantizerTemplateScaling SCALING>
InvertedListScanner* sel12_InvertedListScanner_avx(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate_avx<Codec, SCALING, SIMDWIDTH>;
    using DCClass = DCTemplate_avx<QuantizerClass, Similarity, SIMDWIDTH>;
    return sel2_InvertedListScanner_avx<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity>
InvertedListScanner* sel1_InvertedListScanner_avx(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    switch (sq->qtype) {
        case QuantizerType::QT_8bit_uniform:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec8bit_avx,
                    QuantizerTemplateScaling::UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit_uniform:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec4bit_avx,
                    QuantizerTemplateScaling::UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_8bit:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec8bit_avx,
                    QuantizerTemplateScaling::NON_UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec4bit_avx,
                    QuantizerTemplateScaling::NON_UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_6bit:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec6bit_avx,
                    QuantizerTemplateScaling::NON_UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_fp16:
            return sel2_InvertedListScanner_avx<DCTemplate_avx<
                    QuantizerFP16_avx<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_bf16:
            return sel2_InvertedListScanner_avx<DCTemplate_avx<
                    QuantizerBF16_avx<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_8bit_direct:
            if (sq->d % 16 == 0) {
                return sel2_InvertedListScanner_avx<
                        DistanceComputerByte_avx<Similarity, SIMDWIDTH>>(
                        sq, quantizer, store_pairs, sel, r);
            } else {
                return sel2_InvertedListScanner_avx<DCTemplate_avx<
                        Quantizer8bitDirect_avx<SIMDWIDTH>,
                        Similarity,
                        SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
            }
        case ScalarQuantizer::QT_8bit_direct_signed:
            return sel2_InvertedListScanner_avx<DCTemplate_avx<
                    Quantizer8bitDirectSigned_avx<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
    }

    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <int SIMDWIDTH>
InvertedListScanner* sel0_InvertedListScanner_avx(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner_avx<SimilarityL2_avx<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner_avx<SimilarityIP_avx<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

} // namespace faiss
