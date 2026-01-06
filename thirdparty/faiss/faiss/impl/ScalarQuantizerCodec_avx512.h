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
#include <faiss/impl/ScalarQuantizerCodec_avx.h>
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

struct Codec8bit_avx512 : public Codec8bit_avx {
    static FAISS_ALWAYS_INLINE __m512
    decode_16_components(const uint8_t* code, int i) {
        const __m128i c8 = _mm_loadu_si128((const __m128i_u*)(code + i));
        const __m512i i32 = _mm512_cvtepu8_epi32(c8);
        const __m512 f8 = _mm512_cvtepi32_ps(i32);
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 255.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 255.f);
        return _mm512_fmadd_ps(f8, one_255, half_one_255);
    }
};

struct Codec4bit_avx512 : public Codec4bit_avx {
    static FAISS_ALWAYS_INLINE __m512
    decode_16_components(const uint8_t* code, int i) {
        uint64_t c8 = *(uint64_t*)(code + (i >> 1));
        uint64_t mask = 0x0f0f0f0f0f0f0f0f;
        uint64_t c8ev = c8 & mask;
        uint64_t c8od = (c8 >> 4) & mask;

        // the 8 lower bytes of c8 contain the values
        __m128i c16 =
                _mm_unpacklo_epi8(_mm_set1_epi64x(c8ev), _mm_set1_epi64x(c8od));
        __m256i c8lo = _mm256_cvtepu8_epi32(c16);
        __m256i c8hi = _mm256_cvtepu8_epi32(_mm_srli_si128(c16, 8));
        __m512i i16 = _mm512_castsi256_si512(c8lo);
        i16 = _mm512_inserti32x8(i16, c8hi, 1);
        __m512 f16 = _mm512_cvtepi32_ps(i16);
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 15.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 15.f);
        return _mm512_fmadd_ps(f16, one_255, half_one_255);
    }

    static FAISS_ALWAYS_INLINE __m512i
    decode_16_components_int(const uint8_t* code, int i) {
        __m128i v8 = _mm_loadl_epi64((const __m128i*)(code + (i >> 1)));
        __m128i v16 = _mm_unpacklo_epi8(v8, v8);
        __m512i v512 = _mm512_cvtepu8_epi32(v16);

        // Shift right: 0 for even, 4 for odd
        const __m512i shift_counts = _mm512_setr_epi32(
                0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4);
        v512 = _mm512_srlv_epi32(v512, shift_counts);
        return _mm512_and_si512(v512, _mm512_set1_epi32(0xF));
    }
};

struct Codec6bit_avx512 : public Codec6bit_avx {
    // TODO: can be optimized
    static FAISS_ALWAYS_INLINE __m512
    decode_16_components(const uint8_t* code, int i) {
        /*
        // todo aguzhva: the following piece of code is very fast
        //   for Intel chips. AMD ones will be very slow unless Zen3+

        const uint16_t* data16_0 = (const uint16_t*)(code + (i >> 2) * 3);
        const uint64_t* data64_0 = (const uint64_t*)data16_0;
        const uint64_t val_0 = *data64_0;
        const uint64_t vext_0 = _pdep_u64(val_0, 0x3F3F3F3F3F3F3F3FULL);

        const uint16_t* data16_1 = data16_0 + 3;
        const uint32_t* data32_1 = (const uint32_t*)data16_1;
        const uint64_t val_1 = *data32_1 + ((uint64_t)data16_1[2] << 32);
        const uint64_t vext_1 = _pdep_u64(val_1, 0x3F3F3F3F3F3F3F3FULL);

        const __m128i i8 = _mm_set_epi64x(vext_1, vext_0);
        const __m512i i32 = _mm512_cvtepi8_epi32(i8);
        const __m512 f8 = _mm512_cvtepi32_ps(i32);
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 63.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 63.f);
        return _mm512_fmadd_ps(f8, one_255, half_one_255);
        */

        /*
        // todo aguzhva: another candidate for pdep, which might be faster
        const uint16_t* data16_0 = (const uint16_t*)(code + (i >> 2) * 3);
        const uint64_t* data64_0 = (const uint64_t*)data16_0;
        const uint64_t val_0 = *data64_0;
        const uint64_t vext_0 = _pdep_u64(val_0, 0x3F3F3F3F3F3F3F3FULL);

        const uint32_t* data32_1 = (const uint32_t*)data16_0;
        const uint64_t val_1 = (val_0 >> 48) | (((uint64_t)data32_1[1]) << 16);
        const uint64_t vext_1 = _pdep_u64(val_1, 0x3F3F3F3F3F3F3F3FULL);

        const __m128i i8 = _mm_set_epi64x(vext_1, vext_0);
        const __m512i i32 = _mm512_cvtepi8_epi32(i8);
        const __m512 f8 = _mm512_cvtepi32_ps(i32);
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 63.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 63.f);
        return _mm512_fmadd_ps(f8, one_255, half_one_255);
        */

        // pure AVX512 implementation, slower than pdep one, but has no problems
        // for AMD

        // clang-format off

        // 16 components, 16x6 bit=12 bytes
        const __m128i bit_6v =
                _mm_maskz_loadu_epi8(0b0000111111111111, code + (i >> 2) * 3);
        const __m256i bit_6v_256 = _mm256_broadcast_i32x4(bit_6v);

        // 00 01 02 03 04 05 06 07 08 09 0A 0B 0C 0D 0E 0F
        // 00          01          02          03
        const __m256i shuffle_mask = _mm256_setr_epi16(
                0xFF00, 0x0100, 0x0201, 0xFF02,
                0xFF03, 0x0403, 0x0504, 0xFF05,
                0xFF06, 0x0706, 0x0807, 0xFF08,
                0xFF09, 0x0A09, 0x0B0A, 0xFF0B);
        const __m256i shuffled = _mm256_shuffle_epi8(bit_6v_256, shuffle_mask);

        // 0: xxxxxxxx xx543210
        // 1: xxxx5432 10xxxxxx
        // 2: xxxxxx54 3210xxxx
        // 3: xxxxxxxx 543210xx
        const __m256i shift_right_v = _mm256_setr_epi16(
                0x0U, 0x6U, 0x4U, 0x2U,
                0x0U, 0x6U, 0x4U, 0x2U,
                0x0U, 0x6U, 0x4U, 0x2U,
                0x0U, 0x6U, 0x4U, 0x2U);
        __m256i shuffled_shifted = _mm256_srlv_epi16(shuffled, shift_right_v);

        // remove unneeded bits
        shuffled_shifted =
                _mm256_and_si256(shuffled_shifted, _mm256_set1_epi16(0x003F));

        // scale
        const __m512 f8 =
                _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(shuffled_shifted));
        const __m512 half_one_255 = _mm512_set1_ps(0.5f / 63.f);
        const __m512 one_255 = _mm512_set1_ps(1.f / 63.f);
        return _mm512_fmadd_ps(f8, one_255, half_one_255);

        // clang-format on
    }
};

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

template <class Codec, QuantizerTemplateScaling SCALING, int SIMD>
struct QuantizerTemplate_avx512 {};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, QuantizerTemplateScaling::UNIFORM, 1>
        : public QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::UNIFORM, 1> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::UNIFORM, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, QuantizerTemplateScaling::UNIFORM, 8>
        : public QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::UNIFORM, 8> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::UNIFORM, 8>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, QuantizerTemplateScaling::UNIFORM, 16>
        : public QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::UNIFORM, 8> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::UNIFORM, 8>(d, trained) {}

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i);
        return _mm512_fmadd_ps(
                xi, _mm512_set1_ps(this->vdiff), _mm512_set1_ps(this->vmin));
    }
};

template <>
struct QuantizerTemplate_avx512<
        Codec4bit_avx512,
        QuantizerTemplateScaling::UNIFORM,
        16>
        : public QuantizerTemplate_avx<
                  Codec4bit_avx512,
                  QuantizerTemplateScaling::UNIFORM,
                  8> {
    float final_scale;
    float final_bias;

    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<
                      Codec4bit_avx512,
                      QuantizerTemplateScaling::UNIFORM,
                      8>(d, trained) {
        final_scale = this->vdiff / 15.0f;
        final_bias = this->vmin + this->vdiff * 0.5f / 15.0f;
    }

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512i nibbles = Codec4bit_avx512::decode_16_components_int(code, i);
        __m512 nibbles_f = _mm512_cvtepi32_ps(nibbles);

        return _mm512_fmadd_ps(
                nibbles_f,
                _mm512_set1_ps(final_scale),
                _mm512_set1_ps(final_bias));
    }
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1>
        : public QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::NON_UNIFORM, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, QuantizerTemplateScaling::NON_UNIFORM, 8>
        : public QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::NON_UNIFORM, 8> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::NON_UNIFORM, 8>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, QuantizerTemplateScaling::NON_UNIFORM, 16>
        : public QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::NON_UNIFORM, 8> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, QuantizerTemplateScaling::NON_UNIFORM, 8>(d, trained) {}

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i);
        return _mm512_fmadd_ps(
                xi,
                _mm512_loadu_ps(this->vdiff + i),
                _mm512_loadu_ps(this->vmin + i));
    }
};

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerFP16_avx512 {};

template <>
struct QuantizerFP16_avx512<1> : public QuantizerFP16_avx<1> {
    QuantizerFP16_avx512(size_t d, const std::vector<float>& unused)
            : QuantizerFP16_avx<1>(d, unused) {}
};

template <>
struct QuantizerFP16_avx512<8> : public QuantizerFP16_avx<8> {
    QuantizerFP16_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerFP16_avx<8>(d, trained) {}
};

template <>
struct QuantizerFP16_avx512<16> : public QuantizerFP16_avx<8> {
    QuantizerFP16_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerFP16_avx<8>(d, trained) {}

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i codei = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        return _mm512_cvtph_ps(codei);
    }
};

/*******************************************************************
 * BF16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerBF16_avx512 {};

template <>
struct QuantizerBF16_avx512<1> : public QuantizerBF16_avx<1> {
    QuantizerBF16_avx512(size_t d, const std::vector<float>& unused)
            : QuantizerBF16_avx<1>(d, unused) {}
};

template <>
struct QuantizerBF16_avx512<8> : public QuantizerBF16_avx<8> {
    QuantizerBF16_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerBF16_avx<8>(d, trained) {}
};

template <>
struct QuantizerBF16_avx512<16> : public QuantizerBF16_avx<8> {
    QuantizerBF16_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerBF16_avx<8>(d, trained) {}

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i code_256i = _mm256_loadu_si256((const __m256i*)(code + 2 * i));
        __m512i code_512i = _mm512_cvtepu16_epi32(code_256i);
        code_512i = _mm512_slli_epi32(code_512i, 16);
        return _mm512_castsi512_ps(code_512i);
    }
};

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirect_avx512 {};

template <>
struct Quantizer8bitDirect_avx512<1> : public Quantizer8bitDirect_avx<1> {
    Quantizer8bitDirect_avx512(size_t d, const std::vector<float>& unused)
            : Quantizer8bitDirect_avx<1>(d, unused) {}
};

template <>
struct Quantizer8bitDirect_avx512<8> : public Quantizer8bitDirect_avx<8> {
    Quantizer8bitDirect_avx512(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect_avx<8>(d, trained) {}
};

template <>
struct Quantizer8bitDirect_avx512<16> : public Quantizer8bitDirect_avx<8> {
    Quantizer8bitDirect_avx512(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect_avx<8>(d, trained) {}

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        return _mm512_cvtepi32_ps(y16);                      // 16 * float32
    }
};

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirectSigned_avx512 {};

template <>
struct Quantizer8bitDirectSigned_avx512<1>
        : public Quantizer8bitDirectSigned_avx<1> {
    Quantizer8bitDirectSigned_avx512(size_t d, const std::vector<float>& unused)
            : Quantizer8bitDirectSigned_avx<1>(d, unused) {}
};

template <>
struct Quantizer8bitDirectSigned_avx512<8>
        : public Quantizer8bitDirectSigned_avx<8> {
    Quantizer8bitDirectSigned_avx512(
            size_t d,
            const std::vector<float>& trained)
            : Quantizer8bitDirectSigned_avx<8>(d, trained) {}
};

template <>
struct Quantizer8bitDirectSigned_avx512<16>
        : public Quantizer8bitDirectSigned_avx<8> {
    Quantizer8bitDirectSigned_avx512(
            size_t d,
            const std::vector<float>& trained)
            : Quantizer8bitDirectSigned_avx<8>(d, trained) {}

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m128i x16 = _mm_loadu_si128((__m128i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi32(x16);             // 16 * int32
        __m512i c16 = _mm512_set1_epi32(128);
        __m512i z16 = _mm512_sub_epi32(y16, c16); // subtract 128 from all lanes
        return _mm512_cvtepi32_ps(z16);           // 16 * float32
    }
};

template <int SIMDWIDTH>
SQuantizer* select_quantizer_1_avx512(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case QuantizerType::QT_8bit:
            return new QuantizerTemplate_avx512<
                    Codec8bit_avx512,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_6bit:
            return new QuantizerTemplate_avx512<
                    Codec6bit_avx512,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_4bit:
            return new QuantizerTemplate_avx512<
                    Codec4bit_avx512,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_uniform:
            return new QuantizerTemplate_avx512<
                    Codec8bit_avx512,
                    QuantizerTemplateScaling::UNIFORM,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_4bit_uniform:
            return new QuantizerTemplate_avx512<
                    Codec4bit_avx512,
                    QuantizerTemplateScaling::UNIFORM,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_fp16:
            return new QuantizerFP16_avx512<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_bf16:
            return new QuantizerBF16_avx512<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct:
            return new Quantizer8bitDirect_avx512<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned_avx512<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_1bit_direct:
            // todo: add more SIMDWIDTH support for avx512 if needed
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
struct SimilarityL2_avx512 {};

template <>
struct SimilarityL2_avx512<1> : public SimilarityL2_avx<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    explicit SimilarityL2_avx512(const float* y) : SimilarityL2_avx<1>(y) {}
};

template <>
struct SimilarityL2_avx512<8> : public SimilarityL2_avx<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;

    explicit SimilarityL2_avx512(const float* y) : SimilarityL2_avx<8>(y) {}
};

template <>
struct SimilarityL2_avx512<16> {
    static constexpr int simdwidth = 16;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2_avx512(const float* y) : y(y) {}
    __m512 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16 = _mm512_setzero_ps();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(__m512 x) {
        __m512 yiv = _mm512_loadu_ps(yi);
        yi += 16;
        __m512 tmp = _mm512_sub_ps(yiv, x);
        accu16 = _mm512_fmadd_ps(tmp, tmp, accu16);
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(__m512 x, __m512 y_2) {
        __m512 tmp = _mm512_sub_ps(y_2, x);
        accu16 = _mm512_fmadd_ps(tmp, tmp, accu16);
    }

    FAISS_ALWAYS_INLINE float result_16() {
        return _mm512_reduce_add_ps(accu16);
    }
};

template <int SIMDWIDTH>
struct SimilarityIP_avx512 {};

template <>
struct SimilarityIP_avx512<1> : public SimilarityIP_avx<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    explicit SimilarityIP_avx512(const float* y) : SimilarityIP_avx<1>(y) {}
};

template <>
struct SimilarityIP_avx512<8> : public SimilarityIP_avx<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    explicit SimilarityIP_avx512(const float* y) : SimilarityIP_avx<8>(y) {}
};

template <>
struct SimilarityIP_avx512<16> {
    static constexpr int simdwidth = 16;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP_avx512(const float* y) : y(y) {}

    __m512 accu16;

    FAISS_ALWAYS_INLINE void begin_16() {
        accu16 = _mm512_setzero_ps();
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_16_components(__m512 x) {
        __m512 yiv = _mm512_loadu_ps(yi);
        yi += 16;
        accu16 = _mm512_fmadd_ps(yiv, x, accu16);
    }

    FAISS_ALWAYS_INLINE void add_16_components_2(__m512 x1, __m512 x2) {
        accu16 = _mm512_fmadd_ps(x1, x2, accu16);
    }

    FAISS_ALWAYS_INLINE float result_16() {
        return _mm512_reduce_add_ps(accu16);
    }
};

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, int SIMDWIDTH>
struct DCTemplate_avx512 : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate_avx512<Quantizer, Similarity, 1>
        : public DCTemplate_avx<Quantizer, Similarity, 1> {
    DCTemplate_avx512(size_t d, const std::vector<float>& trained)
            : DCTemplate_avx<Quantizer, Similarity, 1>(d, trained) {}
};

template <class Quantizer, class Similarity>
struct DCTemplate_avx512<Quantizer, Similarity, 8>
        : public DCTemplate_avx<Quantizer, Similarity, 8> {
    DCTemplate_avx512(size_t d, const std::vector<float>& trained)
            : DCTemplate_avx<Quantizer, Similarity, 8>(d, trained) {}
};

template <class Quantizer, class Similarity>
struct DCTemplate_avx512<Quantizer, Similarity, 16> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate_avx512(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            __m512 xi = quant.reconstruct_16_components(code, i);
            sim.add_16_components(xi);
        }
        return sim.result_16();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_16();
        for (size_t i = 0; i < quant.d; i += 16) {
            __m512 x1 = quant.reconstruct_16_components(code1, i);
            __m512 x2 = quant.reconstruct_16_components(code2, i);
            sim.add_16_components_2(x1, x2);
        }
        return sim.result_16();
    }

    void set_query(const float* x) final {
        q = x;
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) final {
        return compute_distance(q, codes + i * code_size);
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

        sim0.begin_16();
        sim1.begin_16();
        sim2.begin_16();
        sim3.begin_16();

        for (size_t i = 0; i < quant.d; i += 16) {
            __m512 xi0 = quant.reconstruct_16_components(code_0, i);
            __m512 xi1 = quant.reconstruct_16_components(code_1, i);
            __m512 xi2 = quant.reconstruct_16_components(code_2, i);
            __m512 xi3 = quant.reconstruct_16_components(code_3, i);
            sim0.add_16_components(xi0);
            sim1.add_16_components(xi1);
            sim2.add_16_components(xi2);
            sim3.add_16_components(xi3);
        }

        dis0 = sim0.result_16();
        dis1 = sim1.result_16();
        dis2 = sim2.result_16();
        dis3 = sim3.result_16();
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

template <class Similarity, bool USE_VNNI>
struct DistanceComputerSQ4UByte_avx512 : SQDistanceComputer {
    using Quantizer = QuantizerTemplate_avx512<
            Codec4bit_avx512,
            QuantizerTemplateScaling::UNIFORM,
            16>;
    using Sim = Similarity;

    Quantizer quant;
    std::vector<uint8_t> q_lo;
    std::vector<uint8_t> q_hi;
    float final_scale_sq;

    DistanceComputerSQ4UByte_avx512(size_t d, const std::vector<float>& trained)
            : quant(d, trained),
              q_lo((d + 1) / 2 + 64, 0),
              q_hi((d + 1) / 2 + 64, 0) {
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
        __m512i acc = _mm512_setzero_si512();
        const size_t d = quant.d;
        const __m512i mask_f = _mm512_set1_epi8(0xF);
        const __m512i one = _mm512_set1_epi16(1);
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        size_t i = 0;
        for (; i + 128 <= d; i += 128) {
            __m512i c512 = _mm512_loadu_si512((const __m512i*)(code + i / 2));

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

        // Handle remaining dimensions
        if (i < d) {
            size_t rem = d - i;
            uint64_t mask_even =
                    (rem + 1) / 2 >= 64 ? -1ULL : (1ULL << ((rem + 1) / 2)) - 1;
            uint64_t mask_odd = rem / 2 >= 64 ? -1ULL : (1ULL << (rem / 2)) - 1;

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

        int32_t sum = _mm512_reduce_add_epi32(acc);
        return sum * final_scale_sq;
    }

    float compute_code_distance_l2(const uint8_t* code1, const uint8_t* code2)
            const {
        __m512i acc = _mm512_setzero_si512();
        const size_t d = quant.d;

        size_t i = 0;
        for (; i + 128 <= d; i += 128) {
            __m512i c1_512 =
                    _mm512_loadu_si512((const __m512i*)(code1 + i / 2));
            __m512i c2_512 =
                    _mm512_loadu_si512((const __m512i*)(code2 + i / 2));

            __m512i c1_nibbles_lo =
                    _mm512_and_si512(c1_512, _mm512_set1_epi8(0xF));
            __m512i c1_nibbles_hi = _mm512_and_si512(
                    _mm512_srli_epi16(c1_512, 4), _mm512_set1_epi8(0xF));

            __m512i c2_nibbles_lo =
                    _mm512_and_si512(c2_512, _mm512_set1_epi8(0xF));
            __m512i c2_nibbles_hi = _mm512_and_si512(
                    _mm512_srli_epi16(c2_512, 4), _mm512_set1_epi8(0xF));

            __m512i diff_lo = _mm512_sub_epi8(c1_nibbles_lo, c2_nibbles_lo);
            __m512i diff_hi = _mm512_sub_epi8(c1_nibbles_hi, c2_nibbles_hi);

            diff_lo = _mm512_abs_epi8(diff_lo);
            diff_hi = _mm512_abs_epi8(diff_hi);

            __m512i sq_lo = _mm512_maddubs_epi16(diff_lo, diff_lo);
            __m512i sq_hi = _mm512_maddubs_epi16(diff_hi, diff_hi);

            __m512i sum_lo = _mm512_madd_epi16(sq_lo, _mm512_set1_epi16(1));
            __m512i sum_hi = _mm512_madd_epi16(sq_hi, _mm512_set1_epi16(1));

            acc = _mm512_add_epi32(acc, sum_lo);
            acc = _mm512_add_epi32(acc, sum_hi);
        }

        // Handle remaining dimensions
        if (i < d) {
            size_t rem = d - i;
            uint64_t mask_even =
                    (rem + 1) / 2 >= 64 ? -1ULL : (1ULL << ((rem + 1) / 2)) - 1;
            uint64_t mask_odd = rem / 2 >= 64 ? -1ULL : (1ULL << (rem / 2)) - 1;

            __m512i c1_512 = _mm512_maskz_loadu_epi8(mask_even, code1 + i / 2);
            __m512i c2_512 = _mm512_maskz_loadu_epi8(mask_even, code2 + i / 2);

            __m512i c1_nibbles_lo =
                    _mm512_and_si512(c1_512, _mm512_set1_epi8(0xF));
            __m512i c1_nibbles_hi = _mm512_and_si512(
                    _mm512_srli_epi16(c1_512, 4), _mm512_set1_epi8(0xF));

            __m512i c2_nibbles_lo =
                    _mm512_and_si512(c2_512, _mm512_set1_epi8(0xF));
            __m512i c2_nibbles_hi = _mm512_and_si512(
                    _mm512_srli_epi16(c2_512, 4), _mm512_set1_epi8(0xF));

            __m512i mask_odd_vec = _mm512_movm_epi8(mask_odd);
            c1_nibbles_hi = _mm512_and_si512(c1_nibbles_hi, mask_odd_vec);
            c2_nibbles_hi = _mm512_and_si512(c2_nibbles_hi, mask_odd_vec);

            __m512i diff_lo = _mm512_sub_epi8(c1_nibbles_lo, c2_nibbles_lo);
            __m512i diff_hi = _mm512_sub_epi8(c1_nibbles_hi, c2_nibbles_hi);

            diff_lo = _mm512_abs_epi8(diff_lo);
            diff_hi = _mm512_abs_epi8(diff_hi);

            __m512i sq_lo = _mm512_maddubs_epi16(diff_lo, diff_lo);
            __m512i sq_hi = _mm512_maddubs_epi16(diff_hi, diff_hi);

            __m512i sum_lo = _mm512_madd_epi16(sq_lo, _mm512_set1_epi16(1));
            __m512i sum_hi = _mm512_madd_epi16(sq_hi, _mm512_set1_epi16(1));

            acc = _mm512_add_epi32(acc, sum_lo);
            acc = _mm512_add_epi32(acc, sum_hi);
        }

        int32_t sum = _mm512_reduce_add_epi32(acc);
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
            float& dis3) const override final {
        if constexpr (USE_VNNI) {
            query_to_codes_batch_4_vnni(
                    code_0, code_1, code_2, code_3, dis0, dis1, dis2, dis3);
        } else {
            query_to_codes_batch_4_avx512(
                    code_0, code_1, code_2, code_3, dis0, dis1, dis2, dis3);
        }
    }

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

        const size_t d = quant.d;
        const __m512i mask_f = _mm512_set1_epi8(0xF);
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        size_t i = 0;
        // 256 dimensions per iteration
        for (; i + 256 <= d; i += 256) {
            // Chunk 0
            __m512i q_lo_0 = _mm512_loadu_si512(q_lo_ptr + i / 2);
            __m512i q_hi_0 = _mm512_loadu_si512(q_hi_ptr + i / 2);

            // Chunk 1
            __m512i q_lo_1 = _mm512_loadu_si512(q_lo_ptr + i / 2 + 64);
            __m512i q_hi_1 = _mm512_loadu_si512(q_hi_ptr + i / 2 + 64);

            auto process_chunk = [&](
                    const uint8_t* code,
                    __m512i& acc,
                    __m512i q_lo,
                    __m512i q_hi,
                    int offset) __attribute__((target("avx512vnni"))) {
                __m512i c512 = _mm512_loadu_si512(
                        (const __m512i*)(code + i / 2 + offset));
                __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
                __m512i nibbles_hi =
                        _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);

                __m512i diff_lo = _mm512_sub_epi8(q_lo, nibbles_lo);
                __m512i diff_hi = _mm512_sub_epi8(q_hi, nibbles_hi);

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

        if (i + 128 <= d) {
            __m512i q_lo_0 = _mm512_loadu_si512(q_lo_ptr + i / 2);
            __m512i q_hi_0 = _mm512_loadu_si512(q_hi_ptr + i / 2);

            auto process_chunk = [&](const uint8_t* code, __m512i& acc)
                    __attribute__((target("avx512vnni"))) {
                __m512i c512 =
                        _mm512_loadu_si512((const __m512i*)(code + i / 2));
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

        // Handle remaining dimensions
        if (i < d) {
            size_t rem = d - i;
            uint64_t mask_even =
                    (rem + 1) / 2 >= 64 ? -1ULL : (1ULL << ((rem + 1) / 2)) - 1;
            uint64_t mask_odd = rem / 2 >= 64 ? -1ULL : (1ULL << (rem / 2)) - 1;

            __m512i q_lo_vec =
                    _mm512_maskz_loadu_epi8(mask_even, q_lo_ptr + i / 2);
            __m512i q_hi_vec =
                    _mm512_maskz_loadu_epi8(mask_odd, q_hi_ptr + i / 2);
            __m512i mask_odd_vec = _mm512_movm_epi8(mask_odd);

            auto process = [&](const uint8_t* code, __m512i& acc)
                    __attribute__((target("avx512vnni"))) {
                __m512i c512 = _mm512_maskz_loadu_epi8(mask_even, code + i / 2);
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

        dis0 = _mm512_reduce_add_epi32(acc0) * final_scale_sq;
        dis1 = _mm512_reduce_add_epi32(acc1) * final_scale_sq;
        dis2 = _mm512_reduce_add_epi32(acc2) * final_scale_sq;
        dis3 = _mm512_reduce_add_epi32(acc3) * final_scale_sq;
    }

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

        const size_t d = quant.d;
        const __m512i mask_f = _mm512_set1_epi8(0xF);
        const __m512i one = _mm512_set1_epi16(1);
        const uint8_t* q_lo_ptr = q_lo.data();
        const uint8_t* q_hi_ptr = q_hi.data();

        size_t i = 0;
        // 256 dimensions per iteration
        for (; i + 256 <= d; i += 256) {
            // Chunk 0
            __m512i q_lo_0 = _mm512_loadu_si512(q_lo_ptr + i / 2);
            __m512i q_hi_0 = _mm512_loadu_si512(q_hi_ptr + i / 2);

            // Chunk 1
            __m512i q_lo_1 = _mm512_loadu_si512(q_lo_ptr + i / 2 + 64);
            __m512i q_hi_1 = _mm512_loadu_si512(q_hi_ptr + i / 2 + 64);

            auto process_chunk = [&](const uint8_t* code,
                                     __m512i& acc,
                                     __m512i q_lo,
                                     __m512i q_hi,
                                     int offset) {
                __m512i c512 = _mm512_loadu_si512(
                        (const __m512i*)(code + i / 2 + offset));
                __m512i nibbles_lo = _mm512_and_si512(c512, mask_f);
                __m512i nibbles_hi =
                        _mm512_and_si512(_mm512_srli_epi16(c512, 4), mask_f);

                __m512i diff_lo = _mm512_sub_epi8(q_lo, nibbles_lo);
                __m512i diff_hi = _mm512_sub_epi8(q_hi, nibbles_hi);

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
                __m512i c512 =
                        _mm512_loadu_si512((const __m512i*)(code + i / 2));
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

        // Handle remaining dimensions
        if (i < d) {
            size_t rem = d - i;
            uint64_t mask_even =
                    (rem + 1) / 2 >= 64 ? -1ULL : (1ULL << ((rem + 1) / 2)) - 1;
            uint64_t mask_odd = rem / 2 >= 64 ? -1ULL : (1ULL << (rem / 2)) - 1;

            __m512i q_lo_vec =
                    _mm512_maskz_loadu_epi8(mask_even, q_lo_ptr + i / 2);
            __m512i q_hi_vec =
                    _mm512_maskz_loadu_epi8(mask_odd, q_hi_ptr + i / 2);
            __m512i mask_odd_vec = _mm512_movm_epi8(mask_odd);

            auto process = [&](const uint8_t* code, __m512i& acc) {
                __m512i c512 = _mm512_maskz_loadu_epi8(mask_even, code + i / 2);
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

        dis0 = _mm512_reduce_add_epi32(acc0) * final_scale_sq;
        dis1 = _mm512_reduce_add_epi32(acc1) * final_scale_sq;
        dis2 = _mm512_reduce_add_epi32(acc2) * final_scale_sq;
        dis3 = _mm512_reduce_add_epi32(acc3) * final_scale_sq;
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
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, int SIMDWIDTH>
struct DistanceComputerByte_avx512 : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte_avx512<Similarity, 1>
        : public DistanceComputerByte_avx<Similarity, 1> {
    DistanceComputerByte_avx512(int d, const std::vector<float>& unused)
            : DistanceComputerByte_avx<Similarity, 1>(d, unused) {}
};

template <class Similarity>
struct DistanceComputerByte_avx512<Similarity, 8>
        : public DistanceComputerByte_avx<Similarity, 8> {
    DistanceComputerByte_avx512(int d, const std::vector<float>& unused)
            : DistanceComputerByte_avx<Similarity, 8>(d, unused) {}
};

template <class Similarity>
struct DistanceComputerByte_avx512<Similarity, 16> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte_avx512(int d, const std::vector<float>&)
            : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        // __m256i accu = _mm256_setzero_ps ();
        __m512i accu = _mm512_setzero_si512();
        for (int i = 0; i < d; i += 32) {
            // load 32 bytes, convert to 16 uint16_t
            __m512i c1 = _mm512_cvtepu8_epi16(
                    _mm256_loadu_si256((__m256i*)(code1 + i)));
            __m512i c2 = _mm512_cvtepu8_epi16(
                    _mm256_loadu_si256((__m256i*)(code2 + i)));
            __m512i prod32;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                prod32 = _mm512_madd_epi16(c1, c2);
            } else {
                __m512i diff = _mm512_sub_epi16(c1, c2);
                prod32 = _mm512_madd_epi16(diff, diff);
            }
            accu = _mm512_add_epi32(accu, prod32);
        }
        return _mm512_reduce_add_epi32(accu);
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
        return compute_distance(q, codes + i * code_size);
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
SQDistanceComputer* select_distance_computer_avx512(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr int SIMDWIDTH = Sim::simdwidth;
    const bool use_vnni = __builtin_cpu_supports("avx512vnni");
    switch (qtype) {
        case QuantizerType::QT_8bit_uniform:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<Codec8bit_avx512, QuantizerTemplateScaling::UNIFORM, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit_uniform:
            if (use_vnni) {
                return new DistanceComputerSQ4UByte_avx512<Sim, true>(
                        d, trained);
            } else {
                return new DistanceComputerSQ4UByte_avx512<Sim, false>(
                        d, trained);
            }

        case QuantizerType::QT_8bit:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<
                            Codec8bit_avx512,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_6bit:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<
                            Codec6bit_avx512,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<
                            Codec4bit_avx512,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_fp16:
            return new DCTemplate_avx512<
                    QuantizerFP16_avx512<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_bf16:
            return new DCTemplate_avx512<
                    QuantizerBF16_avx512<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_8bit_direct:
            if (d % 16 == 0) {
                return new DistanceComputerByte_avx512<Sim, SIMDWIDTH>(
                        d, trained);
            } else {
                return new DCTemplate_avx512<
                        Quantizer8bitDirect_avx512<SIMDWIDTH>,
                        Sim,
                        SIMDWIDTH>(d, trained);
            }

        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate_avx512<
                    Quantizer8bitDirectSigned_avx512<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <class DCClass>
InvertedListScanner* sel2_InvertedListScanner_avx512(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity, class Codec, QuantizerTemplateScaling SCALING>
InvertedListScanner* sel12_InvertedListScanner_avx512(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate_avx512<Codec, SCALING, SIMDWIDTH>;
    using DCClass = DCTemplate_avx512<QuantizerClass, Similarity, SIMDWIDTH>;
    return sel2_InvertedListScanner_avx512<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity>
InvertedListScanner* sel1_InvertedListScanner_avx512(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    switch (sq->qtype) {
        case QuantizerType::QT_8bit_uniform:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec8bit_avx512,
                    QuantizerTemplateScaling::UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit_uniform:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec4bit_avx512,
                    QuantizerTemplateScaling::UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_8bit:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec8bit_avx512,
                    QuantizerTemplateScaling::NON_UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec4bit_avx512,
                    QuantizerTemplateScaling::NON_UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_6bit:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec6bit_avx512,
                    QuantizerTemplateScaling::NON_UNIFORM>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_fp16:
            return sel2_InvertedListScanner_avx512<DCTemplate_avx512<
                    QuantizerFP16_avx512<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_bf16:
            return sel2_InvertedListScanner_avx512<DCTemplate_avx512<
                    QuantizerBF16_avx512<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_8bit_direct:
            if (sq->d % 16 == 0) {
                return sel2_InvertedListScanner_avx512<
                        DistanceComputerByte_avx512<Similarity, SIMDWIDTH>>(
                        sq, quantizer, store_pairs, sel, r);
            } else {
                return sel2_InvertedListScanner_avx512<DCTemplate_avx512<
                        Quantizer8bitDirect_avx512<SIMDWIDTH>,
                        Similarity,
                        SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
            }
        case ScalarQuantizer::QT_8bit_direct_signed:
            return sel2_InvertedListScanner_avx512<DCTemplate_avx512<
                    Quantizer8bitDirectSigned_avx512<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
    }

    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <int SIMDWIDTH>
InvertedListScanner* sel0_InvertedListScanner_avx512(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner_avx512<SimilarityL2_avx512<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner_avx512<SimilarityIP_avx512<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

} // namespace faiss
