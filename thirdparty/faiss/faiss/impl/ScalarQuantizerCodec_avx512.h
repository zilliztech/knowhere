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
};

struct Codec6bit_avx512 : public Codec6bit_avx {
    // TODO: can be optimized
    static FAISS_ALWAYS_INLINE __m512
    decode_16_components(const uint8_t* code, int i) {
        // // todo aguzhva: the following piece of code is very fast
        // //   for Intel chips. AMD ones will be very slow unless Zen3+
        //
        // const uint16_t* data16_0 = (const uint16_t*)(code + (i >> 2) * 3);
        // const uint64_t* data64_0 = (const uint64_t*)data16_0;
        // const uint64_t val_0 = *data64_0;
        // const uint64_t vext_0 = _pdep_u64(val_0, 0x3F3F3F3F3F3F3F3FULL);
        //
        // const uint16_t* data16_1 = data16_0 + 3;
        // const uint32_t* data32_1 = (const uint32_t*)data16_1;
        // const uint64_t val_1 = *data32_1 + ((uint64_t)data16_1[2] << 32);
        // const uint64_t vext_1 = _pdep_u64(val_1, 0x3F3F3F3F3F3F3F3FULL);
        //
        // const __m128i i8 = _mm_set_epi64x(vext_1, vext_0);
        // const __m512i i32 = _mm512_cvtepi8_epi32(i8);
        // const __m512 f8 = _mm512_cvtepi32_ps(i32);
        // const __m512 half_one_255 = _mm512_set1_ps(0.5f / 63.f);
        // const __m512 one_255 = _mm512_set1_ps(1.f / 63.f);
        // return _mm512_fmadd_ps(f8, one_255, half_one_255);

        return _mm512_set_ps(
                decode_component(code, i + 15),
                decode_component(code, i + 14),
                decode_component(code, i + 13),
                decode_component(code, i + 12),
                decode_component(code, i + 11),
                decode_component(code, i + 10),
                decode_component(code, i + 9),
                decode_component(code, i + 8),
                decode_component(code, i + 7),
                decode_component(code, i + 6),
                decode_component(code, i + 5),
                decode_component(code, i + 4),
                decode_component(code, i + 3),
                decode_component(code, i + 2),
                decode_component(code, i + 1),
                decode_component(code, i + 0));
    }
};

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

template <class Codec, bool uniform, int SIMD>
struct QuantizerTemplate_avx512 {};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, true, 1>
        : public QuantizerTemplate_avx<Codec, true, 1> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, true, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, true, 8>
        : public QuantizerTemplate_avx<Codec, true, 8> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, true, 8>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, true, 16>
        : public QuantizerTemplate_avx<Codec, true, 8> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, true, 8>(d, trained) {}

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m512 xi = Codec::decode_16_components(code, i);
        return _mm512_fmadd_ps(
                xi, _mm512_set1_ps(this->vdiff), _mm512_set1_ps(this->vmin));
    }
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, false, 1>
        : public QuantizerTemplate_avx<Codec, false, 1> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, false, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, false, 8>
        : public QuantizerTemplate_avx<Codec, false, 8> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, false, 8>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx512<Codec, false, 16>
        : public QuantizerTemplate_avx<Codec, false, 8> {
    QuantizerTemplate_avx512(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate_avx<Codec, false, 8>(d, trained) {}

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
        __m256i x16 = _mm256_loadu_si256((__m256i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi16(x16);                // 16 * int32
        return _mm512_cvtepi32_ps(y16);                         // 16 * float32
    }
};

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirectSigned_avx512 {};

template <>
struct Quantizer8bitDirectSigned_avx512<1> : public Quantizer8bitDirectSigned_avx<1> {
    Quantizer8bitDirectSigned_avx512(size_t d, const std::vector<float>& unused)
            : Quantizer8bitDirectSigned_avx<1>(d, unused) {}
};

template <>
struct Quantizer8bitDirectSigned_avx512<8> : public Quantizer8bitDirectSigned_avx<8> {
    Quantizer8bitDirectSigned_avx512(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned_avx<8>(d, trained) {}
};

template <>
struct Quantizer8bitDirectSigned_avx512<16> : public Quantizer8bitDirectSigned_avx<8> {
    Quantizer8bitDirectSigned_avx512(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned_avx<8>(d, trained) {}

    FAISS_ALWAYS_INLINE __m512
    reconstruct_16_components(const uint8_t* code, int i) const {
        __m256i x16 = _mm256_loadu_si256((__m256i*)(code + i)); // 16 * int8
        __m512i y16 = _mm512_cvtepu8_epi16(x16);                // 16 * int32
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
                    false,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_6bit:
            return new QuantizerTemplate_avx512<
                    Codec6bit_avx512,
                    false,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_4bit:
            return new QuantizerTemplate_avx512<
                    Codec4bit_avx512,
                    false,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_uniform:
            return new QuantizerTemplate_avx512<
                    Codec8bit_avx512,
                    true,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_4bit_uniform:
            return new QuantizerTemplate_avx512<
                    Codec4bit_avx512,
                    true,
                    SIMDWIDTH>(d, trained);
        case QuantizerType::QT_fp16:
            return new QuantizerFP16_avx512<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_bf16:
            return new QuantizerBF16_avx512<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct:
            return new Quantizer8bitDirect_avx512<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned_avx512<SIMDWIDTH>(d, trained);
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
    switch (qtype) {
        case QuantizerType::QT_8bit_uniform:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<Codec8bit_avx512, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit_uniform:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<Codec4bit_avx512, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_8bit:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<
                            Codec8bit_avx512,
                            false,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_6bit:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<
                            Codec6bit_avx512,
                            false,
                            SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit:
            return new DCTemplate_avx512<
                    QuantizerTemplate_avx512<
                            Codec4bit_avx512,
                            false,
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

template <class Similarity, class Codec, bool uniform>
InvertedListScanner* sel12_InvertedListScanner_avx512(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate_avx512<Codec, uniform, SIMDWIDTH>;
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
                    true>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit_uniform:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec4bit_avx512,
                    true>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_8bit:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec8bit_avx512,
                    false>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec4bit_avx512,
                    false>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_6bit:
            return sel12_InvertedListScanner_avx512<
                    Similarity,
                    Codec6bit_avx512,
                    false>(sq, quantizer, store_pairs, sel, r);
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
