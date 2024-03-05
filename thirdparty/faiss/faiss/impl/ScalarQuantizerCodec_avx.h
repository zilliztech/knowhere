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

template <class Codec, bool uniform, int SIMD>
struct QuantizerTemplate_avx {};

template <class Codec>
struct QuantizerTemplate_avx<Codec, true, 1>
        : public QuantizerTemplate<Codec, true, 1> {
    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, true, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx<Codec, true, 8>
        : public QuantizerTemplate<Codec, true, 1> {
    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, true, 1>(d, trained) {}

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i);
        return _mm256_fmadd_ps(
                xi, _mm256_set1_ps(this->vdiff), _mm256_set1_ps(this->vmin));
    }
};

template <class Codec>
struct QuantizerTemplate_avx<Codec, false, 1>
        : public QuantizerTemplate<Codec, false, 1> {
    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, false, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_avx<Codec, false, 8>
        : public QuantizerTemplate<Codec, false, 1> {
    QuantizerTemplate_avx(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, false, 1>(d, trained) {}

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
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitInRow_avx {};
template <>
struct Quantizer8bitInRow_avx<1> : public Quantizer8bitInRow<1> {
    Quantizer8bitInRow_avx(size_t d, const std::vector<float>& unused)
            : Quantizer8bitInRow(d) {}
};

template <>
struct Quantizer8bitInRow_avx<8> : public Quantizer8bitInRow<1> {
    Quantizer8bitInRow_avx(size_t d, const std::vector<float>& trained)
            : Quantizer8bitInRow<1>(d) {}

    FAISS_ALWAYS_INLINE __m256
    reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32(x8);              // 8 * int32
        return _mm256_cvtepi32_ps(y8);                      // 8 * float32
    }

    FAISS_ALWAYS_INLINE float
    u8_dot_product(const uint8_t* code1, const uint8_t* code2) const {
        auto tmp_d = d;
        const uint8_t* code1_data = code1 + 12;
        const uint8_t* code2_data = code2 + 12;
        __m256i accu = _mm256_setzero_si256();
        while (tmp_d >= 16) {
            __m256i c1 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code1_data)));
            __m256i c2 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code2_data)));
            __m256i dot =  _mm256_madd_epi16(c1, c2);
            accu = _mm256_add_epi32(accu, dot);
            code1_data += 16;
            code2_data += 16;
            tmp_d -= 16;
        }
        if (tmp_d >= 8) {
            __m256i c1 = _mm256_cvtepu8_epi16(
                    _mm_loadl_epi64((__m128i*)(code1_data)));
            __m256i c2 = _mm256_cvtepu8_epi16(
                    _mm_loadl_epi64((__m128i*)(code2_data)));
            __m256i dot =  _mm256_madd_epi16(c1, c2);
            accu = _mm256_add_epi32(accu, dot);
            code1_data += 8;
            code2_data += 8;
            tmp_d -= 8;
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32(sum, _mm256_extractf128_si256(accu, 1));
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        int dot_product = _mm_cvtsi128_si32(sum);

        // can't go into these lines
        while (tmp_d > 0) {
            dot_product += (*code1_data) * (*code2_data);
            code1_data ++;
            code2_data ++;
            tmp_d --;
        }
        return float(dot_product);
    }

    FAISS_ALWAYS_INLINE void dot_product_batch_4(
        const uint8_t* __restrict codex,
        const uint8_t* __restrict codey1,
        const uint8_t* __restrict codey2,
        const uint8_t* __restrict codey3,
        const uint8_t* __restrict codey4,
        float* result1,
        float* result2,
        float* result3,
        float* result4) const {
        uint32_t tmp_d =d;
        const uint8_t* q_code = codex + 12;
        const uint8_t* code1 = codey1 + 12;
        const uint8_t* code2 = codey2 + 12;
        const uint8_t* code3 = codey3 + 12;
        const uint8_t* code4 = codey4 + 12;
        __m256i accu = _mm256_setzero_si256(); 
        while (tmp_d >= 16) {
            __m256i x = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(q_code))); 
            __m256i y1 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code1))); 
            __m256i y2 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code2)));
            __m256i y3 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code3)));
            __m256i y4 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code4)));
            __m256i accu_y1_y2 = _mm256_hadd_epi32(_mm256_madd_epi16(x, y1),  _mm256_madd_epi16(x, y2));
            __m256i accu_y3_y4 = _mm256_hadd_epi32(_mm256_madd_epi16(x, y3),  _mm256_madd_epi16(x, y4));
            accu = _mm256_add_epi32(_mm256_hadd_epi32(accu_y1_y2, accu_y3_y4), accu);
            tmp_d -= 16;
            q_code += 16; code1 +=16; code2 +=16; code3 +=16; code4 +=16; 
        }
        if (tmp_d >= 8) {
            __m256i x = _mm256_cvtepu8_epi16(
                _mm_loadl_epi64((__m128i*)(q_code))); 
            __m256i y1 = _mm256_cvtepu8_epi16(
                _mm_loadl_epi64((__m128i*)(code1))); 
            __m256i y2 = _mm256_cvtepu8_epi16(
                _mm_loadl_epi64((__m128i*)(code2)));
            __m256i y3 = _mm256_cvtepu8_epi16(
                _mm_loadl_epi64((__m128i*)(code3)));
            __m256i y4 = _mm256_cvtepu8_epi16(
                _mm_loadl_epi64((__m128i*)(code4)));
            __m256i accu_y1_y2 = _mm256_hadd_epi32(_mm256_madd_epi16(x, y1),  _mm256_madd_epi16(x, y2));
            __m256i accu_y3_y4 = _mm256_hadd_epi32(_mm256_madd_epi16(x, y3),  _mm256_madd_epi16(x, y4));
            accu = _mm256_add_epi32(_mm256_hadd_epi32(accu_y1_y2, accu_y3_y4), accu);
            tmp_d -= 8;
            q_code += 8; code1 += 8; code2 += 8; code3 += 8; code4 += 8;
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32(sum, _mm256_extractf128_si256(accu, 1));
        uint32_t tmp_res[4];
        _mm_store_si128((__m128i*)tmp_res, sum);
        *result1 = (float)tmp_res[0];*result2 = (float)tmp_res[1];*result3 = (float)tmp_res[2];*result4 = (float)tmp_res[3];
        while(tmp_d > 0) {
            (*result1) += (*code1) * (*q_code); code1++;
            (*result2) += (*code1) * (*q_code); code2++;
            (*result3) += (*code1) * (*q_code); code3++;
            (*result4) += (*code1) * (*q_code); code4++;
            q_code++;
            tmp_d--;
        }
    }

    FAISS_ALWAYS_INLINE uint32_t sum(const uint8_t* code) const {
        const uint8_t* code_data = code + 12;
        __m256i accu = _mm256_setzero_si256(); // 8 * uint32_t result  
        auto tmp_d = d;
        while (tmp_d >= 32) {
            __m256i c1 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code_data)));
            __m256i c2 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code_data + 16)));
            __m256i h_sum = _mm256_add_epi16(c1, c2);
            accu = _mm256_add_epi16(h_sum, accu); 
            code_data += 32;
            tmp_d -= 32;
        }
        if (tmp_d >= 16) {
            __m256i c = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code_data)));
            accu = _mm256_add_epi16(c, accu);
            code_data += 16;
            tmp_d -= 16;
        }
        __m256i lo = _mm256_and_si256(accu, _mm256_set1_epi32(65535));
        __m256i hi = _mm256_srli_epi32(accu, 16);
        accu  = _mm256_add_epi16(lo, hi);
        if (tmp_d >= 8) {
            __m256i c = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(code_data)));
            code_data += 8;
            tmp_d -= 8;
            accu  = _mm256_add_epi16(accu, c);
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32(sum, _mm256_extractf128_si256(accu, 1));
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        uint32_t code_sum = _mm_cvtsi128_si32(sum);
        // can't go these lines
        while (tmp_d > 0) {
            code_sum += *code_data;
            code_data++;
            tmp_d--;
        }
        return code_sum;
    }

    FAISS_ALWAYS_INLINE void sum_batch_4(
            const uint8_t* __restrict code_0,
            const uint8_t* __restrict code_1,
            const uint8_t* __restrict code_2,
            const uint8_t* __restrict code_3,
            uint32_t* result) const {
        uint32_t tmp_d =d;
        const uint8_t* code1 = code_0 + 12;
        const uint8_t* code2 = code_1 + 12;
        const uint8_t* code3 = code_2 + 12;
        const uint8_t* code4 = code_3 + 12;
        __m256i accu = _mm256_setzero_si256(); // dis0, dis1, dist2, dist3, dis0, dis1, dist2, dist3
        while (tmp_d >= 16) {
            __m256i c1 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code1))); // dis0: 16 elements  
            __m256i c2 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code2)));
            __m256i c3 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code3)));
            __m256i c4 = _mm256_cvtepu8_epi16(
            _mm_loadu_si128((__m128i*)(code4)));
            auto sum_c1_c2 = _mm256_hadd_epi16(c1, c2); // dis0: 8 elements  
            auto sum_c3_c4 = _mm256_hadd_epi16(c3, c4);
            auto sum_c1_c2_c3_c4 = _mm256_hadd_epi16(sum_c1_c2, sum_c3_c4); // dis0: 4 elements  
            accu = _mm256_add_epi32(_mm256_blend_epi16(sum_c1_c2_c3_c4, _mm256_setzero_si256(),0xAA), accu);
            accu = _mm256_add_epi32(_mm256_blend_epi16(_mm256_srli_epi32(sum_c1_c2_c3_c4, 16), _mm256_setzero_si256(),0xAA), accu);
            code1 +=16;
            code2 += 16; code3+=16; code4+=16;
            tmp_d-=16;
        }
        if (tmp_d >=8) {
            __m256i c1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(code1)));// c1: 8 elements  
            __m256i c2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(code2)));
            __m256i c3 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(code3)));
            __m256i c4 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(code4)));
            auto sum_c1_c2 = _mm256_hadd_epi32(c1, c2); // c1: 4 elements
            auto sum_c3_c4 = _mm256_hadd_epi32(c3, c4);
            accu = _mm256_add_epi32( _mm256_hadd_epi32(sum_c1_c2, sum_c3_c4), accu);
            code1 += 8; code2 += 8; code3 += 8; code4 += 8;
            tmp_d -= 8;
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32(sum, _mm256_extractf128_si256(accu, 1));
        _mm_store_si128((__m128i*)result, sum);
        while (tmp_d > 0) {
            (*result) += *code1; code1++;
            *(result+4) += *code2; code2++;
            *(result+8) += *code3; code3++;
            *(result+12) += *code4; code4++;
            tmp_d--;
        }
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

template <int SIMDWIDTH>
SQuantizer* select_quantizer_1_avx(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case QuantizerType::QT_8bit:
            return new QuantizerTemplate_avx<Codec8bit_avx, false, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_6bit:
            return new QuantizerTemplate_avx<Codec6bit_avx, false, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_4bit:
            return new QuantizerTemplate_avx<Codec4bit_avx, false, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_8bit_uniform:
            return new QuantizerTemplate_avx<Codec8bit_avx, true, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_4bit_uniform:
            return new QuantizerTemplate_avx<Codec4bit_avx, true, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_fp16:
            return new QuantizerFP16_avx<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct:
            return new Quantizer8bitDirect_avx<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_in_row:
            return new Quantizer8bitInRow_avx<SIMDWIDTH>(d, trained);
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

    FAISS_ALWAYS_INLINE void add_8_components_2(__m256 x, __m256 y) {
        __m256 tmp = _mm256_sub_ps(y, x);
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

        FAISS_PRAGMA_IMPRECISE_LOOP
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

template <class Similarity>
struct DCTemplate_avx<Quantizer8bitInRow_avx<8>, Similarity, 8> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer8bitInRow_avx<8> quant;
    std::vector<uint8_t> query_code;

    DCTemplate_avx(size_t d, const std::vector<float>& trained)
            : quant(d, trained), query_code(d + 12) {}
    
    float compute_code_distance(const uint8_t* code1, const uint8_t* code2) const { 
        auto encode_distace_ip = [&](const uint8_t* code1, const uint8_t* code2) -> float {
            float part1 = quant.u8_dot_product(code1, code2) * quant.code_len(code1) * quant.code_len(code2);
            float part2 = (float)quant.sum(code1) * quant.code_len(code1) * quant.code_bias(code2);
            float part3 = (float)quant.sum(code2) * quant.code_len(code2) * quant.code_bias(code1);
            float part4 = quant.code_bias(code1) *  quant.code_bias(code2) * quant.d;
            float res =  part1 + part2 + part3 + part4;
            return res;
        };
        if constexpr (std::is_same_v<Similarity, SimilarityL2_avx<8>>) {
            auto res = quant.l2_norm(code1) + quant.l2_norm(code2); 
            float dot_product = encode_distace_ip(code1, code2);
            return res - 2.0 * dot_product;
        }
        if constexpr (std::is_same_v<Similarity, SimilarityIP_avx<8>>) {
            return encode_distace_ip(code1, code2);
        }
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_code_distance(query_code.data(), code);
    }

    void query_to_codes_batch_4(const uint8_t* __restrict code_0,
            const uint8_t* __restrict code_1,
            const uint8_t* __restrict code_2,
            const uint8_t* __restrict code_3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const override final {
            uint32_t l1_norm[4];
            quant.sum_batch_4(code_0,code_1,code_2,code_3,l1_norm);
            quant.dot_product_batch_4(query_code.data(), code_0, code_1, code_2, code_3, &dis0, &dis1, &dis2, &dis3);
            auto query_len = quant.code_len(query_code.data());
            auto query_l1_norm = quant.code_len(query_code.data());
            auto query_bias = quant.code_bias(query_code.data());
            
            dis0 = dis0 * query_len * (float)quant.code_len(code_0);
            dis1 = dis1 * query_len * (float)quant.code_len(code_1);
            dis2 = dis2 * query_len * (float)quant.code_len(code_2);
            dis3 = dis3 * query_len * (float)quant.code_len(code_3);
            dis0 += query_l1_norm * query_len * quant.code_bias(code_0) + l1_norm[0] * quant.code_len(code_0) * query_bias;
            dis1 += query_l1_norm * query_len * quant.code_bias(code_1) + l1_norm[1] * quant.code_len(code_1) * query_bias;
            dis2 += query_l1_norm * query_len * quant.code_bias(code_2) + l1_norm[2] * quant.code_len(code_2) * query_bias;
            dis3 += query_l1_norm * query_len * quant.code_bias(code_3) + l1_norm[3] * quant.code_len(code_3) * query_bias;
            dis0 += query_bias * quant.code_bias(code_0) * quant.d;
            dis1 += query_bias * quant.code_bias(code_1) * quant.d;
            dis2 += query_bias * quant.code_bias(code_2) * quant.d;
            dis3 += query_bias * quant.code_bias(code_3) * quant.d;
            if constexpr (std::is_same_v<Similarity, SimilarityL2_avx<8>>) {
                auto query_l2_norm = quant.l2_norm(query_code.data());
                dis0 = query_l2_norm + quant.l2_norm(code_0) - 2 * dis0;
                dis1 = query_l2_norm + quant.l2_norm(code_1) - 2 * dis1;
                dis2 = query_l2_norm + quant.l2_norm(code_2) - 2 * dis2;
                dis3 = query_l2_norm + quant.l2_norm(code_3) - 2 * dis3;
            } 
    }
    void set_query(const float* x) final {
        q = x;
        quant.encode_vector(x, query_code.data());
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
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
                    QuantizerTemplate_avx<Codec8bit_avx, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit_uniform:
            return new DCTemplate_avx<
                    QuantizerTemplate_avx<Codec4bit_avx, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_8bit:
            return new DCTemplate_avx<
                    QuantizerTemplate_avx<Codec8bit_avx, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_6bit:
            return new DCTemplate_avx<
                    QuantizerTemplate_avx<Codec6bit_avx, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit:
            return new DCTemplate_avx<
                    QuantizerTemplate_avx<Codec4bit_avx, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_fp16:
            return new DCTemplate_avx<
                    QuantizerFP16_avx<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);
        
        case QuantizerType::QT_8bit_in_row:
            return new DCTemplate_avx<
                    Quantizer8bitInRow_avx<SIMDWIDTH>,
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

template <class Similarity, class Codec, bool uniform>
InvertedListScanner* sel12_InvertedListScanner_avx(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate_avx<Codec, uniform, SIMDWIDTH>;
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
                    true>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit_uniform:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec4bit_avx,
                    true>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_8bit:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec8bit_avx,
                    false>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec4bit_avx,
                    false>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_6bit:
            return sel12_InvertedListScanner_avx<
                    Similarity,
                    Codec6bit_avx,
                    false>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_fp16:
            return sel2_InvertedListScanner_avx<DCTemplate_avx<
                    QuantizerFP16_avx<SIMDWIDTH>,
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
        case QuantizerType::QT_8bit_in_row:
             return sel2_InvertedListScanner_avx<DCTemplate_avx<
                    Quantizer8bitInRow_avx<SIMDWIDTH>,
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
