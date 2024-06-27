/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arm_neon.h>

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

struct Codec8bit_neon : public Codec8bit {
    static FAISS_ALWAYS_INLINE float32x4x2_t
    decode_8_components(const uint8_t* code, int i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] = decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return {res1, res2};
    }
};

struct Codec4bit_neon : public Codec4bit {
    static FAISS_ALWAYS_INLINE float32x4x2_t
    decode_8_components(const uint8_t* code, int i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] = decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return {res1, res2};
    }
};

struct Codec6bit_neon : public Codec6bit {
    static FAISS_ALWAYS_INLINE float32x4x2_t
    decode_8_components(const uint8_t* code, int i) {
        float32_t result[8] = {};
        for (size_t j = 0; j < 8; j++) {
            result[j] = decode_component(code, i + j);
        }
        float32x4_t res1 = vld1q_f32(result);
        float32x4_t res2 = vld1q_f32(result + 4);
        return {res1, res2};
    }
};

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

template <class Codec, bool uniform, int SIMD>
struct QuantizerTemplate_neon {};

template <class Codec>
struct QuantizerTemplate_neon<Codec, true, 1>
        : public QuantizerTemplate<Codec, true, 1> {
    QuantizerTemplate_neon(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, true, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_neon<Codec, true, 8>
        : public QuantizerTemplate<Codec, true, 1> {
    QuantizerTemplate_neon(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, true, 1>(d, trained) {}

    FAISS_ALWAYS_INLINE float32x4x2_t
    reconstruct_8_components(const uint8_t* code, int i) const {
        float32x4x2_t xi = Codec::decode_8_components(code, i);
        return {
                vfmaq_f32(
                        vdupq_n_f32(this->vmin),
                        xi.val[0],
                        vdupq_n_f32(this->vdiff)),
                vfmaq_f32(
                        vdupq_n_f32(this->vmin),
                        xi.val[1],
                        vdupq_n_f32(this->vdiff))
        };
    }
};

template <class Codec>
struct QuantizerTemplate_neon<Codec, false, 1>
        : public QuantizerTemplate<Codec, false, 1> {
    QuantizerTemplate_neon(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, false, 1>(d, trained) {}
};

template <class Codec>
struct QuantizerTemplate_neon<Codec, false, 8>
        : public QuantizerTemplate<Codec, false, 1> {
    QuantizerTemplate_neon(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, false, 1>(d, trained) {}

    FAISS_ALWAYS_INLINE float32x4x2_t
    reconstruct_8_components(const uint8_t* code, int i) const {
        float32x4x2_t xi = Codec::decode_8_components(code, i);

        float32x4x2_t vmin_8 = vld1q_f32_x2(this->vmin + i);
        float32x4x2_t vdiff_8 = vld1q_f32_x2(this->vdiff + i);

        return {
                vfmaq_f32(vmin_8.val[0], xi.val[0], vdiff_8.val[0]),
                vfmaq_f32(vmin_8.val[1], xi.val[1], vdiff_8.val[1])
        };
    }
};

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerFP16_neon {};

template <>
struct QuantizerFP16_neon<1> : public QuantizerFP16<1> {
    QuantizerFP16_neon(size_t d, const std::vector<float>& unused)
            : QuantizerFP16<1>(d, unused) {}
};

template <>
struct QuantizerFP16_neon<8> : public QuantizerFP16<1> {
    QuantizerFP16_neon(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE float32x4x2_t
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint16x4x2_t codei = vld1_u16_x2((const uint16_t*)(code + 2 * i));
        return {vcvt_f32_f16(vreinterpret_f16_u16(codei.val[0])),
                vcvt_f32_f16(vreinterpret_f16_u16(codei.val[1]))};
    }
};

/*******************************************************************
 * BF16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerBF16_neon {};

template <>
struct QuantizerBF16_neon<1> : public QuantizerBF16<1> {
    QuantizerBF16_neon(size_t d, const std::vector<float>& unused)
            : QuantizerBF16<1>(d, unused) {}
};

template <>
struct QuantizerBF16_neon<8> : public QuantizerBF16<1> {
    QuantizerBF16_neon(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE float32x4x2_t
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint16x4x2_t codei = vld1_u16_x2((const uint16_t*)(code + 2 * i));
        return {vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(codei.val[0]), 16)),
                vreinterpretq_f32_u32(
                        vshlq_n_u32(vmovl_u16(codei.val[1]), 16))};
    }
};

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirect_neon {};

template <>
struct Quantizer8bitDirect_neon<1> : public Quantizer8bitDirect<1> {
    Quantizer8bitDirect_neon(size_t d, const std::vector<float>& unused)
            : Quantizer8bitDirect(d, unused) {}
};

template <>
struct Quantizer8bitDirect_neon<8> : public Quantizer8bitDirect<1> {
    Quantizer8bitDirect_neon(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<1>(d, trained) {}

    FAISS_ALWAYS_INLINE float32x4x2_t
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint8x8_t x8 = vld1_u8((const uint8_t*)(code + i));
        uint16x8_t y8 = vmovl_u8(x8);
        uint16x4_t y8_0 = vget_low_u16(y8);
        uint16x4_t y8_1 = vget_high_u16(y8);

        // convert uint16 -> uint32 -> fp32
        return {vcvtq_f32_u32(vmovl_u16(y8_0)), vcvtq_f32_u32(vmovl_u16(y8_1))};
    }
};

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirectSigned_neon {};

template <>
struct Quantizer8bitDirectSigned_neon<1> : public Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned_neon(size_t d, const std::vector<float>& unused)
            : Quantizer8bitDirectSigned(d, unused) {}
};

template <>
struct Quantizer8bitDirectSigned_neon<8> : public Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned_neon(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<1>(d, trained) {}

    FAISS_ALWAYS_INLINE float32x4x2_t
    reconstruct_8_components(const uint8_t* code, int i) const {
        uint8x8_t x8 = vld1_u8((const uint8_t*)(code + i));
        uint16x8_t y8 = vmovl_u8(x8); // convert uint8 -> uint16
        uint16x4_t y8_0 = vget_low_u16(y8);
        uint16x4_t y8_1 = vget_high_u16(y8);

        float32x4_t z8_0 = vcvtq_f32_u32(
                vmovl_u16(y8_0)); // convert uint16 -> uint32 -> fp32
        float32x4_t z8_1 = vcvtq_f32_u32(vmovl_u16(y8_1));

        // subtract 128 to convert into signed numbers
        return {vsubq_f32(z8_0, vmovq_n_f32(128.0)),
                vsubq_f32(z8_1, vmovq_n_f32(128.0))};
    }
};

template <int SIMDWIDTH>
SQuantizer* select_quantizer_1_neon(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case QuantizerType::QT_8bit:
            return new QuantizerTemplate_neon<Codec8bit_neon, false, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_6bit:
            return new QuantizerTemplate_neon<Codec6bit_neon, false, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_4bit:
            return new QuantizerTemplate_neon<Codec4bit_neon, false, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_8bit_uniform:
            return new QuantizerTemplate_neon<Codec8bit_neon, true, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_4bit_uniform:
            return new QuantizerTemplate_neon<Codec4bit_neon, true, SIMDWIDTH>(
                    d, trained);
        case QuantizerType::QT_fp16:
            return new QuantizerFP16_neon<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_bf16:
            return new QuantizerBF16_neon<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct:
            return new Quantizer8bitDirect_neon<SIMDWIDTH>(d, trained);
        case QuantizerType::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned_neon<SIMDWIDTH>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
}

/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object. The data fields just encapsulate
 * an accumulator.
 */

template <int SIMDWIDTH>
struct SimilarityL2_neon {};

template <>
struct SimilarityL2_neon<1> : public SimilarityL2<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    explicit SimilarityL2_neon(const float* y) : SimilarityL2<1>(y) {}
};

template <>
struct SimilarityL2_neon<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2_neon(const float* y) : y(y) {}
    float32x4x2_t accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8 = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(float32x4x2_t x) {
        float32x4x2_t yiv = vld1q_f32_x2(yi);
        yi += 8;

        float32x4_t sub0 = vsubq_f32(yiv.val[0], x.val[0]);
        float32x4_t sub1 = vsubq_f32(yiv.val[1], x.val[1]);

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], sub0, sub0);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], sub1, sub1);

        accu8 = {accu8_0, accu8_1};
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(
            float32x4x2_t x,
            float32x4x2_t y) {
        float32x4_t sub0 = vsubq_f32(y.val[0], x.val[0]);
        float32x4_t sub1 = vsubq_f32(y.val[1], x.val[1]);

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], sub0, sub0);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], sub1, sub1);

        accu8 = {accu8_0, accu8_1};
    }

    FAISS_ALWAYS_INLINE float result_8() {
        float32x4_t sum_0 = vpaddq_f32(accu8.val[0], accu8.val[0]);
        float32x4_t sum_1 = vpaddq_f32(accu8.val[1], accu8.val[1]);

        float32x4_t sum2_0 = vpaddq_f32(sum_0, sum_0);
        float32x4_t sum2_1 = vpaddq_f32(sum_1, sum_1);
        return vgetq_lane_f32(sum2_0, 0) + vgetq_lane_f32(sum2_1, 0);
    }
};

template <int SIMDWIDTH>
struct SimilarityIP_neon {};

template <>
struct SimilarityIP_neon<1> : public SimilarityIP<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    explicit SimilarityIP_neon(const float* y) : SimilarityIP<1>(y) {}
};

template <>
struct SimilarityIP_neon<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP_neon(const float* y) : y(y) {}

    float32x4x2_t accu8;

    FAISS_ALWAYS_INLINE void begin_8() {
        accu8 = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_8_components(float32x4x2_t x) {
        float32x4x2_t yiv = vld1q_f32_x2(yi);
        yi += 8;

        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], yiv.val[0], x.val[0]);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], yiv.val[1], x.val[1]);
        accu8 = {accu8_0, accu8_1};
    }

    FAISS_ALWAYS_INLINE void add_8_components_2(float32x4x2_t x1, float32x4x2_t x2) {
        float32x4_t accu8_0 = vfmaq_f32(accu8.val[0], x1.val[0], x2.val[0]);
        float32x4_t accu8_1 = vfmaq_f32(accu8.val[1], x1.val[1], x2.val[1]);
        accu8 = {accu8_0, accu8_1};
    }

    FAISS_ALWAYS_INLINE float result_8() {
        float32x4x2_t sum = {
                vpaddq_f32(accu8.val[0], accu8.val[0]),
                vpaddq_f32(accu8.val[1], accu8.val[1])
        };
        float32x4x2_t sum2 = {
                vpaddq_f32(sum.val[0], sum.val[0]),
                vpaddq_f32(sum.val[1], sum.val[1])
        };
        return vgetq_lane_f32(sum2.val[0], 0) + vgetq_lane_f32(sum2.val[1], 0);
    }
};

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, int SIMDWIDTH>
struct DCTemplate_neon : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate_neon<Quantizer, Similarity, 1>
        : public DCTemplate<Quantizer, Similarity, 1> {
    DCTemplate_neon(size_t d, const std::vector<float>& trained)
            : DCTemplate<Quantizer, Similarity, 1>(d, trained) {}
};

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <class Quantizer, class Similarity>
struct DCTemplate_neon<Quantizer, Similarity, 8> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate_neon(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            float32x4x2_t xi = quant.reconstruct_8_components(code, i);
            sim.add_8_components(xi);
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            float32x4x2_t x1 = quant.reconstruct_8_components(code1, i);
            float32x4x2_t x2 = quant.reconstruct_8_components(code2, i);
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
            float32x4x2_t xi0 = quant.reconstruct_8_components(code_0, i);
            float32x4x2_t xi1 = quant.reconstruct_8_components(code_1, i);
            float32x4x2_t xi2 = quant.reconstruct_8_components(code_2, i);
            float32x4x2_t xi3 = quant.reconstruct_8_components(code_3, i);
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
struct DistanceComputerByte_neon : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte_neon<Similarity, 1>
        : public DistanceComputerByte<Similarity, 1> {
    DistanceComputerByte_neon(int d, const std::vector<float>& unused)
            : DistanceComputerByte<Similarity, 1>(d, unused) {}
};

template <class Similarity>
struct DistanceComputerByte_neon<Similarity, 8> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte_neon(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query(const float* x) final {
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
SQDistanceComputer* select_distance_computer_neon(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr int SIMDWIDTH = Sim::simdwidth;
    switch (qtype) {
        case QuantizerType::QT_8bit_uniform:
            return new DCTemplate_neon<
                    QuantizerTemplate_neon<Codec8bit_neon, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit_uniform:
            return new DCTemplate_neon<
                    QuantizerTemplate_neon<Codec4bit_neon, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_8bit:
            return new DCTemplate_neon<
                    QuantizerTemplate_neon<Codec8bit_neon, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_6bit:
            return new DCTemplate_neon<
                    QuantizerTemplate_neon<Codec6bit_neon, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_4bit:
            return new DCTemplate_neon<
                    QuantizerTemplate_neon<Codec4bit_neon, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_fp16:
            return new DCTemplate_neon<
                    QuantizerFP16_neon<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_bf16:
            return new DCTemplate_neon<
                    QuantizerBF16_neon<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case QuantizerType::QT_8bit_direct:
            if (d % 16 == 0) {
                return new DistanceComputerByte_neon<Sim, SIMDWIDTH>(d, trained);
            } else {
                return new DCTemplate_neon<
                        Quantizer8bitDirect_neon<SIMDWIDTH>,
                        Sim,
                        SIMDWIDTH>(d, trained);
            }

        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate_neon<
                    Quantizer8bitDirectSigned_neon<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <class DCClass>
InvertedListScanner* sel2_InvertedListScanner_neon(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity, class Codec, bool uniform>
InvertedListScanner* sel12_InvertedListScanner_neon(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate_neon<Codec, uniform, SIMDWIDTH>;
    using DCClass = DCTemplate_neon<QuantizerClass, Similarity, SIMDWIDTH>;
    return sel2_InvertedListScanner_neon<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity>
InvertedListScanner* sel1_InvertedListScanner_neon(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    switch (sq->qtype) {
        case QuantizerType::QT_8bit_uniform:
            return sel12_InvertedListScanner_neon<
                    Similarity,
                    Codec8bit_neon,
                    true>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit_uniform:
            return sel12_InvertedListScanner_neon<
                    Similarity,
                    Codec4bit_neon,
                    true>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_8bit:
            return sel12_InvertedListScanner_neon<
                    Similarity,
                    Codec8bit_neon,
                    false>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_4bit:
            return sel12_InvertedListScanner_neon<
                    Similarity,
                    Codec4bit_neon,
                    false>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_6bit:
            return sel12_InvertedListScanner_neon<
                    Similarity,
                    Codec6bit_neon,
                    false>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_fp16:
            return sel2_InvertedListScanner_neon<DCTemplate_neon<
                    QuantizerFP16_neon<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_bf16:
            return sel2_InvertedListScanner_neon<DCTemplate_neon<
                    QuantizerBF16_neon<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_8bit_direct:
            if (sq->d % 16 == 0) {
                return sel2_InvertedListScanner_neon<
                        DistanceComputerByte_neon<Similarity, SIMDWIDTH>>(
                        sq, quantizer, store_pairs, sel, r);
            } else {
                return sel2_InvertedListScanner_neon<DCTemplate_neon<
                        Quantizer8bitDirect_neon<SIMDWIDTH>,
                        Similarity,
                        SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
            }
        case ScalarQuantizer::QT_8bit_direct_signed:
            return sel2_InvertedListScanner_neon<DCTemplate_neon<
                    Quantizer8bitDirectSigned_neon<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
    }

    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <int SIMDWIDTH>
InvertedListScanner* sel0_InvertedListScanner_neon(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner_neon<SimilarityL2_neon<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner_neon<SimilarityIP_neon<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

} // namespace faiss
