/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <iostream>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizerOp.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/ScalarQuantizerScanner.h>

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

struct Codec8bit {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        code[i] = (int)(255 * x);
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        return (code[i] + 0.5f) / 255.0f;
    }

    static FAISS_ALWAYS_INLINE int part_num() {
        return 255;
    }

    static FAISS_ALWAYS_INLINE float decode_bias() {
        return 0.5 / 255.0f;
    }
};

struct Codec4bit {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        code[i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }
};

struct Codec6bit {
    static FAISS_ALWAYS_INLINE void encode_component(
            float x,
            uint8_t* code,
            int i) {
        int bits = (int)(x * 63.0);
        code += (i >> 2) * 3;
        switch (i & 3) {
            case 0:
                code[0] |= bits;
                break;
            case 1:
                code[0] |= bits << 6;
                code[1] |= bits >> 2;
                break;
            case 2:
                code[1] |= bits << 4;
                code[2] |= bits >> 4;
                break;
            case 3:
                code[2] |= bits << 2;
                break;
        }
    }

    static FAISS_ALWAYS_INLINE float decode_component(
            const uint8_t* code,
            int i) {
        uint8_t bits = 0x00;
        code += (i >> 2) * 3;
        switch (i & 3) {
            case 0:
                bits = code[0] & 0x3f;
                break;
            case 1:
                bits = code[0] >> 6;
                bits |= (code[1] & 0xf) << 2;
                break;
            case 2:
                bits = code[1] >> 4;
                bits |= (code[2] & 3) << 4;
                break;
            case 3:
                bits = code[2] >> 2;
                break;
        }
        return (bits + 0.5f) / 63.0f;
    }
};

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

template <class Codec, bool uniform, int SIMD>
struct QuantizerTemplate {};

template <class Codec>
struct QuantizerTemplate<Codec, true, 1> : SQuantizer {
    const size_t d;
    const float vmin, vdiff;

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : d(d), vmin(trained[0]), vdiff(trained[1]) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = 0;
            if (vdiff != 0) {
                xi = (x[i] - vmin) / vdiff;
                if (xi < 0) {
                    xi = 0;
                }
                if (xi > 1.0) {
                    xi = 1.0;
                }
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin + xi * vdiff;
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        float xi = Codec::decode_component(code, i);
        return vmin + xi * vdiff;
    }
};

template <class Codec>
struct QuantizerTemplate<Codec, false, 1> : SQuantizer {
    const size_t d;
    const float *vmin, *vdiff;

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : d(d), vmin(trained.data()), vdiff(trained.data() + d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = 0;
            if (vdiff[i] != 0) {
                xi = (x[i] - vmin[i]) / vdiff[i];
                if (xi < 0) {
                    xi = 0;
                }
                if (xi > 1.0) {
                    xi = 1.0;
                }
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin[i] + xi * vdiff[i];
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        float xi = Codec::decode_component(code, i);
        return vmin[i] + xi * vdiff[i];
    }
};

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerFP16 {};

template <>
struct QuantizerFP16<1> : SQuantizer {
    const size_t d;

    QuantizerFP16(size_t d, const std::vector<float>& /* unused */) : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            ((uint16_t*)code)[i] = encode_fp16(x[i]);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = decode_fp16(((uint16_t*)code)[i]);
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return decode_fp16(((uint16_t*)code)[i]);
    }
};

/*******************************************************************
 * 8bit_in_row quantizer
 *******************************************************************/
template <int SIMDWIDTH>
struct Quantizer8bitInRow {};

template <>
struct Quantizer8bitInRow<1> : SQuantizer {
    const size_t d;

    Quantizer8bitInRow(size_t d)
            : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        float vmin = HUGE_VAL;
        float vmax = -HUGE_VAL;
        float l2_norm = 0.0;
        for (size_t i = 0; i < d; i++) {
            if (x[i] < vmin)
                vmin = x[i];
            if (x[i] > vmax)
                vmax = x[i];
            l2_norm += x[i] * x[i];
        }
        auto vdiff = vmax - vmin;
        *((float*)code) = vmin;
        *(float*)(code + 4) = vdiff;
        *(float*)(code + 8) = l2_norm;
        uint8_t* code_data = code + 12;
        for (size_t i = 0; i < d; i++) {
            float xi = (x[i] - vmin)/ vdiff;
            Codec8bit::encode_component(xi, code_data, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        float vmin = *((float*)code);
        float vdiff = *(float*)(code + 4);
        const uint8_t* code_data = code + 12;
        for (size_t i = 0; i < d; i++) {
            auto xi = Codec8bit::decode_component(code_data, i);
            x[i] = vmin + xi * vdiff;
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        float vmin = *((float*)code);
        float vdiff = *(float*)(code + 4);
        const uint8_t* code_data = code + 12;

        auto xi = Codec8bit::decode_component(code_data, i);
        xi = vmin + xi * vdiff;
        return xi;
    }

    FAISS_ALWAYS_INLINE float code_bias(const uint8_t* code) const {
        float vmin = *((float*)code);
        float vdiff = *(float*)(code + 4);
        return vmin + vdiff * Codec8bit::decode_bias();
    }

    FAISS_ALWAYS_INLINE float code_len(const uint8_t* code) const {
        float vdiff = *(float*)(code + 4);
        return vdiff/float(Codec8bit::part_num());
    }

    FAISS_ALWAYS_INLINE float l2_norm(const uint8_t* code) const {
        return *(float*)(code + 8);
    }
    
    FAISS_ALWAYS_INLINE uint32_t sum(const uint8_t* code) const {
        const uint8_t* code_data = code + 12;
        uint32_t sum = 0;
        for (int i = 0; i < d; i++) {
            sum += (uint32_t)code_data[i];
        }
        return sum;
    }

    FAISS_ALWAYS_INLINE const float u8_dot_product(const uint8_t* code1, const uint8_t* code2) const {
        const uint8_t* code1_data = code1 + 12;
        const uint8_t* code2_data = code2 + 12;
        float res = 0.0;
        for (auto i = 0; i < d; i++) {
            res+= (float)code1_data[i] * (float)code2_data[i];
        } 
        return res;
    }
};

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirect {};

template <>
struct Quantizer8bitDirect<1> : SQuantizer {
    const size_t d;

    Quantizer8bitDirect(size_t d, const std::vector<float>& /* unused */)
            : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            code[i] = (uint8_t)x[i];
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = code[i];
        }
    }

    FAISS_ALWAYS_INLINE float reconstruct_component(const uint8_t* code, int i)
            const {
        return code[i];
    }
};

template <int SIMDWIDTH>
SQuantizer* select_quantizer_1(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerTemplate<Codec8bit, false, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_6bit:
            return new QuantizerTemplate<Codec6bit, false, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerTemplate<Codec4bit, false, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerTemplate<Codec8bit, true, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerTemplate<Codec4bit, true, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_fp16:
            return new QuantizerFP16<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_8bit_direct:
            return new Quantizer8bitDirect<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_8bit_in_row:
            return new Quantizer8bitInRow<SIMDWIDTH>(d);
    }
    FAISS_THROW_MSG("unknown qtype");
}

/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object. The data fields just encapsulate
 * an accumulator.
 */

template <int SIMDWIDTH>
struct SimilarityL2 {};

template <>
struct SimilarityL2<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}

    /******* scalar accumulator *******/

    float accu;

    FAISS_ALWAYS_INLINE void begin() {
        accu = 0;
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_component(float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    FAISS_ALWAYS_INLINE void add_component_2(float x1, float x2) {
        float tmp = x1 - x2;
        accu += tmp * tmp;
    }

    FAISS_ALWAYS_INLINE float result() {
        return accu;
    }
};

template <int SIMDWIDTH>
struct SimilarityIP {};

template <>
struct SimilarityIP<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;
    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    FAISS_ALWAYS_INLINE void begin() {
        accu = 0;
        yi = y;
    }

    FAISS_ALWAYS_INLINE void add_component(float x) {
        accu += *yi++ * x;
    }

    FAISS_ALWAYS_INLINE void add_component_2(float x1, float x2) {
        accu += x1 * x2;
    }

    FAISS_ALWAYS_INLINE float result() {
        return accu;
    }
};

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, int SIMDWIDTH>
struct DCTemplate : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, 1> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
            sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_distance(q, code);
    }
};

template <class Similarity>
struct DCTemplate<Quantizer8bitInRow<Similarity::simdwidth>, Similarity, 1> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer8bitInRow<Similarity::simdwidth> quant;
    std::vector<uint8_t> query_code;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d), query_code(d + 12) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2) const {
        auto encode_distace_ip = [&](const uint8_t* code1, const uint8_t* code2) -> float {
            // get encode ip distance:
            // divide into 4 part:
            // decode(x)*decode(y)
            // = sum(x_code*code_len(x) + code_bias(x))*(y_code*code_len(y) + code_bias(y))
            // = x_code*y_code*code_len(x)*code_len(y) + sum(x_code)*code_len(x)*code_bias(y) + code_bias(x)*sum(y_code)*code_len(y) + code_bias(x)*code_bias(y)
            // part1: x_code * y_code * code_len(x) * code_len(y)
            // part2: sum(x_code) * code_len(x) * code_bias(y)
            // part3: sum(y_code) * code_len(y) * code_bias(x)
            // part4: code_bias(x) * code_bias(y) * d
            float part1 = quant.u8_dot_product(code1, code2) * quant.code_len(code1) * quant.code_len(code2);
            float part2 = (float)quant.sum(code1) * quant.code_len(code1) * quant.code_bias(code2);
            float part3 = (float)quant.sum(code2) * quant.code_len(code2) * quant.code_bias(code1);
            float part4 = quant.code_bias(code1) *  quant.code_bias(code2) * quant.d;
            float res =  part1 + part2 + part3 + part4;
            return res;
        };
        if constexpr (std::is_same_v<Similarity, SimilarityL2<1>>) {
            // get encode l2 distace:
            // = x*x + y*y - 2*x*y
            // = x*x + y*y - 2*encode_distace_ip(x, y)
            auto res = quant.l2_norm(code1) + quant.l2_norm(code2); 
            float dot_product = encode_distace_ip(code1, code2);
            return res - 2.0 * dot_product;
        }
        if constexpr (std::is_same_v<Similarity, SimilarityIP<1>>) {
            return encode_distace_ip(code1, code2);
        } 
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
            sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query(const float* x) final {
        q = x;
        quant.encode_vector(x, query_code.data());
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_code_distance(query_code.data(), code);
    }
};

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, int SIMDWIDTH>
struct DistanceComputerByte : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte<Similarity, 1> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

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
SQDistanceComputer* select_distance_computer(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr int SIMDWIDTH = Sim::simdwidth;
    switch (qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<Codec8bit, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<Codec4bit, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_8bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec8bit, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec6bit, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec4bit, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_fp16:
            return new DCTemplate<QuantizerFP16<SIMDWIDTH>, Sim, SIMDWIDTH>(
                    d, trained);

        case ScalarQuantizer::QT_8bit_direct:
            if (d % 16 == 0) {
                return new DistanceComputerByte<Sim, SIMDWIDTH>(d, trained);
            } else {
                return new DCTemplate<
                        Quantizer8bitDirect<SIMDWIDTH>,
                        Sim,
                        SIMDWIDTH>(d, trained);
            }
        case ScalarQuantizer::QT_8bit_in_row:
            return new DCTemplate<
                    Quantizer8bitInRow<SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <class DCClass, int use_sel>
InvertedListScanner* sel3_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (DCClass::Sim::metric_type == METRIC_L2) {
        return new IVFSQScannerL2<DCClass, use_sel>(
                sq->d,
                sq->trained,
                sq->code_size,
                quantizer,
                store_pairs,
                sel,
                r);
    } else if (DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
        return new IVFSQScannerIP<DCClass, use_sel>(
                sq->d, sq->trained, sq->code_size, store_pairs, sel, r);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

template <class DCClass>
InvertedListScanner* sel2_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (sel) {
        if (store_pairs) {
            return sel3_InvertedListScanner<DCClass, 2>(
                    sq, quantizer, store_pairs, sel, r);
        } else {
            return sel3_InvertedListScanner<DCClass, 1>(
                    sq, quantizer, store_pairs, sel, r);
        }
    } else {
        return sel3_InvertedListScanner<DCClass, 0>(
                sq, quantizer, store_pairs, sel, r);
    }
}

template <class Similarity, class Codec, bool uniform>
InvertedListScanner* sel12_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate<Codec, uniform, SIMDWIDTH>;
    using DCClass = DCTemplate<QuantizerClass, Similarity, SIMDWIDTH>;
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity>
InvertedListScanner* sel1_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    switch (sq->qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return sel12_InvertedListScanner<Similarity, Codec8bit, true>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit_uniform:
            return sel12_InvertedListScanner<Similarity, Codec4bit, true>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit:
            return sel12_InvertedListScanner<Similarity, Codec8bit, false>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit:
            return sel12_InvertedListScanner<Similarity, Codec4bit, false>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_6bit:
            return sel12_InvertedListScanner<Similarity, Codec6bit, false>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_fp16:
            return sel2_InvertedListScanner<DCTemplate<
                    QuantizerFP16<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit_direct:
            if (sq->d % 16 == 0) {
                return sel2_InvertedListScanner<
                        DistanceComputerByte<Similarity, SIMDWIDTH>>(
                        sq, quantizer, store_pairs, sel, r);
            } else {
                return sel2_InvertedListScanner<DCTemplate<
                        Quantizer8bitDirect<SIMDWIDTH>,
                        Similarity,
                        SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
            }
        case ScalarQuantizer::QT_8bit_in_row:
            return sel2_InvertedListScanner<DCTemplate<
                    Quantizer8bitInRow<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);

    }

    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <int SIMDWIDTH>
InvertedListScanner* sel0_InvertedListScanner(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner<SimilarityL2<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner<SimilarityIP<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

} // namespace faiss
