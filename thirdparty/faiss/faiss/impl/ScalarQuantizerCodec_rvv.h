/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(__riscv_vector)

#include <riscv_vector.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ScalarQuantizerCodec.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/utils.h>

namespace faiss {

using QuantizerType = ScalarQuantizer::QuantizerType;
using RangeStat = ScalarQuantizer::RangeStat;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;
using SQuantizer = ScalarQuantizer::SQuantizer;

inline size_t get_vlen_f32_m1() {
    return __riscv_vsetvlmax_e32m1();
}
inline size_t get_vlen_f32_m2() {
    return __riscv_vsetvlmax_e32m2();
}
inline size_t get_vlen_f32_m4() {
    return __riscv_vsetvlmax_e32m4();
}

/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

constexpr size_t RVV_CODEC_STACK_THRESHOLD = 512;

struct Codec8bit_rvv : public Codec8bit {
    static FAISS_ALWAYS_INLINE vfloat32m4_t
    decode_components(const uint8_t* code, int i, size_t vl) {
        vuint8m1_t v_u8 = __riscv_vle8_v_u8m1(code + i, vl);
        vuint16m2_t v_u16 = __riscv_vwcvtu_x_x_v_u16m2(v_u8, vl);
        vuint32m4_t v_u32 = __riscv_vwcvtu_x_x_v_u32m4(v_u16, vl);
        vfloat32m4_t v_f32 = __riscv_vfcvt_f_xu_v_f32m4(v_u32, vl);
        vfloat32m4_t one_255 = __riscv_vfmv_v_f_f32m4(1.0f / 255.0f, vl);
        vfloat32m4_t half_one_255 = __riscv_vfmv_v_f_f32m4(0.5f / 255.0f, vl);
        return __riscv_vfmadd_vv_f32m4(v_f32, one_255, half_one_255, vl);
    }
};

struct Codec4bit_rvv : public Codec4bit {
    static FAISS_ALWAYS_INLINE vfloat32m4_t
    decode_components(const uint8_t* code, int i, size_t vl) {
        auto process = [&](uint32_t* unpacked_buf) -> vfloat32m4_t {
            for (size_t j = 0; j < vl; ++j) {
                size_t current_idx = static_cast<size_t>(i) + j;
                const uint8_t byte = code[current_idx / 2];
                unpacked_buf[j] =
                        (current_idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            }
            vuint32m4_t v_u32 = __riscv_vle32_v_u32m4(unpacked_buf, vl);
            vfloat32m4_t v_f32 = __riscv_vfcvt_f_xu_v_f32m4(v_u32, vl);
            vfloat32m4_t one_15 = __riscv_vfmv_v_f_f32m4(1.0f / 15.0f, vl);
            vfloat32m4_t half = __riscv_vfmv_v_f_f32m4(0.5f, vl);
            vfloat32m4_t temp_sum = __riscv_vfadd_vv_f32m4(v_f32, half, vl);
            return __riscv_vfmul_vv_f32m4(temp_sum, one_15, vl);
        };

        if (vl <= RVV_CODEC_STACK_THRESHOLD) {
            std::array<uint32_t, RVV_CODEC_STACK_THRESHOLD> stack_buf{};
            return process(stack_buf.data());
        } else {
            std::vector<uint32_t> heap_buf(vl);
            return process(heap_buf.data());
        }
    }
};

struct Codec6bit_rvv : public Codec6bit {
    static FAISS_ALWAYS_INLINE void decode_components(
            const uint8_t* code,
            int i,
            size_t vl,
            float* out) {
        const size_t max_chunk = __riscv_vsetvlmax_e32m4();

        std::array<uint32_t, RVV_CODEC_STACK_THRESHOLD> unpacked_buf;
        FAISS_THROW_IF_NOT_MSG(
                max_chunk <= RVV_CODEC_STACK_THRESHOLD,
                "RVV max_chunk exceeds stack buffer");

        size_t offset = 0;
        while (offset < vl) {
            const size_t chunk_vl = std::min(vl - offset, max_chunk);

            for (size_t j = 0; j < chunk_vl; ++j) {
                size_t abs_i = static_cast<size_t>(i) + offset + j;
                size_t tab = abs_i / 4;
                size_t q = abs_i % 4;
                const uint8_t* p_grp = code + tab * 3;
                uint32_t x4 = 0;
                if (q == 0) {
                    x4 = p_grp[0] & 0x3F;
                } else if (q == 1) {
                    x4 = ((p_grp[0] >> 6) | (p_grp[1] << 2)) & 0x3F;
                } else if (q == 2) {
                    x4 = ((p_grp[1] >> 4) | (p_grp[2] << 4)) & 0x3F;
                } else {
                    x4 = (p_grp[2] >> 2) & 0x3F;
                }
                unpacked_buf[j] = x4;
            }

            vuint32m4_t v_u32 =
                    __riscv_vle32_v_u32m4(unpacked_buf.data(), chunk_vl);
            vfloat32m4_t v_f32 = __riscv_vfcvt_f_xu_v_f32m4(v_u32, chunk_vl);

            vfloat32m4_t one_63 =
                    __riscv_vfmv_v_f_f32m4(1.0f / 63.0f, chunk_vl);
            vfloat32m4_t half_one_63 =
                    __riscv_vfmv_v_f_f32m4(0.5f / 63.0f, chunk_vl);

            vfloat32m4_t chunk_result = __riscv_vfmadd_vv_f32m4(
                    v_f32, one_63, half_one_63, chunk_vl);

            __riscv_vse32_v_f32m4(out + offset, chunk_result, chunk_vl);

            offset += chunk_vl;
        }
    }
};

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/
template <class Codec, QuantizerTemplateScaling SCALING, int SIMD>
struct QuantizerTemplate_rvv {};

template <class Codec>
struct QuantizerTemplate_rvv<Codec, QuantizerTemplateScaling::UNIFORM, 0>
        : public QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::UNIFORM,
                  1> {
    QuantizerTemplate_rvv(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, QuantizerTemplateScaling::UNIFORM, 1>(
                      d,
                      trained) {}

    FAISS_ALWAYS_INLINE vfloat32m4_t
    reconstruct_components(const uint8_t* code, int i, size_t vl) const {
        vfloat32m4_t xi = Codec::decode_components(code, i, vl);

        vfloat32m4_t v_vmin = __riscv_vfmv_v_f_f32m4(this->vmin, vl);
        vfloat32m4_t v_vdiff = __riscv_vfmv_v_f_f32m4(this->vdiff, vl);

        return __riscv_vfmadd_vv_f32m4(xi, v_vdiff, v_vmin, vl);
    }
};

template <class Codec>
struct QuantizerTemplate_rvv<Codec, QuantizerTemplateScaling::NON_UNIFORM, 0>
        : public QuantizerTemplate<
                  Codec,
                  QuantizerTemplateScaling::NON_UNIFORM,
                  1> {
    QuantizerTemplate_rvv(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec,
                      QuantizerTemplateScaling::NON_UNIFORM,
                      1>(d, trained) {}
    FAISS_ALWAYS_INLINE vfloat32m4_t
    reconstruct_components(const uint8_t* code, int i, size_t vl) const {
        vfloat32m4_t xi = Codec::decode_components(code, i, vl);

        vfloat32m4_t v_vmin = __riscv_vle32_v_f32m4(this->vmin + i, vl);
        vfloat32m4_t v_vdiff = __riscv_vle32_v_f32m4(this->vdiff + i, vl);

        return __riscv_vfmadd_vv_f32m4(xi, v_vdiff, v_vmin, vl);
    }
};

template <>
struct QuantizerTemplate_rvv<
        Codec6bit_rvv,
        QuantizerTemplateScaling::NON_UNIFORM,
        0>
        : public QuantizerTemplate<
                  Codec6bit_rvv,
                  QuantizerTemplateScaling::NON_UNIFORM,
                  1> {
    QuantizerTemplate_rvv(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<
                      Codec6bit_rvv,
                      QuantizerTemplateScaling::NON_UNIFORM,
                      1>(d, trained) {}

    FAISS_ALWAYS_INLINE void reconstruct_components(
            const uint8_t* code,
            int i,
            size_t vl,
            float* out) const {
        Codec6bit_rvv::decode_components(code, i, vl, out);

        const size_t max_chunk = __riscv_vsetvlmax_e32m4();
        size_t offset = 0;

        while (offset < vl) {
            const size_t chunk_vl = std::min(vl - offset, max_chunk);

            vfloat32m4_t xi = __riscv_vle32_v_f32m4(out + offset, chunk_vl);

            vfloat32m4_t v_vmin =
                    __riscv_vle32_v_f32m4(this->vmin + i + offset, chunk_vl);
            vfloat32m4_t v_vdiff =
                    __riscv_vle32_v_f32m4(this->vdiff + i + offset, chunk_vl);

            vfloat32m4_t result =
                    __riscv_vfmadd_vv_f32m4(xi, v_vdiff, v_vmin, chunk_vl);

            __riscv_vse32_v_f32m4(out + offset, result, chunk_vl);

            offset += chunk_vl;
        }
    }
};

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerFP16_rvv {};

template <>
struct QuantizerFP16_rvv<1> : public QuantizerFP16<1> {
    QuantizerFP16_rvv(size_t d, const std::vector<float>& unused)
            : QuantizerFP16<1>(d, unused) {}
};

template <>
struct QuantizerFP16_rvv<0> : public QuantizerFP16<1> {
    QuantizerFP16_rvv(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m2_t
    reconstruct_components(const uint8_t* code, int i, size_t vl) const {
        const _Float16* code_ptr = reinterpret_cast<const _Float16*>(
                code + 2 * static_cast<size_t>(i));
        vfloat16m1_t v_f16 = __riscv_vle16_v_f16m1(code_ptr, vl);
        return __riscv_vfwcvt_f_f_v_f32m2(v_f16, vl);
    }
};

/*******************************************************************
 * BF16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerBF16_rvv {};

template <>
struct QuantizerBF16_rvv<1> : public QuantizerBF16<1> {
    QuantizerBF16_rvv(size_t d, const std::vector<float>& unused)
            : QuantizerBF16<1>(d, unused) {}
};

template <>
struct QuantizerBF16_rvv<0> : public QuantizerBF16<1> {
    QuantizerBF16_rvv(size_t d, const std::vector<float>& trained)
            : QuantizerBF16<1>(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m2_t
    reconstruct_components(const uint8_t* code, int i, size_t vl) const {
        const uint16_t* code_ptr = reinterpret_cast<const uint16_t*>(
                code + 2 * static_cast<size_t>(i));
        vuint16m1_t v_u16 = __riscv_vle16_v_u16m1(code_ptr, vl);
        vuint32m2_t v_u32 = __riscv_vwaddu_vx_u32m2(v_u16, 0, vl);
        vuint32m2_t v_shifted = __riscv_vsll_vx_u32m2(v_u32, 16, vl);
        return __riscv_vreinterpret_v_u32m2_f32m2(v_shifted);
    }
};

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirect_rvv {};
template <>
struct Quantizer8bitDirect_rvv<1> : public Quantizer8bitDirect<1> {
    Quantizer8bitDirect_rvv(size_t d, const std::vector<float>& u)
            : Quantizer8bitDirect(d, u) {}
};
template <>
struct Quantizer8bitDirect_rvv<0> : public Quantizer8bitDirect<1> {
    Quantizer8bitDirect_rvv(size_t d, const std::vector<float>& t)
            : Quantizer8bitDirect<1>(d, t) {}

    FAISS_ALWAYS_INLINE vfloat32m4_t
    reconstruct_components(const uint8_t* code, int i, size_t vl) const {
        vuint8m1_t v_u8 = __riscv_vle8_v_u8m1(code + i, vl);
        vuint16m2_t v_u16 = __riscv_vwcvtu_x_x_v_u16m2(v_u8, vl);
        vuint32m4_t v_u32 = __riscv_vwcvtu_x_x_v_u32m4(v_u16, vl);
        return __riscv_vfcvt_f_xu_v_f32m4(v_u32, vl);
    }
};

/*******************************************************************
 * 8bit_direct_signed quantizer
 *******************************************************************/
template <int SIMDWIDTH>
struct Quantizer8bitDirectSigned_rvv {};

template <>
struct Quantizer8bitDirectSigned_rvv<1> : public Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned_rvv(size_t d, const std::vector<float>& unused)
            : Quantizer8bitDirectSigned(d, unused) {}
};

template <>
struct Quantizer8bitDirectSigned_rvv<0> : public Quantizer8bitDirectSigned<1> {
    Quantizer8bitDirectSigned_rvv(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirectSigned<1>(d, trained) {}

    FAISS_ALWAYS_INLINE vfloat32m4_t
    reconstruct_components(const uint8_t* code, int i, size_t vl) const {
        vuint8m1_t v_u8 = __riscv_vle8_v_u8m1(code + i, vl);
        vuint16m2_t v_u16 = __riscv_vwcvtu_x_x_v_u16m2(v_u8, vl);
        vuint32m4_t v_u32 = __riscv_vwcvtu_x_x_v_u32m4(v_u16, vl);
        vfloat32m4_t v_f32 = __riscv_vfcvt_f_xu_v_f32m4(v_u32, vl);
        vfloat32m4_t c128 = __riscv_vfmv_v_f_f32m4(128.0f, vl);
        return __riscv_vfsub_vv_f32m4(v_f32, c128, vl);
    }
};

template <int SIMDWIDTH>
ScalarQuantizer::SQuantizer* select_quantizer_1_rvv(
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerTemplate_rvv<
                    Codec8bit_rvv,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(dim, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerTemplate_rvv<
                    Codec8bit_rvv,
                    QuantizerTemplateScaling::UNIFORM,
                    SIMDWIDTH>(dim, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerTemplate_rvv<
                    Codec4bit_rvv,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(dim, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerTemplate_rvv<
                    Codec4bit_rvv,
                    QuantizerTemplateScaling::UNIFORM,
                    SIMDWIDTH>(dim, trained);
        case ScalarQuantizer::QT_6bit:
            return new QuantizerTemplate_rvv<
                    Codec6bit_rvv,
                    QuantizerTemplateScaling::NON_UNIFORM,
                    SIMDWIDTH>(dim, trained);
        case ScalarQuantizer::QT_fp16:
            return new QuantizerFP16_rvv<SIMDWIDTH>(dim, trained);
        case ScalarQuantizer::QT_bf16:
            return new QuantizerBF16_rvv<SIMDWIDTH>(dim, trained);
        case ScalarQuantizer::QT_8bit_direct:
            return new Quantizer8bitDirect_rvv<SIMDWIDTH>(dim, trained);
        case ScalarQuantizer::QT_8bit_direct_signed:
            return new Quantizer8bitDirectSigned_rvv<SIMDWIDTH>(dim, trained);
        default:
            FAISS_THROW_FMT("Quantizer type %d not supported", qtype);
    }
    return nullptr;
}

/*******************************************************************
 * Similarity "Tags": Used as template parameters to select metric.
 * These are now stateless.
 *******************************************************************/

template <int SIMDWIDTH>
struct SimilarityL2_rvv {};
template <>
struct SimilarityL2_rvv<0> {
    static constexpr MetricType metric_type = METRIC_L2;
};

template <int SIMDWIDTH>
struct SimilarityIP_rvv {};
template <>
struct SimilarityIP_rvv<0> {
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;
};

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/
template <class Quantizer, class Similarity, int SIMDWIDTH>
struct DCTemplate_rvv : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate_rvv<Quantizer, Similarity, 1>
        : public DCTemplate<Quantizer, Similarity, 1> {
    DCTemplate_rvv(size_t d, const std::vector<float>& trained)
            : DCTemplate<Quantizer, Similarity, 1>(d, trained) {}
};

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <class Quantizer, class Similarity>
struct DCTemplate_rvv<Quantizer, Similarity, 0> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    using Quantizer6bitSpecialized = QuantizerTemplate_rvv<
            Codec6bit_rvv,
            QuantizerTemplateScaling::NON_UNIFORM,
            0>;

    DCTemplate_rvv(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        size_t d = quant.d;
        size_t i = 0;
        const size_t vlmax = __riscv_vsetvlmax_e32m2();

        vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

        for (; i + 4 * vlmax <= d; i += 4 * vlmax) {
            if constexpr (
                    std::is_same_v<Quantizer, QuantizerFP16_rvv<0>> ||
                    std::is_same_v<Quantizer, QuantizerBF16_rvv<0>>) {
                vfloat32m2_t x0 = quant.reconstruct_components(code, i, vlmax);
                vfloat32m2_t x1 =
                        quant.reconstruct_components(code, i + vlmax, vlmax);
                vfloat32m2_t x2 = quant.reconstruct_components(
                        code, i + 2 * vlmax, vlmax);
                vfloat32m2_t x3 = quant.reconstruct_components(
                        code, i + 3 * vlmax, vlmax);

                const float* y_ptr = x + i;
                vfloat32m2_t y0 = __riscv_vle32_v_f32m2(y_ptr, vlmax);
                vfloat32m2_t y1 = __riscv_vle32_v_f32m2(y_ptr + vlmax, vlmax);
                vfloat32m2_t y2 =
                        __riscv_vle32_v_f32m2(y_ptr + 2 * vlmax, vlmax);
                vfloat32m2_t y3 =
                        __riscv_vle32_v_f32m2(y_ptr + 3 * vlmax, vlmax);

                if constexpr (Sim::metric_type == METRIC_L2) {
                    vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(y0, x0, vlmax);
                    vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(y1, x1, vlmax);
                    vfloat32m2_t d2 = __riscv_vfsub_vv_f32m2(y2, x2, vlmax);
                    vfloat32m2_t d3 = __riscv_vfsub_vv_f32m2(y3, x3, vlmax);
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, d0, d0, vlmax);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, d1, d1, vlmax);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, d2, d2, vlmax);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, d3, d3, vlmax);
                } else {
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, y0, x0, vlmax);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, y1, x1, vlmax);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, y2, x2, vlmax);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, y3, x3, vlmax);
                }
            } else {
                vfloat32m4_t x_m4_0, x_m4_1;
                constexpr size_t buf_len = 4 * 128;

                if constexpr (std::is_same_v<
                                      Quantizer,
                                      Quantizer6bitSpecialized>) {
                    std::array<float, buf_len> temp_buf;
                    FAISS_THROW_IF_NOT_MSG(
                            4 * vlmax <= buf_len,
                            "RVV vlmax too large for stack buffer in DCTemplate_rvv");

                    quant.reconstruct_components(
                            code, i, 2 * vlmax, temp_buf.data());
                    quant.reconstruct_components(
                            code,
                            i + 2 * vlmax,
                            2 * vlmax,
                            temp_buf.data() + 2 * vlmax);

                    x_m4_0 = __riscv_vle32_v_f32m4(temp_buf.data(), 2 * vlmax);
                    x_m4_1 = __riscv_vle32_v_f32m4(
                            temp_buf.data() + 2 * vlmax, 2 * vlmax);

                } else {
                    (void)__riscv_vsetvl_e32m4(2 * vlmax);
                    x_m4_0 = quant.reconstruct_components(code, i, 2 * vlmax);
                    x_m4_1 = quant.reconstruct_components(
                            code, i + 2 * vlmax, 2 * vlmax);
                    (void)__riscv_vsetvl_e32m2(vlmax);
                }

                vfloat32m2_t x0 = __riscv_vget_v_f32m4_f32m2(x_m4_0, 0);
                vfloat32m2_t x1 = __riscv_vget_v_f32m4_f32m2(x_m4_0, 1);
                vfloat32m2_t x2 = __riscv_vget_v_f32m4_f32m2(x_m4_1, 0);
                vfloat32m2_t x3 = __riscv_vget_v_f32m4_f32m2(x_m4_1, 1);

                const float* y_ptr = x + i;
                vfloat32m2_t y0 = __riscv_vle32_v_f32m2(y_ptr, vlmax);
                vfloat32m2_t y1 = __riscv_vle32_v_f32m2(y_ptr + vlmax, vlmax);
                vfloat32m2_t y2 =
                        __riscv_vle32_v_f32m2(y_ptr + 2 * vlmax, vlmax);
                vfloat32m2_t y3 =
                        __riscv_vle32_v_f32m2(y_ptr + 3 * vlmax, vlmax);

                if constexpr (Sim::metric_type == METRIC_L2) {
                    vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(y0, x0, vlmax);
                    vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(y1, x1, vlmax);
                    vfloat32m2_t d2 = __riscv_vfsub_vv_f32m2(y2, x2, vlmax);
                    vfloat32m2_t d3 = __riscv_vfsub_vv_f32m2(y3, x3, vlmax);
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, d0, d0, vlmax);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, d1, d1, vlmax);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, d2, d2, vlmax);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, d3, d3, vlmax);
                } else {
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, y0, x0, vlmax);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, y1, x1, vlmax);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, y2, x2, vlmax);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, y3, x3, vlmax);
                }
            }
        }

        for (; i < d;) {
            size_t vl;
            if constexpr (
                    std::is_same_v<Quantizer, QuantizerFP16_rvv<0>> ||
                    std::is_same_v<Quantizer, QuantizerBF16_rvv<0>>) {
                vl = __riscv_vsetvl_e32m2(d - i);
                vfloat32m2_t xi = quant.reconstruct_components(code, i, vl);
                const float* y_ptr = x + i;
                vfloat32m2_t y_rem = __riscv_vle32_v_f32m2(y_ptr, vl);
                if constexpr (Sim::metric_type == METRIC_L2) {
                    vfloat32m2_t diff = __riscv_vfsub_vv_f32m2(y_rem, xi, vl);
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, diff, diff, vl);
                } else {
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, y_rem, xi, vl);
                }
            } else {
                vl = __riscv_vsetvl_e32m4(d - i);

                vfloat32m4_t xi_m4;

                if constexpr (std::is_same_v<
                                      Quantizer,
                                      Quantizer6bitSpecialized>) {
                    if (vl <= RVV_CODEC_STACK_THRESHOLD) {
                        std::array<float, RVV_CODEC_STACK_THRESHOLD> temp_buf;
                        quant.reconstruct_components(
                                code, i, vl, temp_buf.data());
                        xi_m4 = __riscv_vle32_v_f32m4(temp_buf.data(), vl);
                    } else {
                        std::vector<float> temp_buf(vl);
                        quant.reconstruct_components(
                                code, i, vl, temp_buf.data());
                        xi_m4 = __riscv_vle32_v_f32m4(temp_buf.data(), vl);
                    }

                } else {
                    xi_m4 = quant.reconstruct_components(code, i, vl);
                }

                vfloat32m2_t p0 = __riscv_vget_v_f32m4_f32m2(xi_m4, 0);
                vfloat32m2_t p1 = __riscv_vget_v_f32m4_f32m2(xi_m4, 1);

                const float* y_ptr = x + i;

                const size_t vlmax_m2 = __riscv_vsetvlmax_e32m2();
                size_t vl0 = (vl > vlmax_m2) ? vlmax_m2 : vl;
                size_t vl1 = vl - vl0;

                if (vl0 > 0) {
                    vfloat32m2_t y0 = __riscv_vle32_v_f32m2(y_ptr, vl0);
                    if constexpr (Sim::metric_type == METRIC_L2) {
                        vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(y0, p0, vl0);
                        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, d0, d0, vl0);
                    } else {
                        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, y0, p0, vl0);
                    }
                }
                if (vl1 > 0) {
                    vfloat32m2_t y1 = __riscv_vle32_v_f32m2(y_ptr + vl0, vl1);
                    if constexpr (Sim::metric_type == METRIC_L2) {
                        vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(y1, p1, vl1);
                        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, d1, d1, vl1);
                    } else {
                        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, y1, p1, vl1);
                    }
                }
            }
            i += vl;
        }

        vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t s0 =
                __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);
        vfloat32m1_t s1 =
                __riscv_vfredusum_vs_f32m2_f32m1(vacc1, sum_scalar, vlmax);
        vfloat32m1_t s2 =
                __riscv_vfredusum_vs_f32m2_f32m1(vacc2, sum_scalar, vlmax);
        vfloat32m1_t s3 =
                __riscv_vfredusum_vs_f32m2_f32m1(vacc3, sum_scalar, vlmax);

        float f0 = __riscv_vfmv_f_s_f32m1_f32(s0);
        float f1 = __riscv_vfmv_f_s_f32m1_f32(s1);
        float f2 = __riscv_vfmv_f_s_f32m1_f32(s2);
        float f3 = __riscv_vfmv_f_s_f32m1_f32(s3);

        return f0 + f1 + f2 + f3;
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        size_t d = quant.d;
        size_t i = 0;
        const size_t vlmax = __riscv_vsetvlmax_e32m2();

        vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

        for (; i + 4 * vlmax <= d; i += 4 * vlmax) {
            if constexpr (
                    std::is_same_v<Quantizer, QuantizerFP16_rvv<0>> ||
                    std::is_same_v<Quantizer, QuantizerBF16_rvv<0>>) {
                vfloat32m2_t x1_0 =
                        quant.reconstruct_components(code1, i, vlmax);
                vfloat32m2_t x1_1 =
                        quant.reconstruct_components(code1, i + vlmax, vlmax);
                vfloat32m2_t x1_2 = quant.reconstruct_components(
                        code1, i + 2 * vlmax, vlmax);
                vfloat32m2_t x1_3 = quant.reconstruct_components(
                        code1, i + 3 * vlmax, vlmax);
                vfloat32m2_t x2_0 =
                        quant.reconstruct_components(code2, i, vlmax);
                vfloat32m2_t x2_1 =
                        quant.reconstruct_components(code2, i + vlmax, vlmax);
                vfloat32m2_t x2_2 = quant.reconstruct_components(
                        code2, i + 2 * vlmax, vlmax);
                vfloat32m2_t x2_3 = quant.reconstruct_components(
                        code2, i + 3 * vlmax, vlmax);

                if constexpr (Sim::metric_type == METRIC_L2) {
                    vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(x1_0, x2_0, vlmax);
                    vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(x1_1, x2_1, vlmax);
                    vfloat32m2_t d2 = __riscv_vfsub_vv_f32m2(x1_2, x2_2, vlmax);
                    vfloat32m2_t d3 = __riscv_vfsub_vv_f32m2(x1_3, x2_3, vlmax);
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, d0, d0, vlmax);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, d1, d1, vlmax);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, d2, d2, vlmax);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, d3, d3, vlmax);
                } else {
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, x1_0, x2_0, vlmax);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, x1_1, x2_1, vlmax);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, x1_2, x2_2, vlmax);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, x1_3, x2_3, vlmax);
                }
            } else {
                vfloat32m4_t x1_m4_0, x1_m4_1, x2_m4_0, x2_m4_1;

                if constexpr (std::is_same_v<
                                      Quantizer,
                                      Quantizer6bitSpecialized>) {
                    constexpr size_t buf_len = 8 * 128;
                    std::array<float, buf_len> temp_buf;
                    FAISS_THROW_IF_NOT_MSG(
                            8 * vlmax <= buf_len,
                            "RVV vlmax too large for stack buffer in DCTemplate_rvv");

                    quant.reconstruct_components(
                            code1, i, 2 * vlmax, temp_buf.data());
                    quant.reconstruct_components(
                            code1,
                            i + 2 * vlmax,
                            2 * vlmax,
                            temp_buf.data() + 2 * vlmax);
                    quant.reconstruct_components(
                            code2, i, 2 * vlmax, temp_buf.data() + 4 * vlmax);
                    quant.reconstruct_components(
                            code2,
                            i + 2 * vlmax,
                            2 * vlmax,
                            temp_buf.data() + 6 * vlmax);

                    x1_m4_0 = __riscv_vle32_v_f32m4(temp_buf.data(), 2 * vlmax);
                    x1_m4_1 = __riscv_vle32_v_f32m4(
                            temp_buf.data() + 2 * vlmax, 2 * vlmax);
                    x2_m4_0 = __riscv_vle32_v_f32m4(
                            temp_buf.data() + 4 * vlmax, 2 * vlmax);
                    x2_m4_1 = __riscv_vle32_v_f32m4(
                            temp_buf.data() + 6 * vlmax, 2 * vlmax);

                } else {
                    x1_m4_0 = quant.reconstruct_components(code1, i, 2 * vlmax);
                    x1_m4_1 = quant.reconstruct_components(
                            code1, i + 2 * vlmax, 2 * vlmax);
                    x2_m4_0 = quant.reconstruct_components(code2, i, 2 * vlmax);
                    x2_m4_1 = quant.reconstruct_components(
                            code2, i + 2 * vlmax, 2 * vlmax);
                }
                vfloat32m2_t x1_0 = __riscv_vget_v_f32m4_f32m2(x1_m4_0, 0);
                vfloat32m2_t x1_1 = __riscv_vget_v_f32m4_f32m2(x1_m4_0, 1);
                vfloat32m2_t x1_2 = __riscv_vget_v_f32m4_f32m2(x1_m4_1, 0);
                vfloat32m2_t x1_3 = __riscv_vget_v_f32m4_f32m2(x1_m4_1, 1);
                vfloat32m2_t x2_0 = __riscv_vget_v_f32m4_f32m2(x2_m4_0, 0);
                vfloat32m2_t x2_1 = __riscv_vget_v_f32m4_f32m2(x2_m4_0, 1);
                vfloat32m2_t x2_2 = __riscv_vget_v_f32m4_f32m2(x2_m4_1, 0);
                vfloat32m2_t x2_3 = __riscv_vget_v_f32m4_f32m2(x2_m4_1, 1);

                if constexpr (Sim::metric_type == METRIC_L2) {
                    vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(x1_0, x2_0, vlmax);
                    vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(x1_1, x2_1, vlmax);
                    vfloat32m2_t d2 = __riscv_vfsub_vv_f32m2(x1_2, x2_2, vlmax);
                    vfloat32m2_t d3 = __riscv_vfsub_vv_f32m2(x1_3, x2_3, vlmax);
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, d0, d0, vlmax);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, d1, d1, vlmax);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, d2, d2, vlmax);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, d3, d3, vlmax);
                } else {
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, x1_0, x2_0, vlmax);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, x1_1, x2_1, vlmax);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, x1_2, x2_2, vlmax);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, x1_3, x2_3, vlmax);
                }
            }
        }

        for (; i < d;) {
            size_t vl;
            if constexpr (
                    std::is_same_v<Quantizer, QuantizerFP16_rvv<0>> ||
                    std::is_same_v<Quantizer, QuantizerBF16_rvv<0>>) {
                vl = __riscv_vsetvl_e32m2(d - i);
                vfloat32m2_t x1i = quant.reconstruct_components(code1, i, vl);
                vfloat32m2_t x2i = quant.reconstruct_components(code2, i, vl);
                if constexpr (Sim::metric_type == METRIC_L2) {
                    vfloat32m2_t diff = __riscv_vfsub_vv_f32m2(x1i, x2i, vl);
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, diff, diff, vl);
                } else {
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, x1i, x2i, vl);
                }
            } else {
                vl = __riscv_vsetvl_e32m4(d - i);

                vfloat32m4_t x1i_m4, x2i_m4;

                if constexpr (std::is_same_v<
                                      Quantizer,
                                      Quantizer6bitSpecialized>) {
                    if (2 * vl <= RVV_CODEC_STACK_THRESHOLD * 2) {
                        std::array<float, RVV_CODEC_STACK_THRESHOLD * 2>
                                temp_buf;
                        quant.reconstruct_components(
                                code1, i, vl, temp_buf.data());
                        quant.reconstruct_components(
                                code2, i, vl, temp_buf.data() + vl);
                        x1i_m4 = __riscv_vle32_v_f32m4(temp_buf.data(), vl);
                        x2i_m4 =
                                __riscv_vle32_v_f32m4(temp_buf.data() + vl, vl);
                    } else {
                        std::vector<float> temp_buf(2 * vl);
                        quant.reconstruct_components(
                                code1, i, vl, temp_buf.data());
                        quant.reconstruct_components(
                                code2, i, vl, temp_buf.data() + vl);
                        x1i_m4 = __riscv_vle32_v_f32m4(temp_buf.data(), vl);
                        x2i_m4 =
                                __riscv_vle32_v_f32m4(temp_buf.data() + vl, vl);
                    }
                } else {
                    x1i_m4 = quant.reconstruct_components(code1, i, vl);
                    x2i_m4 = quant.reconstruct_components(code2, i, vl);
                }

                vfloat32m2_t p1_0 = __riscv_vget_v_f32m4_f32m2(x1i_m4, 0);
                vfloat32m2_t p1_1 = __riscv_vget_v_f32m4_f32m2(x1i_m4, 1);
                vfloat32m2_t p2_0 = __riscv_vget_v_f32m4_f32m2(x2i_m4, 0);
                vfloat32m2_t p2_1 = __riscv_vget_v_f32m4_f32m2(x2i_m4, 1);

                const size_t vlmax_m2 = __riscv_vsetvlmax_e32m2();
                size_t vl0 = (vl > vlmax_m2) ? vlmax_m2 : vl;
                size_t vl1 = vl - vl0;

                if (vl0 > 0) {
                    if constexpr (Sim::metric_type == METRIC_L2) {
                        vfloat32m2_t diff0 =
                                __riscv_vfsub_vv_f32m2(p1_0, p2_0, vl0);
                        vacc0 = __riscv_vfmacc_vv_f32m2(
                                vacc0, diff0, diff0, vl0);
                    } else {
                        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, p1_0, p2_0, vl0);
                    }
                }
                if (vl1 > 0) {
                    if constexpr (Sim::metric_type == METRIC_L2) {
                        vfloat32m2_t diff1 =
                                __riscv_vfsub_vv_f32m2(p1_1, p2_1, vl1);
                        vacc1 = __riscv_vfmacc_vv_f32m2(
                                vacc1, diff1, diff1, vl1);
                    } else {
                        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, p1_1, p2_1, vl1);
                    }
                }
            }
            i += vl;
        }

        vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        vfloat32m1_t s0 =
                __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax);
        vfloat32m1_t s1 =
                __riscv_vfredusum_vs_f32m2_f32m1(vacc1, sum_scalar, vlmax);
        vfloat32m1_t s2 =
                __riscv_vfredusum_vs_f32m2_f32m1(vacc2, sum_scalar, vlmax);
        vfloat32m1_t s3 =
                __riscv_vfredusum_vs_f32m2_f32m1(vacc3, sum_scalar, vlmax);

        float f0 = __riscv_vfmv_f_s_f32m1_f32(s0);
        float f1 = __riscv_vfmv_f_s_f32m1_f32(s1);
        float f2 = __riscv_vfmv_f_s_f32m1_f32(s2);
        float f3 = __riscv_vfmv_f_s_f32m1_f32(s3);

        return f0 + f1 + f2 + f3;
    }

    void set_query(const float* x) final {
        this->q = x;
    }

    float operator()(idx_t i) final {
        return this->query_to_code(this->codes + i * this->code_size);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                this->codes + i * this->code_size,
                this->codes + j * this->code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_distance(this->q, code);
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
        const size_t vlmax = __riscv_vsetvlmax_e32m2();

        vfloat32m2_t vacc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t vacc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);

        size_t d = quant.d;
        size_t i = 0;

        for (; i < d;) {
            size_t vl;
            if constexpr (
                    std::is_same_v<Quantizer, QuantizerFP16_rvv<0>> ||
                    std::is_same_v<Quantizer, QuantizerBF16_rvv<0>>) {
                vl = __riscv_vsetvl_e32m2(d - i);
                vfloat32m2_t x0 = quant.reconstruct_components(code_0, i, vl);
                vfloat32m2_t x1 = quant.reconstruct_components(code_1, i, vl);
                vfloat32m2_t x2 = quant.reconstruct_components(code_2, i, vl);
                vfloat32m2_t x3 = quant.reconstruct_components(code_3, i, vl);

                vfloat32m2_t y = __riscv_vle32_v_f32m2(this->q + i, vl);
                if constexpr (Sim::metric_type == METRIC_L2) {
                    vfloat32m2_t d0 = __riscv_vfsub_vv_f32m2(y, x0, vl);
                    vfloat32m2_t d1 = __riscv_vfsub_vv_f32m2(y, x1, vl);
                    vfloat32m2_t d2 = __riscv_vfsub_vv_f32m2(y, x2, vl);
                    vfloat32m2_t d3 = __riscv_vfsub_vv_f32m2(y, x3, vl);
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, d0, d0, vl);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, d1, d1, vl);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, d2, d2, vl);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, d3, d3, vl);
                } else {
                    vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, y, x0, vl);
                    vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, y, x1, vl);
                    vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, y, x2, vl);
                    vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, y, x3, vl);
                }
            } else {
                vl = __riscv_vsetvl_e32m4(d - i);

                vfloat32m4_t x0_m4, x1_m4, x2_m4, x3_m4;

                if constexpr (std::is_same_v<
                                      Quantizer,
                                      Quantizer6bitSpecialized>) {
                    if (4 * vl <= RVV_CODEC_STACK_THRESHOLD * 4) {
                        std::array<float, RVV_CODEC_STACK_THRESHOLD * 4>
                                temp_buf;
                        quant.reconstruct_components(
                                code_0, i, vl, temp_buf.data());
                        quant.reconstruct_components(
                                code_1, i, vl, temp_buf.data() + vl);
                        quant.reconstruct_components(
                                code_2, i, vl, temp_buf.data() + 2 * vl);
                        quant.reconstruct_components(
                                code_3, i, vl, temp_buf.data() + 3 * vl);
                        x0_m4 = __riscv_vle32_v_f32m4(temp_buf.data(), vl);
                        x1_m4 = __riscv_vle32_v_f32m4(temp_buf.data() + vl, vl);
                        x2_m4 = __riscv_vle32_v_f32m4(
                                temp_buf.data() + 2 * vl, vl);
                        x3_m4 = __riscv_vle32_v_f32m4(
                                temp_buf.data() + 3 * vl, vl);
                    } else {
                        std::vector<float> temp_buf(4 * vl);
                        quant.reconstruct_components(
                                code_0, i, vl, temp_buf.data());
                        quant.reconstruct_components(
                                code_1, i, vl, temp_buf.data() + vl);
                        quant.reconstruct_components(
                                code_2, i, vl, temp_buf.data() + 2 * vl);
                        quant.reconstruct_components(
                                code_3, i, vl, temp_buf.data() + 3 * vl);
                        x0_m4 = __riscv_vle32_v_f32m4(temp_buf.data(), vl);
                        x1_m4 = __riscv_vle32_v_f32m4(temp_buf.data() + vl, vl);
                        x2_m4 = __riscv_vle32_v_f32m4(
                                temp_buf.data() + 2 * vl, vl);
                        x3_m4 = __riscv_vle32_v_f32m4(
                                temp_buf.data() + 3 * vl, vl);
                    }

                } else {
                    x0_m4 = quant.reconstruct_components(code_0, i, vl);
                    x1_m4 = quant.reconstruct_components(code_1, i, vl);
                    x2_m4 = quant.reconstruct_components(code_2, i, vl);
                    x3_m4 = quant.reconstruct_components(code_3, i, vl);
                }
                vfloat32m2_t x0_p0 = __riscv_vget_v_f32m4_f32m2(x0_m4, 0);
                vfloat32m2_t x0_p1 = __riscv_vget_v_f32m4_f32m2(x0_m4, 1);
                vfloat32m2_t x1_p0 = __riscv_vget_v_f32m4_f32m2(x1_m4, 0);
                vfloat32m2_t x1_p1 = __riscv_vget_v_f32m4_f32m2(x1_m4, 1);
                vfloat32m2_t x2_p0 = __riscv_vget_v_f32m4_f32m2(x2_m4, 0);
                vfloat32m2_t x2_p1 = __riscv_vget_v_f32m4_f32m2(x2_m4, 1);
                vfloat32m2_t x3_p0 = __riscv_vget_v_f32m4_f32m2(x3_m4, 0);
                vfloat32m2_t x3_p1 = __riscv_vget_v_f32m4_f32m2(x3_m4, 1);

                const size_t vlmax_m2 = __riscv_vsetvlmax_e32m2();
                size_t vl0 = (vl > vlmax_m2) ? vlmax_m2 : vl;
                size_t vl1 = vl - vl0;

                if (vl0 > 0) {
                    vfloat32m2_t y0 = __riscv_vle32_v_f32m2(this->q + i, vl0);
                    if constexpr (Sim::metric_type == METRIC_L2) {
                        vfloat32m2_t d0 =
                                __riscv_vfsub_vv_f32m2(y0, x0_p0, vl0);
                        vfloat32m2_t d1 =
                                __riscv_vfsub_vv_f32m2(y0, x1_p0, vl0);
                        vfloat32m2_t d2 =
                                __riscv_vfsub_vv_f32m2(y0, x2_p0, vl0);
                        vfloat32m2_t d3 =
                                __riscv_vfsub_vv_f32m2(y0, x3_p0, vl0);
                        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, d0, d0, vl0);
                        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, d1, d1, vl0);
                        vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, d2, d2, vl0);
                        vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, d3, d3, vl0);
                    } else {
                        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, y0, x0_p0, vl0);
                        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, y0, x1_p0, vl0);
                        vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, y0, x2_p0, vl0);
                        vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, y0, x3_p0, vl0);
                    }
                }
                if (vl1 > 0) {
                    size_t offset = i + vl0;
                    vfloat32m2_t y1 =
                            __riscv_vle32_v_f32m2(this->q + offset, vl1);
                    if constexpr (Sim::metric_type == METRIC_L2) {
                        vfloat32m2_t d0 =
                                __riscv_vfsub_vv_f32m2(y1, x0_p1, vl1);
                        vfloat32m2_t d1 =
                                __riscv_vfsub_vv_f32m2(y1, x1_p1, vl1);
                        vfloat32m2_t d2 =
                                __riscv_vfsub_vv_f32m2(y1, x2_p1, vl1);
                        vfloat32m2_t d3 =
                                __riscv_vfsub_vv_f32m2(y1, x3_p1, vl1);
                        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, d0, d0, vl1);
                        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, d1, d1, vl1);
                        vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, d2, d2, vl1);
                        vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, d3, d3, vl1);
                    } else {
                        vacc0 = __riscv_vfmacc_vv_f32m2(vacc0, y1, x0_p1, vl1);
                        vacc1 = __riscv_vfmacc_vv_f32m2(vacc1, y1, x1_p1, vl1);
                        vacc2 = __riscv_vfmacc_vv_f32m2(vacc2, y1, x2_p1, vl1);
                        vacc3 = __riscv_vfmacc_vv_f32m2(vacc3, y1, x3_p1, vl1);
                    }
                }
            }
            i += vl;
        }

        vfloat32m1_t sum_scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        dis0 = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m2_f32m1(vacc0, sum_scalar, vlmax));
        dis1 = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m2_f32m1(vacc1, sum_scalar, vlmax));
        dis2 = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m2_f32m1(vacc2, sum_scalar, vlmax));
        dis3 = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m2_f32m1(vacc3, sum_scalar, vlmax));
    }
};
FAISS_PRAGMA_IMPRECISE_FUNCTION_END

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/
template <class Similarity, int SIMDWIDTH>
struct DistanceComputerByte_rvv : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte_rvv<Similarity, 1>
        : public DistanceComputerByte<Similarity, 1> {
    DistanceComputerByte_rvv(int d, const std::vector<float>& unused)
            : DistanceComputerByte<Similarity, 1>(d, unused) {}
};

template <class Similarity>
struct DistanceComputerByte_rvv<Similarity, 0> : SQDistanceComputer {
    using Sim = Similarity;
    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte_rvv(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        size_t remaining_d = static_cast<size_t>(d);
        size_t offset = 0;
        uint64_t acc64 = 0;

        while (true) {
            size_t vl = __riscv_vsetvl_e8m1(remaining_d);
            if (vl == 0)
                break;

            vuint8m1_t vx_u = __riscv_vle8_v_u8m1(code1 + offset, vl);
            vuint8m1_t vy_u = __riscv_vle8_v_u8m1(code2 + offset, vl);

            if constexpr (Sim::metric_type == METRIC_L2) {
                vuint16m2_t vx16 = __riscv_vzext_vf2_u16m2(vx_u, vl);
                vuint16m2_t vy16 = __riscv_vzext_vf2_u16m2(vy_u, vl);
                vuint32m4_t vx32 = __riscv_vzext_vf2_u32m4(vx16, vl);
                vuint32m4_t vy32 = __riscv_vzext_vf2_u32m4(vy16, vl);
                vint32m4_t sx32 = __riscv_vreinterpret_v_u32m4_i32m4(vx32);
                vint32m4_t sy32 = __riscv_vreinterpret_v_u32m4_i32m4(vy32);
                vint32m4_t sdiff = __riscv_vsub_vv_i32m4(sx32, sy32, vl);
                vint32m4_t sqr = __riscv_vmul_vv_i32m4(sdiff, sdiff, vl);
                vuint32m4_t sqr_u = __riscv_vreinterpret_v_i32m4_u32m4(sqr);
                vuint32m1_t vsum = __riscv_vmv_s_x_u32m1(0, 1);
                vsum = __riscv_vredsum_vs_u32m4_u32m1(sqr_u, vsum, vl);
                acc64 += static_cast<uint64_t>(__riscv_vmv_x_s_u32m1_u32(vsum));
            } else {
                vuint16m2_t vprod = __riscv_vwmulu_vv_u16m2(vx_u, vy_u, vl);
                vuint32m4_t vprod_w = __riscv_vwaddu_vx_u32m4(vprod, 0, vl);
                vuint32m1_t vsum = __riscv_vmv_s_x_u32m1(0, 1);
                vsum = __riscv_vredsum_vs_u32m4_u32m1(vprod_w, vsum, vl);
                acc64 += static_cast<uint64_t>(__riscv_vmv_x_s_u32m1_u32(vsum));
            }

            offset += vl;
            remaining_d -= vl;
        }
        if (acc64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return std::numeric_limits<int>::max();
        }
        return static_cast<int>(acc64);
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = static_cast<uint8_t>(x[i]);
        }
    }

    float operator()(idx_t i) final {
        return query_to_code(this->codes + i * this->code_size);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                this->codes + i * this->code_size,
                this->codes + j * this->code_size);
    }

    float query_to_code(const uint8_t* code) const override final {
        return compute_code_distance(tmp.data(), code);
    }
};

/*******************************************************************
 * select_distance_computer: runtime selection of template
 * specialization
 *******************************************************************/

template <class Similarity>
ScalarQuantizer::SQDistanceComputer* select_distance_computer_rvv(
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new DCTemplate_rvv<
                    QuantizerTemplate_rvv<
                            Codec8bit_rvv,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            0>,
                    Similarity,
                    0>(dim, trained);

        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate_rvv<
                    QuantizerTemplate_rvv<
                            Codec8bit_rvv,
                            QuantizerTemplateScaling::UNIFORM,
                            0>,
                    Similarity,
                    0>(dim, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate_rvv<
                    QuantizerTemplate_rvv<
                            Codec4bit_rvv,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            0>,
                    Similarity,
                    0>(dim, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            // Fallback to base class for SQ4U to ensure correct COSINE metric
            // handling The generic DCTemplate_rvv computes IP distance for
            // INNER_PRODUCT metric, but SQ4U with COSINE metric requires L2
            // distance computation.
            // TODO: Implement RVV-optimized DistanceComputerSQ4UByte_rvv
            // similar to AVX2 version
            return select_distance_computer<Similarity>(qtype, dim, trained);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate_rvv<
                    QuantizerTemplate_rvv<
                            Codec6bit_rvv,
                            QuantizerTemplateScaling::NON_UNIFORM,
                            0>,
                    Similarity,
                    0>(dim, trained);

        case ScalarQuantizer::QT_fp16:
            return new DCTemplate_rvv<QuantizerFP16_rvv<0>, Similarity, 0>(
                    dim, trained);

        case ScalarQuantizer::QT_bf16:
            return new DCTemplate_rvv<QuantizerBF16_rvv<0>, Similarity, 0>(
                    dim, trained);

        case ScalarQuantizer::QT_8bit_direct:
            return new DCTemplate_rvv<
                    Quantizer8bitDirect_rvv<0>,
                    Similarity,
                    0>(dim, trained);

        case ScalarQuantizer::QT_8bit_direct_signed:
            return new DCTemplate_rvv<
                    Quantizer8bitDirectSigned_rvv<0>,
                    Similarity,
                    0>(dim, trained);

        default:
            FAISS_THROW_FMT("Quantizer type %d not supported", qtype);
            return nullptr;
    }
}

template <class DC>
InvertedListScanner* sel2_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual);

template <class DCClass>
InvertedListScanner* sel2_InvertedListScanner_rvv(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity, class Codec, QuantizerTemplateScaling SCALING>
InvertedListScanner* sel12_InvertedListScanner_rvv(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = 0;
    using QuantizerClass = QuantizerTemplate_rvv<Codec, SCALING, SIMDWIDTH>;
    using DCClass = DCTemplate_rvv<QuantizerClass, Similarity, SIMDWIDTH>;
    return sel2_InvertedListScanner_rvv<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity>
InvertedListScanner* sel1_InvertedListScanner_rvv(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = 0;
    switch (sq->qtype) {
        case QuantizerType::QT_8bit:
            return sel12_InvertedListScanner_rvv<
                    Similarity,
                    Codec8bit_rvv,
                    QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);

        case QuantizerType::QT_8bit_uniform:
            return sel12_InvertedListScanner_rvv<
                    Similarity,
                    Codec8bit_rvv,
                    QuantizerTemplateScaling::UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);

        case QuantizerType::QT_4bit:
            return sel12_InvertedListScanner_rvv<
                    Similarity,
                    Codec4bit_rvv,
                    QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);

        case QuantizerType::QT_4bit_uniform:
            return sel12_InvertedListScanner_rvv<
                    Similarity,
                    Codec4bit_rvv,
                    QuantizerTemplateScaling::UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);

        case QuantizerType::QT_6bit:
            return sel12_InvertedListScanner_rvv<
                    Similarity,
                    Codec6bit_rvv,
                    QuantizerTemplateScaling::NON_UNIFORM>(
                    sq, quantizer, store_pairs, sel, r);
        case QuantizerType::QT_fp16:
            return sel2_InvertedListScanner_rvv<DCTemplate_rvv<
                    QuantizerFP16_rvv<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);

        case QuantizerType::QT_bf16:
            return sel2_InvertedListScanner_rvv<DCTemplate_rvv<
                    QuantizerBF16_rvv<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);

        case QuantizerType::QT_8bit_direct:
            return sel2_InvertedListScanner_rvv<DCTemplate_rvv<
                    Quantizer8bitDirect_rvv<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);

        case QuantizerType::QT_8bit_direct_signed:
            return sel2_InvertedListScanner_rvv<DCTemplate_rvv<
                    Quantizer8bitDirectSigned_rvv<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);

        default:
            FAISS_THROW_MSG("unknown qtype");
            return nullptr;
    }
}

template <int SIMDWIDTH>
InvertedListScanner* select_inverted_list_scanner_rvv(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        size_t /*dim*/,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner_rvv<SimilarityL2_rvv<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner_rvv<SimilarityIP_rvv<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
        return nullptr;
    }
}
} // namespace faiss

#endif // __riscv_vector
