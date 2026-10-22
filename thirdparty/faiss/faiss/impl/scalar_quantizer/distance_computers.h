/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#include <faiss/utils/simd_levels.h>

#include <cstring>
#include <memory>
#include <vector>

#include "simd/hook.h" // knowhere bf16_vec_* kernels (SVE BFDOT / NEON / ref)

namespace faiss {

namespace scalar_quantizer {

using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, SIMDLevel SL>
struct DCTemplate : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, SIMDLevel::NONE> : SQDistanceComputer {
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

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, SIMDLevel SL>
struct DistanceComputerByte : SQDistanceComputer {};

// Byte-domain distance computer for QT_8bit_direct_signed (storage is
// value+128). A level added to the sq-dispatch.h chain without a
// specialization here instantiates this empty template.
template <class Similarity, SIMDLevel SL>
struct DistanceComputerByteSigned : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte<Similarity, SIMDLevel::NONE> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d_in, const std::vector<float>&)
            : d(d_in), tmp(d_in) {}

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

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

/*******************************************************************
 * DistanceComputerBF16Native: QT_bf16 distances computed by Knowhere's
 * BF16 kernels (SVE: BFDOT = BF16 multiply, FP32 accumulate) instead of
 * widening every stored component to FP32 (DCTemplate<QuantizerBF16>).
 * Used by HNSW build (symmetric_dis) and search (query_to_code[s]).
 *
 * BFDOT needs the query in BF16 too. That is lossless when the query was
 * widened from BF16 (BF16 vectors: Knowhere widens them to FP32 before
 * the index sees them). An FP32 query that is not exactly representable
 * in BF16 (FP32 data with a BF16 SQ / refine store) would be rounded, so
 * such queries use `exact` instead: the regular FP32 x BF16 computer.
 *******************************************************************/

namespace detail {
// FP32 -> BF16, round-to-nearest-even. Lossless when x was widened
// from BF16 (Knowhere widens BF16 queries before set_query()).
inline uint16_t fp32_to_bf16_rne(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

// true when every component of x is exactly a BF16 value (low 16 bits zero)
inline bool is_exact_bf16(const float* x, size_t d) {
    uint32_t low = 0;
    for (size_t i = 0; i < d; i++) {
        uint32_t u;
        std::memcpy(&u, x + i, sizeof(u));
        low |= u & 0xFFFFu;
    }
    return low == 0;
}
} // namespace detail

template <class Similarity>
struct DistanceComputerBF16Native : SQDistanceComputer {
    using Sim = Similarity;
    using bf16_t = ::knowhere::bf16;

    size_t d;
    std::vector<uint16_t> qbf; // current query in BF16
    // exact FP32-query x BF16-code computer, for queries that are not BF16
    std::unique_ptr<SQDistanceComputer> exact;
    bool use_exact = false;

    DistanceComputerBF16Native(size_t d_in, SQDistanceComputer* exact_dc)
            : d(d_in), qbf(d_in), exact(exact_dc) {}

    static const bf16_t* as_bf16(const void* p) {
        return reinterpret_cast<const bf16_t*>(p);
    }

    float dist(const void* a, const void* b) const {
        if constexpr (Sim::metric_type == METRIC_INNER_PRODUCT) {
            return ::faiss::cppcontrib::knowhere::bf16_vec_inner_product(
                    as_bf16(a), as_bf16(b), d);
        } else {
            return ::faiss::cppcontrib::knowhere::bf16_vec_L2sqr(
                    as_bf16(a), as_bf16(b), d);
        }
    }

    void set_query(const float* x) final {
        q = x;
        use_exact = !detail::is_exact_bf16(x, d);
        if (use_exact) {
            exact->set_query(x);
            return;
        }
        for (size_t i = 0; i < d; i++) {
            qbf[i] = detail::fp32_to_bf16_rne(x[i]);
        }
    }

    float query_to_code(const uint8_t* code) const final {
        if (use_exact) {
            return exact->query_to_code(code);
        }
        return dist(qbf.data(), code);
    }

    void query_to_codes_batch_4(
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& d0,
            float& d1,
            float& d2,
            float& d3) const override {
        if (use_exact) {
            exact->query_to_codes_batch_4(c0, c1, c2, c3, d0, d1, d2, d3);
            return;
        }
        if constexpr (Sim::metric_type == METRIC_INNER_PRODUCT) {
            ::faiss::cppcontrib::knowhere::bf16_vec_inner_product_batch_4(
                    as_bf16(qbf.data()),
                    as_bf16(c0),
                    as_bf16(c1),
                    as_bf16(c2),
                    as_bf16(c3),
                    d,
                    d0,
                    d1,
                    d2,
                    d3);
        } else {
            ::faiss::cppcontrib::knowhere::bf16_vec_L2sqr_batch_4(
                    as_bf16(qbf.data()),
                    as_bf16(c0),
                    as_bf16(c1),
                    as_bf16(c2),
                    as_bf16(c3),
                    d,
                    d0,
                    d1,
                    d2,
                    d3);
        }
    }

    // Graph construction: neighbour selection / pruning.
    float symmetric_dis(idx_t i, idx_t j) override {
        return dist(codes + i * code_size, codes + j * code_size);
    }
};

/*******************************************************************
 * Selection function
 *******************************************************************/

template <SIMDLevel SL>
SQDistanceComputer* sq_select_distance_computer(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained);

} // namespace scalar_quantizer
} // namespace faiss