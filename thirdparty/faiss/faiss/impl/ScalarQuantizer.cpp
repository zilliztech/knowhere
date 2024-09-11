/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/ScalarQuantizer.h>

#include <algorithm>
#include <cstdio>

#include <faiss/impl/platform_macros.h>
#include <omp.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include <faiss/FaissHook.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*******************************************************************
 * ScalarQuantizer implementation
 *
 * The main source of complexity is to support combinations of 4
 * variants without incurring runtime tests or virtual function calls:
 *
 * - 4 / 8 bits per code component
 * - uniform / non-uniform
 * - IP / L2 distance search
 * - scalar / AVX distance computation
 *
 * The appropriate Quantizer object is returned via select_quantizer
 * that hides the template mess.
 ********************************************************************/

#ifdef __AVX2__
#ifdef __F16C__
#define USE_F16C
#else
#warning \
        "Cannot enable AVX optimizations in scalar quantizer if -mf16c is not set as well"
#endif
#endif

/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer(size_t d, QuantizerType qtype)
        : qtype(qtype),
          rangestat(RangeStat::RS_minmax),
          rangestat_arg(0),
          d(d) {
    set_derived_sizes();
}

ScalarQuantizer::ScalarQuantizer()
        : qtype(QuantizerType::QT_8bit),
          rangestat(RangeStat::RS_minmax),
          rangestat_arg(0),
          d(0),
          bits(0),
          code_size(0) {}

void ScalarQuantizer::set_derived_sizes() {
    switch (qtype) {
        case QuantizerType::QT_8bit:
        case QuantizerType::QT_8bit_uniform:
        case QuantizerType::QT_8bit_direct:
            code_size = d;
            bits = 8;
            break;
        case QuantizerType::QT_4bit:
        case QuantizerType::QT_4bit_uniform:
            code_size = (d + 1) / 2;
            bits = 4;
            break;
        case QuantizerType::QT_6bit:
            code_size = (d * 6 + 7) / 8;
            bits = 6;
            break;
        case QuantizerType::QT_fp16:
            code_size = d * 2;
            bits = 16;
            break;
    }
}

void ScalarQuantizer::train(size_t n, const float* x) {
    int bit_per_dim = qtype == QuantizerType::QT_4bit_uniform ? 4
            : qtype == QuantizerType::QT_4bit                 ? 4
            : qtype == QuantizerType::QT_6bit                 ? 6
            : qtype == QuantizerType::QT_8bit_uniform         ? 8
            : qtype == QuantizerType::QT_8bit                 ? 8
                                                              : -1;

    switch (qtype) {
        case QuantizerType::QT_4bit_uniform:
        case QuantizerType::QT_8bit_uniform:
            train_Uniform(
                    rangestat,
                    rangestat_arg,
                    n * d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QuantizerType::QT_4bit:
        case QuantizerType::QT_8bit:
        case QuantizerType::QT_6bit:
            train_NonUniform(
                    rangestat,
                    rangestat_arg,
                    n,
                    d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QuantizerType::QT_fp16:
        case QuantizerType::QT_8bit_direct:
            // no training necessary
            break;
    }
}

void ScalarQuantizer::train_residual(
        size_t n,
        const float* x,
        Index* quantizer,
        bool by_residual,
        bool verbose) {
    const float* x_in = x;

    // 100k points more than enough
    x = fvecs_maybe_subsample(d, (size_t*)&n, 100000, x, verbose, 1234);

    ScopeDeleter<float> del_x(x_in == x ? nullptr : x);

    if (by_residual) {
        std::vector<Index::idx_t> idx(n);
        quantizer->assign(n, x, idx.data());

        std::vector<float> residuals(n * d);
        quantizer->compute_residual_n(n, x, residuals.data(), idx.data());

        train(n, residuals.data());
    } else {
        train(n, x);
    }
}

void ScalarQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    std::unique_ptr<Quantizer> squant(select_quantizer());

    memset(codes, 0, code_size * n);
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++)
        squant->encode_vector(x + i * d, codes + i * code_size);
}

void ScalarQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    std::unique_ptr<Quantizer> squant(select_quantizer());

#pragma omp parallel for
    for (int64_t i = 0; i < n; i++)
        squant->decode_vector(codes + i * code_size, x + i * d);
}

/*******************************************************************
 * IndexScalarQuantizer/IndexIVFScalarQuantizer scanner object
 *
 * It is an InvertedListScanner, but is designed to work with
 * IndexScalarQuantizer as well.
 ********************************************************************/
Quantizer* ScalarQuantizer::select_quantizer() const {
    /* use hook to decide use AVX512 or not */
    return sq_sel_quantizer(qtype, d, trained);
}

SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
    /* use hook to decide use AVX512 or not */
    return sq_get_distance_computer(metric, qtype, d, trained);
}

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        bool by_residual) const {
    /* use hook to decide use AVX512 or not */
    return sq_sel_inv_list_scanner(mt, this, quantizer, d, store_pairs,
                                   by_residual);
}

} // namespace faiss
