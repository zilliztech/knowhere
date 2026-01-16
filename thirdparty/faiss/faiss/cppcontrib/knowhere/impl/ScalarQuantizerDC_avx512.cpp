/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/impl/ScalarQuantizerDC_avx512.h>
#include <faiss/cppcontrib/knowhere/impl/ScalarQuantizerCodec_avx512.h>



namespace faiss::cppcontrib::knowhere {

/*******************************************************************
 * ScalarQuantizer Distance Computer
 ********************************************************************/

ScalarQuantizer::SQDistanceComputer* sq_get_distance_computer_avx512(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    if (metric == METRIC_L2) {
        if (dim % 16 == 0) {
            return select_distance_computer_avx512<SimilarityL2_avx512<16>>(
                    qtype, dim, trained);
        } else if (dim % 8 == 0) {
            return select_distance_computer_avx512<SimilarityL2_avx512<8>>(
                    qtype, dim, trained);
        } else {
            return select_distance_computer_avx512<SimilarityL2_avx512<1>>(
                    qtype, dim, trained);
        }
    } else {
        if (dim % 16 == 0) {
            return select_distance_computer_avx512<SimilarityIP_avx512<16>>(
                    qtype, dim, trained);
        } else if (dim % 8 == 0) {
            return select_distance_computer_avx512<SimilarityIP_avx512<8>>(
                    qtype, dim, trained);
        } else {
            return select_distance_computer_avx512<SimilarityIP_avx512<1>>(
                    qtype, dim, trained);
        }
    }
}

ScalarQuantizer::SQuantizer* sq_select_quantizer_avx512(
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    if (dim % 16 == 0) {
        return select_quantizer_1_avx512<16>(qtype, dim, trained);
    } else if (dim % 8 == 0) {
        return select_quantizer_1_avx512<8>(qtype, dim, trained);
    } else {
        return select_quantizer_1_avx512<1>(qtype, dim, trained);
    }
}

InvertedListScanner* sq_select_inverted_list_scanner_avx512(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        size_t dim,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (dim % 16 == 0) {
        return sel0_InvertedListScanner_avx512<16>(
                mt, sq, quantizer, store_pairs, sel, by_residual);
    } else if (dim % 8 == 0) {
        return sel0_InvertedListScanner_avx512<8>(
                mt, sq, quantizer, store_pairs, sel, by_residual);
    } else {
        return sel0_InvertedListScanner_avx512<1>(
                mt, sq, quantizer, store_pairs, sel, by_residual);
    }
}

}


