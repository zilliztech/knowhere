/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/impl/ScalarQuantizerCodec_rvv.h>
#include <faiss/cppcontrib/knowhere/impl/ScalarQuantizerDC_rvv.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/*******************************************************************
 * ScalarQuantizer Distance Computer
 ********************************************************************/
ScalarQuantizer::SQDistanceComputer* sq_get_distance_computer_rvv(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
#if defined(__riscv_vector)

    if (metric == METRIC_L2) {
        return select_distance_computer_rvv<SimilarityL2_rvv<0>>(
                qtype, dim, trained);
    } else {
        return select_distance_computer_rvv<SimilarityIP_rvv<0>>(
                qtype, dim, trained);
    }
#else

    if (metric == METRIC_L2) {
        return select_distance_computer_rvv<SimilarityL2_rvv<1>>(
                qtype, dim, trained);
    } else {
        return select_distance_computer_rvv<SimilarityIP_rvv<1>>(
                qtype, dim, trained);
    }
#endif
}

ScalarQuantizer::SQuantizer* sq_select_quantizer_rvv(
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
#if defined(__riscv_vector)

    return select_quantizer_1_rvv<0>(qtype, dim, trained);
#else

    return select_quantizer_1_rvv<1>(qtype, dim, trained);
#endif
}

InvertedListScanner* sq_select_inverted_list_scanner_rvv(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        size_t dim,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
#if defined(__riscv_vector)

    return select_inverted_list_scanner_rvv<0>(
            mt, sq, quantizer, dim, store_pairs, sel, by_residual);
#else

    return select_inverted_list_scanner_rvv<1>(
            mt, sq, quantizer, dim, store_pairs, sel, by_residual);
#endif
}

}
}
} // namespace faiss
