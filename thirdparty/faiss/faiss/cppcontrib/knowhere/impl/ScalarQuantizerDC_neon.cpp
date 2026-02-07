/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/impl/ScalarQuantizerDC_neon.h>
#include <faiss/cppcontrib/knowhere/impl/ScalarQuantizerCodec_neon.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/*******************************************************************
 * ScalarQuantizer Distance Computer
 ********************************************************************/

ScalarQuantizer::SQDistanceComputer* sq_get_distance_computer_neon(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    if (metric == METRIC_L2) {
        if (dim % 8 == 0) {
            return select_distance_computer_neon<SimilarityL2_neon<8>>(
                    qtype, dim, trained);
        } else {
            return select_distance_computer_neon<SimilarityL2_neon<1>>(
                    qtype, dim, trained);
        }
    } else {
        if (dim % 8 == 0) {
            return select_distance_computer_neon<SimilarityIP_neon<8>>(
                    qtype, dim, trained);
        } else {
            return select_distance_computer_neon<SimilarityIP_neon<1>>(
                    qtype, dim, trained);
        }
    }
}

ScalarQuantizer::SQuantizer* sq_select_quantizer_neon(
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    if (dim % 8 == 0) {
        return select_quantizer_1_neon<8>(qtype, dim, trained);
    } else {
        return select_quantizer_1_neon<1>(qtype, dim, trained);
    }
}

InvertedListScanner* sq_select_inverted_list_scanner_neon(
        MetricType mt,
        const ScalarQuantizer *sq,
        const Index *quantizer,
        size_t dim,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (dim % 8 == 0) {
        return sel0_InvertedListScanner_neon<8>(
                mt, sq, quantizer, store_pairs, sel, by_residual);
    } else {
        return sel0_InvertedListScanner_neon<1>(
                mt, sq, quantizer, store_pairs, sel, by_residual);
    }
}

}
}
} // namespace faiss
