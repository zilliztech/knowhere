/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ScalarQuantizerDC.h>
#include <faiss/impl/ScalarQuantizerCodec.h>

namespace faiss {

/*******************************************************************
 * ScalarQuantizer Distance Computer
 ********************************************************************/

/* SSE */
ScalarQuantizer::SQDistanceComputer* sq_get_distance_computer_ref(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    if (metric == METRIC_L2) {
        return select_distance_computer<SimilarityL2<1>>(qtype, dim, trained);
    } else {
        return select_distance_computer<SimilarityIP<1>>(qtype, dim, trained);
    }
}

ScalarQuantizer::SQuantizer* sq_select_quantizer_ref(
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    return select_quantizer_1<1>(qtype, dim, trained);
}

InvertedListScanner* sq_select_inverted_list_scanner_ref(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        size_t dim,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    return sel0_InvertedListScanner<1>(
            mt, sq, quantizer, store_pairs, sel, by_residual);
}

} // namespace faiss
