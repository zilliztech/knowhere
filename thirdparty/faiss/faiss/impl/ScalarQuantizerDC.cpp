/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/ScalarQuantizerDC.h>
#include <faiss/impl/ScalarQuantizerCodec.h>
#include <faiss/utils/hamming_distance/hamdis-inl.h>
#include <faiss/utils/jaccard-inl.h>

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

ScalarQuantizer::SQDistanceComputer* sq_get_hamming_distance_computer_ref(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    return select_hamming_distance_computer(dim, trained);
}

SQDistanceComputer* select_hamming_distance_computer(
        size_t d,
        const std::vector<float>& trained) {
    size_t code_size = (d + 7) / 8;
    switch (code_size) {
        case 4:
            return new BinarySQDistanceComputerWrapper<HammingComputer4>(code_size, trained);
        case 8:
            return new BinarySQDistanceComputerWrapper<HammingComputer8>(code_size, trained);
        case 16:
            return new BinarySQDistanceComputerWrapper<HammingComputer16>(code_size, trained);
        case 20:
            return new BinarySQDistanceComputerWrapper<HammingComputer20>(code_size, trained);
        case 32:
            return new BinarySQDistanceComputerWrapper<HammingComputer32>(code_size, trained);
        case 64:
            return new BinarySQDistanceComputerWrapper<HammingComputer64>(code_size, trained);
        default:
            return new BinarySQDistanceComputerWrapper<HammingComputerDefault>(code_size, trained);
    }
}

ScalarQuantizer::SQDistanceComputer* sq_get_jaccard_distance_computer_ref(
        MetricType metric,
        ScalarQuantizer::QuantizerType qtype,
        size_t dim,
        const std::vector<float>& trained) {
    return select_jaccard_distance_computer(dim, trained);
}

SQDistanceComputer* select_jaccard_distance_computer(
        size_t d,
        const std::vector<float>& trained) {
    size_t code_size = (d + 7) / 8;
    switch (code_size) {
        case 8:
            return new BinarySQDistanceComputerWrapper<JaccardComputer8>(code_size, trained);
        case 16:
            return new BinarySQDistanceComputerWrapper<JaccardComputer16>(code_size, trained);
        case 32:
            return new BinarySQDistanceComputerWrapper<JaccardComputer32>(code_size, trained);
        case 64:
            return new BinarySQDistanceComputerWrapper<JaccardComputer64>(code_size, trained);
        case 128:
            return new BinarySQDistanceComputerWrapper<JaccardComputer128>(code_size, trained);
        case 256:
            return new BinarySQDistanceComputerWrapper<JaccardComputer256>(code_size, trained);
        case 512:
            return new BinarySQDistanceComputerWrapper<JaccardComputer512>(code_size, trained);
        default:
            return new BinarySQDistanceComputerWrapper<JaccardComputerDefault>(code_size, trained);
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
