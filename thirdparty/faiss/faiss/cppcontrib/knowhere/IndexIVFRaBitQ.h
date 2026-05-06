/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/cppcontrib/knowhere/IndexIVF.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// Thin knowhere shim over baseline IndexIVFRaBitQ.
//
// Baseline owns the RaBitQ codec, training, add/search behavior, and
// serialization layout. The only fork-specific behavior kept here is
// the iterator scanner hook: get_InvertedListScanner() wraps baseline's
// anonymous concrete scanner so IVFBaseIteratorWorkspace can keep using
// KnowhereInvertedListScannerHooks::scan_codes_and_return().
struct IndexIVFRaBitQ : ::faiss::IndexIVFRaBitQ {
    IndexIVFRaBitQ(
            Index* quantizer,
            const size_t d,
            const size_t nlist,
            MetricType metric = METRIC_L2,
            bool own_invlists = true,
            uint8_t nb_bits = 1);

    IndexIVFRaBitQ();

    ::faiss::InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
