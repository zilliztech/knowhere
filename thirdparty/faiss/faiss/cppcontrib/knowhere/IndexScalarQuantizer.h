/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <stdint.h>
#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>
#include <faiss/cppcontrib/knowhere/IndexIVF.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/** An IVF implementation where the components of the residuals are
 * encoded with a scalar quantizer. All distance computations
 * are asymmetric, so the encoded vectors are decoded and approximate
 * distances are computed.
 */

// Path-D step 11.4b: derives from `::faiss::IndexIVF` directly.
struct IndexIVFScalarQuantizer : ::faiss::IndexIVF {
    // Baseline scalar quantizer value-type. Fork IVF still inherits
    // from fork IndexIVF (needed for ConcurrentArrayInvertedLists,
    // extended search params, and the 8-arg scan_codes interface), but
    // the SQ state itself is the upstream struct and the scanner
    // returned from get_InvertedListScanner is a fork-interface adapter
    // that forwards distance computation to a baseline
    // SQDistanceComputer.
    ::faiss::ScalarQuantizer sq;

    IndexIVFScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ::faiss::ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = true);

    IndexIVFScalarQuantizer();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    /* standalone codec interface */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /// Path-D step 11.4b: CC dispatch hosted here (instead of fork
    /// IndexIVF base) because the qianya path can leave a non-CC
    /// IndexIVFScalarQuantizer holding a ConcurrentArrayInvertedLists.
    /// IVFSQ is the only non-CC fork IVF leaf that can hit this case.
    /// See IndexScalarQuantizer.cpp for the full rationale.
    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            const idx_t* keys,
            const float* coarse_dis,
            faiss::RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;
};

}
}
} // namespace faiss
