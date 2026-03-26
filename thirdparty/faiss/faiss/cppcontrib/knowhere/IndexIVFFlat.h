/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <stdint.h>
#include <unordered_map>

#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexIVF.h>

#include "knowhere/object.h"

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct IndexIVFFlat : IndexIVF {
    IndexIVFFlat(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2);

    // Be careful with overriding this function, because
    //   renormalized x may be used inside.
    // Overridden by IndexIVFFlatDedup.
    void train(idx_t n, const float* x) override;

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void add_core(
            idx_t n,
            const float* x,
            const float* x_norms,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    IndexIVFFlat();
};

struct IndexIVFFlatCosine : IndexIVFFlat, HasInverseL2Norms {
    IndexIVFFlatCosine(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2);

    IndexIVFFlatCosine();

    void train(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
};

struct IndexIVFFlatCC : IndexIVFFlat {
    IndexIVFFlatCC(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t ssize,
            MetricType = METRIC_L2);

    IndexIVFFlatCC();
};

struct IndexIVFFlatCCCosine : IndexIVFFlatCC, HasInverseL2Norms {
    IndexIVFFlatCCCosine(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t ssize,
            MetricType = METRIC_L2);

    IndexIVFFlatCCCosine();

    void train(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
};

}
}
} // namespace faiss
