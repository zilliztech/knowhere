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

#include <faiss/IndexIVFFlat.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexIVF.h>
#include <faiss/cppcontrib/knowhere/invlists/ConcurrentDirectMap.h>

#include "knowhere/object.h"

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 *
 * Path-D follow-up: derives from baseline `::faiss::IndexIVFFlat`
 * directly. Baseline owns codec/default search helpers; this fork
 * keeps only the knowhere-specific storage, scanner hooks, cosine,
 * iterator, CC, and calc-dist-by-id behavior.
 */
struct IndexIVFFlat : ::faiss::IndexIVFFlat {
    IndexIVFFlat(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2);

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel,
            const IVFSearchParameters* params) const override;

    /// Knowhere-specific: compute distance between one query and a set
    /// of vectors identified by ids, via direct_map lookup + inline
    /// float-code access. Only valid when codes are raw floats (i.e.
    /// IVFFlat and its cosine variant). Supported metrics: L2 and IP.
    /// Path-D step 11 follow-up: moved here from fork::IndexIVF —
    /// `IvfIndexNode::CalcDistByIDs` only invokes it for
    /// `IndexIVFFlat` / `IndexIVFFlatCC` (gated by `if constexpr`),
    /// so making it a fork-IVFFlat-rooted method (overridden by
    /// IndexIVFFlatCC for cc_direct_map access) drops dead inherited
    /// surface from IVFSQ / IVFPQ / IVFRaBitQ / IVFFastScan.
    /// Marked virtual so IndexIVFFlatCC's variant is a proper override.
    virtual void calc_dist_by_ids(
            idx_t n,
            const float* x,
            size_t num_keys,
            const int64_t* keys,
            float* out_dist) const;

    IndexIVFFlat() = default;
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
    /// Path-D step 10.8: CC-specific standalone concurrent direct map.
    /// Populated in parallel with the inherited fork `direct_map`
    /// during add (both get the same `id → LO(list_no, offset)`
    /// entries). Read by `calc_dist_by_ids` and `reconstruct`.
    ///
    /// Currently maintained as a secondary copy — fork's
    /// `direct_map.concurrentArray` is still the authoritative
    /// storage during the transition window. Path-D step 10.9 will
    /// remove the fork `direct_map.ConcurrentArray` variant entirely,
    /// at which point `cc_direct_map` becomes the sole storage.
    ConcurrentDirectMap cc_direct_map;

    IndexIVFFlatCC(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t ssize,
            MetricType = METRIC_L2);

    IndexIVFFlatCC();

    // Path-D step 10.8: override add_core to sync cc_direct_map from
    // the (still-populated) fork direct_map after parent completes.
    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    // Fork-only CC cosine side channel: add vectors while also
    // persisting per-entry norms and populating cc_direct_map. The
    // ordinary IndexIVFFlat root intentionally does not expose this
    // method; non-CC cosine uses a private helper in IndexIVFFlat.cpp.
    void add_core_with_norms(
            idx_t n,
            const float* x,
            const float* x_norms,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr);

    // Path-D step 10.8: override calc_dist_by_ids to read from
    // cc_direct_map instead of the inherited path which reads from
    // fork direct_map. This is the last caller that needs the
    // id→(list_no, offset) lookup for CC variants outside add-time.
    void calc_dist_by_ids(
            idx_t n,
            const float* x,
            size_t num_keys,
            const int64_t* keys,
            float* out_dist) const override;

    // Path-D step 10.9: reconstruct override — fork IndexIVF's base
    // `reconstruct` reads from fork direct_map, which is no longer
    // populated for CC indexes after this step. Override routes the
    // id→(list_no, offset) lookup through cc_direct_map.
    void reconstruct(idx_t key, float* recons) const override;

    // Path-D step 10.4: CC-leaf overrides route through the cc_impl::
    // helpers so the segment-aware scan loop lives in one place rather
    // than on fork::IndexIVF. Bodies just static_cast `this->invlists`
    // to ConcurrentArrayInvertedLists (known concrete type — fork
    // IndexIVFFlatCC ctor swaps in a ConcurrentArrayInvertedLists via
    // replace_invlists) and delegate.
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
