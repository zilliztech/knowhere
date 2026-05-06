/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/IndexIVF.h>            // ::faiss::IndexIVF, ::faiss::SearchParametersIVF, ::faiss::IndexIVFStats
#include <faiss/impl/AuxIndexStructures.h>  // ::faiss::RangeSearchResult
#include <faiss/cppcontrib/knowhere/IndexIVF.h>  // fork IVFSearchParameters alias + fork IndexIVFStats alias

namespace faiss {
namespace cppcontrib {
namespace knowhere {

struct ConcurrentArrayInvertedLists;

namespace cc_impl {

/** Segment-aware knn preassigned-search helper for CC IVF variants.
 *
 * Path-D step 10: single source of truth for the per-list segment scan
 * loop used by IndexIVFFlatCC / IndexIVFScalarQuantizerCC (and their
 * cosine variants). Structurally mirrors
 * `::faiss::IndexIVF::search_preassigned` (baseline) with one extra
 * inner loop that walks each list's segments and calls
 * `scanner->set_list_segment(segment_offset)` per segment so cosine-
 * aware scanners can refresh their per-segment norm cache.
 *
 * @param index  the owning IVF index (reference for inherited fields
 *               like `nlist`, `nprobe`, `parallel_mode`, `metric_type`,
 *               etc.). Pass `*this` from the CC leaf.
 * @param cil    the concrete ConcurrentArrayInvertedLists the CC leaf
 *               owns. The helper reads segments directly off this
 *               typed reference rather than going through
 *               `index.invlists` virtual dispatch — that way the
 *               segment API can later be demoted from virtual to
 *               non-virtual on ConcurrentArrayInvertedLists
 *               (Path-D step 10.10).
 * @param n, x, k, assign, centroid_dis, distances, labels, store_pairs,
 *         params, stats
 *               same semantics as
 *               `::faiss::IndexIVF::search_preassigned`.
 */
void search_preassigned(
        const ::faiss::IndexIVF& index,
        const ConcurrentArrayInvertedLists& cil,
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const ::faiss::SearchParametersIVF* params,
        ::faiss::IndexIVFStats* stats);

/** Segment-aware range-search preassigned helper for CC IVF variants.
 *
 * Same relationship to
 * `::faiss::IndexIVF::range_search_preassigned` as above: segment loop
 * + per-segment `scanner->set_list_segment` call plus the knowhere-
 * specific `max_empty_result_buckets` early-termination tracking.
 */
void range_search_preassigned(
        const ::faiss::IndexIVF& index,
        const ConcurrentArrayInvertedLists& cil,
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        ::faiss::RangeSearchResult* result,
        bool store_pairs,
        const ::faiss::SearchParametersIVF* params,
        ::faiss::IndexIVFStats* stats);

} // namespace cc_impl
} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
