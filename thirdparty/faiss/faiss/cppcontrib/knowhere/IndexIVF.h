/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/cppcontrib/knowhere/invlists/DirectMap.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>

#include "knowhere/object.h"

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// Fork's Level1Quantizer was structurally identical to baseline's
// (same fields, same virtuals, same ctor shapes). Collapsed to an alias
// in Path-D step 7a; the baseline definition is the single source.
using Level1Quantizer = ::faiss::Level1Quantizer;

// As of the SearchParametersIVF baseline-upstreaming change, the fork's
// per-field definition has been collapsed to a type alias of
// faiss::SearchParametersIVF. The baseline struct now carries the three
// knowhere-specific knobs (max_lists_num, ensure_topk_full,
// max_empty_result_buckets) with identical names, types, and defaults, so
// every existing call-site that accesses fields by name continues to work
// unchanged.
using SearchParametersIVF = ::faiss::SearchParametersIVF;
using IVFSearchParameters = ::faiss::SearchParametersIVF;

// Fork's IndexIVFInterface was structurally identical to baseline's
// after Path-D step 0 upstreamed SearchParametersIVF. Collapsed to an
// alias in step 7a.
using IndexIVFInterface = ::faiss::IndexIVFInterface;
using IndexIVFStats = ::faiss::IndexIVFStats;

/** Knowhere-only scanner hooks.
 *
 * Baseline IVF search owns the canonical scanner interface
 * (`::faiss::InvertedListScanner`). These hooks carry the strictly
 * additive behavior needed by knowhere-specific paths:
 *   - `set_list_segment(segment_offset)` virtual (default noop) —
 *     hook for cosine variants to refresh a per-segment
 *     code_norms cache (Path-D step 2). Only invoked explicitly from
 *     CC search paths (cc_impl::); non-CC scanners seed the segment-0
 *     cache from within their own set_list().
 *   - `scan_codes_and_return(...)` virtual — flat-vector result
 *     emission used by the knowhere iterator workspace
 *     (Path-D step 5 TODO about possible iterate_codes subsumption).
 */
struct KnowhereInvertedListScannerHooks {
    /// Following codes come from this segment within the current list.
    /// For non-CC (single-segment ArrayInvertedLists / baseline) lists,
    /// scanners self-seed the segment-0 cache from within set_list().
    virtual void set_list_segment(size_t /* segment_offset */) {}

    /** Scan a set of codes and emit per-code (id, distance) results
     * into a flat vector. Used by the knowhere iterator workspace to
     * stream all per-list candidates to an external consumer rather
     * than top-k heap-reduce them in place.
     *
     * This is a knowhere-specific fork extension — no equivalent
     * virtual exists on baseline faiss::InvertedListScanner. The
     * ::knowhere::DistId element type is deliberately the same
     * struct knowhere uses pervasively for streaming results
     * (iterator APIs, HNSW, sparse, etc.), so no translation is
     * needed at the consumer end.
     *
     * TODO(migration): consider whether iterate_codes() can subsume
     * this entirely. iterate_codes is a baseline virtual with heap-
     * semantics (top-k) rather than flat-streaming, so a direct
     * replacement would require reshaping the iterator workspace to
     * consume heap output. If that reshape is viable, this method
     * can be deleted in favor of baseline iterate_codes + a
     * workspace-side heap-to-flat converter, removing one of the
     * last fork-only scanner virtuals.
     *
     * @param list_size     number of codes to scan
     * @param codes         codes to scan (list_size * code_size)
     * @param ids           corresponding ids (ignored if store_pairs)
     * @param out           output (id, distance) pairs appended here
     */
    virtual void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const;

    virtual ~KnowhereInvertedListScannerHooks() = default;
};

/** Fork InvertedListScanner — now a thin subclass of baseline
 * ::faiss::InvertedListScanner (Path-D step 7b reparent) plus the
 * knowhere-only hooks above.
 *
 * All virtuals that exist on baseline (set_query, set_list,
 * distance_to_code, scan_codes, scan_codes_range, iterate_codes,
 * iterate_codes_range) use the accepted baseline signatures; they are
 * inherited from baseline and overridden by concrete subclasses just
 * like before.
 *
 * The former `mutable size_t scan_cnt` member was removed once
 * baseline moved per-list stats out of scanner-local mutable state.
 * k-NN scanner stats are now reported through
 * ::faiss::ResultHandler::stats, and range-search stats through
 * ::faiss::RangeQueryResult::stats, preserving the documented
 * thread-safety contract on baseline ::faiss::InvertedListScanner.
 *
 * The full definition is placed BEFORE IndexIVF below so that
 * IndexIVF::get_InvertedListScanner can declare a covariant return
 * (`InvertedListScanner*` overriding baseline's
 * `::faiss::InvertedListScanner*` return) — covariance requires the
 * compiler to see the inheritance relationship at the point of
 * declaration, not just a forward declaration.
 */
struct InvertedListScanner
        : ::faiss::InvertedListScanner,
          KnowhereInvertedListScannerHooks {
    // Path-D follow-up: the former `mutable size_t scan_cnt` member
    // was removed once baseline's InvertedListScanner search methods
    // started reporting stats through ResultHandler::stats for k-NN
    // and RangeQueryResult::stats for range search. The only
    // knowhere-specific scanner hooks left here are
    // set_list_segment(...) and scan_codes_and_return(...).

    InvertedListScanner(
            bool store_pairs = false,
            const IDSelector* sel = nullptr)
            : ::faiss::InvertedListScanner() {
        this->store_pairs = store_pairs;
        this->sel = sel;
    }

    ~InvertedListScanner() override {}
};

// Path-D step 11.4b: fork `struct IndexIVF` has been collapsed to an
// alias of baseline. After incremental cleanups in steps 7-11 every
// fork override of an IndexIVF method became either byte-identical to
// baseline or was relocated to a derived class:
//
//   - Method bodies (search, range_search, add_core, get_list_size,
//     get_InvertedListScanner): deleted as byte-identical or trivial
//     duplicates of baseline (steps 11.4 / 11.4b).
//   - encode_vectors pure-virtual redeclaration: deleted (11.4b).
//   - search_preassigned / range_search_preassigned CC dispatch:
//     pushed down to IndexIVFScalarQuantizer (qianya path) and the
//     CC-leaf classes (11.4b).
//   - calc_dist_by_ids: relocated to IndexIVFFlat (11 follow-up).
//   - reconstruct family, sa_code_size/sa_encode, train_encoder,
//     check_compatible_for_merge, merge_from, get_CodePacker,
//     copy_subset_to, search_and_return_codes, to_readonly /
//     is_readonly / check_ids_sorted / dump: deleted across earlier
//     steps as byte-identical or dead.
//   - invlists / direct_map shadows: deleted (11.2 / 11.5).
//
// Fork-only IVF features that needed a real subclass (cosine norm
// storage, CC variants, FastScan, knowhere iterator workspace) live
// on derived classes (IndexIVFFlat / IndexIVFFlatCC / IndexIVFFastScan
// / etc.); they now derive from `::faiss::IndexIVF` directly. Code
// that referred to fork-namespace `IndexIVF` (search-time visitors,
// dynamic_casts in `src/index/ivf/ivf.cc`) continues to compile via
// this alias and works unchanged at runtime — fork-derived classes
// ARE baseline `::faiss::IndexIVF`.
using IndexIVF = ::faiss::IndexIVF;

}
}
} // namespace faiss
