/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IndexIVF.h>

#include <omp.h>
#include <cstdint>
#include <mutex>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <faiss/cppcontrib/knowhere/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <faiss/cppcontrib/knowhere/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>
#include <faiss/cppcontrib/knowhere/utils/distances.h>
#include "knowhere/object.h"
#include "simd/hook.h"



namespace faiss::cppcontrib::knowhere {

using ScopedIds = InvertedLists::ScopedIds;
using ScopedCodes = InvertedLists::ScopedCodes;

// Fork Level1Quantizer was collapsed to an alias of baseline in
// Path-D step 7a; its body definitions live in baseline
// (::faiss::Level1Quantizer) and are not redefined here.

// Path-D step 11.4b: fork IndexIVF struct collapsed to a `using`
// alias of baseline (see header). The 4-arg ctor that installed a
// fork ArrayInvertedLists by default has been replaced — each
// derived class' ctor now does the `own_invlists_in=false` +
// `replace_invlists(new ArrayInvertedLists(...))` dance directly
// against baseline. The default ctor was `= default` and is no
// longer needed.

// add_with_ids body deleted in Path-D step 8b: baseline's inherited
// version is byte-equivalent and now safe (invlists field is non-null
// and virtual dispatch lands on fork overrides).

// Path-D step 11.4b: `add_core` body deleted. Fork body diverged from
// baseline only in cosmetic ways (`size_t` vs `idx_t` loop indices,
// brace style on a single-line `if`); semantics were byte-equivalent.
// Baseline's inherited `::faiss::IndexIVF::add_core` takes over and
// dispatches `encode_vectors` virtually onto fork subclass overrides.

// to_readonly / is_readonly bodies removed: dead since 2026-03-20
// (see IndexIVF.h comment for the audit context).

// Path-D step 11.4b: `search` body deleted. Fork body diverged from
// baseline only in: (a) variable rename `nprobe`/`cur_nprobe` and
// `n,x,distances,labels`/`sub_n,sub_x,sub_distances,sub_labels` inside
// the lambda, (b) older `std::mutex` + `std::string` exception capture
// vs baseline's `std::exception_ptr` + `omp_capture_exception` /
// `omp_rethrow_if_exception`, (c) stats accumulator: fork wrote to its
// own `faiss::cppcontrib::knowhere::indexIVF_stats` global; baseline
// writes to `::faiss::indexIVF_stats`. Fork's global is write-only —
// no readers exist anywhere in knowhere or thirdparty/faiss outside of
// the fork's own writes. Stats now flow through baseline's standard
// `::faiss::indexIVF_stats` (the one queried by `c_api/IndexIVF_c.cpp`).
// Baseline also adds three sanity checks (quantizer/is_trained/invlists
// non-null), which is a strict capability gain.

// Path-D step 11.4b: `search_preassigned` thin wrapper deleted from
// fork IndexIVF base. The runtime CC-invlists check + cc_impl dispatch
// has been pushed down to IndexIVFScalarQuantizer::search_preassigned
// — the only non-CC IVF leaf that can hold a CC invlists at runtime
// (qianya path: serialized IVFSQCC deserializes as plain IVFSQ + CC
// invlists). Other non-CC leaves (IVFFlat / IVFPQ / IVFRaBitQ /
// IVFFastScan) never end up with CC invlists, so they inherit
// baseline's `::faiss::IndexIVF::search_preassigned` directly.
// CC-leaf classes (IndexIVFFlatCC, IndexIVFScalarQuantizerCC) keep
// their own overrides and delegate to cc_impl unconditionally
// (their invlists is statically known to be CC).

// Path-D step 11.4b: `range_search` body deleted. Fork body diverged
// from baseline only in: (a) `nprobe` vs `cur_nprobe` rename, (b) stats
// accumulator (fork wrote to fork's own `indexIVF_stats`; baseline
// writes to `::faiss::indexIVF_stats` — see `search` deletion comment
// above for why this is a no-op for downstream callers). Baseline adds
// quantizer/is_trained sanity checks, a strict capability gain.

// Path-D step 11.4b: `range_search_preassigned` thin wrapper deleted
// from fork IndexIVF base for the same reasons as `search_preassigned`
// above. Dispatch lives on IndexIVFScalarQuantizer (qianya path) and
// the CC leaves; non-CC leaves inherit baseline directly.

// Path-D step 11 follow-up: `calc_dist_by_ids` body moved to
// IndexIVFFlat::calc_dist_by_ids (+ IndexIVFFlatCC shadowing).
// Rationale: knowhere's CalcDistByIDs wrapper only dispatches to
// IVFFlat / IVFFlatCC, so the method belongs on that type rather
// than on the fork IVF base.

// Path-D step 11.4b: `get_InvertedListScanner` body deleted from fork
// IndexIVF base — see header for the full rationale. IVFIteratorWorkspace
// now does the baseline-to-fork dynamic_cast at the call site instead
// of relying on a fork-side covariant return.

// Path-D step 11.4: bodies for `reconstruct`, `reconstruct_n`,
// `sa_code_size`, `sa_encode`, `search_and_reconstruct`,
// `reconstruct_from_offset`, `remove_ids`, `update_vectors`,
// `check_compatible_for_merge`, `merge_from`, `get_CodePacker`,
// `copy_subset_to` were deleted. Fork bodies were byte-identical or
// cosmetic-only diffs against baseline; deleting the overrides means
// baseline's inherited methods take over. Subclass reconstruct_from_offset
// overrides (IVFFlat, IVFSQ, IVFPQ, IVFRaBitQ, IVFPQFastScan) continue
// to be dispatched via baseline's virtual.

// train + train_encoder + train_encoder_num_vectors deleted in Path-D
// step 7b: bytewise-identical to baseline's. Subclasses (IVFPQ, IVFSQ,
// IVFRaBitQ, IVFPQFastScan) override the inherited baseline virtuals
// unchanged, so virtual dispatch still lands on the right encoder.

// check_ids_sorted body removed: dead since 2026-03-20.

// search_and_return_codes body removed: fork override was a byte-
// identical copy of baseline's ::faiss::IndexIVF::search_and_return_codes
// and had zero callers in knowhere C++. Baseline's inherited version
// remains available for faiss Python bindings.

// Dtor body deleted: fork IndexIVF struct collapsed to alias in
// step 11.4b — baseline's `::faiss::IndexIVF::~IndexIVF` is the only
// dtor.

/*************************************************************************
 * InvertedListScanner
 *
 * Path-D step 7b reparented fork InvertedListScanner onto baseline
 * ::faiss::InvertedListScanner. The default bodies for scan_codes,
 * scan_codes_range, iterate_codes, iterate_codes_range now live in
 * baseline and are inherited unchanged. The fork-only additive
 * scan_codes_and_return hook now lives on
 * KnowhereInvertedListScannerHooks.
 *************************************************************************/

void KnowhereInvertedListScannerHooks::scan_codes_and_return(
                size_t list_size,
                const uint8_t* codes,
                const idx_t* ids,
                std::vector<::knowhere::DistId>& out) const {
    FAISS_THROW_MSG("Not implemented.");
}

}
