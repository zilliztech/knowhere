/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/impl/cc_search.h>

#include <omp.h>
#include <cinttypes>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <typeinfo>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>  // omp_capture_exception / omp_rethrow_if_exception
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/utils.h>

#include <atomic>
#include <exception>

#include <faiss/cppcontrib/knowhere/IndexIVF.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {
namespace cc_impl {

// Path-D step 10.3: the segment-aware scan loop for CC IVF variants.
// Lift-and-shift from fork::IndexIVF::search_preassigned /
// range_search_preassigned. Kept deliberately close to
// `::faiss::IndexIVF::search_preassigned` (baseline) so the delta is
// auditable in one place:
//
//   Baseline single-scan per list:
//     scanner->set_list(key, coarse_dis);
//     scanner->scan_codes(list_size, codes, ids, simi, idxi, k);
//
//   CC loop per list (added here):
//     scanner->set_list(key, coarse_dis);
//     for (segment_idx in 0..cil.get_segment_num(key)):
//         scanner->set_list_segment(segment_offset);
//         scanner->scan_codes(segment_size, codes+offset, ids+offset, ...);
//
// The `cil` reference is the concrete ConcurrentArrayInvertedLists — we
// call segment API directly on it so the eventual Path-D step 10.10
// demotion of get_segment_num/size/offset from fork InvertedLists base
// virtuals to non-virtual methods on ConcurrentArrayInvertedLists
// doesn't affect this code.

void search_preassigned(
        const ::faiss::IndexIVF& index,
        const ConcurrentArrayInvertedLists& cil,
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const ::faiss::SearchParametersIVF* params,
        ::faiss::IndexIVFStats* ivf_stats) {
    // Path-D step 11.4b: fork IndexIVF was collapsed to a `using`
    // alias of baseline; no downcast needed.

    // ------------------------------------------------------------------
    // Safety checks (matches baseline ::faiss::IndexIVF::search_preassigned)
    // ------------------------------------------------------------------
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(index.is_trained, "IVF index is not trained");
    FAISS_THROW_IF_NOT_MSG(
            index.invlists, "IVF index has no inverted lists");

    idx_t cur_nprobe = params ? params->nprobe : index.nprobe;
    cur_nprobe = std::min((idx_t)index.nlist, cur_nprobe);
    FAISS_THROW_IF_NOT(cur_nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t cur_max_codes = params ? params->max_codes : index.max_codes;
    const bool ensure_topk_full = params ? params->ensure_topk_full : false;
    // Baseline-style effective budget: with ensure_topk_full, the per-
    // list scan cap is raised to max(max_codes, k) so the probe loop
    // can make progress toward k hits even when the user-set budget is
    // exhausted.
    idx_t effective_max_codes = cur_max_codes;

    ::faiss::IDSelector* sel = params ? params->sel : nullptr;
    const ::faiss::IDSelectorRange* selr =
            dynamic_cast<const ::faiss::IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr; // special IDSelectorRange fast-path
        } else {
            selr = nullptr; // generic IDSelector processing
        }
    }

    FAISS_THROW_IF_NOT_MSG(
            !(sel && store_pairs),
            "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
            !index.invlists->use_iterator ||
                    (cur_max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = ::faiss::CMin<float, idx_t>;
    using HeapForL2 = ::faiss::CMax<float, idx_t>;

    // Baseline-style exception plumbing: preserve original exception
    // types across OMP thread boundaries via std::exception_ptr, and
    // use a memory-safe atomic for the interrupt flag.
    std::exception_ptr ex;
    std::atomic<bool> interrupt{false};

    int pmode = index.parallel_mode & ~index.PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init =
            !(index.parallel_mode & index.PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
            cur_max_codes == 0 || pmode == 0 || pmode == 3,
            "max_codes supported only for parallel_mode = 0 or 3");

    FAISS_THROW_IF_NOT_MSG(
            !ensure_topk_full || pmode == 0 || pmode == 3,
            "ensure_topk_full supported only for parallel_mode = 0 or 3");

    if (cur_max_codes == 0) {
        cur_max_codes = unlimited_list_size;
    }
    effective_max_codes = ensure_topk_full
            ? std::max(cur_max_codes, k)
            : cur_max_codes;

    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? cur_nprobe > 1
                                  : cur_nprobe * n > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        // C++ exceptions that escape an OpenMP parallel region without
        // being caught inside call std::terminate. The outer try/catch
        // covers per-thread setup (scanner creation, set_query); the
        // inner try/catch in scan_one_list covers per-list work. Both
        // use omp_capture_exception to stash the exception and set the
        // interrupt flag; omp_rethrow_if_exception re-raises after the
        // parallel region exits.
        try {
            // Path-D step 11.4b: take ownership in a baseline-typed
            // unique_ptr; dynamic_cast to the knowhere hook interface
            // for access to the fork-only `set_list_segment` virtual
            // (refreshes the cosine norm cache per CC segment).
            std::unique_ptr<::faiss::InvertedListScanner> base_scanner(
                    index.get_InvertedListScanner(store_pairs, sel, params));
            auto* scanner = base_scanner.get();
            auto* hooks = dynamic_cast<KnowhereInvertedListScannerHooks*>(
                    base_scanner.get());
            FAISS_THROW_IF_NOT_MSG(
                    hooks != nullptr,
                    "cc_impl::search_preassigned: scanner does not implement "
                    "knowhere hooks");

            auto init_result = [&](float* simi, idx_t* idxi) {
                if (!do_heap_init) {
                    return;
                }
                if (index.metric_type == METRIC_INNER_PRODUCT) {
                    ::faiss::heap_heapify<HeapForIP>(k, simi, idxi);
                } else {
                    ::faiss::heap_heapify<HeapForL2>(k, simi, idxi);
                }
            };

            auto add_local_results = [&](const float* local_dis,
                                         const idx_t* local_idx,
                                         float* simi,
                                         idx_t* idxi) {
                if (index.metric_type == METRIC_INNER_PRODUCT) {
                    ::faiss::heap_addn<HeapForIP>(
                            k, simi, idxi, local_dis, local_idx, k);
                } else {
                    ::faiss::heap_addn<HeapForL2>(
                            k, simi, idxi, local_dis, local_idx, k);
                }
            };

            auto reorder_result = [&](float* simi, idx_t* idxi) {
                if (!do_heap_init) {
                    return;
                }
                if (index.metric_type == METRIC_INNER_PRODUCT) {
                    ::faiss::heap_reorder<HeapForIP>(k, simi, idxi);
                } else {
                    ::faiss::heap_reorder<HeapForL2>(k, simi, idxi);
                }
            };

            // CC segment-aware list scan.
            //
            // Returns the POST-FILTER scan count (scanner->scan_cnt
            // accumulated across the list's segments). This matches the
            // fork IndexIVF semantics where the per-list nscan in the
            // probe loop is post-filter — required for ensure_topk_full
            // to converge correctly when an IDSelector rejects many
            // candidates. Baseline tracks pre-filter list_size instead;
            // the two differ when a restrictive selector is in play.
            //
            // list_size_max is applied as a raw pre-filter cap across
            // segments: we scan at most that many codes in total before
            // aborting the list. This respects the user's max_codes
            // budget even when scan_cnt (post-filter) hasn't reached it.
            //
            // Note: the IDSelectorRange fast path (baseline uses
            // find_sorted_ids_bounds to skip sections of a sorted id
            // block) is NOT applied in the CC segment loop — sorted-id
            // ranges spanning segment boundaries would need per-segment
            // boundary arithmetic. For now the selr pointer is cleared
            // back to generic IDSelector processing before this call.
            auto scan_one_list = [&](idx_t key,
                                     float coarse_dis_i,
                                     float* simi,
                                     idx_t* idxi,
                                     idx_t list_size_max) {
                if (key < 0) {
                    return (size_t)0;
                }
                FAISS_THROW_IF_NOT_FMT(
                        key < (idx_t)index.nlist,
                        "Invalid key=%" PRId64 " nlist=%zd\n",
                        key,
                        index.nlist);

                if (index.invlists->is_empty(key, inverted_list_context)) {
                    return (size_t)0;
                }

                scanner->set_list(key, coarse_dis_i);
                nlistv++;

                if (index.invlists->use_iterator) {
                    size_t list_size = 0;
                    std::unique_ptr<InvertedListsIterator> it(
                            index.invlists->get_iterator(
                                    key, inverted_list_context));
                    nheap += scanner->iterate_codes(
                            it.get(), simi, idxi, k, list_size);
                    return list_size;
                }

                ::faiss::InvertedListScannerStats list_stats;

                size_t remaining = list_size_max > 0
                        ? static_cast<size_t>(list_size_max)
                        : static_cast<size_t>(unlimited_list_size);

                size_t segment_num = cil.get_segment_num(key);
                for (size_t segment_idx = 0;
                     segment_idx < segment_num && remaining > 0;
                     segment_idx++) {
                    size_t segment_size =
                            cil.get_segment_size(key, segment_idx);
                    if (segment_size == 0) {
                        continue;
                    }
                    size_t to_scan = std::min(segment_size, remaining);
                    size_t segment_offset =
                            cil.get_segment_offset(key, segment_idx);

                    // Path-D step 10.10: direct access on the concrete
                    // CC invlists. ConcurrentArrayInvertedLists's get_ids
                    // / get_codes return deque-chunk-interior pointers;
                    // release_ids / release_codes are no-op defaults, so
                    // the ScopedCodes/ScopedIds RAII wrappers that used
                    // to live here aren't earning anything.
                    const uint8_t* codes =
                            cil.get_codes(key, segment_offset);
                    const idx_t* ids = store_pairs
                            ? nullptr
                            : cil.get_ids(key, segment_offset);
                    hooks->set_list_segment(segment_offset);
                    if (index.metric_type == METRIC_INNER_PRODUCT) {
                        ::faiss::HeapResultHandler<HeapForIP, false> handler(
                                k, simi, idxi);
                        scanner->scan_codes(to_scan, codes, ids, handler);
                        list_stats.scan_cnt += handler.stats.scan_cnt;
                        list_stats.nheap_updates +=
                                handler.stats.nheap_updates;
                    } else {
                        ::faiss::HeapResultHandler<HeapForL2, false> handler(
                                k, simi, idxi);
                        scanner->scan_codes(to_scan, codes, ids, handler);
                        list_stats.scan_cnt += handler.stats.scan_cnt;
                        list_stats.nheap_updates +=
                                handler.stats.nheap_updates;
                    }

                    remaining -= to_scan;
                }

                nheap += list_stats.nheap_updates;
                return list_stats.scan_cnt;
            };

            if (pmode == 0 || pmode == 3) {
#pragma omp for
                for (idx_t i = 0; i < n; i++) {
                    if (interrupt.load(std::memory_order_relaxed)) {
                        continue;
                    }
                    try {
                        scanner->set_query(x + i * index.d);
                        float* simi = distances + i * k;
                        idx_t* idxi = labels + i * k;

                        init_result(simi, idxi);

                        idx_t nscan = 0;

                        for (idx_t ik = 0; ik < cur_nprobe; ik++) {
                            nscan += scan_one_list(
                                    keys[i * cur_nprobe + ik],
                                    coarse_dis[i * cur_nprobe + ik],
                                    simi,
                                    idxi,
                                    effective_max_codes - nscan);

                            if (nscan >= effective_max_codes) {
                                break;
                            }
                        }

                        ndis += nscan;
                        reorder_result(simi, idxi);

                        ::faiss::InterruptCallback::check();
                    } catch (...) {
                        ::faiss::omp_capture_exception(
                                ex, [&] { interrupt = true; });
                    }
                }
            } else if (pmode == 1) {
                std::vector<idx_t> local_idx(k);
                std::vector<float> local_dis(k);

                for (idx_t i = 0; i < n; i++) {
                    scanner->set_query(x + i * index.d);
                    init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                    for (idx_t ik = 0; ik < cur_nprobe; ik++) {
                        try {
                            ndis += scan_one_list(
                                    keys[i * cur_nprobe + ik],
                                    coarse_dis[i * cur_nprobe + ik],
                                    local_dis.data(),
                                    local_idx.data(),
                                    unlimited_list_size);
                        } catch (...) {
                            ::faiss::omp_capture_exception(
                                    ex, [&] { interrupt = true; });
                        }
                    }

                    float* simi = distances + i * k;
                    idx_t* idxi = labels + i * k;
#pragma omp single
                    init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                    {
                        add_local_results(
                                local_dis.data(),
                                local_idx.data(),
                                simi,
                                idxi);
                    }
#pragma omp barrier
#pragma omp single
                    reorder_result(simi, idxi);
                }
            } else if (pmode == 2) {
                std::vector<idx_t> local_idx(k);
                std::vector<float> local_dis(k);

#pragma omp single
                for (int64_t i = 0; i < n; i++) {
                    init_result(distances + i * k, labels + i * k);
                }

#pragma omp for schedule(dynamic)
                for (int64_t ij = 0; ij < n * cur_nprobe; ij++) {
                    try {
                        size_t i = ij / cur_nprobe;

                        scanner->set_query(x + i * index.d);
                        init_result(local_dis.data(), local_idx.data());
                        ndis += scan_one_list(
                                keys[ij],
                                coarse_dis[ij],
                                local_dis.data(),
                                local_idx.data(),
                                unlimited_list_size);
#pragma omp critical
                        {
                            add_local_results(
                                    local_dis.data(),
                                    local_idx.data(),
                                    distances + i * k,
                                    labels + i * k);
                        }
                    } catch (...) {
                        ::faiss::omp_capture_exception(
                                ex, [&] { interrupt = true; });
                    }
                }
#pragma omp single
                for (int64_t i = 0; i < n; i++) {
                    reorder_result(distances + i * k, labels + i * k);
                }
            } else {
                FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
            }
        } catch (...) {
            ::faiss::omp_capture_exception(ex, [&] { interrupt = true; });
        }
    } // parallel section

    // Preserve fork's log-message shape: baseline's `omp_rethrow_if_exception`
    // rethrows the original exception type unchanged, which makes `e.what()`
    // in knowhere's `catch (const std::exception& e) { log e.what(); }`
    // callers (src/index/ivf/ivf.cc) drop the demangled-type prefix. Fork's
    // old IVF bodies formatted the message as
    //   "search interrupted with: <demangled-type>  <original what-msg>"
    // and knowhere log output relied on that. We reproduce the old shape
    // here by rethrowing-to-catch-to-reformat — still benefits from the
    // exception_ptr capture's thread-safe first-one-wins semantics.
    if (ex) {
        try {
            std::rethrow_exception(ex);
        } catch (const std::exception& e) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s  %s",
                    ::faiss::demangle_cpp_symbol(typeid(e).name()).c_str(),
                    e.what());
        } catch (...) {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    // Fork behavior: do NOT auto-fallback to any global indexIVF_stats
    // when ivf_stats is nullptr — callers pass nullptr explicitly to
    // opt out. Baseline upstreamed an auto-fallback to
    // `&indexIVF_stats`; fork intentionally omitted it (see fork
    // IndexIVF.cpp comment at baseline commit 0a00d8137...).
    if (ivf_stats) {
        ivf_stats->nq += n;
        ivf_stats->nlist += nlistv;
        ivf_stats->ndis += ndis;
        ivf_stats->nheap_updates += nheap;
    }
}

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
        ::faiss::IndexIVFStats* stats) {
    // Path-D step 11.4b: fork IndexIVF was collapsed to a `using`
    // alias of baseline; no downcast needed.

    // ------------------------------------------------------------------
    // Safety checks (matches baseline
    // ::faiss::IndexIVF::range_search_preassigned).
    // ------------------------------------------------------------------
    FAISS_THROW_IF_NOT_MSG(index.is_trained, "IVF index is not trained");

    idx_t cur_nprobe = params ? params->nprobe : index.nprobe;
    cur_nprobe = std::min((idx_t)index.nlist, cur_nprobe);
    FAISS_THROW_IF_NOT(cur_nprobe > 0);

    idx_t cur_max_codes = params ? params->max_codes : index.max_codes;
    // Fork semantic: default to 1 (= stop after one empty bucket). Baseline
    // upstreamed the knob but defaults it to 0 (= disabled). We keep fork's
    // default so knowhere callers that don't set it get the knowhere
    // behavior.
    size_t max_empty_result_buckets =
            params ? params->max_empty_result_buckets : 1;
    ::faiss::IDSelector* sel = params ? params->sel : nullptr;

    FAISS_THROW_IF_NOT_MSG(
            !index.invlists->use_iterator ||
                    (cur_max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    FAISS_THROW_IF_NOT_MSG(
            max_empty_result_buckets == 0 || index.parallel_mode == 0,
            "max_empty_result_buckets supported only for parallel_mode = 0");

    size_t nlistv = 0, ndis = 0;

    // Baseline-style exception plumbing via std::exception_ptr preserves
    // original exception types across OMP boundaries.
    std::exception_ptr ex;

    std::vector<::faiss::RangeSearchPartialResult*> all_pres(
            omp_get_max_threads());

    int pmode = index.parallel_mode & ~index.PARALLEL_MODE_NO_HEAP_INIT;
    // don't start parallel section if single query
    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 3           ? false
                     : pmode == 0 ? nx > 1
                     : pmode == 1 ? cur_nprobe > 1
                                  : cur_nprobe * nx > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
    {
        try {
            ::faiss::RangeSearchPartialResult pres(result);
            // Path-D step 11.4b: take ownership in baseline-typed
            // unique_ptr; dynamic_cast for the knowhere-only
            // `set_list_segment` hook.
            std::unique_ptr<::faiss::InvertedListScanner> base_scanner(
                    index.get_InvertedListScanner(store_pairs, sel, params));
            auto* scanner = base_scanner.get();
            auto* hooks = dynamic_cast<KnowhereInvertedListScannerHooks*>(
                    base_scanner.get());
            FAISS_THROW_IF_NOT_MSG(
                    hooks != nullptr,
                    "cc_impl::range_search_preassigned: scanner does not "
                    "implement knowhere hooks");
            all_pres[omp_get_thread_num()] = &pres;

            // CC segment-aware range-list scan.
            //
            // Structurally equivalent to baseline's scan_list_func; the
            // delta is the inner segment loop that accumulates scanned
            // codes across ConcurrentArrayInvertedLists segments and
            // calls `scanner->set_list_segment(segment_offset)` per
            // segment so cosine-aware scanners can refresh their
            // per-segment norm cache.
            //
            // `scanner->set_list(key, coarse_dis)` is called ONCE per
            // list (not per segment) — calling it per segment would
            // reset list_no to the same value repeatedly and any
            // list-level state the scanner caches in set_list. The
            // former fork IVF body called it twice; that was harmless
            // but wasteful.
            auto scan_list_func = [&](size_t i,
                                      size_t ik,
                                      ::faiss::RangeQueryResult& qres) {
                idx_t key = keys[i * cur_nprobe + ik];
                if (key < 0) {
                    return;
                }
                FAISS_THROW_IF_NOT_FMT(
                        key < (idx_t)index.nlist,
                        "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                        key,
                        ik,
                        index.nlist);

                if (index.invlists->is_empty(key, inverted_list_context)) {
                    return;
                }

                ::faiss::InvertedListScannerStats list_stats;
                scanner->set_list(key, coarse_dis[i * cur_nprobe + ik]);
                if (index.invlists->use_iterator) {
                    size_t list_size = 0;
                    std::unique_ptr<InvertedListsIterator> it(
                            index.invlists->get_iterator(
                                    key, inverted_list_context));
                    scanner->iterate_codes_range(
                            it.get(), radius, qres, list_size);
                    qres.stats.scan_cnt += list_size;
                    list_stats.scan_cnt += list_size;
                } else {
                    size_t segment_num = cil.get_segment_num(key);
                    for (size_t segment_idx = 0; segment_idx < segment_num;
                         segment_idx++) {
                        size_t segment_size =
                                cil.get_segment_size(key, segment_idx);
                        if (segment_size == 0) {
                            continue;
                        }
                        size_t segment_offset =
                                cil.get_segment_offset(key, segment_idx);

                        // Path-D step 10.10: direct access (same as in
                        // search_preassigned above).
                        const uint8_t* codes =
                                cil.get_codes(key, segment_offset);
                        const idx_t* ids = cil.get_ids(key, segment_offset);

                        hooks->set_list_segment(segment_offset);
                        const size_t scan_cnt0 = qres.stats.scan_cnt;
                        const size_t heap_updates0 =
                                qres.stats.nheap_updates;
                        scanner->scan_codes_range(
                                segment_size,
                                codes,
                                ids,
                                radius,
                                qres);
                        list_stats.scan_cnt +=
                                qres.stats.scan_cnt - scan_cnt0;
                        list_stats.nheap_updates +=
                                qres.stats.nheap_updates - heap_updates0;
                    }
                }
                nlistv++;
                // Post-filter ndis (matches IndexIVFStats::ndis docstring,
                // "nb of distances computed") — equals raw segment_size
                // sum when no IDSelector is in play.
                ndis += list_stats.scan_cnt;
            };

            if (index.parallel_mode == 0) {
#pragma omp for
                for (idx_t i = 0; i < nx; i++) {
                    try {
                        scanner->set_query(x + i * index.d);

                        ::faiss::RangeQueryResult& qres = pres.new_result(i);

                        // Knowhere-specific: break out of the probe loop
                        // after `max_empty_result_buckets` continuous
                        // probes returned zero new radius matches. The
                        // knob was upstreamed in step 0 but baseline
                        // hasn't wired the break logic yet; fork keeps
                        // its own implementation here.
                        size_t prev_nres = qres.nres;
                        size_t ndup = 0;

                        for (idx_t ik = 0; ik < cur_nprobe; ik++) {
                            scan_list_func(i, ik, qres);

                            if (qres.nres == prev_nres) {
                                ndup++;
                            } else {
                                ndup = 0;
                            }
                            if (max_empty_result_buckets != 0 &&
                                    ndup == max_empty_result_buckets) {
                                break;
                            }
                            prev_nres = qres.nres;
                        }
                    } catch (...) {
                        ::faiss::omp_capture_exception(ex);
                    }
                }
            } else {
                // Other parallel modes (1, 2, 3) disabled in fork; baseline
                // supports 1 and 2 via RangeSearchPartialResult::merge but
                // knowhere has never needed them.
                FAISS_THROW_FMT(
                        "parallel_mode %d not supported\n",
                        index.parallel_mode);
            }

            pres.finalize();
        } catch (...) {
            ::faiss::omp_capture_exception(ex);
        }
    } // parallel section

    // Preserve fork's log-message shape: baseline's `omp_rethrow_if_exception`
    // rethrows the original exception type unchanged, which makes `e.what()`
    // in knowhere's `catch (const std::exception& e) { log e.what(); }`
    // callers (src/index/ivf/ivf.cc) drop the demangled-type prefix. Fork's
    // old IVF bodies formatted the message as
    //   "search interrupted with: <demangled-type>  <original what-msg>"
    // and knowhere log output relied on that. We reproduce the old shape
    // here by rethrowing-to-catch-to-reformat — still benefits from the
    // exception_ptr capture's thread-safe first-one-wins semantics.
    if (ex) {
        try {
            std::rethrow_exception(ex);
        } catch (const std::exception& e) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s  %s",
                    ::faiss::demangle_cpp_symbol(typeid(e).name()).c_str(),
                    e.what());
        } catch (...) {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    // Fork behavior: do NOT auto-fallback to any global indexIVF_stats
    // when stats is nullptr. See note in search_preassigned.
    if (stats) {
        stats->nq += nx;
        stats->nlist += nlistv;
        stats->ndis += ndis;
    }
}

} // namespace cc_impl
} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
