/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IndexIVFFlat.h>

#include <omp.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>

#include "knowhere/object.h"
#include "knowhere/utils.h"
#include "knowhere/bitsetview_idselector.h"

#include <faiss/cppcontrib/knowhere/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/cppcontrib/knowhere/utils/distances.h>
#include <faiss/cppcontrib/knowhere/utils/distances_if.h>
#include <faiss/utils/utils.h>

#include <faiss/cppcontrib/knowhere/MetricType.h>
#include <faiss/cppcontrib/knowhere/impl/cc_search.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>

#include "simd/hook.h"



namespace faiss::cppcontrib::knowhere {

/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexIVFFlat::IndexIVFFlat(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric)
        // Path-D follow-up: this class now derives from baseline
        // `::faiss::IndexIVFFlat` directly. Pass own_invlists=false
        // so baseline doesn't auto-allocate
        // its plain `::faiss::ArrayInvertedLists` — the
        // `replace_invlists` below installs a fork ArrayInvertedLists
        // (NormInvertedLists-capable for cosine variants).
        : ::faiss::IndexIVFFlat(
                  quantizer,
                  d,
                  nlist,
                  metric,
                  /*own_invlists=*/false) {
    replace_invlists(new ArrayInvertedLists(nlist, code_size, false), true);
}

namespace {

// Non-CC cosine norm-bearing add helper. This intentionally stays
// private to this translation unit so ordinary IndexIVFFlat can keep
// a baseline-shaped public surface.
void add_core_with_norms_impl(
        IndexIVFFlat* self,
        idx_t n,
        const float* x,
        const float* x_norms,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(self->is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!self->by_residual);
    assert(self->invlists);
    FAISS_THROW_IF_NOT(x_norms);
    self->direct_map.check_can_add(xids);

    // 5-arg add_entry (with norm) is on the NormInvertedLists
    // interface. Only cosine paths should arrive here.
    NormInvertedLists* norm_il =
            dynamic_cast<NormInvertedLists*>(self->invlists);
    FAISS_THROW_IF_NOT_MSG(
            norm_il,
            "IndexIVFFlatCosine norm add requires a norm-capable invlists");

    int64_t n_add = 0;

    DirectMapAdd dm_adder(self->direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : self->ntotal + i;
                const float* xi = x + i * self->d;
                size_t offset = norm_il->add_entry(
                        list_no,
                        id,
                        (const uint8_t*)xi,
                        x_norms + i,
                        inverted_list_context);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (self->verbose) {
        printf("IndexIVFFlatCosine::add_core_with_norms: added %" PRId64
               " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    self->ntotal += n;
}

}  // anonymous namespace

namespace {

template <MetricType metric, class C, bool use_sel>
struct IVFFlatScanner : InvertedListScanner {
    size_t d;

    /// Back-pointer to the norm-capable invlists this scanner reads
    /// from. Used to self-serve per-segment norm pointers in
    /// set_list_segment, moving code_norms off the scan_codes
    /// parameter list. Path-D step 10.14d: typed as
    /// NormInvertedLists* (the fork-specific norm interface); null
    /// when the index's invlists doesn't implement norms (harmless —
    /// cosine scanner is constructed only for invlists with norm
    /// storage, e.g. fork ArrayInvertedLists with with_norm=true).
    const NormInvertedLists* invlists;

    /// Per-segment norm pointer. Seeded to segment 0 by set_list so
    /// non-CC callers don't need to invoke set_list_segment explicitly;
    /// CC callers re-invoke set_list_segment per segment to refresh it.
    const float* cached_code_norms_for_list = nullptr;

    IVFFlatScanner(
            size_t d,
            bool store_pairs,
            const IDSelector* sel,
            const ::faiss::InvertedLists* invlists)
            : InvertedListScanner(store_pairs, sel),
              d(d),
              invlists(dynamic_cast<const NormInvertedLists*>(invlists)) {
        keep_max = faiss::cppcontrib::knowhere::is_similarity_metric(metric);
    }

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        set_list_segment(0);
    }

    void set_list_segment(size_t segment_offset) override {
        cached_code_norms_for_list = (invlists == nullptr) ? nullptr
                : invlists->get_code_norms(
                        static_cast<size_t>(list_no), segment_offset);
    }

    float distance_to_code(const uint8_t* code) const override {
        const float* yj = (float*)code;
        float dis = metric == METRIC_INNER_PRODUCT
                ? fvec_inner_product(xi, yj, d)
                : fvec_L2sqr(xi, yj, d);
        return dis;
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ::faiss::ResultHandler& handler) const override {
        const float* list_vecs = (const float*)codes;
        const float* norms = cached_code_norms_for_list;
        size_t nup = 0;
        float threshold = handler.threshold;

        // the lambda that filters acceptable elements.
        auto filter =
            [&](const size_t j) { return (!use_sel || sel->is_member(ids[j])); };

        // the lambda that applies a valid element.
        auto apply =
            [&](const float dis_in, const size_t j) {
                const float dis = (norms == nullptr) ? dis_in : (dis_in / norms[j]);
                handler.stats.scan_cnt++;
                if (C::cmp(threshold, dis)) {
                    const int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    if (handler.add_result(dis, id)) {
                        handler.stats.nheap_updates++;
                        nup++;
                        threshold = handler.threshold;
                    }
                }
            };

        if constexpr (metric == METRIC_INNER_PRODUCT) {
            fvec_inner_products_ny_if(
                xi, list_vecs, d, list_size, filter, apply);
        }
        else {
            fvec_L2sqr_ny_if(
                xi, list_vecs, d, list_size, filter, apply);
        }

        return nup;
    }

    void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        const float* list_vecs = (const float*)codes;
        const float* norms = cached_code_norms_for_list;

        // the lambda that filters acceptable elements.
        auto filter = [&](const size_t j) {
            return (!use_sel || sel->is_member(ids[j]));
        };
        // the lambda that applies a valid element.
        auto apply = [&](const float dis_in, const size_t j) {
            const float dis =
                    (norms == nullptr) ? dis_in : (dis_in / norms[j]);
            out.emplace_back(ids[j], dis);
        };
        if constexpr (metric == METRIC_INNER_PRODUCT) {
            fvec_inner_products_ny_if(
                    xi, list_vecs, d, list_size, filter, apply);
        } else {
            fvec_L2sqr_ny_if(xi, list_vecs, d, list_size, filter, apply);
        }
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;
        const float* norms = cached_code_norms_for_list;

        // the lambda that filters acceptable elements.
        auto filter =
            [&](const size_t j) { return (!use_sel || sel->is_member(ids[j])); };

        // the lambda that applies a filtered element.
        auto apply =
            [&](const float dis_in, const size_t j) {
                const float dis = (norms == nullptr) ? dis_in : (dis_in / norms[j]);
                res.stats.scan_cnt++;
                if (C::cmp(radius, dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    res.add(dis, id);
                    res.stats.nheap_updates++;
                }
            };

        if constexpr (metric == METRIC_INNER_PRODUCT) {
            fvec_inner_products_ny_if(
                xi, list_vecs, d, list_size, filter, apply);
        }
        else {
            fvec_L2sqr_ny_if(
                xi, list_vecs, d, list_size, filter, apply);
        }
    }
};

// a custom version for Knowhere
template <MetricType metric, class C, bool use_sel>
struct IVFFlatBitsetViewScanner : InvertedListScanner {
    size_t d;
    ::knowhere::BitsetView bitset;

    /// See IVFFlatScanner::invlists / cached_code_norms_for_list
    /// comments — same self-service pattern, seeded by set_list and
    /// refreshed per-segment via set_list_segment on CC paths.
    const NormInvertedLists* invlists;
    const float* cached_code_norms_for_list = nullptr;

    IVFFlatBitsetViewScanner(
            size_t d,
            bool store_pairs,
            const IDSelector* sel,
            const ::faiss::InvertedLists* invlists)
            : InvertedListScanner(store_pairs, sel),
              d(d),
              invlists(dynamic_cast<const NormInvertedLists*>(invlists)) {
        const auto* bitsetview_sel = dynamic_cast<const ::knowhere::BitsetViewIDSelector*>(sel);
        FAISS_ASSERT_MSG((bitsetview_sel != nullptr), "Unsupported scanner for IVFFlatBitsetViewScanner");

        bitset = bitsetview_sel->bitset_view;
    }

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
        set_list_segment(0);
    }

    void set_list_segment(size_t segment_offset) override {
        cached_code_norms_for_list = (invlists == nullptr) ? nullptr
                : invlists->get_code_norms(
                        static_cast<size_t>(list_no), segment_offset);
    }

    float distance_to_code(const uint8_t* code) const override {
        const float* yj = (float*)code;
        float dis = metric == METRIC_INNER_PRODUCT
                ? fvec_inner_product(xi, yj, d)
                : fvec_L2sqr(xi, yj, d);
        return dis;
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* __restrict codes,
            const idx_t* __restrict ids,
            ::faiss::ResultHandler& handler) const override {
        const float* list_vecs = (const float*)codes;
        const float* norms = cached_code_norms_for_list;
        size_t nup = 0;
        float threshold = handler.threshold;

        // the lambda that filters acceptable elements.
        auto filter =
            [&](const size_t j) { return (!use_sel || !bitset.test(ids[j])); };

        // the lambda that applies a valid element.
        auto apply =
            [&](const float dis_in, const size_t j) {
                const float dis = (norms == nullptr) ? dis_in : (dis_in / norms[j]);
                handler.stats.scan_cnt++;
                if (C::cmp(threshold, dis)) {
                    const int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    if (handler.add_result(dis, id)) {
                        handler.stats.nheap_updates++;
                        nup++;
                        threshold = handler.threshold;
                    }
                }
            };

        if constexpr (metric == METRIC_INNER_PRODUCT) {
            fvec_inner_products_ny_if(
                xi, list_vecs, d, list_size, filter, apply);
        }
        else {
            fvec_L2sqr_ny_if(
                xi, list_vecs, d, list_size, filter, apply);
        }

        return nup;
    }

    void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        const float* list_vecs = (const float*)codes;
        const float* norms = cached_code_norms_for_list;

        // the lambda that filters acceptable elements.
        auto filter = [&](const size_t j) {
            return (!use_sel || !bitset.test(ids[j]));
        };
        // the lambda that applies a valid element.
        auto apply = [&](const float dis_in, const size_t j) {
            const float dis =
                    (norms == nullptr) ? dis_in : (dis_in / norms[j]);
            out.emplace_back(ids[j], dis);
        };
        if constexpr (metric == METRIC_INNER_PRODUCT) {
            fvec_inner_products_ny_if(
                    xi, list_vecs, d, list_size, filter, apply);
        } else {
            fvec_L2sqr_ny_if(xi, list_vecs, d, list_size, filter, apply);
        }
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* __restrict codes,
            const idx_t* __restrict ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;
        const float* norms = cached_code_norms_for_list;

        // the lambda that filters acceptable elements.
        auto filter =
            [&](const size_t j) { return (!use_sel || !bitset.test(ids[j])); };

        // the lambda that applies a filtered element.
        auto apply =
            [&](const float dis_in, const size_t j) {
                const float dis = (norms == nullptr) ? dis_in : (dis_in / norms[j]);
                res.stats.scan_cnt++;
                if (C::cmp(radius, dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    res.add(dis, id);
                    res.stats.nheap_updates++;
                }
            };

        if constexpr (metric == METRIC_INNER_PRODUCT) {
            fvec_inner_products_ny_if(
                xi, list_vecs, d, list_size, filter, apply);
        }
        else {
            fvec_L2sqr_ny_if(
                xi, list_vecs, d, list_size, filter, apply);
        }
    }
};

template <bool use_sel>
InvertedListScanner* get_InvertedListScanner1(
        const IndexIVFFlat* ivf,
        bool store_pairs,
        const IDSelector* sel) {
    // A specialized version for Knowhere.
    //   It is needed to get rid of virtual function calls, because sel
    //   can filter out 99% of samples, so the cost of virtual function calls
    //   becomes noticeable compared to distance computations.
    if (const auto* bitsetview_sel = dynamic_cast<const ::knowhere::BitsetViewIDSelector*>(sel)) {
        if (ivf->metric_type == METRIC_INNER_PRODUCT) {
            return new IVFFlatBitsetViewScanner<
                    METRIC_INNER_PRODUCT,
                    CMin<float, int64_t>,
                    use_sel>(ivf->d, store_pairs, sel, ivf->invlists);
        } else if (ivf->metric_type == METRIC_L2) {
            return new IVFFlatBitsetViewScanner<METRIC_L2, CMax<float, int64_t>, use_sel>(
                    ivf->d, store_pairs, sel, ivf->invlists);
        } else {
            FAISS_THROW_MSG("metric type not supported");
        }
    }

    // default faiss version
    if (ivf->metric_type == METRIC_INNER_PRODUCT) {
        return new IVFFlatScanner<
                METRIC_INNER_PRODUCT,
                CMin<float, int64_t>,
                use_sel>(ivf->d, store_pairs, sel, ivf->invlists);
    } else if (ivf->metric_type == METRIC_L2) {
        return new IVFFlatScanner<METRIC_L2, CMax<float, int64_t>, use_sel>(
                ivf->d, store_pairs, sel, ivf->invlists);
    } else {
        FAISS_THROW_MSG("metric type not supported");
    }
}

} // anonymous namespace

InvertedListScanner* IndexIVFFlat::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    if (sel) {
        return get_InvertedListScanner1<true>(this, store_pairs, sel);
    } else {
        return get_InvertedListScanner1<false>(this, store_pairs, sel);
    }
}

IndexIVFFlatCosine::IndexIVFFlatCosine(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric)
        : IndexIVFFlat(quantizer, d, nlist, metric) {
    replace_invlists(new ArrayInvertedLists(nlist, code_size, true), true);
}

IndexIVFFlatCosine::IndexIVFFlatCosine() {
}

void IndexIVFFlatCosine::train(idx_t n, const float* x) {
    auto x_normalized = ::knowhere::CopyAndNormalizeVecs(x, n, d);
    IndexIVF::train(n, x_normalized.get());
}

void IndexIVFFlatCosine::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    auto x_normalized = std::make_unique<float[]>(n * d);
    std::memcpy(x_normalized.get(), x, n * d * sizeof(float));
    auto norms = ::knowhere::NormalizeVecs(x_normalized.get(), n, d);
    quantizer->assign(n, x_normalized.get(), coarse_idx.get());
    add_core_with_norms_impl(
            this, n, x, norms.data(), xids, coarse_idx.get(), nullptr);
}

IndexIVFFlatCC::IndexIVFFlatCC(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t ssize,
        MetricType metric)
        : IndexIVFFlat(quantizer, d, nlist, metric) {
    replace_invlists(new ConcurrentArrayInvertedLists(nlist, code_size, ssize, false), true);
}

IndexIVFFlatCC::IndexIVFFlatCC() {}

// Path-D step 10.9: CC add_core / add_core_with_norms reimplement the
// IVFFlat encode loop against cc_direct_map (ConcurrentDirectMapAdd)
// rather than the inherited fork `direct_map`. Structurally identical
// to `IndexIVFFlat::add_core_impl` in this file, with DirectMapAdd →
// ConcurrentDirectMapAdd and direct_map → cc_direct_map.
//
// Sharing with the parent body would require templating / function-
// pointer customization of add_core_impl (not worth it for one
// helper). The resulting ~30 LOC duplication per CC add method is the
// price of keeping the CC plumbing self-contained.

void IndexIVFFlatCC::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!by_residual);
    assert(invlists);
    cc_direct_map.check_can_add(xids);

    int64_t n_add = 0;
    ConcurrentDirectMapAdd dm_adder(cc_direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                size_t offset = invlists->add_entry(
                        list_no,
                        id,
                        (const uint8_t*)xi,
                        inverted_list_context);
                dm_adder.add(i, list_no, offset);
                n_add++;
            }
            // ConcurrentDirectMapAdd's ctor pre-resizes to -1 so entries
            // with list_no == -1 stay -1 without an explicit add call
            // (unlike fork DirectMapAdd which takes a dm_adder.add(i, -1, 0)
            // for them).
        }
    }

    if (verbose) {
        printf("IndexIVFFlatCC::add_core: added %" PRId64 " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}

void IndexIVFFlatCC::add_core_with_norms(
        idx_t n,
        const float* x,
        const float* x_norms,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!by_residual);
    assert(invlists);
    cc_direct_map.check_can_add(xids);

    // Path-D step 10.14d: 5-arg add_entry (with code_norm) is on the
    // NormInvertedLists interface. Cache the cast once before the omp
    // loop.
    NormInvertedLists* norm_il = dynamic_cast<NormInvertedLists*>(invlists);
    FAISS_THROW_IF_NOT_MSG(
            norm_il,
            "IndexIVFFlatCC::add_core_with_norms requires a norm-capable invlists");

    int64_t n_add = 0;
    ConcurrentDirectMapAdd dm_adder(cc_direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                const float* xi_norm =
                        (x_norms == nullptr) ? nullptr : (x_norms + i);
                size_t offset = norm_il->add_entry(
                        list_no,
                        id,
                        (const uint8_t*)xi,
                        xi_norm,
                        inverted_list_context);
                dm_adder.add(i, list_no, offset);
                n_add++;
            }
        }
    }

    if (verbose) {
        printf("IndexIVFFlatCC::add_core_with_norms: added %" PRId64
               " / %" PRId64 " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}

void IndexIVFFlatCC::reconstruct(idx_t key, float* recons) const {
    // CC-specific reconstruct: look up the (list_no, offset) in
    // cc_direct_map then use the inherited reconstruct_from_offset
    // (same as fork::IndexIVF::reconstruct's body, but routed through
    // the CC concurrent map since fork::direct_map is no longer
    // populated for CC indexes after step 10.9).
    idx_t lo = cc_direct_map.get(key);
    reconstruct_from_offset(lo_listno(lo), lo_offset(lo), recons);
}

// Knowhere-specific calc_dist_by_ids: query → set-of-IDs distance via
// direct_map lookup + inline float-code access. Only valid when codes
// are raw floats (IVFFlat / IVFFlatCosine; IVFFlatCC shadows below).
// Path-D step 11 follow-up: moved here from fork::IndexIVF — see header
// comment for rationale.
void IndexIVFFlat::calc_dist_by_ids(
        idx_t n,
        const float* x,
        size_t num_keys,
        const int64_t* keys,
        float* __restrict out_dist) const {
    assert(n == 1);

    std::array<const float*, 4> codes_array{};
    std::array<float, 4> norms_array{};
    size_t counter = 0;
    // Norm access goes through the NormInvertedLists interface; works
    // for any invlists that implements it (cosine ArrayInvertedLists),
    // returns null otherwise (plain IVFFlat).
    const auto* norm_il = dynamic_cast<const NormInvertedLists*>(invlists);
    auto has_norms =
            norm_il && norm_il->get_code_norms((size_t)0, (size_t)0) != nullptr;

    for (size_t i = 0; i < num_keys; i++) {
        if (direct_map.type == DirectMap::Type::NoMap) {
            throw std::runtime_error(
                    "NoMap direct map not supported `calculate_dist_by_ids`");
        }
        auto lo = direct_map.get(keys[i]);
        auto list_no = lo_listno(lo);
        auto offset = lo_offset(lo);
        // Cast invlists' raw single-code byte pointer to `const float*`.
        // This is only valid for IVFFlat-family invlists, where each
        // entry's storage IS the raw float vector. The proper
        // generalized way to obtain a vector from an arbitrary IVF
        // index is `reconstruct_from_offset(list_no, offset, dest)`,
        // which dispatches per-class (PQ decode, SQ dequantize, etc.).
        // Hosting this method on fork::IndexIVF historically was a
        // dead inherited method on IVFSQ / IVFPQ / IVFRaBitQ — they
        // would have produced garbage if anyone called it because the
        // pointer-cast assumption doesn't hold for their codes. Moving
        // it here (Path-D step 11 follow-up) makes that restriction
        // explicit in the type system.
        auto codes = (const float*)invlists->get_single_code(list_no, offset);

        codes_array[counter] = codes;
        if (has_norms) {
            norms_array[counter] = norm_il->get_norm(list_no, offset);
        }
        counter++;

        if (counter == 4) {
            counter = 0;
            if (metric_type == METRIC_INNER_PRODUCT) {
                fvec_inner_product_batch_4(
                        x,
                        codes_array[0],
                        codes_array[1],
                        codes_array[2],
                        codes_array[3],
                        d,
                        out_dist[i - 3],
                        out_dist[i - 2],
                        out_dist[i - 1],
                        out_dist[i]);
                if (has_norms) {
                    out_dist[i - 3] /= norms_array[0];
                    out_dist[i - 2] /= norms_array[1];
                    out_dist[i - 1] /= norms_array[2];
                    out_dist[i] /= norms_array[3];
                }
            } else {
                fvec_L2sqr_batch_4(
                        x,
                        codes_array[0],
                        codes_array[1],
                        codes_array[2],
                        codes_array[3],
                        d,
                        out_dist[i - 3],
                        out_dist[i - 2],
                        out_dist[i - 1],
                        out_dist[i]);
            }
        }
    }
    // left over
    for (size_t i = 0; i < counter; i++) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            out_dist[num_keys - counter + i] =
                    fvec_inner_product(x, codes_array[i], d);
            if (has_norms) {
                out_dist[num_keys - counter + i] /= norms_array[i];
            }
        } else {
            out_dist[num_keys - counter + i] = fvec_L2sqr(x, codes_array[i], d);
        }
    }
}

// IndexIVFFlatCC::calc_dist_by_ids — same body shape as
// IndexIVFFlat::calc_dist_by_ids above, but reads cc_direct_map
// instead of fork::direct_map (CC variants leave the standard
// direct_map at NoMap; their id→LO lookup lives in cc_direct_map,
// populated during add_core).
void IndexIVFFlatCC::calc_dist_by_ids(
        idx_t n,
        const float* x,
        size_t num_keys,
        const int64_t* keys,
        float* __restrict out_dist) const {
    assert(n == 1);

    std::array<const float*, 4> codes_array{};
    std::array<float, 4> norms_array{};
    size_t counter = 0;
    // Path-D step 10.14d: norm access via NormInvertedLists interface.
    const auto* norm_il = dynamic_cast<const NormInvertedLists*>(invlists);
    auto has_norms =
            norm_il && norm_il->get_code_norms((size_t)0, (size_t)0) != nullptr;

    for (size_t i = 0; i < num_keys; i++) {
        auto lo = cc_direct_map.get(keys[i]);
        auto list_no = lo_listno(lo);
        auto offset = lo_offset(lo);
        // See IndexIVFFlat::calc_dist_by_ids for why this raw cast to
        // `const float*` is IVFFlat-family-only. The generalized
        // approach is reconstruct_from_offset(...), which would
        // dispatch correctly for PQ/SQ/etc. — but knowhere's
        // CalcDistByIDs only invokes calc_dist_by_ids on IVFFlat /
        // IVFFlatCC, so the cheap inline cast suffices here.
        auto codes = (const float*)invlists->get_single_code(list_no, offset);

        codes_array[counter] = codes;
        if (has_norms) {
            norms_array[counter] = norm_il->get_norm(list_no, offset);
        }
        counter++;

        if (counter == 4) {
            counter = 0;
            if (metric_type == METRIC_INNER_PRODUCT) {
                fvec_inner_product_batch_4(
                        x,
                        codes_array[0],
                        codes_array[1],
                        codes_array[2],
                        codes_array[3],
                        d,
                        out_dist[i - 3],
                        out_dist[i - 2],
                        out_dist[i - 1],
                        out_dist[i]);
                if (has_norms) {
                    out_dist[i - 3] /= norms_array[0];
                    out_dist[i - 2] /= norms_array[1];
                    out_dist[i - 1] /= norms_array[2];
                    out_dist[i] /= norms_array[3];
                }
            } else {
                fvec_L2sqr_batch_4(
                        x,
                        codes_array[0],
                        codes_array[1],
                        codes_array[2],
                        codes_array[3],
                        d,
                        out_dist[i - 3],
                        out_dist[i - 2],
                        out_dist[i - 1],
                        out_dist[i]);
            }
        }
    }
    // leftover
    for (size_t i = 0; i < counter; i++) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            out_dist[num_keys - counter + i] =
                    fvec_inner_product(x, codes_array[i], d);
            if (has_norms) {
                out_dist[num_keys - counter + i] /= norms_array[i];
            }
        } else {
            out_dist[num_keys - counter + i] = fvec_L2sqr(x, codes_array[i], d);
        }
    }
}

// Path-D step 10.4: CC-leaf search/range_search_preassigned delegate
// to cc_impl:: helpers. The `invlists` pointer is always a
// ConcurrentArrayInvertedLists here — set by the ctor via
// replace_invlists — so the static_cast is safe.
void IndexIVFFlatCC::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    const auto* cil = static_cast<const ConcurrentArrayInvertedLists*>(
            this->invlists);
    cc_impl::search_preassigned(
            *this,
            *cil,
            n,
            x,
            k,
            assign,
            centroid_dis,
            distances,
            labels,
            store_pairs,
            params,
            stats);
}

void IndexIVFFlatCC::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        faiss::RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    const auto* cil = static_cast<const ConcurrentArrayInvertedLists*>(
            this->invlists);
    cc_impl::range_search_preassigned(
            *this,
            *cil,
            nx,
            x,
            radius,
            keys,
            coarse_dis,
            result,
            store_pairs,
            params,
            stats);
}

IndexIVFFlatCCCosine::IndexIVFFlatCCCosine(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t ssize,
        MetricType metric)
        : IndexIVFFlatCC(quantizer, d, nlist, ssize, metric) {
    replace_invlists(new ConcurrentArrayInvertedLists(nlist, code_size, ssize, true), true);
}

IndexIVFFlatCCCosine::IndexIVFFlatCCCosine() {
}

void IndexIVFFlatCCCosine::train(idx_t n, const float* x) {
    auto x_normalized = ::knowhere::CopyAndNormalizeVecs(x, n, d);
    IndexIVF::train(n, x_normalized.get());
}

void IndexIVFFlatCCCosine::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    auto x_normalized = std::make_unique<float[]>(n * d);
    std::memcpy(x_normalized.get(), x, n * d * sizeof(float));
    auto norms = ::knowhere::NormalizeVecs(x_normalized.get(), n, d);
    quantizer->assign(n, x_normalized.get(), coarse_idx.get());
    add_core_with_norms(n, x, norms.data(), xids, coarse_idx.get());
}

}
