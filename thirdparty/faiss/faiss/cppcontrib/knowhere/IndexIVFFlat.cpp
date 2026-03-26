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

#include <faiss/cppcontrib/knowhere/FaissHook.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/cppcontrib/knowhere/utils/distances.h>
#include <faiss/cppcontrib/knowhere/utils/distances_if.h>
#include <faiss/utils/utils.h>

#include <faiss/cppcontrib/knowhere/MetricType.h>



namespace faiss::cppcontrib::knowhere {

/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexIVFFlat::IndexIVFFlat(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric)
        : IndexIVF(quantizer, d, nlist, sizeof(float) * d, metric) {
    code_size = sizeof(float) * d;
    by_residual = false;

    replace_invlists(new ArrayInvertedLists(nlist, code_size, false), true);
}

void IndexIVFFlat::train(idx_t n, const float* x) {
    IndexIVF::train(n, x);
}

void IndexIVFFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    quantizer->assign(n, x, coarse_idx.get());
    add_core(n, x, nullptr, xids, coarse_idx.get());
}


IndexIVFFlat::IndexIVFFlat() {
    by_residual = false;
}

void IndexIVFFlat::add_core(
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
    direct_map.check_can_add(xids);

    int64_t n_add = 0;

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                const float* xi_normal = (x_norms == nullptr) ? nullptr : (x_norms + i);
                size_t offset = invlists->add_entry(
                        list_no, id, (const uint8_t*)xi, xi_normal, inverted_list_context);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("IndexIVFFlat::add_core: added %" PRId64 " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}

void IndexIVFFlat::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(!by_residual);
    if (!include_listnos) {
        memcpy(codes, x, code_size * n);
    } else {
        size_t coarse_size = coarse_code_size();
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            uint8_t* code = codes + i * (code_size + coarse_size);
            const float* xi = x + i * d;
            if (list_no >= 0) {
                encode_listno(list_no, code);
                memcpy(code + coarse_size, xi, code_size);
            } else {
                memset(code, 0, code_size + coarse_size);
            }
        }
    }
}

void IndexIVFFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    size_t coarse_size = coarse_code_size();
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = bytes + i * (code_size + coarse_size);
        float* xi = x + i * d;
        memcpy(xi, code + coarse_size, code_size);
    }
}

namespace {

/*
// Baseline implementation that is kept for the reference.
template <MetricType metric, class C, bool use_sel>
struct IVFFlatScanner : InvertedListScanner {
    size_t d;

    IVFFlatScanner(size_t d, bool store_pairs, const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), d(d) {}

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
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
            const float* code_norms,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        const float* list_vecs = (const float*)codes;
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + d * j;
            if (use_sel && !sel->is_member(ids[j])) {
                continue;
            }
            float dis = metric == METRIC_INNER_PRODUCT
                    ? fvec_inner_product(xi, yj, d)
                    : fvec_L2sqr(xi, yj, d);
            if (code_norms) {
                dis /= code_norms[j];
            }
            if (C::cmp(simi[0], dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                heap_replace_top<C>(k, simi, idxi, dis, id);
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + d * j;
            if (use_sel && !sel->is_member(ids[j])) {
                continue;
            }
            float dis = metric == METRIC_INNER_PRODUCT
                    ? fvec_inner_product(xi, yj, d)
                    : fvec_L2sqr(xi, yj, d);
            if (code_norms) {
                dis /= code_norms[j];
            }
            if (C::cmp(radius, dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
            }
        }
    }
};
*/

template <MetricType metric, class C, bool use_sel>
struct IVFFlatScanner : InvertedListScanner {
    size_t d;

    IVFFlatScanner(size_t d, bool store_pairs, const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), d(d) {
        keep_max = faiss::cppcontrib::knowhere::is_similarity_metric(metric);
    }

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
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
            const float* code_norms,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k,
            size_t& scan_cnt) const override {
        const float* list_vecs = (const float*)codes;
        size_t nup = 0;

        // the lambda that filters acceptable elements.
        auto filter =
            [&](const size_t j) { return (!use_sel || sel->is_member(ids[j])); };

        // the lambda that applies a valid element.
        auto apply =
            [&](const float dis_in, const size_t j) {
                const float dis = (code_norms == nullptr) ? dis_in : (dis_in / code_norms[j]);
                scan_cnt++;
                if (C::cmp(simi[0], dis)) {
                    const int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    heap_replace_top<C>(k, simi, idxi, dis, id);
                    nup++;
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
            const float* code_norms,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        const float* list_vecs = (const float*)codes;

        // the lambda that filters acceptable elements.
        auto filter = [&](const size_t j) {
            return (!use_sel || sel->is_member(ids[j]));
        };
        // the lambda that applies a valid element.
        auto apply = [&](const float dis_in, const size_t j) {
            const float dis =
                    (code_norms == nullptr) ? dis_in : (dis_in / code_norms[j]);
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
            const float* code_norms,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;

        // the lambda that filters acceptable elements.
        auto filter =
            [&](const size_t j) { return (!use_sel || sel->is_member(ids[j])); };

        // the lambda that applies a filtered element.
        auto apply =
            [&](const float dis_in, const size_t j) {
                const float dis = (code_norms == nullptr) ? dis_in : (dis_in / code_norms[j]);
                if (C::cmp(radius, dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    res.add(dis, id);
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

    IVFFlatBitsetViewScanner(size_t d, bool store_pairs, const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), d(d) {
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
            const float* __restrict code_norms,
            const idx_t* __restrict ids,
            float* __restrict simi,
            idx_t* __restrict idxi,
            size_t k,
            size_t& scan_cnt) const override {
        const float* list_vecs = (const float*)codes;
        size_t nup = 0;

        // the lambda that filters acceptable elements.
        auto filter =
            [&](const size_t j) { return (!use_sel || !bitset.test(ids[j])); };

        // the lambda that applies a valid element.
        auto apply =
            [&](const float dis_in, const size_t j) {
                const float dis = (code_norms == nullptr) ? dis_in : (dis_in / code_norms[j]);
                scan_cnt++;
                if (C::cmp(simi[0], dis)) {
                    const int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    heap_replace_top<C>(k, simi, idxi, dis, id);
                    nup++;
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
            const float* code_norms,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        const float* list_vecs = (const float*)codes;

        // the lambda that filters acceptable elements.
        auto filter = [&](const size_t j) {
            return (!use_sel || !bitset.test(ids[j]));
        };
        // the lambda that applies a valid element.
        auto apply = [&](const float dis_in, const size_t j) {
            const float dis =
                    (code_norms == nullptr) ? dis_in : (dis_in / code_norms[j]);
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
            const float* __restrict code_norms,
            const idx_t* __restrict ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;

        // the lambda that filters acceptable elements.
        auto filter =
            [&](const size_t j) { return (!use_sel || !bitset.test(ids[j])); };

        // the lambda that applies a filtered element.
        auto apply =
            [&](const float dis_in, const size_t j) {
                const float dis = (code_norms == nullptr) ? dis_in : (dis_in / code_norms[j]);
                if (C::cmp(radius, dis)) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    res.add(dis, id);
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
                    use_sel>(ivf->d, store_pairs, sel);
        } else if (ivf->metric_type == METRIC_L2) {
            return new IVFFlatBitsetViewScanner<METRIC_L2, CMax<float, int64_t>, use_sel>(
                    ivf->d, store_pairs, sel);
        } else {
            FAISS_THROW_MSG("metric type not supported");
        }
    }

    // default faiss version
    if (ivf->metric_type == METRIC_INNER_PRODUCT) {
        return new IVFFlatScanner<
                METRIC_INNER_PRODUCT,
                CMin<float, int64_t>,
                use_sel>(ivf->d, store_pairs, sel);
    } else if (ivf->metric_type == METRIC_L2) {
        return new IVFFlatScanner<METRIC_L2, CMax<float, int64_t>, use_sel>(
                ivf->d, store_pairs, sel);
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

void IndexIVFFlat::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
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
    add_core(n, x, norms.data(), xids, coarse_idx.get());
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
    add_core(n, x, norms.data(), xids, coarse_idx.get());
}

}


