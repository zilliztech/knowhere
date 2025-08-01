/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlat.h>

#include <omp.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>

#include "knowhere/object.h"
#include "knowhere/utils.h"
#include "knowhere/bitsetview_idselector.h"

#include <faiss/IndexFlat.h>

#include <faiss/FaissHook.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/distances_if.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexIVFFlat::IndexIVFFlat(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        bool is_cosine)
        : IndexIVF(quantizer, d, nlist, sizeof(float) * d, metric) {
    this->is_cosine = is_cosine;
    code_size = sizeof(float) * d;
    by_residual = false;

    replace_invlists(new ArrayInvertedLists(nlist, code_size, is_cosine), true);
}

void IndexIVFFlat::train(idx_t n, const float* x) {
    if (is_cosine) {
        auto x_normalized = knowhere::CopyAndNormalizeVecs(x, n, d);
        // use normalized data to train codes for cosine
        IndexIVF::train(n, x_normalized.get());
    } else {
        IndexIVF::train(n, x);
    }
}

void IndexIVFFlat::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    if (is_cosine) {
        auto x_normalized = std::make_unique<float[]>(n * d);
        std::memcpy(x_normalized.get(), x, n * d * sizeof(float));
        auto norms = knowhere::NormalizeVecs(x_normalized.get(), n, d);
        // use normalized data to calculate coarse id
        quantizer->assign(n, x_normalized.get(), coarse_idx.get());
        // add raw data with its norms to inverted list
        add_core(n, x, norms.data(), xids, coarse_idx.get());
    } else {
        quantizer->assign(n, x, coarse_idx.get());
        add_core(n, x, nullptr, xids, coarse_idx.get());
    }
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
        keep_max = is_similarity_metric(metric);
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
            std::vector<knowhere::DistId>& out) const override {
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
    knowhere::BitsetView bitset;

    IVFFlatBitsetViewScanner(size_t d, bool store_pairs, const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), d(d) {
        const auto* bitsetview_sel = dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);
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
            std::vector<knowhere::DistId>& out) const override {
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
    if (const auto* bitsetview_sel = dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
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

IndexIVFFlatCC::IndexIVFFlatCC(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t ssize,
        MetricType metric,
        bool is_cosine)
        : IndexIVFFlat(quantizer, d, nlist, metric, is_cosine) {
    replace_invlists(new ConcurrentArrayInvertedLists(nlist, code_size, ssize, is_cosine), true);
}

IndexIVFFlatCC::IndexIVFFlatCC() {}

/*****************************************
 * IndexIVFFlatDedup implementation
 ******************************************/

IndexIVFFlatDedup::IndexIVFFlatDedup(
        Index* quantizer,
        size_t d,
        size_t nlist_,
        MetricType metric_type)
        : IndexIVFFlat(quantizer, d, nlist_, metric_type) {}

void IndexIVFFlatDedup::train(idx_t n, const float* x) {
    std::unordered_map<uint64_t, idx_t> map;
    std::unique_ptr<float[]> x2(new float[n * d]);

    int64_t n2 = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t hash = hash_bytes((uint8_t*)(x + i * d), code_size);
        if (map.count(hash) &&
            !memcmp(x2.get() + map[hash] * d, x + i * d, code_size)) {
            // is duplicate, skip
        } else {
            map[hash] = n2;
            memcpy(x2.get() + n2 * d, x + i * d, code_size);
            n2++;
        }
    }
    if (verbose) {
        printf("IndexIVFFlatDedup::train: train on %" PRId64
               " points after dedup "
               "(was %" PRId64 " points)\n",
               n2,
               n);
    }
    IndexIVFFlat::train(n2, x2.get());
}

void IndexIVFFlatDedup::add_with_ids(
        idx_t na,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    assert(invlists);
    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "IVFFlatDedup not implemented with direct_map");
    std::unique_ptr<int64_t[]> idx(new int64_t[na]);
    quantizer->assign(na, x, idx.get());

    int64_t n_add = 0, n_dup = 0;

#pragma omp parallel reduction(+ : n_add, n_dup)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < na; i++) {
            int64_t list_no = idx[i];

            if (list_no < 0 || list_no % nt != rank) {
                continue;
            }

            idx_t id = xids ? xids[i] : ntotal + i;
            const float* xi = x + i * d;

            // search if there is already an entry with that id
            InvertedLists::ScopedCodes codes(invlists, list_no);

            int64_t n = invlists->list_size(list_no);
            int64_t offset = -1;
            for (int64_t o = 0; o < n; o++) {
                if (!memcmp(codes.get() + o * code_size, xi, code_size)) {
                    offset = o;
                    break;
                }
            }

            if (offset == -1) { // not found
                invlists->add_entry(list_no, id, (const uint8_t*)xi);
            } else {
                // mark equivalence
                idx_t id2 = invlists->get_single_id(list_no, offset);
                std::pair<idx_t, idx_t> pair(id2, id);

#pragma omp critical
                // executed by one thread at a time
                instances.insert(pair);

                n_dup++;
            }
            n_add++;
        }
    }
    if (verbose) {
        printf("IndexIVFFlat::add_with_ids: added %" PRId64 " / %" PRId64
               " vectors"
               " (out of which %" PRId64 " are duplicates)\n",
               n_add,
               na,
               n_dup);
    }
    ntotal += n_add;
}

void IndexIVFFlatDedup::search_preassigned(
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
    FAISS_THROW_IF_NOT_MSG(
            !store_pairs, "store_pairs not supported in IVFDedup");

    IndexIVFFlat::search_preassigned(
            n, x, k, assign, centroid_dis, distances, labels, false, params);

    std::vector<idx_t> labels2(k);
    std::vector<float> dis2(k);

    for (int64_t i = 0; i < n; i++) {
        idx_t* labels1 = labels + i * k;
        float* dis1 = distances + i * k;
        int64_t j = 0;
        for (; j < k; j++) {
            if (instances.find(labels1[j]) != instances.end()) {
                // a duplicate: special handling
                break;
            }
        }
        if (j < k) {
            // there are duplicates, special handling
            int64_t j0 = j;
            int64_t rp = j;
            while (j < k) {
                auto range = instances.equal_range(labels1[rp]);
                float dis = dis1[rp];
                labels2[j] = labels1[rp];
                dis2[j] = dis;
                j++;
                for (auto it = range.first; j < k && it != range.second; ++it) {
                    labels2[j] = it->second;
                    dis2[j] = dis;
                    j++;
                }
                rp++;
            }
            memcpy(labels1 + j0,
                   labels2.data() + j0,
                   sizeof(labels1[0]) * (k - j0));
            memcpy(dis1 + j0, dis2.data() + j0, sizeof(dis2[0]) * (k - j0));
        }
    }
}

size_t IndexIVFFlatDedup::remove_ids(const IDSelector& sel) {
    std::unordered_map<idx_t, idx_t> replace;
    std::vector<std::pair<idx_t, idx_t>> toadd;
    for (auto it = instances.begin(); it != instances.end();) {
        if (sel.is_member(it->first)) {
            // then we erase this entry
            if (!sel.is_member(it->second)) {
                // if the second is not erased
                if (replace.count(it->first) == 0) {
                    replace[it->first] = it->second;
                } else { // remember we should add an element
                    std::pair<idx_t, idx_t> new_entry(
                            replace[it->first], it->second);
                    toadd.push_back(new_entry);
                }
            }
            it = instances.erase(it);
        } else {
            if (sel.is_member(it->second)) {
                it = instances.erase(it);
            } else {
                ++it;
            }
        }
    }

    instances.insert(toadd.begin(), toadd.end());

    // mostly copied from IndexIVF.cpp

    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "direct map remove not implemented");

    std::vector<int64_t> toremove(nlist);

#pragma omp parallel for
    for (int64_t i = 0; i < nlist; i++) {
        int64_t l0 = invlists->list_size(i), l = l0, j = 0;
        InvertedLists::ScopedIds idsi(invlists, i);
        while (j < l) {
            if (sel.is_member(idsi[j])) {
                if (replace.count(idsi[j]) == 0) {
                    l--;
                    invlists->update_entry(
                            i,
                            j,
                            invlists->get_single_id(i, l),
                            InvertedLists::ScopedCodes(invlists, i, l).get());
                } else {
                    invlists->update_entry(
                            i,
                            j,
                            replace[idsi[j]],
                            InvertedLists::ScopedCodes(invlists, i, j).get());
                    j++;
                }
            } else {
                j++;
            }
        }
        toremove[i] = l0 - l;
    }
    // this will not run well in parallel on ondisk because of possible shrinks
    int64_t nremove = 0;
    for (int64_t i = 0; i < nlist; i++) {
        if (toremove[i] > 0) {
            nremove += toremove[i];
            invlists->resize(i, invlists->list_size(i) - toremove[i]);
        }
    }
    ntotal -= nremove;
    return nremove;
}

void IndexIVFFlatDedup::range_search(
        idx_t,
        const float*,
        float,
        RangeSearchResult*,
        const SearchParameters*) const {
    FAISS_THROW_MSG("not implemented");
}

void IndexIVFFlatDedup::update_vectors(int, const idx_t*, const float*) {
    FAISS_THROW_MSG("not implemented");
}

void IndexIVFFlatDedup::reconstruct_from_offset(int64_t, int64_t, float*)
        const {
    FAISS_THROW_MSG("not implemented");
}

} // namespace faiss
