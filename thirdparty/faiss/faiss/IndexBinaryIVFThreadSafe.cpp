/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexBinaryIVF.h>

#include <omp.h>
#include <cinttypes>
#include <cstdio>

#include <algorithm>
#include <memory>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexLSH.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/binary_distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/jaccard-inl.h>
#include <faiss/utils/utils.h>
#include <cinttypes>
namespace faiss {

void IndexBinaryIVF::search_thread_safe(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        size_t nprobe,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    nprobe = std::min(nlist, nprobe);
    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe]);

    double t0 = getmillisecs();
    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(idx.get(), n * nprobe);

    search_preassigned_thread_safe(
            n,
            x,
            k,
            idx.get(),
            coarse_dis.get(),
            distances,
            labels,
            false,
            nullptr,
            nprobe,
            bitset);
    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexBinaryIVF::search_and_reconstruct_thread_safe(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        uint8_t* recons,
        size_t nprobe) const {
    nprobe = std::min(nlist, nprobe);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

    invlists->prefetch_lists(idx.get(), n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned(
            n,
            x,
            k,
            idx.get(),
            coarse_dis.get(),
            distances,
            labels,
            /* store_pairs */ true);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = 0; j < k; ++j) {
            idx_t ij = i * k + j;
            idx_t key = labels[ij];
            uint8_t* reconstructed = recons + ij * d;
            if (key < 0) {
                // Fill with NaNs
                memset(reconstructed, -1, sizeof(*reconstructed) * d);
            } else {
                int list_no = key >> 32;
                int offset = key & 0xffffffff;

                // Update label to the actual id
                labels[ij] = invlists->get_single_id(list_no, offset);

                reconstruct_from_offset(list_no, offset, reconstructed);
            }
        }
    }
}

namespace {

using idx_t = Index::idx_t;

template <class HammingComputer>
struct IVFBinaryScannerL2 : BinaryInvertedListScanner {
    HammingComputer hc;
    size_t code_size;
    bool store_pairs;

    IVFBinaryScannerL2(size_t code_size, bool store_pairs)
            : code_size(code_size), store_pairs(store_pairs) {}

    void set_query(const uint8_t* query_vector) override {
        hc.set(query_vector, code_size);
    }

    idx_t list_no;
    void set_list(idx_t list_no, uint8_t /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* code) const override {
        return hc.compute(code);
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            int32_t* simi,
            idx_t* idxi,
            size_t k,
            const BitsetView bitset) const override {
        using C = CMax<int32_t, idx_t>;

        size_t nup = 0;
        for (size_t j = 0; j < n; j++) {
            if (bitset.empty() || !bitset.test(ids[j])) {
                float dis = hc.compute(codes);
                if (dis < simi[0]) {
                    idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    heap_replace_top<C>(k, simi, idxi, dis, id);
                    nup++;
                }
            }
            codes += code_size;
        }
        return nup;
    }

    void scan_codes_range(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& result,
            const BitsetView bitset) const override {
        for (size_t j = 0; j < n; j++) {
            if (bitset.empty() || !bitset.test(ids[j])) {
                float dis = hc.compute(codes);
                if (dis < radius) {
                    int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    result.add(dis, id);
                }
            }
            codes += code_size;
        }
    }
};

template <class DistanceComputer, bool store_pairs>
struct IVFBinaryScannerJaccard : BinaryInvertedListScanner {
    DistanceComputer hc;
    size_t code_size;

    IVFBinaryScannerJaccard(size_t code_size) : code_size(code_size) {}

    void set_query(const uint8_t* query_vector) override {
        hc.set(query_vector, code_size);
    }

    idx_t list_no;
    void set_list(idx_t list_no, uint8_t /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* code) const override {
        return hc.compute(code);
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            int32_t* simi,
            idx_t* idxi,
            size_t k,
            const BitsetView bitset) const override {
        using C = CMax<float, idx_t>;
        float* psimi = (float*)simi;
        size_t nup = 0;
        for (size_t j = 0; j < n; j++) {
            if (bitset.empty() || !bitset.test(ids[j])) {
                float dis = hc.compute(codes);
                if (dis < psimi[0]) {
                    idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    heap_replace_top<C>(k, psimi, idxi, dis, id);
                    nup++;
                }
            }
            codes += code_size;
        }
        return nup;
    }

    void scan_codes_range(
            size_t n,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& result,
            const BitsetView bitset) const override {
        for (size_t j = 0; j < n; j++) {
            if (bitset.empty() || !bitset.test(ids[j])) {
                float dis = hc.compute(codes);
                if (dis < radius) {
                    idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                    result.add(dis, id);
                }
            }
            codes += code_size;
        }
    }
};

template <bool store_pairs>
BinaryInvertedListScanner* select_IVFBinaryScannerL2(size_t code_size) {
#define HC(name) return new IVFBinaryScannerL2<name>(code_size, store_pairs)
    switch (code_size) {
        case 4:
            HC(HammingComputer4);
        case 8:
            HC(HammingComputer8);
        case 16:
            HC(HammingComputer16);
        case 20:
            HC(HammingComputer20);
        case 32:
            HC(HammingComputer32);
        case 64:
            HC(HammingComputer64);
        default:
            HC(HammingComputerDefault);
    }
#undef HC
}

template <bool store_pairs>
BinaryInvertedListScanner* select_IVFBinaryScannerJaccard(size_t code_size) {
#define HANDLE_CS(cs)                                                         \
    case cs:                                                                  \
        return new IVFBinaryScannerJaccard<JaccardComputer##cs, store_pairs>( \
                cs);
    switch (code_size) {
        HANDLE_CS(16)
        HANDLE_CS(32)
        HANDLE_CS(64)
        HANDLE_CS(128)
        HANDLE_CS(256)
        HANDLE_CS(512)
        default:
            return new IVFBinaryScannerJaccard<
                    JaccardComputerDefault,
                    store_pairs>(code_size);
    }
#undef HANDLE_CS
}

void search_knn_hamming_heap_thread_safe(
        const IndexBinaryIVF& ivf,
        size_t n,
        const uint8_t* x,
        idx_t k,
        const idx_t* keys,
        const int32_t* coarse_dis,
        int32_t* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        idx_t nprobe,
        const BitsetView bitset) {
    nprobe = params ? params->nprobe : nprobe;
    nprobe = std::min((idx_t)ivf.nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : ivf.max_codes;
    MetricType metric_type = ivf.metric_type;

    // almost verbatim copy from IndexIVF::search_preassigned

    size_t nlistv = 0, ndis = 0, nheap = 0;
    using HeapForIP = CMin<int32_t, idx_t>;
    using HeapForL2 = CMax<int32_t, idx_t>;

#pragma omp parallel if (n > 1) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<BinaryInvertedListScanner> scanner(
                ivf.get_InvertedListScanner(store_pairs));

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* xi = x + i * ivf.code_size;
            scanner->set_query(xi);

            const idx_t* keysi = keys + i * nprobe;
            int32_t* simi = distances + k * i;
            idx_t* idxi = labels + k * i;

            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }

            size_t nscan = 0;

            for (size_t ik = 0; ik < nprobe; ik++) {
                idx_t key = keysi[ik]; /* select the list  */
                if (key < 0) {
                    // not enough centroids for multiprobe
                    continue;
                }
                FAISS_THROW_IF_NOT_FMT(
                        key < (idx_t)ivf.nlist,
                        "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                        key,
                        ik,
                        ivf.nlist);

                scanner->set_list(key, coarse_dis[i * nprobe + ik]);

                nlistv++;

                size_t list_size = ivf.invlists->list_size(key);
                InvertedLists::ScopedCodes scodes(ivf.invlists, key);
                std::unique_ptr<InvertedLists::ScopedIds> sids;
                const Index::idx_t* ids = nullptr;

                if (!store_pairs) {
                    sids.reset(new InvertedLists::ScopedIds(ivf.invlists, key));
                    ids = sids->get();
                }

                nheap += scanner->scan_codes(
                        list_size, scodes.get(), ids, simi, idxi, k, bitset);

                nscan += list_size;
                if (max_codes && nscan >= max_codes)
                    break;
            }

            ndis += nscan;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }

        } // parallel for
    }     // parallel

    indexIVF_stats.nq += n;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nheap_updates += nheap;
}

void search_knn_binary_dis_heap_thread_safe(
        const IndexBinaryIVF& ivf,
        size_t n,
        const uint8_t* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        idx_t nprobe,
        const BitsetView bitset) {
    nprobe = params ? params->nprobe : nprobe;
    nprobe = std::min((idx_t)ivf.nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : ivf.max_codes;
    MetricType metric_type = ivf.metric_type;

    // almost verbatim copy from IndexIVF::search_preassigned

    size_t nlistv = 0, ndis = 0, nheap = 0;
    using HeapForJaccard = CMax<float, idx_t>;

#pragma omp parallel if (n > 1) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<BinaryInvertedListScanner> scanner(
                ivf.get_InvertedListScanner(store_pairs));

#pragma omp for
        for (size_t i = 0; i < n; i++) {
            const uint8_t* xi = x + i * ivf.code_size;
            scanner->set_query(xi);

            const idx_t* keysi = keys + i * nprobe;
            float* simi = distances + k * i;
            idx_t* idxi = labels + k * i;

            heap_heapify<HeapForJaccard>(k, simi, idxi);

            size_t nscan = 0;

            for (size_t ik = 0; ik < nprobe; ik++) {
                idx_t key = keysi[ik]; /* select the list  */
                if (key < 0) {
                    // not enough centroids for multiprobe
                    continue;
                }
                FAISS_THROW_IF_NOT_FMT(
                        key < (idx_t)ivf.nlist,
                        "Invalid key=%" SCNd64 "  at ik=%ld nlist=%ld\n",
                        key,
                        ik,
                        ivf.nlist);

                scanner->set_list(key, (int32_t)coarse_dis[i * nprobe + ik]);

                nlistv++;

                size_t list_size = ivf.invlists->list_size(key);
                InvertedLists::ScopedCodes scodes(ivf.invlists, key);
                std::unique_ptr<InvertedLists::ScopedIds> sids;
                const Index::idx_t* ids = nullptr;

                if (!store_pairs) {
                    sids.reset(new InvertedLists::ScopedIds(ivf.invlists, key));
                    ids = sids->get();
                }

                nheap += scanner->scan_codes(
                        list_size,
                        scodes.get(),
                        ids,
                        (int32_t*)simi,
                        idxi,
                        k,
                        bitset);

                nscan += list_size;
                if (max_codes && nscan >= max_codes)
                    break;
            }

            ndis += nscan;
            heap_reorder<HeapForJaccard>(k, simi, idxi);

        } // parallel for
    }     // parallel

    indexIVF_stats.nq += n;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nheap_updates += nheap;
}

template <class HammingComputer, bool store_pairs>
void search_knn_hamming_count_thread_safe(
        const IndexBinaryIVF& ivf,
        size_t nx,
        const uint8_t* x,
        const idx_t* keys,
        int k,
        int32_t* distances,
        idx_t* labels,
        const IVFSearchParameters* params,
        idx_t nprobe,
        const BitsetView bitset) {
    const int nBuckets = ivf.d + 1;
    std::vector<int> all_counters(nx * nBuckets, 0);
    std::unique_ptr<idx_t[]> all_ids_per_dis(new idx_t[nx * nBuckets * k]);

    nprobe = params ? params->nprobe : nprobe;
    nprobe = std::min((idx_t)ivf.nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : ivf.max_codes;

    std::vector<HCounterState<HammingComputer>> cs;
    for (size_t i = 0; i < nx; ++i) {
        cs.push_back(HCounterState<HammingComputer>(
                all_counters.data() + i * nBuckets,
                all_ids_per_dis.get() + i * nBuckets * k,
                x + i * ivf.code_size,
                ivf.d,
                k));
    }

    size_t nlistv = 0, ndis = 0;

#pragma omp parallel for reduction(+ : nlistv, ndis)
    for (int64_t i = 0; i < nx; i++) {
        const idx_t* keysi = keys + i * nprobe;
        HCounterState<HammingComputer>& csi = cs[i];

        size_t nscan = 0;

        for (size_t ik = 0; ik < nprobe; ik++) {
            idx_t key = keysi[ik]; /* select the list  */
            if (key < 0) {
                // not enough centroids for multiprobe
                continue;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)ivf.nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    ik,
                    ivf.nlist);

            nlistv++;
            size_t list_size = ivf.invlists->list_size(key);
            InvertedLists::ScopedCodes scodes(ivf.invlists, key);
            const uint8_t* list_vecs = scodes.get();
            const Index::idx_t* ids =
                    store_pairs ? nullptr : ivf.invlists->get_ids(key);

            for (size_t j = 0; j < list_size; j++) {
                if (bitset.empty() || !bitset.test(ids[j])) {
                    const uint8_t* yj = list_vecs + ivf.code_size * j;

                    idx_t id = store_pairs ? (key << 32 | j) : ids[j];
                    csi.update_counter(yj, id);
                }
            }
            if (ids)
                ivf.invlists->release_ids(key, ids);

            nscan += list_size;
            if (max_codes && nscan >= max_codes)
                break;
        }
        ndis += nscan;

        int nres = 0;
        for (int b = 0; b < nBuckets && nres < k; b++) {
            for (int l = 0; l < csi.counters[b] && nres < k; l++) {
                labels[i * k + nres] = csi.ids_per_dis[b * k + l];
                distances[i * k + nres] = b;
                nres++;
            }
        }
        while (nres < k) {
            labels[i * k + nres] = -1;
            distances[i * k + nres] = std::numeric_limits<int32_t>::max();
            ++nres;
        }
    }

    indexIVF_stats.nq += nx;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}

template <bool store_pairs>
void search_knn_hamming_count_1(
        const IndexBinaryIVF& ivf,
        size_t nx,
        const uint8_t* x,
        const idx_t* keys,
        int k,
        int32_t* distances,
        idx_t* labels,
        const IVFSearchParameters* params,
        const size_t nprobe,
        const BitsetView bitset) {
    switch (ivf.code_size) {
#define HANDLE_CS(cs)                         \
    case cs:                                  \
        search_knn_hamming_count_thread_safe< \
                HammingComputer##cs,          \
                store_pairs>(                 \
                ivf,                          \
                nx,                           \
                x,                            \
                keys,                         \
                k,                            \
                distances,                    \
                labels,                       \
                params,                       \
                nprobe,                       \
                bitset);                      \
        break;
        HANDLE_CS(4);
        HANDLE_CS(8);
        HANDLE_CS(16);
        HANDLE_CS(20);
        HANDLE_CS(32);
        HANDLE_CS(64);
#undef HANDLE_CS
        default:
            search_knn_hamming_count_thread_safe<
                    HammingComputerDefault,
                    store_pairs>(
                    ivf,
                    nx,
                    x,
                    keys,
                    k,
                    distances,
                    labels,
                    params,
                    nprobe,
                    bitset);
            break;
    }
}

} // namespace

void IndexBinaryIVF::search_preassigned_thread_safe(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        const idx_t* idx,
        const int32_t* coarse_dis,
        int32_t* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        const size_t nprobe,
        const BitsetView bitset) const {
    if (metric_type == METRIC_Jaccard) {
        if (use_heap) {
            float* D = new float[k * n];
            float* c_dis = new float[n * nprobe];
            memcpy(c_dis, coarse_dis, sizeof(float) * n * nprobe);
            search_knn_binary_dis_heap_thread_safe(
                    *this,
                    n,
                    x,
                    k,
                    idx,
                    c_dis,
                    D,
                    labels,
                    store_pairs,
                    params,
                    nprobe,
                    bitset);
            memcpy(distances, D, sizeof(float) * n * k);
            delete[] D;
            delete[] c_dis;
        } else {
            // not implemented
        }
    } else if (
            metric_type == METRIC_Substructure ||
            metric_type == METRIC_Superstructure) {
        // unsupported
    } else {
        if (use_heap) {
            search_knn_hamming_heap_thread_safe(
                    *this,
                    n,
                    x,
                    k,
                    idx,
                    coarse_dis,
                    distances,
                    labels,
                    store_pairs,
                    params,
                    nprobe,
                    bitset);
        } else {
            if (store_pairs) {
                search_knn_hamming_count_1<true>(
                        *this,
                        n,
                        x,
                        idx,
                        k,
                        distances,
                        labels,
                        params,
                        nprobe,
                        bitset);
            } else {
                search_knn_hamming_count_1<false>(
                        *this,
                        n,
                        x,
                        idx,
                        k,
                        distances,
                        labels,
                        params,
                        nprobe,
                        bitset);
            }
        }
    }
}

void IndexBinaryIVF::range_search_thread_safe(
        idx_t n,
        const uint8_t* x,
        float radius,
        RangeSearchResult* res,
        size_t nprobe,
        const BitsetView bitset) const {
    nprobe = std::min(nlist, nprobe);
    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe]);

    double t0 = getmillisecs();
    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(idx.get(), n * nprobe);
    range_search_preassigned_thread_safe(
            n, x, radius, idx.get(), coarse_dis.get(), res, nprobe, bitset);
    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexBinaryIVF::range_search_preassigned_thread_safe(
        idx_t n,
        const uint8_t* x,
        float radius,
        const idx_t* assign,
        const int32_t* centroid_dis,
        RangeSearchResult* res,
        size_t nprobe,
        const BitsetView bitset) const {
    nprobe = std::min(nlist, nprobe);
    bool store_pairs = false;
    size_t nlistv = 0, ndis = 0;

    std::vector<RangeSearchPartialResult*> all_pres(omp_get_max_threads());

#pragma omp parallel reduction(+ : nlistv, ndis)
    {
        RangeSearchPartialResult pres(res);
        std::unique_ptr<BinaryInvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs));
        FAISS_THROW_IF_NOT(scanner.get());

        all_pres[omp_get_thread_num()] = &pres;

        auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult& qres) {
            idx_t key = assign[i * nprobe + ik]; /* select the list  */
            if (key < 0)
                return;
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    ik,
                    nlist);
            const size_t list_size = invlists->list_size(key);

            if (list_size == 0)
                return;

            InvertedLists::ScopedCodes scodes(invlists, key);
            InvertedLists::ScopedIds ids(invlists, key);

            scanner->set_list(key, assign[i * nprobe + ik]);
            nlistv++;
            ndis += list_size;
            scanner->scan_codes_range(
                    list_size, scodes.get(), ids.get(), radius, qres, bitset);
        };

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            scanner->set_query(x + i * code_size);

            RangeQueryResult& qres = pres.new_result(i);
            size_t prev_nres = qres.nres;

            for (size_t ik = 0; ik < nprobe; ik++) {
                scan_list_func(i, ik, qres);
                if (qres.nres == prev_nres) break;
                prev_nres = qres.nres;
            }
        }

        pres.finalize();
    }
    indexIVF_stats.nq += n;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}

} // namespace faiss
