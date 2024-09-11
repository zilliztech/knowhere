/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexBinaryFlat.h>

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/binary_distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>
#include <cstring>

namespace faiss {

IndexBinaryFlat::IndexBinaryFlat(idx_t d) : IndexBinary(d) {}

IndexBinaryFlat::IndexBinaryFlat(idx_t d, MetricType metric)
        : IndexBinary(d, metric) {}

void IndexBinaryFlat::add(idx_t n, const uint8_t* x) {
    xb.insert(xb.end(), x, x + n * code_size);
    ntotal += n;
}

void IndexBinaryFlat::reset() {
    xb.clear();
    ntotal = 0;
}

void IndexBinaryFlat::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const BitsetView bitset) const {
    FAISS_THROW_IF_NOT(k > 0);

    if (metric_type == METRIC_Jaccard) {
        float* D = reinterpret_cast<float*>(distances);
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, D};
        binary_knn_hc(METRIC_Jaccard, &res, x, xb.data(), ntotal, code_size, bitset);
    } else if (metric_type == METRIC_Hamming) {
        int_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        binary_knn_hc(METRIC_Hamming, &res, x, xb.data(), ntotal, code_size, bitset);
    } else if (
            metric_type == METRIC_Substructure ||
            metric_type == METRIC_Superstructure) {
        float* D = reinterpret_cast<float*>(distances);

        // only matched ids will be chosen, not to use heap
        binary_knn_mc(
                metric_type,
                x,
                xb.data(),
                n,
                ntotal,
                k,
                code_size,
                D,
                labels,
                bitset);
    } else {
        FAISS_ASSERT_FMT(false, "invalid metric type %d", (int)metric_type);
    }
}

size_t IndexBinaryFlat::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove(&xb[code_size * j],
                        &xb[code_size * i],
                        sizeof(xb[0]) * code_size);
            }
            j++;
        }
    }
    long nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        xb.resize(ntotal * code_size);
    }
    return nremove;
}

void IndexBinaryFlat::reconstruct(idx_t key, uint8_t* recons) const {
    memcpy(recons, &(xb[code_size * key]), sizeof(*recons) * code_size);
}

void IndexBinaryFlat::range_search(
        idx_t n,
        const uint8_t* x,
        float radius,
        RangeSearchResult* result,
        const BitsetView bitset) const {
    switch (metric_type) {
        case METRIC_Jaccard: {
            binary_range_search<CMin<float, int64_t>, float>(
                    METRIC_Jaccard,
                    x,
                    xb.data(),
                    n,
                    ntotal,
                    radius,
                    code_size,
                    result,
                    bitset);
            break;
        }
        case METRIC_Hamming: {
            binary_range_search<CMin<int, int64_t>, int>(
                    METRIC_Hamming,
                    x,
                    xb.data(),
                    n,
                    ntotal,
                    static_cast<int>(radius),
                    code_size,
                    result,
                    bitset);
            break;
        }
        case METRIC_Superstructure:
            FAISS_THROW_MSG("Superstructure not support range_search");
            break;
        case METRIC_Substructure:
            FAISS_THROW_MSG("Substructure not support range_search");
            break;
        default:
            FAISS_THROW_FMT("Invalid metric type %d\n", (int)metric_type);
            break;
    }
}

} // namespace faiss
