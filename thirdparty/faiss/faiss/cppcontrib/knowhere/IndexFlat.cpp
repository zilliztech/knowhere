/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IndexFlat.h>

#include <faiss/cppcontrib/knowhere/utils/distances.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/Heap.h>

namespace faiss::cppcontrib::knowhere {

void IndexFlat::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    // Fork-specific: METRIC_Jaccard is Tanimoto distance (see header).
    // All other metrics delegate to baseline IndexFlat::search.
    if (metric_type == METRIC_Jaccard) {
        ::faiss::IDSelector* sel = params ? params->sel : nullptr;
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_jaccard(x, get_xb(), d, n, ntotal, &res, sel);
        return;
    }

    ::faiss::IndexFlat::search(n, x, k, distances, labels, params);
}

} // namespace faiss::cppcontrib::knowhere
