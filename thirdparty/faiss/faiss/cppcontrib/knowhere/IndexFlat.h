/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/IndexFlat.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// Thin subclass of baseline faiss::IndexFlat kept in the knowhere namespace
// ONLY to override search() for METRIC_Jaccard.
//
// Rationale: the fork's Jaccard is Tanimoto distance
//   1 - <x,y> / (||x||^2 + ||y||^2 - <x,y>)
// whereas baseline's METRIC_Jaccard is Ruzicka similarity
//   sum_i min(x_i, y_i) / sum_i max(x_i, y_i).
// The two agree on 0/1-valued inputs but diverge in general. Preserving
// Tanimoto keeps backward compatibility with persisted knowhere indexes.
//
// Everything else (L2/IP search, range_search, distance computers,
// reconstruct, sa_encode/decode, IndexFlatIP / IndexFlatL2 / IndexFlat1D
// siblings) is used directly from baseline faiss. IndexFlatCosine extends
// THIS class rather than baseline directly so the Jaccard override remains
// reachable from cosine-normalized data paths.
struct IndexFlat : ::faiss::IndexFlat {
    IndexFlat() = default;
    explicit IndexFlat(idx_t d, MetricType metric = METRIC_L2)
            : ::faiss::IndexFlat(d, metric) {}

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

}
}
} // namespace faiss
