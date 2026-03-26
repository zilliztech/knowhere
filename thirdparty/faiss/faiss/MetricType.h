/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_METRIC_TYPE_H
#define FAISS_METRIC_TYPE_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

/*
/// The metric space for vector comparison for Faiss indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
///
/// NOTE: when adding or removing values, update metric_type_from_int()
///       and metric_type_count() below.
enum MetricType {
    METRIC_INNER_PRODUCT, ///< maximum inner product search
    METRIC_L2,            ///< squared L2 search
    METRIC_L1,            ///< L1 (aka cityblock)
    METRIC_Linf,          ///< infinity distance
    METRIC_Lp,            ///< L_p distance, p is given by a faiss::Index
                          /// metric_arg

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,

    /// sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i)) where a_i, b_i > 0
    METRIC_Jaccard,
    /// Squared Euclidean distance, ignoring NaNs
    METRIC_NaNEuclidean,
    /// Gower's distance - numeric dimensions are in [0,1] and categorical
    /// dimensions are negative integers
    METRIC_GOWER,
};
*/

/// The metric space for vector comparison for Faiss indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.
enum MetricType {
    METRIC_INNER_PRODUCT = 0, ///< maximum inner product search
    METRIC_L2 = 1,            ///< squared L2 search
    METRIC_L1 = 2,            ///< L1 (aka cityblock)
    METRIC_Linf = 3,          ///< infinity distance
    METRIC_Lp = 4,            ///< L_p distance, p is given by a faiss::Index
                              /// metric_arg

    // Note: Faiss 1.7.4 defines METRIC_Jaccard=23,
    //   but Knowhere defines one as 5
    METRIC_Jaccard = 5,       ///< defined as: sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i))
                              ///< where a_i, b_i > 0
    METRIC_Hamming = 7,
    METRIC_Substructure = 8,   ///< Tversky case alpha = 0, beta = 1
    METRIC_Superstructure = 9, ///< Tversky case alpha = 1, beta = 0

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Canberra = 20,
    METRIC_BrayCurtis = 21,
    METRIC_JensenShannon = 22,
    /// Squared Eucliden distance, ignoring NaNs
    METRIC_NaNEuclidean = 24,
    /// Gower's distance - numeric dimensions are in [0,1] and categorical
    /// dimensions are negative integers
    METRIC_GOWER = 25,
    METRIC_MinHash_Jaccard = 26,
};

/// all vector indices are this type
using idx_t = int64_t;

/// this function is used to distinguish between min and max indexes since
/// we need to support similarity and dis-similarity metrics in a flexible way
constexpr bool is_similarity_metric(MetricType metric_type) {
    return ((metric_type == METRIC_INNER_PRODUCT) ||
            (metric_type == METRIC_Jaccard));
}

/// Convert an integer to MetricType with range validation.
/// Throws FaissException if the value is not a valid MetricType.
inline MetricType metric_type_from_int(int x) {
    FAISS_THROW_IF_NOT_FMT(
            (x >= METRIC_INNER_PRODUCT && x <= METRIC_Superstructure) ||
                    (x >= METRIC_Canberra && x <= METRIC_GOWER),
            "invalid metric type %d",
            x);
    return static_cast<MetricType>(x);
}

/// Count of entries in the MetricType enum.
constexpr size_t metric_type_count() {
    return (METRIC_Lp - METRIC_INNER_PRODUCT) + 1 +
            (METRIC_GOWER - METRIC_Canberra) + 1;
}

} // namespace faiss

#endif
