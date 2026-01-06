/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/MetricType.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// Knowhere must override this function until METRIC_Jaccard is fixed.
//   Basically, Knowhere uses `1 - intersection / union`,
//   while Faiss uses `intersection / union`.
//   This leads to the difference in the `is_similarity_metric()` call.

/// this function is used to distinguish between min and max indexes since
/// we need to support similarity and dis-similarity metrics in a flexible way
constexpr bool is_similarity_metric(MetricType metric_type) {
    return metric_type == METRIC_INNER_PRODUCT;
}

}
}
} // namespace faiss

