// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef KNOWHERE_SIMD_SPARSE_SIMD_H
#define KNOWHERE_SIMD_SPARSE_SIMD_H

#include <boost/core/span.hpp>
#include <vector>

#include "knowhere/sparse_utils.h"

namespace knowhere::sparse {

// ============================================================================
// AVX512 SIMD Implementation (16-wide vectorization with hardware scatter)
// ============================================================================
// Implementation in sparse_simd_avx512.cpp (compiled with -mavx512f)
// This function uses runtime CPU detection and is only called when AVX512 is available
template <typename QType>
std::vector<float>
compute_all_distances_avx512(size_t n_rows_internal, const std::vector<std::pair<size_t, float>>& q_vec,
                             const std::vector<boost::span<const table_t>>& inverted_index_ids_spans,
                             const std::vector<boost::span<const QType>>& inverted_index_vals_spans,
                             const DocValueComputer<float>& computer, SparseMetricType metric_type,
                             const boost::span<const float>* doc_len_ratios_spans_ptr);

}  // namespace knowhere::sparse

#endif  // KNOWHERE_SIMD_SPARSE_SIMD_H
