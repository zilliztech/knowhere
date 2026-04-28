// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

#pragma once

#include <faiss/IndexFlatCodes.h>
#include <faiss/MetricType.h>
#include <faiss/impl/DistanceComputer.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/**
 * Storage class for 1-bit-per-dimension binary vectors, with Hamming /
 * Jaccard / Substructure / Superstructure semantics. Acts as the storage
 * under the fork's IndexHNSW family — a direct replacement for the legacy
 * path that routed binary data through faiss::ScalarQuantizer with
 * qtype == QT_1bit_direct.
 *
 * Input/output convention for sa_encode / sa_decode and set_query: the
 * `float*` buffer carries per-byte integer values (0..255) that together
 * represent the bit-packed binary vector. The first code_size entries of
 * each d-float "vector" are meaningful; any remaining lanes are ignored
 * on both encode and decode — decode does not touch them, so callers may
 * allocate exactly code_size floats per vector.
 */
struct IndexBinaryScalarQuantizer : faiss::IndexFlatCodes {
    IndexBinaryScalarQuantizer();

    /// d is the number of binary dimensions. code_size is (d + 7) / 8.
    /// metric must be one of METRIC_Hamming, METRIC_Jaccard,
    /// METRIC_Substructure, METRIC_Superstructure. The index is
    /// considered trained immediately after construction.
    IndexBinaryScalarQuantizer(int d, MetricType metric);

    void
    sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void
    sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /// Returns a size-specialized Hamming or Jaccard computer wired into
    /// the FlatCodesDistanceComputer interface. Uses baseline FAISS
    /// primitives from faiss/utils/hamming.h.
    faiss::FlatCodesDistanceComputer*
    get_FlatCodesDistanceComputer() const override;
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
