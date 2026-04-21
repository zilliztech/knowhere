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

#include <faiss/MetricType.h>
#include <faiss/cppcontrib/knowhere/IndexHNSW.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/**
 * HNSW index with a IndexBinaryScalarQuantizer storage — a replacement for
 * the legacy IndexHNSWSQ(QT_1bit_direct, metric) path. Inherits from
 * IndexHNSW directly (not IndexHNSWSQ) so ctor delegation goes straight
 * to the Index*-storage form.
 *
 * On disk, instances serialize with the same fourcc ("IHNs") and byte
 * layout as IndexHNSWSQ with an inner QT_1bit_direct ScalarQuantizer.
 * Readers materialize either IndexHNSWSQ (for non-binary SQ qtypes) or
 * IndexHNSWBinary (for QT_1bit_direct) depending on the inner storage's
 * qtype — see fork's impl/index_read.cpp for the dispatch.
 */
struct IndexHNSWBinary : IndexHNSW {
    IndexHNSWBinary();

    /// d is the number of binary dimensions (not bytes). metric must be
    /// supported by IndexBinaryScalarQuantizer: METRIC_Hamming, METRIC_Jaccard,
    /// METRIC_Substructure, METRIC_Superstructure.
    IndexHNSWBinary(int d, int M, MetricType metric = METRIC_Hamming);
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
