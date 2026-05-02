/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/cppcontrib/knowhere/IVFIteratorWorkspace.h>

namespace faiss {
struct IndexIVFFastScan;

namespace cppcontrib {
namespace knowhere {

struct IndexScaNN;

/// Iterator workspace for FastScan index types.
/// Uses PImpl to hide AVX2-dependent members (AlignedTable, SIMD handlers)
/// and avoid ODR/ABI mismatch between baseline and AVX2 translation units.
struct IVFFastScanIteratorWorkspace : IVFIteratorWorkspace {
    IVFFastScanIteratorWorkspace(
            const ::faiss::IndexIVFFastScan* index,
            const float* query_data,
            const IVFSearchParameters* params);
    ~IVFFastScanIteratorWorkspace() override;

    void next_batch(size_t current_backup_count) override;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Standalone iterator workspace for IndexScaNN.
/// Wraps an IVFFastScanIteratorWorkspace with optional refinement.
struct ScaNNIteratorWorkspace : IVFIteratorWorkspace {
    /// The inner FastScan workspace.
    std::unique_ptr<IVFIteratorWorkspace> inner;

    ScaNNIteratorWorkspace(const IndexScaNN* scann_index,
                           const float* query_data,
                           const IVFSearchParameters* params);

    void next_batch(size_t current_backup_count) override;
};

}  // namespace knowhere
}  // namespace cppcontrib
}  // namespace faiss
