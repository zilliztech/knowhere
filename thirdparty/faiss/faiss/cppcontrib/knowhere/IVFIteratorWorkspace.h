/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <stdint.h>
#include <memory>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/MetricType.h>
#include <faiss/impl/DistanceComputer.h>

#include "knowhere/object.h"

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// Aliases of the baseline faiss SearchParametersIVF; the fork no longer
// defines its own variant.
using SearchParametersIVF = ::faiss::SearchParametersIVF;
using IVFSearchParameters = ::faiss::SearchParametersIVF;

// Path-D step 11.4b: forward decl `struct IndexIVF;` replaced by an
// alias to baseline. Fork's `struct IndexIVF` was deleted; the alias
// here matches the same alias defined in
// `<faiss/cppcontrib/knowhere/IndexIVF.h>` (allowed under C++ rules
// for equivalent `using` declarations across translation units).
using IndexIVF = ::faiss::IndexIVF;

/// Base workspace for IVF iterator lifecycle.
/// Holds query data, coarse quantization results, and iteration state.
struct IVFIteratorWorkspace {
    IVFIteratorWorkspace() = default;
    IVFIteratorWorkspace(
            const float* query_data,
            const size_t d,
            const IVFSearchParameters* search_params);
    virtual ~IVFIteratorWorkspace();

    /// Scan the next batch of inverted lists, populating this->dists.
    virtual void next_batch(size_t current_backup_count) = 0;

    std::vector<float> query_data; // a copy of a single query
    const IVFSearchParameters* search_params = nullptr;
    size_t nprobe = 0;
    size_t backup_count_threshold = 0;   // count * nprobe / nlist
    std::vector<::knowhere::DistId> dists; // should be cleared after each use
    size_t next_visit_coarse_list_idx = 0;
    std::unique_ptr<float[]> coarse_dis =
            nullptr; // backup coarse centroids distances (heap)
    std::unique_ptr<idx_t[]> coarse_idx =
            nullptr; // backup coarse centroids ids (heap)
    std::unique_ptr<size_t[]> coarse_list_sizes =
            nullptr; // snapshot of the list_size
    std::unique_ptr<DistanceComputer> dis_refine;
};

/// Standalone iterator workspace for base IVF index types
/// (IndexIVFFlat, IndexIVFFlatCC, IndexIVFScalarQuantizerCC, IndexIVFScalarQuantizer).
/// Contains the full init and scanning logic from IndexIVF.
struct IVFBaseIteratorWorkspace : IVFIteratorWorkspace {
    const IndexIVF* ivf_index;

    IVFBaseIteratorWorkspace(const IndexIVF* ivf_index,
                             const float* query_data,
                             const IVFSearchParameters* params);

    void next_batch(size_t current_backup_count) override;
};

}  // namespace knowhere
}  // namespace cppcontrib
}  // namespace faiss
