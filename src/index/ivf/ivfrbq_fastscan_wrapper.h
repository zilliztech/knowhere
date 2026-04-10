// Copyright (C) 2019-2025 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <cstddef>
#include <memory>

#include "faiss/Index.h"
#include "index/ivf/ivf_config.h"
#include "knowhere/expected.h"

// Forward declaration — full header included only in .cc to avoid
// simd_result_handlers.h conflict with knowhere's patched copy.
namespace faiss {
struct IndexIVFRaBitQFastScan;
struct IndexRefine;
}  // namespace faiss

namespace knowhere {

/// Wrapper for the core Faiss IndexIVFRaBitQFastScan backend.
///
/// This mirrors the legacy IVFRaBitQ wrapper structure: ivf.cc should not need
/// to know whether the actual Faiss object is wrapped in an IndexPreTransform
/// and optionally an IndexRefineFlat.
///
/// Inner index is one of:
///   - IndexPreTransform(RR, IndexIVFRaBitQFastScan)
///   - IndexRefineFlat(IndexPreTransform(RR, IndexIVFRaBitQFastScan))
///
/// All types are core Faiss, so serialization/deserialization stays on the
/// core faiss::write_index/read_index path.
struct IndexIVFRaBitQFastScanWrapper : faiss::Index {
    std::unique_ptr<faiss::Index> index;

    explicit IndexIVFRaBitQFastScanWrapper(std::unique_ptr<faiss::Index>&& index_in);

    static expected<std::unique_ptr<IndexIVFRaBitQFastScanWrapper>>
    create(faiss::idx_t d, size_t nlist, const IvfRaBitQFastScanConfig& cfg, faiss::MetricType metric);

    /// Deserialization. Returns nullptr if the index type doesn't match.
    static std::unique_ptr<IndexIVFRaBitQFastScanWrapper>
    from_deserialized(std::unique_ptr<faiss::Index>&& index_in);

    void
    train(faiss::idx_t n, const float* x) override;
    void
    add(faiss::idx_t n, const float* x) override;
    void
    search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
           const faiss::SearchParameters* params) const override;
    void
    range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                 const faiss::SearchParameters* params) const override;
    void
    reset() override;
    void
    merge_from(faiss::Index& otherIndex, faiss::idx_t add_id) override;
    faiss::DistanceComputer*
    get_distance_computer() const override;

    /// Accessor for the underlying FastScan index through the optional
    /// refine/pretransform wrappers.
    faiss::IndexIVFRaBitQFastScan*
    get_fastscan_index();
    const faiss::IndexIVFRaBitQFastScan*
    get_fastscan_index() const;

    bool
    has_refine() const {
        return has_refine_;
    }
    size_t
    get_nlist() const;
    size_t
    size() const;

 private:
    bool has_refine_ = false;
    void
    detect_refine_state();
};

}  // namespace knowhere
