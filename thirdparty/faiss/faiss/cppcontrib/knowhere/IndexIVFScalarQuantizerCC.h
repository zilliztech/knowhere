#pragma once

#include <cstdio>
#include <fstream>
#include <optional>

#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexScalarQuantizer.h>
#include <faiss/cppcontrib/knowhere/invlists/ConcurrentDirectMap.h>
#include <faiss/cppcontrib/knowhere/utils/data_backup_file.h>

#include "knowhere/utils.h"

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/***************************************************
 *IndexIVFScalarQuantizerCC
 ***************************************************/
struct IndexIVFScalarQuantizerCC : IndexIVFScalarQuantizer {
    std::unique_ptr<DataBackFileHandler> raw_data_backup_ = nullptr;

    /// Path-D step 10.8: CC-specific standalone concurrent direct map.
    /// Populated in parallel with the inherited fork `direct_map`
    /// during add. SQ_CC currently has no external reader (the CC
    /// `reconstruct` path goes through `raw_data_backup_` instead of
    /// direct_map, and `calc_dist_by_ids` is not invoked for SQ_CC
    /// per the gate in src/index/ivf/ivf.cc); the field is
    /// populated here for symmetry with IVFFlatCC and so step 10.9
    /// can remove fork's `direct_map.ConcurrentArray` variant
    /// without leaving CC leaves without a thread-safe chunked store.
    ConcurrentDirectMap cc_direct_map;

    IndexIVFScalarQuantizerCC(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t ssize,
            ::faiss::ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = false,
            std::optional<std::string> raw_data_prefix_path = std::nullopt);

    IndexIVFScalarQuantizerCC();

    // Path-D step 10.6: `train` and `add_with_ids` overrides removed —
    // they previously just called up the inheritance chain
    // (IndexIVF::train, IndexIVFScalarQuantizer::add_with_ids) without
    // adding any behavior. Deleting them lets the inherited virtuals
    // dispatch directly with identical semantics.

    /// CC add path: delegate to the parent SQ encoder (which handles
    /// residual + SQ encoding + invlists add_entry + direct_map
    /// bookkeeping inside its own OMP loop) and then, if a raw-data
    /// sidecar is configured, append the original float vectors to it
    /// in insertion order.
    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    /// CC-leaf overrides that route through the cc_impl:: segment-aware
    /// helpers. Path-D step 10.6.
    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            const idx_t* keys,
            const float* coarse_dis,
            faiss::RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    /// Reconstruct original (un-quantized) float vector from the
    /// raw-data sidecar. SQ codes in the invlists are lossy; this path
    /// is how knowhere's GetVectorByIds contract is honored for
    /// IVF_SQ_CC indexes configured with raw_data_store_prefix.
    void reconstruct(idx_t key, float* recons) const override;

    bool with_raw_data();

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;
};

struct IndexIVFScalarQuantizerCCCosine : IndexIVFScalarQuantizerCC, HasInverseL2Norms {
    IndexIVFScalarQuantizerCCCosine(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t ssize,
            ::faiss::ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = false,
            std::optional<std::string> raw_data_prefix_path = std::nullopt);

    IndexIVFScalarQuantizerCCCosine();

    void train(idx_t n, const float* x) override;
    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
};

}
}
} // namespace faiss
