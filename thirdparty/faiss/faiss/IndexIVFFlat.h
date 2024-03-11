/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_FLAT_H
#define FAISS_INDEX_IVF_FLAT_H

#include <stdint.h>
#include <optional>
#include <unordered_map>

#include <faiss/IndexIVF.h>

namespace faiss {
struct IVFFlatIteratorWorkspace {
    IVFFlatIteratorWorkspace(
            const float* query_data,
            const IVFSearchParameters* search_params)
            : query_data(query_data), search_params(search_params) {}

    const float* query_data = nullptr; // single query
    const IVFSearchParameters* search_params = nullptr;
    bool initial_search_done = false;
    std::unique_ptr<float[]> distances = nullptr; // backup distances (heap)
    std::unique_ptr<idx_t[]> labels = nullptr;    // backup ids (heap)
    size_t backup_count = 0;            // scan a new coarse-list when less than backup_count_threshold
    size_t max_backup_count = 0;        
    size_t backup_count_threshold = 0;  // count * nprobe / nlist
    size_t next_visit_coarse_list_idx = 0;
    std::unique_ptr<float[]> coarse_dis = nullptr;   // backup coarse centroids distances (heap)
    std::unique_ptr<idx_t[]> coarse_idx = nullptr;   // backup coarse centroids ids (heap)
    std::unique_ptr<size_t[]> coarse_list_sizes = nullptr;  // snapshot of the list_size
    size_t max_coarse_list_size = 0;
};

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct IndexIVFFlat : IndexIVF {
    IndexIVFFlat(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2,
            bool is_cosine = false);

    void restore_codes(const uint8_t* raw_data, const size_t raw_size);

    // Be careful with overriding this function, because
    //   renormalized x may be used inside.
    // Overridden by IndexIVFFlatDedup.
    void train(idx_t n, const float* x) override;

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void add_core(
            idx_t n,
            const float* x,
            const float* x_norms,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    IndexIVFFlat();

    std::unique_ptr<IVFFlatIteratorWorkspace> getIteratorWorkspace(
            const float* query_data,
            const IVFSearchParameters* ivfsearchParams) const;

    // Unlike regular knn-search, the iterator does not know the size `k` of the
    // returned result.
    //   The workspace will maintain a heap of at least (nprobe/nlist) nodes for
    //   iterator `Next()` operation.
    //   When there are not enough nodes in the heap, iterator will scan the
    //   next coarse list.
    std::optional<std::pair<float, idx_t>> getIteratorNext(
            IVFFlatIteratorWorkspace* workspace) const;
};

struct IndexIVFFlatCC : IndexIVFFlat {
    IndexIVFFlatCC(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t ssize,
            MetricType = METRIC_L2,
            bool is_cosine = false);

    IndexIVFFlatCC();
};

struct IndexIVFFlatDedup : IndexIVFFlat {
    /** Maps ids stored in the index to the ids of vectors that are
     *  the same. When a vector is unique, it does not appear in the
     *  instances map */
    std::unordered_multimap<idx_t, idx_t> instances;

    IndexIVFFlatDedup(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2);

    /// also dedups the training set
    void train(idx_t n, const float* x) override;

    /// implemented for all IndexIVF* classes
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

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

    size_t remove_ids(const IDSelector& sel) override;

    /// not implemented
    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    /// not implemented
    void update_vectors(int nv, const idx_t* idx, const float* v) override;

    /// not implemented
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    IndexIVFFlatDedup() {}
};

} // namespace faiss

#endif
