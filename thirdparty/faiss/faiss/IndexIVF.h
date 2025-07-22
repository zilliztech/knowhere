/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_IVF_H
#define FAISS_INDEX_IVF_H

#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/utils/Heap.h>

#include "knowhere/object.h"

namespace faiss {

/** Encapsulates a quantizer object for the IndexIVF
 *
 * The class isolates the fields that are independent of the storage
 * of the lists (especially training)
 */
struct Level1Quantizer {
    /// quantizer that maps vectors to inverted lists
    Index* quantizer = nullptr;

    /// number of inverted lists
    size_t nlist = 0;

    /**
     * = 0: use the quantizer as index in a kmeans training
     * = 1: just pass on the training set to the train() of the quantizer
     * = 2: kmeans training on a flat index + add the centroids to the quantizer
     */
    char quantizer_trains_alone = 0;
    bool own_fields = false; ///< whether object owns the quantizer

    ClusteringParameters cp; ///< to override default clustering params
    /// to override index used during clustering
    Index* clustering_index = nullptr;

    /// Trains the quantizer and calls train_residual to train sub-quantizers
    void train_q1(
            size_t n,
            const float* x,
            bool verbose,
            MetricType metric_type);

    /// compute the number of bytes required to store list ids
    size_t coarse_code_size() const;
    void encode_listno(idx_t list_no, uint8_t* code) const;
    idx_t decode_listno(const uint8_t* code) const;

    Level1Quantizer(Index* quantizer, size_t nlist);

    Level1Quantizer();

    ~Level1Quantizer();
};

struct SearchParametersIVF : SearchParameters {
    size_t nprobe = 1;    ///< number of probes at query time
    size_t max_codes = 0; ///< max nb of codes to visit to do a query
    ///< indicate whether we should early teriminate before topk results full when search reaches max_codes
    ///< to minimize code change, when users only use nprobe to search, this config does not take affect since we will first retrieve the nearest nprobe buckets
    ///< it is a bit heavy to further retrieve more buckets
    ///< therefore to make sure we get topk results, use nprobe=nlist and use max_codes to narrow down the search range
    size_t max_lists_num = 0; ///< select min{scanned number of (max_codes),
    ///< scanned number of (max_lists_num) to return.}
    bool ensure_topk_full = false;

    ///< during IVF range search, if reach 'max_empty_result_buckets' num of
    ///< continuous buckets with no valid results, terminate range search
    size_t max_empty_result_buckets = 0;

    SearchParameters* quantizer_params = nullptr;

    /// context object to pass to InvertedLists
    void* inverted_list_context = nullptr;

    virtual ~SearchParametersIVF() {}
};

// the new convention puts the index type after SearchParameters
using IVFSearchParameters = SearchParametersIVF;
struct DistanceComputer;
struct IVFIteratorWorkspace {
    IVFIteratorWorkspace() = default;
    IVFIteratorWorkspace(
            const float* query_data,
            const size_t d,
            const IVFSearchParameters* search_params);
    virtual ~IVFIteratorWorkspace();

    std::vector<float> query_data; // a copy of a single query
    const IVFSearchParameters* search_params = nullptr;
    size_t nprobe = 0;
    size_t backup_count_threshold = 0;   // count * nprobe / nlist
    std::vector<knowhere::DistId> dists; // should be cleared after each use
    size_t next_visit_coarse_list_idx = 0;
    std::unique_ptr<float[]> coarse_dis =
            nullptr; // backup coarse centroids distances (heap)
    std::unique_ptr<idx_t[]> coarse_idx =
            nullptr; // backup coarse centroids ids (heap)
    std::unique_ptr<size_t[]> coarse_list_sizes =
            nullptr; // snapshot of the list_size
    std::unique_ptr<DistanceComputer> dis_refine;
};

struct InvertedListScanner;
struct IndexIVFStats;
struct CodePacker;

struct IndexIVFInterface : Level1Quantizer {
    size_t nprobe = 1;    ///< number of probes at query time
    size_t max_codes = 0; ///< max nb of codes to visit to do a query

    explicit IndexIVFInterface(Index* quantizer = nullptr, size_t nlist = 0)
            : Level1Quantizer(quantizer, nlist) {}

    /** search a set of vectors, that are pre-quantized by the IVF
     *  quantizer. Fill in the corresponding heaps with the query
     *  results. The default implementation uses InvertedListScanners
     *  to do the search.
     *
     * @param n      nb of vectors to query
     * @param x      query vectors, size nx * d
     * @param assign coarse quantization indices, size nx * nprobe
     * @param centroid_dis
     *               distances to coarse centroids, size nx * nprobe
     * @param distance
     *               output distances, size n * k
     * @param labels output labels, size n * k
     * @param store_pairs store inv list index + inv list offset
     *                     instead in upper/lower 32 bit of result,
     *                     instead of ids (used for reranking).
     * @param params used to override the object's search parameters
     * @param stats  search stats to be updated (can be null)
     */
    virtual void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const = 0;

    /** Range search a set of vectors, that are pre-quantized by the IVF
     *  quantizer. Fill in the RangeSearchResults results. The default
     * implementation uses InvertedListScanners to do the search.
     *
     * @param n      nb of vectors to query
     * @param x      query vectors, size nx * d
     * @param assign coarse quantization indices, size nx * nprobe
     * @param centroid_dis
     *               distances to coarse centroids, size nx * nprobe
     * @param result Output results
     * @param store_pairs store inv list index + inv list offset
     *                     instead in upper/lower 32 bit of result,
     *                     instead of ids (used for reranking).
     * @param params used to override the object's search parameters
     * @param stats  search stats to be updated (can be null)
     */
    virtual void range_search_preassigned(
            idx_t nx,
            const float* x,
            float radius,
            const idx_t* keys,
            const float* coarse_dis,
            RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const = 0;

    virtual ~IndexIVFInterface() {}
};

/** Index based on a inverted file (IVF)
 *
 * In the inverted file, the quantizer (an Index instance) provides a
 * quantization index for each vector to be added. The quantization
 * index maps to a list (aka inverted list or posting list), where the
 * id of the vector is stored.
 *
 * The inverted list object is required only after trainng. If none is
 * set externally, an ArrayInvertedLists is used automatically.
 *
 * At search time, the vector to be searched is also quantized, and
 * only the list corresponding to the quantization index is
 * searched. This speeds up the search by making it
 * non-exhaustive. This can be relaxed using multi-probe search: a few
 * (nprobe) quantization indices are selected and several inverted
 * lists are visited.
 *
 * Sub-classes implement a post-filtering of the index that refines
 * the distance estimation from the query to databse vectors.
 */
struct IndexIVF : Index, IndexIVFInterface {
    /// Access to the actual data
    InvertedLists* invlists = nullptr;
    bool own_invlists = false;

    size_t code_size = 0; ///< code size per vector in bytes

    /** Parallel mode determines how queries are parallelized with OpenMP
     *
     * 0 (default): split over queries
     * 1: parallelize over inverted lists
     * 2: parallelize over both
     * 3: split over queries with a finer granularity
     *
     * PARALLEL_MODE_NO_HEAP_INIT: binary or with the previous to
     * prevent the heap to be initialized and finalized
     */
    int parallel_mode = 0;
    const int PARALLEL_MODE_NO_HEAP_INIT = 1024;

    /** optional map that maps back ids to invlist entries. This
     *  enables reconstruct() */
    DirectMap direct_map;

    /// do the codes in the invlists encode the vectors relative to the
    /// centroids?
    bool by_residual = true;

    /** The Inverted file takes a quantizer (an Index) on input,
     * which implements the function mapping a vector to a list
     * identifier.
     */
    IndexIVF(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t code_size,
            MetricType metric = METRIC_L2);

    virtual std::unique_ptr<IVFIteratorWorkspace> getIteratorWorkspace(
            const float* query_data,
            const IVFSearchParameters* ivfsearchParams) const;

    // Unlike regular knn-search, the iterator does not know the size `k` of the
    // returned result.
    //   The iterator will maintain a heap of at least (nprobe/nlist) nodes for
    //   iterator `Next()` operation.
    //   When there are not enough nodes in the heap, iterator will scan the
    //   next coarse list.
    virtual void getIteratorNextBatch(
            IVFIteratorWorkspace* workspace,
            size_t current_backup_count) const;

    void reset() override;

    /// Trains the quantizer and calls train_encoder to train sub-quantizers
    void train(idx_t n, const float* x) override;

    /// Calls add_with_ids with NULL ids
    void add(idx_t n, const float* x) override;

    /// default implementation that calls encode_vectors
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    /** Implementation of vector addition where the vector assignments are
     * predefined. The default implementation hands over the code extraction to
     * encode_vectors.
     *
     * @param precomputed_idx    quantization indices for the input vectors
     * (size n)
     */
    virtual void add_core(
            idx_t n,
            const float* x,
            const float* x_norms,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr);

    /** Encodes a set of vectors as they would appear in the inverted lists
     *
     * @param list_nos   inverted list ids as returned by the
     *                   quantizer (size n). -1s are ignored.
     * @param codes      output codes, size n * code_size
     * @param include_listno
     *                   include the list ids in the code (in this case add
     *                   ceil(log8(nlist)) to the code size)
     */
    virtual void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listno = false) const = 0;

    /** Add vectors that are computed with the standalone codec
     *
     * @param codes  codes to add size n * sa_code_size()
     * @param xids   corresponding ids, size n
     */
    void add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids);

    /** Train the encoder for the vectors.
     *
     * If by_residual then it is called with residuals and corresponding assign
     * array, otherwise x is the raw training vectors and assign=nullptr */
    virtual void train_encoder(idx_t n, const float* x, const idx_t* assign);

    /// can be redefined by subclasses to indicate how many training vectors
    /// they need
    virtual idx_t train_encoder_num_vectors() const;

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
            RangeSearchResult* result,
            bool store_pairs = false,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    /** assign the vectors, then call search_preassign */
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    /** Get a scanner for this index (store_pairs means ignore labels)
     *
     * The default search implementation uses this to compute the distances.
     * Use sel instead of params->sel, because sel is initialized with
     * params->sel, but may get overriden by IndexIVF's internal logic.
     */
    virtual InvertedListScanner* get_InvertedListScanner(
            bool store_pairs = false,
            const IDSelector* sel = nullptr,
            const IVFSearchParameters* params = nullptr) const;

    /** reconstruct a vector. Works only if maintain_direct_map is set to 1 or 2
     */
    void reconstruct(idx_t key, float* recons) const override;

    /** Update a subset of vectors.
     *
     * The index must have a direct_map
     *
     * @param nv     nb of vectors to update
     * @param idx    vector indices to update, size nv
     * @param v      vectors of new values, size nv*d
     */
    virtual void update_vectors(int nv, const idx_t* idx, const float* v);

    /** Reconstruct a subset of the indexed vectors.
     *
     * Overrides default implementation to bypass reconstruct() which requires
     * direct_map to be maintained.
     *
     * @param i0     first vector to reconstruct
     * @param ni     nb of vectors to reconstruct
     * @param recons output array of reconstructed vectors, size ni * d
     */
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    /** Similar to search, but also reconstructs the stored vectors (or an
     * approximation in the case of lossy coding) for the search results.
     *
     * Overrides default implementation to avoid having to maintain direct_map
     * and instead fetch the code offsets through the `store_pairs` flag in
     * search_preassigned().
     *
     * @param recons      reconstructed vectors size (n, k, d)
     */
    void search_and_reconstruct(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* recons,
            const SearchParameters* params = nullptr) const override;

    /** Similar to search, but also returns the codes corresponding to the
     * stored vectors for the search results.
     *
     * @param codes      codes (n, k, code_size)
     * @param include_listno
     *                   include the list ids in the code (in this case add
     *                   ceil(log8(nlist)) to the code size)
     */
    void search_and_return_codes(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            uint8_t* recons,
            bool include_listno = false,
            const SearchParameters* params = nullptr) const;

    /** Reconstruct a vector given the location in terms of (inv list index +
     * inv list offset) instead of the id.
     *
     * Useful for reconstructing when the direct_map is not maintained and
     * the inv list offset is computed by search_preassigned() with
     * `store_pairs` set.
     */
    virtual void reconstruct_from_offset(
            int64_t list_no,
            int64_t offset,
            float* recons) const;

    /// Dataset manipulation functions

    size_t remove_ids(const IDSelector& sel) override;

    void check_compatible_for_merge(const Index& otherIndex) const override;

    virtual void merge_from(Index& otherIndex, idx_t add_id) override;

    // returns a new instance of a CodePacker
    virtual CodePacker* get_CodePacker() const;

    /** copy a subset of the entries index to the other index
     * see Invlists::copy_subset_to for the meaning of subset_type
     */
    virtual void copy_subset_to(
            IndexIVF& other,
            InvertedLists::subset_type_t subset_type,
            idx_t a1,
            idx_t a2) const;

    virtual void to_readonly();

    virtual bool is_readonly() const;

    ~IndexIVF() override;

    size_t get_list_size(size_t list_no) const {
        return invlists->list_size(list_no);
    }

    /// are the ids sorted?
    bool check_ids_sorted() const;

    /** initialize a direct map
     *
     * @param new_maintain_direct_map    if true, create a direct map,
     *                                   else clear it
     */
    void make_direct_map(bool new_maintain_direct_map = true, DirectMap::Type type = DirectMap::Type::Array);

    void set_direct_map_type(DirectMap::Type type);

    /// replace the inverted lists, old one is deallocated if own_invlists
    void replace_invlists(InvertedLists* il, bool own = false);

    /* The standalone codec interface (except sa_decode that is specific) */
    size_t sa_code_size() const override;

    /** encode a set of vectors
     * sa_encode will call encode_vector with include_listno=true
     * @param n      nb of vectors to encode
     * @param x      the vectors to encode
     * @param bytes  output array for the codes
     * @return nb of bytes written to codes
     */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void dump();

    IndexIVF();
};

struct RangeQueryResult;

/** Object that handles a query. The inverted lists to scan are
 * provided externally. The object has a lot of state, but
 * distance_to_code and scan_codes can be called in multiple
 * threads */
struct InvertedListScanner {
    idx_t list_no = -1;    ///< remember current list
    bool keep_max = false; ///< keep maximum instead of minimum
    /// store positions in invlists rather than labels
    bool store_pairs;

    /// search in this subset of ids
    const IDSelector* sel;

    InvertedListScanner(
            bool store_pairs = false,
            const IDSelector* sel = nullptr)
            : store_pairs(store_pairs), sel(sel) {}

    /// used in default implementation of scan_codes
    size_t code_size = 0;

    /// from now on we handle this query.
    virtual void set_query(const float* query_vector) = 0;

    /// following codes come from this inverted list
    virtual void set_list(idx_t list_no, float coarse_dis) = 0;

    /// compute a single query-to-code distance
    virtual float distance_to_code(const uint8_t* code) const = 0;

    /** scan a set of codes, compute distances to current query and
     * update heap of results if necessary. Default implemetation
     * calls distance_to_code.
     *
     * @param n      number of codes to scan
     * @param codes  codes to scan (n * code_size)
     * @param ids        corresponding ids (ignored if store_pairs)
     * @param distances  heap distances (size k)
     * @param labels     heap labels (size k)
     * @param k          heap size
     * @param scan_cnt   valid number of codes be scanned
     * @return number of heap updates performed
     */
    virtual size_t scan_codes(
            size_t n,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k,
            size_t& scan_cnt) const;

    /** scan a set of codes, compute distances to current query and
     * return in a vector.
     *
     * @param list_size     number of codes to scan
     * @param codes         codes to scan (list_size * code_size)
     * @param code_norms    norms of code (for cosine)
     * @param ids           corresponding ids (ignored if store_pairs)
     * @param out           output distances and ids
     */
    virtual void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            std::vector<knowhere::DistId>& out) const;

    // same as scan_codes, using an iterator
    virtual size_t iterate_codes(
            InvertedListsIterator* iterator,
            float* distances,
            idx_t* labels,
            size_t k,
            size_t& list_size) const;

    /** scan a set of codes, compute distances to current query and
     * update results if distances are below radius
     *
     * (default implementation fails) */
    virtual void scan_codes_range(
            size_t n,
            const uint8_t* codes,
            const float* code_norms,
            const idx_t* ids,
            float radius,
            RangeQueryResult& result) const;

    // same as scan_codes_range, using an iterator
    virtual void iterate_codes_range(
            InvertedListsIterator* iterator,
            float radius,
            RangeQueryResult& result,
            size_t& list_size) const;

    virtual ~InvertedListScanner() {}
};

// whether to check that coarse quantizers are the same
FAISS_API extern bool check_compatible_for_merge_expensive_check;

struct IndexIVFStats {
    size_t nq;                // nb of queries run
    size_t nlist;             // nb of inverted lists scanned
    size_t ndis;              // nb of distances computed
    size_t nheap_updates;     // nb of times the heap was updated
    double quantization_time; // time spent quantizing vectors (in ms)
    double search_time;       // time spent searching lists (in ms)

    IndexIVFStats() {
        reset();
    }
    void reset();
    void add(const IndexIVFStats& other);
};

// global var that collects them all
FAISS_API extern IndexIVFStats indexIVF_stats;

} // namespace faiss

#endif
