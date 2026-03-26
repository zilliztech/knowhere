/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* All distance functions for L2 and IP distances.
 * The actual functions are implemented in distances.cpp and distances_simd.cpp
 */

#pragma once

#include <stdint.h>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>

#include "knowhere/object.h"

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/***************************************************************************
 * Compute a subset of  distances
 ***************************************************************************/

/** compute the inner product between x and a subset y of ny vectors defined by
 * ids. NOTE: fork-specific — does not assign -INFINITY for filtered entries.
 */
void fvec_inner_products_by_idx(
        float* ip,
        const float* x,
        const float* y,
        const int64_t* ids,
        size_t d,
        size_t nx,
        size_t ny);

/** compute the squared L2 distances between x and a subset y of ny vectors
 * defined by ids. NOTE: fork-specific — does not assign INFINITY for filtered entries.
 */
void fvec_L2sqr_by_idx(
        float* dis,
        const float* x,
        const float* y,
        const int64_t* ids, /* ids of y vecs */
        size_t d,
        size_t nx,
        size_t ny);

void exhaustive_L2sqr_nearest_imp(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        size_t nx,
        size_t ny,
        float* vals,
        int64_t* ids);

/***************************************************************************
 * KNN functions
 ***************************************************************************/

// threshold on nx above which we switch to BLAS to compute distances
FAISS_API extern int distance_compute_blas_threshold;

// block sizes for BLAS distance computations
FAISS_API extern int distance_compute_blas_query_bs;
FAISS_API extern int distance_compute_blas_database_bs;

// above this number of results we switch to a reservoir to collect results
// rather than a heap
FAISS_API extern int distance_compute_min_k_reservoir;

/** Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, w.r.t to max inner product.
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param res  result heap structure, which also provides k. Sorted on output
 */
void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* res,
        const IDSelector* sel = nullptr);

/**  Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, for the inner product metric.
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param distances  output distances, size nq * k
 * @param indexes    output vector ids, size nq * k
 */
void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const IDSelector* sel = nullptr);

/// Convert faiss pair<id, distance> array to knowhere DistId vector.
/// Entries with id < 0 (filtered-out sentinels) are skipped, preserving
/// the caller's pre-initialized values in output.
void pairs_to_distids(
        const std::pair<int64_t, float>* pairs,
        std::vector<::knowhere::DistId>& output,
        size_t n);

void all_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<::knowhere::DistId>& output,
        const IDSelector* sel);

void all_inner_product_distances(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float* output,
        const IDSelector* sel);

/** Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, for the L2 distance
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param res  result heap strcture, which also provides k. Sorted on output
 * @param y_norm2    (optional) norms for the y vectors (nullptr or size ny)
 * @param sel  search in this subset of vectors
 */
void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res,
        const float* y_norm2 = nullptr,
        const IDSelector* sel = nullptr);

/**  Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, for the L2 distance
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size ny * d
 * @param distances  output distances, size nq * k
 * @param indexes    output vector ids, size nq * k
 * @param y_norm2    (optional) norms for the y vectors (nullptr or size ny)
 * @param sel  search in this subset of vectors
 */
void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const float* y_norm2 = nullptr,
        const IDSelector* sel = nullptr);

void all_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<::knowhere::DistId>& output,
        const float* y_norms,
        const IDSelector* sel);

void all_L2sqr_distances(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float* output,
        const float* y_norms,
        const IDSelector* sel);

// Knowhere-specific function
// y_inv_norms: precomputed inverse L2 norms (1/||y||) per database vector,
//   or nullptr to compute on-the-fly.
void knn_cosine(
        const float* x,
        const float* y,
        const float* y_inv_norms,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* ha,
        const IDSelector* sel = nullptr);

void knn_cosine(
        const float* x,
        const float* y,
        const float* y_inv_norms,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const IDSelector* sel = nullptr);

void all_cosine(
        const float* x,
        const float* y,
        const float* y_inv_norms,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<::knowhere::DistId>& output,
        const IDSelector* sel);

void all_cosine_distances(
        const float* x,
        const float* y,
        const float* y_inv_norms,
        size_t d,
        size_t nx,
        size_t ny,
        float* output,
        const IDSelector* sel);

// Knowhere-specific function
void knn_jaccard(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res,
        const IDSelector* sel = nullptr);

/** Find the max inner product neighbors for nx queries in a set of ny vectors
 * indexed by ids. May be useful for re-ranking a pre-selected vector list
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size (max(ids) + 1) * d
 * @param ids  subset of database vectors to consider, size (nx, nsubset)
 * @param res  result structure
 * @param ld_ids stride for the ids array. -1: use nsubset, 0: all queries
 * process the same subset
 */
void knn_inner_products_by_idx(
        const float* x,
        const float* y,
        const int64_t* subset,
        size_t d,
        size_t nx,
        size_t ny,
        size_t nsubset,
        size_t k,
        float* vals,
        int64_t* ids,
        int64_t ld_ids = -1);

/** Find the nearest neighbors for nx queries in a set of ny vectors
 * indexed by ids. May be useful for re-ranking a pre-selected vector list
 *
 * @param x    query vectors, size nx * d
 * @param y    database vectors, size (max(ids) + 1) * d
 * @param subset subset of database vectors to consider, size (nx, nsubset)
 * @param res  rIDesult structure
 * @param ld_subset stride for the subset array. -1: use nsubset, 0: all queries
 * process the same subset
 */
void knn_L2sqr_by_idx(
        const float* x,
        const float* y,
        const int64_t* subset,
        size_t d,
        size_t nx,
        size_t ny,
        size_t nsubset,
        size_t k,
        float* vals,
        int64_t* ids,
        int64_t ld_subset = -1);

void knn_cosine_by_idx(
        const float* x,
        const float* y,
        const float* y_inv_norms,
        const int64_t* subset,
        size_t d,
        size_t nx,
        size_t ny,
        size_t nsubset,
        size_t k,
        float* vals,
        int64_t* ids,
        int64_t ld_ids = -1);

/***************************************************************************
 * Range search
 ***************************************************************************/

/** Return the k nearest neighors of each of the nx vectors x among the ny
 *  vector y, w.r.t to max inner product
 *
 * @param x      query vectors, size nx * d
 * @param y      database vectors, size ny * d
 * @param radius search radius around the x vectors
 * @param result result structure
 */
void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel = nullptr);

/// same as range_search_L2sqr for the inner product similarity
void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel = nullptr);

// Knowhere-specific function
void range_search_cosine(
        const float* x,
        const float* y,
        const float* y_inv_norms,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel = nullptr);

/***************************************************************************
 * elkan
 ***************************************************************************/

/** Return the nearest neighbors of each of the nx vectors x among the ny
 *
 * @param x          query vectors, size nx * d
 * @param y          database vectors, size ny * d
 * @param ids        result array ids
 * @param val        result array value
 * @param tmp_buffer tmporary memory for symmetric matrix data
 * @param sym_dim    dimension of symmetric matrix
 */
void elkan_L2_sse(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        int64_t* ids,
        float* val,
        float* tmp_buffer,
        size_t sym_dim);

/***************************************************************************
 * Templatized versions of distance functions
 ***************************************************************************/

}
}
} // namespace faiss
