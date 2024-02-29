/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/distances.h>
#include <faiss/utils/distances_if.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <omp.h>

#include "knowhere/bitsetview_idselector.h"

#include <faiss/FaissHook.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/utils.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

/***************************************************************************
 * Matrix/vector ops
 ***************************************************************************/

/* Compute the L2 norm of a set of nx vectors */
void fvec_norms_L2(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        nr[i] = sqrtf(fvec_norm_L2sqr(x + i * d, d));
    }
}

void fvec_norms_L2sqr(
        float* __restrict nr,
        const float* __restrict x,
        size_t d,
        size_t nx) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++)
        nr[i] = fvec_norm_L2sqr(x + i * d, d);
}

void fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        float* __restrict xi = x + i * d;

        float nr = fvec_norm_L2sqr(xi, d);

        if (nr > 0) {
            size_t j;
            const float inv_nr = 1.0 / sqrtf(nr);
            for (j = 0; j < d; j++)
                xi[j] *= inv_nr;
        }
    }
}

/***************************************************************************
 * KNN functions
 ***************************************************************************/

namespace {

// Helpers are used in search functions to help specialize various
// performance-related use cases, such as adding some extra
// support for a particular kind of IDSelector classes. This
// may be useful if the lion's share of samples are filtered out.

struct IDSelectorAll {
    inline bool is_member(const size_t idx) const {
        return true;
    }
};

struct IDSelectorHelper {
    const IDSelector* sel;

    inline bool is_member(const size_t idx) const {
        return sel->is_member(idx);
    }
};

struct BitsetViewSelectorHelper {
    // todo aguzhva: use avx gather instruction
    const knowhere::BitsetView bitset;

    inline bool is_member(const size_t idx) const {
        return !bitset.test(idx);
    }
};

/* Find the nearest neighbors for nx queries in a set of ny vectors */

/*
/// Baseline implementation of exhaustive_inner_product_seq
template <class ResultHandler>
void exhaustive_inner_product_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* sel) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;
            resi.begin(i);
            for (size_t j = 0; j < ny; j++) {
                // todo aguzhva: bitset was here
                //if (bitset.empty() || !bitset.test(j)) {
                if (!sel || sel->is_member(j)) {
                    float ip = fvec_inner_product(x_i, y_j, d);
                    resi.add_result(ip, j);
                }
                y_j += d;
            }
            resi.end();
        }
    }
}
*/

// An improved implementation that
// 1. helps the branch predictor,
// 2. computes distances for 4 elements per loop
template <class ResultHandler, class SelectorHelper>
void exhaustive_inner_product_seq(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const SelectorHelper selector) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            resi.begin(i);

            // the lambda that filters acceptable elements.
            auto filter = [&selector](const size_t j) {
                return selector.is_member(j);
            };

            // the lambda that applies a filtered element.
            auto apply = [&resi](const float ip, const idx_t j) {
                resi.add_result(ip, j);
            };

            // compute distances
            fvec_inner_products_ny_if(x_i, y, d, ny, filter, apply);

            resi.end();
        }
    }
}


template <class ResultHandler>
void exhaustive_inner_product_seq(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* __restrict sel) {
    // add different specialized cases here via introducing
    //   helpers which are converted into templates.

    // bitset.empty() translates into sel=nullptr

    if (const auto* bitsetview_sel = dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
        // A specialized case for Knowhere
        auto bitset = bitsetview_sel->bitset_view;
        if (!bitset.empty()) {
            BitsetViewSelectorHelper bitset_helper{bitset};
            exhaustive_inner_product_seq<ResultHandler, BitsetViewSelectorHelper>(
                x, y, d, nx, ny, res, bitset_helper);
            return;
        }
    }
    else if (sel != nullptr) {
        // default Faiss case if sel is defined
        IDSelectorHelper ids_helper{sel};
        exhaustive_inner_product_seq<ResultHandler, IDSelectorHelper>(
            x, y, d, nx, ny, res, ids_helper);
        return;
    }

    // default case if no filter is needed or if it is empty
    IDSelectorAll helper;
    exhaustive_inner_product_seq<ResultHandler, IDSelectorAll>(
        x, y, d, nx, ny, res, helper);
}

/*
// Baseline implementation of exhaustive_L2sqr_seq
template <class ResultHandler>
void exhaustive_L2sqr_seq(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* sel) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;
            resi.begin(i);
            for (size_t j = 0; j < ny; j++) {
                // todo aguzhva: bitset was here
                //if (bitset.empty() || !bitset.test(j)) {
                if (!sel || sel->is_member(j)) {
                    float disij = fvec_L2sqr(x_i, y_j, d);
                    resi.add_result(disij, j);
                }
                y_j += d;
            }
            resi.end();
        }
    }
}
*/

// An improved implementation that
// 1. helps the branch predictor,
// 2. computes distances for 4 elements per loop
template <class ResultHandler, class SelectorHelper>
void exhaustive_L2sqr_seq(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const SelectorHelper selector) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            resi.begin(i);

            // the lambda that filters acceptable elements.
            auto filter = [&selector](const size_t j) {
                return selector.is_member(j);
            };

            // the lambda that applies a filtered element.
            auto apply = [&resi](const float dis, const idx_t j) {
                resi.add_result(dis, j);
            };

            // compute distances
            fvec_L2sqr_ny_if(x_i, y, d, ny, filter, apply);

            resi.end();
        }
    }
}

template <class ResultHandler>
void exhaustive_L2sqr_seq(
        const float* __restrict x,
        const float* __restrict y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* __restrict sel) {
    // add different specialized cases here via introducing
    //   helpers which are converted into templates.

    // bitset.empty() translates into sel=nullptr

    if (const auto* bitsetview_sel = dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
        // A specialized case for Knowhere
        auto bitset = bitsetview_sel->bitset_view;
        if (!bitset.empty()) {
            BitsetViewSelectorHelper bitset_helper{bitset};
            exhaustive_L2sqr_seq<ResultHandler, BitsetViewSelectorHelper>(
                x, y, d, nx, ny, res, bitset_helper);
            return;
        }
    }
    else if (sel != nullptr) {
        // default Faiss case if sel is defined
        IDSelectorHelper ids_helper{sel};
        exhaustive_L2sqr_seq<ResultHandler, IDSelectorHelper>(
            x, y, d, nx, ny, res, ids_helper);
        return;
    }

    // default case if no filter is needed or if it is empty
    IDSelectorAll helper;
    exhaustive_L2sqr_seq<ResultHandler, IDSelectorAll>(
        x, y, d, nx, ny, res, helper);
}

template <class ResultHandler>
void exhaustive_cosine_seq(
        const float* x,
        const float* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* sel) {
    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    int nt = std::min(int(nx), omp_get_max_threads());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (int64_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;
            const float* y_j = y;
            resi.begin(i);
            for (size_t j = 0; j < ny; j++) {
                if (!sel || sel->is_member(j)) {
                    // todo aguzhva: what if a norm == 0 ?
                    float norm =
                        (y_norms != nullptr) ? y_norms[j]
                                             : sqrtf(fvec_norm_L2sqr(y_j, d));
                    float disij = fvec_inner_product(x_i, y_j, d) / norm;
                    resi.add_result(disij, j);
                }
                y_j += d;
            }
            resi.end();
        }
    }
}

/** Find the nearest neighbors for nx queries in a set of ny vectors */
template <class ResultHandler>
void exhaustive_inner_product_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* sel) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }

            res.add_results(j0, j1, ip_block.get(), sel);
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

// distance correction is an operator that can be applied to transform
// the distances
template <class ResultHandler>
void exhaustive_L2sqr_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const float* y_norms = nullptr,
        const IDSelector* sel = nullptr) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
#pragma omp parallel for
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[j] - 2 * ip;

                    // negative values can occur for identical vectors
                    // due to roundoff errors
                    if (dis < 0)
                        dis = 0;

                    *ip_line = dis;
                    ip_line++;
                }
            }
            res.add_results(j0, j1, ip_block.get(), sel);
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

template <class ResultHandler>
void exhaustive_cosine_blas(
        const float* x,
        const float* y,
        const float* y_norms_in,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const IDSelector* sel = nullptr) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> y_norms(new float[ny]);
    std::unique_ptr<float[]> del2;

    if (y_norms_in == nullptr) {
        fvec_norms_L2(y_norms.get(), y, d, ny);
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
#pragma omp parallel for
            for (int64_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line;
                    float dis = (y_norms_in != nullptr) ? ip / y_norms_in[j]
                                                        : ip / y_norms[j];
                    *ip_line = dis;
                    ip_line++;
                }
            }
            res.add_results(j0, j1, ip_block.get(), sel);
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

template <class DistanceCorrection, class ResultHandler>
static void knn_jaccard_blas(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        ResultHandler& res,
        const DistanceCorrection& corr,
        const IDSelector* sel) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    /* block sizes */
    const size_t bs_x = 4096, bs_y = 1024;
    // const size_t bs_x = 16, bs_y = 16;
    float* ip_block = new float[bs_x * bs_y];
    float* x_norms = new float[nx];
    float* y_norms = new float[ny];
    ScopeDeleter<float> del1(ip_block), del3(x_norms), del2(y_norms);

    fvec_norms_L2sqr(x_norms, x, d, nx);
    fvec_norms_L2sqr(y_norms, y, d, ny);

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block,
                       &nyi);
            }

            /* collect minima */
#pragma omp parallel for
            for (size_t i = i0; i < i1; i++) {
                float* ip_line = ip_block + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    if (!sel || sel->is_member(j)) {
                        float ip = *ip_line;
                        float dis = 1.0 - ip / (x_norms[i] + y_norms[j] - ip);

                        // negative values can occur for identical vectors
                        // due to roundoff errors
                        if (dis < 0)
                            dis = 0;

                        dis = corr(dis, i, j);
                        *ip_line = dis;
                    }
                    ip_line++;
                }
            }
            res.add_results(j0, j1, ip_block);
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

} // anonymous namespace

/*******************************************************
 * KNN driver functions
 *******************************************************/
int distance_compute_blas_threshold = 16384;
int distance_compute_blas_query_bs = 4096;
int distance_compute_blas_database_bs = 1024;
int distance_compute_min_k_reservoir = 100;

void knn_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* ha,
        const IDSelector* sel) {
    if (ha->k < distance_compute_min_k_reservoir) {
        HeapResultHandler<CMin<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        if (nx < distance_compute_blas_threshold) {
            exhaustive_inner_product_seq(x, y, d, nx, ny, res, sel);
        } else {
            exhaustive_inner_product_blas(x, y, d, nx, ny, res, sel);
        }
    } else {
        ReservoirResultHandler<CMin<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        if (nx < distance_compute_blas_threshold) {
            exhaustive_inner_product_seq(x, y, d, nx, ny, res, sel);
        } else {
            exhaustive_inner_product_blas(x, y, d, nx, ny, res, sel);
        }
    }
}

// computes and stores all IP distances into output. Output should be
// preallocated of size nx * ny, each element should be initialized to
// {lowest distance, -1}.
void all_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<std::pair<float, int64_t>>& output,
        const IDSelector* sel) {
    CollectAllResultHandler<CMax<float, int64_t>> res(ny, output);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_inner_product_seq(x, y, d, nx, ny, res, sel);
    } else {
        exhaustive_inner_product_blas(x, y, d, nx, ny, res, sel);
    }
}

void knn_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* ha,
        const float* y_norm2,
        const IDSelector* sel) {
    if (ha->k < distance_compute_min_k_reservoir) {
        HeapResultHandler<CMax<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        if (nx < distance_compute_blas_threshold) {
            exhaustive_L2sqr_seq(x, y, d, nx, ny, res, sel);
        } else {
            exhaustive_L2sqr_blas(x, y, d, nx, ny, res, y_norm2, sel);
        }
    } else {
        ReservoirResultHandler<CMax<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        if (nx < distance_compute_blas_threshold) {
            exhaustive_L2sqr_seq(x, y, d, nx, ny, res, sel);
        } else {
            exhaustive_L2sqr_blas(x, y, d, nx, ny, res, y_norm2, sel);
        }
    }
}

// computes and stores all L2 distances into output. Output should be
// preallocated of size nx * ny, each element should be initialized to
// {lowest distance, -1}.
void all_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<std::pair<float, int64_t>>& output,
        const float* y_norms,
        const IDSelector* sel) {
    CollectAllResultHandler<CMax<float, int64_t>> res(ny, output);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_L2sqr_seq(x, y, d, nx, ny, res, sel);
    } else {
        exhaustive_L2sqr_blas(x, y, d, nx, ny, res, y_norms, sel);
    }
}

void knn_cosine(
        const float* x,
        const float* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* ha,
        const IDSelector* sel) {
    if (ha->k < distance_compute_min_k_reservoir) {
        HeapResultHandler<CMin<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        if (nx < distance_compute_blas_threshold) {
            exhaustive_cosine_seq(x, y, y_norms, d, nx, ny, res, sel);
        } else {
            exhaustive_cosine_blas(x, y, y_norms, d, nx, ny, res, sel);
        }
    } else {
        ReservoirResultHandler<CMin<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        if (nx < distance_compute_blas_threshold) {
            exhaustive_cosine_seq(x, y, y_norms, d, nx, ny, res, sel);
        } else {
            exhaustive_cosine_blas(x, y, y_norms, d, nx, ny, res, sel);
        }
    }
}

// computes and stores all cosine distances into output. Output should be
// preallocated of size nx * ny, each element should be initialized to
// {lowest distance, -1}.
void all_cosine(
        const float* x,
        const float* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<std::pair<float, int64_t>>& output,
        const IDSelector* sel) {
    CollectAllResultHandler<CMax<float, int64_t>> res(ny, output);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_cosine_seq(x, y, y_norms, d, nx, ny, res, sel);
    } else {
        exhaustive_cosine_blas(x, y, y_norms, d, nx, ny, res, sel);
    }
}

struct NopDistanceCorrection {
    float operator()(float dis, size_t /*qno*/, size_t /*bno*/) const {
        return dis;
    }
};

void knn_jaccard(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* ha,
        const IDSelector* sel) {
    if (d % 4 != 0) {
        // knn_jaccard_sse(x, y, d, nx, ny, res);
        FAISS_ASSERT_MSG(false, "dim is not multiple of 4!");
    } else {
        NopDistanceCorrection nop;
        HeapResultHandler<CMax<float, int64_t>> res(
                ha->nh, ha->val, ha->ids, ha->k);
        knn_jaccard_blas(x, y, d, nx, ny, res, nop, sel);
    }
}

/***************************************************************************
 * Range search
 ***************************************************************************/

void range_search_L2sqr(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    RangeSearchResultHandler<CMax<float, int64_t>> resh(res, radius);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_L2sqr_seq(x, y, d, nx, ny, resh, sel);
    } else {
        exhaustive_L2sqr_blas(x, y, d, nx, ny, resh, nullptr, sel);
    }
}

void range_search_inner_product(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    RangeSearchResultHandler<CMin<float, int64_t>> resh(res, radius);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_inner_product_seq(x, y, d, nx, ny, resh, sel);
    } else {
        exhaustive_inner_product_blas(x, y, d, nx, ny, resh, sel);
    }
}

void range_search_cosine(
        const float* x,
        const float* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    RangeSearchResultHandler<CMin<float, int64_t>> resh(res, radius);
    if (nx < distance_compute_blas_threshold) {
        exhaustive_cosine_seq(x, y, y_norms, d, nx, ny, resh, sel);
    } else {
        exhaustive_cosine_blas(x, y, y_norms, d, nx, ny, resh, sel);
    }
}

/***************************************************************************
 * compute a subset of  distances
 ***************************************************************************/

/* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_inner_products_by_idx(
        float* __restrict ip,
        const float* x,
        const float* y,
        const int64_t* __restrict ids, /* for y vecs */
        size_t d,
        size_t nx,
        size_t ny) {
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict ipj = ip + j * ny;

        // // baseline version
        // for (size_t i = 0; i < ny; i++) {
        //     if (idsj[i] < 0) {
        //         ipj[i] = -INFINITY;
        //     } else {
        //         ipj[i] = fvec_inner_product(xj, y + d * idsj[i], d);
        //     }
        // }

        // todo aguzhva: this version deviates from the baseline
        //   on not assigning -INFINITY

        // the lambda that filters acceptable elements.
        auto filter = [=](const size_t i) { return (idsj[i] >= 0); };

        // the lambda that applies a filtered element.
        auto apply = [=](const float dis, const size_t i) {
            ipj[i] = dis;
        };

        // compute distances
        fvec_inner_products_ny_by_idx_if(
            xj,
            y,
            idsj,
            d,
            ny,
            filter,
            apply
        );
    }
}

/* compute the inner product between x and a subset y of ny vectors,
   whose indices are given by idy.  */
void fvec_L2sqr_by_idx(
        float* __restrict dis,
        const float* x,
        const float* y,
        const int64_t* __restrict ids, /* ids of y vecs */
        size_t d,
        size_t nx,
        size_t ny) {
#pragma omp parallel for
    for (int64_t j = 0; j < nx; j++) {
        const int64_t* __restrict idsj = ids + j * ny;
        const float* xj = x + j * d;
        float* __restrict disj = dis + j * ny;

        // // baseline version
        // for (size_t i = 0; i < ny; i++) {
        //     if (idsj[i] < 0) {
        //         disj[i] = INFINITY;
        //     } else {
        //         disj[i] = fvec_L2sqr(xj, y + d * idsj[i], d);
        //     }
        // }

        // todo aguzhva: this version deviates from the baseline
        //   on not assigning INFINITY

        // the lambda that filters acceptable elements.
        auto filter = [=](const size_t i) { return (idsj[i] >= 0); };

        // the lambda that applies a filtered element.
        auto apply = [=](const float dis, const size_t i) {
            disj[i] = dis;
        };

        // compute distances
        fvec_L2sqr_ny_by_idx_if(
            xj,
            y,
            idsj,
            d,
            ny,
            filter,
            apply
        );
    }
}

void pairwise_indexed_L2sqr(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis) {
#pragma omp parallel for
    for (int64_t j = 0; j < n; j++) {
        if (ix[j] >= 0 && iy[j] >= 0) {
            dis[j] = fvec_L2sqr(x + d * ix[j], y + d * iy[j], d);
        } else {
            dis[j] = INFINITY;
        }
    }
}

void pairwise_indexed_inner_product(
        size_t d,
        size_t n,
        const float* x,
        const int64_t* ix,
        const float* y,
        const int64_t* iy,
        float* dis) {
#pragma omp parallel for
    for (int64_t j = 0; j < n; j++) {
        if (ix[j] >= 0 && iy[j] >= 0) {
            dis[j] = fvec_inner_product(x + d * ix[j], y + d * iy[j], d);
        } else {
            dis[j] = -INFINITY;
        }
    }
}

/* Find the nearest neighbors for nx queries in a set of ny vectors
   indexed by ids. May be useful for re-ranking a pre-selected vector list */
void knn_inner_products_by_idx(
        const float* x,
        const float* y,
        const int64_t* ids,
        size_t d,
        size_t nx,
        size_t ny,
        float_minheap_array_t* res) {
    size_t k = res->k;

#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        const float* x_ = x + i * d;
        const int64_t* idsi = ids + i * ny;
        size_t j;
        float* __restrict simi = res->get_val(i);
        int64_t* __restrict idxi = res->get_ids(i);
        minheap_heapify(k, simi, idxi);

        for (j = 0; j < ny; j++) {
            if (idsi[j] < 0)
                break;
            float ip = fvec_inner_product(x_, y + d * idsi[j], d);

            if (ip > simi[0]) {
                minheap_replace_top(k, simi, idxi, ip, idsi[j]);
            }
        }
        minheap_reorder(k, simi, idxi);
    }
}

void knn_L2sqr_by_idx(
        const float* x,
        const float* y,
        const int64_t* __restrict ids,
        size_t d,
        size_t nx,
        size_t ny,
        float_maxheap_array_t* res) {
    size_t k = res->k;

#pragma omp parallel for
    for (int64_t i = 0; i < nx; i++) {
        const float* x_ = x + i * d;
        const int64_t* __restrict idsi = ids + i * ny;
        float* __restrict simi = res->get_val(i);
        int64_t* __restrict idxi = res->get_ids(i);
        maxheap_heapify(res->k, simi, idxi);
        for (size_t j = 0; j < ny; j++) {
            float disij = fvec_L2sqr(x_, y + d * idsi[j], d);

            if (disij < simi[0]) {
                maxheap_replace_top(k, simi, idxi, disij, idsi[j]);
            }
        }
        maxheap_reorder(res->k, simi, idxi);
    }
}

void pairwise_L2sqr(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis,
        int64_t ldq,
        int64_t ldb,
        int64_t ldd) {
    if (nq == 0 || nb == 0)
        return;
    if (ldq == -1)
        ldq = d;
    if (ldb == -1)
        ldb = d;
    if (ldd == -1)
        ldd = nb;

    // store in beginning of distance matrix to avoid malloc
    float* b_norms = dis;

#pragma omp parallel for
    for (int64_t i = 0; i < nb; i++)
        b_norms[i] = fvec_norm_L2sqr(xb + i * ldb, d);

#pragma omp parallel for
    for (int64_t i = 1; i < nq; i++) {
        float q_norm = fvec_norm_L2sqr(xq + i * ldq, d);
        for (int64_t j = 0; j < nb; j++)
            dis[i * ldd + j] = q_norm + b_norms[j];
    }

    {
        float q_norm = fvec_norm_L2sqr(xq, d);
        for (int64_t j = 0; j < nb; j++)
            dis[j] += q_norm;
    }

    {
        FINTEGER nbi = nb, nqi = nq, di = d, ldqi = ldq, ldbi = ldb, lddi = ldd;
        float one = 1.0, minus_2 = -2.0;

        sgemm_("Transposed",
               "Not transposed",
               &nbi,
               &nqi,
               &di,
               &minus_2,
               xb,
               &ldbi,
               xq,
               &ldqi,
               &one,
               dis,
               &lddi);
    }
}

void inner_product_to_L2sqr(
        float* __restrict dis,
        const float* nr1,
        const float* nr2,
        size_t n1,
        size_t n2) {
#pragma omp parallel for
    for (int64_t j = 0; j < n1; j++) {
        float* disj = dis + j * n2;
        for (size_t i = 0; i < n2; i++)
            disj[i] = nr1[j] + nr2[i] - 2 * disj[i];
    }
}

// todo aguzhva: Faiss 1.7.4, no longer used in IndexFlat::assign and Clustering.
void elkan_L2_sse(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        int64_t* ids,
        float* val,
        float* tmp_buffer,
        size_t sym_dim) {
    if (nx == 0 || ny == 0) {
        return;
    }

    for (size_t j0 = 0; j0 < ny; j0 += sym_dim) {
        size_t j1 = j0 + sym_dim;
        if (j1 > ny)
            j1 = ny;

        auto Y = [&](size_t i, size_t j) -> float& {
            assert(i != j);
            i -= j0, j -= j0;
            return (i > j) ? tmp_buffer[j + i * (i - 1) / 2]
                           : tmp_buffer[i + j * (j - 1) / 2];
        };

#pragma omp parallel
        {
            int nt = omp_get_num_threads();
            int rank = omp_get_thread_num();
            for (size_t i = j0 + 1 + rank; i < j1; i += nt) {
                const float* y_i = y + i * d;
                for (size_t j = j0; j < i; j++) {
                    const float* y_j = y + j * d;
                    Y(i, j) = fvec_L2sqr(y_i, y_j, d);
                }
            }
        }

#pragma omp parallel for
        for (size_t i = 0; i < nx; i++) {
            const float* x_i = x + i * d;

            int64_t ids_i = j0;
            float val_i = fvec_L2sqr(x_i, y + j0 * d, d);
            float val_i_time_4 = val_i * 4;
            for (size_t j = j0 + 1; j < j1; j++) {
                if (val_i_time_4 <= Y(ids_i, j)) {
                    continue;
                }
                const float* y_j = y + j * d;
                float disij = fvec_L2sqr(x_i, y_j, d / 2);
                if (disij >= val_i) {
                    continue;
                }
                disij += fvec_L2sqr(x_i + d / 2, y_j + d / 2, d - d / 2);
                if (disij < val_i) {
                    ids_i = j;
                    val_i = disij;
                    val_i_time_4 = val_i * 4;
                }
            }

            if (j0 == 0 || val[i] > val_i) {
                val[i] = val_i;
                ids[i] = ids_i;
            }
        }
    }

}

} // namespace faiss
