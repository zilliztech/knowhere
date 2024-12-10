// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License

#include <algorithm>

#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/distances_if.h>
#include <faiss/utils/half_precision_floating_point_distances.h>
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/operands.h"
#include "simd/hook.h"
namespace faiss {
namespace {
template <typename DataType, class BlockResultHandler, class IDSelector>
void half_precision_floating_point_exhaustive_inner_product_impl(
        const DataType* __restrict x,
        const DataType* __restrict y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res,
        const IDSelector& selector) {
    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;

    SingleResultHandler resi(res);
    for (int64_t i = 0; i < nx; i++) {
        const DataType* x_i = x + i * d;
        resi.begin(i);

        // the lambda that applies a filtered element.
        auto apply = [&resi](const float ip, const idx_t j) {
            resi.add_result(ip, j);
        };
        if constexpr (std::is_same_v<IDSelector, IDSelectorArray>) {
            // todo: need more tests about this branch
            auto filter = [](const size_t j) { return true; };
            if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
                fp16_vec_inner_products_ny_by_idx_if(
                        x_i, y, selector.ids, d, selector.n, filter, apply);
            } else {
                bf16_vec_inner_products_ny_by_idx_if(
                        x_i, y, selector.ids, d, selector.n, filter, apply);
            }
        } else {
            // the lambda that filters acceptable elements.
            auto filter = [&selector](const size_t j) {
                return selector.is_member(j);
            };
            if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
                fp16_vec_inner_products_ny_if(x_i, y, d, ny, filter, apply);
            } else {
                bf16_vec_inner_products_ny_if(x_i, y, d, ny, filter, apply);
            }
        }
        resi.end();
    }
}

template <typename DataType, class BlockResultHandler, class IDSelector>
void half_precision_floating_point_exhaustive_L2sqr_seq_impl(
        const DataType* __restrict x,
        const DataType* __restrict y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res,
        const IDSelector& selector) {
    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;

    SingleResultHandler resi(res);
    for (int64_t i = 0; i < nx; i++) {
        const DataType* x_i = x + i * d;
        resi.begin(i);

        // the lambda that applies a filtered element.
        auto apply = [&resi](const float ip, const idx_t j) {
            resi.add_result(ip, j);
        };
        if constexpr (std::is_same_v<IDSelector, IDSelectorArray>) {
            // todo: need more tests about this branch
            auto filter = [](const size_t j) { return true; };
            if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
                fp16_vec_L2sqr_ny_by_idx_if(
                        x_i, y, selector.ids, d, selector->n, filter, apply);
            } else {
                bf16_vec_L2sqr_ny_by_idx_if(
                        x_i, y, selector.ids, d, selector->n, filter, apply);
            }
        } else {
            // the lambda that filters acceptable elements.
            auto filter = [&selector](const size_t j) {
                return selector.is_member(j);
            };
            if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
                fp16_vec_L2sqr_ny_if(x_i, y, d, ny, filter, apply);
            } else {
                bf16_vec_L2sqr_ny_if(x_i, y, d, ny, filter, apply);
            }
        }
        resi.end();
    }
}

template <typename DataType, class BlockResultHandler, class IDSelector>
void half_precision_floating_point_exhaustive_cosine_seq_impl(
        const DataType* __restrict x,
        const DataType* __restrict y,
        const float* __restrict y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res,
        const IDSelector& selector) {
    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;
    typedef float (*NormComputer)(const DataType*, size_t);
    NormComputer norm_computer;
    if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
        norm_computer = fp16_vec_norm_L2sqr;
    } else {
        norm_computer = bf16_vec_norm_L2sqr;
    }
    SingleResultHandler resi(res);
    for (int64_t i = 0; i < nx; i++) {
        const DataType* x_i = x + i * d;
        // distance div x_norm before pushing into the heap
        auto x_norm = sqrtf(norm_computer(x_i, d));
        x_norm = (x_norm == 0.0 ? 1.0 : x_norm);
        auto apply = [&resi, x_norm, y, y_norms, d, norm_computer](
                             const float ip, const idx_t j) {
            float y_norm = (y_norms != nullptr)
                    ? y_norms[j]
                    : sqrtf(norm_computer(y + j * d, d));

            y_norm = (y_norm == 0.0 ? 1.0 : y_norm);
            resi.add_result(ip / (x_norm * y_norm), j);
        };
        resi.begin(i);
        // the lambda that applies a filtered element
        if constexpr (std::is_same_v<IDSelector, IDSelectorArray>) {
            // todo: need more tests about this branch
            auto filter = [](const size_t j) { return true; };
            if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
                fp16_vec_inner_products_ny_by_idx_if(
                        x_i, y, selector.ids, d, selector->n, filter, apply);
            } else {
                bf16_vec_inner_products_ny_by_idx_if(
                        x_i, y, selector.ids, d, selector->n, filter, apply);
            }
        } else {
            // the lambda that filters acceptable elements.
            auto filter = [&selector](const size_t j) {
                return selector.is_member(j);
            };
            if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
                fp16_vec_inner_products_ny_if(x_i, y, d, ny, filter, apply);
            } else {
                bf16_vec_inner_products_ny_if(x_i, y, d, ny, filter, apply);
            }
        }
        resi.end();
    }
}
} // namespace

template <typename DataType>
void half_precision_floating_point_knn_inner_product(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* vals,
        int64_t* ids,
        const IDSelector* sel) {
    int64_t imin = 0;
    if (auto selr = dynamic_cast<const IDSelectorRange*>(sel)) {
        imin = std::max(selr->imin, int64_t(0));
        int64_t imax = std::min(selr->imax, int64_t(ny));
        ny = imax - imin;
        y += d * imin;
        sel = nullptr;
    }
    if (k < distance_compute_min_k_reservoir) {
        HeapBlockResultHandler<CMin<float, int64_t>> res(nx, vals, ids, k);
        if (const auto* sel_bs =
                    dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
            half_precision_floating_point_exhaustive_inner_product_impl(
                    x, y, d, nx, ny, res, *sel_bs);
        } else if (sel == nullptr) {
            half_precision_floating_point_exhaustive_inner_product_impl(
                    x, y, d, nx, ny, res, IDSelectorAll());
        } else {
            half_precision_floating_point_exhaustive_inner_product_impl(
                    x, y, d, nx, ny, res, *sel);
        }
    } else {
        ReservoirBlockResultHandler<CMin<float, int64_t>> res(nx, vals, ids, k);
        if (const auto* sel_bs =
                    dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
            half_precision_floating_point_exhaustive_inner_product_impl(
                    x, y, d, nx, ny, res, *sel_bs);
        } else if (sel == nullptr) {
            half_precision_floating_point_exhaustive_inner_product_impl(
                    x, y, d, nx, ny, res, IDSelectorAll());
        } else {
            half_precision_floating_point_exhaustive_inner_product_impl(
                    x, y, d, nx, ny, res, *sel);
        }
    }

    if (imin != 0) {
        for (size_t i = 0; i < nx * k; i++) {
            if (ids[i] >= 0) {
                ids[i] += imin;
            }
        }
    }
    return;
}

template <typename DataType>
void half_precision_floating_point_all_inner_product(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<knowhere::DistId>& output,
        const IDSelector* sel) {
    CollectAllResultHandler<CMax<float, int64_t>> res(nx, ny, output);
    if (const auto* sel_bs =
                dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
        half_precision_floating_point_exhaustive_inner_product_impl(
                x, y, d, nx, ny, res, *sel_bs);
    } else if (sel == nullptr) {
        half_precision_floating_point_exhaustive_inner_product_impl(
                x, y, d, nx, ny, res, IDSelectorAll());
    } else {
        half_precision_floating_point_exhaustive_inner_product_impl(
                x, y, d, nx, ny, res, *sel);
    }
    return;
}

template <typename DataType>
void half_precision_floating_point_knn_L2sqr(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* vals,
        int64_t* ids,
        const float* y_norm2,
        const IDSelector* sel) {
    int64_t imin = 0;
    if (auto selr = dynamic_cast<const IDSelectorRange*>(sel)) {
        imin = std::max(selr->imin, int64_t(0));
        int64_t imax = std::min(selr->imax, int64_t(ny));
        ny = imax - imin;
        y += d * imin;
        sel = nullptr;
    }
    if (k < distance_compute_min_k_reservoir) {
        HeapBlockResultHandler<CMax<float, int64_t>> res(nx, vals, ids, k);
        if (const auto* sel_bs =
                    dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
            half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                    x, y, d, nx, ny, res, *sel_bs);
        } else if (sel == nullptr) {
            half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                    x, y, d, nx, ny, res, IDSelectorAll());
        } else {
            half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                    x, y, d, nx, ny, res, *sel);
        }
    } else {
        ReservoirBlockResultHandler<CMax<float, int64_t>> res(nx, vals, ids, k);
        if (const auto* sel_bs =
                    dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
            half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                    x, y, d, nx, ny, res, *sel_bs);
        } else if (sel == nullptr) {
            half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                    x, y, d, nx, ny, res, IDSelectorAll());
        } else {
            half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                    x, y, d, nx, ny, res, *sel);
        }
    }
    if (imin != 0) {
        for (size_t i = 0; i < nx * k; i++) {
            if (ids[i] >= 0) {
                ids[i] += imin;
            }
        }
    }
    return;
}

template <typename DataType>
void half_precision_floating_point_all_L2sqr(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<knowhere::DistId>& output,
        const float* y_norms,
        const IDSelector* sel) {
    CollectAllResultHandler<CMax<float, int64_t>> res(nx, ny, output);
    if (const auto* sel_bs =
                dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
        half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                x, y, d, nx, ny, res, *sel_bs);
    } else if (sel == nullptr) {
        half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                x, y, d, nx, ny, res, IDSelectorAll());
    } else {
        half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                x, y, d, nx, ny, res, *sel);
    }
    return;
}

template <typename DataType>
void half_precision_floating_point_knn_cosine(
        const DataType* x,
        const DataType* y,
        const float* y_norm2,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* vals,
        int64_t* ids,
        const IDSelector* sel) {
    int64_t imin = 0;
    if (auto selr = dynamic_cast<const IDSelectorRange*>(sel)) {
        imin = std::max(selr->imin, int64_t(0));
        int64_t imax = std::min(selr->imax, int64_t(ny));
        ny = imax - imin;
        y += d * imin;
        sel = nullptr;
    }
    if (k < distance_compute_min_k_reservoir) {
        HeapBlockResultHandler<CMin<float, int64_t>> res(nx, vals, ids, k);
        if (const auto* sel_bs =
                    dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
            half_precision_floating_point_exhaustive_cosine_seq_impl(
                    x, y, y_norm2, d, nx, ny, res, *sel_bs);
        } else if (sel == nullptr) {
            half_precision_floating_point_exhaustive_cosine_seq_impl(
                    x, y, y_norm2, d, nx, ny, res, IDSelectorAll());
        } else {
            half_precision_floating_point_exhaustive_cosine_seq_impl(
                    x, y, y_norm2, d, nx, ny, res, *sel);
        }
    } else {
        ReservoirBlockResultHandler<CMin<float, int64_t>> res(nx, vals, ids, k);
        if (const auto* sel_bs =
                    dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
            half_precision_floating_point_exhaustive_cosine_seq_impl(
                    x, y, y_norm2, d, nx, ny, res, *sel_bs);
        } else if (sel == nullptr) {
            half_precision_floating_point_exhaustive_cosine_seq_impl(
                    x, y, y_norm2, d, nx, ny, res, IDSelectorAll());
        } else {
            half_precision_floating_point_exhaustive_cosine_seq_impl(
                    x, y, y_norm2, d, nx, ny, res, *sel);
        }
    }
    if (imin != 0) {
        for (size_t i = 0; i < nx * k; i++) {
            if (ids[i] >= 0) {
                ids[i] += imin;
            }
        }
    }
    return;
}

template <typename DataType>
void half_precision_floating_point_all_cosine(
        const DataType* x,
        const DataType* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<knowhere::DistId>& output,
        const IDSelector* sel) {
    CollectAllResultHandler<CMax<float, int64_t>> res(nx, ny, output);
    if (const auto* sel_bs =
                dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
        half_precision_floating_point_exhaustive_cosine_seq_impl(
                x, y, y_norms, d, nx, ny, res, *sel_bs);
    } else if (sel == nullptr) {
        half_precision_floating_point_exhaustive_cosine_seq_impl(
                x, y, y_norms, d, nx, ny, res, IDSelectorAll());
    } else {
        half_precision_floating_point_exhaustive_cosine_seq_impl(
                x, y, y_norms, d, nx, ny, res, *sel);
    }
    return;
}

/***************************************************************************
 * Range search
 ***************************************************************************/
struct RangeSearchResult;

template <typename DataType>
void half_precision_floating_point_range_search_L2sqr(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    RangeSearchBlockResultHandler<CMax<float, int64_t>> resh(res, radius);
    if (const auto* sel_bs =
                dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
        half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                x, y, d, nx, ny, resh, *sel_bs);
    } else if (sel == nullptr) {
        half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                x, y, d, nx, ny, resh, IDSelectorAll());
    } else {
        half_precision_floating_point_exhaustive_L2sqr_seq_impl(
                x, y, d, nx, ny, resh, *sel);
    }
    return;
}

template <typename DataType>
void half_precision_floating_point_range_search_inner_product(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    RangeSearchBlockResultHandler<CMin<float, int64_t>> resh(res, radius);
    if (const auto* sel_bs =
                dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
        half_precision_floating_point_exhaustive_inner_product_impl(
                x, y, d, nx, ny, resh, *sel_bs);
    } else if (sel == nullptr) {
        half_precision_floating_point_exhaustive_inner_product_impl(
                x, y, d, nx, ny, resh, IDSelectorAll());
    } else {
        half_precision_floating_point_exhaustive_inner_product_impl(
                x, y, d, nx, ny, resh, *sel);
    }
    return;
}

// Knowhere-specific function
template <typename DataType>
void half_precision_floating_point_range_search_cosine(
        const DataType* x,
        const DataType* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* res,
        const IDSelector* sel) {
    RangeSearchBlockResultHandler<CMin<float, int64_t>> resh(res, radius);
    if (const auto* sel_bs =
                dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel)) {
        half_precision_floating_point_exhaustive_cosine_seq_impl(
                x, y, y_norms, d, nx, ny, resh, *sel_bs);
    } else if (sel == nullptr) {
        half_precision_floating_point_exhaustive_cosine_seq_impl(
                x, y, y_norms, d, nx, ny, resh, IDSelectorAll());
    } else {
        half_precision_floating_point_exhaustive_cosine_seq_impl(
                x, y, y_norms, d, nx, ny, resh, *sel);
    }
    return;
}

// knn functions
template void faiss::half_precision_floating_point_knn_inner_product<
        knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        size_t,
        size_t,
        size_t,
        size_t,
        float*,
        int64_t*,
        const IDSelector*);
template void faiss::half_precision_floating_point_knn_inner_product<
        knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        size_t,
        size_t,
        size_t,
        size_t,
        float*,
        int64_t*,
        const IDSelector*);
template void faiss::half_precision_floating_point_all_inner_product<
        knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        size_t,
        size_t,
        size_t,
        std::vector<knowhere::DistId>&,
        const IDSelector*);
template void faiss::half_precision_floating_point_all_inner_product<
        knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        size_t,
        size_t,
        size_t,
        std::vector<knowhere::DistId>&,
        const IDSelector*);
template void faiss::half_precision_floating_point_knn_L2sqr<knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        size_t,
        size_t,
        size_t,
        size_t,
        float*,
        int64_t*,
        const float*,
        const IDSelector*);
template void faiss::half_precision_floating_point_knn_L2sqr<knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        size_t,
        size_t,
        size_t,
        size_t,
        float*,
        int64_t*,
        const float*,
        const IDSelector*);
template void faiss::half_precision_floating_point_all_L2sqr<knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        size_t,
        size_t,
        size_t,
        std::vector<knowhere::DistId>&,
        const float*,
        const IDSelector*);
template void faiss::half_precision_floating_point_all_L2sqr<knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        size_t,
        size_t,
        size_t,
        std::vector<knowhere::DistId>&,
        const float*,
        const IDSelector*);
template void faiss::half_precision_floating_point_knn_cosine<knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        const float*,
        size_t,
        size_t,
        size_t,
        size_t,
        float*,
        int64_t*,
        const IDSelector*);
template void faiss::half_precision_floating_point_knn_cosine<knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        const float*,
        size_t,
        size_t,
        size_t,
        size_t,
        float*,
        int64_t*,
        const IDSelector*);
template void faiss::half_precision_floating_point_all_cosine<knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        const float*,
        size_t,
        size_t,
        size_t,
        std::vector<knowhere::DistId>&,
        const IDSelector*);
template void faiss::half_precision_floating_point_all_cosine<knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        const float*,
        size_t,
        size_t,
        size_t,
        std::vector<knowhere::DistId>&,
        const IDSelector*);
// range search functions
template void faiss::half_precision_floating_point_range_search_L2sqr<
        knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        size_t,
        size_t,
        size_t,
        float,
        faiss::RangeSearchResult*,
        const faiss::IDSelector*);
template void faiss::half_precision_floating_point_range_search_L2sqr<
        knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        size_t,
        size_t,
        size_t,
        float,
        faiss::RangeSearchResult*,
        const faiss::IDSelector*);
template void faiss::half_precision_floating_point_range_search_inner_product<
        knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        size_t,
        size_t,
        size_t,
        float,
        faiss::RangeSearchResult*,
        const faiss::IDSelector*);
template void faiss::half_precision_floating_point_range_search_inner_product<
        knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        size_t,
        size_t,
        size_t,
        float,
        faiss::RangeSearchResult*,
        const faiss::IDSelector*);
template void faiss::half_precision_floating_point_range_search_cosine<
        knowhere::fp16>(
        const knowhere::fp16*,
        const knowhere::fp16*,
        const float*,
        size_t,
        size_t,
        size_t,
        float,
        faiss::RangeSearchResult*,
        const faiss::IDSelector*);
template void faiss::half_precision_floating_point_range_search_cosine<
        knowhere::bf16>(
        const knowhere::bf16*,
        const knowhere::bf16*,
        const float*,
        size_t,
        size_t,
        size_t,
        float,
        faiss::RangeSearchResult*,
        const faiss::IDSelector*);
} // namespace faiss