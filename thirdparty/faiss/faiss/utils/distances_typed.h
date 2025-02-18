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

#ifndef FAISS_HALF_PRECISION_FLOATING_POINT_DISTANCES_H
#define FAISS_HALF_PRECISION_FLOATING_POINT_DISTANCES_H
#pragma once
#include <faiss/utils/Heap.h>
#include <stdint.h>
#include <vector>
#include "knowhere/object.h"
namespace faiss {
struct IDSelector;
/***************************************************************************
 * KNN functions
 ***************************************************************************/
// Knowhere-specific function
template <typename DataType>
void knn_inner_product_typed(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const IDSelector* sel = nullptr);

template <typename DataType>
void all_inner_product_typed(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<knowhere::DistId>& output,
        const IDSelector* sel = nullptr);

template <typename DataType>
void knn_L2sqr_typed(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const float* y_norm2 = nullptr,
        const IDSelector* sel = nullptr);

template <typename DataType>
void all_L2sqr_typed(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<knowhere::DistId>& output,
        const float* y_norms = nullptr,
        const IDSelector* sel = nullptr);

template <typename DataType>
void knn_cosine_typed(
        const DataType* x,
        const DataType* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        size_t k,
        float* distances,
        int64_t* indexes,
        const IDSelector* sel = nullptr);

template <typename DataType>
void all_cosine_typed(
        const DataType* x,
        const DataType* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        std::vector<knowhere::DistId>& output,
        const IDSelector* sel = nullptr);

/***************************************************************************
 * Range search
 ***************************************************************************/
struct RangeSearchResult;
template <typename DataType>
void range_search_L2sqr_typed(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel = nullptr);

/// same as range_search_L2sqr for the inner product similarity
template <typename DataType>
void range_search_inner_product_typed(
        const DataType* x,
        const DataType* y,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel = nullptr);

// Knowhere-specific function
template <typename DataType>
void range_search_cosine_typed(
        const DataType* x,
        const DataType* y,
        const float* y_norms,
        size_t d,
        size_t nx,
        size_t ny,
        float radius,
        RangeSearchResult* result,
        const IDSelector* sel = nullptr);
} // namespace faiss
#endif // FAISS_HALF_PRECISION_FLOATING_POINT_DISTANCES_H