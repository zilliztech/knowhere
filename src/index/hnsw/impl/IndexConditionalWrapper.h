// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>

#include "faiss/cppcontrib/knowhere/Index.h"
#include "index/hnsw/faiss_hnsw_config.h"
#include "knowhere/bitsetview.h"

namespace knowhere {

struct HnswSearchThresholds {
    static constexpr float kHnswSearchKnnBFFilterThreshold = 0.93f;
    static constexpr float kHnswSearchRangeBFFilterThreshold = 0.97f;
    static constexpr float kHnswSearchBFTopkThreshold = 0.5f;
};

// Decides whether a brute force should be used instead of a regular HNSW search.
// This may be applicable in case of very large topk values or
//   extremely high filtering levels.
std::optional<bool>
WhetherPerformBruteForceSearch(const faiss::Index* index, const BaseConfig& cfg, const BitsetView& bitset);

// Decides whether a brute force should be used instead of a regular HNSW range search.
// This may be applicable in case of very large topk values or
//   extremely high filtering levels.
std::optional<bool>
WhetherPerformBruteForceRangeSearch(const faiss::Index* index, const FaissHnswConfig& cfg, const BitsetView& bitset);

// first return arg: returns nullptr in case of invalid index
// second return arg: returns whether an index does the refine
//
// `whether_to_enable_refine` allows to enable the refine for the search if the
//    index was trained with the refine.
std::tuple<std::unique_ptr<faiss::cppcontrib::knowhere::Index>, bool>
create_conditional_hnsw_wrapper(faiss::cppcontrib::knowhere::Index* index, const FaissHnswConfig& hnsw_cfg,
                                const bool whether_bf_search, const bool whether_to_enable_refine);

}  // namespace knowhere
