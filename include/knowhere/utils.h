// Copyright (C) 2019-2023 Zilliz. All rights reserved.
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

#include <strings.h>

#include <algorithm>
#include <vector>

#include "knowhere/dataset.h"
#include "knowhere/index_sequence.h"

namespace knowhere {

extern const float FloatAccuracy;

inline bool
IsMetricType(const std::string& str, const knowhere::MetricType& metric_type) {
    return !strcasecmp(str.data(), metric_type.c_str());
}

inline bool
IsFlatIndex(const knowhere::IndexType& index_type) {
    static std::vector<knowhere::IndexType> flat_index_list = {IndexEnum::INDEX_FAISS_IDMAP,
                                                               IndexEnum::INDEX_FAISS_GPU_IDMAP};
    return std::find(flat_index_list.begin(), flat_index_list.end(), index_type) != flat_index_list.end();
}

extern float
NormalizeVec(float* x, int32_t d);

extern std::vector<float>
NormalizeVecs(float* x, size_t rows, int32_t dim);

extern void
Normalize(const DataSet& dataset);

extern std::unique_ptr<float[]>
CopyAndNormalizeVecs(const float* x, size_t rows, int32_t dim);

constexpr inline uint64_t seed = 0xc70f6907UL;

inline uint64_t
hash_vec(const float* x, size_t d) {
    uint64_t h = seed;
    for (size_t i = 0; i < d; ++i) {
        h = h * 13331 + *(uint32_t*)(x + i);
    }
    return h;
}

inline uint64_t
hash_binary_vec(const uint8_t* x, size_t d) {
    size_t len = (d + 7) / 8;
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h = h * 13331 + x[i];
    }
    return h;
}

template <typename T>
inline T
round_down(const T value, const T align) {
    return value / align * align;
}

extern void
ConvertIVFFlatIfNeeded(IndexSequence& binset, const MetricType metric_type, const uint8_t* raw_data,
                       const size_t raw_size);

bool
UseDiskLoad(const std::string& index_type, const std::string& /*version*/);

}  // namespace knowhere
