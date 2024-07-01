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

#include <cstddef>
#include <memory>
#include <vector>

#include "knowhere/bitsetview.h"

namespace knowhere {

inline bool
distance_in_range(const float dist, const float radius, const float range_filter, const bool is_ip) {
    return ((is_ip && radius < dist && dist <= range_filter) || (!is_ip && range_filter <= dist && dist < radius));
}

void
FilterRangeSearchResultForOneNq(std::vector<float>& distances, std::vector<int64_t>& labels, const bool is_ip,
                                const float radius, const float range_filter);

struct RangeSearchResult {
    std::unique_ptr<float[]> distances;
    std::unique_ptr<int64_t[]> labels;
    std::unique_ptr<size_t[]> lims;
};

RangeSearchResult
GetRangeSearchResult(const std::vector<std::vector<float>>& result_distances,
                     const std::vector<std::vector<int64_t>>& result_labels, const bool is_ip, const int64_t nq,
                     const float radius, const float range_filter);

}  // namespace knowhere
