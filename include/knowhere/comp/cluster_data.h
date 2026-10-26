// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once
#include <cstdint>
#include <vector>

namespace knowhere {
// In input row order. The backend owns all permutations; these IDs need not be dense.
struct ClusterData {
    std::vector<int64_t> cluster_ids;
    std::vector<float> cluster_distances;
};
inline constexpr const char* kClusterDataField = "cluster_data";
}  // namespace knowhere
