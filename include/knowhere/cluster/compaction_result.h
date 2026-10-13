// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef KNOWHERE_COMPACTION_RESULT_H
#define KNOWHERE_COMPACTION_RESULT_H

#include <cstdint>
#include <vector>

namespace knowhere {

// A group of indivisible centroid buckets. IDs are dense and match the position
// in CompactionResult::centroid_groups; rows is the sum of its centroid counts.
struct CentroidGroup {
    uint32_t centroid_group_id = 0;
    uint64_t rows = 0;
    std::vector<uint32_t> centroids;
};

// Owned execution result, independent of the input counts and cluster lifetime.
// centroid_counts.size() == centroid_count; their sum equals row_count.
// Every nonempty centroid occurs in exactly one group; empty ones are omitted.
// Row bounds are soft: an indivisible oversized bucket can exceed the target.
// Planner diagnostics belong in logs, not in this execution contract.
struct CompactionResult {
    uint64_t row_count = 0;
    uint32_t centroid_count = 0;
    std::vector<uint64_t> centroid_counts;
    std::vector<CentroidGroup> centroid_groups;
};

}  // namespace knowhere

#endif  // KNOWHERE_COMPACTION_RESULT_H
