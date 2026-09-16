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

#ifndef SEARCH_HINT_H
#define SEARCH_HINT_H

#include <cstdint>
#include <optional>
#include <vector>

namespace knowhere {

// Request-only hints; offsets are public (segment) row IDs, not index storage IDs.
struct SearchHint {
    int64_t offset = 0;
    int64_t count = 0;
    // Finite query-to-centroid distance: smaller values prioritize graph seeds.
    // This is not the distance to an individual row; BF ignores it.
    float id_distance = 0.0f;
};

struct QuerySearchHints {
    std::vector<SearchHint> ranges;
};

// One optional entry per query: nullopt preserves ordinary search; empty ranges
// are an explicit hint with no candidates. Searchers interpret ranges internally.
using BatchSearchHints = std::vector<std::optional<QuerySearchHints>>;

// The single DataSet payload is std::shared_ptr<const BatchSearchHints>.
// Use DataSet::SetSearchHints/GetSearchHints to write/read its immutable snapshot.
constexpr const char* kSearchHintsField = "search_hints";

}  // namespace knowhere

#endif  // SEARCH_HINT_H
