// Copyright (C) 2019-2020 Zilliz. All rights reserved.
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

namespace knowhere {

// SearchHint is a global-index head-index hint for a graph search: the local row
// range of a matched centroid within the segment being searched (id_offset = start
// row, id_range = row count) and the query-to-centroid distance. The graph index
// uses these as entry-point seeds (and an ordering/bound signal) for the query.
//
// This is the shared on-the-wire type carried in the query DataSet under
// kSearchHintsField as std::vector<std::vector<SearchHint>> (outer = per query).
// Producers and consumers must use this exact definition for the std::any
// round-trip to succeed.
struct SearchHint {
    int32_t id_offset = 0;
    int32_t id_range = 0;
    float id_distance = 0.0f;
};

// DataSet meta key for per-query search hints.
constexpr const char* kSearchHintsField = "search_hints";

}  // namespace knowhere

#endif  // SEARCH_HINT_H
