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

// One query has one hint containing its candidate cluster IDs. Layout, row
// ranges and graph seeds belong to the index.
struct SearchHint {
    std::vector<int64_t> clusters;
};
// Entries align with query rows: nullopt preserves ordinary search, while an
// explicit SearchHint with empty clusters has no candidates.
using BatchSearchHints = std::vector<std::optional<SearchHint>>;

// The single DataSet payload is std::shared_ptr<const BatchSearchHints>.
// Use DataSet::SetSearchHints/GetSearchHints to write/read its immutable snapshot.
constexpr const char* kSearchHintsField = "search_hints";

}  // namespace knowhere

#endif  // SEARCH_HINT_H
