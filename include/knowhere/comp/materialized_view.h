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

#include <charconv>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace knowhere {
// MaterializedViewSearchInfo is used to store the search information when performing filtered search (i.e. Materialized
// View - vectors and scalars).
// This information is obtained from expression analysis only, not runtime, so might be inaccurate.
struct MaterializedViewSearchInfo {
    // describes which scalar field is involved during search,
    // and how many categories are touched
    // for example, if we have scalar field `color` with field id `111` and it has three categories: red, green, blue
    // expression `color == "red"`, yields `111 -> 1`
    // expression `color == "red" && color == "green"`, yields `111 -> 2`
    std::unordered_map<int64_t, uint64_t> field_id_to_touched_categories_cnt;

    // whether the search exression has AND (&&) logical operator only
    bool is_pure_and = true;

    // whether the search expression has NOT (!) logical unary operator
    bool has_not = false;
};

// DO NOT CALL THIS FUNCTION MANUALLY
// use `json j = materialized_view_search_info`
void
to_json(nlohmann::json& j, const MaterializedViewSearchInfo& info);

// DO NOT CALL THIS FUNCTION MANUALLY
// use `auto j = j.get<MaterializedViewSearchInfo>() or j[KEY]`
void
from_json(const nlohmann::json& j, MaterializedViewSearchInfo& info);
}  // namespace knowhere
