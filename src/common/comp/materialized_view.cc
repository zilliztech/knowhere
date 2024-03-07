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

#include "knowhere/comp/materialized_view.h"

namespace knowhere {

constexpr std::string_view kFieldIdToTouchedCategoriesCntKey = "field_id_to_touched_categories_cnt";
constexpr std::string_view kIsPureAndKey = "is_pure_and";
constexpr std::string_view kHasNotKey = "has_not";

void
to_json(nlohmann::json& j, const MaterializedViewSearchInfo& info) {
    j = nlohmann::json{{kFieldIdToTouchedCategoriesCntKey, info.field_id_to_touched_categories_cnt},
                       {kIsPureAndKey, info.is_pure_and},
                       {kHasNotKey, info.has_not}};
}

void
from_json(const nlohmann::json& j, MaterializedViewSearchInfo& info) {
    if (j.is_null()) {
        // When the json is null, we return the default value of struct MaterializedViewSearchInfo
        // If `MaterializedViewSearchInfo = j[xxx]` is called, a default constructed MaterializedViewSearchInfo will
        // be created. Therefore the second parameter `info` here should have default values.
        return;
    }

    // if any of the keys is missing, the corresponding field in `info` will have default value
    if (j.contains(kFieldIdToTouchedCategoriesCntKey)) {
        j.at(kFieldIdToTouchedCategoriesCntKey).get_to(info.field_id_to_touched_categories_cnt);
    }
    if (j.contains(kIsPureAndKey)) {
        j.at(kIsPureAndKey).get_to(info.is_pure_and);
    }
    if (j.contains(kHasNotKey)) {
        j.at(kHasNotKey).get_to(info.has_not);
    }
}
}  // namespace knowhere
