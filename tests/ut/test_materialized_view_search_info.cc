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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "knowhere/comp/materialized_view.h"
#include "knowhere/config.h"

constexpr size_t kFieldIdToTouchedCategoriesDefaultSize = 0;
constexpr bool kIsPureAndDefault = true;
constexpr bool kHasNotDefault = false;

TEST_CASE("MaterializedViewSearchInfo", "[comp]") {
    SECTION("Serialize and Deserialize") {
        knowhere::MaterializedViewSearchInfo info;
        info.field_id_to_touched_categories_cnt[1] = 2;
        info.field_id_to_touched_categories_cnt[3] = 4;
        info.is_pure_and = false;
        info.has_not = true;

        // automatically calls to_json and from_json
        knowhere::Json j = info;
        auto info2 = j.get<knowhere::MaterializedViewSearchInfo>();
        REQUIRE(info.field_id_to_touched_categories_cnt == info2.field_id_to_touched_categories_cnt);
        REQUIRE(info.is_pure_and == info2.is_pure_and);
        REQUIRE(info.has_not == info2.has_not);
    }

    auto RequireDefaultVals = [](const knowhere::MaterializedViewSearchInfo& info) {
        REQUIRE(info.field_id_to_touched_categories_cnt.size() == kFieldIdToTouchedCategoriesDefaultSize);
        REQUIRE(info.is_pure_and == kIsPureAndDefault);
        REQUIRE(info.has_not == kHasNotDefault);
    };

    SECTION("Null") {
        knowhere::Json j;
        auto info = j.get<knowhere::MaterializedViewSearchInfo>();
        RequireDefaultVals(info);
    }

    SECTION("Nullptr") {
        knowhere::Json j = nullptr;

        REQUIRE(j.is_null() == true);  // Json(nullptr) same as null

        auto info = j.get<knowhere::MaterializedViewSearchInfo>();
        RequireDefaultVals(info);
    }

    SECTION("Empty json should be treated as null") {
        knowhere::Json j = GENERATE("", "{}", "[]");
        auto info = j.get<knowhere::MaterializedViewSearchInfo>();
        RequireDefaultVals(info);
    }

    SECTION("Json not involve MaterializedViewSearchInfo keys should return defaults") {
        knowhere::Json j = GENERATE(R"("a": [])", R"("a": {})"
                                                  R"("a": {"has_not": false})"
                                                  R"({"xhas_not": false})"
                                                  R"({"has_not": 123})");
        auto info = j.get<knowhere::MaterializedViewSearchInfo>();
        RequireDefaultVals(info);
    }

    SECTION("Partial keys should return defaults for those not involved") {
        SECTION("test field_id_to_touched_categories_cnt") {
            SECTION("from string") {
                // JSON only allows key names to be strings
                // Use arrays to serialize STL unordered_map is pure implementation inside nlohmann json
                auto str = R"(
                  {
                    "field_id_to_touched_categories_cnt": [[1,2]]
                  }
                )";
                knowhere::Json j = knowhere::Json::parse(str);
                auto info = j.get<knowhere::MaterializedViewSearchInfo>();
                REQUIRE(info.field_id_to_touched_categories_cnt.size() == 1);
                REQUIRE(info.field_id_to_touched_categories_cnt[1] == 2);
                REQUIRE(info.is_pure_and == kIsPureAndDefault);
                REQUIRE(info.has_not == kHasNotDefault);
            }

            SECTION("from stl map") {
                decltype(knowhere::MaterializedViewSearchInfo::field_id_to_touched_categories_cnt) expected = {{3, 4}};
                knowhere::Json j;
                j["field_id_to_touched_categories_cnt"] = expected;
                auto info = j.get<knowhere::MaterializedViewSearchInfo>();
                REQUIRE(info.field_id_to_touched_categories_cnt.size() == 1);
                REQUIRE(info.field_id_to_touched_categories_cnt[3] == expected[3]);
                REQUIRE(info.is_pure_and == kIsPureAndDefault);
                REQUIRE(info.has_not == kHasNotDefault);
            }
        }

        SECTION("test is_pure_and") {
            SECTION("from string") {
                auto str = R"(
                  {
                    "is_pure_and": false
                  }
                )";
                knowhere::Json j = knowhere::Json::parse(str);
                auto info = j.get<knowhere::MaterializedViewSearchInfo>();
                REQUIRE(info.field_id_to_touched_categories_cnt.size() == kFieldIdToTouchedCategoriesDefaultSize);
                REQUIRE(info.is_pure_and == false);
                REQUIRE(info.has_not == kHasNotDefault);
            }

            SECTION("from bool") {
                knowhere::Json j;
                j["is_pure_and"] = false;
                auto info = j.get<knowhere::MaterializedViewSearchInfo>();
                REQUIRE(info.field_id_to_touched_categories_cnt.size() == kFieldIdToTouchedCategoriesDefaultSize);
                REQUIRE(info.is_pure_and == false);
                REQUIRE(info.has_not == kHasNotDefault);
            }
        }

        SECTION("test has_not") {
            SECTION("from string") {
                auto str = R"(
                  {
                    "has_not": true
                  }
                )";
                knowhere::Json j = knowhere::Json::parse(str);
                auto info = j.get<knowhere::MaterializedViewSearchInfo>();
                REQUIRE(info.field_id_to_touched_categories_cnt.size() == kFieldIdToTouchedCategoriesDefaultSize);
                REQUIRE(info.is_pure_and == kIsPureAndDefault);
                REQUIRE(info.has_not == true);
            }

            SECTION("from bool") {
                knowhere::Json j;
                j["has_not"] = true;
                auto info = j.get<knowhere::MaterializedViewSearchInfo>();
                REQUIRE(info.field_id_to_touched_categories_cnt.size() == kFieldIdToTouchedCategoriesDefaultSize);
                REQUIRE(info.is_pure_and == kIsPureAndDefault);
                REQUIRE(info.has_not == true);
            }
        }
    }
}
