// Copyright (C) 2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/search_hint.h"
#include "knowhere/dataset.h"

TEST_CASE("Typed DataSet search hints own a stable immutable snapshot", "[search_hint]") {
    knowhere::DataSet dataset;
    REQUIRE(dataset.GetSearchHints() == nullptr);
    knowhere::BatchSearchHints hints = {knowhere::QuerySearchHints{{{5, 2, 1.5f}}}};
    dataset.SetSearchHints(hints);
    hints[0]->ranges[0].offset = 1;
    hints[0]->ranges[0].id_distance = 9.0f;
    auto snapshot = dataset.GetSearchHints();
    REQUIRE(dataset.Get<std::shared_ptr<const knowhere::BatchSearchHints>>(knowhere::kSearchHintsField) == snapshot);
    REQUIRE(snapshot->at(0)->ranges[0].offset == 5);
    REQUIRE(snapshot->at(0)->ranges[0].id_distance == 1.5f);
    dataset.ClearSearchHints();
    REQUIRE(dataset.GetSearchHints() == nullptr);
    REQUIRE(snapshot->at(0)->ranges[0].offset == 5);
    REQUIRE(snapshot->at(0)->ranges[0].id_distance == 1.5f);
    dataset.Set(knowhere::kSearchHintsField, 42);
    REQUIRE_THROWS_AS(dataset.GetSearchHints(), std::invalid_argument);
}

TEST_CASE("DataSet search hints use one metadata payload and preserve query states", "[search_hint]") {
    using namespace knowhere;
    const int64_t offset = int64_t{1} << 33;
    auto hints = std::make_shared<const BatchSearchHints>(
        BatchSearchHints{QuerySearchHints{{{offset, 7, 2.5f}}}, QuerySearchHints{}, std::nullopt});
    DataSet dataset;
    dataset.Set(kSearchHintsField, hints);
    const auto snapshot = dataset.GetSearchHints();
    REQUIRE(snapshot == hints);
    REQUIRE(snapshot->size() == 3);
    REQUIRE(snapshot->at(0)->ranges.at(0).offset == offset);
    REQUIRE(snapshot->at(0)->ranges.at(0).count == 7);
    REQUIRE(snapshot->at(0)->ranges.at(0).id_distance == 2.5f);
    REQUIRE(SearchHint{offset, 7}.id_distance == 0.0f);
    REQUIRE(snapshot->at(1).has_value());
    REQUIRE(snapshot->at(1)->ranges.empty());
    REQUIRE_FALSE(snapshot->at(2).has_value());

    dataset.ClearSearchHints();
    REQUIRE(dataset.Get<std::shared_ptr<const BatchSearchHints>>(kSearchHintsField) == nullptr);
    REQUIRE(dataset.GetSearchHints() == nullptr);
    REQUIRE(snapshot->at(0)->ranges.at(0).offset == offset);

    dataset.SetSearchHints({});
    REQUIRE(dataset.GetSearchHints() != nullptr);
    REQUIRE(dataset.GetSearchHints()->empty());
}
