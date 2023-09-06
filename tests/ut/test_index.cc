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

#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "knowhere/factory.h"
#include "knowhere/index.h"

TEST_CASE("Test Index null check", "[Index]") {
    SECTION("Test Index") {
        {
            knowhere::Index<knowhere::IndexNode> idx;
            CHECK(idx == nullptr);
        }

        {
            auto idx = knowhere::IndexFactory::Instance().Create(knowhere::IndexEnum::INDEX_HNSW);
            CHECK(!(idx == nullptr));
        }

        {
            auto idx = knowhere::IndexFactory::Instance().Create("MEGA");
            CHECK(idx == nullptr);
        }
    }
}
