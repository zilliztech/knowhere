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

#include "catch2/catch_test_macros.hpp"
#include "knowhere/binaryset.h"

TEST_CASE("Test binaryset", "[binaryset]") {
    SECTION("check binaryset interfaces") {
        knowhere::BinarySet binary_set;
        std::shared_ptr<uint8_t[]> sp1(new uint8_t[1]{1});
        binary_set.Append("test", sp1, 1);
        REQUIRE(binary_set.Contains("test"));
        auto binary_get = binary_set.GetByName("test");
        REQUIRE(binary_get->size == 1);
        REQUIRE(binary_get->data != nullptr);
        REQUIRE(binary_set.Size() == 1);

        std::shared_ptr<uint8_t[]> sp2(new uint8_t[2]{1, 2});
        binary_set.Append("test2", sp2, 2);
        REQUIRE(binary_set.Contains("test2"));
        binary_get = binary_set.GetByName("test2");
        REQUIRE(binary_get->size == 2);
        REQUIRE(binary_get->data != nullptr);
        REQUIRE(binary_set.Size() == 3);
    }
}
