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

#include <unordered_set>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/index/index_factory.h"

TEST_CASE("Test index and data type check", "[IndexCheckTest]") {
    SECTION("Test valid") {
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_HNSW,
                                                                   knowhere::VecType::VECTOR_FLOAT) == true);
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_HNSW,
                                                                   knowhere::VecType::VECTOR_BFLOAT16) == true);

#ifndef KNOWHERE_WITH_CARDINAL
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_HNSW,
                                                                   knowhere::VecType::VECTOR_BINARY) == false);
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_HNSW,
                                                                   knowhere::VecType::VECTOR_SPARSE_FLOAT) == false);
#else
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_HNSW,
                                                                   knowhere::VecType::VECTOR_BINARY) == true);
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_HNSW,
                                                                   knowhere::VecType::VECTOR_SPARSE_FLOAT) == true);
#endif
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_DISKANN,
                                                                   knowhere::VecType::VECTOR_FLOAT) == true);
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_DISKANN,
                                                                   knowhere::VecType::VECTOR_FLOAT16) == true);
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_DISKANN,
                                                                   knowhere::VecType::VECTOR_BINARY) == false);
        REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck(knowhere::IndexEnum::INDEX_DISKANN,
                                                                   knowhere::VecType::VECTOR_SPARSE_FLOAT) == false);
    }
}
