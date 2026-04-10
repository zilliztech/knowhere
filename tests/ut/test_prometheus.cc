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

#include <iostream>
#include <string>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/prometheus_client.h"

TEST_CASE("Test prometheus client", "[prometheus client]") {
    SECTION("check get metrics") {
        auto str = knowhere::prometheusClient->GetMetrics();
        std::cout << str << std::endl;
        CHECK(str.length() >= 0);
    }

    SECTION("check index type latency labels") {
        knowhere::ObserveSearchLatencyByIndexType("knowhere", knowhere::IndexEnum::INDEX_FAISS_IDMAP, 12.0);
        knowhere::ObserveBuildLatencyByIndexType("knowhere", knowhere::IndexEnum::INDEX_HNSW, 1.5);

        auto str = knowhere::prometheusClient->GetMetrics();
        CHECK(str.find("search_latency_bucket{index_type=\"FLAT\",module=\"knowhere\"") != std::string::npos);
        CHECK(str.find("build_latency_bucket{index_type=\"HNSW\",module=\"knowhere\"") != std::string::npos);
        CHECK(str.find("index_type=\"FLAT\"") != std::string::npos);
        CHECK(str.find("index_type=\"HNSW\"") != std::string::npos);
    }
}
