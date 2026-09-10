// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/mrl_index_node.h"

namespace {
constexpr int64_t kSourceDim = 4;
constexpr int64_t kMRLDim = 2;

knowhere::Index<knowhere::IndexNode>
CreateMRLIndex(const float* base_data, bool with_refine) {
    auto base_index = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP,
                                                  knowhere::Version::GetDefaultVersion().VersionNumber())
                          .value();
    knowhere::ViewDataOp view_data = [base_data](size_t id) { return base_data + id * kSourceDim; };
    return knowhere::CreateMRLIndex(std::move(base_index), kSourceDim, kMRLDim, knowhere::DataFormatEnum::fp32,
                                    with_refine, std::move(view_data));
}
}  // namespace

TEST_CASE("MRL index searches the prefix and optionally reranks with source vectors", "[mrl]") {
    float base_data[] = {0.0f, 0.0f, 100.0f, 100.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    float query_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto base = knowhere::GenDataSet(2, kSourceDim, base_data);
    auto query = knowhere::GenDataSet(1, kSourceDim, query_data);
    knowhere::Json config = {
        {knowhere::meta::DIM, kMRLDim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, 2},
    };

    SECTION("prefix search") {
        auto index = CreateMRLIndex(base_data, false);
        REQUIRE(index.Build(base, config, false) == knowhere::Status::success);
        auto result = index.Search(query, config, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] == 0);
    }

    SECTION("full-dimensional rerank") {
        auto index = CreateMRLIndex(base_data, true);
        REQUIRE(index.Build(base, config, false) == knowhere::Status::success);
        auto result = index.Search(query, config, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] == 1);

        knowhere::BinarySet binary_set;
        REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
        auto loaded = CreateMRLIndex(base_data, true);
        REQUIRE(loaded.Deserialize(binary_set, config) == knowhere::Status::success);
        auto loaded_result = loaded.Search(query, config, nullptr);
        REQUIRE(loaded_result.has_value());
        REQUIRE(loaded_result.value()->GetIds()[0] == 1);
    }
}

TEST_CASE("MRL cosine refinement preserves full-dimensional norms", "[mrl]") {
    float base_data[] = {1.0f, 0.0f, 10.0f, 0.0f, 0.8f, 0.6f, 0.0f, 0.0f};
    float query_data[] = {1.0f, 0.0f, 0.0f, 0.0f};
    auto base = knowhere::GenDataSet(2, kSourceDim, base_data);
    auto query = knowhere::GenDataSet(1, kSourceDim, query_data);
    knowhere::Json config = {
        {knowhere::meta::DIM, kMRLDim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::COSINE},
        {knowhere::meta::TOPK, 2},
    };

    auto index = CreateMRLIndex(base_data, true);
    REQUIRE(index.Build(base, config, false) == knowhere::Status::success);
    auto result = index.Search(query, config, nullptr);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetIds()[0] == 1);

    knowhere::BinarySet binary_set;
    REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
    auto loaded = CreateMRLIndex(base_data, true);
    REQUIRE(loaded.Deserialize(binary_set, config) == knowhere::Status::success);
    auto loaded_result = loaded.Search(query, config, nullptr);
    REQUIRE(loaded_result.has_value());
    REQUIRE(loaded_result.value()->GetIds()[0] == 1);
}
