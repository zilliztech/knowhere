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
CreateMRLIndex(const float* base_data, bool with_refine,
               const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP) {
    auto base_index = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(index_type, knowhere::Version::GetDefaultVersion().VersionNumber())
                          .value();
    knowhere::ViewDataScopeOp view_data_scope = [base_data](milvus::OpContext*) {
        return [base_data](size_t id) { return base_data + id * kSourceDim; };
    };
    return knowhere::CreateMRLIndex(std::move(base_index), kSourceDim, kMRLDim, knowhere::DataFormatEnum::fp32,
                                    with_refine, std::move(view_data_scope));
}
}  // namespace

TEST_CASE("MRL index searches the prefix and optionally reranks with source vectors", "[mrl]") {
    float base_data[] = {0.0f, 0.0f, 100.0f, 100.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    float query_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto base = knowhere::GenDataSet(2, kSourceDim, base_data);
    auto query = knowhere::GenDataSet(1, kSourceDim, query_data);
    milvus::OpContext op_context;
    knowhere::Json config = {
        {knowhere::meta::DIM, kMRLDim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, 2},
    };

    SECTION("prefix search") {
        auto index = CreateMRLIndex(base_data, false);
        REQUIRE(index.Build(base, config, false, {}, &op_context) == knowhere::Status::success);
        auto result = index.Search(query, config, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] == 0);
    }

    SECTION("full-dimensional rerank") {
        auto index = CreateMRLIndex(base_data, true);
        REQUIRE(index.Build(base, config, false, {}, &op_context) == knowhere::Status::success);
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

TEST_CASE("MRL reports unsupported iterator-based search features", "[mrl]") {
    float base_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float query_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto index = CreateMRLIndex(base_data, false);
    auto query = knowhere::GenDataSet(1, kSourceDim, query_data);
    knowhere::Json config = {{knowhere::meta::METRIC_TYPE, knowhere::metric::L2}};

    auto result = index.AnnIterator(query, config, nullptr);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.what() ==
            "MRL does not support iterator-based searches, including group-by and iterative filtering");
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

TEST_CASE("MRL search maps compact nullable ids after optional refinement", "[mrl]") {
    float base_data[] = {0.0f, 0.0f, 100.0f, 100.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    float raw_data[] = {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 100.0f, 100.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    float query_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int32_t valid_ids[] = {1, 2};
    auto base = knowhere::GenDataSet(2, kSourceDim, base_data);
    base->SetIdMapData(knowhere::IdMapData::FromIds(valid_ids, 2, 3));
    auto query = knowhere::GenDataSet(1, kSourceDim, query_data);
    knowhere::Json config = {
        {knowhere::meta::DIM, kMRLDim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, 2},
    };

    SECTION("prefix search") {
        auto index = CreateMRLIndex(raw_data, false);
        REQUIRE(index.Build(base, config, false) == knowhere::Status::success);
        auto result = index.Search(query, config, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] == 1);
    }

    SECTION("full-dimensional rerank") {
        auto index = CreateMRLIndex(raw_data, true);
        REQUIRE(index.Build(base, config, false) == knowhere::Status::success);
        auto result = index.Search(query, config, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] == 2);
    }
}

TEST_CASE("MRL refinement creates one raw-data scope per search", "[mrl]") {
    float base_data[] = {0.0f, 0.0f, 100.0f, 100.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    float query_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto base = knowhere::GenDataSet(2, kSourceDim, base_data);
    auto query = knowhere::GenDataSet(1, kSourceDim, query_data);
    knowhere::Json config = {
        {knowhere::meta::DIM, kMRLDim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, 2},
    };

    auto active_scopes = std::make_shared<int>(0);
    auto created_scopes = std::make_shared<int>(0);
    knowhere::ViewDataScopeOp view_data_scope = [base_data, active_scopes, created_scopes](milvus::OpContext*) {
        struct ScopeGuard {
            explicit ScopeGuard(int* active) : active(active) {
                ++*active;
            }
            ~ScopeGuard() {
                --*active;
            }
            int* active;
        };
        ++*created_scopes;
        auto guard = std::make_shared<ScopeGuard>(active_scopes.get());
        return [base_data, guard = std::move(guard)](size_t id) { return base_data + id * kSourceDim; };
    };

    auto base_index = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP,
                                                  knowhere::Version::GetDefaultVersion().VersionNumber())
                          .value();
    auto index = knowhere::CreateMRLIndex(std::move(base_index), kSourceDim, kMRLDim, knowhere::DataFormatEnum::fp32,
                                          true, std::move(view_data_scope));
    REQUIRE(index.Build(base, config, false) == knowhere::Status::success);

    REQUIRE(index.Search(query, config, nullptr).has_value());
    REQUIRE(*created_scopes == 1);
    REQUIRE(*active_scopes == 0);
    REQUIRE(index.Search(query, config, nullptr).has_value());
    REQUIRE(*created_scopes == 2);
    REQUIRE(*active_scopes == 0);
}

TEST_CASE("MRL refinement preserves sentinel ids for fully filtered results", "[mrl]") {
    constexpr int64_t row_count = 10;
    float base_data[row_count * kSourceDim] = {};
    float query_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto base = knowhere::GenDataSet(row_count, kSourceDim, base_data);
    auto query = knowhere::GenDataSet(1, kSourceDim, query_data);
    knowhere::Json config = {
        {knowhere::meta::DIM, kMRLDim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, 2},
        {knowhere::indexparam::HNSW_M, 8},
        {knowhere::indexparam::EFCONSTRUCTION, 16},
        {knowhere::indexparam::EF, 16},
    };

    auto index = CreateMRLIndex(base_data, true, knowhere::IndexEnum::INDEX_HNSW);
    REQUIRE(index.Build(base, config, false) == knowhere::Status::success);
    uint8_t bitset_data[]{0xff, 0x03};
    knowhere::BitsetView bitset(bitset_data, row_count);
    auto result = index.Search(query, config, bitset);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetIds()[0] == -1);
    REQUIRE(result.value()->GetIds()[1] == -1);
    auto node_config = index.Node()->CreateConfig();
    static_cast<knowhere::BaseConfig&>(*node_config).k.reset();
    REQUIRE_FALSE(index.Node()->Search(query, std::move(node_config), bitset, nullptr).has_value());
}
