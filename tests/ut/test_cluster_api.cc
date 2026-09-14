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

#include <stdexcept>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/cluster/cluster.h"
#include "knowhere/comp/search_hint.h"

namespace {
using namespace knowhere;

class ClusterApiConfig : public Config {
 public:
    CFG_INT batch_size;
    KNOWHERE_DECLARE_CONFIG(ClusterApiConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(batch_size).set_default(8).set_range(1, 100).for_cluster();
    }
};

class LegacyClusterNode : public ClusterNode {
 public:
    expected<DataSetPtr>
    Train(const DataSet&, const Config&) override {
        return result;
    }

    expected<DataSetPtr>
    Assign(const DataSet& dataset) override {
        input = &dataset;
        return result;
    }

    expected<DataSetPtr>
    GetCentroids() const override {
        return result;
    }

    std::unique_ptr<Config>
    CreateConfig() const override {
        return std::make_unique<ClusterApiConfig>();
    }

    std::string
    Type() const override {
        return "test_cluster_api";
    }

    const DataSet* input = nullptr;
    DataSetPtr result = std::make_shared<DataSet>();
};

class ConfigurableClusterNode : public LegacyClusterNode {
 public:
    using LegacyClusterNode::Assign;

    expected<DataSetPtr>
    Assign(const DataSet& dataset, const Config& cfg) override {
        batch_size = static_cast<const ClusterApiConfig&>(cfg).batch_size.value();
        return LegacyClusterNode::Assign(dataset);
    }

    expected<DataSetPtr>
    BuildCompactionPlan(const DataSet& dataset, const Config& cfg) override {
        return Assign(dataset, cfg);
    }

    Status
    SetCentroids(const DataSet& dataset) override {
        if (throw_on_set) {
            throw std::runtime_error("centroid import failed");
        }
        input = &dataset;
        return Status::success;
    }

    int batch_size = 0;
    bool throw_on_set = false;
};
}  // namespace

TEST_CASE("Cluster extensions preserve legacy assignment", "[cluster_api]") {
    Cluster<ClusterNode> cluster = Cluster<LegacyClusterNode>::Create();
    auto* node = static_cast<LegacyClusterNode*>(cluster.Node());
    DataSet input;

    auto result = cluster.Assign(input, Json::object());
    REQUIRE(result.has_value());
    CHECK(result.value() == node->result);
    CHECK(node->input == &input);
    CHECK(cluster.Assign(input).has_value());
    CHECK(cluster.BuildCompactionPlan(input, Json::object()).error() == Status::not_implemented);
    CHECK(cluster.SetCentroids(input) == Status::not_implemented);
}

TEST_CASE("Cluster extensions validate config before dispatch", "[cluster_api]") {
    Cluster<ClusterNode> cluster = Cluster<ConfigurableClusterNode>::Create();
    auto* node = static_cast<ConfigurableClusterNode*>(cluster.Node());
    DataSet input;

    SECTION("assignment receives config and dataset") {
        auto result = cluster.Assign(input, {{"batch_size", 17}});
        REQUIRE(result.has_value());
        CHECK(result.value() == node->result);
        CHECK(node->batch_size == 17);
        CHECK(node->input == &input);
    }

    SECTION("plan receives config and dataset") {
        auto result = cluster.BuildCompactionPlan(input, {{"batch_size", 23}});
        REQUIRE(result.has_value());
        CHECK(result.value() == node->result);
        CHECK(node->batch_size == 23);
        CHECK(node->input == &input);
    }

    SECTION("invalid config never reaches the node") {
        CHECK(cluster.Assign(input, {{"batch_size", 0}}).error() == Status::out_of_range_in_json);
        CHECK(cluster.BuildCompactionPlan(input, {{"batch_size", 101}}).error() == Status::out_of_range_in_json);
        CHECK(node->input == nullptr);
    }
}

TEST_CASE("SetCentroids forwards input and contains exceptions", "[cluster_api]") {
    Cluster<ClusterNode> cluster = Cluster<ConfigurableClusterNode>::Create();
    auto* node = static_cast<ConfigurableClusterNode*>(cluster.Node());
    DataSet input;

    REQUIRE(cluster.SetCentroids(input) == Status::success);
    CHECK(node->input == &input);
    node->throw_on_set = true;
    CHECK(cluster.SetCentroids(input) == Status::knowhere_inner_error);
}

TEST_CASE("Search hints survive a DataSet metadata round trip", "[cluster_api]") {
    using Hints = std::vector<std::vector<SearchHint>>;
    DataSet dataset;
    dataset.Set(kSearchHintsField, Hints{{{11, 7, 0.5f}}, {{29, 3, 1.0f}, {41, 2, 2.0f}}});

    const auto result = dataset.Get<Hints>(kSearchHintsField);
    REQUIRE(result.size() == 2);
    REQUIRE(result[0].size() == 1);
    REQUIRE(result[1].size() == 2);
    CHECK(result[0][0].id_offset == 11);
    CHECK(result[0][0].id_range == 7);
    CHECK(result[0][0].id_distance == 0.5f);
    CHECK(result[1][1].id_offset == 41);
    CHECK(result[1][1].id_range == 2);
    CHECK(result[1][1].id_distance == 2.0f);
}
