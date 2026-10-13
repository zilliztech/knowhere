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
    AssignWithDistance(const DataSet& dataset, const Config& cfg) override {
        if (throw_on_extended_call) {
            throw std::runtime_error("assignment failed");
        }
        return Assign(dataset, cfg);
    }

    expected<CompactionResult>
    BuildCompactionPlan(const std::vector<uint64_t>& counts, const Config& cfg) override {
        if (throw_on_extended_call) {
            throw std::runtime_error("planning failed");
        }
        batch_size = static_cast<const ClusterApiConfig&>(cfg).batch_size.value();
        CompactionResult plan;
        plan.centroid_count = counts.size();
        plan.centroid_counts = counts;
        for (uint32_t i = 0; i < counts.size(); ++i) {
            plan.row_count += counts[i];
            if (counts[i] != 0) {
                plan.centroid_groups.push_back({static_cast<uint32_t>(plan.centroid_groups.size()), counts[i], {i}});
            }
        }
        return plan;
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
    bool throw_on_extended_call = false;
};
}  // namespace

TEST_CASE("Cluster extensions preserve legacy assignment", "[cluster_api]") {
    Cluster<ClusterNode> cluster = Cluster<LegacyClusterNode>::Create();
    auto* node = static_cast<LegacyClusterNode*>(cluster.Node());
    DataSet input;

    auto result = cluster.Assign(input, Json::object());
    REQUIRE(result.has_value());
    REQUIRE(result.value() == node->result);
    REQUIRE(node->input == &input);
    REQUIRE(cluster.Assign(input).has_value());
    REQUIRE(cluster.BuildCompactionPlan({7, 3}, Json::object()).error() == Status::not_implemented);
    REQUIRE(cluster.AssignWithDistance(input, Json::object()).error() == Status::not_implemented);
    REQUIRE(cluster.SetCentroids(input) == Status::not_implemented);
}

TEST_CASE("Cluster extensions validate config before dispatch", "[cluster_api]") {
    Cluster<ClusterNode> cluster = Cluster<ConfigurableClusterNode>::Create();
    auto* node = static_cast<ConfigurableClusterNode*>(cluster.Node());
    DataSet input;

    SECTION("assignment receives config and dataset") {
        auto result = cluster.Assign(input, {{"batch_size", 17}});
        REQUIRE(result.has_value());
        REQUIRE(result.value() == node->result);
        REQUIRE(node->batch_size == 17);
        REQUIRE(node->input == &input);
    }

    SECTION("distance assignment receives config and dataset") {
        auto result = cluster.AssignWithDistance(input, {{"batch_size", 19}});
        REQUIRE(result.has_value());
        REQUIRE(result.value() == node->result);
        REQUIRE(node->batch_size == 19);
        REQUIRE(node->input == &input);
    }

    SECTION("plan receives counts and config and owns its typed fields") {
        const uint64_t large_count = uint64_t{1} << 33;
        std::vector<uint64_t> counts = {large_count, 0, 3};
        auto result = cluster.BuildCompactionPlan(counts, {{"batch_size", 23}});
        REQUIRE(result.has_value());
        REQUIRE(node->batch_size == 23);
        counts[0] = 1;
        const auto& plan = result.value();
        REQUIRE(plan.row_count == large_count + 3);
        REQUIRE(plan.centroid_count == 3);
        REQUIRE(plan.centroid_counts == std::vector<uint64_t>{large_count, 0, 3});
        REQUIRE(plan.centroid_groups.size() == 2);
        REQUIRE(plan.centroid_groups[0].centroid_group_id == 0);
        REQUIRE(plan.centroid_groups[0].rows == large_count);
        REQUIRE(plan.centroid_groups[0].centroids == std::vector<uint32_t>{0});
        REQUIRE(plan.centroid_groups[1].centroid_group_id == 1);
        REQUIRE(plan.centroid_groups[1].rows == 3);
        REQUIRE(plan.centroid_groups[1].centroids == std::vector<uint32_t>{2});
    }

    SECTION("invalid config never reaches the node") {
        REQUIRE(cluster.Assign(input, {{"batch_size", 0}}).error() == Status::out_of_range_in_json);
        REQUIRE(cluster.BuildCompactionPlan({7, 3}, {{"batch_size", 101}}).error() == Status::out_of_range_in_json);
        REQUIRE(cluster.AssignWithDistance(input, {{"batch_size", 0}}).error() == Status::out_of_range_in_json);
        REQUIRE(node->batch_size == 0);
        REQUIRE(node->input == nullptr);
    }

    SECTION("extended facade methods contain implementation exceptions") {
        node->throw_on_extended_call = true;
        REQUIRE(cluster.BuildCompactionPlan({7, 3}, Json::object()).error() == Status::knowhere_inner_error);
        REQUIRE(cluster.AssignWithDistance(input, Json::object()).error() == Status::knowhere_inner_error);
    }
}

TEST_CASE("SetCentroids forwards input and contains exceptions", "[cluster_api]") {
    Cluster<ClusterNode> cluster = Cluster<ConfigurableClusterNode>::Create();
    auto* node = static_cast<ConfigurableClusterNode*>(cluster.Node());
    DataSet input;

    REQUIRE(cluster.SetCentroids(input) == Status::success);
    REQUIRE(node->input == &input);
    node->throw_on_set = true;
    REQUIRE(cluster.SetCentroids(input) == Status::knowhere_inner_error);
}

TEST_CASE("Search hints survive a DataSet metadata round trip", "[cluster_api]") {
    DataSet dataset;
    dataset.SetSearchHints({QuerySearchHints{{{11, 7, 0.5f}}}, QuerySearchHints{{{29, 3, 1.0f}, {41, 2, 2.5f}}}});

    const auto result = dataset.GetSearchHints();
    REQUIRE(result != nullptr);
    REQUIRE(result->size() == 2);
    REQUIRE(result->at(0)->ranges.size() == 1);
    REQUIRE(result->at(1)->ranges.size() == 2);
    REQUIRE(result->at(0)->ranges[0].offset == 11);
    REQUIRE(result->at(0)->ranges[0].count == 7);
    REQUIRE(result->at(0)->ranges[0].id_distance == 0.5f);
    REQUIRE(result->at(1)->ranges[1].offset == 41);
    REQUIRE(result->at(1)->ranges[1].count == 2);
    REQUIRE(result->at(1)->ranges[1].id_distance == 2.5f);
}
