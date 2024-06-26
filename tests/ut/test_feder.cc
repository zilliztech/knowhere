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

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/feder/DiskANN.h"
#include "knowhere/feder/HNSW.h"
#include "knowhere/feder/IVFFlat.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "utils.h"

void
CheckHnswMeta(const knowhere::DataSetPtr result, int64_t nb, const knowhere::Json& cfg) {
    auto json_info = result->GetJsonInfo();
    auto json_id_set = result->GetJsonIdSet();
    LOG_KNOWHERE_INFO_ << "json_info size: " << json_info.size();
    LOG_KNOWHERE_INFO_ << "json_id_set size: " << json_id_set.size();

    knowhere::feder::hnsw::HNSWMeta meta;
    knowhere::Json j1 = nlohmann::json::parse(json_info);
    nlohmann::from_json(j1, meta);

    REQUIRE(meta.GetEfConstruction() == cfg[knowhere::indexparam::EFCONSTRUCTION]);
    REQUIRE(meta.GetM() == cfg[knowhere::indexparam::HNSW_M]);
    REQUIRE(meta.GetNumElem() == nb);

    auto& hier_graph = meta.GetOverviewHierGraph();
    for (auto& graph : hier_graph) {
        auto& nodes = graph.GetNodes();
        for (auto& node : nodes) {
            REQUIRE(node.id_ >= 0);
            REQUIRE(node.id_ < nb);
            for (auto n : node.neighbors_) {
                REQUIRE(n >= 0);
                REQUIRE(n < nb);
            }
        }
    }

    // check IDSet
    std::unordered_set<int64_t> id_set;
    knowhere::Json j2 = nlohmann::json::parse(json_id_set);
    nlohmann::from_json(j2, id_set);
    LOG_KNOWHERE_INFO_ << "id_set num: " << id_set.size();
    for (auto id : id_set) {
        REQUIRE(id >= 0);
        REQUIRE(id < nb);
    }
}

void
CheckHnswVisitInfo(const knowhere::DataSetPtr result, int64_t nb) {
    auto json_info = result->GetJsonInfo();
    auto json_id_set = result->GetJsonIdSet();
    LOG_KNOWHERE_INFO_ << "json_info size: " << json_info.size();
    LOG_KNOWHERE_INFO_ << "json_id_set size: " << json_id_set.size();

    // check HNSWVisitInfo
    knowhere::feder::hnsw::HNSWVisitInfo visit_info;
    knowhere::Json j1 = nlohmann::json::parse(json_info);
    nlohmann::from_json(j1, visit_info);

    for (auto& level_visit_record : visit_info.GetInfos()) {
        auto& records = level_visit_record.GetRecords();
        for (auto& record : records) {
            auto id_from = std::get<0>(record);
            auto id_to = std::get<1>(record);
            auto dist = std::get<2>(record);
            REQUIRE(id_from >= 0);
            REQUIRE(id_to >= 0);
            REQUIRE(id_from < nb);
            REQUIRE(id_to < nb);
            REQUIRE((dist >= 0.0 || dist == -1.0));
        }
    }

    // check IDSet
    std::unordered_set<int64_t> id_set;
    knowhere::Json j2 = nlohmann::json::parse(json_id_set);
    nlohmann::from_json(j2, id_set);
    LOG_KNOWHERE_INFO_ << "id_set num: " << id_set.size();
    for (auto id : id_set) {
        REQUIRE(id >= 0);
        REQUIRE(id < nb);
    }
}

void
CheckIvfFlatMeta(const knowhere::DataSetPtr result, int64_t nb, const knowhere::Json& cfg) {
    auto json_info = result->GetJsonInfo();
    auto json_id_set = result->GetJsonIdSet();
    LOG_KNOWHERE_INFO_ << "json_info size: " << json_info.size();
    LOG_KNOWHERE_INFO_ << "json_id_set size: " << json_id_set.size();

    // check IVFFlatMeta
    knowhere::feder::ivfflat::IVFFlatMeta meta;
    knowhere::Json j1 = nlohmann::json::parse(json_info);
    nlohmann::from_json(j1, meta);

    REQUIRE(meta.GetNlist() == cfg[knowhere::indexparam::NLIST]);
    REQUIRE(meta.GetDim() == cfg[knowhere::meta::DIM]);
    REQUIRE(meta.GetNtotal() == nb);

    // sum of all cluster nodes should be equal to nb
    auto& clusters = meta.GetClusters();
    std::unordered_set<int64_t> all_id_set;
    REQUIRE(clusters.size() == cfg[knowhere::indexparam::NLIST]);
    for (auto& cluster : clusters) {
        for (auto id : cluster.node_ids_) {
            REQUIRE(id >= 0);
            REQUIRE(id < nb);
            all_id_set.insert(id);
        }
    }
    REQUIRE(all_id_set.size() == (size_t)nb);

    // check IDSet validation
    std::unordered_set<int64_t> id_set;
    knowhere::Json j2 = nlohmann::json::parse(json_id_set);
    nlohmann::from_json(j2, id_set);
    LOG_KNOWHERE_INFO_ << "id_set num: " << id_set.size();
}

TEST_CASE("Test Feder", "[feder]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 1;
    int64_t dim = 128;
    int64_t seed = 42;

    auto version = GenTestVersionList();

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        json[knowhere::meta::TOPK] = 10;
        json[knowhere::meta::TRACE_VISIT] = true;
        return json;
    };

    auto ivfflat_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 4;
        return json;
    };

    auto hnsw_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 8;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 16;
        json[knowhere::indexparam::OVERVIEW_LEVELS] = 2;
        return json;
    };

    const auto train_ds = GenDataSet(nb, dim, seed);
    const auto query_ds = GenDataSet(nq, dim, seed);

    const knowhere::Json conf = base_gen();
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

    SECTION("Test HNSW Feder") {
        auto name = knowhere::IndexEnum::INDEX_HNSW;
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        REQUIRE(idx.Type() == name);

        auto json = hnsw_gen();
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);

        auto res1 = idx.GetIndexMeta(json);
        REQUIRE(res1.has_value());
        CheckHnswMeta(res1.value(), nb, json);

        auto res2 = idx.Search(query_ds, json, nullptr);
        REQUIRE(res2.has_value());
        CheckHnswVisitInfo(res2.value(), nb);

        json[knowhere::meta::RADIUS] = 160000;
        json[knowhere::meta::RANGE_FILTER] = 0;
        auto res3 = idx.RangeSearch(query_ds, json, nullptr);
        REQUIRE(res3.has_value());
        CheckHnswVisitInfo(res3.value(), nb);
    }

    SECTION("Test IVF_FLAT Feder") {
        auto name = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        REQUIRE(idx.Type() == name);

        auto json = ivfflat_gen();
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);

        auto res1 = idx.GetIndexMeta(json);
        REQUIRE(res1.has_value());
        CheckIvfFlatMeta(res1.value(), nb, json);
    }
}
