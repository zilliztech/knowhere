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
#include "faiss/utils/binary_distances.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "utils.h"
namespace {
constexpr float kKnnRecallThreshold = 0.6f;
constexpr float kBruteForceRecallThreshold = 0.99f;
}  // namespace
template <typename data_type>
void
BaseSearchTest() {
    using Catch::Approx;

    const int64_t nb = 10000, nq = 100;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE, knowhere::metric::IP);
    auto topk = GENERATE(as<int64_t>{}, 10);
    auto version = GenTestVersionList();
    std::cout << "cqy::metric " << metric << std::endl;

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto hnsw_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 30;
        json[knowhere::indexparam::EFCONSTRUCTION] = 120;
        json[knowhere::indexparam::EF] = 36;
        return json;
    };

    const auto fp32_train_ds = GenDataSet(nb, dim);
    const auto fp32_query_ds = GenDataSet(nq, dim);
    auto train_ds = knowhere::data_type_conversion<knowhere::fp32, data_type>(*fp32_train_ds);
    auto query_ds = knowhere::data_type_conversion<knowhere::fp32, data_type>(*fp32_query_ds);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<data_type>(train_ds, query_ds, conf, nullptr);

    SECTION("Test half-float Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<data_type>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        knowhere::TimeRecorder rc("cqy: Build index", 2);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        auto time = rc.ElapseFromBegin("done");

        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
        REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);

        knowhere::TimeRecorder knn_rc("cqy: Search knn index", 2);
        auto results = idx.Search(query_ds, json, nullptr);
        auto knn_time = knn_rc.ElapseFromBegin("done");

        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        std::cout << "cqy: recall " << recall << std::endl;
        if (name != knowhere::IndexEnum::INDEX_FAISS_IVFPQ) {
            REQUIRE(recall > kKnnRecallThreshold);
        }
    }

    // SECTION("Test half-float Range Search") {
    //     using std::make_tuple;
    //     auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
    //         make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
    //     }));
    //     auto idx = knowhere::IndexFactory::Instance().Create<data_type>(name, version).value();
    //     auto cfg_json = gen().dump();
    //     CAPTURE(name, cfg_json);
    //     knowhere::Json json = knowhere::Json::parse(cfg_json);
    //     REQUIRE(idx.Type() == name);
    //     REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    //     knowhere::BinarySet bs;
    //     REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
    //     REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);

    //     knowhere::TimeRecorder range_rc("cqy: Search range search", 2);
    //     auto results = idx.RangeSearch(query_ds, json, nullptr);
    //     auto range_time = range_rc.ElapseFromBegin("done");

    //     REQUIRE(results.has_value());
    //     auto ids = results.value()->GetIds();
    //     auto lims = results.value()->GetLims();
    //     auto dis = results.value()->GetDistance();
    //     if (name != knowhere::IndexEnum::INDEX_FAISS_IVFPQ && name != knowhere::IndexEnum::INDEX_FAISS_SCANN) {
    //         for (int i = 0; i < nq; ++i) {
    //             CHECK(ids[lims[i]] == i);
    //         }
    //     }

    // }

    // SECTION("Test half-float Search with Bitset") {
    //     using std::make_tuple;
    //     auto [name, gen, threshold] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>, float>({
    //         make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen, hnswlib::kHnswSearchKnnBFFilterThreshold),
    //     }));
    //     auto idx = knowhere::IndexFactory::Instance().Create<data_type>(name, version).value();
    //     auto cfg_json = gen().dump();
    //     CAPTURE(name, cfg_json);
    //     knowhere::Json json = knowhere::Json::parse(cfg_json);
    //     REQUIRE(idx.Type() == name);
    //     REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    //     std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
    //         GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
    //     const auto bitset_percentages = {0.4f, 0.98f};
    //     for (const float percentage : bitset_percentages) {
    //         for (const auto& gen_func : gen_bitset_funcs) {
    //             auto bitset_data = gen_func(nb, percentage * nb);
    //             knowhere::BitsetView bitset(bitset_data.data(), nb);
    //             knowhere::TimeRecorder bitset_rc("Search knn search with bitset", 2);
    //             auto results = idx.Search(query_ds, json, bitset);
    //             auto bitset_time = bitset_rc.ElapseFromBegin("done");

    //             auto gt = knowhere::BruteForce::Search<data_type>(train_ds, query_ds, json, bitset);
    //             float recall = GetKNNRecall(*gt.value(), *results.value());
    //             std::cout <<"cqy: recall "<< recall<<std::endl;
    //             if (percentage > threshold) {
    //                 REQUIRE(recall > kBruteForceRecallThreshold);
    //             } else {
    //                 REQUIRE(recall > kKnnRecallThreshold);
    //             }
    //         }
    //     }
    // }
}

TEST_CASE("Test Mem Index With fp16/bf16 Vector", "[float metrics]") {
    std::cout << "cqy: fp32 test" << std::endl;
    BaseSearchTest<knowhere::fp32>();
    std::cout << "cqy: fp16 test" << std::endl;
    BaseSearchTest<knowhere::fp16>();
    std::cout << "cqy: bf16 test" << std::endl;
    BaseSearchTest<knowhere::bf16>();
}
