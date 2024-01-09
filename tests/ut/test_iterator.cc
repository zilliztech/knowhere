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
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.6f;
constexpr float kBruteForceRecallThreshold = 0.99f;

knowhere::DataSetPtr
GetKNNResult(const std::vector<std::shared_ptr<knowhere::IndexNode::iterator>>& iterators, int k,
             const knowhere::BitsetView* bitset = nullptr) {
    int nq = iterators.size();
    auto p_id = new int64_t[nq * k];
    auto p_dist = new float[nq * k];
    for (int i = 0; i < nq; ++i) {
        auto& iter = iterators[i];
        for (int j = 0; j < k; ++j) {
            REQUIRE(iter->HasNext());
            auto [id, dist] = iter->Next();
            // if bitset is provided, verify we don't return filtered out points.
            REQUIRE((!bitset || !bitset->test(id)));
            p_id[i * k + j] = id;
            p_dist[i * k + j] = dist;
        }
    }
    return knowhere::GenResultDataSet(nq, k, p_id, p_dist);
}
}  // namespace

// use kNN search to test the correctness of iterator
TEST_CASE("Test Iterator Mem Index With Float Vector", "[float metrics]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;
    auto topk = GENERATE(5, 10, 20);

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    auto hnsw_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 128;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::SEED_EF] = 64;
        return json;
    };

    auto rand = GENERATE(1, 2, 3, 5);

    const auto train_ds = GenDataSet(nb, dim, rand);
    const auto query_ds = GenDataSet(nq, dim, rand + 777);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

    SECTION("Test Search using iterator") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
        REQUIRE(idx.Deserialize(bs) == knowhere::Status::success);
        auto its = idx.AnnIterator(*query_ds, json, nullptr);
        REQUIRE(its.has_value());
        auto results = GetKNNResult(its.value(), topk);
        float recall = GetKNNRecall(*gt.value(), *results);
        REQUIRE(recall > kKnnRecallThreshold);
    }

    SECTION("Test Search with Bitset using iterator") {
        using std::make_tuple;
        auto [name, gen, threshold] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>, float>({
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen, hnswlib::kHnswSearchKnnBFFilterThreshold),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);

        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
            GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
        const auto bitset_percentages = {0.4f, 0.98f};
        for (const float percentage : bitset_percentages) {
            for (const auto& gen_func : gen_bitset_funcs) {
                auto bitset_data = gen_func(nb, percentage * nb);
                knowhere::BitsetView bitset(bitset_data.data(), nb);
                // Iterator doesn't have a fallback to bruteforce mechanism at high filter rate.
                auto its = idx.AnnIterator(*query_ds, json, bitset);
                REQUIRE(its.has_value());
                auto results = GetKNNResult(its.value(), topk, &bitset);
                auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, bitset);
                float recall = GetKNNRecall(*gt.value(), *results);
                REQUIRE(recall > kKnnRecallThreshold);
            }
        }
    }

    SECTION("Test Search with Bitset using iterator insufficient results") {
        using std::make_tuple;
        auto [name, gen, threshold] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>, float>({
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen, hnswlib::kHnswSearchKnnBFFilterThreshold),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);

        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
            GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
        // we want topk but after filtering we have only half of topk points.
        const auto filter_remaining = topk / 2;
        for (const auto& gen_func : gen_bitset_funcs) {
            auto bitset_data = gen_func(nb, nb - filter_remaining);
            knowhere::BitsetView bitset(bitset_data.data(), nb);
            auto its = idx.AnnIterator(*query_ds, json, bitset);
            REQUIRE(its.has_value());
            auto results = GetKNNResult(its.value(), filter_remaining, &bitset);
            // after get those remaining points, iterator should return false for HasNext.
            for (const auto& it : its.value()) {
                REQUIRE(!it->HasNext());
            }
            auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, bitset);
            float recall = GetKNNRecall(*gt.value(), *results);
            REQUIRE(recall > kKnnRecallThreshold);
        }
    }
}

TEST_CASE("Test Iterator Mem Index With Binary Vector", "[float metrics]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 1024;
    const int64_t topk = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::HAMMING, knowhere::metric::JACCARD);
    auto version = GenTestVersionList();

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    auto hnsw_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 128;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::SEED_EF] = 64;
        return json;
    };
    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };

    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
    SECTION("Test Search using iterator") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto its = idx.AnnIterator(*query_ds, json, nullptr);
        REQUIRE(its.has_value());
        auto results = GetKNNResult(its.value(), topk);
        float recall = GetKNNRecall(*gt.value(), *results);
        REQUIRE(recall > kKnnRecallThreshold);
    }
}
