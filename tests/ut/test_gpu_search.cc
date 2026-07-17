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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

#ifdef KNOWHERE_WITH_CUVS

inline knowhere::DataSetPtr
GenNormalizedInt8DataSet(int rows, int dim, int seed = 42) {
    auto* tensor = new knowhere::int8[rows * dim];
    std::vector<float> row(dim);
    for (int i = 0; i < rows; ++i) {
        double norm = 0.0;
        for (int j = 0; j < dim; ++j) {
            row[j] = std::sin(float((i + seed + 1) * (j + 3))) + std::cos(float((i + seed + 5) * (j + 1)));
            norm += double(row[j]) * row[j];
        }
        norm = std::sqrt(norm);

        auto nonzero = false;
        for (int j = 0; j < dim; ++j) {
            auto scaled = norm == 0.0 ? 0 : int(std::llround(double(row[j]) / norm * 127.0));
            scaled = std::max(-127, std::min(127, scaled));
            tensor[i * dim + j] = knowhere::int8(scaled);
            nonzero = nonzero || scaled != 0;
        }
        if (!nonzero) {
            tensor[i * dim] = knowhere::int8(127);
        }
    }

    auto ds = knowhere::GenDataSet(rows, dim, tensor);
    ds->SetIsOwner(true);
    return ds;
}

template <typename T>
void
check_search(const int64_t nb, const int64_t nq, const int64_t dim, const int64_t seed, std::string name, int version,
             const knowhere::Json& conf, float min_recall = 0.9f) {
    auto train_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(nb, dim, seed));
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(nq, dim, seed + 2));

    if (std::is_same_v<T, knowhere::fp16> &&
        (name == knowhere::IndexEnum::INDEX_GPU_IVFFLAT || name == knowhere::IndexEnum::INDEX_CUVS_IVFFLAT)) {
        // IVF-FLAT FP16 distances become too large, so we normalize the dataset
        // https://github.com/rapidsai/cuvs/issues/914
        knowhere::NormalizeDataset<T>(train_ds);
        knowhere::NormalizeDataset<T>(query_ds);
    }

    auto idx = knowhere::IndexFactory::Instance().Create<T>(name, version).value();

    // 1. Self-search
    auto res = idx.Build(train_ds, conf);
    REQUIRE(res == knowhere::Status::success);
    auto results = idx.Search(train_ds, conf, nullptr);
    REQUIRE(results.has_value());

    auto ids = results.value()->GetIds();
    for (int i = 1; i < nq; ++i) {
        CHECK(ids[i] == i);
    }

    // 2. Search a query dataset

    results = idx.Search(query_ds, conf, nullptr);
    REQUIRE(results.has_value());

    auto gt = knowhere::BruteForce::Search<T>(train_ds, query_ds, conf, nullptr);
    REQUIRE(gt.has_value());

    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall >= min_recall);
}

TEST_CASE("Test All GPU Index", "[search]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 1000;
    int64_t dim = 128;
    int64_t seed = 42;

    auto version = GenTestVersionList();

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        json[knowhere::meta::TOPK] = 1;
        json[knowhere::meta::RADIUS] = 10.0;
        json[knowhere::meta::RANGE_FILTER] = 0.0;
        return json;
    };

    auto bruteforce_gen = base_gen;

    auto ivfflat_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 20;
        json[knowhere::indexparam::NPROBE] = 18;
        return json;
    };

    auto ivfpq_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::M] = 0;
        json[knowhere::indexparam::NBITS] = 8;
        return json;
    };

    auto cagra_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = 64;
        json[knowhere::indexparam::GRAPH_DEGREE] = 32;
        json[knowhere::indexparam::ITOPK_SIZE] = 128;
        return json;
    };

    auto cagra_hnsw_gen = [](auto&& upstream_gen) {
        return [upstream_gen]() {
            knowhere::Json json = upstream_gen();
            json[knowhere::indexparam::ADAPT_FOR_CPU] = false;
            json[knowhere::indexparam::EF] = 128;
            return json;
        };
    };

    auto refined_gen = [](auto&& upstream_gen) {
        return [upstream_gen]() {
            knowhere::Json json = upstream_gen();
            json[knowhere::indexparam::REFINE_RATIO] = 1.5;
            json[knowhere::indexparam::CACHE_DATASET_ON_DEVICE] = true;
            return json;
        };
    };

    auto cosine_gen = [](auto&& upstream_gen) {
        return [upstream_gen]() {
            knowhere::Json json = upstream_gen();
            json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
            return json;
        };
    };
    auto hamming_gen = [](auto&& upstream_gen) {
        return [upstream_gen]() {
            knowhere::Json json = upstream_gen();
            json[knowhere::meta::METRIC_TYPE] = knowhere::metric::HAMMING;
            json[knowhere::indexparam::BUILD_ALGO] = "ITERATIVE";
            return json;
        };
    };

    SECTION("Test Gpu Index Search") {
        using std::make_tuple;
        auto [name, gen, min_recall] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>, float>({
            make_tuple(knowhere::IndexEnum::INDEX_GPU_BRUTEFORCE, bruteforce_gen, 0.999f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_IVFFLAT, ivfflat_gen, 0.95f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_IVFFLAT, refined_gen(ivfflat_gen), 0.95f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_IVFPQ, ivfpq_gen, 0.75f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_IVFPQ, refined_gen(ivfpq_gen), 0.75f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_CAGRA, cagra_gen, 0.9f),
            make_tuple(knowhere::IndexEnum::INDEX_GPU_CAGRA, cagra_hnsw_gen(cagra_gen), 0.9f),
        }));

        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json conf = knowhere::Json::parse(cfg_json);
        conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        if (knowhere::IndexFactory::Instance().FeatureCheck(name, knowhere::feature::FLOAT32)) {
            check_search<knowhere::fp32>(nb, nq, dim, seed, name, version, conf, min_recall);
        }
        if (knowhere::IndexFactory::Instance().FeatureCheck(name, knowhere::feature::FP16)) {
            check_search<knowhere::fp16>(nb, nq, dim, seed, name, version, conf, min_recall);
        }
        if (knowhere::IndexFactory::Instance().FeatureCheck(name, knowhere::feature::INT8)) {
            check_search<knowhere::int8>(nb, nq, dim, seed, name, version, conf, min_recall);
        }
    }

    SECTION("Test Gpu Index Search With Bitset") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, bruteforce_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, refined_gen(ivfflat_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, refined_gen(ivfpq_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
            GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
        const auto bitset_percentages = {0.4f, 0.98f};
        for (const float percentage : bitset_percentages) {
            for (const auto& gen_func : gen_bitset_funcs) {
                auto bitset_data = gen_func(nb, percentage * nb);
                knowhere::BitsetView bitset(bitset_data.data(), nb);
                auto results = idx.Search(query_ds, json, bitset);
                REQUIRE(results.has_value());
                auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, bitset);
                float recall = GetKNNRecall(*gt.value(), *results.value());
                if (percentage == 0.98f) {
                    REQUIRE(recall > 0.4f);
                } else {
                    REQUIRE(recall > 0.7f);
                }
            }
        }
    }

    SECTION("Test Gpu Index Search TopK") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, bruteforce_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, refined_gen(ivfflat_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, refined_gen(ivfpq_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_hnsw_gen(cagra_gen)),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        const auto topk_values = {// Tuple with [TopKValue, Threshold]
                                  make_tuple(5, 0.85f), make_tuple(25, 0.85f), make_tuple(100, 0.85f)};

        for (const auto& topKTuple : topk_values) {
            json[knowhere::meta::TOPK] = std::get<0>(topKTuple);
            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
            auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= std::get<1>(topKTuple));
        }
    }

    SECTION("Test Gpu Index Serialize/Deserialize") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, bruteforce_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, refined_gen(ivfflat_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, refined_gen(ivfpq_gen)),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_hnsw_gen(cagra_gen)),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);
        auto idx_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        idx_.Deserialize(bs);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        auto results = idx_.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        // Due to issues with the filtering of invalid values, index 0 is temporarily not being checked.
        for (int i = 1; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test Gpu Index Cagra Adapt For Cpu") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);
        auto idx_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        json[knowhere::indexparam::ADAPT_FOR_CPU] = true;
        json[knowhere::indexparam::EF] = 128;
        idx_.Deserialize(bs, json);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        auto results = idx_.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        // Due to issues with the filtering of invalid values, index 0 is temporarily not being checked.
        for (int i = 1; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test Gpu Index Cagra Adapt For Cpu Without Ef") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);
        auto idx_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        knowhere::Json deser_json = json;
        deser_json[knowhere::indexparam::ADAPT_FOR_CPU] = true;
        // Intentionally NOT setting ef — this is the bug scenario
        idx_.Deserialize(bs, deser_json);
        // Search without ef in params — should not crash
        auto results = idx_.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        for (int i = 1; i < nq; ++i) {
            CHECK(ids[i] == i);
        }
    }

    SECTION("Test Gpu Index Search Simple Bitset") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, bruteforce_gen),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, refined_gen(ivfflat_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, ivfpq_gen),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, refined_gen(ivfpq_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cagra_gen)}));
        auto rows = 64;
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        if (name == knowhere::IndexEnum::INDEX_CUVS_CAGRA) {
            json[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = 32;
            json[knowhere::indexparam::GRAPH_DEGREE] = 32;
            json[knowhere::indexparam::ITOPK_SIZE] = 32;
        }
        auto train_ds = GenDataSet(rows, dim, seed);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        std::vector<uint8_t> bitset_data(8);
        bitset_data[0] = 0b10100010;
        bitset_data[1] = 0b00100011;
        bitset_data[2] = 0b10100010;
        bitset_data[3] = 0b00100111;
        bitset_data[4] = 0b10100000;
        bitset_data[5] = 0b00000000;
        bitset_data[6] = 0b00000010;
        bitset_data[7] = 0b11100011;
        knowhere::BitsetView bitset(bitset_data.data(), rows);
        auto results = idx.Search(train_ds, json, bitset);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, train_ds, json, bitset);
        // Go through the results and check if the id is in the bitset
        for (int i = 0; i < rows; ++i) {
            auto id = results.value()->GetIds()[i];
            if (id == -1) {
                continue;
            }
            REQUIRE(!(bitset_data[id / 8] & (1 << (id % 8))));
        }
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.8f);
    }

    SECTION("Test Gpu Index Search Cosine Metric") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, cosine_gen(bruteforce_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, cosine_gen(ivfflat_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, cosine_gen(ivfpq_gen)),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_IVFPQ, cosine_gen(refined_gen(ivfpq_gen))),
             make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, cosine_gen(cagra_gen))}));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 1);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));
        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall > 0.65f);
    }

    SECTION("Test Gpu Cagra Int8 Cosine Score Range") {
        constexpr auto rows = 512;
        constexpr auto query_rows = 64;
        constexpr auto test_dim = 64;
        constexpr auto topk = 10;

        knowhere::Json json;
        json[knowhere::meta::DIM] = test_dim;
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = 64;
        json[knowhere::indexparam::GRAPH_DEGREE] = 32;
        json[knowhere::indexparam::ITOPK_SIZE] = 64;
        json[knowhere::indexparam::NUM_RANDOM_SAMPLINGS] = 4;

        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::int8>(knowhere::IndexEnum::INDEX_CUVS_CAGRA, version)
                       .value();
        auto train_ds = GenNormalizedInt8DataSet(rows, test_dim, seed);
        auto query_ds = GenNormalizedInt8DataSet(query_rows, test_dim, seed);

        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());

        auto ids = results.value()->GetIds();
        auto distances = results.value()->GetDistance();
        auto max_distance = -1.0f;
        for (auto i = 0; i < query_rows * topk; ++i) {
            if (ids[i] < 0) {
                continue;
            }
            max_distance = std::max(max_distance, distances[i]);
            CHECK(distances[i] >= -1.00001f);
            CHECK(distances[i] <= 1.00001f);
        }
        CHECK(max_distance > 0.5f);
    }

    SECTION("Test Gpu Index Search Hamming Metric") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::IndexEnum::INDEX_CUVS_CAGRA, hamming_gen(cagra_gen))}));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::bin1>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        nb = 1500;  // Reduce dataset size to have less distance = 0 when testing query distance
        auto train_ds = GenBinDataSet(nb, dim, seed);
        auto query_ds = GenBinDataSet(nq, dim, seed + 1);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.Count() == nb);
        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, json, nullptr);
        auto dist = results.value()->GetDistance();
        auto gt_dist = gt.value()->GetDistance();
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall > 0.8f);
        recall = GetKNNRelativeRecall(*gt.value(), *results.value(), true);
        REQUIRE(recall > 0.95f);
        for (int i = 1; i < nq; ++i) {
            // Check query distance
            CHECK(GetRelativeLoss(gt_dist[i], dist[i]) < 0.1f);
        }
    }
}
#endif
