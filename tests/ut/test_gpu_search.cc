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

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
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

// Host-safe header (no device kernels) exposing GpuHnswUploadFaultInjection, the
// test-only hook used by the CUDA-upload fault-injection section below.
#include <faiss/gpu/impl/GpuHnswTypes.h>

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

    // GPU_HNSW tests: build on CPU as HNSW, serialize, deserialize as GPU_HNSW, search on GPU.
    SECTION("Test GPU HNSW Search (CPU build -> GPU search)") {
        // Build a CPU HNSW index
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        // Serialize the CPU index
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        // Deserialize as GPU_HNSW
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        // Count()/Dim() must survive the CPU-copy release after GPU upload;
        // regression guard for returning -1 (segment treated as empty).
        REQUIRE(gpu_idx.Count() == nb);
        REQUIRE(gpu_idx.Dim() == dim);

        // Search on GPU
        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        // Compare against brute force
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.9f);
    }

    SECTION("Test GPU HNSW Search Cosine Metric") {
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 1);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // GPU HNSW cosine recall tracks L2 (measured ~0.98 on L40S); keep the
        // floor in line with the L2/IP sections so real regressions are caught.
        REQUIRE(recall >= 0.80f);
    }

    SECTION("Test GPU HNSW Cosine Metric Omitted From Search Config") {
        // Regression for I1: the GPU node must derive the metric from the built
        // GPU index (via GpuIndexHNSW::isCosine()), NOT from the search config.
        // A search config that omits metric_type defaults to L2, which would
        // skip cosine query normalization and silently return wrong scores.
        // Here the index is COSINE but the search json carries no metric_type;
        // results must match a search whose json explicitly sets COSINE.
        knowhere::Json build_json;
        build_json[knowhere::meta::DIM] = dim;
        build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
        build_json[knowhere::meta::TOPK] = 10;
        build_json[knowhere::indexparam::HNSW_M] = 16;
        build_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        build_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 1);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(cpu_idx.Build(train_ds, build_json) == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

        // Baseline: search with metric_type explicitly set to COSINE.
        auto with_metric = gpu_idx.Search(query_ds, build_json, nullptr);
        REQUIRE(with_metric.has_value());

        // Same search, but metric_type omitted from the config.
        knowhere::Json no_metric_json = build_json;
        no_metric_json.erase(knowhere::meta::METRIC_TYPE);
        auto without_metric = gpu_idx.Search(query_ds, no_metric_json, nullptr);
        REQUIRE(without_metric.has_value());

        // The metric is index-derived, so both searches must be identical.
        const int64_t topk = build_json[knowhere::meta::TOPK].get<int64_t>();
        const auto* ids_a = with_metric.value()->GetIds();
        const auto* ids_b = without_metric.value()->GetIds();
        const auto* dist_a = with_metric.value()->GetDistance();
        const auto* dist_b = without_metric.value()->GetDistance();
        for (int64_t i = 0; i < nq * topk; ++i) {
            REQUIRE(ids_a[i] == ids_b[i]);
            REQUIRE(dist_a[i] == Approx(dist_b[i]).epsilon(1e-5));
        }
    }

    SECTION("Test GPU HNSW Search TopK") {
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        const auto topk_values = {
            std::make_tuple(5, 0.85f),
            std::make_tuple(25, 0.85f),
            std::make_tuple(100, 0.85f),
        };

        for (const auto& [topk, threshold] : topk_values) {
            hnsw_json[knowhere::meta::TOPK] = topk;
            auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
            REQUIRE(results.has_value());
            auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= threshold);
        }
    }

    SECTION("Test GPU HNSW Serialize/Deserialize Round Trip") {
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        // Self-search: each vector should find itself as nearest neighbor.
        // The query set is train_ds (nb rows), so check all nb results — using
        // nq only exercised the first nq of nb vectors.
        auto results = gpu_idx.Search(train_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        int correct = 0;
        for (int i = 0; i < nb; ++i) {
            if (ids[i] == i)
                correct++;
        }
        float self_recall = static_cast<float>(correct) / nb;
        REQUIRE(self_recall >= 0.95f);
    }

    SECTION("Test GPU HNSW SQ8 Deserialization") {
        // Build a CPU HNSW_SQ index, then load as GPU_HNSW
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                           .value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        // GPU_HNSW should accept HNSW_SQ binaries
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // SQ8 has lower recall than flat due to quantization
        REQUIRE(recall >= 0.7f);
    }

    SECTION("Test GPU HNSW INT8 Deserialization") {
        // Build a CPU int8 HNSW (HNSW_SQ QT_8bit_direct_signed storage), then
        // load and search on GPU. int8 vectors stay in their native 1-byte
        // layout on the device (upload_int8_dataset) and are up-converted to
        // fp32 per element in the search kernel. This exercises the native
        // signed-int8 device path, distinct from the unsigned SQ8 test above
        // which routes through the decode-to-fp32 upload branch.
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(GenDataSet(nb, dim, seed));
        auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(GenDataSet(nq, dim, seed + 2));

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::int8>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::int8>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::int8>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // int8 native path should match the CPU int8 HNSW recall closely.
        REQUIRE(recall >= 0.9f);
    }

    SECTION("Test GPU HNSW FP16 Deserialization") {
        // Build a CPU HNSW (fp16 -> HNSW_SQ QT_fp16 storage), then load and
        // search on GPU. fp16 vectors stay in their native 2-byte layout on the
        // device and are up-converted to fp32 per element in the search kernel.
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(GenDataSet(nb, dim, seed));
        auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(GenDataSet(nq, dim, seed + 2));

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp16>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp16>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::fp16>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // fp16 is near-lossless (10-bit mantissa); recall should stay close to
        // the fp32 flat path.
        REQUIRE(recall >= 0.9f);
    }

    SECTION("Test GPU HNSW BF16 Deserialization") {
        // Build a CPU HNSW (bf16 -> HNSW_SQ QT_bf16 storage), then load and
        // search on GPU. bf16 vectors stay in their native 2-byte layout on the
        // device and are up-converted to fp32 per element in the search kernel.
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 1;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(GenDataSet(nb, dim, seed));
        auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(GenDataSet(nq, dim, seed + 2));

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::bf16>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::bf16>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(results.has_value());

        auto gt = knowhere::BruteForce::Search<knowhere::bf16>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // bf16 has a 7-bit mantissa (coarser than fp16) but a wide exponent;
        // recall stays high but with a slightly looser floor.
        REQUIRE(recall >= 0.85f);
    }

    SECTION("Test GPU HNSW non-power-of-two 2*M staging (fp32 + int8 DP4A)") {
        // M=24 => max_degree0 = 2*M = 48, which is NOT a power of two. The
        // parallel bitonic-merge kernel pads the layer-0 staging capacity up to
        // the next power of two (64) and grows the block accordingly. Before the
        // padding fix this configuration hard-failed every GPU search. Cover
        // both the generic fp32-query kernel and the native int8 DP4A kernel
        // (dim=128 is divisible by 4, so int8 L2/IP takes the DP4A path; int8
        // COSINE is re-encoded as fp16 and uses the fp32-query kernel instead).
        const int64_t m = 24;

        SECTION("fp32 cosine, non-pow2 2*M") {
            knowhere::Json hnsw_json;
            hnsw_json[knowhere::meta::DIM] = dim;
            hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
            hnsw_json[knowhere::meta::TOPK] = 10;
            hnsw_json[knowhere::indexparam::HNSW_M] = m;
            hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
            hnsw_json[knowhere::indexparam::EF] = 200;

            auto train_ds = GenDataSet(nb, dim, seed);
            auto query_ds = GenDataSet(nq, dim, seed + 2);

            auto cpu_idx = knowhere::IndexFactory::Instance()
                               .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version)
                               .value();
            REQUIRE(cpu_idx.Build(train_ds, hnsw_json) == knowhere::Status::success);

            knowhere::BinarySet bs;
            cpu_idx.Serialize(bs);

            auto gpu_idx = knowhere::IndexFactory::Instance()
                               .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                               .value();
            REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

            auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
            REQUIRE(results.has_value());

            auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
            REQUIRE(gt.has_value());
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= 0.80f);
        }

        SECTION("int8 IP DP4A, non-pow2 2*M") {
            knowhere::Json hnsw_json;
            hnsw_json[knowhere::meta::DIM] = dim;
            hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
            hnsw_json[knowhere::meta::TOPK] = 10;
            hnsw_json[knowhere::indexparam::HNSW_M] = m;
            hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
            hnsw_json[knowhere::indexparam::EF] = 200;

            auto train_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(GenDataSet(nb, dim, seed));
            auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(GenDataSet(nq, dim, seed + 2));

            auto cpu_idx = knowhere::IndexFactory::Instance()
                               .Create<knowhere::int8>(knowhere::IndexEnum::INDEX_HNSW, version)
                               .value();
            REQUIRE(cpu_idx.Build(train_ds, hnsw_json) == knowhere::Status::success);

            knowhere::BinarySet bs;
            cpu_idx.Serialize(bs);

            auto gpu_idx = knowhere::IndexFactory::Instance()
                               .Create<knowhere::int8>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                               .value();
            REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

            auto results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
            REQUIRE(results.has_value());

            auto gt = knowhere::BruteForce::Search<knowhere::int8>(train_ds, query_ds, hnsw_json, nullptr);
            REQUIRE(gt.has_value());
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= 0.80f);
        }
    }
}

TEST_CASE("Test CPU vs GPU HNSW Comparison", "[gpu_hnsw_compare]") {
    using Catch::Approx;

    int64_t nb = 10000, nq = 100;
    int64_t dim = 128;
    int64_t seed = 42;

    auto version = GenTestVersionList();

    auto run_comparison = [&](const std::string& metric, float min_recall, float max_dist_drift) {
        CAPTURE(metric, nb, nq, dim);

        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = metric;
        hnsw_json[knowhere::meta::TOPK] = 10;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        hnsw_json[knowhere::indexparam::EF] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        // --- Build CPU HNSW ---
        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        // Serialize for GPU deserialization
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        // --- CPU Search with timing ---
        auto cpu_start = std::chrono::high_resolution_clock::now();
        auto cpu_results = cpu_idx.Search(query_ds, hnsw_json, nullptr);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        REQUIRE(cpu_results.has_value());
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // --- Build GPU HNSW from serialized CPU index ---
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        // Warm-up search (first GPU call has kernel launch overhead)
        auto warmup = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        REQUIRE(warmup.has_value());

        // --- GPU Search with timing ---
        auto gpu_start = std::chrono::high_resolution_clock::now();
        auto gpu_results = gpu_idx.Search(query_ds, hnsw_json, nullptr);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        REQUIRE(gpu_results.has_value());
        double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        // --- Brute force ground truth ---
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());

        // --- Recall comparison ---
        float cpu_recall = GetKNNRecall(*gt.value(), *cpu_results.value());
        float gpu_recall = GetKNNRecall(*gt.value(), *gpu_results.value());

        // --- Distance accuracy: compare GPU distances against CPU distances ---
        auto cpu_dist = cpu_results.value()->GetDistance();
        auto gpu_dist = gpu_results.value()->GetDistance();
        auto k = hnsw_json[knowhere::meta::TOPK].get<int64_t>();

        double total_rel_error = 0.0;
        int valid_pairs = 0;
        for (int64_t i = 0; i < nq * k; ++i) {
            if (std::abs(cpu_dist[i]) > 1e-6f) {
                total_rel_error += std::abs((gpu_dist[i] - cpu_dist[i]) / cpu_dist[i]);
                valid_pairs++;
            }
        }
        double avg_rel_dist_error = (valid_pairs > 0) ? (total_rel_error / valid_pairs) : 0.0;

        // --- ID overlap: how many of the same results are returned ---
        auto cpu_ids = cpu_results.value()->GetIds();
        auto gpu_ids = gpu_results.value()->GetIds();
        int id_overlap = 0;
        for (int64_t q = 0; q < nq; ++q) {
            std::set<int64_t> cpu_set(cpu_ids + q * k, cpu_ids + (q + 1) * k);
            for (int64_t j = 0; j < k; ++j) {
                if (cpu_set.count(gpu_ids[q * k + j])) {
                    id_overlap++;
                }
            }
        }
        float id_overlap_ratio = static_cast<float>(id_overlap) / (nq * k);

        // --- Print comparison report ---
        fprintf(stderr, "\n=== CPU vs GPU HNSW Comparison (%s, nb=%ld, nq=%ld, dim=%ld, k=%ld) ===\n", metric.c_str(),
                (long)nb, (long)nq, (long)dim, (long)k);
        fprintf(stderr, "  CPU recall@%ld: %.4f\n", (long)k, cpu_recall);
        fprintf(stderr, "  GPU recall@%ld: %.4f\n", (long)k, gpu_recall);
        fprintf(stderr, "  Recall delta (GPU - CPU): %+.4f\n", gpu_recall - cpu_recall);
        fprintf(stderr, "  CPU search time: %.2f ms\n", cpu_ms);
        fprintf(stderr, "  GPU search time: %.2f ms (includes H2D/D2H transfers)\n", gpu_ms);
        fprintf(stderr, "  Speedup: %.2fx\n", cpu_ms / gpu_ms);
        fprintf(stderr, "  Avg relative distance error: %.6f\n", avg_rel_dist_error);
        fprintf(stderr, "  ID overlap (CPU vs GPU): %.4f (%d/%ld)\n", id_overlap_ratio, id_overlap, (long)(nq * k));
        fprintf(stderr, "===\n\n");

        // --- Assertions ---
        // GPU recall should be close to CPU recall (within a small tolerance)
        REQUIRE(gpu_recall >= min_recall);
        // GPU should return at least 80% of the same IDs as CPU
        REQUIRE(id_overlap_ratio >= 0.8f);
        // Average relative distance error should be small
        REQUIRE(avg_rel_dist_error <= max_dist_drift);
    };

    SECTION("L2 metric comparison") {
        run_comparison(knowhere::metric::L2, 0.85f, 0.05);
    }

    SECTION("IP metric comparison") {
        run_comparison(knowhere::metric::IP, 0.85f, 0.05);
    }

    SECTION("COSINE metric comparison") {
        run_comparison(knowhere::metric::COSINE, 0.60f, 0.10);
    }

    SECTION("Varying ef comparison") {
        knowhere::Json hnsw_json;
        hnsw_json[knowhere::meta::DIM] = dim;
        hnsw_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        hnsw_json[knowhere::meta::TOPK] = 10;
        hnsw_json[knowhere::indexparam::HNSW_M] = 16;
        hnsw_json[knowhere::indexparam::EFCONSTRUCTION] = 200;

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto res = cpu_idx.Build(train_ds, hnsw_json);
        REQUIRE(res == knowhere::Status::success);

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto deser_res = gpu_idx.Deserialize(bs);
        REQUIRE(deser_res == knowhere::Status::success);

        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, hnsw_json, nullptr);
        REQUIRE(gt.has_value());

        fprintf(stderr, "\n=== ef sweep (L2, nb=%ld, nq=%ld, k=10) ===\n", (long)nb, (long)nq);
        fprintf(stderr, "  %6s  %10s  %10s  %10s  %10s\n", "ef", "cpu_recall", "gpu_recall", "cpu_ms", "gpu_ms");

        for (int ef : {16, 32, 64, 128, 256, 512}) {
            hnsw_json[knowhere::indexparam::EF] = ef;

            auto t0 = std::chrono::high_resolution_clock::now();
            auto cpu_res = cpu_idx.Search(query_ds, hnsw_json, nullptr);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto gpu_res = gpu_idx.Search(query_ds, hnsw_json, nullptr);
            auto t2 = std::chrono::high_resolution_clock::now();

            REQUIRE(cpu_res.has_value());
            REQUIRE(gpu_res.has_value());

            float cpu_recall = GetKNNRecall(*gt.value(), *cpu_res.value());
            float gpu_recall = GetKNNRecall(*gt.value(), *gpu_res.value());
            double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            fprintf(stderr, "  %6d  %10.4f  %10.4f  %10.2f  %10.2f\n", ef, cpu_recall, gpu_recall, cpu_ms, gpu_ms);

            // This is a GPU-vs-CPU comparison: GPU HNSW must track the CPU HNSW oracle at
            // every ef, not hit an arbitrary absolute floor. At small ef (e.g. ef=16, ~= k)
            // HNSW recall is legitimately low on both engines (CPU itself is < 0.5 here), so
            // an absolute floor is meaningless; parity with CPU is the correct invariant.
            REQUIRE(gpu_recall >= cpu_recall - 0.05f);
        }
        fprintf(stderr, "===\n\n");
    }
}

// Regression tests for the three Codex P1 findings on the gpu-hnsw-faiss branch:
//   #1 quantized-cosine inverse-norm semantics (native low-precision kernels +
//      SQ8/FP16/BF16 cosine parity),
//   #2 search-vs-reload lifetime safety,
//   #3 phase-separated load-resource accounting.
TEST_CASE("Test GPU HNSW Codex P1 Regressions", "[gpu_hnsw_p1]") {
    using Catch::Approx;

    const int64_t nb = 4000, nq = 200;
    const int64_t dim = 64;
    const int64_t seed = 42;
    const auto version = GenTestVersionList();

    auto sq_json = [&](const std::string& metric, const std::string& sq_type, int64_t topk) {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::HNSW_M] = 16;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 200;
        json[knowhere::indexparam::SQ_TYPE] = sq_type;
        return json;
    };

    // Build an HNSW_SQ index on the fp32 (real, non-mock) node so the serialized
    // storage carries the requested SQ qtype. Deserializing that as GPU_HNSW on
    // the fp32 node selects the native device representation from the storage
    // qtype (FP16 -> half kernel, BF16 -> __nv_bfloat16 kernel, SQ8 -> fp32
    // decode). This bypasses the int8/fp16/bf16 IndexNodeDataMockWrapper, which
    // otherwise converts data to fp32 and routes the fp32 kernel.

    // #1a: native low-precision (FP16/BF16) and SQ8 device paths — L2.
    SECTION("Native low-precision device path (L2): FP16 / BF16 / SQ8") {
        const auto sq_type = GENERATE(std::string("fp16"), std::string("bf16"), std::string("sq8"));
        CAPTURE(sq_type);
        auto json = sq_json(knowhere::metric::L2, sq_type, 10);

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                           .value();
        REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        // Lossy SQ paths; a conservative floor. fp16/bf16 are near-lossless, sq8
        // coarser.
        REQUIRE(recall >= (sq_type == "sq8" ? 0.65f : 0.85f));
    }

    // #4: Milvus loads sealed segments through the mmap/file path
    // (VectorMemIndex::LoadFromFile -> DeserializeFromFile), NOT
    // Deserialize(BinarySet). The eager GPU upload must fire on that path too;
    // otherwise the CPU-built HNSW stays resident in host RAM and the GPU is
    // never used. This was the root cause of the mpd_v2 querynode OOM: each
    // segment's (fp32-expanded) CPU index — ~5.9 GiB — was retained in host
    // anon memory instead of being uploaded to VRAM and freed, so ~14
    // segments/pod overran the memory limit while VRAM stayed near-idle.
    //
    // The discriminating assertion is Size()==0 *before any search*: the eager
    // upload releases indexes[0], so the base Size() (computed from indexes[0])
    // is 0. Pre-fix, DeserializeFromFile skipped the upload and left the CPU
    // copy resident, so Size() reported the full retained host footprint. A
    // recall-only check would NOT catch the regression, because the first
    // Search() lazily uploads and frees the copy — masking the load-time leak
    // that OOMs before any search runs.
    SECTION("DeserializeFromFile uploads to GPU and frees the host copy") {
        const bool enable_mmap = GENERATE(false, true);
        CAPTURE(enable_mmap);
        auto json = sq_json(knowhere::metric::COSINE, "sq8", 10);

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                           .value();
        REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);
        auto binary = bs.GetByName(cpu_idx.Type());
        REQUIRE(binary != nullptr);

        const auto path = std::filesystem::temp_directory_path() /
                          ("knowhere_gpu_hnsw_deserialize_from_file_" + std::to_string(enable_mmap) + ".index");
        std::filesystem::remove(path);
        {
            std::ofstream out(path, std::ios::binary);
            REQUIRE(out.is_open());
            out.write(reinterpret_cast<const char*>(binary->data.get()), binary->size);
        }

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        auto load_json = json;
        load_json["enable_mmap"] = enable_mmap;
        REQUIRE(gpu_idx.DeserializeFromFile(path.string(), load_json) == knowhere::Status::success);

        // Eager upload ran during load (no search yet): host copy released.
        REQUIRE(gpu_idx.Size() == 0);
        REQUIRE(gpu_idx.Count() == nb);
        REQUIRE(gpu_idx.HasRawData(knowhere::metric::COSINE) == false);

        // And the GPU index actually serves searches.
        auto results = gpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
        REQUIRE(gt.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.65f);

        std::filesystem::remove(path);
    }

    // #1b: quantized-cosine recall vs an exact brute-force cosine oracle. The
    // input shares directions but spans widely different magnitudes: cosine must
    // ignore magnitude. The native GPU path L2-normalizes and re-encodes (no
    // inverse-norm buffer), so the correct check is recall against true cosine,
    // not byte/score parity with the legacy inverse-norm CPU encoding. fp16/bf16
    // represent the wide magnitudes accurately and must clear a high recall bar;
    // sq8 (trained QT_8bit over raw vectors) is intrinsically limited on this
    // magnitude spread and is only required to have the GPU track the CPU index
    // (its absolute recall on normal data is gated by [gpu_hnsw_dtype_sq8]).
    SECTION("Quantized cosine recall (CPU vs GPU vs oracle): SQ8 / FP16 / BF16") {
        const auto sq_type = GENERATE(std::string("sq8"), std::string("fp16"), std::string("bf16"));
        CAPTURE(sq_type);
        const int64_t topk = 10;
        auto json = sq_json(knowhere::metric::COSINE, sq_type, topk);

        // Directions from a fixed RNG, then scale each row by a magnitude that
        // spans three orders of magnitude so magnitude cannot be ignored by luck.
        auto make_scaled = [&](int64_t rows, int64_t s) {
            auto ds = GenDataSet(rows, dim, s);
            auto* data = const_cast<float*>(static_cast<const float*>(ds->GetTensor()));
            std::mt19937 rng(static_cast<uint32_t>(s + 7));
            std::uniform_real_distribution<float> mag(0.5f, 500.0f);
            for (int64_t r = 0; r < rows; ++r) {
                float scale = mag(rng);
                for (int64_t d = 0; d < dim; ++d) {
                    data[r * dim + d] *= scale;
                }
            }
            return ds;
        };
        auto train_ds = make_scaled(nb, seed);
        auto query_ds = make_scaled(nq, seed + 2);

        auto cpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                           .value();
        REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
        auto cpu_results = cpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(cpu_results.has_value());

        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);
        auto gpu_results = gpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(gpu_results.has_value());

        // Oracle: exact cosine top-k from a brute-force search over the ORIGINAL
        // (un-quantized) vectors. This is the representation-agnostic ground
        // truth for normalized cosine. The legacy CPU cppcontrib index encodes
        // cosine as direct_signed codes + inverse-L2-norms of the ORIGINAL
        // vectors; the native GPU path L2-normalizes and re-encodes (int8 ->
        // fp16, fp16/bf16 kept), so byte/score parity between the two no longer
        // holds and would be the wrong thing to assert. Both must instead track
        // true cosine recall.
        knowhere::Json bf_json;
        bf_json[knowhere::meta::DIM] = dim;
        bf_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
        bf_json[knowhere::meta::TOPK] = topk;
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, bf_json, nullptr);
        REQUIRE(gt.has_value());

        float cpu_recall = GetKNNRecall(*gt.value(), *cpu_results.value());
        float gpu_recall = GetKNNRecall(*gt.value(), *gpu_results.value());
        CAPTURE(cpu_recall);
        CAPTURE(gpu_recall);

        if (sq_type == "sq8") {
            // sq8 == trained QT_8bit over the RAW (un-normalized) vectors. The
            // input here shares directions but spans 1000x in magnitude, so the
            // single per-dimension trained range is dominated by the large-norm
            // rows and the small-norm rows lose their direction -> cosine recall
            // is intrinsically low for BOTH CPU and GPU. That is a scalar-
            // quantizer limitation on this adversarial magnitude spread, not a
            // GPU defect; absolute sq8-cosine recall on normal-magnitude data is
            // gated by [gpu_hnsw_dtype_sq8]. Here we only require that the GPU
            // faithfully reproduces the CPU index (it decodes the same sq8 codes
            // to fp32), i.e. the GPU is no worse than the CPU oracle-recall.
            REQUIRE(gpu_recall >= cpu_recall - 0.02f);
        } else {
            // fp16/bf16 represent wide-magnitude components accurately, so after
            // normalization the GPU tracks true cosine at high recall. bf16 has
            // fewer mantissa bits, hence a slightly lower bar.
            const float bar = (sq_type == "bf16") ? 0.80f : 0.85f;
            REQUIRE(gpu_recall >= bar);
            REQUIRE(gpu_recall >= cpu_recall - 0.10f);
        }
    }

    // #2: search must not use-after-free a GPU index that a concurrent reload
    // resets/rebuilds. Search continuously from several threads while another
    // thread repeatedly Deserialize()s (reloads) the same index. Run under
    // TSAN/ASAN to catch the lifetime race the atomic<shared_ptr> snapshot fixes.
    SECTION("Search vs concurrent reload (lifetime safety)") {
        auto json = sq_json(knowhere::metric::L2, "fp16", 10);
        json.erase(knowhere::indexparam::SQ_TYPE);  // plain fp32 HNSW for the reload driver
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 2);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

        std::atomic<bool> stop{false};
        std::atomic<bool> failed{false};
        std::vector<std::thread> searchers;
        for (int t = 0; t < 6; ++t) {
            searchers.emplace_back([&]() {
                while (!stop.load(std::memory_order_relaxed)) {
                    auto r = gpu_idx.Search(query_ds, json, nullptr);
                    // A search may race a reload window; success or a benign
                    // "not ready" error are both acceptable, a crash is not.
                    if (!r.has_value() && r.error() != knowhere::Status::empty_index &&
                        r.error() != knowhere::Status::cuda_runtime_error) {
                        failed.store(true, std::memory_order_relaxed);
                    }
                }
            });
        }
        std::thread reloader([&]() {
            for (int i = 0; i < 30 && !stop.load(std::memory_order_relaxed); ++i) {
                if (gpu_idx.Deserialize(bs) != knowhere::Status::success) {
                    failed.store(true, std::memory_order_relaxed);
                }
            }
            stop.store(true, std::memory_order_relaxed);
        });
        reloader.join();
        stop.store(true, std::memory_order_relaxed);
        for (auto& th : searchers) {
            th.join();
        }
        REQUIRE(!failed.load());
        // Index is still usable after the reload storm.
        auto after = gpu_idx.Search(query_ds, json, nullptr);
        REQUIRE(after.has_value());
    }

    // Codex coverage #3: more than four simultaneous searches (the device scratch
    // pool has four slots), with varying nq/k/ef, each validated against its own
    // brute-force ground truth. Exercises scratch-pool contention/serialization.
    SECTION("More than four concurrent searches with varying nq/k/ef") {
        auto build_json = sq_json(knowhere::metric::L2, "fp16", 10);
        build_json.erase(knowhere::indexparam::SQ_TYPE);
        auto train_ds = GenDataSet(nb, dim, seed);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(cpu_idx.Build(train_ds, build_json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);
        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

        struct Case {
            int64_t nq;
            int64_t k;
            int ef;
        };
        const std::vector<Case> cases = {{50, 5, 32},   {120, 10, 64}, {200, 20, 128}, {80, 50, 256},
                                         {30, 10, 512}, {150, 1, 48},  {64, 100, 300}, {90, 25, 96}};
        std::atomic<bool> ok{true};
        std::vector<std::thread> threads;
        for (const auto& c : cases) {
            threads.emplace_back([&, c]() {
                knowhere::Json j;
                j[knowhere::meta::DIM] = dim;
                j[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
                j[knowhere::meta::TOPK] = c.k;
                j[knowhere::indexparam::EF] = c.ef;
                auto q = GenDataSet(c.nq, dim, seed + 100 + c.ef);
                auto r = gpu_idx.Search(q, j, nullptr);
                if (!r.has_value()) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, q, j, nullptr);
                if (!gt.has_value() || GetKNNRecall(*gt.value(), *r.value()) < 0.7f) {
                    ok.store(false, std::memory_order_relaxed);
                }
            });
        }
        for (auto& th : threads) {
            th.join();
        }
        REQUIRE(ok.load());
    }

    // #3: phase-separated load-resource accounting. GPU_HNSW frees its CPU copy
    // after uploading to VRAM, so retained memoryCost is 0 while the transient
    // peak (maxMemoryCost) must be non-zero and cover the serialized read buffer
    // + deserialized compact index (~2x file_size). This estimate is a static
    // computation and needs no live GPU.
    SECTION("Load-resource estimate: memoryCost=0, maxMemoryCost covers peak") {
        knowhere::Json params;
        params[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        params[knowhere::meta::DIM] = dim;

        const uint64_t file_size = static_cast<uint64_t>(nb) * dim * sizeof(float);
        auto gpu_res = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(
            knowhere::IndexEnum::INDEX_GPU_HNSW, version, file_size, nb, dim, params);
        REQUIRE(gpu_res.has_value());
        // Retained host memory is ~0 after the CPU copy is freed.
        REQUIRE(gpu_res.value().memoryCost == 0);
        REQUIRE(gpu_res.value().diskCost == 0);
        // Transient peak is non-zero and at least 2x the on-disk file size
        // (serialized read buffer + deserialized compact index).
        REQUIRE(gpu_res.value().maxMemoryCost > 0);
        REQUIRE(gpu_res.value().maxMemoryCost >= 2 * file_size);

        // The estimate is driven by file_size * 2 (compact int8 upload path:
        // no fp32 expansion staging). For a compressed (e.g. int8) file the
        // peak is 2 * int8_file — NOT the fp32-expanded footprint that the
        // old hnswlib-based estimate used.
        const uint64_t int8_file = static_cast<uint64_t>(nb) * dim * sizeof(int8_t);
        auto compressed_res = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(
            knowhere::IndexEnum::INDEX_GPU_HNSW, version, int8_file, nb, dim, params);
        REQUIRE(compressed_res.has_value());
        REQUIRE(compressed_res.value().maxMemoryCost >= 2 * int8_file);

        // When row/dim metadata is missing (num_rows arrives as 0 from the load
        // info), the estimate still works because it depends only on file_size.
        auto norows_res = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(
            knowhere::IndexEnum::INDEX_GPU_HNSW, version, int8_file, 0, dim, params);
        REQUIRE(norows_res.has_value());
        REQUIRE(norows_res.value().maxMemoryCost >= 2 * int8_file);

        // COSINE reserves more: it reconstructs + re-encodes to fp16 before the
        // upload, so the transient peak is ~4x file + fixed slack, not 2x. The
        // [gpu_hnsw_load_mem] test measures the real ~3.9x runtime peak; the
        // estimate must stay at or above it to keep concurrent loads from
        // over-admitting and OOM-killing the pod.
        knowhere::Json cos_params;
        cos_params[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
        cos_params[knowhere::meta::DIM] = dim;
        auto cos_res = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(
            knowhere::IndexEnum::INDEX_GPU_HNSW, version, int8_file, nb, dim, cos_params);
        REQUIRE(cos_res.has_value());
        REQUIRE(cos_res.value().memoryCost == 0);
        REQUIRE(cos_res.value().maxMemoryCost >= 4 * int8_file);
        // And strictly larger than the native (L2) direct-upload estimate.
        REQUIRE(cos_res.value().maxMemoryCost > compressed_res.value().maxMemoryCost);

        // The device (VRAM) reservation is reported separately and is decoupled
        // from the host transient peak: it must be > 0 and, for cosine, well
        // below the (larger) host peak so GPU admission does not over-reserve.
        REQUIRE(cos_res.value().gpuMemoryCost >= 2 * int8_file);
        REQUIRE(cos_res.value().gpuMemoryCost < cos_res.value().maxMemoryCost);
        REQUIRE(compressed_res.value().gpuMemoryCost >= 2 * int8_file);

        // A plain CPU HNSW keeps its data resident: memoryCost is non-zero and it
        // does not opt into the peak field (maxMemoryCost stays 0 -> Milvus falls
        // back to its 2*memoryCost heuristic).
        auto cpu_res = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(
            knowhere::IndexEnum::INDEX_HNSW, version, file_size, nb, dim, params);
        REQUIRE(cpu_res.has_value());
        REQUIRE(cpu_res.value().memoryCost > 0);
        REQUIRE(cpu_res.value().maxMemoryCost == 0);
    }

    // Codex coverage #5: CUDA-upload fault injection during Deserialize(). Using
    // the test-only GpuHnswUploadFaultInjection hook, force the eager GPU upload
    // to fail both at the very first wrapped CUDA call (immediate) and mid-upload
    // (after several allocations have already succeeded, exercising the device
    // index destructor's partial-allocation cleanup). Assert: (1) the load
    // reports failure, (2) nothing is half-published (a search while re-armed
    // errors cleanly rather than using a partial index or crashing), (3) VRAM
    // returns to baseline after the failed loads + node teardown (no leak), and
    // (4) a subsequent fault-free load succeeds and searches correctly.
    SECTION("CUDA upload fault injection: load fails, nothing half-published, retry succeeds") {
        auto json = sq_json(knowhere::metric::L2, "fp16", 10);
        json.erase(knowhere::indexparam::SQ_TYPE);  // plain fp32 HNSW
        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed + 3);

        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        cpu_idx.Serialize(bs);

        size_t free_before = 0, total = 0;
        REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);

        // fail_at = 1 fails the first cudaMalloc (immediate); fail_at = 4 fails
        // after several buffers are already on the device (mid-upload cleanup).
        for (int fail_at : {1, 4}) {
            CAPTURE(fail_at);
            auto gpu_idx = knowhere::IndexFactory::Instance()
                               .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                               .value();

            faiss::gpu::GpuHnswUploadFaultInjection::arm(fail_at);
            auto st = gpu_idx.Deserialize(bs);
            faiss::gpu::GpuHnswUploadFaultInjection::disarm();
            // (1) load fails rather than reporting a silently-broken segment.
            REQUIRE(st != knowhere::Status::success);

            // (2) no half-published index: re-arm so the lazy-upload retry inside
            // Search() also faults, proving the failed load left no usable index
            // behind and the search path errors cleanly (no crash / no results).
            faiss::gpu::GpuHnswUploadFaultInjection::arm(fail_at);
            auto r = gpu_idx.Search(query_ds, json, nullptr);
            faiss::gpu::GpuHnswUploadFaultInjection::disarm();
            REQUIRE(!r.has_value());
        }

        // (3) no leak: after the failed loads and node teardown, free VRAM must
        // return to ~baseline (allow slack for allocator/context fragmentation).
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        size_t free_after = 0;
        REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
        const size_t slack = static_cast<size_t>(64) * 1024 * 1024;  // 64 MiB
        REQUIRE(free_after + slack >= free_before);

        // (4) retry with the fault disarmed succeeds and returns valid results.
        faiss::gpu::GpuHnswUploadFaultInjection::disarm();
        auto gpu_ok = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                          .value();
        REQUIRE(gpu_ok.Deserialize(bs) == knowhere::Status::success);
        auto r_ok = gpu_ok.Search(query_ds, json, nullptr);
        REQUIRE(r_ok.has_value());
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
        REQUIRE(gt.has_value());
        REQUIRE(GetKNNRecall(*gt.value(), *r_ok.value()) >= 0.8f);
    }
}

// ============================================================================
// Full dtype x metric coverage for the GPU HNSW layer-0 search kernel.
//
// One TEST_CASE per dtype so each runs in its OWN process invocation (a CUDA
// illegal-access in one kernel instantiation corrupts the context and aborts
// the process; separate tags let the harness run every dtype independently and
// still collect per-dtype results instead of losing everything after the first
// crash). Every dtype is exercised for L2, IP and COSINE.
//
// All cases use the SAME config as the fp16 P1 regression that v82 OOBs on
// (nb=4000, dim=64, M=16, ef=200) so the fp32 case below is the control: if
// fp32 (the <float,float,false> kernel) is clean here while fp16
// (<__half,float,false>) faults, the defect is specific to the low-precision
// storage kernel; if fp32 also faults, it is config- rather than dtype-driven.
// dim=64 is divisible by 4, so the int8 IP case takes the DP4A path; int8
// COSINE is re-encoded as fp16 (see ToVanillaHnsw) and uses the fp32-query
// kernel instead.
namespace {
constexpr int64_t kDtypeNb = 4000;
constexpr int64_t kDtypeNq = 200;
constexpr int64_t kDtypeDim = 64;
constexpr int64_t kDtypeSeed = 42;
constexpr int64_t kDtypeTopk = 10;

knowhere::Json
dtype_json(const std::string& metric, const std::string& sq_type) {
    knowhere::Json json;
    json[knowhere::meta::DIM] = kDtypeDim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = kDtypeTopk;
    json[knowhere::indexparam::HNSW_M] = 16;
    json[knowhere::indexparam::EFCONSTRUCTION] = 200;
    json[knowhere::indexparam::EF] = 200;
    if (!sq_type.empty()) {
        json[knowhere::indexparam::SQ_TYPE] = sq_type;
    }
    return json;
}

// Build a CPU index of `index_type` on data type T, serialize, deserialize as
// GPU_HNSW, search, and return recall vs a brute-force reference on T. Shared by
// every per-dtype case so the only variables are the data type, the storage
// index type and the metric.
template <typename T>
float
gpu_dtype_recall(const std::string& index_type, const knowhere::Json& json, int version) {
    auto train_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(kDtypeNb, kDtypeDim, kDtypeSeed));
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(kDtypeNq, kDtypeDim, kDtypeSeed + 2));

    auto cpu_idx = knowhere::IndexFactory::Instance().Create<T>(index_type, version).value();
    REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);

    auto gpu_idx = knowhere::IndexFactory::Instance().Create<T>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());
    auto gt = knowhere::BruteForce::Search<T>(train_ds, query_ds, json, nullptr);
    REQUIRE(gt.has_value());
    return GetKNNRecall(*gt.value(), *results.value());
}

// Filtered-search recall helper. Builds a CPU index of `index_type` on T,
// uploads to GPU_HNSW, then for a given filter `ratio` searches with a random
// delete bitset. Returns {gpu_recall, cpu_recall} where both are measured vs a
// brute-force reference that uses the SAME bitset, so CPU HNSW is the parity
// baseline (not brute force alone). Also asserts no returned GPU id is a
// filtered bit. ratio == 0 exercises the HAS_FILTER=false fast path (empty
// view => data()==nullptr => unfiltered).
template <typename T>
std::pair<float, float>
gpu_filtered_recall(const std::string& index_type, const knowhere::Json& json, int version, float ratio,
                    uint32_t bitset_seed) {
    auto train_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(kDtypeNb, kDtypeDim, kDtypeSeed));
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(kDtypeNq, kDtypeDim, kDtypeSeed + 2));

    auto cpu_idx = knowhere::IndexFactory::Instance().Create<T>(index_type, version).value();
    REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);

    auto gpu_idx = knowhere::IndexFactory::Instance().Create<T>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

    std::vector<uint8_t> bitset_data((kDtypeNb + 7) / 8, 0);
    int64_t filtered_count = 0;
    std::mt19937 rng(bitset_seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int64_t r = 0; r < kDtypeNb; r++) {
        if (u(rng) < ratio) {
            bitset_data[r / 8] |= static_cast<uint8_t>(1 << (r % 8));
            filtered_count++;
        }
    }
    const bool has_filter = filtered_count > 0;
    knowhere::BitsetView bitset =
        has_filter ? knowhere::BitsetView(bitset_data.data(), kDtypeNb, filtered_count) : knowhere::BitsetView();

    auto gpu_res = gpu_idx.Search(query_ds, json, bitset);
    REQUIRE(gpu_res.has_value());
    const int64_t* gids = gpu_res.value()->GetIds();
    for (int64_t i = 0; i < kDtypeNq * kDtypeTopk; i++) {
        const int64_t id = gids[i];
        if (id < 0) {
            continue;
        }
        REQUIRE(id < kDtypeNb);
        if (has_filter) {
            REQUIRE_FALSE((bitset_data[id / 8] & (1 << (id % 8))) != 0);
        }
    }

    auto cpu_res = cpu_idx.Search(query_ds, json, bitset);
    REQUIRE(cpu_res.has_value());
    auto gt = knowhere::BruteForce::Search<T>(train_ds, query_ds, json, bitset);
    REQUIRE(gt.has_value());
    return {GetKNNRecall(*gt.value(), *gpu_res.value()), GetKNNRecall(*gt.value(), *cpu_res.value())};
}
}  // namespace

// FP32 flat storage -> generic <float,float,false> kernel. The config control.
TEST_CASE("GPU HNSW dtype: FP32 (L2/IP/COSINE)", "[gpu_hnsw_dtype_fp32]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);
    float recall = gpu_dtype_recall<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, dtype_json(metric, ""), version);
    REQUIRE(recall >= 0.85f);
}

// FP16 native 2-byte storage -> <__half,float,false> kernel. Built via HNSW_SQ
// on the fp32 node with SQ_TYPE=fp16 so the device selects the native half
// representation (the path the mock-wrapper fp16 tests do NOT reach).
TEST_CASE("GPU HNSW dtype: FP16 native (L2/IP/COSINE)", "[gpu_hnsw_dtype_fp16]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);
    float recall =
        gpu_dtype_recall<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, dtype_json(metric, "fp16"), version);
    REQUIRE(recall >= 0.85f);
}

// BF16 native 2-byte storage -> <__nv_bfloat16,float,false> kernel.
TEST_CASE("GPU HNSW dtype: BF16 native (L2/IP/COSINE)", "[gpu_hnsw_dtype_bf16]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);
    float recall =
        gpu_dtype_recall<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, dtype_json(metric, "bf16"), version);
    // bf16 has a 7-bit mantissa (coarser than fp16); slightly looser floor.
    REQUIRE(recall >= 0.80f);
}

// SQ8 storage -> decoded to fp32 on upload -> generic <float,float,false>
// kernel (distinct from the native half/bf16 paths above).
TEST_CASE("GPU HNSW dtype: SQ8 decoded (L2/IP/COSINE)", "[gpu_hnsw_dtype_sq8]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);
    float recall =
        gpu_dtype_recall<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, dtype_json(metric, "sq8"), version);
    // SQ8 is coarse (8-bit per component); conservative floor.
    REQUIRE(recall >= 0.65f);
}

// INT8 native 1-byte storage. IP with dim%4==0 takes the DP4A kernel
// (<int8_t,int8_t,true>); L2 takes the generic decoded path (<int8_t,float,
// false>). COSINE is re-encoded as fp16 in ToVanillaHnsw (direct_signed
// collapses normalized data) and the int8 query is widened to fp32, so cosine
// uses the fp16 generic path, not DP4A.
TEST_CASE("GPU HNSW dtype: INT8 native (L2/IP/COSINE)", "[gpu_hnsw_dtype_int8]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);
    float recall = gpu_dtype_recall<knowhere::int8>(knowhere::IndexEnum::INDEX_HNSW, dtype_json(metric, ""), version);
    REQUIRE(recall >= 0.85f);
}

// v84 regression: a CPU HNSW_SQ index must load into the *GPU_HNSW_SQ* node.
// GpuHnswSQIndexNode inherits Deserialize from GpuHnswIndexNode, which re-files
// the incoming payload under this node's own Type(). The SQ node's Type() is
// INDEX_GPU_HNSW_SQ and the base loader reads the payload via
// GetByName(Type()); the pre-v84 code aliased under a hardcoded
// INDEX_GPU_HNSW, so the SQ lookup missed and the load failed. Loading into the
// SQ node (not the base GPU_HNSW node) is what exercises the fix.
TEST_CASE("GPU HNSW_SQ node deserialize alias", "[gpu_hnsw_sq_alias]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::COSINE);
    CAPTURE(metric);

    auto json = dtype_json(metric, "sq8");
    auto train_ds = GenDataSet(kDtypeNb, kDtypeDim, kDtypeSeed);
    auto query_ds = GenDataSet(kDtypeNq, kDtypeDim, kDtypeSeed + 2);

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);

    // Load into the GPU_HNSW_SQ node specifically (not the base GPU_HNSW node).
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, nullptr);
    REQUIRE(gt.has_value());
    REQUIRE(GetKNNRecall(*gt.value(), *results.value()) >= 0.65f);
}

// v84: IP/COSINE distances must be returned as the true similarity (larger ==
// closer), matching faiss's METRIC_INNER_PRODUCT contract. The GPU kernel keeps
// a negated, min-first score internally and negates it back on copy-out;
// Knowhere no longer re-negates. This checks the numeric distance value (sign +
// magnitude) against the similarity recomputed from the returned neighbor, which
// recall-only checks cannot catch. A re-introduced double negation would flip
// every score negative and fail here.
TEST_CASE("GPU HNSW IP/COSINE distance sign", "[gpu_hnsw_ip_sign]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);
    const bool is_cosine = (metric == knowhere::metric::COSINE);

    auto json = dtype_json(metric, "");
    auto train_ds = GenDataSet(kDtypeNb, kDtypeDim, kDtypeSeed);
    auto query_ds = GenDataSet(kDtypeNq, kDtypeDim, kDtypeSeed + 2);
    const auto* train = reinterpret_cast<const float*>(train_ds->GetTensor());
    const auto* query = reinterpret_cast<const float*>(query_ds->GetTensor());

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);

    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());
    const int64_t* ids = results.value()->GetIds();
    const float* dist = results.value()->GetDistance();

    int positive_top1 = 0;
    for (int64_t q = 0; q < kDtypeNq; q++) {
        // Distances must be sorted best-first, i.e. descending for a
        // larger-is-closer similarity metric.
        for (int64_t j = 1; j < kDtypeTopk; j++) {
            REQUIRE(dist[q * kDtypeTopk + j] <= dist[q * kDtypeTopk + j - 1] + 1e-4f);
        }
        // Recompute the true similarity between the query and the returned
        // top-1 neighbor and compare to the reported distance.
        int64_t nn = ids[q * kDtypeTopk];
        REQUIRE(nn >= 0);
        REQUIRE(nn < kDtypeNb);
        double dot = 0.0, qn = 0.0, dn = 0.0;
        for (int64_t d = 0; d < kDtypeDim; d++) {
            double qv = query[q * kDtypeDim + d];
            double dv = train[nn * kDtypeDim + d];
            dot += qv * dv;
            qn += qv * qv;
            dn += dv * dv;
        }
        double expected = is_cosine ? dot / (std::sqrt(qn) * std::sqrt(dn) + 1e-12) : dot;
        double got = dist[q * kDtypeTopk];
        REQUIRE(std::fabs(got - expected) <= 1e-2 * (std::fabs(expected) + 1.0));
        if (got > 0.0)
            positive_top1++;
    }
    // For this data the nearest neighbor's similarity is positive for the vast
    // majority of queries; a sign inversion would make all of them negative.
    REQUIRE(positive_top1 > kDtypeNq * 0.8);
}

// v84: every requested result slot must be a real neighbor (no UINT64_MAX/-1 id
// or FLT_MAX sentinel). ef is guaranteed >= k: Knowhere's config bumps an unset
// ef to max(k, 16) and the GPU kernel additionally auto-clamps ef=max(ef,k) as
// defense-in-depth. Sentinel leakage into the last (k-ef) slots was the k>ef
// symptom. ef is intentionally left unset here so the requested topk drives it.
TEST_CASE("GPU HNSW returns k valid results (ef>=k)", "[gpu_hnsw_k_valid]") {
    const auto version = GenTestVersionList();
    const int64_t topk = 100;
    knowhere::Json json;
    json[knowhere::meta::DIM] = kDtypeDim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::HNSW_M] = 16;
    json[knowhere::indexparam::EFCONSTRUCTION] = 200;
    // EF intentionally unset: Knowhere sets ef = max(topk, 16) = topk here.

    auto train_ds = GenDataSet(kDtypeNb, kDtypeDim, kDtypeSeed);
    auto query_ds = GenDataSet(kDtypeNq, kDtypeDim, kDtypeSeed + 2);

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);

    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());
    const int64_t* ids = results.value()->GetIds();
    const float* dist = results.value()->GetDistance();
    for (int64_t i = 0; i < kDtypeNq * topk; i++) {
        REQUIRE(ids[i] >= 0);
        REQUIRE(ids[i] < kDtypeNb);
        REQUIRE(dist[i] < std::numeric_limits<float>::max());
    }
}

// Blocking prerequisite for GPU filtered search: the kernel tests the delete
// bitset by raw FAISS node id, so correctness hinges on
//     GPU node id == storage row offset == BitsetView bit index.
// This proves that mapping empirically with known ordered data and known
// filters, independent of recall. If this ever fails, filtered search silently
// excludes the wrong rows and every downstream filtered test is meaningless.
TEST_CASE("GPU HNSW G-ID mapping: node id == row offset == bitset index", "[gpu_hnsw_id_mapping]") {
    const auto version = GenTestVersionList();
    const int64_t nb = 2000;
    const int64_t dim = 32;

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    json[knowhere::meta::TOPK] = 1;
    json[knowhere::indexparam::HNSW_M] = 16;
    json[knowhere::indexparam::EFCONSTRUCTION] = 200;
    json[knowhere::indexparam::EF] = 200;

    // Randomly generated training data; regardless of order, every row is
    // trivially its own exact nearest neighbor (L2 self-distance 0), so an
    // unfiltered self-query for row i must return id i.
    auto train_ds = GenDataSet(nb, dim, 42);

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);

    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);
    REQUIRE(gpu_idx.Count() == nb);

    // 1) Unfiltered self-query: node id == storage row offset.
    {
        auto res = gpu_idx.Search(train_ds, json, nullptr);
        REQUIRE(res.has_value());
        const int64_t* ids = res.value()->GetIds();
        for (int64_t i = 0; i < nb; i++) {
            REQUIRE(ids[i] == i);
        }
    }

    // 2) Filtered self-query: bit index == storage row offset. Filter every
    //    third row; a returned id must never be a filtered bit, and a filtered
    //    row's own id must be excluded from its self-query.
    {
        std::vector<uint8_t> bitset_data((nb + 7) / 8, 0);
        int64_t filtered_count = 0;
        for (int64_t r = 0; r < nb; r += 3) {
            bitset_data[r / 8] |= static_cast<uint8_t>(1 << (r % 8));
            filtered_count++;
        }
        knowhere::BitsetView bitset(bitset_data.data(), nb, filtered_count);

        auto res = gpu_idx.Search(train_ds, json, bitset);
        REQUIRE(res.has_value());
        const int64_t* ids = res.value()->GetIds();
        for (int64_t i = 0; i < nb; i++) {
            const int64_t id = ids[i];
            if (id == -1) {
                continue;
            }
            const bool id_filtered = (bitset_data[id / 8] & (1 << (id % 8))) != 0;
            REQUIRE_FALSE(id_filtered);
            const bool query_filtered = (bitset_data[i / 8] & (1 << (i % 8))) != 0;
            if (query_filtered) {
                REQUIRE(id != i);
            }
        }
    }
}

// Filtered-search recall/exclusion parity across delete ratios and metrics.
// Exercises the two-tier beam + alpha gate + brute-force fallback. For every
// (metric, ratio) the helper asserts no returned id is a filtered bit; here we
// assert GPU recall tracks the CPU HNSW baseline (both vs the SAME-bitset brute
// force), which is the design's parity requirement. Parity is asserted by
// recall, not bit-identical traversal (the alpha gate is deterministic but not
// CPU encounter-order identical). ratio 0.0 exercises the HAS_FILTER=false fast
// path; 0.3/0.7 cover the mid-range where the alpha-gate order divergence is
// most pronounced; 0.95/0.99 trip the up-front brute-force fallback.
namespace {
void
check_filtered_ratios(const std::string& index_type, const knowhere::Json& json) {
    const auto version = GenTestVersionList();
    uint32_t seed = 123;
    for (const float ratio : {0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.95f, 0.99f}) {
        CAPTURE(ratio);
        auto [gpu_recall, cpu_recall] = gpu_filtered_recall<knowhere::fp32>(index_type, json, version, ratio, seed++);
        CAPTURE(gpu_recall);
        CAPTURE(cpu_recall);
        // Primary parity gate: GPU must not lag CPU HNSW by more than 0.05.
        REQUIRE(gpu_recall >= cpu_recall - 0.05f);
        if (ratio >= 0.93f) {
            // Up-front brute force is exact -> recall should be ~1.0.
            REQUIRE(gpu_recall >= 0.90f);
        }
    }
}
}  // namespace

TEST_CASE("GPU HNSW filtered search recall (FP32 L2/IP/COSINE)", "[gpu_hnsw_filtered_recall]") {
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);
    check_filtered_ratios(knowhere::IndexEnum::INDEX_HNSW, dtype_json(metric, ""));
}

// INT8 native storage: IP with dim%4==0 takes the DP4A brute-force + graph
// path; L2 takes the generic decoded path. COSINE is re-encoded as fp16 and
// uses the fp32-query path. Confirms the filter/BF kernels are threaded through
// both searchHostInt8 (IP) and the fp32 searchHost (L2, cosine).
TEST_CASE("GPU HNSW filtered search recall (INT8 native L2/IP/COSINE)", "[gpu_hnsw_filtered_recall_int8]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);
    uint32_t seed = 777;
    for (const float ratio : {0.0f, 0.5f, 0.9f, 0.99f}) {
        CAPTURE(ratio);
        auto [gpu_recall, cpu_recall] = gpu_filtered_recall<knowhere::int8>(
            knowhere::IndexEnum::INDEX_HNSW, dtype_json(metric, ""), version, ratio, seed++);
        CAPTURE(gpu_recall);
        CAPTURE(cpu_recall);
        REQUIRE(gpu_recall >= cpu_recall - 0.05f);
        if (ratio >= 0.93f) {
            REQUIRE(gpu_recall >= 0.90f);
        }
    }
}

// Edge cases the ratio sweep does not cover: everything filtered (no live rows)
// and k larger than the live-row count. Both must return sentinels for the
// missing slots rather than a filtered id or an error.
TEST_CASE("GPU HNSW filtered search edge cases", "[gpu_hnsw_filtered_edge]") {
    const auto version = GenTestVersionList();
    const int64_t nb = 4000;
    const int64_t dim = 64;
    const int64_t nq = 32;

    auto json = dtype_json(knowhere::metric::L2, "");
    json[knowhere::meta::TOPK] = 10;
    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 44);

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

    // 1) All rows filtered: every result slot must be a sentinel (-1), never a
    //    filtered id, and the search must not error.
    {
        std::vector<uint8_t> bitset_data((nb + 7) / 8, 0xFF);
        knowhere::BitsetView bitset(bitset_data.data(), nb, nb);
        auto res = gpu_idx.Search(query_ds, json, bitset);
        REQUIRE(res.has_value());
        const int64_t* ids = res.value()->GetIds();
        for (int64_t i = 0; i < nq * 10; i++) {
            REQUIRE(ids[i] == -1);
        }
    }

    // 2) k > live_count: only a handful of rows survive; the first `live` slots
    //    per query must be live ids, the remainder sentinels.
    {
        const int64_t live = 3;
        std::vector<uint8_t> bitset_data((nb + 7) / 8, 0xFF);
        for (int64_t r = 0; r < live; r++) {
            bitset_data[r / 8] &= static_cast<uint8_t>(~(1 << (r % 8)));
        }
        knowhere::BitsetView bitset(bitset_data.data(), nb, nb - live);
        auto res = gpu_idx.Search(query_ds, json, bitset);
        REQUIRE(res.has_value());
        const int64_t* ids = res.value()->GetIds();
        for (int64_t q = 0; q < nq; q++) {
            for (int64_t j = 0; j < 10; j++) {
                const int64_t id = ids[q * 10 + j];
                if (id == -1) {
                    continue;
                }
                REQUIRE(id < live);
            }
        }
    }
}

// Visited-bitmap OOM fix: the layer-0 search processes queries in nq-chunks so
// the grow-only visited bitmap (nq * ceil(N/32) * 4 bytes per scratch slot) is
// bounded regardless of batch size / search concurrency. Chunking must be
// result-invariant: each query is independent, and every per-query buffer
// (queries, entry points, neighbors, distances, visited bitmap, BF worklist) is
// indexed by the chunk-local block index against a caller-offset base pointer.
// This runs the SAME search twice -- once with the default cap (single pass) and
// once with GPU_HNSW_BITMAP_BYTES forced small enough to split the batch into
// many chunks -- and asserts the ids and distances match element-for-element.
namespace {
template <typename T>
void
check_chunk_parity(const std::string& index_type, const knowhere::Json& json, const knowhere::BitsetView& bitset) {
    using Catch::Approx;
    const auto version = GenTestVersionList();
    auto train_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(kDtypeNb, kDtypeDim, kDtypeSeed));
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<T>(GenDataSet(kDtypeNq, kDtypeDim, kDtypeSeed + 2));

    auto cpu_idx = knowhere::IndexFactory::Instance().Create<T>(index_type, version).value();
    REQUIRE(cpu_idx.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(cpu_idx.Serialize(bs) == knowhere::Status::success);
    auto gpu_idx = knowhere::IndexFactory::Instance().Create<T>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(bs) == knowhere::Status::success);

    // Baseline: default cap => chunk == nq (single pass).
    ::unsetenv("GPU_HNSW_BITMAP_BYTES");
    ::unsetenv("GPU_HNSW_BITMAP_MB");
    auto base = gpu_idx.Search(query_ds, json, bitset);
    REQUIRE(base.has_value());
    const int64_t n_out = static_cast<int64_t>(kDtypeNq) * kDtypeTopk;
    std::vector<int64_t> base_ids(base.value()->GetIds(), base.value()->GetIds() + n_out);
    std::vector<float> base_dist(base.value()->GetDistance(), base.value()->GetDistance() + n_out);

    // Force chunking: cap the bitmap to ~2 queries' worth so the batch is split
    // into ~nq/2 chunks, exercising the multi-launch offset math and the
    // per-chunk bitmap/worklist resets.
    const long per_query = static_cast<long>((kDtypeNb + 31) / 32) * 4;
    ::setenv("GPU_HNSW_BITMAP_BYTES", std::to_string(2 * per_query).c_str(), 1);
    auto chunked = gpu_idx.Search(query_ds, json, bitset);
    ::unsetenv("GPU_HNSW_BITMAP_BYTES");
    REQUIRE(chunked.has_value());
    const int64_t* cids = chunked.value()->GetIds();
    const float* cdist = chunked.value()->GetDistance();

    for (int64_t i = 0; i < n_out; i++) {
        REQUIRE(cids[i] == base_ids[i]);
        REQUIRE(cdist[i] == Approx(base_dist[i]).epsilon(1e-5));
    }
}

std::vector<uint8_t>
make_random_bitset(float ratio, uint32_t seed, int64_t& filtered_count) {
    std::vector<uint8_t> bd((kDtypeNb + 7) / 8, 0);
    filtered_count = 0;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int64_t r = 0; r < kDtypeNb; r++) {
        if (u(rng) < ratio) {
            bd[r / 8] |= static_cast<uint8_t>(1 << (r % 8));
            filtered_count++;
        }
    }
    return bd;
}
}  // namespace

TEST_CASE("GPU HNSW nq-chunking is result-invariant (visited-bitmap OOM fix)", "[gpu_hnsw_chunk_parity]") {
    const auto metric = GENERATE(knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    CAPTURE(metric);

    // Unfiltered (HAS_FILTER=false fast path).
    SECTION("fp32 unfiltered") {
        check_chunk_parity<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, dtype_json(metric, ""),
                                           knowhere::BitsetView());
    }
    // int8: IP with dim%4==0 takes the DP4A path, where the layer-0 (int8)
    // and upper-layer (fp32) query buffers are distinct, so BOTH chunk offsets
    // must be correct; L2 takes the generic decoded path and COSINE is
    // re-encoded as fp16 (fp32-query path).
    SECTION("int8 unfiltered") {
        check_chunk_parity<knowhere::int8>(knowhere::IndexEnum::INDEX_HNSW, dtype_json(metric, ""),
                                           knowhere::BitsetView());
    }
    // Mid-ratio filter: HAS_FILTER=true two-tier beam + per-chunk worklist reset.
    SECTION("fp32 filtered (~30%)") {
        int64_t fc = 0;
        auto bd = make_random_bitset(0.3f, 7, fc);
        knowhere::BitsetView bitset(bd.data(), kDtypeNb, fc);
        check_chunk_parity<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, dtype_json(metric, ""), bitset);
    }
    // High-ratio filter: trips the up-front brute-force fallback (also chunked).
    SECTION("fp32 filtered (~99% -> up-front BF)") {
        int64_t fc = 0;
        auto bd = make_random_bitset(0.99f, 9, fc);
        knowhere::BitsetView bitset(bd.data(), kDtypeNb, fc);
        check_chunk_parity<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, dtype_json(metric, ""), bitset);
    }
}

// Resident-set size (bytes) of this process, read from /proc/self/statm. Field 2
// is the resident page count; multiply by the page size. Linux-only, which is
// the CI/build platform for the GPU tests.
static size_t
CurrentResidentBytes() {
    std::ifstream statm("/proc/self/statm");
    size_t total_pages = 0;
    size_t resident_pages = 0;
    if (statm >> total_pages >> resident_pages) {
        return resident_pages * static_cast<size_t>(sysconf(_SC_PAGESIZE));
    }
    return 0;
}

// Regression guard for the int8-cosine GPU_HNSW load host-RAM peak (the OOM that
// killed querynodes under concurrent segment loads).
//
// int8-cosine cannot use the native direct-signed device path: L2-normalized
// components shrink as ~1/sqrt(d) and collapse onto a few of the 256 signed
// levels, destroying recall. So ToVanillaHnsw() reconstructs the segment to
// fp32, normalizes, and re-encodes to fp16 before the GPU upload. Done naively
// that materializes one n*d*4 fp32 buffer on top of the source index and the
// growing fp16 output, which is what blew past the load admission reservation
// and OOM-killed the pod. ToVanillaHnsw() streams the reconstruct in bounded
// row chunks so the transient fp32 staging is capped (~256 MiB) instead of the
// whole matrix; the resident peak is then (source index + fp16 output + one
// chunk) rather than (source + full fp32 + fp16 output).
//
// This measures actual resident-set growth during Deserialize() and asserts the
// peak stays within a bound the chunked path meets but the old full-buffer path
// does not. The CUDA context + shared GpuResources are warmed first so their
// (large, fixed) footprint is part of the baseline and excluded from the delta.
TEST_CASE("GPU HNSW int8 cosine load host-RAM peak is bounded", "[gpu_hnsw_load_mem]") {
    const auto version = GenTestVersionList();
    // nb*dim*4 must exceed the 256 MiB chunk so chunking actually engages and
    // the full-buffer path would allocate a distinctly larger fp32 matrix.
    const int64_t nb = 600000;
    const int64_t dim = 384;
    const int64_t seed = 42;

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
    build_json[knowhere::meta::TOPK] = 1;
    build_json[knowhere::indexparam::HNSW_M] = 16;
    build_json[knowhere::indexparam::EFCONSTRUCTION] = 100;
    build_json[knowhere::indexparam::EF] = 64;

    // Warm the CUDA context + shared GpuResources with a tiny int8-cosine load so
    // their resident footprint is established before we baseline.
    {
        auto warm_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(GenDataSet(2000, dim, seed + 9));
        auto warm_cpu =
            knowhere::IndexFactory::Instance().Create<knowhere::int8>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(warm_cpu.Build(warm_ds, build_json) == knowhere::Status::success);
        knowhere::BinarySet warm_bs;
        warm_cpu.Serialize(warm_bs);
        auto warm_gpu = knowhere::IndexFactory::Instance()
                            .Create<knowhere::int8>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                            .value();
        REQUIRE(warm_gpu.Deserialize(warm_bs) == knowhere::Status::success);
    }

    // Build the CPU int8-cosine HNSW, serialize, and drop the CPU index + the
    // training dataset so the baseline holds only the serialized bytes.
    knowhere::BinarySet bs;
    {
        auto train_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(GenDataSet(nb, dim, seed));
        auto cpu_idx =
            knowhere::IndexFactory::Instance().Create<knowhere::int8>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(cpu_idx.Build(train_ds, build_json) == knowhere::Status::success);
        cpu_idx.Serialize(bs);
    }

    uint64_t file_size = 0;
    for (const auto& kv : bs.binary_map_) {
        file_size += kv.second->size;
    }
    REQUIRE(file_size > 0);

    // Let deallocations settle, then baseline.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const size_t baseline = CurrentResidentBytes();
    REQUIRE(baseline > 0);

    // Device free VRAM before the load (context + shared GpuResources already
    // exist from the warm-up, and the warm index has been destroyed), so the
    // free delta after Deserialize isolates this one segment's VRAM footprint.
    size_t cuda_free_before = 0, cuda_total = 0;
    REQUIRE(cudaMemGetInfo(&cuda_free_before, &cuda_total) == cudaSuccess);

    std::atomic<bool> sampling{true};
    std::atomic<size_t> peak_rss{baseline};
    std::thread sampler([&] {
        while (sampling.load(std::memory_order_relaxed)) {
            const size_t r = CurrentResidentBytes();
            size_t prev = peak_rss.load(std::memory_order_relaxed);
            while (r > prev && !peak_rss.compare_exchange_weak(prev, r)) {
            }
            std::this_thread::sleep_for(std::chrono::microseconds(300));
        }
    });

    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::int8>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    const auto st = gpu_idx.Deserialize(bs);

    sampling.store(false, std::memory_order_relaxed);
    sampler.join();
    REQUIRE(st == knowhere::Status::success);

    // Measure retained VRAM while gpu_idx is still alive (the upload has
    // completed and the CPU copy is freed, so this is the steady-state device
    // footprint the querynode GPU admission must reserve per segment).
    size_t cuda_free_after = 0;
    REQUIRE(cudaMemGetInfo(&cuda_free_after, &cuda_total) == cudaSuccess);
    const size_t vram_delta = cuda_free_before > cuda_free_after ? cuda_free_before - cuda_free_after : 0;
    const double vram_ratio = static_cast<double>(vram_delta) / static_cast<double>(file_size);

    const size_t peak = peak_rss.load();
    const size_t peak_delta = peak > baseline ? peak - baseline : 0;
    const double ratio = static_cast<double>(peak_delta) / static_cast<double>(file_size);

    std::ostringstream os;
    os << "[gpu_hnsw_load_mem] file_size=" << file_size << "B baseline=" << baseline << "B peak=" << peak
       << "B peak_delta=" << peak_delta << "B ratio=" << ratio << "x (chunked expected ~3x, full-buffer ~5.5x)"
       << " | VRAM: delta=" << vram_delta << "B vram_ratio=" << vram_ratio
       << "x (per-segment device footprint for GPU admission sizing)";
    WARN(os.str());

    // Chunked reconstruct keeps the transient near (source + fp16 output + one
    // 256 MiB chunk). The old full-buffer path adds a whole n*d*4 fp32 matrix on
    // top, pushing the ratio well past this bound. 4.3x leaves margin for CUDA
    // pinned staging / allocator slack while still failing the full-buffer path.
    CHECK(ratio < 4.3);

    // The measured per-segment device footprint must stay under the 2x file
    // GPU reservation that StaticEstimateLoadResource reports (gpuMemoryCost),
    // so querynode GPU admission does not under-reserve VRAM. int8-cosine
    // measures ~1.7x (fp16 on device); the 2x reservation leaves headroom.
    CHECK(vram_ratio < 2.0);
}
#endif
