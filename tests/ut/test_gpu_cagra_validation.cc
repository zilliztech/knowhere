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

#include <random>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"

#ifdef KNOWHERE_WITH_CUVS

TEST_CASE("Test INT8 CAGRA CPU adaptation validation", "[config][gpu][cagra]") {
    const auto name = GENERATE(knowhere::IndexEnum::INDEX_GPU_CAGRA, knowhere::IndexEnum::INDEX_CUVS_CAGRA);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const auto adaptation = GENERATE(-1, 0, 1);
    CAPTURE(name, adaptation);
    knowhere::Json config{{knowhere::meta::DIM, 32}, {knowhere::meta::METRIC_TYPE, knowhere::metric::L2}};
    if (adaptation != -1) {
        config[knowhere::indexparam::ADAPT_FOR_CPU] = adaptation == 1;
    }
    const auto expected = adaptation == 1 ? knowhere::Status::invalid_args : knowhere::Status::success;
    std::string message;
    REQUIRE(knowhere::IndexStaticFaced<knowhere::int8>::ConfigCheck(name, version, config, message) == expected);
    if (adaptation == 1) {
        REQUIRE(message.find("adapt_for_cpu") != std::string::npos);
        REQUIRE(message.find("INT8") != std::string::npos);
    }

    auto resource =
        knowhere::IndexStaticFaced<knowhere::int8>::EstimateLoadResource(name, version, 4096, 32, 32, config);
    REQUIRE(resource.has_value() == (adaptation != 1));
    if (adaptation == 1) {
        REQUIRE(resource.error() == knowhere::Status::invalid_args);
        REQUIRE(resource.what().find("INT8") != std::string::npos);
    }

    for (const auto stage : {knowhere::TRAIN, knowhere::DESERIALIZE, knowhere::STATIC}) {
        CAPTURE(stage);
        auto int8_config = knowhere::IndexStaticFaced<knowhere::int8>::CreateConfig(name, version);
        REQUIRE(knowhere::Config::Load(*int8_config, config, stage) == expected);
        auto float_config = knowhere::IndexStaticFaced<knowhere::fp32>::CreateConfig(name, version);
        REQUIRE(knowhere::Config::Load(*float_config, config, stage) == knowhere::Status::success);
    }

    if (adaptation == 1) {
        auto index = knowhere::IndexFactory::Instance().Create<knowhere::int8>(name, version).value();
        // Invalid load options must be rejected before reading the artifact or allocating GPU memory.
        REQUIRE(index.Deserialize(knowhere::BinarySet{}, config) == knowhere::Status::invalid_args);
    }
}

TEST_CASE("Test INT8 CAGRA rejects CPU build and load", "[gpu][cagra]") {
    const auto name = GENERATE(knowhere::IndexEnum::INDEX_GPU_CAGRA, knowhere::IndexEnum::INDEX_CUVS_CAGRA);
    const auto explicit_false = GENERATE(false, true);
    CAPTURE(name, explicit_false);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    constexpr int64_t rows = 512;
    constexpr int64_t dim = 32;
    std::mt19937 generator(42);
    std::uniform_int_distribution<int> distribution(-16, 16);
    std::vector<knowhere::int8> vectors(rows * dim);
    for (auto& value : vectors) {
        value = static_cast<knowhere::int8>(distribution(generator));
    }
    auto dataset = knowhere::GenDataSet(rows, dim, vectors.data());
    dataset->SetIsOwner(false);
    knowhere::Json config{{knowhere::meta::DIM, dim},
                          {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
                          {knowhere::meta::TOPK, 1},
                          {knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE, 64},
                          {knowhere::indexparam::GRAPH_DEGREE, 32},
                          {knowhere::indexparam::CACHE_DATASET_ON_DEVICE, true}};
    if (explicit_false) {
        config[knowhere::indexparam::ADAPT_FOR_CPU] = false;
    }
    auto cpu_config = config;
    cpu_config[knowhere::indexparam::ADAPT_FOR_CPU] = true;
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::int8>(name, version).value();
    REQUIRE(index.Train(dataset, cpu_config) == knowhere::Status::invalid_args);
    REQUIRE(index.Build(dataset, cpu_config) == knowhere::Status::invalid_args);
    REQUIRE(index.Build(dataset, config) == knowhere::Status::success);
    knowhere::BinarySet binary;
    REQUIRE(index.Serialize(binary) == knowhere::Status::success);

    auto loaded = knowhere::IndexFactory::Instance().Create<knowhere::int8>(name, version).value();
    REQUIRE(loaded.Deserialize(binary, cpu_config) == knowhere::Status::invalid_args);
    REQUIRE(loaded.Deserialize(binary, config) == knowhere::Status::success);
    REQUIRE(loaded.Count() == rows);
    auto query = knowhere::GenDataSet(1, dim, vectors.data());
    query->SetIsOwner(false);
    auto result = loaded.Search(query, config, nullptr);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetIds()[0] == 0);
    REQUIRE(result.value()->GetDistance()[0] == 0.0f);
}

#endif
