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
#include "knowhere/utils.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.6f;
constexpr float kBruteForceRecallThreshold = 0.99f;
}  // namespace
template <typename data_type>
void
BaseSearchTest() {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);
    auto topk = GENERATE(as<int64_t>{}, 5, 120);
    auto version = GenTestVersionList();

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto ivfflat_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 8;
        return json;
    };

    auto ivfflatcc_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::SSIZE] = 48;
        return json;
    };

    auto ivfsq_gen = ivfflat_gen;

    auto flat_gen = base_gen;

    auto ivfpq_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::M] = 4;
        json[knowhere::indexparam::NBITS] = 8;
        return json;
    };

    auto scann_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::NPROBE] = 14;
        json[knowhere::indexparam::REORDER_K] = 500;
        json[knowhere::indexparam::WITH_RAW_DATA] = true;
        return json;
    };

    auto scann_gen2 = [scann_gen]() {
        knowhere::Json json = scann_gen();
        json[knowhere::indexparam::WITH_RAW_DATA] = false;
        return json;
    };

    auto hnsw_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::HNSW_M] = 128;
        json[knowhere::indexparam::EFCONSTRUCTION] = 200;
        json[knowhere::indexparam::EF] = 200;
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
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen2),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<data_type>(name, version);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
        REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);

        auto results = idx.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        bool scann_without_raw_data =
            (name == knowhere::IndexEnum::INDEX_FAISS_SCANN && scann_gen2().dump() == cfg_json);
        if (name != knowhere::IndexEnum::INDEX_FAISS_IVFPQ && !scann_without_raw_data) {
            REQUIRE(recall > kKnnRecallThreshold);
        }

        if (metric == knowhere::metric::COSINE) {
            if (name != knowhere::IndexEnum::INDEX_FAISS_IVFSQ8 && name != knowhere::IndexEnum::INDEX_FAISS_IVFPQ &&
                !scann_without_raw_data) {
                REQUIRE(CheckDistanceInScope(*results.value(), topk, -1.00001, 1.00001));
            }
        }
    }

    SECTION("Test half-float Range Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen2),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<data_type>(name, version);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
        REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);

        auto results = idx.RangeSearch(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
        auto ids = results.value()->GetIds();
        auto lims = results.value()->GetLims();
        auto dis = results.value()->GetDistance();
        bool scann_without_raw_data =
            (name == knowhere::IndexEnum::INDEX_FAISS_SCANN && scann_gen2().dump() == cfg_json);
        if (name != knowhere::IndexEnum::INDEX_FAISS_IVFPQ && name != knowhere::IndexEnum::INDEX_FAISS_SCANN) {
            for (int i = 0; i < nq; ++i) {
                CHECK(ids[lims[i]] == i);
            }
        }

        if (metric == knowhere::metric::COSINE) {
            if (name != knowhere::IndexEnum::INDEX_FAISS_IVFSQ8 && name != knowhere::IndexEnum::INDEX_FAISS_IVFPQ &&
                !scann_without_raw_data) {
                REQUIRE(CheckDistanceInScope(*results.value(), -1.00001, 1.00001));
            }
        }
    }

    SECTION("Test half-float Search with Bitset") {
        using std::make_tuple;
        auto [name, gen, threshold] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>, float>({
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen, hnswlib::kHnswSearchKnnBFFilterThreshold),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<data_type>(name, version);
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
                auto results = idx.Search(*query_ds, json, bitset);
                auto gt = knowhere::BruteForce::Search<data_type>(train_ds, query_ds, json, bitset);
                float recall = GetKNNRecall(*gt.value(), *results.value());
                if (percentage > threshold) {
                    REQUIRE(recall > kBruteForceRecallThreshold);
                } else {
                    REQUIRE(recall > kKnnRecallThreshold);
                }
            }
        }
    }

    SECTION("Test Serialize/Deserialize") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IDMAP, flat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, ivfflat_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivfflatcc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_SCANN, scann_gen2),
            make_tuple(knowhere::IndexEnum::INDEX_HNSW, hnsw_gen),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create<data_type>(name, version);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(*train_ds, json) == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);

        auto idx_ = knowhere::IndexFactory::Instance().Create<data_type>(name, version);
        idx_.Deserialize(bs);
        auto results = idx_.Search(*query_ds, json, nullptr);
        REQUIRE(results.has_value());
    }
}

TEST_CASE("Test Mem Index With fp16/bf16 Vector", "[float metrics]") {
    BaseSearchTest<knowhere::fp16>();
    BaseSearchTest<knowhere::bf16>();
}
