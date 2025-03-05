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
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/object.h"
#include "simd/hook.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.6f;
constexpr float kBruteForceRecallThreshold = 0.95f;
constexpr int kCosineMaxMissNum = 5;
}  // namespace

TEST_CASE("Test SCANN with data view refiner", "[float metrics]") {
    using Catch::Approx;
    auto version = GenTestVersionList();
    if (!faiss::support_pq_fast_scan) {
        SKIP("pass scann test");
    }

    const int64_t nb = 1000, nq = 10;
    auto metric = GENERATE(as<std::string>{}, knowhere::metric::COSINE, knowhere::metric::IP, knowhere::metric::L2);
    auto topk = GENERATE(as<int64_t>{}, 5, 120);
    auto dim = GENERATE(as<int64_t>{}, 120);

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto scann_gen = [base_gen, topk]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 12;
        json[knowhere::indexparam::REFINE_RATIO] = 4.0;
        json[knowhere::indexparam::SUB_DIM] = 2;
        json[knowhere::indexparam::WITH_RAW_DATA] = true;
        json[knowhere::indexparam::ENSURE_TOPK_FULL] = true;
        return json;
    };

    auto scann_gen1 = [base_gen, topk]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 12;
        json[knowhere::indexparam::REFINE_RATIO] = 4.0;
        json[knowhere::indexparam::SUB_DIM] = 2;
        json[knowhere::indexparam::WITH_RAW_DATA] = true;
        json[knowhere::indexparam::ENSURE_TOPK_FULL] = true;
        json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::BFLOAT16_QUANT;
        json[knowhere::indexparam::REFINE_WITH_QUANT] = true;
        return json;
    };

    auto scann_gen2 = [base_gen, topk]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 12;
        json[knowhere::indexparam::REFINE_RATIO] = 4.0;
        json[knowhere::indexparam::SUB_DIM] = 2;
        json[knowhere::indexparam::WITH_RAW_DATA] = true;
        json[knowhere::indexparam::ENSURE_TOPK_FULL] = true;
        json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::FLOAT16_QUANT;
        json[knowhere::indexparam::REFINE_WITH_QUANT] = true;
        return json;
    };

    auto scann_gen3 = [base_gen, topk]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 12;
        json[knowhere::indexparam::REFINE_RATIO] = 4.0;
        json[knowhere::indexparam::SUB_DIM] = 2;
        json[knowhere::indexparam::WITH_RAW_DATA] = true;
        json[knowhere::indexparam::ENSURE_TOPK_FULL] = true;
        json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::UINT8_QUANT;
        json[knowhere::indexparam::REFINE_WITH_QUANT] = true;
        return json;
    };

    auto scann_gen4 = [base_gen, topk]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 12;
        json[knowhere::indexparam::REFINE_RATIO] = 4.0;
        json[knowhere::indexparam::SUB_DIM] = 2;
        json[knowhere::indexparam::WITH_RAW_DATA] = true;
        json[knowhere::indexparam::ENSURE_TOPK_FULL] = true;
        json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::UINT8_QUANT;
        json[knowhere::indexparam::REFINE_WITH_QUANT] = false;
        return json;
    };

    auto rand = GENERATE(1);
    const auto train_ds = GenDataSet(nb, dim, rand);
    const auto query_ds = GenDataSet(nq, dim, rand + 777);

    auto gen = GENERATE_REF(as<std::function<knowhere::Json()>>{}, scann_gen1, scann_gen2, scann_gen3, scann_gen4);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
    knowhere::ViewDataOp data_view = [&train_ds, data_size = dim](size_t id) {
        auto data = (const float*)train_ds->GetTensor();
        return data + data_size * id;
    };
    auto data_view_pack = knowhere::Pack(data_view);
    SECTION("Accuraccy with refine") {
        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        auto scann_with_dv_refiner =
            knowhere::IndexFactory::Instance()
                .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, version, data_view_pack)
                .value();

        REQUIRE(scann_with_dv_refiner.Type() == knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR);
        REQUIRE(scann_with_dv_refiner.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(scann_with_dv_refiner.Count() == nb);
        REQUIRE(scann_with_dv_refiner.Size() > 0);
        REQUIRE(scann_with_dv_refiner.HasRawData(metric) == false);
        REQUIRE(scann_with_dv_refiner.HasRawData(metric) ==
                knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR,
                                                                       version, cfg_json));

        SECTION("knn search") {
            auto scann_with_dv_refiner_results = scann_with_dv_refiner.Search(query_ds, json, nullptr);
            REQUIRE(scann_with_dv_refiner_results.has_value());
            float recall = GetKNNRecall(*gt.value(), *scann_with_dv_refiner_results.value());
            REQUIRE(recall > kKnnRecallThreshold);

            if (metric == knowhere::metric::COSINE) {
                REQUIRE(CheckDistanceInScope(*scann_with_dv_refiner_results.value(), topk, -1.00001, 1.00001));
            }
        }

        SECTION("knn search with bitset") {
            std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
                GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
            const auto bitset_percentages = {0.22f, 0.98f};
            for (const float percentage : bitset_percentages) {
                for (const auto& gen_func : gen_bitset_funcs) {
                    auto bitset_data = gen_func(nb, percentage * nb);
                    knowhere::BitsetView bitset(bitset_data.data(), nb);
                    auto scann_with_dv_refiner_results = scann_with_dv_refiner.Search(query_ds, json, bitset);
                    auto gt_with_filter =
                        knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, json, bitset);
                    REQUIRE(scann_with_dv_refiner_results.has_value());
                    float recall = GetKNNRecall(*gt_with_filter.value(), *scann_with_dv_refiner_results.value());
                    REQUIRE(recall > kKnnRecallThreshold);
                    if (metric == knowhere::metric::COSINE) {
                        REQUIRE(CheckDistanceInScope(*scann_with_dv_refiner_results.value(), topk, -1.00001, 1.00001));
                    }
                }
            }
        }
    }
}

TEST_CASE("Ensure topk test", "[float metrics]") {
    using Catch::Approx;
    auto version = GenTestVersionList();
    if (!faiss::support_pq_fast_scan) {
        SKIP("pass scann test");
    }

    const int64_t nb = 10000, nq = 10;
    auto metric = GENERATE(as<std::string>{}, knowhere::metric::COSINE, knowhere::metric::IP, knowhere::metric::L2);
    auto topk = nb;
    auto dim = GENERATE(as<int64_t>{}, 120);

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto scann_gen = [base_gen, topk]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 512;
        json[knowhere::indexparam::NPROBE] = 1;
        json[knowhere::indexparam::REFINE_RATIO] = 1.0;
        json[knowhere::indexparam::SUB_DIM] = 2;
        json[knowhere::indexparam::WITH_RAW_DATA] = true;
        json[knowhere::indexparam::ENSURE_TOPK_FULL] = true;
        return json;
    };

    auto rand = GENERATE(1);
    const auto train_ds = GenDataSet(nb, dim, rand);
    const auto query_ds = GenDataSet(nq, dim, rand + 777);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    knowhere::ViewDataOp data_view = [&train_ds, data_size = sizeof(float) * dim](size_t id) {
        auto data = (const char*)train_ds->GetTensor();
        return data + data_size * id;
    };
    auto data_view_pack = knowhere::Pack(data_view);
    auto cfg_json = scann_gen().dump();
    knowhere::Json json = knowhere::Json::parse(cfg_json);

    auto scann_with_dv_refiner =
        knowhere::IndexFactory::Instance()
            .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, version, data_view_pack)
            .value();

    REQUIRE(scann_with_dv_refiner.Type() == knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR);
    REQUIRE(scann_with_dv_refiner.Build(train_ds, json) == knowhere::Status::success);
    REQUIRE(scann_with_dv_refiner.Count() == nb);
    REQUIRE(scann_with_dv_refiner.Size() > 0);
    REQUIRE(scann_with_dv_refiner.HasRawData(metric) == false);
    REQUIRE(scann_with_dv_refiner.HasRawData(metric) ==
            knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, version,
                                                                   cfg_json));
    auto scann_with_dv_refiner_results = scann_with_dv_refiner.Search(query_ds, json, nullptr);
    auto res_ids = scann_with_dv_refiner_results.value()->GetIds();
    // check we can get all vectors in (topk = nb, nprobe = 1 )
    for (auto i = 0; i < nq * topk; i++) {
        REQUIRE(res_ids[i] != -1);
    }
}

template <typename DataType>
void
BaseTest(const knowhere::DataSetPtr train_ds, const knowhere::DataSetPtr query_ds, const int64_t k,
         const knowhere::MetricType metric, const knowhere::Json& conf, const float loss_range = 1.00001) {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto base = knowhere::ConvertToDataTypeIfNeeded<DataType>(train_ds);
    auto query = knowhere::ConvertToDataTypeIfNeeded<DataType>(query_ds);
    auto dim = base->GetDim();
    auto nq = query->GetRows();

    auto knn_gt = knowhere::BruteForce::Search<DataType>(base, query, conf, nullptr);
    knowhere::ViewDataOp data_view = [&base, data_size = dim](size_t id) {
        auto data = (const DataType*)base->GetTensor();
        return data + data_size * id;
    };
    auto data_view_pack = knowhere::Pack(data_view);
    auto scann_with_dv_refiner =
        knowhere::IndexFactory::Instance()
            .Create<DataType>(knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, version, data_view_pack)
            .value();

    REQUIRE(scann_with_dv_refiner.Type() == knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR);
    REQUIRE(scann_with_dv_refiner.Build(base, conf) == knowhere::Status::success);

    REQUIRE(scann_with_dv_refiner.Size() > 0);
    REQUIRE(scann_with_dv_refiner.HasRawData(metric) == false);
    REQUIRE(scann_with_dv_refiner.HasRawData(metric) == knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(
                                                            knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, version, conf));

    // knn search
    auto scann_with_dv_refiner_results = scann_with_dv_refiner.Search(query, conf, nullptr);
    REQUIRE(scann_with_dv_refiner_results.has_value());
    float recall = GetKNNRecall(*knn_gt.value(), *scann_with_dv_refiner_results.value());

    if (metric == knowhere::metric::COSINE) {
        REQUIRE(CheckDistanceInScope(*scann_with_dv_refiner_results.value(), k, -loss_range, loss_range));
    }
    REQUIRE(recall > kKnnRecallThreshold);
    // range search
    auto scann_with_dv_refiner_range_results = scann_with_dv_refiner.RangeSearch(query, conf, nullptr);
    REQUIRE(scann_with_dv_refiner_range_results.has_value());
    auto scann_with_dv_ids = scann_with_dv_refiner_range_results.value()->GetIds();
    auto scann_with_dv_lims = scann_with_dv_refiner_range_results.value()->GetLims();
    if (metric == knowhere::metric::L2 || metric == knowhere::metric::COSINE) {
        for (int i = 0; i < nq; ++i) {
            CHECK(scann_with_dv_ids[scann_with_dv_lims[i]] == i);
        }
    }
}

TEST_CASE("Test difference dim with difference data type", "[multi metrics]") {
    if (!faiss::support_pq_fast_scan) {
        SKIP("pass scann test");
    }
    const int64_t nb = 1000, nq = 10;
    auto metric = GENERATE(as<std::string>{}, knowhere::metric::COSINE, knowhere::metric::IP, knowhere::metric::L2);
    auto topk = GENERATE(as<int64_t>{}, 10);
    auto dim = GENERATE(as<int64_t>{}, 31, 511, 1024);

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.9;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto scann_gen = [base_gen, topk]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 24;
        json[knowhere::indexparam::NPROBE] = 16;
        json[knowhere::indexparam::REFINE_RATIO] = 4.0;
        json[knowhere::indexparam::SUB_DIM] = 2;
        return json;
    };

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);
    auto cfg_json = scann_gen().dump();
    knowhere::Json json = knowhere::Json::parse(cfg_json);
    BaseTest<knowhere::fp32>(train_ds, query_ds, topk, metric, json);
    BaseTest<knowhere::bf16>(train_ds, query_ds, topk, metric, json);
    BaseTest<knowhere::fp16>(train_ds, query_ds, topk, metric, json);
}

TEST_CASE("Test fp16/bf16 with quant refine", "[multi metrics]") {
    if (!faiss::support_pq_fast_scan) {
        SKIP("pass scann test");
    }
    const int64_t nb = 1000, nq = 1;
    auto metric = GENERATE(as<std::string>{}, knowhere::metric::COSINE, knowhere::metric::IP, knowhere::metric::L2);
    auto topk = GENERATE(as<int64_t>{}, 10);
    auto dim = GENERATE(as<int64_t>{}, 120);

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.9;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto scann_gen = [base_gen, topk]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 24;
        json[knowhere::indexparam::NPROBE] = 16;
        json[knowhere::indexparam::REFINE_RATIO] = 4.0;
        json[knowhere::indexparam::SUB_DIM] = 2;
        return json;
    };

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);
    // data type == bfloat16
    // with refine_type = bfloat16, refine_with_quant = false
    {
        auto bf16_json = scann_gen();
        bf16_json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::BFLOAT16_QUANT;
        bf16_json[knowhere::indexparam::REFINE_WITH_QUANT] = false;
        BaseTest<knowhere::bf16>(train_ds, query_ds, topk, metric, bf16_json);
    }
    // with refine_type = bfloat16, refine_with_quant = true
    {
        // some precision loss by the difference between knowhere fp32->bf16 and faiss fp32->bf16
        auto bf16_json = scann_gen();
        bf16_json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::BFLOAT16_QUANT;
        bf16_json[knowhere::indexparam::REFINE_WITH_QUANT] = true;

        BaseTest<knowhere::bf16>(train_ds, query_ds, topk, metric, bf16_json, 1.0001);
    }
    // with refine_type = uint8, refine_with_quant = false
    {
        auto bf16_json = scann_gen();
        bf16_json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::UINT8_QUANT;
        bf16_json[knowhere::indexparam::REFINE_WITH_QUANT] = false;
        BaseTest<knowhere::bf16>(train_ds, query_ds, topk, metric, bf16_json);
    }
    // with refine_type = uint8, refine_with_quant = true
    {
        auto bf16_json = scann_gen();
        bf16_json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::UINT8_QUANT;
        bf16_json[knowhere::indexparam::REFINE_WITH_QUANT] = true;
        BaseTest<knowhere::bf16>(train_ds, query_ds, topk, metric, bf16_json, 1.001);
    }
    // with refine_type = float16, refine_with_quant = false
    {
        auto fp16_json = scann_gen();
        fp16_json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::FLOAT16_QUANT;
        fp16_json[knowhere::indexparam::REFINE_WITH_QUANT] = false;
        BaseTest<knowhere::fp16>(train_ds, query_ds, topk, metric, fp16_json);
    }
    // with refine_type = float16, refine_with_quant = true
    {
        // some precision loss by the difference between knowhere fp32->bf16 and faiss fp32->bf16
        auto fp16_json = scann_gen();
        fp16_json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::FLOAT16_QUANT;
        fp16_json[knowhere::indexparam::REFINE_WITH_QUANT] = true;
        BaseTest<knowhere::fp16>(train_ds, query_ds, topk, metric, fp16_json, 1.0001);
    }
    // with refine_type = uint8, refine_with_quant = false
    {
        auto fp16_json = scann_gen();
        fp16_json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::UINT8_QUANT;
        fp16_json[knowhere::indexparam::REFINE_WITH_QUANT] = false;
        BaseTest<knowhere::fp16>(train_ds, query_ds, topk, metric, fp16_json);
    }
    // with refine_type = uint8, refine_with_quant = true
    {
        auto fp16_json = scann_gen();
        fp16_json[knowhere::indexparam::REFINE_TYPE] = knowhere::RefineType::UINT8_QUANT;
        fp16_json[knowhere::indexparam::REFINE_WITH_QUANT] = true;
        BaseTest<knowhere::fp16>(train_ds, query_ds, topk, metric, fp16_json, 1.001);
    }
}
