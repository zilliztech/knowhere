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

#ifdef KNOWHERE_WITH_SVS

#include <tuple>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/svs/IndexSVSIVF.h"
#include "faiss/svs/IndexSVSVamana.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "utils.h"

TEST_CASE("Test SVS Flat Build and Search", "[svs][flat]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 5;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    // golden results from brute force
    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());
    auto gt_ids = gt.value()->GetIds();

    auto gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    SECTION("Build and KNN Search") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_FLAT, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        REQUIRE(index.Type() == knowhere::IndexEnum::INDEX_SVS_FLAT);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Size() > 0);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());
        auto ids = res.value()->GetIds();
        auto dist = res.value()->GetDistance();

        // brute-force index should have perfect recall
        for (int64_t i = 0; i < nq; i++) {
            REQUIRE(ids[i * topk] == gt_ids[i * topk]);
        }
    }

    SECTION("Search with bitset returns error") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_FLAT, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

        // SVS Flat does not support bitset filtering
        auto filter_bits = GenerateBitsetWithFirstTbitsSet(nb, nb / 2);
        knowhere::BitsetView bitset(filter_bits.data(), nb);

        auto res = index.Search(query_ds, json, bitset);
        REQUIRE(!res.has_value());
        REQUIRE(res.error() == knowhere::Status::not_implemented);
    }

    SECTION("Serialize and Deserialize") {
        knowhere::BinarySet bs;

        // build and serialize
        {
            auto idx =
                knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_FLAT, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
            REQUIRE(index.Serialize(bs) == knowhere::Status::success);
        }

        // deserialize and search
        {
            auto idx =
                knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_FLAT, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Deserialize(bs, json) == knowhere::Status::success);
            REQUIRE(index.Count() == nb);
            REQUIRE(index.Dim() == dim);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());
            auto ids = res.value()->GetIds();

            // results should match original brute-force golden
            for (int64_t i = 0; i < nq; i++) {
                REQUIRE(ids[i * topk] == gt_ids[i * topk]);
            }
        }
    }
}

TEST_CASE("Test SVS Vamana Build and Search", "[svs][vamana]") {
    const int64_t nb = 10000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    // golden results from brute force
    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = 64;
        json[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = 200;
        json[knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE] = 40;
        json[knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY] = 40;
        json[knowhere::indexparam::SVS_ALPHA] = 1.2f;
        json[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("fp32");
        json[knowhere::indexparam::SVS_IS_STATIC] = false;
        return json;
    };

    SECTION("Build and KNN Search") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        REQUIRE(index.Type() == knowhere::IndexEnum::INDEX_SVS_VAMANA);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Size() > 0);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS Vamana recall@" << topk << " = " << recall << " (metric=" << metric << ")";
        REQUIRE(recall >= 0.80f);
    }

    SECTION("Search with bitset") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

        // mask first half of vectors
        auto filter_bits = GenerateBitsetWithFirstTbitsSet(nb, nb / 2);
        knowhere::BitsetView bitset(filter_bits.data(), nb);

        auto res = index.Search(query_ds, json, bitset);
        REQUIRE(res.has_value());

        auto ids = res.value()->GetIds();
        for (int64_t i = 0; i < nq * topk; i++) {
            if (ids[i] >= 0) {
                REQUIRE(ids[i] >= nb / 2);
            }
        }
    }

    SECTION("Serialize and Deserialize") {
        knowhere::BinarySet bs;

        // build and serialize
        knowhere::DataSetPtr first_result;
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA,
                                                                                 version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());
            first_result = res.value();

            REQUIRE(index.Serialize(bs) == knowhere::Status::success);
        }

        // deserialize and search
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA,
                                                                                 version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Deserialize(bs, json) == knowhere::Status::success);
            REQUIRE(index.Count() == nb);
            REQUIRE(index.Dim() == dim);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());

            // results should match pre-serialization
            auto ids_before = first_result->GetIds();
            auto ids_after = res.value()->GetIds();
            for (int64_t i = 0; i < nq * topk; i++) {
                REQUIRE(ids_before[i] == ids_after[i]);
            }
        }
    }
}

TEST_CASE("Test SVS Vamana LeanVec OOD Build and Search", "[svs][vamana][leanvec][ood]") {
    if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
        SKIP("LVQ/LeanVec not available in this SVS runtime build");
    }

    const int64_t nb = 10000, nq = 10, n_train_q = 2000;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);
    // representative query sample for OOD projection training (distinct seed from db/queries)
    const auto train_query_ds = GenDataSet(n_train_q, dim, /*seed=*/7);

    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = 64;
        json[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = 200;
        json[knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE] = 40;
        json[knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY] = 40;
        json[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("leanvec4x8");
        json[knowhere::indexparam::SVS_LEANVEC_OOD] = true;
        json[knowhere::indexparam::SVS_IS_STATIC] = false;
        return json;
    };

    // attach the query-training sample to the train dataset for OOD training
    train_ds->Set(knowhere::meta::TRAIN_QUERY_TENSOR, static_cast<const float*>(train_query_ds->GetTensor()));
    train_ds->Set(knowhere::meta::TRAIN_QUERY_ROWS, static_cast<int64_t>(n_train_q));

    SECTION("Build and KNN Search") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        REQUIRE(index.Type() == knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Size() > 0);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS Vamana LeanVec OOD recall@" << topk << " = " << recall << " (metric=" << metric
                           << ")";
        REQUIRE(recall >= 0.70f);
    }

    SECTION("Build and KNN Search without a query sample") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        // OOD is the default and must work when no query sample is attached
        const auto no_query_ds = CopyDataSet(train_ds, nb);
        REQUIRE(index.Build(no_query_ds, json) == knowhere::Status::success);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS Vamana LeanVec OOD (no query sample) recall@" << topk << " = " << recall
                           << " (metric=" << metric << ")";
        REQUIRE(recall >= 0.70f);
    }

    SECTION("Serialize and Deserialize preserves OOD training data") {
        knowhere::BinarySet bs;
        knowhere::DataSetPtr first_result;
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());
            first_result = res.value();

            REQUIRE(index.Serialize(bs) == knowhere::Status::success);
        }
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Deserialize(bs, json) == knowhere::Status::success);
            REQUIRE(index.Count() == nb);
            REQUIRE(index.Dim() == dim);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());

            auto ids_before = first_result->GetIds();
            auto ids_after = res.value()->GetIds();
            for (int64_t i = 0; i < nq * topk; i++) {
                REQUIRE(ids_before[i] == ids_after[i]);
            }
        }
    }
}

TEST_CASE("Test SVS IVF Build and Search", "[svs][ivf]") {
    const int64_t nb = 10000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::SVS_IVF_NLIST] = 256;
        json[knowhere::indexparam::SVS_IVF_NPROBE] = 48;
        json[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("fp32");
        return json;
    };

    SECTION("Build and KNN Search") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_IVF, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        REQUIRE(index.Type() == knowhere::IndexEnum::INDEX_SVS_IVF);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Size() > 0);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS IVF recall@" << topk << " = " << recall << " (metric=" << metric << ")";
        REQUIRE(recall >= 0.70f);
    }

    SECTION("Search with bitset returns error") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_IVF, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

        // SVS IVF does not support bitset filtering
        auto filter_bits = GenerateBitsetWithFirstTbitsSet(nb, nb / 2);
        knowhere::BitsetView bitset(filter_bits.data(), nb);

        auto res = index.Search(query_ds, json, bitset);
        REQUIRE(!res.has_value());
        REQUIRE(res.error() == knowhere::Status::not_implemented);
    }

    SECTION("Train and Add are rejected as unsupported") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_IVF, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        // SVS IVF is immutable: it must never silently accept rows it does not ingest.
        REQUIRE(index.Train(train_ds, json) == knowhere::Status::not_implemented);
        REQUIRE(index.Add(train_ds, json) == knowhere::Status::not_implemented);

        // Add after a successful Build must also be rejected rather than dropping the rows.
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Count() == nb);
        REQUIRE(index.Add(train_ds, json) == knowhere::Status::not_implemented);
        REQUIRE(index.Count() == nb);
    }

    SECTION("Build honors svs_is_static and k_reorder") {
        auto is_static = GENERATE(true, false);
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_IVF, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        json[knowhere::indexparam::SVS_IS_STATIC] = is_static;
        // k_reorder is a build-time as well as a search-time parameter
        json[knowhere::indexparam::SVS_IVF_K_REORDER] = 2.0f;

        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS IVF recall@" << topk << " = " << recall << " (metric=" << metric
                           << ", is_static=" << is_static << ")";
        REQUIRE(recall >= 0.70f);
    }

    SECTION("Serialize and Deserialize") {
        knowhere::BinarySet bs;
        knowhere::DataSetPtr first_result;
        {
            auto idx =
                knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_IVF, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());
            first_result = res.value();

            REQUIRE(index.Serialize(bs) == knowhere::Status::success);
        }
        {
            auto idx =
                knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_IVF, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Deserialize(bs, json) == knowhere::Status::success);
            REQUIRE(index.Count() == nb);
            REQUIRE(index.Dim() == dim);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());

            auto ids_before = first_result->GetIds();
            auto ids_after = res.value()->GetIds();
            for (int64_t i = 0; i < nq * topk; i++) {
                REQUIRE(ids_before[i] == ids_after[i]);
            }
        }
    }
}

TEST_CASE("Test SVS IVF LeanVec OOD Build and Search", "[svs][ivf][leanvec][ood]") {
    if (!faiss::IndexSVSIVF::is_lvq_leanvec_enabled()) {
        SKIP("LVQ/LeanVec not available in this SVS runtime build");
    }

    const int64_t nb = 10000, nq = 10, n_train_q = 2000;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);
    // representative query sample for OOD projection training (distinct seed from db/queries)
    const auto train_query_ds = GenDataSet(n_train_q, dim, /*seed=*/7);

    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::SVS_IVF_NLIST] = 256;
        json[knowhere::indexparam::SVS_IVF_NPROBE] = 48;
        json[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("leanvec4x8");
        json[knowhere::indexparam::SVS_LEANVEC_OOD] = true;
        return json;
    };

    // attach the query-training sample to the train dataset for OOD training
    train_ds->Set(knowhere::meta::TRAIN_QUERY_TENSOR, static_cast<const float*>(train_query_ds->GetTensor()));
    train_ds->Set(knowhere::meta::TRAIN_QUERY_ROWS, static_cast<int64_t>(n_train_q));

    SECTION("Build and KNN Search") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_IVF_LEANVEC,
                                                                             version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        REQUIRE(index.Type() == knowhere::IndexEnum::INDEX_SVS_IVF_LEANVEC);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Size() > 0);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS IVF LeanVec OOD recall@" << topk << " = " << recall << " (metric=" << metric << ")";
        REQUIRE(recall >= 0.70f);
    }

    SECTION("Build and KNN Search without a query sample") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_IVF_LEANVEC,
                                                                             version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        // OOD is the default and must work when no query sample is attached
        const auto no_query_ds = CopyDataSet(train_ds, nb);
        REQUIRE(index.Build(no_query_ds, json) == knowhere::Status::success);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS IVF LeanVec OOD (no query sample) recall@" << topk << " = " << recall
                           << " (metric=" << metric << ")";
        REQUIRE(recall >= 0.70f);
    }

    SECTION("Serialize and Deserialize preserves OOD training data") {
        knowhere::BinarySet bs;
        knowhere::DataSetPtr first_result;
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_SVS_IVF_LEANVEC, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());
            first_result = res.value();

            REQUIRE(index.Serialize(bs) == knowhere::Status::success);
        }
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
                knowhere::IndexEnum::INDEX_SVS_IVF_LEANVEC, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Deserialize(bs, json) == knowhere::Status::success);
            REQUIRE(index.Count() == nb);
            REQUIRE(index.Dim() == dim);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());

            auto ids_before = first_result->GetIds();
            auto ids_after = res.value()->GetIds();
            for (int64_t i = 0; i < nq * topk; i++) {
                REQUIRE(ids_before[i] == ids_after[i]);
            }
        }
    }
}

TEST_CASE("Test SVS Static Vamana Build and Search", "[svs][vamana][static]") {
    const int64_t nb = 10000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = 64;
        json[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = 200;
        json[knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE] = 40;
        json[knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY] = 40;
        json[knowhere::indexparam::SVS_ALPHA] = 1.2f;
        json[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("fp32");
        json[knowhere::indexparam::SVS_IS_STATIC] = true;
        return json;
    };

    SECTION("Build and KNN Search") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Size() > 0);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS Static Vamana recall@" << topk << " = " << recall << " (metric=" << metric << ")";
        REQUIRE(recall >= 0.80f);
    }

    SECTION("Incremental Add is rejected for a static index") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        // A static index takes all of its data in one shot; a second Add must fail cleanly.
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Count() == nb);

        const auto extra_ds = GenDataSet(100, dim, /*seed=*/11);
        REQUIRE(index.Add(extra_ds, json) == knowhere::Status::not_implemented);
        REQUIRE(index.Count() == nb);
    }

    SECTION("Dynamic index still supports incremental Add") {
        auto idx =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        json[knowhere::indexparam::SVS_IS_STATIC] = false;

        const int64_t extra = 100;
        const auto extra_ds = GenDataSet(extra, dim, /*seed=*/11);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Count() == nb);
        REQUIRE(index.Add(extra_ds, json) == knowhere::Status::success);
        REQUIRE(index.Count() == nb + extra);
    }

    SECTION("Serialize and Deserialize") {
        knowhere::BinarySet bs;
        knowhere::DataSetPtr first_result;
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA,
                                                                                 version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());
            first_result = res.value();

            REQUIRE(index.Serialize(bs) == knowhere::Status::success);
        }
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA,
                                                                                 version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Deserialize(bs, json) == knowhere::Status::success);
            REQUIRE(index.Count() == nb);
            REQUIRE(index.Dim() == dim);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());

            auto ids_before = first_result->GetIds();
            auto ids_after = res.value()->GetIds();
            for (int64_t i = 0; i < nq * topk; i++) {
                REQUIRE(ids_before[i] == ids_after[i]);
            }
        }
    }
}

TEST_CASE("Test SVS Static Vamana LVQ/LeanVec Build and Search", "[svs][vamana][static][leanvec]") {
    if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
        SKIP("LVQ/LeanVec not available in this SVS runtime build");
    }

    const int64_t nb = 10000, nq = 10;
    const int64_t dim = 128;
    const int64_t topk = 10;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto version = GenTestVersionList();

    // {index type, storage kind} pairs for the compressed static variants
    auto variant = GENERATE(table<std::string, std::string>({
        {knowhere::IndexEnum::INDEX_SVS_VAMANA_LVQ, "lvq4x4"},
        {knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC, "leanvec4x8"},
    }));
    const auto& index_type = std::get<0>(variant);
    const auto& storage_kind = std::get<1>(variant);

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = 64;
        json[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = 200;
        json[knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE] = 40;
        json[knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY] = 40;
        json[knowhere::indexparam::SVS_STORAGE_KIND] = storage_kind;
        json[knowhere::indexparam::SVS_IS_STATIC] = true;
        return json;
    };

    SECTION("Build and KNN Search") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version);
        REQUIRE(idx.has_value());
        auto index = idx.value();

        auto cfg_json = gen().dump();
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        REQUIRE(index.Type() == index_type);
        REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(index.Size() > 0);
        REQUIRE(index.Count() == nb);

        auto res = index.Search(query_ds, json, nullptr);
        REQUIRE(res.has_value());

        float recall = GetKNNRecall(*gt.value(), *res.value());
        LOG_KNOWHERE_INFO_ << "SVS Static Vamana " << storage_kind << " recall@" << topk << " = " << recall
                           << " (metric=" << metric << ")";
        REQUIRE(recall >= 0.70f);
    }

    SECTION("Serialize and Deserialize") {
        knowhere::BinarySet bs;
        knowhere::DataSetPtr first_result;
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());
            first_result = res.value();

            REQUIRE(index.Serialize(bs) == knowhere::Status::success);
        }
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version);
            REQUIRE(idx.has_value());
            auto index = idx.value();

            auto cfg_json = gen().dump();
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(index.Deserialize(bs, json) == knowhere::Status::success);
            REQUIRE(index.Count() == nb);
            REQUIRE(index.Dim() == dim);

            auto res = index.Search(query_ds, json, nullptr);
            REQUIRE(res.has_value());

            auto ids_before = first_result->GetIds();
            auto ids_after = res.value()->GetIds();
            for (int64_t i = 0; i < nq * topk; i++) {
                REQUIRE(ids_before[i] == ids_after[i]);
            }
        }
    }
}

namespace {

// The fp16/bf16 LeanVec variants are registered through IndexNodeDataMockWrapper, which rebuilds the train DataSet
// while converting the tensor. This checks that the fp32 OOD query sample survives that conversion instead of being
// dropped and silently falling back to in-distribution training.
template <typename DataType>
void
CheckLeanVecOodThroughMockWrapper(const std::string& index_type, const std::string& metric, int32_t version) {
    const int64_t nb = 10000, nq = 10, n_train_q = 2000;
    const int64_t dim = 128;
    const int64_t topk = 10;

    const auto train_ds_fp32 = GenDataSet(nb, dim);
    const auto query_ds_fp32 = GenDataSet(nq, dim);
    // the OOD query sample is always fp32, regardless of the index data type
    const auto train_query_ds = GenDataSet(n_train_q, dim, /*seed=*/7);

    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds_fp32, query_ds_fp32, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto train_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(train_ds_fp32);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(query_ds_fp32);
    train_ds->Set(knowhere::meta::TRAIN_QUERY_TENSOR, static_cast<const float*>(train_query_ds->GetTensor()));
    train_ds->Set(knowhere::meta::TRAIN_QUERY_ROWS, static_cast<int64_t>(n_train_q));

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("leanvec4x8");
    json[knowhere::indexparam::SVS_LEANVEC_OOD] = true;
    if (index_type == knowhere::IndexEnum::INDEX_SVS_IVF_LEANVEC) {
        json[knowhere::indexparam::SVS_IVF_NLIST] = 256;
        json[knowhere::indexparam::SVS_IVF_NPROBE] = 48;
    } else {
        json[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = 64;
        json[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = 200;
        json[knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE] = 40;
        json[knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY] = 40;
    }

    auto idx = knowhere::IndexFactory::Instance().Create<DataType>(index_type, version);
    REQUIRE(idx.has_value());
    auto index = idx.value();

    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
    REQUIRE(index.Count() == nb);

    auto res = index.Search(query_ds, json, nullptr);
    REQUIRE(res.has_value());

    float recall = GetKNNRecall(*gt.value(), *res.value());
    LOG_KNOWHERE_INFO_ << "SVS " << index_type << " LeanVec OOD (mock wrapper) recall@" << topk << " = " << recall
                       << " (metric=" << metric << ")";
    REQUIRE(recall >= 0.70f);
}

}  // namespace

TEST_CASE("Test SVS LeanVec OOD across the data-type mock wrapper", "[svs][leanvec][ood][mock]") {
    if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
        SKIP("LVQ/LeanVec not available in this SVS runtime build");
    }

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);
    auto index_type = GENERATE(as<std::string>{}, knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC,
                               knowhere::IndexEnum::INDEX_SVS_IVF_LEANVEC);
    auto version = GenTestVersionList();

    SECTION("fp16") {
        CheckLeanVecOodThroughMockWrapper<knowhere::fp16>(index_type, metric, version);
    }

    SECTION("bf16") {
        CheckLeanVecOodThroughMockWrapper<knowhere::bf16>(index_type, metric, version);
    }
}

#endif  // KNOWHERE_WITH_SVS
