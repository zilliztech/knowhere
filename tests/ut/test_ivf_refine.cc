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

#include <string>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {
constexpr const char* kMmapIndexPath = "/tmp/knowhere_dense_mmap_index_test";
constexpr const char* kMmapIndexRefinePath = "/tmp/knowhere_dense_mmap_index_refine_test";
}  // namespace

TEST_CASE("Test Refine Index", "[search][refine]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE, knowhere::metric::L2);
    auto topk = GENERATE(as<int64_t>{}, 5, 120);
    auto version = GenTestVersionList();

    auto base_gen = [=]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    auto ivfflat_gen = [base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 16;
        json[knowhere::indexparam::NPROBE] = 8;
        return json;
    };

    auto ivfsq_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::IVF_SQ_TYPE] = "sq4";
        return json;
    };

    auto ivfsq_refine_gen = [ivfsq_gen]() {
        knowhere::Json json = ivfsq_gen();
        json[knowhere::indexparam::REFINE] = true;
        json[knowhere::indexparam::REFINE_TYPE] = "FP32";
        json[knowhere::indexparam::REFINE_K] = 2.0;
        return json;
    };

    auto ivfpq_gen = [ivfflat_gen]() {
        knowhere::Json json = ivfflat_gen();
        json[knowhere::indexparam::M] = 8;
        json[knowhere::indexparam::NBITS] = 8;
        return json;
    };

    auto ivfpq_refine_gen = [ivfpq_gen]() {
        knowhere::Json json = ivfpq_gen();
        json[knowhere::indexparam::REFINE] = true;
        json[knowhere::indexparam::REFINE_TYPE] = "FP32";
        json[knowhere::indexparam::REFINE_K] = 2.0;
        return json;
    };

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

    SECTION("Test Search") {
        using std::make_tuple;
        auto [name, gen, refine_gen] =
            GENERATE_REF(table<std::string, std::function<knowhere::Json()>, std::function<knowhere::Json()>>(
                {make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, ivfsq_gen, ivfsq_refine_gen),
                 make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFPQ, ivfpq_gen, ivfpq_refine_gen)}));
        knowhere::BinarySet bs;
        // build process
        {
            auto idx_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
            auto idx = idx_expected.value();
            auto cfg_json = gen().dump();
            CAPTURE(name, cfg_json);
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(idx.Type() == name);
            REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
            REQUIRE(idx.Size() > 0);
            REQUIRE(idx.Count() == nb);

            REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

            auto idx_refine_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
            auto idx_refine = idx_refine_expected.value();
            auto refine_cfg_json = refine_gen().dump();
            CAPTURE(name, refine_cfg_json);
            knowhere::Json refine_json = knowhere::Json::parse(refine_cfg_json);
            REQUIRE(idx_refine.Type() == name);
            REQUIRE(idx_refine.Build(train_ds, refine_json) == knowhere::Status::success);
            REQUIRE(idx_refine.Size() > 0);
            REQUIRE(idx_refine.Count() == nb);

            REQUIRE(idx_refine.Serialize(bs) == knowhere::Status::success);
        }
        // search process
        auto load_with_mmap = GENERATE(as<bool>{}, true, false);
        {
            auto idx_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
            auto idx = idx_expected.value();
            auto cfg_json = gen().dump();
            CAPTURE(name, cfg_json);
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            // TODO: qianya(DeserializeFromFile need raw data path. Next pr will remove raw data in ivf sq cc index, and
            // use a knowhere struct to maintain raw data)
            if (load_with_mmap && knowhere::KnowhereCheck::SupportMmapIndexTypeCheck(name)) {
                auto binary = bs.GetByName(idx.Type());
                auto data = binary->data.get();
                auto size = binary->size;
                std::remove(kMmapIndexPath);
                std::ofstream out(kMmapIndexPath, std::ios::binary);
                out.write((const char*)data, size);
                out.close();
                json["enable_mmap"] = true;
                REQUIRE(idx.DeserializeFromFile(kMmapIndexPath, json) == knowhere::Status::success);
            } else {
                REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);
            }

            // TODO: qianya (IVFSQ_CC deserialize casted from the IVFSQ directly, which will cause the hasRawData
            // reference to an uncertain address)
            REQUIRE(idx.HasRawData(json[knowhere::meta::METRIC_TYPE]) ==
                    knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, json));

            auto idx_refine_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
            auto idx_refine = idx_refine_expected.value();
            auto refine_cfg_json = refine_gen().dump();
            CAPTURE(name, refine_cfg_json);
            knowhere::Json refine_json = knowhere::Json::parse(refine_cfg_json);
            if (load_with_mmap && knowhere::KnowhereCheck::SupportMmapIndexTypeCheck(name)) {
                auto binary = bs.GetByName(idx_refine.Type());
                auto data = binary->data.get();
                auto size = binary->size;
                std::remove(kMmapIndexRefinePath);
                std::ofstream out(kMmapIndexRefinePath, std::ios::binary);
                out.write((const char*)data, size);
                out.close();
                refine_json["enable_mmap"] = true;
                REQUIRE(idx_refine.DeserializeFromFile(kMmapIndexRefinePath, refine_json) == knowhere::Status::success);
            } else {
                REQUIRE(idx_refine.Deserialize(bs, refine_json) == knowhere::Status::success);
            }

            REQUIRE(idx_refine.HasRawData(refine_json[knowhere::meta::METRIC_TYPE]) ==
                    knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(name, version, refine_json));

            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
            float recall = GetKNNRecall(*gt.value(), *results.value());

            auto refine_results = idx_refine.Search(query_ds, refine_json, nullptr);
            REQUIRE(refine_results.has_value());
            float refine_recall = GetKNNRecall(*gt.value(), *refine_results.value());

            printf("Test case: %s with refine_type: %s, recall: %f, refine_recall: %f\n", name.c_str(),
                   refine_json["refine_type"].get<std::string>().c_str(), recall, refine_recall);
            REQUIRE(refine_recall >= recall);

            if (metric == knowhere::metric::COSINE) {
                if (name != knowhere::IndexEnum::INDEX_FAISS_IVFSQ8 && name != knowhere::IndexEnum::INDEX_FAISS_IVFPQ) {
                    REQUIRE(CheckDistanceInScope(*results.value(), topk, -1.00001, 1.00001));
                    REQUIRE(CheckDistanceInScope(*refine_results.value(), topk, -1.00001, 1.00001));
                }
            }
        }
        if (load_with_mmap && knowhere::KnowhereCheck::SupportMmapIndexTypeCheck(name)) {
            std::remove(kMmapIndexPath);
            std::remove(kMmapIndexRefinePath);
        }
    }
}
