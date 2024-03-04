// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <future>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/factory.h"
#include "knowhere/utils.h"
#include "utils.h"

TEST_CASE("Test Plugin Index", "[Correctness]") {
    using Catch::Approx;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    int64_t nb = 10000, nq = 1000;
    int64_t dim = 128;
    int64_t top_k = 100;
    int64_t seed = 42;

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = top_k;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto plugin_idx_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json["early_terminate"] = false;
        return json;
    };

    SECTION("Test Build & Serialize & Deserialize & Search Pipeline For Plugin Index ") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple("CustomIdx", plugin_idx_gen),
        }));
        std::string lib_path = "./build/Release/plugin_demo/libcustomIdx.so";
#ifdef CUSTOMIDX_LIB_PATH
#define XSTR(x) STR(x)
#define STR(x) #x
        lib_path = XSTR(CUSTOMIDX_LIB_PATH);
#endif
        auto init_ret = knowhere::KnowherePluginManager::Instance().InitPlugin(lib_path);
        REQUIRE(init_ret);
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto res = idx.Build(*train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.Type() == name);

        knowhere::BinarySet bs;
        // serialize / deserialize test
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
        REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);

        // type/count/dim/size interface test
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Count() == nb);
        REQUIRE(idx.Dim() == dim);
        REQUIRE(idx.Size() > 0);

        {
            // search interface test
            auto query_ds = GenDataSet(nq, dim, seed);

            auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, base_gen(), nullptr);

            auto results = idx.Search(*query_ds, json, nullptr);
            REQUIRE(results.has_value());

            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall > 0.99);
        }

        {
            // rangeSearch interface test
            auto query_ds = GenDataSet(nq, dim, seed);
            auto gt = knowhere::BruteForce::RangeSearch<knowhere::fp32>(train_ds, query_ds, base_gen(), nullptr);

            auto results = idx.RangeSearch(*query_ds, json, nullptr);
            REQUIRE(results.has_value());

            float recall = GetRangeSearchRecall(*gt.value(), *results.value());
            REQUIRE(recall > 0.99);
        }

        {
            // getVectorByIds interface test
            REQUIRE(idx.HasRawData(metric));

            auto ids_ds = GenIdsDataSet(nb, nq);

            auto results = idx.GetVectorByIds(*ids_ds);
            REQUIRE(results.has_value());

            auto xb = (float*)train_ds->GetTensor();

            auto res_rows = results.value()->GetRows();
            auto res_dim = results.value()->GetDim();
            auto res_data = (float*)results.value()->GetTensor();
            REQUIRE(res_rows == nq);
            REQUIRE(res_dim == dim);
            for (int i = 0; i < nq; ++i) {
                const auto id = ids_ds->GetIds()[i];
                for (int j = 0; j < dim; ++j) {
                    REQUIRE(res_data[i * dim + j] == xb[id * dim + j]);
                }
            }
        }
    }
}
