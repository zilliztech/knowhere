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

#include <unordered_set>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/utils/binary_distances.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/bitsetview.h"
#include "knowhere/cluster/cluster_factory.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/log.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.8f;

}  // namespace

// use kNN search to test the correctness of kmeans
TEST_CASE("Test Kmeans With Float Vector", "[float metrics]") {
    using Catch::Approx;

    const int64_t nb = 1000, nq = 10;
    const int64_t dim = 128;
    const int64_t num_clusters = 8;
    auto topk = 1;

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::NUM_CLUSTERS] = num_clusters;
        return json;
    };

    const auto train_ds = GenDataSet(nb, dim);
    const auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, topk},
    };
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

    SECTION("Test Kmeans result") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::ClusterEnum::CLUSTER_KMEANS, base_gen)}));
        auto cluster = knowhere::ClusterFactory::Instance().Create<knowhere::fp32>(name).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(cluster.Type() == name);
        auto res = cluster.Train(*train_ds, json);
        REQUIRE(res.has_value());

        std::vector<std::vector<int64_t>> ids(num_clusters);
        for (int64_t i = 0; i < nb; ++i) {
            auto centroid_id = reinterpret_cast<const uint32_t*>(res.value()->GetTensor())[i];
            REQUIRE(centroid_id < num_clusters);
            ids[centroid_id].push_back(i);
        }
        auto assign_res = cluster.Assign(*query_ds);
        REQUIRE(assign_res.has_value());

        // each query select its nearest cluster as the result
        // like ivfflat choose nprobe=1
        std::vector<std::vector<int64_t>> result(nq);
        for (int64_t i = 0; i < nq; ++i) {
            auto centroid_id = reinterpret_cast<const uint32_t*>(assign_res.value()->GetTensor())[i];
            result[i] = ids[centroid_id];
        }

        float recall = GetKNNRecall(*gt.value(), result);
        LOG_KNOWHERE_INFO_ << "recall: " << recall;
        REQUIRE(recall > kKnnRecallThreshold);
    }
}
