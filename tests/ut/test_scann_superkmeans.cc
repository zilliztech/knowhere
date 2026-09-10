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
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"
#include "utils.h"

namespace {

// Build a SCANN index over train_ds with the given use_super_kmeans value
// and return recall@k against brute-force ground truth.
float
BuildScannAndRecall(const knowhere::DataSetPtr& train_ds, const knowhere::DataSetPtr& query_ds, int64_t nlist,
                    int64_t nprobe, int64_t topk, bool use_super_kmeans) {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_SCANN, version)
                   .value();

    knowhere::Json cfg;
    cfg[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    cfg[knowhere::indexparam::NLIST] = nlist;
    cfg[knowhere::indexparam::NPROBE] = nprobe;
    cfg[knowhere::indexparam::SUB_DIM] = 4;
    cfg[knowhere::indexparam::WITH_RAW_DATA] = false;
    cfg[knowhere::indexparam::USE_SUPER_KMEANS] = use_super_kmeans;

    REQUIRE(idx.Build(train_ds, cfg) == knowhere::Status::success);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    search_cfg[knowhere::meta::TOPK] = topk;
    search_cfg[knowhere::indexparam::NPROBE] = nprobe;
    auto results = idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(
        train_ds, query_ds,
        knowhere::Json{{knowhere::meta::METRIC_TYPE, knowhere::metric::IP}, {knowhere::meta::TOPK, topk}}, nullptr);
    REQUIRE(gt.has_value());

    return GetKNNRecall(*gt.value(), *results.value());
}

}  // namespace

TEST_CASE("SCANN use_super_kmeans default matches Clustering recall", "[scann]") {
    constexpr int64_t nb = 2000;
    constexpr int64_t nq = 100;
    constexpr int64_t dim = 64;
    constexpr int64_t topk = 10;
    constexpr int64_t nlist = 128;
    constexpr int64_t nprobe = 8;

    const auto train_ds = GenDataSet(nb, dim, kSeed);
    const auto query_ds = GenDataSet(nq, dim, kSeed);

    const float super_recall = BuildScannAndRecall(train_ds, query_ds, nlist, nprobe, topk, true);
    const float cluster_recall = BuildScannAndRecall(train_ds, query_ds, nlist, nprobe, topk, false);

    CAPTURE(super_recall, cluster_recall);
    // SuperKMeans coarse quantizer training is recall-equivalent to Clustering.
    // MatchNlist shrinks nlist on this small synthetic set. The enabled build
    // still honors SuperKMeans for the resulting small centroid count; recall
    // equivalence between the two clustering implementations is the invariant.
    REQUIRE(super_recall == Catch::Approx(cluster_recall).margin(0.02f));
}

TEST_CASE("SCANN use_super_kmeans field is honored", "[scann]") {
    // Explicitly disabled must build and search successfully too.
    constexpr int64_t nb = 1000;
    constexpr int64_t nq = 50;
    constexpr int64_t dim = 32;
    constexpr int64_t topk = 10;
    constexpr int64_t nlist = 64;
    constexpr int64_t nprobe = 4;

    const auto train_ds = GenDataSet(nb, dim, kSeed + 1);
    const auto query_ds = GenDataSet(nq, dim, kSeed + 1);

    const float cluster_recall = BuildScannAndRecall(train_ds, query_ds, nlist, nprobe, topk, false);
    REQUIRE(cluster_recall > 0.0f);
}

// SCANN with default use_super_kmeans=true must not fail the build on
// low-dimensional data (e.g. d=16 emb-list scenarios) where SuperKMeans is
// not applicable; it should fall back to Clustering.
TEST_CASE("SCANN low-dim build succeeds with default superkmeans", "[scann]") {
    constexpr int64_t nb = 200;
    constexpr int64_t nq = 20;
    constexpr int64_t dim = 16;
    constexpr int64_t topk = 5;
    constexpr int64_t nlist = 16;
    constexpr int64_t nprobe = 2;

    const auto train_ds = GenDataSet(nb, dim, kSeed + 2);
    const auto query_ds = GenDataSet(nq, dim, kSeed + 2);

    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_SCANN, version)
                   .value();

    knowhere::Json cfg;
    cfg[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    cfg[knowhere::indexparam::NLIST] = nlist;
    cfg[knowhere::indexparam::NPROBE] = nprobe;
    cfg[knowhere::indexparam::SUB_DIM] = 2;
    cfg[knowhere::indexparam::WITH_RAW_DATA] = false;
    // Default use_super_kmeans=true; must fall back to Clustering for d=16.
    REQUIRE(idx.Build(train_ds, cfg) == knowhere::Status::success);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    search_cfg[knowhere::meta::TOPK] = topk;
    search_cfg[knowhere::indexparam::NPROBE] = nprobe;
    auto results = idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
}
