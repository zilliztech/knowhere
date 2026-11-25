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

#include <sys/stat.h>

#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/Clustering.h"
#include "faiss/EMeans.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/SuperKMeans.h"
#include "faiss/cppcontrib/knowhere/utils/binary_distances.h"
#include "faiss/impl/ClusteringHelpers.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/bitsetview.h"
#include "knowhere/cluster/cluster_factory.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/dataset.h"
#include "knowhere/log.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.1f;
constexpr float kKnnMinClusterSizeRatio = 0.1f;
constexpr float kKnnMaxClusterSizeRatio = 10.0f;
}  // namespace

TEST_CASE("E-means closing means and validation", "[cluster][e-means]") {
    faiss::ClusteringParameters cp;
    cp.e_means_epochs = GENERATE(1.0, 2.5);
    const std::vector<float> values{0, 1, 2, 3, 14};
    faiss::EMeans trainer(1, 1, cp);
    trainer.train(values.size(), values.data());
    REQUIRE(trainer.centroids == std::vector<float>{4});

    SECTION("invalid epoch budget") {
        cp.e_means_epochs = GENERATE(0.0, -1.0, 0.01, std::numeric_limits<double>::infinity());
        faiss::EMeans invalid(1, 1, cp);
        REQUIRE_THROWS(invalid.train(values.size(), values.data()));
    }
    SECTION("non-finite input") {
        const float input = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS(trainer.train(1, &input));
    }
    SECTION("too many centroids") {
        faiss::EMeans invalid(1, 6, cp);
        REQUIRE_THROWS(invalid.train(values.size(), values.data()));
    }
}

TEST_CASE("E-means spherical closing and large sums", "[cluster][e-means]") {
    faiss::ClusteringParameters cp;
    cp.e_means_epochs = 2.5;
    cp.spherical = true;
    faiss::EMeans spherical(2, 1, cp);
    const std::vector<float> values{1, 0, 0, 1};
    spherical.train(2, values.data());
    REQUIRE(spherical.centroids[0] == Catch::Approx(std::sqrt(0.5)));
    REQUIRE(spherical.centroids[1] == Catch::Approx(std::sqrt(0.5)));

    cp.spherical = false;
    faiss::EMeans wide(2, 1, cp);
    const std::vector<float> large(256, 1000.0f);
    wide.train(128, large.data());
    REQUIRE((wide.centroids == std::vector<float>{1000.0f, 1000.0f}));
}

TEST_CASE("E-means seeded training separates clusters", "[cluster][e-means]") {
    constexpr size_t n = 256, d = 4, k = 4;
    std::vector<float> values(n * d);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            values[i * d + j] = (i % k) * 10.0f + ((i + j) % 7) * 0.01f;
        }
    }
    faiss::ClusteringParameters cp;
    cp.e_means_epochs = 5;
    cp.nredo = 2;
    faiss::EMeans first(d, k, cp), second(d, k, cp);
    first.train(n, values.data());
    second.train(n, values.data());
    REQUIRE(first.centroids == second.centroids);
    faiss::IndexFlatL2 assigner(d);
    assigner.add(k, first.centroids.data());
    std::vector<float> distances(n);
    std::vector<faiss::idx_t> labels(n);
    assigner.search(n, values.data(), 1, distances.data(), labels.data());
    REQUIRE(std::accumulate(distances.begin(), distances.end(), 0.0) < 1.0);
}

TEST_CASE("E-means takes precedence in IVF coarse training", "[cluster][e-means]") {
    faiss::IndexFlatL2 quantizer(1);
    faiss::IndexIVFFlat index(&quantizer, 1, 1);
    index.cp.e_means_epochs = 2.5;
    index.cp.use_super_kmeans = true;
    index.cp.niter = 0;  // E-means has its own epoch budget.
    const std::vector<float> values{0, 1, 2, 3, 14};
    index.train(values.size(), values.data());
    float centroid;
    quantizer.reconstruct(0, &centroid);
    REQUIRE(centroid == 4.0f);
    index.add(values.size(), values.data());
    float distance;
    faiss::idx_t label;
    index.search(1, &values.back(), 1, &distance, &label);
    REQUIRE(label == 4);
    REQUIRE(distance == 0.0f);
}

// use kNN search to test the correctness of kmeans
TEST_CASE("Test Kmeans With Float Vector", "[float metrics]") {
    using Catch::Approx;

    size_t nb = 100000, nq = 100, nt, ns;
    const int64_t dim = 128;
    const int64_t num_clusters = 10;
    const std::string NUM_CLUSTERS = "num_clusters";
    const std::string NUM_ITER = "num_iter";
    auto topk = 1;
    auto maxTopK = 10;
    auto nprobes = 3;
    auto num_trains = 10;
    auto base_gen = [=]() {
        knowhere::Json json;
        json[NUM_CLUSTERS] = num_clusters;
        json[NUM_ITER] = 7;
        json[knowhere::meta::NUM_BUILD_THREAD] = 1;
        return json;
    };

    std::unique_ptr<float[]> sampled_data;
    auto base_ds = GenDataSet(nb, dim);
    auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, maxTopK},
    };
    knowhere::BitsetView bitset;
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, conf, bitset);

    SECTION("Test Kmeans result") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::ClusterEnum::CLUSTER_KMEANS, base_gen)}));
        auto cluster = knowhere::ClusterFactory::Instance().Create<knowhere::fp32>(name).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(cluster.Type() == name);

        for (int i = 0; i < num_trains; i++) {
            GenRandomSlice<float>((const float*)base_ds->GetTensor(), nb, dim, 0.01, sampled_data, ns);
            auto sample_ds = knowhere::GenDataSet(ns, dim, sampled_data.get());
            auto res = cluster.Train(*sample_ds, json);
            REQUIRE(res.has_value());
        }

        std::vector<std::vector<int64_t>> ids(num_clusters);
        size_t min_cluster_size = nb, max_cluster_size = 0, avg_cluster_size = 0;
        auto res_assign = cluster.Assign(*base_ds);
        REQUIRE(res_assign.has_value());
        for (int64_t i = 0; i < nb; ++i) {
            auto centroid_id = reinterpret_cast<const uint32_t*>(res_assign.value()->GetTensor())[i];
            REQUIRE(centroid_id < num_clusters);
            ids[centroid_id].push_back(i);
        }
        for (size_t i = 0; i < num_clusters; i++) {
            size_t cs = ids[i].size();
            if (cs < min_cluster_size) {
                min_cluster_size = cs;
            }
            if (cs > max_cluster_size) {
                max_cluster_size = cs;
            }
            avg_cluster_size += cs;
        }
        avg_cluster_size /= num_clusters;
        REQUIRE((float(min_cluster_size) / float(avg_cluster_size) > kKnnMinClusterSizeRatio));
        REQUIRE((float(max_cluster_size) / float(avg_cluster_size) < kKnnMaxClusterSizeRatio));
        faiss::IndexFlatL2 index(dim);
        res_assign = cluster.Assign(*query_ds);
        REQUIRE(res_assign.has_value());

        // each query select its nearest cluster as the result
        // like ivfflat choose nprobe=1
        std::vector<std::vector<int64_t>> result_assign(nq);
        for (int64_t i = 0; i < nq; ++i) {
            auto centroid_id = reinterpret_cast<const uint32_t*>(res_assign.value()->GetTensor())[i];
            result_assign[i] = ids[centroid_id];
        }
        res_assign.value().reset();
        float recall_top1 = GetKNNRecall(*gt.value(), result_assign, 1);
        LOG_KNOWHERE_INFO_ << "recall_top1: " << recall_top1;
        float recall_top10 = GetKNNRecall(*gt.value(), result_assign, maxTopK);
        LOG_KNOWHERE_INFO_ << "recall_top10: " << recall_top10;

        result_assign.clear();
        auto res_centroids = cluster.GetCentroids();
        REQUIRE(res_centroids.has_value());
        const float* centroids = static_cast<const float*>(res_centroids.value()->GetTensor());
        const float* queries = static_cast<const float*>(query_ds->GetTensor());
        index.add(num_clusters, centroids);
        nq = 100;
        std::vector<std::vector<int64_t>> result(nq);
        for (int64_t i = 0; i < nq; ++i) {
            std::vector<float> dis(nprobes);
            std::vector<faiss::idx_t> idx(nprobes);
            index.search(1, &queries[i * dim], nprobes, dis.data(), idx.data());
            for (int64_t j = 0; j < nprobes; j++) {
                auto centroid_id = idx[j];
                result[i].insert(result[i].end(), ids[centroid_id].begin(), ids[centroid_id].end());
            }
        }
        float recall = GetKNNRecall(*gt.value(), result, nprobes);
        LOG_KNOWHERE_INFO_ << "recall: " << recall;
        REQUIRE(recall > kKnnRecallThreshold);
    }
}

// Spherical training must return the requested number of unit-normalized centroids.
TEST_CASE("Test SuperKMeans Spherical", "[cluster]") {
    const int d = 64;
    const int k = 16;
    const size_t n = 256;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> x(n * d);
    for (auto& v : x) {
        v = dist(rng);
    }

    faiss::SuperKMeansParameters sp;
    sp.seed = 42;
    sp.niter = 2;
    sp.spherical = true;
    faiss::SuperKMeans sc(d, k, sp);
    REQUIRE_NOTHROW(sc.train(n, x.data()));
    REQUIRE(sc.centroids.size() == static_cast<size_t>(k * d));

    // Centroids must be unit norm under spherical clustering.
    for (int j = 0; j < k; ++j) {
        float norm = 0.f;
        for (int i = 0; i < d; ++i) {
            norm += sc.centroids[j * d + i] * sc.centroids[j * d + i];
        }
        norm = std::sqrt(norm);
        REQUIRE(norm == Catch::Approx(1.f).margin(1e-4));
    }
}

TEST_CASE("Test SuperKMeans with 256 centroids", "[cluster]") {
    constexpr int d = 64;
    constexpr int k = 256;
    constexpr size_t n = 1024;

    std::mt19937 rng(43);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> x(n * d);
    for (auto& v : x) {
        v = dist(rng);
    }

    faiss::SuperKMeansParameters sp;
    sp.seed = 43;
    sp.niter = 2;
    faiss::SuperKMeans clustering(d, k, sp);
    REQUIRE_NOTHROW(clustering.train(n, x.data()));
    REQUIRE(clustering.centroids.size() == static_cast<size_t>(k * d));
}

TEST_CASE("SuperKMeans subsampled training matches explicitly sampled input", "[cluster][superkmeans][upgrade]") {
    constexpr int d = 64, k = 16, n = 2048;
    const bool subsample = GENERATE(false, true);
    const bool faster = GENERATE(false, true);
    const bool spherical = GENERATE(false, true);
    CAPTURE(subsample, faster, spherical);
    std::mt19937 rng(2048);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> x(n * d);
    for (auto& v : x) v = normal(rng);
    const auto original = x;
    faiss::SuperKMeansParameters cp;
    cp.seed = 71;
    cp.niter = 4;
    cp.min_points_per_centroid = 1;
    cp.max_points_per_centroid = subsample ? 16 : n;
    cp.use_faster_subsampling = faster;
    cp.spherical = spherical;
    faiss::idx_t reference_n = n;
    const float* reference_x = x.data();
    std::unique_ptr<uint8_t[]> sampled;
    if (subsample) {
        faiss::Clustering sampling_config(d, k, cp);
        uint8_t* data = nullptr;
        float* weights = nullptr;
        reference_n =
            faiss::detail::subsample_training_set(sampling_config, n, reinterpret_cast<const uint8_t*>(x.data()),
                                                  d * sizeof(float), nullptr, &data, &weights);
        sampled.reset(data);
        REQUIRE(weights == nullptr);
        REQUIRE(reference_n == k * cp.max_points_per_centroid);
        reference_x = reinterpret_cast<const float*>(sampled.get());
    }
    faiss::SuperKMeans actual(d, k, cp), reference(d, k, cp);
    actual.train(n, x.data());
    reference.train(reference_n, reference_x);
    REQUIRE(x == original);
    REQUIRE(actual.iteration_stats.size() == static_cast<size_t>(cp.niter));
    REQUIRE(actual.centroids.size() == static_cast<size_t>(k * d));
    for (size_t i = 0; i < actual.centroids.size(); ++i) {
        REQUIRE(std::isfinite(actual.centroids[i]));
        REQUIRE(actual.centroids[i] == Catch::Approx(reference.centroids[i]).margin(1e-5));
    }
    for (size_t i = 0; i < actual.iteration_stats.size(); ++i) {
        REQUIRE(std::isfinite(actual.iteration_stats[i].obj));
        REQUIRE(actual.iteration_stats[i].obj == Catch::Approx(reference.iteration_stats[i].obj).epsilon(1e-5));
    }
}
