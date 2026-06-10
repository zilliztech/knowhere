// Copyright (C) 2026 6sense Insights Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy at
// http://www.apache.org/licenses/LICENSE-2.0

#ifdef KNOWHERE_WITH_CUVS

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ─── helpers ─────────────────────────────────────────────────────────────────

namespace {

// Build a CPU HNSW_SQ index, serialize it, return the BinarySet.
knowhere::BinarySet
build_hnsw_sq_binset(int nb, int dim, int M, int efC,
                     const std::string& metric,
                     const knowhere::DataSetPtr& train_ds) {
    knowhere::Json build_cfg;
    build_cfg[knowhere::meta::DIM]         = dim;
    build_cfg[knowhere::meta::METRIC_TYPE] = metric;
    build_cfg[knowhere::indexparam::HNSW_M]           = M;
    build_cfg[knowhere::indexparam::EFCONSTRUCTION]   = efC;
    build_cfg[knowhere::indexparam::SQ_TYPE]          = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                       .value();
    REQUIRE(cpu_idx.Build(train_ds, build_cfg) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(cpu_idx.Serialize(binset) == knowhere::Status::success);
    return binset;
}

// Compute overlap between two result sets (fraction of ids in common).
float
overlap(const knowhere::DataSet& a, const knowhere::DataSet& b) {
    int nq   = a.GetRows();
    int k    = a.GetDim();
    REQUIRE(b.GetRows() == nq);
    REQUIRE(b.GetDim() == k);

    auto* a_ids = a.GetIds();
    auto* b_ids = b.GetIds();
    int matched = 0;
    for (int i = 0; i < nq; i++) {
        std::vector<int64_t> av(a_ids + i * k, a_ids + i * k + k);
        std::vector<int64_t> bv(b_ids + i * k, b_ids + i * k + k);
        std::sort(av.begin(), av.end());
        std::sort(bv.begin(), bv.end());
        std::vector<int64_t> common;
        std::set_intersection(av.begin(), av.end(), bv.begin(), bv.end(),
                              std::back_inserter(common));
        matched += static_cast<int>(common.size());
    }
    return static_cast<float>(matched) / (nq * k);
}

}  // anonymous namespace

// ─── tests ───────────────────────────────────────────────────────────────────

TEST_CASE("GPU_HNSW_SQ is registered in IndexFactory", "[gpu_hnsw_sq]") {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version);
    REQUIRE(idx.has_value());
    REQUIRE(idx.value().Type() == knowhere::IndexEnum::INDEX_GPU_HNSW_SQ);
}

TEST_CASE("GPU_HNSW_SQ: build, serialize, deserialize, search (L2)", "[gpu_hnsw_sq]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);

    auto binset = build_hnsw_sq_binset(nb, dim, /*M=*/32, /*efC=*/200, metric, train_ds);

    // Load into GPU index
    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;
    search_cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    // Compare recall against CPU HNSW_SQ
    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                       .value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU HNSW_SQ recall (L2): " << recall);
    REQUIRE(recall >= 0.90f);
}

TEST_CASE("GPU_HNSW_SQ: COSINE metric", "[gpu_hnsw_sq]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::COSINE;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);

    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;
    search_cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    // Brute-force ground truth
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, search_cfg, nullptr);
    REQUIRE(gt.has_value());

    float recall = GetKNNRecall(*gt.value(), *results.value());
    INFO("GPU HNSW_SQ recall vs BruteForce (COSINE): " << recall);
    REQUIRE(recall >= 0.85f);

    // COSINE distances should be in [0, 1] (higher = more similar, normalized)
    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(dists[i] >= -0.01f);  // allow tiny fp noise
        REQUIRE(dists[i] <= 1.01f);
    }
}

TEST_CASE("GPU_HNSW_SQ: IP metric", "[gpu_hnsw_sq]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::IP;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);

    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;
    search_cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                       .value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU HNSW_SQ recall (IP): " << recall);
    REQUIRE(recall >= 0.90f);

    // IP distances should be positive (higher = more similar)
    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(dists[i] >= 0.0f);
    }
}

TEST_CASE("GPU_HNSW_SQ: self-search (query == train) returns self as top-1", "[gpu_hnsw_sq]") {
    constexpr int nb = 5000, dim = 64, k = 1;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto binset   = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;
    search_cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    // Take first 100 vectors from training set as queries
    constexpr int nq = 100;
    auto query_ds = CopyDataSet(train_ds, nq);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto* ids = results.value()->GetIds();
    int self_match = 0;
    for (int i = 0; i < nq; i++) {
        if (ids[i] == i) self_match++;
    }
    float self_recall = static_cast<float>(self_match) / nq;
    INFO("Self-match rate: " << self_recall);
    // SQ8 introduces quantization noise; allow some misses
    REQUIRE(self_recall >= 0.80f);
}

TEST_CASE("GPU_HNSW_SQ: multiple searches use cached GPU index", "[gpu_hnsw_sq]") {
    constexpr int nb = 5000, nq = 100, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset   = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;
    search_cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    // Run three searches — should all succeed and return consistent results
    auto r1 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r1.has_value());
    auto r2 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r2.has_value());
    auto r3 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r3.has_value());

    // Results should be identical across calls (deterministic kernel)
    auto* ids1 = r1.value()->GetIds();
    auto* ids2 = r2.value()->GetIds();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(ids1[i] == ids2[i]);
    }
}

TEST_CASE("GPU_HNSW_SQ: recall scales with ef", "[gpu_hnsw_sq]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset   = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json gt_cfg;
    gt_cfg[knowhere::meta::DIM]         = dim;
    gt_cfg[knowhere::meta::METRIC_TYPE] = metric;
    gt_cfg[knowhere::meta::TOPK]        = k;
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_cfg, nullptr);
    REQUIRE(gt.has_value());

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    float prev_recall = 0.0f;
    for (int ef : {50, 100, 200, 400}) {
        knowhere::Json search_cfg;
        search_cfg[knowhere::meta::DIM]         = dim;
        search_cfg[knowhere::meta::METRIC_TYPE] = metric;
        search_cfg[knowhere::meta::TOPK]        = k;
        search_cfg[knowhere::indexparam::EF]    = ef;
        search_cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                           .value();
        REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
        REQUIRE(results.has_value());

        float recall = GetKNNRecall(*gt.value(), *results.value());
        INFO("ef=" << ef << " recall=" << recall);
        // Recall should be non-decreasing as ef grows
        REQUIRE(recall >= prev_recall - 0.02f);  // allow tiny jitter
        prev_recall = recall;
    }
    // At ef=400, recall should be reasonable
    REQUIRE(prev_recall >= 0.85f);
}

TEST_CASE("GPU_HNSW_SQ: deserialize-again resets GPU cache", "[gpu_hnsw_sq]") {
    constexpr int nb = 3000, nq = 50, dim = 32, k = 5;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset1  = build_hnsw_sq_binset(nb, dim, 16, 100, metric, train_ds);
    // Build a second (slightly different seed) index
    auto train_ds2 = GenDataSet(nb, dim, 7);
    auto binset2   = build_hnsw_sq_binset(nb, dim, 16, 100, metric, train_ds2);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 100;
    search_cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                       .value();

    // Load index 1, search
    REQUIRE(gpu_idx.Deserialize(binset1, search_cfg) == knowhere::Status::success);
    auto r1 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r1.has_value());

    // Reload with index 2, search — should produce different results
    REQUIRE(gpu_idx.Deserialize(binset2, search_cfg) == knowhere::Status::success);
    auto r2 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r2.has_value());

    // Results should differ (different training data)
    auto* ids1 = r1.value()->GetIds();
    auto* ids2 = r2.value()->GetIds();
    int diff = 0;
    for (int i = 0; i < nq * k; i++) {
        if (ids1[i] != ids2[i]) diff++;
    }
    // Very likely to differ with different data; require at least 20% difference
    float diff_rate = static_cast<float>(diff) / (nq * k);
    INFO("Result difference rate after reload: " << diff_rate);
    REQUIRE(diff_rate >= 0.20f);
}

TEST_CASE("GPU_HNSW_SQ: high-dim vectors (384-d, matching Milvus HNSW_SQ production)", "[gpu_hnsw_sq]") {
    constexpr int nb = 50000, nq = 200, dim = 384, k = 10;
    const std::string metric = knowhere::metric::COSINE;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset   = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;
    search_cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW_SQ, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    // Compare against CPU HNSW_SQ
    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version)
                       .value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU recall (384-d COSINE, N=50K): " << recall);
    REQUIRE(recall >= 0.90f);
}

#endif  // KNOWHERE_WITH_CUVS
