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

// Build a CPU HNSW (plain F32) index, serialize it, return the BinarySet.
knowhere::BinarySet
build_hnsw_binset(int nb, int dim, int M, int efC,
                  const std::string& metric,
                  const knowhere::DataSetPtr& train_ds) {
    knowhere::Json build_cfg;
    build_cfg[knowhere::meta::DIM]         = dim;
    build_cfg[knowhere::meta::METRIC_TYPE] = metric;
    build_cfg[knowhere::indexparam::HNSW_M]         = M;
    build_cfg[knowhere::indexparam::EFCONSTRUCTION]  = efC;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version)
                       .value();
    REQUIRE(cpu_idx.Build(train_ds, build_cfg) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(cpu_idx.Serialize(binset) == knowhere::Status::success);
    return binset;
}

}  // anonymous namespace

// ─── tests ───────────────────────────────────────────────────────────────────

TEST_CASE("GPU_HNSW is registered in IndexFactory", "[gpu_hnsw]") {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version);
    REQUIRE(idx.has_value());
    REQUIRE(idx.value().Type() == knowhere::IndexEnum::INDEX_GPU_HNSW);
}

TEST_CASE("GPU_HNSW: build, serialize, deserialize, search (L2)", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);

    auto binset = build_hnsw_binset(nb, dim, /*M=*/32, /*efC=*/200, metric, train_ds);

    // Load into GPU index
    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    // Compare recall against CPU HNSW
    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version)
                       .value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU HNSW recall (L2): " << recall);
    REQUIRE(recall >= 0.90f);
}

TEST_CASE("GPU_HNSW: COSINE metric", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::COSINE;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);

    auto binset = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    // Brute-force ground truth
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, search_cfg, nullptr);
    REQUIRE(gt.has_value());

    float recall = GetKNNRecall(*gt.value(), *results.value());
    INFO("GPU HNSW recall vs BruteForce (COSINE): " << recall);
    REQUIRE(recall >= 0.85f);

    // COSINE distances should be in [0, 1] (higher = more similar, normalized)
    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(dists[i] >= -0.01f);  // allow tiny fp noise
        REQUIRE(dists[i] <= 1.01f);
    }
}

TEST_CASE("GPU_HNSW: IP metric", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::IP;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);

    auto binset = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                       .value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version)
                       .value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU HNSW recall (IP): " << recall);
    REQUIRE(recall >= 0.90f);

    // IP distances should be positive (higher = more similar)
    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(dists[i] >= 0.0f);
    }
}

TEST_CASE("GPU_HNSW: self-search (query == train) returns self as top-1", "[gpu_hnsw]") {
    constexpr int nb = 5000, dim = 64, k = 1;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto binset   = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
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
    // Plain F32 has no quantization noise; should match nearly perfectly
    REQUIRE(self_recall >= 0.90f);
}

TEST_CASE("GPU_HNSW: multiple searches use cached GPU index", "[gpu_hnsw]") {
    constexpr int nb = 5000, nq = 100, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset   = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
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

#endif  // KNOWHERE_WITH_CUVS
