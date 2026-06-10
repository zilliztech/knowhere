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

knowhere::BinarySet
build_hnsw_binset(int nb, int dim, int M, int efC, const std::string& metric,
                  const knowhere::DataSetPtr& train_ds) {
    knowhere::Json cfg;
    cfg[knowhere::meta::DIM]         = dim;
    cfg[knowhere::meta::METRIC_TYPE] = metric;
    cfg[knowhere::indexparam::HNSW_M]        = M;
    cfg[knowhere::indexparam::EFCONSTRUCTION] = efC;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(idx.Build(train_ds, cfg) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);
    return binset;
}

knowhere::BinarySet
build_hnsw_sq_binset(int nb, int dim, int M, int efC, const std::string& metric,
                     const knowhere::DataSetPtr& train_ds) {
    knowhere::Json cfg;
    cfg[knowhere::meta::DIM]         = dim;
    cfg[knowhere::meta::METRIC_TYPE] = metric;
    cfg[knowhere::indexparam::HNSW_M]        = M;
    cfg[knowhere::indexparam::EFCONSTRUCTION] = efC;
    cfg[knowhere::indexparam::SQ_TYPE]        = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    REQUIRE(idx.Build(train_ds, cfg) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);
    return binset;
}

}  // anonymous namespace

// ─── registration ────────────────────────────────────────────────────────────

TEST_CASE("GPU_HNSW is registered in IndexFactory", "[gpu_hnsw]") {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version);
    REQUIRE(idx.has_value());
    REQUIRE(idx.value().Type() == knowhere::IndexEnum::INDEX_GPU_HNSW);
}

// ─── F32 storage tests ───────────────────────────────────────────────────────

TEST_CASE("GPU_HNSW F32: L2 metric recall vs CPU", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
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
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU recall (F32 L2): " << recall);
    REQUIRE(recall >= 0.90f);
}

TEST_CASE("GPU_HNSW F32: COSINE metric recall vs BruteForce", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::COSINE;

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
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, search_cfg, nullptr);
    REQUIRE(gt.has_value());

    float recall = GetKNNRecall(*gt.value(), *results.value());
    INFO("GPU recall vs BruteForce (F32 COSINE): " << recall);
    REQUIRE(recall >= 0.85f);

    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(dists[i] >= -0.01f);
        REQUIRE(dists[i] <= 1.01f);
    }
}

TEST_CASE("GPU_HNSW F32: IP metric recall vs CPU", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::IP;

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
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU recall (F32 IP): " << recall);
    REQUIRE(recall >= 0.90f);

    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(dists[i] >= 0.0f);
    }
}

// ─── SQ8 storage tests ───────────────────────────────────────────────────────

TEST_CASE("GPU_HNSW SQ8: L2 metric recall vs CPU HNSW_SQ", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset   = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU recall (SQ8 L2): " << recall);
    REQUIRE(recall >= 0.90f);
}

TEST_CASE("GPU_HNSW SQ8: COSINE metric recall vs BruteForce", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::COSINE;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset   = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, search_cfg, nullptr);
    REQUIRE(gt.has_value());

    float recall = GetKNNRecall(*gt.value(), *results.value());
    INFO("GPU recall vs BruteForce (SQ8 COSINE): " << recall);
    REQUIRE(recall >= 0.85f);

    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(dists[i] >= -0.01f);
        REQUIRE(dists[i] <= 1.01f);
    }
}

TEST_CASE("GPU_HNSW SQ8: IP metric recall vs CPU HNSW_SQ", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::IP;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset   = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU recall (SQ8 IP): " << recall);
    REQUIRE(recall >= 0.90f);

    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(dists[i] >= 0.0f);
    }
}

// ─── shared behavior tests ───────────────────────────────────────────────────

TEST_CASE("GPU_HNSW F32: self-search returns self as top-1", "[gpu_hnsw]") {
    constexpr int nb = 5000, nq = 100, dim = 64, k = 1;
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
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto query_ds = CopyDataSet(train_ds, nq);
    auto results  = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto* ids = results.value()->GetIds();
    int self_match = 0;
    for (int i = 0; i < nq; i++) {
        if (ids[i] == i) self_match++;
    }
    float self_recall = static_cast<float>(self_match) / nq;
    INFO("Self-match rate (F32): " << self_recall);
    REQUIRE(self_recall >= 0.90f);
}

TEST_CASE("GPU_HNSW SQ8: self-search returns self as top-1", "[gpu_hnsw]") {
    constexpr int nb = 5000, nq = 100, dim = 64, k = 1;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto binset   = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto query_ds = CopyDataSet(train_ds, nq);
    auto results  = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto* ids = results.value()->GetIds();
    int self_match = 0;
    for (int i = 0; i < nq; i++) {
        if (ids[i] == i) self_match++;
    }
    float self_recall = static_cast<float>(self_match) / nq;
    INFO("Self-match rate (SQ8): " << self_recall);
    REQUIRE(self_recall >= 0.80f);  // SQ8 quantization noise allows some misses
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
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto r1 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r1.has_value());
    auto r2 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r2.has_value());
    auto r3 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r3.has_value());

    auto* ids1 = r1.value()->GetIds();
    auto* ids2 = r2.value()->GetIds();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(ids1[i] == ids2[i]);
    }
}

TEST_CASE("GPU_HNSW SQ8: recall scales with ef", "[gpu_hnsw]") {
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

    auto version  = knowhere::Version::GetCurrentVersion().VersionNumber();
    float prev_recall = 0.0f;
    for (int ef : {50, 100, 200, 400}) {
        knowhere::Json search_cfg;
        search_cfg[knowhere::meta::DIM]         = dim;
        search_cfg[knowhere::meta::METRIC_TYPE] = metric;
        search_cfg[knowhere::meta::TOPK]        = k;
        search_cfg[knowhere::indexparam::EF]    = ef;

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
        REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

        auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
        REQUIRE(results.has_value());

        float recall = GetKNNRecall(*gt.value(), *results.value());
        INFO("ef=" << ef << " recall=" << recall);
        REQUIRE(recall >= prev_recall - 0.02f);
        prev_recall = recall;
    }
    REQUIRE(prev_recall >= 0.85f);
}

TEST_CASE("GPU_HNSW: deserialize-again resets GPU cache", "[gpu_hnsw]") {
    constexpr int nb = 3000, nq = 50, dim = 32, k = 5;
    const std::string metric = knowhere::metric::L2;

    auto train_ds1 = GenDataSet(nb, dim, 42);
    auto train_ds2 = GenDataSet(nb, dim, 7);
    auto query_ds  = GenDataSet(nq, dim, 99);
    auto binset1   = build_hnsw_sq_binset(nb, dim, 16, 100, metric, train_ds1);
    auto binset2   = build_hnsw_sq_binset(nb, dim, 16, 100, metric, train_ds2);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM]         = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK]        = k;
    search_cfg[knowhere::indexparam::EF]    = 100;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();

    REQUIRE(gpu_idx.Deserialize(binset1, search_cfg) == knowhere::Status::success);
    auto r1 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r1.has_value());

    REQUIRE(gpu_idx.Deserialize(binset2, search_cfg) == knowhere::Status::success);
    auto r2 = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(r2.has_value());

    auto* ids1 = r1.value()->GetIds();
    auto* ids2 = r2.value()->GetIds();
    int diff = 0;
    for (int i = 0; i < nq * k; i++) {
        if (ids1[i] != ids2[i]) diff++;
    }
    float diff_rate = static_cast<float>(diff) / (nq * k);
    INFO("Result difference rate after reload: " << diff_rate);
    REQUIRE(diff_rate >= 0.20f);
}

TEST_CASE("GPU_HNSW SQ8: high-dim (384-d COSINE, N=50K)", "[gpu_hnsw]") {
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

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    auto cpu_idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU recall (SQ8 384-d COSINE N=50K): " << recall);
    REQUIRE(recall >= 0.90f);
}

#endif  // KNOWHERE_WITH_CUVS
