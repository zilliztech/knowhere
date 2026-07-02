// Copyright (C) 2026 6sense Insights Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy at
// http://www.apache.org/licenses/LICENSE-2.0

#ifdef KNOWHERE_WITH_CUVS

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

// ─── helpers ─────────────────────────────────────────────────────────────────

namespace {

knowhere::BinarySet
build_hnsw_binset(int nb, int dim, int M, int efC, const std::string& metric, const knowhere::DataSetPtr& train_ds) {
    knowhere::Json cfg;
    cfg[knowhere::meta::DIM] = dim;
    cfg[knowhere::meta::METRIC_TYPE] = metric;
    cfg[knowhere::indexparam::HNSW_M] = M;
    cfg[knowhere::indexparam::EFCONSTRUCTION] = efC;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(idx.Build(train_ds, cfg) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);
    return binset;
}

knowhere::BinarySet
build_hnsw_sq_binset(int nb, int dim, int M, int efC, const std::string& metric, const knowhere::DataSetPtr& train_ds) {
    knowhere::Json cfg;
    cfg[knowhere::meta::DIM] = dim;
    cfg[knowhere::meta::METRIC_TYPE] = metric;
    cfg[knowhere::indexparam::HNSW_M] = M;
    cfg[knowhere::indexparam::EFCONSTRUCTION] = efC;
    cfg[knowhere::indexparam::SQ_TYPE] = "SQ8";

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    REQUIRE(idx.Build(train_ds, cfg) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);
    return binset;
}

}  // anonymous namespace

// ─── registration ────────────────────────────────────────────────────────────

TEST_CASE("GPU_HNSW is registered in IndexFactory", "[gpu_hnsw]") {
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version);
    REQUIRE(idx.has_value());
    REQUIRE(idx.value().Type() == knowhere::IndexEnum::INDEX_GPU_HNSW);
}

// ─── F32 storage tests ───────────────────────────────────────────────────────

TEST_CASE("GPU_HNSW F32: L2 metric recall vs CPU", "[gpu_hnsw]") {
    constexpr int nb = 10000, nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
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
    auto binset = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
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
    auto binset = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
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
    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
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
    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
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
    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
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
    auto binset = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto query_ds = CopyDataSet(train_ds, nq);
    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto* ids = results.value()->GetIds();
    int self_match = 0;
    for (int i = 0; i < nq; i++) {
        if (ids[i] == i)
            self_match++;
    }
    float self_recall = static_cast<float>(self_match) / nq;
    INFO("Self-match rate (F32): " << self_recall);
    REQUIRE(self_recall >= 0.90f);
}

TEST_CASE("GPU_HNSW SQ8: self-search returns self as top-1", "[gpu_hnsw]") {
    constexpr int nb = 5000, nq = 100, dim = 64, k = 1;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto query_ds = CopyDataSet(train_ds, nq);
    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());

    auto* ids = results.value()->GetIds();
    int self_match = 0;
    for (int i = 0; i < nq; i++) {
        if (ids[i] == i)
            self_match++;
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
    auto binset = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
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
    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json gt_cfg;
    gt_cfg[knowhere::meta::DIM] = dim;
    gt_cfg[knowhere::meta::METRIC_TYPE] = metric;
    gt_cfg[knowhere::meta::TOPK] = k;
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, gt_cfg, nullptr);
    REQUIRE(gt.has_value());

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    float prev_recall = 0.0f;
    for (int ef : {50, 100, 200, 400}) {
        knowhere::Json search_cfg;
        search_cfg[knowhere::meta::DIM] = dim;
        search_cfg[knowhere::meta::METRIC_TYPE] = metric;
        search_cfg[knowhere::meta::TOPK] = k;
        search_cfg[knowhere::indexparam::EF] = ef;

        auto gpu_idx = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version)
                           .value();
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
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset1 = build_hnsw_sq_binset(nb, dim, 16, 100, metric, train_ds1);
    auto binset2 = build_hnsw_sq_binset(nb, dim, 16, 100, metric, train_ds2);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 100;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();

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
        if (ids1[i] != ids2[i])
            diff++;
    }
    float diff_rate = static_cast<float>(diff) / (nq * k);
    INFO("Result difference rate after reload: " << diff_rate);
    REQUIRE(diff_rate >= 0.20f);
}

// ─── multi-segment lifecycle (absorption) test ───────────────────────────────
// Simulates the Milvus segment lifecycle: two independent segments are searched
// independently, then compacted into one large segment. Verifies recall is
// maintained across the lifecycle and the merged segment matches brute-force.

TEST_CASE("GPU_HNSW: multi-segment lifecycle (absorption)", "[gpu_hnsw]") {
    // Segment A: 5000 vectors, Segment B: 5000 vectors, Merged: 10000 vectors
    constexpr int nb_a = 5000, nb_b = 5000, nb_all = nb_a + nb_b;
    constexpr int nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::COSINE;

    // Generate disjoint datasets with different seeds
    auto ds_a = GenDataSet(nb_a, dim, 42);
    auto ds_b = GenDataSet(nb_b, dim, 77);
    auto query_ds = GenDataSet(nq, dim, 99);

    // Build combined dataset for merged segment and brute-force ground truth
    float* combined = new float[nb_all * dim];
    memcpy(combined, ds_a->GetTensor(), nb_a * dim * sizeof(float));
    memcpy(combined + nb_a * dim, ds_b->GetTensor(), nb_b * dim * sizeof(float));
    auto ds_all = knowhere::GenDataSet(nb_all, dim, combined);
    ds_all->SetIsOwner(true);

    // Build CPU HNSW indexes for each segment and the merged segment
    auto binset_a = build_hnsw_binset(nb_a, dim, 16, 100, metric, ds_a);
    auto binset_b = build_hnsw_binset(nb_b, dim, 16, 100, metric, ds_b);
    auto binset_all = build_hnsw_binset(nb_all, dim, 16, 100, metric, ds_all);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    // --- Phase 1: Search two independent GPU segments ---
    auto gpu_a =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_a.Deserialize(binset_a, search_cfg) == knowhere::Status::success);
    auto res_a = gpu_a.Search(query_ds, search_cfg, nullptr);
    REQUIRE(res_a.has_value());
    REQUIRE(res_a.value()->GetRows() == nq);

    auto gpu_b =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_b.Deserialize(binset_b, search_cfg) == knowhere::Status::success);
    auto res_b = gpu_b.Search(query_ds, search_cfg, nullptr);
    REQUIRE(res_b.has_value());
    REQUIRE(res_b.value()->GetRows() == nq);

    // Both segments should return valid results (non-negative IDs)
    auto* ids_a = res_a.value()->GetIds();
    auto* ids_b = res_b.value()->GetIds();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(ids_a[i] >= 0);
        REQUIRE(ids_a[i] < nb_a);
        REQUIRE(ids_b[i] >= 0);
        REQUIRE(ids_b[i] < nb_b);
    }

    // --- Phase 2: Search merged segment (simulates post-compaction) ---
    auto gpu_all =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_all.Deserialize(binset_all, search_cfg) == knowhere::Status::success);
    auto res_all = gpu_all.Search(query_ds, search_cfg, nullptr);
    REQUIRE(res_all.has_value());

    // --- Phase 3: Verify merged recall vs brute-force ---
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(ds_all, query_ds, search_cfg, nullptr);
    REQUIRE(gt.has_value());

    float merged_recall = GetKNNRecall(*gt.value(), *res_all.value());
    INFO("Merged segment recall vs brute-force: " << merged_recall);
    REQUIRE(merged_recall >= 0.85f);

    // --- Phase 4: Verify merged is better than or equal to either segment alone ---
    // Search each segment against its own brute-force ground truth
    auto gt_a = knowhere::BruteForce::Search<knowhere::fp32>(ds_a, query_ds, search_cfg, nullptr);
    auto gt_b = knowhere::BruteForce::Search<knowhere::fp32>(ds_b, query_ds, search_cfg, nullptr);
    REQUIRE(gt_a.has_value());
    REQUIRE(gt_b.has_value());

    float recall_a = GetKNNRecall(*gt_a.value(), *res_a.value());
    float recall_b = GetKNNRecall(*gt_b.value(), *res_b.value());
    INFO("Segment A recall: " << recall_a << ", Segment B recall: " << recall_b);
    REQUIRE(recall_a >= 0.80f);
    REQUIRE(recall_b >= 0.80f);

    // Merged recall against global GT should be >= the better per-segment recall,
    // because the merged graph sees all candidates.
    float best_segment_recall = std::max(recall_a, recall_b);
    INFO("Merged recall (" << merged_recall << ") vs best segment (" << best_segment_recall << ")");
    // Allow small margin since merging changes graph structure
    REQUIRE(merged_recall >= best_segment_recall - 0.10f);
}

// ─── SQ8 multi-segment lifecycle ─────────────────────────────────────────────

TEST_CASE("GPU_HNSW SQ8: multi-segment lifecycle (absorption)", "[gpu_hnsw]") {
    constexpr int nb_a = 5000, nb_b = 5000, nb_all = nb_a + nb_b;
    constexpr int nq = 200, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto ds_a = GenDataSet(nb_a, dim, 42);
    auto ds_b = GenDataSet(nb_b, dim, 77);
    auto query_ds = GenDataSet(nq, dim, 99);

    float* combined = new float[nb_all * dim];
    memcpy(combined, ds_a->GetTensor(), nb_a * dim * sizeof(float));
    memcpy(combined + nb_a * dim, ds_b->GetTensor(), nb_b * dim * sizeof(float));
    auto ds_all = knowhere::GenDataSet(nb_all, dim, combined);
    ds_all->SetIsOwner(true);

    auto binset_a = build_hnsw_sq_binset(nb_a, dim, 16, 100, metric, ds_a);
    auto binset_b = build_hnsw_sq_binset(nb_b, dim, 16, 100, metric, ds_b);
    auto binset_all = build_hnsw_sq_binset(nb_all, dim, 16, 100, metric, ds_all);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    // Phase 1: Two independent segments
    auto gpu_a =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_a.Deserialize(binset_a, search_cfg) == knowhere::Status::success);
    auto res_a = gpu_a.Search(query_ds, search_cfg, nullptr);
    REQUIRE(res_a.has_value());

    auto gpu_b =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_b.Deserialize(binset_b, search_cfg) == knowhere::Status::success);
    auto res_b = gpu_b.Search(query_ds, search_cfg, nullptr);
    REQUIRE(res_b.has_value());

    // Phase 2: Merged segment
    auto gpu_all =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_all.Deserialize(binset_all, search_cfg) == knowhere::Status::success);
    auto res_all = gpu_all.Search(query_ds, search_cfg, nullptr);
    REQUIRE(res_all.has_value());

    // Phase 3: Merged recall vs brute-force
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(ds_all, query_ds, search_cfg, nullptr);
    REQUIRE(gt.has_value());

    float merged_recall = GetKNNRecall(*gt.value(), *res_all.value());
    INFO("SQ8 merged segment recall vs brute-force: " << merged_recall);
    REQUIRE(merged_recall >= 0.85f);

    // Phase 4: GPU reloads old segment after merged — verify no stale state
    // Reload segment A into the same index object that had the merged index
    REQUIRE(gpu_all.Deserialize(binset_a, search_cfg) == knowhere::Status::success);
    auto res_a2 = gpu_all.Search(query_ds, search_cfg, nullptr);
    REQUIRE(res_a2.has_value());

    // Results should match segment A's original results (GPU cache was reset)
    auto* ids_a1 = res_a.value()->GetIds();
    auto* ids_a2 = res_a2.value()->GetIds();
    int match_count = 0;
    for (int i = 0; i < nq * k; i++) {
        if (ids_a1[i] == ids_a2[i])
            match_count++;
    }
    float consistency = static_cast<float>(match_count) / (nq * k);
    INFO("Reload consistency (segment A original vs reloaded): " << consistency);
    REQUIRE(consistency >= 0.99f);
}

TEST_CASE("GPU_HNSW SQ8: high-dim (384-d COSINE, N=50K)", "[gpu_hnsw]") {
    constexpr int nb = 50000, nq = 200, dim = 384, k = 10;
    const std::string metric = knowhere::metric::COSINE;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);

    auto results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    auto cpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_SQ, version).value();
    REQUIRE(cpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto cpu_results = cpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(cpu_results.has_value());

    float recall = GetKNNRecall(*cpu_results.value(), *results.value());
    INFO("GPU vs CPU recall (SQ8 384-d COSINE N=50K): " << recall);
    REQUIRE(recall >= 0.90f);
}

// ─── INT8 bias decode regression test ─────────────────────────────────────────
// Verifies that QT_8bit_direct_signed codes (biased uint8, code = original + 128)
// are correctly converted to signed int8 before GPU upload (commit 60ca031b).
// Without the fix: kernel reads biased values → garbage distances → R@1 = 0%.

TEST_CASE("GPU_HNSW SQ8: INT8 bias decode correctness (COSINE, 384-d)", "[gpu_hnsw][int8_bias]") {
    constexpr int nb = 10000, nq = 200, dim = 384, k = 1;
    const std::string metric = knowhere::metric::COSINE;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset = build_hnsw_sq_binset(nb, dim, 32, 200, metric, train_ds);

    knowhere::Json search_cfg;
    search_cfg[knowhere::meta::DIM] = dim;
    search_cfg[knowhere::meta::METRIC_TYPE] = metric;
    search_cfg[knowhere::meta::TOPK] = k;
    search_cfg[knowhere::indexparam::EF] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();

    // GPU search
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();
    REQUIRE(gpu_idx.Deserialize(binset, search_cfg) == knowhere::Status::success);
    auto gpu_results = gpu_idx.Search(query_ds, search_cfg, nullptr);
    REQUIRE(gpu_results.has_value());

    // Brute-force ground truth (uses exact float32 distances)
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, search_cfg, nullptr);
    REQUIRE(gt.has_value());

    // R@1 must be >= 0.90 (was 0.00 before the bias decode fix)
    float recall = GetKNNRecall(*gt.value(), *gpu_results.value());
    INFO("GPU INT8 COSINE R@1 (384-d, N=10K): " << recall);
    REQUIRE(recall >= 0.90f);

    // Self-search: query with the first nq training vectors
    auto self_query = CopyDataSet(train_ds, nq);
    auto self_results = gpu_idx.Search(self_query, search_cfg, nullptr);
    REQUIRE(self_results.has_value());

    auto* self_ids = self_results.value()->GetIds();
    int self_match = 0;
    for (int i = 0; i < nq; i++) {
        if (self_ids[i] == i)
            self_match++;
    }
    float self_recall = static_cast<float>(self_match) / nq;
    INFO("GPU INT8 COSINE self-match rate: " << self_recall);
    REQUIRE(self_recall >= 0.80f);
}

// ─── smem overflow clamp test ────────────────────────────────────────────────
// Verifies that ef values exceeding shared memory capacity are clamped rather
// than causing CUDA launch failures. The kernel computes:
//   smem_overhead = sw * max_degree0 * 8 + sw * 4 + 12
//   max_ef = (49152 - smem_overhead) / 12
// For sw=1, max_degree0=32: max_ef = (49152 - 268) / 12 = 4073
// ef=8192 would overflow without the clamp.

TEST_CASE("GPU_HNSW: ef overflow is clamped (no CUDA error)", "[gpu_hnsw][smem_clamp]") {
    constexpr int nb = 5000, nq = 50, dim = 64, k = 10;
    const std::string metric = knowhere::metric::L2;

    auto train_ds = GenDataSet(nb, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 99);
    auto binset = build_hnsw_binset(nb, dim, 32, 200, metric, train_ds);

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto gpu_idx =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_GPU_HNSW, version).value();

    knowhere::Json cfg;
    cfg[knowhere::meta::DIM] = dim;
    cfg[knowhere::meta::METRIC_TYPE] = metric;
    cfg[knowhere::meta::TOPK] = k;

    REQUIRE(gpu_idx.Deserialize(binset, cfg) == knowhere::Status::success);

    // ef=8192 exceeds the 49152-byte smem limit for typical graph configs
    cfg[knowhere::indexparam::EF] = 8192;
    auto results = gpu_idx.Search(query_ds, cfg, nullptr);
    REQUIRE(results.has_value());
    REQUIRE(results.value()->GetRows() == nq);
    REQUIRE(results.value()->GetDim() == k);

    // Results should still be valid (IDs in range, distances non-negative for L2)
    auto* ids = results.value()->GetIds();
    auto* dists = results.value()->GetDistance();
    for (int i = 0; i < nq * k; i++) {
        REQUIRE(ids[i] >= 0);
        REQUIRE(ids[i] < nb);
        REQUIRE(dists[i] >= 0.0f);
    }

    // Recall should still be reasonable (clamped ef is still large)
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, cfg, nullptr);
    REQUIRE(gt.has_value());
    float recall = GetKNNRecall(*gt.value(), *results.value());
    INFO("Recall with clamped ef: " << recall);
    REQUIRE(recall >= 0.85f);
}

#endif  // KNOWHERE_WITH_CUVS
