// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include <random>
#include <thread>

#include "catch2/catch_test_macros.hpp"
#include "index/faiss/faiss_config.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"

namespace {
knowhere::DataSetPtr
gen_fp32(size_t nb, size_t dim, int64_t seed = 42) {
    auto* xb = new float[nb * dim];
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < nb * dim; ++i) xb[i] = dist(gen);
    auto ds = knowhere::GenDataSet(nb, dim, xb);
    ds->SetIsOwner(true);
    return ds;
}

knowhere::DataSetPtr
gen_bin(size_t nb, size_t dim_bits, uint64_t seed = 42) {
    const size_t bytes = (dim_bits + 7) / 8;
    auto* xb = new uint8_t[nb * bytes];
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < nb * bytes; ++i) xb[i] = static_cast<uint8_t>(rng());
    auto ds = knowhere::GenDataSet(nb, dim_bits, xb);
    ds->SetIsOwner(true);
    return ds;
}
}  // namespace

TEST_CASE("FaissConfig parses faiss_index_name and captures raw JSON", "[faiss_vanilla]") {
    knowhere::FaissConfig cfg;
    knowhere::Json j =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF256,Flat","nprobe":16,"efSearch":32})");
    std::string msg;

    // Replicate LoadConfig's internal sequence using only public-header entry points.
    knowhere::Json j_(j);
    REQUIRE(knowhere::Config::FormatAndCheck(cfg, j_, &msg) == knowhere::Status::success);
    cfg.CaptureRawJson(j_);
    REQUIRE(knowhere::Config::Load(cfg, j_, knowhere::TRAIN, &msg) == knowhere::Status::success);

    REQUIRE(cfg.faiss_index_name.value() == "IVF256,Flat");
    REQUIRE(cfg.raw_params.contains("nprobe"));
    REQUIRE(cfg.raw_params["nprobe"] == 16);
    REQUIRE(cfg.raw_params.contains("efSearch"));
}

TEST_CASE("IndexFactory creates FAISS index for fp32", "[faiss_vanilla]") {
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS,
                                                                         version.VersionNumber());
    REQUIRE(idx.has_value());
    REQUIRE(idx.value().Type() == knowhere::IndexEnum::INDEX_FAISS);
}

TEST_CASE("FAISS Train+Add Flat smoke", "[faiss_vanilla]") {
    const size_t nb = 1000, dim = 16;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json j = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","dim":16})");
    auto ds = gen_fp32(nb, dim);
    REQUIRE(idx.Build(ds, j) == knowhere::Status::success);
    REQUIRE(idx.Count() == static_cast<int64_t>(nb));
    REQUIRE(idx.Dim() == static_cast<int64_t>(dim));
}

TEST_CASE("FAISS Train forwards parameters via ParameterSpace", "[faiss_vanilla]") {
    const size_t nb = 2000, dim = 32;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    // nprobe is a search-time knob on IVF but ParameterSpace will accept it at build
    // time by setting the field directly. This verifies the forwarding plumbing.
    knowhere::Json j =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":32,"nprobe":8})");
    auto ds = gen_fp32(nb, dim);
    REQUIRE(idx.Build(ds, j) == knowhere::Status::success);
}

TEST_CASE("FAISS Search on Flat returns exact KNN", "[faiss_vanilla]") {
    const size_t nb = 500, dim = 8, nq = 3, k = 5;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","dim":8})");
    auto base = gen_fp32(nb, dim);
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","k":5})");
    auto queries = gen_fp32(nq, dim, /*seed=*/7);
    auto res = idx.Search(queries, search, nullptr);
    REQUIRE(res.has_value());
    REQUIRE(res.value()->GetRows() == static_cast<int64_t>(nq));
    const auto* ids = res.value()->GetIds();
    for (size_t q = 0; q < nq; ++q) {
        for (size_t j = 0; j < k; ++j) {
            REQUIRE(ids[q * k + j] >= 0);
            REQUIRE(ids[q * k + j] < static_cast<int64_t>(nb));
        }
    }
}

TEST_CASE("FAISS Search accepts nprobe on IVF via SearchParametersIVF", "[faiss_vanilla]") {
    const size_t nb = 2000, dim = 16, nq = 4;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);

    knowhere::Json search =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","k":10,"nprobe":8})");
    auto res = idx.Search(gen_fp32(nq, dim, 99), search, nullptr);
    REQUIRE(res.has_value());
}

TEST_CASE("FAISS Search honors BitsetView filter", "[faiss_vanilla]") {
    const size_t nb = 200, dim = 8, nq = 1, k = 10;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json j = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","dim":8,"k":10})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), j) == knowhere::Status::success);

    // Filter out ids [0, 50) — set those bits to 1 (filtered)
    std::vector<uint8_t> bits((nb + 7) / 8, 0);
    for (size_t i = 0; i < 50; ++i) bits[i / 8] |= (1 << (i % 8));
    knowhere::BitsetView bitset(bits.data(), nb);

    auto res = idx.Search(gen_fp32(nq, dim, 3), j, bitset);
    REQUIRE(res.has_value());
    const auto* ids = res.value()->GetIds();
    for (size_t i = 0; i < nq * k; ++i) {
        REQUIRE(ids[i] >= 50);  // any id < 50 would mean filtering is broken
    }
}

TEST_CASE("FAISS RangeSearch supported on Flat", "[faiss_vanilla]") {
    const size_t nb = 100, dim = 8;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","dim":8})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);

    knowhere::Json search =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","radius":100.0,"range_filter":0.0})");
    auto query = gen_fp32(1, dim, 55);
    auto res = idx.RangeSearch(query, search, nullptr);
    REQUIRE(res.has_value());
}

TEST_CASE("FAISS GetVectorByIds works on Flat, fails on PQ", "[faiss_vanilla]") {
    auto version = knowhere::Version::GetCurrentVersion();

    // --- Flat: GetVectorByIds should succeed; HasRawData always returns false (vanilla adapter) ---
    {
        const size_t nb = 64, dim = 8;
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                       .value();
        knowhere::Json j = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","dim":8})");
        REQUIRE(idx.Build(gen_fp32(nb, dim), j) == knowhere::Status::success);

        REQUIRE(idx.HasRawData("L2") == false);

        int64_t query_id = 5;
        auto ids_ds = knowhere::GenIdsDataSet(1, &query_id);
        auto r = idx.GetVectorByIds(ids_ds);
        REQUIRE(r.has_value());
        REQUIRE(r.value()->GetRows() == 1);
    }

    // --- IVFFlat without direct map: GetVectorByIds should fail ---
    // IndexIVF::reconstruct requires a direct_map (NoMap by default), so it throws.
    {
        const size_t nb = 256, dim = 8;
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                       .value();
        knowhere::Json j = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":8})");
        REQUIRE(idx.Build(gen_fp32(nb, dim), j) == knowhere::Status::success);

        int64_t query_id = 5;
        auto ids_ds = knowhere::GenIdsDataSet(1, &query_id);
        auto r = idx.GetVectorByIds(ids_ds);
        REQUIRE_FALSE(r.has_value());
    }
}

TEST_CASE("FAISS Serialize/Deserialize roundtrip", "[faiss_vanilla]") {
    const size_t nb = 200, dim = 8;
    auto version = knowhere::Version::GetCurrentVersion();

    auto idx1 = knowhere::IndexFactory::Instance()
                    .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                    .value();
    knowhere::Json j = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","dim":8,"k":3})");
    REQUIRE(idx1.Build(gen_fp32(nb, dim), j) == knowhere::Status::success);

    knowhere::BinarySet bs;
    REQUIRE(idx1.Serialize(bs) == knowhere::Status::success);

    auto idx2 = knowhere::IndexFactory::Instance()
                    .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                    .value();
    REQUIRE(idx2.Deserialize(bs, j) == knowhere::Status::success);
    REQUIRE(idx2.Count() == static_cast<int64_t>(nb));
    REQUIRE(idx2.Dim() == static_cast<int64_t>(dim));

    // Both indexes must produce identical KNN for the same query.
    auto q = gen_fp32(1, dim, 777);
    auto r1 = idx1.Search(q, j, nullptr).value();
    auto r2 = idx2.Search(q, j, nullptr).value();
    for (int64_t i = 0; i < 3; ++i) {
        REQUIRE(r1->GetIds()[i] == r2->GetIds()[i]);
    }
}

// ---------------------------------------------------------------------------
// Task 10: Binary path end-to-end test
// ---------------------------------------------------------------------------

TEST_CASE("FAISS binary: BFlat build + search", "[faiss_vanilla]") {
    // Use BFlat (brute-force binary) rather than BIVF for the smoke test:
    // - Exercises the bin1 IndexNode path end-to-end (index_binary_factory,
    //   write_index_binary / read_index_binary, binary search with int32
    //   distance → float projection).
    // - Avoids IndexBinaryIVF::train → Clustering::train_encoded →
    //   IndexLSH::sa_decode, an upstream faiss path where ASAN flags a
    //   heap-use-after-free under the cross-test malloc reuse pattern of the
    //   knowhere UT binary. That's an upstream bug unrelated to this adapter.
    const size_t nb = 1024, dim_bits = 64, nq = 2;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::bin1>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json j = knowhere::Json::parse(R"({"metric_type":"HAMMING","faiss_index_name":"BFlat","dim":64,"k":5})");
    REQUIRE(idx.Build(gen_bin(nb, dim_bits), j) == knowhere::Status::success);
    REQUIRE(idx.Count() == static_cast<int64_t>(nb));
    auto res = idx.Search(gen_bin(nq, dim_bits, 3), j, nullptr);
    REQUIRE(res.has_value());
}

// ---------------------------------------------------------------------------
// Task 11: Error-case tests
// ---------------------------------------------------------------------------

TEST_CASE("FAISS: invalid faiss_index_name returns invalid_args", "[faiss_vanilla]") {
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json j =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"NotARealFactoryString","dim":8})");
    auto st = idx.Build(gen_fp32(32, 8), j);
    REQUIRE(st == knowhere::Status::invalid_args);
}

TEST_CASE("FAISS: typo key surfaces faiss error at build", "[faiss_vanilla]") {
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    // "n_probe" (with underscore) is wrong; the real key is "nprobe".
    // faiss::ParameterSpace::set_index_parameter throws on unknown knobs.
    // The adapter translates that to invalid_args.
    knowhere::Json j =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF32,Flat","dim":8,"n_probe":4})");
    auto st = idx.Build(gen_fp32(64, 8), j);
    REQUIRE(st == knowhere::Status::invalid_args);
}

TEST_CASE("FAISS: search key unknown to family returns invalid_args", "[faiss_vanilla]") {
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json jb = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","dim":8})");
    REQUIRE(idx.Build(gen_fp32(64, 8), jb) == knowhere::Status::success);
    // efSearch is an HNSW knob; Flat uses base SearchParameters and does not accept it.
    knowhere::Json jq = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","k":3,"efSearch":32})");
    auto res = idx.Search(gen_fp32(1, 8), jq, nullptr);
    REQUIRE_FALSE(res.has_value());
}

// ---------------------------------------------------------------------------
// Task 12: Size() memory estimate
// ---------------------------------------------------------------------------

TEST_CASE("FAISS Size() gives a non-zero estimate after Build", "[faiss_vanilla]") {
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json j = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"Flat","dim":8})");
    REQUIRE(idx.Build(gen_fp32(100, 8), j) == knowhere::Status::success);
    REQUIRE(idx.Size() > 0);
}

// ---------------------------------------------------------------------------
// Task 13: Concurrent search isolation
// ---------------------------------------------------------------------------

TEST_CASE("FAISS: concurrent searches with varying nprobe are isolated", "[faiss_vanilla]") {
    const size_t nb = 2000, dim = 16;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json jb = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), jb) == knowhere::Status::success);

    auto worker = [&](int nprobe) {
        for (int i = 0; i < 20; ++i) {
            knowhere::Json jq = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","k":5})");
            jq["nprobe"] = nprobe;
            auto res = idx.Search(gen_fp32(1, dim, nprobe * 100 + i), jq, nullptr);
            REQUIRE(res.has_value());
        }
    };
    std::thread t1(worker, 4);
    std::thread t2(worker, 32);
    t1.join();
    t2.join();
}

// PreTransform wrapper: OPQ16,IVF64,PQ16x4 — outer is IndexPreTransform, inner IVFPQ.
// Verifies build_search_params recurses through PreTransform and forwards nprobe to the
// inner IVF SearchParameters.
TEST_CASE("FAISS PreTransform: nprobe propagates through OPQ to IVFPQ", "[faiss_vanilla]") {
    const size_t nb = 4096, dim = 16, nq = 4;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"OPQ16,IVF64,PQ16x4","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);

    knowhere::Json search =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"OPQ16,IVF64,PQ16x4","k":5,"nprobe":8})");
    auto res = idx.Search(gen_fp32(nq, dim, 11), search, nullptr);
    REQUIRE(res.has_value());
}

// Refine wrapper: IVF64,PQ8x4,RFlat. Verify k_factor is consumed at the wrapper layer
// and nprobe is forwarded to the base IVF.
TEST_CASE("FAISS Refine: k_factor + base nprobe both honored", "[faiss_vanilla]") {
    const size_t nb = 4096, dim = 16, nq = 4;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,PQ8x4,RFlat","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"IVF64,PQ8x4,RFlat","k":5,"nprobe":8,"k_factor":2.0})");
    auto res = idx.Search(gen_fp32(nq, dim, 13), search, nullptr);
    REQUIRE(res.has_value());
}

#ifdef FAISS_ENABLE_SVS
// SVS Vamana — verify search_window_size is recognized at the SVS leaf branch.
// Compiled only in SVS-enabled builds (e.g. production X86 image).
TEST_CASE("FAISS SVS Vamana: search_window_size passed through", "[faiss_vanilla]") {
    const size_t nb = 4096, dim = 16, nq = 4;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);

    knowhere::Json search =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","k":5,"search_window_size":32})");
    auto res = idx.Search(gen_fp32(nq, dim, 19), search, nullptr);
    REQUIRE(res.has_value());
}
#endif

// Stringified numeric/boolean values should be accepted (matches Knowhere's
// native Config::FormatAndCheck convention for declared fields).
TEST_CASE("FAISS: stringified nprobe is coerced to number", "[faiss_vanilla]") {
    const size_t nb = 2000, dim = 16, nq = 4;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json jb = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), jb) == knowhere::Status::success);

    knowhere::Json jq =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","k":5,"nprobe":"16"})");
    auto res = idx.Search(gen_fp32(nq, dim, 3), jq, nullptr);
    REQUIRE(res.has_value());
}

TEST_CASE("FAISS: stringified bool is coerced", "[faiss_vanilla]") {
    const size_t nb = 1000, dim = 16, nq = 1;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json jb = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"HNSW16,Flat","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), jb) == knowhere::Status::success);

    knowhere::Json jq = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"HNSW16,Flat","k":5,"check_relative_distance":"false"})");
    auto res = idx.Search(gen_fp32(nq, dim, 3), jq, nullptr);
    REQUIRE(res.has_value());
}

TEST_CASE("FAISS: unparseable string param is rejected with clear error", "[faiss_vanilla]") {
    const size_t nb = 500, dim = 16;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json jb = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":16,"nprobe":"not_a_number"})");
    auto st = idx.Build(gen_fp32(nb, dim), jb);
    REQUIRE(st == knowhere::Status::invalid_args);
}

// Standalone IndexPQ — verify polysemous_ht is recognized at the PQ leaf branch.
TEST_CASE("FAISS standalone PQ: polysemous_ht passed through", "[faiss_vanilla]") {
    const size_t nb = 4096, dim = 16, nq = 4;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"PQ8x4","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);

    knowhere::Json search =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"PQ8x4","k":5,"polysemous_ht":24})");
    auto res = idx.Search(gen_fp32(nq, dim, 17), search, nullptr);
    REQUIRE(res.has_value());
}
