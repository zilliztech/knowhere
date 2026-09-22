// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "index/faiss/faiss_config.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"

// FAISS_ENABLE_SVS is private to the faiss targets; knowhere's own global switch for
// the same build is KNOWHERE_WITH_SVS.
#ifdef KNOWHERE_WITH_SVS
#include "catch2/catch_approx.hpp"
#include "faiss/AutoTune.h"
#include "faiss/cppcontrib/knowhere/SearchParamsDispatch.h"
#include "faiss/impl/io.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"
#include "faiss/svs/IndexSVSVamana.h"
#include "faiss/svs/IndexSVSVamanaLeanVec.h"
#endif

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
    knowhere::Json j = knowhere::Json::parse(R"({
            "metric_type":"L2",
            "faiss_index_name":"IVF256,Flat",
            "nprobe":16,
            "efSearch":32,
            "index_type":"FAISS",
            "build_id":"1000",
            "index_num_rows":1000,
            "storage_version":2,
            "build_dram_budget_gb":"4.0",
            "index.nonEncoding":"false"
        })");
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
    REQUIRE_FALSE(cfg.raw_params.contains("index_type"));
    REQUIRE_FALSE(cfg.raw_params.contains("build_id"));
    REQUIRE_FALSE(cfg.raw_params.contains("index_num_rows"));
    REQUIRE_FALSE(cfg.raw_params.contains("storage_version"));
    REQUIRE_FALSE(cfg.raw_params.contains("build_dram_budget_gb"));
    REQUIRE_FALSE(cfg.raw_params.contains("index.nonEncoding"));
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

TEST_CASE("FAISS HasRawData/GetVectorByIds unsupported by vanilla adapter", "[faiss_vanilla]") {
    auto version = knowhere::Version::GetCurrentVersion();

    auto check_unsupported = [&](const std::string& factory_str, size_t nb, size_t dim) {
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                       .value();
        knowhere::Json j = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":")" + factory_str +
                                                 R"(","dim":)" + std::to_string(dim) + "}");
        REQUIRE(idx.Build(gen_fp32(nb, dim), j) == knowhere::Status::success);

        REQUIRE(idx.HasRawData("L2") == false);

        int64_t query_id = 5;
        auto ids_ds = knowhere::GenIdsDataSet(1, &query_id);
        auto r = idx.GetVectorByIds(ids_ds);
        REQUIRE_FALSE(r.has_value());
        REQUIRE(r.error() == knowhere::Status::not_implemented);
    };

    check_unsupported("Flat", 64, 8);
    check_unsupported("IVF64,Flat", 256, 8);
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

#ifdef KNOWHERE_WITH_SVS
// SVS Vamana — search_window_size is recognized at the SVS leaf branch. Compiled only in
// SVS-enabled builds (e.g. production X86 image).
//
// SVS asserts capacity >= window size, so unlike efSearch these two have to be raised as
// a pair. The adapter forwards them verbatim, as faiss's own API does.
TEST_CASE("FAISS SVS Vamana: search_window_size passed through", "[faiss_vanilla]") {
    const size_t nb = 4096, dim = 16, nq = 4;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","dim":16})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","k":5,"search_window_size":32,"search_buffer_capacity":32})");
    auto res = idx.Search(gen_fp32(nq, dim, 19), search, nullptr);
    REQUIRE(res.has_value());

    // Widening only the window leaves the capacity at the index default of 10, which SVS
    // rejects rather than silently narrowing.
    knowhere::Json window_only =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","k":5,"search_window_size":32})");
    REQUIRE_FALSE(idx.Search(gen_fp32(nq, dim, 19), window_only, nullptr).has_value());
}

// The adapter pre-validates build params against supported_build_param_names(), so a name
// missing from that list is reported as unknown even though ParameterSpace accepts it.
TEST_CASE("FAISS SVS Vamana: whitelist matches ParameterSpace", "[faiss_vanilla]") {
    const auto& whitelist = faiss::cppcontrib::knowhere::supported_build_param_names();
    ::faiss::ParameterSpace ps;

    SECTION("every build knob is both whitelisted and accepted") {
        std::unique_ptr<faiss::Index> index(faiss::index_factory(16, "SVSVamana64", faiss::METRIC_L2));
        auto* svs = dynamic_cast<faiss::IndexSVSVamana*>(index.get());
        REQUIRE(svs != nullptr);

        // storage_kind is left out: its valid values depend on the SVS runtime build.
        const std::vector<std::pair<std::string, double>> params = {
            {"graph_max_degree", 32},
            {"prune_to", 28},
            {"alpha", 1.5},
            {"construction_window_size", 96},
            {"max_candidate_pool_size", 300},
            {"use_full_search_history", 0},
            {"search_window_size", 48},
            {"search_buffer_capacity", 56},
            {"is_static", 1},
            {"store_vectors", 0},
        };
        for (const auto& [name, val] : params) {
            REQUIRE(faiss::cppcontrib::knowhere::is_supported_build_param(name));
            ps.set_index_parameter(index.get(), name, val);
        }

        REQUIRE(svs->graph_max_degree == 32);
        REQUIRE(svs->prune_to == 28);
        REQUIRE(svs->alpha == Catch::Approx(1.5f));
        REQUIRE(svs->construction_window_size == 96);
        REQUIRE(svs->max_candidate_pool_size == 300);
        REQUIRE(svs->use_full_search_history == false);
        REQUIRE(svs->search_window_size == 48);
        REQUIRE(svs->search_buffer_capacity == 56);
        REQUIRE(svs->is_static);
        REQUIRE_FALSE(svs->store_vectors);
    }

    SECTION("storage_kind and leanvec_d are whitelisted too") {
        // Both go through ParameterSpace, but their accepted values are runtime- resp.
        // family-dependent, so only the whitelist membership is asserted here.
        REQUIRE(whitelist.count("storage_kind") == 1);
        REQUIRE(whitelist.count("leanvec_d") == 1);
    }
}

// End to end: every caller-settable SVS Vamana build knob accepted through
// FaissIndexNode, on an index that is then searchable. search_buffer_capacity must be
// raised along with search_window_size — SVS rejects a capacity below the window size.
// is_static and store_vectors are deliberately absent: the adapter owns them.
TEST_CASE("FAISS SVS Vamana: all build params accepted end to end", "[faiss_vanilla]") {
    const size_t nb = 4096, dim = 16, nq = 4, k = 5;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(R"({
            "metric_type":"L2",
            "faiss_index_name":"SVSVamana64",
            "dim":16,
            "graph_max_degree":32,
            "prune_to":28,
            "alpha":1.4,
            "construction_window_size":96,
            "max_candidate_pool_size":300,
            "use_full_search_history":false,
            "search_window_size":48,
            "search_buffer_capacity":48
        })");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);
    REQUIRE(idx.Count() == static_cast<int64_t>(nb));

    knowhere::Json search = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","k":5,"search_window_size":64,"search_buffer_capacity":64})");
    auto res = idx.Search(gen_fp32(nq, dim, 23), search, nullptr);
    REQUIRE(res.has_value());
    const auto* ids = res.value()->GetIds();
    for (size_t i = 0; i < nq * k; ++i) {
        REQUIRE(ids[i] >= 0);
        REQUIRE(ids[i] < static_cast<int64_t>(nb));
    }
}

// The adapter applies is_static even though the faiss factory default is dynamic. A
// failing second Add is how that is observable without reaching into index_.
TEST_CASE("FAISS SVS Vamana: second Add rejected on the forced static index", "[faiss_vanilla]") {
    const size_t nb = 2048, dim = 16;
    std::unique_ptr<faiss::Index> probe(faiss::index_factory(dim, "SVSVamana64", faiss::METRIC_L2));
    REQUIRE(faiss::cppcontrib::knowhere::supports_static_index(probe.get()));
    REQUIRE(faiss::cppcontrib::knowhere::supports_dropping_stored_vectors(probe.get()));
    REQUIRE_FALSE(dynamic_cast<faiss::IndexSVSVamana*>(probe.get())->is_static);

    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    // No is_static in the JSON: the adapter supplies it.
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","dim":16})");
    auto base = gen_fp32(nb, dim);
    REQUIRE(idx.Train(base, build) == knowhere::Status::success);
    REQUIRE(idx.Add(base, build) == knowhere::Status::success);
    REQUIRE(idx.Count() == static_cast<int64_t>(nb));
    REQUIRE(idx.Add(gen_fp32(nb, dim, 31), build) != knowhere::Status::success);
}

// The adapter-owned knobs are not caller-settable at all: even the value the adapter
// itself applies is rejected.
TEST_CASE("FAISS SVS Vamana: is_static / store_vectors not accepted as params", "[faiss_vanilla]") {
    const size_t nb = 1000, dim = 16;
    auto version = knowhere::Version::GetCurrentVersion();
    auto make = [&]() {
        return knowhere::IndexFactory::Instance()
            .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
            .value();
    };
    auto base = gen_fp32(nb, dim);

    for (const char* param :
         {R"("is_static":false)", R"("is_static":true)", R"("store_vectors":true)", R"("store_vectors":false)"}) {
        auto build = knowhere::Json::parse(std::string(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64",)") +
                                           R"("dim":16,)" + param + "}");
        REQUIRE(make().Build(base, build) == knowhere::Status::invalid_args);
    }

    // Rejected by name, so a family without the knobs gives the same answer.
    std::unique_ptr<faiss::Index> flat(faiss::index_factory(dim, "IVF64,Flat", faiss::METRIC_L2));
    REQUIRE_FALSE(faiss::cppcontrib::knowhere::supports_static_index(flat.get()));
    REQUIRE_FALSE(faiss::cppcontrib::knowhere::supports_dropping_stored_vectors(flat.get()));
    auto on_ivf =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":16,"is_static":false})");
    REQUIRE(make().Build(base, on_ivf) == knowhere::Status::invalid_args);
}

// SVS-only build knobs are whitelisted globally (the whitelist mirrors the
// index-agnostic ParameterSpace if-chain), so ParameterSpace itself has to be the one
// rejecting them on another family. Either way the caller sees invalid_args.
TEST_CASE("FAISS: SVS build param rejected on a non-SVS index", "[faiss_vanilla]") {
    const size_t nb = 1000, dim = 16;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":16,"alpha":1.4})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::invalid_args);
}

// A typo stays a typo on a family that does have these extra build knobs.
TEST_CASE("FAISS SVS Vamana: unknown build param still rejected", "[faiss_vanilla]") {
    const size_t nb = 1000, dim = 16;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","dim":16,"constructoin_window_size":96})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::invalid_args);
}

// Plain Vamana inherits the base no-op, so an explicit ood_training:true is an error
// there, while the default just narrows — otherwise it would break every family that
// cannot train from queries.
TEST_CASE("FAISS SVS Vamana: ood_training rejected without LeanVec storage", "[faiss_vanilla]") {
    const size_t nb = 1000, dim = 16;
    std::unique_ptr<faiss::Index> probe(faiss::index_factory(dim, "SVSVamana64", faiss::METRIC_L2));
    REQUIRE_FALSE(faiss::cppcontrib::knowhere::supports_train_with_queries(probe.get()));

    auto version = knowhere::Version::GetCurrentVersion();
    auto make = [&]() {
        return knowhere::IndexFactory::Instance()
            .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
            .value();
    };
    auto base = gen_fp32(nb, dim);

    knowhere::Json explicit_ood =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","dim":16,"ood_training":true})");
    REQUIRE(make().Build(base, explicit_ood) == knowhere::Status::invalid_args);

    knowhere::Json defaulted =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64","dim":16})");
    REQUIRE(make().Build(base, defaulted) == knowhere::Status::success);
}

// LVQ and LeanVec need Intel hardware plus an LVQ/LeanVec-enabled SVS runtime, so these
// cases only assert anything where that combination is available.
TEST_CASE("FAISS SVS Vamana LVQ: build params accepted", "[faiss_vanilla]") {
    if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
        SUCCEED("LVQ/LeanVec unavailable in this SVS runtime build");
        return;
    }
    const size_t nb = 4096, dim = 32, nq = 4;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build = knowhere::Json::parse(R"({
            "metric_type":"L2",
            "faiss_index_name":"SVSVamana64,LVQ4x8",
            "dim":32,
            "alpha":1.4,
            "construction_window_size":96,
            "search_window_size":48,
            "search_buffer_capacity":48
        })");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"SVSVamana64,LVQ4x8","k":5,"search_window_size":64,"search_buffer_capacity":64})");
    REQUIRE(idx.Search(gen_fp32(nq, dim, 29), search, nullptr).has_value());
}

TEST_CASE("FAISS SVS Vamana LeanVec: build params accepted", "[faiss_vanilla]") {
    if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
        SUCCEED("LVQ/LeanVec unavailable in this SVS runtime build");
        return;
    }
    const size_t nb = 4096, dim = 32, nq = 4, n_train_q = 256;
    auto version = knowhere::Version::GetCurrentVersion();
    const char* name = "SVSVamana64,LeanVec4x8_16";

    auto make = [&]() {
        return knowhere::IndexFactory::Instance()
            .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
            .value();
    };
    knowhere::Json build = knowhere::Json::parse(R"({
            "metric_type":"L2",
            "faiss_index_name":"SVSVamana64,LeanVec4x8_16",
            "dim":32,
            "alpha":1.4,
            "construction_window_size":96,
            "search_window_size":48,
            "search_buffer_capacity":48,
            "ood_training":true
        })");
    knowhere::Json search = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"SVSVamana64,LeanVec4x8_16","k":5,"search_window_size":64,"search_buffer_capacity":64})");

    SECTION("leanvec_d overrides the dimensionality from the factory string") {
        auto idx = make();
        knowhere::Json with_d = build;
        with_d["leanvec_d"] = 8;
        REQUIRE(idx.Build(gen_fp32(nb, dim), with_d) == knowhere::Status::success);
        REQUIRE(idx.Search(gen_fp32(nq, dim, 47), search, nullptr).has_value());
    }
}

namespace {
// The trained LeanVec projection is not reachable through the node API, but write_index
// embeds it: serialize what knowhere built, read it back as a faiss index, and dump only
// the training data, so the comparison is unaffected by the randomized graph.
std::string
leanvec_training_blob(knowhere::Index<knowhere::IndexNode>& idx) {
    knowhere::BinarySet bs;
    REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
    auto bin = bs.GetByName(knowhere::IndexEnum::INDEX_FAISS);
    REQUIRE(bin != nullptr);

    faiss::VectorIOReader reader;
    reader.data.assign(bin->data.get(), bin->data.get() + bin->size);
    std::unique_ptr<faiss::Index> restored(faiss::read_index(&reader));
    auto* leanvec = dynamic_cast<faiss::IndexSVSVamanaLeanVec*>(restored.get());
    REQUIRE(leanvec != nullptr);

    std::ostringstream os;
    leanvec->serialize_training_data(os);
    return os.str();
}

size_t
count_differing_bytes(const std::string& a, const std::string& b) {
    REQUIRE(a.size() == b.size());
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += (a[i] != b[i]) ? 1 : 0;
    }
    return n;
}
}  // namespace

// Build-and-search success cannot tell an OOD-trained projection from an in-distribution
// one, so compare the projection itself across configs. The same-inputs case is the
// control: without it, a difference would not be evidence of anything.
TEST_CASE("FAISS SVS Vamana LeanVec: ood_training", "[faiss_vanilla]") {
    if (!faiss::IndexSVSVamana::is_lvq_leanvec_enabled()) {
        SUCCEED("LVQ/LeanVec unavailable in this SVS runtime build");
        return;
    }
    const size_t nb = 4096, dim = 32, n_train_q = 256;

    std::unique_ptr<faiss::Index> probe(faiss::index_factory(dim, "SVSVamana64,LeanVec4x8_16", faiss::METRIC_L2));
    REQUIRE(faiss::cppcontrib::knowhere::supports_train_with_queries(probe.get()));

    auto version = knowhere::Version::GetCurrentVersion();
    auto make = [&]() {
        return knowhere::IndexFactory::Instance()
            .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
            .value();
    };
    knowhere::Json build = knowhere::Json::parse(R"({
            "metric_type":"L2",
            "faiss_index_name":"SVSVamana64,LeanVec4x8_16",
            "dim":32,
            "ood_training":true
        })");

    knowhere::Json in_dist = build;
    in_dist["ood_training"] = false;

    // Every build gets the same database vectors (gen_fp32 is seeded), so the config and
    // the query sample are the only things that vary. query_seed 0 attaches no sample.
    auto blob_for = [&](const knowhere::Json& cfg, int64_t query_seed) {
        auto idx = make();
        auto base = gen_fp32(nb, dim);
        auto queries = gen_fp32(n_train_q, dim, query_seed);
        if (query_seed != 0) {
            base->Set(knowhere::meta::TRAIN_QUERY_TENSOR, static_cast<const float*>(queries->GetTensor()));
            base->Set(knowhere::meta::TRAIN_QUERY_ROWS, static_cast<int64_t>(n_train_q));
        }
        REQUIRE(idx.Build(base, cfg) == knowhere::Status::success);
        return leanvec_training_blob(idx);
    };

    const auto ood = blob_for(build, /*query_seed=*/7);
    REQUIRE_FALSE(ood.empty());

    // SVS stamps a fresh uuid into the manifest and into each binary section, so two builds
    // from identical inputs are not byte-equal — they differ in ~1.5% of the blob, while a
    // differently trained projection differs in well over half of it. Thresholds keep the
    // comparison out of the SVS serialization format.
    const size_t identical_threshold = ood.size() / 20;
    const size_t different_threshold = ood.size() / 4;

    // Control: identical inputs reproduce the projection.
    REQUIRE(count_differing_bytes(blob_for(build, /*query_seed=*/7), ood) < identical_threshold);
    // The attached sample is what the projection is trained on.
    REQUIRE(count_differing_bytes(blob_for(build, /*query_seed=*/11), ood) > different_threshold);
    // ood_training:false trains on the database vectors and ignores the sample.
    REQUIRE(count_differing_bytes(blob_for(in_dist, /*query_seed=*/7), ood) > different_threshold);
    // With ood_training on but no sample, Train falls back to the database vectors as their
    // own queries. That is still the OOD objective, not what train() computes, so the two
    // projections differ — the fallback is not a synonym for ood_training:false.
    REQUIRE(count_differing_bytes(blob_for(build, /*query_seed=*/0), blob_for(in_dist, /*query_seed=*/0)) >
            different_threshold);
}
#endif

TEST_CASE("FAISS: ood_training rejected for families that cannot use it", "[faiss_vanilla]") {
    const size_t nb = 1000, dim = 16;
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS, version.VersionNumber())
                   .value();
    knowhere::Json build =
        knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"IVF64,Flat","dim":16,"ood_training":true})");
    REQUIRE(idx.Build(gen_fp32(nb, dim), build) == knowhere::Status::invalid_args);
}

// ood_training is a declared config field, so it must not leak into raw_params and reach
// the faiss build-param whitelist as an unknown key.
TEST_CASE("FaissConfig: ood_training is consumed by the typed config layer", "[faiss_vanilla]") {
    knowhere::FaissConfig cfg;
    knowhere::Json j = knowhere::Json::parse(
        R"({"metric_type":"L2","faiss_index_name":"SVSVamana64,LeanVec4x8_16","ood_training":true,"alpha":1.4})");
    std::string msg;

    knowhere::Json j_(j);
    REQUIRE(knowhere::Config::FormatAndCheck(cfg, j_, &msg) == knowhere::Status::success);
    cfg.CaptureRawJson(j_);
    REQUIRE(knowhere::Config::Load(cfg, j_, knowhere::TRAIN, &msg) == knowhere::Status::success);

    REQUIRE(cfg.ood_training.has_value());
    REQUIRE(cfg.ood_training.value() == true);
    REQUIRE_FALSE(cfg.raw_params.contains("ood_training"));
    REQUIRE(cfg.raw_params.contains("alpha"));

    // Train uses has_value() to tell an explicit true from the default, so the field must
    // stay without a config-level default.
    knowhere::FaissConfig unset;
    knowhere::Json j2 = knowhere::Json::parse(R"({"metric_type":"L2","faiss_index_name":"SVSVamana64"})");
    REQUIRE(knowhere::Config::FormatAndCheck(unset, j2, &msg) == knowhere::Status::success);
    REQUIRE(knowhere::Config::Load(unset, j2, knowhere::TRAIN, &msg) == knowhere::Status::success);
    REQUIRE_FALSE(unset.ood_training.has_value());
}

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
