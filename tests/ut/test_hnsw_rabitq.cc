// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <faiss/VectorTransform.h>
#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>
#include <faiss/cppcontrib/knowhere/impl/RaBitQSearch.h>
#include <faiss/cppcontrib/knowhere/impl/StagedDistanceComputer.h>
#include <faiss/cppcontrib/knowhere/index_io.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/io.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "index/hnsw/impl/HnswSearchDispatch.h"
#include "index/hnsw/impl/IndexHNSWRaBitQWrapper.h"
#include "index/hnsw/impl/IndexHNSWWrapper.h"
#include "knowhere/bitsetview.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"

namespace rabitq_search = faiss::cppcontrib::knowhere::rabitq_search;

namespace {
// Independent reproduction of the original full-distance candidate batch:
// four-at-a-time, with scalar tails and no threshold/result state.
struct OriginalFullEvaluation {
    void
    begin(size_t) {
    }
    void
    record(float, int) {
    }
    template <class DC, class Emit>
    size_t
    compute(DC& dc, const size_t* ids, const int*, size_t count, int, Emit&& emit) {
        size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            float a, b, c, d;
            dc.distances_batch_4(ids[i], ids[i + 1], ids[i + 2], ids[i + 3], a, b, c, d);
            emit(i, a);
            emit(i + 1, b);
            emit(i + 2, c);
            emit(i + 3, d);
        }
        for (; i < count; ++i) emit(i, dc(ids[i]));
        return 0;
    }
};
}  // namespace

TEST_CASE("RaBitQ qb4 SIMD matches scalar including masked tails", "[hnsw_rabitq_core]") {
#if defined(__GNUC__) && defined(__x86_64__)
    if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") ||
        !__builtin_cpu_supports("avx512dq") || !__builtin_cpu_supports("avx512vl"))
        return;
    for (size_t bytes : {1, 7, 8, 15, 16, 31, 32, 63, 64, 65, 96, 192, 193}) {
        for (int seed = 0; seed < 8; ++seed) {
            std::vector<uint8_t> data(bytes + 1), query(bytes * 4 + 1);
            for (size_t i = 0; i < data.size(); ++i) data[i] = (i * 31 + seed * 73) % 256;
            for (size_t i = 0; i < query.size(); ++i) query[i] = (i * 17 + seed * 47) % 256;
            const auto expected = faiss::rabitq::bitwise_and_dot_product_with_popcount<faiss::SIMDLevel::NONE>(
                query.data() + 1, data.data() + 1, bytes, 4);
            const auto actual = faiss::rabitq::bitwise_and_dot_product_with_popcount<faiss::SIMDLevel::AVX512>(
                query.data() + 1, data.data() + 1, bytes, 4);
            REQUIRE(actual.dot_product == expected.dot_product);
            REQUIRE(actual.popcount == expected.popcount);
        }
    }
#endif
}

TEST_CASE("RaBitQ SIMD full scorers independently match scalar for multi-bit tails", "[hnsw_rabitq_core]") {
#if defined(__GNUC__) && defined(__x86_64__)
    const bool avx512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
    const bool avx2_supported = __builtin_cpu_supports("avx2");
    for (size_t d : {1, 7, 8, 15, 16, 23, 24, 31, 65, 200, 768, 1536}) {
        for (size_t ex : {1, 2, 3, 4, 5, 6, 7, 8}) {
            for (int seed = 1; seed <= 3; ++seed) {
                CAPTURE(d, ex, seed);
                std::vector<uint8_t> signs((d + 7) / 8), extra((d * ex + 7) / 8 + (ex == 8 ? 0 : 32));
                std::vector<float> query(d);
                for (size_t i = 0; i < signs.size(); ++i) signs[i] = (i * 73 + seed * 19) % 256;
                for (size_t i = 0; i < extra.size(); ++i) extra[i] = (i * 131 + seed * 37) % 256;
                for (size_t i = 0; i < d; ++i) query[i] = std::sin(float(i) * .37f + seed);
                const float cb = -float(1u << ex) + .5f;
                const float ref = faiss::rabitq::multibit::compute_inner_product<faiss::SIMDLevel::NONE>(
                    signs.data(), extra.data(), query.data(), d, ex, cb);
                if (avx512) {
                    const float actual = faiss::rabitq::multibit::compute_inner_product<faiss::SIMDLevel::AVX512>(
                        signs.data(), extra.data(), query.data(), d, ex, cb);
                    REQUIRE(actual == Catch::Approx(ref).epsilon(1e-5).margin(1e-3));
                }
                if (avx2_supported) {
                    const float avx2 = faiss::rabitq::multibit::compute_inner_product<faiss::SIMDLevel::AVX2>(
                        signs.data(), extra.data(), query.data(), d, ex, cb);
                    REQUIRE(avx2 == Catch::Approx(ref).epsilon(1e-5).margin(1e-3));
                }
            }
        }
    }
#endif
}

TEST_CASE("RaBitQ traversal retains all results when k covers the graph", "[hnsw_rabitq_core]") {
    namespace fk = faiss::cppcontrib::knowhere;
    for (const std::string metric : {"L2", "IP", "COSINE"}) {
        CAPTURE(metric);
        const bool similarity = metric != "L2", cosine = metric == "COSINE";
        const auto metric_type = similarity ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
        constexpr int n = 128, dim = 65;
        auto base = GenDataSet(n, dim, 121);
        auto queries = GenDataSet(4, dim, 122);
        const auto* x = static_cast<const float*>(base->GetTensor());
        // Use one connected topology for this full-coverage distance-order test.
        fk::IndexHNSWFlat fp32(dim, 16, faiss::METRIC_L2);
        fp32.add(n, x);
        auto* rq = new faiss::IndexRaBitQ(dim, metric_type, 8);
        rq->qb = 4;
        auto* rr = new faiss::RandomRotationMatrix(dim, dim);
        std::unique_ptr<faiss::IndexPreTransform> storage_owner(cosine ? new fk::IndexPreTransformRaBitQCosine(rr, rq)
                                                                       : new faiss::IndexPreTransform(rr, rq));
        auto& storage = *storage_owner;
        storage.own_fields = true;
        storage.train(n, x);
        storage.add(n, x);
        std::unique_ptr<fk::IndexHNSWRaBitQ> graph_owner(cosine ? new fk::IndexHNSWRaBitQCosine()
                                                                : new fk::IndexHNSWRaBitQ());
        auto& graph = *graph_owner;
        graph.d = dim;
        graph.ntotal = n;
        graph.metric_type = metric_type;
        graph.storage = &storage;
        graph.own_fields = false;
        graph.hnsw = std::move(fp32.hnsw);
        std::unique_ptr<faiss::DistanceComputer> full(storage.get_distance_computer());
        std::vector<float> distances(4 * n);
        std::vector<faiss::idx_t> labels(4 * n);
        rabitq_search::search(graph, 4, static_cast<const float*>(queries->GetTensor()), n, distances.data(),
                              labels.data(), n, true);
        std::vector<float> api_distances(4 * n);
        std::vector<faiss::idx_t> api_labels(4 * n);
        knowhere::IndexHNSWRaBitQWrapper api(&graph);
        knowhere::SearchParametersHNSWWrapper params;
        params.efSearch = n;
        api.search(4, static_cast<const float*>(queries->GetTensor()), n, api_distances.data(), api_labels.data(),
                   &params);
        for (int i = 0; i < 4 * n; ++i) {
            REQUIRE(api_labels[i] == labels[i]);
            REQUIRE(api_distances[i] == Catch::Approx((similarity ? -1.f : 1.f) * distances[i]).margin(1e-5));
        }
        for (int q = 0; q < 4; ++q) {
            full->set_query(static_cast<const float*>(queries->GetTensor()) + q * dim);
            std::vector<std::pair<float, faiss::idx_t>> expected;
            for (int i = 0; i < n; ++i) expected.emplace_back((similarity ? -1.f : 1.f) * (*full)(i), i);
            std::sort(expected.begin(), expected.end());
            for (int i = 0; i < n; ++i) {
                CAPTURE(q, i, distances[q * n + i], expected[i].first);
                REQUIRE(labels[q * n + i] == expected[i].second);
                REQUIRE(distances[q * n + i] == Catch::Approx(expected[i].first).margin(1e-5));
            }
        }
    }
}

TEST_CASE("RaBitQ staged distances preserve metric and cosine threshold semantics", "[hnsw_rabitq]") {
    namespace fk = faiss::cppcontrib::knowhere;
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        const bool cosine = std::string(metric) == "COSINE";
        const bool similarity = std::string(metric) != "L2";
        for (int qb : {0, 4}) {
            auto base = GenDataSet(128, 65, 4201);
            auto query = GenDataSet(4, 65, 4202);
            const auto* x = static_cast<const float*>(base->GetTensor());
            auto* rr = new faiss::RandomRotationMatrix(65, 65);
            auto* rq = new faiss::IndexRaBitQ(65, similarity ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2, 8);
            rq->qb = qb;
            std::unique_ptr<faiss::IndexPreTransform> storage(cosine ? new fk::IndexPreTransformRaBitQCosine(rr, rq)
                                                                     : new faiss::IndexPreTransform(rr, rq));
            storage->own_fields = true;
            storage->train(128, x);
            storage->add(128, x);
            std::unique_ptr<fk::IndexHNSWRaBitQ> graph(cosine ? new fk::IndexHNSWRaBitQCosine()
                                                              : new fk::IndexHNSWRaBitQ());
            graph->d = 65;
            graph->metric_type = rq->metric_type;
            graph->storage = storage.get();
            graph->own_fields = false;
            std::unique_ptr<faiss::DistanceComputer> staged_owner(graph->get_staged_distance_computer());
            auto* staged = dynamic_cast<fk::StagedDistanceComputer*>(staged_owner.get());
            REQUIRE(staged != nullptr);
            std::unique_ptr<faiss::DistanceComputer> full(storage->get_distance_computer());
            std::unique_ptr<faiss::FlatCodesDistanceComputer> raw_owner(rq->get_FlatCodesDistanceComputer());
            auto* raw = dynamic_cast<faiss::RaBitQDistanceComputer*>(raw_owner.get());
            REQUIRE(raw != nullptr);
            std::vector<float> rotated(65);
            for (int q = 0; q < 4; ++q) {
                const auto* v = static_cast<const float*>(query->GetTensor()) + q * 65;
                staged->set_query(v);
                full->set_query(v);
                rr->apply_noalloc(1, v, rotated.data());
                raw->set_query(rotated.data());
                for (int i = 0; i < 128; ++i) {
                    const float expected = (similarity ? -1 : 1) * (*full)(i);
                    REQUIRE((*staged)(i) == Catch::Approx(expected).margin(1e-5));
                    REQUIRE(staged->evaluate(i, std::numeric_limits<float>::infinity()) ==
                            Catch::Approx(expected).margin(1e-5));
                    const float scale = cosine ? dynamic_cast<fk::IndexPreTransformRaBitQCosine*>(storage.get())
                                                         ->get_inverse_l2_norms()[i] /
                                                     std::sqrt(faiss::fvec_norm_L2sqr(v, 65))
                                               : 1;
                    const float estimate =
                        raw->distance_to_code_1bit(raw->codes + i * raw->code_size) * scale * (similarity ? -1 : 1);
                    REQUIRE(staged->evaluate(i, -std::numeric_limits<float>::infinity()) ==
                            Catch::Approx(estimate).margin(1e-5));
                }
                REQUIRE(staged->estimate_count == 256);
                REQUIRE(staged->refine_count == 128);
                // Finite thresholds: compare the scaled inequality against the
                // previous raw-threshold division, away from rounding ties.
                for (int i = 0; i < 16; ++i) {
                    const auto* code = raw->codes + i * raw->code_size;
                    const auto* factors =
                        reinterpret_cast<const faiss::rabitq_utils::SignBitFactorsWithError*>(code + (raw->d + 7) / 8);
                    const float scale = cosine ? dynamic_cast<fk::IndexPreTransformRaBitQCosine*>(storage.get())
                                                         ->get_inverse_l2_norms()[i] /
                                                     std::sqrt(faiss::fvec_norm_L2sqr(v, 65))
                                               : 1.f;
                    const float estimate = raw->distance_to_code_1bit(code);
                    for (float offset : {-10.f, -1.f, .125f, 1.f, 10.f}) {
                        const float threshold = (similarity ? -estimate : estimate) * scale + offset;
                        const bool refine = faiss::rabitq_utils::should_refine_candidate(
                            estimate, factors->f_error, raw->g_error, (similarity ? -threshold : threshold) / scale,
                            similarity);
                        const float expected =
                            (refine ? raw->distance_to_code_full(code) : estimate) * scale * (similarity ? -1.f : 1.f);
                        const auto before = staged->refine_count;
                        REQUIRE(staged->evaluate(i, threshold) == Catch::Approx(expected).margin(1e-5));
                        REQUIRE(staged->refine_count - before == size_t(refine));
                    }
                }
            }
        }
    }
}

TEST_CASE("HNSW RaBitQ metrics and serialized search", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        for (int qb : {0, 4}) {
            CAPTURE(metric, qb);
            auto base = GenDataSet(1024, 65, 3101);
            auto query = GenDataSet(16, 65, 3102);
            const auto* data = static_cast<const float*>(base->GetTensor());
            std::vector<float> original(data, data + 1024 * 65);
            knowhere::Json config = {{"dim", 65},     {"metric_type", metric}, {"k", 20},
                                     {"M", 16},       {"efConstruction", 100}, {"ef", 200},
                                     {"rbq_bits", 8}, {"rbq_bits_query", qb}};
            auto index = knowhere::IndexFactory::Instance()
                             .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                             .value();
            REQUIRE(index.Build(base, config) == knowhere::Status::success);
            REQUIRE(std::equal(original.begin(), original.end(), data));
            auto before = index.Search(query, config, nullptr);
            REQUIRE(before.has_value());
            for (int excluded : {128, 960}) {
                std::vector<uint8_t> mask(128, 0);
                for (int i = 0; i < excluded; ++i) mask[i / 8] |= uint8_t(1u << (i % 8));
                knowhere::BitsetView filter(mask.data(), 1024);
                auto filtered = index.Search(query, config, filter);
                REQUIRE(filtered.has_value());
                for (int i = 0; i < 320; ++i) {
                    REQUIRE(filtered.value()->GetIds()[i] >= excluded);
                    REQUIRE(filtered.value()->GetIds()[i] < 1024);
                }
            }
            for (int q = 0; q < 16; ++q)
                for (int j = 0; j < 20; ++j) {
                    int i = q * 20 + j;
                    REQUIRE(before.value()->GetIds()[i] >= 0);
                    REQUIRE(before.value()->GetIds()[i] < 1024);
                    REQUIRE(std::isfinite(before.value()->GetDistance()[i]));
                    if (j) {
                        if (std::string(metric) == "L2")
                            REQUIRE(before.value()->GetDistance()[i - 1] <= before.value()->GetDistance()[i]);
                        else
                            REQUIRE(before.value()->GetDistance()[i - 1] >= before.value()->GetDistance()[i]);
                    }
                }
            knowhere::BinarySet binary;
            REQUIRE(index.Serialize(binary) == knowhere::Status::success);
            auto loaded = knowhere::IndexFactory::Instance()
                              .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                              .value();
            REQUIRE(loaded.Deserialize(binary, config) == knowhere::Status::success);
            auto after = loaded.Search(query, config, nullptr);
            REQUIRE(after.has_value());
            for (int i = 0; i < 320; ++i) {
                REQUIRE(before.value()->GetIds()[i] == after.value()->GetIds()[i]);
                REQUIRE(before.value()->GetDistance()[i] == after.value()->GetDistance()[i]);
            }
        }
    }
}

TEST_CASE("RaBitQ public search supports all database and query bit widths", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto base = GenDataSet(128, 33, 731);
    auto query = GenDataSet(3, 33, 732);
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        for (int bits = 1; bits <= 9; ++bits) {
            CAPTURE(metric, bits);
            knowhere::Json config = {{"dim", 33}, {"metric_type", metric}, {"k", 10},
                                     {"M", 8},    {"efConstruction", 64},  {"ef", 64}};
            // Also exercise the public default (RBQ1).
            if (bits != 1)
                config["rbq_bits"] = bits;
            auto index = knowhere::IndexFactory::Instance()
                             .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                             .value();
            REQUIRE(index.Build(base, config) == knowhere::Status::success);
            knowhere::BinarySet original;
            REQUIRE(index.Serialize(original) == knowhere::Status::success);
            knowhere::DataSetPtr unquantized;
            for (int qb = 0; qb <= 8; ++qb) {
                CAPTURE(qb);
                config["rbq_bits_query"] = qb;
                auto result = index.Search(query, config, nullptr);
                REQUIRE(result.has_value());
                if (qb == 0)
                    unquantized = result.value();
                if (qb == 1 && bits == 1) {
                    // Detect accidental parameter slicing/default-only routing.
                    bool different = false;
                    for (int i = 0; i < 30; ++i) {
                        different |= result.value()->GetIds()[i] != unquantized->GetIds()[i] ||
                                     result.value()->GetDistance()[i] != unquantized->GetDistance()[i];
                    }
                    REQUIRE(different);
                }
                for (int i = 0; i < 30; ++i) {
                    REQUIRE(result.value()->GetIds()[i] >= 0);
                    REQUIRE(std::isfinite(result.value()->GetDistance()[i]));
                }
                for (int excluded : {16, 124, 128}) {
                    CAPTURE(excluded);
                    std::vector<uint8_t> mask(16, 0);
                    for (int i = 0; i < excluded; ++i) mask[i / 8] |= uint8_t(1u << (i % 8));
                    auto filtered = index.Search(query, config, knowhere::BitsetView(mask.data(), 128));
                    REQUIRE(filtered.has_value());
                    for (int i = 0; i < 30; ++i) {
                        const auto id = filtered.value()->GetIds()[i];
                        REQUIRE((id == -1 || (id >= excluded && id < 128)));
                    }
                }
            }
            // Request qb never mutates serialized storage defaults or codes.
            knowhere::BinarySet after;
            REQUIRE(index.Serialize(after) == knowhere::Status::success);
            REQUIRE(original.binary_map_.size() == after.binary_map_.size());
            for (const auto& [name, bin] : original.binary_map_) {
                const auto copy = after.GetByName(name);
                REQUIRE(bin->size == copy->size);
                REQUIRE(std::memcmp(bin->data.get(), copy->data.get(), bin->size) == 0);
            }
            for (int invalid : {-1, 9}) {
                config["rbq_bits_query"] = invalid;
                REQUIRE_FALSE(index.Search(query, config, nullptr).has_value());
            }
        }
    }
}

TEST_CASE("RaBitQ request qb survives refine range iterator and concurrent search", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto base = GenDataSet(256, 33, 833);
    auto query = GenDataSet(1, 33, 834);
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        for (bool refine : {false, true}) {
            CAPTURE(metric, refine);
            knowhere::Json config = {{"dim", 33}, {"metric_type", metric}, {"k", 10}, {"M", 8}, {"efConstruction", 64},
                                     {"ef", 128}, {"rbq_bits", 9}};
            if (refine) {
                config["refine"] = true;
                config["refine_type"] = "fp16";
                config["refine_k"] = 1.5;
            }
            auto index = knowhere::IndexFactory::Instance()
                             .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                             .value();
            REQUIRE(index.Build(base, config) == knowhere::Status::success);
            std::vector<knowhere::DataSetPtr> references;
            std::vector<std::future<knowhere::DataSetPtr>> jobs;
            for (int qb : {0, 4, 8}) {
                auto request = config;
                request["rbq_bits_query"] = qb;
                auto result = index.Search(query, request, nullptr);
                REQUIRE(result.has_value());
                references.push_back(result.value());
                jobs.push_back(std::async(std::launch::async,
                                          [&, request] { return index.Search(query, request, nullptr).value(); }));
                auto iterators = index.AnnIterator(query, request, nullptr);
                REQUIRE(iterators.has_value());
                auto& it = iterators.value()[0];
                for (int n = 0; n < 10; ++n) {
                    REQUIRE(it->HasNext().value());
                    const auto [id, distance] = it->Next().value();
                    REQUIRE(id >= 0);
                    REQUIRE(id < 256);
                    REQUIRE(std::isfinite(distance));
                }
                request["radius"] = std::string(metric) == "L2" ? 1e6 : -1e6;
                auto range = index.RangeSearch(query, request, nullptr);
                REQUIRE(range.has_value());
                REQUIRE(range.value()->GetLims()[1] > 0);
                request["trace_visit"] = true;
                REQUIRE(index.Search(query, request, nullptr).has_value());
            }
            for (size_t j = 0; j < jobs.size(); ++j) {
                const auto result = jobs[j].get();
                for (int i = 0; i < 10; ++i) {
                    REQUIRE(result->GetIds()[i] == references[j]->GetIds()[i]);
                    REQUIRE(result->GetDistance()[i] == references[j]->GetDistance()[i]);
                }
            }
        }
    }
}

TEST_CASE("RaBitQ 9-bit storage supports floating input formats and loaded request parameters", "[hnsw_rabitq]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto base = GenDataSet(128, 33, 913);
    auto query = GenDataSet(2, 33, 914);
    auto* base_values = const_cast<float*>(static_cast<const float*>(base->GetTensor()));
    auto* query_values = const_cast<float*>(static_cast<const float*>(query->GetTensor()));
    for (int i = 0; i < 128 * 33; ++i) base_values[i] -= 50.f;
    for (int i = 0; i < 2 * 33; ++i) query_values[i] -= 50.f;
    std::fill_n(base_values, 33, 0.f);
    std::fill_n(query_values, 33, 0.f);
    auto exercise = [&](auto tag) {
        using T = decltype(tag);
        auto typed_base = knowhere::data_type_conversion<float, T>(*base);
        auto typed_query = knowhere::data_type_conversion<float, T>(*query);
        for (const auto* metric : {"L2", "IP", "COSINE"}) {
            CAPTURE(metric, sizeof(T));
            knowhere::Json config = {{"dim", 33}, {"metric_type", metric}, {"k", 10}, {"M", 8}, {"efConstruction", 64},
                                     {"ef", 100}, {"rbq_bits", 9}};
            auto index =
                knowhere::IndexFactory::Instance().Create<T>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version).value();
            REQUIRE(index.Build(typed_base, config) == knowhere::Status::success);
            knowhere::BinarySet binary;
            REQUIRE(index.Serialize(binary) == knowhere::Status::success);
            auto loaded =
                knowhere::IndexFactory::Instance().Create<T>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version).value();
            REQUIRE(loaded.Deserialize(binary, config) == knowhere::Status::success);
            for (int qb : {0, 4, 8}) {
                config["rbq_bits_query"] = qb;
                auto a = index.Search(typed_query, config, nullptr);
                auto b = loaded.Search(typed_query, config, nullptr);
                REQUIRE(a.has_value());
                REQUIRE(b.has_value());
                for (int i = 0; i < 20; ++i) {
                    REQUIRE(a.value()->GetIds()[i] == b.value()->GetIds()[i]);
                    REQUIRE(std::isfinite(a.value()->GetDistance()[i]));
                    REQUIRE(a.value()->GetDistance()[i] == b.value()->GetDistance()[i]);
                }
            }
        }
    };
    exercise(knowhere::fp32{});
    exercise(knowhere::fp16{});
    exercise(knowhere::bf16{});
}

TEST_CASE("Generic HNSW parameter factory preserves SQ and PQ searches", "[hnsw_rabitq_regression]") {
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto base = GenDataSet(1024, 32, 931);
    auto query = GenDataSet(2, 32, 932);
    for (const auto* name : {"HNSW_SQ", "HNSW_PQ"}) {
        for (const auto* metric : {"L2", "IP", "COSINE"}) {
            CAPTURE(name, metric);
            knowhere::Json config = {
                {"dim", 32}, {"metric_type", metric}, {"k", 10}, {"M", 8},    {"efConstruction", 64},
                {"ef", 128}, {"sq_type", "SQ8"},      {"m", 4},  {"nbits", 4}};
            auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
            REQUIRE(index.Build(base, config) == knowhere::Status::success);
            auto result = index.Search(query, config, nullptr);
            REQUIRE(result.has_value());
            // Same graph and codes: compare the new default evaluator with
            // the pre-refactor full batch rule, including moderate filtering.
            knowhere::BinarySet binary;
            REQUIRE(index.Serialize(binary) == knowhere::Status::success);
            auto blob = binary.binary_map_.begin()->second;
            faiss::VectorIOReader reader;
            reader.data.assign(blob->data.get(), blob->data.get() + blob->size);
            std::unique_ptr<faiss::Index> decoded(faiss::cppcontrib::knowhere::read_index(&reader));
            auto* graph = dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSW*>(decoded.get());
            REQUIRE(graph != nullptr);
            for (int excluded : {0, 128}) {
                std::vector<uint8_t> bits(128, 0);
                std::fill_n(bits.begin(), excluded / 8, uint8_t(255));
                knowhere::BitsetView filter(bits.data(), 1024, excluded);
                knowhere::BitsetViewIDSelector selector(filter);
                knowhere::SearchParametersHNSWWrapper parameters;
                parameters.efSearch = 128;
                parameters.sel = excluded ? &selector : nullptr;
                parameters.kAlpha = filter.filter_ratio() * 0.7f;
                auto actual = index.Search(query, config, excluded ? filter : knowhere::BitsetView{});
                REQUIRE(actual.has_value());
                for (int q = 0; q < 2; ++q) {
                    std::unique_ptr<faiss::DistanceComputer> dc(graph->storage->get_distance_computer());
                    const bool similarity = std::string(metric) != "L2";
                    if (similarity)
                        dc.reset(new faiss::NegativeDistanceComputer(dc.release()));
                    dc->set_query(static_cast<const float*>(query->GetTensor()) + q * 32);
                    auto visited = faiss::cppcontrib::knowhere::Bitset::create_cleared(1024);
                    float distances[10];
                    faiss::idx_t ids[10];
                    knowhere::search_hnsw_query<OriginalFullEvaluation>(graph->hnsw, *dc, visited, 10, distances, ids,
                                                                        &parameters);
                    for (int j = 0; j < 10; ++j) {
                        REQUIRE(actual.value()->GetIds()[q * 10 + j] == ids[j]);
                        REQUIRE(actual.value()->GetDistance()[q * 10 + j] ==
                                (similarity ? -distances[j] : distances[j]));
                    }
                }
            }
            auto iterators = index.AnnIterator(query, config, nullptr);
            REQUIRE(iterators.has_value());
            REQUIRE(iterators.value()[0]->HasNext().value());
            REQUIRE(std::isfinite(iterators.value()[0]->Next().value().second));
            std::vector<uint8_t> mask(128, 255);
            mask.back() = 0;
            auto filtered = index.Search(query, config, knowhere::BitsetView(mask.data(), 1024));
            REQUIRE(filtered.has_value());
            for (int i = 0; i < 20; ++i) {
                auto id = filtered.value()->GetIds()[i];
                REQUIRE((id == -1 || (id >= 1016 && id < 1024)));
            }
        }
    }
}
