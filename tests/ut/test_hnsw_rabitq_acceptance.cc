// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/VectorTransform.h>
#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>
#include <faiss/cppcontrib/knowhere/IndexRefine.h>
#include <faiss/cppcontrib/knowhere/impl/RaBitQBuildUtils.h>
#include <faiss/cppcontrib/knowhere/impl/StagedDistanceComputer.h>
#include <faiss/cppcontrib/knowhere/index_io.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <set>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "index/hnsw/impl/IndexHNSWWrapper.h"
#include "index/hnsw/impl/RaBitQSearchParameters.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/feder/HNSW.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"

TEST_CASE("RBQ bounded add preserves input slices and rejects invalid sizes", "[hnsw_rabitq_acceptance][rbq_build]") {
    using faiss::cppcontrib::knowhere::rabitq_build::add_in_blocks;
    struct RecordingIndex : faiss::IndexFlatL2 {
        std::vector<std::pair<faiss::idx_t, const float*>> calls;
        RecordingIndex() : faiss::IndexFlatL2(3) {
        }
        void
        add(faiss::idx_t n, const float* x) override {
            calls.emplace_back(n, x);
            ntotal += n;
        }
    };
    std::vector<float> data(8193 * 3);
    for (const auto n : {0, 1, 4095, 4096, 4097, 8193}) {
        RecordingIndex index;
        add_in_blocks(index, n, n ? data.data() : nullptr);
        REQUIRE(index.ntotal == n);
        REQUIRE(index.calls.size() == size_t((n + 4095) / 4096));
        for (size_t i = 0; i < index.calls.size(); ++i) {
            REQUIRE(index.calls[i].first == std::min(4096, n - int(i) * 4096));
            REQUIRE(index.calls[i].second == data.data() + i * 4096 * 3);
        }
    }
    RecordingIndex index;
    add_in_blocks(index, 17, data.data(), 7);
    REQUIRE(index.calls.size() == 3);
    REQUIRE(index.calls.back().first == 3);
    add_in_blocks(index, 2, data.data());
    REQUIRE(index.ntotal == 19);
    const auto calls = index.calls.size();
    REQUIRE_THROWS(add_in_blocks(index, -1, data.data()));
    REQUIRE_THROWS(add_in_blocks(index, 1, data.data(), 0));
    REQUIRE_THROWS(add_in_blocks(index, 1, data.data(), -1));
    REQUIRE_THROWS(add_in_blocks(index, 1, nullptr));
    REQUIRE_THROWS(add_in_blocks(index, std::numeric_limits<faiss::idx_t>::max(), data.data()));
    index.d = 0;
    REQUIRE_THROWS(add_in_blocks(index, 1, data.data()));
    REQUIRE(index.calls.size() == calls);
}

TEST_CASE("RBQ bounded storage encoding preserves codes norms and serialization",
          "[hnsw_rabitq_acceptance][rbq_build]") {
    namespace fk = faiss::cppcontrib::knowhere;
    constexpr int n = 4101, d = 33;
    auto dataset = GenDataSet(n, d, 29091);
    const auto* x = static_cast<const float*>(dataset->GetTensor());
    for (const bool cosine : {false, true}) {
        for (const auto metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
            if (cosine && metric != faiss::METRIC_INNER_PRODUCT)
                continue;
            for (const uint8_t bits : {1, 4, 8, 9}) {
                CAPTURE(cosine, metric, int(bits));
                auto make_storage = [&]() -> std::unique_ptr<faiss::IndexPreTransform> {
                    auto rotation = std::make_unique<faiss::RandomRotationMatrix>(d, d);
                    auto leaf = std::make_unique<faiss::IndexRaBitQ>(d, metric, bits);
                    std::unique_ptr<faiss::IndexPreTransform> storage;
                    if (cosine) {
                        storage = std::make_unique<fk::IndexPreTransformRaBitQCosine>(rotation.get(), leaf.get());
                    } else {
                        storage = std::make_unique<faiss::IndexPreTransform>(rotation.get(), leaf.get());
                    }
                    storage->own_fields = true;
                    rotation.release();
                    leaf.release();
                    storage->train(n, x);
                    return storage;
                };
                auto reference = make_storage();
                auto actual = make_storage();
                // Independent reproduction of the pre-refactor add_to_index loop.
                for (int offset = 0; offset < n; offset += 4096) {
                    reference->add(std::min(4096, n - offset), x + offset * d);
                }
                fk::rabitq_build::add_in_blocks(*actual, n, x);
                REQUIRE(actual->ntotal == n);
                auto* rbq = dynamic_cast<faiss::IndexRaBitQ*>(actual->index);
                auto* expected = dynamic_cast<faiss::IndexRaBitQ*>(reference->index);
                REQUIRE(rbq != nullptr);
                REQUIRE(expected != nullptr);
                REQUIRE(rbq->center == expected->center);
                REQUIRE(rbq->codes == expected->codes);
                if (cosine) {
                    auto* norms = dynamic_cast<fk::IndexPreTransformRaBitQCosine*>(actual.get());
                    REQUIRE(norms != nullptr);
                    norms->validate_norms();
                    for (const int row : {0, 4095, 4096, 4100}) {
                        const auto norm2 = faiss::fvec_norm_L2sqr(x + row * d, d);
                        REQUIRE(norms->get_inverse_l2_norms()[row] == Catch::Approx(1 / std::sqrt(norm2)));
                    }
                }
                faiss::VectorIOWriter before, after;
                fk::write_index(reference.get(), &before);
                fk::write_index(actual.get(), &after);
                REQUIRE(before.data == after.data);
                faiss::VectorIOReader reader;
                reader.data = after.data;
                std::unique_ptr<faiss::Index> loaded(fk::read_index(&reader));
                REQUIRE(loaded->ntotal == n);
                faiss::VectorIOWriter roundtrip;
                fk::write_index(loaded.get(), &roundtrip);
                REQUIRE(roundtrip.data == after.data);
                std::unique_ptr<faiss::DistanceComputer> a(actual->get_distance_computer());
                std::unique_ptr<faiss::DistanceComputer> b(loaded->get_distance_computer());
                a->set_query(x);
                b->set_query(x);
                for (const int row : {0, 4095, 4096, 4100}) REQUIRE((*a)(row) == (*b)(row));
            }
        }
    }
}

TEST_CASE("RBQ bounded add keeps outer refinement in the original input space", "[hnsw_rabitq_acceptance][rbq_build]") {
    namespace fk = faiss::cppcontrib::knowhere;
    constexpr int n = 4101, d = 16;
    auto dataset = GenDataSet(n, d, 29092);
    const auto* x = static_cast<const float*>(dataset->GetTensor());
    faiss::RandomRotationMatrix rotation(d, d);
    faiss::IndexRaBitQ leaf(d, faiss::METRIC_L2, 4);
    faiss::IndexPreTransform storage(&rotation, &leaf);
    fk::IndexRefineFlat refine(&storage);
    refine.train(n, x);
    fk::rabitq_build::add_in_blocks(refine, n, x);
    REQUIRE(refine.ntotal == n);
    REQUIRE(storage.ntotal == n);
    REQUIRE(leaf.ntotal == n);
    REQUIRE(refine.refine_index->ntotal == n);
    std::vector<float> reconstructed(d);
    for (const int row : {0, 4095, 4096, 4100}) {
        refine.reconstruct(row, reconstructed.data());
        REQUIRE(std::equal(reconstructed.begin(), reconstructed.end(), x + row * d));
    }
}

TEST_CASE("RaBitQ memory serialization permits empty transfers without touching null pointers",
          "[hnsw_rabitq_acceptance]") {
    knowhere::MemoryIOWriter writer;
    REQUIRE(writer(nullptr, sizeof(float), 0) == 0);
    REQUIRE(writer(nullptr, 0, 17) == 0);
    REQUIRE(writer.tellg() == 0);
    const uint32_t value = 0x12345678;
    REQUIRE(writer(&value, sizeof(value), 1) == 1);
    std::unique_ptr<uint8_t[]> bytes(writer.data());
    REQUIRE(writer(nullptr, sizeof(float), 0) == 0);
    REQUIRE(writer.tellg() == sizeof(value));
    knowhere::MemoryIOReader reader(bytes.get(), writer.tellg());
    REQUIRE(reader(nullptr, sizeof(float), 0) == 0);
    REQUIRE(reader(nullptr, 0, 17) == 0);
    REQUIRE(reader.tellg() == 0);
    uint32_t copy = 0;
    REQUIRE(reader(&copy, sizeof(copy), 1) == 1);
    REQUIRE(copy == value);
    REQUIRE(reader(nullptr, sizeof(float), 0) == 0);
}

TEST_CASE("RaBitQ byte-aligned bitwise kernels match independent byte oracle", "[hnsw_rabitq_core]") {
    auto exercise = [&](auto level_tag) {
        constexpr auto level = decltype(level_tag)::value;
        for (size_t size : {1, 7, 8, 9, 15, 31, 32, 63, 64, 65, 192})
            for (size_t qb = 1; qb <= 8; ++qb) {
                std::vector<uint8_t> data(size + 1), query(size * qb + 1);
                for (size_t i = 0; i < data.size(); ++i) data[i] = i * 37 + 11;
                for (size_t i = 0; i < query.size(); ++i) query[i] = i * 71 + 19;
                const auto* x = data.data() + 1;
                const auto* q = query.data() + 1;
                uint64_t dot = 0, xor_dot = 0, pop = 0;
                for (size_t i = 0; i < size; ++i) {
                    pop += __builtin_popcount(unsigned(x[i]));
                    for (size_t bit = 0; bit < qb; ++bit) {
                        dot += uint64_t(__builtin_popcount(unsigned(x[i] & q[bit * size + i]))) << bit;
                        xor_dot += uint64_t(__builtin_popcount(unsigned(x[i] ^ q[bit * size + i]))) << bit;
                    }
                }
                REQUIRE(faiss::rabitq::bitwise_and_dot_product<level>(q, x, size, qb) == dot);
                REQUIRE(faiss::rabitq::bitwise_xor_dot_product<level>(q, x, size, qb) == xor_dot);
                REQUIRE(faiss::rabitq::popcount<level>(x, size) == pop);
                auto fused = faiss::rabitq::bitwise_and_dot_product_with_popcount<level>(q, x, size, qb);
                REQUIRE(fused.dot_product == dot);
                REQUIRE(fused.popcount == pop);
            }
    };
    exercise(std::integral_constant<faiss::SIMDLevel, faiss::SIMDLevel::NONE>{});
#if defined(__GNUC__) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2"))
        exercise(std::integral_constant<faiss::SIMDLevel, faiss::SIMDLevel::AVX2>{});
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl"))
        exercise(std::integral_constant<faiss::SIMDLevel, faiss::SIMDLevel::AVX512>{});
#endif
}

TEST_CASE("RaBitQ iterators own query and parameter state while the parent index is retained",
          "[hnsw_rabitq_acceptance]") {
    for (int qb : {0, 4, 8}) {
        // Common IndexIterator borrows the node-owned result IdMap. The parent
        // index must outlive iteration; only request-local objects are released.
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                 knowhere::Version::GetCurrentVersion().VersionNumber())
                         .value();
        auto create_iterator = [&] {
            auto base = GenDataSet(128, 33, 2001);
            auto query = GenDataSet(1, 33, 2002);
            knowhere::Json cfg = {
                {"dim", 33},     {"metric_type", "COSINE"}, {"M", 16}, {"efConstruction", 100}, {"ef", 128}, {"k", 128},
                {"rbq_bits", 9}, {"rbq_bits_query", qb}};
            REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
            auto all = index.Search(query, cfg, nullptr);
            REQUIRE(all.has_value());
            std::vector<float> distance(128);
            for (int i = 0; i < 128; ++i) distance[all.value()->GetIds()[i]] = all.value()->GetDistance()[i];
            auto iterators = index.AnnIterator(query, cfg, nullptr);
            REQUIRE(iterators.has_value());
            cfg["rbq_bits_query"] = 9;
            cfg["radius"] = 0.5;
            REQUIRE_FALSE(index.Search(query, cfg, nullptr).has_value());
            REQUIRE_FALSE(index.AnnIterator(query, cfg, nullptr).has_value());
            REQUIRE_FALSE(index.RangeSearch(query, cfg, nullptr).has_value());
            return std::make_pair(iterators.value()[0], distance);
        };
        auto [iterator, distance] = create_iterator();
        std::set<int64_t> ids;
        while (iterator->HasNext().value()) {
            const auto [id, value] = iterator->Next().value();
            REQUIRE(id >= 0);
            REQUIRE(id < 128);
            REQUIRE(ids.insert(id).second);
            REQUIRE(value == Catch::Approx(distance[id]).margin(1e-5));
        }
        REQUIRE_FALSE(ids.empty());
    }
}

// Opt-in data-backed diagnostic; ordinary CI does not require benchmark files.
TEST_CASE("RaBitQ original COSINE and normalized IP real-data metric diagnostic", "[.hnsw_rabitq_realdata]") {
    const char* data_root = std::getenv("KNOWHERE_RBQ_ACCEPTANCE_DATA");
    REQUIRE(data_root != nullptr);
    constexpr int n = 4096, nq = 16, k = 100;
    auto read_prefix = [&](const std::string& path, int rows, int& dim) {
        std::ifstream input(path, std::ios::binary);
        REQUIRE(input.good());
        uint32_t header[2];
        input.read(reinterpret_cast<char*>(header), sizeof(header));
        REQUIRE(input.good());
        REQUIRE(header[0] >= uint32_t(rows));
        REQUIRE(header[1] > 0);
        REQUIRE(header[1] <= 8192);
        dim = header[1];
        std::vector<float> values(size_t(rows) * dim);
        input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
        REQUIRE(input.good());
        return values;
    };
    for (const auto* dataset : {"cohere", "openai"}) {
        int d, qd;
        const auto prefix = std::string(data_root) + "/" + dataset + "/" + dataset;
        auto raw = read_prefix(prefix + ".fbin", n, d);
        auto raw_q = read_prefix(prefix + "_query.fbin", nq, qd);
        REQUIRE(d == qd);
        auto normalized = raw, normalized_q = raw_q;
        faiss::fvec_renorm_L2(d, n, normalized.data());
        faiss::fvec_renorm_L2(d, nq, normalized_q.data());
        faiss::IndexFlatIP exact(d);
        exact.add(n, normalized.data());
        std::vector<float> gt_distance(nq * k);
        std::vector<faiss::idx_t> gt(nq * k);
        exact.search(nq, normalized_q.data(), k, gt_distance.data(), gt.data());
        for (int bits : {8, 9})
            for (const auto* metric : {"COSINE", "IP"}) {
                CAPTURE(dataset, bits, metric);
                const bool cosine = std::string(metric) == "COSINE";
                auto base = knowhere::GenDataSet(n, d, cosine ? raw.data() : normalized.data());
                auto queries = knowhere::GenDataSet(nq, d, cosine ? raw_q.data() : normalized_q.data());
                base->SetIsOwner(false);
                queries->SetIsOwner(false);
                knowhere::Json cfg = {{"dim", d}, {"metric_type", metric}, {"rbq_bits", bits},
                                      {"M", 30},  {"efConstruction", 360}, {"ef", 500},
                                      {"k", k},   {"rbq_bits_query", 4}};
                auto index = knowhere::IndexFactory::Instance()
                                 .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                         knowhere::Version::GetCurrentVersion().VersionNumber())
                                 .value();
                REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
                auto result = index.Search(queries, cfg, nullptr);
                REQUIRE(result.has_value());
                int hits = 0;
                double error = 0;
                for (int q = 0; q < nq; ++q)
                    for (int i = 0; i < k; ++i) {
                        auto id = result.value()->GetIds()[q * k + i];
                        REQUIRE(id >= 0);
                        REQUIRE(id < n);
                        hits += std::find(gt.begin() + q * k, gt.begin() + (q + 1) * k, id) != gt.begin() + (q + 1) * k;
                        const float expected =
                            faiss::fvec_inner_product(normalized_q.data() + q * d, normalized.data() + id * d, d);
                        const float delta = std::abs(result.value()->GetDistance()[q * k + i] - expected);
                        REQUIRE(delta < 0.03f);
                        error += delta;
                    }
                std::cout << "RBQ_METRIC_DATA dataset=" << dataset << " bits=" << bits << " metric=" << metric
                          << " n=" << n << " nq=" << nq << " recall100=" << double(hits) / (nq * k)
                          << " mean_abs_error=" << error / (nq * k) << '\n';
            }
    }
}

TEST_CASE("RaBitQ same-index full distances agree across available SIMD levels", "[hnsw_rabitq_acceptance]") {
    struct RestoreLevel {
        faiss::SIMDLevel level = faiss::SIMDConfig::get_level();
        ~RestoreLevel() {
            faiss::SIMDConfig::set_level(level);
        }
    } restore;
    auto base = GenDataSet(128, 65, 1981);
    auto query = GenDataSet(1, 65, 1982);
    for (const auto* metric : {"L2", "IP", "COSINE"})
        for (int bits : {1, 4, 8, 9}) {
            knowhere::Json cfg = {{"dim", 65}, {"metric_type", metric}, {"M", 16}, {"efConstruction", 100}, {"ef", 128},
                                  {"k", 128},  {"rbq_bits", bits}};
            auto index = knowhere::IndexFactory::Instance()
                             .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                     knowhere::Version::GetCurrentVersion().VersionNumber())
                             .value();
            REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
            for (int qb : {0, 4, 8}) {
                cfg["rbq_bits_query"] = qb;
                faiss::SIMDConfig::set_level(faiss::SIMDLevel::NONE);
                auto reference = index.Search(query, cfg, nullptr);
                REQUIRE(reference.has_value());
                std::vector<float> distances(128);
                for (int i = 0; i < 128; ++i)
                    distances[reference.value()->GetIds()[i]] = reference.value()->GetDistance()[i];
                for (auto level : {faiss::SIMDLevel::NONE, faiss::SIMDLevel::AVX2, faiss::SIMDLevel::AVX512}) {
                    if (!faiss::SIMDConfig::is_simd_level_available(level))
                        continue;
                    CAPTURE(metric, bits, qb, static_cast<int>(level));
                    faiss::SIMDConfig::set_level(level);
                    auto result = index.Search(query, cfg, nullptr);
                    REQUIRE(result.has_value());
                    std::set<int64_t> ids;
                    for (int i = 0; i < 128; ++i) {
                        auto id = result.value()->GetIds()[i];
                        REQUIRE(id >= 0);
                        REQUIRE(id < 128);
                        REQUIRE(ids.insert(id).second);
                        REQUIRE(result.value()->GetDistance()[i] ==
                                Catch::Approx(distances[id]).epsilon(1e-5).margin(1e-4));
                    }
                }
            }
        }
}

TEST_CASE("RaBitQ advertised refiners rerank the requested expanded candidate set", "[hnsw_rabitq_acceptance]") {
    namespace fk = faiss::cppcontrib::knowhere;
    auto base = GenDataSet(256, 33, 1991);
    auto query = GenDataSet(1, 33, 1992);
    const auto* q = static_cast<const float*>(query->GetTensor());
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        for (const auto* refine : {"SQ4U", "SQ6", "SQ8", "FP16", "BF16", "FP32", "FLAT"}) {
            CAPTURE(metric, refine);
            knowhere::Json cfg = {
                {"dim", 33}, {"metric_type", metric}, {"M", 16},        {"efConstruction", 100}, {"ef", 128},
                {"k", 10},   {"rbq_bits", 4},         {"refine", true}, {"refine_type", refine}, {"refine_k", 1.3}};
            auto index = knowhere::IndexFactory::Instance()
                             .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                     knowhere::Version::GetCurrentVersion().VersionNumber())
                             .value();
            REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
            knowhere::BinarySet binary;
            REQUIRE(index.Serialize(binary) == knowhere::Status::success);
            auto blob = binary.binary_map_.begin()->second;
            faiss::VectorIOReader reader;
            reader.data.assign(blob->data.get(), blob->data.get() + blob->size);
            std::unique_ptr<faiss::Index> decoded(fk::read_index(&reader));
            auto* refiner = dynamic_cast<fk::IndexRefine*>(decoded.get());
            REQUIRE(refiner != nullptr);
            auto* graph = dynamic_cast<fk::IndexHNSWRaBitQ*>(refiner->base_index);
            REQUIRE(graph != nullptr);
            knowhere::IndexHNSWRaBitQWrapper wrapper(graph);
            for (int qb : {0, 4, 8}) {
                cfg["rbq_bits_query"] = qb;
                auto result = index.Search(query, cfg, nullptr);
                REQUIRE(result.has_value());
                knowhere::SearchParametersHNSWRaBitQWrapper params;
                params.efSearch = 128;
                params.storage_params.qb = qb;
                std::vector<float> d(13);
                std::vector<faiss::idx_t> ids(13);
                wrapper.search(1, q, 13, d.data(), ids.data(), &params);
                std::unique_ptr<faiss::DistanceComputer> dc(refiner->refine_index->get_distance_computer());
                dc->set_query(q);
                std::vector<std::pair<float, faiss::idx_t>> expected;
                const float sign = std::string(metric) == "L2" ? 1.f : -1.f;
                for (auto id : ids) {
                    REQUIRE(id >= 0);
                    float distance = (*dc)(id);
                    if (std::string(metric) == "COSINE") {
                        const auto* norms = dynamic_cast<const fk::HasInverseL2Norms*>(graph->storage);
                        REQUIRE(norms != nullptr);
                        distance *= norms->get_inverse_l2_norms()[id] / std::sqrt(faiss::fvec_norm_L2sqr(q, 33));
                    }
                    expected.emplace_back(sign * distance, id);
                }
                std::sort(expected.begin(), expected.end());
                for (int i = 0; i < 10; ++i) {
                    REQUIRE(result.value()->GetIds()[i] == expected[i].second);
                    REQUIRE(result.value()->GetDistance()[i] == Catch::Approx(sign * expected[i].first).margin(1e-5));
                }
            }
        }
    }
}

TEST_CASE("RaBitQ feder trace contains valid edges and retains exhaustive results", "[hnsw_rabitq_acceptance]") {
    auto base = GenDataSet(128, 33, 1951);
    auto query = GenDataSet(1, 33, 1952);
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        for (int bits : {1, 4, 9}) {
            CAPTURE(metric, bits);
            knowhere::Json cfg = {{"dim", 33}, {"metric_type", metric}, {"M", 16}, {"efConstruction", 100}, {"ef", 128},
                                  {"k", 10},   {"rbq_bits", bits}};
            auto index = knowhere::IndexFactory::Instance()
                             .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                     knowhere::Version::GetCurrentVersion().VersionNumber())
                             .value();
            REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
            for (int qb : {0, 4, 8}) {
                cfg["rbq_bits_query"] = qb;
                cfg["trace_visit"] = false;
                auto plain = index.Search(query, cfg, nullptr);
                cfg["trace_visit"] = true;
                auto traced = index.Search(query, cfg, nullptr);
                REQUIRE(plain.has_value());
                REQUIRE(traced.has_value());
                auto info =
                    knowhere::Json::parse(traced.value()->GetJsonInfo()).get<knowhere::feder::hnsw::HNSWVisitInfo>();
                size_t edges = 0;
                for (auto& level : info.GetInfos())
                    for (const auto& edge : level.GetRecords()) {
                        const auto [from, to, distance] = edge;
                        REQUIRE(from >= 0);
                        REQUIRE(from < 128);
                        REQUIRE(to >= 0);
                        REQUIRE(to < 128);
                        REQUIRE(std::isfinite(distance));
                        ++edges;
                    }
                REQUIRE(edges > 0);
                const auto ids = knowhere::Json::parse(traced.value()->GetJsonIdSet()).get<std::set<int64_t>>();
                REQUIRE_FALSE(ids.empty());
                for (auto id : ids) {
                    REQUIRE(id >= 0);
                    REQUIRE(id < 128);
                }
                for (int i = 0; i < 10; ++i) {
                    REQUIRE(plain.value()->GetIds()[i] == traced.value()->GetIds()[i]);
                    REQUIRE(plain.value()->GetDistance()[i] == traced.value()->GetDistance()[i]);
                }
            }
        }
    }
}

TEST_CASE("RaBitQ file serialization rejects truncated and incompatible indexes", "[hnsw_rabitq_acceptance]") {
    namespace fk = faiss::cppcontrib::knowhere;
    auto base = GenDataSet(128, 33, 1931);
    auto query = GenDataSet(2, 33, 1932);
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        CAPTURE(metric);
        knowhere::Json cfg = {{"dim", 33}, {"metric_type", metric}, {"M", 16}, {"efConstruction", 100}, {"ef", 100},
                              {"k", 10},   {"rbq_bits", 9}};
        auto create = [&] {
            return knowhere::IndexFactory::Instance()
                .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                        knowhere::Version::GetCurrentVersion().VersionNumber())
                .value();
        };
        auto index = create();
        REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
        knowhere::BinarySet binary;
        REQUIRE(index.Serialize(binary) == knowhere::Status::success);
        REQUIRE(binary.binary_map_.size() == 1);
        const auto [name, blob] = *binary.binary_map_.begin();
        auto rejected = [&](const std::vector<uint8_t>& bytes) {
            knowhere::BinarySet bad;
            std::shared_ptr<uint8_t[]> copy(new uint8_t[bytes.size()]);
            std::copy(bytes.begin(), bytes.end(), copy.get());
            bad.Append(name, copy, bytes.size());
            auto loaded = create();
            REQUIRE(loaded.Deserialize(bad, cfg) != knowhere::Status::success);
        };
        for (size_t length : {size_t(1), size_t(4), size_t(blob->size / 2), size_t(blob->size - 1)})
            rejected(std::vector<uint8_t>(blob->data.get(), blob->data.get() + length));
        for (const auto* fourcc : {"IHNr", "IHRK", "IHRC", "BAD!"}) {
            std::vector<uint8_t> bytes(blob->data.get(), blob->data.get() + blob->size);
            std::copy_n(fourcc, 4, bytes.begin());
            rejected(bytes);
        }
        faiss::VectorIOReader reader;
        reader.data.assign(blob->data.get(), blob->data.get() + blob->size);
        std::unique_ptr<faiss::Index> decoded(fk::read_index(&reader));
        auto* graph = dynamic_cast<fk::IndexHNSWRaBitQ*>(decoded.get());
        REQUIRE(graph != nullptr);
        auto* storage = const_cast<faiss::IndexPreTransform*>(graph->pretransform_index());
        auto* rq = const_cast<faiss::IndexRaBitQ*>(graph->rabitq_index());
        auto* rr = dynamic_cast<faiss::RandomRotationMatrix*>(storage->chain[0]);
        // Test the serialization boundary, not an outer validator duplicating
        // checks owned by RaBitQ codes and cosine norm storage.
        auto write_decoded = [&] {
            faiss::VectorIOWriter writer;
            fk::write_index(decoded.get(), &writer);
        };
        REQUIRE_NOTHROW(graph->check_storage_compatibility());
        REQUIRE_NOTHROW(write_decoded());
        ++rq->code_size;
        REQUIRE_NOTHROW(graph->check_storage_compatibility());
        REQUIRE_THROWS(write_decoded());
        --rq->code_size;
        auto last = rr->A.back();
        rr->A.pop_back();
        REQUIRE_THROWS(graph->check_storage_compatibility());
        rr->A.push_back(last);
        rr->have_bias = true;
        REQUIRE_THROWS(graph->check_storage_compatibility());
        rr->have_bias = false;
        const auto old_metric = rq->rabitq.metric_type;
        rq->rabitq.metric_type = faiss::METRIC_L1;
        REQUIRE_THROWS(write_decoded());
        rq->rabitq.metric_type = old_metric;
        const auto saved_center = rq->center;
        rq->center.clear();
        REQUIRE_THROWS(write_decoded());
        rq->center = saved_center;
        const auto code_bytes = rq->codes.size();
        const auto last_code = rq->codes[code_bytes - 1];
        rq->codes.resize(code_bytes - 1);
        REQUIRE_THROWS(write_decoded());
        rq->codes.resize(code_bytes);
        rq->codes[code_bytes - 1] = last_code;
        const auto saved_bits = rq->rabitq.nb_bits;
        rq->rabitq.nb_bits = 10;
        REQUIRE_THROWS(write_decoded());
        rq->rabitq.nb_bits = saved_bits;
        const auto saved_qb = rq->qb;
        rq->qb = 9;
        REQUIRE_THROWS(write_decoded());
        rq->qb = saved_qb;
        rq->centered = true;
        REQUIRE_NOTHROW(graph->check_storage_compatibility());
        REQUIRE_THROWS(write_decoded());
        rq->centered = false;
        ++storage->ntotal;
        REQUIRE_THROWS(graph->check_storage_compatibility());
        REQUIRE_THROWS(write_decoded());
        --storage->ntotal;
        REQUIRE_NOTHROW(write_decoded());
        if (auto* cosine = dynamic_cast<fk::IndexHNSWRaBitQCosine*>(graph)) {
            auto* cs = dynamic_cast<fk::IndexPreTransformRaBitQCosine*>(storage);
            auto norm = cs->inverse_norms_storage.inverse_l2_norms.back();
            cs->inverse_norms_storage.inverse_l2_norms.pop_back();
            REQUIRE_NOTHROW(cosine->check_cosine_storage_compatibility());
            REQUIRE_THROWS(write_decoded());
            cs->inverse_norms_storage.inverse_l2_norms.push_back(norm);
            const auto first_norm = cs->inverse_norms_storage.inverse_l2_norms[0];
            cs->inverse_norms_storage.inverse_l2_norms[0] = std::numeric_limits<float>::quiet_NaN();
            REQUIRE_THROWS(write_decoded());
            cs->inverse_norms_storage.inverse_l2_norms[0] = first_norm;
            REQUIRE_NOTHROW(write_decoded());
        }
        // A real file, independently loaded through the public Knowhere API.
        auto pattern = (std::filesystem::temp_directory_path() / "knowhere-rabitq-XXXXXX").string();
        std::vector<char> filename(pattern.begin(), pattern.end());
        filename.push_back('\0');
        const int fd = mkstemp(filename.data());
        REQUIRE(fd >= 0);
        struct RemoveFile {
            std::string path;
            ~RemoveFile() {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        } cleanup{filename.data()};
        std::unique_ptr<FILE, decltype(&std::fclose)> file(fdopen(fd, "w+b"), &std::fclose);
        if (!file)
            close(fd);
        REQUIRE(file != nullptr);
        REQUIRE(std::fwrite(blob->data.get(), 1, blob->size, file.get()) == blob->size);
        REQUIRE(std::fflush(file.get()) == 0);
        const std::string path = filename.data();
        auto loaded = create();
        REQUIRE(loaded.DeserializeFromFile(path, cfg) == knowhere::Status::success);
        for (int qb : {0, 4, 8}) {
            cfg["rbq_bits_query"] = qb;
            auto a = index.Search(query, cfg, nullptr);
            auto b = loaded.Search(query, cfg, nullptr);
            REQUIRE(a.has_value());
            REQUIRE(b.has_value());
            for (int i = 0; i < 20; ++i) {
                REQUIRE(a.value()->GetIds()[i] == b.value()->GetIds()[i]);
                REQUIRE(a.value()->GetDistance()[i] == b.value()->GetDistance()[i]);
            }
        }
        REQUIRE(index.Add(base, cfg) != knowhere::Status::success);
        REQUIRE(index.Count() == 128);
        cfg["enable_mmap"] = true;
        REQUIRE(create().DeserializeFromFile(path, cfg) != knowhere::Status::success);
    }
}

TEST_CASE("RaBitQ rejects valid non-RaBitQ payloads without replacing the live index", "[hnsw_rabitq_acceptance]") {
    auto base = GenDataSet(128, 33, 2031);
    auto query = GenDataSet(2, 33, 2032);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    knowhere::Json cfg = {{"dim", 33}, {"metric_type", "L2"}, {"M", 8},          {"efConstruction", 64}, {"ef", 80},
                          {"k", 10},   {"rbq_bits", 4},       {"sq_type", "SQ8"}};
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>("HNSW_RABITQ", version).value();
    REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
    auto before = index.Search(query, cfg, nullptr);
    REQUIRE(before.has_value());
    for (const auto* wrong_type : {"HNSW", "HNSW_SQ"}) {
        CAPTURE(wrong_type);
        auto wrong = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(wrong_type, version).value();
        REQUIRE(wrong.Build(base, cfg) == knowhere::Status::success);
        knowhere::BinarySet binary;
        REQUIRE(wrong.Serialize(binary) == knowhere::Status::success);
        const auto blob = binary.binary_map_.begin()->second;
        knowhere::BinarySet mislabelled;
        mislabelled.Append("HNSW_RABITQ", blob->data, blob->size);
        REQUIRE(index.Deserialize(mislabelled, cfg) == knowhere::Status::invalid_serialized_index_type);
        auto pattern = (std::filesystem::temp_directory_path() / "knowhere-rabitq-type-XXXXXX").string();
        std::vector<char> filename(pattern.begin(), pattern.end());
        filename.push_back('\0');
        const int fd = mkstemp(filename.data());
        REQUIRE(fd >= 0);
        struct Cleanup {
            std::string path;
            ~Cleanup() {
                std::error_code error;
                std::filesystem::remove(path, error);
            }
        } cleanup{filename.data()};
        std::unique_ptr<FILE, decltype(&std::fclose)> file(fdopen(fd, "wb"), &std::fclose);
        if (!file)
            close(fd);
        REQUIRE(file != nullptr);
        REQUIRE(std::fwrite(blob->data.get(), 1, blob->size, file.get()) == blob->size);
        REQUIRE(std::fflush(file.get()) == 0);
        REQUIRE(index.DeserializeFromFile(filename.data(), cfg) == knowhere::Status::invalid_serialized_index_type);
        auto after = index.Search(query, cfg, nullptr);
        REQUIRE(after.has_value());
        for (int i = 0; i < 20; ++i) {
            REQUIRE(after.value()->GetIds()[i] == before.value()->GetIds()[i]);
            REQUIRE(after.value()->GetDistance()[i] == before.value()->GetDistance()[i]);
        }
    }
}

TEST_CASE("Shared Faiss IVF RaBitQ bits and query parameters survive serialization", "[hnsw_rabitq_regression]") {
    auto base = GenDataSet(512, 33, 1961);
    auto query = GenDataSet(2, 33, 1962);
    auto* x = static_cast<const float*>(base->GetTensor());
    auto* q = static_cast<const float*>(query->GetTensor());
    for (auto metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        for (int bits : {1, 4, 8, 9}) {
            CAPTURE(metric, bits);
            faiss::IndexFlat quantizer(33, metric);
            faiss::IndexIVFRaBitQ index(&quantizer, 33, 4, metric, true, bits);
            index.train(512, x);
            index.add(512, x);
            faiss::VectorIOWriter writer;
            faiss::write_index(&index, &writer);
            faiss::VectorIOReader reader;
            reader.data = writer.data;
            std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));
            for (int qb : {0, 4, 8}) {
                CAPTURE(qb);
                faiss::IVFRaBitQSearchParameters params;
                params.qb = qb;
                params.nprobe = 4;
                std::vector<float> a(20), b(20);
                std::vector<faiss::idx_t> ia(20), ib(20);
                index.search(2, q, 10, a.data(), ia.data(), &params);
                loaded->search(2, q, 10, b.data(), ib.data(), &params);
                REQUIRE(ia == ib);
                for (int i = 0; i < 20; ++i) {
                    REQUIRE(ia[i] >= 0);
                    REQUIRE(ia[i] < 512);
                    REQUIRE(std::isfinite(a[i]));
                    REQUIRE(a[i] == b[i]);
                }
            }
        }
    }
}

TEST_CASE("RaBitQ invalid build parameters are explicitly rejected", "[hnsw_rabitq_acceptance]") {
    auto base = GenDataSet(64, 33, 1971);
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (auto feature : {knowhere::feature::MMAP, knowhere::feature::MV, knowhere::feature::EMB_LIST}) {
        REQUIRE_FALSE(knowhere::IndexFactory::Instance().FeatureCheck(knowhere::IndexEnum::INDEX_HNSW_RABITQ, feature));
    }
    knowhere::Json cfg = {{"dim", 33}, {"metric_type", "L2"}, {"M", 8}, {"efConstruction", 64}, {"rbq_bits", 4}};
    for (const auto& change :
         {knowhere::Json{{"rbq_bits", 0}}, knowhere::Json{{"rbq_bits", 10}}, knowhere::Json{{"metric_type", "HAMMING"}},
          knowhere::Json{{"refine", true}, {"refine_type", "unknown"}}}) {
        auto request = cfg;
        request.update(change);
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                         .value();
        REQUIRE_FALSE(index.IsAdditionalScalarSupported(false));
        REQUIRE_FALSE(index.IsAdditionalScalarSupported(true));
        REQUIRE(index.Build(base, request) != knowhere::Status::success);
    }
}

TEST_CASE("RaBitQ request state remains stable over repeated concurrent load lifecycles", "[hnsw_rabitq_acceptance]") {
    auto base = GenDataSet(256, 33, 1941);
    auto query = GenDataSet(1, 33, 1942);
    for (int cycle = 0; cycle < 4; ++cycle) {
        CAPTURE(cycle);
        knowhere::Json cfg = {
            {"dim", 33}, {"metric_type", "COSINE"},      {"M", 16}, {"efConstruction", 100}, {"ef", 128},
            {"k", 10},   {"rbq_bits", cycle % 2 ? 1 : 9}};
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                 knowhere::Version::GetCurrentVersion().VersionNumber())
                         .value();
        REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
        knowhere::BinarySet original;
        REQUIRE(index.Serialize(original) == knowhere::Status::success);
        auto loaded = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                  knowhere::Version::GetCurrentVersion().VersionNumber())
                          .value();
        REQUIRE(loaded.Deserialize(original, cfg) == knowhere::Status::success);
        std::vector<knowhere::DataSetPtr> refs;
        for (int qb : {0, 4, 8}) {
            auto request = cfg;
            request["rbq_bits_query"] = qb;
            refs.push_back(index.Search(query, request, nullptr).value());
        }
        std::vector<std::future<bool>> jobs;
        for (int task = 0; task < 12; ++task) {
            jobs.push_back(std::async(std::launch::async, [&, task] {
                const int slot = task % 3;
                const int qbs[] = {0, 4, 8};
                auto request = cfg;
                request["rbq_bits_query"] = qbs[slot];
                for (int repeat = 0; repeat < 20; ++repeat) {
                    auto result = (task % 2 ? loaded : index).Search(query, request, nullptr);
                    if (!result.has_value())
                        return false;
                    for (int i = 0; i < 10; ++i)
                        if (result.value()->GetIds()[i] != refs[slot]->GetIds()[i] ||
                            result.value()->GetDistance()[i] != refs[slot]->GetDistance()[i])
                            return false;
                }
                return true;
            }));
        }
        for (auto& job : jobs) REQUIRE(job.get());
        knowhere::BinarySet after;
        REQUIRE(index.Serialize(after) == knowhere::Status::success);
        for (const auto& [name, blob] : original.binary_map_) {
            const auto copy = after.GetByName(name);
            REQUIRE(copy->size == blob->size);
            REQUIRE(std::memcmp(copy->data.get(), blob->data.get(), blob->size) == 0);
        }
    }
}

TEST_CASE("RaBitQ filtered results and exhausted iterators match full-code references", "[hnsw_rabitq_acceptance]") {
    constexpr int n = 256, d = 33, k = 10;
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto base = GenDataSet(n, d, 1901);
    auto query = GenDataSet(1, d, 1902);
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        for (int bits : {1, 4, 9}) {
            CAPTURE(metric, bits);
            knowhere::Json cfg = {{"dim", d}, {"metric_type", metric}, {"M", 16}, {"efConstruction", 100}, {"ef", 128},
                                  {"k", k},   {"rbq_bits", bits}};
            auto index = knowhere::IndexFactory::Instance()
                             .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ, version)
                             .value();
            REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
            knowhere::BinarySet binary;
            REQUIRE(index.Serialize(binary) == knowhere::Status::success);
            REQUIRE(binary.binary_map_.size() == 1);
            const auto blob = binary.binary_map_.begin()->second;
            faiss::VectorIOReader reader;
            reader.data.assign(blob->data.get(), blob->data.get() + blob->size);
            std::unique_ptr<faiss::Index> decoded(faiss::cppcontrib::knowhere::read_index(&reader));
            auto* graph = dynamic_cast<faiss::cppcontrib::knowhere::IndexHNSWRaBitQ*>(decoded.get());
            REQUIRE(graph != nullptr);
            for (int qb : {0, 4, 8}) {
                CAPTURE(qb);
                cfg["rbq_bits_query"] = qb;
                auto all_cfg = cfg;
                all_cfg["k"] = n;
                all_cfg["ef"] = n;
                auto all = index.Search(query, all_cfg, nullptr);
                REQUIRE(all.has_value());
                std::vector<float> reference(n);
                for (int i = 0; i < n; ++i) {
                    REQUIRE(all.value()->GetIds()[i] >= 0);
                    reference[all.value()->GetIds()[i]] = all.value()->GetDistance()[i];
                }
                // An IP graph can contain unreachable vertices. Check iterator
                // completeness against its actual directed graph, not an
                // assumption that every constructed HNSW is strongly connected.
                auto score = [&](int id) { return std::string(metric) == "L2" ? reference[id] : -reference[id]; };
                int nearest = graph->hnsw.entry_point;
                for (int level = graph->hnsw.max_level; level > 0; --level) {
                    bool improved = true;
                    while (improved) {
                        improved = false;
                        size_t begin, end;
                        graph->hnsw.neighbor_range(nearest, level, &begin, &end);
                        for (size_t j = begin; j < end; ++j) {
                            int id = graph->hnsw.neighbors[j];
                            if (id < 0)
                                break;
                            if (score(id) < score(nearest)) {
                                nearest = id;
                                improved = true;
                            }
                        }
                    }
                }
                std::set<int64_t> reachable{nearest};
                std::vector<int> pending{nearest};
                for (size_t i = 0; i < pending.size(); ++i) {
                    size_t begin, end;
                    graph->hnsw.neighbor_range(pending[i], 0, &begin, &end);
                    for (size_t j = begin; j < end; ++j) {
                        int id = graph->hnsw.neighbors[j];
                        if (id < 0)
                            break;
                        if (reachable.insert(id).second)
                            pending.push_back(id);
                    }
                }
                for (int excluded : {0, 64, 200, 235, 236, 238, 239, 248, 249, 252, 256}) {
                    CAPTURE(excluded);
                    std::vector<uint8_t> mask(n / 8, 0);
                    for (int i = 0; i < excluded; ++i) mask[i / 8] |= uint8_t(1u << (i % 8));
                    knowhere::BitsetView filter(mask.data(), n);
                    auto result = index.Search(query, cfg, filter);
                    REQUIRE(result.has_value());
                    std::set<int64_t> ids;
                    std::vector<int64_t> expected;
                    for (int i = 0; i < n; ++i) {
                        if (all.value()->GetIds()[i] >= excluded)
                            expected.push_back(all.value()->GetIds()[i]);
                    }
                    int hits = 0;
                    for (int i = 0; i < k; ++i) {
                        const auto id = result.value()->GetIds()[i];
                        if (i >= n - excluded) {
                            REQUIRE(id == -1);
                            continue;
                        }
                        REQUIRE(id >= excluded);
                        REQUIRE(id < n);
                        REQUIRE(ids.insert(id).second);
                        REQUIRE(result.value()->GetDistance()[i] == Catch::Approx(reference[id]).margin(1e-4));
                        hits += std::find(expected.begin(), expected.begin() + std::min(k, n - excluded), id) !=
                                expected.begin() + std::min(k, n - excluded);
                        if (excluded >= 236)
                            REQUIRE(id == expected[i]);
                    }
                    REQUIRE(ids.size() == size_t(std::min(k, n - excluded)));
                    int reachable_topk = 0;
                    for (int i = 0; i < std::min(k, n - excluded); ++i) reachable_topk += reachable.count(expected[i]);
                    // Finite-ef filtered graph search is approximate. Record its
                    // recall instead of inventing a universal minimum for this
                    // random graph fixture. BF routing above must be exact.
                    std::cout << "RBQ_FILTER_RECALL metric=" << metric << " bits=" << bits << " qb=" << qb
                              << " excluded=" << excluded << " hits=" << hits << " reachable_topk=" << reachable_topk
                              << '\n';
                    if (excluded == 235) {
                        filter.set_filter_count(excluded);
                        // Diagnostic control: same graph and codes, but ordinary
                        // HNSW with full distances (no staged probability window).
                        namespace fk = faiss::cppcontrib::knowhere;
                        auto* rq = const_cast<faiss::IndexRaBitQ*>(graph->rabitq_index());
                        rq->qb = qb;
                        fk::IndexHNSW plain;
                        plain.d = d;
                        plain.ntotal = n;
                        plain.is_trained = true;
                        plain.metric_type = graph->metric_type;
                        plain.hnsw = graph->hnsw;
                        plain.storage = graph->storage;
                        plain.own_fields = false;
                        knowhere::IndexHNSWWrapper wrapper(&plain);
                        knowhere::BitsetViewIDSelector selector(filter);
                        knowhere::SearchParametersHNSWWrapper params;
                        params.efSearch = 128;
                        params.sel = &selector;
                        params.kAlpha = filter.filter_ratio() * 0.7f;
                        std::vector<float> full_dist(k);
                        std::vector<faiss::idx_t> full_ids(k);
                        wrapper.search(1, static_cast<const float*>(query->GetTensor()), k, full_dist.data(),
                                       full_ids.data(), &params);
                        int full_hits = 0;
                        for (auto id : full_ids)
                            full_hits += std::find(expected.begin(), expected.begin() + k, id) != expected.begin() + k;
                        std::cout << "RBQ_FILTER_CONTROL metric=" << metric << " bits=" << bits << " qb=" << qb
                                  << " staged_hits=" << hits << " full_hits=" << full_hits << '\n';
                        if (bits == 1)
                            for (int i = 0; i < k; ++i) {
                                REQUIRE(full_ids[i] == result.value()->GetIds()[i]);
                                REQUIRE(full_dist[i] == Catch::Approx(result.value()->GetDistance()[i]).margin(1e-5));
                            }
                    }

                    auto iterators = index.AnnIterator(query, cfg, filter);
                    REQUIRE(iterators.has_value());
                    std::set<int64_t> iter_ids;
                    auto it = iterators.value()[0];
                    while (it->HasNext().value()) {
                        const auto [id, distance] = it->Next().value();
                        REQUIRE(id >= excluded);
                        REQUIRE(id < n);
                        REQUIRE(iter_ids.insert(id).second);
                        REQUIRE(distance == Catch::Approx(reference[id]).margin(1e-4));
                        REQUIRE(iter_ids.size() <= size_t(n - excluded));
                    }
                    auto expected_reachable = reachable;
                    for (int i = 0; i < excluded; ++i) expected_reachable.erase(i);
                    REQUIRE(iter_ids == expected_reachable);
                    REQUIRE_FALSE(it->HasNext().value());
                }
            }
        }
    }
}

TEST_CASE("RaBitQ range boundaries and range_filter match full-code reference", "[hnsw_rabitq_acceptance]") {
    constexpr int n = 128, d = 33;
    auto base = GenDataSet(n, d, 1911);
    auto query = GenDataSet(1, d, 1912);
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        const bool l2 = std::string(metric) == "L2";
        knowhere::Json cfg = {{"dim", d}, {"metric_type", metric}, {"M", 16}, {"efConstruction", 100}, {"ef", n},
                              {"k", n},   {"rbq_bits", 4}};
        auto index = knowhere::IndexFactory::Instance()
                         .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                 knowhere::Version::GetCurrentVersion().VersionNumber())
                         .value();
        REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
        for (int qb : {0, 4, 8}) {
            cfg["rbq_bits_query"] = qb;
            auto all = index.Search(query, cfg, nullptr);
            REQUIRE(all.has_value());
            for (bool empty : {false, true}) {
                for (bool band : {false, true}) {
                    CAPTURE(metric, qb, empty, band);
                    auto request = cfg;
                    float radius = empty ? all.value()->GetDistance()[0] + (l2 ? -1.f : 1.f)
                                         : (all.value()->GetDistance()[63] + all.value()->GetDistance()[64]) / 2;
                    float lower = (all.value()->GetDistance()[15] + all.value()->GetDistance()[16]) / 2;
                    request["radius"] = radius;
                    if (band && !empty)
                        request["range_filter"] = lower;
                    auto range = index.RangeSearch(query, request, nullptr);
                    REQUIRE(range.has_value());
                    std::set<int64_t> expected, actual;
                    for (int i = 0; i < n; ++i) {
                        float distance = all.value()->GetDistance()[i];
                        if ((l2 ? distance < radius : distance > radius) &&
                            (!(band && !empty) || (l2 ? distance >= lower : distance <= lower)))
                            expected.insert(all.value()->GetIds()[i]);
                    }
                    for (size_t i = 0; i < range.value()->GetLims()[1]; ++i) {
                        REQUIRE(actual.insert(range.value()->GetIds()[i]).second);
                    }
                    REQUIRE(actual == expected);
                }
            }
        }
    }
}

TEST_CASE("RaBitQ FP16 BF16 and FP32 refine return refiner distances", "[hnsw_rabitq_acceptance]") {
    constexpr int n = 128, d = 33, k = 20;
    auto base = GenDataSet(n, d, 1921);
    auto query = GenDataSet(1, d, 1922);
    const auto* x = static_cast<const float*>(base->GetTensor());
    const auto* q = static_cast<const float*>(query->GetTensor());
    for (const auto* metric : {"L2", "IP", "COSINE"}) {
        for (const auto* refine : {"FP16", "BF16", "FP32"}) {
            CAPTURE(metric, refine);
            knowhere::Json cfg = {
                {"dim", d}, {"metric_type", metric}, {"M", 16},        {"efConstruction", 100}, {"ef", n},
                {"k", k},   {"rbq_bits", 4},         {"refine", true}, {"refine_type", refine}, {"refine_k", 1.3}};
            auto index = knowhere::IndexFactory::Instance()
                             .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW_RABITQ,
                                                     knowhere::Version::GetCurrentVersion().VersionNumber())
                             .value();
            REQUIRE(index.Build(base, cfg) == knowhere::Status::success);
            for (int qb : {0, 4, 8}) {
                cfg["rbq_bits_query"] = qb;
                auto result = index.Search(query, cfg, nullptr);
                REQUIRE(result.has_value());
                for (int i = 0; i < k; ++i) {
                    auto id = result.value()->GetIds()[i];
                    REQUIRE(id >= 0);
                    std::vector<float> decoded(d);
                    for (int j = 0; j < d; ++j) {
                        const float value = x[id * d + j];
                        decoded[j] = std::string(refine) == "FP16"   ? float(knowhere::fp16(value))
                                     : std::string(refine) == "BF16" ? faiss::decode_bf16(faiss::encode_bf16(value))
                                                                     : value;
                    }
                    float expected = std::string(metric) == "L2" ? faiss::fvec_L2sqr(q, decoded.data(), d)
                                                                 : faiss::fvec_inner_product(q, decoded.data(), d);
                    if (std::string(metric) == "COSINE")
                        expected /= std::sqrt(faiss::fvec_norm_L2sqr(q, d) * faiss::fvec_norm_L2sqr(x + id * d, d));
                    REQUIRE(result.value()->GetDistance()[i] == Catch::Approx(expected).epsilon(1e-5).margin(1e-4));
                    if (i) {
                        if (std::string(metric) == "L2")
                            REQUIRE(result.value()->GetDistance()[i - 1] <= result.value()->GetDistance()[i]);
                        else
                            REQUIRE(result.value()->GetDistance()[i - 1] >= result.value()->GetDistance()[i]);
                    }
                }
            }
        }
    }
}
