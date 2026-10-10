// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <memory>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/sparse/sindi_inverted_index.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/version.h"

namespace {
using namespace knowhere;
using namespace knowhere::sparse::inverted;
using Row = sparse::SparseRow<float>;

IndexScorerConfig
H2Scorer(bool bm25, float b = .75f) {
    return {.scorer_type = bm25 ? IndexScorerType::BM25 : IndexScorerType::IP,
            .scorer_params = {.bm25 = {.k1 = 1.2f, .b = b, .avgdl = 70.f}}};
}

std::vector<Row>
H2Rows(bool bm25) {
    std::vector<Row> rows;
    const float tf[] = {1, 255, 256, 600, 65535};
    for (size_t i = 0; i < 9 * 1024 + 7; ++i) {
        std::vector<std::pair<uint32_t, float>> terms{{10, bm25 ? tf[i % 5] : .10001f * (i % 29 + 1)}};
        if (i == 2 || i == 9 * 1024 + 3)
            terms.emplace_back(20, bm25 ? 600 : .33333f);
        if (i % 3 == 0)
            terms.emplace_back(30, bm25 ? i % 13 + 1 : .00231f);
        rows.emplace_back(terms);
    }
    return rows;
}

struct H2File {
    char path[64] = "/tmp/knowhere_h2_XXXXXX";
    explicit H2File(const BinaryPtr& data) {
        const auto fd = mkstemp(path);
        REQUIRE(fd >= 0);
        close(fd);
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data->data.get()), data->size);
        REQUIRE(f.good());
    }
    ~H2File() {
        std::remove(path);
    }
};

void
SameHits(const DataSetPtr& a, const DataSetPtr& b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        REQUIRE(a->GetIds()[i] == b->GetIds()[i]);
        if (a->GetIds()[i] >= 0)
            REQUIRE(a->GetDistance()[i] == b->GetDistance()[i]);
    }
}

template <typename Quant>
void
CheckH2Summaries(float b) {
    constexpr bool bm = !std::is_same_v<Quant, fp16>;
    using Index = SindiInvertedIndex<float, Quant>;
    auto rows = H2Rows(bm);
    Index plain(1024), h2(1024, true);
    auto config = H2Scorer(bm, b);
    for (auto* index : {&plain, &h2}) {
        index->set_build_algo("SINDI");
        index->set_build_scorer(config);
        REQUIRE(index->add(rows.data(), rows.size(), 1000000) == Status::success);
    }
    REQUIRE(plain.h2_summary_count() == 0);
    REQUIRE(h2.h2_summary_count() == 22);  // Two dense terms, one term in two of ten windows.
    REQUIRE(h2.size() - plain.size() == h2.h2_summary_bytes());
    REQUIRE(h2.h2_rebuild_seconds() >= 0);
    REQUIRE(h2.accepts_search_scorer(config));
    for (uint32_t term : {10, 20, 30, 999999}) {
        for (uint32_t w = 0; w < 10; ++w) {
            double expected = 0;
            for (size_t i = w * 1024; i < std::min(rows.size(), size_t(w + 1) * 1024); ++i) {
                double dl = 0;
                for (size_t j = 0; j < rows[i].size(); ++j) dl += rows[i][j].val;
                for (size_t j = 0; j < rows[i].size(); ++j) {
                    if (rows[i][j].id != term)
                        continue;
                    double value = rows[i][j].val;
                    if constexpr (bm) {
                        const auto& p = config.scorer_params.bm25;
                        value = value * (double(p.k1) + 1) /
                                (value + double(p.k1) * (1 - double(p.b) + double(p.b) * dl / p.avgdl));
                    } else
                        value = float(fp16(float(value)));
                    expected = std::max(expected, value);
                }
            }
            double actual = h2.h2_window_maximum(term, w);
            REQUIRE(actual >= expected);
            REQUIRE(actual <= expected * (1 + 4e-6) + 1e-30);
        }
    }
    if constexpr (bm) {
        auto changed = config;
        changed.scorer_params.bm25.avgdl += 1;
        REQUIRE_FALSE(h2.accepts_search_scorer(changed));
    }
    MemoryIOWriter on_writer, off_writer;
    REQUIRE(h2.serialize(on_writer) == Status::success);
    REQUIRE(plain.serialize(off_writer) == Status::success);
    std::unique_ptr<uint8_t[]> on_data(on_writer.data()), off_data(off_writer.data());
    REQUIRE(on_writer.tellg() == off_writer.tellg());
    REQUIRE(std::memcmp(on_data.get(), off_data.get(), on_writer.tellg()) == 0);
    Index loaded(4096, true);
    loaded.set_build_scorer(config);
    MemoryIOReader reader(off_data.get(), off_writer.tellg());
    REQUIRE(loaded.deserialize(reader) == Status::success);
    REQUIRE(loaded.h2_summary_count() == h2.h2_summary_count());
    MemoryIOWriter roundtrip;
    REQUIRE(loaded.serialize(roundtrip) == Status::success);
    std::unique_ptr<uint8_t[]> roundtrip_bytes(roundtrip.data());
    REQUIRE(roundtrip.tellg() == off_writer.tellg());
    REQUIRE(std::memcmp(roundtrip_bytes.get(), off_data.get(), roundtrip.tellg()) == 0);
    for (uint32_t term : {10, 20, 30})
        for (uint32_t w = 0; w < 10; ++w) REQUIRE(loaded.h2_window_maximum(term, w) == h2.h2_window_maximum(term, w));
    const uint32_t invalid_version = 99999;
    std::memcpy(off_data.get(), &invalid_version, sizeof(invalid_version));
    MemoryIOReader broken(off_data.get(), off_writer.tellg());
    REQUIRE(loaded.deserialize(broken) == Status::invalid_serialized_index_type);
    REQUIRE(loaded.h2_summary_count() == 0);
    REQUIRE(loaded.h2_summary_bytes() == 0);
    REQUIRE_FALSE(loaded.accepts_search_scorer(config));
    REQUIRE_THROWS(loaded.h2_window_maximum(10, 0));
}
}  // namespace

TEST_CASE("H2 compact summaries match independent represented maxima", "[sindi][h2]") {
    KnowhereConfig::SetBuildThreadPoolSize(8);
    auto quant = GENERATE(0, 1, 2);
    const auto b = GENERATE(0.f, .75f, 1.f);
    if (quant == 0)
        CheckH2Summaries<fp16>(b);
    else if (quant == 1)
        CheckH2Summaries<uint16_t>(b);
    else
        CheckH2Summaries<uint8_t>(b);
}

TEST_CASE("H2 public lifecycle and legacy payloads", "[sindi][h2]") {
    auto quant = GENERATE(std::string("fp16"), std::string("u16"), std::string("u8"), std::string("auto"));
    const int version = GENERATE(10, 11);
    if (version == 10 && (quant == "u8" || quant == "auto"))
        return;
    const bool bm = quant != "fp16";
    auto rows = H2Rows(bm);
    auto data = GenDataSet(rows.size(), 1000000, rows.data());
    data->SetIsSparse(true);
    Row query(std::vector<std::pair<uint32_t, float>>{{10, 1}, {20, .5}, {30, .7}});
    auto queries = GenDataSet(1, 1000000, &query);
    queries->SetIsSparse(true);
    Json cfg = {{"metric_type", bm ? "BM25" : "IP"},
                {"dim", 1000000},
                {"k", 10},
                {"inverted_index_algo", "SINDI"},
                {"quant_type", quant},
                {"sindi_window_size", 1024},
                {"bm25_k1", 1.2},
                {"bm25_b", .75},
                {"bm25_avgdl", 70}};
    auto create = [&]() {
        return IndexFactory::Instance().Create<sparse_u32_f32>(IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version).value();
    };
    auto off = create(), on = create();
    REQUIRE(off.Build(data, cfg) == Status::success);
    cfg["sindi_h2"] = true;
    REQUIRE(on.Build(data, cfg) == Status::success);
    REQUIRE(on.Size() > off.Size());
    auto a = off.Search(queries, cfg, nullptr), c = on.Search(queries, cfg, nullptr);
    REQUIRE(a.has_value());
    REQUIRE(c.has_value());
    SameHits(a.value(), c.value(), 10);
    BinarySet payload;
    REQUIRE(off.Serialize(payload) == Status::success);
    H2File file(payload.GetByName(off.Type()));
    for (bool mmap : {false, true}) {
        auto loaded = create();
        REQUIRE((mmap ? loaded.DeserializeFromFile(file.path, cfg) : loaded.Deserialize(payload, cfg)) ==
                Status::success);
        auto result = loaded.Search(queries, cfg, nullptr);
        REQUIRE(result.has_value());
        SameHits(c.value(), result.value(), 10);
        auto control = create();
        auto disabled = cfg;
        disabled["sindi_h2"] = false;
        REQUIRE((mmap ? control.DeserializeFromFile(file.path, disabled) : control.Deserialize(payload, disabled)) ==
                Status::success);
        REQUIRE(loaded.Size() > control.Size());
    }
    BinarySet on_payload;
    REQUIRE(on.Serialize(on_payload) == Status::success);
    cfg["sindi_h2"] = false;
    auto loaded_off = create();
    REQUIRE(loaded_off.Deserialize(on_payload, cfg) == Status::success);
    auto off_result = loaded_off.Search(queries, cfg, nullptr);
    REQUIRE(off_result.has_value());
    SameHits(c.value(), off_result.value(), 10);
    REQUIRE(loaded_off.Size() < on.Size());
    if (bm)
        for (const char* key : {"bm25_k1", "bm25_b", "bm25_avgdl"}) {
            auto changed = cfg;
            changed[key] = cfg[key].get<double>() * .5;
            auto rejected = on.Search(queries, changed, nullptr);
            REQUIRE_FALSE(rejected.has_value());
            REQUIRE(rejected.error() == Status::invalid_args);
        }
    for (float invalid : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        query.set_at(0, 999999, invalid);  // OOV must also be rejected.
        auto rejected = on.Search(queries, cfg, nullptr);
        REQUIRE_FALSE(rejected.has_value());
        REQUIRE(rejected.error() == Status::invalid_args);
    }
}

TEST_CASE("H2 rejects unsupported index configurations", "[sindi][h2]") {
    Row row(std::vector<std::pair<uint32_t, float>>{{1, 2}});
    auto data = GenDataSet(1, 2, &row);
    data->SetIsSparse(true);
    Json cfg = {{"metric_type", "IP"},
                {"dim", 2},
                {"inverted_index_algo", "SINDI"},
                {"quant_type", "fp16"},
                {"sindi_h2", true}};
    auto check = [&](Json config, int version, const std::string& type) {
        auto index = IndexFactory::Instance().Create<sparse_u32_f32>(type, version).value();
        REQUIRE(index.Build(data, config) != Status::success);
    };
    check(cfg, 9, IndexEnum::INDEX_SPARSE_INVERTED_INDEX);
    check(cfg, 11, IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC);
    cfg["inverted_index_algo"] = "DAAT_MAXSCORE";
    check(cfg, 11, IndexEnum::INDEX_SPARSE_INVERTED_INDEX);
    cfg["inverted_index_algo"] = "SINDI";
    cfg["quant_type"] = "fp32";
    check(cfg, 11, IndexEnum::INDEX_SPARSE_INVERTED_INDEX);
    cfg["metric_type"] = "BM25";
    cfg["quant_type"] = "u32";
    cfg["bm25_k1"] = 1.2;
    cfg["bm25_b"] = .75;
    cfg["bm25_avgdl"] = 2;
    check(cfg, 11, IndexEnum::INDEX_SPARSE_INVERTED_INDEX);
    cfg["quant_type"] = "u8";
    check(cfg, 10, IndexEnum::INDEX_SPARSE_INVERTED_INDEX);
    for (int window : {1023, 65536}) {
        cfg["sindi_window_size"] = window;
        check(cfg, 11, IndexEnum::INDEX_SPARSE_INVERTED_INDEX);
    }
    REQUIRE_THROWS(GrowableSindiInvertedIndexIP(1024, true));
}

TEST_CASE("H2 invalid values leave no derived state", "[sindi][h2]") {
    for (float value :
         {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 70000.f}) {
        Row row(std::vector<std::pair<uint32_t, float>>{{1, value}});
        SindiInvertedIndexIP ip(1024, true);
        ip.set_build_scorer(H2Scorer(false));
        REQUIRE(ip.add(&row, 1, 2) == Status::invalid_args);
        REQUIRE(ip.h2_summary_count() == 0);
        SindiInvertedIndexBM25 bm(1024, true);
        bm.set_build_scorer(H2Scorer(true));
        REQUIRE(bm.add(&row, 1, 2) == Status::invalid_args);
        REQUIRE(bm.h2_summary_count() == 0);
    }
    Row fractional(std::vector<std::pair<uint32_t, float>>{{1, 1.5f}});
    SindiInvertedIndexBM25 bm(1024, true);
    bm.set_build_scorer(H2Scorer(true));
    REQUIRE(bm.add(&fractional, 1, 2) == Status::invalid_args);
    auto invalid = H2Scorer(true);
    invalid.scorer_params.bm25.avgdl = 0;
    bm.set_build_scorer(invalid);
    Row valid(std::vector<std::pair<uint32_t, float>>{{1, 1}});
    REQUIRE(bm.add(&valid, 1, 2) == Status::invalid_args);
}

TEST_CASE("H2 summary rebuild clears previous empty-payload state", "[sindi][h2]") {
    using Index = SindiInvertedIndexBM25;
    Index empty(1024), reused(1024, true);
    auto scorer = H2Scorer(true);
    empty.set_build_scorer(scorer);
    reused.set_build_scorer(scorer);
    empty.set_build_algo("SINDI");
    reused.set_build_algo("SINDI");
    Row row(std::vector<std::pair<uint32_t, float>>{{10, 3}});
    REQUIRE(empty.add(&row, 0, 31) == Status::success);
    REQUIRE(reused.add(&row, 1, 31) == Status::success);
    REQUIRE(reused.h2_summary_count() == 1);
    MemoryIOWriter writer;
    REQUIRE(empty.serialize(writer) == Status::success);
    std::unique_ptr<uint8_t[]> data(writer.data());
    MemoryIOReader reader(data.get(), writer.tellg());
    REQUIRE(reused.deserialize(reader) == Status::success);
    REQUIRE(reused.accepts_search_scorer(scorer));
    REQUIRE(reused.h2_summary_count() == 0);
    REQUIRE(reused.h2_window_maximum(10, 0) == 0);
}

TEST_CASE("H2 summaries cover native unit-weight accumulation", "[sindi][h2]") {
    // Independent bound check using the dispatched kernel, including the
    // clamped-u8 contribution followed by the target's overflow correction.
    const bool u8 = GENERATE(false, true);
    const float b = GENERATE(0.f, .75f, 1.f);
    const auto scorer = H2Scorer(true, b);
    const float p1 = scorer.scorer_params.bm25.k1 + 1;
    const float p2 = scorer.scorer_params.bm25.k1 * (1 - b);
    const float p3 = scorer.scorer_params.bm25.k1 * b / scorer.scorer_params.bm25.avgdl;
    std::vector<Row> rows;
    std::vector<uint16_t> ids, values;
    std::vector<uint8_t> values8;
    std::vector<float> lengths, scores(65, 0);
    const uint16_t tf[] = {1, 255, 256, 600, 65535};
    for (uint16_t i = 0; i < 65; ++i) {
        uint16_t value = tf[i % 5];
        rows.emplace_back(std::vector<std::pair<uint32_t, float>>{{10, float(value)}, {20, float(i * 11)}});
        ids.push_back(i);
        values.push_back(value);
        values8.push_back(std::min<uint16_t>(value, 255));
        lengths.push_back(value + i * 11.f);
    }
    float bound;
    if (u8) {
        SindiInvertedIndexBM25U8 index(1024, true);
        index.set_build_scorer(scorer);
        REQUIRE(index.add(rows.data(), rows.size(), 21) == Status::success);
        bound = index.h2_window_maximum(10, 0);
        sindi::get_bm25_u8_kernels().accumulate(1, values8.data(), ids.data(), ids.size(), scores.data(),
                                                scorer.scorer_params.bm25.k1, b, 70, lengths.data());
        for (size_t i = 0; i < ids.size(); ++i)
            if (values[i] > 255) {
                float norm = p2 + p3 * lengths[i];
                scores[i] += p1 * (values[i] / (values[i] + norm) - 255.f / (255.f + norm));
            }
    } else {
        SindiInvertedIndexBM25 index(1024, true);
        index.set_build_scorer(scorer);
        REQUIRE(index.add(rows.data(), rows.size(), 21) == Status::success);
        bound = index.h2_window_maximum(10, 0);
        sindi::get_bm25_kernels().accumulate(1, values.data(), ids.data(), ids.size(), scores.data(),
                                             scorer.scorer_params.bm25.k1, b, 70, lengths.data());
    }
    for (float score : scores) {
        REQUIRE(std::isfinite(score));
        REQUIRE(score <= bound);
    }
}

namespace {
template <typename Quant>
void
CheckH2Traversal(bool overflow, float ratio, size_t k, int filter_mode) {
    constexpr bool bm = !std::is_same_v<Quant, fp16>;
    using Index = SindiInvertedIndex<float, Quant>;
    constexpr size_t n = 12 * 1024 + 13;
    std::vector<Row> rows;
    for (size_t i = 0; i < n; ++i) {
        const size_t w = i / 1024;
        const bool strong = w == 0 || w == 12;
        float a = bm ? (strong ? (w == 12 ? (overflow ? 600 : 150) : 100) + i % 11 : 1)
                     : (strong ? (w == 12 ? 5.f : 3.f) + float(i % 11) / 32 : .001f);
        float c = bm ? (strong ? (w == 12 ? (overflow ? 600 : 150) : 80) + i % 7 : 1)
                     : (strong ? 2.f + float(i % 7) / 32 : .001f);
        std::vector<std::pair<uint32_t, float>> terms{{10, a}};
        if ((w == 0 || w == 1 || w == 4 || w == 8 || w == 12) && (i % 31 == 0 || w == 12))
            terms.emplace_back(20, bm ? (overflow ? 600.f : 230.f) : (strong ? 4.f : .002f));
        terms.emplace_back(30, c);
        if (bm && !strong)
            terms.emplace_back(40, 60000);  // Original length weakens otherwise overflowing TFs.
        rows.emplace_back(terms);
    }
    Row query(std::vector<std::pair<uint32_t, float>>{{10, 1.3f}, {20, .8f}, {30, .7f}, {999, .5f}});
    auto scorer = H2Scorer(bm);
    Index off(1024), on(1024, true);
    for (auto* index : {&off, &on}) {
        index->set_build_algo("SINDI");
        index->set_build_scorer(scorer);
        REQUIRE(index->add(rows.data(), rows.size(), 1000) == Status::success);
    }
    std::vector<uint8_t> bits((n + 7) / 8, 0);
    for (size_t i = 0; i < n; ++i) {
        if (filter_mode == 2 || (filter_mode == 1 && (i / 1024 == 1 || i / 1024 == 3 || i % 17 == 0)))
            bits[i / 8] |= uint8_t(1u << (i % 8));
    }
    const BitsetView filter = filter_mode ? BitsetView(bits.data(), n) : BitsetView{};
    InvertedIndexSearchParams params{};
    params.algo = InvertedIndexAlgo::SINDI;
    params.scorer_config = scorer;
    params.approx.dim_max_score_ratio = ratio;
    std::vector<float> a(k), b(k);
    std::vector<sparse::label_t> ia(k), ib(k);
    off.search(query, k, a.data(), ia.data(), filter, params);
    on.search(query, k, b.data(), ib.data(), filter, params);
    REQUIRE(ia == ib);
    std::vector<double> oracle(n, 0), sorted, first_window;
    for (size_t i = 0; i < n; ++i) {
        if (bits[i / 8] & (1u << (i % 8)))
            continue;
        double dl = 0;
        for (size_t j = 0; j < rows[i].size(); ++j) dl += rows[i][j].val;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            auto posting = rows[i][j];
            double weight = posting.id == 10   ? double(query[0].val)
                            : posting.id == 20 ? double(query[1].val)
                            : posting.id == 30 ? double(query[2].val)
                                               : 0;
            double value = bm ? posting.val : float(fp16(posting.val));
            if constexpr (bm) {
                const auto& p = scorer.scorer_params.bm25;
                value = value * (double(p.k1) + 1) /
                        (value + double(p.k1) * (1 - double(p.b) + double(p.b) * dl / p.avgdl));
            }
            oracle[i] += weight * value;
        }
        if (oracle[i] > 0) {
            sorted.push_back(oracle[i]);
            if (i < 1024)
                first_window.push_back(oracle[i]);
        }
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    std::sort(first_window.begin(), first_window.end(), std::greater<double>());
    if (filter_mode != 2) {
        // The first window fills the heap. Window four contains sparse terms
        // (and u8 overflow postings), but its entire local bound is below the
        // heap threshold: early termination must still advance all cursors.
        double local = 0;
        for (size_t j = 0; j < query.size(); ++j) local += double(query[j].val) * on.h2_window_maximum(query[j].id, 4);
        REQUIRE(first_window.size() >= k);
        REQUIRE(ratio * local < first_window[k - 1] - 1e-4);
    }
    for (size_t j = 0; j < k; ++j) {
        if (j >= sorted.size()) {
            REQUIRE(ib[j] == -1);
            continue;
        }
        REQUIRE(ib[j] >= 0);
        REQUIRE(a[j] == b[j]);
        REQUIRE(std::abs(double(b[j]) - oracle[ib[j]]) < 1e-4 + 1e-5 * oracle[ib[j]]);
        REQUIRE(std::abs(double(b[j]) - sorted[j]) < 1e-4 + 1e-5 * sorted[j]);
    }
    // Repeat and empty/OOV queries must not retain a previous query's suffix.
    on.search(query, k, b.data(), ib.data(), filter, params);
    REQUIRE(ia == ib);
    Row missing(std::vector<std::pair<uint32_t, float>>{{999, 1}});
    on.search(missing, k, b.data(), ib.data(), filter, params);
    for (auto id : ib) REQUIRE(id == -1);
    Row empty;
    on.search(empty, k, b.data(), ib.data(), filter, params);
    for (auto id : ib) REQUIRE(id == -1);
}
}  // namespace

TEST_CASE("H2 local suffix traversal survives filtered and pruned sparse windows", "[sindi][h2]") {
    auto variant = GENERATE(0, 1, 2, 3);
    auto ratio = GENERATE(1.f, 1.05f);
    auto k = GENERATE(size_t(7), size_t(30));
    auto filter = GENERATE(0, 1, 2);
    if (variant == 0)
        CheckH2Traversal<fp16>(false, ratio, k, filter);
    else if (variant == 1)
        CheckH2Traversal<uint16_t>(true, ratio, k, filter);
    else
        CheckH2Traversal<uint8_t>(variant == 3, ratio, k, filter);
}

namespace {
struct H2Hits {
    std::vector<sparse::label_t> ids;
    std::vector<float> scores;
};

template <typename Index>
H2Hits
H2Search(const Index& index, const Row& query, size_t k, const InvertedIndexSearchParams& params,
         const BitsetView& filter = {}) {
    H2Hits result{std::vector<sparse::label_t>(k), std::vector<float>(k)};
    index.search(query, k, result.scores.data(), result.ids.data(), filter, params);
    return result;
}

void
H2Same(const H2Hits& a, const H2Hits& b) {
    REQUIRE(a.ids == b.ids);
    for (size_t j = 0; j < a.ids.size(); ++j) {
        if (a.ids[j] < 0) {
            REQUIRE(std::isnan(a.scores[j]));
            REQUIRE(std::isnan(b.scores[j]));
        } else
            REQUIRE(a.scores[j] == b.scores[j]);
    }
}

// Independent exhaustive represented-value oracle. Equal-score IDs may be
// selected in any order, but off/on must preserve the existing heap's choice.
template <typename Quant>
void
H2Oracle(const std::vector<Row>& rows, const Row& query, const IndexScorerConfig& scorer, const H2Hits& hits,
         const BitsetView& filter) {
    std::vector<double> scores(rows.size(), 0), sorted;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!filter.empty() && filter.test(i))
            continue;
        double dl = 0;
        for (size_t j = 0; j < rows[i].size(); ++j) dl += rows[i][j].val;
        for (size_t j = 0; j < rows[i].size(); ++j) {
            auto term = rows[i][j];
            // Sealed SINDI omits original values below float epsilon before
            // FP16 conversion; reproduce that storage contract independently.
            if (std::abs(term.val) < std::numeric_limits<float>::epsilon())
                continue;
            for (size_t q = 0; q < query.size(); ++q) {
                if (term.id != query[q].id)
                    continue;
                double value;
                if constexpr (std::is_same_v<Quant, fp16>)
                    value = float(fp16(term.val));
                else {
                    const auto& p = scorer.scorer_params.bm25;
                    value = double(term.val) * (double(p.k1) + 1) /
                            (double(term.val) + double(p.k1) * (1 - double(p.b) + double(p.b) * dl / p.avgdl));
                }
                scores[i] += double(query[q].val) * value;
            }
        }
        if (scores[i] > 0)
            sorted.push_back(scores[i]);
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    std::unordered_set<sparse::label_t> seen;
    for (size_t j = 0; j < hits.ids.size(); ++j) {
        if (j >= sorted.size()) {
            REQUIRE(hits.ids[j] == -1);
            REQUIRE(std::isnan(hits.scores[j]));
            continue;
        }
        REQUIRE(hits.ids[j] >= 0);
        REQUIRE(size_t(hits.ids[j]) < rows.size());
        REQUIRE(seen.insert(hits.ids[j]).second);
        REQUIRE((filter.empty() || !filter.test(hits.ids[j])));
        // Relative tolerance retains sensitivity for tiny FP16 subnormal scores.
        const double tolerance = 2e-5 * sorted[j] + 1e-30;
        REQUIRE(std::abs(double(hits.scores[j]) - sorted[j]) <= tolerance);
        REQUIRE(std::abs(double(hits.scores[j]) - scores[hits.ids[j]]) <= tolerance);
    }
}

template <typename Quant>
void
H2EdgeFixture(const std::vector<Row>& rows, const std::vector<Row>& queries, uint32_t window,
              const IndexScorerConfig& scorer, size_t k, bool concurrent = false) {
    using Index = SindiInvertedIndex<float, Quant>;
    Index off(window), on(window, true);
    for (auto* index : {&off, &on}) {
        index->set_build_algo("SINDI");
        index->set_build_scorer(scorer);
        REQUIRE(index->add(rows.data(), rows.size(), 1000) == Status::success);
    }
    MemoryIOWriter writer;
    REQUIRE(off.serialize(writer) == Status::success);
    auto payload = std::make_shared<Binary>();
    payload->data.reset(writer.data());
    payload->size = writer.tellg();
    H2File file(payload);
    const int fd = open(file.path, O_RDONLY);
    REQUIRE(fd >= 0);
    void* address = mmap(nullptr, payload->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    REQUIRE(address != MAP_FAILED);
    auto unmap = [size = payload->size](uint8_t* p) { munmap(p, size); };
    std::unique_ptr<uint8_t, decltype(unmap)> mapping(static_cast<uint8_t*>(address), unmap);
    MemoryIOReader reader(mapping.get(), payload->size);
    Index loaded(4096, true);
    loaded.set_build_scorer(scorer);
    REQUIRE(loaded.deserialize(reader) == Status::success);
    std::vector<uint8_t> bits((rows.size() + 7) / 8, 0);
    for (size_t i = 0; i < rows.size(); i += 3) bits[i / 8] |= uint8_t(1u << (i % 8));
    for (float ratio : {1.f, 1.05f}) {
        InvertedIndexSearchParams params{};
        params.algo = InvertedIndexAlgo::SINDI;
        params.scorer_config = scorer;
        params.approx.dim_max_score_ratio = ratio;
        for (bool filtered : {false, true}) {
            const BitsetView filter = filtered ? BitsetView(bits.data(), rows.size()) : BitsetView{};
            std::vector<H2Hits> expected;
            for (const auto& query : queries) {
                auto a = H2Search(off, query, k, params, filter);
                auto b = H2Search(on, query, k, params, filter);
                H2Same(a, b);
                H2Oracle<Quant>(rows, query, scorer, b, filter);
                H2Same(b, H2Search(loaded, query, k, params, filter));
                expected.push_back(std::move(b));
            }
            if (concurrent) {
                // No Catch assertions in workers: collect outputs and assert on
                // the test thread after all simultaneous readers finish.
                std::vector<std::future<std::vector<H2Hits>>> workers;
                for (size_t worker = 0; worker < 8; ++worker)
                    workers.push_back(std::async(std::launch::async, [&]() {
                        std::vector<H2Hits> results;
                        for (size_t repeat = 0; repeat < 4; ++repeat)
                            for (const auto& query : queries)
                                results.push_back(H2Search(loaded, query, k, params, filter));
                        return results;
                    }));
                for (auto& worker : workers) {
                    auto results = worker.get();
                    for (size_t i = 0; i < results.size(); ++i) H2Same(expected[i % queries.size()], results[i]);
                }
            }
        }
    }
}
}  // namespace

TEST_CASE("H2 IP rounding ties tiny values and result padding", "[sindi][h2]") {
    const float values[] = {0, 1e-9f, 0x1p-24f, 0x1p-23f, 0x1p-14f, .10001f, .10002f, 1.f, 1.f, 65504.f};
    std::vector<Row> rows;
    for (float v : values) rows.emplace_back(std::vector<std::pair<uint32_t, float>>{{10, v}, {20, v / 2}});
    rows.emplace_back();
    std::vector<Row> queries;
    queries.emplace_back(std::vector<std::pair<uint32_t, float>>{{10, .5f}, {20, .25f}});
    queries.emplace_back(std::vector<std::pair<uint32_t, float>>{{10, 0}});
    queries.emplace_back(std::vector<std::pair<uint32_t, float>>{{999, 1}});
    queries.emplace_back();
    H2EdgeFixture<fp16>(rows, queries, 1024, H2Scorer(false), 20, true);
}

TEST_CASE("H2 full window counts and last document boundaries", "[sindi][h2]") {
    const auto window = GENERATE(1024u, 4096u, 65535u);
    const auto quant = GENERATE(0, 1, 2);
    CAPTURE(window, quant);
    std::vector<Row> rows;
    for (size_t i = 0; i < window + 17; ++i) {
        std::vector<std::pair<uint32_t, float>> terms{{10, float(i % 13 + 1)}};
        if (i == 0 || i == window - 1 || i == window || i == window + 16)
            terms.emplace_back(20, 600);
        rows.emplace_back(terms);
    }
    std::vector<Row> queries;
    queries.emplace_back(std::vector<std::pair<uint32_t, float>>{{10, .7f}, {20, 1.1f}});
    queries.emplace_back(std::vector<std::pair<uint32_t, float>>{{20, 1}});
    if (quant == 0)
        H2EdgeFixture<fp16>(rows, queries, window, H2Scorer(false), 23);
    else if (quant == 1)
        H2EdgeFixture<uint16_t>(rows, queries, window, H2Scorer(true), 23);
    else
        H2EdgeFixture<uint8_t>(rows, queries, window, H2Scorer(true), 23);
}

TEST_CASE("H2 BM25 normalization extremes and concurrent loaded searches", "[sindi][h2]") {
    const auto quant = GENERATE(0, 1);
    const auto b = GENERATE(0.f, .75f, 1.f);
    const auto avgdl = GENERATE(1.f, 1e6f);
    const auto k1 = GENERATE(.01f, 1.2f, 100.f);
    CAPTURE(quant, b, avgdl, k1);
    auto scorer = H2Scorer(true, b);
    scorer.scorer_params.bm25.avgdl = avgdl;
    scorer.scorer_params.bm25.k1 = k1;
    const float tf[] = {1, 255, 256, 600, 65535};
    std::vector<Row> rows;
    for (size_t i = 0; i < 2049; ++i)
        rows.emplace_back(std::vector<std::pair<uint32_t, float>>{
            {10, tf[i % 5]}, {20, tf[(i / 5) % 5]}, {30, i % 2 ? 65535.f : 1.f}});
    std::vector<Row> queries;
    queries.emplace_back(std::vector<std::pair<uint32_t, float>>{{10, .31f}, {20, 1.7f}});
    queries.emplace_back(std::vector<std::pair<uint32_t, float>>{{10, .00001f}, {30, .2f}});
    queries.emplace_back();
    const bool concurrent = b == .75f && avgdl == 1 && k1 == 1.2f;
    if (quant == 0)
        H2EdgeFixture<uint16_t>(rows, queries, 1024, scorer, 10, concurrent);
    else
        H2EdgeFixture<uint8_t>(rows, queries, 1024, scorer, 10, concurrent);
}

TEST_CASE("H2 validation dispatch selection", "[sindi][h2]") {
    const auto ip = sindi::get_ip_kernels().accumulate;
    const auto bm = sindi::get_bm25_kernels().accumulate;
    const auto u8 = sindi::get_bm25_u8_kernels().accumulate;
    REQUIRE(ip != nullptr);
    REQUIRE(bm != nullptr);
    REQUIRE(u8 != nullptr);
    if (const char* mode = std::getenv("KNOWHERE_H2_TEST_DISPATCH")) {
        const bool scalar = std::string(mode) == "scalar";
        REQUIRE((ip == sindi::ip_accumulate_scalar_fp16) == scalar);
        REQUIRE((bm == sindi::bm25_accumulate_scalar_u16) == scalar);
        REQUIRE((u8 == sindi::bm25_accumulate_scalar_u8) == scalar);
    }
}

TEST_CASE("H2 legacy packed posting arrays load at every natural alignment", "[sindi][h2]") {
    const auto dims = GENERATE(1u, 16u, 32u);
    // One window: mask bytes plus six bytes per term place uint32 offsets at
    // odd, two-byte and four-byte aligned positions, respectively.
    std::vector<Row> rows;
    std::vector<std::pair<uint32_t, float>> terms, q;
    for (uint32_t d = 0; d < dims; ++d) {
        terms.emplace_back(d, d + 1);
        q.emplace_back(d, .31f);
    }
    for (size_t i = 0; i < 3; ++i) rows.emplace_back(terms);
    std::vector<Row> queries;
    queries.emplace_back(q);
    H2EdgeFixture<fp16>(rows, queries, 1024, H2Scorer(false), 4);
    H2EdgeFixture<uint16_t>(rows, queries, 1024, H2Scorer(true), 4);
    H2EdgeFixture<uint8_t>(rows, queries, 1024, H2Scorer(true), 4);
}
