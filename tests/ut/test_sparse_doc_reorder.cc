// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/sparse/doc_reorder.h"
#include "index/sparse/inverted_index_format.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {

using knowhere::sparse::inverted::InvertedIndexSectionType;

constexpr int32_t kVersion = 11;

bool
HasSection(const knowhere::BinaryPtr& binary, InvertedIndexSectionType type) {
    knowhere::MemoryIOReader reader(binary->data.get(), binary->size);
    reader.advance(knowhere::sparse::inverted::kInvertedIndexFileHeaderSize);
    uint32_t nr_sections = 0;
    reader.read(&nr_sections, sizeof(uint32_t));
    const auto headers = knowhere::sparse::inverted::read_section_headers(reader, nr_sections);
    return knowhere::sparse::inverted::find_section_header(headers, type) != nullptr;
}

// Four topics with disjoint vocabularies, documents interleaved by topic. Each document
// also carries two rare noise terms that fall below the feature document-frequency cut.
knowhere::DataSetPtr
GenTopicDataSet(int32_t docs_per_topic, std::vector<int>& topic_of) {
    constexpr int kTopics = 4, kTopicTerms = 64, kTermsPerDoc = 12, kNoiseTerms = 100;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> topic_term(0, kTopicTerms - 1), noise_term(0, kNoiseTerms - 1), tf(1, 5);
    const int32_t rows = kTopics * docs_per_topic;
    std::vector<std::map<int32_t, float>> data(rows);
    topic_of.resize(rows);
    for (int32_t i = 0; i < rows; ++i) {
        const int topic = i % kTopics;
        topic_of[i] = topic;
        while (data[i].size() < kTermsPerDoc) {
            data[i][kNoiseTerms + topic * kTopicTerms + topic_term(rng)] = tf(rng);
        }
        for (int j = 0; j < 2; ++j) {
            data[i][noise_term(rng)] = tf(rng);
        }
    }
    return GenSparseDataSet(data, kNoiseTerms + kTopics * kTopicTerms);
}

const knowhere::sparse::SparseRow<float>*
Rows(const knowhere::DataSetPtr& ds) {
    return static_cast<const knowhere::sparse::SparseRow<float>*>(ds->GetTensor());
}

uint32_t
FormatVersion(const knowhere::BinaryPtr& binary) {
    uint32_t version = 0;
    std::memcpy(&version, binary->data.get(), sizeof(version));
    return version;
}

// Every nonzero score per query from an index's iterator, keyed by row id.
std::vector<std::map<int64_t, float>>
AllScores(const knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSetPtr& queries,
          const knowhere::Json& search, const knowhere::BitsetView& bitset) {
    auto iterators = index.AnnIterator(queries, search, bitset);
    REQUIRE(iterators.has_value());
    std::vector<std::map<int64_t, float>> scores(iterators.value().size());
    for (size_t q = 0; q < scores.size(); ++q) {
        auto& it = iterators.value()[q];
        while (it->HasNext().value()) {
            auto [id, dist] = it->Next().value();
            scores[q][id] = dist;
        }
    }
    return scores;
}

bool
Close(float a, float b) {
    return std::abs(a - b) <= 1e-4f * std::max(1.0f, std::abs(a));
}

// Equal scores may come back as different row ids, so the reordered result must have the
// baseline's score at every rank, and each returned row id must have that score in the
// baseline index (oracle). Row ids must be valid, unique and not filtered.
void
RequireSameResults(const knowhere::DataSet& expected, const knowhere::DataSet& actual,
                   const std::vector<std::map<int64_t, float>>& oracle, const knowhere::BitsetView& bitset,
                   int64_t rows) {
    REQUIRE(expected.GetRows() == actual.GetRows());
    REQUIRE(expected.GetDim() == actual.GetDim());
    const auto nq = expected.GetRows();
    const auto k = expected.GetDim();
    size_t returned = 0;
    for (int64_t q = 0; q < nq; ++q) {
        std::set<int64_t> seen;
        for (int64_t j = 0; j < k; ++j) {
            const auto e_id = expected.GetIds()[q * k + j];
            const auto a_id = actual.GetIds()[q * k + j];
            REQUIRE((e_id < 0) == (a_id < 0));
            if (a_id < 0) {
                continue;
            }
            ++returned;
            const auto a_dist = actual.GetDistance()[q * k + j];
            CAPTURE(q, j, e_id, a_id, a_dist);
            REQUIRE(Close(expected.GetDistance()[q * k + j], a_dist));
            REQUIRE(a_id < rows);
            REQUIRE(seen.insert(a_id).second);
            if (!bitset.empty()) {
                REQUIRE(!bitset.test(a_id));
            }
            const auto it = oracle[q].find(a_id);
            REQUIRE(it != oracle[q].end());
            REQUIRE(Close(it->second, a_dist));
        }
    }
    REQUIRE(returned > 0);
}

}  // namespace

TEST_CASE("BP_kmeans document order", "[sparse][sindi][reorder]") {
    using knowhere::sparse::inverted::reorder::bp_kmeans_order;
    using knowhere::sparse::inverted::reorder::BpKmeansOptions;
    using knowhere::sparse::inverted::reorder::BpKmeansStats;

    SECTION("is a deterministic permutation") {
        auto ds = GenSparseDataSet(3000, 400, 0.95);
        BpKmeansStats stats;
        const auto order = bp_kmeans_order(Rows(ds), 3000, BpKmeansOptions{}, &stats);
        REQUIRE(order.size() == 3000);
        REQUIRE(stats.features > 0);
        REQUIRE(stats.swaps > 0);
        auto sorted = order;
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t i = 0; i < sorted.size(); ++i) {
            REQUIRE(sorted[i] == i);
        }
        bool identity = true;
        for (uint32_t i = 0; i < order.size(); ++i) {
            identity &= order[i] == i;
        }
        REQUIRE_FALSE(identity);
        REQUIRE(bp_kmeans_order(Rows(ds), 3000, BpKmeansOptions{}) == order);
    }

    SECTION("groups documents of the same topic") {
        std::vector<int> topic_of;
        constexpr int32_t kDocsPerTopic = 512;
        auto ds = GenTopicDataSet(kDocsPerTopic, topic_of);
        const auto n = topic_of.size();
        const auto order = bp_kmeans_order(Rows(ds), n, BpKmeansOptions{});
        // Before reordering every block of 512 documents mixes all four topics evenly.
        for (size_t block = 0; block < 4; ++block) {
            std::array<int, 4> counts{};
            for (size_t i = block * kDocsPerTopic; i < (block + 1) * kDocsPerTopic; ++i) {
                ++counts[topic_of[order[i]]];
            }
            const int majority = *std::max_element(counts.begin(), counts.end());
            CAPTURE(block, counts[0], counts[1], counts[2], counts[3]);
            REQUIRE(majority >= 0.9 * kDocsPerTopic);
        }
    }

    SECTION("keeps the input order when nothing qualifies") {
        auto ds = GenSparseDataSet(10, 50, 0.5);
        const auto tiny = bp_kmeans_order(Rows(ds), 10, BpKmeansOptions{});
        for (uint32_t i = 0; i < tiny.size(); ++i) {
            REQUIRE(tiny[i] == i);
        }
        BpKmeansOptions no_features;
        no_features.min_df = 1000000;
        auto big = GenSparseDataSet(2000, 300, 0.95);
        const auto order = bp_kmeans_order(Rows(big), 2000, no_features);
        for (uint32_t i = 0; i < order.size(); ++i) {
            REQUIRE(order[i] == i);
        }
    }
}

TEST_CASE("SINDI doc reorder keeps results and row ids", "[sparse][sindi][reorder]") {
    const int32_t nb = 5000, dim = 400, nq = 20;
    auto [metric, quant] = GENERATE(table<std::string, std::string>({
        {knowhere::metric::IP, ""},
        {knowhere::metric::BM25, ""},
        {knowhere::metric::BM25, "u8"},
        {knowhere::metric::BM25, "u4_lut"},
    }));
    const bool bm25 = metric == knowhere::metric::BM25;
    auto train_ds = bm25 ? GenSparseDataSetWithMaxVal(nb, dim, 0.95, 20, true) : GenSparseDataSet(nb, dim, 0.95);
    auto query_ds =
        bm25 ? GenSparseDataSetWithMaxVal(nq, dim, 0.97, 3, false, 99) : GenSparseDataSet(nq, dim, 0.97, 99);
    const auto topk = GENERATE(10, 100);

    knowhere::Json build{{knowhere::meta::DIM, dim},
                         {knowhere::meta::METRIC_TYPE, metric},
                         {knowhere::indexparam::INVERTED_INDEX_ALGO, "SINDI"},
                         {knowhere::meta::BM25_K1, 1.2},
                         {knowhere::meta::BM25_B, 0.75},
                         {knowhere::meta::BM25_AVGDL, 100},
                         {"sindi_window_size", 1024}};
    if (!quant.empty()) {
        build["quant_type"] = quant;
    }
    knowhere::Json search{{knowhere::meta::TOPK, topk},      {knowhere::meta::METRIC_TYPE, metric},
                          {knowhere::meta::BM25_K1, 1.2},    {knowhere::meta::BM25_B, 0.75},
                          {knowhere::meta::BM25_AVGDL, 100}, {"drop_ratio_search", 0}};
    auto reordered_build = build;
    reordered_build["sindi_doc_reorder"] = "bp_kmeans";
    CAPTURE(metric, quant, topk);

    const auto type = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
    auto base = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
    REQUIRE(base.Build(train_ds, build) == knowhere::Status::success);
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
    REQUIRE(idx.Build(train_ds, reordered_build) == knowhere::Status::success);
    REQUIRE(idx.Count() == nb);

    knowhere::BinarySet base_bs, bs;
    REQUIRE(base.Serialize(base_bs) == knowhere::Status::success);
    REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
    REQUIRE_FALSE(HasSection(base_bs.GetByName(type), InvertedIndexSectionType::SINDI_DOC_ORDER));
    REQUIRE(HasSection(bs.GetByName(type), InvertedIndexSectionType::SINDI_DOC_ORDER));
    REQUIRE(FormatVersion(base_bs.GetByName(type)) == knowhere::sparse::inverted::kInvertedIndexFileFormatVersion);
    REQUIRE(FormatVersion(bs.GetByName(type)) == knowhere::sparse::inverted::kInvertedIndexFileFormatVersionDocOrder);

    const knowhere::BitsetView no_filter;
    auto expected = base.Search(query_ds, search, nullptr);
    REQUIRE(expected.has_value());
    const auto oracle = AllScores(base, query_ds, search, no_filter);

    SECTION("search") {
        auto actual = idx.Search(query_ds, search, nullptr);
        REQUIRE(actual.has_value());
        RequireSameResults(*expected.value(), *actual.value(), oracle, no_filter, nb);
    }

    SECTION("search with bitset") {
        const auto filtered = GENERATE(0.1f, 0.5f, 0.9f);
        // A threshold of 1 always takes the posting prefilter, packed U4 included.
        const auto prefilter = GENERATE(false, true);
        CAPTURE(filtered, prefilter);
        auto filtered_search = search;
        if (prefilter) {
            filtered_search["sindi_bm25_filter_threshold"] = 1.0;
            filtered_search["sindi_ip_filter_threshold"] = 1.0;
        }
        auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, filtered * nb);
        knowhere::BitsetView bitset(bitset_data.data(), nb, static_cast<size_t>(filtered * nb));
        auto filtered_expected = base.Search(query_ds, filtered_search, bitset);
        auto actual = idx.Search(query_ds, filtered_search, bitset);
        REQUIRE(filtered_expected.has_value());
        REQUIRE(actual.has_value());
        RequireSameResults(*filtered_expected.value(), *actual.value(), AllScores(base, query_ds, search, bitset),
                           bitset, nb);
    }

    SECTION("a bitmap shorter than the index filters the trailing rows") {
        const int64_t bits = nb - 1000;
        std::vector<uint8_t> bitset_data((bits + 7) / 8, 0);
        knowhere::BitsetView bitset(bitset_data.data(), bits, 0);
        bitset.set_vector_count(nb);
        REQUIRE_FALSE(bitset.empty());
        auto filtered_expected = base.Search(query_ds, search, bitset);
        auto actual = idx.Search(query_ds, search, bitset);
        REQUIRE(filtered_expected.has_value());
        REQUIRE(actual.has_value());
        RequireSameResults(*filtered_expected.value(), *actual.value(), oracle, bitset, bits);
    }

    SECTION("deserialize and deserialize from file") {
        auto loaded = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
        REQUIRE(loaded.Deserialize(bs, build) == knowhere::Status::success);
        auto actual = loaded.Search(query_ds, search, nullptr);
        REQUIRE(actual.has_value());
        RequireSameResults(*expected.value(), *actual.value(), oracle, no_filter, nb);

        char path[] = "/tmp/knowhere_sindi_reorder_XXXXXX";
        const auto fd = mkstemp(path);
        REQUIRE(fd >= 0);
        close(fd);
        {
            auto binary = bs.GetByName(type);
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(binary->data.get()), binary->size);
        }
        auto mapped = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
        REQUIRE(mapped.DeserializeFromFile(path, build) == knowhere::Status::success);
        auto mapped_result = mapped.Search(query_ds, search, nullptr);
        REQUIRE(mapped_result.has_value());
        RequireSameResults(*expected.value(), *mapped_result.value(), oracle, no_filter, nb);
        std::remove(path);
    }

    SECTION("iterator scores are returned by row id") {
        auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, nb / 2);
        knowhere::BitsetView bitset(bitset_data.data(), nb, nb / 2);
        auto base_its = base.AnnIterator(query_ds, search, bitset);
        auto its = idx.AnnIterator(query_ds, search, bitset);
        REQUIRE(base_its.has_value());
        REQUIRE(its.has_value());
        for (int32_t q = 0; q < nq; ++q) {
            std::map<int64_t, float> base_scores, scores;
            auto& base_it = base_its.value()[q];
            auto& it = its.value()[q];
            while (base_it->HasNext().value()) {
                auto [id, dist] = base_it->Next().value();
                base_scores[id] = dist;
            }
            while (it->HasNext().value()) {
                auto [id, dist] = it->Next().value();
                REQUIRE(!bitset.test(id));
                scores[id] = dist;
            }
            REQUIRE(scores.size() == base_scores.size());
            for (const auto& [id, dist] : base_scores) {
                REQUIRE(scores.count(id) == 1);
                REQUIRE(std::abs(scores[id] - dist) <= 1e-5f * std::max(1.0f, std::abs(dist)));
            }
        }
    }
}

TEST_CASE("SINDI doc reorder corrupt order section is rejected", "[sparse][sindi][reorder]") {
    const int32_t nb = 3000, dim = 300;
    auto train_ds = GenSparseDataSetWithMaxVal(nb, dim, 0.95, 20, true);
    knowhere::Json build{{knowhere::meta::DIM, dim},
                         {knowhere::meta::METRIC_TYPE, knowhere::metric::BM25},
                         {knowhere::indexparam::INVERTED_INDEX_ALGO, "SINDI"},
                         {knowhere::meta::BM25_K1, 1.2},
                         {knowhere::meta::BM25_B, 0.75},
                         {knowhere::meta::BM25_AVGDL, 100},
                         {"sindi_doc_reorder", "bp_kmeans"}};
    const auto type = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
    REQUIRE(idx.Build(train_ds, build) == knowhere::Status::success);
    knowhere::BinarySet bs;
    REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
    auto binary = bs.GetByName(type);

    knowhere::MemoryIOReader reader(binary->data.get(), binary->size);
    reader.advance(knowhere::sparse::inverted::kInvertedIndexFileHeaderSize);
    uint32_t nr_sections = 0;
    reader.read(&nr_sections, sizeof(uint32_t));
    const auto headers = knowhere::sparse::inverted::read_section_headers(reader, nr_sections);
    const auto* order =
        knowhere::sparse::inverted::find_section_header(headers, InvertedIndexSectionType::SINDI_DOC_ORDER);
    REQUIRE(order != nullptr);

    auto loaded = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
    REQUIRE(loaded.Deserialize(bs, build) == knowhere::Status::success);

    SECTION("a file without the doc-order version") {
        const uint32_t version = knowhere::sparse::inverted::kInvertedIndexFileFormatVersion;
        std::memcpy(binary->data.get(), &version, sizeof(version));
        auto rejected = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
        REQUIRE(rejected.Deserialize(bs, build) != knowhere::Status::success);
    }

    SECTION("an order that is not a permutation") {
        auto* ids = reinterpret_cast<int32_t*>(binary->data.get() + order->offset + 2 * sizeof(uint32_t));
        ids[1] = ids[0];
        auto rejected = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
        REQUIRE(rejected.Deserialize(bs, build) != knowhere::Status::success);
    }
}

TEST_CASE("SINDI doc reorder parameter validation", "[sparse][sindi][reorder]") {
    const int32_t nb = 1000, dim = 100;
    auto train_ds = GenSparseDataSet(nb, dim, 0.9);
    knowhere::Json build{{knowhere::meta::DIM, dim},
                         {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
                         {knowhere::indexparam::INVERTED_INDEX_ALGO, "SINDI"},
                         {"sindi_doc_reorder", "bp_kmeans"}};
    const auto type = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    auto wrong_value = build;
    wrong_value["sindi_doc_reorder"] = "random";
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
    REQUIRE(idx.Build(train_ds, wrong_value) != knowhere::Status::success);

    auto not_sindi = build;
    not_sindi[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_MAXSCORE";
    auto idx2 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
    REQUIRE(idx2.Build(train_ds, not_sindi) != knowhere::Status::success);

    auto old = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, 10).value();
    REQUIRE(old.Build(train_ds, build) != knowhere::Status::success);

    // IP indexes default to SINDI, so the algorithm may be omitted.
    auto implicit_sindi = build;
    implicit_sindi.erase(knowhere::indexparam::INVERTED_INDEX_ALGO);
    auto idx3 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, kVersion).value();
    REQUIRE(idx3.Build(train_ds, implicit_sindi) == knowhere::Status::success);
}
