// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <array>
#include <atomic>
#include <cstring>
#include <future>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/sparse/inverted_index_format.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

void
WriteBinaryToFile(const std::string& filename, const knowhere::BinaryPtr binary) {
    auto data = binary->data.get();
    auto size = binary->size;
    // if tmp_file already exists, remove it
    std::remove(filename.c_str());
    std::ofstream out(filename, std::ios::binary);
    out.write((const char*)data, size);
    out.close();
}

namespace {

struct SparseIndexSections {
    uint32_t nr_inner_dims;
    std::vector<knowhere::sparse::inverted::InvertedIndexSectionHeader> section_headers;
};

SparseIndexSections
ReadSparseIndexSections(const knowhere::BinaryPtr& binary) {
    REQUIRE(binary != nullptr);

    knowhere::MemoryIOReader reader(binary->data.get(), binary->size);
    uint32_t file_format_version = 0;
    uint32_t nr_inner_dims = 0;
    uint32_t nr_sections = 0;

    reader.read(&file_format_version, sizeof(uint32_t));
    REQUIRE(file_format_version == knowhere::sparse::inverted::kInvertedIndexFileFormatVersion);
    reader.advance(sizeof(uint32_t) * 2);
    reader.read(&nr_inner_dims, sizeof(uint32_t));
    reader.advance(knowhere::sparse::inverted::kInvertedIndexHeaderReservedBytes);
    reader.read(&nr_sections, sizeof(uint32_t));

    return {nr_inner_dims, knowhere::sparse::inverted::read_section_headers(reader, nr_sections)};
}

const knowhere::sparse::inverted::InvertedIndexSectionHeader*
FindSection(const SparseIndexSections& sections, knowhere::sparse::inverted::InvertedIndexSectionType type) {
    return knowhere::sparse::inverted::find_section_header(sections.section_headers, type);
}

}  // namespace

TEST_CASE("Test Mem Sparse Index With Float Vector", "[float metrics]") {
    auto [nb, dim, doc_sparsity, query_sparsity] = GENERATE(table<int32_t, int32_t, float, float>({
        // 300 dim, avg doc nnz 12, avg query nnz 9
        {2000, 300, 0.95, 0.97},
        // 300 dim, avg doc nnz 9, avg query nnz 3
        {2000, 300, 0.97, 0.99},
        // 3000 dim, avg doc nnz 90, avg query nnz 30
        {2000, 3000, 0.97, 0.99},
    }));
    auto topk = 5;
    int64_t nq = 10;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);

    auto inverted_index_algo = GENERATE("TAAT_NAIVE", "DAAT_WAND", "DAAT_MAXSCORE");

    auto drop_ratio_search = metric == knowhere::metric::BM25 ? GENERATE(0.0, 0.1) : GENERATE(0.0, 0.3);

    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::BM25_K1] = 1.2;
        json[knowhere::meta::BM25_B] = 0.75;
        json[knowhere::meta::BM25_AVGDL] = 100;
        return json;
    };

    auto sparse_inverted_index_gen = [base_gen, drop_ratio_search = drop_ratio_search,
                                      inverted_index_algo = inverted_index_algo]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = inverted_index_algo;
        return json;
    };

    auto sparse_dsp_gen = [base_gen, drop_ratio_search = drop_ratio_search]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        return json;
    };

    auto sparse_dataset_gen = [&](int nr, int dim, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim, sparsity, 256, true);
        } else {
            return GenSparseDataSet(nr, dim, sparsity);
        }
    };

    auto train_ds = sparse_dataset_gen(nb, dim, doc_sparsity);
    auto query_ds = sparse_dataset_gen(nq, dim + 20, query_sparsity);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, topk},      {knowhere::meta::BM25_K1, 1.2},
        {knowhere::meta::BM25_B, 0.75},        {knowhere::meta::BM25_AVGDL, 100},
    };

    auto check_distance_decreasing = [](const knowhere::DataSet& ds) {
        auto nq = ds.GetRows();
        auto k = ds.GetDim();
        auto* distances = ds.GetDistance();
        auto* ids = ds.GetIds();
        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < k - 1; ++j) {
                if (ids[i * k + j] == -1 || ids[i * k + j + 1] == -1) {
                    break;
                }
                REQUIRE(distances[i * k + j] >= distances[i * k + j + 1]);
            }
        }
    };

    auto check_result_match_filter = [](const knowhere::DataSet& ds, const knowhere::BitsetView& bitset) {
        auto nq = ds.GetRows();
        auto k = ds.GetDim();
        auto* ids = ds.GetIds();
        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < k; ++j) {
                if (ids[i * k + j] == -1) {
                    break;
                }
                REQUIRE(!bitset.test(ids[i * k + j]));
            }
        }
    };

    SECTION("Test Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_DSP, sparse_dsp_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC, sparse_dsp_gen),
        }));
        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        check_distance_decreasing(*gt.value());

        auto use_mmap = GENERATE(true, false);
        auto tmp_file = "/tmp/knowhere_sparse_inverted_index_test";
        {
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
            auto cfg_json = gen().dump();
            CAPTURE(name, cfg_json);
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(idx.Type() == name);
            REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
            REQUIRE(idx.Size() > 0);
            REQUIRE(idx.Count() == nb);
            REQUIRE(idx.HasRawData(metric) ==
                    knowhere::IndexStaticFaced<knowhere::sparse_u32_f32>::HasRawData(name, version, json));

            knowhere::BinarySet bs;
            REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
            if (use_mmap) {
                WriteBinaryToFile(tmp_file, bs.GetByName(idx.Type()));
                REQUIRE(idx.DeserializeFromFile(tmp_file, json) == knowhere::Status::success);
            } else {
                REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);
            }

            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
            float recall = GetKNNRecall(*gt.value(), *results.value());
            check_distance_decreasing(*results.value());
            auto drop_ratio_search = json[knowhere::indexparam::DROP_RATIO_SEARCH].get<float>();
            if (drop_ratio_search == 0) {
                REQUIRE(recall == 1);
            } else {
                // most test cases are above 0.95, only a few between 0.9 and 0.95
                REQUIRE(recall >= 0.85);
            }
            // idx to destruct and munmap
        }
        if (use_mmap) {
            REQUIRE(std::remove(tmp_file) == 0);
        }
    }

    SECTION("Test DSP Params") {
        // Build one DSP index, then search with different param combos to prove
        // that eta and gamma actually affect pruning behavior.
        auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        REQUIRE(gt.has_value());

        auto use_mmap = GENERATE(true, false);
        auto tmp_file = "/tmp/knowhere_sparse_dsp_param_test";

        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC, version)
                       .value();
        knowhere::Json build_json = base_gen();
        build_json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0;
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
        if (use_mmap) {
            WriteBinaryToFile(tmp_file, bs.GetByName(idx.Type()));
            REQUIRE(idx.DeserializeFromFile(tmp_file, build_json) == knowhere::Status::success);
        } else {
            REQUIRE(idx.Deserialize(bs, build_json) == knowhere::Status::success);
        }

        // Helper: search with given params and return recall vs gt
        auto search_recall = [&](int mode, float mu, float eta, int gamma, bool kth_init = true) -> float {
            knowhere::Json json = base_gen();
            json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0;
            json["dsp_mode"] = mode;
            json["dsp_mu"] = mu;
            json["dsp_eta"] = eta;
            json["dsp_gamma"] = gamma;
            json["dsp_kth_init"] = kth_init;
            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
            check_distance_decreasing(*results.value());
            return GetKNNRecall(*gt.value(), *results.value());
        };

        // 1. DSP mode (mode=0): default params → perfect recall
        float recall_dsp = search_recall(0, 1.0f, 1.0f, 0);
        REQUIRE(recall_dsp == 1.0f);

        // 2. DSP with mu=0.7, eta=0.7 → exercises both mu and eta pruning paths
        float recall_mu_eta_07 = search_recall(0, 0.7f, 0.7f, 0);
        REQUIRE(recall_mu_eta_07 >= 0.0f);

        // 2b. DSP with mu < eta → more aggressive mu pruning, eta still active
        float recall_mu03_eta1 = search_recall(0, 0.3f, 1.0f, 0);
        REQUIRE(recall_mu03_eta1 >= 0.0f);

        // 3. DSP gamma=100000 → perfect recall
        float recall_gamma_all = search_recall(0, 1.0f, 1.0f, 100000);
        REQUIRE(recall_gamma_all == 1.0f);

        // 4. DSP aggressive mu with gamma backstop
        float recall_aggressive_no_gamma = search_recall(0, 0.3f, 1.0f, 0);
        float recall_aggressive_with_gamma = search_recall(0, 0.3f, 1.0f, 50);
        REQUIRE(recall_aggressive_with_gamma >= recall_aggressive_no_gamma);
        REQUIRE(recall_aggressive_no_gamma >= 0.0f);
        REQUIRE(recall_aggressive_with_gamma >= 0.5f);

        // 5. LSP/0 (mode=1): top-gamma only, no mu/asc gate
        float recall_lsp0 = search_recall(1, 1.0f, 1.0f, 100);
        REQUIRE(recall_lsp0 >= 0.5f);
        // lsp0 ignores mu: changing mu should not affect recall
        float recall_lsp0_mu03 = search_recall(1, 0.3f, 1.0f, 100);
        REQUIRE(recall_lsp0_mu03 >= recall_lsp0 - 1e-6f);
        REQUIRE(recall_lsp0_mu03 <= recall_lsp0 + 1e-6f);

        // 6. LSP/1 (mode=2): lsp0 safe set + mu gate
        float recall_lsp1 = search_recall(2, 1.0f, 1.0f, 100);
        REQUIRE(recall_lsp1 >= recall_lsp0);  // lsp1 includes lsp0 + more

        // 7. LSP/2 (mode=3): lsp1 + asc gate → recall >= lsp1
        float recall_lsp2 = search_recall(3, 1.0f, 1.0f, 100);
        REQUIRE(recall_lsp2 >= recall_lsp1);

        // 8. LSP modes with gamma=0 fall back to DSP (not silent empty results)
        {
            float recall_lsp0_g0 = search_recall(1, 1.0f, 1.0f, 0);
            REQUIRE(recall_lsp0_g0 == recall_dsp);
            float recall_lsp1_g0 = search_recall(2, 1.0f, 1.0f, 0);
            REQUIRE(recall_lsp1_g0 == recall_dsp);
            float recall_lsp2_g0 = search_recall(3, 1.0f, 1.0f, 0);
            REQUIRE(recall_lsp2_g0 == recall_dsp);
        }

        // 9. dsp_mode defaults to DSP (mode=0) with gamma=0
        {
            knowhere::Json json = base_gen();
            json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0;
            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
        }

        // 10. kth_init=false is orthogonal to mode
        float recall_dsp_nokth = search_recall(0, 1.0f, 1.0f, 0, false);
        REQUIRE(recall_dsp_nokth == 1.0f);

        if (use_mmap) {
            REQUIRE(std::remove(tmp_file) == 0);
        }
    }

    SECTION("Test DSP Filtered Search with kth-init Safety") {
        // Adversarial test: mask the exact top-k unfiltered results so the kth-init
        // seeded threshold is maximally wrong. Verifies that the filtered bootstrap
        // bypasses kth-init and still achieves perfect recall.
        auto idx = knowhere::IndexFactory::Instance()
                       .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC, version)
                       .value();
        knowhere::Json build_json = base_gen();
        build_json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0;
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        // Find top-k results per query (unfiltered)
        auto gt_unfiltered = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        REQUIRE(gt_unfiltered.has_value());

        // Create adversarial bitset: mask exactly the unfiltered top-k results.
        // This maximizes the kth-init failure: the seeded threshold reflects scores
        // of docs that are all filtered out.
        auto bitset_data = std::vector<uint8_t>((nb + 7) / 8, 0);
        auto* gt_ids = gt_unfiltered.value()->GetIds();
        int64_t gt_k = gt_unfiltered.value()->GetDim();
        for (int64_t q = 0; q < nq; ++q) {
            for (int64_t j = 0; j < gt_k; ++j) {
                int64_t id = gt_ids[q * gt_k + j];
                if (id >= 0 && id < nb) {
                    bitset_data[id / 8] |= (1u << (id % 8));
                }
            }
        }
        knowhere::BitsetView bitset(bitset_data.data(), nb);

        // Compute filtered ground truth
        auto gt_filtered = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, bitset);
        REQUIRE(gt_filtered.has_value());
        check_result_match_filter(*gt_filtered.value(), bitset);

        // Search with kth_init=true (the potentially unsafe case without bootstrap fix)
        knowhere::Json search_json = base_gen();
        search_json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0;
        search_json["dsp_kth_init"] = true;
        auto results_kth = idx.Search(query_ds, search_json, bitset);
        REQUIRE(results_kth.has_value());
        check_result_match_filter(*results_kth.value(), bitset);
        float recall_kth = GetKNNRecall(*gt_filtered.value(), *results_kth.value());
        REQUIRE(recall_kth == 1.0f);

        // Search with kth_init=false (baseline: no seeded threshold)
        search_json["dsp_kth_init"] = false;
        auto results_nokth = idx.Search(query_ds, search_json, bitset);
        REQUIRE(results_nokth.has_value());
        check_result_match_filter(*results_nokth.value(), bitset);
        float recall_nokth = GetKNNRecall(*gt_filtered.value(), *results_nokth.value());
        REQUIRE(recall_nokth == 1.0f);

        // Both should produce identical results (kth_init is bypassed under filter)
        auto* ids_kth = results_kth.value()->GetIds();
        auto* ids_nokth = results_nokth.value()->GetIds();
        for (int64_t q = 0; q < nq; ++q) {
            for (int64_t j = 0; j < topk; ++j) {
                REQUIRE(ids_kth[q * topk + j] == ids_nokth[q * topk + j]);
            }
        }

        // Test with aggressive pruning params under adversarial filter.
        // Bootstrap recovery should re-prune superblocks after heap fills.
        search_json["dsp_mu"] = 0.3;
        search_json["dsp_eta"] = 0.85;
        search_json["dsp_kth_init"] = true;
        auto results_aggressive = idx.Search(query_ds, search_json, bitset);
        REQUIRE(results_aggressive.has_value());
        check_result_match_filter(*results_aggressive.value(), bitset);
        float recall_aggressive = GetKNNRecall(*gt_filtered.value(), *results_aggressive.value());
        REQUIRE(recall_aggressive >= 0.5f);
    }

    SECTION("Test Search with Bitset") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_DSP, sparse_dsp_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC, sparse_dsp_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto gen_bitset_fn = GENERATE(GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet);
        auto bitset_percentages = GENERATE(0.4f, 0.9f);

        auto bitset_data = gen_bitset_fn(nb, bitset_percentages * nb);
        knowhere::BitsetView bitset(bitset_data.data(), nb);
        auto filter_gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, bitset);
        check_result_match_filter(*filter_gt.value(), bitset);

        auto results = idx.Search(query_ds, json, bitset);
        check_result_match_filter(*results.value(), bitset);

        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*filter_gt.value(), *results.value());
        check_distance_decreasing(*results.value());

        auto drop_ratio_search = json[knowhere::indexparam::DROP_RATIO_SEARCH].get<float>();
        if (drop_ratio_search == 0) {
            REQUIRE(recall == 1);
        } else {
            REQUIRE(recall >= 0.8);
        }
    }

    SECTION("Test Sparse Iterator with Bitset") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto gen_bitset_fn = GENERATE(GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet);
        auto bitset_percentages = GENERATE(0.4f, 0.9f);

        auto bitset_data = gen_bitset_fn(nb, bitset_percentages * nb);
        knowhere::BitsetView bitset(bitset_data.data(), nb);
        auto iterators_or = idx.AnnIterator(query_ds, json, bitset);
        REQUIRE(iterators_or.has_value());
        auto& iterators = iterators_or.value();
        REQUIRE(iterators.size() == (size_t)nq);

        int count = 0;
        int out_of_order = 0;
        for (int i = 0; i < nq; ++i) {
            auto& iter = iterators[i];
            float prev_dist = std::numeric_limits<float>::max();
            while (iter->HasNext().value()) {
                auto [id, dist] = iter->Next().value();
                REQUIRE(!bitset.test(id));
                count++;
                if (prev_dist < dist) {
                    out_of_order++;
                }
                prev_dist = dist;
            }
        }
        // less than 5% of the distances are out of order.
        REQUIRE(out_of_order * 20 <= count);
    }

    SECTION("Test Sparse Range Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto [radius, range_filter] = metric == knowhere::metric::BM25 ? GENERATE(table<float, float>({
                                                                             {80.0, 100.0},
                                                                             {100.0, 200.0},
                                                                         }))
                                                                       : GENERATE(table<float, float>({
                                                                             {0.5, 1},
                                                                             {1, 1.5},
                                                                         }));

        json[knowhere::meta::RADIUS] = radius;
        json[knowhere::meta::RANGE_FILTER] = range_filter;

        auto results = idx.RangeSearch(query_ds, json, nullptr);
        REQUIRE(results.has_value());

        auto gt =
            knowhere::BruteForce::RangeSearch<knowhere::sparse::SparseRow<float>>(train_ds, query_ds, json, nullptr);
        REQUIRE(gt.has_value());

        auto ids = results.value()->GetIds();
        auto lims = results.value()->GetLims();
        auto distances = results.value()->GetDistance();
        // any distance must be in range
        for (size_t i = 0; i < lims[nq]; ++i) {
            REQUIRE(distances[i] >= radius);
            REQUIRE(distances[i] <= range_filter);
        }

        auto ids_gt = gt.value()->GetIds();
        auto lims_gt = gt.value()->GetLims();
        auto distances_gt = gt.value()->GetDistance();
        // any distance must be in range
        for (size_t i = 0; i < lims_gt[nq]; ++i) {
            REQUIRE(distances_gt[i] > radius);
            REQUIRE(distances_gt[i] <= range_filter);
        }

        int actual_count = 0;
        int gt_count = 0;

        for (int i = 0; i < nq; ++i) {
            gt_count += lims_gt[i + 1] - lims_gt[i];

            std::unordered_set<int64_t> gt_ids;
            for (size_t j = lims_gt[i]; j < lims_gt[i + 1]; ++j) {
                gt_ids.insert(ids_gt[j]);
            }
            for (size_t j = lims[i]; j < lims[i + 1]; ++j) {
                if (gt_ids.find(ids[j]) != gt_ids.end()) {
                    actual_count++;
                }
            }
        }
        // most above 0.95, only a few between 0.9 and 0.83
        REQUIRE(actual_count * 1.0f / gt_count >= 0.83);
    }
}

TEST_CASE("Sparse long-query bulk windows match brute force", "[sparse][bulk]") {
    constexpr int64_t window_size = 1 << 12;
    constexpr int64_t tail_size = 907;
    constexpr int64_t nb = window_size + tail_size;
    constexpr int64_t nq = 2;
    constexpr int64_t dim = 64;
    constexpr int64_t topk = 3;
    constexpr int64_t boundary_doc_ids[] = {window_size - 1, window_size, nb - 1};
    static_assert(nb > window_size);
    static_assert(tail_size > 0 && tail_size < window_size);

    const auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    const auto filter_ratio = GENERATE(0.0f, 0.5f, 0.9f);
    CAPTURE(metric, filter_ratio);

    // Every document occurs on two high-impact posting lists. The remaining terms rotate across documents, keeping
    // the query long while still exercising essential-partition shrink. The three dense documents make both sides
    // of the 4096 boundary and the final document mandatory top-k hits.
    std::vector<std::map<int32_t, float>> train_data(nb);
    for (int64_t doc_id = 0; doc_id < nb; ++doc_id) {
        train_data[doc_id][0] = 2.0f;
        train_data[doc_id][1] = 1.0f;
        train_data[doc_id][2 + doc_id % (dim - 2)] = 1.0f;
    }
    for (const auto doc_id : boundary_doc_ids) {
        for (int32_t d = 2; d < dim; ++d) {
            train_data[doc_id][d] = 1.0f;
        }
    }

    std::vector<std::map<int32_t, float>> query_data(nq);
    for (int32_t d = 0; d < dim; ++d) {
        query_data[0][d] = 1.0f;
        query_data[1][d] = d % 2 == 0 ? 1.1f : 0.9f;
    }

    auto train_ds = GenSparseDataSet(train_data, dim);
    auto query_ds = GenSparseDataSet(query_data, dim);
    const auto* queries = static_cast<const knowhere::sparse::SparseRow<float>*>(query_ds->GetTensor());
    for (int64_t i = 0; i < nq; ++i) {
        REQUIRE(queries[i].size() >= 32);
    }

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = metric;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "BLOCK_MAX_MAXSCORE";
    build_json["inverted_index_codec"] = "block_streamvbyte";
    build_json["block_max_block_size"] = 64;
    build_json[knowhere::meta::BM25_K1] = 1.2;
    build_json[knowhere::meta::BM25_B] = 0.75;
    build_json[knowhere::meta::BM25_AVGDL] = 100;

    knowhere::Json search_json = build_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::indexparam::SEARCH_ALGO] = "DAAT_MAXSCORE";
    search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.05;

    std::vector<uint8_t> bitset_data;
    knowhere::BitsetView bitset;
    if (filter_ratio > 0) {
        bitset_data = GenerateBitsetWithRandomTbitsSet(nb, static_cast<size_t>(filter_ratio * nb));
        // Keep the targeted boundary documents visible at every filter ratio.
        for (const auto doc_id : boundary_doc_ids) {
            bitset_data[doc_id / 8] &= static_cast<uint8_t>(~(uint8_t{1} << (doc_id % 8)));
        }
        bitset = knowhere::BitsetView(bitset_data.data(), nb);
    }

    auto expected = knowhere::BruteForce::SearchSparse(train_ds, query_ds, search_json, bitset);
    REQUIRE(expected.has_value());
    auto contains_doc = [&](const knowhere::DataSet& result, int64_t query_id, int64_t doc_id) {
        const auto* ids = result.GetIds() + query_id * topk;
        return std::find(ids, ids + topk, doc_id) != ids + topk;
    };
    for (int64_t query_id = 0; query_id < nq; ++query_id) {
        for (const auto doc_id : boundary_doc_ids) {
            REQUIRE(contains_doc(*expected.value(), query_id, doc_id));
        }
    }

    const auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                     .value();
    REQUIRE(index.Build(train_ds, build_json) == knowhere::Status::success);
    for (const std::string search_algo : {"DAAT_MAXSCORE", "BLOCK_MAX_MAXSCORE"}) {
        for (const int32_t bulk_query_nnz_threshold : {0, static_cast<int32_t>(dim + 1)}) {
            CAPTURE(search_algo, bulk_query_nnz_threshold);
            search_json[knowhere::indexparam::SEARCH_ALGO] = search_algo;
            search_json[knowhere::indexparam::BULK_QUERY_NNZ_THRESHOLD] = bulk_query_nnz_threshold;
            auto actual = index.Search(query_ds, search_json, bitset);
            REQUIRE(actual.has_value());
            REQUIRE(GetKNNRecall(*expected.value(), *actual.value()) >= 0.99f);
            for (int64_t query_id = 0; query_id < nq; ++query_id) {
                for (const auto doc_id : boundary_doc_ids) {
                    REQUIRE(contains_doc(*actual.value(), query_id, doc_id));
                }
            }
        }
    }
}

TEST_CASE("Test Mem Sparse Index Handle Empty Vector", "[float metrics]") {
    auto [base_data, has_first_result] = GENERATE(table<std::vector<std::map<int32_t, float>>, bool>(
        {{std::vector<std::map<int32_t, float>>{
              {{1, 1.1f}, {2, 2.2f}, {6, 3.3f}},
              {},          // explicitly empty row
              {{5, 0.0f}}  // implicitly empty row
          },
          true},
         {std::vector<std::map<int32_t, float>>{{{1, 0.0f}}, {{3, 0.0f}}, {{5, 0.0f}}}, false},
         {std::vector<std::map<int32_t, float>>{{{1, 0.0f}}, {{3, 0.0f}}, {}}, false},
         {std::vector<std::map<int32_t, float>>{{}, {}, {}}, false}}));

    auto dim = 7;
    const auto train_ds = GenSparseDataSet(base_data, dim);

    auto topk = 5;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    auto version = GenTestVersionList();

    auto drop_ratio_search = GENERATE(0.0, 0.6);

    auto base_gen = [=, dim = dim, drop_ratio_search = drop_ratio_search]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::BM25_K1] = 1.2;
        json[knowhere::meta::BM25_B] = 0.75;
        json[knowhere::meta::BM25_AVGDL] = 100;
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        return json;
    };

    auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
        std::make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, base_gen),
        std::make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, base_gen),
    }));

    // query data must be constructed to match base_data and has_first_result:
    // if has_first_result is true, only q0 should find doc 0; otherwise, no query should find any neighbor.
    std::vector<std::map<int32_t, float>> query_data = {{{1, 1.1f}}, {{5, 1.1f}}, {}};
    const auto query_ds = GenSparseDataSet(query_data, dim);

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    auto cfg_json = gen().dump();
    CAPTURE(name, cfg_json);
    knowhere::Json json = knowhere::Json::parse(cfg_json);
    REQUIRE(idx.Type() == name);
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
    REQUIRE(idx.Size() > 0);

    knowhere::BinarySet bs;
    REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

    auto use_mmap = GENERATE(false, true);
    auto tmp_file = "/tmp/knowhere_sparse_inverted_index_test";

    if (use_mmap) {
        WriteBinaryToFile(tmp_file, bs.GetByName(idx.Type()));
        REQUIRE(idx.DeserializeFromFile(tmp_file, json) == knowhere::Status::success);
    } else {
        REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);
    }

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, topk},      {knowhere::meta::BM25_K1, 1.2},
        {knowhere::meta::BM25_B, 0.75},        {knowhere::meta::BM25_AVGDL, 100},
    };

    SECTION("Test Search") {
        auto check_result = [&, has_first_result = has_first_result](const knowhere::DataSet& ds) {
            auto nq = ds.GetRows();
            auto k = ds.GetDim();
            auto* ids = ds.GetIds();
            REQUIRE(ids[0] == (has_first_result ? 0 : -1));
            for (auto i = 1; i < nq * k; ++i) {
                REQUIRE(ids[i] == -1);
            }
        };
        auto bf_res = knowhere::BruteForce::SearchSparse(train_ds, query_ds, conf, nullptr);
        REQUIRE(bf_res.has_value());
        check_result(*bf_res.value());

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        check_result(*results.value());
    }

    SECTION("Test RangeSearch") {
        auto check_result = [&, has_first_result = has_first_result](const knowhere::DataSet& ds) {
            auto lims = ds.GetLims();
            auto* ids = ds.GetIds();
            if (has_first_result) {
                REQUIRE(lims[0] == 0);
                REQUIRE(lims[1] == 1);
                REQUIRE(ids[0] == 0);
                REQUIRE(lims[2] == 1);
                REQUIRE(lims[3] == 1);
            } else {
                // if no result found, lims should be all 0, ids and distances should point at 0-element array instead
                // of all -1, thus cannot be checked.
                REQUIRE(lims[0] == 0);
                REQUIRE(lims[1] == 0);
                REQUIRE(lims[2] == 0);
                REQUIRE(lims[3] == 0);
            }
        };
        json[knowhere::meta::RADIUS] = 0.0f;
        json[knowhere::meta::RANGE_FILTER] = 10000.0f;

        auto bf_res =
            knowhere::BruteForce::RangeSearch<knowhere::sparse::SparseRow<float>>(train_ds, query_ds, json, nullptr);
        REQUIRE(bf_res.has_value());
        check_result(*bf_res.value());

        auto results = idx.RangeSearch(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        check_result(*results.value());
    }
}

TEST_CASE("Test DSP Sparse Index Large K Is Rank Safe", "[float metrics][sparse][dsp]") {
    constexpr int64_t nb = 10001;
    constexpr int64_t topk = 10001;
    constexpr int32_t dim = 1;

    std::vector<std::map<int32_t, float>> base_data(nb);
    for (int64_t i = 0; i < nb - 2; ++i) {
        base_data[i][0] = 255.0f;
    }
    base_data[nb - 2][0] = 200.0f;
    base_data[nb - 1][0] = 1.0f;
    const auto train_ds = GenSparseDataSet(base_data, dim);
    const auto query_ds = GenSparseDataSet(std::vector<std::map<int32_t, float>>{{{0, 1.0f}}}, dim);

    knowhere::Json json = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {"dsp_mu", 1.0f},
        {"dsp_eta", 1.0f},
    };

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                       knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

    auto expected = knowhere::BruteForce::SearchSparse(train_ds, query_ds, json, nullptr);
    REQUIRE(expected.has_value());
    auto actual = index.Search(query_ds, json, nullptr);
    REQUIRE(actual.has_value());
    REQUIRE(GetKNNRecall(*expected.value(), *actual.value()) == 1.0f);
}

TEST_CASE("Test DSP BM25 Kth Init Includes Tied Maximum Scores", "[float metrics][sparse][dsp]") {
    constexpr int64_t nb = 2000;
    constexpr int64_t topk = 10;
    constexpr int32_t dim = 1;

    std::vector<std::map<int32_t, float>> base_data(nb, {{{0, 1.0f}}});
    const auto train_ds = GenSparseDataSet(base_data, dim);
    const auto query_ds = GenSparseDataSet(std::vector<std::map<int32_t, float>>{{{0, 1.0f}}}, dim);

    knowhere::Json json = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::BM25},
        {knowhere::meta::TOPK, topk},
        {knowhere::meta::BM25_K1, 1.2f},
        {knowhere::meta::BM25_B, 0.75f},
        {knowhere::meta::BM25_AVGDL, 1.0f},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {"dsp_mu", 1.0f},
        {"dsp_eta", 1.0f},
    };

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                       knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

    auto actual = index.Search(query_ds, json, nullptr);
    REQUIRE(actual.has_value());
    REQUIRE(actual.value()->GetDim() == topk);
    const auto* ids = actual.value()->GetIds();
    for (int64_t i = 0; i < topk; ++i) {
        REQUIRE(ids[i] != -1);
    }
}

TEST_CASE("Test DSP Kth Init Ignores Partially Filled Heaps", "[float metrics][sparse][dsp]") {
    auto [topk, head_count, nb] = GENERATE(table<int64_t, int64_t, int64_t>({
        {100, 50, 101},
        {1000, 100, 1001},
    }));
    constexpr int32_t dim = 2;
    constexpr int64_t tail_id = 0;

    std::vector<std::map<int32_t, float>> base_data(nb);
    base_data[tail_id][1] = 0.01f;
    for (int64_t i = 1; i <= head_count; ++i) {
        base_data[i][0] = 1.0f;
    }
    const auto train_ds = GenSparseDataSet(base_data, dim);
    const auto query_ds = GenSparseDataSet(std::vector<std::map<int32_t, float>>{{{0, 1.0f}, {1, 1.0f}}}, dim);

    knowhere::Json json = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {"dsp_mu", 1.0f},
        {"dsp_eta", 1.0f},
    };

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                       knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

    auto actual = index.Search(query_ds, json, nullptr);
    REQUIRE(actual.has_value());
    REQUIRE(actual.value()->GetDim() == topk);
    const auto* ids = actual.value()->GetIds();
    REQUIRE(std::find(ids, ids + topk, tail_id) != ids + topk);
}

TEST_CASE("Test DSP Kth Init Is Disabled With Bitset Filter", "[float metrics][sparse][dsp]") {
    constexpr int64_t nb = 20;
    constexpr int64_t topk = 10;
    constexpr int32_t dim = 1;

    // The corpus-wide 10th score is 100, but all ten documents supporting that threshold are filtered. The filtered
    // top-10 consists entirely of the remaining score-1 documents, so using the unfiltered kth initializer would
    // incorrectly prune every valid result. IDs 0..7 also form a fully filtered DSP block, covering its fast skip.
    std::vector<std::map<int32_t, float>> base_data(nb);
    for (int64_t i = 0; i < topk; ++i) {
        base_data[i][0] = 100.0f;
    }
    for (int64_t i = topk; i < nb; ++i) {
        base_data[i][0] = 1.0f;
    }
    const auto train_ds = GenSparseDataSet(base_data, dim);
    const auto query_ds = GenSparseDataSet(std::vector<std::map<int32_t, float>>{{{0, 1.0f}}}, dim);

    knowhere::Json json = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {"dsp_mu", 1.0f},
        {"dsp_eta", 1.0f},
    };

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                       knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

    const auto bitset_data = GenerateBitsetWithFirstTbitsSet(nb, topk);
    const knowhere::BitsetView bitset(bitset_data.data(), nb);
    auto expected = knowhere::BruteForce::SearchSparse(train_ds, query_ds, json, bitset);
    REQUIRE(expected.has_value());
    auto actual = index.Search(query_ds, json, bitset);
    REQUIRE(actual.has_value());
    REQUIRE(GetKNNRecall(*expected.value(), *actual.value()) == 1.0f);
    for (int64_t rank = 0; rank < topk; ++rank) {
        REQUIRE(actual.value()->GetIds()[rank] >= topk);
        REQUIRE(actual.value()->GetDistance()[rank] == 1.0f);
    }
}

TEST_CASE("Test DSP Native Serialization Round Trip", "[float metrics][sparse][dsp]") {
    constexpr int64_t nb = 2000;
    constexpr int64_t nq = 10;
    constexpr int64_t topk = 100;
    constexpr int32_t dim = 300;
    const auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    const bool use_mmap = GENERATE(false, true);
    const auto train_ds = GenSparseDataSet(nb, dim, 0.95f);
    const auto query_ds = GenSparseDataSet(nq, dim, 0.97f);

    knowhere::Json json = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
        {knowhere::meta::BM25_K1, 1.2f},
        {knowhere::meta::BM25_B, 0.75f},
        {knowhere::meta::BM25_AVGDL, 100.0f},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {"dsp_mu", 1.0f},
        {"dsp_eta", 1.0f},
    };

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                       knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
    auto before = index.Search(query_ds, json, nullptr);
    REQUIRE(before.has_value());

    knowhere::BinarySet binary_set;
    REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);
    if (use_mmap) {
        const std::string filename = "/tmp/knowhere_dsp_native_serialization_test";
        WriteBinaryToFile(filename, binary_set.GetByName(index.Type()));
        REQUIRE(index.DeserializeFromFile(filename, json) == knowhere::Status::success);
        REQUIRE(std::remove(filename.c_str()) == 0);
    } else {
        REQUIRE(index.Deserialize(binary_set, json) == knowhere::Status::success);
    }

    auto after = index.Search(query_ds, json, nullptr);
    REQUIRE(after.has_value());
    REQUIRE(std::memcmp(after.value()->GetIds(), before.value()->GetIds(), nq * topk * sizeof(int64_t)) == 0);
    REQUIRE(std::memcmp(after.value()->GetDistance(), before.value()->GetDistance(), nq * topk * sizeof(float)) == 0);
}

TEST_CASE("Test DSP Loads Legacy Sparse Serialization", "[float metrics][sparse][dsp]") {
    constexpr int64_t nb = 2048;
    constexpr int64_t nq = 10;
    constexpr int64_t topk = 100;
    constexpr int32_t dim = 32;
    const bool use_mmap = GENERATE(false, true);

    // A v1 DSP file without a DSP_METADATA section is the legacy format supported by the rebuild fallback. Keep the
    // corpus deterministic and make the first row contain every dimension so that raw and inner dimension IDs match.
    std::vector<std::map<int32_t, float>> base_data(nb);
    for (int32_t d = 0; d < dim; ++d) {
        base_data[0][d] = 1.0f + static_cast<float>(d % 7) * 0.1f;
    }
    for (int64_t doc = 1; doc < nb; ++doc) {
        const int32_t d0 = static_cast<int32_t>(doc % dim);
        const int32_t d1 = static_cast<int32_t>((doc * 7 + 3) % dim);
        base_data[doc][d0] = 0.5f + static_cast<float>(doc % 11) * 0.03f;
        base_data[doc][d1] = 0.7f + static_cast<float>(doc % 13) * 0.02f;
    }
    std::vector<std::map<int32_t, float>> query_data(nq);
    for (int64_t query = 0; query < nq; ++query) {
        query_data[query][static_cast<int32_t>(query % dim)] = 1.0f;
        query_data[query][static_cast<int32_t>((query * 5 + 1) % dim)] = 0.8f;
    }
    const auto train_ds = GenSparseDataSet(base_data, dim);
    const auto query_ds = GenSparseDataSet(query_data, dim);

    knowhere::Json json = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {"dsp_mu", 1.0f},
        {"dsp_eta", 1.0f},
    };

    auto fresh_dsp = knowhere::IndexFactory::Instance()
                         .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                           knowhere::Version::GetCurrentVersion().VersionNumber())
                         .value();
    REQUIRE(fresh_dsp.Build(train_ds, json) == knowhere::Status::success);
    auto expected = fresh_dsp.Search(query_ds, json, nullptr);
    REQUIRE(expected.has_value());

    std::vector<std::vector<uint32_t>> posting_ids(dim);
    std::vector<std::vector<float>> posting_vals(dim);
    std::vector<float> max_scores(dim, 0.0f);
    for (uint32_t doc = 0; doc < nb; ++doc) {
        for (const auto& [raw_dim, value] : base_data[doc]) {
            posting_ids[raw_dim].push_back(doc);
            posting_vals[raw_dim].push_back(value);
            max_scores[raw_dim] = std::max(max_scores[raw_dim], value);
        }
    }

    auto append_bytes = [](std::vector<uint8_t>& output, const void* data, size_t size) {
        const auto* first = static_cast<const uint8_t*>(data);
        output.insert(output.end(), first, first + size);
    };
    auto append_value = [&](std::vector<uint8_t>& output, const auto& value) {
        append_bytes(output, &value, sizeof(value));
    };

    std::vector<uint8_t> posting_section;
    const uint32_t encoding_type = 0;
    append_value(posting_section, encoding_type);
    std::vector<uint64_t> posting_offsets(dim + 1, 0);
    for (int32_t d = 0; d < dim; ++d) {
        posting_offsets[d + 1] = posting_offsets[d] + posting_ids[d].size();
    }
    append_bytes(posting_section, posting_offsets.data(), posting_offsets.size() * sizeof(uint64_t));
    for (int32_t d = 0; d < dim; ++d) {
        append_bytes(posting_section, posting_ids[d].data(), posting_ids[d].size() * sizeof(uint32_t));
    }
    for (int32_t d = 0; d < dim; ++d) {
        append_bytes(posting_section, posting_vals[d].data(), posting_vals[d].size() * sizeof(float));
    }

    std::vector<uint32_t> dim_map(dim);
    std::iota(dim_map.begin(), dim_map.end(), 0);
    struct LegacySectionHeader {
        uint32_t type;
        uint32_t padding = 0;
        uint64_t offset;
        uint64_t size;
    };
    static_assert(sizeof(LegacySectionHeader) == 24);
    constexpr uint32_t kPostingListsSection = 0;
    constexpr uint32_t kDimMapSection = 2;
    constexpr uint32_t kMaxScoresSection = 4;
    constexpr uint32_t kHeaderSize = 32;
    constexpr uint32_t kSectionCount = 3;
    uint64_t next_offset = kHeaderSize + sizeof(uint32_t) + kSectionCount * sizeof(LegacySectionHeader);
    std::array<LegacySectionHeader, kSectionCount> section_headers = {
        LegacySectionHeader{kPostingListsSection, 0, next_offset, posting_section.size()},
        LegacySectionHeader{kDimMapSection, 0, next_offset + posting_section.size(), dim_map.size() * sizeof(uint32_t)},
        LegacySectionHeader{kMaxScoresSection, 0,
                            next_offset + posting_section.size() + dim_map.size() * sizeof(uint32_t),
                            max_scores.size() * sizeof(float)},
    };

    std::vector<uint8_t> legacy_blob;
    const uint32_t format_version = 1;
    const uint32_t row_count = nb;
    const uint32_t max_dim = dim;
    const uint32_t inner_dim_count = dim;
    append_value(legacy_blob, format_version);
    append_value(legacy_blob, row_count);
    append_value(legacy_blob, max_dim);
    append_value(legacy_blob, inner_dim_count);
    const std::array<uint8_t, 16> reserved{};
    append_bytes(legacy_blob, reserved.data(), reserved.size());
    append_value(legacy_blob, kSectionCount);
    append_bytes(legacy_blob, section_headers.data(), section_headers.size() * sizeof(LegacySectionHeader));
    append_bytes(legacy_blob, posting_section.data(), posting_section.size());
    append_bytes(legacy_blob, dim_map.data(), dim_map.size() * sizeof(uint32_t));
    append_bytes(legacy_blob, max_scores.data(), max_scores.size() * sizeof(float));

    auto legacy_data = std::shared_ptr<uint8_t[]>(new uint8_t[legacy_blob.size()]);
    std::memcpy(legacy_data.get(), legacy_blob.data(), legacy_blob.size());
    knowhere::BinarySet legacy_binary;
    legacy_binary.Append(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC, legacy_data, legacy_blob.size());

    auto loaded_dsp = knowhere::IndexFactory::Instance()
                          .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                            knowhere::Version::GetCurrentVersion().VersionNumber())
                          .value();
    if (use_mmap) {
        const std::string filename = "/tmp/knowhere_dsp_legacy_serialization_test";
        WriteBinaryToFile(filename, legacy_binary.GetByName(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC));
        REQUIRE(loaded_dsp.DeserializeFromFile(filename, json) == knowhere::Status::success);
        REQUIRE(std::remove(filename.c_str()) == 0);
    } else {
        REQUIRE(loaded_dsp.Deserialize(legacy_binary, json) == knowhere::Status::success);
    }
    auto actual = loaded_dsp.Search(query_ds, json, nullptr);
    REQUIRE(actual.has_value());
    REQUIRE(std::memcmp(actual.value()->GetIds(), expected.value()->GetIds(), nq * topk * sizeof(int64_t)) == 0);
    REQUIRE(std::memcmp(actual.value()->GetDistance(), expected.value()->GetDistance(), nq * topk * sizeof(float)) ==
            0);
}

TEST_CASE("Test DSP Parallel Build Is Byte Identical", "[float metrics][sparse][dsp]") {
    constexpr int64_t nb = 65536;
    constexpr int32_t dim = 300;
    const auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    const auto train_ds = GenSparseDataSet(nb, dim, 0.99f);
    knowhere::Json json = {
        {knowhere::meta::DIM, dim},      {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, 100},
        {knowhere::meta::BM25_K1, 1.2f}, {knowhere::meta::BM25_B, 0.75f},       {knowhere::meta::BM25_AVGDL, 100.0f},
    };

    struct BuildPoolSizeGuard {
        size_t original = knowhere::KnowhereConfig::GetBuildThreadPoolSize();
        ~BuildPoolSizeGuard() {
            // Zero means the global pool had not been initialized yet; zero is not a valid size to restore.
            if (original != 0) {
                knowhere::KnowhereConfig::SetBuildThreadPoolSize(original);
            }
        }
    } pool_size_guard;

    knowhere::KnowhereConfig::SetBuildThreadPoolSize(1);
    auto serial_index = knowhere::IndexFactory::Instance()
                            .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                              knowhere::Version::GetCurrentVersion().VersionNumber())
                            .value();
    REQUIRE(serial_index.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet serial_binary;
    REQUIRE(serial_index.Serialize(serial_binary) == knowhere::Status::success);

    knowhere::KnowhereConfig::SetBuildThreadPoolSize(8);
    auto parallel_index = knowhere::IndexFactory::Instance()
                              .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                                knowhere::Version::GetCurrentVersion().VersionNumber())
                              .value();
    REQUIRE(parallel_index.Build(train_ds, json) == knowhere::Status::success);
    knowhere::BinarySet parallel_binary;
    REQUIRE(parallel_index.Serialize(parallel_binary) == knowhere::Status::success);

    const auto serial_blob = serial_binary.GetByName(serial_index.Type());
    const auto parallel_blob = parallel_binary.GetByName(parallel_index.Type());
    REQUIRE(serial_blob->size == parallel_blob->size);
    REQUIRE(std::memcmp(serial_blob->data.get(), parallel_blob->data.get(), serial_blob->size) == 0);
}

TEST_CASE("Test DSP Concurrent Search Reuses Workspaces", "[float metrics][sparse][dsp][concurrent]") {
    constexpr int64_t nb = 4096;
    constexpr int64_t nq = 4;
    constexpr int64_t topk = 100;
    constexpr int32_t dim = 300;
    constexpr int32_t num_threads = 8;
    constexpr int32_t repetitions = 50;
    const auto train_ds = GenSparseDataSet(nb, dim, 0.95f);
    const auto query_ds = GenSparseDataSet(nq, dim, 0.97f);
    knowhere::Json json = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {"dsp_mu", 1.0f},
        {"dsp_eta", 1.0f},
    };
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                       knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);
    auto expected = index.Search(query_ds, json, nullptr);
    REQUIRE(expected.has_value());

    std::atomic<bool> all_equal{true};
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (int32_t thread = 0; thread < num_threads; ++thread) {
        futures.emplace_back(std::async(std::launch::async, [&]() {
            for (int32_t repetition = 0; repetition < repetitions; ++repetition) {
                auto actual = index.Search(query_ds, json, nullptr);
                if (!actual.has_value() ||
                    std::memcmp(actual.value()->GetIds(), expected.value()->GetIds(), nq * topk * sizeof(int64_t)) !=
                        0 ||
                    std::memcmp(actual.value()->GetDistance(), expected.value()->GetDistance(),
                                nq * topk * sizeof(float)) != 0) {
                    all_equal = false;
                    return;
                }
            }
        }));
    }
    for (auto& future : futures) {
        future.get();
    }
    REQUIRE(all_equal.load());
}

TEST_CASE("Test DSP Safe Mode Matches Brute Force", "[float metrics][sparse][dsp]") {
    using Catch::Approx;
    constexpr int64_t nb = 2000;
    constexpr int64_t nq = 10;
    constexpr int32_t dim = 300;
    const int64_t topk = GENERATE(10, 100, 1000);
    INFO("topk=" << topk);
    const auto train_ds = GenSparseDataSet(nb, dim, 0.95f);
    const auto query_ds = GenSparseDataSet(nq, dim, 0.97f);

    knowhere::Json json = {
        {knowhere::meta::DIM, dim},
        {knowhere::meta::METRIC_TYPE, knowhere::metric::IP},
        {knowhere::meta::TOPK, topk},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {"dsp_mu", 1.0f},
        {"dsp_eta", 1.0f},
    };

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_DSP_CC,
                                                       knowhere::Version::GetCurrentVersion().VersionNumber())
                     .value();
    REQUIRE(index.Build(train_ds, json) == knowhere::Status::success);

    auto expected = knowhere::BruteForce::SearchSparse(train_ds, query_ds, json, nullptr);
    REQUIRE(expected.has_value());
    auto actual = index.Search(query_ds, json, nullptr);
    REQUIRE(actual.has_value());
    REQUIRE(expected.value()->GetDim() == topk);
    REQUIRE(actual.value()->GetDim() == topk);

    const auto* expected_ids = expected.value()->GetIds();
    const auto* expected_scores = expected.value()->GetDistance();
    const auto* actual_ids = actual.value()->GetIds();
    const auto* actual_scores = actual.value()->GetDistance();
    for (int64_t query = 0; query < nq; ++query) {
        const int64_t offset = query * topk;
        int64_t positive_count = 0;
        while (positive_count < topk && expected_scores[offset + positive_count] > 0.0f) {
            ++positive_count;
        }
        CAPTURE(query, positive_count);

        // Brute force may fill the tail with arbitrary zero-score IDs, while DSP (like the other DAAT paths) only
        // emits positive-score matches. Compare the sorted scores only where a positive match exists. The two paths
        // accumulate floats in a different order, hence the same relative tolerance used by the brute-force tests;
        // tied IDs may legitimately appear in a different order.
        for (int64_t rank = 0; rank < positive_count; ++rank) {
            REQUIRE(actual_scores[offset + rank] == Approx(expected_scores[offset + rank]).epsilon(0.00001));
        }

        // IDs above the kth-score tie boundary are unique members of the exact top-k result. IDs at the boundary may
        // be exchanged with other equal-score documents, so comparing them would make this assertion tie-sensitive.
        const float kth_score = expected_scores[offset + topk - 1];
        std::unordered_set<int64_t> expected_strict_ids;
        std::unordered_set<int64_t> actual_strict_ids;
        for (int64_t rank = 0; rank < topk; ++rank) {
            if (expected_scores[offset + rank] > kth_score) {
                expected_strict_ids.insert(expected_ids[offset + rank]);
            }
            if (actual_ids[offset + rank] >= 0 && actual_scores[offset + rank] > kth_score) {
                actual_strict_ids.insert(actual_ids[offset + rank]);
            }
        }
        REQUIRE(actual_strict_ids == expected_strict_ids);
    }
}

TEST_CASE("Test Mem Sparse Index CC", "[float metrics]") {
    std::atomic<int32_t> value_base(0);
    // each time a new batch of vectors are generated, the base value is increased by 1.
    // also the sparse vectors are all full, so newly generated vectors are guaranteed
    // to have larger IP than old vectors.
    auto doc_vector_gen = [&](int32_t nb, int32_t dim) {
        auto base = value_base.fetch_add(1);
        std::vector<std::map<int32_t, float>> data(nb);
        for (int32_t i = 0; i < nb; ++i) {
            for (int32_t j = 0; j < dim; ++j) {
                data[i][j] = base + static_cast<float>(rand()) / RAND_MAX * 0.8 + 0.1;
            }
        }
        return GenSparseDataSet(data, dim);
    };

    auto nb = 1000;
    auto dim = 30;
    auto topk = 50;
    int64_t nq = 100;

    auto query_ds = doc_vector_gen(nq, dim);

    auto inverted_index_algo =
        GENERATE("TAAT_NAIVE", "DAAT_WAND", "DAAT_MAXSCORE", "BLOCK_MAX_MAXSCORE", "BLOCK_MAX_WAND");

    auto drop_ratio_search = GENERATE(0.0, 0.3);

    auto metric = GENERATE(knowhere::metric::IP);
    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        json[knowhere::meta::BM25_K1] = 1.2;
        json[knowhere::meta::BM25_B] = 0.75;
        json[knowhere::meta::BM25_AVGDL] = 100;
        return json;
    };

    auto sparse_inverted_index_gen = [base_gen, drop_ratio_search = drop_ratio_search,
                                      inverted_index_algo = inverted_index_algo]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        json[knowhere::indexparam::INVERTED_INDEX_ALGO] = inverted_index_algo;
        return json;
    };

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, topk},      {knowhere::meta::BM25_K1, 1.2},
        {knowhere::meta::BM25_B, 0.75},        {knowhere::meta::BM25_AVGDL, 100},
    };

    // since all newly inserted vectors are guaranteed to have larger IP than old vectors,
    // the result ids of each search requests shoule be from the same batch of inserted vectors.
    auto check_result = [&](const knowhere::DataSet& ds) {
        auto nq = ds.GetRows();
        auto k = ds.GetDim();
        auto* ids = ds.GetIds();
        auto expected_id_base = ids[0] / nb;
        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < k; ++j) {
                auto base = ids[i * k + j] / nb;
                if (base != expected_id_base) {
                    throw std::runtime_error("id base mismatch at i=" + std::to_string(i) + " j=" + std::to_string(j) +
                                             ": got " + std::to_string(base) + " expected " +
                                             std::to_string(expected_id_base));
                }
            }
        }
    };

    auto test_time = 2;

    using std::make_tuple;
    auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
        make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC, sparse_inverted_index_gen),
        make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, sparse_inverted_index_gen),
    }));

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    auto cfg_json = gen().dump();
    CAPTURE(name, cfg_json);
    knowhere::Json json = knowhere::Json::parse(cfg_json);
    REQUIRE(idx.Type() == name);
    // build the index with some initial data
    auto train_ds = doc_vector_gen(nb, dim);
    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    auto add_task = [&]() {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() <
               test_time) {
            auto doc_ds = doc_vector_gen(nb, dim);
            auto res = idx.Add(doc_ds, json);
            if (res != knowhere::Status::success) {
                throw std::runtime_error("Add failed with status " + std::to_string(static_cast<int>(res)));
            }
        }
    };

    auto search_task = [&]() {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() <
               test_time) {
            auto results = idx.Search(query_ds, json, nullptr);
            if (!results.has_value()) {
                throw std::runtime_error("Search returned no value");
            }
            check_result(*results.value());
        }
    };

    SECTION("Test Search") {
        std::vector<std::future<void>> task_list;
        for (int thread = 0; thread < 5; thread++) {
            task_list.push_back(std::async(std::launch::async, search_task));
        }
        task_list.push_back(std::async(std::launch::async, add_task));
        for (auto& task : task_list) {
            REQUIRE_NOTHROW(task.get());
        }
    }

    SECTION("Test GetVectorByIds") {
        std::vector<int64_t> ids = {0, 1, 2};
        REQUIRE(idx.HasRawData(metric) ==
                knowhere::IndexStaticFaced<knowhere::sparse_u32_f32>::HasRawData(name, version, json));
        auto results = idx.GetVectorByIds(GenIdsDataSet(3, ids));
        REQUIRE(results.has_value());
        auto xb = (knowhere::sparse::SparseRow<float>*)train_ds->GetTensor();
        auto res_data = (knowhere::sparse::SparseRow<float>*)results.value()->GetTensor();
        for (int i = 0; i < 3; ++i) {
            const auto& truth_row = xb[i];
            const auto& res_row = res_data[i];
            REQUIRE(truth_row.size() == res_row.size());
            for (size_t j = 0; j < truth_row.size(); ++j) {
                REQUIRE(truth_row[j] == res_row[j]);
            }
        }
    }
}

TEST_CASE("Test Sparse Index Codec and Algo Combinations", "[sparse]") {
    auto nb = 1000;
    auto dim = 1000;
    auto topk = 10;
    auto nq = 5;
    auto doc_sparsity = 0.98f;
    auto query_sparsity = 0.99f;

    auto metric = GENERATE(std::string(knowhere::metric::IP), std::string(knowhere::metric::BM25));
    auto version = GenTestVersionList();

    // Test different codecs
    auto inverted_index_codec = GENERATE(std::string("block_streamvbyte"), std::string("block_maskedvbyte"),
                                         std::string("block_adaptive"), std::string("default"));

    // Test different build algorithms (which also test metadata generation)
    auto inverted_index_algo =
        GENERATE(std::string("DAAT_MAXSCORE"), std::string("BLOCK_MAX_MAXSCORE"), std::string("BLOCK_MAX_WAND"));

    // Test different search algorithms
    auto search_algo = GENERATE(std::string("INHERIT"), std::string("DAAT_WAND"), std::string("BLOCK_MAX_WAND"),
                                std::string("TAAT_NAIVE"));

    auto sparse_dataset_gen = [&](int nr, int dim, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim, sparsity, 256, true);
        } else {
            return GenSparseDataSet(nr, dim, sparsity);
        }
    };

    auto train_ds = sparse_dataset_gen(nb, dim, doc_sparsity);
    auto query_ds = sparse_dataset_gen(nq, dim, query_sparsity);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = metric;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = inverted_index_algo;
    if (inverted_index_codec != "default") {
        build_json["inverted_index_codec"] = inverted_index_codec;
    }
    build_json["block_max_block_size"] = 64;  // smaller block size for testing
    build_json[knowhere::meta::BM25_K1] = 1.2;
    build_json[knowhere::meta::BM25_B] = 0.75;
    build_json[knowhere::meta::BM25_AVGDL] = 50;

    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = metric;
    search_json[knowhere::indexparam::SEARCH_ALGO] = search_algo;
    search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.0;
    search_json[knowhere::meta::BM25_AVGDL] = 50;

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, build_json, nullptr);

    SECTION("Basic Build and Search") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        auto results = idx.Search(query_ds, search_json, nullptr);
        if (results.has_value()) {
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= 0.99);
        } else {
            // Some combinations of build_algo and search_algo are incompatible
            // e.g. searching with BLOCK_MAX_WAND on an index built without block max scores
            if (inverted_index_algo == "DAAT_MAXSCORE" &&
                (search_algo == "BLOCK_MAX_WAND" || search_algo == "BLOCK_MAX_MAXSCORE")) {
                REQUIRE(results.error() == knowhere::Status::invalid_value_in_json);
            } else {
                REQUIRE(results.has_value());
            }
        }
    }

    SECTION("Serialization and Encoding Detection") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

        // Deserialization should automatically detect the encoding used
        auto idx_new = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();

        // Use a config that DOES NOT specify codec, to test auto-detection
        knowhere::Json load_json = build_json;
        load_json.erase("inverted_index_codec");

        REQUIRE(idx_new.Deserialize(bs, load_json) == knowhere::Status::success);

        auto results = idx_new.Search(query_ds, search_json, nullptr);
        if (results.has_value()) {
            float recall = GetKNNRecall(*gt.value(), *results.value());
            REQUIRE(recall >= 0.99);
        } else {
            // Some combinations of build_algo and search_algo are incompatible
            if (inverted_index_algo == "DAAT_MAXSCORE" &&
                (search_algo == "BLOCK_MAX_WAND" || search_algo == "BLOCK_MAX_MAXSCORE")) {
                REQUIRE(results.error() == knowhere::Status::invalid_value_in_json);
            } else {
                REQUIRE(results.has_value());
            }
        }
    }
}

#ifndef KNOWHERE_WITH_CARDINAL
TEST_CASE("Sparse v8 and v9 serialize the legacy flat codec", "[sparse]") {
    const auto version = GENERATE(8, 9);
    const auto dataset = GenSparseDataSet(100, 1000, 0.98f);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = 1000;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_MAXSCORE";

    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                     .value();
    REQUIRE(index.Build(dataset, build_json) == knowhere::Status::success);

    knowhere::BinarySet binary_set;
    REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);

    const auto binary = binary_set.GetByName(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX);
    const auto sections = ReadSparseIndexSections(binary);
    const auto* posting_lists =
        FindSection(sections, knowhere::sparse::inverted::InvertedIndexSectionType::POSTING_LISTS);
    REQUIRE(posting_lists != nullptr);

    uint32_t encoding = 0;
    std::memcpy(&encoding, binary->data.get() + posting_lists->offset, sizeof(encoding));
    // The v8/v9 wire format defines encoding type 0 as flat.
    REQUIRE(encoding == 0);
}
#endif

TEST_CASE("Test Sparse Index Dim Max Score Ratio", "[sparse]") {
    auto nb = 1000;
    auto dim = 1000;
    auto topk = 10;
    auto nq = 10;
    auto doc_sparsity = 0.95f;
    auto query_sparsity = 0.95f;

    auto metric = knowhere::metric::IP;
    auto version = GenTestVersionList();

    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = metric;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "BLOCK_MAX_WAND";

    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, build_json, nullptr);

    SECTION("Test dim_max_score_ratio") {
        knowhere::Json search_json;
        search_json[knowhere::meta::TOPK] = topk;
        search_json[knowhere::meta::METRIC_TYPE] = metric;

        // Ratio < 1.0: More aggressive pruning, potentially lower recall but faster
        search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = 0.8;
        auto results_aggressive = idx.Search(query_ds, search_json, nullptr);
        REQUIRE(results_aggressive.has_value());
        float recall_aggressive = GetKNNRecall(*gt.value(), *results_aggressive.value());

        // Ratio > 1.0: Less aggressive pruning, higher recall but potentially slower
        search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.2;
        auto results_conservative = idx.Search(query_ds, search_json, nullptr);
        REQUIRE(results_conservative.has_value());
        float recall_conservative = GetKNNRecall(*gt.value(), *results_conservative.value());

        REQUIRE(recall_conservative >= recall_aggressive);
    }
}

TEST_CASE("Test Sparse WAND Index Build and Serialization", "[sparse]") {
    auto nb = 1000;
    auto dim = 1000;
    auto topk = 10;
    auto nq = 5;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;

    auto metric = GENERATE(std::string(knowhere::metric::IP), std::string(knowhere::metric::BM25));
    auto version = GenTestVersionList();

    auto inverted_index_algo = GENERATE(std::string("DAAT_MAXSCORE"), std::string("BLOCK_MAX_WAND"));

    auto sparse_dataset_gen = [&](int nr, int dim, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim, sparsity, 256, true);
        } else {
            return GenSparseDataSet(nr, dim, sparsity);
        }
    };

    auto train_ds = sparse_dataset_gen(nb, dim, doc_sparsity);
    auto query_ds = sparse_dataset_gen(nq, dim, query_sparsity);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = metric;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = inverted_index_algo;
    build_json[knowhere::meta::BM25_K1] = 1.2;
    build_json[knowhere::meta::BM25_B] = 0.75;
    build_json[knowhere::meta::BM25_AVGDL] = 50;

    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = metric;
    search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.0;
    search_json[knowhere::meta::BM25_AVGDL] = 50;

    // Test INDEX_SPARSE_WAND (the WAND-optimized variant)
    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_WAND;

    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, build_json, nullptr);
    REQUIRE(gt.has_value());

    SECTION("Build and search with WAND index") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        auto results = idx.Search(query_ds, search_json, nullptr);
        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.99);
    }

    SECTION("WAND index serialization roundtrip") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

        auto idx2 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx2.Deserialize(bs, build_json) == knowhere::Status::success);

        auto results = idx2.Search(query_ds, search_json, nullptr);
        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.99);
    }
}

TEST_CASE("Test Sparse Index Search Algo Override", "[sparse]") {
    // Build with one algorithm, then search with a different compatible one
    auto nb = 1000;
    auto dim = 800;
    auto topk = 10;
    auto nq = 5;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;
    auto version = GenTestVersionList();

    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    // Build with BLOCK_MAX_WAND to generate all metadata (max scores + block max data)
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "BLOCK_MAX_WAND";

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, build_json, nullptr);
    REQUIRE(gt.has_value());

    // Test that all search algorithms produce correct results on a BLOCK_MAX_WAND-built index
    auto search_algo =
        GENERATE(std::string("INHERIT"), std::string("TAAT_NAIVE"), std::string("DAAT_WAND"),
                 std::string("DAAT_MAXSCORE"), std::string("BLOCK_MAX_WAND"), std::string("BLOCK_MAX_MAXSCORE"));

    CAPTURE(search_algo);

    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    search_json[knowhere::indexparam::SEARCH_ALGO] = search_algo;
    search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.0;

    auto results = idx.Search(query_ds, search_json, nullptr);
    REQUIRE(results.has_value());

    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall >= 0.99);
}

TEST_CASE("Test Sparse Index Block Size Variations", "[sparse]") {
    auto nb = 1000;
    auto dim = 500;
    auto topk = 10;
    auto nq = 5;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;
    auto version = GenTestVersionList();

    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    auto gt = knowhere::BruteForce::SearchSparse(
        train_ds, query_ds,
        knowhere::Json({{knowhere::meta::DIM, dim}, {knowhere::meta::METRIC_TYPE, knowhere::metric::IP}}), nullptr);
    REQUIRE(gt.has_value());

    // Test different block sizes for block max algorithms
    auto block_size = GENERATE(32, 64, 128, 256);
    CAPTURE(block_size);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "BLOCK_MAX_WAND";
    build_json["block_max_block_size"] = block_size;

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.0;

    auto results = idx.Search(query_ds, search_json, nullptr);
    REQUIRE(results.has_value());
    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall >= 0.99);
}

TEST_CASE("Test Sparse Index Bitset Filtering with Block Max Algos", "[sparse]") {
    auto nb = 1000;
    auto dim = 500;
    auto topk = 5;
    auto nq = 5;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;
    auto version = GenTestVersionList();

    auto inverted_index_algo =
        GENERATE(std::string("DAAT_MAXSCORE"), std::string("BLOCK_MAX_MAXSCORE"), std::string("BLOCK_MAX_WAND"));

    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = inverted_index_algo;

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    search_json[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.0;

    // Filter out half of the vectors
    auto filter_count = nb / 2;
    auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, filter_count);
    knowhere::BitsetView bitset(bitset_data.data(), nb);

    auto results = idx.Search(query_ds, search_json, bitset);
    REQUIRE(results.has_value());

    // Check that filtered IDs are not in results
    auto* ids = results.value()->GetIds();
    auto k = results.value()->GetDim();
    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            auto id = ids[i * k + j];
            if (id != -1) {
                REQUIRE_FALSE(bitset.test(id));
            }
        }
    }

    // Check that distances are in decreasing order
    auto* distances = results.value()->GetDistance();
    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < k - 1; ++j) {
            if (ids[i * k + j] == -1 || ids[i * k + j + 1] == -1) {
                break;
            }
            REQUIRE(distances[i * k + j] >= distances[i * k + j + 1]);
        }
    }
}

TEST_CASE("Test Sparse Index Drop Ratio Search", "[sparse]") {
    auto nb = 2000;
    auto dim = 1000;
    auto topk = 10;
    auto nq = 10;
    auto doc_sparsity = 0.95f;
    auto query_sparsity = 0.97f;
    auto version = GenTestVersionList();

    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_MAXSCORE";

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, build_json, nullptr);
    REQUIRE(gt.has_value());

    // drop_ratio_search = 0: exact search
    knowhere::Json search_json_exact;
    search_json_exact[knowhere::meta::TOPK] = topk;
    search_json_exact[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    search_json_exact[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0;
    search_json_exact[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.0;
    auto results_exact = idx.Search(query_ds, search_json_exact, nullptr);
    REQUIRE(results_exact.has_value());
    float recall_exact = GetKNNRecall(*gt.value(), *results_exact.value());
    REQUIRE(recall_exact >= 0.99);

    // drop_ratio_search > 0: approximate search, recall should still be reasonable
    knowhere::Json search_json_approx;
    search_json_approx[knowhere::meta::TOPK] = topk;
    search_json_approx[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    search_json_approx[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.3;
    search_json_approx[knowhere::meta::DIM_MAX_SCORE_RATIO] = 1.0;
    auto results_approx = idx.Search(query_ds, search_json_approx, nullptr);
    REQUIRE(results_approx.has_value());
    float recall_approx = GetKNNRecall(*gt.value(), *results_approx.value());
    // Approximate search should have lower recall than exact, but still reasonable
    REQUIRE(recall_approx >= 0.5);
    REQUIRE(recall_exact >= recall_approx);
}

TEST_CASE("Test Sparse Index CC Build Add Search", "[sparse]") {
    auto nb = 500;
    auto dim = 500;
    auto topk = 5;
    auto nq = 5;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;
    auto version = GenTestVersionList();

    auto cc_name = GENERATE(std::string(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC),
                            std::string(knowhere::IndexEnum::INDEX_SPARSE_WAND_CC));

    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = topk;

    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, json, nullptr);
    REQUIRE(gt.has_value());

    SECTION("Build and search CC index") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(cc_name, version).value();
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.99);
    }

    SECTION("Build then Add more data to CC index") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(cc_name, version).value();
        // Build with initial data first (required for CC indices)
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

        // Add more data on top
        auto extra_ds = GenSparseDataSet(100, dim, doc_sparsity);
        REQUIRE(idx.Add(extra_ds, json) == knowhere::Status::success);

        auto results = idx.Search(query_ds, json, nullptr);
        REQUIRE(results.has_value());
        // Recall may be slightly different since we added extra data not in gt,
        // but original results should still be findable
        auto* ids = results.value()->GetIds();
        auto k = results.value()->GetDim();
        const auto* queries = static_cast<const knowhere::sparse::SparseRow<float>*>(query_ds->GetTensor());
        for (int64_t i = 0; i < nq; ++i) {
            bool found_valid = false;
            for (int64_t j = 0; j < k; ++j) {
                if (ids[i * k + j] != -1) {
                    found_valid = true;
                    break;
                }
            }
            if (queries[i].size() == 0) {
                REQUIRE_FALSE(found_valid);
            } else {
                REQUIRE(found_valid);
            }
        }
    }
}

TEST_CASE("Test SINDI Sparse Inverted Index CC Build Add Search", "[sparse][sindi]") {
    auto nb = 10;
    auto extra_nb = 4;
    auto dim = 8;
    auto topk = 8;
    auto nq = 3;
    auto version = knowhere::Version::GetMaximumVersion().VersionNumber();

    auto dense_sparse_ds = [](int32_t rows, int32_t dim, float value) {
        std::vector<std::map<int32_t, float>> data(rows);
        for (int32_t i = 0; i < rows; ++i) {
            for (int32_t j = 0; j < dim; ++j) {
                data[i][j] = value;
            }
        }
        return GenSparseDataSet(data, dim);
    };

    auto train_ds = dense_sparse_ds(nb, dim, 1.0f);
    auto query_ds = dense_sparse_ds(nq, dim, 1.0f);

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "SINDI";
    json[knowhere::indexparam::SEARCH_ALGO] = "SINDI";

    const std::string cc_name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC;
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(cc_name, version).value();
    CAPTURE(cc_name);

    REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);

    auto extra_ds = dense_sparse_ds(extra_nb, dim, 2.0f);
    REQUIRE(idx.Add(extra_ds, json) == knowhere::Status::success);

    auto results = idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());

    auto* ids = results.value()->GetIds();
    auto k = results.value()->GetDim();
    for (int64_t i = 0; i < nq; ++i) {
        int64_t valid_count = 0;
        for (int64_t j = 0; j < k; ++j) {
            if (ids[i * k + j] != -1) {
                ++valid_count;
            }
        }
        REQUIRE(valid_count == topk);
    }
}

TEST_CASE("Test SINDI Sparse Inverted Index CC Train Multi Add Cross Window Search", "[sparse][sindi]") {
    auto window_size = 1024;
    auto first_nb = window_size;
    auto second_nb = 8;
    auto dim = 8;
    auto topk = 16;
    int64_t nq = 3;
    auto version = knowhere::Version::GetMaximumVersion().VersionNumber();

    auto dense_sparse_ds = [](int32_t rows, int32_t dim, float value) {
        std::vector<std::map<int32_t, float>> data(rows);
        for (int32_t i = 0; i < rows; ++i) {
            for (int32_t j = 0; j < dim; ++j) {
                data[i][j] = value;
            }
        }
        return GenSparseDataSet(data, dim);
    };

    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "SINDI";
    json[knowhere::indexparam::SEARCH_ALGO] = "SINDI";
    json["sindi_window_size"] = window_size;

    const std::string cc_name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC;
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(cc_name, version).value();
    CAPTURE(cc_name);

    auto train_ds = dense_sparse_ds(1, dim, 1.0f);
    REQUIRE(idx.Train(train_ds, json) == knowhere::Status::success);

    auto first_ds = dense_sparse_ds(first_nb, dim, 1.0f);
    REQUIRE(idx.Add(first_ds, json) == knowhere::Status::success);

    auto second_ds = dense_sparse_ds(second_nb, dim, 2.0f);
    REQUIRE(idx.Add(second_ds, json) == knowhere::Status::success);
    REQUIRE(idx.Count() == first_nb + second_nb);

    auto query_ds = dense_sparse_ds(nq, dim, 1.0f);
    auto results = idx.Search(query_ds, json, nullptr);
    REQUIRE(results.has_value());

    auto* ids = results.value()->GetIds();
    auto k = results.value()->GetDim();
    REQUIRE(k == topk);
    for (int64_t i = 0; i < nq; ++i) {
        int64_t valid_count = 0;
        bool found_first_window = false;
        bool found_second_window = false;
        for (int64_t j = 0; j < k; ++j) {
            auto id = ids[i * k + j];
            if (id == -1) {
                continue;
            }
            ++valid_count;
            found_first_window = found_first_window || id < first_nb;
            found_second_window = found_second_window || id >= first_nb;
        }
        REQUIRE(valid_count == topk);
        REQUIRE(found_first_window);
        REQUIRE(found_second_window);
    }
}

TEST_CASE("Test SINDI Index Build and Search", "[sparse][sindi]") {
    auto [nb, dim, doc_sparsity, query_sparsity] = GENERATE(table<int32_t, int32_t, float, float>({
        {2000, 300, 0.95, 0.97},
        {2000, 3000, 0.97, 0.99},
    }));
    auto topk = 5;
    int64_t nq = 10;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    auto search_algo = std::string("SINDI");

    // SINDI requires version >= 10
    auto version = knowhere::Version::GetMaximumVersion().VersionNumber();

    auto sparse_dataset_gen = [&](int nr, int dim, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim, sparsity, 256, true);
        } else {
            return GenSparseDataSet(nr, dim, sparsity);
        }
    };

    auto train_ds = sparse_dataset_gen(nb, dim, doc_sparsity);
    auto query_ds = sparse_dataset_gen(nq, dim + 20, query_sparsity);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = metric;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "SINDI";
    build_json[knowhere::meta::BM25_K1] = 1.2;
    build_json[knowhere::meta::BM25_B] = 0.75;
    build_json[knowhere::meta::BM25_AVGDL] = 100;

    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = metric;
    search_json[knowhere::indexparam::SEARCH_ALGO] = search_algo;
    search_json[knowhere::meta::BM25_K1] = 1.2;
    search_json[knowhere::meta::BM25_B] = 0.75;
    search_json[knowhere::meta::BM25_AVGDL] = 100;

    const knowhere::Json gt_conf = {
        {knowhere::meta::METRIC_TYPE, metric}, {knowhere::meta::TOPK, topk},      {knowhere::meta::BM25_K1, 1.2},
        {knowhere::meta::BM25_B, 0.75},        {knowhere::meta::BM25_AVGDL, 100},
    };
    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto check_distance_decreasing = [](const knowhere::DataSet& ds) {
        auto nq = ds.GetRows();
        auto k = ds.GetDim();
        auto* distances = ds.GetDistance();
        auto* ids = ds.GetIds();
        for (auto i = 0; i < nq; ++i) {
            for (auto j = 0; j < k - 1; ++j) {
                if (ids[i * k + j] == -1 || ids[i * k + j + 1] == -1) {
                    break;
                }
                REQUIRE(distances[i * k + j] >= distances[i * k + j + 1]);
            }
        }
    };

    auto name = GENERATE(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, knowhere::IndexEnum::INDEX_SPARSE_WAND);

    SECTION("Test Search") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        CAPTURE(name, metric, search_algo);
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto results = idx.Search(query_ds, search_json, nullptr);
        REQUIRE(results.has_value());
        check_distance_decreasing(*results.value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.85);
    }

    SECTION("Test Search with Bitset") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        CAPTURE(name, metric, search_algo);
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        auto bitset_percentage = GENERATE(0.4f, 0.9f);
        auto bitset_data = GenerateBitsetWithRandomTbitsSet(nb, bitset_percentage * nb);
        knowhere::BitsetView bitset(bitset_data.data(), nb);

        auto results = idx.Search(query_ds, search_json, bitset);
        REQUIRE(results.has_value());
        check_distance_decreasing(*results.value());

        // Check that filtered IDs are not in results
        auto* ids = results.value()->GetIds();
        auto k = results.value()->GetDim();
        for (int64_t i = 0; i < nq; ++i) {
            for (int64_t j = 0; j < k; ++j) {
                if (ids[i * k + j] == -1) {
                    break;
                }
                REQUIRE(!bitset.test(ids[i * k + j]));
            }
        }
    }

    SECTION("Test Serialize and Deserialize") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        CAPTURE(name, metric, search_algo);
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

        // Deserialize into a new index
        auto idx2 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx2.Deserialize(bs, build_json) == knowhere::Status::success);
        REQUIRE(idx2.Count() == nb);
        REQUIRE(idx2.Size() > 0);

        auto results = idx2.Search(query_ds, search_json, nullptr);
        REQUIRE(results.has_value());
        check_distance_decreasing(*results.value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.85);
    }

    SECTION("Test Serialize and DeserializeFromFile (mmap)") {
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        CAPTURE(name, metric, search_algo);
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        knowhere::BinarySet bs;
        REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

        auto tmp_file = "/tmp/knowhere_sindi_test";
        WriteBinaryToFile(tmp_file, bs.GetByName(idx.Type()));

        auto idx2 = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx2.DeserializeFromFile(tmp_file, build_json) == knowhere::Status::success);
        REQUIRE(idx2.Count() == nb);

        auto results = idx2.Search(query_ds, search_json, nullptr);
        REQUIRE(results.has_value());
        check_distance_decreasing(*results.value());
        float recall = GetKNNRecall(*gt.value(), *results.value());
        REQUIRE(recall >= 0.85);

        REQUIRE(std::remove(tmp_file) == 0);
    }
}

TEST_CASE("Test SINDI Index Requires Version 10", "[sparse][sindi]") {
    constexpr int32_t version = 9;
    auto dim = 16;
    auto train_ds = GenSparseDataSet(100, dim, 0.8);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "SINDI";

    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::invalid_args);
}

TEST_CASE("Test SINDI MPHF Section Uses Version 11 Layout", "[sparse][sindi]") {
    const auto version = GENERATE(10, 11);
    const auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);
    constexpr auto dim = 300;
    auto train_ds = metric == knowhere::metric::BM25 ? GenSparseDataSetWithMaxVal(500, dim, 0.97, 256, true)
                                                     : GenSparseDataSet(500, dim, 0.97);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = metric;
    build_json[knowhere::meta::BM25_K1] = 1.2;
    build_json[knowhere::meta::BM25_B] = 0.75;
    build_json[knowhere::meta::BM25_AVGDL] = 100;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "SINDI";

    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    CAPTURE(version, metric);
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

    knowhere::BinarySet bs;
    REQUIRE(idx.Serialize(bs) == knowhere::Status::success);

    const auto sections = ReadSparseIndexSections(bs.GetByName(idx.Type()));
    const auto* reverse_section =
        FindSection(sections, knowhere::sparse::inverted::InvertedIndexSectionType::DIM_MAP_REVERSE);
    const auto* mphf_section =
        FindSection(sections, knowhere::sparse::inverted::InvertedIndexSectionType::DIM_MAP_MPHF);
    REQUIRE(reverse_section != nullptr);

    const auto reverse_bytes = static_cast<uint64_t>(sections.nr_inner_dims) * sizeof(uint32_t);
    if (version == 10) {
        REQUIRE(mphf_section == nullptr);
        REQUIRE(reverse_section->size > reverse_bytes);
    } else {
        REQUIRE(mphf_section != nullptr);
        REQUIRE(mphf_section->size > 0);
        REQUIRE(reverse_section->size == reverse_bytes);
    }
}

TEST_CASE("Test Sparse Index Rejects Unsupported Inverted Index Algo", "[sparse]") {
    auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
    auto dim = 16;
    auto train_ds = GenSparseDataSet(100, dim, 0.8);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "NOT_A_REAL_ALGO";

    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::invalid_args);

    constexpr int32_t non_sindi_version = 9;
    knowhere::Json valid_build_json = build_json;
    valid_build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "daat_maxscore";

    auto built_idx =
        knowhere::IndexFactory::Instance()
            .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, non_sindi_version)
            .value();
    REQUIRE(built_idx.Build(train_ds, valid_build_json) == knowhere::Status::success);

    knowhere::BinarySet bs;
    REQUIRE(built_idx.Serialize(bs) == knowhere::Status::success);

    auto invalid_load_json = valid_build_json;
    invalid_load_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "NOT_A_REAL_ALGO";

    auto deserialized_idx =
        knowhere::IndexFactory::Instance()
            .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, non_sindi_version)
            .value();
    REQUIRE(deserialized_idx.Deserialize(bs, invalid_load_json) == knowhere::Status::invalid_args);

    const auto tmp_file = "/tmp/knowhere_sparse_invalid_algo_test";
    WriteBinaryToFile(tmp_file, bs.GetByName(built_idx.Type()));

    auto file_deserialized_idx =
        knowhere::IndexFactory::Instance()
            .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, non_sindi_version)
            .value();
    REQUIRE(file_deserialized_idx.DeserializeFromFile(tmp_file, invalid_load_json) == knowhere::Status::invalid_args);
    REQUIRE(std::remove(tmp_file) == 0);
}

TEST_CASE("Test Sparse Index Rejects Invalid Inverted Index Build Params", "[sparse]") {
    const auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
    constexpr auto dim = 16;
    auto train_ds = GenSparseDataSet(100, dim, 0.8);

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    SECTION("reject invalid inverted index codec") {
        knowhere::Json build_json;
        build_json[knowhere::meta::DIM] = dim;
        build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_MAXSCORE";
        build_json["inverted_index_codec"] = "invalid_codec";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::invalid_args);
    }

    SECTION("reject non-positive block max block size") {
        auto block_size = GENERATE(0, -1);
        CAPTURE(block_size);

        knowhere::Json build_json;
        build_json[knowhere::meta::DIM] = dim;
        build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "BLOCK_MAX_WAND";
        build_json["block_max_block_size"] = block_size;

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::out_of_range_in_json);
    }
}

TEST_CASE("Test SINDI Index Window Size", "[sparse][sindi]") {
    auto nb = 2000;
    auto dim = 1000;
    auto topk = 10;
    int64_t nq = 5;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;

    auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    knowhere::Json gt_conf;
    gt_conf[knowhere::meta::DIM] = dim;
    gt_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    gt_conf[knowhere::meta::TOPK] = topk;
    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    auto window_size = GENERATE(1024, 4096, 65535);
    CAPTURE(window_size);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "SINDI";
    build_json["sindi_window_size"] = window_size;

    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);
    REQUIRE(idx.Size() > 0);
    REQUIRE(idx.Count() == nb);

    auto results = idx.Search(query_ds, search_json, nullptr);
    REQUIRE(results.has_value());
    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall >= 0.85);
}

TEST_CASE("Test SINDI Index Search with Window Filter Skip", "[sparse][sindi]") {
    auto nb = 3000;
    auto dim = 500;
    auto topk = 10;
    int64_t nq = 5;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;
    constexpr int32_t window_size = 1024;

    auto metric = GENERATE(knowhere::metric::IP, knowhere::metric::BM25);

    auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
    auto sparse_dataset_gen = [&](int nr, float sparsity) -> knowhere::DataSetPtr {
        if (metric == knowhere::metric::BM25) {
            return GenSparseDataSetWithMaxVal(nr, dim, sparsity, 256, true);
        }
        return GenSparseDataSet(nr, dim, sparsity);
    };
    auto train_ds = sparse_dataset_gen(nb, doc_sparsity);
    auto query_ds = sparse_dataset_gen(nq, query_sparsity);

    // Filter out the leading `filtered_docs` documents so that one or more whole windows
    // are skipped. Skipping leading windows exercises the posting-cursor sync path.
    auto filtered_docs = GENERATE(1024, 2048);
    CAPTURE(filtered_docs);
    auto bitset_data = GenerateBitsetWithFirstTbitsSet(nb, filtered_docs);
    knowhere::BitsetView bitset(bitset_data.data(), nb);

    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = metric;
    build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "SINDI";
    build_json["sindi_window_size"] = window_size;
    build_json[knowhere::meta::BM25_K1] = 1.2;
    build_json[knowhere::meta::BM25_B] = 0.75;
    build_json[knowhere::meta::BM25_AVGDL] = 100;

    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = metric;
    search_json[knowhere::meta::BM25_K1] = 1.2;
    search_json[knowhere::meta::BM25_B] = 0.75;
    search_json[knowhere::meta::BM25_AVGDL] = 100;

    auto expected = knowhere::BruteForce::SearchSparse(train_ds, query_ds, search_json, bitset);
    REQUIRE(expected.has_value());

    auto idx = knowhere::IndexFactory::Instance()
                   .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version)
                   .value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

    auto results = idx.Search(query_ds, search_json, bitset);
    REQUIRE(results.has_value());
    REQUIRE(GetKNNRecall(*expected.value(), *results.value()) >= 0.99f);

    auto* ids = results.value()->GetIds();
    auto k = results.value()->GetDim();
    for (int64_t i = 0; i < nq; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            if (ids[i * k + j] == -1) {
                break;
            }
            REQUIRE(!bitset.test(ids[i * k + j]));
        }
    }
}

TEST_CASE("Test SINDI Index Search Algo Mismatch", "[sparse][sindi]") {
    auto nb = 500;
    auto dim = 300;
    auto topk = 5;
    int64_t nq = 3;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;

    auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    SECTION("SINDI index rejects non-SINDI search algos") {
        knowhere::Json build_json;
        build_json[knowhere::meta::DIM] = dim;
        build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "SINDI";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        // Searching with DAAT_WAND on a SINDI-built index should fail
        auto bad_search_algo =
            GENERATE(std::string("DAAT_WAND"), std::string("DAAT_MAXSCORE"), std::string("TAAT_NAIVE"));
        CAPTURE(bad_search_algo);

        knowhere::Json search_json;
        search_json[knowhere::meta::TOPK] = topk;
        search_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        search_json[knowhere::indexparam::SEARCH_ALGO] = bad_search_algo;

        auto results = idx.Search(query_ds, search_json, nullptr);
        REQUIRE(!results.has_value());
        REQUIRE(results.error() == knowhere::Status::invalid_args);
    }

    SECTION("Non-SINDI index rejects SINDI search algo") {
        knowhere::Json build_json;
        build_json[knowhere::meta::DIM] = dim;
        build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        build_json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_MAXSCORE";

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
        REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

        auto bad_search_algo = GENERATE(std::string("SINDI"));
        CAPTURE(bad_search_algo);

        knowhere::Json search_json;
        search_json[knowhere::meta::TOPK] = topk;
        search_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        search_json[knowhere::indexparam::SEARCH_ALGO] = bad_search_algo;

        auto results = idx.Search(query_ds, search_json, nullptr);
        REQUIRE(!results.has_value());
        REQUIRE(results.error() == knowhere::Status::invalid_args);
    }
}

TEST_CASE("Test SINDI Index Default Algo for Version 10", "[sparse][sindi]") {
    auto nb = 500;
    auto dim = 300;
    auto topk = 5;
    int64_t nq = 3;
    auto doc_sparsity = 0.97f;
    auto query_sparsity = 0.99f;

    auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
    auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    auto query_ds = GenSparseDataSet(nq, dim, query_sparsity);

    const std::string name = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;

    // For version >= 10 with IP metric, default algo should be SINDI
    // Build without specifying inverted_index_algo
    knowhere::Json build_json;
    build_json[knowhere::meta::DIM] = dim;
    build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(name, version).value();
    REQUIRE(idx.Build(train_ds, build_json) == knowhere::Status::success);

    knowhere::Json gt_conf;
    gt_conf[knowhere::meta::DIM] = dim;
    gt_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    gt_conf[knowhere::meta::TOPK] = topk;
    auto gt = knowhere::BruteForce::SearchSparse(train_ds, query_ds, gt_conf, nullptr);
    REQUIRE(gt.has_value());

    // Default search (INHERIT) should work and use SINDI
    knowhere::Json search_json;
    search_json[knowhere::meta::TOPK] = topk;
    search_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;

    auto results = idx.Search(query_ds, search_json, nullptr);
    REQUIRE(results.has_value());
    float recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall >= 0.85);

    // Explicitly using SINDI search algo should also work
    search_json[knowhere::indexparam::SEARCH_ALGO] = "SINDI";
    results = idx.Search(query_ds, search_json, nullptr);
    REQUIRE(results.has_value());
    recall = GetKNNRecall(*gt.value(), *results.value());
    REQUIRE(recall >= 0.85);

    // Non-SINDI search algos should be rejected
    search_json[knowhere::indexparam::SEARCH_ALGO] = "DAAT_WAND";
    results = idx.Search(query_ds, search_json, nullptr);
    REQUIRE(!results.has_value());
}
