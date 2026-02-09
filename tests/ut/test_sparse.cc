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

#include <future>
#include <thread>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
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

    SECTION("Test Search with Bitset") {
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
            while (iter->HasNext()) {
                auto [id, dist] = iter->Next();
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

// ============================================================================
// SparseRow Unit Tests
// ============================================================================

TEST_CASE("SparseRow Basic Operations", "[sparse_utils]") {
    using namespace knowhere::sparse;

    SECTION("Default constructor creates empty row") {
        SparseRow<float> row;
        REQUIRE(row.size() == 0);
        REQUIRE(row.dim() == 0);
        REQUIRE(row.data_byte_size() == 0);
    }

    SECTION("Constructor with count allocates memory") {
        SparseRow<float> row(5);
        REQUIRE(row.size() == 5);
        REQUIRE(row.data() != nullptr);
        REQUIRE(row.data_byte_size() == 5 * SparseRow<float>::element_size());
    }

    SECTION("Constructor from vector of pairs") {
        std::vector<std::pair<table_t, float>> data = {{1, 1.5f}, {3, 2.5f}, {5, 3.5f}};
        SparseRow<float> row(data);
        REQUIRE(row.size() == 3);
        REQUIRE(row[0].id == 1);
        REQUIRE(row[1].id == 3);
        REQUIRE(row[2].id == 5);
        REQUIRE(row.dim() == 6);
    }

    SECTION("Copy constructor performs deep copy") {
        std::vector<std::pair<table_t, float>> data = {{1, 1.0f}, {2, 2.0f}};
        SparseRow<float> original(data);
        SparseRow<float> copy(original);
        REQUIRE(copy.size() == original.size());
        REQUIRE(copy.data() != original.data());
    }

    SECTION("Move constructor transfers ownership") {
        std::vector<std::pair<table_t, float>> data = {{1, 1.0f}};
        SparseRow<float> original(data);
        void* original_data = original.data();
        SparseRow<float> moved(std::move(original));
        REQUIRE(moved.data() == original_data);
    }

    SECTION("set_at modifies elements") {
        SparseRow<float> row(3);
        row.set_at(0, 10, 1.5f);
        row.set_at(1, 20, 2.5f);
        REQUIRE(row[0].id == 10);
        REQUIRE(row[1].id == 20);
    }

    SECTION("set_at throws on out of range") {
        SparseRow<float> row(2);
        REQUIRE_THROWS_AS(row.set_at(5, 1, 1.0f), std::out_of_range);
    }

    SECTION("Dot product operations") {
        std::vector<std::pair<table_t, float>> data1 = {{1, 2.0f}, {2, 3.0f}};
        std::vector<std::pair<table_t, float>> data2 = {{2, 4.0f}, {3, 1.0f}};
        SparseRow<float> row1(data1);
        SparseRow<float> row2(data2);
        float result = row1.dot(row2);
        REQUIRE(result == Catch::Approx(12.0f));  // Only index 2 overlaps: 3*4=12
    }

    SECTION("Swap function") {
        std::vector<std::pair<table_t, float>> data1 = {{1, 1.0f}};
        std::vector<std::pair<table_t, float>> data2 = {{2, 2.0f}, {3, 3.0f}};
        SparseRow<float> row1(data1);
        SparseRow<float> row2(data2);
        swap(row1, row2);
        REQUIRE(row1.size() == 2);
        REQUIRE(row2.size() == 1);
    }
}

TEST_CASE("MaxMinHeap Operations", "[sparse_utils]") {
    using namespace knowhere::sparse;

    SECTION("Basic push and pop") {
        MaxMinHeap<float> heap(3);
        REQUIRE(heap.empty());
        heap.push(1, 10.0f);
        heap.push(2, 20.0f);
        heap.push(3, 30.0f);
        REQUIRE(heap.full());
        REQUIRE(heap.top().val == Catch::Approx(10.0f));
    }

    SECTION("Push larger replaces minimum when full") {
        MaxMinHeap<float> heap(3);
        heap.push(1, 10.0f);
        heap.push(2, 20.0f);
        heap.push(3, 30.0f);
        heap.push(4, 25.0f);  // Should replace 10.0f
        REQUIRE(heap.top().val == Catch::Approx(20.0f));
    }

    SECTION("Push smaller is ignored when full") {
        MaxMinHeap<float> heap(3);
        heap.push(1, 10.0f);
        heap.push(2, 20.0f);
        heap.push(3, 30.0f);
        heap.push(4, 5.0f);  // Should be ignored
        REQUIRE(heap.top().val == Catch::Approx(10.0f));
    }

    SECTION("Pop returns minimum") {
        MaxMinHeap<float> heap(3);
        heap.push(1, 30.0f);
        heap.push(2, 10.0f);
        heap.push(3, 20.0f);
        table_t id = heap.pop();
        REQUIRE(id == 2);
        REQUIRE(heap.size() == 2);
    }
}

TEST_CASE("GrowableVectorView Operations", "[sparse_utils]") {
    using namespace knowhere::sparse;

    SECTION("Basic operations") {
        std::vector<int> storage(5);
        GrowableVectorView<int> view;
        view.initialize(storage.data(), 5 * sizeof(int));
        REQUIRE(view.capacity() == 5);
        REQUIRE(view.size() == 0);

        view.emplace_back(10);
        view.emplace_back(20);
        REQUIRE(view.size() == 2);
        REQUIRE(view[0] == 10);
        REQUIRE(view.at(1) == 20);
    }

    SECTION("Throws on invalid byte_size") {
        std::vector<int> storage(10);
        GrowableVectorView<int> view;
        REQUIRE_THROWS_AS(view.initialize(storage.data(), 7), std::invalid_argument);
    }

    SECTION("Throws on emplace_back when full") {
        std::vector<int> storage(2);
        GrowableVectorView<int> view;
        view.initialize(storage.data(), 2 * sizeof(int));
        view.emplace_back(1);
        view.emplace_back(2);
        REQUIRE_THROWS_AS(view.emplace_back(3), std::out_of_range);
    }

    SECTION("at() bounds checking") {
        std::vector<int> storage(5);
        GrowableVectorView<int> view;
        view.initialize(storage.data(), 5 * sizeof(int));
        view.emplace_back(10);
        REQUIRE_THROWS_AS(view.at(1), std::out_of_range);
    }

    SECTION("Iterator operations") {
        std::vector<int> storage(3);
        GrowableVectorView<int> view;
        view.initialize(storage.data(), 3 * sizeof(int));
        view.emplace_back(1);
        view.emplace_back(2);
        view.emplace_back(3);

        int sum = 0;
        for (int val : view) {
            sum += val;
        }
        REQUIRE(sum == 6);
    }
}

TEST_CASE("DocIdFilterByVector Operations", "[sparse_utils]") {
    using namespace knowhere::sparse;

    SECTION("Empty filter passes all") {
        DocIdFilterByVector filter(std::vector<table_t>{});
        REQUIRE(filter.empty());
        REQUIRE(filter.test(0));
        REQUIRE(filter.test(100));
    }

    SECTION("Filter blocks specified ids") {
        std::vector<table_t> filtered_ids = {2, 5, 8};
        DocIdFilterByVector filter(std::move(filtered_ids));
        REQUIRE(filter.test(0));
        REQUIRE(filter.test(1));
        REQUIRE(!filter.test(2));
        REQUIRE(filter.test(3));
        REQUIRE(!filter.test(5));
        REQUIRE(!filter.test(8));
        REQUIRE(filter.test(10));
    }
}

TEST_CASE("DocValueComputer Functions", "[sparse_utils]") {
    using namespace knowhere::sparse;

    SECTION("Original computer returns value unchanged") {
        auto computer = GetDocValueOriginalComputer<float>();
        REQUIRE(computer(5.0f, 100.0f) == Catch::Approx(5.0f));
    }

    SECTION("BM25 computer applies formula") {
        auto computer = GetDocValueBM25Computer<float>(1.2f, 0.75f, 100.0f);
        float tf = 2.0f, doc_len = 100.0f;
        float expected = tf * 2.2f / (tf + 1.2f * 1.0f);
        REQUIRE(computer(tf, doc_len) == Catch::Approx(expected));
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

    auto inverted_index_algo = GENERATE("TAAT_NAIVE", "DAAT_WAND", "DAAT_MAXSCORE");

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
                REQUIRE(base == expected_id_base);
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
            REQUIRE(res == knowhere::Status::success);
        }
    };

    auto search_task = [&]() {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() <
               test_time) {
            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
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
            task.wait();
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
