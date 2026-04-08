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
#include <stdexcept>
#include <string>
#include <thread>

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
    auto inverted_index_codec =
        GENERATE(std::string("block_streamvbyte"), std::string("block_maskedvbyte"), std::string("default"));

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
        for (int64_t i = 0; i < nq; ++i) {
            bool found_valid = false;
            for (int64_t j = 0; j < k; ++j) {
                if (ids[i * k + j] != -1) {
                    found_valid = true;
                    break;
                }
            }
            REQUIRE(found_valid);
        }
    }
}
