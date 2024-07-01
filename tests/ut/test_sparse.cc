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

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

TEST_CASE("Test Mem Sparse Index With Float Vector", "[float metrics]") {
    auto [nb, dim, doc_sparsity, query_sparsity] = GENERATE(table<int32_t, int32_t, float, float>({
        // 300 dim, avg doc nnz 12, avg query nnz 9
        {2000, 300, 0.95, 0.97},
        // 300 dim, avg doc nnz 9, avg query nnz 3
        {2000, 300, 0.97, 0.99},
        // 3000 dim, avg doc nnz 90, avg query nnz 30
        {20000, 3000, 0.97, 0.99},
    }));
    auto topk = 5;
    int64_t nq = GENERATE(10, 100);

    auto [drop_ratio_build, drop_ratio_search] = GENERATE(table<float, float>({
        {0.0, 0.0},
        {0.0, 0.15},
        {0.15, 0.3},
    }));

    auto metric = knowhere::metric::IP;
    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = topk;
        return json;
    };

    auto sparse_inverted_index_gen = [base_gen, drop_ratio_build = drop_ratio_build,
                                      drop_ratio_search = drop_ratio_search]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_BUILD] = drop_ratio_build;
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        return json;
    };

    const auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);
    // it is possible the query has more dims than the train dataset.
    const auto query_ds = GenSparseDataSet(nq, dim + 20, query_sparsity);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
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
            auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
            auto cfg_json = gen().dump();
            CAPTURE(name, cfg_json);
            knowhere::Json json = knowhere::Json::parse(cfg_json);
            REQUIRE(idx.Type() == name);
            REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
            REQUIRE(idx.Size() > 0);
            REQUIRE(idx.Count() == nb);

            knowhere::BinarySet bs;
            REQUIRE(idx.Serialize(bs) == knowhere::Status::success);
            if (use_mmap) {
                auto binary = bs.GetByName(idx.Type());
                auto data = binary->data.get();
                auto size = binary->size;
                // if tmp_file already exists, remove it
                std::remove(tmp_file);
                std::ofstream out(tmp_file, std::ios::binary);
                out.write((const char*)data, size);
                out.close();
                REQUIRE(idx.DeserializeFromFile(tmp_file, json) == knowhere::Status::success);
            } else {
                REQUIRE(idx.Deserialize(bs, json) == knowhere::Status::success);
            }

            auto results = idx.Search(query_ds, json, nullptr);
            REQUIRE(results.has_value());
            float recall = GetKNNRecall(*gt.value(), *results.value());
            check_distance_decreasing(*results.value());
            auto drop_ratio_build = json[knowhere::indexparam::DROP_RATIO_BUILD].get<float>();
            auto drop_ratio_search = json[knowhere::indexparam::DROP_RATIO_SEARCH].get<float>();
            if (drop_ratio_build == 0 && drop_ratio_search == 0) {
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
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
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

        auto drop_ratio_build = json[knowhere::indexparam::DROP_RATIO_BUILD].get<float>();
        auto drop_ratio_search = json[knowhere::indexparam::DROP_RATIO_SEARCH].get<float>();
        if (drop_ratio_build == 0 && drop_ratio_search == 0) {
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
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
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
        // verify the distances are monotonic decreasing, as INDEX_SPARSE_INVERTED_INDEX and INDEX_SPARSE_WAND
        // performs exausitive search for iterator.
        for (int i = 0; i < nq; ++i) {
            auto& iter = iterators[i];
            float prev_dist = std::numeric_limits<float>::max();
            while (iter->HasNext()) {
                auto [id, dist] = iter->Next();
                REQUIRE(!bitset.test(id));
                REQUIRE(prev_dist >= dist);
                prev_dist = dist;
            }
        }
    }

    SECTION("Test Sparse Range Search") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
        }));

        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(idx.Type() == name);
        REQUIRE(idx.Build(train_ds, json) == knowhere::Status::success);
        REQUIRE(idx.Size() > 0);
        REQUIRE(idx.Count() == nb);

        auto [radius, range_filter] = GENERATE(table<float, float>({
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
        // most above 0.95, only a few between 0.9 and 0.85
        REQUIRE(actual_count * 1.0f / gt_count >= 0.85);
    }
}

TEST_CASE("Test Mem Sparse Index GetVectorByIds", "[float metrics]") {
    auto [nb, dim, doc_sparsity, query_sparsity] = GENERATE(table<int32_t, int32_t, float, float>({
        // 300 dim, avg doc nnz 12, avg query nnz 9
        {2000, 300, 0.95, 0.97},
        // 300 dim, avg doc nnz 9, avg query nnz 3
        {2000, 300, 0.97, 0.99},
        // 3000 dim, avg doc nnz 90, avg query nnz 30
        {20000, 3000, 0.97, 0.99},
    }));
    int64_t nq = GENERATE(10, 100);

    auto [drop_ratio_build, drop_ratio_search] = GENERATE(table<float, float>({
        {0.0, 0.0},
        {0.32, 0.0},
    }));

    auto metric = knowhere::metric::IP;
    auto version = GenTestVersionList();

    auto base_gen = [=, dim = dim]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = 1;
        return json;
    };

    auto sparse_inverted_index_gen = [base_gen, drop_ratio_build = drop_ratio_build,
                                      drop_ratio_search = drop_ratio_search]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::DROP_RATIO_BUILD] = drop_ratio_build;
        json[knowhere::indexparam::DROP_RATIO_SEARCH] = drop_ratio_search;
        return json;
    };

    const auto train_ds = GenSparseDataSet(nb, dim, doc_sparsity);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, 1},
    };

    SECTION("Test GetVectorByIds") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, sparse_inverted_index_gen),
            make_tuple(knowhere::IndexEnum::INDEX_SPARSE_WAND, sparse_inverted_index_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);

        auto ids_ds = GenIdsDataSet(nb, nq);
        REQUIRE(idx.Type() == name);
        auto res = idx.Build(train_ds, json);
        if (!idx.HasRawData(metric)) {
            return;
        }
        REQUIRE(res == knowhere::Status::success);
        knowhere::BinarySet bs;
        idx.Serialize(bs);

        auto idx_new = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        idx_new.Deserialize(bs);

        auto retrieve_task = [&]() {
            auto results = idx_new.GetVectorByIds(ids_ds);
            REQUIRE(results.has_value());
            auto xb = (knowhere::sparse::SparseRow<float>*)train_ds->GetTensor();
            auto res_data = (knowhere::sparse::SparseRow<float>*)results.value()->GetTensor();
            for (int i = 0; i < nq; ++i) {
                const auto id = ids_ds->GetIds()[i];
                const auto& truth_row = xb[id];
                const auto& res_row = res_data[i];
                REQUIRE(truth_row.size() == res_row.size());
                for (size_t j = 0; j < truth_row.size(); ++j) {
                    REQUIRE(truth_row[j] == res_row[j]);
                }
            }
        };

        std::vector<std::future<void>> retrieve_task_list;
        for (int i = 0; i < 20; i++) {
            retrieve_task_list.push_back(std::async(std::launch::async, [&] { return retrieve_task(); }));
        }
        for (auto& task : retrieve_task_list) {
            task.wait();
        }
    }
}
