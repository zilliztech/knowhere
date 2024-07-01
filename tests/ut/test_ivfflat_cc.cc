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

#include <filesystem>
#include <future>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/invlists/InvertedLists.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"

TEST_CASE("Test Build Search Concurrency", "[Concurrency]") {
    using Catch::Approx;

    auto metric = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::COSINE);
    auto version = GenTestVersionList();

    int64_t nb = 10000, nq = 1000;
    int64_t dim = 128;
    int64_t seed = 42;
    int64_t times = 5;
    int64_t top_k = 100;
    int64_t build_task_num = 1;
    int64_t search_task_num = 10;

    auto base_gen = [&]() {
        knowhere::Json json;
        json[knowhere::meta::DIM] = dim;
        json[knowhere::meta::METRIC_TYPE] = metric;
        json[knowhere::meta::TOPK] = top_k;
        json[knowhere::meta::RADIUS] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 10.0 : 0.99;
        json[knowhere::meta::RANGE_FILTER] = knowhere::IsMetricType(metric, knowhere::metric::L2) ? 0.0 : 1.01;
        return json;
    };

    auto ivf_gen = [&base_gen]() {
        knowhere::Json json = base_gen();
        json[knowhere::indexparam::NLIST] = 128;
        json[knowhere::indexparam::NPROBE] = 16;
        json[knowhere::indexparam::ENSURE_TOPK_FULL] = false;
        return json;
    };

    auto ivf_cc_gen = [&ivf_gen]() {
        knowhere::Json json = ivf_gen();
        json[knowhere::meta::NUM_BUILD_THREAD] = 1;
        json[knowhere::indexparam::SSIZE] = 48;
        return json;
    };

    auto ivf_sq_8_cc_gen = [&ivf_cc_gen]() {
        knowhere::Json json = ivf_cc_gen();
        json[knowhere::indexparam::CODE_SIZE] = 8;
        return json;
    };

    SECTION("Test Concurrent Invlists ") {
        size_t nlist = 128;
        size_t code_size = 512;
        size_t segment_size = 1024;

        auto invList = std::make_unique<faiss::ConcurrentArrayInvertedLists>(nlist, code_size, segment_size, true);

        for (size_t i = 0; i < nlist; i++) {
            REQUIRE(invList->list_size(i) == 0);
        }

        size_t dim = code_size / sizeof(float);
        std::vector<size_t> list_size_count(nlist, 0);
        for (int cnt = 0; cnt < times; cnt++) {
            {
                // small batch append
                std::uniform_int_distribution<int> distribution(0, segment_size);
                for (size_t i = 0; i < nlist; i++) {
                    std::mt19937_64 rng(i);
                    int64_t add_size = distribution(rng);
                    std::vector<faiss::idx_t> ids(add_size, i);
                    float value = i;
                    std::vector<float> codes(add_size * dim, value);
                    std::vector<float> code_normals = knowhere::NormalizeVecs(codes.data(), add_size, dim);
                    std::vector<float> origin_codes(add_size * dim, value);
                    invList->add_entries(i, add_size, ids.data(), reinterpret_cast<uint8_t*>(origin_codes.data()),
                                         code_normals.data());
                    list_size_count[i] += add_size;
                    CHECK(invList->list_size(i) == list_size_count[i]);
                }
            }
            {
                // large batch append
                std::uniform_int_distribution<int> distribution(1, 5 * segment_size);
                for (size_t i = 0; i < nlist; i++) {
                    std::mt19937_64 rng(i * i);
                    int64_t add_size = distribution(rng);
                    std::vector<faiss::idx_t> ids(add_size, i);
                    float value = i;
                    std::vector<float> codes(add_size * dim, value);
                    std::vector<float> code_normals = knowhere::NormalizeVecs(codes.data(), add_size, dim);
                    std::vector<float> origin_codes(add_size * dim, value);
                    invList->add_entries(i, add_size, ids.data(), reinterpret_cast<uint8_t*>(origin_codes.data()),
                                         code_normals.data());
                    list_size_count[i] += add_size;
                    CHECK(invList->list_size(i) == list_size_count[i]);
                }
            }
        }
        {
            for (size_t i = 0; i < nlist; i++) {
                auto list_size = list_size_count[i];
                CHECK(invList->get_segment_num(i) == ((list_size / segment_size) + (list_size % segment_size != 0)));
                CHECK(invList->get_segment_size(i, invList->get_segment_num(i) - 1) ==
                      (list_size % segment_size == 0 ? segment_size : list_size % segment_size));
                CHECK(invList->get_segment_offset(i, 0) == 0);

                for (size_t j = 0; j < list_size; j++) {
                    CHECK(*(invList->get_ids(i, j)) == static_cast<int64_t>(i));
                }

                for (size_t j = 0; j < list_size; j++) {
                    for (size_t k = 0; k < code_size; k += sizeof(float)) {
                        float* float_x = (float*)(invList->get_codes(i, j) + k);
                        CHECK(*float_x == (float)(i));

                        float* normal = (float*)(invList->get_code_norms(i, j));
                        if (i == 0) {
                            CHECK(std::abs(*normal - 1.0) < 0.0000001);
                        } else {
                            CHECK(std::abs(*normal - i * std::sqrt(dim)) < 0.001);
                        }
                    }
                }
            }
        }
    }

    SECTION("Test Add & Search & RangeSearch Serialized ") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivf_cc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ_CC, ivf_sq_8_cc_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        auto train_ds = GenDataSet(nb, dim, seed);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.Type() == name);

        auto& build_ds = train_ds;
        auto query_ds = GenDataSet(nq, dim, seed);

        for (int i = 1; i <= times; i++) {
            idx.Add(build_ds, json);
            {
                auto results = idx.Search(query_ds, json, nullptr);
                REQUIRE(results.has_value());
                auto ids = results.value()->GetIds();
                for (int j = 0; j < nq; ++j) {
                    // duplicate result
                    for (int k = 0; k <= i; k++) {
                        CHECK(ids[j * top_k + k] % nb == j);
                    }
                }
            }
            {
                auto results = idx.RangeSearch(query_ds, json, nullptr);
                REQUIRE(results.has_value());
                auto ids = results.value()->GetIds();
                auto lims = results.value()->GetLims();
                for (int j = 0; j < nq; ++j) {
                    for (int k = 0; k <= i; k++) {
                        CHECK(ids[lims[j] + k] % nb == j);
                    }
                }
            }
        }
    }

    SECTION("Test Build & Search Correctness") {
        using std::make_tuple;
        auto [index_name, cc_index_name] = GENERATE_REF(table<std::string, std::string>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC),
        }));
        auto ivf = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_name, version).value();
        auto ivf_cc = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(cc_index_name, version).value();

        knowhere::Json ivf_json = knowhere::Json::parse(ivf_gen().dump());
        knowhere::Json ivf_cc_json = knowhere::Json::parse(ivf_cc_gen().dump());

        auto train_ds = GenDataSet(nb, dim, seed);
        auto query_ds = GenDataSet(nq, dim, seed);

        auto flat_res = ivf.Build(train_ds, ivf_json);
        REQUIRE(flat_res == knowhere::Status::success);
        auto cc_res = ivf_cc.Build(train_ds, ivf_json);
        REQUIRE(cc_res == knowhere::Status::success);

        // test search
        {
            auto flat_results = ivf.Search(query_ds, ivf_json, nullptr);
            REQUIRE(flat_results.has_value());

            auto cc_results = ivf_cc.Search(query_ds, ivf_json, nullptr);
            REQUIRE(cc_results.has_value());

            auto flat_ids = flat_results.value()->GetIds();
            auto cc_ids = cc_results.value()->GetIds();
            for (int i = 0; i < nq; i++) {
                for (int j = 0; j < top_k; j++) {
                    auto id = i * top_k + j;
                    CHECK(flat_ids[id] == cc_ids[id]);
                }
            }
        }
        // test range_search
        {
            auto flat_results = ivf.RangeSearch(query_ds, ivf_json, nullptr);
            REQUIRE(flat_results.has_value());

            auto cc_results = ivf_cc.RangeSearch(query_ds, ivf_json, nullptr);
            REQUIRE(cc_results.has_value());

            auto flat_ids = flat_results.value()->GetIds();
            auto flat_limits = flat_results.value()->GetLims();
            auto cc_ids = cc_results.value()->GetIds();
            auto cc_limits = cc_results.value()->GetLims();
            for (int i = 0; i < nq; i++) {
                CHECK(flat_limits[i] == cc_limits[i]);
                CHECK(flat_limits[i + 1] == cc_limits[i + 1]);
                for (size_t offset = flat_limits[i]; offset < flat_limits[i + 1]; offset++) {
                    CHECK(flat_ids[offset] == cc_ids[offset]);
                }
            }
        }
    }

    SECTION("Test Add & Search & RangeSearch ConCurrent") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>({
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, ivf_cc_gen),
            make_tuple(knowhere::IndexEnum::INDEX_FAISS_IVFSQ_CC, ivf_sq_8_cc_gen),
        }));
        auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        json[knowhere::indexparam::RAW_DATA_STORE_PREFIX] = std::filesystem::current_path().string() + "/";
        auto train_ds = GenDataSet(nb, dim, seed);
        auto res = idx.Build(train_ds, json);
        REQUIRE(res == knowhere::Status::success);
        REQUIRE(idx.Type() == name);

        auto& build_ds = train_ds;
        std::vector<knowhere::DataSetPtr> search_list;
        std::vector<knowhere::DataSetPtr> range_search_list;
        std::vector<knowhere::DataSetPtr> retrieve_search_list;
        for (int i = 0; i < search_task_num; i++) {
            search_list.push_back(GenDataSet(nq, dim, seed));
            range_search_list.push_back(GenDataSet(nq, dim, seed));
        }

        for (int i = 1; i <= times; i++) {
            std::vector<std::future<knowhere::Status>> add_task_list;
            std::vector<std::future<knowhere::expected<knowhere::DataSetPtr>>> search_task_list;
            std::vector<std::future<knowhere::expected<knowhere::DataSetPtr>>> range_search_task_list;
            std::vector<std::future<knowhere::expected<knowhere::DataSetPtr>>> retrieve_task_list;
            for (int j = 0; j < search_task_num; j++) {
                retrieve_search_list.push_back(GenIdsDataSet(nb * i, nq));
            }
            for (int j = 0; j < build_task_num; j++) {
                add_task_list.push_back(
                    std::async(std::launch::async, [&idx, &build_ds, &json] { return idx.Add(build_ds, json); }));
            }
            for (int j = 0; j < search_task_num; j++) {
                auto& query_set = search_list[j];
                search_task_list.push_back(std::async(
                    std::launch::async, [&idx, &query_set, &json] { return idx.Search(query_set, json, nullptr); }));
            }
            for (int j = 0; j < search_task_num; j++) {
                auto& range_query_set = range_search_list[j];
                range_search_task_list.push_back(std::async(std::launch::async, [&idx, &range_query_set, &json] {
                    return idx.RangeSearch(range_query_set, json, nullptr);
                }));
            }
            for (int j = 0; j < search_task_num; j++) {
                auto& retrieve_ids_set = retrieve_search_list[j];
                retrieve_task_list.push_back(std::async(
                    std::launch::async, [&idx, &retrieve_ids_set] { return idx.GetVectorByIds(retrieve_ids_set); }));
            }

            for (auto& task : add_task_list) {
                REQUIRE(task.get() == knowhere::Status::success);
            }

            for (auto& task : search_task_list) {
                auto results = task.get();
                REQUIRE(results.has_value());
                auto ids = results.value()->GetIds();
                for (int j = 0; j < nq; ++j) {
                    // duplicate result
                    for (int k = 0; k < i; k++) {
                        CHECK(ids[j * top_k + k] % nb == j);
                    }
                }
            }
            for (auto& task : range_search_task_list) {
                auto results = task.get();
                REQUIRE(results.has_value());
                auto ids = results.value()->GetIds();
                auto lims = results.value()->GetLims();
                for (int j = 0; j < nq; ++j) {
                    // duplicate result
                    for (int k = 0; k < i; k++) {
                        CHECK(ids[lims[j] + k] % nb == j);
                    }
                }
            }
            for (size_t nt = 0; nt < retrieve_task_list.size(); nt++) {
                auto& task = retrieve_task_list[nt];
                auto results = task.get();
                REQUIRE(results.has_value());
                auto xb = (float*)build_ds->GetTensor();
                auto res_rows = results.value()->GetRows();
                auto res_dim = results.value()->GetDim();
                auto res_data = (float*)results.value()->GetTensor();
                REQUIRE(res_rows == nq);
                REQUIRE(res_dim == dim);
                for (int i = 0; i < nq; ++i) {
                    const auto id = retrieve_search_list[nt]->GetIds()[i];
                    for (int j = 0; j < dim; ++j) {
                        REQUIRE(res_data[i * dim + j] == xb[id * dim + j]);
                    }
                }
            }
        }
    }
}
