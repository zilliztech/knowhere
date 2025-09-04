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

#include <string>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "utils.h"
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif
#include <fstream>
namespace {
std::string kDir = fs::current_path().string() + "/minhash_index_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kIndexDir = kDir + "/index";
std::string kIndexPrefix = kIndexDir + "/";
std::string input_file = kRawDataPath;

constexpr uint32_t kNumRows = 10000;
constexpr uint32_t kNumQueries = 10;
constexpr uint32_t kHashDim = 1024;
constexpr uint32_t kK = 10;
}  // namespace

TEST_CASE("Test MinHashLSHIndexNode with MinHashLSH hit", "[minhash_lsh_index]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kIndexDir));

    auto metric_str = knowhere::metric::MHJACCARD;
    auto version = GenTestVersionList();
    auto hash_bit = GENERATE(as<uint32_t>{}, 32, 64, 128);
    auto use_mmap = GENERATE(as<bool>{}, true, false);
    auto batch_search_flag = GENERATE(as<bool>{}, true, false);
    auto mh_search_with_jaccard = GENERATE(as<bool>{}, true, false);
    size_t bin_vec_dim = kHashDim * hash_bit;
    auto base_gen = [&metric_str, &hash_bit, &mh_search_with_jaccard, &dim = bin_vec_dim]() {
        knowhere::Json json;
        json["dim"] = dim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        json["refine_k"] = int(kK * 4);
        json["mh_lsh_band"] = 32;
        json["mh_element_bit_width"] = hash_bit;
        json["mh_search_with_jaccard"] = mh_search_with_jaccard;
        return json;
    };

    auto build_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = kIndexDir;
        json["data_path"] = kRawDataPath;
        json["mh_lsh_aligned_block_size"] = 4096;
        json["mh_lsh_shared_bloom_filter"] = true;
        json["mh_lsh_bloom_false_positive_prob"] = 0.01;
        json["with_raw_data"] = true;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &use_mmap, &batch_search_flag]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = kIndexDir;
        json["mh_lsh_batch_search"] = batch_search_flag;
        json["hash_code_in_memory"] = !use_mmap;
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str]() {
        knowhere::Json json = base_gen();
        return json;
    };

    knowhere::DataSetPtr lsh_gt_ptr = nullptr;
    auto base_ds = GenBinDataSet(kNumRows, bin_vec_dim, 22);
    auto query_ds = GenBinDataSet(kNumQueries, bin_vec_dim, 22);
    {
        WriteRawDataToDisk<knowhere::bin1>(kRawDataPath, (const knowhere::bin1*)base_ds->GetTensor(), kNumRows,
                                           bin_vec_dim);
        auto base_json = base_gen();
        auto result_knn = knowhere::BruteForce::Search<knowhere::bin1>(base_ds, query_ds, base_json, nullptr);
        lsh_gt_ptr = result_knn.value();
        float lsh_recall = 0;
        auto res_dis = lsh_gt_ptr->GetDistance();
        for (int64_t i = 0; i < query_ds->GetRows(); i++) {
            lsh_recall += (res_dis[i * kK] == 1.0);
        }
        lsh_recall /= query_ds->GetRows();
        REQUIRE(lsh_recall == 1.0);
    }

    SECTION("Basic Test") {
        std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
        auto minhash_index_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;

        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        // build process
        {
            knowhere::DataSetPtr ds_ptr = nullptr;
            auto minhash_index = knowhere::IndexFactory::Instance()
                                     .Create<knowhere::bin1>("MINHASH_LSH", version, minhash_index_index_pack)
                                     .value();
            REQUIRE(minhash_index.Build(ds_ptr, json) == knowhere::Status::success);
            minhash_index.Serialize(binset);
        }
        SECTION("Test search with jaccard distance") {
            // knn search
            auto minhash_index = knowhere::IndexFactory::Instance()
                                     .Create<knowhere::bin1>("MINHASH_LSH", version, minhash_index_index_pack)
                                     .value();
            minhash_index.Deserialize(binset, deserialize_json);

            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = minhash_index.Search(query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            float recall = GetKNNRecall(*lsh_gt_ptr, *res.value());
            REQUIRE(recall == 1.0);
            if (!mh_search_with_jaccard) {
                float lsh_recall = 0;
                auto res_dis = res.value()->GetDistance();
                for (int64_t i = 0; i < query_ds->GetRows(); i++) {
                    lsh_recall += (res_dis[i * kK] == 1.0);
                }
                lsh_recall /= query_ds->GetRows();
                REQUIRE(lsh_recall == 1.0);
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}
