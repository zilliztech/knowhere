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
std::string kIndexPrefix = kIndexDir + "/minhash";

constexpr uint32_t kNumRows = 100000;
constexpr uint32_t kNumQueries = 10;
constexpr uint32_t kDim = 1000;
constexpr uint32_t kLargeDim = 1536;
constexpr uint32_t kK = 1;
constexpr float kKnnRecall = 0.6;
constexpr float kL2RangeAp = 0.9;
constexpr float kIpRangeAp = 0.9;
constexpr float kCosineRangeAp = 0.9;

template <typename DataType>
void
WriteRawDataToDisk(const std::string data_path, const DataType* raw_data, const uint32_t num, const uint32_t dim) {
    std::ofstream writer(data_path.c_str(), std::ios::binary);
    writer.write((char*)&num, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)raw_data, sizeof(DataType) * num * dim);
    writer.close();
}
}  // namespace
template <typename DataType>
inline void
base_search() {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::MHJACCARD);
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::MHJACCARD, kIndexPrefix},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = 1;
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["aligned_block_size"] = 2048;
        json["band"] = 50;
        json["shared_bloom_filter"] = true;
        json["bloom_false_positive_prob"] = 0.1;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["enable_mmap"] = false;
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        return json;
    };

    auto fp32_query_ds = GenDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    auto fp32_base_ds = GenDataSet(kNumRows, kDim, 30);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_query_ds);

    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();
        std::cout << "use BruteForce to get gt.... " << std::endl;
        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        std::cout << "get gt done" << std::endl;
        knn_gt_ptr = result_knn.value();
    }

    SECTION("Test search") {
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
                                     .Create<DataType>("MinHashIndex", version, minhash_index_index_pack)
                                     .value();
            minhash_index.Build(ds_ptr, json);
            minhash_index.Serialize(binset);
        }
        {
            // knn search
            auto minhash_index = knowhere::IndexFactory::Instance()
                                     .Create<DataType>("MinHashIndex", version, minhash_index_index_pack)
                                     .value();
            minhash_index.Deserialize(binset, deserialize_json);

            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            std::cout << "begin of search" << std::endl;
            auto res = minhash_index.Search(query_ds, knn_json, nullptr);
            std::cout << "end of search" << std::endl;
            REQUIRE(res.has_value());
            std::cout << "compare recall" << std::endl;
            auto knn_recall = GetKNNRecall(*knn_gt_ptr, *res.value());
            REQUIRE(knn_recall > kKnnRecall);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test DiskANNIndexNode.", "[minhash_index]") {
    base_search<knowhere::fp32>();
}
