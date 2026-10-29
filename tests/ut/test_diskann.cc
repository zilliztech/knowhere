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

#include <sys/resource.h>

#include <atomic>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>

#include "../DiskANN/include/diskann/defaults.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "diskann/aux_utils.h"
#include "diskann/diskann_gpu.h"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/partition_and_pq.h"
#include "diskann/pq_navigation.h"
#include "diskann/ssd_graph_index.h"
#include "diskann/utils.h"
#include "filemanager/FileManager.h"
#include "filemanager/impl/LocalFileManager.h"
#include "index/diskann/diskann_config.h"
#include "index/diskann/pq_navigation_config.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/expected.h"
#include "knowhere/feder/DiskANN.h"
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
#include <stdexcept>

namespace {
#ifdef KNOWHERE_WITH_CARDINAL
constexpr const char* kNativeDiskANNType = "DISKANN_DEPRECATED";
#else
constexpr const char* kNativeDiskANNType = "DISKANN";
#endif
std::string kDir = fs::current_path().string() + "/diskann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kIPIndexDir = kDir + "/ip_index";
std::string kCOSINEIndexDir = kDir + "/cosine_index";
std::string kL2IndexPrefix = kL2IndexDir + "/l2";
std::string kIPIndexPrefix = kIPIndexDir + "/ip";
std::string kCOSINEIndexPrefix = kCOSINEIndexDir + "/cosine";

std::string kEmbListL2IndexDir = kDir + "/emb_list_l2_index";
std::string kEmbListIPIndexDir = kDir + "/emb_list_ip_index";
std::string kEmbListCOSINEIndexDir = kDir + "/emb_list_cosine_index";
std::string kEmbListL2IndexPrefix = kEmbListL2IndexDir + "/max_sim_l2";
std::string kEmbListIPIndexPrefix = kEmbListIPIndexDir + "/max_sim_ip";
std::string kEmbListCOSINEIndexPrefix = kEmbListCOSINEIndexDir + "/max_sim_cosine";
std::string kEmbListOffsetPath = kDir + "/emb_list_offset.bin";

constexpr uint32_t kNumRows = 1000;
constexpr uint32_t kNumQueries = 10;
constexpr uint32_t kDim = 128;
constexpr uint32_t kLargeDim = 256;
constexpr uint32_t kK = 10;
#ifdef KNOWHERE_WITH_CUVS
// This test is expected to run on a GPU with at least 2 GB of RAM
constexpr uint32_t defaultMaxDegree = 64;
constexpr uint32_t defaultSearchListSize = 64;
#else
constexpr uint32_t defaultMaxDegree = 56;
constexpr uint32_t defaultSearchListSize = 36;
#endif
constexpr float kKnnRecall = 0.9;
constexpr float kEmbListKnnRecall = 0.75;
constexpr float AiSAQKnnRecall = 0.01;
constexpr float kL2RangeAp = 0.9;
constexpr float kIpRangeAp = 0.9;
constexpr float kCosineRangeAp = 0.9;
}  // namespace
TEST_CASE("Valid diskann build params test", "[diskann]") {
    int rows_num = 1000000;
    auto version = GenTestVersionList();

    auto ratio = GENERATE(as<float>{}, 0.01, 0.1, 0.125);

    float pq_code_budget_gb = sizeof(float) * kDim * rows_num * 0.125 / (1024 * 1024 * 1024);
    float search_cache_budget_gb = sizeof(float) * kDim * rows_num * 0.05 / (1024 * 1024 * 1024);

    auto test_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = "L2";
        json["k"] = 100;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 24;
        json["search_list_size"] = 64;
        json["vec_field_size_gb"] = 1.0;
        json["pq_code_budget_gb_ratio"] = ratio;
        json["pq_code_budget_gb"] = pq_code_budget_gb;
        json["build_dram_budget_gb"] = 32.0;
        json["search_cache_budget_gb_ratio"] = ratio;
        json["search_cache_budget_gb"] = search_cache_budget_gb;
        json["beamwidth"] = 8;
        json["min_k"] = 10;
        json["max_k"] = 8000;
        return json;
    };

    SECTION("Dynamic param check") {
        knowhere::Json test_json = test_gen();

        auto cfg = knowhere::IndexStaticFaced<float>::CreateConfig(kNativeDiskANNType, version);
        knowhere::Json json_(test_json);
        std::string msg;
        auto res = knowhere::Config::FormatAndCheck(*cfg, json_, &msg);
        REQUIRE(res == knowhere::Status::success);
        res = knowhere::Config::Load(*cfg, json_, knowhere::PARAM_TYPE::TRAIN, &msg);
        REQUIRE(res == knowhere::Status::success);

        const auto& diskCfg = static_cast<const knowhere::DiskANNConfig&>(*cfg);
        REQUIRE(static_cast<const knowhere::PQNavigationConfig&>(*diskCfg.navigation).pq_code_budget_gb ==
                std::max(pq_code_budget_gb, 1.0f * ratio));
        REQUIRE(diskCfg.search_cache_budget_gb == std::max(search_cache_budget_gb, 1.0f * ratio));
    }
}

TEST_CASE("Invalid diskann params test", "[diskann]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    int rows_num = kNumRows;
    auto version = GenTestVersionList();
    auto test_gen = [rows_num]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = "L2";
        json["k"] = 100;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 32;
        json["search_list_size"] = 64;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * rows_num * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        json["search_cache_budget_gb"] = sizeof(float) * kDim * rows_num * 0.05 / (1024 * 1024 * 1024);
        json["beamwidth"] = 8;
        json["min_k"] = 10;
        json["max_k"] = 8000;
        return json;
    };
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto base_ds = GenDataSet(rows_num, kDim, 30);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, rows_num, kDim);
    // build process
    SECTION("Invalid build params test") {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        knowhere::Json test_json;
        knowhere::Status test_stat;
        // invalid metric type
        test_json = test_gen();
        test_json["metric_type"] = knowhere::metric::JACCARD;
        test_stat = diskann.Build(ds_ptr, test_json);
        REQUIRE(test_stat == knowhere::Status::invalid_metric_type);
        // raw data path not exist
        test_json = test_gen();
        test_json["data_path"] = kL2IndexPrefix + ".temp";
        test_stat = diskann.Build(ds_ptr, test_json);
        REQUIRE(test_stat == knowhere::Status::disk_file_error);
    }

    SECTION("Invalid search params test") {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto binarySet = knowhere::BinarySet();
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        knowhere::Json test_json = test_gen();
        diskann.Build(ds_ptr, test_json);
        diskann.Serialize(binarySet);
        diskann.Deserialize(binarySet, test_gen());

        auto query_ds = GenDataSet(kNumQueries, kDim, 42);

#ifndef KNOWHERE_WITH_CARDINAL
        // search list size < topk
        {
            test_json = test_gen();
            test_json["search_list_size"] = 1;
            auto res = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE_FALSE(res.has_value());
            REQUIRE(res.error() == knowhere::Status::out_of_range_in_json);
        }
#endif
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

template <typename DataType>
inline void
base_search() {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    auto version = GenTestVersionList();
    int max_degree = 56;
    int knn_search_list_size = 36;

#ifdef KNOWHERE_WITH_CUVS
    if (is_gpu_available()) {
        max_degree = defaultMaxDegree;
        knn_search_list_size = defaultSearchListSize;
    }
#endif

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map, max_degree]() {
        knowhere::Json json = base_gen();
        json["quant_type"] = "PQ";
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = max_degree;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map, knn_search_list_size]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = knn_search_list_size;
        json["beamwidth"] = 8;
        return json;
    };

    auto range_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["beamwidth"] = 8;
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
        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<DataType>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }

    SECTION("Test search and range search") {
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;

        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        // build process
        {
            knowhere::DataSetPtr ds_ptr = nullptr;
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
            diskann.Build(ds_ptr, json);
            diskann.Serialize(binset);
        }
        {
            // knn search
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
            diskann.Deserialize(binset, deserialize_json);
            REQUIRE(diskann.HasRawData(metric_str) ==
                    knowhere::IndexStaticFaced<DataType>::HasRawData("DISKANN", version, json));

            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            auto knn_recall = GetKNNRecall(*knn_gt_ptr, *res.value());
            CAPTURE(knn_json.dump());
            REQUIRE(knn_recall > kKnnRecall);
            // knn search without cache file
            {
                std::string cached_nodes_file_path =
                    std::string(build_gen()["index_prefix"]) + std::string("_cached_nodes.bin");
                if (fs::exists(cached_nodes_file_path)) {
                    fs::remove(cached_nodes_file_path);
                }
                auto diskann_tmp =
                    knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
                diskann_tmp.Deserialize(binset, deserialize_json);
                auto knn_search_json = knn_search_gen().dump();
                knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
                auto res = diskann_tmp.Search(query_ds, knn_json, nullptr);
                REQUIRE(res.has_value());
                REQUIRE(GetKNNRecall(*knn_gt_ptr, *res.value()) >= kKnnRecall);
            }
            // knn search with bitset
            std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
                GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
            const auto bitset_percentages = {0.4f, 0.98f};
            const auto bitset_thresholds = {-1.0f, 0.9f};
            for (const float threshold : bitset_thresholds) {
                knn_json["filter_threshold"] = threshold;
                for (const float percentage : bitset_percentages) {
                    for (const auto& gen_func : gen_bitset_funcs) {
                        auto bitset_data = gen_func(kNumRows, percentage * kNumRows);
                        knowhere::BitsetView bitset(bitset_data.data(), kNumRows);
                        auto results = diskann.Search(query_ds, knn_json, bitset);
                        auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, knn_json, bitset);
                        float recall = GetKNNRecall(*gt.value(), *results.value());
                        if (percentage == 0.98f) {
                            REQUIRE(recall >= 0.9f);
                        } else {
                            REQUIRE(recall >= kKnnRecall);
                        }
                    }
                }
            }

            // A fully filtered search returns before doing any I/O. Verify that
            // the early return still restores the scratch slot to thread_data;
            // cal_size() includes the number of currently available slots.
            auto fully_filtered_query = knowhere::ConvertToDataTypeIfNeeded<DataType>(GenDataSet(1, kDim, 43));
            auto fully_filtered_data = GenerateBitsetWithFirstTbitsSet(kNumRows, kNumRows);
            knowhere::BitsetView fully_filtered_bitset(fully_filtered_data.data(), kNumRows);
            const auto size_before_fully_filtered_search = diskann.Size();
            auto fully_filtered_results = diskann.Search(fully_filtered_query, knn_json, fully_filtered_bitset);
            REQUIRE(fully_filtered_results.has_value());
            for (uint32_t i = 0; i < kK; ++i) {
                REQUIRE(fully_filtered_results.value()->GetIds()[i] == -1);
            }
            REQUIRE(diskann.Size() == size_before_fully_filtered_search);

            // range search process
            auto range_search_json = range_search_gen().dump();
            knowhere::Json range_json = knowhere::Json::parse(range_search_json);
            auto range_search_res = diskann.RangeSearch(query_ds, range_json, nullptr);
            REQUIRE(range_search_res.has_value());
            auto ap = GetRangeSearchRecall(*range_search_gt_ptr, *range_search_res.value());
            float standard_ap = metric_range_ap_map[metric_str];
            REQUIRE(ap > standard_ap);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test DiskANNIndexNode.", "[diskann]") {
    base_search<knowhere::fp32>();
}

// At high filter ratios the DiskANN iterator ranks the unfiltered points by navigation distance and refines them from
//   disk instead of walking the graph, so it must return every unfiltered point. Results are exact within a refine
//   window and ordered by navigation distance across windows, so the top-k must match brute force when all
//   unfiltered points fit in one window.
template <typename DataType>
inline void
iterator_brute_force_search(const std::string& quant_type = "PQ") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP, knowhere::metric::COSINE);
    auto version = GenTestVersionList();
    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };

    knowhere::Json json;
    json["dim"] = kDim;
    json["metric_type"] = metric_str;
    json["k"] = kK;
    json["index_prefix"] = metric_dir_map[metric_str];
    json["data_path"] = kRawDataPath;
    json["max_degree"] = defaultMaxDegree;
    json["search_list_size"] = 128;
    json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
    json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
    json["build_dram_budget_gb"] = 32.0;
    json["quant_type"] = quant_type;
    CAPTURE(metric_str, quant_type);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(GenDataSet(kNumRows, kDim, 30));
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(GenDataSet(kNumQueries, kDim, 42));
    WriteRawDataToDisk<DataType>(kRawDataPath, static_cast<const DataType*>(base_ds->GetTensor()), kNumRows, kDim);

    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    knowhere::BinarySet binset;
    {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
        REQUIRE(diskann.Build(ds_ptr, json) == knowhere::Status::success);
        REQUIRE(diskann.Serialize(binset) == knowhere::Status::success);
    }
    auto diskann = knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
    REQUIRE(diskann.Deserialize(binset, json) == knowhere::Status::success);

    // (filter_threshold, filtered ratio). -1 uses the iterator default (97%); 0.5 checks that the search param
    //   overrides it, and leaves enough unfiltered points to span several refine windows.
    constexpr size_t kSearchListSize = 16;
    // lsearch * kBruteForceTopkRefineExpansionFactor
    constexpr size_t kRefineWindow = kSearchListSize * 2;
    const std::vector<std::pair<float, float>> cases = {{-1.0f, 0.98f}, {-1.0f, 0.99f}, {0.5f, 0.6f}};
    std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
        GenerateBitsetWithFirstTbitsSet, GenerateBitsetWithRandomTbitsSet};
    for (const auto& [threshold, percentage] : cases) {
        for (const auto& gen_func : gen_bitset_funcs) {
            json["filter_threshold"] = threshold;
            json["search_list_size"] = kSearchListSize;
            CAPTURE(threshold, percentage);
            const size_t n_filtered = percentage * kNumRows;
            const size_t n_unfiltered = kNumRows - n_filtered;
            auto bitset_data = gen_func(kNumRows, n_filtered);
            knowhere::BitsetView bitset(bitset_data.data(), kNumRows);

            auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, json, bitset);
            REQUIRE(gt.has_value());
            const auto gt_ids = gt.value()->GetIds();
            const auto gt_dists = gt.value()->GetDistance();
            const size_t expected_k = std::min<size_t>(kK, n_unfiltered);

            auto its = diskann.AnnIterator(query_ds, json, bitset);
            REQUIRE(its.has_value());
            size_t n_hit = 0;
            for (uint32_t i = 0; i < kNumQueries; ++i) {
                auto& it = its.value()[i];
                std::unordered_set<int64_t> gt_topk(gt_ids + i * kK, gt_ids + i * kK + expected_k);
                std::unordered_set<int64_t> returned;
                while (it->HasNext().value()) {
                    auto [id, dist] = it->Next().value();
                    REQUIRE(!bitset.test(id));
                    REQUIRE(returned.insert(id).second);
                    if (returned.size() == 1 && id == gt_ids[i * kK]) {
                        REQUIRE(dist == Catch::Approx(gt_dists[i * kK]).epsilon(0.01));
                    }
                    if (returned.size() <= expected_k && gt_topk.count(id) > 0) {
                        n_hit++;
                    }
                }
                REQUIRE(returned.size() == n_unfiltered);
            }
            if (n_unfiltered <= kRefineWindow) {
                const float recall = static_cast<float>(n_hit) / (kNumQueries * expected_k);
                REQUIRE(recall >= 0.98f);
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test DiskANN iterator falls back to brute force at high filter ratio", "[diskann]") {
    SECTION("fp32") {
        iterator_brute_force_search<knowhere::fp32>();
    }
    SECTION("fp32 RaBitQ navigation") {
        iterator_brute_force_search<knowhere::fp32>("RABITQ");
    }
    SECTION("fp16") {
        iterator_brute_force_search<knowhere::fp16>();
    }
    SECTION("bf16") {
        iterator_brute_force_search<knowhere::bf16>();
    }
}

TEST_CASE("Test DiskANN CalcDistByIDs with all vectors cached", "[diskann]") {
    fs::remove_all(kDir);
    REQUIRE_NOTHROW(fs::create_directories(kCOSINEIndexDir));

#ifdef KNOWHERE_WITH_CARDINAL
    // This case verifies native DiskANN's cached-node file contract.
    // Cardinal-backed DISKANN keeps its cache inside Cardinal storage instead.
    constexpr const char* diskann_type = "DISKANN_DEPRECATED";
#else
    constexpr const char* diskann_type = "DISKANN";
#endif

    auto version = GenTestVersionList();
    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(GenDataSet(kNumRows, kDim, 30));
    auto base_ptr = static_cast<const knowhere::fp16*>(base_ds->GetTensor());
    WriteRawDataToDisk<knowhere::fp16>(kRawDataPath, base_ptr, kNumRows, kDim);

    knowhere::Json build_json;
    build_json["dim"] = kDim;
    build_json["metric_type"] = knowhere::metric::COSINE;
    build_json["index_prefix"] = kCOSINEIndexPrefix;
    build_json["data_path"] = kRawDataPath;
    build_json["max_degree"] = 56;
    build_json["search_list_size"] = 128;
    build_json["pq_code_budget_gb"] = sizeof(knowhere::fp16) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
    build_json["search_cache_budget_gb"] = sizeof(knowhere::fp16) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
    build_json["build_dram_budget_gb"] = 32.0;

    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto diskann =
        knowhere::IndexFactory::Instance().Create<knowhere::fp16>(diskann_type, version, diskann_index_pack).value();
    REQUIRE(diskann.Build(nullptr, build_json) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(diskann.Serialize(binset) == knowhere::Status::success);

    knowhere::Json deserialize_json;
    deserialize_json["dim"] = kDim;
    deserialize_json["metric_type"] = knowhere::metric::COSINE;
    deserialize_json["index_prefix"] = kCOSINEIndexPrefix;
    deserialize_json["search_cache_budget_gb"] = build_json["search_cache_budget_gb"];

    auto loaded_diskann =
        knowhere::IndexFactory::Instance().Create<knowhere::fp16>(diskann_type, version, diskann_index_pack).value();
    REQUIRE(loaded_diskann.Deserialize(binset, deserialize_json) == knowhere::Status::success);

    std::unique_ptr<uint32_t[]> cached_ids;
    size_t num_cached_ids = 0;
    size_t cached_id_dim = 0;
    diskann::load_bin<uint32_t>(diskann::get_cached_nodes_file(kCOSINEIndexPrefix), cached_ids, num_cached_ids,
                                cached_id_dim);
    REQUIRE(num_cached_ids > 0);
    REQUIRE(cached_id_dim == 1);

    const int64_t cached_id = cached_ids[0];
    auto query_ds = knowhere::GenDataSet(1, kDim, base_ptr + cached_id * kDim);
    auto result = loaded_diskann.CalcDistByIDs(query_ds, nullptr, &cached_id, 1, true);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDistance()[0] == Catch::Approx(1.0f).margin(0.001f));

    fs::remove_all(kDir);
}

#ifdef KNOWHERE_WITH_CUVS
TEST_CASE("Test DiskANN Build Index", "[diskann]") {
    auto version = GenTestVersionList();
    knowhere::Status test_stat;
    if (is_gpu_available()) {
        for (const uint32_t dim : {kDim}) {
            fs::remove_all(kDir);
            fs::remove(kDir);
            REQUIRE_NOTHROW(fs::create_directories(kL2IndexDir));

            auto base_gen = [=] {
                knowhere::Json json;
                json["dim"] = dim;
                json["metric_type"] = knowhere::metric::L2;
                json["k"] = kK;
                return json;
            };

            auto build_gen = [=]() {
                knowhere::Json json = base_gen();
                json["index_prefix"] = kL2IndexPrefix;
                json["data_path"] = kRawDataPath;
                json["max_degree"] = 64;
                json["search_list_size"] = 128;
                json["pq_code_budget_gb"] = sizeof(float) * dim * kNumRows * 0.03125 / (1024 * 1024 * 1024);
                json["build_dram_budget_gb"] = 0.4;
                return json;
            };

            auto base_ds = GenDataSet(kNumRows, dim, 30);
            auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
            WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, dim);

            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto diskann_index_pack = knowhere::Pack(file_manager);

            knowhere::DataSetPtr ds_ptr = nullptr;
            auto diskann = knowhere::IndexFactory::Instance()
                               .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                               .value();
            auto build_json = build_gen().dump();
            knowhere::Json json = knowhere::Json::parse(build_json);
            diskann.Build(ds_ptr, json);

            SECTION("Invalid build params for GPU test") {
                json["max_degree"] = 56;
                json["index_prefix"] = kL2IndexDir + "/l2a";
                test_stat = diskann.Build(ds_ptr, json);
                REQUIRE(test_stat == knowhere::Status::success);
                json["max_degree"] = 64;
                json["search_list_size"] = kK;
                json["index_prefix"] = kL2IndexDir + "/l2b";
                test_stat = diskann.Build(ds_ptr, json);
                REQUIRE(test_stat == knowhere::Status::success);
                json["max_degree"] = 64;
                json["search_list_size"] = 100;
                json["index_prefix"] = kL2IndexDir + "/l2c";
                test_stat = diskann.Build(ds_ptr, json);
                REQUIRE(test_stat == knowhere::Status::success);
            }
        }
        fs::remove_all(kDir);
        fs::remove(kDir);
    }
}
#endif

template <typename DataType>
inline void
emb_list_search() {
    auto version = GenTestEmbListVersionList();
    int max_degree = 56;
    int knn_search_list_size = 36;

#ifdef KNOWHERE_WITH_CUVS
    if (is_gpu_available()) {
        max_degree = defaultMaxDegree;
        knn_search_list_size = defaultSearchListSize;
    }
#endif

    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kEmbListL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kEmbListIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kEmbListCOSINEIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::MAX_SIM_COSINE, knowhere::metric::MAX_SIM_IP,
                               knowhere::metric::MAX_SIM_L2);

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::MAX_SIM_L2, kEmbListL2IndexPrefix},
        {knowhere::metric::MAX_SIM_IP, kEmbListIPIndexPrefix},
        {knowhere::metric::MAX_SIM_COSINE, kEmbListCOSINEIndexPrefix},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map, max_degree]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["emb_list_offset_file_path"] = kEmbListOffsetPath;
        json["max_degree"] = max_degree;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map, knn_search_list_size]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = knn_search_list_size;
        json["beamwidth"] = 8;
        json["retrieval_ann_ratio"] = 3.0f;
        return json;
    };

    int each_el_len = 10;
    int num_el = int((kNumRows + each_el_len - 1) / each_el_len);
    auto fp32_query_ds = GenQueryEmbListDataSet(kNumQueries, kDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    auto fp32_base_ds = GenEmbListDataSet(kNumRows, kDim, 42, each_el_len);
    auto emb_list_offset = GenEmbListOffset(kNumRows, each_el_len);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_query_ds);

    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kDim);
        WriteEmbListOffsetToDisk(kEmbListOffsetPath, emb_list_offset.data(), emb_list_offset.size());
        // generate the gt of knn search
        auto base_json = base_gen();
        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
    }

    SECTION("Test EmbList knn search") {
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;

        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        // build process
        {
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
            diskann.Build(nullptr, json);
            diskann.Serialize(binset);
        }
        {
            // knn search
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
            diskann.Deserialize(binset, deserialize_json);
            REQUIRE(diskann.HasRawData(metric_str) ==
                    knowhere::IndexStaticFaced<DataType>::HasRawData("DISKANN", version, json));

            auto knn_search_json = knn_search_gen().dump();
            knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
            auto res = diskann.Search(query_ds, knn_json, nullptr);
            REQUIRE(res.has_value());
            auto knn_recall = GetKNNRecall(*knn_gt_ptr, *res.value());
            CAPTURE(knn_json.dump());
            REQUIRE(knn_recall > kEmbListKnnRecall);
            // knn search without cache file
            {
                std::string cached_nodes_file_path =
                    std::string(build_gen()["index_prefix"]) + std::string("_cached_nodes.bin");
                if (fs::exists(cached_nodes_file_path)) {
                    fs::remove(cached_nodes_file_path);
                }
                auto diskann_tmp =
                    knowhere::IndexFactory::Instance().Create<DataType>("DISKANN", version, diskann_index_pack).value();
                diskann_tmp.Deserialize(binset, deserialize_json);
                auto knn_search_json = knn_search_gen().dump();
                knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
                auto res = diskann_tmp.Search(query_ds, knn_json, nullptr);
                REQUIRE(res.has_value());
                REQUIRE(GetKNNRecall(*knn_gt_ptr, *res.value()) >= kEmbListKnnRecall);
            }
            // knn search with bitset
            std::vector<std::function<std::vector<uint8_t>(size_t, float, size_t)>> gen_bitset_funcs = {
                GenerateBitsetByPartition};
            const auto bitset_percentages = {0.1f, 0.5f, 0.9f, 0.98f};
            for (const float percentage : bitset_percentages) {
                for (const auto& gen_func : gen_bitset_funcs) {
                    auto bitset_data = gen_func(num_el, 1 - percentage, 1);
                    knowhere::BitsetView bitset(bitset_data.data(), num_el);
                    auto results = diskann.Search(query_ds, knn_json, bitset);
                    auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, knn_json, bitset);
                    float recall = GetKNNRecall(*gt.value(), *results.value());
                    REQUIRE(recall >= kEmbListKnnRecall);
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

#ifndef KNOWHERE_WITH_CARDINAL
TEST_CASE("Test DISKANN for EmbList", "[diskann]") {
    emb_list_search<knowhere::fp32>();
}
#endif  // !KNOWHERE_WITH_CARDINAL

// This test case only check L2
TEST_CASE("Test DiskANN GetVectorByIds", "[diskann]") {
    auto version = GenTestVersionList();
    for (const uint32_t dim : {kDim, kLargeDim}) {
        fs::remove_all(kDir);
        fs::remove(kDir);
        REQUIRE_NOTHROW(fs::create_directories(kL2IndexDir));

        auto base_gen = [=] {
            knowhere::Json json;
            json["dim"] = dim;
            json["metric_type"] = knowhere::metric::L2;
            json["k"] = kK;
            return json;
        };

        auto build_gen = [=]() {
            knowhere::Json json = base_gen();
            json["index_prefix"] = kL2IndexPrefix;
            json["data_path"] = kRawDataPath;
            json["max_degree"] = 32;
            json["search_list_size"] = kK;
            json["pq_code_budget_gb"] = sizeof(float) * dim * kNumRows * 0.03125 / (1024 * 1024 * 1024);
            json["build_dram_budget_gb"] = 32.0;
            return json;
        };

        auto query_ds = GenDataSet(kNumQueries, dim, 42);
        auto base_ds = GenDataSet(kNumRows, dim, 30);
        auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
        WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, dim);

        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);

        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("DISKANN", version, diskann_index_pack).value();
        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        diskann.Build(ds_ptr, json);
        knowhere::BinarySet binset;
        diskann.Serialize(binset);
        {
            std::vector<double> cache_sizes = {0, 1.0f * sizeof(float) * dim * kNumRows * 0.125 / (1024 * 1024 * 1024)};
            for (const auto cache_size : cache_sizes) {
                auto deserialize_gen = [&base_gen, cache = cache_size]() {
                    knowhere::Json json = base_gen();
                    json["index_prefix"] = kL2IndexPrefix;
                    json["search_cache_budget_gb"] = cache;
                    return json;
                };
                knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                auto index = knowhere::IndexFactory::Instance()
                                 .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                                 .value();
                auto ret = index.Deserialize(binset, deserialize_json);
                REQUIRE(ret == knowhere::Status::success);

                REQUIRE(diskann.HasRawData(knowhere::metric::L2) ==
                        knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData("DISKANN", version, json));

                if (!diskann.HasRawData(knowhere::metric::L2)) {
                    continue;
                }

                std::vector<double> ids_sizes = {1, kNumRows * 0.2, kNumRows * 0.7, kNumRows};
                for (const auto ids_size : ids_sizes) {
                    LOG_KNOWHERE_DEBUG_ << "Testing dim = " << dim << ", cache_size = " << cache_size
                                        << ", ids_size = " << ids_size;
                    auto ids_ds = GenIdsDataSet(ids_size, ids_size);
                    auto results = index.GetVectorByIds(ids_ds);
                    REQUIRE(results.has_value());
                    auto xb = (float*)base_ds->GetTensor();
                    auto data = (float*)results.value()->GetTensor();
                    for (size_t i = 0; i < ids_size; ++i) {
                        auto id = ids_ds->GetIds()[i];
                        for (size_t j = 0; j < dim; ++j) {
                            REQUIRE(data[i * dim + j] == xb[id * dim + j]);
                        }
                    }
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test_AiSAQ_dynamic_cache", "[diskann]") {
    std::string index_type = "AISAQ";
    constexpr uint32_t kNumRowsTest = 10000;
    constexpr uint32_t kNumQueriesTest = 10;
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));
    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::COSINE, knowhere::metric::IP, knowhere::metric::L2);
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 64;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRowsTest * 0.125 * 0.5 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = 0;
        json["build_dram_budget_gb"] = 16.0;
        json["rearrange"] = true;
        json["inline_pq"] = 0;
        json["pq_cache_size"] = 0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = 0;
        json["pq_cache_size"] = 0;
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = defaultSearchListSize;
        json["beamwidth"] = 4;
        json["vectors_beamwidth"] = 4;
        return json;
    };

    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto fp32_base_ds = GenDataSet(kNumRowsTest, kLargeDim, 30);
    auto fp32_query_ds = GenDataSet(kNumQueriesTest, kLargeDim, 42);

    knowhere::BinarySet binset;
    auto build_json = build_gen().dump();
    knowhere::Json json = knowhere::Json::parse(build_json);
    knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp32>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp32>(fp32_query_ds);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;

    {
        auto base_ptr = static_cast<const knowhere::fp32*>(base_ds->GetTensor());
        WriteRawDataToDisk<knowhere::fp32>(kRawDataPath, base_ptr, kNumRowsTest, kDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();

        auto result_knn = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<knowhere::fp32>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }
    // build process
    {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();

        diskann.Build(ds_ptr, json);
        diskann.Serialize(binset);
    }
    {
        auto knn_search_json = knn_search_gen().dump();
        knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Deserialize(binset, deserialize_json);

        knn_json["pq_read_page_cache_size"] = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto results = diskann.Search(query_ds, knn_json, nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "*********************** First Run time: " << duration.count() << " ms"
                  << " read_page_cache_size: 0 KiB" << std::endl;
        uint32_t read_page_cache_size_limit = /*diskann::defaults::MAX_PQ_READ_PAGE_CACHE_SIZE*/ 204800;
        for (uint32_t read_page_cache_size = 0; read_page_cache_size < read_page_cache_size_limit;
             read_page_cache_size += 4096) {
            knn_json["pq_read_page_cache_size"] = read_page_cache_size;
            auto start = std::chrono::high_resolution_clock::now();
            auto loop_results = diskann.Search(query_ds, knn_json, nullptr);
            REQUIRE(loop_results.has_value());
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "*********************** Run time: " << duration.count() << " ms"
                      << " read_page_cache_size: " << read_page_cache_size / 1024 << " KiB" << std::endl;
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}
// This test case only check L2
TEST_CASE("Test_AiSAQ_GetVectorByIds", "[diskann]") {
    auto version = GenTestVersionList();
    for (const uint32_t dim : {kDim, kLargeDim}) {
        fs::remove_all(kDir);
        fs::remove(kDir);
        REQUIRE_NOTHROW(fs::create_directories(kL2IndexDir));

        auto base_gen = [&] {
            knowhere::Json json;
            json[knowhere::meta::RETRIEVE_FRIENDLY] = true;
            json["dim"] = dim;
            json["metric_type"] = knowhere::metric::L2;
            json["k"] = kK;
            return json;
        };

        auto build_gen = [&]() {
            knowhere::Json json = base_gen();
            json["index_prefix"] = kL2IndexPrefix;
            json["data_path"] = kRawDataPath;
            json["max_degree"] = 32;
            json["search_list_size"] = kK;
            json["pq_code_budget_gb"] = sizeof(float) * dim * kNumRows * 0.03125 / (1024 * 1024 * 1024);
            json["build_dram_budget_gb"] = 32.0;
            json["rearrange"] = false;
            return json;
        };

        auto query_ds = GenDataSet(kNumQueries, dim, 42);
        auto base_ds = GenDataSet(kNumRows, dim, 30);
        auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
        WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, dim);

        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);

        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>("AISAQ", version, diskann_index_pack).value();
        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        diskann.Build(ds_ptr, json);
        knowhere::BinarySet binset;
        diskann.Serialize(binset);
        {
            std::vector<double> cache_sizes = {0};
            for (const auto cache_size : cache_sizes) {
                auto deserialize_gen = [&base_gen, cache = cache_size]() {
                    knowhere::Json json = base_gen();
                    json["index_prefix"] = kL2IndexPrefix;
                    json["search_cache_budget_gb"] = cache;
                    json["pq_cache_size"] = 1024;
                    json["pq_read_page_cache_size"] = 0;
                    return json;
                };
                knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                auto index = knowhere::IndexFactory::Instance()
                                 .Create<knowhere::fp32>("AISAQ", version, diskann_index_pack)
                                 .value();
                auto ret = index.Deserialize(binset, deserialize_json);
                REQUIRE(ret == knowhere::Status::success);

                REQUIRE(diskann.HasRawData(knowhere::metric::L2) ==
                        knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData("AISAQ", version, json));

                std::vector<double> ids_sizes = {1, kNumRows * 0.2, kNumRows * 0.7, kNumRows};
                for (const auto ids_size : ids_sizes) {
                    LOG_KNOWHERE_DEBUG_ << "Testing dim = " << dim << ", cache_size = " << cache_size
                                        << ", ids_size = " << ids_size;
                    auto ids_ds = GenIdsDataSet(ids_size, ids_size);
                    auto results = index.GetVectorByIds(ids_ds);
                    REQUIRE(results.has_value());
                    auto xb = (float*)base_ds->GetTensor();
                    auto data = (float*)results.value()->GetTensor();
                    for (size_t i = 0; i < ids_size; ++i) {
                        auto id = ids_ds->GetIds()[i];
                        for (size_t j = 0; j < dim; ++j) {
                            REQUIRE(data[i * dim + j] == xb[id * dim + j]);
                        }
                    }
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

template <typename DataType>
inline void
base_AiSAQ_param_test() {
    std::string index_type = "AISAQ";
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = knowhere::metric::L2;
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 64;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 * 0.5 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 16.0;
        json["disk_pq_dims"] = 0;
        json["accelerate_build"] = false;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["warm_up"] = true;
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = defaultSearchListSize;
        json["beamwidth"] = 4;
        json["vectors_beamwidth"] = 4;
        return json;
    };

    auto knn_search_without_beamwidth_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_list_size"] = 36;
        json["vectors_beamwidth"] = 4;
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

        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<DataType>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }

    SECTION("Test search and range search") {
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);
        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
        knowhere::BinarySet binset;
        auto build_json = build_gen().dump();
        knowhere::Json json = knowhere::Json::parse(build_json);
        LOG_KNOWHERE_DEBUG_ << "========== build_json =================";
        LOG_KNOWHERE_DEBUG_ << json.dump();
        LOG_KNOWHERE_DEBUG_ << "===========================";
        LOG_KNOWHERE_DEBUG_ << "========== deserialize_json =================";
        LOG_KNOWHERE_DEBUG_ << deserialize_json.dump();
        LOG_KNOWHERE_DEBUG_ << "===========================";
        knowhere::Status test_stat;
        knowhere::Json test_json;
        // build process
        {
            knowhere::DataSetPtr ds_ptr = GenDataSet(kNumRows, kDim, 30);
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>(index_type, version, diskann_index_pack).value();
            LOG_KNOWHERE_INFO_ << "Test invalid metric_type parameter";
            diskann.Dim();
            diskann.Size();
            diskann.Count();
            test_json = build_gen();
            test_json["metric_type"] = knowhere::metric::JACCARD;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::invalid_metric_type);
            // raw data path not exist
            LOG_KNOWHERE_INFO_ << "Test data path not exist";
            test_json = build_gen();
            test_json.erase("data_path");
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::invalid_param_in_json);
            test_json = build_gen();
            test_json["data_path"] = kL2IndexPrefix + ".temp";
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::disk_file_error);
            LOG_KNOWHERE_INFO_ << "Test pq_cache_size parameter value more than maximum";
            test_json = build_gen();
            test_json["pq_cache_size"] = diskann::defaults::MAX_PQ_CACHE_SIZE + 1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test pq_cache_size parameter value less than minimum";
            test_json = build_gen();
            test_json["pq_cache_size"] = -1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test num_entry_points parameter value more than maximum";
            test_json = build_gen();
            test_json["num_entry_points"] = diskann::defaults::MAX_NUM_ENTRY_POINTS + 1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test num_entry_points parameter value less than minimum";
            test_json = build_gen();
            test_json["num_entry_points"] = -1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test inline_pq parameter value less than minimum";
            test_json = build_gen();
            test_json["inline_pq"] = -2;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test disk_pq_dims parameter value less than minimum";
            test_json = build_gen();
            test_json["disk_pq_dims"] = -2;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);
            LOG_KNOWHERE_INFO_ << "Test disk_pq_dims parameter value more than maximum";
            test_json = build_gen();
            test_json["disk_pq_dims"] = kDim + 2;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);
            LOG_KNOWHERE_INFO_ << "Test inline_pq parameter value more than max_degree";
            test_json = build_gen();
            test_json["inline_pq"] = 68;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);
            LOG_KNOWHERE_INFO_ << "Test max_degree parameter value more than maximum";
            test_json = build_gen();
            test_json["max_degree"] = diskann::defaults::MAX_AISAQ_MAX_DEGREE + 1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);
            test_json = build_gen();
            test_json["search_list_size"] = diskann::defaults::MAX_AISAQ_SEARCH_LIST_SIZE + 1;
            test_stat = diskann.Build(ds_ptr, test_json);
            REQUIRE(test_stat == knowhere::Status::aisaq_error);

            test_stat = diskann.Train(ds_ptr, json);
            REQUIRE(test_stat == knowhere::Status::not_implemented);
            test_stat = diskann.Add(ds_ptr, json);
            REQUIRE(test_stat == knowhere::Status::not_implemented);
            test_stat = diskann.DeserializeFromFile("DeserializeFile", json);
            REQUIRE(test_stat == knowhere::Status::not_implemented);
            diskann.Build(ds_ptr, json);
            diskann.Serialize(binset);
            diskann.Dim();
            diskann.Size();
            diskann.Count();
            diskann.Type();
        }
        {
            // knn search
            auto diskann =
                knowhere::IndexFactory::Instance().Create<DataType>(index_type, version, diskann_index_pack).value();
            deserialize_json.erase("index_prefix");
            test_stat = diskann.Deserialize(binset, deserialize_json);
            REQUIRE(test_stat == knowhere::Status::invalid_param_in_json);
            test_json = knn_search_gen();
            auto search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::empty_index);
            deserialize_json["index_prefix"] = metric_dir_map[metric_str];
            deserialize_json["metric_type"] = knowhere::metric::JACCARD;
            test_stat = diskann.Deserialize(binset, deserialize_json);
            REQUIRE(test_stat == knowhere::Status::invalid_metric_type);
            deserialize_json["metric_type"] = knowhere::metric::L2;
            diskann.Deserialize(binset, deserialize_json);
            diskann.Deserialize(binset, deserialize_json);
            REQUIRE(diskann.HasRawData(metric_str) ==
                    knowhere::IndexStaticFaced<DataType>::HasRawData(index_type, version, json));

            LOG_KNOWHERE_INFO_ << "Test default beamwidth";
            test_json = knn_search_without_beamwidth_gen();
            test_json["vectors_beamwidth"] = 4;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::aisaq_error);
            LOG_KNOWHERE_INFO_ << "Test vectors_beamwidth parameter value less than minimum";
            test_json = knn_search_gen();
            test_json["vectors_beamwidth"] = 0;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test pq_read_page_cache_size parameter value less than minimum";
            test_json = knn_search_gen();
            test_json["pq_read_page_cache_size"] = -1;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test pq_read_page_cache_size parameter value more than maximum";
            test_json = knn_search_gen();
            test_json["pq_read_page_cache_size"] = diskann::defaults::MAX_PQ_READ_PAGE_CACHE_SIZE + 1024;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test beamwidth parameter value less than minimum";
            test_json = knn_search_gen();
            test_json["beamwidth"] = 0;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test beamwidth parameter value more than maximum";
            test_json = knn_search_gen();
            test_json["beamwidth"] = diskann::defaults::MAX_AISAQ_BEAMWIDTH + 1;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test vectors_beamwidth parameter value more than maximum "
                               << diskann::defaults::MAX_AISAQ_VECTORS_BEAMWIDTH;
            test_json = knn_search_gen();
            test_json["vectors_beamwidth"] = diskann::defaults::MAX_AISAQ_VECTORS_BEAMWIDTH + 1;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::out_of_range_in_json);
            LOG_KNOWHERE_INFO_ << "Test feder parameter";
            test_json = knn_search_gen();
            test_json["trace_visit"] = true;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            auto one_query_ds = GenDataSet(1, kLargeDim, 42);
            auto test_query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(one_query_ds);
            test_json = knn_search_gen();
            test_json["trace_visit"] = true;
            search_stat = diskann.Search(test_query_ds, test_json, nullptr);
            LOG_KNOWHERE_INFO_ << "Test invalid metric parameter for search operation";
            test_json = knn_search_gen();
            test_json["metric_type"] = knowhere::metric::JACCARD;
            search_stat = diskann.Search(query_ds, test_json, nullptr);
            REQUIRE(search_stat.error() == knowhere::Status::invalid_metric_type);
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test_AiSAQ_Params", "[diskann]") {
    base_AiSAQ_param_test<knowhere::fp32>();
}

int64_t
check_node_size(std::string metric_type, int64_t dim, int64_t pq_bytes, int64_t pq_cache_size,
                int64_t num_entry_points) {
    int64_t size = 0;
    if (metric_type == "COSINE")
        size += dim * 4;
    if (num_entry_points > 0)
        size += num_entry_points * pq_bytes;
    if (pq_cache_size > 0)
        size += pq_cache_size;
    return size;
}

template <typename DataType>
inline void
base_AiSAQ_search() {
    std::string index_type = "AISAQ";
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::COSINE, knowhere::metric::IP, knowhere::metric::L2);
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kLargeDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 64;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kLargeDim * kNumRows * 0.5 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = 0;
        json["disk_pq_dims"] = kLargeDim;
        json["build_dram_budget_gb"] = 16.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["search_cache_budget_gb"] = 0;
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["beamwidth"] = 4;
        json["vectors_beamwidth"] = 1;
        return json;
    };

    auto fp32_query_ds = GenDataSet(kNumQueries, kLargeDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    auto fp32_base_ds = GenDataSet(kNumRows, kLargeDim, 30);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_query_ds);

    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kLargeDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();

        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<DataType>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }
    const auto rearrange_list = {true, false};
    const auto inline_pq_list = {64, -1, 0};
    const auto pq_cache_size_list = {0, 4096};
    const auto num_entry_points_list = {0, 100};
    const auto bitset_percentages = {0.05f, 0.20f, 0.40f, 0.60f, 0.80f, 0.9f};
    const auto bitset_thresholds = {0.9f};
    const auto l_search_dict = {128};
    const auto pq_read_page_cache_size_list = {0, 1048576};
    SECTION("Test search and range search") {
        for (const bool rearrange : rearrange_list) {
            for (const int inline_pq : inline_pq_list) {
                for (const int pq_cache_size : pq_cache_size_list) {
                    for (const int num_entry_points : num_entry_points_list) {
                        std::shared_ptr<milvus::FileManager> file_manager =
                            std::make_shared<milvus::LocalFileManager>();
                        auto diskann_index_pack = knowhere::Pack(file_manager);
                        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                        knowhere::BinarySet binset;
                        auto build_json = build_gen().dump();
                        knowhere::Json json = knowhere::Json::parse(build_json);
                        json["rearrange"] = rearrange;
                        json["inline_pq"] = inline_pq;
                        json["pq_cache_size"] = pq_cache_size;
                        json["num_entry_points"] = num_entry_points;
                        deserialize_json["rearrange"] = rearrange;
                        deserialize_json["pq_cache_size"] = pq_cache_size;
                        deserialize_json["num_entry_points"] = num_entry_points;
                        LOG_KNOWHERE_DEBUG_ << "========== build_json =================";
                        LOG_KNOWHERE_DEBUG_ << json.dump();
                        LOG_KNOWHERE_DEBUG_ << "===========================";
                        LOG_KNOWHERE_DEBUG_ << "========== deserialize_json =================";
                        LOG_KNOWHERE_DEBUG_ << deserialize_json.dump();
                        LOG_KNOWHERE_DEBUG_ << "===========================";

                        // build process
                        knowhere::DataSetPtr ds_ptr = nullptr;
                        auto diskann = knowhere::IndexFactory::Instance()
                                           .Create<DataType>(index_type, version, diskann_index_pack)
                                           .value();

                        diskann.Build(ds_ptr, json);
                        diskann.Serialize(binset);
                        // knn search;
                        diskann.Deserialize(binset, deserialize_json);

                        REQUIRE(diskann.HasRawData(metric_str) ==
                                knowhere::IndexStaticFaced<DataType>::HasRawData(index_type, version, json));

                        auto knn_search_json = knn_search_gen().dump();
                        knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);

                        // knn search with bitset
                        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
                            GenerateBitsetWithRandomTbitsSet};
                        for (const int pq_read_page_cache_size : pq_read_page_cache_size_list) {
                            for (const int l_search : l_search_dict) {
                                knn_json["search_list_size"] = l_search;
                                knn_json["pq_read_page_cache_size"] = pq_read_page_cache_size;
                                std::cout << " Test parameters: "
                                          << "metric type: " << metric_str << " rearrange: " << rearrange
                                          << " node cache size: " << deserialize_json["search_cache_budget_gb"] << " Gb"
                                          << " inline_pq: " << inline_pq << " pq_cache_size: " << pq_cache_size
                                          << " pq_read_page_cache_size: " << pq_read_page_cache_size
                                          << " num_entry_points: " << num_entry_points << " vectors_beamwidth: 4"
                                          << " search_list_size: " << l_search << std::endl;
                                for (const float threshold : bitset_thresholds) {
                                    knn_json["filter_threshold"] = threshold;
                                    for (const float percentage : bitset_percentages) {
                                        for (const auto& gen_func : gen_bitset_funcs) {
                                            auto bitset_data = gen_func(kNumRows, percentage * kNumRows);
                                            knowhere::BitsetView bitset(bitset_data.data(), kNumRows);
                                            auto results = diskann.Search(query_ds, knn_json, bitset);
                                            auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds,
                                                                                             knn_json, bitset);
                                            float recall = GetKNNRecall(*gt.value(), *results.value());
                                            std::cout << " Test the map is " << percentage * 100
                                                      << "% full -  Recall :" << recall * 100
                                                      << "%; threshold: " << threshold << " l_search: " << l_search
                                                      << std::endl;
                                            if (percentage == 0.98f) {
                                                REQUIRE(recall >= 0.9f);
                                            } else {
                                                REQUIRE(recall >= AiSAQKnnRecall);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        LOG_KNOWHERE_DEBUG_ << "========== Index clean =================";
                        fs::remove_all(kL2IndexDir);
                        REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
                        fs::remove_all(kIPIndexDir);
                        REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
                        fs::remove_all(kCOSINEIndexDir);
                        REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));
                    }
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test_AiSAQ_IndexNode", "[diskann]") {
    base_AiSAQ_search<knowhere::fp32>();
}

template <typename DataType>
inline void
cached_node_AiSAQ_search() {
    std::string index_type = "AISAQ";
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));

    auto metric_str = knowhere::metric::L2;
    auto version = GenTestVersionList();

    std::unordered_map<knowhere::MetricType, std::string> metric_dir_map = {
        {knowhere::metric::L2, kL2IndexPrefix},
        {knowhere::metric::IP, kIPIndexPrefix},
        {knowhere::metric::COSINE, kCOSINEIndexPrefix},
    };
    std::unordered_map<knowhere::MetricType, float> metric_range_ap_map = {
        {knowhere::metric::L2, kL2RangeAp},
        {knowhere::metric::IP, kIpRangeAp},
        {knowhere::metric::COSINE, kCosineRangeAp},
    };

    auto base_gen = [&metric_str]() {
        knowhere::Json json;
        json["dim"] = kLargeDim;
        json["metric_type"] = metric_str;
        json["k"] = kK;
        if (metric_str == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else if (metric_str == knowhere::metric::IP) {
            json["radius"] = CFG_FLOAT::value_type(350000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        } else {
            json["radius"] = 0.75f;
            json["range_filter"] = 1.0f;
        }
        return json;
    };

    auto build_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 64;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kLargeDim * kNumRows * 0.5 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kLargeDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["disk_pq_dims"] = kLargeDim;
        json["build_dram_budget_gb"] = 16.0;
        return json;
    };

    auto deserialize_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["use_bfs_cache"] = true;
        json["search_cache_budget_gb"] = sizeof(float) * kLargeDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        return json;
    };

    auto knn_search_gen = [&base_gen, &metric_str, &metric_dir_map]() {
        knowhere::Json json = base_gen();
        json["index_prefix"] = metric_dir_map[metric_str];
        json["beamwidth"] = 8;
        json["vectors_beamwidth"] = 1;
        return json;
    };

    auto fp32_query_ds = GenDataSet(kNumQueries, kLargeDim, 42);
    knowhere::DataSetPtr knn_gt_ptr = nullptr;
    knowhere::DataSetPtr range_search_gt_ptr = nullptr;
    auto fp32_base_ds = GenDataSet(kNumRows, kLargeDim, 30);

    auto base_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_base_ds);
    auto query_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(fp32_query_ds);

    {
        auto base_ptr = static_cast<const DataType*>(base_ds->GetTensor());
        WriteRawDataToDisk<DataType>(kRawDataPath, base_ptr, kNumRows, kLargeDim);

        // generate the gt of knn search and range search
        auto base_json = base_gen();

        auto result_knn = knowhere::BruteForce::Search<DataType>(base_ds, query_ds, base_json, nullptr);
        knn_gt_ptr = result_knn.value();
        auto result_range = knowhere::BruteForce::RangeSearch<DataType>(base_ds, query_ds, base_json, nullptr);
        range_search_gt_ptr = result_range.value();
    }
    const auto rearrange_list = {false, true};
    const auto inline_pq_list = {64, 10, 0};
    const auto pq_cache_size_list = {0};
    const auto num_entry_points_list = {0};
    const auto bitset_percentages = {0.80f};
    const auto bitset_thresholds = {0.9f};
    const auto l_search_dict = {128};
    const auto pq_read_page_cache_size_list = {0};
    SECTION("Test search and range search") {
        for (const bool rearrange : rearrange_list) {
            for (const int inline_pq : inline_pq_list) {
                for (const int pq_cache_size : pq_cache_size_list) {
                    for (const int num_entry_points : num_entry_points_list) {
                        std::shared_ptr<milvus::FileManager> file_manager =
                            std::make_shared<milvus::LocalFileManager>();
                        auto diskann_index_pack = knowhere::Pack(file_manager);
                        knowhere::Json deserialize_json = knowhere::Json::parse(deserialize_gen().dump());
                        knowhere::BinarySet binset;
                        auto build_json = build_gen().dump();
                        knowhere::Json json = knowhere::Json::parse(build_json);
                        json["rearrange"] = rearrange;
                        json["inline_pq"] = inline_pq;
                        json["pq_cache_size"] = pq_cache_size;
                        json["num_entry_points"] = num_entry_points;
                        LOG_KNOWHERE_DEBUG_ << "========== build_json =================";
                        LOG_KNOWHERE_DEBUG_ << json.dump();
                        LOG_KNOWHERE_DEBUG_ << "===========================";
                        LOG_KNOWHERE_DEBUG_ << "========== deserialize_json =================";
                        LOG_KNOWHERE_DEBUG_ << deserialize_json.dump();
                        LOG_KNOWHERE_DEBUG_ << "===========================";

                        // build process
                        knowhere::DataSetPtr ds_ptr = nullptr;
                        auto diskann = knowhere::IndexFactory::Instance()
                                           .Create<DataType>(index_type, version, diskann_index_pack)
                                           .value();

                        diskann.Build(ds_ptr, json);
                        diskann.Serialize(binset);
                        // knn search;
                        if (rearrange)
                            deserialize_json["use_bfs_cache"] = true;
                        else
                            deserialize_json["use_bfs_cache"] = false;
                        diskann.Deserialize(binset, deserialize_json);
                        diskann.Size();
                        REQUIRE(diskann.HasRawData(metric_str) ==
                                knowhere::IndexStaticFaced<DataType>::HasRawData(index_type, version, json));

                        auto knn_search_json = knn_search_gen().dump();
                        knowhere::Json knn_json = knowhere::Json::parse(knn_search_json);

                        // knn search with bitset
                        std::vector<std::function<std::vector<uint8_t>(size_t, size_t)>> gen_bitset_funcs = {
                            GenerateBitsetWithRandomTbitsSet};
                        for (const int pq_read_page_cache_size : pq_read_page_cache_size_list) {
                            for (const int l_search : l_search_dict) {
                                knn_json["search_list_size"] = l_search;
                                knn_json["pq_read_page_cache_size"] = pq_read_page_cache_size;
                                std::cout << " Test parameters: "
                                          << "metric type: " << metric_str << " rearrange: " << rearrange
                                          << " node cache size: " << deserialize_json["search_cache_budget_gb"] << " Gb"
                                          << " inline_pq: " << inline_pq << " pq_cache_size: " << pq_cache_size
                                          << " pq_read_page_cache_size: " << pq_read_page_cache_size
                                          << " num_entry_points: " << num_entry_points << " vectors_beamwidth: 4"
                                          << " search_list_size: " << l_search << std::endl;
                                for (const float threshold : bitset_thresholds) {
                                    knn_json["filter_threshold"] = threshold;
                                    for (const float percentage : bitset_percentages) {
                                        for (const auto& gen_func : gen_bitset_funcs) {
                                            auto bitset_data = gen_func(kNumRows, percentage * kNumRows);
                                            knowhere::BitsetView bitset(bitset_data.data(), kNumRows);
                                            auto results = diskann.Search(query_ds, knn_json, bitset);
                                            auto gt = knowhere::BruteForce::Search<DataType>(base_ds, query_ds,
                                                                                             knn_json, bitset);
                                            float recall = GetKNNRecall(*gt.value(), *results.value());
                                            std::cout << " Test the map is " << percentage * 100
                                                      << "% full -  Recall :" << recall * 100
                                                      << "%; threshold: " << threshold << " l_search: " << l_search
                                                      << std::endl;
                                            if (percentage == 0.98f) {
                                                REQUIRE(recall >= 0.9f);
                                            } else {
                                                REQUIRE(recall >= AiSAQKnnRecall);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    LOG_KNOWHERE_DEBUG_ << "========== Index clean =================";
                    fs::remove_all(kL2IndexDir);
                    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));
                    fs::remove_all(kIPIndexDir);
                    REQUIRE_NOTHROW(fs::create_directory(kIPIndexDir));
                    fs::remove_all(kCOSINEIndexDir);
                    REQUIRE_NOTHROW(fs::create_directory(kCOSINEIndexDir));
                }
            }
        }
    }
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("Test_AiSAQ_CachedNode", "[diskann]") {
    cached_node_AiSAQ_search<knowhere::fp32>();
}

TEST_CASE("Test DiskANN Search Cancellation", "[diskann][cancellation]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kL2IndexDir));

    auto version = GenTestVersionList();
    auto index_type = GENERATE(as<std::string>{}, "DISKANN", "AISAQ");

    auto build_gen = [&index_type]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 32;
        json["search_list_size"] = 128;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        if (index_type == "AISAQ") {
            json["rearrange"] = true;
            json["inline_pq"] = 0;
            json["pq_cache_size"] = 0;
            json["num_entry_points"] = 100;
        }
        return json;
    };

    auto deserialize_gen = [&index_type]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kL2IndexPrefix;
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        if (index_type == "AISAQ") {
            json["pq_cache_size"] = 0;
        }
        return json;
    };

    auto search_gen = [&index_type]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kL2IndexPrefix;
        json["search_list_size"] = defaultSearchListSize;
        json["beamwidth"] = 8;
        if (index_type == "AISAQ") {
            json["vectors_beamwidth"] = 4;
        }
        return json;
    };

    // Prepare data and build index
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto query_ds = GenDataSet(kNumQueries, kDim, 42);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    knowhere::BinarySet binset;

    // Build index
    {
        knowhere::DataSetPtr ds_ptr = nullptr;
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Build(ds_ptr, build_gen());
        diskann.Serialize(binset);
    }

    SECTION("Test Search with cancellation from another thread") {
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Deserialize(binset, deserialize_gen());

        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());

        std::atomic<bool> search_started{false};
        std::atomic<bool> search_finished{false};

        std::thread search_thread([&]() {
            search_started = true;
            auto results = diskann.Search(query_ds, search_gen(), nullptr, &op_context);
            search_finished = true;
            (void)results;
        });

        while (!search_started) {
            std::this_thread::yield();
        }

        cs.requestCancellation();
        search_thread.join();

        REQUIRE(search_finished);
    }

    SECTION("Test Search with pre-cancelled context should return error") {
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Deserialize(binset, deserialize_gen());

        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());

        cs.requestCancellation();
        bool is_cancelled = op_context.cancellation_token.isCancellationRequested();
        REQUIRE(is_cancelled);

        auto results = diskann.Search(query_ds, search_gen(), nullptr, &op_context);
        REQUIRE(!results.has_value());
        // A cancellation the caller asked for is reported as such, not as a
        // DiskANN or AiSAQ engine failure. Cardinal still reports its own
        // inner error until the matching change lands there.
        REQUIRE((results.error() == knowhere::Status::cancelled ||
                 results.error() == knowhere::Status::cardinal_inner_error));
    }

    SECTION("Test Search without cancellation should succeed") {
        auto diskann =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack).value();
        diskann.Deserialize(binset, deserialize_gen());

        // Search without OpContext should succeed
        auto results = diskann.Search(query_ds, search_gen(), nullptr);
        REQUIRE(results.has_value());

        // Search with non-cancelled OpContext should also succeed∏
        folly::CancellationSource cs;
        milvus::OpContext op_context(cs.getToken());
        auto results2 = diskann.Search(query_ds, search_gen(), nullptr, &op_context);
        REQUIRE(results2.has_value());
    }

    fs::remove_all(kDir);
    fs::remove(kDir);
}

namespace {
class PQNavigationTestDirectory {
 public:
    explicit PQNavigationTestDirectory(const std::string& name) : path(kDir + "/" + name) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~PQNavigationTestDirectory() {
        fs::remove_all(path);
    }
    fs::path path;
};

// Generate the existing PQ format through the navigation-aware build pipeline.
// Compatibility with the unchanged main-branch writer is checked separately.
class NativePQFixture : public PQNavigationTestDirectory {
 public:
    explicit NativePQFixture(const std::string& name) : PQNavigationTestDirectory(name) {
        constexpr size_t rows = 512, dim = 8, nq = 4;
        std::mt19937 rng(20261014);
        std::vector<float> base(rows * dim), queries(nq * dim);
        for (auto& value : base) value = float(rng() & 65535) / 32768.0f - 1.0f;
        for (size_t q = 0; q < nq; ++q) {
            std::copy_n(base.data() + (q * 113 + 7) * dim, dim, queries.data() + q * dim);
        }
        diskann::save_bin<float>((path / "base.bin").string(), base.data(), rows, dim);
        diskann::save_bin<float>((path / "queries.bin").string(), queries.data(), nq, dim);
        diskann::BuildConfig config;
        config.data_file_path = (path / "base.bin").string();
        config.index_file_path = (path / "index").string();
        config.compare_metric = diskann::Metric::L2;
        config.max_degree = 16;
        config.search_list_size = 64;
        diskann::PQNavigationStore navigation(true, false, {double(rows * 2) / (1024 * 1024 * 1024)});
        config.index_mem_gb = 1;
        REQUIRE(diskann::build_disk_index<float>(config, navigation) == 0);
    }
};

class LegacyPQFileManager : public milvus::LocalFileManager {
 public:
    explicit LegacyPQFileManager(fs::path source) : source_(std::move(source)) {
    }
    bool
    LoadFile(const std::string& filename) override {
        loaded.push_back(filename);
        if (fs::path(filename).filename() == missing_file) {
            return false;
        }
        const auto source = source_ / fs::path(filename).filename();
        if (!fs::exists(source)) {
            return false;
        }
        fs::copy_file(source, filename, fs::copy_options::overwrite_existing);
        return true;
    }
    std::optional<bool>
    IsExisted(const std::string& filename) override {
        if (fail_exists)
            return std::nullopt;
        return fs::exists(source_ / fs::path(filename).filename());
    }
    bool fail_exists = false;
    std::string missing_file;
    std::vector<std::string> loaded;

 private:
    fs::path source_;
};
}  // namespace

TEST_CASE("DiskANN SSD configuration does not select a navigation codec", "[diskann][navigation]") {
    knowhere::DiskANNSSDConfig config;
    knowhere::Json request = {{"metric_type", "L2"},
                              {"dim", 8},
                              {"index_prefix", "unused_index"},
                              {"data_path", "unused_data"},
                              {"vec_field_size_gb", 2.0},
                              {"search_cache_budget_gb", 0.1},
                              {"search_cache_budget_gb_ratio", 0.2},
                              {"quant_type", "UNKNOWN"},
                              {"rbq_bits", 0}};
    REQUIRE(knowhere::Config::Load(config, request, knowhere::PARAM_TYPE::TRAIN) == knowhere::Status::success);
    REQUIRE(config.search_list_size == 128);
    REQUIRE(config.search_cache_budget_gb.value() == Catch::Approx(0.4));
    for (const auto* field : {"quant_type", "pq_code_budget_gb", "rbq_bits"}) {
        REQUIRE(config.__DICT__.count(field) == 0);
    }
    knowhere::DiskANNConfig native;
    REQUIRE(knowhere::Config::Load(native, request, knowhere::PARAM_TYPE::TRAIN) == knowhere::Status::invalid_args);
}

TEST_CASE("DiskANN navigation codec selection", "[diskann][navigation]") {
    const auto version = GenTestVersionList();
    const auto phase = GENERATE(knowhere::PARAM_TYPE::TRAIN, knowhere::PARAM_TYPE::DESERIALIZE);
    knowhere::Json request = {
        {"metric_type", "L2"}, {"dim", 8}, {"index_prefix", "unused_index"}, {"data_path", "unused_data"}};
    auto omitted = knowhere::IndexStaticFaced<knowhere::fp32>::CreateConfig(kNativeDiskANNType, version);
    REQUIRE(knowhere::Config::Load(*omitted, request, phase) == knowhere::Status::success);
    REQUIRE_FALSE(static_cast<const knowhere::DiskANNConfig&>(*omitted).quant_type.has_value());
    if (phase == knowhere::PARAM_TYPE::TRAIN)
        REQUIRE(std::string(
                    static_cast<const knowhere::DiskANNConfig&>(*omitted).navigation->CreateForBuild()->CodecName()) ==
                "PQ");
    else
        REQUIRE_FALSE(static_cast<const knowhere::DiskANNConfig&>(*omitted).navigation);
    for (const auto* codec : {"PQ", "RABITQ"}) {
        request["quant_type"] = codec;
        const bool pq = std::string(codec) == "PQ";
        request["pq_code_budget_gb"] = pq ? knowhere::Json("0.25") : knowhere::Json(-1);
        request["pq_code_budget_gb_ratio"] = pq ? knowhere::Json("0.5") : knowhere::Json(-1);
        request["vec_field_size_gb"] = 2.0;
        request["rbq_bits"] = pq ? knowhere::Json(10) : knowhere::Json("3");
        auto selected = knowhere::IndexStaticFaced<knowhere::fp32>::CreateConfig(kNativeDiskANNType, version);
        REQUIRE(knowhere::Config::Load(*selected, request, phase) == knowhere::Status::success);
        const auto& selected_config = static_cast<const knowhere::DiskANNConfig&>(*selected);
        REQUIRE(selected_config.quant_type.value() == codec);
        if (phase == knowhere::PARAM_TYPE::TRAIN) {
            REQUIRE(std::string(selected_config.navigation->CreateForBuild()->CodecName()) == codec);
            knowhere::Json metadata;
            selected_config.navigation->AppendBuildMetadata(metadata);
            REQUIRE(metadata.at(pq ? "pq_code_budget_gb" : "rbq_bits") == (pq ? 1.0 : 3.0));
            std::string error;
            REQUIRE(knowhere::IndexStaticFaced<knowhere::fp32>::ConfigCheck(kNativeDiskANNType, version, request,
                                                                            error) == knowhere::Status::success);
            if (pq)
                REQUIRE(knowhere::Config::Load(*selected, {{"k", 8}, {"rbq_bits_query", 9}},
                                               knowhere::PARAM_TYPE::SEARCH) == knowhere::Status::success);
        }
    }
    for (const auto* codec : {"TQ", ""}) {
        request["quant_type"] = codec;
        auto selected = knowhere::IndexStaticFaced<knowhere::fp32>::CreateConfig(kNativeDiskANNType, version);
        std::string error;
        const bool rejects_config = std::string(codec).empty() || phase == knowhere::PARAM_TYPE::TRAIN;
        REQUIRE(knowhere::Config::Load(*selected, request, phase, &error) ==
                (rejects_config ? knowhere::Status::invalid_args : knowhere::Status::success));
        if (rejects_config)
            REQUIRE(error == "Unsupported DiskANN navigation codec: " + std::string(codec));
        auto manager = std::make_shared<LegacyPQFileManager>(fs::path{});
        auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(manager));
        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(kNativeDiskANNType, version, pack).value();
        const auto rejected =
            phase == knowhere::PARAM_TYPE::TRAIN ? index.Build(nullptr, request) : index.Deserialize({}, request);
        REQUIRE(rejected ==
                (rejects_config ? knowhere::Status::invalid_args : knowhere::Status::invalid_serialized_index_type));
        REQUIRE(manager->loaded.empty());
    }
}
TEST_CASE("PQ navigation matches the original lookup", "[diskann][navigation]") {
    NativePQFixture fixture("pq_lookup_source");
    const auto prefix = (fixture.path / "index").string();
    auto model = diskann::CreateNavigationStore("PQ");
    REQUIRE(model->Load(prefix) == 0);
    size_t rows, chunks, nq, dimension;
    std::unique_ptr<uint8_t[]> codes;
    std::unique_ptr<float[]> queries;
    diskann::load_bin<uint8_t>(diskann::get_pq_compressed_filename(prefix), codes, rows, chunks);
    diskann::load_bin<float>((fixture.path / "queries.bin").string(), queries, nq, dimension);
    REQUIRE(model->Count() == rows);
    REQUIRE(model->Dimension() == dimension);
    diskann::FixedChunkPQTable native;
    native.load_pq_centroid_bin(diskann::get_pq_pivots_filename(prefix).c_str(), chunks);
    // Build-cache capacities need not make the allocation a multiple of 256.
    const auto capacity = GENERATE(size_t(40), size_t(diskann::defaults::MAX_GRAPH_DEGREE));
    auto first = model->CreateDistanceComputer(ROUND_UP(dimension, 8), capacity);
    auto second = model->CreateDistanceComputer(ROUND_UP(dimension, 8), capacity);
    std::vector<float> lut(256 * chunks);
    for (size_t q = 0; q < nq; ++q) {
        auto& computer = q % 2 ? second : first;
        computer->set_query(queries.get() + q * dimension);
        native.populate_chunk_distances(queries.get() + q * dimension, lut.data());
        for (const size_t count : {size_t(1), size_t(17), capacity}) {
            std::vector<unsigned> ids(count);
            for (size_t i = 0; i < count; ++i) ids[i] = (i * 7 + q) % rows;
            std::vector<uint8_t> gathered(count * chunks);
            std::vector<float> expected(count), actual(count);
            diskann::aggregate_coords(ids.data(), count, codes.get(), chunks, gathered.data());
            diskann::pq_dist_lookup(gathered.data(), count, chunks, lut.data(), expected.data());
            computer->compute_distances(ids.data(), count, actual.data());
            REQUIRE(actual == expected);
            computer->compute_distances(ids.data(), count, actual.data(), 0.0f);
            REQUIRE(actual == expected);
        }
    }

    first->set_query(queries.get());
    second->set_query(queries.get() + dimension);
    const unsigned id = 0;
    for (size_t q = 0; q < 2; ++q) {
        native.populate_chunk_distances(queries.get() + q * dimension, lut.data());
        float expected, actual;
        diskann::pq_dist_lookup(codes.get(), 1, chunks, lut.data(), &expected);
        auto& computer = q == 0 ? first : second;
        computer->compute_distances(&id, 1, &actual);
        REQUIRE(actual == expected);
    }

    // AiSAQ's header-only model retains ordinary -> rearranged fallback and
    // scores reader-provided codes without loading the full codes into RAM.
    for (const bool rearranged : {false, true}) {
        PQNavigationTestDirectory directory(rearranged ? "pq_reader_rearranged" : "pq_reader");
        const auto reader_prefix = (directory.path / "index").string();
        std::vector<std::string> loaded;
        if (!rearranged) {
            fs::copy_file(diskann::get_pq_compressed_filename(prefix),
                          diskann::get_pq_compressed_filename(reader_prefix));
        }
        diskann::PQNavigationStore reader_model(false, rearranged);
        std::vector<std::string> artifacts;
        reader_model.GetNecessaryFilenames(reader_prefix, artifacts);
        REQUIRE(std::find(artifacts.begin(), artifacts.end(),
                          rearranged ? diskann::get_pq_compressed_rearranged_filename(reader_prefix)
                                     : diskann::get_pq_compressed_filename(reader_prefix)) != artifacts.end());
        reader_model.GetNecessaryFilenames(reader_prefix, loaded, diskann::NavigationFilePurpose::Load);
        for (const auto& file : loaded) {
            auto source = fixture.path / fs::path(file).filename();
            if (file == diskann::get_pq_compressed_rearranged_filename(reader_prefix)) {
                source = diskann::get_pq_compressed_filename(prefix);
            }
            fs::copy_file(source, file);
        }
        REQUIRE(reader_model.Load(reader_prefix) == 0);
        REQUIRE(reader_model.Count() == rows);
        REQUIRE(reader_model.Dimension() == dimension);
        REQUIRE(std::find(loaded.begin(), loaded.end(), diskann::get_pq_compressed_filename(reader_prefix)) ==
                loaded.end());
        auto computer = reader_model.CreateDistanceComputer(ROUND_UP(dimension, 8));
        computer->set_query(queries.get());
        native.populate_chunk_distances(queries.get(), lut.data());
        float expected, actual;
        diskann::pq_dist_lookup(codes.get(), 1, chunks, lut.data(), &expected);
        static_cast<diskann::PQNavigationDistanceComputer&>(*computer).EvaluateCodes(codes.get(), 1, &actual);
        REQUIRE(actual == expected);
    }
}

TEST_CASE("DiskANN loads legacy PQ files and localizes navigation assets", "[diskann][navigation]") {
    const auto version = GenTestVersionList();
    const bool explicit_pq = GENERATE(false, true);
    const bool with_metadata = GENERATE(false, true);
    NativePQFixture fixture("pq_load_source");
    const auto source_prefix = (fixture.path / "index").string();
    if (with_metadata)
        diskann::WriteNavigationMetadata(source_prefix, "PQ");
    PQNavigationTestDirectory directory("pq_legacy_load");
    const auto prefix = (directory.path / "index").string();
    auto manager = std::make_shared<LegacyPQFileManager>(fixture.path);
    SECTION("Successful legacy load") {
    }
    SECTION("Missing navigation file preserves the file error") {
        manager->missing_file = "index_pq_pivots.bin_chunk_offsets.bin";
    }
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(manager));
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(kNativeDiskANNType, version, pack).value();
    SECTION("A navigation-only artifact prevents overwriting an existing index") {
        for (const bool metadata_only : {false, true}) {
            const auto artifact = metadata_only ? diskann::GetNavigationMetadataFilename(prefix)
                                                : diskann::get_pq_pivots_filename(prefix);
            if (metadata_only)
                diskann::WriteNavigationMetadata(prefix, "PQ");
            else
                fs::copy_file(diskann::get_pq_pivots_filename(source_prefix), artifact);
            REQUIRE(index.Build(nullptr, {{"metric_type", "L2"},
                                          {"dim", 8},
                                          {"index_prefix", prefix},
                                          {"data_path", (fixture.path / "base.bin").string()},
                                          {"max_degree", 16},
                                          {"search_list_size", 64},
                                          {"pq_code_budget_gb", 1e-6},
                                          {"build_dram_budget_gb", 1.0}}) == knowhere::Status::disk_file_error);
            REQUIRE(manager->loaded.empty());
            fs::remove(artifact);
        }
        return;
    }
    SECTION("Source absence overrides stale cached metadata") {
        fs::remove(diskann::GetNavigationMetadataFilename(source_prefix));
        diskann::WriteNavigationMetadata(prefix, "TQ");
    }
    knowhere::Json load = {
        {"metric_type", "L2"}, {"index_prefix", prefix}, {"search_cache_budget_gb", 0.0}, {"warm_up", false}};
    if (explicit_pq) {
        load["quant_type"] = "PQ";
    }
    const auto status = index.Deserialize({}, load);
    if (!manager->missing_file.empty()) {
        REQUIRE(status == knowhere::Status::disk_file_error);
        return;
    }
    REQUIRE(status == knowhere::Status::success);
    const auto metadata = diskann::GetNavigationMetadataFilename(prefix);
    REQUIRE(std::count(manager->loaded.begin(), manager->loaded.end(), metadata) ==
            (fs::exists(diskann::GetNavigationMetadataFilename(source_prefix)) ? 1 : 0));
    for (const auto& file : fs::directory_iterator(fixture.path)) {
        if (file.path().filename().string().rfind("index_pq_", 0) == 0) {
            const auto destination = (directory.path / file.path().filename()).string();
            REQUIRE(std::count(manager->loaded.begin(), manager->loaded.end(), destination) == 1);
        }
    }
    size_t nq, dim, rows, base_dim;
    std::unique_ptr<float[]> queries, base;
    diskann::load_bin<float>((fixture.path / "queries.bin").string(), queries, nq, dim);
    diskann::load_bin<float>((fixture.path / "base.bin").string(), base, rows, base_dim);
    REQUIRE(index.Count() == int64_t(rows));
    REQUIRE(index.Dim() == int64_t(dim));
    auto dataset = knowhere::GenDataSet(nq, dim, queries.get());
    knowhere::Json search = {{"metric_type", "L2"}, {"dim", dim}, {"k", 5}, {"search_list_size", 64}, {"beamwidth", 4}};
    auto result = index.Search(dataset, search, nullptr);
    REQUIRE(result.has_value());
    for (size_t q = 0; q < nq; ++q) {
        REQUIRE(result.value()->GetIds()[q * 5] == int64_t(q * 113 + 7));
        for (size_t i = 0; i < 5; ++i) {
            const auto id = result.value()->GetIds()[q * 5 + i];
            REQUIRE(id >= 0);
            REQUIRE(id < int64_t(rows));
            float reference = 0;
            for (size_t d = 0; d < dim; ++d) {
                const float delta = queries[q * dim + d] - base[id * base_dim + d];
                reference += delta * delta;
            }
            REQUIRE(result.value()->GetDistance()[q * 5 + i] == Catch::Approx(reference).margin(1e-5));
        }
    }

    search["search_list_size"] = 16;
    auto iterators = index.AnnIterator(dataset, search, nullptr, false);
    REQUIRE(iterators.has_value());
    REQUIRE(iterators.value().size() == nq);
    for (size_t q = 0; q < nq; ++q) {
        auto& iterator = iterators.value()[q];
        std::unordered_set<int64_t> visited;
        // Consume beyond a single lsearch batch, with several iterators alive.
        for (size_t i = 0; i < 80; ++i) {
            REQUIRE(iterator->HasNext().value());
            auto next = iterator->Next();
            REQUIRE(next.has_value());
            const auto [id, distance] = next.value();
            REQUIRE(id >= 0);
            REQUIRE(id < int64_t(rows));
            REQUIRE(visited.insert(id).second);
            float reference = 0;
            for (size_t d = 0; d < dim; ++d) {
                const float delta = queries[q * dim + d] - base[id * base_dim + d];
                reference += delta * delta;
            }
            REQUIRE(distance == Catch::Approx(reference).margin(1e-5));
        }
    }
}

TEST_CASE("Navigation metadata errors never fall back to legacy PQ", "[diskann][navigation]") {
    const auto version = GenTestVersionList();
    const auto type = GENERATE(kNativeDiskANNType, "AISAQ");
    NativePQFixture source("metadata_source");
    PQNavigationTestDirectory destination("metadata_destination");
    const auto source_prefix = (source.path / "index").string();
    const auto prefix = (destination.path / "index").string();
    const auto source_metadata = diskann::GetNavigationMetadataFilename(source_prefix);
    diskann::WriteNavigationMetadata(source_prefix, "PQ");
    auto manager = std::make_shared<LegacyPQFileManager>(source.path);
    auto expected = knowhere::Status::invalid_serialized_index_type;
    knowhere::Json config = {{"metric_type", "L2"}, {"index_prefix", prefix}, {"search_cache_budget_gb", 0.0}};
    auto overwrite = [&](size_t offset, uint8_t byte) {
        std::fstream file(source_metadata, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(offset);
        file.put(char(byte));
        REQUIRE(bool(file));
    };
    SECTION("Unknown codec with complete PQ files present") {
        diskann::WriteNavigationMetadata(source_prefix, "TQ");
    }
    SECTION("Unsupported version") {
        overwrite(8, 2);
    }
    SECTION("Invalid magic") {
        overwrite(0, 'X');
    }
    SECTION("Truncated header") {
        fs::resize_file(source_metadata, 10);
    }
    SECTION("Oversized codec name") {
        overwrite(12, 255);
    }
    SECTION("Truncated codec name") {
        fs::resize_file(source_metadata, 17);
    }
    SECTION("Trailing bytes") {
        std::ofstream file(source_metadata, std::ios::binary | std::ios::app);
        file.put('X');
    }
    SECTION("Existence query fails") {
        manager->fail_exists = true;
        expected = knowhere::Status::disk_file_error;
    }
    SECTION("Metadata localization fails") {
        manager->missing_file = fs::path(source_metadata).filename().string();
        expected = knowhere::Status::disk_file_error;
    }
    SECTION("Explicit codec disagrees with the stored type") {
        config["quant_type"] = "RABITQ";
    }
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(manager));
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(type, version, pack).value();
    REQUIRE(index.Deserialize({}, config) == expected);
    REQUIRE(manager->loaded.size() == (manager->fail_exists ? 0 : 1));
    if (!manager->loaded.empty())
        REQUIRE(manager->loaded.front() == diskann::GetNavigationMetadataFilename(prefix));
}

TEST_CASE("Navigation metadata loads with a fresh local file manager", "[diskann][navigation]") {
    const auto version = GenTestVersionList();
    NativePQFixture source("metadata_local_source");
    const auto prefix = (source.path / "index").string();
    diskann::WriteNavigationMetadata(prefix, "PQ");
    auto manager = std::make_shared<milvus::LocalFileManager>();
    REQUIRE(manager->IsExisted(diskann::GetNavigationMetadataFilename(prefix)) == false);
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(manager));
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(kNativeDiskANNType, version, pack).value();
    knowhere::Json config = {{"metric_type", "L2"}, {"index_prefix", prefix}, {"search_cache_budget_gb", 0.0}};
    REQUIRE(index.Deserialize({}, config) == knowhere::Status::success);
    REQUIRE(index.Count() == 512);
    REQUIRE(index.Dim() == 8);
    config["quant_type"] = "TQ";
    REQUIRE(index.Deserialize({}, config) == knowhere::Status::invalid_serialized_index_type);
    REQUIRE(index.Count() == 512);
}

namespace {
// A different encoding with no PQ files: exercises the common build/cache
// contract without introducing another production codec or a factory hook.
class RawBuildNavigation final : public diskann::NavigationStore {
 public:
    explicit RawBuildNavigation(bool sample, std::optional<size_t> cache_memory = 0)
        : sample_(sample), cache_memory_(cache_memory) {
    }
    const char*
    CodecName() const override {
        return "TEST_RAW";
    }
    bool
    IsCompatible(knowhere::DataFormatEnum, bool) const override {
        return true;
    }
    int
    ValidateBuild() const override {
        return 0;
    }
    bool
    NeedsTrainingSample() const override {
        return sample_;
    }
    int
    Build(const diskann::NavigationBuildContext& context) override {
        metric = context.metric;
        diskann::load_bin<float>(context.data_path, data_, count_, dimension_);
        metadata_matches = context.count == count_ && context.dimension == dimension_ &&
                           context.data_type == knowhere::DataFormatEnum::fp32;
        training_count = context.training_count;
        shared_sample_matches =
            context.training_data == nullptr
                ? training_count == 0
                : training_count == count_ &&
                      std::equal(data_.get(), data_.get() + count_ * dimension_, context.training_data);
        return 0;
    }
    std::optional<size_t>
    BuildCacheMemorySize(const std::string&) const override {
        return cache_memory_;
    }
    int
    Load(const std::string&) override {
        ++loads;
        return 0;
    }
    void
    GetNecessaryFilenames(const std::string&, std::vector<std::string>&,
                          diskann::NavigationFilePurpose) const override {
    }
    uint64_t
    Count() const override {
        return count_;
    }
    uint64_t
    Dimension() const override {
        return dimension_;
    }
    size_t
    MemorySize() const override {
        return count_ * dimension_ * sizeof(float);
    }
    size_t
    QueryMemorySize(uint64_t aligned_dimension) const override {
        return aligned_dimension * sizeof(float);
    }
    std::unique_ptr<diskann::NavigationDistanceComputer>
    CreateDistanceComputer(uint64_t, uint64_t capacity) const override {
        struct Computer final : diskann::NavigationDistanceComputer {
            Computer(const RawBuildNavigation& store, size_t capacity)
                : store(store), capacity(capacity), query(store.dimension_) {
            }
            void
            set_query(const float* input) override {
                std::copy_n(input, query.size(), query.data());
            }
            void
            compute_distances(const unsigned* ids, uint64_t count, float* distances) override {
                if (count > capacity)
                    throw std::runtime_error("cache exceeded its batch capacity");
                for (size_t i = 0; i < count; ++i) {
                    float distance = 0;
                    for (size_t d = 0; d < query.size(); ++d) {
                        const float delta = query[d] - store.data_[ids[i] * query.size() + d];
                        distance += delta * delta;
                    }
                    distances[i] = distance;
                }
            }
            void
            compute_distances(const unsigned* ids, uint64_t count, float* distances, float threshold) override {
                if (std::isfinite(threshold))
                    ++store.bounded_batches;
                else
                    ++store.unbounded_batches;
                compute_distances(ids, count, distances);
            }
            const RawBuildNavigation& store;
            size_t capacity;
            std::vector<float> query;
        };
        if (loads != 1)
            throw std::runtime_error("cache model was not loaded once after the memory gate");
        ++computers;
        return std::make_unique<Computer>(*this, capacity);
    }
    diskann::Metric metric;
    bool metadata_matches = false, shared_sample_matches = false;
    size_t training_count = 0, loads = 0;
    mutable std::atomic<size_t> computers{0};
    mutable size_t bounded_batches = 0, unbounded_batches = 0;

 private:
    bool sample_;
    std::optional<size_t> cache_memory_;
    std::unique_ptr<float[]> data_;
    size_t count_ = 0, dimension_ = 0;
};
}  // namespace

TEST_CASE("DiskANN build and cache do not require navigation PQ", "[diskann][navigation][navigation-build]") {
    const auto metric = GENERATE(diskann::Metric::L2, diskann::Metric::INNER_PRODUCT, diskann::Metric::COSINE);
    const auto disk_chunks = GENERATE(0, 2);
    const auto needs_sample = GENERATE(false, true);
    PQNavigationTestDirectory directory("independent_build");
    constexpr size_t rows = 512, dim = 8;
    auto dataset = GenDataSet(rows, dim, 123);
    const auto source = (directory.path / "base.bin").string();
    diskann::save_bin<const float>(source, static_cast<const float*>(dataset->GetTensor()), rows, dim);
    diskann::BuildConfig config;
    config.data_file_path = source;
    config.index_file_path = (directory.path / "index").string();
    config.compare_metric = metric;
    config.max_degree = 16;
    config.search_list_size = 64;
    config.index_mem_gb = 1;
    config.disk_pq_dims = disk_chunks;
    config.num_nodes_to_cache = 16;
    RawBuildNavigation navigation(needs_sample);
    REQUIRE(diskann::build_disk_index<float>(config, navigation) == 0);
    REQUIRE(navigation.metadata_matches);
    REQUIRE(navigation.metric == metric);
    REQUIRE(navigation.Dimension() == dim + (metric == diskann::Metric::INNER_PRODUCT));
    REQUIRE(navigation.shared_sample_matches);
    REQUIRE(navigation.training_count == ((needs_sample || disk_chunks) ? rows : 0));
    REQUIRE(navigation.loads == 1);
    REQUIRE(navigation.computers > 0);
    REQUIRE_FALSE(fs::exists(diskann::get_pq_pivots_filename(config.index_file_path)));
    REQUIRE_FALSE(fs::exists(diskann::get_pq_compressed_filename(config.index_file_path)));
    REQUIRE(fs::exists(diskann::get_disk_index_filename(config.index_file_path)));
    REQUIRE(fs::exists(config.index_file_path + "_disk.index_pq_pivots.bin") == bool(disk_chunks));
    std::unique_ptr<uint32_t[]> cached;
    size_t count, columns;
    diskann::load_bin<uint32_t>(diskann::get_cached_nodes_file(config.index_file_path), cached, count, columns);
    REQUIRE(count == config.num_nodes_to_cache);
    REQUIRE(columns == 1);
    std::unordered_set<uint32_t> ids(cached.get(), cached.get() + count);
    REQUIRE(ids.size() == count);
    for (auto id : ids) REQUIRE(id < rows);
}

TEST_CASE("DiskANN supplies navigation bounds only to bounded graph searches", "[diskann][navigation]") {
    const bool cached = GENERATE(false, true);
    PQNavigationTestDirectory directory("navigation_bounds");
    constexpr size_t rows = 512, dim = 8;
    auto base = GenDataSet(rows, dim, 173);
    auto query = GenDataSet(1, dim, 174);
    diskann::BuildConfig config;
    config.data_file_path = (directory.path / "base.bin").string();
    config.index_file_path = (directory.path / "index").string();
    config.compare_metric = diskann::Metric::L2;
    config.max_degree = 16;
    config.search_list_size = 64;
    config.index_mem_gb = 1;
    diskann::save_bin<const float>(config.data_file_path, static_cast<const float*>(base->GetTensor()), rows, dim);
    auto navigation = std::make_unique<RawBuildNavigation>(false, std::nullopt);
    REQUIRE(diskann::build_disk_index<float>(config, *navigation) == 0);
    REQUIRE(navigation->Load(config.index_file_path) == 0);
    auto& observed = *navigation;
    auto reader = std::make_shared<LinuxAlignedFileReader>();
    diskann::SSDGraphIndex<float> index(reader, diskann::Metric::L2);
    REQUIRE(index.load(1, config.index_file_path.c_str(), std::move(navigation)) == 0);
    if (cached) {
        std::vector<uint32_t> nodes;
        index.cache_bfs_levels(rows, nodes);
        index.load_cache_list(nodes);
    }
    const auto* input = static_cast<const float*>(query->GetTensor());
    int64_t ids[8];
    float scores[8];
    // Both disk and cached neighborhoods must receive a finite threshold once
    // L is full, and no threshold before that. Scores stay unchanged.
    diskann::QueryStats stats;
    index.cached_beam_search(input, 8, 16, ids, scores, 4, false, &stats);
    REQUIRE((cached ? stats.n_cache_hits > 0 : stats.n_cache_hits == 0));
    REQUIRE(observed.bounded_batches > 0);
    REQUIRE(observed.unbounded_batches > 0);
    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(ids[i] >= 0);
        float expected = 0;
        for (size_t d = 0; d < dim; ++d) {
            const float delta = input[d] - static_cast<const float*>(base->GetTensor())[ids[i] * dim + d];
            expected += delta * delta;
        }
        REQUIRE(scores[i] == Catch::Approx(expected));
    }
    const auto bounded = observed.bounded_batches;
    std::vector<uint8_t> mask(rows / 8, 0x55);
    index.cached_beam_search(input, 8, 16, ids, scores, 4, false, nullptr, nullptr,
                             knowhere::BitsetView(mask.data(), rows, rows / 2), 0.0f);
    auto workspace = index.getIteratorWorkspace(input, 16, 4, -1, nullptr);
    index.getIteratorNextBatch(workspace.get());
    REQUIRE(observed.bounded_batches == bounded);
}

TEST_CASE("PQ build cache accepts nonaligned gather buffer sizes", "[diskann][navigation][pq-build-cache]") {
    PQNavigationTestDirectory directory("pq_build_cache_alignment");
    // IP adds one dimension: ROUND_UP(128 + 1, 8) * 56 = 7616 bytes,
    // which is not a multiple of the allocator's 256-byte alignment.
    constexpr size_t rows = 512, dim = 128;
    auto dataset = GenDataSet(rows, dim, 123);
    diskann::BuildConfig config;
    config.data_file_path = (directory.path / "base.bin").string();
    config.index_file_path = (directory.path / "index").string();
    config.compare_metric = diskann::Metric::INNER_PRODUCT;
    config.max_degree = 56;
    config.search_list_size = 64;
    config.index_mem_gb = 1;
    config.num_nodes_to_cache = 16;
    diskann::save_bin<const float>(config.data_file_path, static_cast<const float*>(dataset->GetTensor()), rows, dim);
    diskann::PQNavigationStore navigation(true, false, {double(rows * 16) / (1 << 30)});
    REQUIRE(diskann::build_disk_index<float>(config, navigation) == 0);
    std::unique_ptr<uint32_t[]> cached;
    size_t count, columns;
    diskann::load_bin<uint32_t>(diskann::get_cached_nodes_file(config.index_file_path), cached, count, columns);
    REQUIRE(count == config.num_nodes_to_cache);
    REQUIRE(columns == 1);
    std::unordered_set<uint32_t> ids(cached.get(), cached.get() + count);
    REQUIRE(ids.size() == count);
    for (auto id : ids) REQUIRE(id < rows);
}

TEST_CASE("DiskANN build cache respects missing assets and memory gate", "[diskann][navigation][navigation-build]") {
    const auto available = GENERATE(false, true);
    PQNavigationTestDirectory directory("build_cache_gate");
    auto dataset = GenDataSet(64, 8, 123);
    const auto source = (directory.path / "base.bin").string();
    diskann::save_bin<const float>(source, static_cast<const float*>(dataset->GetTensor()), 64, 8);
    diskann::BuildConfig config;
    config.data_file_path = source;
    config.index_file_path = (directory.path / "index").string();
    config.max_degree = 8;
    config.search_list_size = 32;
    config.index_mem_gb = 1;
    config.num_nodes_to_cache = 8;
    RawBuildNavigation navigation(false, available ? std::optional<size_t>(size_t(2) << 30) : std::nullopt);
    REQUIRE(diskann::build_disk_index<float>(config, navigation) == 0);
    REQUIRE(navigation.loads == 0);
    REQUIRE(navigation.computers == 0);
    REQUIRE_FALSE(fs::exists(diskann::get_cached_nodes_file(config.index_file_path)));
}

TEST_CASE("PQ navigation build preserves encoding and cache accounting", "[diskann][navigation][navigation-build]") {
    // Include metric-prepared FP32 data and both existing low-precision L2 inputs.
    const auto mode = GENERATE(0, 1, 2, 3, 4);
    PQNavigationTestDirectory directory("pq_build_equivalence");
    constexpr size_t rows = 512, dim = 8;
    const auto metric = mode == 1   ? diskann::Metric::INNER_PRODUCT
                        : mode == 2 ? diskann::Metric::COSINE
                                    : diskann::Metric::L2;
    auto dataset = GenDataSet(rows, dim, 123);
    auto check = [&](auto value) {
        using T = decltype(value);
        std::vector<T> base(rows * dim);
        const auto* source = static_cast<const float*>(dataset->GetTensor());
        for (size_t i = 0; i < base.size(); ++i) base[i] = T(source[i]);
        auto data_path = (directory.path / "base.bin").string();
        diskann::save_bin<T>(data_path, base.data(), rows, dim);
        if constexpr (std::is_same_v<T, float>) {
            const auto prepared = (directory.path / "prepared.bin").string();
            if (metric == diskann::Metric::INNER_PRODUCT) {
                diskann::prepare_base_for_inner_products<T>(data_path, prepared);
                data_path = prepared;
            } else if (metric == diskann::Metric::COSINE) {
                diskann::prepare_base_for_cosine<T>(data_path, prepared);
                data_path = prepared;
            }
        }
        std::unique_ptr<float[]> sample;
        size_t sample_count, sample_dim;
        gen_random_slice<T>(data_path, 1.0, sample, sample_count, sample_dim);
        const auto prefix = (directory.path / "index").string();
        diskann::PQNavigationStore navigation(true, false, {double(rows * 2) / (1024 * 1024 * 1024)});
        REQUIRE(navigation.Build({prefix, data_path, knowhere::datatype_v<T>, metric, rows, sample_dim, sample.get(),
                                  sample_count}) == 0);
        const auto pivots = diskann::get_pq_pivots_filename(prefix);
        // Reference uses the original native encoder and the written codebook.
        const auto reference = (directory.path / "reference.bin").string();
        generate_pq_data_from_pivots<T>(data_path, 256, 2, pivots, reference);
        std::unique_ptr<uint8_t[]> actual, expected;
        size_t actual_rows, chunks, expected_rows, expected_chunks;
        diskann::load_bin<uint8_t>(diskann::get_pq_compressed_filename(prefix), actual, actual_rows, chunks);
        diskann::load_bin<uint8_t>(reference, expected, expected_rows, expected_chunks);
        REQUIRE(actual_rows == rows);
        REQUIRE(chunks == 2);
        REQUIRE(expected_rows == rows);
        REQUIRE(expected_chunks == chunks);
        REQUIRE(std::equal(actual.get(), actual.get() + rows * chunks, expected.get()));
        std::unique_ptr<float[]> centroid;
        size_t centroid_rows, centroid_columns;
        diskann::load_bin<float>(diskann::get_pq_centroid_filename(pivots), centroid, centroid_rows, centroid_columns);
        REQUIRE(centroid_rows == sample_dim);
        REQUIRE(centroid_columns == 1);
        for (size_t d = 0; d < sample_dim; ++d) {
            float mean = 0;
            if (metric == diskann::Metric::L2) {
                for (size_t i = 0; i < sample_count; ++i) mean += sample[i * sample_dim + d];
                mean /= sample_count;
            }
            REQUIRE(centroid[d] == mean);
        }
        REQUIRE(navigation.BuildCacheMemorySize(prefix).value() ==
                fs::file_size(pivots) + fs::file_size(diskann::get_pq_compressed_filename(prefix)));
        REQUIRE(navigation.Load(prefix) == 0);
        REQUIRE(navigation.Count() == rows);
        REQUIRE(navigation.Dimension() == sample_dim);
        fs::remove(diskann::get_pq_compressed_filename(prefix));
        REQUIRE_FALSE(navigation.BuildCacheMemorySize(prefix).has_value());
    };
    if (mode == 3)
        check(knowhere::fp16{});
    else if (mode == 4)
        check(knowhere::bf16{});
    else
        check(float{});
}

TEST_CASE("PQ build rejects invalid budget before SSD encoding", "[diskann][navigation][navigation-build]") {
    PQNavigationTestDirectory directory("pq_build_validation");
    auto dataset = GenDataSet(64, 8, 123);
    diskann::BuildConfig config;
    config.data_file_path = (directory.path / "base.bin").string();
    config.index_file_path = (directory.path / "index").string();
    config.index_mem_gb = 1;
    config.disk_pq_dims = 2;
    diskann::save_bin<const float>(config.data_file_path, static_cast<const float*>(dataset->GetTensor()), 64, 8);
    REQUIRE(diskann::PQNavigationStore::CodeSize(512, 8, {256.0 / (1 << 30)}) == 1);
    REQUIRE(diskann::PQNavigationStore::CodeSize(512, 8, {8192.0 / (1 << 30)}) == 8);
    REQUIRE(diskann::PQNavigationStore::CodeSize(512, 1024, {1048576.0 / (1 << 30)}) ==
            diskann::defaults::MAX_PQ_CHUNKS);
    diskann::PQNavigationStore navigation;
    REQUIRE(diskann::build_disk_index<float>(config, navigation) == -1);
    REQUIRE_FALSE(fs::exists(config.index_file_path + "_disk.index_pq_pivots.bin"));
    REQUIRE_FALSE(fs::exists(diskann::get_pq_pivots_filename(config.index_file_path)));
    REQUIRE_FALSE(fs::exists(diskann::get_disk_index_filename(config.index_file_path)));
}

TEST_CASE("DiskANN common metadata does not own navigation parameters", "[diskann][navigation][navigation-metadata]") {
    knowhere::feder::diskann::DiskANNBuildConfig common{};
    common.disk_pq_dims = 7;
    knowhere::Json params = common;
    REQUIRE(params.at("disk_pq_dims") == 7);
    REQUIRE_FALSE(params.contains("pq_code_budget_gb"));

    auto navigation = knowhere::CreateNavigationConfig("PQ");
    navigation->AppendBuildMetadata(params);
    REQUIRE_FALSE(params.contains("pq_code_budget_gb"));
    static_cast<knowhere::PQNavigationConfig&>(*navigation).pq_code_budget_gb = 0.25f;
    navigation->AppendBuildMetadata(params);
    REQUIRE(params.at("pq_code_budget_gb") == 0.25f);
    REQUIRE(params.at("disk_pq_dims") == 7);
    REQUIRE_THROWS_AS(knowhere::CreateNavigationConfig("TQ"), diskann::NavigationTypeError);
}

TEST_CASE("DiskANN and AiSAQ delegate PQ metadata to navigation", "[diskann][navigation][navigation-metadata]") {
    const auto type = GENERATE(kNativeDiskANNType, "AISAQ");
    PQNavigationTestDirectory directory("navigation_metadata");
    constexpr size_t rows = 512, dim = 8;
    auto base = GenDataSet(rows, dim, 123);
    const auto source = (directory.path / "base.bin").string();
    const auto prefix = (directory.path / "index").string();
    diskann::save_bin<const float>(source, static_cast<const float*>(base->GetTensor()), rows, dim);
    knowhere::Json build = {{"metric_type", "L2"},
                            {"dim", dim},
                            {"data_path", source},
                            {"index_prefix", prefix},
                            {"max_degree", 16},
                            {"search_list_size", 64},
                            {"build_dram_budget_gb", 1.0},
                            {"disk_pq_dims", 0},
                            {"pq_code_budget_gb", double(rows) / (1 << 30)},
                            {"pq_code_budget_gb_ratio", 0.0625},
                            {"vec_field_size_gb", double(rows * dim * sizeof(float)) / (1 << 30)}};
    if (std::string(type) == "AISAQ")
        build["inline_pq"] = -1;
    auto manager = std::make_shared<milvus::LocalFileManager>();
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(manager));
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(type, GenTestVersionList(), pack).value();
    REQUIRE(index.Build(nullptr, build) == knowhere::Status::success);
    REQUIRE(index.Deserialize({}, {{"metric_type", "L2"}, {"index_prefix", prefix}, {"search_cache_budget_gb", 0.0}}) ==
            knowhere::Status::success);
    // Exercise the modified Node implementation with a prepared build config;
    // the existing FEDER request parser is outside this navigation refactor.
    auto config = index.Node()->CreateConfig();
    REQUIRE(knowhere::Config::Load(*config, build, knowhere::PARAM_TYPE::TRAIN) == knowhere::Status::success);
    auto result = index.Node()->GetIndexMeta(std::move(config));
    REQUIRE(result.has_value());
    const auto metadata = knowhere::Json::parse(result.value()->GetJsonInfo());
    const auto& params = metadata.at("build_params_");
    REQUIRE(params.at("data_path") == source);
    REQUIRE(params.at("max_degree") == 16);
    REQUIRE(params.at("search_list_size") == 64);
    REQUIRE(params.at("build_dram_budget_gb") == 1.0);
    REQUIRE(params.at("disk_pq_dims") == 0);
    REQUIRE(params.at("accelerate_build") == false);
    REQUIRE(metadata.at("num_elem_") == rows);
    REQUIRE_FALSE(metadata.at("entry_points_").empty());
    // Preserve the existing flat PQ JSON field with the ratio-adjusted value.
    REQUIRE(params.at("pq_code_budget_gb") == double(rows * 2) / (1 << 30));
}
