// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cachinglayer/Manager.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {

uint64_t
get_params_hash(const std::vector<int32_t>& params) {
    std::hash<int32_t> h;
    std::hash<uint64_t> h64;

    uint64_t result = 0;

    for (const auto value : params) {
        result = h64((result ^ h(value)) + 17);
    }

    return result;
}

template <typename T>
void
match_datasets(const knowhere::DataSetPtr& baseline, const knowhere::DataSetPtr& candidate, const int64_t* const ids) {
    REQUIRE(baseline != nullptr);
    REQUIRE(candidate != nullptr);
    REQUIRE(baseline->GetDim() == candidate->GetDim());
    REQUIRE(baseline->GetRows() == candidate->GetRows());

    const int64_t dim = baseline->GetDim();
    const int64_t rows = candidate->GetRows();

    const T* const baseline_data = reinterpret_cast<const T*>(baseline->GetTensor());
    const T* const candidate_data = reinterpret_cast<const T*>(candidate->GetTensor());

    for (int64_t i = 0; i < rows; i++) {
        const int64_t id = ids[i];
        for (int64_t j = 0; j < dim; j++) {
            REQUIRE(baseline_data[id * dim + j] == candidate_data[i * dim + j]);
        }
    }
}

// make sure the two iterators return the same results.
inline bool
check_same_iterator(const knowhere::IndexNode::IteratorPtr& iter1, const knowhere::IndexNode::IteratorPtr& iter2) {
    size_t count = 0;
    while (iter1->HasNext()) {
        REQUIRE(iter2->HasNext());
        auto [id1, dist1] = iter1->Next();
        auto [id2, dist2] = iter2->Next();
        count++;
        if (id1 != id2 || dist1 != dist2) {
            return false;
        }
    }
    printf("Total number of iterator->Next() calls: %ld\n", count);
    REQUIRE(!iter2->HasNext());
    return true;
}

struct FileIOWriter {
    std::fstream fs;
    std::string name;

    explicit FileIOWriter(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::out | std::ios::binary);
    }

    ~FileIOWriter() {
        fs.close();
    }

    size_t
    operator()(void* ptr, size_t size) {
        fs.write(reinterpret_cast<char*>(ptr), size);
        return size;
    }
};

struct FileIOReader {
    std::fstream fs;
    std::string name;

    explicit FileIOReader(const std::string& fname) {
        name = fname;
        fs = std::fstream(name, std::ios::in | std::ios::binary);
    }

    ~FileIOReader() {
        fs.close();
    }

    size_t
    operator()(void* ptr, size_t size) {
        fs.read(reinterpret_cast<char*>(ptr), size);
        return size;
    }

    size_t
    size() {
        fs.seekg(0, fs.end);
        size_t len = fs.tellg();
        fs.seekg(0, fs.beg);
        return len;
    }
};

void
write_index(knowhere::Index<knowhere::IndexNode>& index, const std::string& filename, const knowhere::Json& conf) {
    FileIOWriter writer(filename);

    knowhere::BinarySet binary_set;
    index.Serialize(binary_set);

    const auto& m = binary_set.binary_map_;
    for (auto it = m.begin(); it != m.end(); ++it) {
        const std::string& name = it->first;
        size_t name_size = name.length();
        const knowhere::BinaryPtr data = it->second;
        size_t data_size = data->size;

        writer(&name_size, sizeof(name_size));
        writer(&data_size, sizeof(data_size));
        writer((void*)name.c_str(), name_size);
        writer(data->data.get(), data_size);
    }
}

void
read_index(knowhere::Index<knowhere::IndexNode>& index, const std::string& filename, const knowhere::Json& conf) {
    FileIOReader reader(filename);
    int64_t file_size = reader.size();
    if (file_size < 0) {
        throw std::exception();
    }

    knowhere::BinarySet binary_set;
    int64_t offset = 0;
    while (offset < file_size) {
        size_t name_size, data_size;
        reader(&name_size, sizeof(size_t));
        offset += sizeof(size_t);
        reader(&data_size, sizeof(size_t));
        offset += sizeof(size_t);

        std::string name;
        name.resize(name_size);
        reader(name.data(), name_size);
        offset += name_size;
        auto data = new uint8_t[data_size];
        reader(data, data_size);
        offset += data_size;

        std::shared_ptr<uint8_t[]> data_ptr(data);
        binary_set.Append(name, data_ptr, data_size);
    }

    index.Deserialize(binary_set, conf);
}

template <typename T>
knowhere::Index<knowhere::IndexNode>
create_index(const std::string& index_type, const std::string& index_file_name,
             const knowhere::DataSetPtr& default_ds_ptr, const knowhere::Json& conf, const bool mv_only_enable,
             const std::optional<std::string>& additional_name = std::nullopt) {
    std::string additional_name_s = additional_name.value_or("");

    printf("Creating %sindex \"%s\"\n", additional_name_s.c_str(), index_type.c_str());

    auto version = GenTestEmbListVersionList();
    auto index = knowhere::IndexFactory::Instance().Create<T>(index_type, version);

    try {
        printf("Reading %sindex file: %s\n", additional_name_s.c_str(), index_file_name.c_str());

        read_index(index.value(), index_file_name, conf);
    } catch (...) {
        printf("Building %sindex all on %ld vectors\n", additional_name_s.c_str(), default_ds_ptr->GetRows());

        auto base = knowhere::ConvertToDataTypeIfNeeded<T>(default_ds_ptr);

        if (mv_only_enable) {
            base->Set(knowhere::meta::SCALAR_INFO,
                      default_ds_ptr->Get<std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>>(
                          knowhere::meta::SCALAR_INFO));
        }

        StopWatch sw;
        index.value().Build(base, conf);
        double elapsed = sw.elapsed();
        printf("Building %sindex took %f msec\n", additional_name_s.c_str(), elapsed);

        printf("Writing %sindex file: %s\n", additional_name_s.c_str(), index_file_name.c_str());

        write_index(index.value(), index_file_name, conf);
    }

    return index.value();
}

template <typename T>
std::string
get_index_name(const std::string& ann_test_name, const std::string& index_type, const std::vector<int32_t>& params) {
    std::string params_str = "";
    for (size_t i = 0; i < params.size(); i++) {
        params_str += "_" + std::to_string(params[i]);
    }
    if constexpr (std::is_same_v<T, knowhere::fp32>) {
        return ann_test_name + "_" + index_type + params_str + "_fp32" + ".index";
    } else if constexpr (std::is_same_v<T, knowhere::fp16>) {
        return ann_test_name + "_" + index_type + params_str + "_fp16" + ".index";
    } else if constexpr (std::is_same_v<T, knowhere::bf16>) {
        return ann_test_name + "_" + index_type + params_str + "_bf16" + ".index";
    } else if constexpr (std::is_same_v<T, knowhere::int8>) {
        return ann_test_name + "_" + index_type + params_str + "_int8" + ".index";
    } else {
        return ann_test_name + "_" + index_type + params_str + ".index";
    }
}

//
const std::string ann_test_name_ = "faiss_hnsw";

bool
index_support_int8(const knowhere::Json& conf) {
    const std::string index_type = conf[knowhere::meta::INDEX_TYPE].get<std::string>();
    return knowhere::IndexFactory::Instance().FeatureCheck(index_type, knowhere::feature::INT8);
}

//
template <typename T>
std::string
test_emb_list_index(const knowhere::DataSetPtr& default_ds_ptr, const knowhere::DataSetPtr& query_ds_ptr,
                    const knowhere::DataSetPtr& golden_result, const std::vector<int32_t>& index_params,
                    const knowhere::Json& conf, const bool mv_only_enable, const knowhere::BitsetView bitset_view,
                    const bool load_index_only = false) {
    const std::string index_type = conf[knowhere::meta::INDEX_TYPE].get<std::string>();

    // load indices
    std::string index_file_name = get_index_name<T>(ann_test_name_, index_type, index_params);

    // training data
    auto default_t_ds_ptr = knowhere::ConvertToDataTypeIfNeeded<T>(default_ds_ptr);

    // our index
    // first, we create an index and save it
    auto index = create_index<T>(index_type, index_file_name, default_ds_ptr, conf, mv_only_enable);

    // // then, we force it to be loaded in order to test load & save
    auto index_loaded = create_index<T>(index_type, index_file_name, default_ds_ptr, conf, mv_only_enable);

    // query
    auto query_t_ds_ptr = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds_ptr);

    StopWatch sw_search;
    auto result = load_index_only ? index_loaded.Search(query_t_ds_ptr, conf, bitset_view)
                                  : index.Search(query_t_ds_ptr, conf, bitset_view);
    double search_elapsed = sw_search.elapsed();
    printf("search cost: %f\n", search_elapsed);

    auto result_loaded = index_loaded.Search(query_t_ds_ptr, conf, bitset_view);

    // calc recall
    auto recall = GetKNNRecall(*golden_result, *result.value());
    printf("recall: %f\n", recall);
    auto recall_loaded = GetKNNRecall(*golden_result, *result_loaded.value());
    const float target_recall = 0.75;
    REQUIRE(recall >= target_recall);
    REQUIRE(recall_loaded >= target_recall);
    REQUIRE(recall == recall_loaded);

    return index_file_name;
}

}  // namespace

TEST_CASE("Search for EMBList Indices (Float)", "Benchmark and validation on float vectors") {
    // various constants and restrictions

    // metrics to test
    const std::vector<std::string> DISTANCE_TYPES = {"MAX_SIM_IP", "MAX_SIM_COSINE", "MAX_SIM_L2"};

    // for unit tests
    const std::vector<int32_t> DIMS = {4};
    const std::vector<int32_t> NBS = {256};
    const int32_t NQ = 10;
    const int32_t TOPK = 16;

    const std::vector<bool> MV_ONLYs = {false, true};

    // SQ params
    const std::vector<std::string> SQ_TYPES = {"SQ6", "SQ8", "BF16", "FP16"};
    // accepted refines for a given SQ type for a FP32 data type
    std::unordered_map<std::string, std::vector<std::string>> SQ_ALLOWED_REFINES_FP32 = {
        {"SQ6", {"SQ8", "BF16", "FP16", "FLAT"}},
        {"SQ8", {"BF16", "FP16", "FLAT"}},
        {"BF16", {"FLAT"}},
        {"FP16", {"FLAT"}}};
    // accepted refines for a given SQ type for a FP16 data type
    std::unordered_map<std::string, std::vector<std::string>> SQ_ALLOWED_REFINES_FP16 = {
        {"SQ6", {"SQ8", "FP16"}}, {"SQ8", {"FP16"}}, {"BF16", {}}, {"FP16", {}}};
    // accepted refines for a given SQ type for a BF16 data type
    std::unordered_map<std::string, std::vector<std::string>> SQ_ALLOWED_REFINES_BF16 = {
        {"SQ6", {"SQ8", "BF16"}}, {"SQ8", {"BF16"}}, {"BF16", {}}, {"FP16", {}}};

    // PQ params
    const std::vector<int32_t> NBITS = {8};
    // accepted refines for PQ for FP32 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_FP32 = {{"SQ6", "SQ8", "BF16", "FP16", "FLAT"}};
    // accepted refines for PQ for FP16 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_FP16 = {{"SQ6", "SQ8", "FP16"}};
    // accepted refines for PQ for BF16 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_BF16 = {{"SQ6", "SQ8", "BF16"}};

    // random bitset rates
    // 0.0 means unfiltered, 1.0 means all filtered out
    const std::vector<float> BITSET_RATES = {0.0f, 0.5f, 0.95f, 1.0f};

    // create base json config
    knowhere::Json default_conf;

    default_conf[knowhere::indexparam::HNSW_M] = 16;
    default_conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    default_conf[knowhere::indexparam::EF] = 64;
    default_conf[knowhere::meta::TOPK] = TOPK;
    default_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 3.0f;
    default_conf[knowhere::indexparam::NLIST] = 6;
    default_conf[knowhere::indexparam::NPROBE] = 4;

    SECTION("HNSW FLAT") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(params);
                    // vector_id -> emb_list_id
                    // [0...9] -> 0, [10...19] -> 1, [20...29] -> 2, ...
                    int each_el_len = 10;
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListDataSet(nb, dim, rng_seed, each_el_len);
                    // emb_list_id -> partition_id
                    // [0, 3, 6, ...] -> 0, [1, 4, 7, ...] -> 1, [2, 5, 8, ...] -> 2, ...
                    int partition_num = 3;
                    printf("num_el: %d, each_el_len: %d, partition_num: %d\n", num_el, each_el_len, partition_num);
                    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info =
                        GenerateScalarInfoWithStep(nb, partition_num, each_el_len);

                    for (const bool mv_only_enable : MV_ONLYs) {
                        printf("with mv only enabled : %d\n", mv_only_enable);
                        if (mv_only_enable) {
                            default_ds_ptr->Set(knowhere::meta::SCALAR_INFO, scalar_info);
                        }

                        std::vector<std::string> index_files;
                        std::string index_file;

                        // test various bitset rates
                        for (const float bitset_rate : BITSET_RATES) {
                            printf("bitset_rate: %f\n", bitset_rate);
                            const std::vector<uint8_t> bitset_data = GenerateBitsetByPartition(
                                num_el, 1.0f - bitset_rate, mv_only_enable ? partition_num : 1);

                            // initialize bitset_view.
                            // provide a default one if nbits_set == 0
                            knowhere::BitsetView bitset_view = nullptr;
                            if (bitset_rate != 0.0f || mv_only_enable) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), num_el);
                            }

                            // get a golden result
                            // auto golden_result = golden_index.Search(query_ds_ptr, conf, bitset_view);
                            auto golden_result = knowhere::BruteForce::Search<knowhere::fp32>(
                                default_ds_ptr, query_ds_ptr, conf, bitset_view);

                            // fp32 candidate
                            printf(
                                "\nProcessing EMBList HNSW,Flat fp32 for %s distance, dim=%d, nrows=%d, %d%% points "
                                "filtered "
                                "out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file =
                                test_emb_list_index<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                    params, conf, mv_only_enable, bitset_view);
                            index_files.emplace_back(index_file);

                            // fp16 candidate
                            printf(
                                "\nProcessing EMBList HNSW,Flat fp16 for %s distance, dim=%d, nrows=%d, %d%% points "
                                "filtered "
                                "out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file =
                                test_emb_list_index<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                    params, conf, mv_only_enable, bitset_view);
                            index_files.emplace_back(index_file);

                            // bf16 candidate
                            printf(
                                "\nProcessing EMBList HNSW,Flat bf16 for %s distance, dim=%d, nrows=%d, %d%% points "
                                "filtered "
                                "out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file =
                                test_emb_list_index<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                    params, conf, mv_only_enable, bitset_view);
                            index_files.emplace_back(index_file);

                            if (index_support_int8(conf)) {
                                // int8 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,Flat int8 for %s distance, dim=%d, nrows=%d, %d%% "
                                    "points "
                                    "filtered "
                                    "out\n",
                                    DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::int8>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);
                            }
                            std::remove(get_index_name<knowhere::fp32>(ann_test_name_, index_type, params).c_str());
                            std::remove(get_index_name<knowhere::fp16>(ann_test_name_, index_type, params).c_str());
                            std::remove(get_index_name<knowhere::bf16>(ann_test_name_, index_type, params).c_str());
                            if (index_support_int8(conf)) {
                                std::remove(get_index_name<knowhere::int8>(ann_test_name_, index_type, params).c_str());
                            }
                        }
                        for (auto index : index_files) {
                            std::remove(index.c_str());
                        }
                    }
                }
            }
        }
    }

#ifdef KNOWHERE_WITH_CARDINAL
    SECTION("CARDINAL_TIERED") {
        static const int64_t mb = 1024 * 1024;
        milvus::cachinglayer::Manager::ConfigureTieredStorage(
            {CacheWarmupPolicy::CacheWarmupPolicy_Disable, CacheWarmupPolicy::CacheWarmupPolicy_Disable,
             CacheWarmupPolicy::CacheWarmupPolicy_Disable, CacheWarmupPolicy::CacheWarmupPolicy_Disable},
            {1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb}, true, true, {10, false, 30});
        const std::string& index_type = knowhere::IndexEnum::INDEX_CARDINAL_TIERED;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(params);
                    // vector_id -> emb_list_id
                    // [0...9] -> 0, [10...19] -> 1, [20...29] -> 2, ...
                    int each_el_len = 10;
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListDataSet(nb, dim, rng_seed, each_el_len);
                    // emb_list_id -> partition_id
                    // [0, 3, 6, ...] -> 0, [1, 4, 7, ...] -> 1, [2, 5, 8, ...] -> 2, ...
                    int partition_num = 3;
                    printf("num_el: %d, each_el_len: %d, partition_num: %d\n", num_el, each_el_len, partition_num);
                    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info =
                        GenerateScalarInfoWithStep(nb, partition_num, each_el_len);

                    for (const bool mv_only_enable : MV_ONLYs) {
                        printf("with mv only enabled : %d\n", mv_only_enable);
                        if (mv_only_enable) {
                            // continue;
                            default_ds_ptr->Set(knowhere::meta::SCALAR_INFO, scalar_info);
                        }

                        std::vector<std::string> index_files;
                        std::string index_file;

                        // test various bitset rates
                        for (const float bitset_rate : BITSET_RATES) {
                            printf("bitset_rate: %f\n", bitset_rate);
                            const std::vector<uint8_t> bitset_data = GenerateBitsetByPartition(
                                num_el, 1.0f - bitset_rate, mv_only_enable ? partition_num : 1);

                            // initialize bitset_view.
                            // provide a default one if nbits_set == 0
                            knowhere::BitsetView bitset_view = nullptr;
                            if (bitset_rate != 0.0f || mv_only_enable) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), num_el);
                            }

                            // get a golden result
                            auto golden_result = knowhere::BruteForce::Search<knowhere::fp32>(
                                default_ds_ptr, query_ds_ptr, conf, bitset_view);

                            // fp32 candidate
                            printf(
                                "\nProcessing EMBList CARDINAL_TIERED fp32 for %s distance, dim=%d, nrows=%d, %d%% "
                                "points filtered out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file =
                                test_emb_list_index<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                    params, conf, mv_only_enable, bitset_view, true);
                            index_files.emplace_back(index_file);

                            // fp16 candidate
                            printf(
                                "\nProcessing EMBList CARDINAL_TIERED fp16 for %s distance, dim=%d, nrows=%d, %d%% "
                                "points filtered out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file =
                                test_emb_list_index<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                    params, conf, mv_only_enable, bitset_view, true);
                            index_files.emplace_back(index_file);

                            // bf16 candidate
                            printf(
                                "\nProcessing EMBList CARDINAL_TIERED bf16 for %s distance, dim=%d, nrows=%d, %d%% "
                                "points filtered out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file =
                                test_emb_list_index<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                    params, conf, mv_only_enable, bitset_view, true);
                            index_files.emplace_back(index_file);

                            if (!index_support_int8(conf)) {
                                // int8 candidate
                                printf(
                                    "\nProcessing EMBList CARDINAL_TIERED int8 for %s distance, dim=%d, nrows=%d, %d%% "
                                    "points filtered out\n",
                                    DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::int8>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);
                            }
                            std::remove(get_index_name<knowhere::fp32>(ann_test_name_, index_type, params).c_str());
                            std::remove(get_index_name<knowhere::fp16>(ann_test_name_, index_type, params).c_str());
                            std::remove(get_index_name<knowhere::bf16>(ann_test_name_, index_type, params).c_str());
                            if (index_support_int8(conf)) {
                                std::remove(get_index_name<knowhere::int8>(ann_test_name_, index_type, params).c_str());
                            }
                        }
                        for (auto index : index_files) {
                            std::remove(index.c_str());
                        }
                    }
                }
            }
        }
    }
#endif

    SECTION("HNSW SQ") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW_SQ;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json base_conf = default_conf;
                    base_conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    base_conf[knowhere::meta::DIM] = dim;
                    base_conf[knowhere::meta::ROWS] = nb;
                    base_conf[knowhere::meta::INDEX_TYPE] = index_type;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(params);
                    // vector_id -> emb_list_id
                    // [0...9] -> 0, [10...19] -> 1, [20...29] -> 2, ...
                    int each_el_len = 10;
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListDataSet(nb, dim, rng_seed, each_el_len);
                    // emb_list_id -> partition_id
                    // [0, 3, 6, ...] -> 0, [1, 4, 7, ...] -> 1, [2, 5, 8, ...] -> 2, ...
                    int partition_num = 3;
                    printf("num_el: %d, each_el_len: %d, partition_num: %d\n", num_el, each_el_len, partition_num);
                    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info =
                        GenerateScalarInfoWithStep(nb, partition_num, each_el_len);

                    for (const bool mv_only_enable : MV_ONLYs) {
#ifdef KNOWHERE_WITH_CARDINAL
                        if (mv_only_enable) {
                            continue;
                        }
#endif
                        printf("with mv only enabled : %d\n", mv_only_enable);
                        if (mv_only_enable) {
                            default_ds_ptr->Set(knowhere::meta::SCALAR_INFO, scalar_info);
                        }

                        std::vector<std::string> index_files;
                        std::string index_file;

                        // test various bitset rates
                        for (const float bitset_rate : BITSET_RATES) {
                            printf("bitset_rate: %f\n", bitset_rate);
                            const std::vector<uint8_t> bitset_data = GenerateBitsetByPartition(
                                num_el, 1.0f - bitset_rate, mv_only_enable ? partition_num : 1);

                            // initialize bitset_view.
                            // provide a default one if nbits_set == 0
                            knowhere::BitsetView bitset_view = nullptr;
                            if (bitset_rate != 0.0f || mv_only_enable) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), num_el, num_el * bitset_rate);
                            }

                            // get a golden result
                            // auto golden_result = golden_index.Search(query_ds_ptr, conf, bitset_view);
                            auto golden_result = knowhere::BruteForce::Search<knowhere::fp32>(
                                default_ds_ptr, query_ds_ptr, base_conf, bitset_view);

                            // go SQ
                            for (size_t i_sq_type = 0; i_sq_type < SQ_TYPES.size(); i_sq_type++) {
                                knowhere::Json conf = base_conf;
                                const std::string sq_type = SQ_TYPES[i_sq_type];
                                conf[knowhere::indexparam::SQ_TYPE] = sq_type;

                                std::vector<int32_t> params = {(int)distance_type, dim, nb, (int)i_sq_type};

                                // fp32 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,SQ(%s) fp32 for %s distance, dim=%d, nrows=%d, %d%% "
                                    "points "
                                    "filtered "
                                    "out\n",
                                    sq_type.c_str(), DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::fp32>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                // fp16 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,SQ(%s) fp16 for %s distance, dim=%d, nrows=%d, %d%% "
                                    "points "
                                    "filtered "
                                    "out\n",
                                    sq_type.c_str(), DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::fp16>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                // bf16 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,SQ(%s) bf16 for %s distance, dim=%d, nrows=%d, %d%% "
                                    "points "
                                    "filtered "
                                    "out\n",
                                    sq_type.c_str(), DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::bf16>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                if (index_support_int8(conf)) {
                                    // int8 candidate
                                    printf(
                                        "\nProcessing EMBList HNSW,SQ(%s) int8 for %s distance, dim=%d, nrows=%d, %d%% "
                                        "points "
                                        "filtered "
                                        "out\n",
                                        sq_type.c_str(), DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                        int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::int8>(default_ds_ptr, query_ds_ptr,
                                                                                     golden_result.value(), params,
                                                                                     conf, mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }

                                // test refines for FP32
                                {
                                    const auto& allowed_refs = SQ_ALLOWED_REFINES_FP32[sq_type];
                                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size();
                                         allowed_ref_idx++) {
                                        auto conf_refine = conf;
                                        conf_refine["refine"] = true;
                                        conf_refine["refine_k"] = 1.5;

                                        const std::string allowed_ref = allowed_refs[allowed_ref_idx];
                                        conf_refine["refine_type"] = allowed_ref;

                                        std::vector<int32_t> params_refine = {(int)distance_type, dim, nb,
                                                                              (int)i_sq_type, (int)allowed_ref_idx};

                                        printf(
                                            "\nProcessing EMBList HNSW,SQ(%s) with %s refine, fp32 for %s distance, "
                                            "dim=%d, "
                                            "nrows=%d, %d%% points filtered out\n",
                                            sq_type.c_str(), allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(),
                                            dim, nb, int(bitset_rate * 100));

                                        index_file = test_emb_list_index<knowhere::fp32>(
                                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine,
                                            conf_refine, mv_only_enable, bitset_view);
                                        index_files.emplace_back(index_file);
                                    }
                                }

                                // test refines for FP16
                                {
                                    const auto& allowed_refs = SQ_ALLOWED_REFINES_FP16[sq_type];
                                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size();
                                         allowed_ref_idx++) {
                                        auto conf_refine = conf;
                                        conf_refine["refine"] = true;
                                        conf_refine["refine_k"] = 1.5;

                                        const std::string allowed_ref = allowed_refs[allowed_ref_idx];
                                        conf_refine["refine_type"] = allowed_ref;

                                        std::vector<int32_t> params_refine = {(int)distance_type, dim, nb,
                                                                              (int)i_sq_type, (int)allowed_ref_idx};

                                        printf(
                                            "\nProcessing EMBList HNSW,SQ(%s) with %s refine, fp16 for %s distance, "
                                            "dim=%d, "
                                            "nrows=%d, %d%% points filtered out\n",
                                            sq_type.c_str(), allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(),
                                            dim, nb, int(bitset_rate * 100));

                                        index_file = test_emb_list_index<knowhere::fp16>(
                                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine,
                                            conf_refine, mv_only_enable, bitset_view);
                                        index_files.emplace_back(index_file);
                                    }
                                }

                                // test refines for BF16
                                {
                                    const auto& allowed_refs = SQ_ALLOWED_REFINES_BF16[sq_type];
                                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size();
                                         allowed_ref_idx++) {
                                        auto conf_refine = conf;
                                        conf_refine["refine"] = true;
                                        conf_refine["refine_k"] = 1.5;

                                        const std::string allowed_ref = allowed_refs[allowed_ref_idx];
                                        conf_refine["refine_type"] = allowed_ref;

                                        std::vector<int32_t> params_refine = {(int)distance_type, dim, nb,
                                                                              (int)i_sq_type, (int)allowed_ref_idx};

                                        printf(
                                            "\nProcessing EMBList HNSW,SQ(%s) with %s refine, bf16 for %s distance, "
                                            "dim=%d, "
                                            "nrows=%d, %d%% points filtered out\n",
                                            sq_type.c_str(), allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(),
                                            dim, nb, int(bitset_rate * 100));

                                        index_file = test_emb_list_index<knowhere::bf16>(
                                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine,
                                            conf_refine, mv_only_enable, bitset_view);
                                        index_files.emplace_back(index_file);
                                    }
                                }
                            }

                            for (auto index : index_files) {
                                std::remove(index.c_str());
                            }
                        }
                    }
                }
            }
        }
    }

    SECTION("HNSW PQ") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW_PQ;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : {16}) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json base_conf = default_conf;
                    base_conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    base_conf[knowhere::meta::DIM] = dim;
                    base_conf[knowhere::meta::ROWS] = nb;
                    base_conf[knowhere::meta::INDEX_TYPE] = index_type;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(params);
                    // vector_id -> emb_list_id
                    // [0...9] -> 0, [10...19] -> 1, [20...29] -> 2, ...
                    int each_el_len = 10;
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListDataSet(nb, dim, rng_seed, each_el_len);
                    // emb_list_id -> partition_id
                    // [0, 3, 6, ...] -> 0, [1, 4, 7, ...] -> 1, [2, 5, 8, ...] -> 2, ...
                    int partition_num = 3;
                    printf("num_el: %d, each_el_len: %d, partition_num: %d\n", num_el, each_el_len, partition_num);
                    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info =
                        GenerateScalarInfoWithStep(nb, partition_num, each_el_len);

                    // accelerate the test by only testing with mv_only_enable = false
                    for (const bool mv_only_enable : {false}) {
#ifdef KNOWHERE_WITH_CARDINAL
                        if (mv_only_enable) {
                            continue;
                        }
#endif
                        printf("with mv only enabled : %d\n", mv_only_enable);
                        if (mv_only_enable) {
                            default_ds_ptr->Set(knowhere::meta::SCALAR_INFO, scalar_info);
                        }

                        std::vector<std::string> index_files;
                        std::string index_file;

                        // test various bitset rates
                        for (const float bitset_rate : BITSET_RATES) {
                            printf("bitset_rate: %f\n", bitset_rate);
                            const std::vector<uint8_t> bitset_data = GenerateBitsetByPartition(
                                num_el, 1.0f - bitset_rate, mv_only_enable ? partition_num : 1);

                            // initialize bitset_view.
                            // provide a default one if nbits_set == 0
                            knowhere::BitsetView bitset_view = nullptr;
                            if (bitset_rate != 0.0f || mv_only_enable) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), num_el, num_el * bitset_rate);
                            }

                            // get a golden result
                            // auto golden_result = golden_index.Search(query_ds_ptr, conf, bitset_view);
                            auto golden_result = knowhere::BruteForce::Search<knowhere::fp32>(
                                default_ds_ptr, query_ds_ptr, base_conf, bitset_view);

                            // go SQ
                            for (size_t nbits_type = 0; nbits_type < NBITS.size(); nbits_type++) {
                                const int pq_m = 8;

                                knowhere::Json conf = base_conf;
                                conf[knowhere::indexparam::NBITS] = NBITS[nbits_type];
                                conf[knowhere::indexparam::M] = pq_m;

                                std::vector<int32_t> params = {(int)distance_type, dim, nb, pq_m, (int)nbits_type};

                                // fp32 candidate
                                printf(
                                    "\nProcessing HNSW,PQ%dx%d fp32 for %s distance, dim=%d, nrows=%d, %d%% points "
                                    "filtered out\n",
                                    pq_m, NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::fp32>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                // fp16 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,PQ%dx%d fp16 for %s distance, dim=%d, nrows=%d, %d%% "
                                    "points "
                                    "filtered "
                                    "out\n",
                                    pq_m, NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::fp16>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                // bf16 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,PQ%dx%d bf16 for %s distance, dim=%d, nrows=%d, %d%% "
                                    "points "
                                    "filtered "
                                    "out\n",
                                    pq_m, NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::bf16>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                if (index_support_int8(conf)) {
                                    // int8 candidate
                                    printf(
                                        "\nProcessing EMBList HNSW,PQ%dx%d int8 for %s distance, dim=%d, nrows=%d, "
                                        "%d%% "
                                        "points "
                                        "filtered "
                                        "out\n",
                                        pq_m, NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                        int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::int8>(default_ds_ptr, query_ds_ptr,
                                                                                     golden_result.value(), params,
                                                                                     conf, mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }

                                // test refines for FP32
                                for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP32.size();
                                     allowed_ref_idx++) {
                                    auto conf_refine = conf;
                                    conf_refine["refine"] = true;
                                    conf_refine["refine_k"] = 1.5;

                                    const std::string allowed_ref = PQ_ALLOWED_REFINES_FP32[allowed_ref_idx];
                                    conf_refine["refine_type"] = allowed_ref;

                                    std::vector<int32_t> params_refine = {
                                        (int)distance_type, dim, nb, pq_m, (int)nbits_type, (int)allowed_ref_idx};

                                    printf(
                                        "\nProcessing EMBList HNSW,PQ%dx%d with %s refine, fp32 for %s distance, "
                                        "dim=%d, "
                                        "nrows=%d, %d%% points filtered out\n",
                                        pq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                        DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::fp32>(
                                        default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine,
                                        mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }

                                // test refines for FP16
                                for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP16.size();
                                     allowed_ref_idx++) {
                                    auto conf_refine = conf;
                                    conf_refine["refine"] = true;
                                    conf_refine["refine_k"] = 1.5;

                                    const std::string allowed_ref = PQ_ALLOWED_REFINES_FP16[allowed_ref_idx];
                                    conf_refine["refine_type"] = allowed_ref;

                                    std::vector<int32_t> params_refine = {
                                        (int)distance_type, dim, nb, pq_m, (int)nbits_type, (int)allowed_ref_idx};

                                    printf(
                                        "\nProcessing EMBList HNSW,PQ%dx%d with %s refine, fp16 for %s distance, "
                                        "dim=%d, "
                                        "nrows=%d, %d%% points filtered out\n",
                                        pq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                        DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::fp16>(
                                        default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine,
                                        mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }

                                // test refines for BF16
                                for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_BF16.size();
                                     allowed_ref_idx++) {
                                    auto conf_refine = conf;
                                    conf_refine["refine"] = true;
                                    conf_refine["refine_k"] = 1.5;

                                    const std::string allowed_ref = PQ_ALLOWED_REFINES_BF16[allowed_ref_idx];
                                    conf_refine["refine_type"] = allowed_ref;

                                    std::vector<int32_t> params_refine = {
                                        (int)distance_type, dim, nb, pq_m, (int)nbits_type, (int)allowed_ref_idx};

                                    printf(
                                        "\nProcessing EMBList HNSW,PQ%dx%d with %s refine, bf16 for %s distance, "
                                        "dim=%d, "
                                        "nrows=%d, %d%% points filtered out\n",
                                        pq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                        DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::bf16>(
                                        default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine,
                                        mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }
                            }
                        }

                        for (auto index : index_files) {
                            std::remove(index.c_str());
                        }
                    }
                }
            }
        }
    }

    SECTION("HNSW PRQ") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW_PRQ;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : {16}) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json base_conf = default_conf;
                    base_conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    base_conf[knowhere::meta::DIM] = dim;
                    base_conf[knowhere::meta::ROWS] = nb;
                    base_conf[knowhere::meta::INDEX_TYPE] = index_type;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(params);
                    // vector_id -> emb_list_id
                    // [0...9] -> 0, [10...19] -> 1, [20...29] -> 2, ...
                    int each_el_len = 10;
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListDataSet(nb, dim, rng_seed, each_el_len);
                    // emb_list_id -> partition_id
                    // [0, 3, 6, ...] -> 0, [1, 4, 7, ...] -> 1, [2, 5, 8, ...] -> 2, ...
                    int partition_num = 3;
                    printf("num_el: %d, each_el_len: %d, partition_num: %d\n", num_el, each_el_len, partition_num);
                    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info =
                        GenerateScalarInfoWithStep(nb, partition_num, each_el_len);

                    // accelerate the test by only testing with mv_only_enable = false
                    for (const bool mv_only_enable : {false}) {
#ifdef KNOWHERE_WITH_CARDINAL
                        if (mv_only_enable) {
                            continue;
                        }
#endif
                        printf("with mv only enabled : %d\n", mv_only_enable);
                        if (mv_only_enable) {
                            default_ds_ptr->Set(knowhere::meta::SCALAR_INFO, scalar_info);
                        }

                        std::vector<std::string> index_files;
                        std::string index_file;

                        // test various bitset rates
                        for (const float bitset_rate : BITSET_RATES) {
                            printf("bitset_rate: %f\n", bitset_rate);
                            const std::vector<uint8_t> bitset_data = GenerateBitsetByPartition(
                                num_el, 1.0f - bitset_rate, mv_only_enable ? partition_num : 1);

                            // initialize bitset_view.
                            // provide a default one if nbits_set == 0
                            knowhere::BitsetView bitset_view = nullptr;
                            if (bitset_rate != 0.0f || mv_only_enable) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), num_el, num_el * bitset_rate);
                            }

                            // get a golden result
                            // auto golden_result = golden_index.Search(query_ds_ptr, conf, bitset_view);
                            auto golden_result = knowhere::BruteForce::Search<knowhere::fp32>(
                                default_ds_ptr, query_ds_ptr, base_conf, bitset_view);

                            // go SQ
                            for (size_t nbits_type = 0; nbits_type < NBITS.size(); nbits_type++) {
                                const int prq_m = 4;
                                const int prq_num = 2;

                                knowhere::Json conf = base_conf;
                                conf[knowhere::indexparam::NBITS] = NBITS[nbits_type];
                                conf[knowhere::indexparam::M] = prq_m;
                                conf[knowhere::indexparam::PRQ_NUM] = prq_num;

                                std::vector<int32_t> params = {(int)distance_type, dim, nb, prq_m, prq_num,
                                                               (int)nbits_type};

                                // fp32 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,PRQ%dx%dx%d fp32 for %s distance, dim=%d, nrows=%d, "
                                    "%d%% points "
                                    "filtered out\n",
                                    prq_num, prq_m, NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::fp32>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                // fp16 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,PRQ%dx%dx%d fp16 for %s distance, dim=%d, nrows=%d, "
                                    "%d%% "
                                    "points "
                                    "filtered "
                                    "out\n",
                                    prq_num, prq_m, NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::fp16>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                // bf16 candidate
                                printf(
                                    "\nProcessing EMBList HNSW,PRQ%dx%dx%d bf16 for %s distance, dim=%d, nrows=%d, "
                                    "%d%% "
                                    "points "
                                    "filtered "
                                    "out\n",
                                    prq_num, prq_m, NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                    int(bitset_rate * 100));

                                index_file = test_emb_list_index<knowhere::bf16>(default_ds_ptr, query_ds_ptr,
                                                                                 golden_result.value(), params, conf,
                                                                                 mv_only_enable, bitset_view);
                                index_files.emplace_back(index_file);

                                if (index_support_int8(conf)) {
                                    // int8 candidate
                                    printf(
                                        "\nProcessing EMBList HNSW,PRQ%dx%dx%d int8 for %s distance, dim=%d, nrows=%d, "
                                        "%d%% "
                                        "points "
                                        "filtered "
                                        "out\n",
                                        prq_num, prq_m, NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim,
                                        nb, int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::int8>(default_ds_ptr, query_ds_ptr,
                                                                                     golden_result.value(), params,
                                                                                     conf, mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }

                                // test refines for FP32
                                for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP32.size();
                                     allowed_ref_idx++) {
                                    auto conf_refine = conf;
                                    conf_refine["refine"] = true;
                                    conf_refine["refine_k"] = 1.5;

                                    const std::string allowed_ref = PQ_ALLOWED_REFINES_FP32[allowed_ref_idx];
                                    conf_refine["refine_type"] = allowed_ref;

                                    std::vector<int32_t> params_refine = {
                                        (int)distance_type,  dim, nb, prq_m, prq_num, (int)nbits_type,
                                        (int)allowed_ref_idx};

                                    printf(
                                        "\nProcessing EMBList HNSW,PRQ%dx%dx%d with %s refine, fp32 for %s distance, "
                                        "dim=%d, "
                                        "nrows=%d, %d%% points filtered out\n",
                                        prq_num, prq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                        DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::fp32>(
                                        default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine,
                                        mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }

                                // test refines for FP16
                                for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP16.size();
                                     allowed_ref_idx++) {
                                    auto conf_refine = conf;
                                    conf_refine["refine"] = true;
                                    conf_refine["refine_k"] = 1.5;

                                    const std::string allowed_ref = PQ_ALLOWED_REFINES_FP16[allowed_ref_idx];
                                    conf_refine["refine_type"] = allowed_ref;

                                    std::vector<int32_t> params_refine = {
                                        (int)distance_type,  dim, nb, prq_m, prq_num, (int)nbits_type,
                                        (int)allowed_ref_idx};

                                    printf(
                                        "\nProcessing EMBList HNSW,PRQ%dx%dx%d with %s refine, fp16 for %s distance, "
                                        "dim=%d, "
                                        "nrows=%d, %d%% points filtered out\n",
                                        prq_num, prq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                        DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::fp16>(
                                        default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine,
                                        mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }

                                // test refines for BF16
                                for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_BF16.size();
                                     allowed_ref_idx++) {
                                    auto conf_refine = conf;
                                    conf_refine["refine"] = true;
                                    conf_refine["refine_k"] = 1.5;

                                    const std::string allowed_ref = PQ_ALLOWED_REFINES_BF16[allowed_ref_idx];
                                    conf_refine["refine_type"] = allowed_ref;

                                    std::vector<int32_t> params_refine = {
                                        (int)distance_type,  dim, nb, prq_m, prq_num, (int)nbits_type,
                                        (int)allowed_ref_idx};

                                    printf(
                                        "\nProcessing EMBList HNSW,PRQ%dx%dx%d with %s refine, bf16 for %s distance, "
                                        "dim=%d, "
                                        "nrows=%d, %d%% points filtered out\n",
                                        prq_num, prq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                        DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                    index_file = test_emb_list_index<knowhere::bf16>(
                                        default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine,
                                        mv_only_enable, bitset_view);
                                    index_files.emplace_back(index_file);
                                }
                            }
                        }

                        for (auto index : index_files) {
                            std::remove(index.c_str());
                        }
                    }
                }
            }
        }
    }

    SECTION("IVFFLAT") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
        // ivfflat emb list search need direct_map, which only generate when deserializing index.
        bool test_load_index_only = true;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate query dataset
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(params);
                    int each_el_len = 10;
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListDataSet(nb, dim, rng_seed, each_el_len);

                    std::vector<std::string> index_files;
                    std::string index_file;

                    for (const float bitset_rate : BITSET_RATES) {
                        printf("bitset_rate: %f\n", bitset_rate);
                        const std::vector<uint8_t> bitset_data =
                            GenerateBitsetByPartition(num_el, 1.0f - bitset_rate, 1);
                        knowhere::BitsetView bitset_view = nullptr;
                        if (bitset_rate != 0.0f) {
                            bitset_view = knowhere::BitsetView(bitset_data.data(), num_el);
                        }
                        auto golden_result = knowhere::BruteForce::Search<knowhere::fp32>(default_ds_ptr, query_ds_ptr,
                                                                                          conf, bitset_view);
                        // only test fp32
                        printf(
                            "Processing EMBList IVFFLAT fp32 for %s distance, dim=%d, nrows=%d, %d%% points filtered "
                            "out\n",
                            DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));
                        index_file =
                            test_emb_list_index<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                params, conf, false, bitset_view, test_load_index_only);
                        index_files.emplace_back(index_file);
                    }

                    for (auto index : index_files) {
                        std::remove(index.c_str());
                    }
                }
            }
        }
    }
}

TEST_CASE("Search for EMBList Indices (Binary)", "Benchmark and validation on binary vectors") {
    const std::vector<std::string> DISTANCE_TYPES = {"MAX_SIM_HAMMING", "MAX_SIM_JACCARD"};

    const std::vector<int32_t> DIMS = {32};
    const std::vector<int32_t> NBS = {256};
    const int32_t NQ = 10;
    const int32_t TOPK = 16;

    const std::vector<bool> MV_ONLYs = {false, true};

    const std::vector<float> BITSET_RATES = {0.0f, 0.5f, 0.95f, 1.0f};

    knowhere::Json default_conf;

    default_conf[knowhere::indexparam::HNSW_M] = 16;
    default_conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    default_conf[knowhere::indexparam::EF] = 64;
    default_conf[knowhere::meta::TOPK] = TOPK;
    default_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 3.0f;

    // vector_id -> emb_list_id
    // [0...9] -> 0; [10...19] -> 1; [20...29] -> 2; ...
    int each_el_len = 10;
    // emb_list_id -> partition_id
    // [0, 3, 6, ...] -> 0; [1, 4, 7, ...] -> 1; [2, 5, 8, ...] -> 2
    int partition_num = 3;

    SECTION("HNSW FLAT") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate query dataset
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListBinDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    // generate base dataset
                    std::vector<int32_t> params = {(int)distance_type, dim, nb};
                    const uint64_t rng_seed = get_params_hash(params);
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListBinDataSet(nb, dim, rng_seed, each_el_len);

                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    printf("conf: %s\n", conf.dump().c_str());

                    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info =
                        GenerateScalarInfoWithStep(nb, partition_num, each_el_len);

                    for (const bool mv_only_enable : MV_ONLYs) {
                        printf("mv_only_enable: %d\n", mv_only_enable);
                        if (mv_only_enable) {
                            default_ds_ptr->Set(knowhere::meta::SCALAR_INFO, scalar_info);
                        }

                        // test various bitset rates
                        for (const float bitset_rate : BITSET_RATES) {
                            printf("bitset_rate: %f\n", bitset_rate);
                            const std::vector<uint8_t> bitset_data = GenerateBitsetByPartition(
                                num_el, 1.0f - bitset_rate, mv_only_enable ? partition_num : 1);

                            // initialize bitset_view.
                            // provide a default one if nbits_set == 0
                            knowhere::BitsetView bitset_view = nullptr;
                            if (bitset_rate != 0.0f || mv_only_enable) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), num_el);
                            }

                            // get a golden result
                            auto golden_result = knowhere::BruteForce::Search<knowhere::bin1>(
                                default_ds_ptr, query_ds_ptr, conf, bitset_view);

                            printf(
                                "\nProcessing EMBList HNSW,Flat bin1 for %s distance, dim=%d, nrows=%d, %d%% points "
                                "filtered "
                                "out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            auto index_file =
                                test_emb_list_index<knowhere::bin1>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                    params, conf, mv_only_enable, bitset_view);
                            std::remove(index_file.c_str());
                        }
                    }
                }
            }
        }
    }

#ifdef KNOWHERE_WITH_CARDINAL
    SECTION("CARDINAL_TIERED") {
        static const int64_t mb = 1024 * 1024;
        milvus::cachinglayer::Manager::ConfigureTieredStorage(
            {CacheWarmupPolicy::CacheWarmupPolicy_Disable, CacheWarmupPolicy::CacheWarmupPolicy_Disable,
             CacheWarmupPolicy::CacheWarmupPolicy_Disable, CacheWarmupPolicy::CacheWarmupPolicy_Disable},
            {1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb}, true, true, {10, false, 30});
        const std::string& index_type = knowhere::IndexEnum::INDEX_CARDINAL_TIERED;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate query dataset
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListBinDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    // generate base dataset
                    std::vector<int32_t> params = {(int)distance_type, dim, nb};
                    const uint64_t rng_seed = get_params_hash(params);
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListBinDataSet(nb, dim, rng_seed, each_el_len);

                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    printf("conf: %s\n", conf.dump().c_str());

                    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info =
                        GenerateScalarInfoWithStep(nb, partition_num, each_el_len);

                    for (const bool mv_only_enable : MV_ONLYs) {
                        printf("mv_only_enable: %d\n", mv_only_enable);
                        if (mv_only_enable) {
                            default_ds_ptr->Set(knowhere::meta::SCALAR_INFO, scalar_info);
                        }

                        // test various bitset rates
                        for (const float bitset_rate : BITSET_RATES) {
                            printf("bitset_rate: %f\n", bitset_rate);
                            const std::vector<uint8_t> bitset_data = GenerateBitsetByPartition(
                                num_el, 1.0f - bitset_rate, mv_only_enable ? partition_num : 1);

                            // initialize bitset_view.
                            // provide a default one if nbits_set == 0
                            knowhere::BitsetView bitset_view = nullptr;
                            if (bitset_rate != 0.0f || mv_only_enable) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), num_el);
                            }

                            // get a golden result
                            auto golden_result = knowhere::BruteForce::Search<knowhere::bin1>(
                                default_ds_ptr, query_ds_ptr, conf, bitset_view);

                            printf(
                                "\nProcessing EMBList CARDINAL_TIERED bin1 for %s distance, dim=%d, nrows=%d, %d%% "
                                "points filtered out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            auto index_file =
                                test_emb_list_index<knowhere::bin1>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                    params, conf, mv_only_enable, bitset_view, true);
                            std::remove(index_file.c_str());
                        }
                    }
                }
            }
        }
    }
#endif
}

TEST_CASE("Test with some empty emb list", "[empty_emb_list]") {
    // metrics to test
    const std::vector<std::string> DISTANCE_TYPES = {"MAX_SIM_IP", "MAX_SIM_L2", "MAX_SIM_COSINE"};

    // for unit tests
    const std::vector<int32_t> DIMS = {4};
    const std::vector<int32_t> NBS = {256};
    const int32_t NQ = 10;
    const int32_t TOPK = 16;

    // random bitset rates
    // 0.0 means unfiltered, 1.0 means all filtered out
    const std::vector<float> BITSET_RATES = {0.0f, 0.5f, 0.95f, 1.0f};

    // create base json config
    knowhere::Json default_conf;

    default_conf[knowhere::indexparam::HNSW_M] = 16;
    default_conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    default_conf[knowhere::indexparam::EF] = 64;
    default_conf[knowhere::meta::TOPK] = TOPK;
    default_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 3.0f;
    default_conf[knowhere::indexparam::NLIST] = 6;
    default_conf[knowhere::indexparam::NPROBE] = 6;

    int each_el_len = 10;

    SECTION("HNSW FLAT") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate query dataset
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;
                    // generate base dataset
                    std::vector<int32_t> params = {(int)distance_type, dim, nb};
                    const uint64_t rng_seed = get_params_hash(params);
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListDataSetWithSomeEmpty(nb, dim, rng_seed, each_el_len);

                    for (const float bitset_rate : BITSET_RATES) {
                        printf("bitset_rate: %f\n", bitset_rate);
                        const std::vector<uint8_t> bitset_data =
                            GenerateBitsetByPartition(num_el, 1.0f - bitset_rate, 1);
                        knowhere::BitsetView bitset_view = nullptr;
                        if (bitset_rate != 0.0f) {
                            bitset_view = knowhere::BitsetView(bitset_data.data(), num_el);
                        }

                        auto golden_result = knowhere::BruteForce::Search<knowhere::fp32>(default_ds_ptr, query_ds_ptr,
                                                                                          conf, bitset_view);
                        printf(
                            "\nProcessing EMBList HNSW,Flat fp32 for %s distance, dim=%d, nrows=%d, %d%% points "
                            "filtered "
                            "out\n",
                            DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));
                        auto index_file = test_emb_list_index<knowhere::fp32>(
                            default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf, false, bitset_view);
                        std::remove(index_file.c_str());
                    }
                }
            }
        }
    }
}

template <typename DataType>
void
EmbListAddTest(const knowhere::DataSetPtr train_ds_in, const knowhere::DataSetPtr query_ds,
               const knowhere::MetricType metric, const knowhere::Json& conf, const size_t each_el_len) {
    auto train_ds = knowhere::ConvertToDataTypeIfNeeded<DataType>(train_ds_in);
    auto partition_num = 3;
    auto train_ds_list = SplitEmbListDataSet<DataType>(train_ds, partition_num, each_el_len);
    auto query = knowhere::ConvertToDataTypeIfNeeded<DataType>(query_ds);
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto rows = train_ds->GetRows();
    auto num_el = (rows + each_el_len - 1) / each_el_len;

    auto index = knowhere::IndexFactory::Instance().Create<DataType>(conf[knowhere::meta::INDEX_TYPE], version).value();
    for (size_t i = 0; i < train_ds_list.size(); i++) {
        auto& base = train_ds_list[i];
        if (i == 0) {
            REQUIRE(index.Build(base, conf, false) == knowhere::Status::success);
        } else {
            REQUIRE(index.Add(base, conf, false) == knowhere::Status::success);
        }
    }

    auto knn_gt = knowhere::BruteForce::Search<DataType>(train_ds, query, conf, nullptr);

    const auto bitset_percentages = {0.0f, 0.5f, 0.9f, 0.98f};
    for (const float percentage : bitset_percentages) {
        auto bitset_data = GenerateBitsetByPartition(num_el, 1 - percentage, 1);
        knowhere::BitsetView bitset(bitset_data.data(), num_el);
        auto knn_gt = knowhere::BruteForce::Search<DataType>(train_ds, query, conf, bitset);
        auto res = index.Search(query, conf, bitset);
        REQUIRE(res.has_value());
        float recall = GetKNNRecall(*knn_gt.value(), *res.value());
        printf("bitset_rate: %f, recall: %f\n", percentage, recall);
        REQUIRE(recall > 0.75f);
    }
}

TEST_CASE("Test growing with emb list", "[growing]") {
    const std::vector<std::string> DISTANCE_TYPES = {"MAX_SIM_IP", "MAX_SIM_L2", "MAX_SIM_COSINE"};
    const std::vector<int32_t> DIMS = {4};
    const std::vector<int32_t> NBS = {1000};
    const int32_t NQ = 10;
    const int32_t TOPK = 16;
    const int32_t each_el_len = 10;

    uint64_t seed = 42;

    SECTION("IVFFLAT_CC") {
        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    auto total_nb = nb * 2;
                    auto train_ds = GenEmbListDataSet(total_nb, dim, seed, each_el_len);
                    auto query_ds = GenQueryEmbListDataSet(NQ, dim, seed);
                    knowhere::Json conf;
                    conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::TOPK] = TOPK;
                    conf[knowhere::indexparam::NLIST] = 16;
                    conf[knowhere::indexparam::NPROBE] = 12;

                    EmbListAddTest<knowhere::fp32>(train_ds, query_ds, conf[knowhere::meta::METRIC_TYPE], conf,
                                                   each_el_len);
                }
            }
        }
    }
}

TEST_CASE("Test brute force search on chunk", "[on_chunk]") {
    const std::vector<int32_t> DIMS = {8};
    const std::vector<int32_t> NBS = {1000};
    const int32_t NQ = 10;
    const int32_t TOPK = 16;
    const int32_t each_el_len = 10;
    const std::vector<float> BITSET_RATES = {0.1f, 0.5f, 0.95f, 1.0f};

    knowhere::Json default_conf;
    default_conf[knowhere::meta::TOPK] = TOPK;

    uint64_t seed = 42;

    float CHUNK_RECALL_THRESHOLD = 0.999f;

    SECTION("Dense Vector Search on Chunk") {
        const std::vector<std::string> DISTANCE_TYPES = {"L2", "IP", "COSINE"};
        for (const auto& distance_type : DISTANCE_TYPES) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    auto train_ds = GenEmbListDataSet(nb, dim, seed, each_el_len);
                    auto query_ds = GenDataSet(NQ, dim, seed);

                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = distance_type;
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;

                    auto golden_result =
                        knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

                    // use the same seed to generate same float vectors
                    auto chunk_train_ds = GenChunkDataSet(nb, dim, seed, each_el_len);

                    auto chunk_result =
                        knowhere::BruteForce::Search<knowhere::fp32>(chunk_train_ds, query_ds, conf, nullptr);
                    REQUIRE(chunk_result.has_value());
                    float recall = GetKNNRecall(*golden_result.value(), *chunk_result.value());
                    printf("recall: %f\n", recall);
                    REQUIRE(recall > CHUNK_RECALL_THRESHOLD);

                    // with filter
                    for (const float bitset_rate : BITSET_RATES) {
                        const int32_t filter_out_bits = nb * bitset_rate;
                        const std::vector<uint8_t> bitset_data = GenerateBitsetWithRandomTbitsSet(nb, filter_out_bits);
                        knowhere::BitsetView bitset_view =
                            knowhere::BitsetView(bitset_data.data(), nb, filter_out_bits);

                        auto golden_result =
                            knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, bitset_view);
                        auto chunk_result =
                            knowhere::BruteForce::Search<knowhere::fp32>(chunk_train_ds, query_ds, conf, bitset_view);
                        REQUIRE(chunk_result.has_value());
                        float recall = GetKNNRecall(*golden_result.value(), *chunk_result.value());
                        printf("bitset_rate: %f, recall: %f\n", bitset_rate, recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }

                    // with datatype bf16
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(chunk_train_ds);

                        auto golden_result_typed =
                            knowhere::BruteForce::Search<knowhere::bf16>(train_ds_typed, query_ds_typed, conf, nullptr);
                        auto chunk_result_typed = knowhere::BruteForce::Search<knowhere::bf16>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_typed.has_value());
                        float recall = GetKNNRecall(*golden_result_typed.value(), *chunk_result_typed.value());
                        printf("datatype [bf16], recall: %f\n", recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }

                    // with datatype fp16
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(chunk_train_ds);

                        auto golden_result_typed =
                            knowhere::BruteForce::Search<knowhere::bf16>(train_ds_typed, query_ds_typed, conf, nullptr);
                        auto chunk_result_typed = knowhere::BruteForce::Search<knowhere::bf16>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_typed.has_value());
                        float recall = GetKNNRecall(*golden_result_typed.value(), *chunk_result_typed.value());
                        printf("datatype [fp16], recall: %f\n", recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }

                    // with datatype int8
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(chunk_train_ds);

                        auto golden_result_typed =
                            knowhere::BruteForce::Search<knowhere::int8>(train_ds_typed, query_ds_typed, conf, nullptr);
                        auto chunk_result_typed = knowhere::BruteForce::Search<knowhere::int8>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_typed.has_value());
                        float recall = GetKNNRecall(*golden_result_typed.value(), *chunk_result_typed.value());
                        printf("datatype [int8], recall: %f\n", recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }
                }
            }
        }
    }

    SECTION("Binary VectorSearch on Chunk") {
        const std::vector<std::string> DISTANCE_TYPES = {"JACCARD", "HAMMING"};
        for (const auto& distance_type : DISTANCE_TYPES) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    auto train_ds = GenEmbListBinDataSet(nb, dim, seed, each_el_len);
                    auto query_ds = GenBinDataSet(NQ, dim, seed);

                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = distance_type;
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;

                    auto golden_result =
                        knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, conf, nullptr);

                    // use the same seed to generate same float vectors
                    auto chunk_train_ds = GenChunkBinDataSet(nb, dim, seed, each_el_len);

                    auto chunk_result =
                        knowhere::BruteForce::Search<knowhere::bin1>(chunk_train_ds, query_ds, conf, nullptr);
                    REQUIRE(chunk_result.has_value());
                    float recall = GetKNNRecall(*golden_result.value(), *chunk_result.value());
                    printf("recall: %f\n", recall);
                    REQUIRE(recall > CHUNK_RECALL_THRESHOLD);

                    // with filter
                    for (const float bitset_rate : BITSET_RATES) {
                        const int32_t filter_out_bits = nb * bitset_rate;
                        const std::vector<uint8_t> bitset_data = GenerateBitsetWithRandomTbitsSet(nb, filter_out_bits);
                        knowhere::BitsetView bitset_view =
                            knowhere::BitsetView(bitset_data.data(), nb, filter_out_bits);

                        auto golden_result =
                            knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, conf, bitset_view);
                        auto chunk_result =
                            knowhere::BruteForce::Search<knowhere::bin1>(chunk_train_ds, query_ds, conf, bitset_view);
                        REQUIRE(chunk_result.has_value());
                        float recall = GetKNNRecall(*golden_result.value(), *chunk_result.value());
                        printf("bitset_rate: %f, recall: %f\n", bitset_rate, recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }
                }
            }
        }
    }

    SECTION("Dense EmbList Search on Chunk") {
        const std::vector<std::string> DISTANCE_TYPES = {"MAX_SIM_IP", "MAX_SIM_L2", "MAX_SIM_COSINE"};
        for (const auto& distance_type : DISTANCE_TYPES) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    auto train_ds = GenEmbListDataSet(nb, dim, seed, each_el_len);
                    auto query_ds = GenEmbListDataSet(NQ, dim, seed);
                    auto num_el = (nb + each_el_len - 1) / each_el_len;

                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = distance_type;
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;

                    auto golden_result =
                        knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, nullptr);

                    // use the same seed to generate same float vectors
                    auto chunk_train_ds = GenChunkDataSet(nb, dim, seed, each_el_len);

                    auto chunk_result =
                        knowhere::BruteForce::Search<knowhere::fp32>(chunk_train_ds, query_ds, conf, nullptr);
                    REQUIRE(chunk_result.has_value());
                    float recall = GetKNNRecall(*golden_result.value(), *chunk_result.value());
                    printf("recall: %f\n", recall);
                    REQUIRE(recall > CHUNK_RECALL_THRESHOLD);

                    // with filter
                    for (const float bitset_rate : BITSET_RATES) {
                        const int32_t filter_out_bits = num_el * bitset_rate;
                        const std::vector<uint8_t> bitset_data =
                            GenerateBitsetByPartition(num_el, 1.0f - bitset_rate, 1);
                        knowhere::BitsetView bitset_view =
                            knowhere::BitsetView(bitset_data.data(), num_el, filter_out_bits);

                        auto golden_result =
                            knowhere::BruteForce::Search<knowhere::fp32>(train_ds, query_ds, conf, bitset_view);
                        auto chunk_result =
                            knowhere::BruteForce::Search<knowhere::fp32>(chunk_train_ds, query_ds, conf, bitset_view);
                        REQUIRE(chunk_result.has_value());
                        float recall = GetKNNRecall(*golden_result.value(), *chunk_result.value());
                        printf("bitset_rate: %f, recall: %f\n", bitset_rate, recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }

                    // with datatype bf16
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(chunk_train_ds);

                        auto golden_result_typed =
                            knowhere::BruteForce::Search<knowhere::bf16>(train_ds_typed, query_ds_typed, conf, nullptr);
                        auto chunk_result_typed = knowhere::BruteForce::Search<knowhere::bf16>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_typed.has_value());
                        float recall = GetKNNRecall(*golden_result_typed.value(), *chunk_result_typed.value());
                        printf("datatype [bf16], recall: %f\n", recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }

                    // with datatype fp16
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(chunk_train_ds);

                        auto golden_result_typed =
                            knowhere::BruteForce::Search<knowhere::bf16>(train_ds_typed, query_ds_typed, conf, nullptr);
                        auto chunk_result_typed = knowhere::BruteForce::Search<knowhere::bf16>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_typed.has_value());
                        float recall = GetKNNRecall(*golden_result_typed.value(), *chunk_result_typed.value());
                        printf("datatype [fp16], recall: %f\n", recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }

                    // with datatype int8
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(chunk_train_ds);

                        auto golden_result_typed =
                            knowhere::BruteForce::Search<knowhere::int8>(train_ds_typed, query_ds_typed, conf, nullptr);
                        auto chunk_result_typed = knowhere::BruteForce::Search<knowhere::int8>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_typed.has_value());
                        float recall = GetKNNRecall(*golden_result_typed.value(), *chunk_result_typed.value());
                        printf("datatype [int8], recall: %f\n", recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }
                }
            }
        }
    }

    SECTION("Binary EmbList Search on Chunk") {
        const std::vector<std::string> DISTANCE_TYPES = {"MAX_SIM_HAMMING", "MAX_SIM_JACCARD"};

        for (const auto& distance_type : DISTANCE_TYPES) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    auto train_ds = GenEmbListBinDataSet(nb, dim, seed, each_el_len);
                    auto query_ds = GenEmbListBinDataSet(NQ, dim, seed);
                    auto num_el = (nb + each_el_len - 1) / each_el_len;

                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = distance_type;
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;

                    auto golden_result =
                        knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, conf, nullptr);

                    // use the same seed to generate same float vectors
                    auto chunk_train_ds = GenChunkBinDataSet(nb, dim, seed, each_el_len);

                    auto chunk_result =
                        knowhere::BruteForce::Search<knowhere::bin1>(chunk_train_ds, query_ds, conf, nullptr);
                    REQUIRE(chunk_result.has_value());
                    float recall = GetKNNRecall(*golden_result.value(), *chunk_result.value());
                    printf("recall: %f\n", recall);
                    REQUIRE(recall > CHUNK_RECALL_THRESHOLD);

                    // with filter
                    for (const float bitset_rate : BITSET_RATES) {
                        const int32_t filter_out_bits = num_el * bitset_rate;
                        const std::vector<uint8_t> bitset_data =
                            GenerateBitsetByPartition(num_el, 1.0f - bitset_rate, 1);
                        knowhere::BitsetView bitset_view =
                            knowhere::BitsetView(bitset_data.data(), num_el, filter_out_bits);

                        auto golden_result =
                            knowhere::BruteForce::Search<knowhere::bin1>(train_ds, query_ds, conf, bitset_view);
                        auto chunk_result =
                            knowhere::BruteForce::Search<knowhere::bin1>(chunk_train_ds, query_ds, conf, bitset_view);
                        REQUIRE(chunk_result.has_value());
                        float recall = GetKNNRecall(*golden_result.value(), *chunk_result.value());
                        printf("bitset_rate: %f, recall: %f\n", bitset_rate, recall);
                        REQUIRE(recall > CHUNK_RECALL_THRESHOLD);
                    }
                }
            }
        }
    }
}

TEST_CASE("Test brute force anniterator on chunk", "[on_chunk]") {
    const std::vector<int32_t> DIMS = {8};
    const std::vector<int32_t> NBS = {1000};
    const int32_t NQ = 10;
    const int32_t TOPK = 16;
    const int32_t each_el_len = 10;
    const std::vector<float> BITSET_RATES = {0.1f, 0.5f, 0.95f, 1.0f};

    knowhere::Json default_conf;
    default_conf[knowhere::meta::TOPK] = TOPK;

    uint64_t seed = 42;

    SECTION("Dense AnnIterator on Chunk") {
        const std::vector<std::string> DISTANCE_TYPES = {"L2", "IP", "COSINE"};
        for (const auto& distance_type : DISTANCE_TYPES) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    auto train_ds = GenEmbListDataSet(nb, dim, seed, each_el_len);
                    auto query_ds = GenDataSet(NQ, dim, seed);
                    // use the same seed to generate same float vectors
                    auto chunk_train_ds = GenChunkDataSet(nb, dim, seed, each_el_len);

                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = distance_type;
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;

                    auto golden_result_iter_or =
                        knowhere::BruteForce::AnnIterator<knowhere::fp32>(train_ds, query_ds, conf, nullptr);
                    REQUIRE(golden_result_iter_or.has_value());
                    auto golden_result_iter = golden_result_iter_or.value();

                    auto chunk_result_iter_or =
                        knowhere::BruteForce::AnnIterator<knowhere::fp32>(chunk_train_ds, query_ds, conf, nullptr);
                    REQUIRE(chunk_result_iter_or.has_value());
                    auto chunk_result_iter = chunk_result_iter_or.value();

                    for (auto j = 0; j < NQ; ++j) {
                        REQUIRE(check_same_iterator(golden_result_iter[j], chunk_result_iter[j]));
                    }

                    // with filter
                    for (const float bitset_rate : BITSET_RATES) {
                        const int32_t filter_out_bits = nb * bitset_rate;
                        const std::vector<uint8_t> bitset_data = GenerateBitsetWithRandomTbitsSet(nb, filter_out_bits);
                        knowhere::BitsetView bitset_view =
                            knowhere::BitsetView(bitset_data.data(), nb, filter_out_bits);

                        auto golden_result_iter_or =
                            knowhere::BruteForce::AnnIterator<knowhere::fp32>(train_ds, query_ds, conf, bitset_view);
                        REQUIRE(golden_result_iter_or.has_value());
                        auto golden_result_iter = golden_result_iter_or.value();

                        auto chunk_result_iter_or = knowhere::BruteForce::AnnIterator<knowhere::fp32>(
                            chunk_train_ds, query_ds, conf, bitset_view);
                        REQUIRE(chunk_result_iter_or.has_value());
                        auto chunk_result_iter = chunk_result_iter_or.value();

                        for (auto j = 0; j < NQ; ++j) {
                            REQUIRE(check_same_iterator(golden_result_iter[j], chunk_result_iter[j]));
                        }
                    }

                    // with datatype bf16
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::bf16>(chunk_train_ds);

                        auto golden_result_iter_or = knowhere::BruteForce::AnnIterator<knowhere::bf16>(
                            train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(golden_result_iter_or.has_value());
                        auto golden_result_iter = golden_result_iter_or.value();

                        auto chunk_result_iter_or = knowhere::BruteForce::AnnIterator<knowhere::bf16>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_iter_or.has_value());
                        auto chunk_result_iter = chunk_result_iter_or.value();

                        for (auto j = 0; j < NQ; ++j) {
                            REQUIRE(check_same_iterator(golden_result_iter[j], chunk_result_iter[j]));
                        }
                    }

                    // with datatype fp16
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(chunk_train_ds);

                        auto golden_result_iter_or = knowhere::BruteForce::AnnIterator<knowhere::fp16>(
                            train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(golden_result_iter_or.has_value());
                        auto golden_result_iter = golden_result_iter_or.value();

                        auto chunk_result_iter_or = knowhere::BruteForce::AnnIterator<knowhere::fp16>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_iter_or.has_value());
                        auto chunk_result_iter = chunk_result_iter_or.value();

                        for (auto j = 0; j < NQ; ++j) {
                            REQUIRE(check_same_iterator(golden_result_iter[j], chunk_result_iter[j]));
                        }
                    }

                    // with datatype int8
                    {
                        auto train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(chunk_train_ds);
                        auto query_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(query_ds);
                        auto chunk_train_ds_typed = knowhere::ConvertToDataTypeIfNeeded<knowhere::int8>(chunk_train_ds);

                        auto golden_result_iter_or = knowhere::BruteForce::AnnIterator<knowhere::int8>(
                            train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(golden_result_iter_or.has_value());
                        auto golden_result_iter = golden_result_iter_or.value();

                        auto chunk_result_iter_or = knowhere::BruteForce::AnnIterator<knowhere::int8>(
                            chunk_train_ds_typed, query_ds_typed, conf, nullptr);
                        REQUIRE(chunk_result_iter_or.has_value());
                        auto chunk_result_iter = chunk_result_iter_or.value();

                        for (auto j = 0; j < NQ; ++j) {
                            REQUIRE(check_same_iterator(golden_result_iter[j], chunk_result_iter[j]));
                        }
                    }
                }
            }
        }
    }

    SECTION("Binary AnnIterator on Chunk") {
        const std::vector<std::string> DISTANCE_TYPES = {"JACCARD", "HAMMING"};

        for (const auto& distance_type : DISTANCE_TYPES) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    auto train_ds = GenEmbListBinDataSet(nb, dim, seed, each_el_len);
                    auto query_ds = GenBinDataSet(NQ, dim, seed);
                    // use the same seed to generate same float vectors
                    auto chunk_train_ds = GenChunkBinDataSet(nb, dim, seed, each_el_len);

                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = distance_type;
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;

                    auto golden_result_iter_or =
                        knowhere::BruteForce::AnnIterator<knowhere::bin1>(train_ds, query_ds, conf, nullptr);
                    REQUIRE(golden_result_iter_or.has_value());
                    auto golden_result_iter = golden_result_iter_or.value();

                    auto chunk_result_iter_or =
                        knowhere::BruteForce::AnnIterator<knowhere::bin1>(chunk_train_ds, query_ds, conf, nullptr);
                    REQUIRE(chunk_result_iter_or.has_value());
                    auto chunk_result_iter = chunk_result_iter_or.value();

                    for (auto j = 0; j < NQ; ++j) {
                        REQUIRE(check_same_iterator(golden_result_iter[j], chunk_result_iter[j]));
                    }

                    // with filter
                    for (const float bitset_rate : BITSET_RATES) {
                        const int32_t filter_out_bits = nb * bitset_rate;
                        const std::vector<uint8_t> bitset_data = GenerateBitsetWithRandomTbitsSet(nb, filter_out_bits);
                        knowhere::BitsetView bitset_view =
                            knowhere::BitsetView(bitset_data.data(), nb, filter_out_bits);

                        auto golden_result_iter_or =
                            knowhere::BruteForce::AnnIterator<knowhere::bin1>(train_ds, query_ds, conf, nullptr);
                        REQUIRE(golden_result_iter_or.has_value());
                        auto golden_result_iter = golden_result_iter_or.value();

                        auto chunk_result_iter_or =
                            knowhere::BruteForce::AnnIterator<knowhere::bin1>(chunk_train_ds, query_ds, conf, nullptr);
                        REQUIRE(chunk_result_iter_or.has_value());
                        auto chunk_result_iter = chunk_result_iter_or.value();

                        for (auto j = 0; j < NQ; ++j) {
                            REQUIRE(check_same_iterator(golden_result_iter[j], chunk_result_iter[j]));
                        }
                    }
                }
            }
        }
    }
}
