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

knowhere::DataSetPtr
GenEmbListDataSet(int rows, int dim, const uint64_t seed = 42, int each_el_len = 10) {
    std::mt19937 rng(seed);
    // use int type to cover test cases for fb16, bf16, int8
    std::uniform_int_distribution<> distrib(-100.0, 100.0);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) {
        ts[i] = (float)distrib(rng);
    }
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    auto ptr = std::make_unique<size_t[]>(size_t(rows / each_el_len) + 2);
    size_t i = 0;
    for (; i * each_el_len < (size_t)rows; i += 1) {
        ptr[i] = i * each_el_len;
    }
    ptr[i] = (size_t)rows;
    ds->SetLims(std::move(ptr));
    ds->SetIsOwner(true);
    return ds;
}

knowhere::DataSetPtr
GenQueryEmbListDataSet(int rows, int dim, const uint64_t seed = 42) {
    std::mt19937 rng(seed);
    // use int type to cover test cases for fb16, bf16, int8
    std::uniform_int_distribution<> distrib(-100.0, 100.0);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) {
        ts[i] = (float)distrib(rng);
    }
    auto ds = knowhere::GenDataSet(rows, dim, ts);
    auto ptr = std::make_unique<size_t[]>(int(sqrt(rows)) + 2);
    size_t i = 0;
    while (true) {
        if (i * i >= (size_t)rows) {
            break;
        }
        ptr[i] = i * i;
        i += 1;
    }
    ptr[i] = (size_t)rows;
    ds->SetLims(std::move(ptr));
    ds->SetIsOwner(true);
    return ds;
}

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

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
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
test_hnsw(const knowhere::DataSetPtr& default_ds_ptr, const knowhere::DataSetPtr& query_ds_ptr,
          const knowhere::DataSetPtr& golden_result, const std::vector<int32_t>& index_params,
          const knowhere::Json& conf, const bool mv_only_enable, const knowhere::BitsetView bitset_view) {
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
    auto result = index.Search(query_t_ds_ptr, conf, bitset_view);
    double search_elapsed = sw_search.elapsed();
    printf("search cost: %f\n", search_elapsed);

    auto result_loaded = index_loaded.Search(query_t_ds_ptr, conf, bitset_view);

    // calc recall
    auto recall = GetKNNRecall(*golden_result, *result.value());
    printf("recall: %f\n", recall);
    auto recall_loaded = GetKNNRecall(*golden_result, *result_loaded.value());
    REQUIRE(recall >= 0.8);
    REQUIRE(recall_loaded >= 0.8);
    REQUIRE(recall == recall_loaded);

    return index_file_name;
}

}  // namespace

TEST_CASE("Search for EMBList HNSW Indices", "Benchmark and validation") {
    // various constants and restrictions

    // metrics to test
    const std::vector<std::string> DISTANCE_TYPES = {"MAX_SIM"};

    // for unit tests
    const std::vector<int32_t> DIMS = {64};
    const std::vector<int32_t> NBS = {1000};
    const int32_t NQ = 10;
    const int32_t TOPK = 10;

    const std::vector<bool> MV_ONLYs = {false, true};

    // const std::vector<std::string> SQ_TYPES = {"SQ6", "SQ8", "BF16", "FP16"};

    // random bitset rates
    // 0.0 means unfiltered, 1.0 means all filtered out
    const std::vector<float> BITSET_RATES = {0.0f, 0.1f, 0.5f, 0.95f, 1.0f};

    // create base json config
    knowhere::Json default_conf;

    default_conf[knowhere::indexparam::HNSW_M] = 16;
    default_conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    default_conf[knowhere::indexparam::EF] = 64;
    default_conf[knowhere::meta::TOPK] = TOPK;
    default_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 1.0f;

    SECTION("FLAT") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_EMB_LIST_HNSW;
        const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

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

                    std::vector<int32_t> golden_params = {(int)distance_type, dim, nb};
                    std::vector<int32_t> params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(golden_params);
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
                                default_ds_ptr, query_ds_ptr, conf, bitset_view);

                            // fp32 candidate
                            printf(
                                "\nProcessing HNSW,Flat fp32 for %s distance, dim=%d, nrows=%d, %d%% points filtered "
                                "out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file = test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                   params, conf, mv_only_enable, bitset_view);
                            index_files.emplace_back(index_file);

                            // fp16 candidate
                            printf(
                                "\nProcessing HNSW,Flat fp16 for %s distance, dim=%d, nrows=%d, %d%% points filtered "
                                "out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file = test_hnsw<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                   params, conf, mv_only_enable, bitset_view);
                            index_files.emplace_back(index_file);

                            // bf16 candidate
                            printf(
                                "\nProcessing HNSW,Flat bf16 for %s distance, dim=%d, nrows=%d, %d%% points filtered "
                                "out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file = test_hnsw<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                   params, conf, mv_only_enable, bitset_view);
                            index_files.emplace_back(index_file);

                            if (index_support_int8(conf)) {
                                // int8 candidate
                                printf(
                                    "\nProcessing HNSW,Flat int8 for %s distance, dim=%d, nrows=%d, %d%% points "
                                    "filtered "
                                    "out\n",
                                    DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                                index_file =
                                    test_hnsw<knowhere::int8>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                              params, conf, mv_only_enable, bitset_view);
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
}
