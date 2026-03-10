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
#include "knowhere/index/emb_list_strategy.h"
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
    size_t number_of_imprecise_distances = 0;
    double max_abs_relative_error = 0;
    while (iter1->HasNext()) {
        REQUIRE(iter2->HasNext());
        auto [id1, dist1] = iter1->Next();
        auto [id2, dist2] = iter2->Next();
        count++;

        // Perform a precise match of ids
        if (id1 != id2) {
            return false;
        }

        // Perform an approximate match of distances.
        // Differences may occur because of a different compiler
        //   code in a regular distance computation code and a
        //   batch4 distance computation code for -ffast-math.
        if (dist1 != dist2) {
            number_of_imprecise_distances += 1;

            double relative_error = std::abs(((double)dist1 - (double)dist2) / (double)dist2);
            if (relative_error > 1e-6) {
                return false;
            }

            max_abs_relative_error = std::max(max_abs_relative_error, relative_error);
        }
    }
    printf("Total number of iterator->Next() calls: %ld\n", count);
    if (number_of_imprecise_distances > 0) {
        printf("Total number of imprecise distances: %ld\n", number_of_imprecise_distances);
        printf("Max abs relative error exponent: %f\n", std::log10(max_abs_relative_error));
    }
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

    SECTION("HNSW FLAT MUVERA") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW;

        // test multiple MUVERA parameter combinations: {num_projections, num_repeats, strategy_id}
        const std::vector<std::tuple<int32_t, int32_t, int32_t>> MUVERA_PARAMS = {
            {3, 5, 1},
            {4, 3, 3},
        };

        for (const auto& [num_proj, num_rep, strategy_id] : MUVERA_PARAMS) {
            for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
                for (const int32_t dim : DIMS) {
                    const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                    auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                    for (const int32_t nb : NBS) {
                        knowhere::Json conf = default_conf;
                        conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                        conf[knowhere::meta::DIM] = dim;
                        conf[knowhere::meta::ROWS] = nb;
                        conf[knowhere::meta::INDEX_TYPE] = index_type;
                        conf["emb_list_strategy"] = "muvera";
                        conf["muvera_num_projections"] = num_proj;
                        conf["muvera_num_repeats"] = num_rep;
                        conf["muvera_seed"] = 42;

                        // strategy_id differentiates filename per param combo
                        std::vector<int32_t> params = {(int)distance_type, dim, nb, strategy_id};

                        const uint64_t rng_seed = get_params_hash(params);
                        int each_el_len = 10;
                        int num_el = int(nb / each_el_len) + 1;
                        auto default_ds_ptr = GenEmbListDataSet(nb, dim, rng_seed, each_el_len);

                        for (const float bitset_rate : BITSET_RATES) {
                            printf("bitset_rate: %f\n", bitset_rate);
                            const std::vector<uint8_t> bitset_data =
                                GenerateBitsetByPartition(num_el, 1.0f - bitset_rate, 1);

                            knowhere::BitsetView bitset_view = nullptr;
                            if (bitset_rate != 0.0f) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), num_el);
                            }

                            auto golden_result = knowhere::BruteForce::Search<knowhere::fp32>(
                                default_ds_ptr, query_ds_ptr, conf, bitset_view);

                            printf(
                                "\nProcessing EMBList HNSW,Flat MUVERA(proj=%d,rep=%d) fp32 for %s distance, "
                                "dim=%d, nrows=%d, %d%% points filtered out\n",
                                num_proj, num_rep, DISTANCE_TYPES[distance_type].c_str(), dim, nb,
                                int(bitset_rate * 100));

                            auto index_file = test_emb_list_index<knowhere::fp32>(
                                default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf, false, bitset_view);
                            std::remove(index_file.c_str());
                        }
                    }
                }
            }
        }
    }

    SECTION("HNSW FLAT MUVERA no rerank") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW;
        const float target_recall = 0.5f;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;
                    conf["emb_list_strategy"] = "muvera";
                    conf["muvera_num_projections"] = 3;
                    conf["muvera_num_repeats"] = 5;
                    conf["muvera_seed"] = 42;
                    conf["emb_list_rerank"] = false;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb, 4};

                    const uint64_t rng_seed = get_params_hash(params);
                    int each_el_len = 10;
                    int num_el = int(nb / each_el_len) + 1;
                    auto default_ds_ptr = GenEmbListDataSet(nb, dim, rng_seed, each_el_len);

                    auto version = GenTestEmbListVersionList();
                    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version).value();
                    index.Build(default_ds_ptr, conf);

                    // serialize + deserialize round-trip
                    knowhere::BinarySet binset;
                    index.Serialize(binset);
                    auto index_loaded =
                        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version).value();
                    index_loaded.Deserialize(binset, conf);

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
                            "\nProcessing EMBList HNSW,Flat MUVERA(no rerank) fp32 for %s distance, dim=%d, "
                            "nrows=%d, %d%% points filtered out\n",
                            DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                        auto result = index.Search(query_ds_ptr, conf, bitset_view);
                        REQUIRE(result.has_value());
                        auto recall = GetKNNRecall(*golden_result.value(), *result.value());
                        printf("recall: %f\n", recall);
                        REQUIRE(recall >= target_recall);

                        auto result_loaded = index_loaded.Search(query_ds_ptr, conf, bitset_view);
                        REQUIRE(result_loaded.has_value());
                        auto recall_loaded = GetKNNRecall(*golden_result.value(), *result_loaded.value());
                        REQUIRE(recall_loaded >= target_recall);
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
            {1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb}, true, true, {10, false, 30},
            std::chrono::milliseconds(1000));
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
            {1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb}, true, true, {10, false, 30},
            std::chrono::milliseconds(1000));
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

    SECTION("HNSW FLAT MUVERA") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenQueryEmbListDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;
                    conf["emb_list_strategy"] = "muvera";
                    conf["muvera_num_projections"] = 3;
                    conf["muvera_num_repeats"] = 5;
                    conf["muvera_seed"] = 42;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb, 1};
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
                            "\nProcessing EMBList HNSW,Flat MUVERA fp32 (empty) for %s distance, dim=%d, nrows=%d, "
                            "%d%% points filtered out\n",
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

TEST_CASE("EmbList Serialization", "Strategy and IndexNode serialization/deserialization tests") {
    const int32_t DIM = 4;
    const int32_t NB = 64;
    const int32_t NQ = 4;
    const int32_t TOPK = 5;
    const int32_t EACH_EL_LEN = 8;

    auto default_ds_ptr = GenEmbListDataSet(NB, DIM, 42, EACH_EL_LEN);
    auto query_ds_ptr = GenQueryEmbListDataSet(NQ, DIM, 99);

    knowhere::Json base_conf;
    base_conf[knowhere::indexparam::HNSW_M] = 16;
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    base_conf[knowhere::indexparam::EF] = 64;
    base_conf[knowhere::meta::TOPK] = TOPK;
    base_conf[knowhere::meta::DIM] = DIM;
    base_conf[knowhere::meta::ROWS] = NB;
    base_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    base_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    base_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 3.0f;

    SECTION("Strategy-level: TokenANN serialize/deserialize roundtrip") {
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        cfg.dim = DIM;
        auto strategy_or = knowhere::CreateEmbListStrategy("tokenann", cfg);
        REQUIRE(strategy_or.has_value());
        auto& strategy = strategy_or.value();

        // Prepare data
        const size_t* lims = default_ds_ptr->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + NB / EACH_EL_LEN + 1));
        auto prep_result = strategy->PrepareDataForBuild(default_ds_ptr, doc_offset, cfg);
        REQUIRE(prep_result.has_value());

        // Serialize
        std::shared_ptr<uint8_t[]> data;
        int64_t size = 0;
        REQUIRE(strategy->Serialize(data, size) == knowhere::Status::success);
        REQUIRE(size > 0);

        // Deserialize into new instance
        auto strategy2_or = knowhere::CreateEmbListStrategy("tokenann", cfg);
        REQUIRE(strategy2_or.has_value());
        auto& strategy2 = strategy2_or.value();
        REQUIRE(strategy2->Deserialize(data.get(), size, cfg) == knowhere::Status::success);

        // Verify
        REQUIRE(strategy2->GetDocCount() == strategy->GetDocCount());

        auto offset1 = strategy->GetEmbListOffset();
        auto offset2 = strategy2->GetEmbListOffset();
        REQUIRE(offset1->offset == offset2->offset);
    }

    SECTION("Strategy-level: TokenANN legacy format compatibility") {
        // Legacy format: [size_t count][size_t[count] offsets] (no magic)
        std::vector<size_t> offsets = {0, 8, 16, 24, 32, 40, 48, 56, 64};
        size_t num_offsets = offsets.size();
        size_t blob_size = sizeof(size_t) + num_offsets * sizeof(size_t);
        auto blob = std::make_unique<uint8_t[]>(blob_size);
        std::memcpy(blob.get(), &num_offsets, sizeof(size_t));
        std::memcpy(blob.get() + sizeof(size_t), offsets.data(), num_offsets * sizeof(size_t));

        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        auto strategy_or = knowhere::CreateEmbListStrategy("tokenann", cfg);
        REQUIRE(strategy_or.has_value());
        REQUIRE(strategy_or.value()->Deserialize(blob.get(), blob_size, cfg) == knowhere::Status::success);
        REQUIRE(strategy_or.value()->GetDocCount() == 8);
        auto loaded_offset = strategy_or.value()->GetEmbListOffset();
        REQUIRE(loaded_offset->offset == offsets);
    }

    SECTION("Strategy-level: TokenANN version validation") {
        // New format with unsupported version (version=99)
        constexpr int32_t kMagic = 0x544F4B41;
        constexpr int32_t kBadVersion = 99;
        std::vector<size_t> offsets = {0, 8, 16};
        size_t num_offsets = offsets.size();
        size_t blob_size = 2 * sizeof(int32_t) + sizeof(size_t) + num_offsets * sizeof(size_t);
        auto blob = std::make_unique<uint8_t[]>(blob_size);
        uint8_t* ptr = blob.get();
        std::memcpy(ptr, &kMagic, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &kBadVersion, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_offsets, sizeof(size_t));
        std::memcpy(ptr + sizeof(size_t), offsets.data(), num_offsets * sizeof(size_t));

        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        auto strategy_or = knowhere::CreateEmbListStrategy("tokenann", cfg);
        REQUIRE(strategy_or.has_value());
        REQUIRE(strategy_or.value()->Deserialize(blob.get(), blob_size, cfg) == knowhere::Status::emb_list_inner_error);
    }

    SECTION("Strategy-level: MUVERA serialize/deserialize roundtrip") {
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        cfg.muvera_num_projections = 3;
        cfg.muvera_num_repeats = 2;
        cfg.muvera_seed = 42;
        auto strategy_or = knowhere::CreateEmbListStrategy("muvera", cfg);
        REQUIRE(strategy_or.has_value());
        auto& strategy = strategy_or.value();

        const size_t* lims = default_ds_ptr->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + NB / EACH_EL_LEN + 1));
        auto prep_result = strategy->PrepareDataForBuild(default_ds_ptr, doc_offset, cfg);
        REQUIRE(prep_result.has_value());

        std::shared_ptr<uint8_t[]> data;
        int64_t size = 0;
        REQUIRE(strategy->Serialize(data, size) == knowhere::Status::success);

        auto strategy2_or = knowhere::CreateEmbListStrategy("muvera", cfg);
        REQUIRE(strategy2_or.has_value());
        REQUIRE(strategy2_or.value()->Deserialize(data.get(), size, cfg) == knowhere::Status::success);

        REQUIRE(strategy2_or.value()->GetDocCount() == strategy->GetDocCount());

        REQUIRE(strategy2_or.value()->GetEmbListOffset()->offset == strategy->GetEmbListOffset()->offset);
    }

    SECTION("Strategy-level: MUVERA magic validation") {
        // Corrupt magic
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        cfg.muvera_num_projections = 3;
        cfg.muvera_num_repeats = 2;
        cfg.muvera_seed = 42;
        auto strategy_or = knowhere::CreateEmbListStrategy("muvera", cfg);
        REQUIRE(strategy_or.has_value());
        auto& strategy = strategy_or.value();

        const size_t* lims = default_ds_ptr->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + NB / EACH_EL_LEN + 1));
        strategy->PrepareDataForBuild(default_ds_ptr, doc_offset, cfg);

        std::shared_ptr<uint8_t[]> data;
        int64_t size = 0;
        strategy->Serialize(data, size);

        // Corrupt magic byte
        data.get()[0] = 0xFF;
        auto strategy2_or = knowhere::CreateEmbListStrategy("muvera", cfg);
        REQUIRE(strategy2_or.value()->Deserialize(data.get(), size, cfg) == knowhere::Status::emb_list_inner_error);
    }

    SECTION("Strategy-level: MUVERA version validation") {
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        cfg.muvera_num_projections = 3;
        cfg.muvera_num_repeats = 2;
        cfg.muvera_seed = 42;
        auto strategy_or = knowhere::CreateEmbListStrategy("muvera", cfg);
        REQUIRE(strategy_or.has_value());

        const size_t* lims = default_ds_ptr->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + NB / EACH_EL_LEN + 1));
        strategy_or.value()->PrepareDataForBuild(default_ds_ptr, doc_offset, cfg);

        std::shared_ptr<uint8_t[]> data;
        int64_t size = 0;
        strategy_or.value()->Serialize(data, size);

        // Overwrite version (offset 4) with unsupported version
        int32_t bad_version = 99;
        std::memcpy(data.get() + sizeof(int32_t), &bad_version, sizeof(int32_t));

        auto strategy2_or = knowhere::CreateEmbListStrategy("muvera", cfg);
        REQUIRE(strategy2_or.value()->Deserialize(data.get(), size, cfg) == knowhere::Status::emb_list_inner_error);
    }

    SECTION("Strategy-level: LEMUR serialize/deserialize roundtrip") {
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        cfg.lemur_hidden_dim = 16;
        cfg.lemur_num_train_samples = 32;
        cfg.lemur_num_epochs = 2;
        cfg.lemur_batch_size = 16;
        cfg.lemur_learning_rate = 0.001f;
        cfg.lemur_seed = 42;
        cfg.lemur_num_layers = 1;
        auto strategy_or = knowhere::CreateEmbListStrategy("lemur", cfg);
        REQUIRE(strategy_or.has_value());
        auto& strategy = strategy_or.value();

        const size_t* lims = default_ds_ptr->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + NB / EACH_EL_LEN + 1));
        auto prep_result = strategy->PrepareDataForBuild(default_ds_ptr, doc_offset, cfg);
        REQUIRE(prep_result.has_value());

        std::shared_ptr<uint8_t[]> data;
        int64_t size = 0;
        REQUIRE(strategy->Serialize(data, size) == knowhere::Status::success);

        auto strategy2_or = knowhere::CreateEmbListStrategy("lemur", cfg);
        REQUIRE(strategy2_or.has_value());
        REQUIRE(strategy2_or.value()->Deserialize(data.get(), size, cfg) == knowhere::Status::success);

        REQUIRE(strategy2_or.value()->GetDocCount() == strategy->GetDocCount());

        REQUIRE(strategy2_or.value()->GetEmbListOffset()->offset == strategy->GetEmbListOffset()->offset);
    }

    SECTION("Strategy-level: LEMUR magic validation") {
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        cfg.lemur_hidden_dim = 16;
        cfg.lemur_num_train_samples = 32;
        cfg.lemur_num_epochs = 2;
        cfg.lemur_batch_size = 16;
        cfg.lemur_learning_rate = 0.001f;
        cfg.lemur_seed = 42;
        cfg.lemur_num_layers = 1;
        auto strategy_or = knowhere::CreateEmbListStrategy("lemur", cfg);
        REQUIRE(strategy_or.has_value());

        const size_t* lims = default_ds_ptr->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + NB / EACH_EL_LEN + 1));
        strategy_or.value()->PrepareDataForBuild(default_ds_ptr, doc_offset, cfg);

        std::shared_ptr<uint8_t[]> data;
        int64_t size = 0;
        strategy_or.value()->Serialize(data, size);

        data.get()[0] = 0xFF;
        auto strategy2_or = knowhere::CreateEmbListStrategy("lemur", cfg);
        REQUIRE(strategy2_or.value()->Deserialize(data.get(), size, cfg) == knowhere::Status::emb_list_inner_error);
    }

    SECTION("Strategy-level: LEMUR version validation") {
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        cfg.lemur_hidden_dim = 16;
        cfg.lemur_num_train_samples = 32;
        cfg.lemur_num_epochs = 2;
        cfg.lemur_batch_size = 16;
        cfg.lemur_learning_rate = 0.001f;
        cfg.lemur_seed = 42;
        cfg.lemur_num_layers = 1;
        auto strategy_or = knowhere::CreateEmbListStrategy("lemur", cfg);
        REQUIRE(strategy_or.has_value());

        const size_t* lims = default_ds_ptr->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + NB / EACH_EL_LEN + 1));
        strategy_or.value()->PrepareDataForBuild(default_ds_ptr, doc_offset, cfg);

        std::shared_ptr<uint8_t[]> data;
        int64_t size = 0;
        strategy_or.value()->Serialize(data, size);

        int32_t bad_version = 99;
        std::memcpy(data.get() + sizeof(int32_t), &bad_version, sizeof(int32_t));

        auto strategy2_or = knowhere::CreateEmbListStrategy("lemur", cfg);
        REQUIRE(strategy2_or.value()->Deserialize(data.get(), size, cfg) == knowhere::Status::emb_list_inner_error);
    }

    SECTION("Factory: CreateEmbListStrategy known and unknown types") {
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";

        REQUIRE(knowhere::CreateEmbListStrategy("tokenann", cfg).has_value());
        REQUIRE(knowhere::CreateEmbListStrategy("muvera", cfg).has_value());
        REQUIRE(knowhere::CreateEmbListStrategy("lemur", cfg).has_value());
        REQUIRE(knowhere::CreateEmbListStrategy("", cfg).has_value());  // empty defaults to tokenann
        REQUIRE_FALSE(knowhere::CreateEmbListStrategy("unknown_strategy", cfg).has_value());
    }

    SECTION("IndexNode-level: TokenANN BinarySet roundtrip") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        // Verify EMB_LIST_META key exists
        auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
        REQUIRE(meta_bin != nullptr);

        // Deserialize into new index and search
        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(index2.Deserialize(binset, conf) == knowhere::Status::success);

        auto result1 = index.Search(query_ds_ptr, conf, nullptr);
        auto result2 = index2.Search(query_ds_ptr, conf, nullptr);
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        // Results should be identical
        const auto* ids1 = result1.value()->GetIds();
        const auto* ids2 = result2.value()->GetIds();
        auto num_q = result1.value()->GetRows();
        for (int64_t i = 0; i < num_q * TOPK; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
        }
    }

    SECTION("IndexNode-level: MUVERA BinarySet roundtrip with raw index") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;
        conf["emb_list_strategy"] = "muvera";
        conf["muvera_num_projections"] = 3;
        conf["muvera_num_repeats"] = 2;
        conf["muvera_seed"] = 42;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        // Verify both EMB_LIST_META and EMB_LIST_RAW_INDEX keys exist
        REQUIRE(binset.GetByName(knowhere::meta::EMB_LIST_META) != nullptr);
        REQUIRE(binset.GetByName(knowhere::meta::EMB_LIST_RAW_INDEX) != nullptr);

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(index2.Deserialize(binset, conf) == knowhere::Status::success);

        auto result1 = index.Search(query_ds_ptr, conf, nullptr);
        auto result2 = index2.Search(query_ds_ptr, conf, nullptr);
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        const auto* ids1 = result1.value()->GetIds();
        const auto* ids2 = result2.value()->GetIds();
        auto num_q = result1.value()->GetRows();
        for (int64_t i = 0; i < num_q * TOPK; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
        }
    }

    SECTION("IndexNode-level: LEMUR BinarySet roundtrip with raw index") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;
        conf["emb_list_strategy"] = "lemur";
        conf["lemur_hidden_dim"] = 32;
        conf["lemur_num_train_samples"] = 1000;
        conf["lemur_num_epochs"] = 2;
        conf["lemur_batch_size"] = 16;
        conf["lemur_learning_rate"] = 0.001f;
        conf["lemur_seed"] = 42;
        conf["lemur_num_layers"] = 1;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        REQUIRE(binset.GetByName(knowhere::meta::EMB_LIST_META) != nullptr);
        REQUIRE(binset.GetByName(knowhere::meta::EMB_LIST_RAW_INDEX) != nullptr);

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(index2.Deserialize(binset, conf) == knowhere::Status::success);

        auto result1 = index.Search(query_ds_ptr, conf, nullptr);
        auto result2 = index2.Search(query_ds_ptr, conf, nullptr);
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        const auto* ids1 = result1.value()->GetIds();
        const auto* ids2 = result2.value()->GetIds();
        auto num_q = result1.value()->GetRows();
        for (int64_t i = 0; i < num_q * TOPK; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
        }
    }

    SECTION("File-based: TokenANN DeserializeFromFile roundtrip") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        std::string base_index_file = "/tmp/test_emb_list_tokenann_base.index";
        std::string meta_file = "/tmp/test_emb_list_tokenann_meta.bin";
        {
            // Write base index as raw faiss data (not BinarySet wire format)
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            REQUIRE(hnsw_bin != nullptr);
            std::ofstream base_out(base_index_file, std::ios::binary);
            base_out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);

            auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
            REQUIRE(meta_bin != nullptr);
            std::ofstream meta_out(meta_file, std::ios::binary);
            meta_out.write(reinterpret_cast<const char*>(meta_bin->data.get()), meta_bin->size);
        }

        knowhere::Json load_conf = conf;
        load_conf["emb_list_meta_file_path"] = meta_file;
        load_conf["enable_mmap"] = false;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto status = index2.DeserializeFromFile(base_index_file, load_conf);

        std::remove(base_index_file.c_str());
        std::remove(meta_file.c_str());

        REQUIRE(status == knowhere::Status::success);
        auto result1 = index.Search(query_ds_ptr, conf, nullptr);
        auto result2 = index2.Search(query_ds_ptr, conf, nullptr);
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        const auto* ids1 = result1.value()->GetIds();
        const auto* ids2 = result2.value()->GetIds();
        auto num_q = result1.value()->GetRows();
        for (int64_t i = 0; i < num_q * TOPK; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
        }
    }

    SECTION("File-based: TokenANN DeserializeFromFile with mmap") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        std::string base_index_file = "/tmp/test_emb_list_tokenann_mmap.index";
        std::string meta_file = "/tmp/test_emb_list_tokenann_mmap_meta.bin";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream base_out(base_index_file, std::ios::binary);
            base_out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);

            auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
            std::ofstream meta_out(meta_file, std::ios::binary);
            meta_out.write(reinterpret_cast<const char*>(meta_bin->data.get()), meta_bin->size);
        }

        knowhere::Json load_conf = conf;
        load_conf["emb_list_meta_file_path"] = meta_file;
        load_conf["enable_mmap"] = true;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto status = index2.DeserializeFromFile(base_index_file, load_conf);

        std::remove(base_index_file.c_str());
        std::remove(meta_file.c_str());

        REQUIRE(status == knowhere::Status::success);
        auto result1 = index.Search(query_ds_ptr, conf, nullptr);
        auto result2 = index2.Search(query_ds_ptr, conf, nullptr);
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        const auto* ids1 = result1.value()->GetIds();
        const auto* ids2 = result2.value()->GetIds();
        auto num_q = result1.value()->GetRows();
        for (int64_t i = 0; i < num_q * TOPK; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
        }
    }

    SECTION("File-based: meta file path empty returns error") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        index.Serialize(binset);

        std::string base_index_file = "/tmp/test_emb_list_empty_meta.index";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream out(base_index_file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);
        }

        knowhere::Json load_conf = conf;
        load_conf["enable_mmap"] = false;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto status = index2.DeserializeFromFile(base_index_file, load_conf);
        std::remove(base_index_file.c_str());

        REQUIRE(status != knowhere::Status::success);
    }

    SECTION("File-based: meta file not found returns error") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        index.Serialize(binset);

        std::string base_index_file = "/tmp/test_emb_list_nofile.index";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream out(base_index_file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);
        }

        knowhere::Json load_conf = conf;
        load_conf["emb_list_meta_file_path"] = "/tmp/nonexistent_meta_file.bin";
        load_conf["enable_mmap"] = false;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto status = index2.DeserializeFromFile(base_index_file, load_conf);
        std::remove(base_index_file.c_str());

        REQUIRE(status != knowhere::Status::success);
    }

    SECTION("File-based: MUVERA raw index file path empty returns error") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;
        conf["emb_list_strategy"] = "muvera";
        conf["muvera_num_projections"] = 3;
        conf["muvera_num_repeats"] = 2;
        conf["muvera_seed"] = 42;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        index.Serialize(binset);

        std::string base_index_file = "/tmp/test_emb_list_muvera_noraw.index";
        std::string meta_file = "/tmp/test_emb_list_muvera_noraw_meta.bin";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream base_out(base_index_file, std::ios::binary);
            base_out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);

            auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
            std::ofstream meta_out(meta_file, std::ios::binary);
            meta_out.write(reinterpret_cast<const char*>(meta_bin->data.get()), meta_bin->size);
        }

        knowhere::Json load_conf = conf;
        load_conf["emb_list_meta_file_path"] = meta_file;
        load_conf["enable_mmap"] = false;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto status = index2.DeserializeFromFile(base_index_file, load_conf);

        std::remove(base_index_file.c_str());
        std::remove(meta_file.c_str());

        REQUIRE(status != knowhere::Status::success);
    }

    SECTION("File-based: MUVERA DeserializeFromFile roundtrip") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;
        conf["emb_list_strategy"] = "muvera";
        conf["muvera_num_projections"] = 3;
        conf["muvera_num_repeats"] = 2;
        conf["muvera_seed"] = 42;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        std::string base_index_file = "/tmp/test_emb_list_muvera_file.index";
        std::string meta_file = "/tmp/test_emb_list_muvera_file_meta.bin";
        std::string raw_index_file = "/tmp/test_emb_list_muvera_file_raw.index";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream base_out(base_index_file, std::ios::binary);
            base_out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);

            auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
            std::ofstream meta_out(meta_file, std::ios::binary);
            meta_out.write(reinterpret_cast<const char*>(meta_bin->data.get()), meta_bin->size);

            auto raw_bin = binset.GetByName(knowhere::meta::EMB_LIST_RAW_INDEX);
            REQUIRE(raw_bin != nullptr);
            std::ofstream raw_out(raw_index_file, std::ios::binary);
            raw_out.write(reinterpret_cast<const char*>(raw_bin->data.get()), raw_bin->size);
        }

        for (bool enable_mmap : {false, true}) {
            knowhere::Json load_conf = conf;
            load_conf["emb_list_meta_file_path"] = meta_file;
            load_conf["emb_list_raw_index_file_path"] = raw_index_file;
            load_conf["enable_mmap"] = enable_mmap;

            auto index2 = knowhere::IndexFactory::Instance()
                              .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version)
                              .value();
            auto status = index2.DeserializeFromFile(base_index_file, load_conf);
            REQUIRE(status == knowhere::Status::success);

            auto result1 = index.Search(query_ds_ptr, conf, nullptr);
            auto result2 = index2.Search(query_ds_ptr, conf, nullptr);
            REQUIRE(result1.has_value());
            REQUIRE(result2.has_value());

            const auto* ids1 = result1.value()->GetIds();
            const auto* ids2 = result2.value()->GetIds();
            auto num_q = result1.value()->GetRows();
            for (int64_t i = 0; i < num_q * TOPK; ++i) {
                REQUIRE(ids1[i] == ids2[i]);
            }
        }

        std::remove(base_index_file.c_str());
        std::remove(meta_file.c_str());
        std::remove(raw_index_file.c_str());
    }

    SECTION("File-based: LEMUR DeserializeFromFile roundtrip") {
        // Use larger dataset: LEMUR needs num_docs >= hidden_dim for OLS
        const int32_t LEMUR_NB = 512;
        auto lemur_ds_ptr = GenEmbListDataSet(LEMUR_NB, DIM, 42, EACH_EL_LEN);
        auto lemur_query_ds_ptr = GenQueryEmbListDataSet(NQ, DIM, 99);

        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;
        conf[knowhere::meta::ROWS] = LEMUR_NB;
        conf["emb_list_strategy"] = "lemur";
        conf["lemur_hidden_dim"] = 32;
        conf["lemur_num_train_samples"] = 1000;
        conf["lemur_num_epochs"] = 2;
        conf["lemur_batch_size"] = 16;
        conf["lemur_learning_rate"] = 0.001f;
        conf["lemur_seed"] = 42;
        conf["lemur_num_layers"] = 1;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(lemur_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        std::string base_index_file = "/tmp/test_emb_list_lemur_file.index";
        std::string meta_file = "/tmp/test_emb_list_lemur_file_meta.bin";
        std::string raw_index_file = "/tmp/test_emb_list_lemur_file_raw.index";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream base_out(base_index_file, std::ios::binary);
            base_out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);

            auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
            std::ofstream meta_out(meta_file, std::ios::binary);
            meta_out.write(reinterpret_cast<const char*>(meta_bin->data.get()), meta_bin->size);

            auto raw_bin = binset.GetByName(knowhere::meta::EMB_LIST_RAW_INDEX);
            REQUIRE(raw_bin != nullptr);
            std::ofstream raw_out(raw_index_file, std::ios::binary);
            raw_out.write(reinterpret_cast<const char*>(raw_bin->data.get()), raw_bin->size);
        }

        knowhere::Json load_conf = conf;
        load_conf["emb_list_meta_file_path"] = meta_file;
        load_conf["emb_list_raw_index_file_path"] = raw_index_file;
        load_conf["enable_mmap"] = false;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(index2.DeserializeFromFile(base_index_file, load_conf) == knowhere::Status::success);

        auto result1 = index.Search(lemur_query_ds_ptr, conf, nullptr);
        auto result2 = index2.Search(lemur_query_ds_ptr, conf, nullptr);
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        const auto* ids1 = result1.value()->GetIds();
        const auto* ids2 = result2.value()->GetIds();
        auto num_q = result1.value()->GetRows();
        for (int64_t i = 0; i < num_q * TOPK; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
        }

        std::remove(base_index_file.c_str());
        std::remove(meta_file.c_str());
        std::remove(raw_index_file.c_str());
    }

    SECTION("Strategy-level: LEMUR num_docs vs hidden_dim") {
        // Test LEMUR with different num_docs relative to hidden_dim
        knowhere::BaseConfig cfg;
        cfg.metric_type = "MAX_SIM_IP";
        cfg.lemur_hidden_dim = 16;
        cfg.lemur_num_train_samples = 32;
        cfg.lemur_num_epochs = 2;
        cfg.lemur_batch_size = 16;
        cfg.lemur_learning_rate = 0.001f;
        cfg.lemur_seed = 42;
        cfg.lemur_num_layers = 1;

        // Case 1: num_docs >= hidden_dim (64 docs, hidden_dim=16) — should succeed
        {
            const int32_t big_nb = 512;
            const int32_t el_len = 8;  // 512/8 = 64 docs
            auto ds = GenEmbListDataSet(big_nb, DIM, 42, el_len);
            const size_t* lims = ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
            knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + big_nb / el_len + 1));

            auto strategy_or = knowhere::CreateEmbListStrategy("lemur", cfg);
            REQUIRE(strategy_or.has_value());
            auto result = strategy_or.value()->PrepareDataForBuild(ds, doc_offset, cfg);
            REQUIRE(result.has_value());
        }

        // Case 2: num_docs < hidden_dim (4 docs, hidden_dim=16) — should still succeed
        // (regularization makes ZtZ positive definite even when underdetermined)
        {
            const int32_t small_nb = 32;
            const int32_t el_len = 8;  // 32/8 = 4 docs
            auto ds = GenEmbListDataSet(small_nb, DIM, 42, el_len);
            const size_t* lims = ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
            knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + small_nb / el_len + 1));

            auto strategy_or = knowhere::CreateEmbListStrategy("lemur", cfg);
            REQUIRE(strategy_or.has_value());
            auto result = strategy_or.value()->PrepareDataForBuild(ds, doc_offset, cfg);
            REQUIRE(result.has_value());
        }

        // Case 3: single doc — edge case
        {
            const int32_t tiny_nb = 8;
            const int32_t el_len = 8;  // 8/8 = 1 doc
            auto ds = GenEmbListDataSet(tiny_nb, DIM, 42, el_len);
            const size_t* lims = ds->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
            knowhere::EmbListOffset doc_offset(std::vector<size_t>(lims, lims + tiny_nb / el_len + 1));

            auto strategy_or = knowhere::CreateEmbListStrategy("lemur", cfg);
            REQUIRE(strategy_or.has_value());
            auto result = strategy_or.value()->PrepareDataForBuild(ds, doc_offset, cfg);
            REQUIRE(result.has_value());
        }
    }

    SECTION("File-based: LEMUR DeserializeFromFile with mmap") {
        const int32_t LEMUR_NB = 512;
        auto lemur_ds_ptr = GenEmbListDataSet(LEMUR_NB, DIM, 42, EACH_EL_LEN);
        auto lemur_query_ds_ptr = GenQueryEmbListDataSet(NQ, DIM, 99);

        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;
        conf[knowhere::meta::ROWS] = LEMUR_NB;
        conf["emb_list_strategy"] = "lemur";
        conf["lemur_hidden_dim"] = 32;
        conf["lemur_num_train_samples"] = 1000;
        conf["lemur_num_epochs"] = 2;
        conf["lemur_batch_size"] = 16;
        conf["lemur_learning_rate"] = 0.001f;
        conf["lemur_seed"] = 42;
        conf["lemur_num_layers"] = 1;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(lemur_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        std::string base_index_file = "/tmp/test_emb_list_lemur_mmap.index";
        std::string meta_file = "/tmp/test_emb_list_lemur_mmap_meta.bin";
        std::string raw_index_file = "/tmp/test_emb_list_lemur_mmap_raw.index";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream base_out(base_index_file, std::ios::binary);
            base_out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);

            auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
            std::ofstream meta_out(meta_file, std::ios::binary);
            meta_out.write(reinterpret_cast<const char*>(meta_bin->data.get()), meta_bin->size);

            auto raw_bin = binset.GetByName(knowhere::meta::EMB_LIST_RAW_INDEX);
            REQUIRE(raw_bin != nullptr);
            std::ofstream raw_out(raw_index_file, std::ios::binary);
            raw_out.write(reinterpret_cast<const char*>(raw_bin->data.get()), raw_bin->size);
        }

        knowhere::Json load_conf = conf;
        load_conf["emb_list_meta_file_path"] = meta_file;
        load_conf["emb_list_raw_index_file_path"] = raw_index_file;
        load_conf["enable_mmap"] = true;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        REQUIRE(index2.DeserializeFromFile(base_index_file, load_conf) == knowhere::Status::success);

        auto result1 = index.Search(lemur_query_ds_ptr, conf, nullptr);
        auto result2 = index2.Search(lemur_query_ds_ptr, conf, nullptr);
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        const auto* ids1 = result1.value()->GetIds();
        const auto* ids2 = result2.value()->GetIds();
        auto num_q = result1.value()->GetRows();
        for (int64_t i = 0; i < num_q * TOPK; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
        }

        std::remove(base_index_file.c_str());
        std::remove(meta_file.c_str());
        std::remove(raw_index_file.c_str());
    }

    SECTION("File-based: LEMUR raw index file path empty returns error") {
        const int32_t LEMUR_NB = 512;
        auto lemur_ds_ptr = GenEmbListDataSet(LEMUR_NB, DIM, 42, EACH_EL_LEN);

        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;
        conf[knowhere::meta::ROWS] = LEMUR_NB;
        conf["emb_list_strategy"] = "lemur";
        conf["lemur_hidden_dim"] = 32;
        conf["lemur_num_train_samples"] = 1000;
        conf["lemur_num_epochs"] = 2;
        conf["lemur_batch_size"] = 16;
        conf["lemur_learning_rate"] = 0.001f;
        conf["lemur_seed"] = 42;
        conf["lemur_num_layers"] = 1;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(lemur_ds_ptr, conf);

        knowhere::BinarySet binset;
        index.Serialize(binset);

        std::string base_index_file = "/tmp/test_emb_list_lemur_noraw.index";
        std::string meta_file = "/tmp/test_emb_list_lemur_noraw_meta.bin";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream base_out(base_index_file, std::ios::binary);
            base_out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);

            auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
            std::ofstream meta_out(meta_file, std::ios::binary);
            meta_out.write(reinterpret_cast<const char*>(meta_bin->data.get()), meta_bin->size);
        }

        knowhere::Json load_conf = conf;
        load_conf["emb_list_meta_file_path"] = meta_file;
        load_conf["enable_mmap"] = false;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto status = index2.DeserializeFromFile(base_index_file, load_conf);

        std::remove(base_index_file.c_str());
        std::remove(meta_file.c_str());

        REQUIRE(status != knowhere::Status::success);
    }

    SECTION("File-based: TokenANN legacy meta file compatibility") {
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        // Write base index file
        std::string base_index_file = "/tmp/test_emb_list_tokenann_legacy.index";
        std::string meta_file = "/tmp/test_emb_list_tokenann_legacy_meta.bin";
        {
            auto hnsw_bin = binset.GetByName(knowhere::IndexEnum::INDEX_HNSW);
            std::ofstream base_out(base_index_file, std::ios::binary);
            base_out.write(reinterpret_cast<const char*>(hnsw_bin->data.get()), hnsw_bin->size);
        }

        // Write legacy meta file: [size_t count][size_t[count] offsets]
        // Extract offsets from the original dataset
        const size_t* lims = default_ds_ptr->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        size_t num_docs = NB / EACH_EL_LEN;
        std::vector<size_t> offsets(lims, lims + num_docs + 1);
        {
            std::ofstream meta_out(meta_file, std::ios::binary);
            size_t count = offsets.size();
            meta_out.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
            meta_out.write(reinterpret_cast<const char*>(offsets.data()), count * sizeof(size_t));
        }

        knowhere::Json load_conf = conf;
        load_conf["emb_list_meta_file_path"] = meta_file;
        load_conf["enable_mmap"] = false;

        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto status = index2.DeserializeFromFile(base_index_file, load_conf);

        std::remove(base_index_file.c_str());
        std::remove(meta_file.c_str());

        REQUIRE(status == knowhere::Status::success);

        auto result1 = index.Search(query_ds_ptr, conf, nullptr);
        auto result2 = index2.Search(query_ds_ptr, conf, nullptr);
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());

        const auto* ids1 = result1.value()->GetIds();
        const auto* ids2 = result2.value()->GetIds();
        auto num_q = result1.value()->GetRows();
        for (int64_t i = 0; i < num_q * TOPK; ++i) {
            REQUIRE(ids1[i] == ids2[i]);
        }
    }

    SECTION("ParseEmbListMetaHeader: new format parsing") {
        // Build a valid EMB_LIST_META blob: [magic][type_len][type][strategy_blob]
        std::string strategy_type = "muvera";
        std::vector<uint8_t> fake_blob = {0x01, 0x02, 0x03, 0x04};
        size_t type_len = strategy_type.size();
        int64_t total_size = sizeof(int32_t) + sizeof(size_t) + type_len + fake_blob.size();
        auto data = std::make_unique<uint8_t[]>(total_size);
        uint8_t* ptr = data.get();

        int32_t magic = knowhere::kEmbListMetaMagic;
        std::memcpy(ptr, &magic, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &type_len, sizeof(size_t));
        ptr += sizeof(size_t);
        std::memcpy(ptr, strategy_type.data(), type_len);
        ptr += type_len;
        std::memcpy(ptr, fake_blob.data(), fake_blob.size());

        // Use the full BinarySet roundtrip to verify parsing works correctly.
        // We verify by deserializing through the IndexNode path.
        // Direct ParseEmbListMetaHeader is private, so test via integration.

        // Verify: build a real index, serialize, check the meta key can be parsed back
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;
        conf["emb_list_strategy"] = "muvera";
        conf["muvera_num_projections"] = 3;
        conf["muvera_num_repeats"] = 2;
        conf["muvera_seed"] = 42;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        index.Serialize(binset);

        // Verify EMB_LIST_META starts with the correct magic
        auto meta_bin = binset.GetByName(knowhere::meta::EMB_LIST_META);
        REQUIRE(meta_bin != nullptr);
        int32_t read_magic = 0;
        std::memcpy(&read_magic, meta_bin->data.get(), sizeof(int32_t));
        REQUIRE(read_magic == knowhere::kEmbListMetaMagic);

        // Verify strategy type is embedded after magic
        const uint8_t* meta_ptr = meta_bin->data.get() + sizeof(int32_t);
        size_t read_type_len = 0;
        std::memcpy(&read_type_len, meta_ptr, sizeof(size_t));
        meta_ptr += sizeof(size_t);
        std::string read_type(reinterpret_cast<const char*>(meta_ptr), read_type_len);
        REQUIRE(read_type == "muvera");
    }

    SECTION("ParseEmbListMetaHeader: legacy format defaults to tokenann") {
        // Legacy format has no magic — first bytes are a size_t count.
        // When magic doesn't match kEmbListMetaMagic, it defaults to tokenann.
        // Verify via a full BinarySet roundtrip with tokenann (which should use new format now).

        // Build a legacy-format blob manually: [size_t count][size_t[count] offsets]
        std::vector<size_t> offsets = {0, 10, 20, 30};
        size_t num_offsets = offsets.size();
        size_t blob_size = sizeof(size_t) + num_offsets * sizeof(size_t);
        auto blob_data = std::shared_ptr<uint8_t[]>(new uint8_t[blob_size]);
        std::memcpy(blob_data.get(), &num_offsets, sizeof(size_t));
        std::memcpy(blob_data.get() + sizeof(size_t), offsets.data(), num_offsets * sizeof(size_t));

        // Verify the first 4 bytes don't match the magic
        int32_t first_bytes = 0;
        std::memcpy(&first_bytes, blob_data.get(), sizeof(int32_t));
        REQUIRE(first_bytes != knowhere::kEmbListMetaMagic);

        // Put this into a BinarySet and try to deserialize as TokenANN
        // Build a real tokenann index first to get a valid base index
        auto version = GenTestEmbListVersionList();
        auto conf = base_conf;

        auto index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        index.Build(default_ds_ptr, conf);

        knowhere::BinarySet binset;
        index.Serialize(binset);

        // Replace EMB_LIST_META with legacy format blob
        binset.binary_map_.erase(knowhere::meta::EMB_LIST_META);
        binset.Append(knowhere::meta::EMB_LIST_META, blob_data, blob_size);

        // Deserialize — should detect legacy format and use tokenann
        auto index2 =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version).value();
        auto status = index2.Deserialize(binset, conf);
        REQUIRE(status == knowhere::Status::success);
    }
}
