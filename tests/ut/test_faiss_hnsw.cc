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
#include <chrono>
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
#include "knowhere/comp/thread_pool.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {

template <typename T = float>
void
normalize_vector(T* l2_mag, T* vec_norm, int index) {
    T sqrt_norm = std::sqrt(*l2_mag);
    if (sqrt_norm == 0) {
        throw std::invalid_argument("ERROR: norm value is 0.");
    }
    vec_norm[index] = sqrt_norm;
    *l2_mag = 0;
}

knowhere::DataSetPtr
GenDataSet(int rows, int dim, const uint64_t seed = 42) {
    std::mt19937 rng(static_cast<int>(47));
    std::uniform_real_distribution<float> distrib_real;
    float* ts = new float[rows * dim];
    float* ts_normed = new float[rows];
    float l2_mag = 0;

    for (int i = 0; i < rows * dim; ++i) {
        ts[i] = distrib_real(rng);
        l2_mag += ts[i] * ts[i];
        if ((i + 1) % dim == 0) {
            normalize_vector(&l2_mag, ts_normed, i / dim);
        }
    }

    int j = 0;
    for (int i = 0; i < dim * rows; ++i) {
        ts[i] /= ts_normed[j];
        if ((i + 1) % dim == 0) {
            ++j;
        }
    }

    auto ds = knowhere::GenDataSet(rows, dim, ts);
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

        auto start = std::chrono::high_resolution_clock::now();
        index.value().Build(base, conf);
        double elapsed = sw.elapsed();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);  // double 초 변환

        std::cout << "+++++++++++++++++++++++++Elapsed time: " << duration.count() << " seconds" << std::endl;

        printf("Building %sindex took %f s\n", additional_name_s.c_str(), elapsed);

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

    // then, we force it to be loaded in order to test load & save
    auto index_loaded = create_index<T>(index_type, index_file_name, default_ds_ptr, conf, mv_only_enable);

    // query
    auto query_t_ds_ptr = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds_ptr);

    StopWatch sw_search;
    auto startSearch = std::chrono::high_resolution_clock::now();

    auto result = index.Search(query_t_ds_ptr, conf, bitset_view);
    auto endSearch = std::chrono::high_resolution_clock::now();
    double search_elapsed = sw_search.elapsed();

    auto durationSearch = std::chrono::duration<double>(endSearch - startSearch);  // double 초 변환

    std::cout << "+++++++++++++++++++++++++Elapsed time: " << durationSearch.count() << " seconds" << std::endl;

    auto result_loaded = index_loaded.Search(query_t_ds_ptr, conf, bitset_view);

    // calc recall
    auto recall = GetKNNRecall(*golden_result, *result.value());
    auto recall_loaded = GetKNNRecall(*golden_result, *result_loaded.value());
    printf("Recall is %f, %f. Search took %f s\n", recall, recall_loaded, search_elapsed);

    // REQUIRE(recall >= 0.8);
    // REQUIRE(recall_loaded >= 0.8);
    REQUIRE(recall == recall_loaded);

    // test HasRawData()
    auto metric_type = conf[knowhere::meta::METRIC_TYPE];
    REQUIRE(index_loaded.HasRawData(metric_type) ==
            knowhere::IndexStaticFaced<T>::HasRawData(index_type,
                                                      knowhere::Version::GetCurrentVersion().VersionNumber(), conf));

    // test GetVectorByIds()
    if (index_loaded.HasRawData(metric_type)) {
        const auto rows = default_t_ds_ptr->GetRows();

        int64_t* ids = new int64_t[rows];
        for (int64_t i = 0; i < rows; i++) {
            ids[i] = i;
        }

        auto ids_ds = knowhere::GenIdsDataSet(rows, ids);
        ids_ds->SetIsOwner(true);

        auto vectors = index_loaded.GetVectorByIds(ids_ds);
        REQUIRE(vectors.has_value());

        match_datasets<T>(default_t_ds_ptr, vectors.value(), ids);
    }
    return index_file_name;
}

//
template <typename T>
std::string
test_hnsw_range(const knowhere::DataSetPtr& default_ds_ptr, const knowhere::DataSetPtr& query_ds_ptr,
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

    // then, we force it to be loaded in order to test load & save
    auto index_loaded = create_index<T>(index_type, index_file_name, default_ds_ptr, conf, mv_only_enable);

    // query
    auto query_t_ds_ptr = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds_ptr);

    // perform the search
    StopWatch sw_range_search;
    auto result = index.RangeSearch(query_t_ds_ptr, conf, bitset_view);
    double range_search_elapsed = sw_range_search.elapsed();

    auto result_loaded = index_loaded.RangeSearch(query_t_ds_ptr, conf, bitset_view);

    // compute the recall
    float recall = GetRangeSearchRecall(*golden_result, *result.value());
    float recall_loaded = GetRangeSearchRecall(*golden_result, *result_loaded.value());

    printf("Recall is %f, %f. Filtered %zd of %zd (on average). Search took %f ms\n", recall, recall_loaded,
           result.value()->GetLims()[query_ds_ptr->GetRows()] / query_ds_ptr->GetRows(), default_t_ds_ptr->GetRows(),
           range_search_elapsed);

    REQUIRE(recall >= 0.8);
    REQUIRE(recall_loaded >= 0.8);
    REQUIRE(recall == recall_loaded);

    // test HasRawData()
    auto metric_type = conf[knowhere::meta::METRIC_TYPE];
    REQUIRE(index_loaded.HasRawData(metric_type) ==
            knowhere::IndexStaticFaced<T>::HasRawData(index_type,
                                                      knowhere::Version::GetCurrentVersion().VersionNumber(), conf));

    // test GetVectorByIds()
    if (index_loaded.HasRawData(metric_type)) {
        const auto rows = default_t_ds_ptr->GetRows();

        int64_t* ids = new int64_t[rows];
        for (int64_t i = 0; i < rows; i++) {
            ids[i] = i;
        }

        auto ids_ds = knowhere::GenIdsDataSet(rows, ids);
        ids_ds->SetIsOwner(true);

        auto vectors = index_loaded.GetVectorByIds(ids_ds);
        REQUIRE(vectors.has_value());

        match_datasets<T>(default_t_ds_ptr, vectors.value(), ids);
    }
    return index_file_name;
}

}  // namespace

TEST_CASE("Search for FAISS HNSW Indices", "Benchmark and validation") {
    // various constants and restrictions

    // metrics to test
    // const std::vector<std::string> DISTANCE_TYPES = {"L2", "IP", "COSINE"};
    //    const std::vector<std::string> DISTANCE_TYPES = {"IP"};  // use IP distances
    const std::vector<std::string> DISTANCE_TYPES = {"L2"};

    // // for benchmarking
    // const std::vector<int32_t> DIMS = {13, 16, 27};
    // const std::vector<int32_t> NBS = {16384, 9632 + 16384};
    // const int32_t NQ = 256;
    // const int32_t TOPK = 64;

    // for unit tests
    const std::vector<int32_t> DIMS = {1024};  // dimensions in hnsw
                                               //    const std::vector<int32_t> NBS = {100000};  // max_elements in hnsw
                                               //    const std::vector<int32_t> NBS = {256};    // max_elements in hnsw
                                               //    const std::vector<size_t> NBS = {256};
    const std::vector<int32_t> NBS = {100000};  // max_elements in hnsw
                                                // const std::vector<int32_t> NBS = {100};  // max_elements in hnsw

    const int32_t NQ = 100000;  // number of queries
                                //    const int32_t NQ = 100;  // number of queries

    const int32_t TOPK = 1;  // number of neighbors

    //    const std::vector<bool> MV_ONLYs = {false, true};
    const std::vector<bool> MV_ONLYs = {false};  // see certain data types (filtering)

    const std::vector<std::string> SQ_TYPES = {};  // Scalar Quantization (make vector data smaller types)

    // const std::vector<std::string> SQ_TYPES = {"SQ6", "SQ8", "BF16",
    //                                            "FP16"};  // Scalar Quantization (make vector data smaller types)

    // random bitset rates
    // 0.0 means unfiltered, 1.0 means all filtered out
    //    const std::vector<float> BITSET_RATES = {0.0f, 0.1f, 0.5f, 0.95f, 1.0f};
    // const std::vector<float> BITSET_RATES = {0.0f, 0.5f};
    const std::vector<float> BITSET_RATES = {0.0f};

    // todo: enable 10 and 12 bits when the PQ training code is provided
    // const std::vector<int32_t> NBITS = {8, 10, 12};
    const std::vector<int32_t> NBITS = {8};

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

    // accepted refines for PQ for FP32 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_FP32 = {{"SQ6", "SQ8", "BF16", "FP16", "FLAT"}};

    // accepted refines for PQ for FP16 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_FP16 = {{"SQ6", "SQ8", "FP16"}};

    // accepted refines for PQ for BF16 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_BF16 = {{"SQ6", "SQ8", "BF16"}};

    // create base json config
    knowhere::Json default_conf;

    default_conf[knowhere::indexparam::HNSW_M] = 32;  // hnsw , M parameter
    // default_conf[knowhere::indexparam::EFCONSTRUCTION] = 260;  // hnsw , efConstruction parameter
    // default_conf[knowhere::indexparam::EF] = 130;              // hnsw , ef parameter
    default_conf[knowhere::meta::TOPK] = TOPK;  // number of neighbors

    default_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;  // hnsw , efConstruction parameter
    default_conf[knowhere::indexparam::EF] = 100;              // hnsw , ef parameter

    std::cout << " hnsw_m: " << default_conf[knowhere::indexparam::HNSW_M]
              << " efConstruction: " << default_conf[knowhere::indexparam::EFCONSTRUCTION]
              << " ef: " << default_conf[knowhere::indexparam::EF] << " IP distance " << std::endl;

    // create golden indices for search
    {
        const std::string golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    knowhere::Json conf = default_conf;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;

                    std::vector<int32_t> golden_params = {(int)distance_type, dim, nb};

                    const uint64_t rng_seed = get_params_hash(golden_params);
                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);

                    // create or load a golden index
                    std::string golden_index_file_name =
                        get_index_name<knowhere::fp32>(ann_test_name_, golden_index_type, golden_params);

                    create_index<knowhere::fp32>(golden_index_type, golden_index_file_name, default_ds_ptr, conf, false,
                                                 "golden ");
                }
            }
        }
    }

    SECTION("FLAT") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_HNSW;
        const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenDataSet(NQ, dim, query_rng_seed);

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
                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);

                    // create or load a golden index
                    std::string golden_index_file_name =
                        get_index_name<knowhere::fp32>(ann_test_name_, golden_index_type, golden_params);

                    auto golden_index = create_index<knowhere::fp32>(golden_index_type, golden_index_file_name,
                                                                     default_ds_ptr, conf, false, "golden ");

                    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info =
                        GenerateScalarInfo(nb);
                    auto partition_size = scalar_info[0][0].size();  // will be masked by partition key value

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
                            const int32_t nbits_set = mv_only_enable
                                                          ? partition_size + (nb - partition_size) * bitset_rate
                                                          : nb * bitset_rate;
                            const std::vector<uint8_t> bitset_data =
                                mv_only_enable ? GenerateBitsetByScalarInfoAndFirstTBits(scalar_info[0][0], nb, 0)
                                               : GenerateBitsetWithRandomTbitsSet(nb, nbits_set);

                            // initialize bitset_view.
                            // provide a default one if nbits_set == 0
                            knowhere::BitsetView bitset_view = nullptr;
                            if (nbits_set != 0) {
                                bitset_view = knowhere::BitsetView(bitset_data.data(), nb, nb - nbits_set);
                            }

                            // get a golden result
                            auto golden_result = golden_index.Search(query_ds_ptr, conf, bitset_view);

                            // fp32 candidate
                            printf(
                                "\nProcessing HNSW,Flat fp32 for %s distance, dim=%d, nrows=%d, %d%% points filtered "
                                "out\n",
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb, int(bitset_rate * 100));

                            index_file = test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                                   params, conf, mv_only_enable, bitset_view);
                            index_files.emplace_back(index_file);

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
