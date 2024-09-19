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
#include "utils.h"

namespace {

knowhere::DataSetPtr
GenDataSet(int rows, int dim, const uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib(-1.0, 1.0);
    float* ts = new float[rows * dim];
    for (int i = 0; i < rows * dim; ++i) {
        ts[i] = (float)distrib(rng);
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

float
CalcRecall(const int64_t* g_ids, const int64_t* ids, int32_t nq, int32_t k) {
    int32_t hit = 0;
    for (int32_t i = 0; i < nq; i++) {
        std::unordered_set<int32_t> ground(g_ids + i * k, g_ids + (i + 1) * k);
        for (int32_t j = 0; j < k; j++) {
            auto id = ids[i * k + j];
            if (ground.count(id) > 0) {
                hit++;
            }
        }
    }
    return (hit * 1.0f / (nq * k));
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
             const knowhere::DataSetPtr& default_ds_ptr, const knowhere::Json& conf,
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
    } else {
        return ann_test_name + "_" + index_type + params_str + ".index";
    }
}

//
const std::string ann_test_name_ = "faiss_hnsw";

//
template <typename T>
void
test_hnsw(const knowhere::DataSetPtr& default_ds_ptr, const knowhere::DataSetPtr& query_ds_ptr,
          const knowhere::DataSetPtr& golden_result, const std::vector<int32_t>& index_params,
          const knowhere::Json& conf, bool expected_raw_data) {
    const std::string index_type = conf[knowhere::meta::INDEX_TYPE].get<std::string>();

    // load indices
    std::string index_file_name = get_index_name<T>(ann_test_name_, index_type, index_params);

    // training data
    auto default_t_ds_ptr = knowhere::ConvertToDataTypeIfNeeded<T>(default_ds_ptr);

    // our index
    // first, we create an index and save it
    auto index = create_index<T>(index_type, index_file_name, default_ds_ptr, conf);

    // then, we force it to be loaded in order to test load & save
    auto index_loaded = create_index<T>(index_type, index_file_name, default_ds_ptr, conf);

    // query
    auto query_t_ds_ptr = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds_ptr);

    auto result = index.Search(query_t_ds_ptr, conf, nullptr);
    auto result_loaded = index_loaded.Search(query_t_ds_ptr, conf, nullptr);

    // calc recall
    auto recall = CalcRecall(golden_result->GetIds(), result.value()->GetIds(), query_t_ds_ptr->GetRows(),
                             conf[knowhere::meta::TOPK].get<size_t>());

    auto recall_loaded = CalcRecall(golden_result->GetIds(), result_loaded.value()->GetIds(), query_t_ds_ptr->GetRows(),
                                    conf[knowhere::meta::TOPK].get<size_t>());

    printf("Recall is %f, %f\n", recall, recall_loaded);

    REQUIRE(recall >= 0.8);
    REQUIRE(recall_loaded >= 0.8);
    REQUIRE(recall == recall_loaded);

    // test HasRawData()
    auto metric_type = conf[knowhere::meta::METRIC_TYPE];
    REQUIRE(index_loaded.HasRawData(metric_type) == expected_raw_data);
    REQUIRE(knowhere::IndexStaticFaced<T>::HasRawData(
                index_type, knowhere::Version::GetCurrentVersion().VersionNumber(), conf) == expected_raw_data);

    // test GetVectorByIds()
    if (expected_raw_data) {
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
}

}  // namespace

TEST_CASE("FAISS HNSW Indices", "Benchmark and validation") {
    // various constants and restrictions

    // metrics to test
    const std::vector<std::string> DISTANCE_TYPES = {"L2", "IP", "COSINE"};

    // // for benchmarking
    // const std::vector<int32_t> DIMS = {13, 16, 27};
    // const std::vector<int32_t> NBS = {16384, 9632 + 16384};
    // const int32_t NQ = 256;
    // const int32_t TOPK = 64;

    // for unit tests
    const std::vector<int32_t> DIMS = {4};
    const std::vector<int32_t> NBS = {256};
    const int32_t NQ = 16;
    const int32_t TOPK = 16;
    const std::vector<std::string> SQ_TYPES = {"SQ6", "SQ8", "BF16", "FP16"};

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

    default_conf[knowhere::indexparam::HNSW_M] = 16;
    default_conf[knowhere::indexparam::EFCONSTRUCTION] = 96;
    default_conf[knowhere::indexparam::EF] = 64;
    default_conf[knowhere::meta::TOPK] = TOPK;

    // create golden indices
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

                    // create a golden index
                    std::string golden_index_file_name =
                        get_index_name<knowhere::fp32>(ann_test_name_, golden_index_type, golden_params);

                    create_index<knowhere::fp32>(golden_index_type, golden_index_file_name, default_ds_ptr, conf,
                                                 "golden ");
                }
            }
        }
    }

    // I'd like to have a sequential process here, because every item in the loop
    //   is parallelized on its own

    SECTION("FLAT") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_HNSW_FLAT;
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

                    // get a golden result
                    std::string golden_index_file_name =
                        get_index_name<knowhere::fp32>(ann_test_name_, golden_index_type, golden_params);

                    auto golden_index = create_index<knowhere::fp32>(golden_index_type, golden_index_file_name,
                                                                     default_ds_ptr, conf, "golden ");

                    auto golden_result = golden_index.Search(query_ds_ptr, conf, nullptr);

                    // fp32 candidate
                    printf("\nProcessing HNSW,Flat fp32 for %s distance, dim=%d, nrows=%d\n",
                           DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                    test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf, true);

                    // fp16 candidate
                    printf("\nProcessing HNSW,Flat fp16 for %s distance, dim=%d, nrows=%d\n",
                           DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                    test_hnsw<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf, true);

                    // bf32 candidate
                    printf("\nProcessing HNSW,Flat bf16 for %s distance, dim=%d, nrows=%d\n",
                           DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                    test_hnsw<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf, true);
                }
            }
        }
    }

    SECTION("SQ") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_HNSW_SQ;
        const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    // create golden conf
                    knowhere::Json conf_golden = default_conf;
                    conf_golden[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf_golden[knowhere::meta::DIM] = dim;
                    conf_golden[knowhere::meta::ROWS] = nb;

                    std::vector<int32_t> golden_params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(golden_params);
                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);

                    // get a golden result
                    std::string golden_index_file_name =
                        get_index_name<knowhere::fp32>(ann_test_name_, golden_index_type, golden_params);

                    auto golden_index = create_index<knowhere::fp32>(golden_index_type, golden_index_file_name,
                                                                     default_ds_ptr, conf_golden, "golden ");

                    auto golden_result = golden_index.Search(query_ds_ptr, conf_golden, nullptr);

                    // go SQ
                    for (size_t i_sq_type = 0; i_sq_type < SQ_TYPES.size(); i_sq_type++) {
                        knowhere::Json conf = conf_golden;
                        conf[knowhere::meta::INDEX_TYPE] = index_type;

                        const std::string sq_type = SQ_TYPES[i_sq_type];
                        conf[knowhere::indexparam::SQ_TYPE] = sq_type;

                        std::vector<int32_t> params = {(int)distance_type, dim, nb, (int)i_sq_type};

                        // fp32 candidate
                        printf("\nProcessing HNSW,SQ(%s) fp32 for %s distance, dim=%d, nrows=%d\n", sq_type.c_str(),
                               DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  false);

                        // fp16 candidate
                        printf("\nProcessing HNSW,SQ(%s) fp16 for %s distance, dim=%d, nrows=%d\n", sq_type.c_str(),
                               DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  sq_type == "FP16");

                        // bf16 candidate
                        printf("\nProcessing HNSW,SQ(%s) bf16 for %s distance, dim=%d, nrows=%d\n", sq_type.c_str(),
                               DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  sq_type == "BF16");

                        // test refines for FP32
                        {
                            const auto& allowed_refs = SQ_ALLOWED_REFINES_FP32[sq_type];
                            for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size(); allowed_ref_idx++) {
                                auto conf_refine = conf;
                                conf_refine["refine"] = true;
                                conf_refine["refine_k"] = 1.5;

                                const std::string allowed_ref = allowed_refs[allowed_ref_idx];
                                conf_refine["refine_type"] = allowed_ref;

                                std::vector<int32_t> params_refine = {(int)distance_type, dim, nb, (int)i_sq_type,
                                                                      (int)allowed_ref_idx};

                                // fp32 candidate
                                printf(
                                    "\nProcessing HNSW,SQ(%s) with %s refine, fp32 for %s distance, dim=%d, nrows=%d\n",
                                    sq_type.c_str(), allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(), dim,
                                    nb);

                                test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                          params_refine, conf_refine, allowed_ref == "FLAT");
                            }
                        }

                        // test refines for FP16
                        {
                            const auto& allowed_refs = SQ_ALLOWED_REFINES_FP16[sq_type];
                            for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size(); allowed_ref_idx++) {
                                auto conf_refine = conf;
                                conf_refine["refine"] = true;
                                conf_refine["refine_k"] = 1.5;

                                const std::string allowed_ref = allowed_refs[allowed_ref_idx];
                                conf_refine["refine_type"] = allowed_ref;

                                std::vector<int32_t> params_refine = {(int)distance_type, dim, nb, (int)i_sq_type,
                                                                      (int)allowed_ref_idx};

                                // fp16 candidate
                                printf(
                                    "\nProcessing HNSW,SQ(%s) with %s refine, fp16 for %s distance, dim=%d, nrows=%d\n",
                                    sq_type.c_str(), allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(), dim,
                                    nb);

                                test_hnsw<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                          params_refine, conf_refine, allowed_ref == "FP16");
                            }
                        }

                        // test refines for BF16
                        {
                            const auto& allowed_refs = SQ_ALLOWED_REFINES_BF16[sq_type];
                            for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size(); allowed_ref_idx++) {
                                auto conf_refine = conf;
                                conf_refine["refine"] = true;
                                conf_refine["refine_k"] = 1.5;

                                const std::string allowed_ref = allowed_refs[allowed_ref_idx];
                                conf_refine["refine_type"] = allowed_ref;

                                std::vector<int32_t> params_refine = {(int)distance_type, dim, nb, (int)i_sq_type,
                                                                      (int)allowed_ref_idx};

                                // bf16 candidate
                                printf(
                                    "\nProcessing HNSW,SQ(%s) with %s refine, bf16 for %s distance, dim=%d, nrows=%d\n",
                                    sq_type.c_str(), allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(), dim,
                                    nb);

                                test_hnsw<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                          params_refine, conf_refine, allowed_ref == "BF16");
                            }
                        }
                    }
                }
            }
        }
    }

    SECTION("PQ") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_HNSW_PQ;
        const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : {16}) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    // set up a golden cfg
                    knowhere::Json conf_golden = default_conf;
                    conf_golden[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf_golden[knowhere::meta::DIM] = dim;
                    conf_golden[knowhere::meta::ROWS] = nb;

                    std::vector<int32_t> golden_params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(golden_params);
                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);

                    // get a golden result
                    std::string golden_index_file_name =
                        get_index_name<knowhere::fp32>(ann_test_name_, golden_index_type, golden_params);

                    auto golden_index = create_index<knowhere::fp32>(golden_index_type, golden_index_file_name,
                                                                     default_ds_ptr, conf_golden, "golden ");

                    auto golden_result = golden_index.Search(query_ds_ptr, conf_golden, nullptr);

                    // go PQ
                    for (size_t nbits_type = 0; nbits_type < NBITS.size(); nbits_type++) {
                        const int pq_m = 8;

                        knowhere::Json conf = conf_golden;
                        conf[knowhere::meta::INDEX_TYPE] = index_type;
                        conf[knowhere::indexparam::NBITS] = NBITS[nbits_type];
                        conf[knowhere::indexparam::M] = pq_m;

                        std::vector<int32_t> params = {(int)distance_type, dim, nb, pq_m, (int)nbits_type};

                        // test fp32 candidate
                        printf("\nProcessing HNSW,PQ%dx%d fp32 for %s distance, dim=%d, nrows=%d\n", pq_m,
                               NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  false);

                        // test fp16 candidate
                        printf("\nProcessing HNSW,PQ%dx%d fp16 for %s distance, dim=%d, nrows=%d\n", pq_m,
                               NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  false);

                        // test bf16 candidate
                        printf("\nProcessing HNSW,PQ%dx%d bf16 for %s distance, dim=%d, nrows=%d\n", pq_m,
                               NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  false);

                        // test refines for fp32
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP32.size();
                             allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;

                            const std::string allowed_ref = PQ_ALLOWED_REFINES_FP32[allowed_ref_idx];
                            conf_refine["refine_type"] = allowed_ref;

                            std::vector<int32_t> params_refine = {(int)distance_type,  dim, nb, pq_m, (int)nbits_type,
                                                                  (int)allowed_ref_idx};

                            // test fp32 candidate
                            printf("\nProcessing HNSW,PQ%dx%d with %s refine, fp32 for %s distance, dim=%d, nrows=%d\n",
                                   pq_m, NBITS[nbits_type], allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(),
                                   dim, nb);

                            test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                      params_refine, conf_refine, allowed_ref == "FLAT");
                        }

                        // test refines for fp16
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP16.size();
                             allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;

                            const std::string allowed_ref = PQ_ALLOWED_REFINES_FP16[allowed_ref_idx];
                            conf_refine["refine_type"] = allowed_ref;

                            std::vector<int32_t> params_refine = {(int)distance_type,  dim, nb, pq_m, (int)nbits_type,
                                                                  (int)allowed_ref_idx};

                            // test fp16 candidate
                            printf("\nProcessing HNSW,PQ%dx%d with %s refine, fp16 for %s distance, dim=%d, nrows=%d\n",
                                   pq_m, NBITS[nbits_type], allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(),
                                   dim, nb);

                            test_hnsw<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                      params_refine, conf_refine, allowed_ref == "FP16");
                        }

                        // test refines for bf16
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_BF16.size();
                             allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;

                            const std::string allowed_ref = PQ_ALLOWED_REFINES_BF16[allowed_ref_idx];
                            conf_refine["refine_type"] = allowed_ref;

                            std::vector<int32_t> params_refine = {(int)distance_type,  dim, nb, pq_m, (int)nbits_type,
                                                                  (int)allowed_ref_idx};

                            // test bf16 candidate
                            printf("\nProcessing HNSW,PQ%dx%d with %s refine, bf16 for %s distance, dim=%d, nrows=%d\n",
                                   pq_m, NBITS[nbits_type], allowed_ref.c_str(), DISTANCE_TYPES[distance_type].c_str(),
                                   dim, nb);

                            test_hnsw<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                      params_refine, conf_refine, allowed_ref == "BF16");
                        }
                    }
                }
            }
        }
    }

    SECTION("PRQ") {
        const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_HNSW_PRQ;
        const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : {16}) {
                // generate a query
                const uint64_t query_rng_seed = get_params_hash({(int)distance_type, dim});
                auto query_ds_ptr = GenDataSet(NQ, dim, query_rng_seed);

                for (const int32_t nb : NBS) {
                    // set up a golden cfg
                    knowhere::Json conf_golden = default_conf;
                    conf_golden[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf_golden[knowhere::meta::DIM] = dim;
                    conf_golden[knowhere::meta::ROWS] = nb;

                    std::vector<int32_t> golden_params = {(int)distance_type, dim, nb};

                    // generate a default dataset
                    const uint64_t rng_seed = get_params_hash(golden_params);
                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);

                    // get a golden result
                    std::string golden_index_file_name =
                        get_index_name<knowhere::fp32>(ann_test_name_, golden_index_type, golden_params);

                    auto golden_index = create_index<knowhere::fp32>(golden_index_type, golden_index_file_name,
                                                                     default_ds_ptr, conf_golden, "golden ");

                    auto golden_result = golden_index.Search(query_ds_ptr, conf_golden, nullptr);

                    // go PRQ
                    for (size_t nbits_type = 0; nbits_type < NBITS.size(); nbits_type++) {
                        const int prq_m = 4;
                        const int prq_num = 2;

                        knowhere::Json conf = conf_golden;
                        conf[knowhere::meta::INDEX_TYPE] = index_type;
                        conf[knowhere::indexparam::NBITS] = NBITS[nbits_type];
                        conf[knowhere::indexparam::M] = prq_m;
                        conf[knowhere::indexparam::PRQ_NUM] = prq_num;

                        std::vector<int32_t> params = {(int)distance_type, dim, nb, prq_m, prq_num, (int)nbits_type};

                        // test fp32 candidate
                        printf("\nProcessing HNSW,PRQ%dx%dx%d fp32 for %s distance, dim=%d, nrows=%d\n", prq_num, prq_m,
                               NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  false);

                        // test fp16 candidate
                        printf("\nProcessing HNSW,PRQ%dx%dx%d fp16 for %s distance, dim=%d, nrows=%d\n", prq_num, prq_m,
                               NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  false);

                        // test bf16 candidate
                        printf("\nProcessing HNSW,PRQ%dx%dx%d bf16 for %s distance, dim=%d, nrows=%d\n", prq_num, prq_m,
                               NBITS[nbits_type], DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                        test_hnsw<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf,
                                                  false);

                        // test fp32 refines
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP32.size();
                             allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;

                            const std::string allowed_ref = PQ_ALLOWED_REFINES_FP32[allowed_ref_idx];
                            conf_refine["refine_type"] = allowed_ref;

                            std::vector<int32_t> params_refine = {
                                (int)distance_type, dim, nb, prq_m, prq_num, (int)nbits_type, (int)allowed_ref_idx};

                            //
                            printf(
                                "\nProcessing HNSW,PRQ%dx%dx%d with %s refine, fp32 for %s distance, dim=%d, "
                                "nrows=%d\n",
                                prq_num, prq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                            // test a candidate
                            test_hnsw<knowhere::fp32>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                      params_refine, conf_refine, allowed_ref == "FLAT");
                        }

                        // test fp16 refines
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP16.size();
                             allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;

                            const std::string allowed_ref = PQ_ALLOWED_REFINES_FP16[allowed_ref_idx];
                            conf_refine["refine_type"] = allowed_ref;

                            std::vector<int32_t> params_refine = {
                                (int)distance_type, dim, nb, prq_m, prq_num, (int)nbits_type, (int)allowed_ref_idx};

                            //
                            printf(
                                "\nProcessing HNSW,PRQ%dx%dx%d with %s refine, fp16 for %s distance, dim=%d, "
                                "nrows=%d\n",
                                prq_num, prq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                            // test a candidate
                            test_hnsw<knowhere::fp16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                      params_refine, conf_refine, allowed_ref == "FP16");
                        }

                        // test bf16 refines
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_BF16.size();
                             allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;

                            const std::string allowed_ref = PQ_ALLOWED_REFINES_BF16[allowed_ref_idx];
                            conf_refine["refine_type"] = allowed_ref;

                            std::vector<int32_t> params_refine = {
                                (int)distance_type, dim, nb, prq_m, prq_num, (int)nbits_type, (int)allowed_ref_idx};

                            //
                            printf(
                                "\nProcessing HNSW,PRQ%dx%dx%d with %s refine, bf16 for %s distance, dim=%d, "
                                "nrows=%d\n",
                                prq_num, prq_m, NBITS[nbits_type], allowed_ref.c_str(),
                                DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                            // test a candidate
                            test_hnsw<knowhere::bf16>(default_ds_ptr, query_ds_ptr, golden_result.value(),
                                                      params_refine, conf_refine, allowed_ref == "BF16");
                        }
                    }
                }
            }
        }
    }
}
