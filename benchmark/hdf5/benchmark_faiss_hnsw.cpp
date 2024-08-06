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

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "benchmark_knowhere.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"


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

// unlike other benchmarks, this one operates on a synthetic data
//   and verifies the correctness of many-many variants of FAISS HNSW indices.
class Benchmark_Faiss_Hnsw : public Benchmark_knowhere, public ::testing::Test {
public:
    template<typename T>
    void test_hnsw(
        const knowhere::DataSetPtr& default_ds_ptr,
        const knowhere::DataSetPtr& query_ds_ptr,
        const knowhere::DataSetPtr& golden_result,
        const std::vector<int32_t>& index_params,
        const knowhere::Json& conf
    ) {
        const std::string index_type = conf[knowhere::meta::INDEX_TYPE].get<std::string>();

        // load indices
        std::string index_file_name = get_index_name<T>(
            ann_test_name_, index_type, index_params);

        // our index
        // first, we create an index and save it
        auto index = create_index<T>(
            index_type,
            index_file_name,
            default_ds_ptr,
            conf
        );

        // then, we force it to be loaded in order to test load & save
        auto index_loaded = create_index<T>(
            index_type,
            index_file_name,
            default_ds_ptr,
            conf
        );

        auto query_t_ds_ptr = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds_ptr);

        auto result = index.Search(query_t_ds_ptr, conf, nullptr);
        auto result_loaded = index_loaded.Search(query_t_ds_ptr, conf, nullptr);

        // calc recall
        auto recall = this->CalcRecall(
            golden_result->GetIds(),
            result.value()->GetIds(),
            query_t_ds_ptr->GetRows(),
            conf[knowhere::meta::TOPK].get<size_t>()
        );

        auto recall_loaded = this->CalcRecall(
            golden_result->GetIds(),
            result_loaded.value()->GetIds(),
            query_t_ds_ptr->GetRows(),
            conf[knowhere::meta::TOPK].get<size_t>()
        );

        printf("Recall is %f, %f\n", recall, recall_loaded);

        ASSERT_GE(recall, 0.9);
        ASSERT_GE(recall_loaded, 0.9);
    }

protected:
    void
    SetUp() override {
        T0_ = elapsed();
        set_ann_test_name("faiss_hnsw");

        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX2);

        cfg_[knowhere::indexparam::HNSW_M] = 16;
        cfg_[knowhere::indexparam::EFCONSTRUCTION] = 96;
        cfg_[knowhere::indexparam::EF] = 64;
        cfg_[knowhere::meta::TOPK] = TOPK;

        // create baseline indices here
        CreateGoldenIndices();
    }

    void CreateGoldenIndices() {
        const std::string golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

        uint64_t rng_seed = 1;
        for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
            for (const int32_t dim : DIMS) {
                for (const int32_t nb : NBS) {
                    knowhere::Json conf = cfg_;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb};

                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);
                    rng_seed += 1;

                    // create a golden index
                    std::string golden_index_file_name = get_index_name<knowhere::fp32>(
                        ann_test_name_, golden_index_type, params);

                    create_index<knowhere::fp32>(
                        golden_index_type,
                        golden_index_file_name,
                        default_ds_ptr,
                        conf,
                        "golden "
                    );
                }
            }
        }
    }

    const std::vector<std::string> DISTANCE_TYPES = {"L2", "IP", "COSINE"};
    const std::vector<int32_t> DIMS = {13, 16, 27};
    const std::vector<int32_t> NBS = {16384, 9632 + 16384};
    const int32_t NQ = 256;
    const int32_t TOPK = 64;

    const std::vector<std::string> SQ_TYPES = {"SQ6", "SQ8", "BF16", "FP16"};

    // todo: enable 10 and 12 bits when the PQ training code is provided
    // const std::vector<int32_t> NBITS = {8, 10, 12};
    const std::vector<int32_t> NBITS = {8};

    // accepted refines for a given SQ type for a FP32 data type
    std::unordered_map<std::string, std::vector<std::string>> SQ_ALLOWED_REFINES_FP32 = {
        { "SQ6", {"SQ8", "BF16", "FP16", "FLAT"} },
        { "SQ8", {"BF16", "FP16", "FLAT"} },
        { "BF16", {"FLAT"} },
        { "FP16", {"FLAT"} }
    };

    // accepted refines for a given SQ type for a FP16 data type
    std::unordered_map<std::string, std::vector<std::string>> SQ_ALLOWED_REFINES_FP16 = {
        { "SQ6", {"SQ8", "FP16"} },
        { "SQ8", {"FP16"} },
        { "BF16", {} },
        { "FP16", {} }
    };

    // accepted refines for a given SQ type for a BF16 data type
    std::unordered_map<std::string, std::vector<std::string>> SQ_ALLOWED_REFINES_BF16 = {
        { "SQ6", {"SQ8", "BF16"} },
        { "SQ8", {"BF16"} },
        { "BF16", {} },
        { "FP16", {} }
    };

    // accepted refines for PQ for FP32 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_FP32 = {
        {"SQ6", "SQ8", "BF16", "FP16", "FLAT"}
    };

    // accepted refines for PQ for FP16 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_FP16 = {
        {"SQ6", "SQ8", "FP16"}
    };

    // accepted refines for PQ for BF16 data type
    std::vector<std::string> PQ_ALLOWED_REFINES_BF16 = {
        {"SQ6", "SQ8", "BF16"}
    };
};


//
TEST_F(Benchmark_Faiss_Hnsw, TEST_HNSWFLAT) {
    const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_HNSW_FLAT;
    const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    uint64_t rng_seed = 1;
    for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
        for (const int32_t dim : DIMS) {
            auto query_ds_ptr = GenDataSet(NQ, dim, rng_seed + 1234567);

            for (const int32_t nb : NBS) {
                knowhere::Json conf = cfg_;
                conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                conf[knowhere::meta::DIM] = dim;
                conf[knowhere::meta::ROWS] = nb;
                conf[knowhere::meta::INDEX_TYPE] = index_type;

                std::vector<int32_t> params = {(int)distance_type, dim, nb};

                // generate a default dataset
                auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);
                rng_seed += 1;

                // get a golden result
                std::string golden_index_file_name = get_index_name<knowhere::fp32>(
                    ann_test_name_, golden_index_type, params);

                auto golden_index = create_index<knowhere::fp32>(
                    golden_index_type,
                    golden_index_file_name,
                    default_ds_ptr,
                    conf,
                    "golden "
                );

                auto golden_result = golden_index.Search(query_ds_ptr, conf, nullptr);


                // fp32 candidate
                printf("\nProcessing HNSW,Flat fp32 for %s distance, dim=%d, nrows=%d\n",
                    DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                test_hnsw<knowhere::fp32>(
                    default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                // fp16 candidate
                printf("\nProcessing HNSW,Flat fp16 for %s distance, dim=%d, nrows=%d\n",
                    DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                test_hnsw<knowhere::fp16>(
                    default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                // bf32 candidate
                printf("\nProcessing HNSW,Flat bf16 for %s distance, dim=%d, nrows=%d\n",
                    DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                test_hnsw<knowhere::bf16>(
                    default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);
            }
        }
    }
}

//
TEST_F(Benchmark_Faiss_Hnsw, TEST_HNSWSQ) {
    const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_HNSW_SQ;
    const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    uint64_t rng_seed = 1;
    for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
        for (const int32_t dim : DIMS) {
            auto query_ds_ptr = GenDataSet(NQ, dim, rng_seed + 1234567);

            for (const int32_t nb : NBS) {
                for (size_t sq_type = 0; sq_type < SQ_TYPES.size(); sq_type++) {
                    knowhere::Json conf = cfg_;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;
                    conf[knowhere::indexparam::SQ_TYPE] = SQ_TYPES[sq_type];

                    std::vector<int32_t> params = {(int)distance_type, dim, nb, (int)sq_type};

                    // generate a default dataset
                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);
                    rng_seed += 1;

                    // get a golden result
                    std::string golden_index_file_name = get_index_name<knowhere::fp32>(
                        ann_test_name_, golden_index_type, params);

                    auto golden_index = create_index<knowhere::fp32>(
                        golden_index_type,
                        golden_index_file_name,
                        default_ds_ptr,
                        conf,
                        "golden "
                    );

                    auto golden_result = golden_index.Search(query_ds_ptr, conf, nullptr);


                    // fp32 candidate
                    printf("\nProcessing HNSW,SQ(%s) fp32 for %s distance, dim=%d, nrows=%d\n",
                        SQ_TYPES[sq_type].c_str(), DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                    test_hnsw<knowhere::fp32>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                    // fp16 candidate
                    printf("\nProcessing HNSW,SQ(%s) fp16 for %s distance, dim=%d, nrows=%d\n",
                        SQ_TYPES[sq_type].c_str(), DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                    test_hnsw<knowhere::fp16>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                    // bf16 candidate
                    printf("\nProcessing HNSW,SQ(%s) bf16 for %s distance, dim=%d, nrows=%d\n",
                        SQ_TYPES[sq_type].c_str(), DISTANCE_TYPES[distance_type].c_str(), dim, nb);

                    test_hnsw<knowhere::bf16>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);


                    // test refines for FP32
                    {
                        const auto& allowed_refs = SQ_ALLOWED_REFINES_FP32[SQ_TYPES[sq_type]];
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size(); allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;
                            conf_refine["refine_type"] = allowed_refs[allowed_ref_idx];

                            std::vector<int32_t> params_refine =
                                {(int)distance_type, dim, nb, (int)sq_type, (int)allowed_ref_idx};

                            // fp32 candidate
                            printf("\nProcessing HNSW,SQ(%s) with %s refine, fp32 for %s distance, dim=%d, nrows=%d\n",
                                SQ_TYPES[sq_type].c_str(),
                                allowed_refs[allowed_ref_idx].c_str(),
                                DISTANCE_TYPES[distance_type].c_str(),
                                dim,
                                nb);

                            test_hnsw<knowhere::fp32>(
                                default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                        }
                    }

                    // test refines for FP16
                    {
                        const auto& allowed_refs = SQ_ALLOWED_REFINES_FP16[SQ_TYPES[sq_type]];
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size(); allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;
                            conf_refine["refine_type"] = allowed_refs[allowed_ref_idx];

                            std::vector<int32_t> params_refine =
                                {(int)distance_type, dim, nb, (int)sq_type, (int)allowed_ref_idx};

                            // fp16 candidate
                            printf("\nProcessing HNSW,SQ(%s) with %s refine, fp16 for %s distance, dim=%d, nrows=%d\n",
                                SQ_TYPES[sq_type].c_str(),
                                allowed_refs[allowed_ref_idx].c_str(),
                                DISTANCE_TYPES[distance_type].c_str(),
                                dim,
                                nb);

                            test_hnsw<knowhere::fp16>(
                                default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                        }
                    }

                    // test refines for BF16
                    {
                        const auto& allowed_refs = SQ_ALLOWED_REFINES_BF16[SQ_TYPES[sq_type]];
                        for (size_t allowed_ref_idx = 0; allowed_ref_idx < allowed_refs.size(); allowed_ref_idx++) {
                            auto conf_refine = conf;
                            conf_refine["refine"] = true;
                            conf_refine["refine_k"] = 1.5;
                            conf_refine["refine_type"] = allowed_refs[allowed_ref_idx];

                            std::vector<int32_t> params_refine =
                                {(int)distance_type, dim, nb, (int)sq_type, (int)allowed_ref_idx};

                            // bf16 candidate
                            printf("\nProcessing HNSW,SQ(%s) with %s refine, bf16 for %s distance, dim=%d, nrows=%d\n",
                                SQ_TYPES[sq_type].c_str(),
                                allowed_refs[allowed_ref_idx].c_str(),
                                DISTANCE_TYPES[distance_type].c_str(),
                                dim,
                                nb);

                            test_hnsw<knowhere::bf16>(
                                default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                        }
                    }
                }
            }
        }
    }
}


//
TEST_F(Benchmark_Faiss_Hnsw, TEST_HNSWPQ) {
    const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_HNSW_PQ;
    const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    uint64_t rng_seed = 1;
    for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
        for (const int32_t dim : {16}) {
            auto query_ds_ptr = GenDataSet(NQ, dim, rng_seed + 1234567);

            for (const int32_t nb : NBS) {
                for (size_t nbits_type = 0; nbits_type < NBITS.size(); nbits_type++) {
                    const int pq_m = 4;

                    knowhere::Json conf = cfg_;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;
                    conf[knowhere::indexparam::NBITS] = NBITS[nbits_type];
                    conf[knowhere::indexparam::M] = pq_m;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb, pq_m, (int)nbits_type};

                    // generate a default dataset
                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);
                    rng_seed += 1;

                    // get a golden result
                    std::string golden_index_file_name = get_index_name<knowhere::fp32>(
                        ann_test_name_, golden_index_type, params);

                    auto golden_index = create_index<knowhere::fp32>(
                        golden_index_type,
                        golden_index_file_name,
                        default_ds_ptr,
                        conf,
                        "golden "
                    );

                    auto golden_result = golden_index.Search(query_ds_ptr, conf, nullptr);

                    // test fp32 candidate
                    printf("\nProcessing HNSW,PQ%dx%d fp32 for %s distance, dim=%d, nrows=%d\n",
                        pq_m,
                        NBITS[nbits_type],
                        DISTANCE_TYPES[distance_type].c_str(),
                        dim,
                        nb);

                    test_hnsw<knowhere::fp32>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                    // test fp16 candidate
                    printf("\nProcessing HNSW,PQ%dx%d fp16 for %s distance, dim=%d, nrows=%d\n",
                        pq_m,
                        NBITS[nbits_type],
                        DISTANCE_TYPES[distance_type].c_str(),
                        dim,
                        nb);

                    test_hnsw<knowhere::fp16>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                    // test bf16 candidate
                    printf("\nProcessing HNSW,PQ%dx%d bf16 for %s distance, dim=%d, nrows=%d\n",
                        pq_m,
                        NBITS[nbits_type],
                        DISTANCE_TYPES[distance_type].c_str(),
                        dim,
                        nb);

                    test_hnsw<knowhere::bf16>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                    // test refines for fp32
                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP32.size(); allowed_ref_idx++) {
                        auto conf_refine = conf;
                        conf_refine["refine"] = true;
                        conf_refine["refine_k"] = 1.5;
                        conf_refine["refine_type"] = PQ_ALLOWED_REFINES_FP32[allowed_ref_idx];

                        std::vector<int32_t> params_refine =
                            {(int)distance_type, dim, nb, pq_m, (int)nbits_type, (int)allowed_ref_idx};

                        // test fp32 candidate
                        printf("\nProcessing HNSW,PQ%dx%d with %s refine, fp32 for %s distance, dim=%d, nrows=%d\n",
                            pq_m,
                            NBITS[nbits_type],
                            PQ_ALLOWED_REFINES_FP32[allowed_ref_idx].c_str(),
                            DISTANCE_TYPES[distance_type].c_str(),
                            dim,
                            nb);

                        test_hnsw<knowhere::fp32>(
                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                    }

                    // test refines for fp16
                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP16.size(); allowed_ref_idx++) {
                        auto conf_refine = conf;
                        conf_refine["refine"] = true;
                        conf_refine["refine_k"] = 1.5;
                        conf_refine["refine_type"] = PQ_ALLOWED_REFINES_FP16[allowed_ref_idx];

                        std::vector<int32_t> params_refine =
                            {(int)distance_type, dim, nb, pq_m, (int)nbits_type, (int)allowed_ref_idx};

                        // test fp16 candidate
                        printf("\nProcessing HNSW,PQ%dx%d with %s refine, fp16 for %s distance, dim=%d, nrows=%d\n",
                            pq_m,
                            NBITS[nbits_type],
                            PQ_ALLOWED_REFINES_FP16[allowed_ref_idx].c_str(),
                            DISTANCE_TYPES[distance_type].c_str(),
                            dim,
                            nb);

                        test_hnsw<knowhere::fp16>(
                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                    }

                    // test refines for bf16
                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_BF16.size(); allowed_ref_idx++) {
                        auto conf_refine = conf;
                        conf_refine["refine"] = true;
                        conf_refine["refine_k"] = 1.5;
                        conf_refine["refine_type"] = PQ_ALLOWED_REFINES_BF16[allowed_ref_idx];

                        std::vector<int32_t> params_refine =
                            {(int)distance_type, dim, nb, pq_m, (int)nbits_type, (int)allowed_ref_idx};

                        // test bf16 candidate
                        printf("\nProcessing HNSW,PQ%dx%d with %s refine, bf16 for %s distance, dim=%d, nrows=%d\n",
                            pq_m,
                            NBITS[nbits_type],
                            PQ_ALLOWED_REFINES_BF16[allowed_ref_idx].c_str(),
                            DISTANCE_TYPES[distance_type].c_str(),
                            dim,
                            nb);

                        test_hnsw<knowhere::bf16>(
                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                    }
                }
            }
        }
    }
}

//
TEST_F(Benchmark_Faiss_Hnsw, TEST_HNSWPRQ) {
    const std::string& index_type = knowhere::IndexEnum::INDEX_FAISS_HNSW_PRQ;
    const std::string& golden_index_type = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    uint64_t rng_seed = 1;
    for (size_t distance_type = 0; distance_type < DISTANCE_TYPES.size(); distance_type++) {
        for (const int32_t dim : {16}) {
            auto query_ds_ptr = GenDataSet(NQ, dim, rng_seed + 1234567);

            for (const int32_t nb : NBS) {
                for (size_t nbits_type = 0; nbits_type < NBITS.size(); nbits_type++) {
                    const int prq_m = 4;
                    const int prq_num = 2;

                    knowhere::Json conf = cfg_;
                    conf[knowhere::meta::METRIC_TYPE] = DISTANCE_TYPES[distance_type];
                    conf[knowhere::meta::DIM] = dim;
                    conf[knowhere::meta::ROWS] = nb;
                    conf[knowhere::meta::INDEX_TYPE] = index_type;
                    conf[knowhere::indexparam::NBITS] = NBITS[nbits_type];
                    conf[knowhere::indexparam::M] = prq_m;
                    conf[knowhere::indexparam::PRQ_NUM] = prq_num;

                    std::vector<int32_t> params = {(int)distance_type, dim, nb, prq_m, prq_num, (int)nbits_type};

                    // generate a default dataset
                    auto default_ds_ptr = GenDataSet(nb, dim, rng_seed);
                    rng_seed += 1;

                    // get a golden result
                    std::string golden_index_file_name = get_index_name<knowhere::fp32>(
                        ann_test_name_, golden_index_type, params);

                    auto golden_index = create_index<knowhere::fp32>(
                        golden_index_type,
                        golden_index_file_name,
                        default_ds_ptr,
                        conf,
                        "golden "
                    );

                    auto golden_result = golden_index.Search(query_ds_ptr, conf, nullptr);

                    // test fp32 candidate
                    printf("\nProcessing HNSW,PRQ%dx%dx%d fp32 for %s distance, dim=%d, nrows=%d\n",
                        prq_num,
                        prq_m,
                        NBITS[nbits_type],
                        DISTANCE_TYPES[distance_type].c_str(),
                        dim,
                        nb);

                    test_hnsw<knowhere::fp32>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                    // test fp16 candidate
                    printf("\nProcessing HNSW,PRQ%dx%dx%d fp16 for %s distance, dim=%d, nrows=%d\n",
                        prq_num,
                        prq_m,
                        NBITS[nbits_type],
                        DISTANCE_TYPES[distance_type].c_str(),
                        dim,
                        nb);

                    test_hnsw<knowhere::fp16>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                    // test bf16 candidate
                    printf("\nProcessing HNSW,PRQ%dx%dx%d bf16 for %s distance, dim=%d, nrows=%d\n",
                        prq_num,
                        prq_m,
                        NBITS[nbits_type],
                        DISTANCE_TYPES[distance_type].c_str(),
                        dim,
                        nb);

                    test_hnsw<knowhere::bf16>(
                        default_ds_ptr, query_ds_ptr, golden_result.value(), params, conf);

                    // test fp32 refines
                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP32.size(); allowed_ref_idx++) {
                        auto conf_refine = conf;
                        conf_refine["refine"] = true;
                        conf_refine["refine_k"] = 1.5;
                        conf_refine["refine_type"] = PQ_ALLOWED_REFINES_FP32[allowed_ref_idx];

                        std::vector<int32_t> params_refine =
                            {(int)distance_type, dim, nb, prq_m, prq_num, (int)nbits_type, (int)allowed_ref_idx};

                        //
                        printf("\nProcessing HNSW,PRQ%dx%dx%d with %s refine, fp32 for %s distance, dim=%d, nrows=%d\n",
                            prq_num,
                            prq_m,
                            NBITS[nbits_type],
                            PQ_ALLOWED_REFINES_FP32[allowed_ref_idx].c_str(),
                            DISTANCE_TYPES[distance_type].c_str(),
                            dim,
                            nb);

                        // test a candidate
                        test_hnsw<knowhere::fp32>(
                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                    }

                    // test fp16 refines
                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_FP16.size(); allowed_ref_idx++) {
                        auto conf_refine = conf;
                        conf_refine["refine"] = true;
                        conf_refine["refine_k"] = 1.5;
                        conf_refine["refine_type"] = PQ_ALLOWED_REFINES_FP16[allowed_ref_idx];

                        std::vector<int32_t> params_refine =
                            {(int)distance_type, dim, nb, prq_m, prq_num, (int)nbits_type, (int)allowed_ref_idx};

                        //
                        printf("\nProcessing HNSW,PRQ%dx%dx%d with %s refine, fp16 for %s distance, dim=%d, nrows=%d\n",
                            prq_num,
                            prq_m,
                            NBITS[nbits_type],
                            PQ_ALLOWED_REFINES_FP16[allowed_ref_idx].c_str(),
                            DISTANCE_TYPES[distance_type].c_str(),
                            dim,
                            nb);

                        // test a candidate
                        test_hnsw<knowhere::fp16>(
                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                    }

                    // test bf16 refines
                    for (size_t allowed_ref_idx = 0; allowed_ref_idx < PQ_ALLOWED_REFINES_BF16.size(); allowed_ref_idx++) {
                        auto conf_refine = conf;
                        conf_refine["refine"] = true;
                        conf_refine["refine_k"] = 1.5;
                        conf_refine["refine_type"] = PQ_ALLOWED_REFINES_BF16[allowed_ref_idx];

                        std::vector<int32_t> params_refine =
                            {(int)distance_type, dim, nb, prq_m, prq_num, (int)nbits_type, (int)allowed_ref_idx};

                        //
                        printf("\nProcessing HNSW,PRQ%dx%dx%d with %s refine, bf16 for %s distance, dim=%d, nrows=%d\n",
                            prq_num,
                            prq_m,
                            NBITS[nbits_type],
                            PQ_ALLOWED_REFINES_BF16[allowed_ref_idx].c_str(),
                            DISTANCE_TYPES[distance_type].c_str(),
                            dim,
                            nb);

                        // test a candidate
                        test_hnsw<knowhere::bf16>(
                            default_ds_ptr, query_ds_ptr, golden_result.value(), params_refine, conf_refine);
                    }
                }
            }
        }
    }
}

