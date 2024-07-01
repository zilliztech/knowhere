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

#include <gtest/gtest.h>

#include <vector>

#include "benchmark_knowhere.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"

class Benchmark_binary : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_binary_idmap(const knowhere::Json& cfg) {
        auto conf = cfg;

        printf("\n[%0.3f s] %s | %s \n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
            for (auto k : TOPKs_) {
                conf[knowhere::meta::TOPK] = k;
                CALC_TIME_SPAN(auto result = index_.value().Search(ds_ptr, conf, nullptr));
                auto ids = result.value()->GetIds();
                float recall = CalcRecall(ids, nq, k);
                printf("  nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", nq, k, t_diff, recall);
                std::fflush(stdout);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_binary_ivf(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto nlist = conf[knowhere::indexparam::NLIST].get<int64_t>();

        printf("\n[%0.3f s] %s | %s | nlist=%ld\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               nlist);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            conf[knowhere::indexparam::NPROBE] = nprobe;
            for (auto nq : NQs_) {
                knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    conf[knowhere::meta::TOPK] = k;
                    CALC_TIME_SPAN(auto result = index_.value().Search(ds_ptr, conf, nullptr));
                    auto ids = result.value()->GetIds();
                    float recall = CalcRecall(ids, nq, k);
                    printf("  nprobe = %4d, nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", nprobe, nq, k, t_diff,
                           recall);
                    std::fflush(stdout);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_binary_hnsw(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto M = conf[knowhere::indexparam::HNSW_M].get<int64_t>();
        auto efc = conf[knowhere::indexparam::EFCONSTRUCTION].get<int64_t>();

        printf("\n[%0.3f s] %s | %s | M=%ld | efc=%ld\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               M, efc);
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            conf[knowhere::indexparam::EF] = ef;
            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    conf[knowhere::meta::TOPK] = k;
                    CALC_TIME_SPAN(auto result = index_.value().Search(ds_ptr, conf, nullptr));
                    auto ids = result.value()->GetIds();
                    float recall = CalcRecall(ids, nq, k);
                    printf("  ef = %4d, nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n", ef, nq, k, t_diff, recall);
                    std::fflush(stdout);
                }
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        // set_ann_test_name("sift-128-euclidean");
        set_ann_test_name("sift-4096-hamming");
        parse_ann_test_name();
        load_hdf5_data<true>();

        cfg_[knowhere::meta::METRIC_TYPE] = metric_type_;
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX2);
        printf("faiss::distance_compute_blas_threshold: %ld\n", knowhere::KnowhereConfig::GetBlasThreshold());
    }

    void
    TearDown() override {
        free_all();
    }

 protected:
    const std::vector<int32_t> NQs_ = {10000};
    const std::vector<int32_t> TOPKs_ = {100};

    // IVF index params
    const std::vector<int32_t> NLISTs_ = {1024};
    const std::vector<int32_t> NPROBEs_ = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    // HNSW index params
    const std::vector<int32_t> HNSW_Ms_ = {16};
    const std::vector<int32_t> EFCONs_ = {200};
    const std::vector<int32_t> EFs_ = {128, 256, 512};
};

// This testcase can be used to generate binary sift1m HDF5 file
// Following these steps:
//   1. set_ann_test_name("sift-128-euclidean")
//   2. use load_hdf5_data<false>();
//   3. change metric type to expected value (hamming/jaccard/tanimoto) manually
//   4. specify the hdf5 file name to generate
//   5. run this testcase
#if 0
TEST_F(Benchmark_binary, TEST_CREATE_BINARY_HDF5) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP;

    knowhere::Config conf = cfg_;
    std::string index_file_name = get_index_name({});

    // use sift1m data as binary data
    dim_ *= 32;
    metric_type_ = knowhere::metric::HAMMING;
    knowhere::SetMetaMetricType(conf, metric_type_);

    create_index(index_file_name, conf);
    index_->Load(binary_set_);

    knowhere::DatasetPtr ds_ptr = knowhere::GenDataset(nq_, dim_, xq_);
    knowhere::SetMetaTopk(conf, gt_k_);
    auto result = index_->Query(ds_ptr, conf, nullptr);

    auto gt_ids = knowhere::GetDatasetIDs(result);
    auto gt_dist = knowhere::GetDatasetDistance(result);

    auto gt_ids_int = new int32_t[gt_k_ * nq_];
    for (int32_t i = 0; i < gt_k_ * nq_; i++) {
        gt_ids_int[i] = gt_ids[i];
    }

    assert(dim_ == 4096);
    assert(nq_ == 10000);
    assert(gt_k_ == 100);
    hdf5_write<true>("sift-4096-hamming.hdf5", dim_/32, gt_k_, xb_, nb_, xq_, nq_, gt_ids_int, gt_dist);

    delete[] gt_ids_int;
}
#endif

TEST_F(Benchmark_binary, TEST_BINARY_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP;

    knowhere::Json conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_index(index_file_name, conf);
    test_binary_idmap(conf);
}

TEST_F(Benchmark_binary, TEST_BINARY_IVF_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT;

    knowhere::Json conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = nlist;
        std::string index_file_name = get_index_name({nlist});
        create_index(index_file_name, conf);
        test_binary_ivf(conf);
    }
}

TEST_F(Benchmark_binary, TEST_BINARY_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Json conf = cfg_;
    for (auto M : HNSW_Ms_) {
        conf[knowhere::indexparam::HNSW_M] = M;
        for (auto efc : EFCONs_) {
            conf[knowhere::indexparam::EFCONSTRUCTION] = efc;
            std::string index_file_name = get_index_name({M, efc});
            create_index(index_file_name, conf);
            test_binary_hnsw(conf);
        }
    }
}
