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
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"

class Benchmark_binary_range : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_binary_idmap(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto radius = conf.at(knowhere::meta::RADIUS).get<float>();

        printf("\n[%0.3f s] %s | %s, radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               radius);
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
            CALC_TIME_SPAN(auto result = index_.value().RangeSearch(ds_ptr, conf, nullptr));
            auto ids = result.value()->GetIds();
            auto distances = result.value()->GetDistance();
            auto lims = result.value()->GetLims();
            CheckDistance(metric_type_, ids, distances, lims, nq);
            float recall = CalcRecall(ids, lims, nq);
            float accuracy = CalcAccuracy(ids, lims, nq);
            printf("  nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f\n", nq, t_diff, recall, accuracy);
            std::fflush(stdout);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_binary_ivf(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto nlist = conf[knowhere::indexparam::NLIST].get<int64_t>();
        auto radius = conf.at(knowhere::meta::RADIUS).get<float>();

        printf("\n[%0.3f s] %s | %s | nlist=%ld, radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), nlist, radius);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            conf[knowhere::indexparam::NPROBE] = nprobe;
            for (auto nq : NQs_) {
                knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                CALC_TIME_SPAN(auto result = index_.value().RangeSearch(ds_ptr, conf, nullptr));
                auto ids = result.value()->GetIds();
                auto lims = result.value()->GetLims();
                float recall = CalcRecall(ids, lims, nq);
                float accuracy = CalcAccuracy(ids, lims, nq);
                printf("  nprobe = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f\n", nprobe, nq, t_diff, recall,
                       accuracy);
                std::fflush(stdout);
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
        auto radius = conf.at(knowhere::meta::RADIUS).get<float>();

        printf("\n[%0.3f s] %s | %s | M=%ld | efc=%ld, radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), M, efc, radius);
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            conf[knowhere::indexparam::EF] = ef;
            for (auto nq : NQs_) {
                knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                CALC_TIME_SPAN(auto result = index_.value().RangeSearch(ds_ptr, conf, nullptr));
                auto ids = result.value()->GetIds();
                auto lims = result.value()->GetLims();
                float recall = CalcRecall(ids, lims, nq);
                float accuracy = CalcAccuracy(ids, lims, nq);
                printf("  ef = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f\n", ef, nq, t_diff, recall,
                       accuracy);
                std::fflush(stdout);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
#if 0  // used when create range sift HDF5
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();
#else
        set_ann_test_name("sift-4096-hamming-range");
        parse_ann_test_name_with_range();
        load_hdf5_data_range<true>();
#endif

        cfg_[knowhere::meta::METRIC_TYPE] = metric_type_;
        cfg_[knowhere::meta::RADIUS] = *gt_radius_;
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX2);
        printf("faiss::distance_compute_blas_threshold: %ld\n", knowhere::KnowhereConfig::GetBlasThreshold());
    }

    void
    TearDown() override {
        free_all();
    }

 protected:
    const std::vector<int32_t> NQs_ = {10000};

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
//   2. use parse_ann_test_name() and load_hdf5_data<false>()
//   3. set expected distance calculation API for RunRangeSearchBF
//   4. specify the hdf5 file name to generate
//   5. run this testcase
#if 0
TEST_F(Benchmark_binary_range, TEST_CREATE_HDF5) {
    // use sift1m data as binary data
    dim_ *= 32;

    // set this radius to get about 1M result dataset for 10k nq
    const float radius = 291.0;

    std::vector<int64_t> golden_labels;
    std::vector<float> golden_distances;
    std::vector<size_t> golden_lims;
    RunBinaryRangeSearchBF<CMin<float>>(golden_labels, golden_distances, golden_lims, metric_type_,
                                        (const uint8_t*)xb_, nb_, (const uint8_t*)xq_, nq_, dim_, radius, nullptr);

    // convert golden_lims and golden_ids to int32
    std::vector<int32_t> golden_lims_int(nq_ + 1);
    for (int32_t i = 0; i <= nq_; i++) {
        golden_lims_int[i] = golden_lims[i];
    }

    auto elem_cnt = golden_lims[nq_];
    std::vector<int32_t> golden_ids_int(elem_cnt);
    for (int32_t i = 0; i < elem_cnt; i++) {
        golden_ids_int[i] = golden_labels[i];
    }

    assert(dim_ == 4096);
    assert(nq_ == 10000);
    hdf5_write_range<true>("sift-4096-hamming-range.hdf5", dim_/32, xb_, nb_, xq_, nq_, radius,
                           golden_lims_int.data(), golden_ids_int.data(), golden_distances.data());
}
#endif

TEST_F(Benchmark_binary_range, TEST_BINARY_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP;

    knowhere::Json conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_index(index_file_name, conf);
    test_binary_idmap(conf);
}

TEST_F(Benchmark_binary_range, TEST_BINARY_IVF_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT;

    knowhere::Json conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = nlist;
        std::string index_file_name = get_index_name({nlist});
        create_index(index_file_name, conf);
        test_binary_ivf(conf);
    }
}

TEST_F(Benchmark_binary_range, TEST_BINARY_HNSW) {
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
