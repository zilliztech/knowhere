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
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/dataset.h"

const int32_t GPU_DEVICE_ID = 0;

constexpr uint32_t kNumRows = 10000;
constexpr uint32_t kNumQueries = 100;
constexpr uint32_t kDim = 128;
constexpr uint32_t kK = 10;
constexpr float kL2KnnRecall = 0.8;

class Benchmark_float_range_bitset : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_ivf(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto radius = conf[knowhere::meta::RADIUS].get<float>();

        printf("\n[%0.3f s] %s | %s | radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               radius);
        printf("================================================================================\n");
        for (auto per : PERCENTs_) {
            auto bitset_data = GenRandomBitset(nb_, nb_ * per / 100);
            knowhere::BitsetView bitset(bitset_data.data(), nb_);

            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                auto g_result = golden_index_.value().RangeSearch(ds_ptr, conf, bitset);
                auto g_ids = g_result.value()->GetIds();
                auto g_lims = g_result.value()->GetLims();
                CALC_TIME_SPAN(auto result = index_.value().RangeSearch(ds_ptr, conf, bitset));
                auto ids = result.value()->GetIds();
                auto lims = result.value()->GetLims();
                float recall = CalcRecall(g_ids, g_lims, ids, lims, nq);
                float accuracy = CalcAccuracy(g_ids, g_lims, ids, lims, nq);
                printf("  bitset_per = %3d%%, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f, L@ = %.2f\n", per, nq,
                       t_diff, recall, accuracy, lims[nq] / (float)nq);
                std::fflush(stdout);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    void
    test_hnsw(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto radius = conf[knowhere::meta::RADIUS].get<float>();

        printf("\n[%0.3f s] %s | %s | radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               radius);
        printf("================================================================================\n");
        for (auto per : PERCENTs_) {
            auto bitset_data = GenRandomBitset(nb_, nb_ * per / 100);
            knowhere::BitsetView bitset(bitset_data.data(), nb_);

            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                auto g_result = golden_index_.value().RangeSearch(ds_ptr, conf, bitset);
                auto g_ids = g_result.value()->GetIds();
                auto g_lims = g_result.value()->GetLims();
                CALC_TIME_SPAN(auto result = index_.value().RangeSearch(ds_ptr, conf, bitset));
                auto ids = result.value()->GetIds();
                auto lims = result.value()->GetLims();
                float recall = CalcRecall(g_ids, g_lims, ids, lims, nq);
                float accuracy = CalcAccuracy(g_ids, g_lims, ids, lims, nq);
                printf("  bitset_per = %3d%%, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f, L@ = %.2f\n", per, nq,
                       t_diff, recall, accuracy, lims[nq] / (float)nq);
                std::fflush(stdout);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

#ifdef KNOWHERE_WITH_DISKANN
    void
    test_diskann(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto radius = conf[knowhere::meta::RADIUS].get<float>();

        printf("\n[%0.3f s] %s | %s | radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               radius);
        printf("================================================================================\n");
        for (auto per : PERCENTs_) {
            auto bitset_data = GenRandomBitset(nb_, nb_ * per / 100);
            knowhere::BitsetView bitset(bitset_data.data(), nb_);
            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                auto g_result = golden_index_.value().RangeSearch(ds_ptr, conf, bitset);
                auto g_ids = g_result.value()->GetIds();
                auto g_lims = g_result.value()->GetLims();
                CALC_TIME_SPAN(auto result = index_.value().RangeSearch(ds_ptr, conf, bitset));
                auto ids = result.value()->GetIds();
                auto lims = result.value()->GetLims();
                float recall = CalcRecall(g_ids, g_lims, ids, lims, nq);
                float accuracy = CalcAccuracy(g_ids, g_lims, ids, lims, nq);
                printf("  bitset_per = %3d%%, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f, L@ = %.2f\n", per, nq,
                       t_diff, recall, accuracy, lims[nq] / (float)nq);
                std::fflush(stdout);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }
#endif

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        set_ann_test_name("sift-128-euclidean-range");
        parse_ann_test_name_with_range();
        load_hdf5_data_range<false>();

        cfg_[knowhere::meta::METRIC_TYPE] = metric_type_;
        cfg_[knowhere::meta::RADIUS] = *gt_radius_;
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX2);
        printf("faiss::distance_compute_blas_threshold: %ld\n", knowhere::KnowhereConfig::GetBlasThreshold());

        create_golden_index(cfg_);
    }

    void
    TearDown() override {
        free_all();
    }

 protected:
    const std::vector<int32_t> NQs_ = {10000};
    const std::vector<int32_t> TOPKs_ = {100};
    const std::vector<int32_t> PERCENTs_ = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    // IVF index params
    // const std::vector<int32_t> NLISTs_ = {1024};
    // const std::vector<int32_t> NPROBEs_ = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

    // IVFPQ index params
    // const std::vector<int32_t> Ms_ = {8, 16, 32};
    // const int32_t NBITS_ = 8;

    // HNSW index params
    // const std::vector<int32_t> HNSW_Ms_ = {16};
    // const std::vector<int32_t> EFCONs_ = {200};
    // const std::vector<int32_t> EFs_ = {128, 256, 512};
};

TEST_F(Benchmark_float_range_bitset, TEST_IVF_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    knowhere::Json conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_index(index_file_name, conf);
    test_ivf(conf);
}

TEST_F(Benchmark_float_range_bitset, TEST_IVF_SQ8) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    knowhere::Json conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_index(index_file_name, conf);
    test_ivf(conf);
}

TEST_F(Benchmark_float_range_bitset, TEST_IVF_PQ) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFPQ;

    knowhere::Json conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_index(index_file_name, conf);
    test_ivf(conf);
}

TEST_F(Benchmark_float_range_bitset, TEST_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Json conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_index(index_file_name, conf);
    test_hnsw(conf);
}

#ifdef KNOWHERE_WITH_DISKANN
TEST_F(Benchmark_float_range_bitset, TEST_DISKANN) {
    index_type_ = knowhere::IndexEnum::INDEX_DISKANN;

    knowhere::Json conf = cfg_;

    conf["index_prefix"] = (metric_type_ == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
    conf["data_path"] = kRawDataPath;
    conf["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
    conf["build_dram_budget_gb"] = 32.0;

    fs::create_directory(kDir);
    fs::create_directory(kL2IndexDir);
    fs::create_directory(kIPIndexDir);

    WriteRawDataToDisk(kRawDataPath, (const float*)xb_, (const uint32_t)nb_, (const uint32_t)dim_);

    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    index_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type_, version, diskann_index_pack);
    printf("[%.3f s] Building all on %d vectors\n", get_time_diff(), nb_);
    knowhere::DataSetPtr ds_ptr = nullptr;
    index_.value().Build(ds_ptr, conf);

    knowhere::BinarySet binset;
    index_.value().Serialize(binset);
    index_.value().Deserialize(binset, conf);

    test_diskann(conf);
}
#endif
