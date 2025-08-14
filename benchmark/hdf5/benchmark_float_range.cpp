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
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/dataset.h"

class Benchmark_float_range : public Benchmark_knowhere, public ::testing::Test {
 public:
    template <typename T>
    void
    test_idmap(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto radius = conf.at(knowhere::meta::RADIUS).get<float>();

        std::string data_type_str = get_data_type_name<T>();
        printf("\n[%0.3f s] %s | %s(%s) | radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               data_type_str.c_str(), radius);
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
            auto query = knowhere::ConvertToDataTypeIfNeeded<T>(ds_ptr);
            CALC_TIME_SPAN(auto result = index_.value().RangeSearch(query, conf, nullptr));
            auto ids = result.value()->GetIds();
            auto distances = result.value()->GetDistance();
            auto lims = result.value()->GetLims();
            CheckDistance(metric_type_, ids, distances, lims, nq);
            float recall = CalcRecall(ids, lims, nq);
            float accuracy = CalcAccuracy(ids, lims, nq);
            printf("  nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f, L@ = %.2f\n", nq, TDIFF_, recall, accuracy,
                   lims[nq] / (float)nq);
            std::fflush(stdout);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    template <typename T>
    void
    test_ivf(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto nlist = conf[knowhere::indexparam::NLIST].get<int64_t>();
        auto radius = conf.at(knowhere::meta::RADIUS).get<float>();

        std::string data_type_str = get_data_type_name<T>();
        printf("\n[%0.3f s] %s | %s(%s) | nlist=%ld, radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), data_type_str.c_str(), nlist, radius);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            conf[knowhere::indexparam::NPROBE] = nprobe;
            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                auto query = knowhere::ConvertToDataTypeIfNeeded<T>(ds_ptr);
                CALC_TIME_SPAN(auto result = index_.value().RangeSearch(query, conf, nullptr));
                auto ids = result.value()->GetIds();
                auto lims = result.value()->GetLims();
                float recall = CalcRecall(ids, lims, nq);
                float accuracy = CalcAccuracy(ids, lims, nq);
                printf("  nprobe = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f, L@ = %.2f\n", nprobe, nq,
                       TDIFF_, recall, accuracy, lims[nq] / (float)nq);
                std::fflush(stdout);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    template <typename T>
    void
    test_hnsw(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto M = conf[knowhere::indexparam::HNSW_M].get<int64_t>();
        auto efc = conf[knowhere::indexparam::EFCONSTRUCTION].get<int64_t>();
        auto radius = conf.at(knowhere::meta::RADIUS).get<float>();

        std::string data_type_str = get_data_type_name<T>();
        printf("\n[%0.3f s] %s | %s(%s) | M=%ld | efc=%ld, radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), data_type_str.c_str(), M, efc, radius);
        printf("================================================================================\n");
        for (auto ef : EFs_) {
            conf[knowhere::indexparam::EF] = ef;
            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                auto query = knowhere::ConvertToDataTypeIfNeeded<T>(ds_ptr);
                CALC_TIME_SPAN(auto result = index_.value().RangeSearch(query, conf, nullptr));
                auto ids = result.value()->GetIds();
                auto lims = result.value()->GetLims();
                float recall = CalcRecall(ids, lims, nq);
                float accuracy = CalcAccuracy(ids, lims, nq);
                printf("  ef = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f, L@ = %.2f\n", ef, nq, TDIFF_,
                       recall, accuracy, lims[nq] / (float)nq);
                std::fflush(stdout);
            }
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

#ifdef KNOWHERE_WITH_DISKANN
    template <typename T>
    void
    test_diskann(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto radius = conf.at(knowhere::meta::RADIUS).get<float>();

        std::string data_type_str = get_data_type_name<T>();
        printf("\n[%0.3f s] %s | %s(%s) | radius=%.3f\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               data_type_str.c_str(), radius);
        printf("================================================================================\n");
        for (auto search_list_size : SEARCH_LISTs_) {
            conf["search_list_size"] = search_list_size;
            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                auto query = knowhere::ConvertToDataTypeIfNeeded<T>(ds_ptr);
                CALC_TIME_SPAN(auto result = index_.value().RangeSearch(query, conf, nullptr));
                auto ids = result.value()->GetIds();
                auto lims = result.value()->GetLims();
                float recall = CalcRecall(ids, lims, nq);
                float accuracy = CalcAccuracy(ids, lims, nq);
                printf("  search_list_size = %4d, nq = %4d, elapse = %6.3fs, R@ = %.4f, A@ = %.4f, L@ = %.2f\n",
                       search_list_size, nq, TDIFF_, recall, accuracy, lims[nq] / (float)nq);
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
#if 0  // used when create range sift HDF5
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();
#else
        set_ann_test_name("sift-128-euclidean-range");
        parse_ann_test_name_with_range();
        load_hdf5_data_range<knowhere::fp32>();
#endif

        cfg_[knowhere::meta::METRIC_TYPE] = metric_type_;
        cfg_[knowhere::meta::RADIUS] = *gt_radius_;
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AVX2);
        knowhere::KnowhereConfig::SetBuildThreadPoolSize(default_build_thread_num);
        knowhere::KnowhereConfig::SetSearchThreadPoolSize(default_search_thread_num);
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

    // IVFPQ index params
    const std::vector<int32_t> Ms_ = {8, 16, 32};
    const int32_t NBITS_ = 8;

    // HNSW index params
    const std::vector<int32_t> HNSW_Ms_ = {16};
    const std::vector<int32_t> EFCONs_ = {200};
    const std::vector<int32_t> EFs_ = {16, 32, 64, 128, 256, 512};

    // DISKANN index params
    const std::vector<int32_t> SEARCH_LISTs_ = {100, 200, 400};
};

#if 0
// This testcase can be used to generate HDF5 file
// Following these steps:
//   1. set_ann_test_name, eg. "sift-128-euclidean" or "glove-200-angular"
//   2. use parse_ann_test_name() and load_hdf5_data<false>()
//   3. comment SetMetaRadius()
//   4. set radius to a right value
//   5. use RunFloatRangeSearchBF<CMin<float>> for L2, or RunFloatRangeSearchBF<CMax<float>> for IP
//   6. specify the hdf5 file name to generate
//   7. run this testcase
TEST_F(Benchmark_float_range, TEST_CREATE_HDF5) {
    // set this radius to get about 1M result dataset for 10k nq
    const float radius = 186.0 * 186.0;
    const float range_filter = 0.0;

    std::vector<int64_t> golden_labels;
    std::vector<float> golden_distances;
    std::vector<size_t> golden_lims;
    RunFloatRangeSearchBF(golden_labels, golden_distances, golden_lims, metric_type_,
                          (const float*)xb_, nb_, (const float*)xq_, nq_, dim_, radius, range_filter, nullptr);

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

    assert(dim_ == 128);
    assert(nq_ == 10000);
    hdf5_write_range<false>("sift-128-euclidean-range.hdf5", dim_, xb_, nb_, xq_, nq_, high_bound,
                            golden_lims_int.data(), golden_ids_int.data(), golden_distances.data());
}

// This testcase is to convert sift-128-euclidean for VECTOR_INT8
// In the original SIFT dataset, the numerical range for training and testing data is [0, 218].
// After subtracting 110 from each value, the range of values becomes [-110, 108], then each
// FLOAT data can be converted to INT8 without loss for VECTOR-INT8 testing.
TEST_F(Benchmark_float_range, TEST_CREATE_HDF5_FOR_VECTOR_INT8) {
    std::vector<float> xb_new(nb_ * dim_);
    for (int32_t i = 0; i <= nb_ * dim_; i++) {
        xb_new[i] = *((float*)xb_ + i) - 110;
    }

    std::vector<float> xq_new(nq_ * dim_);
    for (int32_t i = 0; i <= nq_ * dim_; i++) {
        xq_new[i] = *((float*)xq_ + i) - 110;
    }

    assert(dim_ == 128);
    assert(nq_ == 10000);
    hdf5_write_range<knowhere::fp32>("sift-128-euclidean-range-new.hdf5", dim_, xb_new.data(), nb_, xq_new.data(), nq_,
                                     *gt_radius_, gt_lims_, gt_ids_, gt_dist_);
}
#endif

#define TEST_INDEX(NAME, T, X)              \
    index_file_name = get_index_name<T>(X); \
    create_index<T>(index_file_name, conf); \
    test_##NAME<T>(conf)

TEST_F(Benchmark_float_range, TEST_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    std::vector<int32_t> params = {};

    TEST_INDEX(idmap, knowhere::fp32, params);
    TEST_INDEX(idmap, knowhere::fp16, params);
    TEST_INDEX(idmap, knowhere::bf16, params);
}

TEST_F(Benchmark_float_range, TEST_IVF_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = nlist;
        std::vector<int32_t> params = {nlist};

        TEST_INDEX(ivf, knowhere::fp32, params);
        TEST_INDEX(ivf, knowhere::fp16, params);
        TEST_INDEX(ivf, knowhere::bf16, params);
    }
}

TEST_F(Benchmark_float_range, TEST_IVF_SQ8) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = nlist;
        std::vector<int32_t> params = {nlist};

        TEST_INDEX(ivf, knowhere::fp32, params);
        TEST_INDEX(ivf, knowhere::fp16, params);
        TEST_INDEX(ivf, knowhere::bf16, params);
    }
}

TEST_F(Benchmark_float_range, TEST_IVF_PQ) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFPQ;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::NBITS] = NBITS_;
    for (auto m : Ms_) {
        conf[knowhere::indexparam::M] = m;
        for (auto nlist : NLISTs_) {
            conf[knowhere::indexparam::NLIST] = nlist;
            std::vector<int32_t> params = {nlist, m};

            TEST_INDEX(ivf, knowhere::fp32, params);
            TEST_INDEX(ivf, knowhere::fp16, params);
            TEST_INDEX(ivf, knowhere::bf16, params);
        }
    }
}

TEST_F(Benchmark_float_range, TEST_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    for (auto M : HNSW_Ms_) {
        conf[knowhere::indexparam::HNSW_M] = M;
        for (auto efc : EFCONs_) {
            conf[knowhere::indexparam::EFCONSTRUCTION] = efc;
            std::vector<int32_t> params = {M, efc};

            TEST_INDEX(hnsw, knowhere::fp32, params);
            TEST_INDEX(hnsw, knowhere::fp16, params);
            TEST_INDEX(hnsw, knowhere::bf16, params);
        }
    }
}

#ifdef KNOWHERE_WITH_DISKANN
TEST_F(Benchmark_float_range, TEST_DISKANN) {
    index_type_ = knowhere::IndexEnum::INDEX_DISKANN;

    knowhere::Json conf = cfg_;

    conf[knowhere::meta::INDEX_PREFIX] = (metric_type_ == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
    conf[knowhere::meta::DATA_PATH] = kRawDataPath;
    conf[knowhere::indexparam::MAX_DEGREE] = 56;
    conf[knowhere::indexparam::PQ_CODE_BUDGET_GB] = sizeof(float) * dim_ * nb_ * 0.125 / (1024 * 1024 * 1024);
    conf[knowhere::indexparam::BUILD_DRAM_BUDGET_GB] = 32.0;
    conf[knowhere::indexparam::SEARCH_CACHE_BUDGET_GB] = 0;
    conf[knowhere::indexparam::BEAMWIDTH] = 8;

    fs::create_directory(kDir);
    fs::create_directory(kL2IndexDir);
    fs::create_directory(kIPIndexDir);

    WriteRawDataToDisk(kRawDataPath, (const float*)xb_, (const uint32_t)nb_, (const uint32_t)dim_);

    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);

    index_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        index_type_, knowhere::Version::GetCurrentVersion().VersionNumber(), diskann_index_pack);
    printf("[%.3f s] Building all on %d vectors\n", get_time_diff(), nb_);
    knowhere::DataSetPtr ds_ptr = nullptr;
    index_.value().Build(ds_ptr, conf);

    knowhere::BinarySet binset;
    index_.value().Serialize(binset);
    index_.value().Deserialize(binset, conf);

    test_diskann<knowhere::fp32>(conf);
}
#endif
