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

class Benchmark_float : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_idmap(const knowhere::Json& cfg) {
        auto conf = cfg;

        printf("\n[%0.3f s] %s | %s \n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        printf("================================================================================\n");
        for (auto nq : NQs_) {
            auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
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
    test_ivf(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto nlist = conf[knowhere::indexparam::NLIST].get<int64_t>();

        printf("\n[%0.3f s] %s | %s | nlist=%ld\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(),
               nlist);
        printf("================================================================================\n");
        for (auto nprobe : NPROBEs_) {
            conf[knowhere::indexparam::NPROBE] = nprobe;
            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
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
    test_hnsw(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto M = conf[knowhere::indexparam::HNSW_M].get<int64_t>();
        auto efConstruction = conf[knowhere::indexparam::EFCONSTRUCTION].get<int64_t>();

        printf("\n[%0.3f s] %s | %s | M=%ld | efConstruction=%ld\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), M, efConstruction);
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

#ifdef KNOWHERE_WITH_DISKANN
    void
    test_diskann(const knowhere::Json& cfg) {
        auto conf = cfg;

        printf("\n[%0.3f s] %s | %s \n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        printf("================================================================================\n");
        for (auto search_list_size : SEARCH_LISTs_) {
            conf["search_list_size"] = search_list_size;
            for (auto nq : NQs_) {
                auto ds_ptr = knowhere::GenDataSet(nq, dim_, xq_);
                for (auto k : TOPKs_) {
                    conf[knowhere::meta::TOPK] = k;
                    CALC_TIME_SPAN(auto result = index_.value().Search(ds_ptr, conf, nullptr));
                    auto ids = result.value()->GetIds();
                    float recall = CalcRecall(ids, nq, k);
                    printf("  search_list_size = %4d, nq = %4d, k = %4d, elapse = %6.3fs, R@ = %.4f\n",
                           search_list_size, nq, k, t_diff, recall);
                    std::fflush(stdout);
                }
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
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();

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

    // IVFPQ index params
    const std::vector<int32_t> Ms_ = {8, 16, 32};
    const int32_t NBITS_ = 8;

    // HNSW index params
    const std::vector<int32_t> HNSW_Ms_ = {16};
    const std::vector<int32_t> EFCONs_ = {200};
    const std::vector<int32_t> EFs_ = {128, 256, 512};

    // DISKANN index params
    const std::vector<int32_t> SEARCH_LISTs_ = {100, 200, 400};
};

TEST_F(Benchmark_float, TEST_IDMAP) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    knowhere::Json conf = cfg_;
    std::string index_file_name = get_index_name({});
    create_index(index_file_name, conf);
    test_idmap(conf);
}

TEST_F(Benchmark_float, TEST_IVF_FLAT) {
#ifdef KNOWHERE_WITH_RAFT
    index_type_ = knowhere::IndexEnum::INDEX_RAFT_IVFFLAT;
#else
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
#endif

    knowhere::Json conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = nlist;
        std::string index_file_name = get_index_name({nlist});
        create_index(index_file_name, conf);
        test_ivf(conf);
    }
}

TEST_F(Benchmark_float, TEST_IVF_SQ8) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;

    knowhere::Json conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = nlist;
        std::string index_file_name = get_index_name({nlist});
        create_index(index_file_name, conf);
        test_ivf(conf);
    }
}

TEST_F(Benchmark_float, TEST_IVF_PQ) {
#ifdef KNOWHERE_WITH_RAFT
    index_type_ = knowhere::IndexEnum::INDEX_RAFT_IVFPQ;
#else
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFPQ;
#endif

    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::NBITS] = NBITS_;
    for (auto m : Ms_) {
        conf[knowhere::indexparam::M] = m;
        for (auto nlist : NLISTs_) {
            conf[knowhere::indexparam::NLIST] = nlist;
            std::string index_file_name = get_index_name({nlist, m});
            create_index(index_file_name, conf);
            test_ivf(conf);
        }
    }
}

TEST_F(Benchmark_float, TEST_HNSW) {
    index_type_ = knowhere::IndexEnum::INDEX_HNSW;

    knowhere::Json conf = cfg_;
    for (auto M : HNSW_Ms_) {
        conf[knowhere::indexparam::HNSW_M] = M;
        for (auto efc : EFCONs_) {
            conf[knowhere::indexparam::EFCONSTRUCTION] = efc;
            std::string index_file_name = get_index_name({M, efc});
            create_index(index_file_name, conf);
            test_hnsw(conf);
        }
    }
}

#ifdef KNOWHERE_WITH_DISKANN
TEST_F(Benchmark_float, TEST_DISKANN) {
    index_type_ = knowhere::IndexEnum::INDEX_DISKANN;

    knowhere::Json conf = cfg_;

    conf["index_prefix"] = (metric_type_ == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
    conf["data_path"] = kRawDataPath;
    conf["max_degree"] = 56;
    conf["pq_code_budget_gb"] = sizeof(float) * dim_ * nb_ * 0.125 / (1024 * 1024 * 1024);
    conf["build_dram_budget_gb"] = 32.0;
    conf["search_cache_budget_gb"] = 0;
    conf["beamwidth"] = 8;

    fs::create_directory(kDir);
    fs::create_directory(kL2IndexDir);
    fs::create_directory(kIPIndexDir);

    WriteRawDataToDisk(kRawDataPath, (const float*)xb_, (const uint32_t)nb_, (const uint32_t)dim_);

    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);

    index_ = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        index_type_, knowhere::Version::GetCurrentVersion().VersionNumber(), diskann_index_pack);
    printf("[%.3f s] Building all on %d vectors\n", get_time_diff(), nb_);
    knowhere::DataSetPtr ds_ptr = nullptr;
    index_.value().Build(ds_ptr, conf);

    knowhere::BinarySet binset;
    index_.value().Serialize(binset);
    index_.value().Deserialize(binset, conf);

    test_diskann(conf);
}
#endif
