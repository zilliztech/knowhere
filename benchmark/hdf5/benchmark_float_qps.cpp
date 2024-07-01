// Copyright (C) 2019-2020 Zilliz. All rights reserved.
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

#include <thread>
#include <vector>

#include "benchmark_knowhere.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"

const int32_t GPU_DEVICE_ID = 0;

class Benchmark_float_qps : public Benchmark_knowhere, public ::testing::Test {
 public:
    void
    test_ivf(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto nlist = conf[knowhere::indexparam::NLIST].get<int32_t>();

        auto find_smallest_nprobe = [&](float expected_recall) -> int32_t {
            conf[knowhere::meta::TOPK] = topk_;
            auto ds_ptr = knowhere::GenDataSet(nq_, dim_, xq_);

            int32_t left = 1, right = 256, nprobe;
            float recall;
            while (left <= right) {
                nprobe = left + (right - left) / 2;
                conf[knowhere::indexparam::NPROBE] = nprobe;

                auto result = index_.value().Search(ds_ptr, conf, nullptr);
                recall = CalcRecall(result.value()->GetIds(), nq_, topk_);
                printf("[%0.3f s] iterate IVF param for recall %.4f: nlist=%d, nprobe=%4d, k=%d, R@=%.4f\n",
                       get_time_diff(), expected_recall, nlist, nprobe, topk_, recall);
                std::fflush(stdout);
                if (std::abs(recall - expected_recall) <= 0.0001) {
                    return nprobe;
                }
                if (recall < expected_recall) {
                    left = nprobe + 1;
                } else {
                    right = nprobe - 1;
                }
            }
            return left;
        };

        for (auto expected_recall : EXPECTED_RECALLs_) {
            auto nprobe = find_smallest_nprobe(expected_recall);
            conf[knowhere::indexparam::NPROBE] = nprobe;
            conf[knowhere::meta::TOPK] = topk_;

            printf("\n[%0.3f s] %s | %s | nlist=%d, nprobe=%d, k=%d, R@=%.4f\n", get_time_diff(),
                   ann_test_name_.c_str(), index_type_.c_str(), nlist, nprobe, topk_, expected_recall);
            printf("================================================================================\n");
            for (auto thread_num : THREAD_NUMs_) {
                CALC_TIME_SPAN(task(conf, thread_num, nq_));
                printf("  thread_num = %2d, elapse = %6.3fs, VPS = %.3f\n", thread_num, t_diff, nq_ / t_diff);
                std::fflush(stdout);
            }
            printf("================================================================================\n");
            printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        }
    }

    void
    test_cagra(const knowhere::Json& cfg) {
        auto conf = cfg;

        auto find_smallest_max_iters = [&](float expected_recall) -> int32_t {
            auto ds_ptr = knowhere::GenDataSet(nq_, dim_, xq_);
            auto left = 32;
            auto right = 256;
            auto max_iterations = left;

            float recall;
            while (left <= right) {
                max_iterations = left + (right - left) / 2;
                conf[knowhere::indexparam::MAX_ITERATIONS] = max_iterations;

                auto result = index_.value().Search(ds_ptr, conf, nullptr);
                recall = CalcRecall(result.value()->GetIds(), nq_, topk_);
                printf(
                    "[%0.3f s] iterate CAGRA param for recall %.4f: max_iterations=%d, k=%d, "
                    "R@=%.4f\n",
                    get_time_diff(), expected_recall, max_iterations, topk_, recall);
                std::fflush(stdout);
                if (std::abs(recall - expected_recall) <= 0.0001) {
                    return max_iterations;
                }
                if (recall < expected_recall) {
                    left = max_iterations + 1;
                } else {
                    right = max_iterations - 1;
                }
            }
            return left;
        };

        for (auto expected_recall : EXPECTED_RECALLs_) {
            conf[knowhere::indexparam::ITOPK_SIZE] = ((int{topk_} + 32 - 1) / 32) * 32;
            conf[knowhere::meta::TOPK] = topk_;
            conf[knowhere::indexparam::MAX_ITERATIONS] = find_smallest_max_iters(expected_recall);

            printf(
                "\n[%0.3f s] %s | %s | k=%d, "
                "R@=%.4f\n",
                get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(), topk_, expected_recall);
            printf("================================================================================\n");
            for (auto thread_num : THREAD_NUMs_) {
                CALC_TIME_SPAN(task(conf, thread_num, nq_));
                printf("  thread_num = %2d, elapse = %6.3fs, VPS = %.3f\n", thread_num, t_diff, nq_ / t_diff);
                std::fflush(stdout);
            }
            printf("================================================================================\n");
            printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        }
    }

    void
    test_hnsw(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto M = conf[knowhere::indexparam::HNSW_M].get<int32_t>();
        auto efConstruction = conf[knowhere::indexparam::EFCONSTRUCTION].get<int32_t>();

        auto find_smallest_ef = [&](float expected_recall) -> int32_t {
            conf[knowhere::meta::TOPK] = topk_;
            auto ds_ptr = knowhere::GenDataSet(nq_, dim_, xq_);

            int32_t left = topk_, right = 1024, ef;
            float recall;
            while (left <= right) {
                ef = left + (right - left) / 2;
                conf[knowhere::indexparam::EF] = ef;

                auto result = index_.value().Search(ds_ptr, conf, nullptr);
                recall = CalcRecall(result.value()->GetIds(), nq_, topk_);
                printf("[%0.3f s] iterate HNSW param for expected recall %.4f: M=%d, efc=%d, ef=%4d, k=%d, R@=%.4f\n",
                       get_time_diff(), expected_recall, M, efConstruction, ef, topk_, recall);
                std::fflush(stdout);
                if (std::abs(recall - expected_recall) <= 0.0001) {
                    return ef;
                }
                if (recall < expected_recall) {
                    left = ef + 1;
                } else {
                    right = ef - 1;
                }
            }
            return left;
        };

        for (auto expected_recall : EXPECTED_RECALLs_) {
            auto ef = find_smallest_ef(expected_recall);
            conf[knowhere::indexparam::EF] = ef;
            conf[knowhere::meta::TOPK] = topk_;

            printf("\n[%0.3f s] %s | %s | M=%d | efConstruction=%d, ef=%d, k=%d, R@=%.4f\n", get_time_diff(),
                   ann_test_name_.c_str(), index_type_.c_str(), M, efConstruction, ef, topk_, expected_recall);
            printf("================================================================================\n");
            for (auto thread_num : THREAD_NUMs_) {
                CALC_TIME_SPAN(task(conf, thread_num, nq_));
                printf("  thread_num = %2d, elapse = %6.3fs, VPS = %.3f\n", thread_num, t_diff, nq_ / t_diff);
                std::fflush(stdout);
            }
            printf("================================================================================\n");
            printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        }
    }

    void
    test_scann(const knowhere::Json& cfg) {
        auto conf = cfg;

        const auto reorder_k = conf[knowhere::indexparam::REORDER_K].get<int32_t>();
        const auto with_raw_data = conf[knowhere::indexparam::WITH_RAW_DATA].get<bool>();
        auto nlist = conf[knowhere::indexparam::NLIST].get<int32_t>();

        auto find_smallest_nprobe = [&](float expected_recall) -> int32_t {
            conf[knowhere::meta::TOPK] = topk_;
            auto ds_ptr = knowhere::GenDataSet(nq_, dim_, xq_);

            int32_t left = 1, right = 256, nprobe;
            float recall;
            while (left <= right) {
                nprobe = left + (right - left) / 2;
                conf[knowhere::indexparam::NPROBE] = nprobe;

                auto result = index_.value().Search(ds_ptr, conf, nullptr);
                recall = CalcRecall(result.value()->GetIds(), nq_, topk_);
                printf(
                    "[%0.3f s] iterate scann param for recall %.4f: nlist=%d, nprobe=%4d, reorder_k=%d, "
                    "with_raw_data=%d, k=%d, R@=%.4f\n",
                    get_time_diff(), expected_recall, nlist, nprobe, reorder_k, with_raw_data ? 1 : 0, topk_, recall);
                std::fflush(stdout);
                if (std::abs(recall - expected_recall) <= 0.0001) {
                    return nprobe;
                }
                if (recall < expected_recall) {
                    left = nprobe + 1;
                } else {
                    right = nprobe - 1;
                }
            }
            return left;
        };

        for (auto expected_recall : EXPECTED_RECALLs_) {
            auto nprobe = find_smallest_nprobe(expected_recall);
            conf[knowhere::indexparam::NPROBE] = nprobe;
            conf[knowhere::meta::TOPK] = topk_;

            printf("\n[%0.3f s] %s | %s | nlist=%d, nprobe=%d, reorder_k=%d, with_raw_data=%d, k=%d, R@=%.4f\n",
                   get_time_diff(), ann_test_name_.c_str(), index_type_.c_str(), nlist, nprobe, reorder_k,
                   with_raw_data ? 1 : 0, topk_, expected_recall);
            printf("================================================================================\n");
            for (auto thread_num : THREAD_NUMs_) {
                CALC_TIME_SPAN(task(conf, thread_num, nq_));
                printf("  thread_num = %2d, elapse = %6.3fs, VPS = %.3f\n", thread_num, t_diff, nq_ / t_diff);
                std::fflush(stdout);
            }
            printf("================================================================================\n");
            printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        }
    }

 private:
    void
    task(const knowhere::Json& conf, int32_t worker_num, int32_t nq_total) {
        auto worker = [&](int32_t idx_start, int32_t num) {
            num = std::min(num, nq_total - idx_start);
            for (int32_t i = 0; i < num; i++) {
                knowhere::DataSetPtr ds_ptr = knowhere::GenDataSet(1, dim_, (const float*)xq_ + (idx_start + i) * dim_);
                index_.value().Search(ds_ptr, conf, nullptr);
            }
        };

        std::vector<std::thread> thread_vector(worker_num);
        for (int32_t i = 0; i < worker_num; i++) {
            int32_t idx_start, req_num;
            req_num = nq_total / worker_num;
            if (nq_total % worker_num != 0) {
                req_num++;
            }
            idx_start = req_num * i;
            thread_vector[i] = std::thread(worker, idx_start, req_num);
        }
        for (int32_t i = 0; i < worker_num; i++) {
            thread_vector[i].join();
        }
    }

 protected:
    void
    SetUp() override {
        T0_ = elapsed();
        set_ann_test_name("sift-128-euclidean");
        parse_ann_test_name();
        load_hdf5_data<false>();

        cfg_[knowhere::meta::METRIC_TYPE] = metric_type_;
        knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);
#ifdef KNOWHERE_WITH_GPU
        knowhere::KnowhereConfig::InitGPUResource(GPU_DEVICE_ID, 2);
        cfg_[knowhere::meta::DEVICE_ID] = GPU_DEVICE_ID;
#endif
#ifdef KNOWHERE_WITH_RAFT
        knowhere::KnowhereConfig::SetRaftMemPool();
#endif
    }

    void
    TearDown() override {
        free_all();
#ifdef KNOWHERE_WITH_GPU
        knowhere::KnowhereConfig::FreeGPUResource();
#endif
    }

 protected:
    const int32_t topk_ = 100;
    const std::vector<float> EXPECTED_RECALLs_ = {0.8, 0.95};
    const std::vector<int32_t> THREAD_NUMs_ = {1, 2, 4, 8};

    // IVF index params
    const std::vector<int32_t> NLISTs_ = {1024};

    // IVFPQ index params
    const std::vector<int32_t> Ms_ = {8, 16, 32};
    const int32_t NBITS_ = 8;

    // HNSW index params
    const std::vector<int32_t> HNSW_Ms_ = {16};
    const std::vector<int32_t> EFCONs_ = {100};

    // SCANN index params
    const std::vector<int32_t> SCANN_REORDER_K = {256, 512, 768, 1024};
    const std::vector<bool> SCANN_WITH_RAW_DATA = {true};

    // CAGRA index params
    const std::vector<int32_t> GRAPH_DEGREE_ = {32, 64};
};

TEST_F(Benchmark_float_qps, TEST_IVF_FLAT) {
#ifdef KNOWHERE_WITH_GPU
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_GPU_IVFFLAT;
#elif KNOWHERE_WITH_RAFT
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

TEST_F(Benchmark_float_qps, TEST_IVF_SQ8) {
#ifdef KNOWHERE_WITH_GPU
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_GPU_IVFSQ8;
#else
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IVFSQ8;
#endif

    knowhere::Json conf = cfg_;
    for (auto nlist : NLISTs_) {
        conf[knowhere::indexparam::NLIST] = nlist;
        std::string index_file_name = get_index_name({nlist});
        create_index(index_file_name, conf);
        test_ivf(conf);
    }
}

TEST_F(Benchmark_float_qps, TEST_IVF_PQ) {
#ifdef KNOWHERE_WITH_GPU
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_GPU_IVFPQ;
#elif KNOWHERE_WITH_RAFT
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

TEST_F(Benchmark_float_qps, TEST_HNSW) {
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

TEST_F(Benchmark_float_qps, TEST_SCANN) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_SCANN;

    knowhere::Json conf = cfg_;
    for (auto reorder_k : SCANN_REORDER_K) {
        if (reorder_k < topk_) {
            continue;
        }
        conf[knowhere::indexparam::REORDER_K] = reorder_k;
        for (auto nlist : NLISTs_) {
            conf[knowhere::indexparam::NLIST] = nlist;
            for (const auto with_raw_data : SCANN_WITH_RAW_DATA) {
                conf[knowhere::indexparam::WITH_RAW_DATA] = with_raw_data;
                std::string index_file_name = get_index_name({nlist, reorder_k, with_raw_data});
                create_index(index_file_name, conf);
                test_scann(conf);
            }
        }
    }
}
TEST_F(Benchmark_float_qps, TEST_CAGRA) {
    index_type_ = knowhere::IndexEnum::INDEX_RAFT_CAGRA;
    knowhere::Json conf = cfg_;
    for (auto gd : GRAPH_DEGREE_) {
        conf[knowhere::indexparam::GRAPH_DEGREE] = gd;
        conf[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = gd;
        conf[knowhere::indexparam::MAX_ITERATIONS] = 64;
        std::string index_file_name = get_index_name({gd});
        create_index(index_file_name, conf);
        test_cagra(conf);
    }
}
