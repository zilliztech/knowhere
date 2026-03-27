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

class Benchmark_svs_float_qps : public Benchmark_knowhere, public ::testing::Test {
 public:
    template <typename T>
    void
    test_flat(const knowhere::Json& cfg) {
        auto conf = cfg;

        float expected_recall = 1.0f;
        conf[knowhere::meta::TOPK] = topk_;

        std::string data_type_str = get_data_type_name<T>();
        printf("\n[%0.3f s] %s | %s(%s) | k=%d, R@=%.4f\n", get_time_diff(), ann_test_name_.c_str(),
               index_type_.c_str(), data_type_str.c_str(), topk_, expected_recall);
        printf("================================================================================\n");
        for (auto thread_num : THREAD_NUMs_) {
            CALC_TIME_SPAN(task<T>(conf, thread_num, nq_));
            printf("  thread_num = %2d, elapse = %6.3fs, VPS = %.3f\n", thread_num, TDIFF_, nq_ / TDIFF_);
            std::fflush(stdout);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

 private:
    template <typename T>
    void
    task(const knowhere::Json& conf, int32_t worker_num, int32_t nq_total) {
        auto worker = [&](int32_t idx_start, int32_t num) {
            num = std::min(num, nq_total - idx_start);
            for (int32_t i = 0; i < num; i++) {
                auto ds_ptr = knowhere::GenDataSet(1, dim_, (const float*)xq_ + (idx_start + i) * dim_);
                auto query = knowhere::ConvertToDataTypeIfNeeded<T>(ds_ptr);
                index_.value().Search(query, conf, nullptr);
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
        load_hdf5_data<knowhere::fp32>();

        cfg_[knowhere::meta::METRIC_TYPE] = metric_type_;
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
    const int32_t topk_ = 100;
    const std::vector<int32_t> THREAD_NUMs_ = {8};
};

#define TEST_INDEX(NAME, T, X)              \
    index_file_name = get_index_name<T>(X); \
    create_index<T>(index_file_name, conf); \
    test_##NAME<T>(conf)

TEST_F(Benchmark_svs_float_qps, TEST_FAISS_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_FAISS_IDMAP;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    std::vector<int32_t> params = {};

    TEST_INDEX(flat, knowhere::fp32, params);
}

#ifdef KNOWHERE_WITH_SVS
TEST_F(Benchmark_svs_float_qps, TEST_SVS_FLAT) {
    index_type_ = knowhere::IndexEnum::INDEX_SVS_FLAT;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    std::vector<int32_t> params = {};

    TEST_INDEX(flat, knowhere::fp32, params);
}
#endif
