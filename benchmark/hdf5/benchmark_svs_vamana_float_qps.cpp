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

class Benchmark_svs_vamana_float_qps : public Benchmark_knowhere, public ::testing::Test {
 public:
    template <typename T>
    void
    test_svs_vamana(const knowhere::Json& cfg) {
        auto conf = cfg;
        auto degree = conf[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE].get<int32_t>();
        auto storage = conf[knowhere::indexparam::SVS_STORAGE_KIND].get<std::string>();
        std::string data_type_str = get_data_type_name<T>();

        auto find_smallest_search_window = [&](float expected_recall) -> std::tuple<int32_t, float> {
            std::unordered_map<int32_t, float> recall_map;
            conf[knowhere::meta::TOPK] = topk_;
            auto ds_ptr = knowhere::GenDataSet(nq_, dim_, xq_);
            auto query = knowhere::ConvertToDataTypeIfNeeded<T>(ds_ptr);

            int32_t left = topk_, right = 512, sw;
            float recall;
            while (left <= right) {
                sw = left + (right - left) / 2;
                conf[knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE] = sw;
                conf[knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY] = sw;

                auto result = index_.value().Search(query, conf, nullptr);
                recall = CalcRecall(result.value()->GetIds(), nq_, topk_);
                recall_map[sw] = recall;
                printf("[%0.3f s] iterate SVS Vamana param for recall %.4f: degree=%d, storage=%s, sw=%4d, k=%d, "
                       "R@=%.4f\n",
                       get_time_diff(), expected_recall, degree, storage.c_str(), sw, topk_, recall);
                std::fflush(stdout);
                if (std::abs(recall - expected_recall) <= 0.0001) {
                    return {sw, recall_map[sw]};
                }
                if (recall < expected_recall) {
                    left = sw + 1;
                } else {
                    right = sw - 1;
                }
            }
            return {left, recall_map.count(left) ? recall_map[left] : recall_map[sw]};
        };

        for (auto expected_recall : EXPECTED_RECALLs_) {
            auto [sw, recall] = find_smallest_search_window(expected_recall);
            conf[knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE] = sw;
            conf[knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY] = sw;
            conf[knowhere::meta::TOPK] = topk_;

            printf("\n[%0.3f s] %s | %s(%s) | degree=%d, storage=%s, sw=%d, k=%d, R@=%.4f\n", get_time_diff(),
                   ann_test_name_.c_str(), index_type_.c_str(), data_type_str.c_str(), degree, storage.c_str(), sw,
                   topk_, recall);
            printf("================================================================================\n");
            for (auto thread_num : THREAD_NUMs_) {
                CALC_TIME_SPAN(task<T>(conf, thread_num, nq_));
                printf("  thread_num = %2d, elapse = %6.3fs, VPS = %.3f\n", thread_num, TDIFF_, nq_ / TDIFF_);
                std::fflush(stdout);
            }
            printf("================================================================================\n");
            printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
        }
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
    const std::vector<float> EXPECTED_RECALLs_ = {0.8, 0.95};
    const std::vector<int32_t> THREAD_NUMs_ = {8};

    // SVS Vamana index params
    const std::vector<int32_t> GRAPH_DEGREEs_ = {32, 64};
    const int32_t CONSTRUCTION_WINDOW_SIZE_ = 200;
};

#define TEST_INDEX(NAME, T, X)              \
    index_file_name = get_index_name<T>(X); \
    create_index<T>(index_file_name, conf); \
    test_##NAME<T>(conf)

#ifdef KNOWHERE_WITH_SVS

TEST_F(Benchmark_svs_vamana_float_qps, TEST_SVS_VAMANA_FP32) {
    index_type_ = knowhere::IndexEnum::INDEX_SVS_VAMANA;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = CONSTRUCTION_WINDOW_SIZE_;
    conf[knowhere::indexparam::SVS_ALPHA] = 1.2f;

    for (auto degree : GRAPH_DEGREEs_) {
        conf[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = degree;
        conf[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("fp32");
        std::vector<int32_t> params = {degree};

        TEST_INDEX(svs_vamana, knowhere::fp32, params);
    }
}

TEST_F(Benchmark_svs_vamana_float_qps, TEST_SVS_VAMANA_FP16) {
    index_type_ = knowhere::IndexEnum::INDEX_SVS_VAMANA;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = CONSTRUCTION_WINDOW_SIZE_;
    conf[knowhere::indexparam::SVS_ALPHA] = 1.2f;

    for (auto degree : GRAPH_DEGREEs_) {
        conf[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = degree;
        conf[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("fp16");
        std::vector<int32_t> params = {degree};

        TEST_INDEX(svs_vamana, knowhere::fp32, params);
    }
}

TEST_F(Benchmark_svs_vamana_float_qps, TEST_SVS_VAMANA_SQI8) {
    index_type_ = knowhere::IndexEnum::INDEX_SVS_VAMANA;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = CONSTRUCTION_WINDOW_SIZE_;
    conf[knowhere::indexparam::SVS_ALPHA] = 1.2f;

    for (auto degree : GRAPH_DEGREEs_) {
        conf[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = degree;
        conf[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("sqi8");
        std::vector<int32_t> params = {degree};

        TEST_INDEX(svs_vamana, knowhere::fp32, params);
    }
}

TEST_F(Benchmark_svs_vamana_float_qps, TEST_SVS_VAMANA_LVQ4x4) {
    index_type_ = knowhere::IndexEnum::INDEX_SVS_VAMANA_LVQ;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = CONSTRUCTION_WINDOW_SIZE_;
    conf[knowhere::indexparam::SVS_ALPHA] = 1.2f;

    for (auto degree : GRAPH_DEGREEs_) {
        conf[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = degree;
        conf[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("lvq4x4");
        std::vector<int32_t> params = {degree};

        TEST_INDEX(svs_vamana, knowhere::fp32, params);
    }
}

TEST_F(Benchmark_svs_vamana_float_qps, TEST_SVS_VAMANA_LVQ4x8) {
    index_type_ = knowhere::IndexEnum::INDEX_SVS_VAMANA_LVQ;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = CONSTRUCTION_WINDOW_SIZE_;
    conf[knowhere::indexparam::SVS_ALPHA] = 1.2f;

    for (auto degree : GRAPH_DEGREEs_) {
        conf[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = degree;
        conf[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("lvq4x8");
        std::vector<int32_t> params = {degree};

        TEST_INDEX(svs_vamana, knowhere::fp32, params);
    }
}

TEST_F(Benchmark_svs_vamana_float_qps, TEST_SVS_VAMANA_LEANVEC4x4) {
    index_type_ = knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = CONSTRUCTION_WINDOW_SIZE_;
    conf[knowhere::indexparam::SVS_ALPHA] = 1.2f;
    conf[knowhere::indexparam::SVS_LEANVEC_DIM] = 0;

    for (auto degree : GRAPH_DEGREEs_) {
        conf[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = degree;
        conf[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("leanvec4x4");
        std::vector<int32_t> params = {degree};

        TEST_INDEX(svs_vamana, knowhere::fp32, params);
    }
}

TEST_F(Benchmark_svs_vamana_float_qps, TEST_SVS_VAMANA_LEANVEC8x8) {
    index_type_ = knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC;

    std::string index_file_name;
    knowhere::Json conf = cfg_;
    conf[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = CONSTRUCTION_WINDOW_SIZE_;
    conf[knowhere::indexparam::SVS_ALPHA] = 1.2f;
    conf[knowhere::indexparam::SVS_LEANVEC_DIM] = 0;

    for (auto degree : GRAPH_DEGREEs_) {
        conf[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = degree;
        conf[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("leanvec8x8");
        std::vector<int32_t> params = {degree};

        TEST_INDEX(svs_vamana, knowhere::fp32, params);
    }
}

#endif  // KNOWHERE_WITH_SVS
