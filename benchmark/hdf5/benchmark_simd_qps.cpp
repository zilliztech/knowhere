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
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "src/simd/hook.h"

typedef void (*worker)(knowhere::DataSetPtr, knowhere::DataSetPtr, int32_t, int32_t, float*);

class Benchmark_simd_qps : public Benchmark_knowhere, public ::testing::Test {
 public:
    template <typename T>
    void
    test_simd(std::string worker_name, worker worker_func, float* dist = nullptr) {
        const int32_t thread_num = 8;
        std::string data_type_str = get_data_type_name<T>();

        auto base_ds_ptr = knowhere::GenDataSet(nb_, dim_, xb_);
        auto base = knowhere::ConvertToDataTypeIfNeeded<T>(base_ds_ptr);

        auto query_ds_ptr = knowhere::GenDataSet(nq_, dim_, xq_);
        auto query = knowhere::ConvertToDataTypeIfNeeded<T>(query_ds_ptr);

        printf("\n[%0.3f s] %s (%s) \n", get_time_diff(), ann_test_name_.c_str(), data_type_str.c_str());
        printf("================================================================================\n");
        for (auto simd_type : SIMD_TYPEs_) {
            std::string simd_str = knowhere::KnowhereConfig::SetSimdType(simd_type);
            CALC_TIME_SPAN(task<T>(base, query, worker_func, thread_num, dist));
            printf("  func = %10s, simd_type = %7s, elapse = %6.3fs, VPS = %.3f\n", worker_name.c_str(),
                   simd_str.c_str(), TDIFF_, nq_ / TDIFF_);
            std::fflush(stdout);
        }
        printf("================================================================================\n");
        printf("[%.3f s] Test '%s/%s' done\n\n", get_time_diff(), ann_test_name_.c_str(), index_type_.c_str());
    }

    template <typename T>
    static void
    ip_worker(knowhere::DataSetPtr base, knowhere::DataSetPtr query, int32_t start, int32_t num, float* dist) {
        auto dim = base->GetDim();
        auto nb = base->GetRows();
        auto xb = (const T*)base->GetTensor();

        auto nq = query->GetRows();
        auto xq = (const T*)query->GetTensor();

        num = std::min<int32_t>(num, nq - start);
        for (int32_t i = 0; i < num; i++) {
            const size_t offset = (start + i) * dim;
            const T* x = xq + offset;
            for (int32_t j = 0; j < nb; j++) {
                const T* y = xb + j * dim;
                if constexpr (std::is_same_v<T, knowhere::fp32>) {
                    auto d = faiss::fvec_inner_product(x, y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::fp16>) {
                    auto d = faiss::fp16_vec_inner_product(x, y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::bf16>) {
                    auto d = faiss::bf16_vec_inner_product(x, y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::int8>) {
                    auto d = faiss::int8_vec_inner_product(x, y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                }
            }
        }
    }

    template <typename T>
    static void
    l2_worker(knowhere::DataSetPtr base, knowhere::DataSetPtr query, int32_t start, int32_t num, float* dist) {
        auto dim = base->GetDim();
        auto nb = base->GetRows();
        auto xb = (const T*)base->GetTensor();

        auto nq = query->GetRows();
        auto xq = (const T*)query->GetTensor();

        num = std::min<int32_t>(num, nq - start);
        for (int32_t i = 0; i < num; i++) {
            const size_t offset = (start + i) * dim;
            const T* x = xq + offset;
            for (int32_t j = 0; j < nb; j++) {
                const T* y = xb + j * dim;
                if constexpr (std::is_same_v<T, knowhere::fp32>) {
                    auto d = faiss::fvec_L2sqr(x, y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::fp16>) {
                    auto d = faiss::fp16_vec_L2sqr(x, y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::bf16>) {
                    auto d = faiss::bf16_vec_L2sqr(x, y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::int8>) {
                    auto d = faiss::int8_vec_L2sqr(x, y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                }
            }
        }
    }

    template <typename T>
    static void
    l2_norm_worker(knowhere::DataSetPtr base, knowhere::DataSetPtr query, int32_t start, int32_t num, float* dist) {
        auto dim = base->GetDim();
        auto nb = base->GetRows();
        auto xb = (const T*)base->GetTensor();

        auto nq = query->GetRows();

        num = std::min<int32_t>(num, nq - start);
        for (int32_t i = 0; i < num; i++) {
            for (int32_t j = 0; j < nb; j++) {
                const T* y = xb + j * dim;
                if constexpr (std::is_same_v<T, knowhere::fp32>) {
                    auto d = faiss::fvec_norm_L2sqr(y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::fp16>) {
                    auto d = faiss::fp16_vec_norm_L2sqr(y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::bf16>) {
                    auto d = faiss::bf16_vec_norm_L2sqr(y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                } else if constexpr (std::is_same_v<T, knowhere::int8>) {
                    auto d = faiss::int8_vec_norm_L2sqr(y, dim);
                    if (dist) {
                        dist[(start + i) * nb + j] = d;
                    }
                }
            }
        }
    }

    template <typename T>
    static void
    ip_batch_4_worker(knowhere::DataSetPtr base, knowhere::DataSetPtr query, int32_t start, int32_t num, float* dist) {
        auto dim = base->GetDim();
        auto nb = base->GetRows();
        auto xb = (const T*)base->GetTensor();

        auto nq = query->GetRows();
        auto xq = (const T*)query->GetTensor();

        num = std::min<int32_t>(num, nq - start);
        for (int32_t i = 0; i < num; i++) {
            const size_t offset = (start + i) * dim;
            const T* x = xq + offset;
            for (int32_t j = 0; j < nb; j += 4) {
                const T* y = xb + j * dim;
                if constexpr (std::is_same_v<T, knowhere::fp32>) {
                    if (dist) {
                        faiss::fvec_inner_product_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim,
                                                          dist[(start + i) * nb + j], dist[(start + i) * nb + j + 1],
                                                          dist[(start + i) * nb + j + 2],
                                                          dist[(start + i) * nb + j + 3]);
                    } else {
                        float d0, d1, d2, d3;
                        faiss::fvec_inner_product_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, d0, d1, d2, d3);
                    }
                } else if constexpr (std::is_same_v<T, knowhere::fp16>) {
                    if (dist) {
                        faiss::fp16_vec_inner_product_batch_4(
                            x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, dist[(start + i) * nb + j],
                            dist[(start + i) * nb + j + 1], dist[(start + i) * nb + j + 2],
                            dist[(start + i) * nb + j + 3]);
                    } else {
                        float d0, d1, d2, d3;
                        faiss::fp16_vec_inner_product_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, d0, d1, d2,
                                                              d3);
                    }
                } else if constexpr (std::is_same_v<T, knowhere::bf16>) {
                    if (dist) {
                        faiss::bf16_vec_inner_product_batch_4(
                            x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, dist[(start + i) * nb + j],
                            dist[(start + i) * nb + j + 1], dist[(start + i) * nb + j + 2],
                            dist[(start + i) * nb + j + 3]);
                    } else {
                        float d0, d1, d2, d3;
                        faiss::bf16_vec_inner_product_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, d0, d1, d2,
                                                              d3);
                    }
                } else if constexpr (std::is_same_v<T, knowhere::int8>) {
                    if (dist) {
                        faiss::int8_vec_inner_product_batch_4(
                            x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, dist[(start + i) * nb + j],
                            dist[(start + i) * nb + j + 1], dist[(start + i) * nb + j + 2],
                            dist[(start + i) * nb + j + 3]);
                    } else {
                        float d0, d1, d2, d3;
                        faiss::int8_vec_inner_product_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, d0, d1, d2,
                                                              d3);
                    }
                }
            }
        }
    }

    template <typename T>
    static void
    l2_batch_4_worker(knowhere::DataSetPtr base, knowhere::DataSetPtr query, int32_t start, int32_t num, float* dist) {
        auto dim = base->GetDim();
        auto nb = base->GetRows();
        auto xb = (const T*)base->GetTensor();

        auto nq = query->GetRows();
        auto xq = (const T*)query->GetTensor();

        num = std::min<int32_t>(num, nq - start);
        for (int32_t i = 0; i < num; i++) {
            const size_t offset = (start + i) * dim;
            const T* x = xq + offset;
            for (int32_t j = 0; j < nb; j += 4) {
                const T* y = xb + j * dim;
                if constexpr (std::is_same_v<T, knowhere::fp32>) {
                    if (dist) {
                        faiss::fvec_L2sqr_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim,
                                                  dist[(start + i) * nb + j], dist[(start + i) * nb + j + 1],
                                                  dist[(start + i) * nb + j + 2], dist[(start + i) * nb + j + 3]);
                    } else {
                        float d0, d1, d2, d3;
                        faiss::fvec_L2sqr_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, d0, d1, d2, d3);
                    }
                } else if constexpr (std::is_same_v<T, knowhere::fp16>) {
                    if (dist) {
                        faiss::fp16_vec_L2sqr_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim,
                                                      dist[(start + i) * nb + j], dist[(start + i) * nb + j + 1],
                                                      dist[(start + i) * nb + j + 2], dist[(start + i) * nb + j + 3]);
                    } else {
                        float d0, d1, d2, d3;
                        faiss::fp16_vec_L2sqr_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, d0, d1, d2, d3);
                    }
                } else if constexpr (std::is_same_v<T, knowhere::bf16>) {
                    if (dist) {
                        faiss::bf16_vec_L2sqr_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim,
                                                      dist[(start + i) * nb + j], dist[(start + i) * nb + j + 1],
                                                      dist[(start + i) * nb + j + 2], dist[(start + i) * nb + j + 3]);
                    } else {
                        float d0, d1, d2, d3;
                        faiss::bf16_vec_L2sqr_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, d0, d1, d2, d3);
                    }
                } else if constexpr (std::is_same_v<T, knowhere::int8>) {
                    if (dist) {
                        faiss::int8_vec_L2sqr_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim,
                                                      dist[(start + i) * nb + j], dist[(start + i) * nb + j + 1],
                                                      dist[(start + i) * nb + j + 2], dist[(start + i) * nb + j + 3]);
                    } else {
                        float d0, d1, d2, d3;
                        faiss::int8_vec_L2sqr_batch_4(x, y, y + dim, y + 2 * dim, y + 3 * dim, dim, d0, d1, d2, d3);
                    }
                }
            }
        }
    }

 private:
    template <typename T>
    void
    task(knowhere::DataSetPtr base, knowhere::DataSetPtr query, worker worker_func, int32_t worker_num, float* dist) {
        auto nq = query->GetRows();

        std::vector<std::thread> thread_vector(worker_num);
        for (int32_t i = 0; i < worker_num; i++) {
            int32_t idx_start, req_num;
            req_num = nq / worker_num;
            if (nq % worker_num != 0) {
                req_num++;
            }
            idx_start = req_num * i;
            thread_vector[i] = std::thread(worker_func, base, query, idx_start, req_num, dist);
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

        nb_ = 100000;
        nq_ = 800;
    }

    void
    TearDown() override {
        free_all();
    }

 protected:
    const std::vector<int32_t> THREAD_NUMs_ = {8};
    const std::vector<knowhere::KnowhereConfig::SimdType> SIMD_TYPEs_ = {
#if defined(__x86_64__)
        knowhere::KnowhereConfig::SimdType::AVX512,
        knowhere::KnowhereConfig::SimdType::AVX2,
        knowhere::KnowhereConfig::SimdType::SSE4_2,
#endif
        knowhere::KnowhereConfig::SimdType::GENERIC,
    };
};

TEST_F(Benchmark_simd_qps, TEST_SIMD) {
    using T1 = knowhere::fp32;
    test_simd<T1>("IP", ip_worker<T1>);
    test_simd<T1>("L2", l2_worker<T1>);
    test_simd<T1>("L2_NORM", l2_norm_worker<T1>);
    test_simd<T1>("IP_BATCH_4", ip_batch_4_worker<T1>);
    test_simd<T1>("L2_BATCH_4", l2_batch_4_worker<T1>);

    using T2 = knowhere::fp16;
    test_simd<T2>("IP", ip_worker<T2>);
    test_simd<T2>("L2", l2_worker<T2>);
    test_simd<T2>("L2_NORM", l2_norm_worker<T2>);
    test_simd<T2>("IP_BATCH_4", ip_batch_4_worker<T2>);
    test_simd<T2>("L2_BATCH_4", l2_batch_4_worker<T2>);

    using T3 = knowhere::bf16;
    test_simd<T3>("IP", ip_worker<T3>);
    test_simd<T3>("L2", l2_worker<T3>);
    test_simd<T3>("L2_NORM", l2_norm_worker<T3>);
    test_simd<T3>("IP_BATCH_4", ip_batch_4_worker<T3>);
    test_simd<T3>("L2_BATCH_4", l2_batch_4_worker<T3>);

    using T4 = knowhere::int8;
    test_simd<T4>("IP", ip_worker<T4>);
    test_simd<T4>("L2", l2_worker<T4>);
    test_simd<T4>("L2_NORM", l2_norm_worker<T4>);
    test_simd<T4>("IP_BATCH_4", ip_batch_4_worker<T4>);
    test_simd<T4>("L2_BATCH_4", l2_batch_4_worker<T4>);
}
