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

#pragma once

#include <math.h>
#include <sys/time.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

#define CALC_TIME_SPAN(X) \
    T1_ = elapsed();      \
    X;                    \
    TDIFF_ = elapsed() - T1_;

class Benchmark_base {
 public:
    inline void
    normalize(float* arr, int32_t nq, int32_t dim) {
#pragma omp parallel for
        for (int32_t i = 0; i < nq; i++) {
            double vecLen = 0.0, inv_vecLen = 0.0;
            for (int32_t j = 0; j < dim; j++) {
                double val = arr[i * dim + j];
                vecLen += val * val;
            }
            inv_vecLen = 1.0 / std::sqrt(vecLen);
            for (int32_t j = 0; j < dim; j++) {
                arr[i * dim + j] = (float)(arr[i * dim + j] * inv_vecLen);
            }
        }
    }

    static inline double
    elapsed() {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

    double
    get_time_diff() {
        return elapsed() - T0_;
    }

    /* Distance checking is only meaningful for binary index or FLAT index */
    void
    CheckDistance(const std::string& metric_type, const int64_t* ids, const float* distances, const size_t* lims,
                  int32_t nq) {
        const float FLOAT_DIFF = 0.00001;
        for (int32_t i = 0; i < nq; i++) {
            std::unordered_set<int32_t> gt_ids_set(gt_ids_ + gt_lims_[i], gt_ids_ + gt_lims_[i + 1]);
            std::unordered_map<int32_t, float> gt_map;
            for (auto j = gt_lims_[i]; j < gt_lims_[i + 1]; j++) {
                gt_map[gt_ids_[j]] = gt_dist_[j];
            }
            for (auto j = lims[i]; j < lims[i + 1]; j++) {
                if (gt_ids_set.count(ids[j]) > 0) {
                    ASSERT_LT(std::abs(distances[j] - gt_map[ids[j]]), FLOAT_DIFF);
                }
            }
        }
    }

    float
    CalcRecall(const int64_t* ids, int32_t nq, int32_t k) {
        int32_t min_k = std::min(gt_k_, k);
        int32_t hit = 0;
        for (int32_t i = 0; i < nq; i++) {
            std::unordered_set<int32_t> ground(gt_ids_ + i * gt_k_, gt_ids_ + i * gt_k_ + min_k);
            for (int32_t j = 0; j < min_k; j++) {
                auto id = ids[i * k + j];
                if (ground.count(id) > 0) {
                    hit++;
                }
            }
        }
        return (hit * 1.0f / (nq * min_k));
    }

    static float
    CalcRecall(const int64_t* g_ids, const int64_t* ids, int32_t nq, int32_t k) {
        int32_t hit = 0;
        for (int32_t i = 0; i < nq; i++) {
            std::unordered_set<int32_t> ground(g_ids + i * k, g_ids + (i + 1) * k);
            for (int32_t j = 0; j < k; j++) {
                auto id = ids[i * k + j];
                if (ground.count(id) > 0) {
                    hit++;
                }
            }
        }
        return (hit * 1.0f / (nq * k));
    }

    float
    CalcRecall(const int64_t* ids, int32_t nq_start, int32_t step, int32_t k) {
        assert(nq_start + step <= 10000);
        int32_t min_k = std::min(gt_k_, k);
        int32_t hit = 0;
        for (int32_t i = 0; i < step; i++) {
            std::unordered_set<int32_t> ground(gt_ids_ + (i + nq_start) * gt_k_,
                                               gt_ids_ + (i + nq_start) * gt_k_ + min_k);
            for (int32_t j = 0; j < min_k; j++) {
                auto id = ids[i * k + j];
                if (ground.count(id) > 0) {
                    hit++;
                }
            }
        }
        return (hit * 1.0f / (step * min_k));
    }

    int32_t
    CalcHits(const int64_t* ids, const size_t* lims, int32_t nq) {
        int32_t hit = 0;
        for (int32_t i = 0; i < nq; i++) {
            std::unordered_set<int64_t> gt_ids_set(gt_ids_ + gt_lims_[i], gt_ids_ + gt_lims_[i + 1]);
            for (auto j = lims[i]; j < lims[i + 1]; j++) {
                if (gt_ids_set.count(ids[j]) > 0) {
                    hit++;
                }
            }
        }
        return hit;
    }

    int32_t
    CalcHits(const int64_t* g_ids, const size_t* g_lims, const int64_t* ids, const size_t* lims, int32_t nq) {
        int32_t hit = 0;
        for (int32_t i = 0; i < nq; i++) {
            std::unordered_set<int64_t> gt_ids_set(g_ids + g_lims[i], g_ids + g_lims[i + 1]);
            for (auto j = lims[i]; j < lims[i + 1]; j++) {
                if (gt_ids_set.count(ids[j]) > 0) {
                    hit++;
                }
            }
        }
        return hit;
    }

    int32_t
    CalcHits(const int64_t* ids, const size_t* lims, int32_t start, int32_t num) {
        int32_t hit = 0;
        for (int32_t i = 0; i < num; i++) {
            std::unordered_set<int64_t> gt_ids_set(gt_ids_ + gt_lims_[start + i], gt_ids_ + gt_lims_[start + i + 1]);
            for (auto j = lims[i]; j < lims[i + 1]; j++) {
                if (gt_ids_set.count(ids[j]) > 0) {
                    hit++;
                }
            }
        }
        return hit;
    }

    float
    CalcRecall(const int64_t* ids, const size_t* lims, int32_t nq) {
        int32_t hit = CalcHits(ids, lims, nq);
        return (hit * 1.0f / gt_lims_[nq]);
    }

    float
    CalcAccuracy(const int64_t* ids, const size_t* lims, int32_t nq) {
        int32_t hit = CalcHits(ids, lims, nq);
        return (hit * 1.0f / lims[nq]);
    }

    float
    CalcRecall(const int64_t* g_ids, const size_t* g_lims, const int64_t* ids, const size_t* lims, int32_t nq) {
        int32_t hit = CalcHits(g_ids, g_lims, ids, lims, nq);
        return (hit * 1.0f / g_lims[nq]);
    }

    float
    CalcAccuracy(const int64_t* g_ids, const size_t* g_lims, const int64_t* ids, const size_t* lims, int32_t nq) {
        int32_t hit = CalcHits(g_ids, g_lims, ids, lims, nq);
        return (hit * 1.0f / lims[nq]);
    }

    void
    free_all() {
        if (xb_ != nullptr) {
            delete[] (float*)xb_;
        }
        if (xq_ != nullptr) {
            delete[] (float*)xq_;
        }
        if (gt_radius_ != nullptr) {
            delete[] gt_radius_;
        }
        if (gt_lims_ != nullptr) {
            delete[] gt_lims_;
        }
        if (gt_ids_ != nullptr) {
            delete[] gt_ids_;
        }
        if (gt_dist_ != nullptr) {
            delete[] gt_dist_;
        }
    }

 protected:
    double T0_, T1_, TDIFF_;
    std::string metric_type_;
    int32_t dim_;
    void* xb_ = nullptr;
    void* xq_ = nullptr;
    int32_t nb_;
    int32_t nq_;
    float* gt_radius_ = nullptr;  // ground-truth radius
    int32_t* gt_lims_ = nullptr;  // ground-truth lims
    int32_t* gt_ids_ = nullptr;   // ground-truth labels
    float* gt_dist_ = nullptr;    // ground-truth distances
    int32_t gt_k_;
};
