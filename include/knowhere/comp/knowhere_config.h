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

#ifndef COMP_KNOWHERE_CONFIG_H
#define COMP_KNOWHERE_CONFIG_H

#include <string>
#include <vector>

#ifdef KNOWHERE_WITH_DISKANN
#include "diskann/aio_context_pool.h"
#endif

namespace knowhere {

class KnowhereConfig {
 public:
    static void
    ShowVersion();

    /**
     * set SIMD type
     */
    enum SimdType {
        AUTO = 0,  // enable all and depend on the system
        AVX512,    // only enable AVX512
        AVX2,      // only enable AVX2
        SSE4_2,    // only enable SSE4_2
        GENERIC,   // use arithmetic instead of SIMD
    };

    static std::string
    SetSimdType(const SimdType simd_type);

    /**
     *The purpose of this interface is: part of the sealed indexes default to using bf16 as the base data to achieve
     *higher capacity; to ensure consistency in computation between growing and sealed, it is necessary to maintain the
     *same precision in growing calculations as in sealed.
     */
    static void
    EnablePatchForComputeFP32AsBF16();

    static void
    DisablePatchForComputeFP32AsBF16();

    /**
     * Set openblas threshold
     *   if nq < use_blas_threshold, calculated by omp
     *   else, calculated by openblas
     */
    static void
    SetBlasThreshold(const int64_t use_blas_threshold);

    static int64_t
    GetBlasThreshold();

    /**
     * set Clustering early stop [0, 100]
     *   It is to reduce the number of iterations of K-means.
     *   Between each two iterations, if the optimization rate < early_stop_threshold, stop
     *   And if early_stop_threshold = 0, won't early stop
     */
    static void
    SetEarlyStopThreshold(const double early_stop_threshold);

    static double
    GetEarlyStopThreshold();

    /**
     * set Clustering type
     */
    enum ClusteringType {
        K_MEANS = 0,        // k-means (default)
        K_MEANS_PLUS_PLUS,  // k-means++
    };

    static void
    SetClusteringType(const ClusteringType clustering_type);

    /**
     * The numebr of maximum parallel disk reads per thread.
     * On Linux, the default limit of `aio-max-nr` is 65536, so the product of `num_threads` and `max_events` (default
     * 128) should not exceed this value. You can type `sudo sysctl -a | grep fs.aio-max-nr` on your terminal to see
     * what is your default limit. If you want to raise the default limit, you can type `sudo sysctl -w fs.aio-max-nr=X`
     * on your terminal. This function returns true if validation passes. Otherwise, it returns false.
     */
    static bool
    SetAioContextPool(size_t num_ctx);

    static void
    SetBuildThreadPoolSize(size_t num_threads);
    static size_t
    GetBuildThreadPoolSize();

    static void
    SetSearchThreadPoolSize(size_t num_threads);
    static size_t
    GetSearchThreadPoolSize();

    /**
     * init GPU Resource
     */
    static void
    InitGPUResource(int64_t gpu_id, int64_t res_num = 2);

    /**
     * free GPU Resource
     */
    static void
    FreeGPUResource();
    /*
        static void
        SettingRaftMemPool(std::size_t init_size, std::size_t max_size);
    */
    static void
    SetRaftMemPool(size_t init_size, size_t max_size);

    /**
     * Initialize RAFT with defaults
     */
    static void
    SetRaftMemPool();
};

}  // namespace knowhere

#endif /* COMP_KNOWHERE_CONFIG_H */
