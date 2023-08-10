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

#include "knowhere/comp/knowhere_config.h"

#include <string>

#ifdef KNOWHERE_WITH_DISKANN
#include "diskann/aio_context_pool.h"
#endif
#include "faiss/Clustering.h"
#include "faiss/utils/distances.h"
#include "knowhere/log.h"
#ifdef KNOWHERE_WITH_GPU
#include "index/gpu/gpu_res_mgr.h"
#endif
#include "simd/hook.h"
#ifdef KNOWHERE_WITH_RAFT
#include "common/raft/raft_utils.h"
#endif

namespace knowhere {

void
KnowhereConfig::ShowVersion() {
#define XSTR(x) STR(x)
#define STR(x) #x

    std::string msg = "Knowhere Version: ";

#ifdef KNOWHERE_VERSION
    msg = msg + XSTR(KNOWHERE_VERSION);
#ifdef KNOWHERE_WITH_RAFT
    msg = msg + "-gpu";
#endif
#else
    msg = msg + " unknown";
#endif

#ifndef NDEBUG
    msg = msg + " (DEBUG)";
#endif

    LOG_KNOWHERE_INFO_ << msg;
}

std::string
KnowhereConfig::SetSimdType(const SimdType simd_type) {
#ifdef __x86_64__
    if (simd_type == SimdType::AUTO) {
        faiss::use_avx512 = true;
        faiss::use_avx2 = true;
        faiss::use_sse4_2 = true;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::AUTO";
    } else if (simd_type == SimdType::AVX512) {
        faiss::use_avx512 = true;
        faiss::use_avx2 = true;
        faiss::use_sse4_2 = true;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::AVX512";
    } else if (simd_type == SimdType::AVX2) {
        faiss::use_avx512 = false;
        faiss::use_avx2 = true;
        faiss::use_sse4_2 = true;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::AVX2";
    } else if (simd_type == SimdType::SSE4_2) {
        faiss::use_avx512 = false;
        faiss::use_avx2 = false;
        faiss::use_sse4_2 = true;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::SSE4_2";
    } else if (simd_type == SimdType::GENERIC) {
        faiss::use_avx512 = false;
        faiss::use_avx2 = false;
        faiss::use_sse4_2 = false;
        LOG_KNOWHERE_INFO_ << "FAISS expect simdType::GENERIC";
    }
#endif
    std::string simd_str;
    faiss::fvec_hook(simd_str);
    LOG_KNOWHERE_INFO_ << "FAISS hook " << simd_str;
    return simd_str;
}

void
KnowhereConfig::SetBlasThreshold(const int64_t use_blas_threshold) {
    LOG_KNOWHERE_INFO_ << "Set faiss::distance_compute_blas_threshold to " << use_blas_threshold;
    faiss::distance_compute_blas_threshold = static_cast<int>(use_blas_threshold);
}

int64_t
KnowhereConfig::GetBlasThreshold() {
    return faiss::distance_compute_blas_threshold;
}

void
KnowhereConfig::SetEarlyStopThreshold(const double early_stop_threshold) {
    LOG_KNOWHERE_INFO_ << "Set faiss::early_stop_threshold to " << early_stop_threshold;
    faiss::early_stop_threshold = early_stop_threshold;
}

double
KnowhereConfig::GetEarlyStopThreshold() {
    return faiss::early_stop_threshold;
}

void
KnowhereConfig::SetClusteringType(const ClusteringType clustering_type) {
    LOG_KNOWHERE_INFO_ << "Set faiss::clustering_type to " << clustering_type;
    switch (clustering_type) {
        case ClusteringType::K_MEANS:
        default:
            faiss::clustering_type = faiss::ClusteringType::K_MEANS;
            break;
        case ClusteringType::K_MEANS_PLUS_PLUS:
            faiss::clustering_type = faiss::ClusteringType::K_MEANS_PLUS_PLUS;
            break;
    }
}

bool
KnowhereConfig::SetAioContextPool(size_t num_ctx) {
#ifdef KNOWHERE_WITH_DISKANN
    return AioContextPool::InitGlobalAioPool(num_ctx, default_max_events);
#endif
    return true;
}

void
KnowhereConfig::InitGPUResource(int64_t gpu_id, int64_t res_num) {
#ifdef KNOWHERE_WITH_GPU
    LOG_KNOWHERE_INFO_ << "init GPU resource for gpu id " << gpu_id << ", resource num " << res_num;
    knowhere::GPUParams gpu_params(res_num);
    knowhere::GPUResMgr::GetInstance().InitDevice(gpu_id, gpu_params);
    knowhere::GPUResMgr::GetInstance().Init();
#endif
}

void
KnowhereConfig::FreeGPUResource() {
#ifdef KNOWHERE_WITH_GPU
    LOG_KNOWHERE_INFO_ << "free GPU resource";
    knowhere::GPUResMgr::GetInstance().Free();
#endif
}
void
KnowhereConfig::SetRaftMemPool(size_t init_size, size_t max_size) {
#ifdef KNOWHERE_WITH_RAFT
    raft_utils::set_mem_pool_size(init_size, max_size);
#endif
}

}  // namespace knowhere
