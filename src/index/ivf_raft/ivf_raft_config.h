/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef IVF_RAFT_CONFIG_H
#define IVF_RAFT_CONFIG_H

#include "index/ivf/ivf_config.h"
#include "knowhere/config.h"

namespace knowhere {

class RaftIvfFlatConfig : public IvfFlatConfig {
 public:
    CFG_INT kmeans_n_iters;
    CFG_FLOAT kmeans_trainset_fraction;
    CFG_BOOL adaptive_centers;
    KNOHWERE_DECLARE_CONFIG(RaftIvfFlatConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(k)
            .set_default(10)
            .description("search for top k similar vector.")
            .set_range(1, 256)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(kmeans_n_iters)
            .description("iterations to search for kmeans centers")
            .set_default(20)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(kmeans_trainset_fraction)
            .description("fraction of data to use in kmeans building")
            .set_default(0.5)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(adaptive_centers)
            .description("update centroids with new data")
            .set_default(false)
            .for_train();
    }
};

class RaftIvfPqConfig : public IvfPqConfig {
 public:
    CFG_INT streams_per_device;
    CFG_INT kmeans_n_iters;
    CFG_FLOAT kmeans_trainset_fraction;
    CFG_INT m;
    CFG_STRING codebook_kind;
    CFG_BOOL force_random_rotation;
    CFG_BOOL conservative_memory_allocation;
    CFG_STRING lut_dtype;
    CFG_STRING internal_distance_dtype;
    CFG_FLOAT preferred_shmem_carveout;
    KNOHWERE_DECLARE_CONFIG(RaftIvfPqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(k)
            .set_default(10)
            .description("search for top k similar vector.")
            .set_range(1, 1024)
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(streams_per_device)
            .description("CUDA streams per GPU")
            .set_default(1)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(kmeans_n_iters)
            .description("iterations to search for kmeans centers")
            .set_default(20)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(kmeans_trainset_fraction)
            .description("fraction of data to use in kmeans building")
            .set_default(0.5)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(m).description("dimension after compression by PQ").set_default(0).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(codebook_kind)
            .description("how PQ codebooks are created")
            .set_default("PER_SUBSPACE")
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(force_random_rotation)
            .description("always perform random rotation")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(conservative_memory_allocation)
            .description("use minimal possible GPU memory")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(lut_dtype)
            .description("Data type for lookup table")
            .set_default("CUDA_R_32F")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(internal_distance_dtype)
            .description("Data type for distance storage")
            .set_default("CUDA_R_32F")
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(preferred_shmem_carveout)
            .description("preferred fraction of memory for shmem vs L1")
            .set_default(1.0)
            .for_search();
    }
};

}  // namespace knowhere
#endif /* IVF_CONFIG_H */
