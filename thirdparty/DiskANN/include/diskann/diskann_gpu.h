// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.
/*
* Copyright (c) 2024, NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* MODIFICATIONS:
* [21/08/2025] - [Gili Buzaglo]:
* Import and adjust methods from cuvs library example source to knowhere
* (vamana_build_and_write and read_bin_dataset)
*
* source repository: https://github.com/rapidsai/cuvs
* release tag: v25.06.00
* commit id: 2e20f1ec8a7e2325cdcf49d289879c2624e5bb5d
* File name: https://github.com/rapidsai/cuvs/blob/v25.06.00/examples/cpp/src/vamana_example.cu
* Function name: vamana_build_and_write
* Lines: 30 – 63
*
* source repository: https://github.com/rapidsai/cuvs
* release tag: v25.06.00
* commit id: 2e20f1ec8a7e2325cdcf49d289879c2624e5bb5d
* File name: https://github.com/rapidsai/cuvs/blob/v25.06.00/examples/cpp/src/common.cuh
* Function name: read_bin_dataset
* Lines: 106 - 129
*/

#ifndef THIRDPARTY_DISKANN_INCLUDE_DISKANN_DISKANN_GPU_H_
#define THIRDPARTY_DISKANN_INCLUDE_DISKANN_DISKANN_GPU_H_
#pragma once

#ifdef KNOWHERE_WITH_CUVS
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>

template <typename T, typename idxT = std::size_t>
raft::device_matrix<T, idxT> read_bin_dataset(const raft::device_resources& dev_resources,
                                              const std::string& fname);

template <typename T>
void vamana_build_and_write(raft::device_resources const& dev_resources,
                            raft::device_matrix_view<const T, uint32_t> dataset,
                            std::string out_fname,
                            int degree,
                            int visited_size,
                            float max_fraction,
                            int iters);
bool is_gpu_available();

void kmeans_gpu(
    raft::resources& dev_resources,
    const float* h_chunk_data,
    size_t num_train,
    size_t chunk_size,
    size_t num_centers,
    int max_iter,
    float* h_centroids_out,
	bool is_balanced=false);


int predict_gpu(raft::resources& dev_resources,
				const float* h_data,
                size_t n_samples,
                size_t dim,
                const float* h_centroids,
                size_t n_clusters,
				int* h_labels);

template <typename T>
int brute_force_gpu(raft::resources& dev_resources,
		const T* h_data,
        size_t n_samples,
        size_t dim,
		size_t k,
        const T* queries,
        size_t n_queries,
		int64_t* h_labels);

void gpu_get_mem_info(raft::resources &dev_resources, size_t &gpu_free_mem,size_t &gpu_total_mem);

#endif




#endif /* THIRDPARTY_DISKANN_INCLUDE_DISKANN_DISKANN_GPU_H_ */
