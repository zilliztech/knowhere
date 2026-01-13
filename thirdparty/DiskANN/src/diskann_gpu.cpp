// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <cuda_runtime.h>
#include <cuvs/neighbors/vamana.hpp>
#include "diskann/diskann_gpu.h"
#include "knowhere/log.h"
#include <fstream>
#include <iostream>
#include <raft/core/resources.hpp>
#include <cuvs/neighbors/brute_force.hpp>

template<typename T, typename idxT>
raft::device_matrix<T, idxT> read_bin_dataset(
		const raft::device_resources &dev_resources, const std::string &fname) {
	// Open file
	std::ifstream datafile(fname, std::ios::binary);
	if (!datafile) {
		throw std::runtime_error("Failed to open file: " + fname);
	}

	// Read header
	uint32_t N = 0;
	uint32_t dim = 0;
	datafile.read(reinterpret_cast<char*>(&N), sizeof(uint32_t));
	datafile.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    LOG_KNOWHERE_INFO_ << "Read in file - N: " << N << ", dim: " << dim;

	std::size_t total = static_cast<std::size_t>(N) * dim;
	if (total == 0) {
		throw std::runtime_error("Invalid dataset: total size is zero");
	}

	// Allocate pinned host memory
	T *pinned_data = nullptr;
	RAFT_CUDA_TRY(
			cudaMallocHost(reinterpret_cast<void**>(&pinned_data),
					total * sizeof(T)));

	// Read data into pinned memory
	datafile.read(reinterpret_cast<char*>(pinned_data), total * sizeof(T));
	if (!datafile) {
		cudaFreeHost(pinned_data);
		throw std::runtime_error("Failed to read vector data from file");
	}
	datafile.close();

	// Create device matrix
	auto dataset = raft::make_device_matrix<T, idxT>(dev_resources, N, dim);

	// Copy to device using raft::copy
	raft::copy(dataset.data_handle(), pinned_data, total,
			raft::resource::get_cuda_stream(dev_resources));

	cudaFreeHost(pinned_data);
	return dataset;
}

template<typename T>
void vamana_build_and_write(raft::device_resources const &dev_resources,
		raft::device_matrix_view<const T, uint32_t> dataset,
		std::string out_fname, int degree, int visited_size, float max_fraction,
		int iters) {
	using namespace cuvs::neighbors;

	// use default index parameters
	vamana::index_params index_params;
	index_params.max_fraction = max_fraction;
	index_params.visited_size = visited_size;
	index_params.graph_degree = degree;
	index_params.vamana_iters = iters;

	LOG_KNOWHERE_INFO_ << "Building Vamana index (search graph)";

	auto start = std::chrono::system_clock::now();
	auto index = vamana::build(dev_resources, index_params, dataset);
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	LOG_KNOWHERE_INFO_ << "Vamana index has " << index.size() << " vectors";
	LOG_KNOWHERE_INFO_ << "Vamana graph has degree " << index.graph_degree()
			<< ", graph size [" << index.graph().extent(0) << ", "
			<< index.graph().extent(1) << "]";

	LOG_KNOWHERE_INFO_ << "Time to build index: " << elapsed_seconds.count() << "s\n";
	// Output index to file
	serialize(dev_resources, out_fname, index);
}

void kmeans_gpu(raft::resources &dev_resources, const float *h_chunk_data,
		size_t num_train, size_t dim, size_t num_centers, int max_iter,
		float *h_centroids_out, bool is_balanced/*=false*/) {
	// KMeans parameters
	cuvs::cluster::kmeans::params km_params;
	int64_t n_iter;
	float inertia;
	km_params.max_iter = max_iter;
	km_params.n_init = 1;
	km_params.n_clusters = num_centers;

	// Allocate device matrices
	auto d_data = raft::make_device_matrix<float>(dev_resources, num_train, dim);
	auto d_centroids = raft::make_device_matrix<float>(dev_resources, num_centers, dim);

	// Copy input chunk to device asynchronously
	raft::copy(d_data.data_handle(), h_chunk_data, num_train * dim, raft::resource::get_cuda_stream(dev_resources));
	// Run KMeans (fit uses RAFT-managed streams internally)
	if(!is_balanced) {
		cuvs::cluster::kmeans::fit(dev_resources, km_params, d_data.view(), std::nullopt,
				d_centroids.view(),
				raft::make_host_scalar_view<float, int64_t>(&inertia),
				raft::make_host_scalar_view<int64_t, int64_t>(&n_iter));
	}else {
		//use balance kmeans
		cuvs::cluster::kmeans::balanced_params b_p;
		b_p.n_iters=100;
		cuvs::cluster::kmeans::fit(dev_resources, b_p, d_data.view(), d_centroids.view());
	}
	// Copy centroids back to host asynchronously
	raft::copy(h_centroids_out, d_centroids.data_handle(), num_centers * dim,
			raft::resource::get_cuda_stream(dev_resources));
}


int predict_gpu(raft::resources& handle,
                const float* h_data,
                size_t n_samples,
                size_t dim,
                const float* h_centroids,
                size_t n_clusters,
                int* h_labels) {
    if (!is_gpu_available()) return -1;

    // Allocate device buffers
    auto d_data      = raft::make_device_matrix<float>(handle, n_samples, dim);
    auto d_centroids = raft::make_device_matrix<float>(handle, n_clusters, dim);
    auto d_labels    = raft::make_device_vector<int>(handle, n_samples);

    // Copy to device
    raft::copy(d_data.data_handle(), h_data, n_samples * dim,
               raft::resource::get_cuda_stream(handle));
    raft::copy(d_centroids.data_handle(), h_centroids, n_clusters * dim,
               raft::resource::get_cuda_stream(handle));

    float inertia = 0.0f;
    auto inertia_view = raft::make_host_scalar_view(&inertia);

    cuvs::cluster::kmeans::params params;
    params.n_clusters = static_cast<int>(n_clusters);
    params.metric     = cuvs::distance::DistanceType::L2Expanded;

    cuvs::cluster::kmeans::predict(
        handle,
        params,
        d_data.view(),
        std::nullopt,
        d_centroids.view(),
        d_labels.view(),
        /*normalize_weight=*/false,
        inertia_view);

    // Copy back
    raft::copy(h_labels, d_labels.data_handle(), n_samples,
               raft::resource::get_cuda_stream(handle));
    raft::resource::sync_stream(handle);

    return 0;
}

bool is_gpu_available() {
	int count = 0;
	return (cudaGetDeviceCount(&count) == cudaSuccess) && (count > 0);
}

void gpu_get_mem_info(raft::resources &dev_resources, size_t &gpu_free_mem,
		size_t &gpu_total_mem) {
    int dev = raft::resource::get_device_id(dev_resources);

    int prev_dev;
    cudaGetDevice(&prev_dev);

    cudaSetDevice(dev);

    cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);

    // restore previous device
    cudaSetDevice(prev_dev);
}

template<typename T>
int brute_force_gpu(raft::resources &dev_resources, const T *h_data,
		size_t n_samples, size_t dim, size_t k, const T *queries,
		size_t n_queries, int64_t *h_labels) {
	using namespace cuvs::neighbors;

	brute_force::index_params index_params;
	brute_force::search_params search_params;
	auto dataset = raft::make_device_matrix<T, int64_t>(dev_resources, n_samples, dim);
	auto queries_matrix = raft::make_device_matrix<T, int64_t>(dev_resources, n_queries, dim);

	raft::copy(dataset.data_handle(), h_data, n_samples * dim,
			raft::resource::get_cuda_stream(dev_resources));
	raft::copy(queries_matrix.data_handle(), queries, n_queries * dim,
			raft::resource::get_cuda_stream(dev_resources));

	auto index = brute_force::build(dev_resources, index_params,
			raft::make_const_mdspan(dataset.view()));

	auto neighbors = raft::make_device_matrix<int64_t, int64_t>(dev_resources,
			n_queries, k);
	auto distances = raft::make_device_matrix<T, int64_t>(dev_resources, n_queries, k);

	brute_force::search(dev_resources, search_params, index,
			raft::make_const_mdspan(queries_matrix.view()), neighbors.view(), distances.view());
	// Copy back
	raft::copy(h_labels, neighbors.data_handle(), n_queries * k,
			raft::resource::get_cuda_stream(dev_resources));
	return 0;

}

template raft::device_matrix<float, std::size_t> read_bin_dataset<float,
		std::size_t>(const raft::device_resources &dev_resources,
		const std::string &fname);

template raft::device_matrix<uint8_t, std::size_t> read_bin_dataset<uint8_t,
		std::size_t>(const raft::device_resources &dev_resources,
		const std::string &fname);

template void vamana_build_and_write<float>(
		raft::device_resources const &dev_resources,
		raft::device_matrix_view<const float, uint32_t> dataset,
		std::string out_fname, int degree, int visited_size, float max_fraction,
		int iters);

template void vamana_build_and_write<uint8_t>(
		raft::device_resources const &dev_resources,
		raft::device_matrix_view<const uint8_t, uint32_t> dataset,
		std::string out_fname, int degree, int visited_size, float max_fraction,
		int iters);

template int brute_force_gpu<float>(raft::resources& dev_resources,
		const float* h_data,
        size_t n_samples,
        size_t dim,
		size_t k,
        const float* queries,
        size_t n_queries,
		int64_t* h_labels);

