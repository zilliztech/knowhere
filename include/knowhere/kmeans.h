// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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

#include <cstdlib>
#include <fstream>
#include <string_view>
#include <vector>

#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/file_manager.h"

namespace knowhere::kmeans {
template <typename VecT = float>
class KMeans {
 public:
    KMeans(size_t K, size_t dim, bool verbose = true) : dim_(dim), n_centroids_(K), verbose_(verbose) {
    }

    void
    fit(const VecT* vecs, size_t n, size_t max_iter = 10, uint32_t random_state = 0, std::string_view init = "random",
        std::string_view algorithm = "lloyd");

    void
    computeClosestCentroid(const VecT* vecs, size_t n, const VecT* centroids, uint32_t* closest_centroid,
                           float* closest_centroid_distance);

    std::unique_ptr<VecT[]>&
    get_centroids() {
        return centroids_;
    }

    std::unique_ptr<uint32_t[]>&
    get_centroid_id_mapping() {
        return centroid_id_mapping_;
    }

    ~KMeans() {
    }

 private:
    size_t dim_, n_centroids_;
    std::unique_ptr<VecT[]> centroids_;
    std::unique_ptr<uint32_t[]> centroid_id_mapping_;

    bool verbose_ = true;

    void
    initRandom(const VecT* train_data, size_t n_train, uint32_t random_state);

    void
    initKMeanspp(const VecT* train_data, size_t n_train, uint32_t random_state);

    float
    lloyds_iter(const VecT* train_data, std::vector<std::vector<uint32_t>>& closest_docs, uint32_t* closest_centroids,
                float* closest_centroid_distancessize_t, size_t n_train, uint32_t random_state,
                bool compute_residual = false);

    void
    split_clusters(std::vector<int>& hassign, size_t n_train, uint32_t random_state);

    void
    elkan_L2(const VecT* x, const VecT* y, size_t d, size_t nx, size_t ny, uint32_t* ids, float* val);

    void
    exhaustive_L2sqr_blas(const VecT* x, const VecT* y, size_t d, size_t nx, size_t ny, uint32_t* ids, float* val);
};

template <typename VecT>
expected<DataSetPtr>
ClusteringMajorCompaction(const DataSet& dataset, const uint32_t num_clusters);

template <typename VecT>
expected<DataSetPtr>
ClusteringDataAssign(const DataSet& dataset, const VecT* centroids, const uint32_t num_clusters);

}  // namespace knowhere::kmeans
