//  Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not
//  use this file except in compliance with the License. You may obtain a copy
//  of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
//  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
//  License for the specific language governing permissions and limitations
//  under the License
#include "knowhere/kmeans.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "faiss/utils/distances.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/dataset.h"
#include "knowhere/log.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"
#include "simd/hook.h"

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */
int
sgemm_(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n, FINTEGER* k, const float* alpha,
       const float* a, FINTEGER* lda, const float* b, FINTEGER* ldb, float* beta, float* c, FINTEGER* ldc);

void
openblas_set_num_threads(int num_threads);
}

namespace knowhere::kmeans {

template <typename VecT>
void
KMeans<VecT>::exhaustive_L2sqr_blas(const VecT* x, const VecT* y, size_t d, size_t nx, size_t ny, uint32_t* ids,
                                    float* val) {
    static_assert(std::is_same_v<VecT, float>, "sgemm only support float now");
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return;

    // inside blas call only use one thread, it is more efficient to parallel outside
    openblas_set_num_threads(1);
    /* block sizes */
    const size_t bs_x = faiss::distance_compute_blas_query_bs;
    const size_t bs_y = faiss::distance_compute_blas_database_bs;

    std::unique_ptr<float[]> ip_block = std::make_unique<float[]>(bs_x * bs_y);
    std::unique_ptr<float[]> x_norms = std::make_unique<float[]>(nx);
    std::unique_ptr<float[]> y_norms = std::make_unique<float[]>(ny);

    for (size_t i = 0; i < nx; i++) {
        x_norms[i] = faiss::fvec_norm_L2sqr(x + i * d, d);
    }

    for (size_t i = 0; i < ny; i++) {
        y_norms[i] = faiss::fvec_norm_L2sqr(y + i * d, d);
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose", "Not transpose", &nyi, &nxi, &di, &one, y + j0 * d, &di, x + i0 * d, &di, &zero,
                       ip_block.get(), &nyi);
            }
            for (size_t i = i0; i < i1; i++) {
                float* ip_line = ip_block.get() + (i - i0) * (j1 - j0);

                for (size_t j = j0; j < j1; j++) {
                    float ip = *ip_line;
                    float dis = x_norms[i] + y_norms[j] - 2 * ip;

                    // negative values can occur for identical vectors
                    // due to roundoff errors
                    if (dis < 0)
                        dis = 0;

                    *ip_line = dis;
                    ip_line++;
                    if (j == 0) {
                        ids[i] = j;
                        val[i] = dis;
                    } else if (dis < val[i]) {
                        ids[i] = j;
                        val[i] = dis;
                    }
                }
            }
        }
    }
}

template <typename VecT>
void
KMeans<VecT>::elkan_L2(const VecT* x, const VecT* y, size_t d, size_t nx, size_t ny, uint32_t* ids, float* val) {
    if (nx == 0 || ny == 0) {
        return;
    }
    const size_t bs_y = 256;
    auto data = std::make_unique<float[]>(bs_y * (bs_y - 1) / 2);

    for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
        size_t j1 = j0 + bs_y;
        if (j1 > ny) {
            j1 = ny;
        }

        auto Y = [&](size_t i, size_t j) -> float& {
            assert(i != j);
            i -= j0, j -= j0;
            return (i > j) ? data[j + i * (i - 1) / 2] : data[i + j * (j - 1) / 2];
        };
        for (size_t i = j0 + 1; i < j1; ++i) {
            const VecT* y_i = y + i * d;
            for (size_t j = j0; j < i; j++) {
                const VecT* y_j = y + j * d;
                Y(i, j) = faiss::fvec_L2sqr(y_i, y_j, d);
            }
        }

        for (size_t i = 0; i < nx; i++) {
            const VecT* x_i = x + i * d;

            int64_t ids_i = j0;
            float val_i = faiss::fvec_L2sqr(x_i, y + j0 * d, d);
            float val_i_time_4 = val_i * 4;
            for (size_t j = j0 + 1; j < j1; j++) {
                if (val_i_time_4 <= Y(ids_i, j)) {
                    continue;
                }
                const VecT* y_j = y + j * d;
                float disij = faiss::fvec_L2sqr(x_i, y_j, d / 2);
                if (disij >= val_i) {
                    continue;
                }
                disij += faiss::fvec_L2sqr(x_i + d / 2, y_j + d / 2, d - d / 2);
                if (disij < val_i) {
                    ids_i = j;
                    val_i = disij;
                    val_i_time_4 = val_i * 4;
                }
            }

            if (j0 == 0 || val[i] > val_i) {
                val[i] = val_i;
                ids[i] = ids_i;
            }
        }
    }
}

template <typename VecT>
void
KMeans<VecT>::fit(const VecT* vecs, size_t n, size_t max_iter, uint32_t random_state, std::string_view init,
                  std::string_view algorithm) {
    centroids_ = std::make_unique<VecT[]>(n_centroids_ * dim_);
    knowhere::TimeRecorder build_time("Kmeans cost", 2);

    if (init == "random") {
        initRandom(vecs, n, random_state);
    } else if (init == "kmeans++") {
        initKMeanspp(vecs, n, random_state);
    } else {
        throw std::runtime_error(std::string("Init method: ") + std::string(init) + " not supported yet.");
    }
    LOG_KNOWHERE_INFO_ << " n_centroids: " << n_centroids_ << " dim: " << dim_;

    float old_loss = std::numeric_limits<float>::max();
    std::vector<std::vector<uint32_t>> closest_docs(n_centroids_);
    centroid_id_mapping_ = std::make_unique<uint32_t[]>(n);
    auto closest_centroid_distance = std::make_unique<float[]>(n);

    for (size_t iter = 1; iter <= max_iter; ++iter) {
        if (algorithm == "lloyd") {
            auto loss = lloyds_iter(vecs, closest_docs, centroid_id_mapping_.get(), closest_centroid_distance.get(), n,
                                    random_state, verbose_);

            if (verbose_) {
                LOG_KNOWHERE_INFO_ << "Iter [" << iter << "/" << max_iter << "], loss: " << loss;
            }
            if (verbose_ &&
                ((loss < std::numeric_limits<float>::epsilon()) || ((iter != 1) && ((old_loss - loss) / loss) < 0))) {
                LOG_KNOWHERE_INFO_ << "Residuals unchanged: " << old_loss << " becomes " << loss
                                   << ". Early termination.";
                break;
            }
            old_loss = loss;
        } else {
            throw std::runtime_error(std::string("Algorithm: ") + std::string(algorithm) + " not supported yet.");
        }
    }
    build_time.RecordSection("total iteration");
}

template <typename VecT>
void
KMeans<VecT>::initRandom(const VecT* train_data, size_t n_train, uint32_t random_state) {
    std::unordered_set<uint32_t> picked;
    std::mt19937 rng(random_state);
    for (int64_t j = static_cast<int64_t>(n_train) - static_cast<int64_t>(n_centroids_);
         j < static_cast<int64_t>(n_train); ++j) {
        uint32_t tmp = std::uniform_int_distribution<uint32_t>(0, j)(rng);
        if (picked.count(tmp)) {
            tmp = j;
        }
        picked.insert(tmp);
        std::memcpy(centroids_.get() + (j - static_cast<int64_t>(n_train) + static_cast<int64_t>(n_centroids_)) * dim_,
                    train_data + tmp * dim_, dim_ * sizeof(VecT));
    }
}

template <typename VecT>
void
KMeans<VecT>::initKMeanspp(const VecT* train_data, size_t n_train, uint32_t random_state) {
    std::vector<size_t> picked;
    std::mt19937 rng(random_state);
    std::uniform_real_distribution<> distribution(0, 1);
    std::uniform_int_distribution<size_t> int_dist(0, n_train - 1);
    size_t init_id = int_dist(rng);
    size_t num_picked = 1;

    LOG_KNOWHERE_INFO_ << "init kmeans++ start";
    picked.push_back(init_id);
    std::memcpy(centroids_.get(), train_data + init_id * dim_, dim_ * sizeof(VecT));

    auto dist = std::make_unique<float[]>(n_train);
    faiss::fvec_L2sqr_ny(dist.get(), train_data + init_id * dim_, train_data, dim_, n_train);

    double dart_val;
    size_t tmp_pivot;
    bool sum_flag = false;

    while (num_picked < n_centroids_) {
        dart_val = distribution(rng);

        double sum = 0;
        for (size_t i = 0; i < n_train; i++) {
            sum = sum + static_cast<double>(dist[i]);
        }

        if (sum < 1e-6) {
            sum_flag = true;
        }

        dart_val *= sum;

        double prefix_sum = 0;
        for (size_t i = 0; i < n_train; i++) {
            tmp_pivot = i;
            if (dart_val >= prefix_sum && dart_val < prefix_sum + static_cast<double>(dist[i])) {
                break;
            }

            prefix_sum += static_cast<double>(dist[i]);
        }

        if (std::find(picked.begin(), picked.end(), tmp_pivot) != picked.end() && sum_flag == false) {
            continue;
        }
        picked.push_back(tmp_pivot);
        std::memcpy(centroids_.get() + num_picked * dim_, train_data + tmp_pivot * dim_, dim_ * sizeof(VecT));

        faiss::fvec_L2sqr_ny(dist.get(), train_data + init_id * dim_, train_data, dim_, n_train);
        num_picked++;
    }
    LOG_KNOWHERE_INFO_ << "init kmeans++ done.";
}

template <typename VecT>
void
KMeans<VecT>::split_clusters(std::vector<int>& hassign, size_t n_train, uint32_t random_state) {
    /* Take care of void clusters */
    size_t nsplit = 0;
    constexpr float EPS = 1.0 / 1024;
    std::mt19937 mt(random_state);
    for (size_t ci = 0; ci < n_centroids_; ci++) {
        if (hassign[ci] == 0) { /* need to redefine a centroid */
            size_t cj;
            for (cj = 0;; cj = (cj + 1) % n_centroids_) {
                /* probability to pick this cluster for split */
                float p = (hassign[cj] - 1.0) / (float)(n_train - n_centroids_);
                float r = mt() / float(mt.max());
                if (r < p) {
                    break; /* found our cluster to be split */
                }
            }
            std::memcpy(centroids_.get() + ci * dim_, centroids_.get() + cj * dim_, sizeof(VecT) * dim_);

            /* small symmetric pertubation */
            for (size_t j = 0; j < dim_; j++) {
                if (j % 2 == 0) {
                    centroids_[ci * dim_ + j] = centroids_[ci * dim_ + j] * (1 + EPS);
                    centroids_[cj * dim_ + j] = centroids_[cj * dim_ + j] * (1 - EPS);
                } else {
                    centroids_[ci * dim_ + j] = centroids_[ci * dim_ + j] * (1 - EPS);
                    centroids_[cj * dim_ + j] = centroids_[cj * dim_ + j] * (1 + EPS);
                }
            }

            /* assume even split of the cluster */
            hassign[ci] = hassign[cj] / 2;
            hassign[cj] -= hassign[ci];
            nsplit++;
            LOG_KNOWHERE_INFO_ << ci << " " << cj << " " << hassign[ci] << " " << hassign[cj];
        }
    }
    LOG_KNOWHERE_INFO_ << "there are " << nsplit << " splits";
}

template <typename VecT>
float
KMeans<VecT>::lloyds_iter(const VecT* train_data, std::vector<std::vector<uint32_t>>& closest_docs,
                          uint32_t* closest_centroid, float* closest_centroid_distance, size_t n_train,
                          uint32_t random_state, bool compute_residual) {
    float losses = 0.0;

    for (size_t c = 0; c < n_centroids_; ++c) {
        closest_docs[c].clear();
    }

    computeClosestCentroid(train_data, n_train, centroids_.get(), closest_centroid, closest_centroid_distance);
    for (size_t i = 0; i < n_train; ++i) {
        closest_docs[closest_centroid[i]].push_back(i);
    }
    std::memset((void*)centroids_.get(), 0x0, n_centroids_ * dim_ * sizeof(VecT));
    std::vector<int> hassign(n_centroids_, 0);

    for (size_t c = 0; c < n_centroids_; ++c) {
        hassign[c] = closest_docs[c].size();
        if (closest_docs[c].empty()) {
            continue;
        }
        std::vector<double> centroids_tmp(dim_, 0.0);
        for (auto i : closest_docs[c]) {
            for (size_t j = 0; j < dim_; ++j) {
                centroids_tmp[j] += double(train_data[i * dim_ + j]);
            }
        }
        for (size_t j = 0; j < dim_; ++j) {
            centroids_[c * dim_ + j] = VecT(centroids_tmp[j] / closest_docs[c].size());
        }
    }
    if (compute_residual) {
        for (size_t i = 0; i < n_train; ++i) {
            losses += closest_centroid_distance[i];
        }
    }
    split_clusters(hassign, n_train, random_state);
    return losses;
}

template <typename VecT>
void
KMeans<VecT>::computeClosestCentroid(const VecT* vecs, size_t n, const VecT* centroids, uint32_t* closest_centroid,
                                     float* closest_centroid_distance) {
    auto pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    constexpr int block_size = 8192;
    size_t block_num = DIV_ROUND_UP(n, block_size);
    futures.reserve(block_num);
    for (size_t i = 0; i < block_num; ++i) {
        size_t start = i * block_size;
        size_t end = std::min(n, (i + 1) * block_size);
        futures.emplace_back(pool->push([&, start, end]() {
            if (std::is_same_v<VecT, float>) {
                exhaustive_L2sqr_blas(vecs + start * dim_, centroids, dim_, end - start, n_centroids_,
                                      closest_centroid + start, closest_centroid_distance + start);
            } else {
                elkan_L2(vecs + start * dim_, centroids, dim_, end - start, n_centroids_, closest_centroid + start,
                         closest_centroid_distance + start);
            }
        }));
    }
    for (auto& future : futures) {
        future.wait();
    }
}

// currently only support float
template class KMeans<float>;
// template class KMeans<bf16>;
// template class KMeans<fp16>;

// train_data and num_clusters
// return centroids and centroid_id_mapping
template <typename VecT>
expected<DataSetPtr>
ClusteringMajorCompaction(const DataSet& dataset, const uint32_t num_clusters) {
    auto rows = dataset.GetRows();
    auto tensor = dataset.GetTensor();
    auto dim = dataset.GetDim();

    LOG_KNOWHERE_INFO_ << "total vector num: " << rows << " dim: " << dim;

    KMeans<VecT> kMeans(num_clusters, dim);
    kMeans.fit((const VecT*)tensor, rows);
    auto& centroids = kMeans.get_centroids();
    auto& centroid_id_mapping = kMeans.get_centroid_id_mapping();

    return GenResultDataSet(dim, centroids.release(), rows, centroid_id_mapping.release());
}

template <typename VecT>
expected<DataSetPtr>
ClusteringDataAssign(const DataSet& dataset, const VecT* centroids, const uint32_t num_clusters) {
    auto rows = dataset.GetRows();
    auto tensor = dataset.GetTensor();
    auto dim = dataset.GetDim();
    auto centroid_id_mapping = std::make_unique<uint32_t[]>(rows);
    auto closest_centroid_distance = std::make_unique<float[]>(rows);
    KMeans<VecT> kMeans(num_clusters, dim);
    kMeans.computeClosestCentroid((const VecT*)tensor, rows, centroids, centroid_id_mapping.get(),
                                  closest_centroid_distance.get());
    return GenResultDataSet(rows, centroid_id_mapping.release());
}

template expected<DataSetPtr>
ClusteringMajorCompaction<float>(const DataSet& dataset, const uint32_t num_clusters);

template expected<DataSetPtr>
ClusteringDataAssign<float>(const DataSet& dataset, const float* centroids, const uint32_t num_clusters);

}  // namespace knowhere::kmeans
