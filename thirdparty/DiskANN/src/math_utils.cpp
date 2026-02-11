// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <limits>
#include <memory>
#include <unordered_set>
#include <malloc.h>
#include <diskann/math_utils.h>
#include "diskann/logger.h"
#include "diskann/utils.h"

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n,
           FINTEGER* k, const float* alpha, const float* a, FINTEGER* lda,
           const float* b, FINTEGER* ldb, float* beta, float* c, FINTEGER* ldc);
};

namespace math_utils {
  namespace {
    static constexpr const char* kNoTranspose = "Not transpose";
    static constexpr const char* kTranspose = "Transpose";
  };  // namespace

  float calc_distance(const float* vec_1, const float* vec_2, size_t dim) {
    float dist = 0;
    for (size_t j = 0; j < dim; j++) {
      dist += (vec_1[j] - vec_2[j]) * (vec_1[j] - vec_2[j]);
    }
    return dist;
  }

  // compute l2-squared norms of data stored in row major num_points * dim,
  // needs
  // to be pre-allocated
  void compute_vecs_l2sq(float* vecs_l2sq, const float* data, const size_t num_points,
                         const size_t dim) {
    for (int64_t n_iter = 0; n_iter < (_s64) num_points; n_iter++) {
      vecs_l2sq[n_iter] =
          calc_distance(data + (n_iter * dim), data + (n_iter * dim), dim);
    }
  }

  void elkan_L2(const float* x, const float* y, size_t d, size_t nx, size_t ny,
                uint32_t* ids) {
    if (nx == 0 || ny == 0) {
      return;
    }
    const size_t bs_y = 256;
    float* data = (float*) malloc((bs_y * (bs_y - 1) / 2) * sizeof(float));
    float* val = (float*) malloc(nx * sizeof(float));
    for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
      size_t j1 = j0 + bs_y;
      if (j1 > ny)
        j1 = ny;

      auto Y = [&](size_t i, size_t j) -> float& {
        assert(i != j);
        i -= j0, j -= j0;
        return (i > j) ? data[j + i * (i - 1) / 2] : data[i + j * (j - 1) / 2];
      };

      for (size_t i = j0 + 1; i < j1; ++i) {
        const float* y_i = y + i * d;
        for (size_t j = j0; j < i; j++) {
          const float* y_j = y + j * d;
          Y(i, j) = calc_distance(y_i, y_j, d);
        }
      }

      for (size_t i = 0; i < nx; i++) {
        const float* x_i = x + i * d;

        int64_t ids_i = j0;
        float   val_i = calc_distance(x_i, y + j0 * d, d);
        float   val_i_time_4 = val_i * 4;
        for (size_t j = j0 + 1; j < j1; j++) {
          if (val_i_time_4 <= Y(ids_i, j)) {
            continue;
          }
          const float* y_j = y + j * d;
          float        disij = calc_distance(x_i, y_j, d / 2);
          if (disij >= val_i) {
            continue;
          }
          disij += calc_distance(x_i + d / 2, y_j + d / 2, d - d / 2);
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
    free(val);
    free(data);
  }

  void rotate_data_randomly(float* data, size_t num_points, size_t dim,
                            float* rot_mat, float*& new_mat,
                            bool transpose_rot) {
    char* transpose = const_cast<char*>(kNoTranspose);
    if (transpose_rot) {
      diskann::cout << "Transposing rotation matrix.." << std::flush;
      transpose = const_cast<char*>(kTranspose);
    }
    diskann::cout << "done Rotating data with random matrix.." << std::flush;

    float    one = 1.0, zero = 0.0;
    FINTEGER m = num_points, finteger_dim = dim;
    sgemm_(kNoTranspose, transpose, &m, &finteger_dim, &finteger_dim, &one,
           data, &finteger_dim, rot_mat, &finteger_dim, &zero, new_mat,
           &finteger_dim);

    diskann::cout << "done." << std::endl;
  }

  // calculate k closest centers to data of num_points * dim (row major)
  // centers is num_centers * dim (row major)
  // data_l2sq has pre-computed squared norms of data
  // centers_l2sq has pre-computed squared norms of centers
  // pre-allocated center_index will contain id of nearest center
  // pre-allocated dist_matrix shound be num_points * num_centers and contain
  // squared distances
  // Default value of k is 1

  // Ideally used only by compute_closest_centers
  void compute_closest_centers_in_block(
      const float* const data, const size_t num_points, const size_t dim,
      const float* const centers, const size_t num_centers,
      const float* const docs_l2sq, const float* const centers_l2sq,
      uint32_t* center_index, float* const dist_matrix, size_t k) {
    if (k > num_centers) {
      diskann::cout << "ERROR: k (" << k << ") > num_center(" << num_centers
                    << ")" << std::endl;
      return;
    }

    auto ones_a = std::make_unique<float[]>(num_centers);
    auto ones_b = std::make_unique<float[]>(num_points);

    for (size_t i = 0; i < num_centers; i++) {
      ones_a[i] = 1.0;
    }
    for (size_t i = 0; i < num_points; i++) {
      ones_b[i] = 1.0;
    }

    float    one = 1, zero = 0, minus_two = -2.0f;
    FINTEGER m = num_points, n = num_centers, finteger_one = 1,
             finteger_dim = dim;
    sgemm_(kNoTranspose, kTranspose, &m, &n, &finteger_one, &one, docs_l2sq,
           &finteger_one, ones_a.get(), &finteger_one, &zero, dist_matrix, &n);

    sgemm_(kNoTranspose, kTranspose, &m, &n, &finteger_one, &one, ones_b.get(),
           &finteger_one, centers_l2sq, &finteger_one, &one, dist_matrix, &n);

    sgemm_(kNoTranspose, kTranspose, &m, &n, &finteger_dim, &minus_two, data,
           &finteger_dim, centers, &finteger_dim, &one, dist_matrix, &n);

    if (k == 1) {
      for (int64_t i = 0; i < (_s64) num_points; i++) {
        float  min = std::numeric_limits<float>::max();
        float* current = dist_matrix + (i * num_centers);
        for (size_t j = 0; j < num_centers; j++) {
          if (current[j] < min) {
            center_index[i] = (uint32_t) j;
            min = current[j];
          }
        }
      }
    } else {
      for (int64_t i = 0; i < (_s64) num_points; i++) {
        std::vector<PivotContainer> top_k_vec;
        float*                      current = dist_matrix + (i * num_centers);
        for (size_t j = 0; j < num_centers; j++) {
          top_k_vec.emplace_back(j, -current[j]);
        }
        std::nth_element(top_k_vec.begin(), top_k_vec.begin() + k - 1,
                         top_k_vec.end());
        std::sort(top_k_vec.begin(), top_k_vec.begin() + k);
        for (size_t j = 0; j < k; j++) {
          center_index[i * k + j] = top_k_vec[j].piv_id;
        }
      }
    }
  }

  // Given data in num_points * new_dim row major
  // Pivots stored in full_pivot_data as num_centers * new_dim row major
  // Calculate the k closest pivot for each point and store it in vector
  // closest_centers_ivf (row major, num_points*k) (which needs to be allocated
  // outside) Additionally, if inverted index is not null (and pre-allocated),
  // it
  // will return inverted index for each center, assuming each of the inverted
  // indices is an empty vector. Additionally, if pts_norms_squared is not null,
  // then it will assume that point norms are pre-computed and use those values

  void compute_closest_centers(const float* data, size_t num_points, size_t dim,
                               const float* pivot_data, size_t num_centers,
                               size_t k, int64_t* closest_centers_ivf,
                               std::vector<size_t>*      inverted_index,
                               float* pts_norms_squared) {
    if (k > num_centers) {
      diskann::cout << "ERROR: k (" << k << ") > num_center(" << num_centers
                    << ")" << std::endl;
      return;
    }

    bool is_norm_given_for_pts = (pts_norms_squared != NULL);

    auto pivs_norms_squared = std::make_unique<float>(num_centers);
    std::unique_ptr<float[]> pts_norms_squared_deleter = nullptr;
    if (!is_norm_given_for_pts) {
      pts_norms_squared_deleter = std::make_unique<float[]>(num_points);
      pts_norms_squared = pts_norms_squared_deleter.get();
    }

    size_t PAR_BLOCK_SIZE = num_points;
    size_t N_BLOCKS = (num_points % PAR_BLOCK_SIZE) == 0
                          ? (num_points / PAR_BLOCK_SIZE)
                          : (num_points / PAR_BLOCK_SIZE) + 1;

    if (!is_norm_given_for_pts)
      math_utils::compute_vecs_l2sq(pts_norms_squared, data, num_points, dim);
    math_utils::compute_vecs_l2sq(pivs_norms_squared.get(), pivot_data, num_centers,
                                  dim);
    auto closest_centers = std::make_unique<uint32_t[]>(PAR_BLOCK_SIZE * k);
    auto distance_matrix = std::make_unique<float[]>(num_centers * PAR_BLOCK_SIZE);

    for (size_t cur_blk = 0; cur_blk < N_BLOCKS; cur_blk++) {
      const float* data_cur_blk = data + cur_blk * PAR_BLOCK_SIZE * dim;
      size_t num_pts_blk =
          std::min(PAR_BLOCK_SIZE, num_points - cur_blk * PAR_BLOCK_SIZE);
      float* pts_norms_blk = pts_norms_squared + cur_blk * PAR_BLOCK_SIZE;

      math_utils::compute_closest_centers_in_block(
          data_cur_blk, num_pts_blk, dim, pivot_data, num_centers,
          pts_norms_blk, pivs_norms_squared.get(), closest_centers.get(),
          distance_matrix.get(), k);

      int64_t blk_st = cur_blk * PAR_BLOCK_SIZE;
      int64_t blk_ed =
          std::min((_s64) num_points, (_s64) ((cur_blk + 1) * PAR_BLOCK_SIZE));
      for (int64_t j = blk_st; j < blk_ed; j++) {
        for (size_t l = 0; l < k; l++) {
          size_t this_center_id =
              closest_centers[(j - cur_blk * PAR_BLOCK_SIZE) * k + l];
          closest_centers_ivf[j * k + l] = (int64_t) this_center_id;
        }
      }
      if (inverted_index != NULL) {
        for (size_t j = 0; j < num_pts_blk * k; ++j) {
          inverted_index[closest_centers[j]].push_back(blk_st + j / k);
        }
      }
    }
  }

  // if to_subtract is 1, will subtract nearest center from each row. Else will
  // add. Output will be in data_load iself.
  // Nearest centers need to be provided in closst_centers.
  void process_residuals(float* data_load, size_t num_points, size_t dim,
                         float* cur_pivot_data, size_t num_centers,
                         uint32_t* closest_centers, bool to_subtract) {
    diskann::cout << "Processing residuals of " << num_points << " points in "
                  << dim << " dimensions using " << num_centers << " centers "
                  << std::endl;
    for (int64_t n_iter = 0; n_iter < (_s64) num_points; n_iter++) {
      for (size_t d_iter = 0; d_iter < dim; d_iter++) {
        if (to_subtract == 1)
          data_load[n_iter * dim + d_iter] =
              data_load[n_iter * dim + d_iter] -
              cur_pivot_data[closest_centers[n_iter] * dim + d_iter];
        else
          data_load[n_iter * dim + d_iter] =
              data_load[n_iter * dim + d_iter] +
              cur_pivot_data[closest_centers[n_iter] * dim + d_iter];
      }
    }
  }

}  // namespace math_utils

namespace kmeans {

  // run Lloyds one iteration
  // Given data in row major num_points * dim, and centers in row major
  // num_centers * dim And squared lengths of data points, output the closest
  // center to each data point, update centers, and also return inverted index.
  // If
  // closest_centers == NULL, will allocate memory and return. Similarly, if
  // closest_docs == NULL, will allocate memory and return.

  float lloyds_iter(const float* data, size_t num_points, size_t dim,
                    float* centers, size_t num_centers,
                    std::unique_ptr<std::vector<size_t>[]>& closest_docs,
                    std::unique_ptr<uint32_t[]>&          closest_center) {
    bool compute_residual = true;
    // Timer timer;

    bool ret_closest_center = true;
    bool ret_closest_docs = true;
    if (closest_center == nullptr) {
      closest_center = std::make_unique<uint32_t[]>(num_points);
      ret_closest_center = false;
    }
    if (closest_docs == nullptr) {
      closest_docs = std::make_unique<std::vector<size_t>[]>(num_centers);
      ret_closest_docs = false;
    } else {
      for (size_t c = 0; c < num_centers; ++c)
        closest_docs[c].clear();
    }

    math_utils::elkan_L2(data, centers, dim, num_points, num_centers,
                         closest_center.get());
    for (size_t i = 0; i < num_points; ++i) {
      closest_docs[closest_center[i]].push_back(i);
    }

    for (int64_t c = 0; c < (_s64) num_centers; ++c) {
      float*  center = centers + (size_t) c * (size_t) dim;
      auto cluster_sum = std::make_unique<double[]>(dim);
      for (size_t i = 0; i < dim; i++)
        cluster_sum[i] = 0.0;
      for (size_t i = 0; i < closest_docs[c].size(); i++) {
        const float* current = data + ((closest_docs[c][i]) * dim);
        for (size_t j = 0; j < dim; j++) {
          cluster_sum[j] += (double) current[j];
        }
      }
      if (closest_docs[c].size() > 0) {
        for (size_t i = 0; i < dim; i++)
          center[i] =
              (float) (cluster_sum[i] / ((double) closest_docs[c].size()));
      }
    }

    float residual = 0.0;
    if (compute_residual) {
      size_t BUF_PAD = 32;
      size_t CHUNK_SIZE = 2 * 8192;
      size_t nchunks =
          num_points / CHUNK_SIZE + (num_points % CHUNK_SIZE == 0 ? 0 : 1);
      std::vector<float> residuals(nchunks * BUF_PAD, 0.0);

      for (int64_t chunk = 0; chunk < (_s64) nchunks; ++chunk)
        for (size_t d = chunk * CHUNK_SIZE;
             d < num_points && d < (chunk + 1) * CHUNK_SIZE; ++d)
          residuals[chunk * BUF_PAD] += math_utils::calc_distance(
              data + (d * dim),
              centers + (size_t) closest_center[d] * (size_t) dim, dim);

      for (size_t chunk = 0; chunk < nchunks; ++chunk)
        residual += residuals[chunk * BUF_PAD];
    }

    if (!ret_closest_docs) {
      closest_docs.reset();
    }
    if (!ret_closest_center) {
      closest_center.reset();
    }
    return residual;
  }

  // Run Lloyds until max_reps or stopping criterion
  // If you pass NULL for closest_docs and closest_center, it will NOT return
  // the
  // results, else it will assume appriate allocation as closest_docs = new
  // vector<size_t> [num_centers], and closest_center = new size_t[num_points]
  // Final centers are output in centers as row major num_centers * dim
  //
  float run_lloyds(const float* data, size_t num_points, size_t dim,
                   float* centers, const size_t num_centers,
                   const size_t max_reps, std::vector<size_t>* closest_docs,
                   uint32_t* closest_center) {
    float residual = std::numeric_limits<float>::max();

    std::unique_ptr<std::vector<size_t>[]> closest_docs_deleter = nullptr;
    std::unique_ptr<uint32_t[]> closest_center_deleter = nullptr;
    if (closest_docs == nullptr) {
      closest_docs_deleter = std::make_unique<std::vector<size_t>[]>(num_centers);
      closest_docs = closest_docs_deleter.get();
    }
    if (closest_center == nullptr) {
      closest_center_deleter  = std::make_unique<uint32_t[]>(num_points);
      closest_center = closest_center_deleter.get();
    }

    float old_residual;
    // Timer timer;
    for (size_t i = 0; i < max_reps; ++i) {
      old_residual = residual;

      residual = lloyds_iter(data, num_points, dim, centers, num_centers,
                             closest_docs_deleter, closest_center_deleter);

      LOG_KNOWHERE_DEBUG_ << "Lloyd's iter " << i
                          << "  dist_sq residual: " << residual;

      if (((i != 0) && ((old_residual - residual) / residual) < 0.00001) ||
          (residual < std::numeric_limits<float>::epsilon())) {
        LOG_KNOWHERE_DEBUG_ << "Residuals unchanged: " << old_residual
                            << " becomes " << residual
                            << ". Early termination.";
        break;
      }
    }
    return residual;
  }

  // assumes memory allocated for pivot_data as new
  // float[num_centers*dim]
  // and select randomly num_centers points as pivots
  void selecting_pivots(const float* data, size_t num_points, size_t dim,
                        float* pivot_data, size_t num_centers) {
    std::unordered_set<size_t> picked;
    std::random_device         rd;
    auto                       x = rd();
    LOG_KNOWHERE_DEBUG_ << "Selecting " << num_centers << " pivots from "
                        << num_points << " points using "
                        << "random seed " << x;
    std::mt19937                          generator(x);
    std::uniform_int_distribution<size_t> distribution(0, num_points - 1);

    size_t tmp_pivot;
    for (size_t j = num_points - num_centers; j < num_points; j++) {
      tmp_pivot = std::uniform_int_distribution<size_t>(0, j)(generator);
      if (picked.count(tmp_pivot)) {
        tmp_pivot = j;
      }
      picked.insert(tmp_pivot);
      std::memcpy(pivot_data + (j - num_points + num_centers) * dim,
                  data + tmp_pivot * dim, dim * sizeof(float));
    }
  }

  void kmeanspp_selecting_pivots(const float* data, size_t num_points, size_t dim,
                                 float* pivot_data, size_t num_centers) {
    if (num_points > 1 << 23) {
      diskann::cout << "ERROR: n_pts " << num_points
                    << " currently not supported for k-means++, maximum is "
                       "8388608. Falling back to random pivot "
                       "selection."
                    << std::endl;
      selecting_pivots(data, num_points, dim, pivot_data, num_centers);
      return;
    }

    std::stringstream   stream;
    std::vector<size_t> picked;
    std::random_device  rd;
    auto                x = rd();

    stream << "Selecting " << num_centers << " pivots from " << num_points
           << " points using "
           << "random seed " << x;

    std::mt19937                          generator(x);
    std::uniform_real_distribution<>      distribution(0, 1);
    std::uniform_int_distribution<size_t> int_dist(0, num_points - 1);
    size_t                                init_id = int_dist(generator);
    size_t                                num_picked = 1;

    auto dist = std::make_unique<float[]>(num_points);

    for (int64_t i = 0; i < (_s64) num_points; i++) {
      dist[i] =
          math_utils::calc_distance(data + i * dim, data + init_id * dim, dim);
      if (std::isinf(dist[i])) {
        selecting_pivots(data, num_points, dim, pivot_data, num_centers);
        return;
      }
    }

    picked.push_back(init_id);
    std::memcpy(pivot_data, data + init_id * dim, dim * sizeof(float));

    double dart_val;
    size_t tmp_pivot;
    bool   sum_flag = false;

    while (num_picked < num_centers) {
      dart_val = distribution(generator);

      double sum = 0;
      for (size_t i = 0; i < num_points; i++) {
        sum = sum + dist[i];
      }
      if (sum == 0)
        sum_flag = true;

      dart_val *= sum;

      double prefix_sum = 0;
      for (size_t i = 0; i < (num_points); i++) {
        tmp_pivot = i;
        if (dart_val >= prefix_sum && dart_val < prefix_sum + dist[i]) {
          break;
        }

        prefix_sum += dist[i];
      }

      if (std::find(picked.begin(), picked.end(), tmp_pivot) != picked.end() &&
          (sum_flag == false))
        continue;
      picked.push_back(tmp_pivot);
      std::memcpy(pivot_data + num_picked * dim, data + tmp_pivot * dim,
                  dim * sizeof(float));

      for (int64_t i = 0; i < (_s64) num_points; i++) {
        dist[i] = (std::min) (dist[i],
                              math_utils::calc_distance(
                                  data + i * dim, data + tmp_pivot * dim, dim));
      }
      num_picked++;
      if (num_picked % 32 == 0)
        stream << "." << std::flush;
    }
    stream << "done.";
    LOG_KNOWHERE_DEBUG_ << stream.str();
  }

}  // namespace kmeans
