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

#include "faiss/utils/distances_if.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/config.h"
#include "knowhere/operands.h"
namespace knowhere::minhash {
using idx_t = faiss::idx_t;
using KeyType = uint64_t;
using ValueType = idx_t;
struct KVPair {
    KeyType Key;
    ValueType Value;
};

constexpr int kBatch = 4096;
constexpr int kQueryBatch = 64;
constexpr int kQueryBandBatch = 4;
using idx_t = faiss::idx_t;

struct MinHashLSHResultHandler {
    idx_t* ids_list_;
    float* dis_list_;
    size_t topk_;
    size_t counter_ = 0;
    MinHashLSHResultHandler(idx_t* res_ids, float* res_dis, size_t topk)
        : ids_list_(res_ids), dis_list_(res_dis), topk_(topk) {
        for (size_t i = 0; i < topk_; i++) {
            ids_list_[i] = -1;
            dis_list_[i] = 0.0f;
        }
    }
    bool
    find(const idx_t id) {
        return (id != -1) && std::find(ids_list_, ids_list_ + topk_, id) != ids_list_ + topk_;
    }
    void
    push(const idx_t id, const float dis) {
        if (this->full()) {
            return;
        }
        if (id == -1 || dis < 0.000001f) {
            return;
        }
        if (topk_ > 1 && find(id)) {
            return;
        }
        ids_list_[counter_] = id;
        dis_list_[counter_] = dis;
        counter_++;
    }
    bool
    full() {
        return topk_ == counter_;
    }
    size_t
    count() {
        return counter_;
    }
};

// get the best minhash lsh params (b, r), number of bands and size of each bands
std::pair<size_t, size_t>
OptimizeMinHashLSHParams(size_t original_dim, size_t band);

inline minhash::KeyType
GetHashKey(const char* data, size_t size /*in bytes*/, size_t band, size_t band_i) {
    const size_t r = size / band;
    auto band_i_data = data + r * band_i;
    return faiss::calculate_hash((const char*)band_i_data, r);
}

std::shared_ptr<minhash::KVPair[]>
GenHashKV(const char* data, size_t rows, size_t data_size, size_t data_element_size, size_t band_num, size_t band_size);

std::shared_ptr<minhash::KVPair[]>
GenTransposedHashKV(const char* data, size_t rows, size_t data_size, size_t data_element_size, size_t band_num,
                    size_t band_size);

void
SortHashKV(const std::shared_ptr<KVPair[]> kv_code, size_t rows, size_t band);

Status
MinhashConfigCheck(const size_t dim, const DataFormatEnum data_type, const uint32_t fun_type, const BaseConfig* cfg,
                   const BitsetView* bitset);
/**
 * @brief returning LSH band hit results on a specified number of vectors as MinHash LSH does
 *
 * @param x Pointer to query vector data containing vectors to search for
 * @param y Pointer to base dataset
 * @param size_in_bytes minhash vector size in bytes
 * @param element_size_in_bytes sizeof (minhash vector element)
 * @param mh_lsh_band Number of LSH bands, used to control the balance between recall and precision
 * @param mh_lsh_r Size of LSH band
 * @param ny Number of vectors in the base dataset
 * @param topk Number of most similar results to return
 * @param bitset Bitset view for filtering vectors that should not be searched (marked vectors will be skipped)
 * @param vals Output parameter: array to store returned similarity scores
 * @param ids Output parameter: array to store returned vector IDs
 */
void
MinHashLSHHitByNy(const char* x, const char* y, size_t size_in_bytes, size_t element_size_in_bytes, size_t mh_lsh_band,
                  size_t mh_lsh_r, size_t ny, size_t topk, const BitsetView& bitset, float* vals, int64_t* ids);
/**
 * @brief Returns the top-k vectors with smallest Jaccard distances to the query MinHash vector
 *
 * This function computes exact Jaccard similarity between MinHash vectors and returns the k most
 * similar vectors based on Jaccard distance (smaller distance means higher similarity).
 *
 * @param x Pointer to query MinHash vector data
 * @param y Pointer to base dataset
 * @param length minhash vector length in bytes
 * @param element_size Size of each MinHash element in bytes
 * @param ny Number of vectors in the base dataset
 * @param topk Number of nearest neighbors to return (top-k with smallest Jaccard distances)
 * @param bitset Bitset view for vector filtering
 * @param vals Output parameter: array to store Jaccard distances (smaller values indicate higher similarity)
 * @param ids Output parameter: array to store corresponding vector IDs
 */
void
MinHashJaccardKNNSearchByNy(const char* x, const char* y, size_t length, size_t element_size, size_t ny, size_t topk,
                            const BitsetView& bitset, float* vals, int64_t* ids);
/**
 * @brief Performs MinHash KNN search using Jaccard similarity on vectors with specified ID list
 *
 * @param x Pointer to query data
 * @param y Pointer to base dataset
 * @param sel_ids Array of specified vector IDs to search
 * @param length Data length
 * @param element_size Size of each element in bytes
 * @param sel_ids_num Number of selected IDs
 * @param topk Number of most similar results to return
 * @param res_vals Output parameter: array to store similarity scores
 * @param res_ids Output parameter: array to store corresponding result vector IDs
 */
void
MinHashJaccardKNNSearchByIDs(const char* x, const char* y, const int64_t* sel_ids, size_t length, size_t element_size,
                             size_t sel_ids_num, size_t topk, float* res_vals, int64_t* res_ids);

/**
 * @brief General MinHash vector search
 *
 * @param x Pointer to query vectors
 * @param y Pointer to base dataset vectors
 * @param size_in_bytes minhash vector size in bytes
 * @param element_size_in_bytes sizeof (minhash vector element)
 * @param mh_lsh_band Number of LSH bands for locality sensitive hashing
 * @param mh_lsh_r Size of LSH band for locality sensitive hashing
 * @param nx Number of query vectors
 * @param ny Number of base vectors
 * @param topk Number of most similar results to return for each query
 * @param mh_search_with_jaccard Flag to enable Jaccard similarity calculation in search
 * @param bitset Bitset view for filtering vectors during search
 * @param distances Output parameter: array to store similarity scores
 * @param labels Output parameter: array to store corresponding vector IDs
 * @return Status indicating success or failure of the operation
 */
Status
MinHashVecSearch(const char* x, const char* y, size_t size_in_bytes, size_t element_size_in_bytes, size_t mh_lsh_band,
                 size_t mh_lsh_r, size_t nx, size_t ny, size_t topk, bool mh_search_with_jaccard,
                 const BitsetView& bitset, float* distances, int64_t* labels);

}  // namespace knowhere::minhash
