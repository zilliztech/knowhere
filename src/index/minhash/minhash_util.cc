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
#include "index/minhash/minhash_util.h"
namespace knowhere::minhash {
namespace {
using JcaccardSim = faiss::CMin<float, idx_t>;
using DIST1FUNC = float (*)(const char* x, const char* y, size_t element_length, size_t element_size);
using DIST4FUNC = void (*)(const char* x, const char* y0, const char* y1, const char* y2, const char* y3, size_t size,
                           size_t element_size, float& dis0, float& dis1, float& dis2, float& dis3);
inline float
minhash_jaccard_native(const char* x, const char* y, size_t element_length, size_t element_size) {
    float res = 0;
    for (size_t i = 0; i < element_length; i++) {
        const char* x_b = x + element_size * i;
        const char* y_b = y + element_size * i;
        res += (std::memcmp(x_b, y_b, element_size) == 0) ? 1.0f : 0.0f;
    }
    return float(res) / float(element_length);
}

inline void
minhash_jaccard_batch_4_native(const char* x, const char* y0, const char* y1, const char* y2, const char* y3,
                               size_t size, size_t element_size, float& dis0, float& dis1, float& dis2, float& dis3) {
    dis0 = dis1 = dis2 = dis3 = 0.0f;
    for (size_t i = 0; i < size; i++) {
        const char* x_b = x + element_size * i;
        const char* y0_b = y0 + element_size * i;
        const char* y1_b = y1 + element_size * i;
        const char* y2_b = y2 + element_size * i;
        const char* y3_b = y3 + element_size * i;
        dis0 += (std::memcmp(x_b, y0_b, element_size) == 0) ? 1.0f : 0.0f;
        dis1 += (std::memcmp(x_b, y1_b, element_size) == 0) ? 1.0f : 0.0f;
        dis2 += (std::memcmp(x_b, y2_b, element_size) == 0) ? 1.0f : 0.0f;
        dis3 += (std::memcmp(x_b, y3_b, element_size) == 0) ? 1.0f : 0.0f;
    }
    dis0 /= size;
    dis1 /= size;
    dis2 /= size;
    dis3 /= size;
    return;
}

// use minhash jaccard distance
struct MinHashJaccardComputer : faiss::DistanceComputer {
    const char* base;
    const char* q;
    size_t element_length;  // minhash vector dim
    DIST1FUNC dist1;
    DIST4FUNC dist4;
    size_t element_size;  // minhash vector element size(in bytes)
    size_t vec_size;      // total minhash vector size
    MinHashJaccardComputer(const char* x, const size_t l, const size_t es)
        : base(x), element_length(l), element_size(es) {
        if (element_size == 4) {
            dist1 = faiss::u32_jaccard_distance;
            dist4 = faiss::u32_jaccard_distance_batch_4;
        } else if (element_size == 8) {
            dist1 = faiss::u64_jaccard_distance;
            dist4 = faiss::u64_jaccard_distance_batch_4;
        } else {
            dist1 = &minhash_jaccard_native;
            dist4 = &minhash_jaccard_batch_4_native;
        }
        vec_size = element_size * element_length;
    }
    void
    set_query(const float* x) override {
        q = reinterpret_cast<const char*>(x);
    }
    float
    distance_to_code(const void* x) {
        return dist1(q, (const char*)x, element_length, element_size);
    }
    float
    operator()(idx_t i) override {
        return distance_to_code(base + i * vec_size);
    }
    void
    distances_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3, float& dis0, float& dis1,
                      float& dis2, float& dis3) override {
        const char* y_0 = base + idx0 * vec_size;
        const char* y_1 = base + idx1 * vec_size;
        const char* y_2 = base + idx2 * vec_size;
        const char* y_3 = base + idx3 * vec_size;
        dist4(q, y_0, y_1, y_2, y_3, element_length, element_size, dis0, dis1, dis2, dis3);
    }
    float
    symmetric_dis(idx_t i, idx_t j) override {
        return dist1(base + i * vec_size, base + j * vec_size, element_length, element_size);
    }
};

/**
 * @brief Optimized MinHash LSH search with top1 results using hash table lookup
 *
 * Performs LSH band matching using binary search for efficient hash matching.
 * Uses multi-threading for batch processing of queries.
 */
Status
minhash_lsh_hit_with_topk1_opt_search(const char* x, const char* y, size_t u8_dim, size_t mh_lsh_band, size_t nx,
                                      size_t ny, size_t topk, const BitsetView& bitset, float* distances,
                                      int64_t* labels) {
    std::vector<minhash::KeyType> base_hash_k;
    std::vector<minhash::ValueType> base_hash_v;
    base_hash_k.reserve(ny * mh_lsh_band);
    base_hash_v.reserve(ny * mh_lsh_band);
    {
        auto base_kv = minhash::GenTransposedHashKV((const char*)y, ny, u8_dim, mh_lsh_band);
        minhash::SortHashKV(base_kv, ny, mh_lsh_band);
        for (auto i = 0; i < ny * mh_lsh_band; i++) {
            base_hash_k.emplace_back(base_kv[i].Key);
            base_hash_v.emplace_back(base_kv[i].Value);
        }
    }
    std::vector<minhash::KeyType> query_hash_k;
    query_hash_k.reserve(nx * mh_lsh_band);
    {
        auto query_kv = minhash::GenHashKV((const char*)x, nx, u8_dim, mh_lsh_band);
        for (auto i = 0; i < nx * mh_lsh_band; i++) {
            query_hash_k.emplace_back(query_kv[i].Key);
        }
    }

    std::vector<MinHashLSHResultHandler> all_res;
    all_res.reserve(nx);
    for (size_t i = 0; i < nx; i++) {
        all_res.emplace_back(labels + i * topk, distances + i * topk, topk);
    }
    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futs;
    size_t run_times = (nx + kQueryBatch - 1) / kQueryBatch;
    futs.reserve(run_times);
    for (size_t row = 0; row < run_times; ++row) {
        futs.emplace_back(pool->push([&query_hash_k, &base_hash_k, &base_hash_v, &all_res, mh_lsh_band, ny,
                                      query_beg = row * kQueryBatch,
                                      query_end = std::min((size_t)((row + 1) * kQueryBatch), (size_t)nx)]() {
            for (size_t query_id = query_beg; query_id < query_end; query_id++) {
                auto query_key = query_hash_k.data() + mh_lsh_band * query_id;
                for (auto i = 0; i < mh_lsh_band; i++) {
                    auto hit_id = faiss::u64_binary_search_eq(base_hash_k.data() + i * ny, ny, query_key[i]);
                    if (hit_id != -1) {
                        all_res[query_id].push(base_hash_v[hit_id], 1.0f);
                        while (!all_res[query_id].full() && hit_id < mh_lsh_band &&
                               base_hash_k[i * mh_lsh_band + hit_id] == query_key[i]) {
                            all_res[query_id].push(base_hash_v[hit_id], 1.0f);
                            hit_id++;
                        }
                        if (all_res[query_id].full()) {
                            break;
                        }
                    }
                }
            }
        }));
    }
    RETURN_IF_ERROR(WaitAllSuccess(futs));
    return Status::success;
}

/**
 * @brief Batch processing function for MinHash vector search
 *
 * Processes multiple query vectors in batches using either LSH band matching
 * or exact Jaccard distance computation based on configuration.
 */
Status
minhash_ny_batch_search(const char* x, const char* y, size_t u8_dim, size_t mh_lsh_band, size_t mh_element_bit_width,
                        size_t nx, size_t ny, size_t topk, bool mh_search_with_jaccard, const BitsetView& bitset,
                        float* distances, int64_t* labels) {
    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<Status>> futs;
    const int batch_size = 128;
    int num_batches = (nx + batch_size - 1) / batch_size;
    futs.reserve(num_batches);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        futs.emplace_back(pool->push([&, batch_idx] {
            ThreadPool::ScopedSearchOmpSetter setter(1);
            int start_query = batch_idx * batch_size;
            int end_query = std::min(size_t(start_query + batch_size), nx);

            for (int i = start_query; i < end_query; ++i) {
                auto cur_labels = labels + topk * i;
                auto cur_distances = distances + topk * i;

                if (mh_search_with_jaccard) {
                    size_t hash_element_size = mh_element_bit_width / 8;  // in bytes
                    size_t hash_element_length = u8_dim / hash_element_size;
                    auto cur_query = (const char*)x + u8_dim * i;
                    MinHashJaccardKNNSearchByNy(cur_query, (const char*)y, hash_element_length, hash_element_size, ny,
                                                topk, bitset, cur_distances, cur_labels);
                } else {
                    auto cur_query = (const char*)x + u8_dim * i;
                    MinHashLSHHitByNy(cur_query, (const char*)y, u8_dim, mh_lsh_band, ny, topk, bitset, cur_distances,
                                      cur_labels);
                }
            }
            return Status::success;
        }));
    }

    RETURN_IF_ERROR(WaitAllSuccess(futs));
    return Status::success;
}
}  // namespace

std::shared_ptr<minhash::KVPair[]>
GenHashKV(const char* data, size_t rows, size_t data_size, size_t band) {
    auto res_kv = std::shared_ptr<minhash::KVPair[]>(new minhash::KVPair[band * rows]);
    auto batch_num = (rows + kBatch - 1) / kBatch;
    auto build_pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    for (size_t i = 0; i < batch_num; i++) {
        futures.emplace_back(build_pool->push([&, idx = i]() {
            auto beg_id = idx * kBatch;
            auto end_id = std::min((idx + 1) * kBatch, rows);
            for (size_t j = beg_id; j < end_id; j++) {
                const char* data_j = data + data_size * j;
                size_t b = 0;
                for (; b + 4 <= band; b += 4) {
                    res_kv.get()[j * band + b] = {GetHashKey(data_j, data_size, band, b), minhash::ValueType(j)};
                    res_kv.get()[j * band + b + 1] = {GetHashKey(data_j, data_size, band, b + 1),
                                                      minhash::ValueType(j)};
                    res_kv.get()[j * band + b + 2] = {GetHashKey(data_j, data_size, band, b + 2),
                                                      minhash::ValueType(j)};
                    res_kv.get()[j * band + b + 3] = {GetHashKey(data_j, data_size, band, b + 3),
                                                      minhash::ValueType(j)};
                }
                for (; b < band; b++) {
                    minhash::KVPair kv = {GetHashKey(data_j, data_size, band, b), minhash::ValueType(j)};
                    res_kv.get()[j * band + b] = kv;
                }
            }
        }));
    }
    WaitAllSuccess(futures);
    return res_kv;
}

std::shared_ptr<minhash::KVPair[]>
GenTransposedHashKV(const char* data, size_t rows, size_t data_size, size_t band) {
    auto res_kv = std::shared_ptr<minhash::KVPair[]>(new minhash::KVPair[band * rows]);
    auto batch_num = (rows + kBatch - 1) / kBatch;
    auto build_pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    for (size_t i = 0; i < batch_num; i++) {
        futures.emplace_back(build_pool->push([&, idx = i]() {
            auto beg_id = idx * kBatch;
            auto end_id = std::min((idx + 1) * kBatch, rows);
            for (size_t j = beg_id; j < end_id; j++) {
                const char* data_j = data + data_size * j;
                size_t b = 0;
                for (; b + 4 <= band; b += 4) {
                    res_kv.get()[b * rows + j] = {GetHashKey(data_j, data_size, band, b), minhash::ValueType(j)};
                    res_kv.get()[(b + 1) * rows + j] = {GetHashKey(data_j, data_size, band, b + 1),
                                                        minhash::ValueType(j)};
                    res_kv.get()[(b + 2) * rows + j] = {GetHashKey(data_j, data_size, band, b + 2),
                                                        minhash::ValueType(j)};
                    res_kv.get()[(b + 3) * rows + j] = {GetHashKey(data_j, data_size, band, b + 3),
                                                        minhash::ValueType(j)};
                }
                for (; b < band; b++) {
                    minhash::KVPair kv = {GetHashKey(data_j, data_size, band, b), minhash::ValueType(j)};
                    res_kv.get()[b * rows + j] = kv;
                }
            }
        }));
    }
    WaitAllSuccess(futures);
    return res_kv;
}

void
SortHashKV(const std::shared_ptr<KVPair[]> kv_code, size_t rows, size_t band) {
    auto build_pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    for (size_t i = 0; i < band; i++) {
        futures.emplace_back(build_pool->push([&, idx = i]() {
            std::sort(kv_code.get() + rows * idx, kv_code.get() + rows * (idx + 1),
                      [](const KVPair& a, const KVPair& b) { return a.Key < b.Key; });
        }));
    }
    WaitAllSuccess(futures);
}

void
MinHashLSHHitByNy(const char* x, const char* y, size_t dim, size_t mh_lsh_band, size_t ny, size_t topk,
                  const BitsetView& bitset, float* vals, int64_t* ids) {
    MinHashLSHResultHandler res(ids, vals, topk);
    for (size_t i = 0; i < ny; i++) {
        if (bitset.empty() || !bitset.test(i)) {
            res.push(i, faiss::minhash_lsh_hit(x, y + dim * i, dim, mh_lsh_band));
            if (res.full()) {
                break;
            }
        }
    }
}

void
MinHashJaccardKNNSearchByNy(const char* x, const char* y, size_t length, size_t element_size, size_t ny, size_t topk,
                            const BitsetView& bitset, float* vals, int64_t* ids) {
    // init
    for (size_t i = 0; i < topk; i++) {
        vals[i] = 0.0f;
        ids[i] = -1;
    }
    auto computer = std::make_shared<MinHashJaccardComputer>(y, length, element_size);
    computer->set_query((const float*)x);
    auto filter = [&](const size_t j) { return (bitset.empty() || !bitset.test(j)); };

    // the lambda that applies a valid element.
    auto apply = [&](const float dis_in, const size_t j) {
        if (JcaccardSim::cmp(vals[0], dis_in)) {
            faiss::heap_replace_top<JcaccardSim>(topk, vals, ids, dis_in, j);
        }
    };
    faiss::distance_compute_if(ny, computer.get(), filter, apply);
    faiss::heap_reorder<JcaccardSim>(topk, vals, ids);
}

void
MinHashJaccardKNNSearchByIDs(const char* x, const char* y, const int64_t* sel_ids, size_t length, size_t element_size,
                             size_t sel_ids_num, size_t topk, float* res_vals, int64_t* res_ids) {
    // init
    for (size_t i = 0; i < topk; i++) {
        res_vals[i] = 0.0;
        res_ids[i] = -1;
    }
    auto apply = [&](const float dis_in, const size_t j) {
        if (JcaccardSim::cmp(res_vals[0], dis_in)) {
            faiss::heap_replace_top<JcaccardSim>(topk, res_vals, res_ids, dis_in, j);
        }
    };
    auto computer = std::make_shared<MinHashJaccardComputer>(y, length, element_size);
    computer->set_query((const float*)x);
    size_t i = 0;
    float dis0, dis1, dis2, dis3;
    for (; i + 4 < sel_ids_num; i += 4) {
        computer->distances_batch_4(sel_ids[i], sel_ids[i + 1], sel_ids[i + 2], sel_ids[i + 3], dis0, dis1, dis2, dis3);
        apply(dis0, sel_ids[i]);
        apply(dis1, sel_ids[i + 1]);
        apply(dis2, sel_ids[i + 2]);
        apply(dis3, sel_ids[i + 3]);
    }
    while (i < sel_ids_num) {
        auto dis = computer->operator()(sel_ids[i]);
        apply(dis, sel_ids[i]);
        i++;
    }
    faiss::heap_reorder<JcaccardSim>(topk, res_vals, res_ids);
}

Status
MinHashVecSearch(const char* x, const char* y, size_t u8_dim, size_t mh_lsh_band, size_t mh_element_bit_width,
                 size_t nx, size_t ny, size_t topk, bool mh_search_with_jaccard, const BitsetView& bitset,
                 float* distances, int64_t* labels) {
    if (topk == 1 && !mh_search_with_jaccard) {  // most common case : topk = 1 and mh_search_with_jaccard == false
        return minhash_lsh_hit_with_topk1_opt_search(x, y, u8_dim, mh_lsh_band, nx, ny, topk, bitset, distances,
                                                     labels);
    } else {
        return minhash_ny_batch_search(x, y, u8_dim, mh_lsh_band, mh_element_bit_width, nx, ny, topk,
                                       mh_search_with_jaccard, bitset, distances, labels);
    }
}

Status
MinhashConfigCheck(const size_t dim, const DataFormatEnum data_type, const uint32_t fun_type, const BaseConfig* cfg,
                   const BitsetView* bitset) {
    if (dim % 8 != 0) {
        LOG_KNOWHERE_ERROR_ << "binary vector dim should be divisible by 8.";
        return Status::invalid_metric_type;
    }
    if (data_type != DataFormatEnum::bin1) {
        LOG_KNOWHERE_ERROR_ << "Metric MH_JACCARD only support fp32.";
        return Status::invalid_metric_type;
    }
    uint32_t invalid_type = ~(PARAM_TYPE::TRAIN | PARAM_TYPE::SEARCH);
    if ((fun_type & invalid_type) != 0) {
        LOG_KNOWHERE_ERROR_ << "Metric MH_JACCARD only support search and train.";
        return Status::not_implemented;
    }
    if (fun_type & PARAM_TYPE::TRAIN) {
        size_t mh_d = cfg->mh_lsh_band.value();
        if (dim % mh_d != 0) {
            LOG_KNOWHERE_ERROR_ << "Metric MH_JACCARD not supported for dim % mh_lsh_band != 0.";
            return Status::not_implemented;
        }
    }
    return Status::success;
}
}  // namespace knowhere::minhash
