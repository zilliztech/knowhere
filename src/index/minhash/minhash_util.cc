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
namespace knowhere {
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

inline float
minhash_lsh_hit(const char* x, const char* y, size_t size, size_t mh_lsh_band) {
    size_t r = size / (mh_lsh_band);
    for (size_t i = 0; i < mh_lsh_band; i++) {
        const char* x_b = x + r * i;
        const char* y_b = y + r * i;
        if (std::memcmp(x_b, y_b, r) == 0) {
            return 1.0f;
        }
    }
    return 0.0f;
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
        const char* xb_0 = base + idx0 * vec_size;
        const char* xb_1 = base + idx1 * vec_size;
        const char* xb_2 = base + idx2 * vec_size;
        const char* xb_3 = base + idx3 * vec_size;
        dist4(q, xb_0, xb_1, xb_2, xb_3, element_length, element_size, dis0, dis1, dis2, dis3);
    }
    float
    symmetric_dis(idx_t i, idx_t j) override {
        return dist1(base + i * vec_size, base + j * vec_size, element_length, element_size);
    }
};
}  // namespace

void
minhash_lsh_hit_ny(const char* x, const char* y, size_t dim, size_t mh_lsh_band, size_t ny, size_t topk,
                   const BitsetView& bitset, float* vals, int64_t* ids) {
    MinHashLSHResultHandler res(ids, vals, topk);
    for (size_t i = 0; i < ny; i++) {
        if (bitset.empty() || !bitset.test(i)) {
            res.push(i, minhash_lsh_hit(x, y + dim * i, dim, mh_lsh_band));
            if (res.full()) {
                break;
            }
        }
    }
}

void
minhash_jaccard_knn_ny(const char* x, const char* y, size_t length, size_t element_size, size_t ny, size_t topk,
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
}

void
minhash_jaccard_knn_ny_by_ids(const char* x, const char* y, const int64_t* sel_ids, size_t length, size_t element_size,
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
}  // namespace knowhere
