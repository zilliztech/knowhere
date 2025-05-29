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
#include "knowhere/config.h"
#include "knowhere/operands.h"
namespace knowhere {
Status
MinhashConfigCheck(const size_t dim, const DataFormatEnum data_type, const uint32_t fun_type, const BaseConfig* cfg,
                   const BitsetView* bitset);

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
        if (id == -1 || dis < 0.000001f)
            return;
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

void
minhash_lsh_hit_ny(const char* x, const char* y, size_t dim, size_t band, size_t ny, size_t topk,
                   const BitsetView& bitset, float* vals, int64_t* ids);

void
minhash_jaccard_knn_ny(const char* x, const char* y, size_t length, size_t element_size, size_t ny, size_t topk,
                       const BitsetView& bitset, float* vals, int64_t* ids);

void
minhash_jaccard_knn_ny_by_ids(const char* x, const char* y, const int64_t* sel_ids, size_t length, size_t element_size,
                              size_t sel_ids_num, size_t topk, float* res_vals, int64_t* res_ids);
}  // namespace knowhere
