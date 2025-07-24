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

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "knowhere/bitsetview.h"

namespace knowhere {

class EmbListOffset {
 public:
    EmbListOffset(const size_t* lims, size_t rows) {
        size_t idx = 0;
        assert(lims[idx] == 0);
        assert(rows > 0);
        while (lims[idx] < rows) {
            offset.push_back(lims[idx]);
            idx++;
        }
        assert(lims[idx] == rows);
        offset.push_back(lims[idx]);
    }

    EmbListOffset(std::vector<size_t>& offset_) {
        assert(offset_.size() > 0);
        assert(offset_[0] == 0);
        offset.resize(offset_.size());
        std::memcpy(offset.data(), offset_.data(), offset_.size() * sizeof(size_t));
    }

    EmbListOffset(std::vector<uint32_t>& offset_) {
        assert(offset_.size() > 0);
        assert(offset_[0] == 0);
        offset.resize(offset_.size());
        offset.assign(offset_.begin(), offset_.end());
    }

    EmbListOffset(std::vector<size_t>&& offset_) {
        assert(offset_[0] == 0);
        offset = std::move(offset_);
    }
    // get the emb_list id of the i-th vector
    size_t
    get_el_id(size_t vid) const {
        auto it = std::upper_bound(offset.begin(), offset.end(), vid);
        return std::distance(offset.begin(), it) - 1;
    }

    std::vector<int64_t>
    get_vids(size_t el_id) const {
        assert(el_id < offset.size() - 1);
        std::vector<int64_t> vids(offset[el_id + 1] - offset[el_id]);
        std::iota(vids.begin(), vids.end(), (int64_t)offset[el_id]);
        return vids;
    }

    inline size_t
    num_el() const {
        assert(offset.size() > 0);
        return offset.size() - 1;
    }

    std::vector<size_t> offset;
};

inline std::vector<size_t>
convert_lims_to_vector(const size_t* lims, size_t rows) {
    std::vector<size_t> offset;
    size_t idx = 0;
    assert(lims[idx] == 0);
    assert(rows > 0);
    while (lims[idx] <= rows) {
        offset.push_back(lims[idx]);
        idx++;
    }
    assert(lims[idx - 1] == rows);
    return offset;
}

inline std::optional<float>
find_max_in_range(const float* dists, int start_idx, int end_idx) {
    float max_v = std::numeric_limits<float>::lowest();
    if (start_idx < 0 || end_idx <= start_idx) {
        LOG_KNOWHERE_WARNING_ << "invalid range, start_idx: " << start_idx << ", end_idx: " << end_idx;
        return std::nullopt;
    }
    for (auto i = start_idx; i < end_idx; ++i) {
        if (dists[i] > max_v) {
            max_v = dists[i];
        }
    }
    return max_v;
}

inline std::optional<float>
get_sum_max_sim(const float* dists, const size_t nq, const size_t el_len) {
    float score = 0.0f;
    for (size_t i = 0; i < nq; i++) {
        auto max_v = find_max_in_range(dists, i * el_len, (i + 1) * el_len);
        if (max_v.has_value()) {
            score += max_v.value();
        } else {
            return std::nullopt;
        }
    }
    return score;
}

inline std::optional<float>
get_ordered_sum_max_sim(const float* dists, const size_t nq, const size_t el_len) {
    if (nq == 0 || el_len == 0) {
        LOG_KNOWHERE_WARNING_ << "invalid nq or el_len, nq: " << nq << ", el_len: " << el_len;
        return std::nullopt;
    }
    std::vector<float> scores(el_len, 0.0f);
    for (size_t i = 0; i < nq; i++) {
        scores[0] += dists[i * el_len];
        for (size_t j = 1; j < el_len; j++) {
            scores[j] = std::max(scores[j - 1], scores[j] + dists[i * el_len + j]);
        }
    }
    return scores[el_len - 1];
}

}  // namespace knowhere
