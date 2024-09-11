// Copyright (C) 2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef DEVICE_BITSET_H
#define DEVICE_BITSET_H

#include "knowhere/bitsetview.h"
#include "raft/core/device_mdarray.hpp"
#include "raft/core/device_resources.hpp"
#include "raft/util/cudart_utils.hpp"

namespace knowhere {

struct DeviceBitsetView {
    __device__ __host__
    DeviceBitsetView(const DeviceBitsetView& other)
        : bits_{other.data()}, num_bits_{other.size()} {
    }
    __device__ __host__
    DeviceBitsetView(const uint8_t* data, size_t num_bits = size_t{})
        : bits_{data}, num_bits_{num_bits} {
    }

    __device__ __host__ bool
    empty() const {
        return num_bits_ == 0;
    }

    __device__ __host__ size_t
    size() const {
        return num_bits_;
    }

    __device__ __host__ size_t
    byte_size() const {
        return (num_bits_ + 8 - 1) >> 3;
    }

    __device__ __host__ const uint8_t*
    data() const {
        return bits_;
    }

    __device__ bool
    test(int64_t index) const {
        auto result = false;
        if (index < num_bits_) {
            result = bits_[index >> 3] & (0x1 << (index & 0x7));
        }
        return result;
    }

 private:
    const uint8_t* bits_ = nullptr;
    size_t num_bits_ = 0;
};

struct DeviceBitset {
    DeviceBitset(raft::device_resources& res, BitsetView const& other)
        : storage_{[&res, &other]() {
              auto result = raft::make_device_vector<uint8_t, uint32_t>(res, other.byte_size());
              if (!other.empty()) {
                  raft::copy(result.data_handle(), other.data(), other.byte_size(), res.get_stream());
              }
              return result;
          }()},
          num_bits_{other.size()} {
    }

    auto
    view() {
        return DeviceBitsetView{storage_.data_handle(), num_bits_};
    }

 private:
    raft::device_vector<uint8_t> storage_;
    size_t num_bits_;
};

}  // namespace knowhere

#endif /* DEVICE_BITSET_H */
