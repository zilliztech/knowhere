// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef ADAPTIVE_STORE_H
#define ADAPTIVE_STORE_H

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "knowhere/array_store.h"
#include "knowhere/mmap.h"

namespace knowhere {

template <typename T>
class AdaptiveStore {
 public:
    static_assert(std::is_integral_v<T>, "AdaptiveStore requires an integral value type");
    static_assert(std::is_signed_v<T>, "AdaptiveStore uses -1 as the invalid value");

    using ArrayType = typename ArrayStore<T>::Type;

    AdaptiveStore() = default;

    // Lookup table with dense-array or sparse-map storage. Sealed heap storage
    // may use sparse mode; mmap and growing storage use dense arrays.
    void
    SetType(ArrayType type) {
        type_ = type;
        dense_values_.SetType(type_);
    }

    void
    SetMmapFilePathGenerator(MmapFilePathGenerator mmap_file_paths) {
        mmap_file_paths_ = std::move(mmap_file_paths);
    }

    void
    Set(const T* keys, size_t value_count, size_t key_count) {
        if (value_count != 0 && keys == nullptr) {
            throw std::runtime_error("adaptive store keys are null");
        }
        Clear();
        dense_values_.SetType(type_);
        // keys[value] is the lookup key for dense value "value".
        use_sparse_ = type_ == ArrayType::ARRAY && mmap_file_paths_.empty() && value_count != 0 && key_count != 0 &&
                      static_cast<double>(value_count) / static_cast<double>(key_count) <= kSparseMaxFillRatio;

        if (use_sparse_) {
            sparse_values_.reserve(value_count);
            for (size_t value = 0; value < value_count; ++value) {
                const auto key = keys[value];
                CheckKey(key, 0, key_count);
                CheckValue(value);
                const auto mapped_value = static_cast<T>(value);
                auto insert_result = sparse_values_.emplace(key, mapped_value);
                if (!insert_result.second) {
                    throw std::runtime_error("adaptive store contains duplicate key");
                }
            }
            return;
        }

        std::vector<T> values(key_count, kInvalidValue);
        FillValues(keys, value_count, 0, key_count, 0, values);
        if (type_ == ArrayType::ARRAY) {
            dense_values_.Set(values.data(), values.size(), NextArrayMmapFilePath());
            return;
        }
        dense_values_.Append(values.data(), values.size());
    }

    T
    Get(T key) const {
        if (key < 0) {
            return kInvalidValue;
        }
        if (use_sparse_) {
            auto iter = sparse_values_.find(key);
            return iter == sparse_values_.end() ? kInvalidValue : iter->second;
        }
        const auto offset = static_cast<size_t>(key);
        return offset < dense_values_.size() ? dense_values_[offset] : kInvalidValue;
    }

    void
    Append(const T* keys, size_t value_count, size_t key_begin, size_t key_count, size_t value_begin) {
        if (value_count != 0 && keys == nullptr) {
            throw std::runtime_error("adaptive store keys are null");
        }
        if (type_ != ArrayType::APPEND_ARRAY || use_sparse_) {
            throw std::runtime_error("adaptive store append is only supported by append dense storage");
        }
        // Append extends one contiguous key range.
        const auto key_end = key_begin + key_count;
        if (dense_values_.size() != key_begin) {
            throw std::runtime_error("adaptive store append key range is not contiguous");
        }

        std::vector<T> values(key_count, kInvalidValue);
        FillValues(keys, value_count, key_begin, key_end, value_begin, values);
        dense_values_.Append(values.data(), values.size());
    }

    void
    Clear() {
        dense_values_.Clear();
        sparse_values_ = {};
        use_sparse_ = false;
    }

 private:
    static constexpr T kInvalidValue = static_cast<T>(-1);
    static constexpr double kSparseMaxFillRatio = 0.10;

    static void
    CheckKey(T key, size_t key_begin, size_t key_end) {
        if (key < 0) {
            throw std::runtime_error("adaptive store key is negative");
        }
        const auto key_offset = static_cast<size_t>(key);
        if (key_offset < key_begin || key_offset >= key_end) {
            throw std::runtime_error("adaptive store key is out of range");
        }
    }

    static void
    CheckValue(size_t value) {
        if (value > static_cast<size_t>(std::numeric_limits<T>::max())) {
            throw std::runtime_error("adaptive store value overflows");
        }
    }

    static void
    FillValues(const T* keys, size_t value_count, size_t key_begin, size_t key_end, size_t value_begin,
               std::vector<T>& values) {
        for (size_t value = 0; value < value_count; ++value) {
            const auto key = keys[value];
            CheckKey(key, key_begin, key_end);
            CheckValue(value_begin + value);
            const auto key_offset = static_cast<size_t>(key);
            const auto local_offset = key_offset - key_begin;
            if (values[local_offset] != kInvalidValue) {
                throw std::runtime_error("adaptive store contains duplicate key");
            }
            values[local_offset] = static_cast<T>(value_begin + value);
        }
    }

    std::string
    NextArrayMmapFilePath() {
        return mmap_file_paths_.Next(this);
    }

    ArrayStore<T> dense_values_;
    std::unordered_map<T, T> sparse_values_;
    bool use_sparse_ = false;
    ArrayType type_ = ArrayType::ARRAY;
    MmapFilePathGenerator mmap_file_paths_;
};

}  // namespace knowhere

#endif /* ADAPTIVE_STORE_H */
