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

#ifndef OBJECT_H
#define OBJECT_H

#include <atomic>
#include <cassert>
#include <iostream>

#include "knowhere/file_manager.h"

namespace knowhere {

template <typename I, typename T>
struct IdVal {
    I id;
    T val;

    IdVal() = default;
    IdVal(I id, T val) : id(id), val(val) {
    }

    inline friend bool
    operator<(const IdVal<I, T>& lhs, const IdVal<I, T>& rhs) {
        return lhs.val < rhs.val || (lhs.val == rhs.val && lhs.id < rhs.id);
    }

    inline friend bool
    operator>(const IdVal<I, T>& lhs, const IdVal<I, T>& rhs) {
        return !(lhs < rhs) && !(lhs == rhs);
    }

    inline friend bool
    operator==(const IdVal<I, T>& lhs, const IdVal<I, T>& rhs) {
        return lhs.id == rhs.id && lhs.val == rhs.val;
    }
};

class Object {
 public:
    Object() = default;
    Object(const std::nullptr_t value) {
        assert(value == nullptr);
    }
    inline uint32_t
    Ref() const {
        return ref_counts_.load(std::memory_order_relaxed);
    }
    inline void
    DecRef() {
        ref_counts_.fetch_sub(1, std::memory_order_relaxed);
    }
    inline void
    IncRef() {
        ref_counts_.fetch_add(1, std::memory_order_relaxed);
    }
    virtual ~Object() {
    }

 private:
    mutable std::atomic_uint32_t ref_counts_ = 1;
};

template <typename T>
class Pack : public Object {
    static_assert(std::is_same_v<T, std::shared_ptr<knowhere::FileManager>>,
                  "IndexPack only support std::shared_ptr<knowhere::FileManager> by far.");

 public:
    Pack() {
    }
    Pack(T package) : package_(package) {
    }
    T
    GetPack() const {
        return package_;
    }
    ~Pack() {
    }

 private:
    T package_;
};

}  // namespace knowhere
#endif /* OBJECT_H */
