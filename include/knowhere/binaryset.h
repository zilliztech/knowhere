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

#ifndef BINARYSET_H
#define BINARYSET_H

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace knowhere {

struct Binary {
    std::shared_ptr<uint8_t[]> data;
    int64_t size = 0;
};
using BinaryPtr = std::shared_ptr<Binary>;

inline uint8_t*
CopyBinary(const BinaryPtr& bin) {
    uint8_t* newdata = new uint8_t[bin->size];
    std::memcpy(newdata, bin->data.get(), bin->size);
    return newdata;
}

class BinarySet {
 public:
    BinaryPtr
    GetByName(const std::string& name) const {
        if (Contains(name)) {
            return binary_map_.at(name);
        }
        return nullptr;
    }

    // This API is used to be compatible with knowhere-1.x.
    // It tries each key name one by one, and returns the first matched.
    BinaryPtr
    GetByNames(const std::vector<std::string>& names) const {
        for (auto& name : names) {
            if (Contains(name)) {
                return binary_map_.at(name);
            }
        }
        return nullptr;
    }

    void
    Append(const std::string& name, BinaryPtr binary) {
        binary_map_[name] = std::move(binary);
    }

    void
    Append(const std::string& name, std::shared_ptr<uint8_t[]> data, int64_t size) {
        auto binary = std::make_shared<Binary>();
        binary->data = data;
        binary->size = size;
        binary_map_[name] = std::move(binary);
    }

    BinaryPtr
    Erase(const std::string& name) {
        BinaryPtr result = nullptr;
        auto it = binary_map_.find(name);
        if (it != binary_map_.end()) {
            result = it->second;
            binary_map_.erase(it);
        }
        return result;
    }

    void
    clear() {
        binary_map_.clear();
    }

    bool
    Contains(const std::string& key) const {
        return binary_map_.find(key) != binary_map_.end();
    }

    // Return the total size of all binary data in binary set.
    size_t
    Size() const {
        size_t size = 0;
        for (auto& pair : binary_map_) {
            if (pair.second != nullptr) {
                size += pair.second->size;
            }
        }
        return size;
    }

 public:
    std::map<std::string, BinaryPtr> binary_map_;
};

using BinarySetPtr = std::shared_ptr<BinarySet>;
}  // namespace knowhere

#endif /* BINARYSET_H */
