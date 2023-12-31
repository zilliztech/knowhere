// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "io/memory_io.h"

#include <cstring>

namespace knowhere {

// TODO(linxj): Get From Config File
static size_t magic_num = 2;

size_t
MemoryIOWriter::operator()(const void* ptr, size_t size, size_t nitems) {
    auto total_need = size * nitems + rp_;

    if (!data_) {  // data == nullptr
        total_ = total_need * magic_num;
        rp_ = size * nitems;
        data_ = new uint8_t[total_];
        memcpy(data_, ptr, rp_);
        return nitems;
    }

    if (total_need > total_) {
        total_ = total_need * magic_num;
        auto new_data = new uint8_t[total_];
        memcpy(new_data, data_, rp_);
        delete[] data_;
        data_ = new_data;

        memcpy((data_ + rp_), ptr, size * nitems);
        rp_ = total_need;
    } else {
        memcpy((data_ + rp_), ptr, size * nitems);
        rp_ = total_need;
    }

    return nitems;
}

size_t
MemoryIOReader::operator()(void* ptr, size_t size, size_t nitems) {
    if (rp_ >= total_) {
        return 0;
    }
    size_t nremain = (total_ - rp_) / size;
    if (nremain < nitems) {
        nitems = nremain;
    }
    memcpy(ptr, (data_ + rp_), size * nitems);
    rp_ += size * nitems;
    return nitems;
}

}  // namespace knowhere
