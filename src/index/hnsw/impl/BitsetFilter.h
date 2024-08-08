// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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

#include <faiss/MetricType.h>

#include "knowhere/bitsetview.h"

namespace knowhere {

// specialized override for knowhere
struct BitsetFilter {
    // contains disabled nodes.
    knowhere::BitsetView bitset_view;

    inline BitsetFilter(knowhere::BitsetView bitset_view_) : bitset_view{bitset_view_} {
    }

    inline bool
    allowed(const faiss::idx_t idx) const {
        // there's no check for bitset_view.empty() by design
        return !bitset_view.test(idx);
    }
};

}  // namespace knowhere
