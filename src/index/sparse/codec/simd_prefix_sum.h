// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
// on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
// for the specific language governing permissions and limitations under the License.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "index/sparse/codec/simd_bitpacking_kernel.h"

namespace knowhere::sparse::inverted::simd_prefix_sum {

inline void
integrate_doc_id_gaps(uint32_t* values, size_t n, uint32_t previous_value) noexcept {
    assert(values != nullptr || n == 0);
    if (n < 4) {
        for (size_t i = 0; i < n; ++i) {
            previous_value += values[i] + 1;
            values[i] = previous_value;
        }
        return;
    }
    knowhere_simd_integrate_doc_id_gaps(values, n, previous_value);
}

}  // namespace knowhere::sparse::inverted::simd_prefix_sum
