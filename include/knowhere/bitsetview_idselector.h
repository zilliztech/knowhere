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

#include <faiss/impl/IDSelector.h>

#include "knowhere/bitsetview.h"

namespace knowhere {

struct BitsetViewIDSelector final : faiss::IDSelector {
    const BitsetView bitset_view;
    const size_t id_offset;

    inline BitsetViewIDSelector(BitsetView bitset_view, const size_t offset = 0)
        : bitset_view{bitset_view}, id_offset(offset) {
    }

    inline bool
    is_member(faiss::idx_t id) const override final {
        // it is by design that bitset_view.empty() is not tested here
        return (!bitset_view.test(id + id_offset));
    }
};

struct BitsetViewWithMappingIDSelector final : faiss::IDSelector {
    const BitsetView bitset_view;
    const uint32_t* out_id_mapping;
    const size_t id_offset;

    inline BitsetViewWithMappingIDSelector(BitsetView bitset_view, const uint32_t* out_id_mapping,
                                           const size_t offset = 0)
        : bitset_view{bitset_view}, out_id_mapping(out_id_mapping), id_offset(offset) {
    }

    inline bool
    is_member(faiss::idx_t id) const override final {
        // it is by design that out_id_mapping == nullptr is not tested here
        return (!bitset_view.test(out_id_mapping[id + id_offset]));
    }
};

}  // namespace knowhere
