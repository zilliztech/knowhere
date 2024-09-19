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

#include <faiss/impl/HNSW.h>

namespace knowhere {

// a visitor that does nothing
struct DummyVisitor {
    using storage_idx_t = faiss::HNSW::storage_idx_t;

    inline void
    visit_level(const int level) {
        // does nothing
    }

    inline void
    visit_edge(const int level, const storage_idx_t node_from, const storage_idx_t node_to, const float distance) {
        // does nothing
    }
};

}  // namespace knowhere
