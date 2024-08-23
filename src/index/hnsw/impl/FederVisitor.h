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

#include "knowhere/feder/HNSW.h"

namespace knowhere {

// a default feder visitor
struct FederVisitor {
    using storage_idx_t = faiss::HNSW::storage_idx_t;

    // a non-owning pointer
    knowhere::feder::hnsw::FederResult* feder = nullptr;

    inline FederVisitor(knowhere::feder::hnsw::FederResult* const feder_v) : feder{feder_v} {
    }

    //
    inline void
    visit_level(const int level) {
        if (feder != nullptr) {
            feder->visit_info_.AddLevelVisitRecord(level);
        }
    }

    //
    inline void
    visit_edge(const int level, const storage_idx_t node_from, const storage_idx_t node_to, const float distance) {
        if (feder != nullptr) {
            feder->visit_info_.AddVisitRecord(level, node_from, node_to, distance);
            feder->id_set_.insert(node_from);
            feder->id_set_.insert(node_to);
        }
    }
};

}  // namespace knowhere
