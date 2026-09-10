// Copyright (C) 2026 Zilliz. All rights reserved.
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

#include <faiss/cppcontrib/knowhere/IndexHNSWRaBitQ.h>

#include "index/hnsw/impl/IndexHNSWWrapper.h"

namespace knowhere {

// Owns the per-request storage parameters; no shared index state is changed.
struct SearchParametersHNSWRaBitQWrapper : SearchParametersHNSWWrapper {
    faiss::RaBitQSearchParameters storage_params;

    faiss::DistanceComputer*
    storage_distance_computer(const faiss::Index* index) const override {
        const auto* rbq = dynamic_cast<const faiss::cppcontrib::knowhere::IndexHNSWRaBitQ*>(index);
        FAISS_THROW_IF_NOT_MSG(rbq, "RaBitQ search parameters require RaBitQ storage");
        auto* dc = rbq->get_staged_distance_computer(&storage_params);
        // The staged adapter is smaller-is-better; BF expects public metric units.
        return index->metric_type == faiss::METRIC_INNER_PRODUCT ? new faiss::NegativeDistanceComputer(dc) : dc;
    }
};

}  // namespace knowhere
