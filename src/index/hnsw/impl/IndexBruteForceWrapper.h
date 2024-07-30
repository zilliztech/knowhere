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

#include <faiss/Index.h>
#include <faiss/cppcontrib/knowhere/IndexWrapper.h>

namespace knowhere {

// override a search procedure to perform a brute-force search.
struct IndexBruteForceWrapper : public faiss::cppcontrib::knowhere::IndexWrapper {
    IndexBruteForceWrapper(faiss::Index* underlying_index);

    /// entry point for search
    void
    search(faiss::idx_t n, const float* x, faiss::idx_t k, float* distances, faiss::idx_t* labels,
           const faiss::SearchParameters* params) const override;
};

}  // namespace knowhere
