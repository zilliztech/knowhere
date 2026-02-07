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

#include <faiss/cppcontrib/knowhere/Index.h>
#include <faiss/cppcontrib/knowhere/IndexWrapper.h>
#include <faiss/impl/DistanceComputer.h>

namespace knowhere {

// overrides a distance compute function
struct IndexWrapperCosine : public faiss::cppcontrib::knowhere::IndexWrapper {
    // a non-owning pointer
    const float* inverse_l2_norms;

    // norms are external
    IndexWrapperCosine(faiss::cppcontrib::knowhere::Index* index, const float* inverse_l2_norms_in);

    faiss::DistanceComputer*
    get_distance_computer() const override;
};

}  // namespace knowhere
