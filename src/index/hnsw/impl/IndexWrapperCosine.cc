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

#include "index/hnsw/impl/IndexWrapperCosine.h"

#include <faiss/cppcontrib/knowhere/IndexCosine.h>

namespace knowhere {

// a wrapper that overrides a distance computer
IndexWrapperCosine::IndexWrapperCosine(faiss::cppcontrib::knowhere::Index* index, const float* inverse_l2_norms_in)
    : faiss::cppcontrib::knowhere::IndexWrapper(index), inverse_l2_norms{inverse_l2_norms_in} {
}

faiss::DistanceComputer*
IndexWrapperCosine::get_distance_computer() const {
    return new faiss::cppcontrib::knowhere::WithCosineNormDistanceComputer(
        inverse_l2_norms, index->d, std::unique_ptr<faiss::DistanceComputer>(index->get_distance_computer()));
}

}  // namespace knowhere
