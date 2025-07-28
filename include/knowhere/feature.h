// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <cstdint>

#ifndef FEATURE_H
#define FEATURE_H

// these features have been report to outside (milvus); pls sync the feature code when it needs to be changed.
namespace knowhere::feature {
// vector datatype support : binary
constexpr uint64_t BINARY = 1UL << 0;
// vector datatype support : float32
constexpr uint64_t FLOAT32 = 1UL << 1;
// vector datatype support : fp16
constexpr uint64_t FP16 = 1UL << 2;
// vector datatype support : bf16
constexpr uint64_t BF16 = 1UL << 3;
// vector datatype support : sparse_float32
constexpr uint64_t SPARSE_FLOAT32 = 1UL << 4;
// vector datatype support : int8
constexpr uint64_t INT8 = 1UL << 5;

// This flag indicates that the index is a emb list index
constexpr uint64_t EMB_LIST = 1UL << 15;
// This flag indicates that there is no need to create any index structure (build stage can be skipped)
constexpr uint64_t NO_TRAIN = 1UL << 16;
// This flag indicates that the index defaults to KNN search, meaning the recall is 100% (no precision loss compared
// with original data)
constexpr uint64_t KNN = 1UL << 17;
// This flag indicates that the index search stage will be performed on GPU (need GPU devices)
constexpr uint64_t GPU = 1UL << 18;
// This flag indicates that the index support using mmap manage its mainly memory, which can significant improve the
// capacity
constexpr uint64_t MMAP = 1UL << 19;
// This flag indicates that the index support using materialized view to accelerate filtering search
constexpr uint64_t MV = 1UL << 20;
// This flag indicates that the index need disk during search stage
constexpr uint64_t DISK = 1UL << 21;

constexpr uint64_t NONE = 0UL;

constexpr uint64_t ALL_TYPE = BINARY | FLOAT32 | FP16 | BF16 | SPARSE_FLOAT32 | INT8;
constexpr uint64_t ALL_DENSE_TYPE = BINARY | FLOAT32 | FP16 | BF16 | INT8;
constexpr uint64_t ALL_DENSE_FLOAT_TYPE = FLOAT32 | FP16 | BF16;

constexpr uint64_t NO_TRAIN_INDEX = NO_TRAIN;
constexpr uint64_t GPU_KNN_FLOAT_INDEX = FLOAT32 | GPU | KNN;
constexpr uint64_t GPU_ANN_FLOAT_INDEX = FLOAT32 | GPU;
}  // namespace knowhere::feature
#endif /* FEATURE_H */
