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

#include "knowhere/utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "faiss/cppcontrib/knowhere/IndexIVFFlat.h"
#include "faiss/impl/FaissException.h"
#include "faiss/index_io.h"
#include "io/memory_io.h"
#include "knowhere/log.h"
#include "simd/hook.h"

namespace knowhere {

const float FloatAccuracy = 0.00001;
template <typename DataType>
float
GetL2Norm(const DataType* x, int32_t d) {
    float norm_l2_sqr = 0.0;
    if constexpr (std::is_same_v<DataType, fp32>) {
        norm_l2_sqr = faiss::cppcontrib::knowhere::fvec_norm_L2sqr(x, d);
    } else if constexpr (std::is_same_v<DataType, fp16>) {
        norm_l2_sqr = faiss::cppcontrib::knowhere::fp16_vec_norm_L2sqr(x, d);
    } else if constexpr (std::is_same_v<DataType, bf16>) {
        norm_l2_sqr = faiss::cppcontrib::knowhere::bf16_vec_norm_L2sqr(x, d);
    } else {
        KNOWHERE_THROW_MSG("Unknown Datatype");
    }

    if (norm_l2_sqr > 0 && std::abs(1.0f - norm_l2_sqr) > FloatAccuracy) {
        float norm_l2 = std::sqrt(norm_l2_sqr);
        return norm_l2;
    }
    return 1.0f;
}

template <typename DataType>
std::vector<float>
GetL2Norms(const DataType* x, int32_t d, int32_t n) {
    std::vector<float> norms(n);
    for (auto i = 0; i < n; i++) {
        auto x_i = x + d * i;
        norms[i] = GetL2Norm(x_i, d);
    }
    return norms;
}
// normalize one vector and return its norm
template <typename DataType>
float
NormalizeVec(DataType* x, int32_t d) {
    float norm_l2_sqr = 0.0;
    if constexpr (std::is_same_v<DataType, fp32>) {
        norm_l2_sqr = faiss::cppcontrib::knowhere::fvec_norm_L2sqr(x, d);
    } else if constexpr (std::is_same_v<DataType, fp16>) {
        norm_l2_sqr = faiss::cppcontrib::knowhere::fp16_vec_norm_L2sqr(x, d);
    } else if constexpr (std::is_same_v<DataType, bf16>) {
        norm_l2_sqr = faiss::cppcontrib::knowhere::bf16_vec_norm_L2sqr(x, d);
    } else {
        KNOWHERE_THROW_MSG("Unknown Datatype");
    }

    if (norm_l2_sqr > 0 && std::abs(1.0f - norm_l2_sqr) > FloatAccuracy) {
        float norm_l2 = std::sqrt(norm_l2_sqr);
        for (int32_t i = 0; i < d; i++) {
            x[i] = (DataType)((float)x[i] / norm_l2);
        }
        return norm_l2;
    }
    return 1.0f;
}

// normalize all vectors and return their norms
template <typename DataType>
std::vector<float>
NormalizeVecs(DataType* x, size_t rows, int32_t dim) {
    std::vector<float> norms(rows);
    for (size_t i = 0; i < rows; i++) {
        norms[i] = NormalizeVec(x + i * dim, dim);
    }
    return norms;
}

// copy and return normalized vectors
template <typename DataType>
std::unique_ptr<DataType[]>
CopyAndNormalizeVecs(const DataType* x, size_t rows, int32_t dim) {
    auto x_normalized = std::make_unique<DataType[]>(rows * dim);
    std::copy_n(x, rows * dim, x_normalized.get());
    NormalizeVecs(x_normalized.get(), rows, dim);
    return x_normalized;
}

template <typename DataType>
void
NormalizeDataset(const DataSetPtr dataset) {
    auto rows = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto data = (DataType*)dataset->GetTensor();

    LOG_KNOWHERE_DEBUG_ << "vector normalize, rows " << rows << ", dim " << dim;
    NormalizeVecs<DataType>(data, rows, dim);
}

template <typename DataType>
std::tuple<DataSetPtr, std::vector<float>>
CopyAndNormalizeDataset(const DataSetPtr dataset) {
    auto rows = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto data = (DataType*)dataset->GetTensor();

    LOG_KNOWHERE_DEBUG_ << "vector normalize, rows " << rows << ", dim " << dim;

    auto x_normalized = new DataType[rows * dim];
    std::copy_n(data, rows * dim, x_normalized);
    auto norms = NormalizeVecs<DataType>(x_normalized, rows, dim);
    auto normalize_bs = GenDataSet(rows, dim, x_normalized);
    normalize_bs->SetIsOwner(true);
    return std::make_tuple(normalize_bs, norms);
}

bool
UseDiskLoad(const std::string& index_type, const int32_t& version) {
#ifdef KNOWHERE_WITH_CARDINAL
    if (version == 0) {
        return !index_type.compare(IndexEnum::INDEX_DISKANN);
    } else {
        return !index_type.compare(IndexEnum::INDEX_DISKANN) || !index_type.compare(IndexEnum::INDEX_HNSW) ||
               !index_type.compare(IndexEnum::INDEX_MINHASH_LSH) ||
               !index_type.compare(IndexEnum::INDEX_CARDINAL_TIERED);
    }
#else
    return !index_type.compare(IndexEnum::INDEX_DISKANN) || !index_type.compare(IndexEnum::INDEX_MINHASH_LSH) ||
           !index_type.compare(IndexEnum::INDEX_AISAQ);
#endif
}

template float
GetL2Norm<fp32>(const fp32* x, int32_t d);
template float
GetL2Norm<fp16>(const fp16* x, int32_t d);
template float
GetL2Norm<bf16>(const bf16* x, int32_t d);

template std::vector<float>
GetL2Norms<fp32>(const fp32* x, int32_t d, int32_t n);
template std::vector<float>
GetL2Norms<fp16>(const fp16* x, int32_t d, int32_t n);
template std::vector<float>
GetL2Norms<bf16>(const bf16* x, int32_t d, int32_t n);

template float
NormalizeVec<fp32>(fp32* x, int32_t d);
template float
NormalizeVec<fp16>(fp16* x, int32_t d);
template float
NormalizeVec<bf16>(bf16* x, int32_t d);

template std::vector<float>
NormalizeVecs<fp32>(fp32* x, size_t rows, int32_t dim);
template std::vector<float>
NormalizeVecs<fp16>(fp16* x, size_t rows, int32_t dim);
template std::vector<float>
NormalizeVecs<bf16>(bf16* x, size_t rows, int32_t dim);

template std::unique_ptr<fp32[]>
CopyAndNormalizeVecs(const fp32* x, size_t rows, int32_t dim);
template std::unique_ptr<fp16[]>
CopyAndNormalizeVecs(const fp16* x, size_t rows, int32_t dim);
template std::unique_ptr<bf16[]>
CopyAndNormalizeVecs(const bf16* x, size_t rows, int32_t dim);

template void
NormalizeDataset<fp32>(const DataSetPtr dataset);
template void
NormalizeDataset<fp16>(const DataSetPtr dataset);
template void
NormalizeDataset<bf16>(const DataSetPtr dataset);

template std::tuple<DataSetPtr, std::vector<float>>
CopyAndNormalizeDataset<fp32>(const DataSetPtr dataset);
template std::tuple<DataSetPtr, std::vector<float>>
CopyAndNormalizeDataset<fp16>(const DataSetPtr dataset);
template std::tuple<DataSetPtr, std::vector<float>>
CopyAndNormalizeDataset<bf16>(const DataSetPtr dataset);
}  // namespace knowhere
