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

#include <strings.h>

#include <algorithm>
#include <fstream>
#include <vector>

#include "knowhere/binaryset.h"
#include "knowhere/dataset.h"
#include "knowhere/operands.h"

namespace knowhere {

extern const float FloatAccuracy;

inline bool
IsMetricType(const std::string& str, const knowhere::MetricType& metric_type) {
    return !strcasecmp(str.data(), metric_type.c_str());
}

inline bool
IsFlatIndex(const knowhere::IndexType& index_type) {
    static std::vector<knowhere::IndexType> flat_index_list = {
        IndexEnum::INDEX_FAISS_IDMAP, IndexEnum::INDEX_FAISS_GPU_IDMAP, IndexEnum::INDEX_GPU_BRUTEFORCE};
    return std::find(flat_index_list.begin(), flat_index_list.end(), index_type) != flat_index_list.end();
}

template <typename DataType>
float
GetL2Norm(const DataType* x, int32_t d);

template <typename DataType>
std::vector<float>
GetL2Norms(const DataType* x, int32_t d, int32_t n);

template <typename DataType>
extern float
NormalizeVec(DataType* x, int32_t d);

template <typename DataType>
extern std::vector<float>
NormalizeVecs(DataType* x, size_t rows, int32_t dim);

template <typename DataType>
extern std::unique_ptr<DataType[]>
CopyAndNormalizeVecs(const DataType* x, size_t rows, int32_t dim);

template <typename DataType>
extern void
NormalizeDataset(const DataSetPtr dataset);

template <typename DataType>
extern std::tuple<DataSetPtr, std::vector<float>>
CopyAndNormalizeDataset(const DataSetPtr dataset);

constexpr inline uint64_t seed = 0xc70f6907UL;

inline uint64_t
hash_vec(const float* x, size_t d) {
    uint64_t h = seed;
    for (size_t i = 0; i < d; ++i) {
        h = h * 13331 + *(uint32_t*)(x + i);
    }
    return h;
}

inline uint64_t
hash_u8_vec(const uint8_t* x, size_t d) {
    uint64_t h = seed;
    for (size_t i = 0; i < d; ++i) {
        h = h * 13331 + *(x + i);
    }
    return h;
}

inline uint64_t
hash_binary_vec(const uint8_t* x, size_t d) {
    size_t len = (d + 7) / 8;
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h = h * 13331 + x[i];
    }
    return h;
}

inline uint64_t
hash_half_precision_float(const void* x, size_t d) {
    uint64_t h = seed;
    auto u16_x = (uint16_t*)(x);
    for (size_t i = 0; i < d; ++i) {
        h = h * 13331 + u16_x[i];
    }
    return h;
}

template <typename DataType>
inline std::string
GetKey(const std::string& name) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    if (std::is_same_v<DataType, fp32>) {
        return name + std::string("_fp32");
    } else if (std::is_same_v<DataType, fp16>) {
        return name + std::string("_fp16");
    } else if (std::is_same_v<DataType, bf16>) {
        return name + std::string("_bf16");
    } else if (std::is_same_v<DataType, bin1>) {
        return name + std::string("_bin1");
    } else if (std::is_same_v<DataType, int8>) {
        return name + std::string("_int8");
    }
}

template <typename InType, typename OutType>
inline DataSetPtr
data_type_conversion(const DataSet& src, const std::optional<int64_t> start = std::nullopt,
                     const std::optional<int64_t> count = std::nullopt,
                     const std::optional<int64_t> count_dim = std::nullopt) {
    auto in_dim = src.GetDim();
    auto out_dim = count_dim.value_or(in_dim);
    auto rows = src.GetRows();

    // check the acceptable range
    int64_t start_row = start.value_or(0);
    if (start_row < 0 || start_row >= rows) {
        return nullptr;
    }

    int64_t count_rows = count.value_or(rows - start_row);
    if (count_rows < 0 || start_row + count_rows > rows) {
        return nullptr;
    }

    // map
    auto* des_data = new OutType[out_dim * count_rows];

    // only do memset() for intrinsic data types, such as float;
    // for fp16/bf16, they will be initialized by the constructor
    if constexpr (std::is_arithmetic_v<OutType>) {
        std::memset(des_data, 0, sizeof(OutType) * out_dim * count_rows);
    }

    auto* src_data = (const InType*)src.GetTensor();
    for (auto i = 0; i < count_rows; i++) {
        for (auto d = 0; d < in_dim; d++) {
            des_data[i * out_dim + d] = (OutType)src_data[(start_row + i) * in_dim + d];
        }
    }

    auto des = std::make_shared<DataSet>();
    des->SetRows(count_rows);
    des->SetDim(out_dim);
    des->SetTensor(des_data);
    des->SetIsOwner(true);
    des->SetTensorBeginId(src.GetTensorBeginId() + start_row);

    // for emb_list
    auto lims = src.GetLims();
    if (lims != nullptr) {
        size_t lims_size = 0;
        while (lims[lims_size] < (size_t)rows && lims_size < (size_t)rows) {
            lims_size++;
        }
        assert(lims[lims_size] == (size_t)rows);  // the last lims should be the rows
        auto lims_data = std::make_unique<size_t[]>(lims_size + 1);
        for (size_t i = 0; i < lims_size + 1; i++) {
            lims_data[i] = lims[i];
        }
        des->SetLims(std::move(lims_data));
    }
    return des;
}

// Convert DataSet from DataType to float
// * no start, no count, float -> returns the source without cloning
// * no start, no count, no float -> returns a clone with a different type
// * start, no count -> returns a clone that starts from a given row 'start'
// * no start, count -> returns a clone that starts from a row 0 and has 'count' rows
// * start, count -> returns a clone that start from a given row 'start' and has 'count' rows
// * invalid start, count values -> returns nullptr
template <typename DataType>
inline DataSetPtr
ConvertFromDataTypeIfNeeded(const DataSetPtr& ds, const std::optional<int64_t> start = std::nullopt,
                            const std::optional<int64_t> count = std::nullopt,
                            const std::optional<int64_t> count_dim = std::nullopt) {
    if constexpr (std::is_same_v<DataType, typename MockData<DataType>::type>) {
        if (!start.has_value() && !count.has_value() && (!count_dim.has_value() || ds->GetDim() == count_dim.value())) {
            return ds;
        }
    }

    return data_type_conversion<DataType, typename MockData<DataType>::type>(*ds, start, count, count_dim);
}

// Convert DataSet from float to DataType
template <typename DataType>
inline DataSetPtr
ConvertToDataTypeIfNeeded(const DataSetPtr& ds, const std::optional<int64_t> start = std::nullopt,
                          const std::optional<int64_t> count = std::nullopt,
                          const std::optional<int64_t> count_dim = std::nullopt) {
    if constexpr (std::is_same_v<DataType, typename MockData<DataType>::type>) {
        if (!start.has_value() && !count.has_value() && (!count_dim.has_value() || ds->GetDim() == count_dim.value())) {
            return ds;
        }
    }

    return data_type_conversion<typename MockData<DataType>::type, DataType>(*ds, start, count, count_dim);
}

template <typename T>
inline T
round_down(const T value, const T align) {
    return value / align * align;
}

bool
UseDiskLoad(const std::string& index_type, const int32_t& /*version*/);

template <typename T, typename W>
static void
writeBinaryPOD(W& out, const T& podRef) {
    out.write((char*)&podRef, sizeof(T));
}

template <typename T, typename R>
static void
readBinaryPOD(R& in, T& podRef) {
    in.read((char*)&podRef, sizeof(T));
}

template <typename T>
inline void
load_vec_meta(const std::string& bin_file, size_t& rows, size_t& dim) {
    uint32_t u32_rows, u32_dim;
    std::ifstream file(bin_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("fail to open file: " + bin_file);
    }
    size_t autual_file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(&u32_rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&u32_dim), sizeof(uint32_t));
    rows = u32_rows;
    dim = u32_dim;
    // check data dim and size
    size_t expect_file_size = 0;
    if constexpr (std::is_same_v<T, bin1>) {
        if (dim % 8 != 0) {
            throw std::runtime_error("fail to load binary vector base file, dim % 8 != 0 ");
        }
        expect_file_size = rows * dim / 8 + 2 * sizeof(uint32_t);
    } else {
        expect_file_size = rows * dim * sizeof(T) + 2 * sizeof(uint32_t);
    }
    if (autual_file_size != expect_file_size) {
        throw std::runtime_error("fail to get raw data meta, file size mismatch of raw data file.");
    }
}

template <typename T>
inline void
load_vec_data(const std::string& bin_file, std::unique_ptr<char[]>& data, size_t& npts, size_t& dim) {
    std::ifstream file(bin_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("fail to open file: " + bin_file);
    }
    size_t autual_file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    uint32_t n, d;
    file.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
    npts = n;
    dim = d;
    size_t expect_file_size = 0;
    // check data dim and size
    if constexpr (std::is_same_v<T, bin1>) {
        if (dim % 8 != 0) {
            throw std::runtime_error("fail to load binary vector base file, dim % 8 != 0 ");
        }
        expect_file_size = npts * dim / 8 + 2 * sizeof(uint32_t);
    } else {
        expect_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    }
    if (autual_file_size != expect_file_size) {
        throw std::runtime_error("fail to load raw data, file size mismatch of raw data file.");
    }
    uint64_t total_size = dim * npts / 8;
    data = std::make_unique<char[]>(total_size);
    file.read(reinterpret_cast<char*>(data.get()), total_size);
}

// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t)(X) / (Y)) * (Y))

}  // namespace knowhere
