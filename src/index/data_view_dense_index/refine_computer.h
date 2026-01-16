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

// knowhere-specific indices
#pragma once

#include "faiss/cppcontrib/knowhere/impl/ScalarQuantizer.h"
#include "faiss/cppcontrib/knowhere/invlists/InvertedLists.h"
#include "faiss/impl/DistanceComputer.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/object.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"
#include "simd/hook.h"

namespace knowhere {
using idx_t = faiss::idx_t;
namespace {
inline bool
convert_data(const void* in_data, float* out_data, const DataFormatEnum data_type, const size_t n, const size_t dim) {
    // todo: use simd to convert, these codes cost much cpu time
    if (data_type == DataFormatEnum::fp16) {
        auto fp16_data = (const fp16*)in_data;
        for (size_t i = 0; i < n; i++) {
            for (size_t d = 0; d < dim; d++) {
                out_data[i * dim + d] = (float)(fp16_data[i * dim + d]);
            }
        }
        return true;
    } else if (data_type == DataFormatEnum::bf16) {
        auto bf16_data = (const bf16*)in_data;
        for (size_t i = 0; i < n; i++) {
            for (size_t d = 0; d < dim; d++) {
                out_data[i * dim + d] = (float)(bf16_data[i * dim + d]);
            }
        }
        return true;
    }
    return false;
}
}  // namespace
/*
Quantify the streaming data with a thread safe mode, only support fp32 vector
 */
struct QuantRefine {
 public:
    QuantRefine(size_t d, DataFormatEnum data_type, RefineType refine_type, MetricType metric)
        : origin_data_type(data_type), refine_type(refine_type) {
        if (metric == metric::IP) {
            metric_type = faiss::MetricType::METRIC_INNER_PRODUCT;
        } else if (metric == metric::L2) {
            metric_type = faiss::MetricType::METRIC_L2;
        } else {
            throw std::runtime_error("QuantRefine only support metric L2 and IP.");
        }
        switch (refine_type) {
            case RefineType::UINT8_QUANT:
                quantizer = new faiss::cppcontrib::knowhere::ScalarQuantizer(
                    d, faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType::QT_8bit);
                break;
            case RefineType::BFLOAT16_QUANT:
                quantizer = new faiss::cppcontrib::knowhere::ScalarQuantizer(
                    d, faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType::QT_bf16);
                break;
            case RefineType::FLOAT16_QUANT:
                quantizer = new faiss::cppcontrib::knowhere::ScalarQuantizer(
                    d, faiss::cppcontrib::knowhere::ScalarQuantizer::QuantizerType::QT_fp16);
                break;
            default:
                throw std::runtime_error("Fail to generate quant for refiner if refine_type == RefineType::DATA_VIEW");
                break;
        }
        storage = new faiss::cppcontrib::knowhere::ConcurrentArrayInvertedLists(list_num, quantizer->code_size,
                                                                                segment_size, false);
    }
    void
    Train(const void* train_data, const size_t n) {
        if (origin_data_type == DataFormatEnum::fp32) {
            quantizer->train(n, (const float*)train_data);
        } else {
            auto fp32_x = std::unique_ptr<float[]>(new float[n * quantizer->d]);
            if (convert_data(train_data, fp32_x.get(), origin_data_type, n, quantizer->d) != true) {
                throw std::runtime_error("fail to convert data to fp32 type.");
            }
            quantizer->train(n, fp32_x.get());
        }
    }

    void
    Add(const void* data, const idx_t* ids, const size_t n) {
        auto codes = std::make_unique<uint8_t[]>(n * quantizer->code_size);
        if (origin_data_type == DataFormatEnum::fp32) {
            quantizer->compute_codes((const float*)data, codes.get(), n);
            storage->add_entries(key, n, ids, codes.get());
        } else {
            auto fp32_x = std::unique_ptr<float[]>(new float[n * quantizer->d]);
            if (convert_data(data, fp32_x.get(), origin_data_type, n, quantizer->d) != true) {
                throw std::runtime_error("fail to convert data to fp32 type.");
            }
            quantizer->compute_codes(fp32_x.get(), codes.get(), n);
            storage->add_entries(key, n, ids, codes.get());
        }
    }

    const uint8_t*
    GetCode(size_t id) {
        return storage->get_codes(key, id);
    }
    faiss::MetricType
    GetMetric() {
        return metric_type;
    }
    std::unique_ptr<faiss::cppcontrib::knowhere::ScalarQuantizer::SQDistanceComputer>
    GetQuantComputer() {
        return std::unique_ptr<faiss::cppcontrib::knowhere::ScalarQuantizer::SQDistanceComputer>(
            quantizer->get_distance_computer(metric_type));
    }
    DataFormatEnum
    GetOriginDataType() {
        return origin_data_type;
    }

    ~QuantRefine() {
        if (storage != nullptr) {
            delete storage;
        }
        if (quantizer != nullptr) {
            delete quantizer;
        }
    }

 private:
    static constexpr size_t key = 0;
    static constexpr size_t list_num = 1;
    static constexpr size_t segment_size = 48;
    faiss::cppcontrib::knowhere::ScalarQuantizer* quantizer = nullptr;
    faiss::cppcontrib::knowhere::InvertedLists* storage = nullptr;
    faiss::MetricType metric_type;
    DataFormatEnum origin_data_type;
    RefineType refine_type;
};

// refine computer only use in single thread
template <bool NeedNormalize = false>
struct QuantDataDistanceComputer : faiss::DistanceComputer {
    std::vector<float> query_buf;
    std::shared_ptr<QuantRefine> quant_data;
    std::unique_ptr<faiss::cppcontrib::knowhere::ScalarQuantizer::SQDistanceComputer> qc;
    float q_norm;
    size_t dim;

    QuantDataDistanceComputer(const std::shared_ptr<QuantRefine> quant_data, const size_t dim,
                              const float* query = nullptr)
        : quant_data(quant_data), dim(dim) {
        qc = quant_data->GetQuantComputer();
        if (query != nullptr) {
            set_query(query);
        }
        return;
    }

    void
    set_query(const float* x) override {
        if (quant_data->GetOriginDataType() == DataFormatEnum::fp32) {
            qc->q = x;
            if constexpr (NeedNormalize) {
                q_norm = GetL2Norm(x, dim);
            }
        } else {
            query_buf.resize(dim);
            if (convert_data(x, query_buf.data(), quant_data->GetOriginDataType(), 1, dim) != true) {
                throw std::runtime_error("fail to convert data to fp32 type.");
            }
            qc->q = query_buf.data();
            if constexpr (NeedNormalize) {
                q_norm = GetL2Norm(query_buf.data(), dim);
            }
        }
    }

    float
    operator()(idx_t i) override {
        auto code_i = quant_data->GetCode(i);
        if constexpr (NeedNormalize) {
            return qc->distance_to_code(code_i) / q_norm;
        } else {
            return qc->distance_to_code(code_i);
        }
    }

    void
    distances_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3, float& dis0, float& dis1,
                      float& dis2, float& dis3) override {
        auto code_0 = quant_data->GetCode(idx0);
        auto code_1 = quant_data->GetCode(idx1);
        auto code_2 = quant_data->GetCode(idx2);
        auto code_3 = quant_data->GetCode(idx3);
        qc->query_to_codes_batch_4(code_0, code_1, code_2, code_3, dis0, dis1, dis2, dis3);
        if constexpr (NeedNormalize) {
            dis0 /= q_norm;
            dis1 /= q_norm;
            dis2 /= q_norm;
            dis3 /= q_norm;
        }
        return;
    }

    /// compute distance between two stored vectors
    float
    symmetric_dis(idx_t i, idx_t j) override {
        // todo: support symmetric_dis after making compute_code_distance function to a virtual function
        throw std::runtime_error("symmetric_dis() not support for QuantDataDistanceComputer");
    }
};

template <typename DataType, typename Distance1, typename Distance4, bool NeedNormalize = false>
struct DataViewDistanceComputer : faiss::DistanceComputer {
    ViewDataOp view_data;
    size_t dim;
    const DataType* q;
    Distance1 dist1;
    Distance4 dist4;
    float q_norm;

    DataViewDistanceComputer(const ViewDataOp& view_data, const size_t dim, Distance1 dist1, Distance4 dist4,
                             const DataType* query = nullptr, std::optional<float> query_norm = std::nullopt)
        : view_data(view_data), dim(dim), dist1(dist1), dist4(dist4) {
        if (query != nullptr) {
            this->set_query((const float*)query, query_norm);
        }
        return;
    }

    // convert x to float* for override, still use DataType to get distance
    void
    set_query(const float* x) override {
        q = (const DataType*)x;
        if constexpr (NeedNormalize) {
            q_norm = GetL2Norm(q, dim);
        }
    }

    void
    set_query(const float* x, std::optional<float> x_norm = std::nullopt) {
        q = (const DataType*)x;
        if constexpr (NeedNormalize) {
            q_norm = x_norm.value_or(GetL2Norm(q, dim));
        }
    }

    float
    operator()(idx_t i) override {
        auto code = view_data(i);
        return distance_to_code(code);
    }

    float
    distance_to_code(const void* x) {
        if constexpr (NeedNormalize) {
            return dist1(q, (const DataType*)x, dim) / q_norm;
        } else {
            return dist1(q, (const DataType*)x, dim);
        }
    }

    void
    distances_batch_4(const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3, float& dis0, float& dis1,
                      float& dis2, float& dis3) override {
        auto x0 = (DataType*)view_data(idx0);
        auto x1 = (DataType*)view_data(idx1);
        auto x2 = (DataType*)view_data(idx2);
        auto x3 = (DataType*)view_data(idx3);
        dist4(q, x0, x1, x2, x3, dim, dis0, dis1, dis2, dis3);
        if constexpr (NeedNormalize) {
            dis0 /= q_norm;
            dis1 /= q_norm;
            dis2 /= q_norm;
            dis3 /= q_norm;
        }
    }

    /// compute distance between two stored vectors
    float
    symmetric_dis(idx_t i, idx_t j) override {
        auto x = (DataType*)view_data(i);
        auto y = (DataType*)view_data(j);
        return dist1(x, y, dim);
    }
};

static std::unique_ptr<faiss::DistanceComputer>
SelectDataViewComputer(const ViewDataOp& view_data, const DataFormatEnum& data_type, const knowhere::MetricType& metric,
                       const size_t dim, bool is_cosine, const std::shared_ptr<QuantRefine> quant = nullptr) {
    if (quant) {
        if (is_cosine) {
            return std::unique_ptr<faiss::DistanceComputer>(new QuantDataDistanceComputer<true>(quant, dim));
        } else {
            return std::unique_ptr<faiss::DistanceComputer>(new QuantDataDistanceComputer<false>(quant, dim));
        }
    } else if (data_type == DataFormatEnum::fp16) {
        if (metric == metric::IP) {
            if (is_cosine) {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<fp16, decltype(faiss::cppcontrib::knowhere::fp16_vec_inner_product),
                                                 decltype(faiss::cppcontrib::knowhere::fp16_vec_inner_product_batch_4),
                                                 true>(view_data, dim,
                                                       faiss::cppcontrib::knowhere::fp16_vec_inner_product,
                                                       faiss::cppcontrib::knowhere::fp16_vec_inner_product_batch_4));
            } else {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<fp16, decltype(faiss::cppcontrib::knowhere::fp16_vec_inner_product),
                                                 decltype(faiss::cppcontrib::knowhere::fp16_vec_inner_product_batch_4),
                                                 false>(view_data, dim,
                                                        faiss::cppcontrib::knowhere::fp16_vec_inner_product,
                                                        faiss::cppcontrib::knowhere::fp16_vec_inner_product_batch_4));
            }
        } else {
            return std::unique_ptr<faiss::DistanceComputer>(
                new DataViewDistanceComputer<fp16, decltype(faiss::cppcontrib::knowhere::fp16_vec_L2sqr),
                                             decltype(faiss::cppcontrib::knowhere::fp16_vec_L2sqr_batch_4)>(
                    view_data, dim, faiss::cppcontrib::knowhere::fp16_vec_L2sqr,
                    faiss::cppcontrib::knowhere::fp16_vec_L2sqr_batch_4));
        }
    } else if (data_type == DataFormatEnum::bf16) {
        if (metric == metric::IP) {
            if (is_cosine) {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<bf16, decltype(faiss::cppcontrib::knowhere::bf16_vec_inner_product),
                                                 decltype(faiss::cppcontrib::knowhere::bf16_vec_inner_product_batch_4),
                                                 true>(view_data, dim,
                                                       faiss::cppcontrib::knowhere::bf16_vec_inner_product,
                                                       faiss::cppcontrib::knowhere::bf16_vec_inner_product_batch_4));
            } else {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<bf16, decltype(faiss::cppcontrib::knowhere::bf16_vec_inner_product),
                                                 decltype(faiss::cppcontrib::knowhere::bf16_vec_inner_product_batch_4),
                                                 false>(view_data, dim,
                                                        faiss::cppcontrib::knowhere::bf16_vec_inner_product,
                                                        faiss::cppcontrib::knowhere::bf16_vec_inner_product_batch_4));
            }
        } else {
            return std::unique_ptr<faiss::DistanceComputer>(
                new DataViewDistanceComputer<bf16, decltype(faiss::cppcontrib::knowhere::bf16_vec_L2sqr),
                                             decltype(faiss::cppcontrib::knowhere::bf16_vec_L2sqr_batch_4)>(
                    view_data, dim, faiss::cppcontrib::knowhere::bf16_vec_L2sqr,
                    faiss::cppcontrib::knowhere::bf16_vec_L2sqr_batch_4));
        }
    } else if (data_type == DataFormatEnum::fp32) {
        if (metric == metric::IP) {
            if (is_cosine) {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<fp32, decltype(faiss::cppcontrib::knowhere::fvec_inner_product),
                                                 decltype(faiss::cppcontrib::knowhere::fvec_inner_product_batch_4),
                                                 true>(view_data, dim, faiss::cppcontrib::knowhere::fvec_inner_product,
                                                       faiss::cppcontrib::knowhere::fvec_inner_product_batch_4));
            } else {
                return std::unique_ptr<faiss::DistanceComputer>(
                    new DataViewDistanceComputer<fp32, decltype(faiss::cppcontrib::knowhere::fvec_inner_product),
                                                 decltype(faiss::cppcontrib::knowhere::fvec_inner_product_batch_4),
                                                 false>(view_data, dim, faiss::cppcontrib::knowhere::fvec_inner_product,
                                                        faiss::cppcontrib::knowhere::fvec_inner_product_batch_4));
            }
        } else {
            return std::unique_ptr<faiss::DistanceComputer>(
                new DataViewDistanceComputer<fp32, decltype(faiss::cppcontrib::knowhere::fvec_L2sqr),
                                             decltype(faiss::cppcontrib::knowhere::fvec_L2sqr_batch_4)>(
                    view_data, dim, faiss::cppcontrib::knowhere::fvec_L2sqr,
                    faiss::cppcontrib::knowhere::fvec_L2sqr_batch_4));
        }
    } else {
        return nullptr;
    }
}
}  // namespace knowhere
