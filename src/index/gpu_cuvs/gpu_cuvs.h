/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GPU_CUVS_H
#define GPU_CUVS_H

#include <cstdint>
#include <exception>
#include <fstream>
#include <istream>
#include <numeric>
#include <thread>
#include <tuple>
#include <vector>

#include "common/cuvs/integration/cuvs_knowhere_config.hpp"
#include "common/cuvs/integration/cuvs_knowhere_index.hpp"
#include "common/cuvs/integration/type_mappers.hpp"
#include "common/cuvs/proto/cuvs_index_kind.hpp"
#include "index/gpu_cuvs/gpu_cuvs_brute_force_config.h"
#include "index/gpu_cuvs/gpu_cuvs_cagra_config.h"
#include "index/gpu_cuvs/gpu_cuvs_ivf_flat_config.h"
#include "index/gpu_cuvs/gpu_cuvs_ivf_pq_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

auto static constexpr cuda_concurrent_size_per_device = std::uint32_t{4};

template <cuvs_proto::cuvs_index_kind K>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<cuvs_proto::cuvs_index_kind::brute_force> {
    using Type = GpuCuvsBruteForceConfig;
};

template <>
struct KnowhereConfigType<cuvs_proto::cuvs_index_kind::ivf_flat> {
    using Type = GpuCuvsIvfFlatConfig;
};

template <>
struct KnowhereConfigType<cuvs_proto::cuvs_index_kind::ivf_pq> {
    using Type = GpuCuvsIvfPqConfig;
};

template <>
struct KnowhereConfigType<cuvs_proto::cuvs_index_kind::cagra> {
    using Type = GpuCuvsCagraConfig;
};

template <typename DataType, cuvs_proto::cuvs_index_kind K>
struct GpuCuvsIndexNode : public IndexNode {
    auto static constexpr index_kind = K;
    using knowhere_config_type = typename KnowhereConfigType<index_kind>::Type;
    using data_type = typename cuvs_knowhere::cuvs_data_type_mapper<DataType>::data_type;

    GpuCuvsIndexNode(int32_t, const Object& object) : index_{} {
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        auto result = Status::success;
        auto cuvs_cfg = cuvs_knowhere::cuvs_knowhere_config{};
        try {
            cuvs_cfg = to_cuvs_knowhere_config(static_cast<const knowhere_config_type&>(*cfg));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << e.what();
            result = Status::invalid_args;
        }
        if (index_.is_trained()) {
            result = Status::index_already_trained;
        }
        if (result == Status::success) {
            auto rows = dataset->GetRows();
            auto dim = dataset->GetDim();
            auto const* data = reinterpret_cast<const data_type*>(dataset->GetTensor());
            try {
                index_.train(cuvs_cfg, data, rows, dim);
                index_.synchronize(true);
            } catch (const std::exception& e) {
                LOG_KNOWHERE_ERROR_ << e.what();
                result = Status::cuvs_inner_error;
            }
        }
        return result;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        auto result = Status::success;
        auto cuvs_cfg = cuvs_knowhere::cuvs_knowhere_config{};
        auto err_msg = std::string{};
        try {
            cuvs_cfg = to_cuvs_knowhere_config(static_cast<const knowhere_config_type&>(*cfg));
        } catch (const std::exception& e) {
            err_msg = std::string{e.what()};
            LOG_KNOWHERE_ERROR_ << e.what();
            result = Status::invalid_args;
        }
        if (result == Status::success) {
            try {
                auto rows = dataset->GetRows();
                auto dim = dataset->GetDim();
                auto const* data = reinterpret_cast<const data_type*>(dataset->GetTensor());
                auto search_result =
                    index_.search(cuvs_cfg, data, rows, dim, bitset.data(), bitset.byte_size(), bitset.size());
                std::this_thread::yield();
                index_.synchronize();
                return GenResultDataSet(rows, cuvs_cfg.k, std::get<0>(search_result), std::get<1>(search_result));
            } catch (const std::exception& e) {
                err_msg = std::string{e.what()};
                LOG_KNOWHERE_ERROR_ << e.what();
                result = Status::cuvs_inner_error;
            }
        }
        return expected<DataSetPtr>::Err(result, err_msg.c_str());
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "RangeSearch not implemented");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    Status
    Serialize(BinarySet& binset) const override {
        auto result = Status::success;
        std::stringbuf buf;
        if (!index_.is_trained()) {
            result = Status::empty_index;
        } else {
            std::ostream os(&buf);

            try {
                index_.serialize(os);
                index_.synchronize(true);
            } catch (const std::exception& e) {
                LOG_KNOWHERE_ERROR_ << e.what();
                result = Status::cuvs_inner_error;
            }
            os.flush();
        }
        if (result == Status::success) {
            std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[buf.str().size()]);
            memcpy(index_binary.get(), buf.str().c_str(), buf.str().size());
            binset.Append(this->Type(), index_binary, buf.str().size());
        }
        return result;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config>) override {
        auto result = Status::success;
        std::stringbuf buf;
        auto binary = binset.GetByName(this->Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            result = Status::invalid_binary_set;
        } else {
            buf.sputn((char*)binary->data.get(), binary->size);
            std::istream is(&buf);
            result = DeserializeFromStream(is);
        }
        return result;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config>) override {
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<knowhere_config_type>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    int64_t
    Dim() const override {
        return index_.dim();
    }

    int64_t
    Size() const override {
        return 0;
    }

    int64_t
    Count() const override {
        return index_.size();
    }

    std::string
    Type() const override {
        if constexpr (index_kind == cuvs_proto::cuvs_index_kind::brute_force) {
            return knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE;
        } else if constexpr (index_kind == cuvs_proto::cuvs_index_kind::ivf_flat) {
            return knowhere::IndexEnum::INDEX_CUVS_IVFFLAT;
        } else if constexpr (index_kind == cuvs_proto::cuvs_index_kind::ivf_pq) {
            return knowhere::IndexEnum::INDEX_CUVS_IVFPQ;
        } else if constexpr (index_kind == cuvs_proto::cuvs_index_kind::cagra) {
            return knowhere::IndexEnum::INDEX_CUVS_CAGRA;
        }
    }

    using cuvs_knowhere_index_type = typename cuvs_knowhere::cuvs_knowhere_index<K, DataType>;

 protected:
    cuvs_knowhere_index_type index_;

    Status
    DeserializeFromStream(std::istream& stream) {
        auto result = Status::success;
        try {
            index_ = cuvs_knowhere_index_type::deserialize(stream);
            index_.synchronize(true);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << e.what();
            result = Status::cuvs_inner_error;
        }
        stream.sync();
        return result;
    }
};

template <typename DataType>
using GpuCuvsBruteForceIndexNode = GpuCuvsIndexNode<DataType, cuvs_proto::cuvs_index_kind::brute_force>;
template <typename DataType>
using GpuCuvsIvfFlatIndexNode = GpuCuvsIndexNode<DataType, cuvs_proto::cuvs_index_kind::ivf_flat>;
template <typename DataType>
using GpuCuvsIvfPqIndexNode = GpuCuvsIndexNode<DataType, cuvs_proto::cuvs_index_kind::ivf_pq>;
template <typename DataType>
using GpuCuvsCagraIndexNode = GpuCuvsIndexNode<DataType, cuvs_proto::cuvs_index_kind::cagra>;

}  // namespace knowhere

#endif /* GPU_CUVS_H */
