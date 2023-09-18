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

#ifndef IVF_RAFT_CUH
#define IVF_RAFT_CUH

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <raft/neighbors/specializations.cuh>

#include "common/raft/raft_utils.h"
#include "common/raft_metric.h"
#include "fmt/core.h"
#include "index/ivf_raft/ivf_raft_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/device_bitset.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "raft/core/device_resources.hpp"
#include "thrust/execution_policy.h"
#include "thrust/logical.h"
#include "thrust/sequence.h"

namespace knowhere {

namespace raft_detail {
struct raft_results {
    raft_results(raft::device_resources& res)
        : ids_{raft::make_device_matrix<std::int64_t, std::int64_t>(res, 0, 0)},
          dists_{raft::make_device_matrix<float, std::int64_t>(res, 0, 0)} {
    }
    raft_results(raft::device_resources& res, std::int64_t rows, std::int64_t k)
        : ids_{raft::make_device_matrix<std::int64_t, std::int64_t>(res, rows, k)},
          dists_{raft::make_device_matrix<float, std::int64_t>(res, rows, k)} {
    }
    auto
    ids() {
        return ids_.view();
    }
    auto
    dists() {
        return dists_.view();
    }
    auto
    ids_data() {
        return ids_.data_handle();
    }
    auto
    dists_data() {
        return dists_.data_handle();
    }

    auto
    rows() const {
        return ids_.extent(0);
    }
    auto
    k() const {
        return ids_.extent(1);
    }

 private:
    raft::device_matrix<std::int64_t, std::int64_t> ids_;
    raft::device_matrix<float, std::int64_t> dists_;
};

auto constexpr static const MAX_IVF_FLAT_K = 256;

__global__ void
postprocess_device_results(bool* enough_valid, int64_t* ids, float* dists, int64_t rows, int k, int target_k,
                           DeviceBitsetView bitset) {
    __shared__ int invalid_count[1];
    for (auto row_index = blockIdx.x; row_index < rows; row_index += gridDim.x) {
        if (threadIdx.x == 0) {
            invalid_count[0] = 0;
        }
        __syncthreads();
        // First, replace all invalid IDs with -1
        for (auto col_index = threadIdx.x; col_index < k; col_index += blockDim.x) {
            auto elem_index = row_index * k + col_index;
            auto cur_id = ids[elem_index];
            auto invalid_id = bitset.test(cur_id);
            if (invalid_id) {  // TODO(wphicks): assuming the branch is worth it here;
                               // should analyze perf.
                atomicAdd_block(invalid_count, 1);
            }
            invalid_id |= cur_id == std::numeric_limits<int64_t>::max();
            ids[elem_index] = int(invalid_id) * -1 + int(!invalid_id) * ids[elem_index];
        }
        __syncthreads();

        // Now move all valid results to the front of the row
        auto cur_valid_index = row_index * k;
        for (auto col_index = 0; col_index < k; ++col_index) {
            auto elem_index = row_index * k + col_index;
            // Just do one row per block for now; can improve this later
            if (ids[elem_index] != -1 && threadIdx.x == 0) {
                // Swap valid id to an earlier place in the row
                ids[cur_valid_index] = ids[elem_index];
                if (elem_index != cur_valid_index) {
                    ids[elem_index] = -1;
                    // Only count elements invalidated by the bitset. These are the
                    // elements that will require a swap as opposed to just appearing at
                    // the end of the row
                }
                dists[cur_valid_index++] = dists[elem_index];
            }
        }
        // Check if we have enough valid results
        if (threadIdx.x == 0) {
            enough_valid[row_index] = (k - invalid_count[0] >= target_k);
        }
    }
}

/* The following kernel is used to copy data from one row-major 2D array to
 * another. If the input array has more columns than the output array, the
 * rows will be truncated to the length of the output array. Similarly, the
 * number of rows will be truncated if the output has fewer rows. If the output
 * array has more rows/columns than the input, the output will be padded with
 * entries of -1. */
template <typename T, typename IndexT>
__global__ void
slice(T* out, T* in, IndexT out_rows, IndexT out_cols, IndexT in_rows, IndexT in_cols) {
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < out_rows * out_cols; i += blockDim.x * gridDim.x) {
        auto row_index = i / out_cols;
        auto col_index = i % out_cols;
        if (row_index < in_rows && col_index < in_cols) {
            out[i] = in[row_index * in_cols + col_index];
        } else {
            out[i] = -1;
        }
    }
}

}  // namespace raft_detail

namespace detail {
using raft_ivf_flat_index = raft::neighbors::ivf_flat::index<float, std::int64_t>;
using raft_ivf_pq_index = raft::neighbors::ivf_pq::index<std::int64_t>;

namespace codebook {
auto static constexpr const PER_SUBSPACE = "PER_SUBSPACE";
auto static constexpr const PER_CLUSTER = "PER_CLUSTER";
}  // namespace codebook

inline expected<raft::neighbors::ivf_pq::codebook_gen>
str_to_codebook_gen(std::string const& str) {
    static const std::unordered_map<std::string, raft::neighbors::ivf_pq::codebook_gen> name_map = {
        {codebook::PER_SUBSPACE, raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE},
        {codebook::PER_CLUSTER, raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER},
    };

    auto it = name_map.find(str);
    if (it == name_map.end())
        return expected<raft::neighbors::ivf_pq::codebook_gen>::Err(Status::invalid_args,
                                                                    fmt::format("invalid arguments: {}", str));
    return it->second;
}

namespace cuda_type {
auto static constexpr const CUDA_R_16F = "CUDA_R_16F";
auto static constexpr const CUDA_C_16F = "CUDA_C_16F";
auto static constexpr const CUDA_R_16BF = "CUDA_R_16BF";
auto static constexpr const CUDA_C_16BF = "CUDA_C_16BF";
auto static constexpr const CUDA_R_32F = "CUDA_R_32F";
auto static constexpr const CUDA_C_32F = "CUDA_C_32F";
auto static constexpr const CUDA_R_64F = "CUDA_R_64F";
auto static constexpr const CUDA_C_64F = "CUDA_C_64F";
auto static constexpr const CUDA_R_8I = "CUDA_R_8I";
auto static constexpr const CUDA_C_8I = "CUDA_C_8I";
auto static constexpr const CUDA_R_8U = "CUDA_R_8U";
auto static constexpr const CUDA_C_8U = "CUDA_C_8U";
auto static constexpr const CUDA_R_32I = "CUDA_R_32I";
auto static constexpr const CUDA_C_32I = "CUDA_C_32I";
auto static constexpr const CUDA_R_8F_E4M3 = "CUDA_R_8F_E4M3";
auto static constexpr const CUDA_R_8F_E5M2 = "CUDA_R_8F_E5M2";
}  // namespace cuda_type

inline expected<cudaDataType_t>
str_to_cuda_dtype(std::string const& str) {
    static const std::unordered_map<std::string, cudaDataType_t> name_map = {
        {cuda_type::CUDA_R_16F, CUDA_R_16F},   {cuda_type::CUDA_C_16F, CUDA_C_16F},
        {cuda_type::CUDA_R_16BF, CUDA_R_16BF}, {cuda_type::CUDA_C_16BF, CUDA_C_16BF},
        {cuda_type::CUDA_R_32F, CUDA_R_32F},   {cuda_type::CUDA_C_32F, CUDA_C_32F},
        {cuda_type::CUDA_R_64F, CUDA_R_64F},   {cuda_type::CUDA_C_64F, CUDA_C_64F},
        {cuda_type::CUDA_R_8I, CUDA_R_8I},     {cuda_type::CUDA_C_8I, CUDA_C_8I},
        {cuda_type::CUDA_R_8U, CUDA_R_8U},     {cuda_type::CUDA_C_8U, CUDA_C_8U},
        {cuda_type::CUDA_R_32I, CUDA_R_32I},   {cuda_type::CUDA_C_32I, CUDA_C_32I},
        // not support, when we use cuda 11.6
        //{cuda_type::CUDA_R_8F_E4M3, CUDA_R_8F_E4M3}, {cuda_type::CUDA_R_8F_E5M2, CUDA_R_8F_E5M2},

    };

    auto it = name_map.find(str);
    if (it == name_map.end())
        return expected<cudaDataType_t>::Err(Status::invalid_args, fmt::format("invalid arguments: {}", str));
    return it->second;
}

}  // namespace detail

template <typename T>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<detail::raft_ivf_flat_index> {
    typedef RaftIvfFlatConfig Type;
};
template <>
struct KnowhereConfigType<detail::raft_ivf_pq_index> {
    typedef RaftIvfPqConfig Type;
};

template <typename T>
class RaftIvfIndexNode : public IndexNode {
 public:
    RaftIvfIndexNode(const Object& object) : device_id_{-1}, gpu_index_{} {
    }

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        auto ivf_raft_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);
        if (gpu_index_) {
            LOG_KNOWHERE_WARNING_ << "index is already trained";
            return Status::index_already_trained;
        }
        try {
            RANDOM_CHOOSE_DEVICE_WITH_ASSIGN(this->device_id_);
            raft_utils::device_setter with_this_device(this->device_id_);
            raft_utils::init_gpu_resources();

            auto metric = Str2RaftMetricType(ivf_raft_cfg.metric_type.value());
            if (!metric.has_value()) {
                LOG_KNOWHERE_WARNING_ << "please check metric value: " << ivf_raft_cfg.metric_type.value();
                return metric.error();
            }
            if (metric.value() != raft::distance::DistanceType::L2Expanded &&
                metric.value() != raft::distance::DistanceType::InnerProduct) {
                LOG_KNOWHERE_WARNING_ << "selected metric not supported in RAFT IVF indexes: "
                                      << ivf_raft_cfg.metric_type.value();
                return Status::invalid_metric_type;
            }
            auto& res = raft_utils::get_raft_resources();

            auto rows = dataset.GetRows();
            auto dim = dataset.GetDim();
            auto* data = reinterpret_cast<float const*>(dataset.GetTensor());

            auto data_gpu = raft::make_device_matrix<float, std::int64_t>(res, rows, dim);
            RAFT_CUDA_TRY(cudaMemcpyAsync(data_gpu.data_handle(), data, data_gpu.size() * sizeof(float),
                                          cudaMemcpyDefault, res.get_stream().value()));
            if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
                auto build_params = raft::neighbors::ivf_flat::index_params{};
                build_params.metric = metric.value();
                build_params.n_lists = ivf_raft_cfg.nlist.value();
                build_params.kmeans_n_iters = ivf_raft_cfg.kmeans_n_iters.value();
                build_params.kmeans_trainset_fraction = ivf_raft_cfg.kmeans_trainset_fraction.value();
                build_params.adaptive_centers = ivf_raft_cfg.adaptive_centers.value();
                gpu_index_ = raft::neighbors::ivf_flat::build<float, std::int64_t>(res, build_params, data_gpu.view());
            } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
                auto build_params = raft::neighbors::ivf_pq::index_params{};
                build_params.metric = metric.value();
                build_params.n_lists = ivf_raft_cfg.nlist.value();
                build_params.pq_bits = ivf_raft_cfg.nbits.value();
                build_params.kmeans_n_iters = ivf_raft_cfg.kmeans_n_iters.value();
                build_params.kmeans_trainset_fraction = ivf_raft_cfg.kmeans_trainset_fraction.value();
                build_params.pq_dim = ivf_raft_cfg.m.value();
                auto codebook_kind = detail::str_to_codebook_gen(ivf_raft_cfg.codebook_kind.value());
                if (!codebook_kind.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check codebook kind: " << ivf_raft_cfg.codebook_kind.value();
                    return codebook_kind.error();
                }
                build_params.codebook_kind = codebook_kind.value();
                build_params.force_random_rotation = ivf_raft_cfg.force_random_rotation.value();
                gpu_index_ = raft::neighbors::ivf_pq::build<float, std::int64_t>(res, build_params, data_gpu.view());
            }
            dim_ = dim;
            counts_ = rows;
            res.sync_stream();

        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
            return Status::raft_inner_error;
        }
        return Status::success;
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        if (!gpu_index_) {
            return Status::index_not_trained;
        }
        try {
            RAFT_EXPECTS(this->device_id_ != -1, "call data add before index train.");
            raft_utils::device_setter with_this_device{this->device_id_};
            auto rows = dataset.GetRows();
            auto dim = dataset.GetDim();
            auto* data = reinterpret_cast<float const*>(dataset.GetTensor());

            raft_utils::init_gpu_resources();
            auto& res = raft_utils::get_raft_resources();

            // TODO(wphicks): Clean up transfer with raft
            // buffer objects when available
            auto data_gpu = raft::make_device_matrix<float, std::int64_t>(res, rows, dim);
            RAFT_CUDA_TRY(cudaMemcpyAsync(data_gpu.data_handle(), data, data_gpu.size() * sizeof(float),
                                          cudaMemcpyDefault, res.get_stream().value()));

            auto indices = rmm::device_uvector<std::int64_t>(rows, res.get_stream());
            thrust::sequence(res.get_thrust_policy(), indices.begin(), indices.end(), gpu_index_->size());

            if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
                raft::neighbors::ivf_flat::extend<float, std::int64_t>(
                    res, raft::make_const_mdspan(data_gpu.view()),
                    std::make_optional(
                        raft::make_device_vector_view<const std::int64_t, std::int64_t>(indices.data(), rows)),
                    gpu_index_.value());
            } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
                raft::neighbors::ivf_pq::extend<float, std::int64_t>(
                    res, raft::make_const_mdspan(data_gpu.view()),
                    std::make_optional(
                        raft::make_device_matrix_view<const std::int64_t, std::int64_t>(indices.data(), rows, 1)),
                    gpu_index_.value());
            } else {
                static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
            }
            dim_ = dim;
            counts_ = rows;
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
            return Status::raft_inner_error;
        }

        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto ivf_raft_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
        auto output_size = rows * ivf_raft_cfg.k.value();
        auto ids = std::unique_ptr<std::int64_t[]>(new std::int64_t[output_size]);
        auto dis = std::unique_ptr<float[]>(new float[output_size]);
        try {
            RAFT_EXPECTS(this->device_id_ != -1, "device id is -1, when call search");
            raft_utils::device_setter with_this_device{this->device_id_};
            auto& res_ = raft_utils::get_raft_resources();

            // TODO(wphicks): Clean up transfer with raft
            // buffer objects when available
            auto data_gpu = raft::make_device_matrix<float, std::int64_t>(res_, rows, dim);
            raft::copy(data_gpu.data_handle(), data, data_gpu.size(), res_.get_stream());

            auto gpu_results = raft_detail::raft_results{res_};
            auto gpu_bitset = DeviceBitset{res_, bitset};

            if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
                auto search_params = raft::neighbors::ivf_flat::search_params{};
                search_params.n_probes = std::min<uint32_t>(ivf_raft_cfg.nprobe.value(), gpu_index_->n_lists());
                auto search_k = std::min(
                    ivf_raft_cfg.k.value() + (bitset.count() * ivf_raft_cfg.k.value() / counts_),
                    std::min(static_cast<uint64_t>(counts_), static_cast<uint64_t>(raft_detail::MAX_IVF_FLAT_K)));
                gpu_results = RawSearch(res_, raft::make_const_mdspan(data_gpu.view()), search_params, search_k,
                                        ivf_raft_cfg.k.value(), gpu_bitset.view());
            } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
                auto search_params = raft::neighbors::ivf_pq::search_params{};
                search_params.n_probes = std::min<uint32_t>(ivf_raft_cfg.nprobe.value(), gpu_index_->n_lists());
                auto lut_dtype = detail::str_to_cuda_dtype(ivf_raft_cfg.lut_dtype.value());
                if (!lut_dtype.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check lookup dtype: " << ivf_raft_cfg.lut_dtype.value();
                    return expected<DataSetPtr>::Err(
                        lut_dtype.error(), fmt::format("invalid lookup dtype: {}", ivf_raft_cfg.lut_dtype.value()));
                }
                if (lut_dtype.value() != CUDA_R_32F && lut_dtype.value() != CUDA_R_16F &&
                    lut_dtype.value() != CUDA_R_8U) {
                    LOG_KNOWHERE_WARNING_ << "selected lookup dtype not supported: " << ivf_raft_cfg.lut_dtype.value();
                    return expected<DataSetPtr>::Err(Status::invalid_args,
                                                     fmt::format("invalid lookup dtype: {}", lut_dtype.value()));
                }
                search_params.lut_dtype = lut_dtype.value();
                auto internal_distance_dtype = detail::str_to_cuda_dtype(ivf_raft_cfg.internal_distance_dtype.value());
                if (!internal_distance_dtype.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check internal distance dtype: "
                                          << ivf_raft_cfg.internal_distance_dtype.value();
                    return expected<DataSetPtr>::Err(internal_distance_dtype.error(),
                                                     fmt::format("invalid internal distance dtype: {}",
                                                                 ivf_raft_cfg.internal_distance_dtype.value()));
                }
                if (internal_distance_dtype.value() != CUDA_R_32F && internal_distance_dtype.value() != CUDA_R_16F) {
                    LOG_KNOWHERE_WARNING_ << "selected internal distance dtype not supported: "
                                          << ivf_raft_cfg.internal_distance_dtype.value();
                    return expected<DataSetPtr>::Err(
                        Status::invalid_args, fmt::format("invalid lookup dtype: {}", internal_distance_dtype.value()));
                }
                search_params.internal_distance_dtype = internal_distance_dtype.value();
                search_params.preferred_shmem_carveout = search_params.preferred_shmem_carveout;
                gpu_results = RawSearch(res_, raft::make_const_mdspan(data_gpu.view()), search_params,
                                        ivf_raft_cfg.k.value(), ivf_raft_cfg.k.value(), gpu_bitset.view());
            } else {
                static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
            }

            if (gpu_results.k() != ivf_raft_cfg.k.value()) {
                auto new_gpu_results = raft_detail::raft_results{res_, gpu_results.rows(), ivf_raft_cfg.k.value()};
                raft_detail::slice<<<1024, 256, 0, res_.get_stream().value()>>>(
                    new_gpu_results.ids_data(), gpu_results.ids_data(), new_gpu_results.rows(), new_gpu_results.k(),
                    gpu_results.rows(), gpu_results.k());
                res_.sync_stream();
                gpu_results = new_gpu_results;
            }
            raft::copy(ids.get(), gpu_results.ids_data(), output_size, res_.get_stream());
            raft::copy(dis.get(), gpu_results.dists_data(), output_size, res_.get_stream());
            res_.sync_stream();

        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
            return expected<DataSetPtr>::Err(Status::raft_inner_error, e.what());
        }

        return GenResultDataSet(rows, ivf_raft_cfg.k.value(), ids.release(), dis.release());
    }

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "RangeSearch not implemented");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            return !IsMetricType(metric_type, metric::COSINE);
        }
        if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            return false;
        }
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!gpu_index_.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty RaftIvfIndex.";
            return Status::empty_index;
        }

        RAFT_EXPECTS(this->device_id_ != -1, "index serialize before trained.");
        std::stringbuf buf;

        std::ostream os(&buf);

        os.write((char*)(&this->dim_), sizeof(this->dim_));
        os.write((char*)(&this->counts_), sizeof(this->counts_));
        os.write((char*)(&this->device_id_), sizeof(this->device_id_));

        raft_utils::device_setter with_this_device{device_id_};
        auto& res = raft_utils::get_raft_resources();

        if constexpr (std::is_same_v<T, detail::raft_ivf_flat_index>) {
            raft::neighbors::ivf_flat::serialize<float, std::int64_t>(res, os, *gpu_index_);
        }
        if constexpr (std::is_same_v<T, detail::raft_ivf_pq_index>) {
            raft::neighbors::ivf_pq::serialize<std::int64_t>(res, os, *gpu_index_);
        }
        res.sync_stream();

        os.flush();
        std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[buf.str().size()]);

        memcpy(index_binary.get(), buf.str().c_str(), buf.str().size());
        binset.Append(this->Type(), index_binary, buf.str().size());
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        std::stringbuf buf;
        auto binary = binset.GetByName(this->Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }
        buf.sputn((char*)binary->data.get(), binary->size);
        std::istream is(&buf);

        is.read((char*)(&this->dim_), sizeof(this->dim_));
        is.read((char*)(&this->counts_), sizeof(this->counts_));
        // device_id from binset is useless, will gen device id from global
        // status
        is.read((char*)(&this->device_id_), sizeof(this->device_id_));
        MIN_LOAD_CHOOSE_DEVICE_WITH_ASSIGN(this->device_id_, binary->size);
        raft_utils::device_setter with_this_device{this->device_id_};

        raft_utils::init_gpu_resources();
        auto& res = raft_utils::get_raft_resources();

        if constexpr (std::is_same_v<T, detail::raft_ivf_flat_index>) {
            T index_ = raft::neighbors::ivf_flat::deserialize<float, std::int64_t>(res, is);
            res.sync_stream();
            is.sync();
            gpu_index_ = T(std::move(index_));
        }
        if constexpr (std::is_same_v<T, detail::raft_ivf_pq_index>) {
            T index_ = raft::neighbors::ivf_pq::deserialize<std::int64_t>(res, is);
            res.sync_stream();
            is.sync();
            gpu_index_ = T(std::move(index_));
        }

        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) {
        LOG_KNOWHERE_ERROR_ << "RaftIvfIndex doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<typename KnowhereConfigType<T>::Type>();
    }

    int64_t
    Dim() const override {
        return dim_;
    }

    int64_t
    Size() const override {
        return 0;
    }

    int64_t
    Count() const override {
        return counts_;
    }

    std::string
    Type() const override {
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            return knowhere::IndexEnum::INDEX_RAFT_IVFFLAT;
        }
        if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            return knowhere::IndexEnum::INDEX_RAFT_IVFPQ;
        }
    }

 private:
    int device_id_ = -1;
    int64_t dim_ = 0;
    int64_t counts_ = 0;
    std::optional<T> gpu_index_;

    template <typename raft_search_params_t>
    raft_detail::raft_results
    RawSearch(raft::device_resources& res, raft::device_matrix_view<const float, std::int64_t> queries,
              raft_search_params_t const& search_params, int k, int target_k, DeviceBitsetView const& bitset) const {
        auto max_k = counts_;
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            k = std::min(k, raft_detail::MAX_IVF_FLAT_K);
            max_k = std::min(max_k, int64_t{raft_detail::MAX_IVF_FLAT_K});
        }
        auto result = raft_detail::raft_results{res, queries.extent(0), k};
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            raft::neighbors::ivf_flat::search<float, std::int64_t>(res, search_params, *gpu_index_, queries,
                                                                   result.ids(), result.dists());
        } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            raft::neighbors::ivf_pq::search<float, std::int64_t>(res, search_params, *gpu_index_, queries, result.ids(),
                                                                 result.dists());
        } else {
            static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
        }

        auto blocks = std::min(queries.extent(0), int64_t{1024});
        auto threads = std::min(k, 256);
        auto warp_remainder = threads % 32;
        if (warp_remainder != 0) {
            threads += (32 - warp_remainder);
        }
        auto enough_valid = raft::make_device_vector<bool>(res, queries.extent(0));

        raft_detail::postprocess_device_results<<<blocks, threads, 0, res.get_stream().value()>>>(
            enough_valid.data_handle(), result.ids_data(), result.dists_data(), queries.extent(0), k, target_k, bitset);

        if (k < max_k && !thrust::all_of(res.get_thrust_policy(), enough_valid.data_handle(),
                                         enough_valid.data_handle() + queries.extent(0), thrust::identity<bool>())) {
            result = RawSearch(res, queries, search_params, std::min(int64_t{k * 2}, max_k), target_k, bitset);
        }
        return result;
    }
};

}  // namespace knowhere
#endif /* IVF_RAFT_CUH */
