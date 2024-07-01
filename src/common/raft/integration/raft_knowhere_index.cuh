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
#pragma once
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <raft/core/bitset.cuh>
#include <raft/core/copy.cuh>
#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/serialize.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/sample_filter.cuh>
#include <tuple>
#include <type_traits>

#include "common/raft/integration/raft_knowhere_index.hpp"
#include "common/raft/proto/raft_index.cuh"
#include "common/raft/proto/raft_index_kind.hpp"

namespace raft_knowhere {
namespace detail {

// This helper struct maps the generic type of RAFT index to the specific
// instantiation of that index used within knowhere.
template <bool B, raft_proto::raft_index_kind IndexKind>
struct raft_index_type_mapper : std::false_type {};

template <>
struct raft_index_type_mapper<true, raft_proto::raft_index_kind::brute_force> : std::true_type {
    using data_type = raft_data_t<raft_proto::raft_index_kind::brute_force>;
    using indexing_type = raft_indexing_t<raft_proto::raft_index_kind::brute_force>;
    using type = raft_proto::raft_index<raft::neighbors::brute_force::index, data_type>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};
template <>
struct raft_index_type_mapper<true, raft_proto::raft_index_kind::ivf_flat> : std::true_type {
    using data_type = raft_data_t<raft_proto::raft_index_kind::ivf_flat>;
    using indexing_type = raft_indexing_t<raft_proto::raft_index_kind::ivf_flat>;
    using type = raft_proto::raft_index<raft::neighbors::ivf_flat::index, data_type, indexing_type>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};
template <>
struct raft_index_type_mapper<true, raft_proto::raft_index_kind::ivf_pq> : std::true_type {
    using data_type = raft_data_t<raft_proto::raft_index_kind::ivf_pq>;
    using indexing_type = raft_indexing_t<raft_proto::raft_index_kind::ivf_pq>;
    using type = raft_proto::raft_index<raft::neighbors::ivf_pq::index, indexing_type>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};
template <>
struct raft_index_type_mapper<true, raft_proto::raft_index_kind::cagra> : std::true_type {
    using data_type = raft_data_t<raft_proto::raft_index_kind::cagra>;
    using indexing_type = raft_indexing_t<raft_proto::raft_index_kind::cagra>;
    using type = raft_proto::raft_index<raft::neighbors::cagra::index, data_type, indexing_type>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};

template <typename T, typename U, typename V>
struct check_valid_entry {
    __device__ __host__
    check_valid_entry(U max_distance, V max_id)
        : max_distance_(max_distance), max_id_(max_id) {
    }
    __device__ auto
    operator()(T id_distance) {
        auto id = thrust::get<0>(id_distance);
        auto distance = thrust::get<1>(id_distance);
        return distance >= max_distance_ || distance < 0 || id >= max_id_;
    }

 private:
    U max_distance_;
    V max_id_;
};

}  // namespace detail

template <raft_proto::raft_index_kind IndexKind>
using raft_index_t = typename detail::raft_index_type_mapper<true, IndexKind>::type;

template <raft_proto::raft_index_kind IndexKind>
using raft_index_params_t = typename detail::raft_index_type_mapper<true, IndexKind>::index_params_type;
template <raft_proto::raft_index_kind IndexKind>
using raft_search_params_t = typename detail::raft_index_type_mapper<true, IndexKind>::search_params_type;

// Metrics are passed between knowhere and RAFT as strings to avoid tight
// coupling between the implementation details of either one.
[[nodiscard]] inline auto
metric_string_to_raft_distance_type(std::string const& metric_string) {
    auto result = raft::distance::DistanceType::L2Expanded;
    if (metric_string == "L2") {
        result = raft::distance::DistanceType::L2Expanded;
    } else if (metric_string == "L2SqrtExpanded") {
        result = raft::distance::DistanceType::L2SqrtExpanded;
    } else if (metric_string == "CosineExpanded") {
        result = raft::distance::DistanceType::CosineExpanded;
    } else if (metric_string == "L1") {
        result = raft::distance::DistanceType::L1;
    } else if (metric_string == "L2Unexpanded") {
        result = raft::distance::DistanceType::L2Unexpanded;
    } else if (metric_string == "L2SqrtUnexpanded") {
        result = raft::distance::DistanceType::L2SqrtUnexpanded;
    } else if (metric_string == "IP") {
        result = raft::distance::DistanceType::InnerProduct;
    } else if (metric_string == "Linf") {
        result = raft::distance::DistanceType::Linf;
    } else if (metric_string == "Canberra") {
        result = raft::distance::DistanceType::Canberra;
    } else if (metric_string == "LpUnexpanded") {
        result = raft::distance::DistanceType::LpUnexpanded;
    } else if (metric_string == "CorrelationExpanded") {
        result = raft::distance::DistanceType::CorrelationExpanded;
    } else if (metric_string == "JACCARD") {
        result = raft::distance::DistanceType::JaccardExpanded;
    } else if (metric_string == "HellingerExpanded") {
        result = raft::distance::DistanceType::HellingerExpanded;
    } else if (metric_string == "Haversine") {
        result = raft::distance::DistanceType::Haversine;
    } else if (metric_string == "BrayCurtis") {
        result = raft::distance::DistanceType::BrayCurtis;
    } else if (metric_string == "JensenShannon") {
        result = raft::distance::DistanceType::JensenShannon;
    } else if (metric_string == "HAMMING") {
        result = raft::distance::DistanceType::HammingUnexpanded;
    } else if (metric_string == "KLDivergence") {
        result = raft::distance::DistanceType::KLDivergence;
    } else if (metric_string == "RusselRaoExpanded") {
        result = raft::distance::DistanceType::RusselRaoExpanded;
    } else if (metric_string == "DiceExpanded") {
        result = raft::distance::DistanceType::DiceExpanded;
    } else if (metric_string == "Precomputed") {
        result = raft::distance::DistanceType::Precomputed;
    } else {
        RAFT_FAIL("Unrecognized metric type %s", metric_string.c_str());
    }
    return result;
}

[[nodiscard]] inline auto
codebook_string_to_raft_codebook_gen(std::string const& codebook_string) {
    auto result = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    if (codebook_string == "PER_SUBSPACE") {
        result = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    } else if (codebook_string == "PER_CLUSTER") {
        result = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
    } else {
        RAFT_FAIL("Unrecognized codebook type %s", codebook_string.c_str());
    }
    return result;
}
[[nodiscard]] inline auto
build_algo_string_to_cagra_build_algo(std::string const& algo_string) {
    auto result = raft::neighbors::cagra::graph_build_algo::IVF_PQ;
    if (algo_string == "IVF_PQ") {
        result = raft::neighbors::cagra::graph_build_algo::IVF_PQ;
    } else if (algo_string == "NN_DESCENT") {
        result = raft::neighbors::cagra::graph_build_algo::NN_DESCENT;
    } else {
        RAFT_FAIL("Unrecognized CAGRA build algo %s", algo_string.c_str());
    }
    return result;
}

[[nodiscard]] inline auto
search_algo_string_to_cagra_search_algo(std::string const& algo_string) {
    auto result = raft::neighbors::cagra::search_algo::AUTO;
    if (algo_string == "SINGLE_CTA") {
        result = raft::neighbors::cagra::search_algo::SINGLE_CTA;
    } else if (algo_string == "MULTI_CTA") {
        result = raft::neighbors::cagra::search_algo::MULTI_CTA;
    } else if (algo_string == "MULTI_KERNEL") {
        result = raft::neighbors::cagra::search_algo::MULTI_KERNEL;
    } else if (algo_string == "AUTO") {
        result = raft::neighbors::cagra::search_algo::AUTO;
    } else {
        RAFT_FAIL("Unrecognized CAGRA search algo %s", algo_string.c_str());
    }
    return result;
}

[[nodiscard]] inline auto
hashmap_mode_string_to_cagra_hashmap_mode(std::string const& mode_string) {
    auto result = raft::neighbors::cagra::hash_mode::AUTO;
    if (mode_string == "HASH") {
        result = raft::neighbors::cagra::hash_mode::HASH;
    } else if (mode_string == "SMALL") {
        result = raft::neighbors::cagra::hash_mode::SMALL;
    } else if (mode_string == "AUTO") {
        result = raft::neighbors::cagra::hash_mode::AUTO;
    } else {
        RAFT_FAIL("Unrecognized CAGRA hash mode %s", mode_string.c_str());
    }
    return result;
}

[[nodiscard]] inline auto
dtype_string_to_cuda_dtype(std::string const& dtype_string) {
    auto result = CUDA_R_32F;
    if (dtype_string == "CUDA_R_16F") {
        result = CUDA_R_16F;
    } else if (dtype_string == "CUDA_C_16F") {
        result = CUDA_C_16F;
    } else if (dtype_string == "CUDA_R_16BF") {
        result = CUDA_R_16BF;
    } else if (dtype_string == "CUDA_R_32F") {
        result = CUDA_R_32F;
    } else if (dtype_string == "CUDA_C_32F") {
        result = CUDA_C_32F;
    } else if (dtype_string == "CUDA_R_64F") {
        result = CUDA_R_64F;
    } else if (dtype_string == "CUDA_C_64F") {
        result = CUDA_C_64F;
    } else if (dtype_string == "CUDA_R_8I") {
        result = CUDA_R_8I;
    } else if (dtype_string == "CUDA_C_8I") {
        result = CUDA_C_8I;
    } else if (dtype_string == "CUDA_R_8U") {
        result = CUDA_R_8U;
    } else if (dtype_string == "CUDA_C_8U") {
        result = CUDA_C_8U;
    } else if (dtype_string == "CUDA_R_32I") {
        result = CUDA_R_32I;
    } else if (dtype_string == "CUDA_C_32I") {
        result = CUDA_C_32I;
#if __CUDACC_VER_MAJOR__ >= 12
    } else if (dtype_string == "CUDA_R_8F_E4M3") {
        result = CUDA_R_8F_E4M3;
    } else if (dtype_string == "CUDA_R_8F_E5M2") {
        result = CUDA_R_8F_E5M2;
#endif
    } else {
        RAFT_FAIL("Unrecognized dtype %s", dtype_string.c_str());
    }
    return result;
}

// Given a generic config without RAFT symbols, convert to RAFT index build
// parameters
template <raft_proto::raft_index_kind IndexKind>
[[nodiscard]] auto
config_to_index_params(raft_knowhere_config const& raw_config) {
    RAFT_EXPECTS(raw_config.index_type == IndexKind, "Incorrect index type for this index");
    auto config = validate_raft_knowhere_config(raw_config);
    auto result = raft_index_params_t<IndexKind>{};

    result.metric = metric_string_to_raft_distance_type(config.metric_type);
    result.metric_arg = config.metric_arg;
    result.add_data_on_build = config.add_data_on_build;

    if constexpr (IndexKind == raft_proto::raft_index_kind::ivf_flat ||
                  IndexKind == raft_proto::raft_index_kind::ivf_pq) {
        result.n_lists = *(config.nlist);
        result.kmeans_n_iters = *(config.kmeans_n_iters);
        result.kmeans_trainset_fraction = *(config.kmeans_trainset_fraction);
        result.conservative_memory_allocation = *(config.conservative_memory_allocation);
    }
    if constexpr (IndexKind == raft_proto::raft_index_kind::ivf_flat) {
        result.adaptive_centers = *(config.adaptive_centers);
    }
    if constexpr (IndexKind == raft_proto::raft_index_kind::ivf_pq) {
        result.pq_dim = *(config.m);
        result.pq_bits = *(config.nbits);
        result.codebook_kind = codebook_string_to_raft_codebook_gen(*(config.codebook_kind));
        result.force_random_rotation = *(config.force_random_rotation);
    }
    if constexpr (IndexKind == raft_proto::raft_index_kind::cagra) {
        result.intermediate_graph_degree = *(config.intermediate_graph_degree);
        result.graph_degree = *(config.graph_degree);
        result.build_algo = build_algo_string_to_cagra_build_algo(*(config.build_algo));
        result.nn_descent_niter = *(config.nn_descent_niter);
    }
    return result;
}

// Given a generic config without RAFT symbols, convert to RAFT index search
// parameters
template <raft_proto::raft_index_kind IndexKind>
[[nodiscard]] auto
config_to_search_params(raft_knowhere_config const& raw_config) {
    RAFT_EXPECTS(raw_config.index_type == IndexKind, "Incorrect index type for this index");
    auto config = validate_raft_knowhere_config(raw_config);
    auto result = raft_search_params_t<IndexKind>{};
    if constexpr (IndexKind == raft_proto::raft_index_kind::ivf_flat ||
                  IndexKind == raft_proto::raft_index_kind::ivf_pq) {
        result.n_probes = *(config.nprobe);
    }
    if constexpr (IndexKind == raft_proto::raft_index_kind::ivf_pq) {
        result.lut_dtype = dtype_string_to_cuda_dtype(*(config.lookup_table_dtype));
        result.internal_distance_dtype = dtype_string_to_cuda_dtype(*(config.internal_distance_dtype));
        result.preferred_shmem_carveout = *(config.preferred_shmem_carveout);
    }
    if constexpr (IndexKind == raft_proto::raft_index_kind::cagra) {
        result.max_queries = *(config.max_queries);
        result.itopk_size = *(config.itopk_size);
        result.max_iterations = *(config.max_iterations);
        result.algo = search_algo_string_to_cagra_search_algo(*(config.search_algo));
        result.team_size = *(config.team_size);
        result.search_width = *(config.search_width);
        result.min_iterations = *(config.min_iterations);
        result.thread_block_size = *(config.thread_block_size);
        result.hashmap_mode = hashmap_mode_string_to_cagra_hashmap_mode(*(config.hashmap_mode));
        result.hashmap_min_bitlen = *(config.hashmap_min_bitlen);
        result.hashmap_max_fill_rate = *(config.hashmap_max_fill_rate);
    }
    return result;
}

inline auto const&
get_device_resources_without_mempool(int device_id = raft::device_setter::get_current_device()) {
    auto thread_local res = std::vector<raft::device_resources>([]() {
        int device_count;
        RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
        return device_count;
    }());

    return res[device_id];
}

inline auto
select_device_id() {
    auto static device_count = []() {
        auto result = 0;
        RAFT_CUDA_TRY(cudaGetDeviceCount(&result));
        RAFT_EXPECTS(result != 0, "No CUDA devices found");
        return result;
    }();
    auto static index_counter = std::atomic<int>{0};
    // Use round-robin assignment to distribute indexes across devices
    auto result = index_counter.fetch_add(1) % device_count;
    return result;
}

// This struct is used to connect knowhere to a RAFT index. The implementation
// is provided here, but this header should never be directly included in
// another knowhere header. This ensures that RAFT symbols are not exposed in
// any knowhere header.
template <raft_proto::raft_index_kind IndexKind>
struct raft_knowhere_index<IndexKind>::impl {
    auto static constexpr index_kind = IndexKind;
    using data_type = raft_data_t<index_kind>;
    using indexing_type = raft_indexing_t<index_kind>;
    using input_indexing_type = raft_input_indexing_t<index_kind>;
    using raft_index_type = raft_index_t<index_kind>;

    impl() {
    }

    auto
    is_trained() const {
        return index_.has_value();
    }
    [[nodiscard]] auto
    size() const {
        auto result = std::int64_t{};
        if (is_trained()) {
            result = index_->size();
        }
        return result;
    }
    [[nodiscard]] auto
    dim() const {
        auto result = std::int64_t{};
        if (is_trained()) {
            result = index_->dim();
        }
        return result;
    }

    void
    train(raft_knowhere_config const& config, data_type const* data, knowhere_indexing_type row_count,
          knowhere_indexing_type feature_count) {
        auto scoped_device = raft::device_setter{device_id};
        auto index_params = config_to_index_params<index_kind>(config);
        if constexpr (index_kind == raft_proto::raft_index_kind::ivf_flat ||
                      index_kind == raft_proto::raft_index_kind::ivf_pq) {
            index_params.n_lists = std::min(knowhere_indexing_type(index_params.n_lists), row_count);
        }
        auto const& res = get_device_resources_without_mempool();
        auto host_data = raft::make_host_matrix_view(data, row_count, feature_count);
        if constexpr (index_kind == raft_proto::raft_index_kind::ivf_flat) {
            device_dataset_storage =
                raft::make_device_matrix<data_type, input_indexing_type>(res, row_count, feature_count);
            auto device_data = device_dataset_storage->view();
            raft::copy(res, device_data, host_data);
            index_ = raft_index_type::template build<data_type, indexing_type, input_indexing_type>(
                res, index_params, raft::make_const_mdspan(device_data));
            if (!config.cache_dataset_on_device) {
                device_dataset_storage = std::nullopt;
            }
        } else {
            if (config.cache_dataset_on_device) {
                device_dataset_storage =
                    raft::make_device_matrix<data_type, input_indexing_type>(res, row_count, feature_count);
                auto device_data = device_dataset_storage->view();
                raft::copy(res, device_data, host_data);
                index_ = raft_index_type::template build<data_type, indexing_type, input_indexing_type>(
                    res, index_params, raft::make_const_mdspan(device_data));
            } else {
                index_ = raft_index_type::template build<data_type, indexing_type, input_indexing_type>(
                    res, index_params, raft::make_const_mdspan(host_data));
            }
        }
    }

    void
    add(data_type const* data, knowhere_indexing_type row_count, knowhere_indexing_type feature_count,
        knowhere_indexing_type const* new_ids) {
        if constexpr (index_kind == raft_proto::raft_index_kind::brute_force) {
            if (index_) {
                RAFT_FAIL("RAFT brute force does not support adding vectors after training");
            }
        } else if constexpr (index_kind == raft_proto::raft_index_kind::cagra) {
            if (index_) {
                RAFT_FAIL("CAGRA does not support adding vectors after training");
            }
        } else if constexpr (index_kind == raft_proto::raft_index_kind::ivf_pq) {
            if (index_) {
                RAFT_FAIL("IVFPQ does not support adding vectors after training");
            }
        } else {
            if (index_) {
                auto const& res = get_device_resources_without_mempool();
                raft::resource::set_workspace_to_pool_resource(res);
                auto host_data = raft::make_host_matrix_view(data, row_count, feature_count);
                device_dataset_storage =
                    raft::make_device_matrix<data_type, input_indexing_type>(res, row_count, feature_count);
                auto device_data = device_dataset_storage->view();
                raft::copy(res, device_data, host_data);
                auto device_ids_storage = std::optional<raft::device_vector<indexing_type, input_indexing_type>>{};
                if (new_ids != nullptr) {
                    auto host_ids = raft::make_host_vector_view(new_ids, row_count);
                    device_ids_storage = raft::make_device_vector<indexing_type, input_indexing_type>(res, row_count);
                    raft::copy(res, device_ids_storage->view(), host_ids);
                }

                if (device_ids_storage) {
                    index_ = raft_index_type::extend(
                        res, raft::make_const_mdspan(device_data),
                        std::make_optional(raft::make_const_mdspan(device_ids_storage->view())), *index_);
                } else {
                    index_ = raft_index_type::extend(
                        res, raft::make_const_mdspan(device_data),
                        std::optional<raft::device_vector_view<indexing_type const, input_indexing_type>>{}, *index_);
                }
            } else {
                RAFT_FAIL("Index has not yet been trained");
            }
        }
    }

    auto
    search(raft_knowhere_config const& config, data_type const* data, knowhere_indexing_type row_count,
           knowhere_indexing_type feature_count, knowhere_bitset_data_type const* bitset_data,
           knowhere_bitset_indexing_type bitset_byte_size, knowhere_bitset_indexing_type bitset_size) const {
        auto scoped_device = raft::device_setter{device_id};
        auto const& res = raft::device_resources_manager::get_device_resources();
        auto k = knowhere_indexing_type(config.k);
        auto search_params = config_to_search_params<index_kind>(config);

        auto host_data = raft::make_host_matrix_view(data, row_count, feature_count);
        auto device_data_storage =
            raft::make_device_matrix<data_type, input_indexing_type>(res, row_count, feature_count);
        raft::copy(res, device_data_storage.view(), host_data);

        auto device_bitset =
            std::optional<raft::core::bitset<knowhere_bitset_data_type, knowhere_bitset_indexing_type>>{};
        auto k_tmp = k;

        if (bitset_data != nullptr && bitset_byte_size != 0) {
            device_bitset =
                raft::core::bitset<knowhere_bitset_data_type, knowhere_bitset_indexing_type>(res, bitset_size);
            raft::copy(res, device_bitset->to_mdspan(), raft::make_host_vector_view(bitset_data, bitset_byte_size));
            if constexpr (index_kind == raft_proto::raft_index_kind::brute_force) {
                k_tmp += device_bitset->count(res);
                if (k_tmp == k) {
                    device_bitset = std::nullopt;
                }
                k_tmp = std::min(k_tmp, size());
            }
            if (device_bitset) {
                device_bitset->flip(res);
            }
        }

        auto output_size = row_count * k;
        auto ids = std::unique_ptr<knowhere_indexing_type[]>(new knowhere_indexing_type[output_size]);
        auto distances = std::unique_ptr<knowhere_data_type[]>(new knowhere_data_type[output_size]);

        auto host_ids = raft::make_host_matrix_view(ids.get(), row_count, k);
        auto host_distances = raft::make_host_matrix_view(distances.get(), row_count, k);

        auto device_ids_storage = raft::make_device_matrix<indexing_type, input_indexing_type>(res, row_count, k_tmp);
        auto device_distances_storage = raft::make_device_matrix<data_type, input_indexing_type>(res, row_count, k_tmp);
        auto device_ids = device_ids_storage.view();
        auto device_distances = device_distances_storage.view();

        RAFT_EXPECTS(index_, "Index has not yet been trained");
        auto dataset_view = device_dataset_storage
                                ? std::make_optional(device_dataset_storage->view())
                                : std::optional<raft::device_matrix_view<const data_type, input_indexing_type>>{};

        if (device_bitset) {
            raft_index_type::search(
                res, *index_, search_params, raft::make_const_mdspan(device_data_storage.view()), device_ids,
                device_distances, config.refine_ratio, input_indexing_type{}, dataset_view,
                raft::neighbors::filtering::bitset_filter<knowhere_bitset_data_type, knowhere_bitset_indexing_type>{
                    device_bitset->view()});
        } else {
            raft_index_type::search(res, *index_, search_params, raft::make_const_mdspan(device_data_storage.view()),
                                    device_ids, device_distances, config.refine_ratio, input_indexing_type{},
                                    dataset_view);
        }

        auto device_knowhere_ids_storage =
            std::optional<raft::device_matrix<knowhere_indexing_type, input_indexing_type>>{};
        auto device_knowhere_ids = [&device_knowhere_ids_storage, &res, row_count, k_tmp, device_ids]() {
            if constexpr (std::is_signed_v<indexing_type>) {
                return device_ids;
            } else {
                device_knowhere_ids_storage =
                    raft::make_device_matrix<knowhere_indexing_type, input_indexing_type>(res, row_count, k_tmp);
                raft::copy(res, device_knowhere_ids_storage->view(), device_ids);
                return device_knowhere_ids_storage->view();
            }
        }();

        auto max_distance = std::nextafter(std::numeric_limits<data_type>::max(), 0.0f);
        thrust::replace_if(
            raft::resource::get_thrust_policy(res),
            thrust::device_ptr<typename decltype(device_knowhere_ids)::value_type>(device_knowhere_ids.data_handle()),
            thrust::device_ptr<typename decltype(device_knowhere_ids)::value_type>(device_knowhere_ids.data_handle() +
                                                                                   device_knowhere_ids.size()),
            thrust::make_zip_iterator(thrust::make_tuple(
                thrust::device_ptr<typename decltype(device_knowhere_ids)::value_type>(
                    device_knowhere_ids.data_handle()),
                thrust::device_ptr<typename decltype(device_distances)::value_type>(device_distances.data_handle()))),
            detail::check_valid_entry<thrust::tuple<typename decltype(device_knowhere_ids)::value_type,
                                                    typename decltype(device_distances)::value_type>,
                                      decltype(max_distance), knowhere_indexing_type>{max_distance,
                                                                                      knowhere_indexing_type(size())},
            typename decltype(device_knowhere_ids)::value_type{-1});

        if constexpr (index_kind == raft_proto::raft_index_kind::brute_force) {
            if (k_tmp > k) {
                for (auto i = 0; i < host_ids.extent(0); ++i) {
                    raft::copy(res, raft::make_host_vector_view(host_ids.data_handle() + i * host_ids.extent(1), k),
                               raft::make_device_vector_view(
                                   device_knowhere_ids.data_handle() + i * device_knowhere_ids.extent(1), k));
                    raft::copy(
                        res,
                        raft::make_host_vector_view(host_distances.data_handle() + i * host_distances.extent(1), k),
                        raft::make_device_vector_view(device_distances.data_handle() + i * device_distances.extent(1),
                                                      k));
                }
            } else {
                raft::copy(res, host_ids, device_knowhere_ids);
                raft::copy(res, host_distances, device_distances);
            }
        } else {
            raft::copy(res, host_ids, device_knowhere_ids);
            raft::copy(res, host_distances, device_distances);
        }
        return std::make_tuple(ids.release(), distances.release());
    }
    void
    range_search() const {
        RAFT_FAIL("Range search not yet implemented for RAFT indexes");
    }
    void
    get_vector_by_id() const {
        RAFT_FAIL("Vector reconstruction not yet implemented for RAFT indexes");
    }
    void
    serialize(std::ostream& os) const {
        auto scoped_device = raft::device_setter{device_id};
        auto const& res = get_device_resources_without_mempool();
        RAFT_EXPECTS(index_, "Index has not yet been trained");
        raft_index_type::template serialize<data_type, indexing_type>(res, os, *index_);
        if (device_dataset_storage) {
            raft::serialize_scalar(res, os, true);
            raft::serialize_scalar(res, os, device_dataset_storage->extent(0));
            raft::serialize_scalar(res, os, device_dataset_storage->extent(1));
            raft::serialize_mdspan(res, os, device_dataset_storage->view());
        } else {
            raft::serialize_scalar(res, os, false);
        }
    }

    void
    serialize_to_hnswlib(std::ostream& os) const {
        // only carga can save to hnswlib format
        if constexpr (index_kind == raft_proto::raft_index_kind::cagra) {
            auto scoped_device = raft::device_setter{device_id};
            auto const& res = get_device_resources_without_mempool();
            RAFT_EXPECTS(index_, "Index has not yet been trained");
            raft_index_type::template serialize_to_hnswlib<data_type, indexing_type>(res, os, *index_);
            raft::serialize_scalar(res, os, false);
        }
    }

    auto static deserialize(std::istream& is) {
        auto new_device_id = select_device_id();
        auto scoped_device = raft::device_setter{new_device_id};
        auto const& res = get_device_resources_without_mempool();
        auto des_index = raft_index_type::template deserialize<data_type, indexing_type>(res, is);

        auto dataset = std::optional<raft::device_matrix<data_type, input_indexing_type>>{};
        auto has_dataset = raft::deserialize_scalar<bool>(res, is);
        if (has_dataset) {
            auto rows = raft::deserialize_scalar<input_indexing_type>(res, is);
            auto cols = raft::deserialize_scalar<input_indexing_type>(res, is);
            dataset = raft::make_device_matrix<data_type, input_indexing_type>(res, rows, cols);
            raft::deserialize_mdspan(res, is, dataset->view());
            if constexpr (index_kind == raft_proto::raft_index_kind::brute_force ||
                          index_kind == raft_proto::raft_index_kind::cagra) {
                raft_index_type::template update_dataset<data_type, input_indexing_type>(
                    res, des_index, raft::make_const_mdspan(dataset->view()));
            }
        }
        return std::make_unique<typename raft_knowhere_index<index_kind>::impl>(std::move(des_index), new_device_id,
                                                                                std::move(dataset));
    }

    void
    synchronize(bool is_without_mempool = false) const {
        auto scoped_device = raft::device_setter{device_id};
        if (is_without_mempool) {
            get_device_resources_without_mempool().sync_stream();

        } else {
            raft::device_resources_manager::get_device_resources().sync_stream();
        }
    }
    impl(raft_index_type&& index, int new_device_id,
         std::optional<raft::device_matrix<data_type, input_indexing_type>>&& dataset)
        : index_{std::move(index)}, device_id{new_device_id}, device_dataset_storage{std::move(dataset)} {
    }

 private:
    std::optional<raft_index_type> index_ = std::nullopt;
    int device_id = select_device_id();
    std::optional<raft::device_matrix<data_type, input_indexing_type>> device_dataset_storage = std::nullopt;
};

template <raft_proto::raft_index_kind IndexKind>
raft_knowhere_index<IndexKind>::raft_knowhere_index() : pimpl{new raft_knowhere_index<IndexKind>::impl()} {
}

template <raft_proto::raft_index_kind IndexKind>
raft_knowhere_index<IndexKind>::~raft_knowhere_index<IndexKind>() = default;

template <raft_proto::raft_index_kind IndexKind>
raft_knowhere_index<IndexKind>::raft_knowhere_index(raft_knowhere_index<IndexKind>&& other)
    : pimpl{std::move(other.pimpl)} {
}

template <raft_proto::raft_index_kind IndexKind>
raft_knowhere_index<IndexKind>&
raft_knowhere_index<IndexKind>::operator=(raft_knowhere_index<IndexKind>&& other) {
    pimpl = std::move(other.pimpl);
    return *this;
}

template <raft_proto::raft_index_kind IndexKind>
bool
raft_knowhere_index<IndexKind>::is_trained() const {
    return pimpl->is_trained();
}

template <raft_proto::raft_index_kind IndexKind>
std::int64_t
raft_knowhere_index<IndexKind>::size() const {
    return pimpl->size();
}

template <raft_proto::raft_index_kind IndexKind>
std::int64_t
raft_knowhere_index<IndexKind>::dim() const {
    return pimpl->dim();
}

template <raft_proto::raft_index_kind IndexKind>
void
raft_knowhere_index<IndexKind>::train(raft_knowhere_config const& config, data_type const* data,
                                      knowhere_indexing_type row_count, knowhere_indexing_type feature_count) {
    return pimpl->train(config, data, row_count, feature_count);
}
template <raft_proto::raft_index_kind IndexKind>
void
raft_knowhere_index<IndexKind>::add(data_type const* data, knowhere_indexing_type row_count,
                                    knowhere_indexing_type feature_count, knowhere_indexing_type const* new_ids) {
    return pimpl->add(data, row_count, feature_count, new_ids);
}
template <raft_proto::raft_index_kind IndexKind>
std::tuple<knowhere_indexing_type*, knowhere_data_type*>
raft_knowhere_index<IndexKind>::search(raft_knowhere_config const& config, data_type const* data,
                                       knowhere_indexing_type row_count, knowhere_indexing_type feature_count,
                                       knowhere_bitset_data_type const* bitset_data,
                                       knowhere_bitset_indexing_type bitset_byte_size,
                                       knowhere_bitset_indexing_type bitset_size) const {
    return pimpl->search(config, data, row_count, feature_count, bitset_data, bitset_byte_size, bitset_size);
}

template <raft_proto::raft_index_kind IndexKind>
void
raft_knowhere_index<IndexKind>::range_search() const {
    return pimpl->range_search();
}

template <raft_proto::raft_index_kind IndexKind>
void
raft_knowhere_index<IndexKind>::get_vector_by_id() const {
    return pimpl->get_vector_by_id();
}

template <raft_proto::raft_index_kind IndexKind>
void
raft_knowhere_index<IndexKind>::serialize(std::ostream& os) const {
    return pimpl->serialize(os);
}

template <raft_proto::raft_index_kind IndexKind>
void
raft_knowhere_index<IndexKind>::serialize_to_hnswlib(std::ostream& os) const {
    return pimpl->serialize_to_hnswlib(os);
}

template <raft_proto::raft_index_kind IndexKind>
raft_knowhere_index<IndexKind>
raft_knowhere_index<IndexKind>::deserialize(std::istream& is) {
    return raft_knowhere_index<IndexKind>(raft_knowhere_index<IndexKind>::impl::deserialize(is));
}

template <raft_proto::raft_index_kind IndexKind>
void
raft_knowhere_index<IndexKind>::synchronize(bool is_without_mempool) const {
    return pimpl->synchronize(is_without_mempool);
}

}  // namespace raft_knowhere
