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
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <cstdint>
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <istream>
#include <optional>
#include <ostream>
#include <raft/core/copy.hpp>
#include <raft/core/logger.hpp>
#include <type_traits>

#include "common/cuvs/proto/cuvs_index_kind.hpp"

namespace cuvs_proto {

namespace detail {
template <cuvs_index_kind index_kind, template <typename...> typename index_template>
struct template_matches_index_kind : std::false_type {};

template <>
struct template_matches_index_kind<cuvs_index_kind::brute_force, cuvs::neighbors::brute_force::index> : std::true_type {
};

template <>
struct template_matches_index_kind<cuvs_index_kind::ivf_flat, cuvs::neighbors::ivf_flat::index> : std::true_type {};

template <>
struct template_matches_index_kind<cuvs_index_kind::ivf_pq, cuvs::neighbors::ivf_pq::index> : std::true_type {};

template <>
struct template_matches_index_kind<cuvs_index_kind::cagra, cuvs::neighbors::cagra::index> : std::true_type {};

template <cuvs_index_kind index_kind, template <typename...> typename index_template>
auto static constexpr template_matches_index_kind_v = template_matches_index_kind<index_kind, index_template>::value;

}  // namespace detail

template <template <typename...> typename underlying_index_type, typename... cuvs_index_args>
struct cuvs_index {
    using vector_index_type = underlying_index_type<cuvs_index_args...>;
    auto static constexpr vector_index_kind = []() {
        if constexpr (detail::template_matches_index_kind_v<cuvs_index_kind::brute_force, underlying_index_type>) {
            return cuvs_index_kind::brute_force;
        } else if constexpr (detail::template_matches_index_kind_v<cuvs_index_kind::ivf_flat, underlying_index_type>) {
            return cuvs_index_kind::ivf_flat;
        } else if constexpr (detail::template_matches_index_kind_v<cuvs_index_kind::ivf_pq, underlying_index_type>) {
            return cuvs_index_kind::ivf_pq;
        } else if constexpr (detail::template_matches_index_kind_v<cuvs_index_kind::cagra, underlying_index_type>) {
            return cuvs_index_kind::cagra;
        } else {
            static_assert(detail::template_matches_index_kind_v<cuvs_index_kind::brute_force, underlying_index_type>,
                          "Unsupported index template passed to cuvs_index");
        }
    }();

    using index_params_type = std::conditional_t<
        vector_index_kind == cuvs_index_kind::brute_force, cuvs::neighbors::brute_force::index_params,
        std::conditional_t<
            vector_index_kind == cuvs_index_kind::ivf_flat, cuvs::neighbors::ivf_flat::index_params,
            std::conditional_t<
                vector_index_kind == cuvs_index_kind::ivf_pq, cuvs::neighbors::ivf_pq::index_params,
                std::conditional_t<vector_index_kind == cuvs_index_kind::cagra, cuvs::neighbors::cagra::index_params,
                                   // Should never get here; precluded by static assertion above
                                   cuvs::neighbors::index_params>>>>;
    using search_params_type = std::conditional_t<
        vector_index_kind == cuvs_index_kind::brute_force, cuvs::neighbors::brute_force::search_params,
        std::conditional_t<
            vector_index_kind == cuvs_index_kind::ivf_flat, cuvs::neighbors::ivf_flat::search_params,
            std::conditional_t<
                vector_index_kind == cuvs_index_kind::ivf_pq, cuvs::neighbors::ivf_pq::search_params,
                std::conditional_t<vector_index_kind == cuvs_index_kind::cagra, cuvs::neighbors::cagra::search_params,
                                   // Should never get here; precluded by static assertion above
                                   cuvs::neighbors::search_params>>>>;

    [[nodiscard]] auto&
    get_vector_index() {
        return vector_index_;
    }
    [[nodiscard]] auto const&
    get_vector_index() const {
        return vector_index_;
    }
    [[nodiscard]] auto
    size() const {
        return vector_index_.size();
    }
    [[nodiscard]] auto
    dim() const {
        return vector_index_.dim();
    }

    template <typename T, typename IdxT, typename InputIdxT, typename DataMdspanT>
    auto static build(raft::resources const& res, index_params_type const& index_params, DataMdspanT data) {
        if constexpr (std::is_same_v<DataMdspanT, raft::host_matrix_view<T const, IdxT>>) {
            if constexpr (vector_index_kind == cuvs_index_kind::brute_force) {
                return cuvs_index<underlying_index_type, cuvs_index_args...>{
                    cuvs::neighbors::brute_force::build(res, index_params, data)};
            } else if constexpr (vector_index_kind == cuvs_index_kind::cagra) {
                return cuvs_index<underlying_index_type, cuvs_index_args...>{
                    cuvs::neighbors::cagra::build(res, index_params, data)};
            } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_pq) {
                return cuvs_index<underlying_index_type, cuvs_index_args...>{
                    cuvs::neighbors::ivf_pq::build(res, index_params, data)};
            } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_flat) {
                return cuvs_index<underlying_index_type, cuvs_index_args...>{
                    cuvs::neighbors::ivf_flat::build(res, index_params, data)};
            }
        } else {
            if constexpr (vector_index_kind == cuvs_index_kind::brute_force) {
                return cuvs_index<underlying_index_type, cuvs_index_args...>{
                    cuvs::neighbors::brute_force::build(res, index_params, data)};
            } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_flat) {
                return cuvs_index<underlying_index_type, cuvs_index_args...>{
                    cuvs::neighbors::ivf_flat::build(res, index_params, data)};
            } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_pq) {
                return cuvs_index<underlying_index_type, cuvs_index_args...>{
                    cuvs::neighbors::ivf_pq::build(res, index_params, data)};
            } else if constexpr (vector_index_kind == cuvs_index_kind::cagra) {
                return cuvs_index<underlying_index_type, cuvs_index_args...>{
                    cuvs::neighbors::cagra::build(res, index_params, data)};
            }
        }
    }

    template <typename T, typename IdxT, typename InputIdxT,
              typename FilterT = cuvs::neighbors::filtering::none_sample_filter>
    auto static search(raft::resources const& res, cuvs_index<underlying_index_type, cuvs_index_args...> const& index,
                       search_params_type const& search_params, raft::device_matrix_view<T const, InputIdxT> queries,
                       raft::device_matrix_view<IdxT, InputIdxT> neighbors,
                       raft::device_matrix_view<float, InputIdxT> distances, float refine_ratio = 1.0f,
                       InputIdxT k_offset = InputIdxT{},
                       std::optional<raft::device_matrix_view<const T, InputIdxT>> dataset = std::nullopt,
                       FilterT filter = cuvs::neighbors::filtering::none_sample_filter{}) {
        auto const& underlying_index = index.get_vector_index();

        auto k = neighbors.extent(1);
        auto k_tmp = k + k_offset;
        if (refine_ratio > 1.0f) {
            k_tmp *= refine_ratio;
        }

        auto neighbors_tmp = neighbors;
        auto distances_tmp = distances;
        auto neighbors_storage = std::optional<raft::device_matrix<IdxT, InputIdxT>>{};
        auto distances_storage = std::optional<raft::device_matrix<float, InputIdxT>>{};

        if (k_tmp > k) {
            neighbors_storage = raft::make_device_matrix<IdxT, InputIdxT>(res, queries.extent(0), k_tmp);
            neighbors_tmp = neighbors_storage->view();
            distances_storage = raft::make_device_matrix<float, InputIdxT>(res, queries.extent(0), k_tmp);
            distances_tmp = distances_storage->view();
        }

        if constexpr (vector_index_kind == cuvs_index_kind::brute_force) {
            cuvs::neighbors::brute_force::search(res, search_params, underlying_index, queries, neighbors_tmp,
                                                 distances_tmp, filter);
        } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_flat) {
            cuvs::neighbors::ivf_flat::search(res, search_params, underlying_index, queries, neighbors_tmp,
                                              distances_tmp, filter);
        } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_pq) {
            cuvs::neighbors::ivf_pq::search(res, search_params, underlying_index, queries, neighbors_tmp, distances_tmp,
                                            filter);
        } else if constexpr (vector_index_kind == cuvs_index_kind::cagra) {
            cuvs::neighbors::cagra::search(res, search_params, underlying_index, queries, neighbors_tmp, distances_tmp,
                                           filter);
        }
        if (refine_ratio > 1.0f) {
            if (dataset.has_value()) {
                if constexpr (std::is_same_v<IdxT, InputIdxT>) {
                    cuvs::neighbors::refine(res, *dataset, queries, raft::make_const_mdspan(neighbors_tmp), neighbors,
                                            distances, underlying_index.metric());
                } else {
                    cuvs::neighbors::refine(
                        res,
                        raft::make_device_matrix_view(dataset->data_handle(), InputIdxT(dataset->extent(0)),
                                                      InputIdxT(dataset->extent(1))),
                        raft::make_device_matrix_view(queries.data_handle(), InputIdxT(queries.extent(0)),
                                                      InputIdxT(queries.extent(1))),
                        raft::make_const_mdspan(raft::make_device_matrix_view(neighbors_tmp.data_handle(),
                                                                              InputIdxT(neighbors_tmp.extent(0)),
                                                                              InputIdxT(neighbors_tmp.extent(1)))),
                        raft::make_device_matrix_view(neighbors.data_handle(), InputIdxT(neighbors.extent(0)),
                                                      InputIdxT(neighbors.extent(1))),
                        raft::make_device_matrix_view(distances.data_handle(), InputIdxT(distances.extent(0)),
                                                      InputIdxT(distances.extent(1))),
                        underlying_index.metric());
                }
            } else {
                RAFT_LOG_WARN("Refinement requested, but no dataset provided. Ignoring refinement request.");
            }
        }
    }

    template <typename T, typename IdxT, typename InputIdxT>
    auto static extend(raft::resources const& res, raft::device_matrix_view<T const, InputIdxT> new_vectors,
                       std::optional<raft::device_vector_view<IdxT const, InputIdxT>> new_ids,
                       cuvs_index<underlying_index_type, cuvs_index_args...>& index) {
        auto const& underlying_index = index.get_vector_index();

        if constexpr (vector_index_kind == cuvs_index_kind::brute_force) {
            // TODO(wphicks): Implement brute force extend
            RAFT_FAIL("Brute force implements no extend method");
        } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_flat) {
            return cuvs_index{cuvs::neighbors::ivf_flat::extend(res, new_vectors, new_ids, underlying_index)};
        } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_pq) {
            return cuvs_index{cuvs::neighbors::ivf_pq::extend(res, new_vectors, new_ids, underlying_index)};
        } else if constexpr (vector_index_kind == cuvs_index_kind::cagra) {
            RAFT_FAIL("CAGRA implements no extend method");
        }
    }

    template <typename T, typename IdxT>
    void static serialize(raft::resources const& res, std::ostream& os,
                          cuvs_index<underlying_index_type, cuvs_index_args...> const& index,
                          bool include_dataset = true) {
        auto const& underlying_index = index.get_vector_index();

        if constexpr (vector_index_kind == cuvs_index_kind::brute_force) {
            return cuvs::neighbors::brute_force::serialize(res, os, underlying_index, include_dataset);
        } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_flat) {
            return cuvs::neighbors::ivf_flat::serialize(res, os, underlying_index);
        } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_pq) {
            return cuvs::neighbors::ivf_pq::serialize(res, os, underlying_index);
        } else if constexpr (vector_index_kind == cuvs_index_kind::cagra) {
            return cuvs::neighbors::cagra::serialize(res, os, underlying_index, include_dataset);
        }
    }

    template <typename T, typename IdxT>
    void static serialize_to_hnswlib(raft::resources const& res, std::ostream& os,
                                     cuvs_index<underlying_index_type, cuvs_index_args...> const& index,
                                     bool include_dataset = true) {
        auto const& underlying_index = index.get_vector_index();
        if constexpr (vector_index_kind == cuvs_index_kind::cagra) {
            return cuvs::neighbors::cagra::serialize_to_hnswlib(res, os, underlying_index);
        }
    }

    template <typename T, typename IdxT>
    auto static deserialize(raft::resources const& res, std::istream& is) {
        if constexpr (vector_index_kind == cuvs_index_kind::brute_force) {
            cuvs::neighbors::brute_force::index<T, float> loaded_index(res);
            cuvs::neighbors::brute_force::deserialize(res, is, &loaded_index);
            return cuvs_index{std::forward<decltype(loaded_index)>(loaded_index)};
        } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_flat) {
            cuvs::neighbors::ivf_flat::index<T, IdxT> loaded_index(res);
            cuvs::neighbors::ivf_flat::deserialize(res, is, &loaded_index);
            return cuvs_index{std::forward<decltype(loaded_index)>(loaded_index)};
        } else if constexpr (vector_index_kind == cuvs_index_kind::ivf_pq) {
            cuvs::neighbors::ivf_pq::index<IdxT> loaded_index(res);
            cuvs::neighbors::ivf_pq::deserialize(res, is, &loaded_index);
            return cuvs_index{std::forward<decltype(loaded_index)>(loaded_index)};
        } else if constexpr (vector_index_kind == cuvs_index_kind::cagra) {
            cuvs::neighbors::cagra::index<T, IdxT> loaded_index(res);
            cuvs::neighbors::cagra::deserialize(res, is, &loaded_index);
            return cuvs_index{std::forward<decltype(loaded_index)>(loaded_index)};
        }
    }

    template <typename T, typename InputIdxT>
    void static update_dataset(raft::resources const& res, cuvs_index<underlying_index_type, cuvs_index_args...>& index,
                               raft::device_matrix_view<T const, InputIdxT> data) {
        if constexpr (vector_index_kind == cuvs_index_kind::brute_force ||
                      vector_index_kind == cuvs_index_kind::cagra) {
            index.get_vector_index().update_dataset(res, data);
        } else {
            RAFT_FAIL("update_dataset is not supported for this index type");
        }
    }

 private:
    vector_index_type vector_index_;

    explicit cuvs_index(vector_index_type&& vector_index) : vector_index_{std::move(vector_index)} {
    }
};

}  // namespace cuvs_proto
