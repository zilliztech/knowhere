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
#include <istream>
#include <optional>
#include <ostream>
#include <raft/core/logger.hpp>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/brute_force_serialize.cuh>
#include <raft/neighbors/brute_force_types.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_serialize.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_serialize.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <type_traits>

#include "common/raft/proto/raft_index_kind.hpp"

namespace raft_proto {

auto static const RAFT_NAME = raft::RAFT_NAME;

namespace detail {
template <raft_index_kind index_kind, template <typename...> typename index_template>
struct template_matches_index_kind : std::false_type {};

template <>
struct template_matches_index_kind<raft_index_kind::brute_force, raft::neighbors::brute_force::index> : std::true_type {
};

template <>
struct template_matches_index_kind<raft_index_kind::ivf_flat, raft::neighbors::ivf_flat::index> : std::true_type {};

template <>
struct template_matches_index_kind<raft_index_kind::ivf_pq, raft::neighbors::ivf_pq::index> : std::true_type {};

template <>
struct template_matches_index_kind<raft_index_kind::cagra, raft::neighbors::cagra::index> : std::true_type {};

template <raft_index_kind index_kind, template <typename...> typename index_template>
auto static constexpr template_matches_index_kind_v = template_matches_index_kind<index_kind, index_template>::value;

// Note: The following are not at all general or properly SFINAE-guarded. They
// should be replaced if the post_filter proto is required for upstreaming.
template <typename mdspan_t>
auto
mdspan_begin(mdspan_t data) {
    return thrust::device_ptr<typename mdspan_t::value_type>(data.data_handle());
}
template <typename mdspan_t>
auto
mdspan_end(mdspan_t data) {
    return thrust::device_ptr<typename mdspan_t::value_type>(data.data_handle() + data.size());
}

template <typename mdspan_t>
auto
mdspan_begin_row(mdspan_t data, std::size_t row) {
    return thrust::device_ptr<typename mdspan_t::value_type>(data.data_handle() + row * data.extent(1));
}
template <typename mdspan_t>
auto
mdspan_end_row(mdspan_t data, std::size_t row) {
    return thrust::device_ptr<typename mdspan_t::value_type>(data.data_handle() + (row + 1) * data.extent(1));
}

template <typename index_mdspan_t, typename distance_mdspan_t, typename filter_lambda_t>
void
post_filter(raft::resources const& res, filter_lambda_t const& sample_filter, index_mdspan_t index_mdspan,
            distance_mdspan_t distance_mdspan) {
    auto counter = thrust::counting_iterator<decltype(index_mdspan.extent(0))>(0);
    // TODO (wphicks): This could be rolled into the stable_partition calls
    // below, but I am not sure whether or not that would be a net benefit. This
    // deserves some benchmarking unless pre-filtering gets in before we revisit
    // this.
    thrust::for_each(raft::resource::get_thrust_policy(res),
                     thrust::make_zip_iterator(
                         thrust::make_tuple(counter, mdspan_begin(index_mdspan), mdspan_begin(distance_mdspan))),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         counter + index_mdspan.size(), mdspan_end(index_mdspan), mdspan_end(distance_mdspan))),
                     [=] __device__(auto& index_id_distance) {
                         auto index = thrust::get<0>(index_id_distance);
                         auto& id = thrust::get<1>(index_id_distance);
                         auto& distance = thrust::get<2>(index_id_distance);
                         if (!sample_filter(index / index_mdspan.extent(1), id)) {
                             id = std::numeric_limits<std::remove_reference_t<decltype(id)>>::max();
                             distance = std::numeric_limits<std::remove_reference_t<decltype(distance)>>::max();
                         }
                     });
    for (auto i = 0; i < index_mdspan.extent(0); ++i) {
        auto id_row_begin = mdspan_begin_row(index_mdspan, i);
        auto id_row_end = mdspan_end_row(index_mdspan, i);
        auto distance_row_begin = mdspan_begin_row(distance_mdspan, i);
        auto distance_row_end = mdspan_end_row(distance_mdspan, i);
        thrust::stable_partition(
            raft::resource::get_thrust_policy(res),
            thrust::make_zip_iterator(thrust::make_tuple(id_row_begin, distance_row_begin)),
            thrust::make_zip_iterator(thrust::make_tuple(id_row_end, distance_row_end)),
            [=] __device__(auto& id_distance) {
                return thrust::get<0>(id_distance) !=
                       std::numeric_limits<std::remove_reference_t<decltype(thrust::get<0>(id_distance))>>::max();
            });
    }
}

}  // namespace detail

template <template <typename...> typename underlying_index_type, typename... raft_index_args>
struct raft_index {
    using vector_index_type = underlying_index_type<raft_index_args...>;
    auto static constexpr vector_index_kind = []() {
        if constexpr (detail::template_matches_index_kind_v<raft_index_kind::brute_force, underlying_index_type>) {
            return raft_index_kind::brute_force;
        } else if constexpr (detail::template_matches_index_kind_v<raft_index_kind::ivf_flat, underlying_index_type>) {
            return raft_index_kind::ivf_flat;
        } else if constexpr (detail::template_matches_index_kind_v<raft_index_kind::ivf_pq, underlying_index_type>) {
            return raft_index_kind::ivf_pq;
        } else if constexpr (detail::template_matches_index_kind_v<raft_index_kind::cagra, underlying_index_type>) {
            return raft_index_kind::cagra;
        } else {
            static_assert(detail::template_matches_index_kind_v<raft_index_kind::brute_force, underlying_index_type>,
                          "Unsupported index template passed to raft_index");
        }
    }();

    using index_params_type = std::conditional_t<
        vector_index_kind == raft_index_kind::brute_force, raft::neighbors::brute_force::index_params,
        std::conditional_t<
            vector_index_kind == raft_index_kind::ivf_flat, raft::neighbors::ivf_flat::index_params,
            std::conditional_t<
                vector_index_kind == raft_index_kind::ivf_pq, raft::neighbors::ivf_pq::index_params,
                std::conditional_t<vector_index_kind == raft_index_kind::cagra, raft::neighbors::cagra::index_params,
                                   // Should never get here; precluded by static assertion above
                                   raft::neighbors::brute_force::index_params>>>>;
    using search_params_type = std::conditional_t<
        vector_index_kind == raft_index_kind::brute_force, raft::neighbors::brute_force::search_params,
        std::conditional_t<
            vector_index_kind == raft_index_kind::ivf_flat, raft::neighbors::ivf_flat::search_params,
            std::conditional_t<
                vector_index_kind == raft_index_kind::ivf_pq, raft::neighbors::ivf_pq::search_params,
                std::conditional_t<vector_index_kind == raft_index_kind::cagra, raft::neighbors::cagra::search_params,
                                   // Should never get here; precluded by static assertion above
                                   raft::neighbors::brute_force::search_params>>>>;

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
        if constexpr (std::is_same_v<DataMdspanT, raft::host_matrix_view<T const, InputIdxT>>) {
            if constexpr (vector_index_kind == raft_index_kind::brute_force) {
                return raft_index<underlying_index_type, raft_index_args...>{
                    raft::neighbors::brute_force::build<T>(res, index_params, data)};
            } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
                return raft_index<underlying_index_type, raft_index_args...>{
                    raft::neighbors::cagra::build<T>(res, index_params, data)};
            } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
                return raft_index<underlying_index_type, raft_index_args...>{raft::neighbors::ivf_pq::build<T, IdxT>(
                    res, index_params, data.handle(), data.extent(0), data.extent(1))};
            } else {
                RAFT_FAIL("IVF flat does not support construction from host data");
            }
        } else {
            if constexpr (vector_index_kind == raft_index_kind::brute_force) {
                return raft_index<underlying_index_type, raft_index_args...>{
                    raft::neighbors::brute_force::build<T>(res, index_params, data)};
            } else if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
                return raft_index<underlying_index_type, raft_index_args...>{
                    raft::neighbors::ivf_flat::build<T, IdxT>(res, index_params, data)};
            } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
                return raft_index<underlying_index_type, raft_index_args...>{
                    raft::neighbors::ivf_pq::build<T, IdxT>(res, index_params, data)};
            } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
                return raft_index<underlying_index_type, raft_index_args...>{
                    raft::neighbors::cagra::build<T>(res, index_params, data)};
            }
        }
    }

    template <typename T, typename IdxT, typename InputIdxT, typename FilterT = std::nullptr_t>
    auto static search(raft::resources const& res, raft_index<underlying_index_type, raft_index_args...> const& index,
                       search_params_type const& search_params, raft::device_matrix_view<T const, InputIdxT> queries,
                       raft::device_matrix_view<IdxT, InputIdxT> neighbors,
                       raft::device_matrix_view<float, InputIdxT> distances, float refine_ratio = 1.0f,
                       InputIdxT k_offset = InputIdxT{},
                       std::optional<raft::device_matrix_view<const T, InputIdxT>> dataset = std::nullopt,
                       FilterT filter = nullptr) {
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

        if constexpr (vector_index_kind == raft_index_kind::brute_force) {
            raft::neighbors::brute_force::search<T>(res, search_params, underlying_index, queries, neighbors_tmp,
                                                    distances_tmp);
            if constexpr (!std::is_same_v<FilterT, std::nullptr_t>) {
                // TODO(wphicks): This can be replaced once prefiltering is
                // implemented for brute force upstream
                detail::post_filter(res, filter, neighbors_tmp, distances_tmp);
            }
        } else if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
            if constexpr (std::is_same_v<FilterT, std::nullptr_t>) {
                raft::neighbors::ivf_flat::search<T, IdxT>(res, search_params, underlying_index, queries, neighbors_tmp,
                                                           distances_tmp);
            } else {
                raft::neighbors::ivf_flat::search_with_filtering<T, IdxT>(res, search_params, underlying_index, queries,
                                                                          neighbors_tmp, distances_tmp, filter);
            }
        } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
            if constexpr (std::is_same_v<FilterT, std::nullptr_t>) {
                raft::neighbors::ivf_pq::search<T, IdxT>(res, search_params, underlying_index, queries, neighbors_tmp,
                                                         distances_tmp);
            } else {
                raft::neighbors::ivf_pq::search_with_filtering<T, IdxT>(res, search_params, underlying_index, queries,
                                                                        neighbors_tmp, distances_tmp, filter);
            }
        } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
            if constexpr (std::is_same_v<FilterT, std::nullptr_t>) {
                raft::neighbors::cagra::search<T, IdxT>(res, search_params, underlying_index, queries, neighbors_tmp,
                                                        distances_tmp);
            } else {
                raft::neighbors::cagra::search_with_filtering<T, IdxT>(res, search_params, underlying_index, queries,
                                                                       neighbors_tmp, distances_tmp, filter);
            }
        }
        if (refine_ratio > 1.0f) {
            if (dataset.has_value()) {
                if constexpr (std::is_same_v<IdxT, InputIdxT>) {
                    raft::neighbors::refine(res, *dataset, queries, raft::make_const_mdspan(neighbors_tmp), neighbors,
                                            distances, underlying_index.metric());
                } else {
                    // https://github.com/rapidsai/raft/issues/1950
                    raft::neighbors::refine(
                        res,
                        raft::make_device_matrix_view(dataset->data_handle(), IdxT(dataset->extent(0)),
                                                      IdxT(dataset->extent(1))),
                        raft::make_device_matrix_view(queries.data_handle(), IdxT(queries.extent(0)),
                                                      IdxT(queries.extent(1))),
                        raft::make_const_mdspan(raft::make_device_matrix_view(
                            neighbors_tmp.data_handle(), IdxT(neighbors_tmp.extent(0)), IdxT(neighbors_tmp.extent(1)))),
                        raft::make_device_matrix_view(neighbors.data_handle(), IdxT(neighbors.extent(0)),
                                                      IdxT(neighbors.extent(1))),
                        raft::make_device_matrix_view(distances.data_handle(), IdxT(distances.extent(0)),
                                                      IdxT(distances.extent(1))),
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
                       raft_index<underlying_index_type, raft_index_args...>& index) {
        auto const& underlying_index = index.get_vector_index();

        if constexpr (vector_index_kind == raft_index_kind::brute_force) {
            // TODO(wphicks): Implement brute force extend
            RAFT_FAIL("Brute force implements no extend method");
        } else if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
            return raft_index{raft::neighbors::ivf_flat::extend<T, IdxT>(res, new_vectors, new_ids, underlying_index)};
        } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
            return raft_index{raft::neighbors::ivf_pq::extend<T, IdxT>(res, new_vectors, new_ids, underlying_index)};
        } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
            RAFT_FAIL("CAGRA implements no extend method");
        }
    }

    template <typename T, typename IdxT>
    void static serialize(raft::resources const& res, std::ostream& os,
                          raft_index<underlying_index_type, raft_index_args...> const& index,
                          bool include_dataset = true) {
        auto const& underlying_index = index.get_vector_index();

        if constexpr (vector_index_kind == raft_index_kind::brute_force) {
            return raft::neighbors::brute_force::serialize<T>(res, os, underlying_index, include_dataset);
        } else if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
            return raft::neighbors::ivf_flat::serialize<T, IdxT>(res, os, underlying_index);
        } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
            return raft::neighbors::ivf_pq::serialize<IdxT>(res, os, underlying_index);
        } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
            return raft::neighbors::cagra::serialize<T, IdxT>(res, os, underlying_index, include_dataset);
        }
    }

    template <typename T, typename IdxT>
    auto static deserialize(raft::resources const& res, std::istream& is) {
        if constexpr (vector_index_kind == raft_index_kind::brute_force) {
            return raft_index{raft::neighbors::brute_force::deserialize<T>(res, is)};
        } else if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
            return raft_index{raft::neighbors::ivf_flat::deserialize<T, IdxT>(res, is)};
        } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
            return raft_index{raft::neighbors::ivf_pq::deserialize<IdxT>(res, is)};
        } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
            return raft_index{raft::neighbors::cagra::deserialize<T, IdxT>(res, is)};
        }
    }

    template <typename T, typename InputIdxT>
    void static update_dataset(raft::resources const& res, raft_index<underlying_index_type, raft_index_args...>& index,
                               raft::device_matrix_view<T const, InputIdxT> data) {
        if constexpr (vector_index_kind == raft_index_kind::brute_force ||
                      vector_index_kind == raft_index_kind::cagra) {
            index.get_vector_index().update_dataset(res, data);
        } else {
            RAFT_FAIL("update_dataset is not supported for this index type");
        }
    }

 private:
    vector_index_type vector_index_;

    explicit raft_index(vector_index_type&& vector_index) : vector_index_{std::move(vector_index)} {
    }
};

}  // namespace raft_proto
