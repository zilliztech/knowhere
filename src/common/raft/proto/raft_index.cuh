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
#include <cstddef>
#include <cstdint>
#include <istream>
#include <optional>
#include <ostream>
#include <type_traits>
#include <raft/core/logger.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_flat_serialize.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_serialize.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/cagra_serialize.cuh>
#include <raft/neighbors/cagra_types.hpp>
#include "common/raft/proto/raft_index_kind.hpp"

namespace raft_proto {

auto static const RAFT_NAME = raft::RAFT_NAME;

namespace detail {
template <raft_index_kind index_kind, template<typename...> typename index_template>
struct template_matches_index_kind : std::false_type{};

template<>
struct template_matches_index_kind<raft_index_kind::ivf_flat, raft::neighbors::ivf_flat::index> : std::true_type{};

template<>
struct template_matches_index_kind<raft_index_kind::ivf_pq, raft::neighbors::ivf_pq::index> : std::true_type{};

template<>
struct template_matches_index_kind<raft_index_kind::cagra, raft::neighbors::cagra::index> : std::true_type{};

template <raft_index_kind index_kind, template<typename...> typename index_template>
auto static constexpr template_matches_index_kind_v = template_matches_index_kind<index_kind, index_template>::value;

}

template <template <typename...> typename underlying_index_type, typename... raft_index_args>
struct raft_index {
  using vector_index_type = underlying_index_type<raft_index_args...>;
  auto static constexpr vector_index_kind = []() {
    if constexpr (detail::template_matches_index_kind_v<raft_index_kind::ivf_flat, underlying_index_type>) {
      return raft_index_kind::ivf_flat;
    } else if constexpr (detail::template_matches_index_kind_v<raft_index_kind::ivf_pq, underlying_index_type>) {
      return raft_index_kind::ivf_pq;
    } else if constexpr (detail::template_matches_index_kind_v<raft_index_kind::cagra, underlying_index_type>) {
      return raft_index_kind::cagra;
    } else {
      static_assert(detail::template_matches_index_kind_v<
        raft_index_kind::ivf_flat, underlying_index_type
      >, "Unsupported index template passed to raft_index");
    }
  }();

  using index_params_type = std::conditional_t<
    vector_index_kind == raft_index_kind::ivf_flat,
    raft::neighbors::ivf_flat::index_params,
    std::conditional_t<
      vector_index_kind == raft_index_kind::ivf_pq,
      raft::neighbors::ivf_pq::index_params,
      std::conditional_t<
        vector_index_kind == raft_index_kind::cagra,
        raft::neighbors::cagra::index_params,
        // Should never get here; precluded by static assertion above
        raft::neighbors::ivf_flat::index_params
      >
    >
  >;
  using search_params_type = std::conditional_t<
    vector_index_kind == raft_index_kind::ivf_flat,
    raft::neighbors::ivf_flat::search_params,
    std::conditional_t<
      vector_index_kind == raft_index_kind::ivf_pq,
      raft::neighbors::ivf_pq::search_params,
      std::conditional_t<
        vector_index_kind == raft_index_kind::cagra,
        raft::neighbors::cagra::search_params,
        // Should never get here; precluded by static assertion above
        raft::neighbors::ivf_flat::search_params
      >
    >
  >;

  [[nodiscard]] auto& get_vector_index() { return vector_index_; }
  [[nodiscard]] auto const& get_vector_index() const { return vector_index_; }
  [[nodiscard]] auto size() const {
    return vector_index_.size();
  }
  [[nodiscard]] auto dim() const {
    return vector_index_.dim();
  }

  template <typename T, typename IdxT, typename InputIdxT>
  auto static build(
    raft::resources const& res,
    index_params_type const& index_params,
    raft::device_matrix_view<
      T const, InputIdxT
    > data
  ) {
    if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
      return raft_index<underlying_index_type, raft_index_args...>{
        raft::neighbors::ivf_flat::build<T, IdxT>(
          res,
          index_params,
          data
        )
      };
    } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
      return raft_index<underlying_index_type, raft_index_args...>{
        raft::neighbors::ivf_pq::build<T, IdxT>(
          res,
          index_params,
          data
        )
      };
    } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
      return raft_index<underlying_index_type, raft_index_args...>{
        raft::neighbors::cagra::build<T>(
          res,
          index_params,
          data
        )
      };
    }
  }

  template <typename T, typename IdxT, typename InputIdxT, typename FilterT=std::nullptr_t>
  auto static search(
    raft::resources const& res,
    raft_index<underlying_index_type, raft_index_args...> const& index,
    search_params_type const& search_params,
    raft::device_matrix_view<T const, InputIdxT> queries,
    raft::device_matrix_view<IdxT, InputIdxT> neighbors,
    raft::device_matrix_view<float, InputIdxT> distances,
    float refine_ratio = 1.0f,
    InputIdxT k_offset=InputIdxT{},
    std::optional<raft::device_matrix_view<const T, InputIdxT>> dataset = std::nullopt,
    FilterT filter = nullptr
  ) {
    auto const& underlying_index = index.get_vector_index();

    auto k = neighbors.extent(1);
    auto k_tmp = k + k_offset;
    if (refine_ratio > 1.0f) {
      k_tmp *= refine_ratio;
    }

    auto neighbors_tmp = neighbors;
    auto distances_tmp = distances;
    auto neighbors_storage = std::optional<raft::device_matrix<IdxT,
         InputIdxT>>{};
    auto distances_storage = std::optional<raft::device_matrix<float,
         InputIdxT>>{};

    if (k_tmp > k) {
      neighbors_storage = raft::make_device_matrix<IdxT, InputIdxT>(
        res,
        queries.extent(0),
        k_tmp
      );
      neighbors_tmp = neighbors_storage->view();
      distances_storage = raft::make_device_matrix<float, InputIdxT>(
        res,
        queries.extent(0),
        k_tmp
      );
      distances_tmp = distances_storage->view();
    }

    if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
      if constexpr (std::is_same_v<FilterT, std::nullptr_t>){
        raft::neighbors::ivf_flat::search<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp
        );
      } else {
        raft::neighbors::ivf_flat::search_with_filtering<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp,
          filter
        );
      }
    } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
      if constexpr (std::is_same_v<FilterT, std::nullptr_t>){
        raft::neighbors::ivf_pq::search<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp
        );
      } else {
        raft::neighbors::ivf_pq::search_with_filtering<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp,
          filter
        );
      }
    } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
      if constexpr (std::is_same_v<FilterT, std::nullptr_t>){
        raft::neighbors::cagra::search<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp
        );
      } else {
        raft::neighbors::cagra::search_with_filtering<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp,
          filter
        );
      }
    }
    if (refine_ratio > 1.0f) {
      if constexpr (vector_index_kind != raft_index_kind::cagra) {
        if (dataset.has_value()) {
          raft::neighbors::refine(
            res,
            *dataset,
            queries,
            raft::make_const_mdspan(neighbors_tmp),
            neighbors,
            distances,
            underlying_index.metric()
          );
        } else {
          RAFT_LOG_WARN(
            "Refinement requested, but no dataset provided. "
            "Ignoring refinement request."
          );
        }
      } else {
        // TODO(wphicks): Determine why CAGRA refinement fails to compile
        RAFT_LOG_WARN(
          "Refinement requested, but refinement is not yet implemented for "
          "CAGRA. Ignoring refinement request."
        );
      }
    }
  }

  template <typename T, typename IdxT, typename InputIdxT>
  auto static extend(
    raft::resources const& res,
    raft::device_matrix_view<T const, InputIdxT> new_vectors,
    std::optional<raft::device_vector_view<IdxT const, InputIdxT>> new_ids,
    raft_index<underlying_index_type, raft_index_args...>& index
  ) {
    auto const& underlying_index = index.get_vector_index();

    if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
      return raft_index{raft::neighbors::ivf_flat::extend<T, IdxT>(
        res,
        new_vectors,
        new_ids,
        underlying_index
      )};
    } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
      return raft_index{raft::neighbors::ivf_pq::extend<T, IdxT>(
        res,
        new_vectors,
        new_ids,
        underlying_index
      )};
    } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
      RAFT_FAIL("CAGRA implements no extend method");
    }
  }

  template <typename T, typename IdxT>
  void static serialize(
    raft::resources const& res,
    std::ostream& os,
    raft_index<underlying_index_type, raft_index_args...> const& index
  ) {
    auto const& underlying_index = index.get_vector_index();

    if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
      return raft::neighbors::ivf_flat::serialize<T, IdxT>(
        res,
        os,
        underlying_index
      );
    } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
      return raft::neighbors::ivf_pq::serialize<IdxT>(
        res,
        os,
        underlying_index
      );
    } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
      return raft::neighbors::cagra::serialize<T, IdxT>(
        res,
        os,
        underlying_index
      );
    }
  }

  template <typename T, typename IdxT>
  auto static deserialize(
    raft::resources const& res,
    std::istream& is
  ) {
    if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
      return raft_index{raft::neighbors::ivf_flat::deserialize<T, IdxT>(
        res,
        is
      )};
    } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
      return raft_index{raft::neighbors::ivf_pq::deserialize<IdxT>(
        res,
        is
      )};
    } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
      return raft_index{raft::neighbors::cagra::deserialize<T, IdxT>(
        res,
        is
      )};
    }
  }

 private:
  vector_index_type vector_index_;

  explicit raft_index(vector_index_type&& vector_index)
    : vector_index_{std::move(vector_index)} {}
};

/*template <typename T, typename IdxT, template <typename...> typename underlying_index, typename... raft_index_args>
struct tiered_index {
  using raft_index_type = raft_index<underlying_index, raft_index_args...>;
  using data_type = T;
  using index_type = IdxT;

  tiered_index (
    raft::device_resources const& res,
    raft_index_type::index_params_type const& index_params,
    raft::device_matrix_view<T const, IdxT> data,
    IdxT max_staged_vectors = IdxT{}
  ) : index_params_{index_params},
      vector_index_{},
      dataset_{
        [&res, data]() {
          auto result = raft::make_device_matrix<T, IdxT>(
            res,
            data.extent(0),
            data.extent(1)
          );
          raft::copy(res, result.view(), data);
          return result;
        }()
      },
      staged_count{data.extent(0)},
      max_staged_count{max_staged_vectors} {
      if (staged_count >= max_staged_count) {
        commit(res);
      }
    }

  template <typename FilterT=std::nullptr_t>
  auto search(
    raft::device_resources const& res,
    search_params_type const& search_params,
    raft::device_matrix_view<T const, IdxT> queries,
    raft::device_matrix_view<IdxT, IdxT> neighbors,
    raft::device_matrix_view<float, IdxT> distances,
    float refine_ratio = 1.0f,
    FilterT filter = nullptr
  ) const {
    auto k = neighbors.extent(1);
    auto k_committed = k;
    if (refine_ratio > 1.0f) {
      k_committed *= refine_ratio;
    }
    auto k_total = k_committed + staged_count;

    auto neighbors_committed = neighbors;
    auto distances_committed = distances;
    auto neighbors_staged = neighbors;
    auto distances_staged = distances;
    auto neighbors_storage = std::optional<raft::device_matrix<IdxT, IdxT>>{};
    auto distances_storage = std::optional<raft::device_matrix<float, IdxT>>{};
    if (k_total > k) {
      neighbors_storage = raft::make_device_matrix<IdxT, IdxT>(
        res,
        queries.extent(0),
        k_total
      );
      neighbors_committed = raft::make_device_matrix_view(
          new_dataset.data_handle(),
          queries.extent(0),
          dataset_.extent(1)
        ),
      distances_storage = raft::make_device_matrix<float, IdxT>(
        res,
        queries.extent(0),
        k_total
      );
      distances_tmp = distances_storage.view();
    }
    raft::neighbors::refine(
      res,
      *dataset,
      queries,
      neighbors_tmp,
      neighbors,
      distances,
      underlying_index.metric()
    );
    return raft_index_type::search(
      res,
      vector_index_,
      search_params,
      queries,
      neighbors,
      distances,
      1.0f,
      0,  // TODO (wphicks): Pass offset down when batching available
      get_entire_dataset(),
      filter
    );
  }

  template <typename T, typename IdxT>
  void extend(
    raft::device_resources const& res,
    raft::device_matrix_view<T const, IdxT> new_vectors,
    std::optional<raft::device_matrix_view<IdxT const, IdxT>> new_ids
  ) {
    if (size() + new_vectors.extent(0) > capacity()) {
      reserve(res, size() + std::max(size(), new_vectors.extent(0)));
    }
    staged_count += new_vectors.extent(0);
    if (staged_count >= max_staged_vectors) {
      commit(res);
    }
  }

  auto size() const {
    return committed_count + staged_count;
  }

  void reserve(raft::resources const& res, std::size_type new_capacity) {
    if (new_capacity > capacity()) {
      auto new_dataset = raft::make_device_matrix<T, IdxT>(
        res,
        new_capacity,
        dataset_.extent(1)
      );
      raft::copy(
        res, 
        raft::make_device_matrix_view(
          new_dataset.data_handle(),
          size(),
          dataset_.extent(1)
        ),
        get_entire_dataset()
      );
    }
  }

 private:
  raft_index_type::index_params_type index_params_;
  std::optional<raft_index_type> vector_index_;

  raft::device_matrix<T, IdxT> dataset_;
  IdxT committed_count;
  IdxT max_staged_count;
  IdxT build_size = IdxT{};
  IdxT staged_count = IdxT{};

  auto commit(raft::resources const& res) {
    if (
      (!vector_index_ && staged_count > max_staged_count) ||
      build_size * 2 < size()
    ) {
      vector_index_ = raft_index_type::build(res, index_params_, dataset_.view());
      committed_count = dataset_.extent(0);
      build_size = committed_count;
    } else {
      raft_index_type::extend(res, vector_index_, get_staged_dataset());
      committed_count += staged_count;
    }
    staged_count = IdxT{};
  }

  auto capacity() const {
    return dataset_.extent(0);
  }

  auto get_committed_dataset() {
    return raft::make_device_matrix_view(
      dataset_.data_handle(),
      committed_count,
      dataset_.extent(1)
    );
  }

  auto get_staged_dataset() {
    return raft::make_device_matrix_view(
      dataset_.data_handle() + committed_count * dataset_.extent(1),
      staged_count,
      dataset_.extent(1)
    );
  }

  auto get_entire_dataset() {
    return raft::make_device_matrix_view(
      dataset_.data_handle(),
      size(),
      dataset_.extent(1)
    );
  }
}; */

}  // namespace raft_proto
