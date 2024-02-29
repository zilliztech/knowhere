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
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra.cuh>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/sample_filter.cuh>

#include "common/raft/proto/raft_index_kind.hpp"

namespace raft_proto {
namespace detail {
template <raft_proto::raft_index_kind K, typename T, typename IdxT>
using index_instantiation = std::conditional_t<
    K == raft_proto::raft_index_kind::ivf_flat, raft::neighbors::ivf_flat::index<T, IdxT>,
    std::conditional_t<
        K == raft_proto::raft_index_kind::ivf_pq, raft::neighbors::ivf_pq::index<IdxT>,
        std::conditional_t<K == raft_proto::raft_index_kind::cagra, raft::neighbors::cagra::index<T, IdxT>,
                           raft::neighbors::ivf_flat::index<T, IdxT>>>>;
}  // namespace detail
}  // namespace raft_proto

#define RAFT_FILTERED_SEARCH_TEMPLATE(index_type, T, IdxT, InpIdxT, DistT, BitsetDataT, BitsetIdxT)                   \
    template void search_with_filtering<T, IdxT, raft::neighbors::filtering::bitset_filter<BitsetDataT, BitsetIdxT>>( \
        raft::resources const&, search_params const&,                                                                 \
        raft_proto::detail::index_instantiation<raft_proto::raft_index_kind::index_type, T, IdxT> const&,             \
        raft::device_matrix_view<const T, InpIdxT>, raft::device_matrix_view<IdxT, InpIdxT>,                          \
        raft::device_matrix_view<DistT, InpIdxT>, raft::neighbors::filtering::bitset_filter<BitsetDataT, BitsetIdxT>)

#define RAFT_FILTERED_SEARCH_INSTANTIATION(index_type, T, IdxT, InpIdxT, DistT, BitsetDataT, BitsetIdxT) \
    namespace raft::neighbors::index_type {                                                              \
    RAFT_FILTERED_SEARCH_TEMPLATE(index_type, T, IdxT, InpIdxT, DistT, BitsetDataT, BitsetIdxT);         \
    }

#define RAFT_FILTERED_SEARCH_EXTERN(index_type, T, IdxT, InpIdxT, DistT, BitsetDataT, BitsetIdxT) \
    namespace raft::neighbors::index_type {                                                       \
    RAFT_FILTERED_SEARCH_TEMPLATE(index_type, T, IdxT, InpIdxT, DistT, BitsetDataT, BitsetIdxT);  \
    }
