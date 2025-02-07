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
#include <cstdint>
#include <type_traits>

#include "common/cuvs/proto/cuvs_index_kind.hpp"

namespace cuvs_knowhere {

using knowhere_distance_type = float;
using knowhere_indexing_type = std::int64_t;
using knowhere_bitset_data_type = std::uint8_t;
using knowhere_bitset_indexing_type = std::uint32_t;
using knowhere_bitset_internal_data_type = std::uint32_t;
using knowhere_bitset_internal_indexing_type = std::int64_t;

namespace detail {

template <bool B, cuvs_proto::cuvs_index_kind IndexKind>
struct cuvs_io_type_mapper : std::false_type {};

template <>
struct cuvs_io_type_mapper<true, cuvs_proto::cuvs_index_kind::brute_force> : std::true_type {
    using data_type = float;
    using indexing_type = std::int64_t;
    using input_indexing_type = std::int64_t;
};

template <>
struct cuvs_io_type_mapper<true, cuvs_proto::cuvs_index_kind::ivf_flat> : std::true_type {
    using data_type = float;
    using indexing_type = std::int64_t;
    using input_indexing_type = std::int64_t;
};

template <>
struct cuvs_io_type_mapper<true, cuvs_proto::cuvs_index_kind::ivf_pq> : std::true_type {
    using data_type = float;
    using indexing_type = std::int64_t;
    using input_indexing_type = std::uint32_t;
};

template <>
struct cuvs_io_type_mapper<true, cuvs_proto::cuvs_index_kind::cagra> : std::true_type {
    using data_type = float;
    using indexing_type = std::uint32_t;
    using input_indexing_type = std::int64_t;
};
}  // namespace detail

template <cuvs_proto::cuvs_index_kind IndexKind>
using cuvs_indexing_t = typename detail::cuvs_io_type_mapper<true, IndexKind>::indexing_type;

template <cuvs_proto::cuvs_index_kind IndexKind>
using cuvs_input_indexing_t = typename detail::cuvs_io_type_mapper<true, IndexKind>::input_indexing_type;

}  // namespace cuvs_knowhere
