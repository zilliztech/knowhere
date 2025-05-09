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
#include <cuda_fp16.h>

#include <cstdint>

#include "common/cuvs/integration/cuvs_knowhere_config.hpp"
#include "common/cuvs/integration/type_mappers.hpp"
#include "common/cuvs/proto/cuvs_index_kind.hpp"
#include "knowhere/operands.h"

namespace cuvs_knowhere {

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
struct cuvs_knowhere_index {
    auto static constexpr index_kind = IndexKind;

    using data_type = typename cuvs_data_type_mapper<DataType>::data_type;
    using indexing_type = cuvs_indexing_t<index_kind>;
    using input_indexing_type = cuvs_input_indexing_t<index_kind>;

    cuvs_knowhere_index();
    ~cuvs_knowhere_index();

    cuvs_knowhere_index(cuvs_knowhere_index&& other);
    cuvs_knowhere_index&
    operator=(cuvs_knowhere_index&& other);

    bool
    is_trained() const;
    std::int64_t
    size() const;
    std::int64_t
    dim() const;
    void
    train(cuvs_knowhere_config const&, data_type const*, knowhere_indexing_type, knowhere_indexing_type);
    std::tuple<knowhere_indexing_type*, knowhere_distance_type*>
    search(cuvs_knowhere_config const& config, data_type const* data, knowhere_indexing_type row_count,
           knowhere_indexing_type feature_count, knowhere_bitset_data_type const* bitset_data = nullptr,
           knowhere_bitset_indexing_type bitset_byte_size = knowhere_bitset_indexing_type{},
           knowhere_bitset_indexing_type bitset_size = knowhere_bitset_indexing_type{}) const;
    void
    range_search() const;
    void
    get_vector_by_id() const;
    void
    serialize(std::ostream& os) const;
    void
    serialize_to_hnswlib(std::ostream& os) const;
    static cuvs_knowhere_index<IndexKind, DataType>
    deserialize(std::istream& is);
    void
    synchronize(bool is_without_mempool = false) const;

 private:
    // Use a private implementation to completely separate knowhere headers from
    // cuVS headers
    struct impl;
    std::unique_ptr<impl> pimpl;

    cuvs_knowhere_index(std::unique_ptr<impl>&& new_pimpl) : pimpl{std::move(new_pimpl)} {
    }
};

extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::brute_force, knowhere::fp32>;
extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::brute_force, knowhere::fp16>;

extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_flat, knowhere::fp32>;
extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_flat, knowhere::fp16>;
extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_flat, knowhere::int8>;

extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_pq, knowhere::fp32>;
extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_pq, knowhere::fp16>;
extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_pq, knowhere::int8>;

extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::cagra, knowhere::fp32>;
extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::cagra, knowhere::fp16>;
extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::cagra, knowhere::int8>;
extern template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::cagra, knowhere::bin1>;

}  // namespace cuvs_knowhere
