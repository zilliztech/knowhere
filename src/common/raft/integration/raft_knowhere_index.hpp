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

#include "common/raft/integration/raft_knowhere_config.hpp"
#include "common/raft/integration/type_mappers.hpp"
#include "common/raft/proto/raft_index_kind.hpp"
namespace raft_knowhere {

template <raft_proto::raft_index_kind IndexKind>
struct raft_knowhere_index {
    auto static constexpr index_kind = IndexKind;

    using data_type = raft_data_t<index_kind>;
    using indexing_type = raft_indexing_t<index_kind>;
    using input_indexing_type = raft_input_indexing_t<index_kind>;

    raft_knowhere_index();
    ~raft_knowhere_index();

    raft_knowhere_index(raft_knowhere_index&& other);
    raft_knowhere_index&
    operator=(raft_knowhere_index&& other);

    bool
    is_trained() const;
    std::int64_t
    size() const;
    std::int64_t
    dim() const;
    void
    train(raft_knowhere_config const&, data_type const*, knowhere_indexing_type, knowhere_indexing_type);
    void
    add(data_type const* data, knowhere_indexing_type row_count, knowhere_indexing_type feature_count,
        knowhere_indexing_type const* new_ids = nullptr);
    std::tuple<knowhere_indexing_type*, knowhere_data_type*>
    search(raft_knowhere_config const& config, data_type const* data, knowhere_indexing_type row_count,
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
    static raft_knowhere_index<IndexKind>
    deserialize(std::istream& is);
    void
    synchronize(bool is_without_mempool = false) const;

 private:
    // Use a private implementation to completely separate knowhere headers from
    // RAFT headers
    struct impl;
    std::unique_ptr<impl> pimpl;

    raft_knowhere_index(std::unique_ptr<impl>&& new_pimpl) : pimpl{std::move(new_pimpl)} {
    }
};

extern template struct raft_knowhere_index<raft_proto::raft_index_kind::brute_force>;
extern template struct raft_knowhere_index<raft_proto::raft_index_kind::ivf_flat>;
extern template struct raft_knowhere_index<raft_proto::raft_index_kind::ivf_pq>;
extern template struct raft_knowhere_index<raft_proto::raft_index_kind::cagra>;

}  // namespace raft_knowhere
