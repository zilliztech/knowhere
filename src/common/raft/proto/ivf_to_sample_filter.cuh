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
namespace raft_proto {

template <typename index_t, typename filter_t>
struct ivf_to_sample_filter {
  const index_t* const* inds_ptrs_;
  const filter_t next_filter_;

  ivf_to_sample_filter(const index_t* const* inds_ptrs, const filter_t next_filter)
    : inds_ptrs_{inds_ptrs}, next_filter_{next_filter} {}

  inline __host__ __device__ bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const
  {
    return next_filter_(query_ix, inds_ptrs_[cluster_ix][sample_ix]);
  }
};

} // namespace raft_proto
