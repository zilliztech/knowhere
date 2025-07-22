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
#include "common/cuvs/integration/cuvs_knowhere_index.cuh"
#include "common/cuvs/proto/cuvs_index_kind.hpp"

namespace cuvs_knowhere {
template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_flat, knowhere::fp32>;
template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_flat, knowhere::fp16>;
template struct cuvs_knowhere_index<cuvs_proto::cuvs_index_kind::ivf_flat, knowhere::int8>;
}  // namespace cuvs_knowhere
