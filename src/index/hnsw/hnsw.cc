// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "index/hnsw/hnsw.h"

#include <new>
#include <numeric>

#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"
#include "index/hnsw/hnsw_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/config.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/feder/HNSW.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"

namespace knowhere {

template class HnswIndexNode<knowhere::fp32, hnswlib::QuantType::None>;
template class HnswIndexNode<knowhere::fp16, hnswlib::QuantType::None>;
template class HnswIndexNode<knowhere::bf16, hnswlib::QuantType::None>;

KNOWHERE_SIMPLE_REGISTER_DENSE_ALL_GLOBAL(HNSWLIB_DEPRECATED, HnswIndexNode,
                                          knowhere::feature::MMAP | knowhere::feature::MV)

}  // namespace knowhere
