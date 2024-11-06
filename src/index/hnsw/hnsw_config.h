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

#ifndef HNSW_CONFIG_H
#define HNSW_CONFIG_H

#include "index/hnsw/base_hnsw_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"

namespace knowhere {

class HnswConfig : public BaseHnswConfig {};

}  // namespace knowhere

#endif /* HNSW_CONFIG_H */
