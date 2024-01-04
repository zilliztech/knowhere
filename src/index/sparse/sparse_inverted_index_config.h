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

#ifndef SPARSE_INVERTED_INDEX_CONFIG_H
#define SPARSE_INVERTED_INDEX_CONFIG_H

#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"

namespace knowhere {

class SparseInvertedIndexConfig : public BaseConfig {
 public:
    CFG_FLOAT drop_ratio_build;
    CFG_FLOAT drop_ratio_search;
    CFG_INT refine_factor;
    KNOHWERE_DECLARE_CONFIG(SparseInvertedIndexConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(drop_ratio_build)
            .description("drop ratio for build")
            .set_default(0.0f)
            .set_range(0.0f, 1.0f)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(drop_ratio_search)
            .description("drop ratio for search")
            .set_default(0.0f)
            .set_range(0.0f, 1.0f)
            .for_search()
            .for_range_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_factor)
            .description("refine factor")
            .set_default(10)
            .for_search()
            .for_range_search();
    }
};  // class SparseInvertedIndexConfig

}  // namespace knowhere

#endif  // SPARSE_INVERTED_INDEX_CONFIG_H
