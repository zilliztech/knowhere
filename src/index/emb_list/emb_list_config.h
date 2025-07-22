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

#ifndef EMB_LIST_HNSW_CONFIG_H
#define EMB_LIST_HNSW_CONFIG_H

#include "index/hnsw/faiss_hnsw_config.h"
#include "knowhere/config.h"

namespace knowhere {

class EmbListHNSWConfig : public FaissHnswFlatConfig {
 public:
    CFG_FLOAT retrieval_ann_ratio;
    KNOHWERE_DECLARE_CONFIG(EmbListHNSWConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(retrieval_ann_ratio)
            .description("")
            .set_default(1.0f)
            .set_range(0.01f, 10.0f)
            .for_search();
    }
};

}  // namespace knowhere

#endif /* EMB_LIST_HNSW_CONFIG_H */
