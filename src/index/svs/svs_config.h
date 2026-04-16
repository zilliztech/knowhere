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

#ifndef SVS_CONFIG_H
#define SVS_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

class SvsVamanaConfig : public BaseConfig {
 public:
    // Maximum degree of the Vamana graph. Larger values produce higher-quality graphs but increase memory and build
    // time. Typical values are between 32 and 128.
    CFG_INT svs_graph_max_degree;
    // Window size used during graph construction. Larger values improve graph quality at the cost of build time.
    CFG_INT svs_construction_window_size;
    // Window size used during search. Larger values improve recall at the cost of latency.
    CFG_INT svs_search_window_size;
    // Buffer capacity for the search priority queue.
    CFG_INT svs_search_buffer_capacity;
    // Pruning parameter. Default is 1.2 for L2 distance, 0.95 for inner product.
    CFG_FLOAT svs_alpha;
    // Data storage format: "fp32", "fp16", "sqi8".
    CFG_STRING svs_storage_kind;
    KNOHWERE_DECLARE_CONFIG(SvsVamanaConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_graph_max_degree)
            .description("maximum degree of the Vamana graph.")
            .set_default(32)
            .set_range(4, 256)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_construction_window_size)
            .description("window size used during graph construction.")
            .set_default(128)
            .set_range(1, 10000)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_search_window_size)
            .description("window size used during search.")
            .set_default(64)
            .set_range(1, 10000)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_search_buffer_capacity)
            .description("buffer capacity for the search priority queue.")
            .set_default(64)
            .set_range(1, 10000)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_alpha)
            .description(
                "pruning parameter for graph construction. Default depends on metric: 1.2 for L2, 0.95 for IP/COSINE.")
            .allow_empty_without_default()
            .set_range(0.0f, 10.0f)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_storage_kind)
            .description("data storage format: fp32, fp16, sqi8.")
            .set_default("fp32")
            .for_train();
    }
};

class SvsVamanaLvqConfig : public SvsVamanaConfig {
 public:
    KNOHWERE_DECLARE_CONFIG(SvsVamanaLvqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_graph_max_degree)
            .description("maximum degree of the Vamana graph.")
            .set_range(4, 256)
            .set_default(32)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_construction_window_size)
            .description("window size used during graph construction.")
            .set_default(128)
            .set_range(1, 10000)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_search_window_size)
            .description("window size used during search.")
            .set_default(64)
            .set_range(1, 10000)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_search_buffer_capacity)
            .description("buffer capacity for the search priority queue.")
            .set_default(64)
            .set_range(1, 10000)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_alpha)
            .description(
                "pruning parameter for graph construction. Default depends on metric: 1.2 for L2, 0.95 for IP/COSINE.")
            .allow_empty_without_default()
            .set_range(0.0f, 10.0f)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_storage_kind)
            .description("LVQ storage format: lvq4x0, lvq4x4, lvq4x8.")
            .set_default("lvq4x4")
            .for_train();
    }
};

class SvsVamanaLeanVecConfig : public SvsVamanaConfig {
 public:
    // Dimensionality for LeanVec compression. Default is d/2 (set to 0 to use default).
    CFG_INT svs_leanvec_dim;
    KNOHWERE_DECLARE_CONFIG(SvsVamanaLeanVecConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_graph_max_degree)
            .description("maximum degree of the Vamana graph.")
            .set_range(4, 256)
            .set_default(64)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_construction_window_size)
            .description("window size used during graph construction.")
            .set_default(128)
            .set_range(1, 10000)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_search_window_size)
            .description("window size used during search.")
            .set_default(64)
            .set_range(1, 10000)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_search_buffer_capacity)
            .description("buffer capacity for the search priority queue.")
            .set_default(64)
            .set_range(1, 10000)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_alpha)
            .description(
                "pruning parameter for graph construction. Default depends on metric: 1.2 for L2, 0.95 for IP/COSINE.")
            .allow_empty_without_default()
            .set_range(0.0f, 10.0f)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_storage_kind)
            .description("LeanVec storage format: leanvec4x4, leanvec4x8, leanvec8x8.")
            .set_default("leanvec4x4")
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(svs_leanvec_dim)
            .description("dimensionality for LeanVec compression. 0 means d/2.")
            .set_default(0)
            .set_range(0, 65536)
            .for_train();
    }
};

}  // namespace knowhere
#endif /* SVS_CONFIG_H */
