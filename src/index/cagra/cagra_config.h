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
#include "knowhere/config.h"
namespace knowhere {

class CagraConfig : public BaseConfig {
 public:
    CFG_INT intermediate_graph_degree;
    CFG_INT graph_degree;
    CFG_INT itopk_size;
    CFG_LIST gpu_ids;
    CFG_INT max_queries;
    KNOHWERE_DECLARE_CONFIG(CagraConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(intermediate_graph_degree)
            .set_default(128)
            .description("degree of input graph for pruning.")
            .for_train()
            .set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(graph_degree)
            .set_default(64)
            .description("degree of output graph.")
            .for_search()
            .set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(itopk_size)
            .set_default(64)
            .description("number of intermediate search results retained during the search.")
            .for_search()
            .set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(gpu_ids)
            .description("gpu device ids")
            .set_default({
                0,
            })
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_queries).description("query batch size.").set_default(1).for_search();
    }
};

}  // namespace knowhere
