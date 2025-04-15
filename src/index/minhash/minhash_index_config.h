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

#ifndef MINHASH_CONFIG_H
#define MINHASH_CONFIG_H

#include "knowhere/config.h"

namespace knowhere {

class MinHashConfig : public BaseConfig {
 public:
    CFG_INT aligned_block_size;
    CFG_INT band;
    CFG_BOOL enable_mmap;
    CFG_BOOL shared_bloom_filter;
    CFG_FLOAT bloom_false_positive_prob;
    KNOHWERE_DECLARE_CONFIG(MinHashConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(aligned_block_size)
            .description("the degree of the graph index.")
            .set_default(4096)
            .set_range(1024, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(band)
            .description("the degree of the graph index.")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(enable_mmap)
            .description("hash code is mmap mdode.")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(shared_bloom_filter)
            .description("whether to use one bloom filter for all band")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(bloom_false_positive_prob)
            .description("whether to use one bloom filter for all band")
            .set_default(0.01)
            .set_range(0.0, 1.0)
            .for_train();
    }
};
}  // namespace knowhere

#endif /* MINHASH_CONFIG_H */
