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
    CFG_BOOL hash_code_in_mem;
    CFG_BOOL shared_bloom_filter;
    CFG_FLOAT bloom_false_positive_prob;
    CFG_BOOL with_raw_data;
    CFG_INT refine_k;
    CFG_BOOL batch_search;
    KNOHWERE_DECLARE_CONFIG(MinHashConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(aligned_block_size)
            .description("decide the data format in file")
            .set_default(4096)
            .set_range(4096, std::numeric_limits<CFG_INT::value_type>::max())
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(hash_code_in_mem)
            .description("hash code is in ann-rss or in mmap file.")
            .set_default(true)
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(shared_bloom_filter)
            .description("whether to use one bloom filter for all bands")
            .set_default(false)
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(bloom_false_positive_prob)
            .description("false posibilty of bloom filter")
            .set_default(0.01)
            .set_range(0.0, 1.0)
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_k)
            .description("only useful in search_with_jaccard, the search topk of minhash lsh.")
            .set_default(1)
            .set_range(1, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(with_raw_data)
            .description("if with_raw_data = true, index will keep raw data in the index.")
            .set_default(false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(batch_search)
            .description("search query in batch, useful in limit cpu and hash_code_in_mem = false.")
            .set_default(false)
            .for_search();
    }
};
}  // namespace knowhere

#endif /* MINHASH_CONFIG_H */
