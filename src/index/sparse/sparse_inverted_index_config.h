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
    CFG_FLOAT wand_bm25_max_score_ratio;
    CFG_STRING inverted_index_algo;
    KNOHWERE_DECLARE_CONFIG(SparseInvertedIndexConfig) {
        // NOTE: drop_ratio_build has been deprecated, it won't change anything
        KNOWHERE_CONFIG_DECLARE_FIELD(drop_ratio_build)
            .description("drop ratio for build")
            .set_default(0.0f)
            .set_range(0.0f, 1.0f, true, false)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(drop_ratio_search)
            .description("drop ratio for search")
            .set_default(0.0f)
            .set_range(0.0f, 1.0f, true, false)
            .for_search()
            .for_range_search()
            .for_iterator();
        KNOWHERE_CONFIG_DECLARE_FIELD(refine_factor)
            .description("refine factor")
            .set_default(10)
            .for_search()
            .for_range_search();
        /**
         * The term frequency part of score of BM25 is:
         * tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
         * as more documents being added to the collection, avgdl can also
         * change. In WAND index we precompute and cache this score in order to
         * speed up the search process, but if avgdl changes, we need to
         * re-compute such score which is expensive. To avoid this, we upscale
         * the max score by a ratio to compensate for avgdl changes. This will
         * make the max score larger than the actual max score, it makes the
         * filtering less aggressive, but guarantees the correctness.
         * The larger the ratio, the less aggressive the filtering is.
         */
        KNOWHERE_CONFIG_DECLARE_FIELD(wand_bm25_max_score_ratio)
            .set_range(1.0, 1.3)
            .set_default(1.05)
            .description("ratio to upscale max score to compensate for avgdl changes")
            .for_train()
            .for_deserialize()
            .for_deserialize_from_file();
        KNOWHERE_CONFIG_DECLARE_FIELD(inverted_index_algo)
            .description("inverted index algorithm")
            .set_default("DAAT_MAXSCORE")
            .for_train_and_search()
            .for_deserialize()
            .for_deserialize_from_file();
    }
};  // class SparseInvertedIndexConfig

}  // namespace knowhere

#endif  // SPARSE_INVERTED_INDEX_CONFIG_H
