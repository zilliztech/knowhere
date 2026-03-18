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

#include "knowhere/index/emb_list_strategy.h"

#include <algorithm>
#include <limits>
#include <queue>

#include "knowhere/index/index_node.h"
#include "knowhere/log.h"

namespace knowhere {

EmbListStrategyPtr
CreateTokenANNEmbListStrategy();
EmbListStrategyPtr
CreateMuveraEmbListStrategy();
EmbListStrategyPtr
CreateLemurEmbListStrategy();

expected<EmbListStrategyPtr>
CreateEmbListStrategy(const std::string& strategy_type, const BaseConfig& config) {
    if (strategy_type == meta::EMB_LIST_STRATEGY_TOKENANN || strategy_type.empty()) {
        return CreateTokenANNEmbListStrategy();
    }
    if (strategy_type == meta::EMB_LIST_STRATEGY_MUVERA) {
        return CreateMuveraEmbListStrategy();
    }
    if (strategy_type == meta::EMB_LIST_STRATEGY_LEMUR) {
        return CreateLemurEmbListStrategy();
    }
    LOG_KNOWHERE_ERROR_ << "Unknown emb_list strategy: " << strategy_type;
    return expected<EmbListStrategyPtr>::Err(Status::invalid_args, "unknown emb_list strategy: " + strategy_type);
}

Status
RerankByCalcDistByIDs(const std::vector<int64_t>& candidate_docs, const DataSetPtr& query_dataset, size_t nq, int32_t k,
                      bool larger_is_closer, bool is_cosine, const std::shared_ptr<EmbListOffset>& emb_list_offset,
                      const IndexNode* index, const BitsetView& bitset, milvus::OpContext* op_context,
                      const EmbListAggFunc& agg_func, int64_t* out_ids, float* out_dists, size_t& out_doc_vecs,
                      size_t& out_distance_computations) {
    std::priority_queue<DistId, std::vector<DistId>, std::greater<>> minheap;
    std::priority_queue<DistId, std::vector<DistId>, std::less<>> maxheap;

    for (int64_t doc_id : candidate_docs) {
        auto vids = emb_list_offset->get_vids((size_t)doc_id);
        if (vids.empty()) {
            // Empty embedding lists (0 vectors) can appear as rerank candidates because some strategies
            // (e.g., MUVERA) encode every doc into the ANN index regardless of whether it has vectors.
            // An empty doc's FDE encoding is all zeros, which can still be retrieved by ANN search.
            // Since there are no vectors to compute distances against, skip it.
            continue;
        }
        out_doc_vecs += vids.size();
        out_distance_computations += nq * vids.size();
        auto bf_search_res =
            index->CalcDistByIDs(query_dataset, bitset, vids.data(), vids.size(), is_cosine, op_context);
        if (!bf_search_res.has_value()) {
            LOG_KNOWHERE_ERROR_ << "CalcDistByIDs failed for doc " << doc_id << ": " << bf_search_res.what();
            return Status::emb_list_inner_error;
        }
        const auto* bf_dists = bf_search_res.value()->GetDistance();
        auto score = agg_func(bf_dists, nq, vids.size(), larger_is_closer);
        if (!score.has_value()) {
            LOG_KNOWHERE_ERROR_ << "agg_func failed for doc " << doc_id;
            return Status::emb_list_inner_error;
        }

        if (larger_is_closer) {
            if (minheap.size() < (size_t)k) {
                minheap.emplace(doc_id, score.value());
            } else if (score.value() > minheap.top().val) {
                minheap.pop();
                minheap.emplace(doc_id, score.value());
            }
        } else {
            if (maxheap.size() < (size_t)k) {
                maxheap.emplace(doc_id, score.value());
            } else if (score.value() < maxheap.top().val) {
                maxheap.pop();
                maxheap.emplace(doc_id, score.value());
            }
        }
    }

    // Extract results in sorted order
    size_t result_count = larger_is_closer ? minheap.size() : maxheap.size();
    for (size_t j = 0; j < result_count; ++j) {
        size_t idx = result_count - j - 1;
        if (larger_is_closer) {
            out_ids[idx] = minheap.top().id;
            out_dists[idx] = minheap.top().val;
            minheap.pop();
        } else {
            out_ids[idx] = maxheap.top().id;
            out_dists[idx] = maxheap.top().val;
            maxheap.pop();
        }
    }

    // Fill remaining with invalid values
    std::fill(out_ids + result_count, out_ids + k, -1);
    std::fill(out_dists + result_count, out_dists + k,
              larger_is_closer ? std::numeric_limits<float>::lowest() : std::numeric_limits<float>::max());

    return Status::success;
}

}  // namespace knowhere
