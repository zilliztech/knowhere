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

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_set>

#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

// TokenANNEmbListStrategy indexes all vectors and aggregates scores at search time.
class TokenANNEmbListStrategy : public EmbListStrategy {
 public:
    std::string
    Type() const override {
        return meta::EMB_LIST_STRATEGY_TOKENANN;
    }

    expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) override {
        emb_list_offset_ = std::make_shared<EmbListOffset>(doc_offset.offset);
        return std::optional<DataSetPtr>(dataset);
    }

    bool
    NeedsBaseIndexIDMap() const override {
        return true;  // needs vector_id -> doc_id mapping for bitset filtering
    }

    expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, int32_t k, const BaseConfig& config,
           const EmbListSearchContext& ctx) const override {
        if (!emb_list_offset_) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "emb_list_offset not initialized");
        }

        auto metric_or = ParseEmbListMetric(config);
        if (!metric_or.has_value()) {
            return expected<DataSetPtr>::Err(metric_or.error(), metric_or.what());
        }
        auto& mi = metric_or.value();
        LOG_KNOWHERE_DEBUG_ << "search emb_list with sub metric_type: " << mi.sub_metric_type;

        auto dim = query_dataset->GetDim();
        auto num_q_el = query_offset.num_el();

        auto query_code_size_or = ctx.get_query_code_size(query_dataset);
        if (!query_code_size_or.has_value()) {
            LOG_KNOWHERE_ERROR_ << "could not get query code size";
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "could not get query code size");
        }
        auto query_code_size = query_code_size_or.value();

        // Stage 1: Batch ANN search to retrieve top k' vectors per query vector
        auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
        if (retrieval_ann_ratio <= 0.0f) {
            auto err_msg = "retrieval_ann_ratio could not be less than or equal to 0";
            LOG_KNOWHERE_WARNING_ << err_msg;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, err_msg);
        }
        int32_t vec_topk =
            std::min(std::max((int32_t)(k * retrieval_ann_ratio), 1), (int32_t)emb_list_offset_->offset.back());

        auto stage1_start = std::chrono::high_resolution_clock::now();

        auto ann_search_res = ctx.ann_search(query_dataset, vec_topk);
        if (!ann_search_res.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed ANN search: " << ann_search_res.what();
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "failed ANN search");
        }
        const auto* stage1_ids = ann_search_res.value()->GetIds();

        auto stage1_end = std::chrono::high_resolution_clock::now();
        double stage1_ms = std::chrono::duration<double, std::milli>(stage1_end - stage1_start).count();
        auto total_query_vecs = query_dataset->GetRows();
        LOG_KNOWHERE_DEBUG_ << "[TokenANN] Stage1 ANN search: " << stage1_ms << " ms"
                            << ", num_query_docs=" << num_q_el << ", num_query_vecs=" << total_query_vecs << ", k=" << k
                            << ", vec_topk=" << vec_topk << ", index_docs=" << emb_list_offset_->num_el()
                            << ", index_vecs=" << (emb_list_offset_->offset.back());

        auto ids = std::make_unique<int64_t[]>(num_q_el * k);
        auto dists = std::make_unique<float[]>(num_q_el * k);

        // Stage 2: For each query doc, collect unique docs from stage 1, then rerank
        auto stage2_start = std::chrono::high_resolution_clock::now();
        size_t total_candidates = 0;
        size_t total_distance_computations = 0;
        size_t total_doc_vecs = 0;

        std::vector<int64_t> candidate_docs;
        std::unordered_set<size_t> el_ids_set;
        for (size_t i = 0; i < num_q_el; i++) {
            auto start_offset = query_offset.offset[i];
            auto end_offset = query_offset.offset[i + 1];
            auto nq = end_offset - start_offset;

            // Collect unique doc IDs from stage 1 results
            el_ids_set.clear();
            for (size_t j = start_offset * vec_topk; j < end_offset * vec_topk; j++) {
                if (stage1_ids[j] < 0) {
                    continue;
                }
                el_ids_set.emplace(emb_list_offset_->get_el_id((size_t)stage1_ids[j]));
            }

            // Convert to vector for RerankByCalcDistByIDs
            candidate_docs.clear();
            for (const auto& el_id : el_ids_set) {
                if (el_id < emb_list_offset_->num_el()) {
                    candidate_docs.push_back((int64_t)el_id);
                }
            }
            total_candidates += candidate_docs.size();

            // Compute aggregated score for each candidate
            auto tensor = (const char*)query_dataset->GetTensor();
            size_t tensor_offset = start_offset * query_code_size;
            auto bf_query_dataset = GenDataSet(nq, dim, tensor + tensor_offset);

            auto status = RerankByCalcDistByIDs(candidate_docs, bf_query_dataset, nq, k, mi.larger_is_closer,
                                                mi.is_cosine, emb_list_offset_, ctx, mi.agg_func, ids.get() + i * k,
                                                dists.get() + i * k, total_doc_vecs, total_distance_computations);

            if (status != Status::success) {
                return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "rerank distance computation error");
            }
        }

        auto stage2_end = std::chrono::high_resolution_clock::now();
        double stage2_ms = std::chrono::duration<double, std::milli>(stage2_end - stage2_start).count();
        LOG_KNOWHERE_DEBUG_ << "[TokenANN] Stage2 Rerank: " << stage2_ms << " ms"
                            << ", total_candidates=" << total_candidates
                            << ", avg_candidates_per_query=" << (num_q_el > 0 ? (double)total_candidates / num_q_el : 0)
                            << ", total_dist_comps=" << total_distance_computations << ", avg_dist_comps_per_query="
                            << (num_q_el > 0 ? (double)total_distance_computations / num_q_el : 0);

        return GenResultDataSet((int64_t)num_q_el, (int64_t)k, std::move(ids), std::move(dists));
    }

    Status
    Serialize(std::shared_ptr<uint8_t[]>& data, int64_t& size) const override {
        // Blob format: [magic][version][offsets]
        constexpr int32_t kMagic = 0x544F4B41;  // "TOKA"
        constexpr int32_t kVersion = 1;
        size_t offset_size = EmbListOffsetByteSize(emb_list_offset_);
        size = static_cast<int64_t>(2 * sizeof(int32_t) + offset_size);
        data = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
        uint8_t* ptr = data.get();

        std::memcpy(ptr, &kMagic, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &kVersion, sizeof(int32_t));
        ptr += sizeof(int32_t);
        SerializeEmbListOffsetToBytes(emb_list_offset_, ptr);
        return Status::success;
    }

    Status
    Deserialize(const uint8_t* data, int64_t size, const BaseConfig& config) override {
        constexpr int32_t kMagic = 0x544F4B41;  // "TOKA"
        const uint8_t* ptr = data;

        int32_t magic = 0;
        std::memcpy(&magic, ptr, sizeof(int32_t));

        if (magic == kMagic) {
            // New format: [magic][version][offsets]
            ptr += sizeof(int32_t);
            int32_t version = 0;
            std::memcpy(&version, ptr, sizeof(int32_t));
            ptr += sizeof(int32_t);
            if (version > 1) {
                LOG_KNOWHERE_ERROR_ << "TokenANN: unsupported serialization version " << version
                                    << " (max supported: 1)";
                return Status::emb_list_inner_error;
            }
            DeserializeEmbListOffsetFromBytes(ptr, emb_list_offset_);
        } else {
            // Legacy format: [count][offsets] (no magic/version)
            DeserializeEmbListOffsetFromBytes(data, emb_list_offset_);
        }
        return Status::success;
    }

    Status
    SetEmbListOffset(std::shared_ptr<EmbListOffset> offset) override {
        emb_list_offset_ = std::move(offset);
        return Status::success;
    }

    std::shared_ptr<EmbListOffset>
    GetEmbListOffset() const override {
        return emb_list_offset_;
    }

    int64_t
    GetDocCount() const override {
        return emb_list_offset_ ? emb_list_offset_->num_el() : 0;
    }

 private:
    std::shared_ptr<EmbListOffset> emb_list_offset_;
};

EmbListStrategyPtr
CreateTokenANNEmbListStrategy() {
    return std::make_unique<TokenANNEmbListStrategy>();
}

}  // namespace knowhere
