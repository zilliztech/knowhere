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
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/comp/time_recorder.h"
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

// TokenANNEmbListStrategy indexes all vectors and aggregates scores at search time.
class TokenANNEmbListStrategy : public EmbListStrategy {
 public:
    [[nodiscard]] std::string
    Type() const override {
        return meta::EMB_LIST_STRATEGY_TOKENANN;
    }

    expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) override {
        emb_list_offset_ = std::make_shared<EmbListOffset>(doc_offset.offset);
        return std::optional<DataSetPtr>(dataset);
    }

    [[nodiscard]] bool
    NeedsBaseIndexIDMap() const override {
        return true;  // needs vector_id -> doc_id mapping for bitset filtering
    }

    expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, const IndexNode* index,
           std::unique_ptr<Config> cfg, const BitsetView& bitset, milvus::OpContext* op_context) const override {
        if (!emb_list_offset_) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "emb_list_offset not initialized");
        }

        // Parse config and metric
        auto& config = static_cast<BaseConfig&>(*cfg);
        auto k = config.k.value();
        auto metric_or = ParseEmbListMetric(config);
        if (!metric_or.has_value()) {
            return expected<DataSetPtr>::Err(metric_or.error(), metric_or.what());
        }
        auto& mi = metric_or.value();
        LOG_KNOWHERE_DEBUG_ << "search emb_list with sub metric_type: " << mi.sub_metric_type;

        auto dim = query_dataset->GetDim();
        auto num_q_el = query_offset.num_el();

        auto query_code_size_opt = index->GetQueryCodeSize(query_dataset);
        if (!query_code_size_opt.has_value()) {
            LOG_KNOWHERE_ERROR_ << "could not get query code size";
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "could not get query code size");
        }
        auto query_code_size = query_code_size_opt.value();

        // Stage 1: Batch ANN search to retrieve top k' vectors per query vector
        auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
        if (retrieval_ann_ratio <= 0.0f) {
            auto err_msg = "retrieval_ann_ratio could not be less than or equal to 0";
            LOG_KNOWHERE_WARNING_ << err_msg;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, err_msg);
        }
        int32_t vec_topk =
            std::min(std::max((int32_t)(k * retrieval_ann_ratio), 1), (int32_t)emb_list_offset_->offset.back());

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        knowhere_search_emb_list_retrieval_ann_ratio.Observe(retrieval_ann_ratio);
        TimeRecorder rc("Emb List Search - 1st round ann search");
#endif

        config.k = vec_topk;
        config.metric_type = mi.sub_metric_type;
        auto ann_search_res = index->Search(query_dataset, std::move(cfg), bitset, op_context);
        if (!ann_search_res.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed ANN search: " << ann_search_res.what();
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "failed ANN search");
        }
        const auto* stage1_ids = ann_search_res.value()->GetIds();

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        auto time = rc.ElapseFromBegin("done");
        time *= 0.001;
        knowhere_search_emb_list_1st_ann_latency.Observe(time);
#endif

        LOG_KNOWHERE_DEBUG_ << "[TokenANN] Stage1 ANN search"
                            << ", num_query_docs=" << num_q_el << ", num_query_vecs=" << query_dataset->GetRows()
                            << ", k=" << k << ", vec_topk=" << vec_topk << ", index_docs=" << emb_list_offset_->num_el()
                            << ", index_vecs=" << (emb_list_offset_->offset.back());

        auto ids = std::make_unique<int64_t[]>(num_q_el * k);
        auto dists = std::make_unique<float[]>(num_q_el * k);

        // Stage 2: For each query doc, collect unique docs from stage 1, then rerank
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        TimeRecorder rc2("Emb List Search - 2nd round bf and agg");
#endif
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
                if (el_id >= emb_list_offset_->num_el()) {
                    LOG_KNOWHERE_ERROR_ << "Invalid el_id: " << el_id;
                    return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid emb_list id");
                }
                candidate_docs.push_back((int64_t)el_id);
            }
            total_candidates += candidate_docs.size();

            // Compute aggregated score for each candidate
            auto tensor = (const char*)query_dataset->GetTensor();
            size_t tensor_offset = start_offset * query_code_size;
            auto bf_query_dataset = GenDataSet(nq, dim, tensor + tensor_offset);

            auto status =
                RerankByCalcDistByIDs(candidate_docs, bf_query_dataset, nq, k, mi.larger_is_closer, mi.is_cosine,
                                      emb_list_offset_, index, bitset, op_context, mi.agg_func, ids.get() + i * k,
                                      dists.get() + i * k, total_doc_vecs, total_distance_computations);

            if (status != Status::success) {
                return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "rerank distance computation error");
            }
        }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        auto time2 = rc2.ElapseFromBegin("done");
        time2 *= 0.001;
        knowhere_search_emb_list_2nd_bf_agg_latency.Observe(time2);
#endif

        LOG_KNOWHERE_DEBUG_ << "[TokenANN] Stage2 Rerank"
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

    [[nodiscard]] std::shared_ptr<EmbListOffset>
    GetEmbListOffset() const override {
        return emb_list_offset_;
    }

    [[nodiscard]] int64_t
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
