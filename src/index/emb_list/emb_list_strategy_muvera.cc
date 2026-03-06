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
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "knowhere/comp/index_param.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "simd/hook.h"

namespace knowhere {

// SimHash projects vectors onto random hyperplanes and uses the signs of
// projections to form a binary hash code, which maps to a partition index.
// Similar vectors are likely to fall into the same partition.
class SimHash {
 public:
    SimHash(int32_t dim, int32_t num_projections, int32_t seed) : dim_(dim), num_projections_(num_projections) {
        GenerateProjectionMatrix(seed);
    }

    // Compute partition index for a vector, returns value in [0, 2^num_projections)
    int32_t
    GetPartitionIndex(const float* vec) const {
        int32_t index = 0;
        for (int32_t p = 0; p < num_projections_; ++p) {
            const float* proj = projection_matrix_.data() + p * dim_;
            float dot = faiss::cppcontrib::knowhere::fvec_inner_product(proj, vec, dim_);
            if (dot >= 0) {
                index |= (1 << p);
            }
        }
        return index;
    }

    const std::vector<float>&
    GetProjectionMatrix() const {
        return projection_matrix_;
    }

    void
    SetProjectionMatrix(std::vector<float>&& matrix) {
        projection_matrix_ = std::move(matrix);
    }

 private:
    void
    GenerateProjectionMatrix(int32_t seed) {
        projection_matrix_.resize(num_projections_ * dim_);

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (int32_t i = 0; i < num_projections_ * dim_; ++i) {
            projection_matrix_[i] = dist(rng);
        }

        LOG_KNOWHERE_DEBUG_ << "SimHash: generated projection matrix [" << num_projections_ << " x " << dim_
                            << "] with seed " << seed;
    }

    int32_t dim_;
    int32_t num_projections_;
    std::vector<float> projection_matrix_;  // [num_projections * dim] row-major
};

// MUVERA uses Fixed Dimensional Encoding to convert variable-length multi-vector
// documents into fixed-length single vectors for efficient ANN retrieval,
// followed by exact MaxSim reranking.
class MuveraEmbListStrategy : public EmbListStrategy {
 public:
    std::string
    Type() const override {
        return meta::EMB_LIST_STRATEGY_MUVERA;
    }

    expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) override {
        // 1. Read config
        num_projections_ = config.muvera_num_projections.value();
        num_repeats_ = config.muvera_num_repeats.value();
        seed_ = config.muvera_seed.value();
        original_dim_ = dataset->GetDim();
        num_docs_ = doc_offset.num_el();
        if (num_docs_ > std::numeric_limits<int32_t>::max()) {
            return expected<std::optional<DataSetPtr>>::Err(Status::emb_list_inner_error,
                                                            "num_docs exceeds int32 limit");
        }

        // 2. Compute num_buckets = 2^num_projections
        num_buckets_ = 1 << num_projections_;

        // 3. Initialize SimHash instances (different seed for each repeat)
        simhash_instances_.clear();
        simhash_instances_.reserve(num_repeats_);
        for (int32_t r = 0; r < num_repeats_; ++r) {
            simhash_instances_.emplace_back(original_dim_, num_projections_, seed_ + r);
        }

        // 4. Compute encoded dimension
        encoded_dim_ = num_repeats_ * num_buckets_ * original_dim_;

        LOG_KNOWHERE_INFO_ << "MUVERA PrepareDataForBuild: num_docs=" << num_docs_ << ", original_dim=" << original_dim_
                           << ", num_projections=" << num_projections_ << ", num_buckets=" << num_buckets_
                           << ", num_repeats=" << num_repeats_ << ", encoded_dim=" << encoded_dim_;

        // 5. Allocate encoded data buffer and perform FDE encoding
        // Document uses mean aggregation (asymmetric with query's sum)
        auto encoded_data = std::make_unique<float[]>(num_docs_ * encoded_dim_);
        const float* raw_data = static_cast<const float*>(dataset->GetTensor());

        EncodeFDE(raw_data, doc_offset, encoded_data.get(), num_docs_, /*use_mean=*/true);

        LOG_KNOWHERE_INFO_ << "MUVERA FDE encoding completed";

        // 6. Store doc_offset for reranking
        emb_list_offset_ = std::make_shared<EmbListOffset>(doc_offset.offset);

        // 7. Create encoded dataset
        auto encoded_dataset = GenDataSet(num_docs_, encoded_dim_, encoded_data.release());

        return std::optional<DataSetPtr>(encoded_dataset);
    }

    bool
    NeedsBaseIndexIDMap() const override {
        return false;
    }

    bool
    NeedsRawVectorStorage() const override {
        return true;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, int32_t k, const BaseConfig& config,
           const EmbListSearchContext& ctx) const override {
        // 1. Validate state
        if (!emb_list_offset_) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "MUVERA not initialized");
        }

        // 2. Parse metric type
        auto metric_or = ParseEmbListMetric(config);
        if (!metric_or.has_value()) {
            return expected<DataSetPtr>::Err(metric_or.error(), metric_or.what());
        }
        auto& mi = metric_or.value();

        LOG_KNOWHERE_DEBUG_ << "MUVERA Search: sub_metric=" << mi.sub_metric_type;

        // 3. FDE encode query documents
        // Query uses sum aggregation (asymmetric with document's mean)
        auto num_query_docs = query_offset.num_el();
        auto query_data = static_cast<const float*>(query_dataset->GetTensor());

        auto encode_start = std::chrono::high_resolution_clock::now();
        std::vector<float> encoded_queries(num_query_docs * encoded_dim_);
        EncodeFDE(query_data, query_offset, encoded_queries.data(), num_query_docs, /*use_mean=*/false);
        auto encode_end = std::chrono::high_resolution_clock::now();
        double encode_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

        LOG_KNOWHERE_DEBUG_ << "[MUVERA] Stage1 FDE encode: " << encode_ms << " ms, num_query_docs=" << num_query_docs;

        // 4. ANN search with encoded queries
        bool do_rerank = config.emb_list_rerank.value_or(true);
        auto encoded_query_dataset = GenDataSet(num_query_docs, encoded_dim_, encoded_queries.data());

        // If reranking, retrieve more candidates; otherwise just retrieve k
        int32_t ann_k;
        if (do_rerank) {
            auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
            ann_k = std::min(std::max(static_cast<int32_t>(k * retrieval_ann_ratio), 1),
                            static_cast<int32_t>(num_docs_));
        } else {
            ann_k = std::min(k, static_cast<int32_t>(num_docs_));
        }

        auto ann_start = std::chrono::high_resolution_clock::now();
        auto ann_result = ctx.ann_search(encoded_query_dataset, ann_k);
        if (!ann_result.has_value()) {
            LOG_KNOWHERE_ERROR_ << "MUVERA ANN search failed: " << ann_result.what();
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "ANN search failed");
        }
        auto ann_end = std::chrono::high_resolution_clock::now();
        double ann_ms = std::chrono::duration<double, std::milli>(ann_end - ann_start).count();
        const auto* ann_ids = ann_result.value()->GetIds();

        LOG_KNOWHERE_DEBUG_ << "[MUVERA] Stage2 ANN search: " << ann_ms << " ms, ann_k=" << ann_k
                            << ", encoded_dim=" << encoded_dim_ << ", rerank=" << do_rerank;

        // 5. If reranking disabled, directly return ANN results
        if (!do_rerank) {
            auto ids = std::make_unique<int64_t[]>(num_query_docs * k);
            auto dists = std::make_unique<float[]>(num_query_docs * k);
            const auto* ann_dists = ann_result.value()->GetDistance();

            for (size_t q = 0; q < num_query_docs; ++q) {
                for (int32_t i = 0; i < k; ++i) {
                    if (i < ann_k) {
                        ids[q * k + i] = ann_ids[q * ann_k + i];
                        dists[q * k + i] = ann_dists[q * ann_k + i];
                    } else {
                        ids[q * k + i] = -1;
                        dists[q * k + i] = mi.larger_is_closer ? -std::numeric_limits<float>::infinity()
                                                               : std::numeric_limits<float>::infinity();
                    }
                }
            }
            LOG_KNOWHERE_DEBUG_ << "[MUVERA] Reranking disabled, returning ANN results directly";
            return GenResultDataSet(static_cast<int64_t>(num_query_docs), static_cast<int64_t>(k), std::move(ids),
                                    std::move(dists));
        }

        // 6. MaxSim reranking
        auto ids = std::make_unique<int64_t[]>(num_query_docs * k);
        auto dists = std::make_unique<float[]>(num_query_docs * k);

        auto rerank_start = std::chrono::high_resolution_clock::now();
        size_t total_candidates = 0;
        size_t total_distance_computations = 0;
        size_t total_doc_vecs = 0;
        size_t total_query_vecs = 0;

        for (size_t q = 0; q < num_query_docs; ++q) {
            size_t q_vec_start = query_offset.offset[q];
            size_t q_vec_end = query_offset.offset[q + 1];
            size_t nq = q_vec_end - q_vec_start;
            total_query_vecs += nq;

            // Collect candidate doc IDs (no dedup needed — ANN returns doc-level IDs)
            std::vector<int64_t> candidate_docs;
            candidate_docs.reserve(ann_k);
            for (int32_t i = 0; i < ann_k; ++i) {
                int64_t doc_id = ann_ids[q * ann_k + i];
                if (doc_id >= 0 && doc_id < num_docs_) {
                    candidate_docs.push_back(doc_id);
                }
            }
            total_candidates += candidate_docs.size();

            // Build query dataset for this query document
            auto bf_query_dataset = GenDataSet(nq, original_dim_, query_data + q_vec_start * original_dim_);

            auto status = RerankByCalcDistByIDs(candidate_docs, bf_query_dataset, nq, k, mi.larger_is_closer,
                                                mi.is_cosine, emb_list_offset_, ctx, mi.agg_func, ids.get() + q * k,
                                                dists.get() + q * k, total_doc_vecs, total_distance_computations);

            if (status != Status::success) {
                return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "rerank distance computation error");
            }
        }

        auto rerank_end = std::chrono::high_resolution_clock::now();
        double rerank_ms = std::chrono::duration<double, std::milli>(rerank_end - rerank_start).count();

        double avg_doc_len = total_candidates > 0 ? static_cast<double>(total_doc_vecs) / total_candidates : 0;
        double avg_query_len = num_query_docs > 0 ? static_cast<double>(total_query_vecs) / num_query_docs : 0;
        LOG_KNOWHERE_DEBUG_ << "[MUVERA] Stage3 Rerank: " << rerank_ms << " ms"
                            << ", total_candidates=" << total_candidates << ", avg_doc_len=" << avg_doc_len
                            << ", avg_query_len=" << avg_query_len
                            << ", total_dist_comps=" << total_distance_computations;

        return GenResultDataSet(static_cast<int64_t>(num_query_docs), static_cast<int64_t>(k), std::move(ids),
                                std::move(dists));
    }

    Status
    Serialize(std::shared_ptr<uint8_t[]>& data, int64_t& size) const override {
        // Blob format: [magic][version][config fields...][offsets]
        constexpr int32_t kMagic = 0x4D555652;  // "MUVR"
        constexpr int32_t kVersion = 1;
        size_t config_size = 2 * sizeof(int32_t) + 4 * sizeof(int32_t) + sizeof(int64_t);
        size_t offset_size = EmbListOffsetByteSize(emb_list_offset_);
        size = static_cast<int64_t>(config_size + offset_size);
        data = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
        uint8_t* ptr = data.get();

        std::memcpy(ptr, &kMagic, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &kVersion, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_projections_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_repeats_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &seed_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &original_dim_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_docs_, sizeof(int64_t));
        ptr += sizeof(int64_t);

        SerializeEmbListOffsetToBytes(emb_list_offset_, ptr);

        LOG_KNOWHERE_INFO_ << "MUVERA Serialize: config + offsets";
        return Status::success;
    }

    Status
    Deserialize(const uint8_t* data, int64_t size, const BaseConfig& config) override {
        // 1. Deserialize config parameters
        constexpr size_t kExpectedConfigSize = 2 * sizeof(int32_t) + 4 * sizeof(int32_t) + sizeof(int64_t);
        if (size < static_cast<int64_t>(kExpectedConfigSize)) {
            LOG_KNOWHERE_ERROR_ << "MUVERA: blob too small: " << size << " < " << kExpectedConfigSize;
            return Status::emb_list_inner_error;
        }

        constexpr int32_t kMagic = 0x4D555652;  // "MUVR"
        const uint8_t* ptr = data;

        int32_t magic = 0;
        std::memcpy(&magic, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        if (magic != kMagic) {
            LOG_KNOWHERE_ERROR_ << "MUVERA: invalid magic number 0x" << std::hex << magic;
            return Status::emb_list_inner_error;
        }

        int32_t version = 0;
        std::memcpy(&version, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        if (version > 1) {
            LOG_KNOWHERE_ERROR_ << "MUVERA: unsupported serialization version " << version << " (max supported: 1)";
            return Status::emb_list_inner_error;
        }

        std::memcpy(&num_projections_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_repeats_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&seed_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&original_dim_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_docs_, ptr, sizeof(int64_t));
        ptr += sizeof(int64_t);

        if (num_docs_ > std::numeric_limits<int32_t>::max()) {
            LOG_KNOWHERE_ERROR_ << "MUVERA: num_docs " << num_docs_ << " exceeds int32 limit";
            return Status::emb_list_inner_error;
        }

        // 2. Recompute derived values
        num_buckets_ = 1 << num_projections_;
        encoded_dim_ = num_repeats_ * num_buckets_ * original_dim_;

        // 3. Rebuild SimHash instances with the same seeds
        simhash_instances_.clear();
        simhash_instances_.reserve(num_repeats_);
        for (int32_t r = 0; r < num_repeats_; ++r) {
            simhash_instances_.emplace_back(original_dim_, num_projections_, seed_ + r);
        }

        LOG_KNOWHERE_INFO_ << "MUVERA Deserialize config (v" << version << "): num_projections=" << num_projections_
                           << ", num_repeats=" << num_repeats_ << ", seed=" << seed_
                           << ", original_dim=" << original_dim_ << ", num_docs=" << num_docs_
                           << ", encoded_dim=" << encoded_dim_;

        // 4. Deserialize document offsets
        DeserializeEmbListOffsetFromBytes(ptr, emb_list_offset_);

        LOG_KNOWHERE_INFO_ << "MUVERA Deserialize completed";
        return Status::success;
    }

    int32_t
    GetIndexedDim() const override {
        return encoded_dim_;
    }

    int64_t
    GetDocCount() const override {
        return num_docs_;
    }

    std::shared_ptr<EmbListOffset>
    GetEmbListOffset() const override {
        return emb_list_offset_;
    }

 private:
    int32_t num_projections_ = 0;
    int32_t num_buckets_ = 0;  // = 2^num_projections
    int32_t num_repeats_ = 0;
    int32_t seed_ = 0;
    int32_t original_dim_ = 0;
    int32_t encoded_dim_ = 0;  // = num_repeats * num_buckets * original_dim
    int64_t num_docs_ = 0;

    std::vector<SimHash> simhash_instances_;  // one per repetition

    // Storage for reranking
    std::shared_ptr<EmbListOffset> emb_list_offset_;  // document offsets

    // FDE encoding: converts multi-vector documents to fixed-length vectors
    // @param use_mean: if true, use mean aggregation; if false, use sum aggregation
    void
    EncodeFDE(const float* raw_data, const EmbListOffset& offset, float* encoded_data, int64_t num_items,
              bool use_mean) const {
        std::memset(encoded_data, 0, num_items * encoded_dim_ * sizeof(float));

        // Bucket counts for mean aggregation: [num_items, num_repeats, num_buckets]
        std::vector<int32_t> bucket_counts;
        if (use_mean) {
            bucket_counts.resize(num_items * num_repeats_ * num_buckets_, 0);
        }

        for (int64_t item_id = 0; item_id < num_items; ++item_id) {
            size_t vec_start = offset.offset[item_id];
            size_t vec_end = offset.offset[item_id + 1];
            float* item_encoded = encoded_data + item_id * encoded_dim_;
            int32_t* item_counts = use_mean ? bucket_counts.data() + item_id * num_repeats_ * num_buckets_ : nullptr;

            for (int32_t r = 0; r < num_repeats_; ++r) {
                float* repeat_encoded = item_encoded + r * num_buckets_ * original_dim_;
                int32_t* repeat_counts = use_mean ? item_counts + r * num_buckets_ : nullptr;

                for (size_t vec_idx = vec_start; vec_idx < vec_end; ++vec_idx) {
                    const float* vec = raw_data + vec_idx * original_dim_;
                    int32_t bucket_idx = simhash_instances_[r].GetPartitionIndex(vec);
                    float* bucket = repeat_encoded + bucket_idx * original_dim_;

                    // SIMD accumulation: bucket[i] += vec[i]
                    faiss::cppcontrib::knowhere::fvec_madd(original_dim_, vec, 1.0f, bucket, bucket);
                    if (use_mean) {
                        repeat_counts[bucket_idx]++;
                    }
                }

                // Mean aggregation: divide by count
                if (use_mean) {
                    for (int32_t b = 0; b < num_buckets_; ++b) {
                        if (repeat_counts[b] > 1) {
                            float* bucket = repeat_encoded + b * original_dim_;
                            float inv_count = 1.0f / repeat_counts[b];
                            for (int32_t d = 0; d < original_dim_; ++d) {
                                bucket[d] *= inv_count;
                            }
                        }
                    }
                }
            }
        }
    }
};

EmbListStrategyPtr
CreateMuveraEmbListStrategy() {
    return std::make_unique<MuveraEmbListStrategy>();
}

}  // namespace knowhere
