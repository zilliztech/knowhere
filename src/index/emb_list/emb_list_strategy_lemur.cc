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

#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "knowhere/comp/index_param.h"
#include "knowhere/comp/task.h"
#include "knowhere/config.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/log.h"
#include "knowhere/object.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"
#include "simd/hook.h"
#include "simple_mlp.h"

namespace knowhere {

// Find max value in an array
static inline float
FindMax(const float* data, size_t len) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < len; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    return max_val;
}

/**
 * @brief LEMUR (Learned Multi-Vector Retrieval) strategy.
 *
 * LEMUR learns a neural network to compress multi-vector documents into
 * fixed-dimensional representations. Unlike MUVERA's random projections,
 * LEMUR is data-aware and can adapt to the corpus distribution.
 *
 * Training:
 *   1. Sample vectors from corpus as training inputs
 *   2. Compute MaxSim between sampled vectors and all documents as labels
 *   3. Train MLP: input(dim) -> hidden(hidden_dim) -> output(num_docs)
 *   4. W matrix = output_layer weights, shape [num_docs, hidden_dim]
 *
 * Search:
 *   1. Extract query features: feature_extractor(query_vectors)
 *   2. Aggregate query features (sum/mean)
 *   3. Approximate scoring: query_feat @ W.T
 *   4. ANN search on W to find candidates
 *   5. MaxSim reranking on candidates
 */
class LemurEmbListStrategy : public EmbListStrategy {
 public:
    std::string
    Type() const override {
        return meta::EMB_LIST_STRATEGY_LEMUR;
    }

    expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) override {
        if (!dataset) {
            return expected<std::optional<DataSetPtr>>::Err(Status::emb_list_inner_error,
                                                            "LEMUR requires non-null dataset for training");
        }
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. Read config
        hidden_dim_ = config.lemur_hidden_dim.value();
        num_train_samples_ = config.lemur_num_train_samples.value();
        num_epochs_ = config.lemur_num_epochs.value();
        batch_size_ = config.lemur_batch_size.value();
        learning_rate_ = config.lemur_learning_rate.value();
        seed_ = config.lemur_seed.value();
        num_layers_ = config.lemur_num_layers.value();

        original_dim_ = dataset->GetDim();
        num_docs_ = doc_offset.num_el();
        if (num_docs_ > std::numeric_limits<int32_t>::max()) {
            return expected<std::optional<DataSetPtr>>::Err(Status::emb_list_inner_error,
                                                            "num_docs exceeds int32 limit");
        }
        size_t total_vectors = doc_offset.offset.back();

        // Parse metric for ComputeMaxSimLabels
        auto metric_or = ParseEmbListMetric(config);
        if (!metric_or.has_value()) {
            return expected<std::optional<DataSetPtr>>::Err(metric_or.error(), metric_or.what());
        }
        is_l2_ = (metric_or.value().sub_metric_type == metric::L2);

        LOG_KNOWHERE_INFO_ << "LEMUR PrepareDataForBuild: num_docs=" << num_docs_ << ", total_vectors=" << total_vectors
                           << ", original_dim=" << original_dim_ << ", hidden_dim=" << hidden_dim_
                           << ", num_train_samples=" << num_train_samples_ << ", epochs=" << num_epochs_;

        // 2. Store doc_offset for reranking
        emb_list_offset_ = std::make_shared<EmbListOffset>(doc_offset.offset);
        const float* raw_data = static_cast<const float*>(dataset->GetTensor());

        // 3. Sample training vectors (reservoir sampling: O(actual_samples) memory).
        std::mt19937 rng(seed_);
        int32_t actual_samples = std::min(num_train_samples_, static_cast<int32_t>(total_vectors));
        // Memory check: peak usage is ~2 × actual_samples × num_docs × 4B (y_train + y_train_raw)
        // plus actual_samples × original_dim × 4B (X_train). Cap at 4GB to avoid OOM.
        constexpr size_t kMaxTrainingMemoryBytes = 4ULL * 1024 * 1024 * 1024;  // 4GB
        size_t label_matrix_bytes = static_cast<size_t>(actual_samples) * num_docs_ * sizeof(float);
        size_t estimated_peak_bytes =
            2 * label_matrix_bytes + static_cast<size_t>(actual_samples) * original_dim_ * sizeof(float);
        if (estimated_peak_bytes > kMaxTrainingMemoryBytes) {
            LOG_KNOWHERE_ERROR_ << "LEMUR: estimated training memory " << (estimated_peak_bytes >> 20)
                                << " MB exceeds limit " << (kMaxTrainingMemoryBytes >> 20) << " MB"
                                << " (actual_samples=" << actual_samples << ", num_docs=" << num_docs_ << ")";
            return expected<std::optional<DataSetPtr>>::Err(
                Status::emb_list_inner_error,
                "LEMUR training memory exceeds 4GB limit, reduce num_train_samples or num_docs");
        }

        std::vector<int32_t> sample_indices(actual_samples);
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        for (size_t i = actual_samples; i < total_vectors; ++i) {
            std::uniform_int_distribution<size_t> dist(0, i);
            size_t j = dist(rng);
            if (j < static_cast<size_t>(actual_samples)) {
                sample_indices[j] = static_cast<int32_t>(i);
            }
        }

        std::vector<float> X_train(actual_samples * original_dim_);
        for (int32_t i = 0; i < actual_samples; ++i) {
            std::memcpy(X_train.data() + i * original_dim_, raw_data + sample_indices[i] * original_dim_,
                        original_dim_ * sizeof(float));
        }

        LOG_KNOWHERE_INFO_ << "LEMUR: Sampled " << actual_samples << " vectors for training";

        // 4. Compute training labels (MaxSim for each sample vector against each document)
        auto label_start = std::chrono::high_resolution_clock::now();
        std::vector<float> y_train(actual_samples * num_docs_);
        ComputeMaxSimLabels(X_train.data(), actual_samples, raw_data, doc_offset, y_train.data());
        auto label_end = std::chrono::high_resolution_clock::now();
        double label_ms = std::chrono::duration<double, std::milli>(label_end - label_start).count();

        LOG_KNOWHERE_INFO_ << "LEMUR: Computed MaxSim labels in " << label_ms << " ms";

        // 5. Save raw labels for OLS (original LEMUR uses raw MaxSim, not normalized)
        std::vector<float> y_train_raw = y_train;

        // 6. Normalize labels (z-score normalization for stable MLP training)
        float label_mean = 0.0f, label_std = 0.0f;
        size_t label_count = actual_samples * num_docs_;
        for (size_t i = 0; i < label_count; ++i) {
            label_mean += y_train[i];
        }
        label_mean /= label_count;

        for (size_t i = 0; i < label_count; ++i) {
            float diff = y_train[i] - label_mean;
            label_std += diff * diff;
        }
        label_std = std::sqrt(label_std / label_count);
        if (label_std < 1e-6f) {
            label_std = 1.0f;
        }

        for (size_t i = 0; i < label_count; ++i) {
            y_train[i] = (y_train[i] - label_mean) / label_std;
        }
        LOG_KNOWHERE_INFO_ << "LEMUR: Label normalization - mean=" << label_mean << ", std=" << label_std;

        // 7. Train MLP
        // SimpleMLP(input_dim, output_dim, hidden_dim, final_hidden_dim, num_layers, seed)
        auto train_start = std::chrono::high_resolution_clock::now();
        mlp_ = std::make_unique<SimpleMLP>(original_dim_, static_cast<int32_t>(num_docs_), hidden_dim_, hidden_dim_,
                                           num_layers_, seed_);
        float final_loss =
            mlp_->Train(X_train.data(), y_train.data(), actual_samples, num_epochs_, batch_size_, learning_rate_);
        mlp_->ReleaseTrainingBuffers();
        auto train_end = std::chrono::high_resolution_clock::now();
        double train_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();

        LOG_KNOWHERE_INFO_ << "LEMUR: MLP training completed in " << train_ms << " ms, final_loss=" << final_loss;

        // 8. Compute W using pseudoinverse (fit_corpus step from original LEMUR)
        // W = pinv(Z) @ Y, where Z = features of sampled vectors, Y = MaxSim labels
        // This is a least-squares solution: find W such that Z @ W^T ≈ Y
        auto ols_start = std::chrono::high_resolution_clock::now();
        final_hidden_dim_ = mlp_->FinalHiddenDim();

        // Extract features for all training samples: Z [num_samples, hidden_dim]
        std::vector<float> Z(actual_samples * final_hidden_dim_);
        mlp_->ExtractFeatures(X_train.data(), actual_samples, Z.data());

        // Compute W = pinv(Z) @ Y using normal equations: W = (Z^T Z)^{-1} Z^T Y
        // For numerical stability, we solve the normal equations directly
        // Z^T Z: [hidden_dim, hidden_dim]
        // Z^T Y: [hidden_dim, num_docs]
        std::vector<float> ZtZ(final_hidden_dim_ * final_hidden_dim_, 0.0f);
        std::vector<float> ZtY(final_hidden_dim_ * num_docs_, 0.0f);

        // Compute Z^T @ Z using BLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, final_hidden_dim_, final_hidden_dim_, actual_samples, 1.0f,
                    Z.data(), final_hidden_dim_, Z.data(), final_hidden_dim_, 0.0f, ZtZ.data(), final_hidden_dim_);

        // Add regularization for numerical stability: ZtZ += lambda * I
        const float lambda = 1e-4f;
        for (int32_t i = 0; i < final_hidden_dim_; ++i) {
            ZtZ[i * final_hidden_dim_ + i] += lambda;
        }

        // Compute Z^T @ Y using BLAS (Y is raw MaxSim labels, matching original LEMUR)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, final_hidden_dim_, num_docs_, actual_samples, 1.0f,
                    Z.data(), final_hidden_dim_, y_train_raw.data(), num_docs_, 0.0f, ZtY.data(), num_docs_);

        // Solve (Z^T Z) @ W^T = Z^T Y using Cholesky decomposition
        // Since ZtZ is symmetric positive definite (with regularization), use Cholesky
        // L @ L^T = ZtZ, then solve L @ L^T @ X = ZtY

        // Cholesky decomposition: ZtZ = L @ L^T (in-place, lower triangular)
        // ZtZ is positive definite after regularization, so this should always succeed.
        for (int32_t i = 0; i < final_hidden_dim_; ++i) {
            for (int32_t j = 0; j <= i; ++j) {
                float sum = ZtZ[i * final_hidden_dim_ + j];
                for (int32_t k = 0; k < j; ++k) {
                    sum -= ZtZ[i * final_hidden_dim_ + k] * ZtZ[j * final_hidden_dim_ + k];
                }
                if (i == j) {
                    if (sum <= 0.0f) {
                        return expected<std::optional<DataSetPtr>>::Err(
                            Status::emb_list_inner_error,
                            "LEMUR: Cholesky decomposition failed at diagonal " + std::to_string(i));
                    }
                    ZtZ[i * final_hidden_dim_ + i] = std::sqrt(sum);
                } else {
                    ZtZ[i * final_hidden_dim_ + j] = sum / ZtZ[j * final_hidden_dim_ + j];
                }
            }
        }

        // Solve L @ L^T @ W^T = ZtY using BLAS triangular solve (in-place on ZtY)
        cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, final_hidden_dim_, num_docs_,
                    1.0f, ZtZ.data(), final_hidden_dim_, ZtY.data(), num_docs_);
        cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, final_hidden_dim_, num_docs_, 1.0f,
                    ZtZ.data(), final_hidden_dim_, ZtY.data(), num_docs_);
        // ZtY now holds W^T [final_hidden_dim, num_docs]

        auto ols_end = std::chrono::high_resolution_clock::now();
        double ols_ms = std::chrono::duration<double, std::milli>(ols_end - ols_start).count();

        LOG_KNOWHERE_INFO_ << "LEMUR: Computed W via pseudoinverse (OLS) [" << num_docs_ << " x " << final_hidden_dim_
                           << "] in " << ols_ms << " ms";

        // 9. Create dataset from W for ANN indexing (transpose W^T to W)
        auto W_data = std::make_unique<float[]>(num_docs_ * final_hidden_dim_);
        for (int64_t d = 0; d < num_docs_; ++d) {
            for (int32_t h = 0; h < final_hidden_dim_; ++h) {
                W_data[d * final_hidden_dim_ + h] = ZtY[h * num_docs_ + d];
            }
        }
        auto W_dataset = GenDataSet(num_docs_, final_hidden_dim_, W_data.release());
        W_dataset->SetIsOwner(true);

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        LOG_KNOWHERE_INFO_ << "LEMUR PrepareDataForBuild completed in " << total_ms << " ms";

        return std::optional<DataSetPtr>(W_dataset);
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
        if (!mlp_ || !emb_list_offset_) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "LEMUR not initialized");
        }

        // 2. Parse metric type
        auto metric_or = ParseEmbListMetric(config);
        if (!metric_or.has_value()) {
            return expected<DataSetPtr>::Err(metric_or.error(), metric_or.what());
        }
        auto& mi = metric_or.value();

        auto num_query_docs = query_offset.num_el();
        const float* query_data = static_cast<const float*>(query_dataset->GetTensor());

        LOG_KNOWHERE_DEBUG_ << "LEMUR Search: num_query_docs=" << num_query_docs << ", k=" << k;

        // 3. Extract query features and aggregate
        auto feat_start = std::chrono::high_resolution_clock::now();

        // Batch extract features for all query tokens at once
        size_t total_query_tokens = query_offset.offset[num_query_docs];
        if (total_query_tokens > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "total query tokens exceeds int32 limit");
        }
        std::vector<float> all_feats(total_query_tokens * final_hidden_dim_);
        mlp_->ExtractFeatures(query_data, static_cast<int32_t>(total_query_tokens), all_feats.data());

        // Aggregate per query doc: sum over tokens
        std::vector<float> query_feats(num_query_docs * final_hidden_dim_, 0.0f);
        for (size_t q = 0; q < num_query_docs; ++q) {
            size_t q_vec_start = query_offset.offset[q];
            size_t q_vec_end = query_offset.offset[q + 1];
            size_t nq = q_vec_end - q_vec_start;

            float* q_feat = query_feats.data() + q * final_hidden_dim_;
            const float* token_feats = all_feats.data() + q_vec_start * final_hidden_dim_;
            for (size_t t = 0; t < nq; ++t) {
                for (int32_t d = 0; d < final_hidden_dim_; ++d) {
                    q_feat[d] += token_feats[t * final_hidden_dim_ + d];
                }
            }
        }
        auto feat_end = std::chrono::high_resolution_clock::now();
        double feat_ms = std::chrono::duration<double, std::milli>(feat_end - feat_start).count();

        LOG_KNOWHERE_DEBUG_ << "[LEMUR] Stage1 Feature extraction: " << feat_ms << " ms";

        // 4. ANN search on W
        bool do_rerank = config.emb_list_rerank.value_or(true);
        auto query_feat_dataset = GenDataSet(num_query_docs, final_hidden_dim_, query_feats.data());

        int32_t ann_k;
        if (do_rerank) {
            auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
            ann_k =
                std::min(std::max(static_cast<int32_t>(k * retrieval_ann_ratio), 1), static_cast<int32_t>(num_docs_));
        } else {
            ann_k = std::min(k, static_cast<int32_t>(num_docs_));
        }

        auto ann_start = std::chrono::high_resolution_clock::now();
        auto ann_result = ctx.ann_search(query_feat_dataset, ann_k);
        if (!ann_result.has_value()) {
            LOG_KNOWHERE_ERROR_ << "LEMUR ANN search failed: " << ann_result.what();
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "ANN search failed");
        }
        auto ann_end = std::chrono::high_resolution_clock::now();
        double ann_ms = std::chrono::duration<double, std::milli>(ann_end - ann_start).count();
        const auto* ann_ids = ann_result.value()->GetIds();

        LOG_KNOWHERE_DEBUG_ << "[LEMUR] Stage2 ANN search: " << ann_ms << " ms, ann_k=" << ann_k;

        // 5. If reranking disabled, return ANN results directly
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
            return GenResultDataSet(static_cast<int64_t>(num_query_docs), static_cast<int64_t>(k), std::move(ids),
                                    std::move(dists));
        }

        // 6. MaxSim reranking via CalcDistByIDs
        auto ids = std::make_unique<int64_t[]>(num_query_docs * k);
        auto dists = std::make_unique<float[]>(num_query_docs * k);

        auto rerank_start = std::chrono::high_resolution_clock::now();

        // Statistics for logging
        size_t total_candidates = 0;
        size_t total_doc_vecs = 0;
        size_t total_query_vecs = 0;
        size_t total_distance_computations = 0;

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
        LOG_KNOWHERE_DEBUG_ << "[LEMUR] Stage3 Rerank: " << rerank_ms << " ms"
                            << ", total_candidates=" << total_candidates << ", avg_doc_len=" << avg_doc_len
                            << ", avg_query_len=" << avg_query_len
                            << ", total_dist_comps=" << total_distance_computations;

        return GenResultDataSet(static_cast<int64_t>(num_query_docs), static_cast<int64_t>(k), std::move(ids),
                                std::move(dists));
    }

    Status
    Serialize(std::shared_ptr<uint8_t[]>& data, int64_t& size) const override {
        // Blob format: [config][mlp_flag][mlp_data (feature_extractor only, no W_out)][offsets]
        // Config: [magic][version][hidden_dim][final_hidden_dim][num_layers][original_dim][num_docs]
        constexpr int32_t kMagic = 0x4C454D52;  // "LEMR"
        constexpr int32_t kVersion = 1;
        size_t config_size = 6 * sizeof(int32_t) + sizeof(int64_t);  // magic + version + 4 fields + num_docs

        // Calculate MLP size (feature_extractor weights only)
        int32_t has_mlp = mlp_ ? 1 : 0;
        size_t mlp_size = 0;
        if (mlp_) {
            const auto& fc_weights = mlp_->GetFcWeights();
            const auto& fc_biases = mlp_->GetFcBiases();
            const auto& ln_gammas = mlp_->GetLnGammas();
            const auto& ln_betas = mlp_->GetLnBetas();

            mlp_size = sizeof(int32_t);  // num_layers
            for (int32_t i = 0; i < num_layers_; ++i) {
                mlp_size += 4 * sizeof(size_t);
                mlp_size += fc_weights[i].size() * sizeof(float);
                mlp_size += fc_biases[i].size() * sizeof(float);
                mlp_size += ln_gammas[i].size() * sizeof(float);
                mlp_size += ln_betas[i].size() * sizeof(float);
            }
        }

        size_t offset_size = EmbListOffsetByteSize(emb_list_offset_);
        size = static_cast<int64_t>(config_size + sizeof(int32_t) + mlp_size + offset_size);
        data = std::shared_ptr<uint8_t[]>(new uint8_t[size]);
        uint8_t* ptr = data.get();

        // 1. Config
        std::memcpy(ptr, &kMagic, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &kVersion, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &hidden_dim_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &final_hidden_dim_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_layers_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &original_dim_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_docs_, sizeof(int64_t));
        ptr += sizeof(int64_t);

        // 2. MLP flag + feature_extractor weights (no W_out)
        std::memcpy(ptr, &has_mlp, sizeof(int32_t));
        ptr += sizeof(int32_t);
        if (mlp_) {
            const auto& fc_weights = mlp_->GetFcWeights();
            const auto& fc_biases = mlp_->GetFcBiases();
            const auto& ln_gammas = mlp_->GetLnGammas();
            const auto& ln_betas = mlp_->GetLnBetas();

            std::memcpy(ptr, &num_layers_, sizeof(int32_t));
            ptr += sizeof(int32_t);

            for (int32_t i = 0; i < num_layers_; ++i) {
                size_t sz = fc_weights[i].size();
                std::memcpy(ptr, &sz, sizeof(size_t));
                ptr += sizeof(size_t);
                std::memcpy(ptr, fc_weights[i].data(), sz * sizeof(float));
                ptr += sz * sizeof(float);

                sz = fc_biases[i].size();
                std::memcpy(ptr, &sz, sizeof(size_t));
                ptr += sizeof(size_t);
                std::memcpy(ptr, fc_biases[i].data(), sz * sizeof(float));
                ptr += sz * sizeof(float);

                sz = ln_gammas[i].size();
                std::memcpy(ptr, &sz, sizeof(size_t));
                ptr += sizeof(size_t);
                std::memcpy(ptr, ln_gammas[i].data(), sz * sizeof(float));
                ptr += sz * sizeof(float);

                sz = ln_betas[i].size();
                std::memcpy(ptr, &sz, sizeof(size_t));
                ptr += sizeof(size_t);
                std::memcpy(ptr, ln_betas[i].data(), sz * sizeof(float));
                ptr += sz * sizeof(float);
            }
        }

        // 3. Offsets
        SerializeEmbListOffsetToBytes(emb_list_offset_, ptr);

        LOG_KNOWHERE_DEBUG_ << "LEMUR Serialize completed";
        return Status::success;
    }

    Status
    Deserialize(const uint8_t* data, int64_t size, const BaseConfig& config) override {
        constexpr int32_t kMagic = 0x4C454D52;  // "LEMR"
        const uint8_t* ptr = data;
        const uint8_t* end = data + size;

        // 1. Read magic and version
        constexpr size_t kMinConfigSize = 6 * sizeof(int32_t) + sizeof(int64_t);  // must match Serialize
        if (size < static_cast<int64_t>(kMinConfigSize)) {
            LOG_KNOWHERE_ERROR_ << "LEMUR: blob too small: " << size << " < " << kMinConfigSize;
            return Status::emb_list_inner_error;
        }

        int32_t magic = 0;
        std::memcpy(&magic, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        if (magic != kMagic) {
            LOG_KNOWHERE_ERROR_ << "LEMUR: invalid magic number 0x" << std::hex << magic;
            return Status::emb_list_inner_error;
        }

        int32_t version = 0;
        std::memcpy(&version, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);

        // 2. Deserialize config: [hidden_dim][final_hidden_dim][num_layers][original_dim][num_docs(i64)]
        if (version > 1) {
            LOG_KNOWHERE_ERROR_ << "LEMUR: unsupported serialization version " << version << " (max supported: 1)";
            return Status::emb_list_inner_error;
        }

        std::memcpy(&hidden_dim_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&final_hidden_dim_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_layers_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&original_dim_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_docs_, ptr, sizeof(int64_t));
        ptr += sizeof(int64_t);

        LOG_KNOWHERE_INFO_ << "LEMUR Deserialize config (v" << version << "): hidden_dim=" << hidden_dim_
                           << ", final_hidden_dim=" << final_hidden_dim_ << ", num_layers=" << num_layers_
                           << ", original_dim=" << original_dim_ << ", num_docs=" << num_docs_;

        if (num_docs_ > std::numeric_limits<int32_t>::max()) {
            LOG_KNOWHERE_ERROR_ << "LEMUR: num_docs " << num_docs_ << " exceeds int32 limit";
            return Status::emb_list_inner_error;
        }

        // 3. Deserialize MLP weights
        int32_t has_mlp = 0;
        std::memcpy(&has_mlp, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);

        if (has_mlp) {
            const uint8_t* mlp_ptr = ptr;
            const uint8_t* mlp_end = end;

            int32_t saved_num_layers;
            std::memcpy(&saved_num_layers, mlp_ptr, sizeof(int32_t));
            mlp_ptr += sizeof(int32_t);

            if (saved_num_layers != num_layers_) {
                LOG_KNOWHERE_ERROR_ << "LEMUR: saved num_layers " << saved_num_layers << " != expected " << num_layers_;
                return Status::emb_list_inner_error;
            }

            std::vector<std::vector<float>> fc_weights(saved_num_layers);
            std::vector<std::vector<float>> fc_biases(saved_num_layers);
            std::vector<std::vector<float>> ln_gammas(saved_num_layers);
            std::vector<std::vector<float>> ln_betas(saved_num_layers);

            for (int32_t i = 0; i < saved_num_layers; ++i) {
                size_t sz;

                if (mlp_ptr + sizeof(size_t) > mlp_end) {
                    LOG_KNOWHERE_ERROR_ << "LEMUR: MLP binary truncated at layer " << i;
                    return Status::emb_list_inner_error;
                }
                std::memcpy(&sz, mlp_ptr, sizeof(size_t));
                mlp_ptr += sizeof(size_t);
                if (mlp_ptr + sz * sizeof(float) > mlp_end) {
                    LOG_KNOWHERE_ERROR_ << "LEMUR: MLP binary truncated at fc_weights layer " << i;
                    return Status::emb_list_inner_error;
                }
                fc_weights[i].resize(sz);
                std::memcpy(fc_weights[i].data(), mlp_ptr, sz * sizeof(float));
                mlp_ptr += sz * sizeof(float);

                if (mlp_ptr + sizeof(size_t) > mlp_end) {
                    LOG_KNOWHERE_ERROR_ << "LEMUR: MLP binary truncated at layer " << i;
                    return Status::emb_list_inner_error;
                }
                std::memcpy(&sz, mlp_ptr, sizeof(size_t));
                mlp_ptr += sizeof(size_t);
                if (mlp_ptr + sz * sizeof(float) > mlp_end) {
                    LOG_KNOWHERE_ERROR_ << "LEMUR: MLP binary truncated at fc_biases layer " << i;
                    return Status::emb_list_inner_error;
                }
                fc_biases[i].resize(sz);
                std::memcpy(fc_biases[i].data(), mlp_ptr, sz * sizeof(float));
                mlp_ptr += sz * sizeof(float);

                if (mlp_ptr + sizeof(size_t) > mlp_end) {
                    LOG_KNOWHERE_ERROR_ << "LEMUR: MLP binary truncated at layer " << i;
                    return Status::emb_list_inner_error;
                }
                std::memcpy(&sz, mlp_ptr, sizeof(size_t));
                mlp_ptr += sizeof(size_t);
                if (mlp_ptr + sz * sizeof(float) > mlp_end) {
                    LOG_KNOWHERE_ERROR_ << "LEMUR: MLP binary truncated at ln_gammas layer " << i;
                    return Status::emb_list_inner_error;
                }
                ln_gammas[i].resize(sz);
                std::memcpy(ln_gammas[i].data(), mlp_ptr, sz * sizeof(float));
                mlp_ptr += sz * sizeof(float);

                if (mlp_ptr + sizeof(size_t) > mlp_end) {
                    LOG_KNOWHERE_ERROR_ << "LEMUR: MLP binary truncated at layer " << i;
                    return Status::emb_list_inner_error;
                }
                std::memcpy(&sz, mlp_ptr, sizeof(size_t));
                mlp_ptr += sizeof(size_t);
                if (mlp_ptr + sz * sizeof(float) > mlp_end) {
                    LOG_KNOWHERE_ERROR_ << "LEMUR: MLP binary truncated at ln_betas layer " << i;
                    return Status::emb_list_inner_error;
                }
                ln_betas[i].resize(sz);
                std::memcpy(ln_betas[i].data(), mlp_ptr, sz * sizeof(float));
                mlp_ptr += sz * sizeof(float);
            }

            mlp_ = std::make_unique<SimpleMLP>(original_dim_, static_cast<int32_t>(num_docs_), hidden_dim_,
                                               final_hidden_dim_, num_layers_, 0);
            mlp_->SetFcWeights(fc_weights);
            mlp_->SetFcBiases(fc_biases);
            mlp_->SetLnGammas(ln_gammas);
            mlp_->SetLnBetas(ln_betas);
            mlp_->MarkAsTrained();

            ptr = mlp_ptr;
        }

        // 4. Deserialize document offsets
        DeserializeEmbListOffsetFromBytes(ptr, emb_list_offset_);

        LOG_KNOWHERE_DEBUG_ << "LEMUR Deserialize completed";
        return Status::success;
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
    /**
     * @brief Compute MaxSim labels for training (chunk-parallel, fused sgemm+reduce).
     *
     * For each sample vector v and each document d:
     *   label[v, d] = MaxSim(v, d) = max over all tokens t in d: IP(v, t)
     *
     * Splits raw_data into doc-aligned chunks. Each thread pool task computes a small
     * sgemm for its chunk and immediately reduces to MaxSim, avoiding a massive
     * intermediate IP matrix.
     */
    void
    ComputeMaxSimLabels(const float* samples, int32_t num_samples, const float* raw_data, const EmbListOffset& offset,
                        float* labels) const {
        ScopedBLASThreads blas_guard(1);

        auto pool = ThreadPool::GetGlobalSearchThreadPool();
        size_t num_docs = offset.num_el();
        size_t total_vecs = offset.offset.back();

        // For L2 metric: score = 2*IP(v,t) - ||t||^2, take max over tokens
        // (equivalent to min L2, since ||v||^2 is constant per sample and cancels in z-score normalization)
        std::vector<float> token_sq_norms;
        if (is_l2_) {
            token_sq_norms.resize(total_vecs);
            for (size_t i = 0; i < total_vecs; ++i) {
                const float* vec = raw_data + i * original_dim_;
                float norm = 0.0f;
                for (int32_t d = 0; d < original_dim_; ++d) {
                    norm += vec[d] * vec[d];
                }
                token_sq_norms[i] = norm;
            }
        }

        // Initialize labels to -inf
        std::fill(labels, labels + static_cast<size_t>(num_samples) * num_docs,
                  -std::numeric_limits<float>::infinity());

        // Build doc-aligned chunks: each chunk contains complete docs.
        // Target ~50K vectors per chunk so each thread's sgemm output (kSampleBatchSize * chunk_vecs * 4B)
        // fits comfortably in L3 cache (~100MB per thread at kSampleBatchSize=512).
        constexpr size_t kTargetChunkVecs = 50000;
        struct Chunk {
            size_t doc_start;  // first doc index
            size_t doc_end;    // one past last doc index
            size_t vec_start;  // first vector index
            size_t vec_end;    // one past last vector index
        };
        std::vector<Chunk> chunks;
        {
            size_t cur_doc = 0;
            while (cur_doc < num_docs) {
                size_t chunk_doc_start = cur_doc;
                size_t chunk_vec_start = offset.offset[cur_doc];
                size_t chunk_vec_end = chunk_vec_start;

                while (cur_doc < num_docs) {
                    chunk_vec_end = offset.offset[cur_doc + 1];
                    cur_doc++;
                    if (chunk_vec_end - chunk_vec_start >= kTargetChunkVecs) {
                        break;
                    }
                }
                chunks.push_back({chunk_doc_start, cur_doc, chunk_vec_start, chunk_vec_end});
            }
        }

        // Process samples in batches to limit per-thread memory.
        // Each thread allocates kSampleBatchSize * chunk_vecs * 4B (~100MB at 512 * 50K).
        constexpr int32_t kSampleBatchSize = 512;

        for (int32_t sample_start = 0; sample_start < num_samples; sample_start += kSampleBatchSize) {
            int32_t sample_end = std::min(sample_start + kSampleBatchSize, num_samples);
            int32_t batch_samples = sample_end - sample_start;
            const float* batch_samples_data = samples + sample_start * original_dim_;

            // Parallel over chunks: each task does small sgemm + immediate reduce
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(chunks.size());

            for (const auto& chunk : chunks) {
                futs.emplace_back(pool->push([&, chunk, batch_samples, sample_start, batch_samples_data]() {
                    size_t chunk_vecs = chunk.vec_end - chunk.vec_start;

                    // sgemm computes IP: (batch_samples, dim) × (dim, chunk_vecs)
                    std::vector<float> local_ips(batch_samples * chunk_vecs);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_samples, chunk_vecs, original_dim_, 1.0f,
                                batch_samples_data, original_dim_, raw_data + chunk.vec_start * original_dim_,
                                original_dim_, 0.0f, local_ips.data(), chunk_vecs);

                    // For L2: convert IP to negative L2 score: 2*IP - ||t||^2
                    if (is_l2_) {
                        for (int32_t s = 0; s < batch_samples; ++s) {
                            float* ip_row = local_ips.data() + s * chunk_vecs;
                            for (size_t v = 0; v < chunk_vecs; ++v) {
                                ip_row[v] = 2.0f * ip_row[v] - token_sq_norms[chunk.vec_start + v];
                            }
                        }
                    }

                    // Immediate reduce: MaxSim per doc within this chunk
                    for (int32_t s = 0; s < batch_samples; ++s) {
                        const float* ip_row = local_ips.data() + s * chunk_vecs;
                        float* label_row = labels + (sample_start + s) * num_docs;

                        for (size_t d = chunk.doc_start; d < chunk.doc_end; ++d) {
                            size_t doc_vec_start = offset.offset[d] - chunk.vec_start;
                            size_t doc_vec_end = offset.offset[d + 1] - chunk.vec_start;
                            label_row[d] = FindMax(ip_row + doc_vec_start, doc_vec_end - doc_vec_start);
                        }
                    }
                }));
            }

            WaitAllSuccess(futs);
        }
    }

 private:
    // Config
    int32_t hidden_dim_ = 256;
    int32_t final_hidden_dim_ = 256;
    int32_t num_layers_ = 2;
    int32_t num_train_samples_ = 10000;
    int32_t num_epochs_ = 50;
    int32_t batch_size_ = 64;
    float learning_rate_ = 0.001f;
    int32_t seed_ = 42;

    int32_t original_dim_ = 0;
    int64_t num_docs_ = 0;
    bool is_l2_ = false;

    // MLP model
    std::unique_ptr<SimpleMLP> mlp_;

    // Storage for reranking
    std::shared_ptr<EmbListOffset> emb_list_offset_;
};

EmbListStrategyPtr
CreateLemurEmbListStrategy() {
    return std::make_unique<LemurEmbListStrategy>();
}

}  // namespace knowhere
