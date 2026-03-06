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

#ifndef EMB_LIST_STRATEGY_H
#define EMB_LIST_STRATEGY_H

#include <cstring>
#include <functional>
#include <memory>
#include <string>

#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/dataset.h"
#include "knowhere/emb_list_utils.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"

namespace knowhere {

/**
 * @brief Simple iterator interface for incremental ANN result fetching.
 */
class AnnResultIterator {
 public:
    virtual ~AnnResultIterator() = default;
    virtual std::pair<int64_t, float>
    Next() = 0;
    virtual bool
    HasNext() = 0;
};
using AnnResultIteratorPtr = std::shared_ptr<AnnResultIterator>;

/**
 * @brief Context providing callbacks and resources for strategy search.
 *
 * Strategies have full control over search flow and can use these callbacks
 * as needed. Not all strategies use all callbacks.
 */
struct EmbListSearchContext {
    /**
     * @brief Execute ANN search on the underlying index.
     *
     * @param query Query dataset
     * @param k Number of results per query
     * @return Search results [nq, k] with ids and distances
     */
    std::function<expected<DataSetPtr>(const DataSetPtr query, int32_t k)> ann_search;

    /**
     * @brief Get ANN iterators for incremental result fetching.
     *
     * Returns one iterator per query vector. Use this when you need to collect
     * results incrementally (e.g., until enough unique docs are found).
     *
     * @param query Query dataset [nq, dim]
     * @return Vector of iterators, one per query
     */
    std::function<expected<std::vector<AnnResultIteratorPtr>>(const DataSetPtr query)> ann_iterator;

    /**
     * @brief Calculate distances between query vectors and indexed vectors by IDs.
     *
     * @param query Query vectors [nq, dim]
     * @param ids Vector IDs to compute distances for
     * @param ids_len Number of IDs
     * @param is_cosine Whether to use cosine similarity
     * @return Distance matrix [nq, ids_len]
     */
    std::function<expected<DataSetPtr>(const DataSetPtr query, const int64_t* ids, size_t ids_len, bool is_cosine)>
        calc_distance_by_ids;

    /**
     * @brief Retrieve raw vectors by their IDs.
     *
     * @param ids Vector IDs to retrieve
     * @param ids_len Number of IDs
     * @return Raw vectors [ids_len, dim]
     */
    std::function<expected<DataSetPtr>(const int64_t* ids, size_t ids_len)> get_vectors_by_ids;

    /**
     * @brief Get total count of indexed items.
     */
    std::function<int64_t()> get_index_count;

    /**
     * @brief Get query code size for the dataset.
     *
     * @param dataset Query dataset
     * @return Code size in bytes, or error
     */
    std::function<expected<size_t>(const DataSetPtr dataset)> get_query_code_size;

    /**
     * @brief Filtering bitset (document level for most strategies).
     */
    BitsetView bitset;
};

/**
 * @brief Parsed metric info shared by all EmbList strategies.
 */
struct EmbListMetricInfo {
    std::string el_metric_type;
    std::string sub_metric_type;
    EmbListAggFunc agg_func;
    bool larger_is_closer;
    bool is_cosine;
};

/**
 * @brief Parse metric type from config into a shared struct.
 *
 * Extracts el_metric_type, sub_metric_type, agg_func, larger_is_closer, is_cosine
 * from the config's metric_type string. Used by all EmbList strategies.
 */
inline expected<EmbListMetricInfo>
ParseEmbListMetric(const BaseConfig& config) {
    auto metric_type = config.metric_type.value();

    auto el_metric_type_or = get_el_metric_type(metric_type);
    if (!el_metric_type_or.has_value()) {
        return expected<EmbListMetricInfo>::Err(Status::emb_list_inner_error, "invalid metric type: " + metric_type);
    }
    auto el_metric_type = el_metric_type_or.value();

    auto el_agg_func_or = get_emb_list_agg_func(el_metric_type);
    if (!el_agg_func_or.has_value()) {
        return expected<EmbListMetricInfo>::Err(
            Status::emb_list_inner_error, "invalid emb list aggregation function for metric type: " + el_metric_type);
    }

    auto sub_metric_type_or = get_sub_metric_type(metric_type);
    if (!sub_metric_type_or.has_value()) {
        return expected<EmbListMetricInfo>::Err(Status::emb_list_inner_error, "invalid metric type: " + metric_type);
    }
    auto sub_metric_type = sub_metric_type_or.value();

    bool larger_is_closer =
        !(sub_metric_type == metric::L2 || sub_metric_type == metric::HAMMING || sub_metric_type == metric::JACCARD);
    bool is_cosine = (sub_metric_type == metric::COSINE);

    return EmbListMetricInfo{
        std::move(el_metric_type), std::move(sub_metric_type), el_agg_func_or.value(), larger_is_closer, is_cosine,
    };
}

/**
 * @brief Rerank candidate documents by computing exact distances via CalcDistByIDs.
 *
 * Shared rerank logic used by all EmbList strategies (TokenANN, MUVERA, LEMUR).
 * For each candidate document, retrieves its vector IDs, computes distances to
 * query vectors via ctx.calc_distance_by_ids, and aggregates scores using agg_func.
 *
 * @return Status::success or Status::emb_list_inner_error on failure
 */
inline Status
RerankByCalcDistByIDs(const std::vector<int64_t>& candidate_docs, const DataSetPtr& query_dataset, size_t nq, int32_t k,
                      bool larger_is_closer, bool is_cosine, const std::shared_ptr<EmbListOffset>& emb_list_offset,
                      const EmbListSearchContext& ctx, const EmbListAggFunc& agg_func, int64_t* out_ids,
                      float* out_dists, size_t& out_doc_vecs, size_t& out_distance_computations) {
    bool has_error = false;
    std::string error_msg;

    auto compute_score = [&](int64_t doc_id) -> std::optional<float> {
        if (has_error) {
            return std::nullopt;
        }

        auto vids = emb_list_offset->get_vids((size_t)doc_id);
        out_doc_vecs += vids.size();
        out_distance_computations += nq * vids.size();
        auto bf_search_res = ctx.calc_distance_by_ids(query_dataset, vids.data(), vids.size(), is_cosine);
        if (!bf_search_res.has_value()) {
            has_error = true;
            error_msg = bf_search_res.what();
            return std::nullopt;
        }
        const auto* bf_dists = bf_search_res.value()->GetDistance();
        return agg_func(bf_dists, nq, vids.size(), larger_is_closer);
    };

    RerankCandidates(candidate_docs, k, larger_is_closer, compute_score, out_ids, out_dists);

    if (has_error) {
        return Status::emb_list_inner_error;
    }
    return Status::success;
}

/**
 * @brief Serialize EmbListOffset to raw bytes: [size_t count][size_t[count] offsets]
 */
inline void
SerializeEmbListOffsetToBytes(const std::shared_ptr<EmbListOffset>& emb_list_offset, uint8_t* ptr) {
    size_t num_offsets = emb_list_offset->offset.size();
    std::memcpy(ptr, &num_offsets, sizeof(size_t));
    std::memcpy(ptr + sizeof(size_t), emb_list_offset->offset.data(), num_offsets * sizeof(size_t));
}

inline size_t
EmbListOffsetByteSize(const std::shared_ptr<EmbListOffset>& emb_list_offset) {
    return sizeof(size_t) + emb_list_offset->offset.size() * sizeof(size_t);
}

/**
 * @brief Deserialize EmbListOffset from raw bytes: [size_t count][size_t[count] offsets]
 * @return Number of bytes consumed
 */
inline size_t
DeserializeEmbListOffsetFromBytes(const uint8_t* ptr, std::shared_ptr<EmbListOffset>& out) {
    size_t num_offsets = 0;
    std::memcpy(&num_offsets, ptr, sizeof(size_t));
    std::vector<size_t> offset(num_offsets);
    std::memcpy(offset.data(), ptr + sizeof(size_t), num_offsets * sizeof(size_t));
    out = std::make_shared<EmbListOffset>(std::move(offset));
    return sizeof(size_t) + num_offsets * sizeof(size_t);
}

/**
 * @brief Abstract interface for EmbList encoding/search strategies.
 *
 * EmbList (Embedding List) represents multi-vector documents where each document
 * consists of multiple vectors. Different strategies handle these differently:
 *
 * - TokenANN: Index all vectors, search at vector level, aggregate to document level
 * - MUVERA: Encode each document to single vector (FDE), search encoded vectors, rerank with MaxSim
 * - PLAID: Use centroid-based retrieval with inverted index, then exact MaxSim
 *
 * Strategies have full control over the search flow and can implement arbitrary
 * multi-stage pipelines.
 */
class EmbListStrategy {
 public:
    virtual ~EmbListStrategy() = default;

    /**
     * @brief Strategy type identifier
     */
    virtual std::string
    Type() const = 0;

    /**
     * @brief Prepare data for building the underlying ANN index.
     *
     * @param dataset Original dataset containing all vectors [N, dim]
     * @param doc_offset Document offsets defining vector groupings
     * @param config Build configuration
     * @return Dataset to be indexed by underlying ANN index, or nullopt if no ANN index needed
     *
     * TokenANN: returns original dataset (index all N vectors)
     * MUVERA: returns FDE-encoded dataset [M, encoded_dim] where M = num_docs
     * PLAID: returns centroid vectors for ANN indexing
     */
    virtual expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) = 0;

    /**
     * @brief Called after underlying ANN index is built.
     *
     * Allows strategy to perform post-build setup (e.g., storing raw data,
     * building additional indexes like inverted index for PLAID).
     *
     * @param dataset Original dataset
     * @param doc_offset Document offsets
     * @param config Build configuration
     * @return Status
     */
    virtual Status
    OnBuildComplete(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) {
        return Status::success;
    }

    /**
     * @brief Check if strategy needs ID mapping for bitset filtering.
     *
     * TokenANN strategy needs vector_id -> doc_id mapping for 1-hop bitset check.
     * MUVERA/PLAID don't need this since they index at document/centroid level.
     */
    virtual bool
    NeedsBaseIndexIDMap() const = 0;

    /**
     * @brief Check if strategy needs raw vector storage at IndexNode level.
     *
     * MUVERA/LEMUR encode documents into different representations for ANN search,
     * so the base index doesn't hold raw vectors. They need a separate raw vector
     * store for reranking via CalcDistByIDs.
     * TokenANN indexes raw vectors directly, so it doesn't need this.
     */
    virtual bool
    NeedsRawVectorStorage() const {
        return false;
    }

    /**
     * @brief Execute search with full control over the search flow.
     *
     * Strategy controls the entire search pipeline and can implement arbitrary
     * multi-stage retrieval (e.g., PLAID's centroid -> inverted index -> exact scoring).
     *
     * @param query_dataset Original query dataset containing query vectors
     * @param query_offset Query document offsets (queries can also be multi-vector)
     * @param k Number of documents to return per query
     * @param config Search configuration
     * @param ctx Search context providing callbacks (ANN search, distance calc, etc.)
     * @return Search results [num_query_docs, k] with document IDs and scores
     */
    virtual expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, int32_t k, const BaseConfig& config,
           const EmbListSearchContext& ctx) const = 0;

    /**
     * @brief Serialize strategy-specific data to a single blob.
     *
     * Each strategy defines its own blob format (e.g., config + offsets + weights).
     * The blob is opaque to IndexNode — only the strategy knows its internal layout.
     *
     * @param[out] data Serialized blob
     * @param[out] size Blob size in bytes
     * @return Status
     */
    virtual Status
    Serialize(std::shared_ptr<uint8_t[]>& data, int64_t& size) const = 0;

    /**
     * @brief Deserialize strategy-specific data from a single blob.
     *
     * @param data Raw bytes (strategy blob only, without magic/type prefix)
     * @param size Blob size in bytes
     * @param config Configuration for deserialization
     * @return Status
     */
    virtual Status
    Deserialize(const uint8_t* data, int64_t size, const BaseConfig& config) = 0;

    /**
     * @brief Get emb_list offset structure (shared).
     *
     * Returns nullptr if strategy doesn't maintain emb_list offsets.
     */
    virtual std::shared_ptr<EmbListOffset>
    GetEmbListOffset() const {
        return nullptr;
    }

    /**
     * @brief Get the dimension of vectors indexed by underlying ANN.
     *
     * TokenANN: original dim
     * MUVERA: encoded dim
     * PLAID: centroid dim (same as original)
     *
     * Returns 0 if strategy doesn't use ANN index.
     */
    virtual int32_t
    GetIndexedDim() const = 0;

    /**
     * @brief Get number of documents.
     */
    virtual int64_t
    GetDocCount() const = 0;
};

using EmbListStrategyPtr = std::unique_ptr<EmbListStrategy>;

// Magic number for EMB_LIST_META format: [magic][type_len][type][strategy_blob]
constexpr int32_t kEmbListMetaMagic = 0x454C4D46;  // "ELMF"

/**
 * @brief Factory function to create EmbList strategy.
 *
 * @param strategy_type Strategy type: "tokenann", "muvera", "lemur", etc.
 * @param config Configuration containing strategy-specific parameters
 * @return Strategy instance or error
 */
expected<EmbListStrategyPtr>
CreateEmbListStrategy(const std::string& strategy_type, const BaseConfig& config);

}  // namespace knowhere

#endif /* EMB_LIST_STRATEGY_H */
