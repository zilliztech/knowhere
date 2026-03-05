#pragma once

#include <fcntl.h>
#include <sys/mman.h>

#include <boost/core/make_span.hpp>
#include <boost/core/span.hpp>
#include <cstdint>

#include "index/sparse/block_max_data.h"
#include "index/sparse/scorer.h"
#include "index/sparse/searcher/block_max_maxscore.h"
#include "index/sparse/searcher/block_max_wand.h"
#include "index/sparse/searcher/daat_maxscore.h"
#include "index/sparse/searcher/daat_wand.h"
#include "index/sparse/searcher/taat_naive.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/operands.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere::sparse::inverted {

enum class InvertedIndexAlgo : uint32_t {
    TAAT_NAIVE = 0,
    DAAT_MAXSCORE = 1,
    DAAT_WAND = 2,
    BLOCK_MAX_MAXSCORE = 3,
    BLOCK_MAX_WAND = 4,
};

enum class InvertedIndexEncoding : uint32_t {
    FLAT = 0,
    BLOCK_STREAMVBYTE = 1,
    BLOCK_MASKEDVBYTE = 2,
    FIXED_DOCID_WINDOWS = 3
};

enum class InvertedIndexSectionType : uint32_t {
    POSTING_LISTS = 0,
    METRIC_PARAMS = 1,
    DIM_MAP = 2,
    ROW_SUMS = 3,
    MAX_SCORES_PER_DIM = 4,
    BLOCK_MAX_SCORES = 5,
    PROMETHEUS_BUILD_STATS = 6
};

enum class InvertedIndexPrometheusBuildStats : uint32_t { DATASET_NNZ_STATS = 0, POSTING_LIST_LENGTH_STATS = 1 };

struct InvertedIndexSectionHeader {
    InvertedIndexSectionType type;
    uint64_t offset;
    uint64_t size;
};

/**
 * @brief Metadata for the inverted index
 *
 * This struct contains metadata used in the build phase of the inverted index.
 * Currently it has two components:
 * 1. Sum of values in each row, used by BM25
 * 2. Maximum score for each dimension, used by DAAT_MAXSCORE and DAAT_WAND
 * 3. Block max scores, used by BLOCK_MAX_MAXSCORE and BLOCK_MAX_WAND
 * The flags indicate which components are present.
 */
struct InvertedIndexMetaData {
    // Flags indicating which metadata components are present
    static constexpr uint32_t FLAG_NONE = 0;
    static constexpr uint32_t FLAG_HAS_ROW_SUMS = 1 << 0;            // Row sums are present
    static constexpr uint32_t FLAG_HAS_MAX_SCORES_PER_DIM = 1 << 1;  // Maximum scores per dimension are present
    static constexpr uint32_t FLAG_HAS_BLOCK_MAX_SCORES = 1 << 2;    // Block max scores are present
    using MetaDataFlags = uint32_t;
    MetaDataFlags flags_{FLAG_NONE};

    std::vector<float> row_sums_;
    std::vector<float> max_score_per_dim_;
    BlockMaxData block_max_data_;
};

/**
 * @brief Statistics for the build process, which will be used to generate the prometheus metrics
 */
struct InvertedIndexBuildStats {
    std::vector<uint32_t> dataset_nnz_stats_;
    std::vector<uint32_t> posting_list_length_stats_;
};

struct InvertedIndexSearchParams {
    InvertedIndexAlgo algo;
    IndexScorerConfig scorer_config;

    struct {
        float drop_ratio_search;
        float dim_max_score_ratio;
    } approx;
};

template <typename DType>
class InvertedIndex {
 public:
    InvertedIndex() = default;

    virtual ~InvertedIndex() = default;

    /**
     * @brief Get total size of the index in bytes
     */
    [[nodiscard]] virtual size_t
    size() const = 0;

    /**
     * @brief Add sparse vectors to the index
     *
     * @param data Array of sparse vectors to add
     * @param rows Number of vectors to add
     * @param dim Dimensionality of the vectors
     * @return Status success or error code
     */
    virtual Status
    add(const SparseRow<DType>* data, size_t rows, int64_t dim) = 0;

    /**
     * @brief Build index from raw data format, which is compatible with the old index format
     *
     * @param reader Reader containing serialized index data
     * @param enable_mmap Whether to use file backed memory mapping
     * @param backed_filename File to use for memory mapping if enabled
     * @return Status success if deserialization succeeds, error code otherwise
     */
    virtual Status
    build_from_raw_data(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename) = 0;

    /**
     * @brief Convert index to raw data format, which is compatible with the old index format
     * Convert the index data in the following layout:
     * 1. Number of rows (size_t)
     * 2. Number of columns (size_t)
     * 3. Deprecated value threshold (DType)
     * 4. For each row:
     *    - Row length (size_t)
     *    - Row data (array of dimension-value pairs)
     *
     * @param writer Writer to serialize the index to
     * @return Status success if conversion succeeds, error code otherwise
     */
    virtual Status
    convert_to_raw_data(MemoryIOWriter& writer) const = 0;

    /**
     * @brief Serialize the index to a binary format
     *
     * @param writer Writer to serialize the index to
     * @return Status success if serialization succeeds, error code otherwise
     */
    virtual Status
    serialize(MemoryIOWriter& writer) const = 0;

    /**
     * @brief Deserialize the index from a binary format
     *
     * @param reader Reader to deserialize the index from
     * @return Status success if deserialization succeeds, error code otherwise
     */
    virtual Status
    deserialize(MemoryIOReader& reader) = 0;

    /**
     * @brief Search for the top k nearest neighbors
     *
     * @param query The query sparse row
     * @param k The number of nearest neighbors to search for
     * @param distances Array to store the distances to the nearest neighbors
     * @param labels Array to store the labels of the nearest neighbors
     * @param bitset The bitset view of the query
     * @param search_params The search parameters
     */
    virtual void
    search(const SparseRow<DType>& query, size_t k, float* distances, label_t* labels, const BitsetView& bitset,
           const InvertedIndexSearchParams& search_params) const = 0;

    /**
     * @brief Get all distances for a query
     *
     * @param query The query sparse row
     * @param bitset The bitset view of the query
     * @param search_params The search parameters
     * @return std::vector<float> The distances
     */
    virtual std::vector<float>
    get_all_distances(const SparseRow<DType>& query, const BitsetView& bitset,
                      const InvertedIndexSearchParams& search_params) const = 0;

    /**
     * @brief Set the build scorer type and parameters
     *
     * @param scorer_config Scorer config
     */
    void
    set_build_scorer(const IndexScorerConfig& scorer_config) {
        if (scorer_config.scorer_type == IndexScorerType::BM25) {
            meta_data_.flags_ |= InvertedIndexMetaData::FLAG_HAS_ROW_SUMS;
            build_scorer_ = std::make_shared<BM25IndexScorer>(
                scorer_config.scorer_params.bm25.k1, scorer_config.scorer_params.bm25.b,
                scorer_config.scorer_params.bm25.avgdl, meta_data_.row_sums_);
        } else {
            build_scorer_ = std::make_shared<IPIndexScorer>();
        }
    }

    /**
     * @brief Set the build algorithm of inverted index
     *
     * @param build_algo The build algorithm
     * @param args Additional arguments for this algorithm
     */
    template <typename... Args>
    void
    set_build_algo(const std::string& build_algo, Args... args) {
        if (build_algo == "DAAT_MAXSCORE") {
            build_algo_ = InvertedIndexAlgo::DAAT_MAXSCORE;
            meta_data_.flags_ |= InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM;
        } else if (build_algo == "DAAT_WAND") {
            build_algo_ = InvertedIndexAlgo::DAAT_WAND;
            meta_data_.flags_ |= InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM;
        } else if (build_algo == "BLOCK_MAX_MAXSCORE") {
            build_algo_ = InvertedIndexAlgo::BLOCK_MAX_MAXSCORE;
            meta_data_.flags_ |= InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM;
            meta_data_.flags_ |= InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES;
            if constexpr (sizeof...(args) > 0) {
                meta_data_.block_max_data_.block_size_ = std::get<0>(std::forward_as_tuple(args...));
            } else {
                LOG_KNOWHERE_WARNING_ << "No block size provided, using default block size 128";
                meta_data_.block_max_data_.block_size_ = 128;
            }
        } else if (build_algo == "BLOCK_MAX_WAND") {
            build_algo_ = InvertedIndexAlgo::BLOCK_MAX_WAND;
            meta_data_.flags_ |= InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM;
            meta_data_.flags_ |= InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES;
            if constexpr (sizeof...(args) > 0) {
                meta_data_.block_max_data_.block_size_ = std::get<0>(std::forward_as_tuple(args...));
            } else {
                LOG_KNOWHERE_WARNING_ << "No block size provided, using default block size 128";
                meta_data_.block_max_data_.block_size_ = 128;
            }
        } else {
            build_algo_ = InvertedIndexAlgo::TAAT_NAIVE;
        }
    }

    /**
     * @brief Get the build algorithm of inverted index
     *
     * @return InvertedIndexAlgo The build algorithm
     */
    [[nodiscard]] InvertedIndexAlgo
    get_build_algo() const {
        return build_algo_;
    }

    /**
     * @brief Get the row sums
     *
     * @return const std::vector<float>& The row sums
     */
    [[nodiscard]] const std::vector<float>&
    get_row_sums() const {
        return meta_data_.row_sums_;
    }

    /**
     * @brief Get the block max data cursor
     *
     * @param dim_id The dimension ID
     * @return BlockMaxDataCursor The block max data cursor
     */
    [[nodiscard]] BlockMaxDataCursor
    get_block_max_data_cursor(uint32_t dim_id) const {
        uint32_t start = dim_id == 0 ? 0 : this->meta_data_.block_max_data_.block_offsets_[dim_id - 1];
        uint32_t num = this->meta_data_.block_max_data_.block_offsets_[dim_id] - start;
        return {this->meta_data_.block_max_data_.block_max_ids_.subspan(start, num),
                this->meta_data_.block_max_data_.block_max_scores_.subspan(start, num)};
    }

    /**
     * @brief Check if maximum scores per dimension are present
     *
     * @return true if maximum scores per dimension are present, false otherwise
     */
    [[nodiscard]] bool
    has_max_scores_per_dim() const {
        // meta_data_.max_score_per_dim_.size() > 0 is not necessary,
        // because the index could be empty
        return meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM;
    }

    /**
     * @brief Check if block max scores are present
     *
     * @return true if block max scores are present, false otherwise
     */
    [[nodiscard]] bool
    has_block_max_scores() const {
        // meta_data_.block_max_data_.block_max_ids_.size() > 0 is not necessary,
        // because the index could be empty
        return meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES;
    }

    /**
     * @brief Get the maximum score for a dimension
     *
     * @param dim_id The dimension ID
     * @param dim_val The value of the dimension
     * @return float The maximum score
     */
    float
    get_dim_max_score(std::uint32_t dim_id, DType dim_val) const {
        return dim_val * meta_data_.max_score_per_dim_[dim_id];
    }

    /**
     * @brief Get the scorer config
     *
     * @return IndexScorerConfig The scorer config of the index
     */
    [[nodiscard]] const IndexScorerConfig&
    get_scorer_config() const {
        return build_scorer_->config();
    }

    /**
     * @brief Get the number of rows in the index
     *
     * @return size_t The number of rows
     */
    [[nodiscard]] size_t
    nr_rows() const {
        return nr_rows_;
    };

    /**
     * @brief Get the number of columns in the index
     *
     * @return size_t The number of columns
     */
    [[nodiscard]] size_t
    nr_cols() const {
        return max_dim_;
    };

 protected:
    /**
     * @brief Get the threshold of the query
     *
     * @param values The values of the query
     * @param drop_ratio The drop ratio
     * @return DType The threshold
     */
    DType
    get_threshold(std::vector<DType>& values, float drop_ratio) const {
        // drop_ratio is in [0, 1) thus drop_count is guaranteed to be less
        // than values.size().
        auto drop_count = static_cast<size_t>(drop_ratio * values.size());
        if (drop_count == 0) {
            return 0;
        }
        auto pos = values.begin() + drop_count;
        std::nth_element(values.begin(), pos, values.end());
        return *pos;
    }

    /**
     * @brief Parse the query
     *
     * @param query The query
     * @param drop_ratio_search The drop ratio
     * @return std::vector<std::pair<uint32_t, float>> The parsed query
     */
    std::vector<std::pair<uint32_t, float>>
    parse_query(const SparseRow<DType>& query, float drop_ratio_search) const {
        DType q_threshold = 0;
        if (drop_ratio_search != 0) {
            std::vector<DType> values(query.size());
            for (size_t i = 0; i < query.size(); ++i) {
                values[i] = std::abs(query[i].val);
            }
            q_threshold = get_threshold(values, drop_ratio_search);
        }

        std::vector<std::pair<uint32_t, float>> filtered_query;
        for (size_t i = 0; i < query.size(); ++i) {
            auto [dim, val] = query[i];
            auto dim_it = dim_map_.find(dim);
            if (dim_it == dim_map_.cend() || std::abs(val) < q_threshold) {
                continue;
            }
            filtered_query.emplace_back(dim_it->second, val);
        }

        return filtered_query;
    }

    // Number of vectors in the index
    std::uint32_t nr_rows_{0};

    // Maximum dimension seen
    std::uint32_t max_dim_{0};

    // Maps external dimension numbers to internal dimension numbers
    std::unordered_map<uint32_t, uint32_t> dim_map_;
    uint32_t nr_inner_dims_{0};

    // The algorithm used to build the index
    InvertedIndexAlgo build_algo_;

    // The scorer used to build the index
    std::shared_ptr<IndexScorer> build_scorer_;

    // Meta data for the index, which could be used by the searcher
    InvertedIndexMetaData meta_data_;

    // Statistics for the build process, which will be used to generate the prometheus metrics
    InvertedIndexBuildStats build_stats_;

    std::int64_t index_id_{0};
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    prometheus::Gauge* index_size_gauge_{nullptr};
    prometheus::Histogram* index_dataset_nnz_len_histogram_{nullptr};
    prometheus::Histogram* index_posting_list_len_histogram_{nullptr};
#endif
};

template <typename IndexType, typename DType>
class CRTPInvertedIndex : public InvertedIndex<DType> {
 public:
    CRTPInvertedIndex(std::string index_type) {
        this->index_id_ =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        this->index_size_gauge_ = &sparse_inverted_index_size_family.Add(
            {{"index_id", std::to_string(this->index_id_)}, {"index_type", index_type}});
        this->index_dataset_nnz_len_histogram_ = &sparse_dataset_nnz_len_family.Add(
            {{"index_id", std::to_string(this->index_id_)}, {"index_type", index_type}}, defaultBuckets);
        this->index_posting_list_len_histogram_ = &sparse_inverted_index_posting_list_len_family.Add(
            {{"index_id", std::to_string(this->index_id_)}, {"index_type", index_type}}, defaultBuckets);
#endif
    }

    virtual ~CRTPInvertedIndex() {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        sparse_inverted_index_size_family.Remove(this->index_size_gauge_);
        sparse_dataset_nnz_len_family.Remove(this->index_dataset_nnz_len_histogram_);
        sparse_inverted_index_posting_list_len_family.Remove(this->index_posting_list_len_histogram_);
#endif
    }

    Status
    convert_to_raw_data(MemoryIOWriter& writer) const override;

    void
    search(const SparseRow<DType>& query, size_t k, float* distances, label_t* labels, const BitsetView& bitset,
           const InvertedIndexSearchParams& search_params) const override;

    std::vector<float>
    get_all_distances(const SparseRow<DType>& query, const BitsetView& bitset,
                      const InvertedIndexSearchParams& search_params) const override;
};

template <typename IndexType, typename DType>
Status
CRTPInvertedIndex<IndexType, DType>::convert_to_raw_data(MemoryIOWriter& writer) const {
    /**
     * Layout:
     *
     * 1. size_t rows
     * 2. size_t cols
     * 3. DType value_threshold_ (deprecated)
     * 4. for each row:
     *     1. size_t len
     *     2. for each non-zero value:
     *        1. table_t idx
     *        2. DType val (if quantized, the quantized value of val is stored as a DType with precision loss)
     *
     * Data are densely packed in serialized bytes and no padding is added.
     */
    float deprecated_value_threshold = 0;
    size_t rows = this->nr_rows_;
    size_t cols = this->max_dim_;
    writeBinaryPOD(writer, rows);
    writeBinaryPOD(writer, cols);
    writeBinaryPOD(writer, deprecated_value_threshold);

    auto dim_map_reverse = std::unordered_map<uint32_t, table_t>();
    for (const auto& [dim, dim_id] : this->dim_map_) {
        dim_map_reverse[dim_id] = dim;
    }

    std::vector<std::vector<std::pair<table_t, DType>>> raw_rows(this->nr_rows_);

    BitsetView bitset(nullptr, 0);
    for (size_t i = 0; i < this->dim_map_.size(); ++i) {
        auto plist_iter = static_cast<const IndexType*>(this)->get_dim_plist_cursor(i, bitset);
        while (plist_iter.valid()) {
            raw_rows[plist_iter.vec_id()].push_back(std::make_pair(dim_map_reverse[i], plist_iter.val()));
            plist_iter.next();
        }
    }

    for (size_t i = 0; i < this->nr_rows_; ++i) {
        writeBinaryPOD(writer, raw_rows[i].size());
        if (raw_rows[i].size() > 0) {
            writer.write(SparseRow<DType>(raw_rows[i]).data(), raw_rows[i].size() * SparseRow<DType>::element_size());
        }
    }

    return Status::success;
}

template <typename IndexType, typename DType>
std::vector<float>
CRTPInvertedIndex<IndexType, DType>::get_all_distances(const SparseRow<DType>& query, const BitsetView& bitset,
                                                       const InvertedIndexSearchParams& search_params) const {
    if (query.size() == 0) {
        return {};
    }

    std::vector<DType> values(query.size());
    for (size_t i = 0; i < query.size(); ++i) {
        values[i] = std::abs(query[i].val);
    }

    auto q_vec = this->parse_query(query, search_params.approx.drop_ratio_search);

    std::vector<float> distances(this->nr_rows_, 0.0f);

    std::shared_ptr<IndexScorer> search_scorer;
    if (search_params.scorer_config.scorer_type == IndexScorerType::BM25) {
        search_scorer = std::make_shared<BM25IndexScorer>(
            search_params.scorer_config.scorer_params.bm25.k1, search_params.scorer_config.scorer_params.bm25.b,
            search_params.scorer_config.scorer_params.bm25.avgdl, this->get_row_sums());
    } else {
        search_scorer = std::make_shared<IPIndexScorer>();
    }

    const auto* self = static_cast<const IndexType*>(this);
    for (const auto& [dim_id, dim_val] : q_vec) {
        auto index_cursor = self->get_dim_plist_cursor(dim_id, bitset);
        auto scorer = search_scorer->dim_scorer(dim_val);
        while (index_cursor.valid()) {
            distances[index_cursor.vec_id()] += scorer(index_cursor.vec_id(), index_cursor.val());
            index_cursor.next();
        }
    }

    return distances;
}

template <typename IndexType, typename DType>
void
CRTPInvertedIndex<IndexType, DType>::search(const SparseRow<DType>& query, size_t k, float* distances, label_t* labels,
                                            const BitsetView& bitset,
                                            const InvertedIndexSearchParams& search_params) const {
    std::shared_ptr<IndexScorer> search_scorer;
    if (search_params.scorer_config.scorer_type == IndexScorerType::BM25) {
        search_scorer = std::make_shared<BM25IndexScorer>(
            search_params.scorer_config.scorer_params.bm25.k1, search_params.scorer_config.scorer_params.bm25.b,
            search_params.scorer_config.scorer_params.bm25.avgdl, this->meta_data_.row_sums_);
    } else {
        search_scorer = std::make_shared<IPIndexScorer>();
    }

    std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
    std::fill(labels, labels + k, -1);

    if (query.size() == 0) {
        return;
    }

    auto q_vec = this->parse_query(query, search_params.approx.drop_ratio_search);
    if (q_vec.empty()) {
        return;
    }

    auto process_search_results = [&](auto& searcher) {
        auto topk = searcher.topk();
        size_t cnt = topk.size();
        for (size_t i = 0; i < cnt; ++i) {
            distances[i] = topk[i].first;
            labels[i] = topk[i].second;
        }
    };

    switch (search_params.algo) {
        case InvertedIndexAlgo::DAAT_WAND: {
            DaatWandSearcher<std::remove_reference_t<IndexType>> searcher(*static_cast<const IndexType*>(this), q_vec,
                                                                          search_scorer, k, this->nr_rows_, bitset,
                                                                          search_params.approx.dim_max_score_ratio);
            searcher.search();
            process_search_results(searcher);
            break;
        }
        case InvertedIndexAlgo::DAAT_MAXSCORE: {
            DaatMaxScoreSearcher<std::remove_reference_t<IndexType>> searcher(
                *static_cast<const IndexType*>(this), q_vec, search_scorer, k, this->nr_rows_, bitset,
                search_params.approx.dim_max_score_ratio);
            searcher.search();
            process_search_results(searcher);
            break;
        }
        case InvertedIndexAlgo::TAAT_NAIVE: {
            TaatNaiveSearcher<std::remove_reference_t<IndexType>> searcher(*static_cast<const IndexType*>(this), q_vec,
                                                                           search_scorer, k, this->nr_rows_, bitset);
            searcher.search();
            process_search_results(searcher);
            break;
        }
        case InvertedIndexAlgo::BLOCK_MAX_MAXSCORE: {
            BlockMaxMaxScoreSearcher<std::remove_reference_t<IndexType>> searcher(
                *static_cast<const IndexType*>(this), q_vec, search_scorer, k, this->nr_rows_, bitset,
                search_params.approx.dim_max_score_ratio);
            searcher.search();
            process_search_results(searcher);
            break;
        }
        case InvertedIndexAlgo::BLOCK_MAX_WAND: {
            BlockMaxWandSearcher<std::remove_reference_t<IndexType>> searcher(
                *static_cast<const IndexType*>(this), q_vec, search_scorer, k, this->nr_rows_, bitset,
                search_params.approx.dim_max_score_ratio);
            searcher.search();
            process_search_results(searcher);
            break;
        }
        default:
            LOG_KNOWHERE_ERROR_ << "Unsupported search algorithm";
    }
}

/**
 * @brief Convert a value to its quantized representation
 *
 * @param val The value to quantize
 * @return QType The quantized value
 *
 * If QType and DType are different, performs quantization by clamping to QType range.
 * If QType and DType are the same, returns val unchanged.
 */
template <typename DType, typename QType>
QType
get_quant_val(DType val) {
    if constexpr (!std::is_same_v<QType, DType>) {
        if (std::is_same_v<QType, fp16> && std::is_same_v<DType, float>) {
            return static_cast<QType>(val);
        }
        const DType max_val = static_cast<DType>(std::numeric_limits<QType>::max());
        if (val >= max_val) {
            return std::numeric_limits<QType>::max();
        } else if (val <= std::numeric_limits<QType>::min()) {
            return std::numeric_limits<QType>::min();
        } else {
            return static_cast<QType>(val);
        }
    } else {
        return val;
    }
}

}  // namespace knowhere::sparse::inverted
