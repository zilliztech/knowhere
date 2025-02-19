#ifndef KNOWHERE_SPARSE_BASE_INVERTED_INDEX_H
#define KNOWHERE_SPARSE_BASE_INVERTED_INDEX_H

#include <fcntl.h>
#include <sys/mman.h>

#include <filesystem>
#include <fstream>

#include "index/sparse/inverted/pisa/index_scorer.h"
#include "index/sparse/inverted/pisa/searcher/daat_maxscore.h"
#include "index/sparse/inverted/pisa/searcher/daat_wand.h"
#include "index/sparse/inverted/pisa/searcher/taat_naive.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/operands.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere::sparse {

enum class InvertedIndexAlgo { TAAT_NAIVE, DAAT_MAXSCORE, DAAT_WAND };

/**
 * @brief Metadata for the inverted index
 *
 * This struct contains metadata used in the build phase of the inverted index.
 * Currently it has two components:
 * 1. Sum of values in each row, used by BM25
 * 2. Maximum score for each dimension, used by DAAT_MAXSCORE and DAAT_WAND
 *
 * The flags indicate which components are present.
 */

struct InvertedIndexMetaData {
    // Flags indicating which metadata components are present
    static constexpr uint32_t FLAG_NONE = 0;
    static constexpr uint32_t FLAG_HAS_ROW_SUMS = 1 << 0;            // Row sums are present
    static constexpr uint32_t FLAG_HAS_MAX_SCORES_PER_DIM = 1 << 1;  // Maximum scores per dimension are present
    using MetaDataFlags = uint32_t;
    MetaDataFlags flags_{FLAG_NONE};

    InvertedIndexAlgo build_algo_;
    std::vector<float> row_sums_;
    std::vector<float> max_score_per_dim_;
};

struct InvertedIndexSearchParams {
    InvertedIndexAlgo algo;
    SparseMetricType metric_type;
    SparseMetricParams metric_params;
    struct {
        float drop_ratio_search;
        float dim_max_score_ratio;
    } approx;
};

template <typename DType>
class InvertedIndex {
 public:
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
     * @brief Build index from raw data format
     *
     * @param reader Reader containing serialized index data
     * @param enable_mmap Whether to use file backed memory mapping
     * @param backed_filename File to use for memory mapping if enabled
     * @return Status success if deserialization succeeds, error code otherwise
     */
    virtual Status
    build_from_raw_data(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename) = 0;

    /**
     * @brief Convert index to raw data format
     *
     * Convert the index data in the following layout:
     * 1. Number of rows (size_t)
     * 2. Number of columns (size_t)
     * 3. Deprecated value threshold (DType)
     * 4. For each row:
     *    - Row length (size_t)
     *    - Row data (array of dimension-value pairs)
     */
    virtual Status
    convert_to_raw_data(MemoryIOWriter& writer) const = 0;

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
     * @brief Check if the search parameters are valid
     *
     * @param search_params Search parameters
     * @return Status success if valid, error code otherwise
     */
    Status
    valid_search_check(const InvertedIndexSearchParams& search_params) {
        // check search algo
        if (search_params.algo != InvertedIndexAlgo::DAAT_WAND &&
            search_params.algo != InvertedIndexAlgo::DAAT_MAXSCORE &&
            search_params.algo != InvertedIndexAlgo::TAAT_NAIVE) {
            return Status::invalid_value_in_json;
        }

        // check search metric type
        if (search_params.metric_type != metric_type_) {
            LOG_KNOWHERE_ERROR_ << "Search metric type must be same as built index";
            return Status::invalid_metric_type;
        }

        if (!(meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) &&
            (search_params.algo == InvertedIndexAlgo::DAAT_MAXSCORE ||
             search_params.algo == InvertedIndexAlgo::DAAT_WAND)) {
            LOG_KNOWHERE_ERROR_ << "search algorithm DAAT_MAXSCORE and DAAT_WAND requires max_score_per_dim";
            return Status::invalid_value_in_json;
        }

        // check search metric params
        if (search_params.metric_type == SparseMetricType::METRIC_BM25 &&
            (search_params.algo == InvertedIndexAlgo::DAAT_WAND ||
             search_params.algo == InvertedIndexAlgo::DAAT_MAXSCORE)) {
            if (search_params.metric_params.bm25.k1 != metric_params_.bm25.k1 ||
                search_params.metric_params.bm25.b != metric_params_.bm25.b) {
                LOG_KNOWHERE_ERROR_ << "BM25 parameters k1 and b in search config must be same as built index";
                return Status::invalid_value_in_json;
            }
        }

        return Status::success;
    }

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
     * @brief Set the metric type and parameters
     *
     * @param metric_type Metric type
     * @param metric_params Metric parameters
     */
    void
    set_metric(const SparseMetricType& metric_type, const SparseMetricParams& metric_params) {
        metric_type_ = metric_type;
        metric_params_ = metric_params;
    }

    void
    set_build_algo(const std::string& build_algo) {
        if (build_algo == "DAAT_MAXSCORE") {
            meta_data_.build_algo_ = InvertedIndexAlgo::DAAT_MAXSCORE;
        } else if (build_algo == "DAAT_WAND") {
            meta_data_.build_algo_ = InvertedIndexAlgo::DAAT_WAND;
        } else {
            meta_data_.build_algo_ = InvertedIndexAlgo::TAAT_NAIVE;
        }
    }

    /**
     * @brief Set the flags
     *
     * @param flags Flags
     */
    void
    set_metadata_flags(sparse::InvertedIndexMetaData::MetaDataFlags flags) {
        meta_data_.flags_ = flags;
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
     * @brief Get the metric type
     *
     * @return SparseMetricType The metric type
     */
    [[nodiscard]] SparseMetricType
    get_metric_type() const {
        return metric_type_;
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

    [[nodiscard]] InvertedIndexAlgo
    build_algo() const {
        return meta_data_.build_algo_;
    }

    // Meta data for the index, which could be used by the searcher
    InvertedIndexMetaData meta_data_;

 protected:
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

    // Number of documents in the index
    std::uint32_t nr_rows_{0};

    // Maximum dimension seen
    std::uint32_t max_dim_{0};

    // Maps external dimension numbers to internal dimension numbers
    std::unordered_map<uint32_t, uint32_t> dim_map_;

    // Metric type
    SparseMetricType metric_type_{SparseMetricType::METRIC_IP};

    // Metric params
    SparseMetricParams metric_params_;
};

template <typename IndexType, typename DType>
class CRTPInvertedIndex : public InvertedIndex<DType> {
 public:
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
        auto plist_iter = static_cast<const IndexType*>(this)->get_plist_cursor(i, bitset);
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

    std::shared_ptr<pisa::IndexScorer> index_scorer;
    if (search_params.metric_type == SparseMetricType::METRIC_BM25) {
        index_scorer = std::make_shared<pisa::BM25IndexScorer>(
            search_params.metric_params.bm25.k1, search_params.metric_params.bm25.b,
            search_params.metric_params.bm25.avgdl, this->meta_data_.row_sums_);
    } else {
        index_scorer = std::make_shared<pisa::IPIndexScorer>();
    }

    auto cursors = pisa::make_scored_cursors(*static_cast<const IndexType*>(this), q_vec, index_scorer, bitset);
    for (auto& cursor : cursors) {
        while (cursor.valid()) {
            distances[cursor.vec_id()] += cursor.score();
            cursor.next();
        }
    }

    return distances;
}

template <typename IndexType, typename DType>
void
CRTPInvertedIndex<IndexType, DType>::search(const SparseRow<DType>& query, size_t k, float* distances, label_t* labels,
                                            const BitsetView& bitset,
                                            const InvertedIndexSearchParams& search_params) const {
    std::shared_ptr<pisa::IndexScorer> index_scorer;
    if (search_params.metric_type == SparseMetricType::METRIC_BM25) {
        index_scorer = std::make_shared<pisa::BM25IndexScorer>(
            search_params.metric_params.bm25.k1, search_params.metric_params.bm25.b,
            search_params.metric_params.bm25.avgdl, this->meta_data_.row_sums_);
    } else {
        index_scorer = std::make_shared<pisa::IPIndexScorer>();
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
            pisa::DaatWandSearcher<std::remove_reference_t<IndexType>> searcher(
                *static_cast<const IndexType*>(this), q_vec, index_scorer, k, this->nr_rows_, bitset);
            searcher.search();
            process_search_results(searcher);
            break;
        }
        case InvertedIndexAlgo::DAAT_MAXSCORE: {
            pisa::DaatMaxScoreSearcher<std::remove_reference_t<IndexType>> searcher(
                *static_cast<const IndexType*>(this), q_vec, index_scorer, k, this->nr_rows_, bitset);
            searcher.search();
            process_search_results(searcher);
            break;
        }
        case InvertedIndexAlgo::TAAT_NAIVE: {
            pisa::TaatNaiveSearcher<std::remove_reference_t<IndexType>> searcher(
                *static_cast<const IndexType*>(this), q_vec, index_scorer, k, this->nr_rows_, bitset);
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

class BinaryContainer {
 public:
    virtual ~BinaryContainer() = default;
    virtual void
    append(const uint8_t* values, size_t count) = 0;
    virtual void
    write_at(size_t index, const uint8_t* values, size_t count) = 0;
    virtual void
    resize(size_t new_size) = 0;
    virtual uint8_t*
    data() = 0;
    virtual void
    seal() = 0;
    [[nodiscard]] virtual size_t
    size() const = 0;
};

class FileBinaryContainer : public BinaryContainer {
 public:
    explicit FileBinaryContainer(const std::filesystem::path& filepath) : filepath_(filepath) {
        file_.open(filepath, std::ios::binary | std::ios::in | std::ios::out);

        if (!file_) {
            std::ofstream create_file(filepath, std::ios::binary);
            create_file.close();
            file_.open(filepath, std::ios::binary | std::ios::in | std::ios::out);
            if (!file_) {
                throw std::runtime_error("Failed to open file: " + filepath.string());
            }
        }

        file_.seekg(0, std::ios::end);
        size_ = file_.tellg() / sizeof(uint8_t);
        file_.seekg(0);
    }

    void
    append(const uint8_t* values, size_t count) override {
        file_.seekp(size_ * sizeof(uint8_t));
        file_.write(reinterpret_cast<const char*>(values), count * sizeof(uint8_t));
        size_ += count;
    }

    void
    write_at(size_t index, const uint8_t* values, size_t count) override {
        if (index + count > size_) {
            resize(index + count);
        }
        file_.seekp(index * sizeof(uint8_t));
        file_.write(reinterpret_cast<const char*>(values), count * sizeof(uint8_t));
    }

    void
    resize(size_t new_size) override {
        file_.close();
        std::filesystem::resize_file(filepath_, new_size * sizeof(uint8_t));
        file_.open(filepath_, std::ios::binary | std::ios::in | std::ios::out);
        size_ = new_size;
    }

    void
    seal() override {
        if (!sealed_) {
            file_.flush();
            file_.close();
            sealed_ = true;
        }
    }

    [[nodiscard]] size_t
    size() const override {
        return size_;
    }

    // Only call this function after sealing the container
    [[nodiscard]] uint8_t*
    data() override {
        if (!sealed_) {
            throw std::runtime_error("FileBinaryContainer is not sealed");
        }

        fd_ = ::open(filepath_.c_str(), O_RDWR);
        if (fd_ == -1) {
            throw std::runtime_error("Failed to open file: " + filepath_.string());
        }

        // Map the file
        data_ = ::mmap(nullptr, size_ * sizeof(uint8_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            throw std::runtime_error("Failed to mmap file");
        }

        return static_cast<uint8_t*>(data_);
    }

    ~FileBinaryContainer() override {
        if (!sealed_) {
            if (file_.is_open()) {
                file_.close();
            }
        } else {
            if (fd_ != -1) {
                ::munmap(data_, size_ * sizeof(uint8_t));
                ::close(fd_);
                data_ = nullptr;
            }
        }
        std::filesystem::remove(filepath_);
    }

 private:
    std::filesystem::path filepath_;
    mutable std::fstream file_;
    bool sealed_{false};
    int fd_{-1};
    void* data_{nullptr};
    size_t size_{};
};

template <typename T>
class AlignedAllocator {
 public:
    using value_type = T;
    static constexpr size_t alignment = 128;

    T*
    allocate(size_t n) {
        if (auto ptr = std::aligned_alloc(alignment, n * sizeof(T))) {
            return static_cast<T*>(ptr);
        }
        throw std::bad_alloc();
    }

    void
    deallocate(T* p, size_t) {
        std::free(p);
    }
};

class MemBinaryContainer : public BinaryContainer {
 public:
    explicit MemBinaryContainer() = default;

    void
    append(const uint8_t* values, size_t count) override {
        data_.insert(data_.end(), values, values + count);
    }

    void
    write_at(size_t index, const uint8_t* values, size_t count) override {
        if (index + count > data_.size()) {
            data_.resize(index + count);
        }
        std::copy(values, values + count, data_.begin() + index);
    }

    void
    resize(size_t new_size) override {
        data_.resize(new_size);
    }

    [[nodiscard]] size_t
    size() const override {
        return data_.size();
    }

    void
    seal() override {
        // nothing to do
    }

    [[nodiscard]] uint8_t*
    data() override {
        return data_.data();
    }

 private:
    std::vector<uint8_t, AlignedAllocator<uint8_t>> data_;
};

}  // namespace knowhere::sparse
#endif  // KNOWHERE_SPARSE_BASE_INVERTED_INDEX_H
