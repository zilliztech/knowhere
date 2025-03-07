#ifndef KNOWHERE_SPARSE_GROWABLE_INVERTED_INDEX_H
#define KNOWHERE_SPARSE_GROWABLE_INVERTED_INDEX_H

#include "index/sparse/inverted/pisa/index_scorer.h"
#include "inverted_index.h"
#include "knowhere/utils.h"

namespace knowhere::sparse {

template <typename DType, typename QType>
class GrowableInvertedIndexCursor {
 public:
    GrowableInvertedIndexCursor(const std::vector<table_t>& plist_ids, const std::vector<QType>& plist_vals,
                                size_t universe, BitsetView bitset)
        : plist_ids_(plist_ids),
          plist_vals_(plist_vals),
          plist_size_(plist_ids.size()),
          universe_(universe),
          bitset_(bitset) {
        reset();
    }
    GrowableInvertedIndexCursor(const GrowableInvertedIndexCursor& rhs) = delete;
    GrowableInvertedIndexCursor(GrowableInvertedIndexCursor&& rhs) noexcept = default;

    void
    reset() {
        pos_ = 0;
        skip_filtered_ids();
        update_cur_vec_id();
    }

    void
    next() {
        ++pos_;
        skip_filtered_ids();
        update_cur_vec_id();
    }

    void
    next_geq(table_t vec_id) {
        while (pos_ < plist_size_ && plist_ids_[pos_] < vec_id) {
            ++pos_;
        }
        skip_filtered_ids();
        update_cur_vec_id();
    }

    [[nodiscard]] table_t
    vec_id() const {
        return cur_vec_id_;
    }

    [[nodiscard]] QType
    val() const {
        return plist_vals_[pos_];
    }

    [[nodiscard]] bool
    valid() const {
        return cur_vec_id_ != universe_;
    }

 private:
    void
    skip_filtered_ids() {
        while (pos_ < plist_size_ && !bitset_.empty() && bitset_.test(plist_ids_[pos_])) {
            ++pos_;
        }
    }

    void
    update_cur_vec_id() {
        cur_vec_id_ = (pos_ >= plist_size_) ? universe_ : plist_ids_[pos_];
    }

    const std::vector<table_t>& plist_ids_;
    const std::vector<QType>& plist_vals_;
    const size_t plist_size_;
    const size_t universe_;
    const BitsetView bitset_;

    size_t pos_{0};
    table_t cur_vec_id_{0};
};

/**
 * @brief Dynamic in-memory inverted index for sparse vectors that supports incremental updates
 *
 * This index allows dynamically adding new vectors after construction. All data is stored in memory.
 *
 * @tparam DType Type of the original vector values (e.g. float)
 * @tparam QType Type used for quantized values in the index (e.g. float)
 */
template <typename DType, typename QType>
class GrowableInvertedIndex : public CRTPInvertedIndex<GrowableInvertedIndex<DType, QType>, DType> {
 public:
    using posting_list_iterator = GrowableInvertedIndexCursor<DType, QType>;

    GrowableInvertedIndex() = default;
    ~GrowableInvertedIndex() = default;
    GrowableInvertedIndex(const GrowableInvertedIndex& rhs) = delete;
    GrowableInvertedIndex(GrowableInvertedIndex&& rhs) noexcept = default;
    GrowableInvertedIndex&
    operator=(const GrowableInvertedIndex& rhs) = delete;
    GrowableInvertedIndex&
    operator=(GrowableInvertedIndex&& rhs) noexcept = default;

    [[nodiscard]] size_t
    size() const override {
        size_t res = sizeof(*this);

        res += this->dim_map_.size() * (sizeof(typename decltype(this->dim_map_)::key_type) +
                                        sizeof(typename decltype(this->dim_map_)::mapped_type));

        res += sizeof(typename decltype(posting_lists_ids_)::value_type) * posting_lists_ids_.size();
        for (const auto& ids : posting_lists_ids_) {
            res += ids.size() * sizeof(uint32_t);
        }
        res += sizeof(typename decltype(posting_lists_vals_)::value_type) * posting_lists_vals_.size();
        for (const auto& vals : posting_lists_vals_) {
            res += vals.size() * sizeof(QType);
        }

        if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
            res += this->meta_data_.row_sums_.size() * sizeof(float);
        }

        if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
            res += this->meta_data_.max_score_per_dim_.size() * sizeof(float);
        }

        return res;
    }

    Status
    add(const SparseRow<DType>* data, size_t rows, int64_t dim) override;

    Status
    build_from_raw_data(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename) override;

    [[nodiscard]] posting_list_iterator
    get_plist_cursor(uint32_t dim_id, const BitsetView& bitset) const {
        return posting_list_iterator(posting_lists_ids_[dim_id], posting_lists_vals_[dim_id], this->nr_rows_, bitset);
    }

 private:
    /**
     * @brief Add a single sparse vector to the index
     *
     * @param row The sparse vector to add
     * @param row_id ID to assign to this vector
     */
    void
    add_row_to_index(const SparseRow<DType>& row, std::uint32_t row_id);

    // Inverted posting lists storing vector IDs
    std::vector<std::vector<std::uint32_t>> posting_lists_ids_;

    // Inverted posting lists storing corresponding values
    std::vector<std::vector<QType>> posting_lists_vals_;
};

template <typename DType, typename QType>
Status
GrowableInvertedIndex<DType, QType>::add(const SparseRow<DType>* data, size_t rows, int64_t dim) {
    if (dim > this->max_dim_) {
        this->max_dim_ = dim;
    }

    for (size_t i = 0; i < rows; ++i) {
        add_row_to_index(data[i], this->nr_rows_ + i);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        for (size_t i = 0; i < rows; ++i) {
            float row_sum = 0.0f;
            for (size_t j = 0; j < data[i].size(); ++j) {
                auto [dim, val] = data[i][j];
                row_sum += val;
            }
            this->meta_data_.row_sums_.emplace_back(row_sum);
        }
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        std::unique_ptr<pisa::IndexScorer> scorer;
        if (this->metric_type_ == SparseMetricType::METRIC_BM25) {
            scorer =
                std::make_unique<pisa::BM25IndexScorer>(this->metric_params_.bm25.k1, this->metric_params_.bm25.b,
                                                        this->metric_params_.bm25.avgdl, this->meta_data_.row_sums_);
        } else {
            scorer = std::make_unique<pisa::IPIndexScorer>();
        }

        this->meta_data_.max_score_per_dim_.resize(this->dim_map_.size(), 0.0f);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                auto [dim, val] = data[i][j];
                if (this->dim_map_.find(dim) == this->dim_map_.end()) {
                    continue;
                }
                float score = scorer->vec_score(this->nr_rows_ + i, val);
                this->meta_data_.max_score_per_dim_[this->dim_map_[dim]] =
                    std::max(this->meta_data_.max_score_per_dim_[this->dim_map_[dim]], score);
            }
        }
    }

    this->nr_rows_ += rows;

    return Status::success;
}

template <typename DType, typename QType>
Status
GrowableInvertedIndex<DType, QType>::build_from_raw_data(MemoryIOReader& reader, bool enable_mmap,
                                                         const std::string& backed_filename) {
    float deprecated_value_threshold = 0.0f;
    int64_t rows = 0;
    size_t cols = 0;

    // previous versions used the signness of rows to indicate whether to
    // use wand. now we use a template parameter to control this thus simply
    // take the absolute value of rows.
    readBinaryPOD(reader, rows);
    this->nr_rows_ = std::abs(rows);
    // dim should not be exceed uint32_t
    readBinaryPOD(reader, cols);
    this->max_dim_ = cols;
    readBinaryPOD(reader, deprecated_value_threshold);

    for (uint32_t i = 0; i < this->nr_rows_; ++i) {
        size_t count;
        readBinaryPOD(reader, count);
        SparseRow<DType> raw_row = SparseRow<DType>(count);
        if (count > 0) {
            reader.read(raw_row.data(), count * SparseRow<DType>::element_size());
        }
        add_row_to_index(raw_row, i);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        std::unique_ptr<pisa::IndexScorer> scorer;
        if (this->metric_type_ == SparseMetricType::METRIC_BM25) {
            scorer =
                std::make_unique<pisa::BM25IndexScorer>(this->metric_params_.bm25.k1, this->metric_params_.bm25.b,
                                                        this->metric_params_.bm25.avgdl, this->meta_data_.row_sums_);
        } else {
            scorer = std::make_unique<pisa::IPIndexScorer>();
        }

        this->meta_data_.max_score_per_dim_.resize(this->dim_map_.size(), 0.0f);

        for (size_t i = 0; i < this->dim_map_.size(); ++i) {
            auto ids = posting_lists_ids_[i];
            auto vals = posting_lists_vals_[i];
            for (size_t j = 0; j < ids.size(); ++j) {
                auto score = scorer->vec_score(ids[j], vals[j]);
                this->meta_data_.max_score_per_dim_[i] = std::max(this->meta_data_.max_score_per_dim_[i], score);
            }
        }
    }

    return Status::success;
}

template <typename DType, typename QType>
inline void
GrowableInvertedIndex<DType, QType>::add_row_to_index(const SparseRow<DType>& row, std::uint32_t vec_id) {
    float row_sum = 0.0f;
    for (size_t j = 0; j < row.size(); ++j) {
        auto [dim, val] = row[j];
        row_sum += val;

        // Skip values equals to or close enough to zero (which is little to the total IP score).
        if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
            continue;
        }

        auto dim_it = this->dim_map_.find(dim);
        if (dim_it == this->dim_map_.cend()) {
            dim_it = this->dim_map_.insert({dim, this->dim_map_.size()}).first;
            posting_lists_ids_.emplace_back();
            posting_lists_vals_.emplace_back();
        }

        posting_lists_ids_[dim_it->second].emplace_back(vec_id);
        posting_lists_vals_[dim_it->second].emplace_back(get_quant_val<DType, QType>(val));
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        this->meta_data_.row_sums_.push_back(row_sum);
    }
}

}  // namespace knowhere::sparse

#endif  // KNOWHERE_SPARSE_GROWABLE_INVERTED_INDEX_H
