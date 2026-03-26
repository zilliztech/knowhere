#pragma once

#include "index/sparse/inverted_index.h"
#include "index/sparse/scorer.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

template <typename DType, typename QType>
class GrowableInvertedIndexCursor {
 public:
    GrowableInvertedIndexCursor(const std::vector<uint32_t>& plist_ids, const std::vector<QType>& plist_vals,
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
    next_geq(uint32_t vec_id) {
        while (pos_ < plist_size_ && plist_ids_[pos_] < vec_id) {
            ++pos_;
        }
        skip_filtered_ids();
        update_cur_vec_id();
    }

    [[nodiscard]] uint32_t
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

    const std::vector<uint32_t>& plist_ids_;
    const std::vector<QType>& plist_vals_;
    const size_t plist_size_;
    const size_t universe_;
    BitsetView bitset_;

    size_t pos_{0};
    uint32_t cur_vec_id_{0};
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

    GrowableInvertedIndex() : CRTPInvertedIndex<GrowableInvertedIndex<DType, QType>, DType>("growableinverted") {
    }

    GrowableInvertedIndex(const GrowableInvertedIndex& rhs) = delete;
    GrowableInvertedIndex(GrowableInvertedIndex&& rhs) noexcept = default;
    GrowableInvertedIndex&
    operator=(const GrowableInvertedIndex& rhs) = delete;
    GrowableInvertedIndex&
    operator=(GrowableInvertedIndex&& rhs) noexcept = default;

    [[nodiscard]] size_t
    size() const override {
        size_t res = sizeof(*this);

        res += this->nr_inner_dims_ * (sizeof(typename decltype(this->dim_map_)::key_type) +
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

        if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
            res += this->meta_data_.block_max_data_.container_->size();
        }

        return res;
    }

    Status
    add(const SparseRow<DType>* data, size_t rows, int64_t dim) override;

    Status
    build_from_raw_data(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename) override {
        return Status::not_implemented;
    }

    Status
    serialize(MemoryIOWriter& writer) const override {
        return Status::not_implemented;
    }

    Status
    deserialize(MemoryIOReader& reader) override {
        return Status::not_implemented;
    }

    [[nodiscard]] posting_list_iterator
    get_dim_plist_cursor(uint32_t dim_id, const BitsetView& bitset) const {
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

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        this->meta_data_.row_sums_.resize(this->nr_rows_ + rows);
    }

    for (size_t i = 0; i < rows; ++i) {
        add_row_to_index(data[i], this->nr_rows_ + i);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        this->meta_data_.max_score_per_dim_.resize(this->nr_inner_dims_, 0.0f);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                auto [dim_id, val] = data[i][j];
                if (this->dim_map_.find(dim_id) == this->dim_map_.end()) {
                    continue;
                }
                float score = this->build_scorer_->vec_score(this->nr_rows_ + i, val);
                this->meta_data_.max_score_per_dim_[this->dim_map_[dim_id]] =
                    std::max(this->meta_data_.max_score_per_dim_[this->dim_map_[dim_id]], score);
            }
        }
    }

    this->nr_rows_ += rows;

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    this->index_size_gauge_->Set((double)size() / 1024.0 / 1024.0);
#endif

    return Status::success;
}

template <typename DType, typename QType>
inline void
GrowableInvertedIndex<DType, QType>::add_row_to_index(const SparseRow<DType>& row, std::uint32_t vec_id) {
    float row_sum = 0.0f;
    for (size_t i = 0; i < row.size(); ++i) {
        auto [dim, val] = row[i];
        row_sum += val;

        // Skip values equals to or close enough to zero (which is little to the total IP score).
        if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
            continue;
        }

        auto dim_it = this->dim_map_.find(dim);
        if (dim_it == this->dim_map_.cend()) {
            dim_it = this->dim_map_.insert({dim, this->nr_inner_dims_++}).first;
            posting_lists_ids_.emplace_back();
            posting_lists_vals_.emplace_back();
        }

        posting_lists_ids_[dim_it->second].emplace_back(vec_id);
        posting_lists_vals_[dim_it->second].emplace_back(get_quant_val<DType, QType>(val));
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    this->index_dataset_nnz_len_histogram_->Observe(row.size());
#endif

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        this->meta_data_.row_sums_[vec_id] = row_sum;
    }
}

}  // namespace knowhere::sparse::inverted
