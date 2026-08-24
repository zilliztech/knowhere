#pragma once

#include <algorithm>

#include "index/sparse/block_max_data.h"
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
        const auto begin = plist_ids_.begin() + std::min(pos_, plist_size_);
        pos_ = static_cast<size_t>(std::lower_bound(begin, plist_ids_.end(), vec_id) - plist_ids_.begin());
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
class GrowableInvertedIndex : public CRTPInvertedIndex<GrowableInvertedIndex<DType, QType>, DType, true> {
 public:
    using posting_list_iterator = GrowableInvertedIndexCursor<DType, QType>;

    GrowableInvertedIndex() : CRTPInvertedIndex<GrowableInvertedIndex<DType, QType>, DType, true>("growableinverted") {
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

        res += this->dim_map_.byte_size();

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
            // Growable index keeps block-max data in per-dim vectors instead of the
            // contiguous container used by the sealed indexes, so size() must account
            // for those vectors directly rather than dereferencing block_max_data_.container_
            // (which is never allocated here).
            res += sizeof(typename decltype(block_max_ids_per_dim_)::value_type) * block_max_ids_per_dim_.size();
            for (const auto& ids : block_max_ids_per_dim_) {
                res += ids.size() * sizeof(uint32_t);
            }
            res += sizeof(typename decltype(block_max_scores_per_dim_)::value_type) * block_max_scores_per_dim_.size();
            for (const auto& scores : block_max_scores_per_dim_) {
                res += scores.size() * sizeof(float);
            }
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

    /**
     * @brief Get a block-max data cursor for a dimension.
     */
    [[nodiscard]] BlockMaxDataCursor
    get_block_max_data_cursor(uint32_t dim_id) const {
        const auto& ids = block_max_ids_per_dim_[dim_id];
        const auto& scores = block_max_scores_per_dim_[dim_id];
        return {std::span<uint32_t>(const_cast<uint32_t*>(ids.data()), ids.size()),
                std::span<float>(const_cast<float*>(scores.data()), scores.size())};
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

    /**
     * @brief Incrementally extend the block-max data for a single dimension with one newly
     * appended posting list entry.
     *
     * Blocks are defined purely by position within the (append-only, docid-sorted) posting
     * list, exactly like the batch index builders. The last entry of block_max_ids_per_dim_/
     * block_max_scores_per_dim_ always represents the current, possibly still-open, last
     * block and is updated in place until it fills up; a new entry is only appended once a
     * new block starts.
     *
     * @param dim_id Inner dimension id whose posting list was just extended
     * @param vec_id Vector id of the newly appended posting list entry
     * @param score Score contribution of the newly appended entry
     * @param block_size Number of posting list entries per block
     */
    void
    update_block_max(uint32_t dim_id, uint32_t vec_id, float score, size_t block_size) {
        auto& ids = block_max_ids_per_dim_[dim_id];
        auto& scores = block_max_scores_per_dim_[dim_id];
        const size_t pos = posting_lists_ids_[dim_id].size() - 1;
        const size_t block_idx = pos / block_size;
        if (block_idx == ids.size()) {
            ids.emplace_back(vec_id);
            scores.emplace_back(score);
        } else {
            ids.back() = vec_id;
            scores.back() = std::max(scores.back(), score);
        }
    }

    // Inverted posting lists storing vector IDs
    std::vector<std::vector<std::uint32_t>> posting_lists_ids_;

    // Inverted posting lists storing corresponding values
    std::vector<std::vector<QType>> posting_lists_vals_;

    // Growable counterpart of InvertedIndexMetaData::block_max_data_: for each inner dim,
    // the max score of every closed block plus the still-open trailing block, keyed by the
    // last vector id observed in that block. Only populated when
    // InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES is set.
    std::vector<std::vector<std::uint32_t>> block_max_ids_per_dim_;
    std::vector<std::vector<float>> block_max_scores_per_dim_;
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
        this->meta_data_.resize_max_score_per_dim(this->nr_inner_dims_, 0.0f);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                auto [dim_id, val] = data[i][j];
                auto inner_dim = this->dim_map_.lookup(dim_id);
                if (!inner_dim.has_value()) {
                    continue;
                }
                float score = this->build_scorer_->vec_score(this->nr_rows_ + i, val);
                this->meta_data_.max_score_per_dim_[inner_dim.value()] =
                    std::max(this->meta_data_.max_score_per_dim_[inner_dim.value()], score);
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
    const bool build_block_max = this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES;

    // BM25 scoring (used below for block-max data) needs the row's total sum finalized before
    // scoring any of its entries, so compute and publish it up front instead of accumulating it
    // while iterating (which is what the non-block-max code path used to do, since it never
    // needed a fully-finalized row_sum until after this function returned).
    float row_sum = 0.0f;
    for (size_t i = 0; i < row.size(); ++i) {
        row_sum += row[i].val;
    }
    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        this->meta_data_.row_sums_[vec_id] = row_sum;
    }

    const size_t block_size = build_block_max ? std::max<size_t>(this->meta_data_.block_max_data_.block_size_, 1) : 0;

    for (size_t i = 0; i < row.size(); ++i) {
        auto [dim, val] = row[i];

        // Skip values equals to or close enough to zero (which is little to the total IP score).
        if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
            continue;
        }

        auto inner_dim = this->dim_map_.lookup(dim);
        if (!inner_dim.has_value()) {
            inner_dim = this->dim_map_.append_legacy_entry(dim);
            this->nr_inner_dims_ = this->dim_map_.size();
            posting_lists_ids_.emplace_back();
            posting_lists_vals_.emplace_back();
            if (build_block_max) {
                block_max_ids_per_dim_.emplace_back();
                block_max_scores_per_dim_.emplace_back();
            }
        }

        const auto dim_id = inner_dim.value();
        posting_lists_ids_[dim_id].emplace_back(vec_id);
        posting_lists_vals_[dim_id].emplace_back(get_quant_val<DType, QType>(val));

        if (build_block_max) {
            // Must be called immediately after the posting list append above so that
            // posting_lists_ids_[dim_id].size() - 1 (read inside update_block_max) reflects
            // the position of exactly this entry.
            update_block_max(dim_id, vec_id, this->build_scorer_->vec_score(vec_id, val), block_size);
        }
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    this->index_dataset_nnz_len_histogram_->Observe(row.size());
#endif
}

}  // namespace knowhere::sparse::inverted
