#ifndef KNOWHERE_SPARSE_FLATTEN_INVERTED_INDEX_H
#define KNOWHERE_SPARSE_FLATTEN_INVERTED_INDEX_H

#include "gsl/span"
#include "index/sparse/inverted/pisa/index_scorer.h"
#include "inverted_index.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere::sparse {

template <typename DType, typename QType>
class FlattenInvertedIndexCursor {
 public:
    FlattenInvertedIndexCursor(gsl::span<uint32_t> plist_ids, gsl::span<QType> plist_vals, size_t universe,
                               BitsetView bitset)
        : plist_ids_(plist_ids),
          plist_vals_(plist_vals),
          plist_size_(plist_ids.size()),
          universe_(universe),
          bitset_(bitset) {
        reset();
    }

    FlattenInvertedIndexCursor(const FlattenInvertedIndexCursor& rhs) = delete;
    FlattenInvertedIndexCursor(FlattenInvertedIndexCursor&& rhs) noexcept = default;

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

    gsl::span<uint32_t> plist_ids_;
    gsl::span<QType> plist_vals_;
    const size_t plist_size_;
    const size_t universe_;
    const BitsetView bitset_;

    size_t pos_{0};
    table_t cur_vec_id_{0};
};

/**
 * @brief Fixed flatten inverted index for sparse vectors
 *
 * This index is immutable after construction - no new vectors can be added.
 * All data is stored as flat arrays with raw values.
 *
 * @tparam DType Type of the original vector values (e.g. float, uint32_t)
 * @tparam QType Type used for quantized values in the index (e.g. float, uint16_t)
 */
template <typename DType, typename QType>
class FlattenInvertedIndex : public CRTPInvertedIndex<FlattenInvertedIndex<DType, QType>, DType> {
 public:
    using posting_list_iterator = FlattenInvertedIndexCursor<DType, QType>;

    FlattenInvertedIndex() = default;
    ~FlattenInvertedIndex() = default;
    FlattenInvertedIndex(const FlattenInvertedIndex& rhs) = delete;
    FlattenInvertedIndex(FlattenInvertedIndex&& rhs) noexcept = default;
    FlattenInvertedIndex&
    operator=(const FlattenInvertedIndex& rhs) = delete;
    FlattenInvertedIndex&
    operator=(FlattenInvertedIndex&& rhs) noexcept = default;

    [[nodiscard]] size_t
    size() const override {
        size_t res = sizeof(*this);

        res += this->dim_map_.size() * (sizeof(typename decltype(this->dim_map_)::key_type) +
                                        sizeof(typename decltype(this->dim_map_)::mapped_type));

        if (raw_index_container_) {
            res += raw_index_container_->size();
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
    add(const SparseRow<DType>* data, size_t rows, int64_t dim) override {
        LOG_KNOWHERE_ERROR_ << "FlattenInvertedIndex does not support add";
        return Status::invalid_index_error;
    }

    Status
    build_from_raw_data(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename) override;

    [[nodiscard]] posting_list_iterator
    get_plist_cursor(uint32_t dim_id, const BitsetView& bitset) const {
        auto endpoint = this->raw_index_offsets_[dim_id];
        auto size = this->raw_index_offsets_[dim_id + 1] - endpoint;
        return posting_list_iterator(this->raw_index_ids_.subspan(endpoint, size),
                                     this->raw_index_vals_.subspan(endpoint, size), this->nr_rows_, bitset);
    }

 private:
    /**
     * @brief Add a single sparse vector to the index
     *
     * @param row The sparse vector to add
     * @param row_id ID to assign to this vector
     * @param curr_offsets Current offsets into each dimension's inverted list,
     *                     used to track where to insert the next value for each dimension.
     *                     The offsets are updated as values are added.
     *
     * This function takes a sparse vector and adds its non-zero elements to the inverted index.
     * For each non-zero element:
     * 1. Looks up the dimension mapping
     * 2. Adds the row_id to the inverted list for that dimension
     * 3. Adds the quantized value to the corresponding values list
     *
     * Zero values are skipped since they don't contribute to similarity scores.
     * Throws if a dimension is encountered that wasn't seen during index construction.
     */
    void
    add_row_to_index(const SparseRow<DType>& row, std::uint32_t row_id, std::vector<size_t>& curr_offsets);

    /**
     * @brief Build the raw index from the serialized data
     *
     * @param reader Reader containing serialized index data to analyze
     * @param enable_mmap Whether to use file backed memory mapping
     * @param backed_filename File to use for memory mapping if enabled
     *
     * This function builds the raw index from the serialized data.
     * It first reads the number of rows and the maximum dimension seen.
     * Then it calculates the memory requirements for the index structures.
     * If memory mapping is enabled, it creates a memory mapped file to store the data.
     * Otherwise, it uses heap memory.
     *
     * The memory is allocated for:
     * - Inverted lists storing vector IDs (inverted_index_ids_)
     * - Inverted lists storing quantized values (inverted_index_vals_)
     */
    void
    build_raw_index(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename);

    std::unique_ptr<BinaryContainer> raw_index_container_;

    // Inverted lists start offsets
    // Each dimension's inverted list is stored contiguously in a flattened array
    // The start offset of each dimension's list is stored in raw_index_offsets_
    gsl::span<size_t> raw_index_offsets_;

    // Inverted lists storing vector IDs
    gsl::span<uint32_t> raw_index_ids_;

    // Inverted lists storing corresponding values
    gsl::span<QType> raw_index_vals_;
};

template <typename DType, typename QType>
void
FlattenInvertedIndex<DType, QType>::build_raw_index(MemoryIOReader& reader, bool enable_mmap,
                                                    const std::string& backed_filename) {
    const auto saved_reader_loc = reader.tellg();
    const auto nnz = (reader.remaining() - (this->nr_rows_ * sizeof(size_t))) / SparseRow<DType>::element_size();

    std::unordered_map<uint32_t, size_t> plist_cnts;
    for (size_t i = 0; i < this->nr_rows_; ++i) {
        size_t count;
        readBinaryPOD(reader, count);
        if (count == 0) {
            continue;
        }
        for (size_t j = 0; j < count; ++j) {
            uint32_t dim;
            readBinaryPOD(reader, dim);
            if (this->dim_map_.find(dim) == this->dim_map_.end()) {
                this->dim_map_[dim] = this->dim_map_.size();
            }
            plist_cnts[this->dim_map_[dim]]++;
            reader.advance(sizeof(DType));
        }
    }

    // reset reader to the saved beginning
    reader.seekg(saved_reader_loc);

    // calculate raw index byte size
    auto raw_index_ids_byte_sz = nnz * sizeof(uint32_t);
    auto raw_index_vals_byte_sz = nnz * sizeof(QType);
    auto raw_index_offsets_byte_sz = (this->dim_map_.size() + 1) * sizeof(size_t);
    auto raw_index_byte_sz = raw_index_ids_byte_sz + raw_index_vals_byte_sz + raw_index_offsets_byte_sz;
    auto container_byte_sz = (raw_index_byte_sz % AlignedAllocator<uint8_t>::alignment == 0)
                                 ? raw_index_byte_sz
                                 : raw_index_byte_sz + AlignedAllocator<uint8_t>::alignment -
                                       (raw_index_byte_sz % AlignedAllocator<uint8_t>::alignment);

    if (enable_mmap) {
        raw_index_container_ = std::make_unique<FileBinaryContainer>(backed_filename + ".raw_index");
    } else {
        raw_index_container_ = std::make_unique<MemBinaryContainer>();
    }

    raw_index_container_->resize(container_byte_sz);
    raw_index_container_->seal();

    auto data = raw_index_container_->data();
    raw_index_ids_ = gsl::span<uint32_t>(reinterpret_cast<uint32_t*>(data), nnz);
    raw_index_vals_ = gsl::span<QType>(reinterpret_cast<QType*>(data + raw_index_ids_byte_sz), nnz);
    raw_index_offsets_ = gsl::span<size_t>(
        reinterpret_cast<size_t*>(data + raw_index_ids_byte_sz + raw_index_vals_byte_sz), this->dim_map_.size() + 1);

    std::size_t offset = 0;
    for (size_t i = 0; i < this->dim_map_.size(); ++i) {
        raw_index_offsets_[i] = offset;
        offset += plist_cnts[i];
    }
    raw_index_offsets_[this->dim_map_.size()] = offset;

    std::vector<size_t> curr_offsets(this->dim_map_.size());
    for (size_t i = 0; i < this->dim_map_.size(); ++i) {
        curr_offsets[i] = raw_index_offsets_[i];
    }

    for (size_t i = 0; i < this->nr_rows_; ++i) {
        size_t count;
        readBinaryPOD(reader, count);
        SparseRow<DType> raw_row = SparseRow<DType>(count);
        if (count > 0) {
            reader.read(raw_row.data(), count * SparseRow<DType>::element_size());
        }
        add_row_to_index(raw_row, i, curr_offsets);
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
            auto offset = this->raw_index_offsets_[i];
            size_t count = this->raw_index_offsets_[i + 1] - offset;
            auto ids = this->raw_index_ids_.subspan(offset, count);
            auto vals = this->raw_index_vals_.subspan(offset, count);
            for (size_t j = 0; j < count; ++j) {
                auto score = scorer->vec_score(ids[j], vals[j]);
                this->meta_data_.max_score_per_dim_[i] = std::max(this->meta_data_.max_score_per_dim_[i], score);
            }
        }
    }
}

template <typename DType, typename QType>
Status
FlattenInvertedIndex<DType, QType>::build_from_raw_data(MemoryIOReader& reader, bool enable_mmap,
                                                        const std::string& backed_filename) {
    float deprecated_value_threshold = 0.0f;
    int64_t rows = 0;
    size_t cols = 0;

    // previous versions used the signness of rows to indicate whether to
    // use wand. now we use a template parameter to control this thus simply
    // take the absolute value of rows.
    readBinaryPOD(reader, rows);
    this->nr_rows_ = std::abs(rows);
    readBinaryPOD(reader, cols);
    this->max_dim_ = cols;
    readBinaryPOD(reader, deprecated_value_threshold);

    // build raw index to raw_index_ids_, raw_index_vals_, raw_index_offsets_ and dim_map_
    build_raw_index(reader, enable_mmap, backed_filename);

    return Status::success;
}

template <typename DType, typename QType>
inline void
FlattenInvertedIndex<DType, QType>::add_row_to_index(const SparseRow<DType>& row, std::uint32_t vec_id,
                                                     std::vector<size_t>& curr_offsets) {
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
            throw std::runtime_error("unexpected vector dimension in FlattenInvertedIndex");
        }

        auto offset = curr_offsets[dim_it->second]++;
        raw_index_ids_[offset] = vec_id;
        raw_index_vals_[offset] = get_quant_val<DType, QType>(val);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        this->meta_data_.row_sums_.push_back(row_sum);
    }
}

}  // namespace knowhere::sparse

#endif  // KNOWHERE_SPARSE_FLATTEN_INVERTED_INDEX_H
