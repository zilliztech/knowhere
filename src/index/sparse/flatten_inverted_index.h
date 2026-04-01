#pragma once
#include <boost/core/make_span.hpp>
#include <boost/core/span.hpp>
#include <iostream>
#include <vector>

#include "index/sparse/inverted_index.h"
#include "index/sparse/scorer.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere::sparse::inverted {

template <typename DType, typename QType>
class FlattenInvertedIndexCursor {
 public:
    FlattenInvertedIndexCursor(boost::span<uint32_t> plist_ids, boost::span<QType> plist_vals, size_t universe,
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
        if (pos_ >= plist_size_) {
            cur_vec_id_ = universe_;
        } else {
            cur_vec_id_ = plist_ids_[pos_];
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(plist_vals_.data() + pos_, 0, 3);
#endif
        }
    }

    boost::span<uint32_t> plist_ids_;
    boost::span<QType> plist_vals_;
    const size_t plist_size_;
    const size_t universe_;
    BitsetView bitset_;

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

    static constexpr uint64_t current_index_file_format_version_ = 1;

    FlattenInvertedIndex() : CRTPInvertedIndex<FlattenInvertedIndex<DType, QType>, DType>("flatteninverted") {
    }

    FlattenInvertedIndex(const FlattenInvertedIndex& rhs) = delete;
    FlattenInvertedIndex(FlattenInvertedIndex&& rhs) noexcept = default;
    FlattenInvertedIndex&
    operator=(const FlattenInvertedIndex& rhs) = delete;
    FlattenInvertedIndex&
    operator=(FlattenInvertedIndex&& rhs) noexcept = default;

    [[nodiscard]] size_t
    size() const override {
        size_t res = sizeof(*this);

        res += this->nr_inner_dims_ * (sizeof(typename decltype(this->dim_map_)::key_type) +
                                       sizeof(typename decltype(this->dim_map_)::mapped_type));

        res += raw_index_offsets_.size() * sizeof(size_t);
        res += raw_index_ids_.size() * sizeof(uint32_t);
        res += raw_index_vals_.size() * sizeof(QType);

        if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
            res += this->meta_data_.row_sums_.size() * sizeof(float);
        }

        if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
            res += this->meta_data_.max_score_per_dim_.size() * sizeof(float);
        }

        if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
            res += this->meta_data_.block_max_data_.block_max_ids_.size() * sizeof(uint32_t);
            res += this->meta_data_.block_max_data_.block_max_scores_.size() * sizeof(float);
            res += this->meta_data_.block_max_data_.block_offsets_.size() * sizeof(size_t);
        }

        return res;
    }

    Status
    add(const SparseRow<DType>* data, size_t rows, int64_t dim) override;

    Status
    build_from_raw_data(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename) override;

    Status
    serialize(MemoryIOWriter& writer) const override;

    Status
    deserialize(MemoryIOReader& reader) override;

    [[nodiscard]] posting_list_iterator
    get_dim_plist_cursor(uint32_t dim_id, const BitsetView& bitset) const {
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

    /**
     * @brief Build the block max data from the raw index
     *
     * @param enable_mmap Whether to use file backed memory mapping
     * @param backed_filename File to use for memory mapping if enabled
     */
    void
    build_block_max_data(bool enable_mmap, const std::string& backed_filename);

    std::unique_ptr<BinaryContainer> index_container_;

    // Inverted lists start offsets
    // Each dimension's inverted list is stored contiguously in a flattened array
    // The start offset of each dimension's list is stored in raw_index_offsets_
    boost::span<size_t> raw_index_offsets_;

    // Inverted lists storing vector IDs
    boost::span<uint32_t> raw_index_ids_;

    // Inverted lists storing corresponding values
    boost::span<QType> raw_index_vals_;
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
                this->dim_map_[dim] = this->nr_inner_dims_++;
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
    auto raw_index_offsets_byte_sz = (this->nr_inner_dims_ + 1) * sizeof(size_t);
    auto container_byte_sz = raw_index_ids_byte_sz + raw_index_vals_byte_sz + raw_index_offsets_byte_sz;

    if (enable_mmap) {
        index_container_ = std::make_unique<FileBinaryContainer>(backed_filename + ".raw_index");
    } else {
        index_container_ = std::make_unique<MemBinaryContainer>();
    }

    index_container_->resize(container_byte_sz);
    index_container_->seal();

    auto* data = index_container_->data();
    raw_index_offsets_ = boost::span<size_t>(reinterpret_cast<size_t*>(data), this->nr_inner_dims_ + 1);
    raw_index_ids_ = boost::span<uint32_t>(reinterpret_cast<uint32_t*>(data + raw_index_offsets_byte_sz), nnz);
    raw_index_vals_ =
        boost::span<QType>(reinterpret_cast<QType*>(data + raw_index_ids_byte_sz + raw_index_offsets_byte_sz), nnz);

    std::size_t offset = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        raw_index_offsets_[i] = offset;
        offset += plist_cnts[i];
    }
    raw_index_offsets_[this->nr_inner_dims_] = offset;

    std::vector<size_t> curr_offsets(this->nr_inner_dims_);
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        curr_offsets[i] = raw_index_offsets_[i];
    }

    auto build_progress_interval = this->nr_rows_ / 10;
    for (size_t i = 0; i < this->nr_rows_; ++i) {
        if (build_progress_interval > 0 && i % build_progress_interval == 0) {
            LOG_KNOWHERE_INFO_ << "FlattenInvertedIndex[index_id=" << this->index_id_
                               << "] building progress: " << (i / build_progress_interval * 10) << "%";
        }
        size_t count = 0;
        readBinaryPOD(reader, count);
        SparseRow<DType> raw_row = SparseRow<DType>(count);
        if (count > 0) {
            reader.read(raw_row.data(), count * SparseRow<DType>::element_size());
        }
        add_row_to_index(raw_row, i, curr_offsets);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        this->meta_data_.max_score_per_dim_.resize(this->nr_inner_dims_, 0.0f);

        for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
            auto offset = this->raw_index_offsets_[i];
            size_t count = this->raw_index_offsets_[i + 1] - offset;
            auto ids = this->raw_index_ids_.subspan(offset, count);
            auto vals = this->raw_index_vals_.subspan(offset, count);
            for (size_t j = 0; j < count; ++j) {
                auto score = this->build_scorer_->vec_score(ids[j], vals[j]);
                this->meta_data_.max_score_per_dim_[i] = std::max(this->meta_data_.max_score_per_dim_[i], score);
            }
        }
    }

    LOG_KNOWHERE_INFO_ << "FlattenInvertedIndex[index_id=" << this->index_id_ << "] building progress: 100%";
}

template <typename DType, typename QType>
void
FlattenInvertedIndex<DType, QType>::build_block_max_data(bool enable_mmap, const std::string& backed_filename) {
    if (enable_mmap) {
        this->meta_data_.block_max_data_.container_ =
            std::make_unique<FileBinaryContainer>(backed_filename + ".block_max_data");
    } else {
        this->meta_data_.block_max_data_.container_ = std::make_unique<MemBinaryContainer>();
    }

    size_t total_blocks = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        size_t count = this->raw_index_offsets_[i + 1] - this->raw_index_offsets_[i];
        total_blocks +=
            (count + this->meta_data_.block_max_data_.block_size_ - 1) / this->meta_data_.block_max_data_.block_size_;
    }

    this->meta_data_.block_max_data_.container_->resize(this->nr_inner_dims_ * sizeof(size_t) +
                                                        total_blocks * (sizeof(uint32_t) + sizeof(float)));
    this->meta_data_.block_max_data_.container_->seal();

    auto block_max_data_container_data_ = this->meta_data_.block_max_data_.container_->data();

    size_t container_offset = 0;
    this->meta_data_.block_max_data_.block_offsets_ = boost::span<size_t>(
        reinterpret_cast<size_t*>(block_max_data_container_data_ + container_offset), this->nr_inner_dims_);
    container_offset += this->nr_inner_dims_ * sizeof(size_t);
    this->meta_data_.block_max_data_.block_max_ids_ = boost::span<uint32_t>(
        reinterpret_cast<uint32_t*>(block_max_data_container_data_ + container_offset), total_blocks);
    container_offset += total_blocks * sizeof(uint32_t);
    this->meta_data_.block_max_data_.block_max_scores_ =
        boost::span<float>(reinterpret_cast<float*>(block_max_data_container_data_ + container_offset), total_blocks);
    container_offset += total_blocks * sizeof(float);
    assert(container_offset == this->meta_data_.block_max_data_.container_->size());

    size_t block_max_offset = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        auto offset = this->raw_index_offsets_[i];
        size_t count = this->raw_index_offsets_[i + 1] - offset;
        auto ids = this->raw_index_ids_.subspan(offset, count);
        auto vals = this->raw_index_vals_.subspan(offset, count);

        float block_max_score = 0.0f;
        for (size_t j = 0; j < count; ++j) {
            if (j != 0 && (j % this->meta_data_.block_max_data_.block_size_) == 0) {
                this->meta_data_.block_max_data_.block_max_ids_[block_max_offset] = ids[j] - 1;
                this->meta_data_.block_max_data_.block_max_scores_[block_max_offset] = block_max_score;
                ++block_max_offset;
                block_max_score = 0.0f;
            }
            block_max_score = std::max(block_max_score, this->build_scorer_->vec_score(ids[j], vals[j]));
        }
        this->meta_data_.block_max_data_.block_max_ids_[block_max_offset] = ids[count - 1];
        this->meta_data_.block_max_data_.block_max_scores_[block_max_offset] = block_max_score;
        ++block_max_offset;
        this->meta_data_.block_max_data_.block_offsets_[i] = block_max_offset;
    }

    assert(block_max_offset == total_blocks);
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

    // build block max data if the flag is set
    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        build_block_max_data(enable_mmap, backed_filename);
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    this->index_size_gauge_->Set((double)size() / 1024.0 / 1024.0);
#endif

    return Status::success;
}

template <typename DType, typename QType>
inline void
FlattenInvertedIndex<DType, QType>::add_row_to_index(const SparseRow<DType>& row, std::uint32_t vec_id,
                                                     std::vector<size_t>& curr_offsets) {
    float row_sum = 0.0f;
    for (size_t j = 0; j < row.size(); ++j) {
        auto [dim, val] = row[j];
        // Skip values equals to or close enough to zero (which is little to the total IP score).
        if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
            continue;
        }

        row_sum += val;

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

template <typename DType, typename QType>
Status
FlattenInvertedIndex<DType, QType>::add(const SparseRow<DType>* data, size_t rows, int64_t dim) {
    std::unordered_map<uint32_t, size_t> plist_cnts;
    size_t total_nnz = 0;

    if (this->nr_rows_ != 0) {
        LOG_KNOWHERE_ERROR_ << "FlattenInvertedIndex is already built, and cannot be added to again.";
        return Status::invalid_index_error;
    }

    this->nr_rows_ = rows;
    this->max_dim_ = dim;

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
    this->build_stats_.dataset_nnz_stats_.resize(rows);
#endif

    for (uint32_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            auto [dim, val] = data[i][j];
            // Skip values equals to or close enough to zero (which is little to the total IP score).
            if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
                continue;
            }
            if (this->dim_map_.find(dim) == this->dim_map_.end()) {
                this->dim_map_[dim] = this->nr_inner_dims_++;
            }
            plist_cnts[this->dim_map_[dim]]++;
            ++total_nnz;
        }
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
        this->build_stats_.dataset_nnz_stats_[i] = data[i].size();
#endif
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
    this->build_stats_.posting_list_length_stats_.resize(this->nr_inner_dims_);
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        this->build_stats_.posting_list_length_stats_[i] = plist_cnts[i];
    }
#endif

    // calculate raw index byte size
    auto raw_index_offsets_byte_sz = (this->nr_inner_dims_ + 1) * sizeof(size_t);
    auto raw_index_ids_byte_sz = total_nnz * sizeof(uint32_t);
    auto raw_index_vals_byte_sz = total_nnz * sizeof(QType);
    auto raw_index_byte_sz = raw_index_ids_byte_sz + raw_index_vals_byte_sz + raw_index_offsets_byte_sz;

    this->index_container_ = std::make_unique<MemBinaryContainer>();

    this->index_container_->resize(raw_index_byte_sz);
    this->index_container_->seal();

    auto* buffer = this->index_container_->data();
    this->raw_index_offsets_ = boost::span<size_t>(reinterpret_cast<size_t*>(buffer), this->nr_inner_dims_ + 1);
    buffer += raw_index_offsets_byte_sz;
    this->raw_index_ids_ = boost::span<uint32_t>(reinterpret_cast<uint32_t*>(buffer), total_nnz);
    buffer += raw_index_ids_byte_sz;
    this->raw_index_vals_ = boost::span<QType>(reinterpret_cast<QType*>(buffer), total_nnz);

    std::size_t offset = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        this->raw_index_offsets_[i] = offset;
        offset += plist_cnts[i];
    }
    this->raw_index_offsets_[this->nr_inner_dims_] = offset;

    std::vector<size_t> curr_offsets(this->nr_inner_dims_);
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        curr_offsets[i] = this->raw_index_offsets_[i];
    }

    for (size_t i = 0; i < this->nr_rows_; ++i) {
        add_row_to_index(data[i], i, curr_offsets);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        this->meta_data_.max_score_per_dim_.resize(this->nr_inner_dims_, 0.0f);

        for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
            auto offset = this->raw_index_offsets_[i];
            size_t count = this->raw_index_offsets_[i + 1] - offset;
            auto ids = this->raw_index_ids_.subspan(offset, count);
            auto vals = this->raw_index_vals_.subspan(offset, count);
            for (size_t j = 0; j < count; ++j) {
                auto score = this->build_scorer_->vec_score(ids[j], vals[j]);
                this->meta_data_.max_score_per_dim_[i] = std::max(this->meta_data_.max_score_per_dim_[i], score);
            }
        }
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        build_block_max_data(false, "");
    }

    return Status::success;
}

template <typename DType, typename QType>
Status
FlattenInvertedIndex<DType, QType>::serialize(MemoryIOWriter& writer) const {
    // Serialized format:
    // 1. Index Header (36 bytes):
    //    - index_format_version (uint32_t): Version of the index format, currently 1
    //    - nr_rows (uint32_t): Number of rows in the index
    //    - max_dim (uint32_t): Number of columns, or maximum dimension ID
    //    - nr_inner_dims (uint32_t): Number of inner dimensions
    //    - reserved (16 bytes): Reserved for future use
    //
    // 2. Section Headers Table:
    //    - nr_sections (uint32_t): Number of sections
    //    - section_headers[nr_sections]: Array of section headers, each containing:
    //      - type (InvertedIndexSectionType): Type of the section
    //      - padding (uint32_t): Padding to align the section header to 8 bytes
    //      - offset (uint64_t): Offset of the section from the beginning of the file
    //      - size (uint64_t): Size of the section in bytes
    //
    // 3. Posting Lists Section:
    //    - index_encoding_type (uint32_t): Type of encoding used, only flat is supported for FlattenInvertedIndex
    //    - encoded_index_data: Flattened posting lists
    //
    // 4. Dimension Map Section:
    //    - dim_map_reverse[nr_inner_dims]: Array mapping internal dimension IDs to original dimensions
    //
    // 5. Optional Row Sums Section (if FLAG_HAS_ROW_SUMS is set):
    //    - row_sums[nr_rows]: Array of row sums (float)
    //
    // 6. Optional Max Scores Per Dimension Section (if FLAG_HAS_MAX_SCORES_PER_DIM is set):
    //    - max_score_per_dim[nr_inner_dims]: Array of maximum scores per dimension (float)
    //
    // 7. Optional Block Max Scores Section (if FLAG_HAS_BLOCK_MAX_SCORES is set):
    //    - total_blocks (size_t): Total number of blocks
    //    - block_size (uint32_t): Size of each block
    //    - block_max_data: Serialized block max data

    // write index header data
    const uint32_t index_format_version = current_index_file_format_version_;

    // Index File Header
    writer.write(&index_format_version, sizeof(uint32_t));  // index format version
    writer.write(&this->nr_rows_, sizeof(uint32_t));        // number of rows
    writer.write(&this->max_dim_, sizeof(uint32_t));        // number of cols, or maximum dimension id
    writer.write(&this->nr_inner_dims_, sizeof(uint32_t));  // number of inner dimensions
    auto reserved = std::array<uint8_t, 16>();              // reserved for future use
    writer.write(reserved.data(), reserved.size());

    // Section Headers Table
    uint32_t nr_sections = 2;  // base sections: inverted index and dim map
    // Count additional sections based on flags in a single operation
    nr_sections += ((this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) != 0) +
                   ((this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) != 0) +
                   ((this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) != 0);
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
    // use a section to store some build stats for prometheus
    nr_sections += 1;
#endif
    writer.write(&nr_sections, sizeof(uint32_t));

    // since writer doesn't support seekp() for now, calculate all sizes of each sections first
    InvertedIndexSectionHeader section_headers[nr_sections];
    uint64_t used_offset = sizeof(InvertedIndexSectionHeader) * nr_sections + 36;
    section_headers[0].type = InvertedIndexSectionType::POSTING_LISTS;
    section_headers[0].offset = used_offset;
    section_headers[0].size = sizeof(uint32_t) + index_container_->size();
    used_offset += section_headers[0].size;

    section_headers[1].type = InvertedIndexSectionType::DIM_MAP;
    section_headers[1].offset = used_offset;
    section_headers[1].size = sizeof(uint32_t) * this->nr_inner_dims_;
    used_offset += section_headers[1].size;

    auto curr_section_idx = 2;
    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        section_headers[curr_section_idx].type = InvertedIndexSectionType::ROW_SUMS;
        section_headers[curr_section_idx].offset = used_offset;
        section_headers[curr_section_idx].size = sizeof(float) * this->nr_rows_;
        used_offset += section_headers[curr_section_idx].size;
        curr_section_idx++;
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        section_headers[curr_section_idx].type = InvertedIndexSectionType::MAX_SCORES_PER_DIM;
        section_headers[curr_section_idx].offset = used_offset;
        section_headers[curr_section_idx].size = sizeof(float) * this->nr_inner_dims_;
        used_offset += section_headers[curr_section_idx].size;
        curr_section_idx++;
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        section_headers[curr_section_idx].type = InvertedIndexSectionType::BLOCK_MAX_SCORES;
        section_headers[curr_section_idx].offset = used_offset;
        section_headers[curr_section_idx].size =
            sizeof(size_t) + sizeof(uint32_t) + this->meta_data_.block_max_data_.container_->size();
        used_offset += section_headers[curr_section_idx].size;
        curr_section_idx++;
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
    section_headers[curr_section_idx].type = InvertedIndexSectionType::PROMETHEUS_BUILD_STATS;
    section_headers[curr_section_idx].offset = used_offset;
    section_headers[curr_section_idx].size =
        sizeof(uint32_t) * this->nr_rows_ + sizeof(uint32_t) * this->nr_inner_dims_;
    used_offset += section_headers[curr_section_idx].size;
    curr_section_idx++;
#endif

    assert(curr_section_idx == nr_sections);

    // write section headers table
    writer.write(section_headers, sizeof(InvertedIndexSectionHeader), nr_sections);

    // write index encoding type and encoded index
    uint32_t index_encoding_type = static_cast<uint32_t>(InvertedIndexEncoding::FLAT);
    writer.write(&index_encoding_type, sizeof(uint32_t));
    writer.write(index_container_->data(), index_container_->size());

    // write dim map
    auto dim_map_reverse = std::vector<uint32_t>(this->nr_inner_dims_);
    for (const auto& [dim, dim_id] : this->dim_map_) {
        dim_map_reverse[dim_id] = dim;
    }
    writer.write(dim_map_reverse.data(), sizeof(uint32_t), this->nr_inner_dims_);

    // write index meta data
    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        writer.write(this->meta_data_.row_sums_.data(), sizeof(float), this->nr_rows_);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        writer.write(this->meta_data_.max_score_per_dim_.data(), sizeof(float), this->nr_inner_dims_);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        size_t total_blocks = this->meta_data_.block_max_data_.block_max_ids_.size();
        writer.write(&total_blocks, sizeof(size_t));
        writer.write(&this->meta_data_.block_max_data_.block_size_, sizeof(uint32_t));
        writer.write(this->meta_data_.block_max_data_.container_->data(),
                     this->meta_data_.block_max_data_.container_->size());
    }

    // write prometheus build stats
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
    writer.write(this->build_stats_.dataset_nnz_stats_.data(), sizeof(uint32_t), this->nr_rows_);
    writer.write(this->build_stats_.posting_list_length_stats_.data(), sizeof(uint32_t), this->nr_inner_dims_);
#endif

    return Status::success;
}

template <typename DType, typename QType>
Status
FlattenInvertedIndex<DType, QType>::deserialize(MemoryIOReader& reader) {
    auto file_header_handler = [&]() {
        uint32_t index_format_version = 0;
        reader.read(&index_format_version, sizeof(uint32_t));
        // for now we only support version 1
        if (index_format_version != current_index_file_format_version_) {
            return Status::invalid_serialized_index_type;
        }

        reader.read(&this->nr_rows_, sizeof(uint32_t));
        reader.read(&this->max_dim_, sizeof(uint32_t));
        reader.read(&this->nr_inner_dims_, sizeof(uint32_t));
        // skip reserved bytes
        reader.advance(16);

        return Status::success;
    };

    auto sections_handler = [&]() {
        uint32_t nr_sections = 0;
        reader.read(&nr_sections, sizeof(uint32_t));
        size_t sec_table_offset = reader.tellg();

        for (uint32_t i = 0; i < nr_sections; ++i) {
            InvertedIndexSectionHeader section_header;
            reader.seekg(sec_table_offset);
            reader.read(&section_header, sizeof(InvertedIndexSectionHeader));
            sec_table_offset += sizeof(InvertedIndexSectionHeader);

            switch (section_header.type) {
                case InvertedIndexSectionType::POSTING_LISTS: {
                    reader.seekg(section_header.offset);
                    // check index encoding type
                    uint32_t index_encoding_type = 0;
                    reader.read(&index_encoding_type, sizeof(uint32_t));
                    if (index_encoding_type != static_cast<uint32_t>(InvertedIndexEncoding::FLAT)) {
                        return Status::invalid_serialized_index_type;
                    }
                    this->raw_index_offsets_ = boost::span<size_t>(
                        reinterpret_cast<size_t*>(reader.data() + reader.tellg()), this->nr_inner_dims_ + 1);
                    reader.advance(sizeof(size_t) * (this->nr_inner_dims_ + 1));
                    auto nnz = this->raw_index_offsets_[this->nr_inner_dims_];
                    this->raw_index_ids_ =
                        boost::span<uint32_t>(reinterpret_cast<uint32_t*>(reader.data() + reader.tellg()), nnz);
                    reader.advance(nnz * sizeof(uint32_t));
                    this->raw_index_vals_ =
                        boost::span<QType>(reinterpret_cast<QType*>(reader.data() + reader.tellg()), nnz);
                    reader.advance(nnz * sizeof(QType));
                    // deserialize will use the memory from reader, so containers are not needed
                    // explicitly assign nullptr to them
                    this->index_container_ = nullptr;
                    break;
                }
                case InvertedIndexSectionType::DIM_MAP: {
                    reader.seekg(section_header.offset);
                    for (uint32_t i = 0; i < this->nr_inner_dims_; ++i) {
                        uint32_t dim = 0;
                        reader.read(&dim, sizeof(uint32_t));
                        this->dim_map_[dim] = i;
                    }
                    break;
                }
                case InvertedIndexSectionType::ROW_SUMS: {
                    reader.seekg(section_header.offset);
                    this->meta_data_.row_sums_.resize(this->nr_rows_);
                    reader.read(this->meta_data_.row_sums_.data(), sizeof(float), this->nr_rows_);
                    break;
                }
                case InvertedIndexSectionType::MAX_SCORES_PER_DIM: {
                    reader.seekg(section_header.offset);
                    this->meta_data_.max_score_per_dim_.resize(this->nr_inner_dims_);
                    reader.read(this->meta_data_.max_score_per_dim_.data(), sizeof(float), this->nr_inner_dims_);
                    break;
                }
                case InvertedIndexSectionType::BLOCK_MAX_SCORES: {
                    reader.seekg(section_header.offset);
                    size_t total_blocks = 0;
                    reader.read(&total_blocks, sizeof(size_t));
                    reader.read(&this->meta_data_.block_max_data_.block_size_, sizeof(uint32_t));
                    this->meta_data_.block_max_data_.block_offsets_ = boost::span<size_t>(
                        reinterpret_cast<size_t*>(reader.data() + reader.tellg()), this->nr_inner_dims_);
                    reader.advance(this->nr_inner_dims_ * sizeof(size_t));
                    this->meta_data_.block_max_data_.block_max_ids_ = boost::span<uint32_t>(
                        reinterpret_cast<uint32_t*>(reader.data() + reader.tellg()), total_blocks);
                    reader.advance(total_blocks * sizeof(uint32_t));
                    this->meta_data_.block_max_data_.block_max_scores_ =
                        boost::span<float>(reinterpret_cast<float*>(reader.data() + reader.tellg()), total_blocks);
                    this->meta_data_.block_max_data_.container_ = nullptr;
                    reader.advance(total_blocks * sizeof(float));
                    break;
                }
                case InvertedIndexSectionType::PROMETHEUS_BUILD_STATS: {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
                    reader.seekg(section_header.offset);
                    auto dataset_nnz_stats = std::vector<uint32_t>(this->nr_rows_);
                    reader.read(dataset_nnz_stats.data(), sizeof(uint32_t), this->nr_rows_);
                    auto posting_list_length_stats = std::vector<uint32_t>(this->nr_inner_dims_);
                    reader.read(posting_list_length_stats.data(), sizeof(uint32_t), this->nr_inner_dims_);
                    for (size_t i = 0; i < this->nr_rows_; ++i) {
                        this->index_dataset_nnz_len_histogram_->Observe(dataset_nnz_stats[i]);
                    }
                    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
                        this->index_posting_list_len_histogram_->Observe(posting_list_length_stats[i]);
                    }
#endif
                    break;
                }
                default:
                    // skip unknown sections
                    break;
            }
        }

        return Status::success;
    };

    if (auto status = file_header_handler(); status != Status::success) {
        return status;
    }

    if (auto status = sections_handler(); status != Status::success) {
        return status;
    }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    this->index_size_gauge_->Set((double)size() / 1024.0 / 1024.0);
#endif

    return Status::success;
}

}  // namespace knowhere::sparse::inverted
