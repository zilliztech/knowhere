#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <unordered_set>

#include "index/sparse/codec/block_codec.h"
#include "index/sparse/inverted_index.h"
#include "index/sparse/inverted_index_build.h"
#include "index/sparse/inverted_index_format.h"
#include "index/sparse/parallel_build.h"
#include "index/sparse/scorer.h"
#include "knowhere/bitsetview.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere::sparse::inverted {

// Empty posting lists are never materialized. Encoded count value 0 (wire byte 0x80) therefore marks the singleton
// short form whose logical count is one. Its sole document ID occupies the posting block max-ID slot. BM25 stores
// TF - 1 after that slot as an untagged internal varint, including an explicit zero for TF == 1; IP keeps its regular
// raw value payload.
inline constexpr uint32_t kSingletonShortFormSizeMarker = 0;

struct PostingListSizeInfo {
    uint32_t size;
    bool is_singleton_short_form;
};

[[nodiscard]] inline PostingListSizeInfo
posting_list_size_info(uint32_t encoded_size) noexcept {
    const bool is_singleton_short_form = encoded_size == kSingletonShortFormSizeMarker;
    return {is_singleton_short_form ? 1U : encoded_size, is_singleton_short_form};
}

// Internal little-endian base-128 varint. Unlike unsigned LEB128 or Lucene VInt, bit 7 marks the final byte: it is
// clear on continuation bytes and set on the terminator (for example, 0 -> 0x80 and 128 -> 0x00 0x81).
inline void
varint_encode(uint32_t val, std::vector<uint8_t>& out) {
    while (val >= 128) {
        out.push_back(static_cast<uint8_t>(val & 0x7F));
        val >>= 7;
    }
    out.push_back(static_cast<uint8_t>(val | 0x80));
}

inline const uint8_t*
varint_decode(const uint8_t* in, uint32_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t v = 0;
        unsigned int shift = 0;
        for (;;) {
            uint8_t c = *in++;
            v += (static_cast<uint32_t>(c & 0x7F) << shift);
            if (c & 0x80) {
                *out++ = v;
                break;
            }
            shift += 7;
        }
    }
    return in;
}

template <typename VType>
class BlockInvertedIndexCursor {
 public:
    enum class BlockFilterState : uint8_t { AllValid, AllFiltered, Mixed };

    BlockInvertedIndexCursor(const BlockCodecPtr& block_codec, std::uint8_t const* data, std::uint32_t universe,
                             BitsetView bitset, uint32_t initial_lower_bound = 0,
                             uint32_t valid_upper_bound = std::numeric_limits<uint32_t>::max())
        : universe_(universe),
          block_codec_(block_codec),
          block_size_(block_codec->block_size()),
          bitset_(bitset),
          valid_upper_bound_(std::min(universe, valid_upper_bound)) {
        uint32_t encoded_size = 0;
        base_ = varint_decode(data, &encoded_size, 1);
        const auto size_info = posting_list_size_info(encoded_size);
        n_ = size_info.size;
        is_singleton_short_form_ = size_info.is_singleton_short_form;
        assert(!is_singleton_short_form_ || block_codec_->supports_singleton_short_form());
        assert(n_ > 0 && n_ <= universe_);
        nr_blocks_ = (n_ + block_size_ - 1) / block_size_;

        const size_t block_maxids_size = sizeof(uint32_t) * nr_blocks_;
        const size_t block_offsets_size = sizeof(uint32_t) * (nr_blocks_ - 1);
        block_maxids_ = base_;
        block_offsets_ = block_maxids_ + block_maxids_size;
        blocks_data_ = block_offsets_ + block_offsets_size;

        ids_buf_.resize(block_size_);
        vals_buf_.resize(block_size_);
        reset(initial_lower_bound);
    }

    void
    reset(uint32_t lower_bound = 0) {
        if (lower_bound >= valid_upper_bound_ || lower_bound > block_maxid(nr_blocks_ - 1)) {
            set_invalid();
            return;
        }
        decode_vecids_block(lower_bound_block(0, lower_bound));
        advance_to(lower_bound);
        if (cur_vec_id_ >= valid_upper_bound_) {
            set_invalid();
            return;
        }
        skip_filtered_ids();
    }

    void
    next_raw() {
        ++pos_in_block_;
        if (pos_in_block_ == cur_block_size_) [[unlikely]] {
            if (cur_block_ + 1 == nr_blocks_) {
                cur_vec_id_ = universe_;
                return;
            }
            decode_vecids_block(cur_block_ + 1);
        } else {
            cur_vec_id_ += ids_buf_[pos_in_block_] + 1;
        }
        if (cur_vec_id_ >= valid_upper_bound_) {
            set_invalid();
        }
    }

    void
    next() {
        next_raw();
        skip_filtered_ids();
    }

    /**
     * Moves to the next vector, counting from the current position,
     * with the ID equal to or greater than `lower_bound`.
     *
     * In particular, if called with a value that is less than or equal
     * to the current vector ID, the position will not change.
     */
    void
    next_geq(uint32_t lower_bound) {
        if (lower_bound >= valid_upper_bound_) {
            set_invalid();
            return;
        }
        if (lower_bound > cur_block_maxid_) [[unlikely]] {
            if (lower_bound > block_maxid(nr_blocks_ - 1)) {
                cur_vec_id_ = universe_;
                return;
            }
            decode_vecids_block(lower_bound_block(cur_block_ + 1, lower_bound));
        }

        advance_to(lower_bound);

        skip_filtered_ids();
    }

    [[nodiscard]] uint32_t
    vec_id() const {
        return cur_vec_id_;
    }

    VType
    val() {
        if (!vals_decoded_) {
            decode_vals_block();
        }

        // Only the uint32_t/BM25 value path stores TF - 1 through the block codec; IP values remain verbatim.
        if constexpr (std::is_same_v<VType, uint32_t>) {
            return vals_buf_[pos_in_block_] + 1;
        } else {
            return vals_buf_[pos_in_block_];
        }
    }

    [[nodiscard]] uint32_t
    position() const {
        return cur_block_ * block_size_ + pos_in_block_;
    }

    [[nodiscard]] uint32_t
    block_maxid(uint32_t blk_idx) const {
        return load_u32(block_maxids_ + sizeof(uint32_t) * blk_idx);
    }

    void
    skip_filtered_ids() {
        while (cur_block_filter_state_ == BlockFilterState::Mixed && cur_vec_id_ < universe_ &&
               bitset_.test(cur_vec_id_)) {
            next_raw();
        }
    }

    [[nodiscard]] BlockFilterState
    classify_filter_range(uint32_t blkid) const {
        if (bitset_.empty()) {
            return BlockFilterState::AllValid;
        }

        const size_t begin = blkid == 0 ? 0 : static_cast<size_t>(block_maxid(blkid - 1)) + 1;
        const size_t end = std::min<size_t>(valid_upper_bound_, static_cast<size_t>(block_maxid(blkid)) + 1);
        constexpr size_t kMaxRangeScanBits = 1U << 16;
        if (end <= begin || end > bitset_.size() || end - begin > kMaxRangeScanBits) {
            return BlockFilterState::Mixed;
        }

        return bitset_.range_all_filtered(begin, end) ? BlockFilterState::AllFiltered : BlockFilterState::Mixed;
    }

    void
    set_invalid() {
        cur_block_ = nr_blocks_;
        pos_in_block_ = 0;
        cur_block_maxid_ = universe_;
        cur_block_size_ = 0;
        cur_vec_id_ = universe_;
        vals_block_data_ = nullptr;
        vals_decoded_ = false;
    }

    [[nodiscard]] static uint32_t
    load_u32(const uint8_t* data) {
        uint32_t value = 0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    [[nodiscard]] uint32_t
    block_offset(uint32_t blk_idx) const {
        return load_u32(block_offsets_ + sizeof(uint32_t) * blk_idx);
    }

    [[nodiscard]] uint32_t
    lower_bound_block(uint32_t first, uint32_t lower_bound) const {
        uint32_t last = nr_blocks_;
        while (first < last) {
            const uint32_t middle = first + (last - first) / 2;
            if (block_maxid(middle) < lower_bound) {
                first = middle + 1;
            } else {
                last = middle;
            }
        }
        return first;
    }

    void
    advance_to(uint32_t lower_bound) {
        while (cur_vec_id_ < lower_bound) {
            cur_vec_id_ += ids_buf_[++pos_in_block_] + 1;
        }
    }

    void
    decode_vecids_block(uint32_t blkid) {
        while (blkid < nr_blocks_) {
            const size_t block_begin = blkid == 0 ? 0 : static_cast<size_t>(block_maxid(blkid - 1)) + 1;
            if (block_begin >= valid_upper_bound_) {
                set_invalid();
                return;
            }
            cur_block_filter_state_ = classify_filter_range(blkid);
            if (cur_block_filter_state_ != BlockFilterState::AllFiltered) {
                break;
            }
            ++blkid;
        }
        if (blkid == nr_blocks_) {
            set_invalid();
            return;
        }

        const uint32_t endpoint = blkid != 0U ? block_offset(blkid - 1) : 0;
        uint8_t const* block_data = blocks_data_ + endpoint;
        cur_block_size_ = ((blkid + 1) * block_size_ <= n_) ? block_size_ : (n_ % block_size_);
        cur_block_maxid_ = block_maxid(blkid);

        if (is_singleton_short_form_) [[unlikely]] {
            assert(n_ == 1 && nr_blocks_ == 1 && blkid == 0 && cur_block_size_ == 1);
            assert(cur_block_maxid_ < universe_);
            ids_buf_[0] = cur_block_maxid_;
            vals_block_data_ = block_data;
        } else {
            vals_block_data_ = block_codec_->decode(block_data, ids_buf_.data(), cur_block_size_);

            // Materialize only the first document ID. The remaining decoded values stay as gaps and are integrated
            // lazily as the cursor advances, avoiding a full-block prefix sum.
            const uint32_t block_base = blkid == 0 ? 0 : block_maxid(blkid - 1) + 1;
            ids_buf_[0] += block_base;
        }
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(vals_block_data_, 0, 3);
#endif

        cur_block_ = blkid;
        pos_in_block_ = 0;
        cur_vec_id_ = ids_buf_[0];
        vals_decoded_ = false;
    }

    void
    decode_vals_block() {
        if constexpr (std::is_same_v<VType, uint32_t>) {
            uint8_t const* next_block = vals_block_data_;
            if (is_singleton_short_form_) [[unlikely]] {
                next_block = varint_decode(vals_block_data_, vals_buf_.data(), 1);
            } else {
                next_block = block_codec_->decode(vals_block_data_, vals_buf_.data(), cur_block_size_);
            }
#if defined(__GNUC__) || defined(__clang__)
            __builtin_prefetch(next_block, 0, 3);
#endif
        } else {
            std::memcpy(vals_buf_.data(), vals_block_data_, cur_block_size_ * sizeof(VType));
        }
        vals_decoded_ = true;
    }

    [[nodiscard]] bool
    valid() const {
        return cur_vec_id_ != universe_;
    }

    uint32_t n_{0};
    uint8_t const* base_{nullptr};
    uint32_t nr_blocks_{0};
    uint8_t const* block_maxids_{nullptr};
    uint8_t const* block_offsets_{nullptr};
    uint8_t const* blocks_data_{nullptr};
    uint32_t universe_{0};

    uint32_t cur_block_{0};
    uint32_t pos_in_block_{0};
    uint32_t cur_block_maxid_{0};
    uint32_t cur_block_size_{0};
    uint32_t cur_vec_id_{0};

    uint8_t const* vals_block_data_{nullptr};
    bool vals_decoded_{false};

    std::vector<uint32_t> ids_buf_;
    std::vector<VType> vals_buf_;
    BlockCodecPtr block_codec_;
    std::size_t block_size_;
    bool is_singleton_short_form_{false};
    BitsetView bitset_;
    uint32_t valid_upper_bound_{0};
    BlockFilterState cur_block_filter_state_{BlockFilterState::AllValid};
};

template <typename DType, typename QType, IndexScorerType MetricType>
class BlockInvertedIndex : public CRTPInvertedIndex<BlockInvertedIndex<DType, QType, MetricType>, DType> {
 public:
    // IP metric: values stored as raw bytes in blocks.
    // BM25 metric: values stored with block codec compression as uint32_t.
    static constexpr bool kIsIPMetric = MetricType == IndexScorerType::IP;
    using posting_list_iterator = BlockInvertedIndexCursor<std::conditional_t<kIsIPMetric, QType, uint32_t>>;

    explicit BlockInvertedIndex(BlockCodecPtr block_codec)
        : CRTPInvertedIndex<BlockInvertedIndex<DType, QType, MetricType>, DType>("blockinverted"),
          block_codec_(block_codec) {
    }

    BlockInvertedIndex(const BlockInvertedIndex& rhs) = delete;
    BlockInvertedIndex(BlockInvertedIndex&& rhs) noexcept = default;
    BlockInvertedIndex&
    operator=(const BlockInvertedIndex& rhs) = delete;
    BlockInvertedIndex&
    operator=(BlockInvertedIndex&& rhs) noexcept = default;

    [[nodiscard]] size_t
    size() const override {
        size_t res = sizeof(*this);

        res += this->dim_map_.byte_size();

        res += posting_blocks_dim_offsets_.size() * sizeof(size_t);
        res += posting_blocks_data_.size();

        const auto& flags = this->meta_data_.flags_;

        if (flags & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
            res += this->meta_data_.row_sums_.size() * sizeof(float);
        }

        if (flags & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
            res += this->meta_data_.max_score_per_dim_.size() * sizeof(float);
        }

        if (flags & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
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
        const auto begin = this->posting_blocks_dim_offsets_[dim_id];
        auto* data = this->posting_blocks_data_.data() + begin;
        return posting_list_iterator(this->block_codec_, data, this->nr_rows_, bitset);
    }

    [[nodiscard]] posting_list_iterator
    get_dim_plist_cursor(uint32_t dim_id, const BitsetView& bitset, uint32_t initial_lower_bound,
                         uint32_t valid_upper_bound) const {
        const auto begin = this->posting_blocks_dim_offsets_[dim_id];
        auto* data = this->posting_blocks_data_.data() + begin;
        return posting_list_iterator(this->block_codec_, data, this->nr_rows_, bitset, initial_lower_bound,
                                     valid_upper_bound);
    }

 private:
    /**
     * @brief Add a single sparse vector to the index
     *
     * @param raw_row The sparse vector to add
     * @param vec_id ID to assign to this vector
     * @param raw_index_ids Inverted lists storing vector IDs
     * @param raw_index_vals Inverted lists storing quantized values
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
    add_row_to_index(const SparseRow<DType>& raw_row, uint32_t vec_id, std::span<uint32_t>& raw_index_ids,
                     std::span<QType>& raw_index_vals, std::vector<size_t>& curr_offsets) {
        float row_sum = 0.0f;

        for (size_t j = 0; j < raw_row.size(); ++j) {
            auto [dim, val] = raw_row[j];
            // Skip values equals to or close enough to zero (which is little to the total IP score).
            if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
                continue;
            }

            row_sum += val;

            auto inner_dim = this->dim_map_.lookup(dim);
            if (!inner_dim.has_value()) {
                throw std::runtime_error("unexpected vector dimension in BlockInvertedIndex");
            }

            auto offset = curr_offsets[inner_dim.value()]++;
            raw_index_ids[offset] = vec_id;
            raw_index_vals[offset] = get_quant_val<DType, QType>(val);
        }

        if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
            this->meta_data_.row_sums_.push_back(row_sum);
        }
    }

    /**
     * @brief Build the raw index from the serialized data
     *
     * @param reader Reader containing serialized index data to analyze
     * @param raw_index_container Container to store the raw index
     * @param raw_index_ids Inverted lists storing vector IDs
     * @param raw_index_vals Inverted lists storing quantized values
     * @param raw_index_offsets Inverted lists storing offsets into each dimension's inverted list
     * @param enable_mmap Whether to use file backed memory mapping
     * @param backed_filename File to use for memory mapping if enabled
     *
     * This function starts at the serialized row payload after the caller has read the row and dimension headers. It
     * first scans the rows to count postings per external dimension, then allocates and fills the flattened raw
     * arrays.
     * If memory mapping is enabled, it creates a memory mapped file to store the data.
     * Otherwise, it uses heap memory.
     *
     * The memory is allocated for the flattened raw_index_ids/raw_index_vals arrays and raw_index_offsets table.
     */
    void
    build_raw_index(MemoryIOReader& reader, std::unique_ptr<BinaryContainer>& raw_index_container,
                    std::span<uint32_t>& raw_index_ids, std::span<QType>& raw_index_vals,
                    std::span<size_t>& raw_index_offsets, bool enable_mmap, const std::string& backed_filename);

    /**
     * @brief Build the block max data from the raw index
     *
     * @param raw_index_ids Inverted lists storing vector IDs
     * @param raw_index_vals Inverted lists storing quantized values
     * @param raw_index_offsets Inverted lists storing offsets into each dimension's inverted list
     * @param enable_mmap Whether to use file backed memory mapping
     * @param backed_filename File to use for memory mapping if enabled
     */
    void
    build_block_max_data(std::span<uint32_t> raw_index_ids, std::span<QType> raw_index_vals,
                         std::span<size_t> raw_index_offsets, bool enable_mmap, const std::string& backed_filename);

    /**
     * @brief Encode the posting list into a binary format
     *
     * @param out_buf Output buffer to store the encoded posting list
     * @param vec_ids Inverted lists storing vector IDs
     * @param vals Inverted lists storing quantized values
     */
    void
    encode_posting_list(std::vector<uint8_t>& out_buf, std::span<uint32_t> vec_ids, std::span<QType> vals);

    /**
     * @brief Build the block compressed index from the raw index
     *
     * @param raw_index_ids Inverted lists storing vector IDs
     * @param raw_index_vals Inverted lists storing quantized values
     * @param raw_index_offsets Inverted lists storing offsets into each dimension's inverted list
     * @param enable_mmap Whether to use file backed memory mapping
     * @param backed_filename File to use for memory mapping if enabled
     *
     * This function builds the block compressed index from the raw index.
     * It first creates a postings container to store the block compressed index.
     * Then it writes the endpoints of each dimension's inverted list to the postings container.
     * Finally, it writes the block compressed index to the postings container.
     *
     * The postings container is a memory mapped file if memory mapping is enabled,
     * otherwise it is a heap allocated container.
     */
    void
    build_block_index(std::span<uint32_t>& raw_index_ids, std::span<QType>& raw_index_vals,
                      std::span<size_t>& raw_index_offsets, bool enable_mmap, const std::string& backed_filename);

    std::unique_ptr<BinaryContainer> index_container_;

    // Inverted lists start offsets
    // Each dimension's inverted list is stored contiguously in a flattened array
    // The start offset of each dimension's list is stored in posting_blocks_dim_offsets_
    std::span<size_t> posting_blocks_dim_offsets_;

    // Posting headers and encoded document-ID/value blocks for all dimensions, followed by a decoder guard.
    std::span<uint8_t> posting_blocks_data_;

    // Block codec
    BlockCodecPtr block_codec_;
};

template <typename DType, typename QType, IndexScorerType MetricType>
void
BlockInvertedIndex<DType, QType, MetricType>::build_raw_index(MemoryIOReader& reader,
                                                              std::unique_ptr<BinaryContainer>& raw_index_container,
                                                              std::span<uint32_t>& raw_index_ids,
                                                              std::span<QType>& raw_index_vals,
                                                              std::span<size_t>& raw_index_offsets, bool enable_mmap,
                                                              const std::string& backed_filename) {
    const auto saved_reader_loc = reader.tellg();
    const auto nnz = (reader.remaining() - (this->nr_rows_ * sizeof(size_t))) / SparseRow<DType>::element_size();

    std::unordered_set<uint32_t> external_dims;
    for (uint32_t i = 0; i < this->nr_rows_; ++i) {
        size_t count = 0;
        readBinaryPOD(reader, count);
        if (count == 0) {
            continue;
        }
        for (size_t j = 0; j < count; ++j) {
            uint32_t dim = 0;
            readBinaryPOD(reader, dim);
            external_dims.insert(dim);
            reader.advance(sizeof(DType));
        }
    }

    this->dim_map_.build_from_external_dims(external_dims);
    this->nr_inner_dims_ = this->dim_map_.size();

    // reset reader to the saved beginning
    reader.seekg(saved_reader_loc);

    std::vector<size_t> plist_cnts(this->nr_inner_dims_, 0);
    for (uint32_t i = 0; i < this->nr_rows_; ++i) {
        size_t count = 0;
        readBinaryPOD(reader, count);
        if (count == 0) {
            continue;
        }
        for (size_t j = 0; j < count; ++j) {
            uint32_t dim = 0;
            readBinaryPOD(reader, dim);
            auto inner_dim = this->dim_map_.lookup(dim);
            if (!inner_dim.has_value()) {
                throw std::runtime_error("unexpected vector dimension in BlockInvertedIndex raw data");
            }
            plist_cnts[inner_dim.value()]++;
            reader.advance(sizeof(DType));
        }
    }

    // reset reader to the saved beginning
    reader.seekg(saved_reader_loc);

    // calculate raw index byte size
    auto raw_index_ids_byte_sz = nnz * sizeof(uint32_t);
    auto raw_index_vals_byte_sz = nnz * sizeof(QType);
    auto raw_index_offsets_byte_sz = (this->nr_inner_dims_ + 1) * sizeof(size_t);
    auto raw_index_byte_sz = raw_index_ids_byte_sz + raw_index_vals_byte_sz + raw_index_offsets_byte_sz;

    if (enable_mmap) {
        raw_index_container = std::make_unique<FileBinaryContainer>(backed_filename + ".raw_index");
    } else {
        raw_index_container = std::make_unique<MemBinaryContainer>();
    }

    raw_index_container->resize(raw_index_byte_sz);
    raw_index_container->seal();

    auto* data = raw_index_container->data();
    raw_index_ids = std::span<uint32_t>(reinterpret_cast<uint32_t*>(data), nnz);
    raw_index_vals = std::span<QType>(reinterpret_cast<QType*>(data + raw_index_ids_byte_sz), nnz);
    raw_index_offsets = std::span<size_t>(
        reinterpret_cast<size_t*>(data + raw_index_ids_byte_sz + raw_index_vals_byte_sz), this->nr_inner_dims_ + 1);

    std::size_t offset = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        raw_index_offsets[i] = offset;
        offset += plist_cnts[i];
    }
    raw_index_offsets[this->nr_inner_dims_] = offset;

    std::vector<size_t> curr_offsets(this->nr_inner_dims_);
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        curr_offsets[i] = raw_index_offsets[i];
    }

    for (size_t i = 0; i < this->nr_rows_; ++i) {
        size_t count = 0;
        readBinaryPOD(reader, count);
        SparseRow<DType> raw_row = SparseRow<DType>(count);
        if (count > 0) {
            reader.read(raw_row.data(), count * SparseRow<DType>::element_size());
        }
        add_row_to_index(raw_row, i, raw_index_ids, raw_index_vals, curr_offsets);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        this->meta_data_.resize_max_score_per_dim(this->nr_inner_dims_, 0.0f);

        for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
            auto offset = raw_index_offsets[i];
            size_t count = raw_index_offsets[i + 1] - offset;
            auto ids = raw_index_ids.subspan(offset, count);
            auto vals = raw_index_vals.subspan(offset, count);
            for (size_t j = 0; j < count; ++j) {
                auto score = this->build_scorer_->vec_score(ids[j], vals[j]);
                this->meta_data_.max_score_per_dim_[i] = std::max(this->meta_data_.max_score_per_dim_[i], score);
            }
        }
    }
}

template <typename DType, typename QType, IndexScorerType MetricType>
void
BlockInvertedIndex<DType, QType, MetricType>::build_block_max_data(std::span<uint32_t> raw_index_ids,
                                                                   std::span<QType> raw_index_vals,
                                                                   std::span<size_t> raw_index_offsets,
                                                                   bool enable_mmap,
                                                                   const std::string& backed_filename) {
    if (enable_mmap) {
        this->meta_data_.block_max_data_.container_ =
            std::make_unique<FileBinaryContainer>(backed_filename + ".block_max_data");
    } else {
        this->meta_data_.block_max_data_.container_ = std::make_unique<MemBinaryContainer>();
    }

    const auto block_size = this->meta_data_.block_max_data_.block_size_;
    size_t total_blocks = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        const auto posting_count = raw_index_offsets[i + 1] - raw_index_offsets[i];
        total_blocks += (posting_count + block_size - 1) / block_size;
    }

    this->meta_data_.block_max_data_.container_->resize(this->nr_inner_dims_ * sizeof(size_t) +
                                                        total_blocks * (sizeof(uint32_t) + sizeof(float)));
    this->meta_data_.block_max_data_.container_->seal();

    uint8_t* data = this->meta_data_.block_max_data_.container_->data();

    size_t container_offset = 0;
    this->meta_data_.block_max_data_.block_offsets_ =
        std::span<size_t>(reinterpret_cast<size_t*>(data + container_offset), this->nr_inner_dims_);
    container_offset += this->nr_inner_dims_ * sizeof(size_t);
    this->meta_data_.block_max_data_.block_max_ids_ =
        std::span<uint32_t>(reinterpret_cast<uint32_t*>(data + container_offset), total_blocks);
    container_offset += total_blocks * sizeof(uint32_t);
    this->meta_data_.block_max_data_.block_max_scores_ =
        std::span<float>(reinterpret_cast<float*>(data + container_offset), total_blocks);
    container_offset += total_blocks * sizeof(float);
    assert(container_offset == this->meta_data_.block_max_data_.container_->size());

    size_t next_block_offset = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        const auto posting_count = raw_index_offsets[i + 1] - raw_index_offsets[i];
        next_block_offset += (posting_count + block_size - 1) / block_size;
        this->meta_data_.block_max_data_.block_offsets_[i] = next_block_offset;
    }
    assert(next_block_offset == total_blocks);

    parallel_for(this->nr_inner_dims_, [&](size_t i) {
        const auto posting_offset = raw_index_offsets[i];
        const auto posting_count = raw_index_offsets[i + 1] - posting_offset;
        const auto ids = raw_index_ids.subspan(posting_offset, posting_count);
        const auto vals = raw_index_vals.subspan(posting_offset, posting_count);

        size_t block_index = i == 0 ? 0 : this->meta_data_.block_max_data_.block_offsets_[i - 1];
        float block_max_score = 0.0f;
        for (size_t j = 0; j < posting_count; ++j) {
            if (j != 0 && (j % block_size) == 0) {
                this->meta_data_.block_max_data_.block_max_ids_[block_index] = ids[j] - 1;
                this->meta_data_.block_max_data_.block_max_scores_[block_index] = block_max_score;
                ++block_index;
                block_max_score = 0.0f;
            }
            block_max_score = std::max(block_max_score, this->build_scorer_->vec_score(ids[j], vals[j]));
        }
        this->meta_data_.block_max_data_.block_max_ids_[block_index] = ids.back();
        this->meta_data_.block_max_data_.block_max_scores_[block_index] = block_max_score;
        assert(block_index + 1 == this->meta_data_.block_max_data_.block_offsets_[i]);
    });
}

template <typename DType, typename QType, IndexScorerType MetricType>
void
BlockInvertedIndex<DType, QType, MetricType>::encode_posting_list(std::vector<uint8_t>& out_buf,
                                                                  std::span<uint32_t> vec_ids, std::span<QType> vals) {
    // Posting list layout:
    // +----------------+------------------------------------------+
    // | encoded_count | internal varint; 0 marks a singleton     |
    // +----------------+------------------------------------------+
    // | block_maxids  | uint32_t[nr_blocks]: max vector ID/block |
    // +----------------+------------------------------------------+
    // | block_ends    | uint32_t[nr_blocks-1]: block end offsets |
    // +----------------+------------------------------------------+
    // | blocks        | uint8_t[]: encoded posting data          |
    // +----------------+------------------------------------------+
    // block_ends[i] is the end of block i and therefore the start of block i + 1; block 0 starts implicitly at 0.
    // A codec with the singleton short form stores no doc-ID payload and reuses block_maxids[0] as the sole document
    // ID.
    // BM25 follows it with an untagged internal varint containing TF - 1, including zero for TF == 1. IP follows it
    // with its regular raw value payload.
    const uint32_t block_sz = block_codec_->block_size();

    const size_t list_sz = vec_ids.size();
    const bool use_singleton_short_form = list_sz == 1 && block_codec_->supports_singleton_short_form();
    uint32_t singleton_stored_value = 0;
    uint32_t encoded_list_size = static_cast<uint32_t>(list_sz);
    if (use_singleton_short_form) {
        encoded_list_size = kSingletonShortFormSizeMarker;
        if constexpr (!kIsIPMetric) {
            singleton_stored_value = static_cast<uint32_t>(get_quant_val<DType, QType>(vals.front() - 1));
        }
    }

    size_t nr_blocks = (list_sz + block_sz - 1) / block_sz;
    out_buf.reserve(sizeof(size_t) + sizeof(int32_t) * nr_blocks + sizeof(uint32_t) * (nr_blocks - 1) + 64);

    varint_encode(encoded_list_size, out_buf);

    size_t begin_block_maxids = out_buf.size();
    size_t begin_block_endpoints = begin_block_maxids + sizeof(uint32_t) * nr_blocks;
    size_t begin_blocks = begin_block_endpoints + sizeof(uint32_t) * (nr_blocks - 1);
    out_buf.resize(begin_blocks);

    auto* ids_it = vec_ids.data();
    auto* vals_it = vals.data();

    std::vector<uint32_t> ids_buf(block_sz);
    std::vector<QType> ip_vals_buf(kIsIPMetric ? block_sz : 0);
    std::vector<uint32_t> bm25_vals_buf(kIsIPMetric ? 0 : block_sz);
    uint32_t last_vecid = UINT32_MAX;

    for (size_t b = 0; b < nr_blocks; ++b) {
        uint32_t cur_block_size = ((b + 1) * block_sz <= list_sz) ? block_sz : (list_sz % block_sz);

        for (size_t i = 0; i < cur_block_size; ++i) {
            uint32_t vecid(*ids_it++);
            ids_buf[i] = vecid - last_vecid - 1;
            last_vecid = vecid;
        }
        std::memcpy(out_buf.data() + begin_block_maxids + sizeof(uint32_t) * b, &last_vecid, sizeof(last_vecid));

        if (!use_singleton_short_form) {
            block_codec_->encode_doc_ids(ids_buf.data(), cur_block_size, out_buf);
        }

        if constexpr (kIsIPMetric) {
            for (size_t i = 0; i < cur_block_size; ++i) {
                ip_vals_buf[i] = *vals_it++;
            }
            out_buf.insert(out_buf.end(), reinterpret_cast<uint8_t*>(ip_vals_buf.data()),
                           reinterpret_cast<uint8_t*>(ip_vals_buf.data() + cur_block_size));
        } else if (use_singleton_short_form) {
            assert(cur_block_size == 1 && b == 0);
            varint_encode(singleton_stored_value, out_buf);
            ++vals_it;
        } else {
            for (size_t i = 0; i < cur_block_size; ++i) {
                bm25_vals_buf[i] = get_quant_val<DType, QType>(*vals_it++ - 1);
            }
            block_codec_->encode(bm25_vals_buf.data(), cur_block_size, out_buf);
        }

        if (b != nr_blocks - 1) {
            const uint32_t endpoint = static_cast<uint32_t>(out_buf.size() - begin_blocks);
            std::memcpy(out_buf.data() + begin_block_endpoints + sizeof(uint32_t) * b, &endpoint, sizeof(endpoint));
        }
    }
}

template <typename DType, typename QType, IndexScorerType MetricType>
void
BlockInvertedIndex<DType, QType, MetricType>::build_block_index(std::span<uint32_t>& raw_index_ids,
                                                                std::span<QType>& raw_index_vals,
                                                                std::span<size_t>& raw_index_offsets, bool enable_mmap,
                                                                const std::string& backed_filename) {
    // fill the postings container
    if (enable_mmap) {
        index_container_ = std::make_unique<FileBinaryContainer>(backed_filename + ".block_index");
    } else {
        index_container_ = std::make_unique<MemBinaryContainer>();
    }

    std::vector<std::vector<uint8_t>> encoded_posting_lists(this->nr_inner_dims_);
    parallel_for(this->nr_inner_dims_, [&](size_t i) {
        auto offset = raw_index_offsets[i];
        size_t count = raw_index_offsets[i + 1] - offset;
        encode_posting_list(encoded_posting_lists[i], raw_index_ids.subspan(offset, count),
                            raw_index_vals.subspan(offset, count));
    });

    std::vector<size_t> posting_offsets(this->nr_inner_dims_ + 1, 0);
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        posting_offsets[i + 1] = posting_offsets[i] + encoded_posting_lists[i].size();
    }

    // This is a workaround to streamvbyte decode having to sometimes look beyond the buffer due to some SIMD loads.
    constexpr size_t padding_size = 16;
    const auto offsets_byte_size = posting_offsets.size() * sizeof(size_t);
    index_container_->resize(offsets_byte_size + posting_offsets.back() + padding_size);
    index_container_->seal();

    auto data_ptr = index_container_->data();
    std::memcpy(data_ptr, posting_offsets.data(), offsets_byte_size);
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        std::memcpy(data_ptr + offsets_byte_size + posting_offsets[i], encoded_posting_lists[i].data(),
                    encoded_posting_lists[i].size());
    }
    std::memset(data_ptr + offsets_byte_size + posting_offsets.back(), 0, padding_size);

    posting_blocks_dim_offsets_ = std::span<size_t>(reinterpret_cast<size_t*>(data_ptr), this->nr_inner_dims_ + 1);
    posting_blocks_data_ =
        std::span<uint8_t>(data_ptr + offsets_byte_size, index_container_->size() - offsets_byte_size);
}

template <typename DType, typename QType, IndexScorerType MetricType>
Status
BlockInvertedIndex<DType, QType, MetricType>::add(const SparseRow<DType>* data, size_t rows, int64_t dim) {
    if (this->nr_rows_ != 0) {
        LOG_KNOWHERE_ERROR_ << "BlockInvertedIndex is already built, and cannot be added to again.";
        return Status::invalid_index_error;
    }

    this->nr_rows_ = rows;
    this->max_dim_ = dim;

    LOG_KNOWHERE_INFO_ << "BlockInvertedIndex build started: rows=" << rows << ", max_dim=" << dim
                       << ", metric=" << (kIsIPMetric ? "IP" : "BM25") << ", codec=" << block_codec_->get_name()
                       << ", codec_block_size=" << block_codec_->block_size();

    std::vector<float>* row_sums = nullptr;
    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        this->meta_data_.row_sums_.resize(this->nr_rows_);
        row_sums = &this->meta_data_.row_sums_;
    }

    std::vector<uint32_t>* dataset_nnz_stats = nullptr;
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    this->build_stats_.dataset_nnz_stats_.resize(rows);
    dataset_nnz_stats = &this->build_stats_.dataset_nnz_stats_;
#endif

    auto row_scan = scan_rows_for_build(data, rows, dataset_nnz_stats, row_sums);
    LOG_KNOWHERE_INFO_ << "BlockInvertedIndex row scan completed: external_dims=" << row_scan.external_dims.size();

    this->dim_map_.build_from_external_dims(row_scan.external_dims);
    this->nr_inner_dims_ = this->dim_map_.size();
    auto posting_plan =
        prepare_posting_build_plan(std::move(row_scan.posting_counts_by_worker), this->dim_map_, this->nr_inner_dims_);
    row_scan = {};
    const auto total_nnz = posting_plan.total_postings();

    size_t min_posting_count = total_nnz == 0 ? 0 : std::numeric_limits<size_t>::max();
    size_t max_posting_count = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        const auto posting_count = posting_plan.posting_count(i);
        min_posting_count = std::min(min_posting_count, posting_count);
        max_posting_count = std::max(max_posting_count, posting_count);
    }
    LOG_KNOWHERE_INFO_ << "BlockInvertedIndex posting plan completed: inner_dims=" << this->nr_inner_dims_
                       << ", total_postings=" << total_nnz << ", min_posting_length=" << min_posting_count
                       << ", max_posting_length=" << max_posting_count;

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    this->build_stats_.posting_list_length_stats_.resize(this->nr_inner_dims_);
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        this->build_stats_.posting_list_length_stats_[i] = posting_plan.posting_count(i);
    }
#endif

    // calculate raw index byte size
    auto raw_index_ids_byte_sz = total_nnz * sizeof(uint32_t);
    auto raw_index_vals_byte_sz = total_nnz * sizeof(QType);
    auto raw_index_offsets_byte_sz = (this->nr_inner_dims_ + 1) * sizeof(size_t);
    auto raw_index_byte_sz = raw_index_ids_byte_sz + raw_index_vals_byte_sz + raw_index_offsets_byte_sz;

    LOG_KNOWHERE_INFO_ << "BlockInvertedIndex allocating raw postings: bytes=" << raw_index_byte_sz
                       << ", ids_bytes=" << raw_index_ids_byte_sz << ", values_bytes=" << raw_index_vals_byte_sz
                       << ", offsets_bytes=" << raw_index_offsets_byte_sz;

    auto raw_index_container = std::make_unique<MemBinaryContainer>();

    raw_index_container->resize(raw_index_byte_sz);
    raw_index_container->seal();

    auto* buffer = raw_index_container->data();
    auto raw_index_ids = std::span<uint32_t>(reinterpret_cast<uint32_t*>(buffer), total_nnz);
    auto raw_index_vals = std::span<QType>(reinterpret_cast<QType*>(buffer + raw_index_ids_byte_sz), total_nnz);
    auto raw_index_offsets = std::span<size_t>(
        reinterpret_cast<size_t*>(buffer + raw_index_ids_byte_sz + raw_index_vals_byte_sz), this->nr_inner_dims_ + 1);

    std::copy(posting_plan.posting_offsets.begin(), posting_plan.posting_offsets.end(), raw_index_offsets.begin());
    fill_postings_by_worker(data, rows, this->dim_map_, raw_index_ids, raw_index_vals, posting_plan,
                            [](DType val) { return get_quant_val<DType, QType>(val); });
    posting_plan = {};
    LOG_KNOWHERE_INFO_ << "BlockInvertedIndex raw postings filled: total_postings=" << total_nnz;

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        this->meta_data_.resize_max_score_per_dim(this->nr_inner_dims_, 0.0f);

        parallel_for(this->nr_inner_dims_, [&](size_t i) {
            const auto posting_offset = raw_index_offsets[i];
            const auto posting_count = raw_index_offsets[i + 1] - posting_offset;
            const auto ids = raw_index_ids.subspan(posting_offset, posting_count);
            const auto vals = raw_index_vals.subspan(posting_offset, posting_count);
            float max_score = 0.0f;
            for (size_t j = 0; j < posting_count; ++j) {
                max_score = std::max(max_score, this->build_scorer_->vec_score(ids[j], vals[j]));
            }
            this->meta_data_.max_score_per_dim_[i] = max_score;
        });
        LOG_KNOWHERE_INFO_ << "BlockInvertedIndex max scores per dimension built: count=" << this->nr_inner_dims_;
    }
    // build block max data if the flag is set
    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        build_block_max_data(raw_index_ids, raw_index_vals, raw_index_offsets, false, "");
        LOG_KNOWHERE_INFO_ << "BlockInvertedIndex block max data built: blocks="
                           << this->meta_data_.block_max_data_.block_max_ids_.size()
                           << ", bytes=" << this->meta_data_.block_max_data_.container_->size();
    }
    // build block compressed index to postings_data_ and postings_endpoints_
    build_block_index(raw_index_ids, raw_index_vals, raw_index_offsets, false, "");
    const auto compressed_bytes = index_container_->size();
    const double compression_ratio =
        raw_index_byte_sz == 0 ? 0.0 : static_cast<double>(compressed_bytes) / static_cast<double>(raw_index_byte_sz);
    LOG_KNOWHERE_INFO_ << "BlockInvertedIndex block postings encoded: bytes=" << compressed_bytes
                       << ", compressed_to_raw_ratio=" << compression_ratio;
    LOG_KNOWHERE_INFO_ << "BlockInvertedIndex build completed: rows=" << rows << ", inner_dims=" << this->nr_inner_dims_
                       << ", total_postings=" << total_nnz << ", index_bytes=" << size();
    return Status::success;
}

template <typename DType, typename QType, IndexScorerType MetricType>
Status
BlockInvertedIndex<DType, QType, MetricType>::build_from_raw_data(MemoryIOReader& reader, bool enable_mmap,
                                                                  const std::string& backed_filename) {
    float deprecated_value_threshold = 0.0f;
    int64_t rows = 0;
    size_t cols = 0;

    // Previous versions used the sign of rows to indicate whether WAND metadata was present. The current format
    // records that in metadata, so ignore the legacy sign and use the absolute row count.
    readBinaryPOD(reader, rows);
    this->nr_rows_ = std::abs(rows);
    readBinaryPOD(reader, cols);
    this->max_dim_ = cols;
    readBinaryPOD(reader, deprecated_value_threshold);

    std::unique_ptr<BinaryContainer> raw_index_container;
    std::span<uint32_t> raw_index_ids;
    std::span<QType> raw_index_vals;
    std::span<size_t> raw_index_offsets;

    LOG_KNOWHERE_INFO_ << "Building raw index from raw data";
    // build raw index to raw_index_ids, raw_index_vals, raw_index_offsets and dim_map_
    build_raw_index(reader, raw_index_container, raw_index_ids, raw_index_vals, raw_index_offsets, enable_mmap,
                    backed_filename);

    LOG_KNOWHERE_INFO_ << "Building block max data";
    // build block max data if the flag is set
    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        build_block_max_data(raw_index_ids, raw_index_vals, raw_index_offsets, enable_mmap, backed_filename);
    }

    LOG_KNOWHERE_INFO_ << "Building block compressed index";
    // build block compressed index to postings_data_ and postings_endpoints_
    build_block_index(raw_index_ids, raw_index_vals, raw_index_offsets, enable_mmap, backed_filename);

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    this->index_size_gauge_->Set((double)size() / 1024.0 / 1024.0);
#endif

    return Status::success;
}

template <typename DType, typename QType, IndexScorerType MetricType>
Status
BlockInvertedIndex<DType, QType, MetricType>::serialize(MemoryIOWriter& writer) const {
    const uint32_t index_format_version = kInvertedIndexFileFormatVersion;
    auto index_encoding_type = [&]() -> uint32_t {
        if (this->block_codec_->get_name() == "block_streamvbyte") {
            return static_cast<uint32_t>(InvertedIndexEncoding::BLOCK_STREAMVBYTE);
        } else if (this->block_codec_->get_name() == "block_maskedvbyte") {
            return static_cast<uint32_t>(InvertedIndexEncoding::BLOCK_MASKEDVBYTE);
        } else if (this->block_codec_->get_name() == "block_adaptive") {
            return static_cast<uint32_t>(InvertedIndexEncoding::BLOCK_ADAPTIVE);
        } else {
            throw std::runtime_error("Unsupported index encoding type for BlockInvertedIndex");
        }
    }();

    writer.write(&index_format_version, sizeof(uint32_t));
    writer.write(&this->nr_rows_, sizeof(uint32_t));
    writer.write(&this->max_dim_, sizeof(uint32_t));
    writer.write(&this->nr_inner_dims_, sizeof(uint32_t));
    auto reserved = std::array<uint8_t, kInvertedIndexHeaderReservedBytes>();
    writer.write(reserved.data(), reserved.size());

    uint32_t nr_sections = 2;  // base sections: posting lists and dim map reverse
    constexpr auto dim_map_storage = DimMapMphfStorage::SeparateSection;
    nr_sections += ((this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) != 0) +
                   ((this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) != 0) +
                   ((this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) != 0) +
                   this->dim_map_.has_mphf_section(dim_map_storage);
    writer.write(&nr_sections, sizeof(uint32_t));

    std::vector<InvertedIndexSectionHeader> section_headers(nr_sections);

    uint64_t used_offset = first_section_offset(nr_sections);
    section_headers[0].type = InvertedIndexSectionType::POSTING_LISTS;
    section_headers[0].size = sizeof(InvertedIndexEncoding) + index_container_->size();
    assign_section_offset(section_headers[0], used_offset);

    section_headers[1].type = InvertedIndexSectionType::DIM_MAP_REVERSE;
    section_headers[1].size = this->dim_map_.reverse_section_size(dim_map_storage);
    assign_section_offset(section_headers[1], used_offset);

    auto curr_section_idx = 2;
    if (this->dim_map_.has_mphf_section(dim_map_storage)) {
        section_headers[curr_section_idx].type = InvertedIndexSectionType::DIM_MAP_MPHF;
        section_headers[curr_section_idx].size = this->dim_map_.mphf_section_size(dim_map_storage);
        assign_section_offset(section_headers[curr_section_idx], used_offset);
        curr_section_idx++;
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        section_headers[curr_section_idx].type = InvertedIndexSectionType::ROW_SUMS;
        section_headers[curr_section_idx].size = sizeof(float) * this->nr_rows_;
        assign_section_offset(section_headers[curr_section_idx], used_offset);
        curr_section_idx++;
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        section_headers[curr_section_idx].type = InvertedIndexSectionType::MAX_SCORES_PER_DIM;
        section_headers[curr_section_idx].size = sizeof(float) * this->nr_inner_dims_;
        assign_section_offset(section_headers[curr_section_idx], used_offset);
        curr_section_idx++;
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        section_headers[curr_section_idx].type = InvertedIndexSectionType::BLOCK_MAX_SCORES;
        section_headers[curr_section_idx].size =
            sizeof(size_t) + sizeof(uint32_t) + this->meta_data_.block_max_data_.container_->size();
        assign_section_offset(section_headers[curr_section_idx], used_offset);
        curr_section_idx++;
    }

    writer.write(section_headers.data(), sizeof(InvertedIndexSectionHeader), nr_sections);

    write_padding_until(writer, section_headers[0].offset);
    writer.write(&index_encoding_type, sizeof(uint32_t));
    writer.write(index_container_->data(), index_container_->size());

    write_padding_until(writer, section_headers[1].offset);
    this->dim_map_.write_reverse_section(writer, dim_map_storage);

    curr_section_idx = 2;
    if (this->dim_map_.has_mphf_section(dim_map_storage)) {
        write_padding_until(writer, section_headers[curr_section_idx].offset);
        this->dim_map_.write_mphf_section(writer, dim_map_storage);
        curr_section_idx++;
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_ROW_SUMS) {
        write_padding_until(writer, section_headers[curr_section_idx].offset);
        writer.write(this->meta_data_.row_sums_.data(), sizeof(float), this->nr_rows_);
        curr_section_idx++;
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        write_padding_until(writer, section_headers[curr_section_idx].offset);
        writer.write(this->meta_data_.max_score_per_dim_.data(), sizeof(float), this->nr_inner_dims_);
        curr_section_idx++;
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        write_padding_until(writer, section_headers[curr_section_idx].offset);
        size_t total_blocks = this->meta_data_.block_max_data_.block_max_ids_.size();
        writer.write(&total_blocks, sizeof(size_t));
        writer.write(&this->meta_data_.block_max_data_.block_size_, sizeof(uint32_t));
        writer.write(this->meta_data_.block_max_data_.container_->data(),
                     this->meta_data_.block_max_data_.container_->size());
        curr_section_idx++;
    }

    return Status::success;
}

template <typename DType, typename QType, IndexScorerType MetricType>
Status
BlockInvertedIndex<DType, QType, MetricType>::deserialize(MemoryIOReader& reader) {
    auto file_header_handler = [&]() {
        uint32_t index_format_version = 0;
        reader.read(&index_format_version, sizeof(uint32_t));
        // for now we only support version 1
        if (index_format_version != kInvertedIndexFileFormatVersion) {
            return Status::invalid_serialized_index_type;
        }

        reader.read(&this->nr_rows_, sizeof(uint32_t));
        reader.read(&this->max_dim_, sizeof(uint32_t));
        reader.read(&this->nr_inner_dims_, sizeof(uint32_t));
        // skip reserved bytes
        reader.advance(kInvertedIndexHeaderReservedBytes);

        return Status::success;
    };

    auto sections_handler = [&]() {
        uint32_t nr_sections = 0;
        reader.read(&nr_sections, sizeof(uint32_t));
        const auto section_headers = read_section_headers(reader, nr_sections);
        if (auto status = this->dim_map_.load_sections(reader, section_headers, this->nr_inner_dims_,
                                                       DimMapMphfStorage::SeparateSection);
            status != Status::success) {
            return status;
        }

        for (const auto& section_header : section_headers) {
            switch (section_header.type) {
                case InvertedIndexSectionType::POSTING_LISTS: {
                    reader.seekg(section_header.offset);
                    // check index encoding type
                    uint32_t index_encoding_type = 0;
                    reader.read(&index_encoding_type, sizeof(uint32_t));
                    if (index_encoding_type == static_cast<uint32_t>(InvertedIndexEncoding::FLAT)) {
                        LOG_KNOWHERE_ERROR_
                            << "BlockInvertedIndex cannot deserialize FLAT-encoded data. "
                            << "The index file was built with FLAT encoding but is being loaded as block-compressed.";
                        return Status::invalid_serialized_index_type;
                    }
                    if (index_encoding_type == static_cast<uint32_t>(InvertedIndexEncoding::BLOCK_STREAMVBYTE) &&
                        this->block_codec_->get_name() != "block_streamvbyte") {
                        return Status::invalid_serialized_index_type;
                    }
                    if (index_encoding_type == static_cast<uint32_t>(InvertedIndexEncoding::BLOCK_MASKEDVBYTE) &&
                        this->block_codec_->get_name() != "block_maskedvbyte") {
                        return Status::invalid_serialized_index_type;
                    }
                    if (index_encoding_type == static_cast<uint32_t>(InvertedIndexEncoding::BLOCK_ADAPTIVE) &&
                        this->block_codec_->get_name() != "block_adaptive") {
                        return Status::invalid_serialized_index_type;
                    }
                    // construct posting blocks dim offsets
                    this->posting_blocks_dim_offsets_ = std::span<size_t>(
                        reinterpret_cast<size_t*>(reader.data() + reader.tellg()), this->nr_inner_dims_ + 1);
                    reader.advance(sizeof(size_t) * (this->nr_inner_dims_ + 1));
                    // construct posting blocks data
                    size_t posting_blocks_data_size =
                        section_header.size - sizeof(uint32_t) - sizeof(size_t) * (this->nr_inner_dims_ + 1);
                    this->posting_blocks_data_ =
                        std::span<uint8_t>(reader.data() + reader.tellg(), posting_blocks_data_size);
                    // deserialize will use the memory from reader, so containers are not needed
                    // explicitly assign nullptr to them
                    this->index_container_ = nullptr;
                    break;
                }
                case InvertedIndexSectionType::DIM_MAP_REVERSE:
                case InvertedIndexSectionType::DIM_MAP_MPHF: {
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
                    const auto max_score_bytes = static_cast<uint64_t>(this->nr_inner_dims_) * sizeof(float);
                    if (section_header.size < max_score_bytes) {
                        LOG_KNOWHERE_ERROR_ << "Sparse inverted index MAX_SCORES_PER_DIM section is truncated, "
                                               "section_size="
                                            << section_header.size << ", expected_bytes=" << max_score_bytes;
                        return Status::invalid_serialized_index_type;
                    }
                    this->meta_data_.set_max_score_per_dim_view(
                        reinterpret_cast<float*>(reader.data() + reader.tellg()), this->nr_inner_dims_);
                    reader.advance(static_cast<size_t>(max_score_bytes));
                    break;
                }
                case InvertedIndexSectionType::BLOCK_MAX_SCORES: {
                    reader.seekg(section_header.offset);
                    size_t total_blocks = 0;
                    reader.read(&total_blocks, sizeof(size_t));
                    reader.read(&this->meta_data_.block_max_data_.block_size_, sizeof(uint32_t));
                    this->meta_data_.block_max_data_.block_offsets_ = std::span<size_t>(
                        reinterpret_cast<size_t*>(reader.data() + reader.tellg()), this->nr_inner_dims_);
                    reader.advance(this->nr_inner_dims_ * sizeof(size_t));
                    this->meta_data_.block_max_data_.block_max_ids_ =
                        std::span<uint32_t>(reinterpret_cast<uint32_t*>(reader.data() + reader.tellg()), total_blocks);
                    reader.advance(total_blocks * sizeof(uint32_t));
                    this->meta_data_.block_max_data_.block_max_scores_ =
                        std::span<float>(reinterpret_cast<float*>(reader.data() + reader.tellg()), total_blocks);
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
                    log_uint32_stats("BlockInvertedIndex", "dataset_nnz", dataset_nnz_stats);
                    log_uint32_stats("BlockInvertedIndex", "posting_list_length", posting_list_length_stats);
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
