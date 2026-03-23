#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <boost/core/span.hpp>
#include <cstring>

#include "index/sparse/codec/block_codec.h"
#include "index/sparse/inverted_index.h"
#include "index/sparse/scorer.h"
#include "knowhere/log.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"

namespace knowhere::sparse::inverted {

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
    BlockInvertedIndexCursor(const BlockCodecPtr& block_codec, std::uint8_t const* data, std::uint32_t universe,
                             BitsetView bitset)
        : base_(varint_decode(data, &n_, 1)),
          nr_blocks_((n_ + block_codec->block_size() - 1) / block_codec->block_size()),
          block_maxids_(base_),
          block_offsets_(block_maxids_ + sizeof(int32_t) * nr_blocks_),
          blocks_data_(block_offsets_ + sizeof(uint32_t) * (nr_blocks_ - 1)),
          universe_(universe),
          block_codec_(block_codec),
          block_size_(block_codec->block_size()),
          bitset_(bitset) {
        ids_buf_.resize(block_size_);
        vals_buf_.resize(block_size_);
        reset();
    }

    void
    reset() {
        decode_vecids_block(0);
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
    }

    void
    next() {
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
        if (lower_bound > cur_block_maxid_) [[unlikely]] {
            if (lower_bound > block_maxid(nr_blocks_ - 1)) {
                cur_vec_id_ = universe_;
                return;
            }
            uint32_t block = cur_block_ + 1;
            while (block_maxid(block) < lower_bound) {
                ++block;
            }
            decode_vecids_block(block);
        }

        while (cur_vec_id_ < lower_bound) {
            cur_vec_id_ += ids_buf_[++pos_in_block_] + 1;
            assert(pos_in_block_ < cur_block_size_);
        }

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

        // now only uint32_t is compression supported
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
        return ((uint32_t const*)block_maxids_)[blk_idx];
    }

    void
    skip_filtered_ids() {
        while (!bitset_.empty() && cur_vec_id_ < universe_ && bitset_.test(cur_vec_id_)) {
            next_raw();
        }
    }

    void
    decode_vecids_block(uint32_t blkid) {
        uint32_t endpoint = blkid != 0U ? ((uint32_t const*)block_offsets_)[blkid - 1] : 0;
        uint8_t const* block_data = blocks_data_ + endpoint;
        cur_block_size_ = ((blkid + 1) * block_size_ <= n_) ? block_size_ : (n_ % block_size_);
        uint32_t cur_base = (blkid != 0U ? block_maxid(blkid - 1) : uint32_t(-1)) + 1;
        cur_block_maxid_ = block_maxid(blkid);
        vals_block_data_ = block_codec_->decode(block_data, ids_buf_.data(), cur_block_size_);
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(vals_block_data_, 0, 3);
#endif

        ids_buf_[0] += cur_base;

        cur_block_ = blkid;
        pos_in_block_ = 0;
        cur_vec_id_ = ids_buf_[0];
        vals_decoded_ = false;
    }

    void
    decode_vals_block() {
        if constexpr (std::is_same_v<VType, uint32_t>) {
            uint8_t const* next_block = block_codec_->decode(vals_block_data_, vals_buf_.data(), cur_block_size_);
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
    BitsetView bitset_;
};

template <typename DType, typename QType, IndexScorerType MetricType>
class BlockInvertedIndex : public CRTPInvertedIndex<BlockInvertedIndex<DType, QType, MetricType>, DType> {
 public:
    // IP metric: values stored as raw bytes in blocks.
    // BM25 metric: values stored with block codec compression as uint32_t.
    static constexpr bool kIsIPMetric = MetricType == IndexScorerType::IP;
    using posting_list_iterator = BlockInvertedIndexCursor<std::conditional_t<kIsIPMetric, QType, uint32_t>>;

    static constexpr uint64_t current_index_file_format_version_ = 1;

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

        res += this->nr_inner_dims_ * (sizeof(typename decltype(this->dim_map_)::key_type) +
                                       sizeof(typename decltype(this->dim_map_)::mapped_type));

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
        auto endpoint = this->posting_blocks_dim_offsets_[dim_id];
        auto* data = this->posting_blocks_data_.data() + endpoint;
        return posting_list_iterator(this->block_codec_, data, this->nr_rows_, bitset);
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
    add_row_to_index(const SparseRow<DType>& raw_row, uint32_t vec_id, boost::span<uint32_t>& raw_index_ids,
                     boost::span<QType>& raw_index_vals, std::vector<size_t>& curr_offsets) {
        float row_sum = 0.0f;

        for (size_t j = 0; j < raw_row.size(); ++j) {
            auto [dim, val] = raw_row[j];
            // Skip values equals to or close enough to zero (which is little to the total IP score).
            if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
                continue;
            }

            row_sum += val;

            auto dim_it = this->dim_map_.find(dim);
            if (dim_it == this->dim_map_.cend()) {
                throw std::runtime_error("unexpected vector dimension in BlockInvertedIndex");
            }

            auto offset = curr_offsets[dim_it->second]++;
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
     * This function builds the raw index from the serialized data.
     * It first reads the number of rows and the maximum dimension seen.
     * Then it calculates the memory requirements for the index structures.
     * If memory mapping is enabled, it creates a memory mapped file to store the data.
     * Otherwise, it uses heap memory.
     *
     * The memory is allocated for:
     * - Inverted lists storing vector IDs (inverted_index_ids_)
     * - Inverted lists storing values (inverted_index_vals_)
     */
    void
    build_raw_index(MemoryIOReader& reader, std::unique_ptr<BinaryContainer>& raw_index_container,
                    boost::span<uint32_t>& raw_index_ids, boost::span<QType>& raw_index_vals,
                    boost::span<size_t>& raw_index_offsets, bool enable_mmap, const std::string& backed_filename);

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
    build_block_max_data(boost::span<uint32_t> raw_index_ids, boost::span<QType> raw_index_vals,
                         boost::span<size_t> raw_index_offsets, bool enable_mmap, const std::string& backed_filename);

    /**
     * @brief Encode the posting list into a binary format
     *
     * @param out_buf Output buffer to store the encoded posting list
     * @param vec_ids Inverted lists storing vector IDs
     * @param vals Inverted lists storing quantized values
     */
    void
    encode_posting_list(std::vector<uint8_t>& out_buf, boost::span<uint32_t> vec_ids, boost::span<QType> vals);

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
    build_block_index(boost::span<uint32_t>& raw_index_ids, boost::span<QType>& raw_index_vals,
                      boost::span<size_t>& raw_index_offsets, bool enable_mmap, const std::string& backed_filename);

    std::unique_ptr<BinaryContainer> index_container_;

    // Inverted lists start offsets
    // Each dimension's inverted list is stored contiguously in a flattened array
    // The start offset of each dimension's list is stored in posting_blocks_dim_offsets_
    boost::span<size_t> posting_blocks_dim_offsets_;

    // Inverted lists storing all vector blocks
    boost::span<uint8_t> posting_blocks_data_;

    // Block codec
    BlockCodecPtr block_codec_;
};

template <typename DType, typename QType, IndexScorerType MetricType>
void
BlockInvertedIndex<DType, QType, MetricType>::build_raw_index(MemoryIOReader& reader,
                                                              std::unique_ptr<BinaryContainer>& raw_index_container,
                                                              boost::span<uint32_t>& raw_index_ids,
                                                              boost::span<QType>& raw_index_vals,
                                                              boost::span<size_t>& raw_index_offsets, bool enable_mmap,
                                                              const std::string& backed_filename) {
    const auto saved_reader_loc = reader.tellg();
    const auto nnz = (reader.remaining() - (this->nr_rows_ * sizeof(size_t))) / SparseRow<DType>::element_size();

    std::unordered_map<uint32_t, size_t> plist_cnts;
    for (uint32_t i = 0; i < this->nr_rows_; ++i) {
        size_t count = 0;
        readBinaryPOD(reader, count);
        if (count == 0) {
            continue;
        }
        for (size_t j = 0; j < count; ++j) {
            uint32_t dim = 0;
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
    auto raw_index_byte_sz = raw_index_ids_byte_sz + raw_index_vals_byte_sz + raw_index_offsets_byte_sz;

    if (enable_mmap) {
        raw_index_container = std::make_unique<FileBinaryContainer>(backed_filename + ".raw_index");
    } else {
        raw_index_container = std::make_unique<MemBinaryContainer>();
    }

    raw_index_container->resize(raw_index_byte_sz);
    raw_index_container->seal();

    auto* data = raw_index_container->data();
    raw_index_ids = boost::span<uint32_t>(reinterpret_cast<uint32_t*>(data), nnz);
    raw_index_vals = boost::span<QType>(reinterpret_cast<QType*>(data + raw_index_ids_byte_sz), nnz);
    raw_index_offsets = boost::span<size_t>(
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
        this->meta_data_.max_score_per_dim_.resize(this->nr_inner_dims_, 0.0f);

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
BlockInvertedIndex<DType, QType, MetricType>::build_block_max_data(boost::span<uint32_t> raw_index_ids,
                                                                   boost::span<QType> raw_index_vals,
                                                                   boost::span<size_t> raw_index_offsets,
                                                                   bool enable_mmap,
                                                                   const std::string& backed_filename) {
    if (enable_mmap) {
        this->meta_data_.block_max_data_.container_ =
            std::make_unique<FileBinaryContainer>(backed_filename + ".block_max_data");
    } else {
        this->meta_data_.block_max_data_.container_ = std::make_unique<MemBinaryContainer>();
    }

    size_t total_blocks = 0;
    for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
        size_t count = raw_index_offsets[i + 1] - raw_index_offsets[i];
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
        auto offset = raw_index_offsets[i];
        size_t count = raw_index_offsets[i + 1] - offset;
        auto ids = raw_index_ids.subspan(offset, count);
        auto vals = raw_index_vals.subspan(offset, count);

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

template <typename DType, typename QType, IndexScorerType MetricType>
void
BlockInvertedIndex<DType, QType, MetricType>::encode_posting_list(std::vector<uint8_t>& out_buf,
                                                                  boost::span<uint32_t> vec_ids,
                                                                  boost::span<QType> vals) {
    // Posting list layout:
    // +----------------+------------------------------------------+
    // | list_sz       | uint32_t: total number of postings       |
    // +----------------+------------------------------------------+
    // | block_maxids  | int32_t[nr_blocks]: max vector id/block  |
    // +----------------+------------------------------------------+
    // | block_ends    | uint32_t[nr_blocks-1]: block end offsets |
    // +----------------+------------------------------------------+
    // | blocks        | uint8_t[]: encoded posting data          |
    // +----------------+------------------------------------------+
    // Note: First block end offset is omitted (always 0)
    size_t list_sz = vec_ids.size();
    varint_encode(list_sz, out_buf);

    uint32_t block_sz = block_codec_->block_size();
    size_t nr_blocks = (list_sz + block_sz - 1) / block_sz;
    size_t begin_block_maxids = out_buf.size();
    size_t begin_block_endpoints = begin_block_maxids + sizeof(int32_t) * nr_blocks;
    size_t begin_blocks = begin_block_endpoints + sizeof(uint32_t) * (nr_blocks - 1);
    out_buf.resize(begin_blocks);

    auto* ids_it = vec_ids.begin();
    auto* vals_it = vals.begin();

    std::vector<uint32_t> ids_buf(block_sz);
    int32_t last_vecid(-1);
    for (size_t b = 0; b < nr_blocks; ++b) {
        uint32_t cur_block_size = ((b + 1) * block_sz <= list_sz) ? block_sz : (list_sz % block_sz);

        for (size_t i = 0; i < cur_block_size; ++i) {
            uint32_t vecid(*ids_it++);
            ids_buf[i] = vecid - last_vecid - 1;
            last_vecid = vecid;
        }
        std::memcpy(out_buf.data() + begin_block_maxids + sizeof(int32_t) * b, &last_vecid, sizeof(last_vecid));

        block_codec_->encode(ids_buf.data(), cur_block_size, out_buf);

        if constexpr (kIsIPMetric) {
            std::vector<QType> vals_buf(cur_block_size);
            for (size_t i = 0; i < cur_block_size; ++i) {
                vals_buf[i] = *vals_it++;
            }
            out_buf.insert(out_buf.end(), reinterpret_cast<uint8_t*>(vals_buf.data()),
                           reinterpret_cast<uint8_t*>(vals_buf.data() + cur_block_size));
        } else {
            std::vector<uint32_t> vals_buf(cur_block_size);
            for (size_t i = 0; i < cur_block_size; ++i) {
                vals_buf[i] = get_quant_val<DType, QType>(*vals_it++ - 1);
            }
            block_codec_->encode(vals_buf.data(), cur_block_size, out_buf);
        }

        if (b != nr_blocks - 1) {
            uint32_t endpoint = out_buf.size() - begin_blocks;
            std::memcpy(out_buf.data() + begin_block_endpoints + sizeof(uint32_t) * b, &endpoint, sizeof(endpoint));
        }
    }
}

template <typename DType, typename QType, IndexScorerType MetricType>
void
BlockInvertedIndex<DType, QType, MetricType>::build_block_index(boost::span<uint32_t>& raw_index_ids,
                                                                boost::span<QType>& raw_index_vals,
                                                                boost::span<size_t>& raw_index_offsets,
                                                                bool enable_mmap, const std::string& backed_filename) {
    // fill the postings container
    if (enable_mmap) {
        index_container_ = std::make_unique<FileBinaryContainer>(backed_filename + ".block_index");
    } else {
        index_container_ = std::make_unique<MemBinaryContainer>();
    }

    // write the endpoints of each dimension's inverted list
    index_container_->resize((this->nr_inner_dims_ + 1) * sizeof(size_t));
    size_t endpoint = 0;

    index_container_->write_at(0, reinterpret_cast<uint8_t*>(&endpoint), sizeof(size_t));

    for (uint32_t i = 0; i < this->nr_inner_dims_; ++i) {
        auto offset = raw_index_offsets[i];
        size_t count = raw_index_offsets[i + 1] - offset;
        std::vector<uint8_t> out_buf;
        encode_posting_list(out_buf, raw_index_ids.subspan(offset, count), raw_index_vals.subspan(offset, count));
        index_container_->append(out_buf.data(), out_buf.size());
        endpoint += out_buf.size();
        index_container_->write_at((i + 1) * sizeof(size_t), reinterpret_cast<uint8_t*>(&endpoint), sizeof(size_t));
    }

    // This is a workaround to QMX codex having to sometimes look beyond the buffer due to some SIMD loads.
    std::array<uint8_t, 15> padding{};
    index_container_->append(padding.data(), padding.size());

    index_container_->seal();

    auto data_ptr = index_container_->data();
    posting_blocks_dim_offsets_ = boost::span<size_t>(reinterpret_cast<size_t*>(data_ptr), this->nr_inner_dims_ + 1);
    posting_blocks_data_ = boost::span<uint8_t>(data_ptr + sizeof(size_t) * (this->nr_inner_dims_ + 1),
                                                index_container_->size() - sizeof(size_t) * (this->nr_inner_dims_ + 1));
}

template <typename DType, typename QType, IndexScorerType MetricType>
Status
BlockInvertedIndex<DType, QType, MetricType>::add(const SparseRow<DType>* data, size_t rows, int64_t dim) {
    std::unordered_map<uint32_t, size_t> plist_cnts;
    size_t total_nnz = 0;

    if (this->nr_rows_ != 0) {
        LOG_KNOWHERE_ERROR_ << "BlockInvertedIndex is already built, and cannot be added to again.";
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
    auto raw_index_ids_byte_sz = total_nnz * sizeof(uint32_t);
    auto raw_index_vals_byte_sz = total_nnz * sizeof(QType);
    auto raw_index_offsets_byte_sz = (this->nr_inner_dims_ + 1) * sizeof(size_t);
    auto raw_index_byte_sz = raw_index_ids_byte_sz + raw_index_vals_byte_sz + raw_index_offsets_byte_sz;

    auto raw_index_container = std::make_unique<MemBinaryContainer>();

    raw_index_container->resize(raw_index_byte_sz);
    raw_index_container->seal();

    auto* buffer = raw_index_container->data();
    auto raw_index_ids = boost::span<uint32_t>(reinterpret_cast<uint32_t*>(buffer), total_nnz);
    auto raw_index_vals = boost::span<QType>(reinterpret_cast<QType*>(buffer + raw_index_ids_byte_sz), total_nnz);
    auto raw_index_offsets = boost::span<size_t>(
        reinterpret_cast<size_t*>(buffer + raw_index_ids_byte_sz + raw_index_vals_byte_sz), this->nr_inner_dims_ + 1);

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
        add_row_to_index(data[i], i, raw_index_ids, raw_index_vals, curr_offsets);
    }

    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_MAX_SCORES_PER_DIM) {
        this->meta_data_.max_score_per_dim_.resize(this->nr_inner_dims_, 0.0f);

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

    // build block max data if the flag is set
    if (this->meta_data_.flags_ & InvertedIndexMetaData::FLAG_HAS_BLOCK_MAX_SCORES) {
        build_block_max_data(raw_index_ids, raw_index_vals, raw_index_offsets, false, "");
    }

    // build block compressed index to postings_data_ and postings_endpoints_
    build_block_index(raw_index_ids, raw_index_vals, raw_index_offsets, false, "");
    return Status::success;
}

template <typename DType, typename QType, IndexScorerType MetricType>
Status
BlockInvertedIndex<DType, QType, MetricType>::build_from_raw_data(MemoryIOReader& reader, bool enable_mmap,
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

    std::unique_ptr<BinaryContainer> raw_index_container;
    boost::span<uint32_t> raw_index_ids;
    boost::span<QType> raw_index_vals;
    boost::span<size_t> raw_index_offsets;

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
    //      - offset (uint64_t): Offset of the section from the beginning of the file
    //      - size (uint64_t): Size of the section in bytes
    //
    // 3. Posting Lists Section:
    //    - index_encoding_type (uint32_t): Type of encoding used
    //    - encoded_index_data: Block-encoded posting lists
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
    auto index_encoding_type = [&]() -> uint32_t {
        if (this->block_codec_->get_name() == "block_streamvbyte") {
            return static_cast<uint32_t>(InvertedIndexEncoding::BLOCK_STREAMVBYTE);
        } else if (this->block_codec_->get_name() == "block_maskedvbyte") {
            return static_cast<uint32_t>(InvertedIndexEncoding::BLOCK_MASKEDVBYTE);
        } else {
            throw std::runtime_error("Unsupported index encoding type for BlockInvertedIndex");
        }
    }();

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
    writer.write(&nr_sections, sizeof(uint32_t));

    // since writer doesn't support seekp() for now, calculate all sizes of each sections first
    InvertedIndexSectionHeader section_headers[nr_sections];
    uint64_t used_offset = sizeof(InvertedIndexSectionHeader) * nr_sections + 36;
    section_headers[0].type = InvertedIndexSectionType::POSTING_LISTS;
    section_headers[0].offset = used_offset;
    section_headers[0].size = sizeof(InvertedIndexEncoding) + index_container_->size();
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

    // write section headers table
    writer.write(section_headers, sizeof(InvertedIndexSectionHeader), nr_sections);

    // write index encoding type and encoded index
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

    return Status::success;
}

template <typename DType, typename QType, IndexScorerType MetricType>
Status
BlockInvertedIndex<DType, QType, MetricType>::deserialize(MemoryIOReader& reader) {
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
                    // construct posting blocks dim offsets
                    this->posting_blocks_dim_offsets_ = boost::span<size_t>(
                        reinterpret_cast<size_t*>(reader.data() + reader.tellg()), this->nr_inner_dims_ + 1);
                    reader.advance(sizeof(size_t) * (this->nr_inner_dims_ + 1));
                    // construct posting blocks data
                    size_t posting_blocks_data_size =
                        section_header.size - sizeof(uint32_t) - sizeof(size_t) * (this->nr_inner_dims_ + 1);
                    this->posting_blocks_data_ =
                        boost::span<uint8_t>(reader.data() + reader.tellg(), posting_blocks_data_size);
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
