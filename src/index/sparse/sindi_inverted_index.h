// Based on the SINDI algorithm for sparse vector search.
// Reference: https://arxiv.org/abs/2509.08395

#pragma once

#include <algorithm>
#include <boost/core/span.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <queue>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "index/sparse/aligned_allocator.h"
#include "index/sparse/inverted_index.h"
#include "index/sparse/scorer.h"
#include "index/sparse/sindi_simd.h"
#include "knowhere/bitsetview.h"
#include "knowhere/operands.h"
#include "simd/hook.h"

namespace knowhere::sparse::inverted {

/**
 * @brief Dynamic in-memory inverted index for sparse vectors that supports
 * incremental updates
 *
 * This index allows dynamically adding new vectors after construction. All data
 * is stored in memory.
 *
 * @tparam DType Type of the original vector values (e.g. float)
 */
template <typename DataType, typename QuantType>
class SindiInvertedIndex : public InvertedIndex<DataType> {
 public:
    static_assert(std::is_same_v<QuantType, knowhere::fp16> || std::is_same_v<QuantType, uint16_t>,
                  "QuantType must be fp16 (for IP) or uint16_t (for BM25)");

    static constexpr uint32_t min_window_size = 1024;
    static constexpr uint32_t max_window_size = 65535;

    SindiInvertedIndex(uint32_t window_size) : window_size_(std::clamp(window_size, min_window_size, max_window_size)) {
    }

    SindiInvertedIndex(const SindiInvertedIndex& rhs) = delete;
    SindiInvertedIndex(SindiInvertedIndex&& rhs) noexcept = default;
    SindiInvertedIndex&
    operator=(const SindiInvertedIndex& rhs) = delete;
    SindiInvertedIndex&
    operator=(SindiInvertedIndex&& rhs) noexcept = default;

    static constexpr uint64_t current_index_file_format_version_ = 1;

    [[nodiscard]] size_t
    size() const noexcept {
        size_t res = sizeof(*this);

        // Global posting lists
        res += plists_dim_offsets_span_.size() *
               sizeof(typename std::decay_t<decltype(plists_dim_offsets_span_)>::value_type);
        for (size_t dim_id = 0; dim_id < total_plists_ids_spans_.size(); ++dim_id) {
            res += total_plists_ids_spans_[dim_id].size() *
                   sizeof(typename std::decay_t<decltype(total_plists_ids_spans_[dim_id])>::value_type);
            res += total_plists_vals_spans_[dim_id].size() *
                   sizeof(typename std::decay_t<decltype(total_plists_vals_spans_[dim_id])>::value_type);
        }

        // Window sizes encoding (per-dim window nnz) and bitset of formats
        res += plists_wnnzs_fmts_msk_span_.size() *
               sizeof(typename std::decay_t<decltype(plists_wnnzs_fmts_msk_span_)>::value_type);
        res += plists_window_nnzs_spans_.size() *
               sizeof(typename std::decay_t<decltype(plists_window_nnzs_spans_)>::value_type);
        for (const auto& wspan : plists_window_nnzs_spans_) {
            res += wspan.size() * sizeof(typename std::decay_t<decltype(wspan)>::value_type);
        }

        res += this->nr_inner_dims_ * sizeof(typename decltype(this->dim_map_)::key_type) +
               sizeof(typename decltype(this->dim_map_)::mapped_type);

        // Row sums for BM25 support
        res += row_sums_span_.size() * sizeof(float);

        return res;
    }

    void
    append_window_indexes(const SparseRow<DataType>* data, size_t rows) {
        // Initialize global posting lists
        total_plists_ids_.resize(this->dim_map_.size());
        total_plists_vals_.resize(this->dim_map_.size());
        total_plists_ids_spans_.resize(this->dim_map_.size());
        total_plists_vals_spans_.resize(this->dim_map_.size());

        plists_dim_offsets_.resize(this->dim_map_.size() + 1, 0);
        plists_window_nnzs_.resize(this->dim_map_.size());
        plists_window_nnzs_spans_.resize(this->dim_map_.size());
        plists_wnnzs_fmts_msk_.resize((this->dim_map_.size() + 7) / 8);
        std::fill(plists_wnnzs_fmts_msk_.begin(), plists_wnnzs_fmts_msk_.end(), static_cast<uint8_t>(0));

        nr_windows_ = (this->nr_rows_ + rows + window_size_ - 1) / window_size_;
        window_index_plists_sz_.resize(nr_windows_);
        window_index_plists_sz_spans_.resize(nr_windows_);

        for (auto& wif : window_index_plists_sz_) {
            wif.resize(this->dim_map_.size());
        }

        std::vector<size_t> prev_offsets(this->dim_map_.size(), 0);
        std::vector<int32_t> cnt_nonempty_windows(this->dim_map_.size(), 0);

        for (size_t vecid = 0; vecid < rows; ++vecid) {
            for (size_t j = 0; j < data[vecid].size(); ++j) {
                auto [dim, val] = data[vecid][j];
                if (std::abs(val) < std::numeric_limits<DataType>::epsilon()) {
                    continue;
                }
                auto dim_id = this->dim_map_[dim];
                total_plists_ids_[dim_id].push_back(static_cast<uint16_t>((vecid + this->nr_rows_) % window_size_));
                total_plists_vals_[dim_id].push_back(static_cast<QuantType>(static_cast<float>(val)));
            }

            if ((this->nr_rows_ + vecid + 1) % window_size_ == 0) {
                for (size_t dim_id = 0; dim_id < this->dim_map_.size(); ++dim_id) {
                    auto delta_sz = total_plists_ids_[dim_id].size() - prev_offsets[dim_id];
                    window_index_plists_sz_[curr_widx_][dim_id] = static_cast<uint16_t>(delta_sz);
                    if (delta_sz > 0) {
                        cnt_nonempty_windows[dim_id] += 1;
                    }
                    prev_offsets[dim_id] = total_plists_ids_[dim_id].size();
                }
                ++curr_widx_;
                // When we've just closed the last existing window, curr_widx_ may point
                // beyond our current window count. It will become valid when a new window
                // is opened by subsequent add() calls.
            }
        }

        // If the last window is only partially filled, record its offsets as well,
        // so that window_index_plists_sz_ stores per-window nnz consistently.
        if ((this->nr_rows_ + rows) % window_size_ != 0 && curr_widx_ < nr_windows_) {
            for (size_t dim_id = 0; dim_id < this->dim_map_.size(); ++dim_id) {
                auto delta_sz = total_plists_ids_[dim_id].size() - prev_offsets[dim_id];
                window_index_plists_sz_[curr_widx_][dim_id] = static_cast<uint16_t>(delta_sz);
                if (delta_sz > 0) {
                    cnt_nonempty_windows[dim_id] += 1;
                }
                prev_offsets[dim_id] = total_plists_ids_[dim_id].size();
            }
        }

        // Update global posting list spans, and record posting list offsets for each dim
        uint32_t total_postings = 0;
        for (size_t dim_id = 0; dim_id < this->dim_map_.size(); ++dim_id) {
            plists_dim_offsets_[dim_id] = total_postings;
            total_plists_ids_spans_[dim_id] =
                boost::span<const uint16_t>(total_plists_ids_[dim_id].data(), total_plists_ids_[dim_id].size());
            total_plists_vals_spans_[dim_id] =
                boost::span<const QuantType>(total_plists_vals_[dim_id].data(), total_plists_vals_[dim_id].size());
            total_postings += total_plists_ids_[dim_id].size();
        }
        plists_dim_offsets_[this->dim_map_.size()] = total_postings;
        plists_dim_offsets_span_ = boost::span<const uint32_t>(plists_dim_offsets_.data(), plists_dim_offsets_.size());

        // Update window_index_plists_sz_spans_
        for (size_t wid = 0; wid < nr_windows_; ++wid) {
            window_index_plists_sz_spans_[wid] =
                boost::span<const uint16_t>(window_index_plists_sz_[wid].data(), window_index_plists_sz_[wid].size());
        }

        // Encode window nnzs with sparse/dense format selection
        // Sparse format: (wid, wnnz) pairs when cnt_nonempty * 4 < nr_windows * 2
        // Dense format: one uint16_t per window otherwise
        auto encode_dim_nnzs = [&](size_t dimid, int32_t cnt_nonempty) {
            plists_window_nnzs_[dimid].clear();
            auto* mskptr = plists_wnnzs_fmts_msk_.data() + (dimid >> 3);
            const bool use_sparse =
                static_cast<size_t>(cnt_nonempty) * sizeof(uint32_t) < nr_windows_ * sizeof(uint16_t);

            if (use_sparse) {
                // Sparse format: [wid | wnnz] packed into 32 bits
                *mskptr |= static_cast<uint8_t>(0x1u << (dimid & 0x7));
                plists_window_nnzs_[dimid].resize(static_cast<size_t>(cnt_nonempty) * sizeof(uint32_t));
                auto* pnnz_ptr = plists_window_nnzs_[dimid].data();
                const auto wnnz_bits = 32 - __builtin_clz(window_size_);
                for (size_t wid = 0; wid < nr_windows_; ++wid) {
                    auto wnnz = window_index_plists_sz_spans_[wid][dimid];
                    if (wnnz > 0) {
                        auto val = (static_cast<uint32_t>(wid) << wnnz_bits) | wnnz;
                        std::memcpy(pnnz_ptr, &val, sizeof(uint32_t));
                        pnnz_ptr += sizeof(uint32_t);
                    }
                }
            } else {
                // Dense format: one uint16_t per window
                *mskptr &= static_cast<uint8_t>(~(0x1u << (dimid & 0x7)));
                plists_window_nnzs_[dimid].resize(nr_windows_ * sizeof(uint16_t));
                auto* pnnz_ptr = plists_window_nnzs_[dimid].data();
                for (size_t wid = 0; wid < nr_windows_; ++wid) {
                    auto wnnz = window_index_plists_sz_spans_[wid][dimid];
                    std::memcpy(pnnz_ptr, &wnnz, sizeof(uint16_t));
                    pnnz_ptr += sizeof(uint16_t);
                }
            }
            plists_window_nnzs_spans_[dimid] =
                boost::span<const uint8_t>(plists_window_nnzs_[dimid].data(), plists_window_nnzs_[dimid].size());
        };

        for (size_t dimid = 0; dimid < this->dim_map_.size(); ++dimid) {
            encode_dim_nnzs(dimid, cnt_nonempty_windows[dimid]);
        }

        plists_wnnzs_fmts_msk_span_ =
            boost::span<const uint8_t>(plists_wnnzs_fmts_msk_.data(), plists_wnnzs_fmts_msk_.size());
    }

    Status
    add(const SparseRow<DataType>* data, size_t rows, int64_t dim) {
        this->max_dim_ = std::max(this->max_dim_, static_cast<uint32_t>(dim));
        // update dim_map_
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                auto [dim, val] = data[i][j];
                if (std::abs(val) < std::numeric_limits<DataType>::epsilon()) {
                    continue;
                }
                if (this->dim_map_.find(dim) == this->dim_map_.end()) {
                    this->dim_map_[dim] = this->dim_map_.size();
                }
            }
        }
        this->nr_inner_dims_ = this->dim_map_.size();

        // update window inverted indexes
        append_window_indexes(data, rows);

        this->nr_rows_ += rows;

        // Compute row sums for BM25 support (only for uint16_t QuantType)
        if constexpr (std::is_same_v<QuantType, uint16_t>) {
            // Reserve capacity for existing + new rows to avoid reallocations
            row_sums_.reserve(row_sums_.size() + rows);
            for (size_t i = 0; i < rows; ++i) {
                float row_sum = 0.0f;
                for (size_t j = 0; j < data[i].size(); ++j) {
                    const auto [dim, val] = data[i][j];
                    if (std::abs(val) < std::numeric_limits<DataType>::epsilon()) {
                        continue;
                    }
                    row_sum += static_cast<float>(val);
                }
                row_sums_.push_back(row_sum);
            }
            row_sums_span_ = boost::span<const float>(row_sums_.data(), row_sums_.size());
        }

        return Status::success;
    }

    [[nodiscard]] Status
    build_from_raw_data(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename) {
        return Status::not_implemented;
    }

    [[nodiscard]] Status
    serialize(MemoryIOWriter& writer) const {
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
        //    - window_size (uint32_t): Size of each window
        //    - nr_windows (uint32_t): Num of windows
        //    - plists_window_nnzs encoding mask (uint8[(nr_inner_dims+7)/8])
        //    - plists_window_nnzs per dim, concatenated
        //    - plists_dim_offsets per dim, uint32[nr_dims+1], the last one is the total size
        //    - total_plists_ids per dim, concatenated
        //    - total_plists_vals per dim, concatenated
        //
        // 4. Dimension Map Section:
        //    - dim_map_reverse[nr_inner_dims]: Array mapping internal dimension IDs to original dimensions
        //
        // 5. Optional Row Sums Section (for BM25 support, only when QuantType is uint16_t):
        //    - row_sums[nr_rows]: Array of row sums (float)
        const uint32_t index_format_version = current_index_file_format_version_;
        writer.write(&index_format_version, sizeof(uint32_t));
        writer.write(&this->nr_rows_, sizeof(uint32_t));
        writer.write(&this->max_dim_, sizeof(uint32_t));
        writer.write(&this->nr_inner_dims_, sizeof(uint32_t));
        auto reserved = std::array<uint8_t, 16>();
        writer.write(reserved.data(), reserved.size());

        // Determine if we need to write row sums (BM25 support)
        const bool has_row_sums = !row_sums_span_.empty();

        uint32_t nr_sections = 2;  // base sections: inverted index and dim map
        if (has_row_sums) {
            nr_sections += 1;  // add row sums section
        }
        writer.write(&nr_sections, sizeof(uint32_t));

        const size_t nr_dims = this->nr_inner_dims_;

        std::vector<InvertedIndexSectionHeader> section_headers(nr_sections);
        // 32 bytes header (16 bytes reserved + 16 bytes index_format_version, nr_rows, max_dim, nr_inner_dims)
        // + 4 bytes nr_sections + section headers table
        uint64_t used_offset = 32 + sizeof(uint32_t) + sizeof(InvertedIndexSectionHeader) * nr_sections;
        section_headers[0].type = InvertedIndexSectionType::POSTING_LISTS;
        section_headers[0].offset = used_offset;
        section_headers[0].size = [&, this]() -> uint64_t {
            // Layout:
            //   [encoding_type(uint32), window_size(uint32), nr_windows(uint32)]
            //   [plists_window_nnzs encoding mask (uint8[(nr_inner_dims+7)/8])]
            //   [plists_window_nnzs per dim, concatenated]
            //   [plists_dim_offsets per dim, uint32[nr_dims+1]], the last one is the total size
            //   [total_plists_ids per dim, concatenated]
            //   [total_plists_vals per dim, concatenated]
            size_t res = sizeof(uint32_t) * 3;  // header

            if (nr_dims == 0 || nr_windows_ == 0) {
                return res;
            }

            // plists windows sizes mask
            const size_t mask_sz = (nr_dims + 7) / 8;
            res += mask_sz * sizeof(uint8_t);

            // Per-dimension window nnzs and derive total postings per dimension
            for (size_t dimid = 0; dimid < nr_dims; ++dimid) {
                const auto& nnz_span = plists_window_nnzs_spans_[dimid];
                // span size header (32 bits) + payload (span.size() bytes)
                res += sizeof(uint32_t);
                res += nnz_span.size();
            }

            // plists_dim_offsets
            res += sizeof(uint32_t) * (nr_dims + 1);

            // Docids and QuantType values
            auto total_postings = plists_dim_offsets_span_[nr_dims];
            res += total_postings * sizeof(uint16_t);   // ids
            res += total_postings * sizeof(QuantType);  // vals

            return res;
        }();
        used_offset += section_headers[0].size;

        section_headers[1].type = InvertedIndexSectionType::DIM_MAP;
        section_headers[1].offset = used_offset;
        section_headers[1].size = sizeof(uint32_t) * this->nr_inner_dims_;
        used_offset += section_headers[1].size;

        // Add row sums section header if needed (BM25 support)
        if (has_row_sums) {
            section_headers[2].type = InvertedIndexSectionType::ROW_SUMS;
            section_headers[2].offset = used_offset;
            section_headers[2].size = sizeof(float) * this->nr_rows_;
            used_offset += section_headers[2].size;
        }

        writer.write(section_headers.data(), sizeof(InvertedIndexSectionHeader), nr_sections);

        uint32_t index_encoding_type = static_cast<uint32_t>(InvertedIndexEncoding::FIXED_DOCID_WINDOWS);
        writer.write(&index_encoding_type, sizeof(uint32_t));
        writer.write(&this->window_size_, sizeof(uint32_t));
        writer.write(&this->nr_windows_, sizeof(uint32_t));

        // write plists_woffsets_formats_mask and plists_window_nnzs
        writer.write(plists_wnnzs_fmts_msk_span_.data(), sizeof(uint8_t), plists_wnnzs_fmts_msk_span_.size());
        for (size_t dimid = 0; dimid < nr_dims; ++dimid) {
            const auto& span = plists_window_nnzs_spans_[dimid];
            uint32_t span_sz = static_cast<uint32_t>(span.size());
            writer.write(&span_sz, sizeof(uint32_t));
            writer.write(span.data(), sizeof(uint8_t), span.size());
        }

        // write plists_dim_offsets
        writer.write(plists_dim_offsets_span_.data(), sizeof(uint32_t), plists_dim_offsets_span_.size());

        // write total_plists_ids / vals per dim, concatenated
        if (nr_windows_ > 0 && nr_dims > 0) {
            // ids
            for (size_t dim_id = 0; dim_id < nr_dims; ++dim_id) {
                uint32_t len = plists_dim_offsets_span_[dim_id + 1] - plists_dim_offsets_span_[dim_id];
                if (!total_plists_ids_.empty()) {
                    writer.write(total_plists_ids_[dim_id].data(), sizeof(uint16_t), len);
                } else if (!total_plists_ids_spans_.empty()) {
                    writer.write(total_plists_ids_spans_[dim_id].data(), sizeof(uint16_t), len);
                }
            }
            // vals
            for (size_t dim_id = 0; dim_id < nr_dims; ++dim_id) {
                uint32_t len = plists_dim_offsets_span_[dim_id + 1] - plists_dim_offsets_span_[dim_id];
                if (!total_plists_vals_.empty()) {
                    writer.write(total_plists_vals_[dim_id].data(), sizeof(QuantType), len);
                } else if (!total_plists_vals_spans_.empty()) {
                    writer.write(total_plists_vals_spans_[dim_id].data(), sizeof(QuantType), len);
                }
            }
        }

        // write dim map
        auto dim_map_reverse = std::vector<uint32_t>(this->nr_inner_dims_);
        for (const auto& [dim, dim_id] : this->dim_map_) {
            dim_map_reverse[dim_id] = dim;
        }
        writer.write(dim_map_reverse.data(), sizeof(uint32_t), this->nr_inner_dims_);

        // write row sums if present (BM25 support)
        if (has_row_sums) {
            writer.write(row_sums_span_.data(), sizeof(float), row_sums_span_.size());
        }

        return Status::success;
    }

    [[nodiscard]] Status
    deserialize(MemoryIOReader& reader) {
        auto file_header_handler = [&, this]() {
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

            // if there are zero rows, there should be no inner dims, something is wrong
            if (this->nr_rows_ == 0 && this->nr_inner_dims_ != 0) {
                return Status::invalid_serialized_index_type;
            }

            return Status::success;
        };

        auto sections_handler = [&, this]() {
            uint32_t nr_sections = 0;
            reader.read(&nr_sections, sizeof(uint32_t));
            // Allow 3 sections (base) or 4 sections (with row sums for BM25)
            if (nr_sections < 3 || nr_sections > 4) {
                return Status::invalid_serialized_index_type;
            }
            size_t sec_table_offset = reader.tellg();

            for (uint32_t i = 0; i < nr_sections; ++i) {
                InvertedIndexSectionHeader section_header;
                reader.seekg(sec_table_offset);
                reader.read(&section_header, sizeof(InvertedIndexSectionHeader));
                sec_table_offset += sizeof(InvertedIndexSectionHeader);

                // Log high-level section info
                LOG_KNOWHERE_INFO_ << "SindiInvertedIndex::deserialize section[" << i
                                   << "] type=" << static_cast<uint32_t>(section_header.type)
                                   << " offset=" << section_header.offset << " size_bytes=" << section_header.size;

                switch (section_header.type) {
                    case InvertedIndexSectionType::POSTING_LISTS: {
                        reader.seekg(section_header.offset);
                        // check index encoding type
                        uint32_t index_encoding_type = 0;
                        reader.read(&index_encoding_type, sizeof(uint32_t));
                        if (index_encoding_type != static_cast<uint32_t>(InvertedIndexEncoding::FIXED_DOCID_WINDOWS)) {
                            return Status::invalid_serialized_index_type;
                        }

                        // check window params
                        reader.read(&this->window_size_, sizeof(uint32_t));
                        reader.read(&this->nr_windows_, sizeof(uint32_t));

                        if (this->window_size_ == 0 || this->window_size_ >= 65536) {
                            LOG_KNOWHERE_INFO_ << "SindiInvertedIndex::deserialize invalid window_size_="
                                               << this->window_size_;
                            return Status::invalid_serialized_index_type;
                        }
                        uint32_t expected_windows =
                            (this->nr_rows_ == 0)
                                ? 0u
                                : static_cast<uint32_t>((this->nr_rows_ + this->window_size_ - 1) / this->window_size_);
                        if (this->nr_windows_ != expected_windows) {
                            LOG_KNOWHERE_INFO_ << "SindiInvertedIndex::deserialize nr_windows_=" << this->nr_windows_
                                               << " != expected_windows=" << expected_windows;
                            return Status::invalid_serialized_index_type;
                        }

                        const size_t nr_dims = this->nr_inner_dims_;
                        const uint64_t bytes_header = static_cast<uint64_t>(sizeof(uint32_t) * 3);

                        window_index_plists_sz_.clear();
                        window_index_plists_sz_spans_.clear();
                        total_plists_ids_.clear();
                        total_plists_vals_.clear();
                        total_plists_ids_spans_.assign(nr_dims, {});
                        total_plists_vals_spans_.assign(nr_dims, {});
                        plists_window_nnzs_.clear();
                        plists_window_nnzs_spans_.assign(nr_dims, {});
                        plists_wnnzs_fmts_msk_.clear();

                        // plists window sizes: encoding mask
                        const size_t mask_sz = (nr_dims + 7) / 8;
                        uint64_t bytes_mask = 0;
                        if (mask_sz > 0) {
                            const uint8_t* mask_base = reinterpret_cast<const uint8_t*>(reader.data() + reader.tellg());
                            plists_wnnzs_fmts_msk_span_ = boost::span<const uint8_t>(mask_base, mask_sz);
                            reader.advance(mask_sz);
                            bytes_mask = static_cast<uint64_t>(mask_sz) * sizeof(uint8_t);
                        } else {
                            plists_wnnzs_fmts_msk_span_ = {};
                        }

                        // Per-dimension window nnzs
                        uint64_t bytes_win_nnzs = 0;
                        for (size_t dimid = 0; dimid < nr_dims; ++dimid) {
                            uint32_t span_sz = 0;
                            reader.read(&span_sz, sizeof(uint32_t));
                            const uint8_t* dbase = reinterpret_cast<const uint8_t*>(reader.data() + reader.tellg());
                            plists_window_nnzs_spans_[dimid] = boost::span<const uint8_t>(dbase, span_sz);
                            reader.advance(static_cast<size_t>(span_sz) * sizeof(uint8_t));
                            bytes_win_nnzs += sizeof(uint32_t) + static_cast<size_t>(span_sz) * sizeof(uint8_t);
                        }

                        // plists dim offsets
                        plists_dim_offsets_span_ = boost::span<const uint32_t>(
                            reinterpret_cast<const uint32_t*>(reader.data() + reader.tellg()), nr_dims + 1);
                        reader.advance((nr_dims + 1) * sizeof(uint32_t));
                        auto total_postings = plists_dim_offsets_span_[nr_dims];

                        // ids region (per-dim contiguous, concatenated)
                        const uint16_t* ids_region = reinterpret_cast<const uint16_t*>(reader.data() + reader.tellg());
                        const uint64_t bytes_ids = static_cast<uint64_t>(total_postings) * sizeof(uint16_t);
                        size_t base = 0;
                        for (size_t dim_id = 0; dim_id < nr_dims; ++dim_id) {
                            uint32_t len = plists_dim_offsets_span_[dim_id + 1] - plists_dim_offsets_span_[dim_id];
                            total_plists_ids_spans_[dim_id] = boost::span<const uint16_t>(ids_region + base, len);
                            base += len;
                        }
                        reader.advance(total_postings * sizeof(uint16_t));

                        // vals region (per-dim contiguous, concatenated)
                        const QuantType* vals_region =
                            reinterpret_cast<const QuantType*>(reader.data() + reader.tellg());
                        base = 0;
                        const uint64_t bytes_vals = static_cast<uint64_t>(total_postings) * sizeof(QuantType);
                        for (size_t dim_id = 0; dim_id < nr_dims; ++dim_id) {
                            uint32_t len = plists_dim_offsets_span_[dim_id + 1] - plists_dim_offsets_span_[dim_id];
                            total_plists_vals_spans_[dim_id] = boost::span<const QuantType>(vals_region + base, len);
                            base += len;
                        }
                        reader.advance(total_postings * sizeof(QuantType));

                        // Log breakdown for POSTING_LISTS section
                        LOG_KNOWHERE_DEBUG_ << "SindiInvertedIndex::deserialize POSTING_LISTS breakdown: "
                                            << " header_bytes=" << bytes_header << " mask_bytes=" << bytes_mask
                                            << " window_nnzs_bytes=" << bytes_win_nnzs
                                            << " dim_offsets_bytes=" << (nr_dims + 1) * sizeof(uint32_t)
                                            << " ids_bytes=" << bytes_ids << " vals_bytes=" << bytes_vals
                                            << " total_section_bytes=" << section_header.size;

                        break;
                    }
                    case InvertedIndexSectionType::DIM_MAP: {
                        reader.seekg(section_header.offset);
                        uint32_t dim = 0;
                        std::unordered_set<uint32_t> seen_dims;
                        seen_dims.reserve(this->nr_inner_dims_);
                        for (uint32_t i = 0; i < this->nr_inner_dims_; ++i) {
                            reader.read(&dim, sizeof(uint32_t));
                            // validate dim id range
                            if (dim > this->max_dim_) {
                                return Status::invalid_serialized_index_type;
                            }
                            // validate uniqueness
                            auto [_, inserted] = seen_dims.insert(dim);
                            if (!inserted) {
                                return Status::invalid_serialized_index_type;
                            }
                            this->dim_map_[dim] = i;
                        }

                        // Log breakdown for DIM_MAP section (single contiguous array)
                        const uint64_t dim_map_bytes = static_cast<uint64_t>(this->nr_inner_dims_) * sizeof(uint32_t);
                        LOG_KNOWHERE_DEBUG_ << "SindiInvertedIndex::deserialize DIM_MAP breakdown: "
                                            << "dim_map_bytes=" << dim_map_bytes
                                            << " total_section_bytes=" << section_header.size;
                        break;
                    }
                    case InvertedIndexSectionType::ROW_SUMS: {
                        // Row sums section for BM25
                        reader.seekg(section_header.offset);
                        row_sums_span_ = boost::span<const float>(
                            reinterpret_cast<const float*>(reader.data() + section_header.offset), this->nr_rows_);
                        reader.advance(sizeof(float) * this->nr_rows_);

                        // Log breakdown for ROW_SUMS section
                        const uint64_t row_sums_bytes = static_cast<uint64_t>(this->nr_rows_) * sizeof(float);
                        LOG_KNOWHERE_DEBUG_ << "SindiInvertedIndex::deserialize ROW_SUMS breakdown: "
                                            << "row_sums_bytes=" << row_sums_bytes
                                            << " total_section_bytes=" << section_header.size;
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

        return Status::success;
    }

    void
    search(const SparseRow<DataType>& query, size_t k, float* distances, label_t* labels, const BitsetView& bitset,
           const InvertedIndexSearchParams& search_params) const {
        std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
        std::fill(labels, labels + k, -1);

        if (query.size() == 0) {
            return;
        }

        auto q_vec = this->parse_query(query, search_params.approx.drop_ratio_search);
        if (q_vec.empty()) {
            return;
        }

        knowhere::ResultMinHeap<float, uint32_t> topk_q(k);
        std::vector<float> wscores_final_vec(window_size_, 0.0f);
        float* wscores_final = wscores_final_vec.data();

        float threshold = 0.0f;

        const uint32_t wnnz_bits = 32 - __builtin_clz(window_size_);
        const uint32_t wnnz_mask = (1u << wnnz_bits) - 1;

        // Initialize posting list cursors for each query term
        // Cursors track position in posting lists and handle window-by-window iteration
        std::vector<PostingCursor> cursors;
        cursors.reserve(q_vec.size());
        for (auto& [qid, qval] : q_vec) {
            const uint8_t* wnnz_buf = nullptr;
            size_t wnnz_buf_sz = 0;
            if (!plists_window_nnzs_spans_.empty() && qid < plists_window_nnzs_spans_.size()) {
                const auto& wnnz_buf_span = plists_window_nnzs_spans_[qid];
                wnnz_buf = wnnz_buf_span.data();
                wnnz_buf_sz = wnnz_buf_span.size();
            }

            bool is_sparse = !plists_wnnzs_fmts_msk_span_.empty() &&
                             ((plists_wnnzs_fmts_msk_span_[qid >> 3] & static_cast<uint8_t>(0x1u << (qid & 0x7))) != 0);

            cursors.emplace_back(static_cast<uint32_t>(qid),            // dim_id
                                 qval,                                  // qval
                                 total_plists_ids_spans_[qid].data(),   // ids_base
                                 total_plists_vals_spans_[qid].data(),  // vals_base
                                 wnnz_buf,                              // wnnz_buf
                                 wnnz_buf_sz,                           // wnnz_buf_sz
                                 is_sparse,                             // is_sparse
                                 0,                                     // cursor
                                 0,                                     // offset
                                 wnnz_bits,                             // wnnz_bits
                                 wnnz_mask                              // wnnz_mask
            );
        }

        // Main search loop: iterate over windows and process each window
        if constexpr (std::is_same_v<QuantType, knowhere::fp16>) {
            const auto scatter_fn = sindi::get_ip_kernels().accumulate;
            const auto batch_insert_fn = sindi::get_ip_kernels().batch_insert;

            for (size_t widx = 0; widx < nr_windows_; ++widx) {
                const size_t docid_start = window_size_ * widx;
                const uint32_t curr_window_size =
                    std::min(window_size_, static_cast<uint32_t>(this->nr_rows_ - docid_start));
                std::fill_n(wscores_final, curr_window_size, 0);

                for (auto& cur : cursors) {
                    if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                        continue;
                    }

                    auto [soff, wnnz] = cur.advance_window(widx);
                    if (wnnz == 0) {
                        continue;
                    }

                    const uint16_t* plist_ids = cur.ids_base + soff;
                    const QuantType* plist_vals = cur.vals_base + soff;

                    scatter_fn(cur.qval, plist_vals, plist_ids, wnnz, wscores_final);
                }

                batch_insert_fn(wscores_final, docid_start, curr_window_size, topk_q, threshold, bitset);
            }
        } else {
            const auto accumulate_fn = sindi::get_bm25_kernels().accumulate;
            const auto batch_insert_fn = sindi::get_bm25_kernels().batch_insert;

            const float bm25_k1 = search_params.scorer_config.scorer_params.bm25.k1;
            const float bm25_b = search_params.scorer_config.scorer_params.bm25.b;
            const float bm25_avgdl = search_params.scorer_config.scorer_params.bm25.avgdl;
            const float* row_sums_ptr = row_sums_span_.data();

            for (size_t widx = 0; widx < nr_windows_; ++widx) {
                const size_t docid_start = window_size_ * widx;
                const uint32_t curr_window_size =
                    std::min(window_size_, static_cast<uint32_t>(this->nr_rows_ - docid_start));
                std::fill_n(wscores_final, curr_window_size, 0);

                for (auto& cur : cursors) {
                    if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                        continue;
                    }

                    auto [soff, wnnz] = cur.advance_window(widx);
                    if (wnnz == 0) {
                        continue;
                    }

                    const uint16_t* plist_ids = cur.ids_base + soff;
                    const QuantType* plist_vals = cur.vals_base + soff;

                    accumulate_fn(cur.qval, plist_vals, plist_ids, wnnz, wscores_final, bm25_k1, bm25_b, bm25_avgdl,
                                  row_sums_ptr + docid_start);
                }

                batch_insert_fn(wscores_final, docid_start, curr_window_size, topk_q, threshold, bitset);
            }
        }

        topk_q.Finalize();
        const auto& topk_vec = topk_q.Results();
        for (size_t i = 0; i < topk_vec.size(); ++i) {
            auto [score, vid] = topk_vec[i];
            distances[i] = score;
            labels[i] = vid;
        }
    }

    [[nodiscard]] std::vector<float>
    get_all_distances(const SparseRow<DataType>& query, const BitsetView& bitset,
                      const InvertedIndexSearchParams& search_params) const override {
        if (query.size() == 0) {
            return {};
        }

        auto q_vec = this->parse_query(query, search_params.approx.drop_ratio_search);
        if (q_vec.empty()) {
            return {};
        }

        const uint32_t wnnz_bits = 32 - __builtin_clz(window_size_);
        const uint32_t wnnz_mask = (1u << wnnz_bits) - 1;

        // Initialize posting list cursors for each query term
        std::vector<PostingCursor> cursors;
        cursors.reserve(q_vec.size());
        for (auto& [qid, qval] : q_vec) {
            const uint8_t* wnnz_buf = nullptr;
            size_t wnnz_buf_sz = 0;
            if (!plists_window_nnzs_spans_.empty() && qid < plists_window_nnzs_spans_.size()) {
                const auto& nnz_span = plists_window_nnzs_spans_[qid];
                wnnz_buf = nnz_span.data();
                wnnz_buf_sz = nnz_span.size();
            }

            bool is_sparse = !plists_wnnzs_fmts_msk_span_.empty() &&
                             ((plists_wnnzs_fmts_msk_span_[qid >> 3] & static_cast<uint8_t>(0x1u << (qid & 0x7))) != 0);

            cursors.emplace_back(static_cast<uint32_t>(qid),            // dim_id
                                 qval,                                  // qval
                                 total_plists_ids_spans_[qid].data(),   // ids_base
                                 total_plists_vals_spans_[qid].data(),  // vals_base
                                 wnnz_buf,                              // wnnz_buf
                                 wnnz_buf_sz,                           // wnnz_buf_sz
                                 is_sparse,                             // is_sparse
                                 0,                                     // cursor
                                 0,                                     // offset
                                 wnnz_bits,                             // wnnz_bits
                                 wnnz_mask                              // wnnz_mask
            );
        }

        std::vector<float> distances(this->nr_rows_, 0.0f);

        // Cache SIMD kernel function pointer and process windows
        // Use compile-time dispatch to select appropriate kernel
        if constexpr (std::is_same_v<QuantType, knowhere::fp16>) {
            // IP scoring path
            const auto scatter_fn = sindi::get_ip_kernels().accumulate;

            for (size_t widx = 0; widx < nr_windows_; ++widx) {
                const size_t docid_start = window_size_ * widx;
                float* wscores_final = distances.data() + docid_start;

                for (auto& cur : cursors) {
                    if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                        continue;
                    }

                    auto [soff, wnnz] = cur.advance_window(widx);
                    if (wnnz == 0) {
                        continue;
                    }

                    const uint16_t* plist_ids = cur.ids_base + soff;
                    const QuantType* plist_vals = cur.vals_base + soff;

                    scatter_fn(cur.qval, plist_vals, plist_ids, wnnz, wscores_final);
                }
            }
        } else {
            // BM25 scoring path
            const auto scatter_fn = sindi::get_bm25_kernels().accumulate;

            // Extract BM25 parameters
            const float bm25_k1 = search_params.scorer_config.scorer_params.bm25.k1;
            const float bm25_b = search_params.scorer_config.scorer_params.bm25.b;
            const float bm25_avgdl = search_params.scorer_config.scorer_params.bm25.avgdl;
            const float* row_sums_ptr = row_sums_span_.data();

            for (size_t widx = 0; widx < nr_windows_; ++widx) {
                const size_t docid_start = window_size_ * widx;
                float* wscores_final = distances.data() + docid_start;

                for (auto& cur : cursors) {
                    if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                        continue;
                    }

                    auto [soff, wnnz] = cur.advance_window(widx);
                    if (wnnz == 0) {
                        continue;
                    }

                    const uint16_t* plist_ids = cur.ids_base + soff;
                    const QuantType* plist_vals = cur.vals_base + soff;

                    scatter_fn(cur.qval, plist_vals, plist_ids, wnnz, wscores_final, bm25_k1, bm25_b, bm25_avgdl,
                               row_sums_ptr + docid_start);
                }
            }
        }

        // Apply bitset filter: zero out scores for filtered documents
        if (!bitset.empty()) {
            for (size_t i = 0; i < distances.size(); ++i) {
                if (bitset.test(static_cast<int64_t>(i))) {
                    distances[i] = 0.0f;
                }
            }
        }

        return distances;
    }

    /**
     * @brief Get the row sums span for BM25 scoring
     *
     * @return boost::span<const float> The row sums span (empty if not BM25 index)
     */
    [[nodiscard]] boost::span<const float>
    get_row_sums_span() const noexcept {
        return row_sums_span_;
    }

    [[nodiscard]] Status
    convert_to_raw_data(MemoryIOWriter& writer) const {
        return Status::not_implemented;
    }

 private:
    // Global posting lists (per dimension, concatenated across windows)
    using aligned_u16_vec = std::vector<uint16_t, aligned_allocator<uint16_t, 64>>;
    using aligned_quant_vec = std::vector<QuantType, aligned_allocator<QuantType, 64>>;
    std::vector<aligned_u16_vec> total_plists_ids_;
    std::vector<aligned_quant_vec> total_plists_vals_;
    std::vector<boost::span<const uint16_t>> total_plists_ids_spans_;
    std::vector<boost::span<const QuantType>> total_plists_vals_spans_;

    // Window sizes encoding (per-dim window nnz) and bitset of formats
    // Bit=1 means sparse format (wid, wnnz pairs), Bit=0 means dense format (one entry per window)
    std::vector<uint8_t> plists_wnnzs_fmts_msk_;
    boost::span<const uint8_t> plists_wnnzs_fmts_msk_span_;
    std::vector<std::vector<uint16_t>> window_index_plists_sz_;
    std::vector<boost::span<const uint16_t>> window_index_plists_sz_spans_;
    std::vector<std::vector<uint8_t>> plists_window_nnzs_;
    std::vector<boost::span<const uint8_t>> plists_window_nnzs_spans_;
    std::vector<uint32_t> plists_dim_offsets_;
    boost::span<const uint32_t> plists_dim_offsets_span_;

    // Row sums only needed for BM25
    std::vector<float> row_sums_;
    boost::span<const float> row_sums_span_;

    uint32_t window_size_{max_window_size};
    uint32_t nr_windows_{0};
    uint32_t curr_widx_{0};

    // Cursor for iterating over a dimension's posting list by windows
    struct PostingCursor {
        uint32_t dim_id{0};
        float qval{0.0f};
        const uint16_t* ids_base{nullptr};
        const QuantType* vals_base{nullptr};
        const uint8_t* wnnz_buf{nullptr};
        size_t wnnz_buf_sz{0};
        bool is_sparse{false};
        size_t cursor{0};       // for sparse format: byte offset into (wid, wnnz) pairs
        uint32_t offset{0};     // global postings offset for this dim
        uint32_t wnnz_bits{0};  // bit width for wnnz encoding
        uint32_t wnnz_mask{0};  // mask for extracting wnnz

        PostingCursor() = default;
        PostingCursor(uint32_t dim_id, float qval, const uint16_t* ids_base, const QuantType* vals_base,
                      const uint8_t* wnnz_buf, size_t wnnz_buf_sz, bool is_sparse, size_t cursor, uint32_t offset,
                      uint32_t wnnz_bits, uint32_t wnnz_mask)
            : dim_id(dim_id),
              qval(qval),
              ids_base(ids_base),
              vals_base(vals_base),
              wnnz_buf(wnnz_buf),
              wnnz_buf_sz(wnnz_buf_sz),
              is_sparse(is_sparse),
              cursor(cursor),
              offset(offset),
              wnnz_bits(wnnz_bits),
              wnnz_mask(wnnz_mask) {
        }

        // advance to next window, returns {start_offset, wnnz} for current window
        inline std::pair<uint32_t, uint16_t>
        advance_window(size_t widx) {
            uint32_t soff = offset;
            uint16_t wnnz = 0;
            if (is_sparse) {
                if (cursor + sizeof(uint32_t) <= wnnz_buf_sz) {
                    auto wval = *reinterpret_cast<const uint32_t*>(wnnz_buf + cursor);
                    auto wid = wval >> wnnz_bits;
                    if (wid == widx) {
                        wnnz = static_cast<uint16_t>(wval & wnnz_mask);
                        offset += wnnz;
                        cursor += sizeof(uint32_t);
                    }
                }
            } else {
                wnnz = *reinterpret_cast<const uint16_t*>(wnnz_buf + widx * sizeof(uint16_t));
                offset += wnnz;
            }
            return {soff, wnnz};
        }
    };
};

using SindiInvertedIndexIP = SindiInvertedIndex<float, knowhere::fp16>;
using SindiInvertedIndexBM25 = SindiInvertedIndex<float, uint16_t>;

}  // namespace knowhere::sparse::inverted
