// Based on the SINDI algorithm for sparse vector search.
// Reference: https://arxiv.org/abs/2509.08395

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <queue>
#include <span>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "index/sparse/aligned_allocator.h"
#include "index/sparse/inverted_index.h"
#include "index/sparse/inverted_index_format.h"
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
template <typename DataType, typename QuantType, bool AllowIncremental = false>
class SindiInvertedIndex : public DimMapInvertedIndex<DataType, AllowIncremental> {
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

    [[nodiscard]] size_t
    size() const noexcept override {
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

        res += this->dim_map_.byte_size();

        // Row sums for BM25 support
        res += row_sums_span_.size() * sizeof(float);

        return res;
    }

    void
    set_legacy_dim_map_mphf_trailer_workaround(bool enabled) {
        legacy_dim_map_mphf_trailer_workaround_ = enabled;
    }

    void
    append_window_indexes(const SparseRow<DataType>* data, size_t rows) {
        const size_t dim_count = this->nr_inner_dims_;
        total_plists_ids_.resize(dim_count);
        total_plists_vals_.resize(dim_count);
        total_plists_ids_spans_.resize(dim_count);
        total_plists_vals_spans_.resize(dim_count);

        plists_dim_offsets_.resize(dim_count + 1, 0);
        plists_window_nnzs_.resize(dim_count);
        plists_window_nnzs_spans_.resize(dim_count);
        plists_wnnzs_fmts_msk_.resize((dim_count + 7) / 8);
        std::fill(plists_wnnzs_fmts_msk_.begin(), plists_wnnzs_fmts_msk_.end(), static_cast<uint8_t>(0));

        nr_windows_ = (this->nr_rows_ + rows + window_size_ - 1) / window_size_;
        window_index_plists_sz_.resize(nr_windows_);
        window_index_plists_sz_spans_.resize(nr_windows_);

        for (auto& wif : window_index_plists_sz_) {
            wif.resize(dim_count);
        }

        for (size_t vecid = 0; vecid < rows; ++vecid) {
            const uint32_t global_vecid = static_cast<uint32_t>(this->nr_rows_ + vecid);
            const uint32_t widx = global_vecid / window_size_;
            const uint16_t local_id = static_cast<uint16_t>(global_vecid % window_size_);
            for (size_t j = 0; j < data[vecid].size(); ++j) {
                auto [dim, val] = data[vecid][j];
                if (std::abs(val) < std::numeric_limits<DataType>::epsilon()) {
                    continue;
                }
                auto inner_dim = this->dim_map_.lookup(dim);
                if (!inner_dim.has_value()) {
                    throw std::runtime_error("unexpected vector dimension in SindiInvertedIndex");
                }
                auto dim_id = inner_dim.value();
                total_plists_ids_[dim_id].push_back(local_id);
                total_plists_vals_[dim_id].push_back(static_cast<QuantType>(static_cast<float>(val)));
                ++window_index_plists_sz_[widx][dim_id];
            }
        }

        // Update global posting list spans, and record posting list offsets for each dim
        uint32_t total_postings = 0;
        for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
            plists_dim_offsets_[dim_id] = total_postings;
            total_plists_ids_spans_[dim_id] =
                std::span<const uint16_t>(total_plists_ids_[dim_id].data(), total_plists_ids_[dim_id].size());
            total_plists_vals_spans_[dim_id] =
                std::span<const QuantType>(total_plists_vals_[dim_id].data(), total_plists_vals_[dim_id].size());
            total_postings += total_plists_ids_[dim_id].size();
        }
        plists_dim_offsets_[dim_count] = total_postings;
        plists_dim_offsets_span_ = std::span<const uint32_t>(plists_dim_offsets_.data(), plists_dim_offsets_.size());

        // Update window_index_plists_sz_spans_
        for (size_t wid = 0; wid < nr_windows_; ++wid) {
            window_index_plists_sz_spans_[wid] =
                std::span<const uint16_t>(window_index_plists_sz_[wid].data(), window_index_plists_sz_[wid].size());
        }

        // Encode window nnzs with sparse/dense format selection
        // Sparse format: (wid, wnnz) pairs when cnt_nonempty * 4 < nr_windows * 2
        // Dense format: one uint16_t per window otherwise
        auto encode_dim_nnzs = [&](size_t dimid) {
            plists_window_nnzs_[dimid].clear();
            auto* mskptr = plists_wnnzs_fmts_msk_.data() + (dimid >> 3);
            int32_t cnt_nonempty = 0;
            for (size_t wid = 0; wid < nr_windows_; ++wid) {
                if (window_index_plists_sz_spans_[wid][dimid] > 0) {
                    ++cnt_nonempty;
                }
            }
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
                std::span<const uint8_t>(plists_window_nnzs_[dimid].data(), plists_window_nnzs_[dimid].size());
        };

        for (size_t dimid = 0; dimid < dim_count; ++dimid) {
            encode_dim_nnzs(dimid);
        }

        plists_wnnzs_fmts_msk_span_ =
            std::span<const uint8_t>(plists_wnnzs_fmts_msk_.data(), plists_wnnzs_fmts_msk_.size());
    }

    Status
    add(const SparseRow<DataType>* data, size_t rows, int64_t dim) override {
        if constexpr (!AllowIncremental) {
            if (this->nr_rows_ != 0) {
                return Status::not_implemented;
            }
        }

        const size_t old_nr_rows = this->nr_rows_;
        this->max_dim_ = std::max(this->max_dim_, static_cast<uint32_t>(dim));

        if constexpr (AllowIncremental) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < data[i].size(); ++j) {
                    const auto [dim, val] = data[i][j];
                    if (std::abs(val) < std::numeric_limits<DataType>::epsilon()) {
                        continue;
                    }
                    this->dim_map_.append_legacy_entry(dim);
                }
            }
            this->nr_inner_dims_ = this->dim_map_.size();
        } else {
            std::unordered_set<uint32_t> external_dims;
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < data[i].size(); ++j) {
                    const auto [dim, val] = data[i][j];
                    if (std::abs(val) < std::numeric_limits<DataType>::epsilon()) {
                        continue;
                    }
                    external_dims.insert(dim);
                }
            }
            this->dim_map_.build_from_external_dims(external_dims);
            this->nr_inner_dims_ = this->dim_map_.size();
        }

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
                    const auto abs_val = std::abs(val);
                    if (abs_val < std::numeric_limits<DataType>::epsilon()) {
                        continue;
                    }
                    row_sum += static_cast<float>(abs_val);
                }
                row_sums_.push_back(row_sum);
            }
            row_sums_span_ = std::span<const float>(row_sums_.data(), row_sums_.size());
        }

        // Incrementally update max score per dimension for early termination optimization.
        // For IP: max_score = max(abs(val)) across quantized postings.
        // For BM25: max_score = max((k1+1)*tf / (tf + k1*(1-b+b*dl/avgdl))) across postings.
        max_scores_per_dim_.resize(this->nr_inner_dims_);
        if constexpr (std::is_same_v<QuantType, knowhere::fp16>) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < data[i].size(); ++j) {
                    const auto [dim, val] = data[i][j];
                    const auto abs_val = std::abs(val);
                    if (abs_val < std::numeric_limits<DataType>::epsilon()) {
                        continue;
                    }
                    auto inner_dim = this->dim_map_.lookup(dim);
                    if (!inner_dim.has_value()) {
                        continue;
                    }
                    const auto dim_id = inner_dim.value();
                    if (abs_val > max_scores_per_dim_[dim_id]) {
                        max_scores_per_dim_[dim_id] = abs_val;
                    }
                }
            }
        } else {
            // BM25 scoring: compute exact max BM25 score per dimension
            // For each posting, compute: (k1+1)*tf / (tf + k1*(1-b+b*dl/avgdl))
            // using the actual document length (dl) from row_sums
            const auto& cfg = this->build_scorer_->config();
            const float k1 = cfg.scorer_params.bm25.k1;
            const float b = cfg.scorer_params.bm25.b;
            const float avgdl = cfg.scorer_params.bm25.avgdl;
            const float p1 = k1 + 1.0f;
            const float p2 = k1 * (1.0f - b);
            const float p3 = k1 * b / avgdl;

            for (size_t i = 0; i < rows; ++i) {
                const size_t global_docid = old_nr_rows + i;
                const float dl = row_sums_span_[global_docid];
                for (size_t j = 0; j < data[i].size(); ++j) {
                    const auto [dim, val] = data[i][j];
                    const auto abs_val = std::abs(val);
                    if (abs_val < std::numeric_limits<DataType>::epsilon()) {
                        continue;
                    }
                    auto inner_dim = this->dim_map_.lookup(dim);
                    if (!inner_dim.has_value()) {
                        continue;
                    }
                    const auto dim_id = inner_dim.value();
                    const float tf = static_cast<float>(abs_val);
                    const float bm25_score = p1 * tf / (tf + p2 + p3 * dl);
                    if (bm25_score > max_scores_per_dim_[dim_id]) {
                        max_scores_per_dim_[dim_id] = bm25_score;
                    }
                }
            }
        }
        max_scores_per_dim_span_ = std::span<const float>(max_scores_per_dim_.data(), max_scores_per_dim_.size());

        return Status::success;
    }

    [[nodiscard]] Status
    build_from_raw_data(MemoryIOReader& reader, bool enable_mmap, const std::string& backed_filename) override {
        return Status::not_implemented;
    }

    [[nodiscard]] Status
    serialize(MemoryIOWriter& writer) const override {
        if constexpr (AllowIncremental) {
            LOG_KNOWHERE_ERROR_ << "SindiInvertedIndex incremental mode does not support serialize";
            return Status::not_implemented;
        }

        if constexpr (!AllowIncremental) {
            const uint32_t index_format_version = kInvertedIndexFileFormatVersion;
            writer.write(&index_format_version, sizeof(uint32_t));
            writer.write(&this->nr_rows_, sizeof(uint32_t));
            writer.write(&this->max_dim_, sizeof(uint32_t));
            writer.write(&this->nr_inner_dims_, sizeof(uint32_t));
            auto reserved = std::array<uint8_t, kInvertedIndexHeaderReservedBytes>();
            writer.write(reserved.data(), reserved.size());

            const bool has_row_sums = !row_sums_span_.empty();
            const auto dim_map_storage = legacy_dim_map_mphf_trailer_workaround_ ? DimMapMphfStorage::LegacyTrailer
                                                                                 : DimMapMphfStorage::SeparateSection;

            uint32_t nr_sections = 3;  // base sections: inverted index, dim map and max scores per dim
            if (this->dim_map_.has_mphf_section(dim_map_storage)) {
                nr_sections += 1;
            }
            if (has_row_sums) {
                nr_sections += 1;
            }
            writer.write(&nr_sections, sizeof(uint32_t));

            const size_t nr_dims = this->nr_inner_dims_;

            std::vector<InvertedIndexSectionHeader> section_headers(nr_sections);
            uint64_t used_offset = first_section_offset(nr_sections);
            section_headers[0].type = InvertedIndexSectionType::POSTING_LISTS;
            section_headers[0].size = [&, this]() -> uint64_t {
                size_t res = sizeof(uint32_t) * 3;

                const size_t mask_sz = (nr_dims + 7) / 8;
                res += mask_sz * sizeof(uint8_t);

                for (size_t dimid = 0; dimid < nr_dims; ++dimid) {
                    const auto& nnz_span = plists_window_nnzs_spans_[dimid];
                    res += sizeof(uint32_t);
                    res += nnz_span.size();
                }

                res += sizeof(uint32_t) * (nr_dims + 1);

                if (nr_windows_ == 0) {
                    return res;
                }

                auto total_postings = plists_dim_offsets_span_[nr_dims];
                res += total_postings * sizeof(uint16_t);
                res += total_postings * sizeof(QuantType);

                return res;
            }();
            assign_section_offset(section_headers[0], used_offset);

            section_headers[1].type = InvertedIndexSectionType::DIM_MAP_REVERSE;
            section_headers[1].size = this->dim_map_.reverse_section_size(dim_map_storage);
            assign_section_offset(section_headers[1], used_offset);

            size_t curr_section_idx = 2;
            if (this->dim_map_.has_mphf_section(dim_map_storage)) {
                section_headers[curr_section_idx].type = InvertedIndexSectionType::DIM_MAP_MPHF;
                section_headers[curr_section_idx].size = this->dim_map_.mphf_section_size(dim_map_storage);
                assign_section_offset(section_headers[curr_section_idx], used_offset);
                curr_section_idx++;
            }

            section_headers[curr_section_idx].type = InvertedIndexSectionType::MAX_SCORES_PER_DIM;
            section_headers[curr_section_idx].size = sizeof(float) * this->nr_inner_dims_;
            assign_section_offset(section_headers[curr_section_idx], used_offset);
            curr_section_idx++;

            if (has_row_sums) {
                section_headers[curr_section_idx].type = InvertedIndexSectionType::ROW_SUMS;
                section_headers[curr_section_idx].size = sizeof(float) * this->nr_rows_;
                assign_section_offset(section_headers[curr_section_idx], used_offset);
                curr_section_idx++;
            }
            assert(curr_section_idx == nr_sections);

            writer.write(section_headers.data(), sizeof(InvertedIndexSectionHeader), nr_sections);

            uint32_t index_encoding_type = static_cast<uint32_t>(InvertedIndexEncoding::FIXED_DOCID_WINDOWS);
            write_padding_until(writer, section_headers[0].offset);
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

            write_padding_until(writer, section_headers[1].offset);
            this->dim_map_.write_reverse_section(writer, dim_map_storage);

            curr_section_idx = 2;
            if (this->dim_map_.has_mphf_section(dim_map_storage)) {
                write_padding_until(writer, section_headers[curr_section_idx].offset);
                this->dim_map_.write_mphf_section(writer, dim_map_storage);
                curr_section_idx++;
            }

            write_padding_until(writer, section_headers[curr_section_idx].offset);
            writer.write(max_scores_per_dim_span_.data(), sizeof(float), max_scores_per_dim_span_.size());
            curr_section_idx++;

            if (has_row_sums) {
                write_padding_until(writer, section_headers[curr_section_idx].offset);
                writer.write(row_sums_span_.data(), sizeof(float), row_sums_span_.size());
                curr_section_idx++;
            }

            return Status::success;
        }
    }

    [[nodiscard]] Status
    deserialize(MemoryIOReader& reader) override {
        if constexpr (AllowIncremental) {
            LOG_KNOWHERE_ERROR_ << "SindiInvertedIndex incremental mode does not support deserialize";
            return Status::not_implemented;
        }

        if constexpr (!AllowIncremental) {
            auto file_header_handler = [&, this]() {
                uint32_t index_format_version = 0;
                reader.read(&index_format_version, sizeof(uint32_t));
                if (index_format_version != kInvertedIndexFileFormatVersion) {
                    return Status::invalid_serialized_index_type;
                }

                reader.read(&this->nr_rows_, sizeof(uint32_t));
                reader.read(&this->max_dim_, sizeof(uint32_t));
                reader.read(&this->nr_inner_dims_, sizeof(uint32_t));
                reader.advance(kInvertedIndexHeaderReservedBytes);

                // if there are zero rows, there should be no inner dims, something is wrong
                if (this->nr_rows_ == 0 && this->nr_inner_dims_ != 0) {
                    return Status::invalid_serialized_index_type;
                }

                return Status::success;
            };

            uint32_t nr_sections = 0;
            uint64_t posting_list_section_bytes = 0;
            uint64_t total_postings = 0;
            uint64_t window_nnz_bytes = 0;
            auto sections_handler = [&, this]() {
                reader.read(&nr_sections, sizeof(uint32_t));
                if (nr_sections < 3) {
                    return Status::invalid_serialized_index_type;
                }
                const auto section_headers = read_section_headers(reader, nr_sections);
                const auto dim_map_storage =
                    find_section_header(section_headers, InvertedIndexSectionType::DIM_MAP_MPHF) == nullptr
                        ? DimMapMphfStorage::LegacyTrailer
                        : DimMapMphfStorage::SeparateSection;
                if (auto status =
                        this->dim_map_.load_sections(reader, section_headers, this->nr_inner_dims_, dim_map_storage);
                    status != Status::success) {
                    return status;
                }

                for (size_t i = 0; i < section_headers.size(); ++i) {
                    const auto& section_header = section_headers[i];

                    // Log high-level section info
                    LOG_KNOWHERE_INFO_ << "SindiInvertedIndex::deserialize section[" << i
                                       << "] type=" << static_cast<uint32_t>(section_header.type)
                                       << " offset=" << section_header.offset << " size_bytes=" << section_header.size;

                    switch (section_header.type) {
                        case InvertedIndexSectionType::POSTING_LISTS: {
                            reader.seekg(section_header.offset);
                            posting_list_section_bytes = section_header.size;
                            // check index encoding type
                            uint32_t index_encoding_type = 0;
                            reader.read(&index_encoding_type, sizeof(uint32_t));
                            if (index_encoding_type !=
                                static_cast<uint32_t>(InvertedIndexEncoding::FIXED_DOCID_WINDOWS)) {
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
                                    : static_cast<uint32_t>((this->nr_rows_ + this->window_size_ - 1) /
                                                            this->window_size_);
                            if (this->nr_windows_ != expected_windows) {
                                LOG_KNOWHERE_INFO_
                                    << "SindiInvertedIndex::deserialize nr_windows_=" << this->nr_windows_
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
                                const uint8_t* mask_base =
                                    reinterpret_cast<const uint8_t*>(reader.data() + reader.tellg());
                                plists_wnnzs_fmts_msk_span_ = std::span<const uint8_t>(mask_base, mask_sz);
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
                                plists_window_nnzs_spans_[dimid] = std::span<const uint8_t>(dbase, span_sz);
                                reader.advance(static_cast<size_t>(span_sz) * sizeof(uint8_t));
                                bytes_win_nnzs += sizeof(uint32_t) + static_cast<size_t>(span_sz) * sizeof(uint8_t);
                            }

                            // plists dim offsets
                            plists_dim_offsets_span_ = std::span<const uint32_t>(
                                reinterpret_cast<const uint32_t*>(reader.data() + reader.tellg()), nr_dims + 1);
                            reader.advance((nr_dims + 1) * sizeof(uint32_t));
                            total_postings = plists_dim_offsets_span_[nr_dims];

                            // Validate total_postings against section size
                            const uint64_t bytes_dim_offsets = static_cast<uint64_t>(nr_dims + 1) * sizeof(uint32_t);
                            const uint64_t bytes_postings_data =
                                static_cast<uint64_t>(total_postings) * (sizeof(uint16_t) + sizeof(QuantType));
                            const uint64_t expected_section_bytes =
                                bytes_header + bytes_mask + bytes_win_nnzs + bytes_dim_offsets + bytes_postings_data;
                            if (expected_section_bytes != section_header.size) {
                                LOG_KNOWHERE_INFO_ << "SindiInvertedIndex::deserialize POSTING_LISTS size mismatch: "
                                                   << "expected=" << expected_section_bytes
                                                   << " actual=" << section_header.size;
                                return Status::invalid_serialized_index_type;
                            }

                            // ids region (per-dim contiguous, concatenated)
                            const uint16_t* ids_region =
                                reinterpret_cast<const uint16_t*>(reader.data() + reader.tellg());
                            const uint64_t bytes_ids = static_cast<uint64_t>(total_postings) * sizeof(uint16_t);
                            size_t base = 0;
                            for (size_t dim_id = 0; dim_id < nr_dims; ++dim_id) {
                                uint32_t len = plists_dim_offsets_span_[dim_id + 1] - plists_dim_offsets_span_[dim_id];
                                total_plists_ids_spans_[dim_id] = std::span<const uint16_t>(ids_region + base, len);
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
                                total_plists_vals_spans_[dim_id] = std::span<const QuantType>(vals_region + base, len);
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
                            window_nnz_bytes = bytes_win_nnzs;

                            break;
                        }
                        case InvertedIndexSectionType::DIM_MAP_REVERSE:
                        case InvertedIndexSectionType::DIM_MAP_MPHF: {
                            break;
                        }
                        case InvertedIndexSectionType::MAX_SCORES_PER_DIM: {
                            reader.seekg(section_header.offset);
                            max_scores_per_dim_span_ = std::span<const float>(
                                reinterpret_cast<const float*>(reader.data() + section_header.offset),
                                this->nr_inner_dims_);
                            reader.advance(sizeof(float) * this->nr_inner_dims_);
                            break;
                        }
                        case InvertedIndexSectionType::ROW_SUMS: {
                            // Row sums section for BM25
                            reader.seekg(section_header.offset);
                            row_sums_span_ = std::span<const float>(
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

            LOG_KNOWHERE_INFO_ << "SindiInvertedIndex::deserialize stats: rows=" << this->nr_rows_
                               << " max_dim=" << this->max_dim_ << " inner_dims=" << this->nr_inner_dims_
                               << " sections=" << nr_sections << " window_size=" << this->window_size_
                               << " windows=" << this->nr_windows_ << " postings=" << total_postings
                               << " posting_list_bytes=" << posting_list_section_bytes
                               << " window_nnz_bytes=" << window_nnz_bytes
                               << " dim_map_reverse_bytes=" << this->dim_map_.reverse_size_bytes()
                               << " mphf_bytes=" << this->dim_map_.mphf_serialized_size() << " byte_size=" << size();

            return Status::success;
        }
    }

    void
    search(const SparseRow<DataType>& query, size_t k, float* distances, label_t* labels, const BitsetView& bitset,
           const InvertedIndexSearchParams& search_params) const override {
        std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
        std::fill(labels, labels + k, -1);

        if (query.size() == 0) {
            return;
        }

        auto q_vec = parse_query_with_dim_map(query, this->dim_map_, search_params.approx.drop_ratio_search);
        if (q_vec.empty()) {
            return;
        }

        // Compute max contributions for each query term.
        // and sort query terms by max contributions descending for early termination
        std::vector<float> max_contributions(q_vec.size());
        for (size_t i = 0; i < q_vec.size(); ++i) {
            const auto& [qid, qval] = q_vec[i];
            float dim_max = (qid < max_scores_per_dim_span_.size()) ? max_scores_per_dim_span_[qid] : 0.0f;
            max_contributions[i] = qval * dim_max;
        }

        // Sort q_vec by max contribution descending (process high-impact terms first)
        std::vector<size_t> sorted_indices(q_vec.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
                  [&max_contributions](size_t a, size_t b) { return max_contributions[a] > max_contributions[b]; });

        // Reorder q_vec and max_contributions according to sorted indices
        std::vector<std::pair<uint32_t, float>> sorted_q_vec(q_vec.size());
        std::vector<float> sorted_max_contributions(q_vec.size());
        for (size_t i = 0; i < q_vec.size(); ++i) {
            sorted_q_vec[i] = q_vec[sorted_indices[i]];
            sorted_max_contributions[i] = max_contributions[sorted_indices[i]];
        }

        // Compute suffix sums of max contributions for early termination
        // suffix_sum[i] = sum of max_contributions from index i to end
        std::vector<float> suffix_sum(q_vec.size() + 1, 0.0f);
        for (int i = static_cast<int>(q_vec.size()) - 1; i >= 0; --i) {
            suffix_sum[i] = suffix_sum[i + 1] + sorted_max_contributions[i];
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
        cursors.reserve(sorted_q_vec.size());
        for (auto& [qid, qval] : sorted_q_vec) {
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

                float curr_max_score = 0.0f;
                bool skip_window_calc = false;
                for (size_t ci = 0; ci < cursors.size(); ++ci) {
                    auto& cur = cursors[ci];
                    if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                        continue;
                    }

                    auto [soff, wnnz] = cur.advance_window(widx);

                    if (skip_window_calc || wnnz == 0) {
                        continue;
                    }

                    // Skip window calculation if current max + remaining max contributions <= threshold
                    if (curr_max_score + search_params.approx.dim_max_score_ratio * suffix_sum[ci] <= threshold) {
                        skip_window_calc = true;
                        continue;
                    }

                    const uint16_t* plist_ids = cur.ids_base + soff;
                    const QuantType* plist_vals = cur.vals_base + soff;

                    float dispatch_max = scatter_fn(cur.qval, plist_vals, plist_ids, wnnz, wscores_final);
                    if (dispatch_max > curr_max_score) {
                        curr_max_score = dispatch_max;
                    }
                }

                if (curr_max_score > threshold) {
                    batch_insert_fn(wscores_final, docid_start, curr_window_size, topk_q, threshold, bitset);
                }
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

                float curr_max_score = 0.0f;
                bool skip_window_calc = false;
                for (size_t ci = 0; ci < cursors.size(); ++ci) {
                    auto& cur = cursors[ci];
                    if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                        continue;
                    }

                    auto [soff, wnnz] = cur.advance_window(widx);

                    if (skip_window_calc || wnnz == 0) {
                        continue;
                    }

                    // Skip window calculation if current max + remaining max contributions <= threshold
                    if (curr_max_score + search_params.approx.dim_max_score_ratio * suffix_sum[ci] <= threshold) {
                        skip_window_calc = true;
                        continue;
                    }

                    const uint16_t* plist_ids = cur.ids_base + soff;
                    const QuantType* plist_vals = cur.vals_base + soff;

                    float dispatch_max = accumulate_fn(cur.qval, plist_vals, plist_ids, wnnz, wscores_final, bm25_k1,
                                                       bm25_b, bm25_avgdl, row_sums_ptr + docid_start);
                    if (dispatch_max > curr_max_score) {
                        curr_max_score = dispatch_max;
                    }
                }

                if (curr_max_score > threshold) {
                    batch_insert_fn(wscores_final, docid_start, curr_window_size, topk_q, threshold, bitset);
                }
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

        auto q_vec = parse_query_with_dim_map(query, this->dim_map_, search_params.approx.drop_ratio_search);
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
     * @return std::span<const float> The row sums span (empty if not BM25 index)
     */
    [[nodiscard]] std::span<const float>
    get_row_sums_span() const noexcept {
        return row_sums_span_;
    }

    [[nodiscard]] Status
    convert_to_raw_data(MemoryIOWriter& writer) const override {
        return Status::not_implemented;
    }

 private:
    // Global posting lists (per dimension, concatenated across windows)
    using aligned_u16_vec = std::vector<uint16_t, aligned_allocator<uint16_t, 64>>;
    using aligned_quant_vec = std::vector<QuantType, aligned_allocator<QuantType, 64>>;
    std::vector<aligned_u16_vec> total_plists_ids_;
    std::vector<aligned_quant_vec> total_plists_vals_;
    std::vector<std::span<const uint16_t>> total_plists_ids_spans_;
    std::vector<std::span<const QuantType>> total_plists_vals_spans_;

    // Window sizes encoding (per-dim window nnz) and bitset of formats
    // Bit=1 means sparse format (wid, wnnz pairs), Bit=0 means dense format (one entry per window)
    std::vector<uint8_t> plists_wnnzs_fmts_msk_;
    std::span<const uint8_t> plists_wnnzs_fmts_msk_span_;
    std::vector<std::vector<uint16_t>> window_index_plists_sz_;
    std::vector<std::span<const uint16_t>> window_index_plists_sz_spans_;
    std::vector<std::vector<uint8_t>> plists_window_nnzs_;
    std::vector<std::span<const uint8_t>> plists_window_nnzs_spans_;
    std::vector<uint32_t> plists_dim_offsets_;
    std::span<const uint32_t> plists_dim_offsets_span_;
    std::vector<float> max_scores_per_dim_;
    std::span<const float> max_scores_per_dim_span_;

    // Row sums only needed for BM25
    std::vector<float> row_sums_;
    std::span<const float> row_sums_span_;

    bool legacy_dim_map_mphf_trailer_workaround_{true};
    uint32_t window_size_{max_window_size};
    uint32_t nr_windows_{0};

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
                    uint32_t wval;
                    std::memcpy(&wval, wnnz_buf + cursor, sizeof(uint32_t));
                    auto wid = wval >> wnnz_bits;
                    if (wid == widx) {
                        wnnz = static_cast<uint16_t>(wval & wnnz_mask);
                        offset += wnnz;
                        cursor += sizeof(uint32_t);
                    }
                }
            } else {
                std::memcpy(&wnnz, wnnz_buf + widx * sizeof(uint16_t), sizeof(uint16_t));
                offset += wnnz;
            }
            return {soff, wnnz};
        }
    };
};

using SindiInvertedIndexIP = SindiInvertedIndex<float, knowhere::fp16>;
using SindiInvertedIndexBM25 = SindiInvertedIndex<float, uint16_t>;
using GrowableSindiInvertedIndexIP = SindiInvertedIndex<float, knowhere::fp16, true>;
using GrowableSindiInvertedIndexBM25 = SindiInvertedIndex<float, uint16_t, true>;

}  // namespace knowhere::sparse::inverted
