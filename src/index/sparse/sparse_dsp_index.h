// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef SPARSE_DSP_INDEX_H
#define SPARSE_DSP_INDEX_H

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <boost/core/span.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"
#include "simd/instruction_set.h"
#include "simd/sparse_simd.h"

namespace knowhere::sparse {

// Section types for DSP index serialization format
enum class DspSectionType : uint32_t {
    POSTING_LISTS = 0,
    METRIC_PARAMS = 1,
    DIM_MAP = 2,
    ROW_SUMS = 3,
    MAX_SCORES_PER_DIM = 4,
    PROMETHEUS_BUILD_STATS = 5,
    DSP_METADATA = 6
};

struct DspSectionHeader {
    DspSectionType type;
    uint64_t offset;
    uint64_t size;
};

struct DspBuildStats {
    std::vector<uint32_t> dataset_nnz_stats_;
    std::vector<uint32_t> posting_list_length_stats_;
};

// Superblock selection modes for DSP search.
enum class DspSearchMode : int {
    DSP = 0,   // dual-threshold (mu, eta) + optional top-gamma backstop
    LSP0 = 1,  // top-gamma from ub>=theta, no mu/asc gate (recommended for SPLADE)
    LSP1 = 2,  // LSP/0 + mu-overestimation gate (ub>theta/mu)
    LSP2 = 3,  // LSP/1 + asc gate (ub>theta/mu || asc>theta/eta)
};

struct DspSearchParams {
    int refine_factor;
    float drop_ratio_search;
    float dim_max_score_ratio;
    DspSearchMode dsp_mode = DspSearchMode::DSP;
    float dsp_mu = 1.0f;
    float dsp_eta = 1.0f;
    int dsp_gamma = 0;
    bool dsp_kth_init = true;
    float dsp_kth_alpha = 1.0f;
};

// Type-erased base for DspIndex so that the index node can hold either mmapped or non-mmapped variant.
template <typename T>
class DspIndexBase {
 public:
    virtual ~DspIndexBase() = default;
    virtual Status SerializeV0(MemoryIOWriter& writer) const = 0;
    virtual Status DeserializeV0(MemoryIOReader& reader, int map_flags, const std::string& supplement_target_filename) = 0;
    virtual Status Serialize(MemoryIOWriter& writer) const = 0;
    virtual Status Deserialize(MemoryIOReader& reader) = 0;
    virtual Status Train(const SparseRow<T>* data, size_t rows) = 0;
    virtual Status Add(const SparseRow<T>* data, size_t rows, int64_t dim) = 0;
    virtual void Search(const SparseRow<T>& query, size_t k, float* distances, label_t* labels,
                        const BitsetView& bitset, const DocValueComputer<T>& computer,
                        DspSearchParams& params) const = 0;
    virtual std::vector<float> GetAllDistances(const SparseRow<T>& query, float drop_ratio_search,
                                               const BitsetView& bitset, const DocValueComputer<T>& computer) const = 0;
    virtual float GetRawDistance(const label_t vec_id, const SparseRow<T>& query,
                                const DocValueComputer<T>& computer) const = 0;
    virtual expected<DocValueComputer<T>> GetDocValueComputer(const BaseConfig& cfg) const = 0;
    [[nodiscard]] virtual size_t size() const = 0;
    [[nodiscard]] virtual size_t n_rows() const = 0;
    [[nodiscard]] virtual size_t n_cols() const = 0;
    virtual void SetBM25Params(float k1, float b, float avgdl) = 0;
};

// DSP (Dynamic Superblock Pruning) index for fast sparse vector search.
//
// DSP index structure:
// - u8 quantized block max scores
// - u16 upper bound accumulators with AVX-512 SIMD
// - Counting sort (bucket sort) for block ordering by upper bound
// - Forward index with two-pointer merge scoring
// - Two-level hierarchy: superblocks for coarse pruning, subblocks for scoring
template <typename DType, typename QType, bool mmapped = false>
class DspIndex : public DspIndexBase<DType> {
 public:
    template <typename U>
    using Vector = std::conditional_t<mmapped, GrowableVectorView<U>, std::vector<U>>;

    static constexpr uint32_t kSubblockSize = 8;
    static constexpr uint32_t kSuperblockSize = 512;
    static constexpr uint32_t kStride = kSuperblockSize / kSubblockSize;  // 64
    static constexpr uint32_t kSimdWidth = 32;                            // AVX-512 processes 32 u16 values

    explicit DspIndex(SparseMetricType metric_type) : metric_type_(metric_type) {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        // for now, use timestamp as index_id
        index_id_ = std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());
        index_size_gauge_ = &sparse_inverted_index_size_family.Add({{"index_id", index_id_}, {"index_type", "dsp"}});
        index_dataset_nnz_len_histogram_ =
            &sparse_dataset_nnz_len_family.Add({{"index_id", index_id_}, {"index_type", "dsp"}}, defaultBuckets);
        index_posting_list_len_histogram_ = &sparse_inverted_index_posting_list_len_family.Add(
            {{"index_id", index_id_}, {"index_type", "dsp"}}, defaultBuckets);
#endif
    }

    ~DspIndex() {
        if constexpr (mmapped) {
            if (map_ != nullptr) {
                auto res = munmap(map_, map_byte_size_);
                if (res != 0) {
                    LOG_KNOWHERE_ERROR_ << "Failed to munmap when deleting sparse DspIndex: " << strerror(errno);
                }
                map_ = nullptr;
                map_byte_size_ = 0;
            }
            if (map_fd_ != -1) {
                close(map_fd_);
                map_fd_ = -1;
            }
        }
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        if (index_size_gauge_ != nullptr) {
            sparse_inverted_index_size_family.Remove(index_size_gauge_);
        }
        if (index_dataset_nnz_len_histogram_ != nullptr) {
            sparse_dataset_nnz_len_family.Remove(index_dataset_nnz_len_histogram_);
        }
        if (index_posting_list_len_histogram_ != nullptr) {
            sparse_inverted_index_posting_list_len_family.Remove(index_posting_list_len_histogram_);
        }
#endif
    }

    void
    SetBM25Params(float k1, float b, float avgdl) override {
        bm25_params_ = std::make_unique<BM25Params>(k1, b, avgdl);
    }

    expected<DocValueComputer<float>>
    GetDocValueComputer(const BaseConfig& cfg) const override {
        auto metric_type = cfg.metric_type;
        if (metric_type_ != SparseMetricType::METRIC_BM25) {
            if (metric_type.has_value() && !IsMetricType(metric_type.value(), metric::IP)) {
                auto msg =
                    "metric type not match, expected: " + std::string(metric::IP) + ", got: " + metric_type.value();
                return expected<DocValueComputer<float>>::Err(Status::invalid_metric_type, msg);
            }
            return GetDocValueOriginalComputer<float>();
        }
        if (metric_type.has_value() && !IsMetricType(metric_type.value(), metric::BM25)) {
            auto msg =
                "metric type not match, expected: " + std::string(metric::BM25) + ", got: " + metric_type.value();
            return expected<DocValueComputer<float>>::Err(Status::invalid_metric_type, msg);
        }
        if (!cfg.bm25_avgdl.has_value()) {
            return expected<DocValueComputer<float>>::Err(Status::invalid_args,
                                                          "avgdl must be supplied during searching");
        }
        auto avgdl = cfg.bm25_avgdl.value();
        avgdl = std::max(avgdl, 1.0f);
        if ((cfg.bm25_k1.has_value() && cfg.bm25_k1.value() != bm25_params_->k1) ||
            ((cfg.bm25_b.has_value() && cfg.bm25_b.value() != bm25_params_->b))) {
            return expected<DocValueComputer<float>>::Err(Status::invalid_args,
                                                          "search time k1/b must equal load time config.");
        }
        return GetDocValueBM25Computer<float>(bm25_params_->k1, bm25_params_->b, avgdl);
    }

    Status
    Train(const SparseRow<DType>* data, size_t rows) override {
        if constexpr (mmapped) {
            throw std::invalid_argument("mmapped DspIndex does not support Train");
        } else {
            return Status::success;
        }
    }

    Status
    Add(const SparseRow<DType>* data, size_t rows, int64_t dim) override {
        if constexpr (mmapped) {
            throw std::invalid_argument("mmapped DspIndex does not support Add");
        } else {
            auto current_rows = n_rows_internal_;
            if ((size_t)dim > max_dim_) {
                max_dim_ = dim;
            }

            if (metric_type_ == SparseMetricType::METRIC_BM25) {
                bm25_params_->row_sums.reserve(current_rows + rows);
            }
            for (size_t i = 0; i < rows; ++i) {
                add_row_to_index(data[i], current_rows + i);
            }
            n_rows_internal_ += rows;

            nr_inner_dims_ = dim_map_.size();

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            build_stats_.posting_list_length_stats_.resize(nr_inner_dims_);
            for (size_t i = 0; i < nr_inner_dims_; ++i) {
                build_stats_.posting_list_length_stats_[i] = inverted_index_ids_[i].size();
            }
#endif

            inverted_index_ids_spans_.clear();
            inverted_index_vals_spans_.clear();
            inverted_index_ids_spans_.reserve(nr_inner_dims_);
            inverted_index_vals_spans_.reserve(nr_inner_dims_);

            for (size_t i = 0; i < nr_inner_dims_; ++i) {
                inverted_index_ids_spans_.emplace_back(inverted_index_ids_[i].data(), inverted_index_ids_[i].size());
                inverted_index_vals_spans_.emplace_back(inverted_index_vals_[i].data(), inverted_index_vals_[i].size());
            }

            if (max_score_in_dim_.size() > 0) {
                max_score_in_dim_spans_ = boost::span<const float>(max_score_in_dim_.data(), max_score_in_dim_.size());
            }

            if (metric_type_ == SparseMetricType::METRIC_BM25) {
                bm25_params_->row_sums_spans_ =
                    boost::span<const float>(bm25_params_->row_sums.data(), bm25_params_->row_sums.size());
            }

            build_dsp_metadata();
            return Status::success;
        }
    }

    Status
    SerializeV0(MemoryIOWriter& writer) const override {
        DType deprecated_value_threshold = 0;
        writeBinaryPOD(writer, n_rows_internal_);
        writeBinaryPOD(writer, max_dim_);
        writeBinaryPOD(writer, deprecated_value_threshold);

        auto dim_map_reverse = std::unordered_map<uint32_t, table_t>();
        for (const auto& [dim, dim_id] : dim_map_) {
            dim_map_reverse[dim_id] = dim;
        }

        std::vector<size_t> row_sizes(n_rows_internal_, 0);
        for (auto inverted_index_ids_span : inverted_index_ids_spans_) {
            for (const auto& id : inverted_index_ids_span) {
                row_sizes[id]++;
            }
        }

        std::vector<SparseRow<DType>> raw_rows(n_rows_internal_);
        for (size_t i = 0; i < n_rows_internal_; ++i) {
            raw_rows[i] = std::move(SparseRow<DType>(row_sizes[i]));
        }

        for (size_t i = 0; i < inverted_index_ids_spans_.size(); ++i) {
            const auto& ids = inverted_index_ids_spans_[i];
            const auto& vals = inverted_index_vals_spans_[i];
            const auto dim = dim_map_reverse[i];
            for (size_t j = 0; j < ids.size(); ++j) {
                raw_rows[ids[j]].set_at(raw_rows[ids[j]].size() - row_sizes[ids[j]], dim, vals[j]);
                --row_sizes[ids[j]];
            }
        }

        for (table_t vec_id = 0; vec_id < n_rows_internal_; ++vec_id) {
            writeBinaryPOD(writer, raw_rows[vec_id].size());
            if (raw_rows[vec_id].size() > 0) {
                writer.write(raw_rows[vec_id].data(), raw_rows[vec_id].size() * SparseRow<DType>::element_size());
            }
        }

        return Status::success;
    }

    Status
    DeserializeV0(MemoryIOReader& reader, int map_flags, const std::string& supplement_target_filename) override {
        DType deprecated_value_threshold;
        int64_t rows;
        readBinaryPOD(reader, rows);
        rows = std::abs(rows);
        readBinaryPOD(reader, max_dim_);
        readBinaryPOD(reader, deprecated_value_threshold);

        if constexpr (mmapped) {
            RETURN_IF_ERROR(PrepareMmap(reader, rows, map_flags, supplement_target_filename));
        } else {
            if (metric_type_ == SparseMetricType::METRIC_BM25) {
                bm25_params_->row_sums.reserve(rows);
            }
        }

        auto load_progress_interval = rows / 10;
        for (int64_t i = 0; i < rows; ++i) {
            if (load_progress_interval > 0 && i % load_progress_interval == 0) {
                LOG_KNOWHERE_INFO_ << "Sparse DspIndex loading progress: " << (i / load_progress_interval * 10) << "%";
            }

            size_t count;
            readBinaryPOD(reader, count);
            SparseRow<DType> raw_row;
            if constexpr (mmapped) {
                raw_row = std::move(SparseRow<DType>(count, reader.data() + reader.tellg(), false));
                reader.advance(count * SparseRow<DType>::element_size());
            } else {
                raw_row = std::move(SparseRow<DType>(count));
                if (count > 0) {
                    reader.read(raw_row.data(), count * SparseRow<DType>::element_size());
                }
            }
            add_row_to_index(raw_row, i);
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            index_dataset_nnz_len_histogram_->Observe(count);
#endif
        }
        LOG_KNOWHERE_INFO_ << "Sparse DspIndex loading progress: 100%";

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        for (size_t i = 0; i < dim_map_.size(); ++i) {
            index_posting_list_len_histogram_->Observe(inverted_index_ids_[i].size());
        }
        index_size_gauge_->Set((double)size() / 1024.0 / 1024.0);
#endif

        n_rows_internal_ = rows;
        nr_inner_dims_ = dim_map_.size();

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        build_stats_.posting_list_length_stats_.resize(nr_inner_dims_);
        for (size_t i = 0; i < nr_inner_dims_; ++i) {
            build_stats_.posting_list_length_stats_[i] = inverted_index_ids_[i].size();
        }
#endif
        inverted_index_ids_spans_.reserve(nr_inner_dims_);
        inverted_index_vals_spans_.reserve(nr_inner_dims_);
        for (size_t i = 0; i < nr_inner_dims_; ++i) {
            inverted_index_ids_spans_.emplace_back(inverted_index_ids_[i].data(), inverted_index_ids_[i].size());
            inverted_index_vals_spans_.emplace_back(inverted_index_vals_[i].data(), inverted_index_vals_[i].size());
        }

        if (max_score_in_dim_.size() > 0) {
            max_score_in_dim_spans_ = boost::span<const float>(max_score_in_dim_.data(), max_score_in_dim_.size());
        }

        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            bm25_params_->row_sums_spans_ =
                boost::span<const float>(bm25_params_->row_sums.data(), bm25_params_->row_sums.size());
        }

        build_dsp_metadata();
        return Status::success;
    }

    Status
    Serialize(MemoryIOWriter& writer) const override {
        const uint32_t index_format_version = 1;

        // Index File Header (v1)
        writer.write(&index_format_version, sizeof(uint32_t));
        writer.write(&n_rows_internal_, sizeof(uint32_t));
        writer.write(&max_dim_, sizeof(uint32_t));
        writer.write(&nr_inner_dims_, sizeof(uint32_t));
        auto reserved = std::array<uint8_t, index_file_v1_header_reserved_size>();
        writer.write(reserved.data(), reserved.size());

        // Phase 1: Collect section metadata
        std::vector<std::pair<DspSectionType, uint64_t>> section_meta;

        uint64_t posting_lists_size = sizeof(uint32_t);                 // encoding type
        posting_lists_size += sizeof(uint64_t) * (nr_inner_dims_ + 1);  // dim offsets
        for (size_t i = 0; i < nr_inner_dims_; ++i) {
            posting_lists_size += inverted_index_ids_spans_[i].size() * sizeof(uint32_t) +
                                  inverted_index_vals_spans_[i].size() * sizeof(QType);
        }
        section_meta.emplace_back(DspSectionType::POSTING_LISTS, posting_lists_size);
        section_meta.emplace_back(DspSectionType::DIM_MAP, sizeof(uint32_t) * nr_inner_dims_);

        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            section_meta.emplace_back(DspSectionType::ROW_SUMS, sizeof(float) * n_rows_internal_);
        }
        if (max_score_in_dim_spans_.size() > 0) {
            section_meta.emplace_back(DspSectionType::MAX_SCORES_PER_DIM, sizeof(float) * nr_inner_dims_);
        }
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
        section_meta.emplace_back(DspSectionType::PROMETHEUS_BUILD_STATS,
                                  sizeof(uint32_t) * n_rows_internal_ + sizeof(uint32_t) * nr_inner_dims_);
#endif

        // Append DSP metadata section
        if (n_subblocks_ > 0) {
            uint64_t dsp_size = 0;
            dsp_size += 4 * sizeof(uint32_t);  // header: version, n_subblocks, n_superblocks, n_sb_padded

            for (uint32_t d = 0; d < nr_inner_dims_; ++d) {
                const auto& bm = dim_block_max_[d];
                dsp_size += sizeof(uint32_t);                        // n_block_ids
                dsp_size += sizeof(uint32_t);                        // n_logical
                dsp_size += 4;                                       // kth[4]
                dsp_size += bm.block_ids.size() * sizeof(uint32_t);  // block_ids
                dsp_size += sizeof(uint32_t);                        // packed_size
                dsp_size += bm.max_scores.size() * sizeof(uint8_t);  // u8 max_scores
            }

            uint32_t spb_total = spb_block_ids_.size();
            dsp_size += sizeof(uint32_t);                         // spb_total
            dsp_size += (nr_inner_dims_ + 1) * sizeof(uint32_t);  // spb_dim_offsets
            dsp_size += spb_total * sizeof(uint32_t);             // spb_block_ids
            dsp_size += spb_total * sizeof(float);                // spb_max_vals
            dsp_size += spb_total * sizeof(float);                // spb_asc_vals

            uint32_t total_terms = fwd_term_ids_.size();
            uint32_t total_entries = fwd_doc_offsets_.size();
            dsp_size += sizeof(uint32_t);                       // total_terms
            dsp_size += sizeof(uint32_t);                       // total_entries
            dsp_size += (n_subblocks_ + 1) * sizeof(uint32_t);  // fwd_block_term_offsets
            dsp_size += total_terms * sizeof(uint32_t);         // fwd_term_ids
            dsp_size += (total_terms + 1) * sizeof(uint32_t);   // fwd_term_entry_offsets
            dsp_size += total_entries * sizeof(uint8_t);        // fwd_doc_offsets
            dsp_size += total_entries * sizeof(float);          // fwd_scores

            section_meta.emplace_back(DspSectionType::DSP_METADATA, dsp_size);
        }

        // Phase 2: Build headers with offsets and write section table
        uint32_t nr_sections = static_cast<uint32_t>(section_meta.size());
        writer.write(&nr_sections, sizeof(uint32_t));

        std::vector<DspSectionHeader> section_headers(nr_sections);
        uint64_t used_offset = index_file_v1_header_size + sizeof(uint32_t) + sizeof(DspSectionHeader) * nr_sections;
        for (uint32_t i = 0; i < nr_sections; ++i) {
            section_headers[i].type = section_meta[i].first;
            section_headers[i].offset = used_offset;
            section_headers[i].size = section_meta[i].second;
            used_offset += section_meta[i].second;
        }
        writer.write(section_headers.data(), sizeof(DspSectionHeader), nr_sections);

        // Write posting lists
        uint32_t index_encoding_type = 0;
        writer.write(&index_encoding_type, sizeof(uint32_t));
        std::vector<uint64_t> inverted_index_offsets(nr_inner_dims_ + 1);
        inverted_index_offsets[0] = 0;
        for (size_t i = 1; i <= nr_inner_dims_; ++i) {
            inverted_index_offsets[i] = inverted_index_offsets[i - 1] + inverted_index_ids_spans_[i - 1].size();
        }
        writer.write(inverted_index_offsets.data(), sizeof(uint64_t), inverted_index_offsets.size());
        for (size_t i = 0; i < nr_inner_dims_; ++i) {
            writer.write(inverted_index_ids_spans_[i].data(), sizeof(uint32_t), inverted_index_ids_spans_[i].size());
        }
        for (size_t i = 0; i < nr_inner_dims_; ++i) {
            writer.write(inverted_index_vals_spans_[i].data(), sizeof(QType), inverted_index_vals_spans_[i].size());
        }

        // Write dim map
        auto dim_map_reverse = std::vector<uint32_t>(nr_inner_dims_);
        for (const auto& [dim, dim_id] : dim_map_) {
            dim_map_reverse[dim_id] = dim;
        }
        writer.write(dim_map_reverse.data(), sizeof(uint32_t), nr_inner_dims_);

        // Write row sums (BM25)
        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            writer.write(bm25_params_->row_sums_spans_.data(), sizeof(float), n_rows_internal_);
        }

        // Write max scores per dim
        if (max_score_in_dim_spans_.size() > 0) {
            writer.write(max_score_in_dim_spans_.data(), sizeof(float), nr_inner_dims_);
        }

        // Write prometheus build stats
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
        writer.write(build_stats_.dataset_nnz_stats_.data(), sizeof(uint32_t), n_rows_internal_);
        writer.write(build_stats_.posting_list_length_stats_.data(), sizeof(uint32_t), nr_inner_dims_);
#endif

        // Write DSP metadata section
        if (n_subblocks_ > 0) {
            uint32_t dsp_version = 1;
            writer.write(&dsp_version, sizeof(uint32_t));
            writer.write(&n_subblocks_, sizeof(uint32_t));
            writer.write(&n_superblocks_, sizeof(uint32_t));
            writer.write(&n_sb_padded_, sizeof(uint32_t));

            for (uint32_t d = 0; d < nr_inner_dims_; ++d) {
                const auto& bm = dim_block_max_[d];
                uint32_t n_block_ids = bm.block_ids.size();
                uint32_t n_logical = bm.n_logical;
                writer.write(&n_block_ids, sizeof(uint32_t));
                writer.write(&n_logical, sizeof(uint32_t));
                writer.write(bm.kth, 4);
                if (n_block_ids > 0) {
                    writer.write(bm.block_ids.data(), sizeof(uint32_t), n_block_ids);
                }
                uint32_t packed_size = bm.max_scores.size();
                writer.write(&packed_size, sizeof(uint32_t));
                if (packed_size > 0) {
                    writer.write(bm.max_scores.data(), sizeof(uint8_t), packed_size);
                }
            }

            uint32_t spb_total = spb_block_ids_.size();
            writer.write(&spb_total, sizeof(uint32_t));
            writer.write(spb_dim_offsets_.data(), sizeof(uint32_t), nr_inner_dims_ + 1);
            writer.write(spb_block_ids_.data(), sizeof(uint32_t), spb_total);
            writer.write(spb_max_vals_.data(), sizeof(float), spb_total);
            writer.write(spb_asc_vals_.data(), sizeof(float), spb_total);

            uint32_t total_terms = fwd_term_ids_.size();
            uint32_t total_entries = fwd_doc_offsets_.size();
            writer.write(&total_terms, sizeof(uint32_t));
            writer.write(&total_entries, sizeof(uint32_t));
            writer.write(fwd_block_term_offsets_.data(), sizeof(uint32_t), n_subblocks_ + 1);
            writer.write(fwd_term_ids_.data(), sizeof(uint32_t), total_terms);
            writer.write(fwd_term_entry_offsets_.data(), sizeof(uint32_t), total_terms + 1);
            writer.write(fwd_doc_offsets_.data(), sizeof(uint8_t), total_entries);
            writer.write(fwd_scores_.data(), sizeof(float), total_entries);
        }

        return Status::success;
    }

    Status
    Deserialize(MemoryIOReader& reader) override {
        dsp_loaded_ = false;

        // Read file header
        uint32_t index_format_version = 0;
        reader.read(&index_format_version, sizeof(uint32_t));
        if (index_format_version != 1) {
            return Status::invalid_serialized_index_type;
        }

        reader.read(&n_rows_internal_, sizeof(uint32_t));
        reader.read(&max_dim_, sizeof(uint32_t));
        reader.read(&nr_inner_dims_, sizeof(uint32_t));
        reader.advance(index_file_v1_header_reserved_size);

        // Read sections
        uint32_t nr_sections = 0;
        reader.read(&nr_sections, sizeof(uint32_t));
        size_t sec_table_offset = reader.tellg();

        for (uint32_t i = 0; i < nr_sections; ++i) {
            DspSectionHeader section_header;
            reader.seekg(sec_table_offset);
            reader.read(&section_header, sizeof(DspSectionHeader));
            sec_table_offset += sizeof(DspSectionHeader);

            switch (section_header.type) {
                case DspSectionType::POSTING_LISTS: {
                    reader.seekg(section_header.offset);
                    uint32_t index_encoding_type = 0;
                    reader.read(&index_encoding_type, sizeof(uint32_t));
                    if (index_encoding_type != 0) {
                        return Status::invalid_serialized_index_type;
                    }
                    auto inverted_index_offsets_span = boost::span<const uint64_t>(
                        reinterpret_cast<uint64_t*>(reader.data() + reader.tellg()), nr_inner_dims_ + 1);
                    reader.advance(sizeof(uint64_t) * (nr_inner_dims_ + 1));
                    inverted_index_ids_spans_.resize(nr_inner_dims_);
                    inverted_index_vals_spans_.resize(nr_inner_dims_);
                    for (size_t j = 0; j < nr_inner_dims_; ++j) {
                        inverted_index_ids_spans_[j] = boost::span<const uint32_t>(
                            reinterpret_cast<uint32_t*>(reader.data() + reader.tellg()),
                            inverted_index_offsets_span[j + 1] - inverted_index_offsets_span[j]);
                        reader.advance(inverted_index_ids_spans_[j].size() * sizeof(uint32_t));
                    }
                    for (size_t j = 0; j < nr_inner_dims_; ++j) {
                        inverted_index_vals_spans_[j] = boost::span<const QType>(
                            reinterpret_cast<QType*>(reader.data() + reader.tellg()),
                            inverted_index_offsets_span[j + 1] - inverted_index_offsets_span[j]);
                        reader.advance(inverted_index_vals_spans_[j].size() * sizeof(QType));
                    }
                    break;
                }
                case DspSectionType::DIM_MAP: {
                    reader.seekg(section_header.offset);
                    for (uint32_t j = 0; j < nr_inner_dims_; ++j) {
                        uint32_t dim = 0;
                        reader.read(&dim, sizeof(uint32_t));
                        dim_map_[dim] = j;
                    }
                    break;
                }
                case DspSectionType::ROW_SUMS: {
                    reader.seekg(section_header.offset);
                    bm25_params_->row_sums_spans_ = boost::span<const float>(
                        reinterpret_cast<float*>(reader.data() + section_header.offset), n_rows_internal_);
                    reader.advance(sizeof(float) * n_rows_internal_);
                    break;
                }
                case DspSectionType::MAX_SCORES_PER_DIM: {
                    reader.seekg(section_header.offset);
                    max_score_in_dim_spans_ = boost::span<const float>(
                        reinterpret_cast<float*>(reader.data() + section_header.offset), nr_inner_dims_);
                    reader.advance(sizeof(float) * nr_inner_dims_);
                    break;
                }
                case DspSectionType::PROMETHEUS_BUILD_STATS: {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
                    reader.seekg(section_header.offset);
                    auto dataset_nnz_stats = std::vector<uint32_t>(n_rows_internal_);
                    reader.read(dataset_nnz_stats.data(), sizeof(uint32_t), n_rows_internal_);
                    auto posting_list_length_stats = std::vector<uint32_t>(nr_inner_dims_);
                    reader.read(posting_list_length_stats.data(), sizeof(uint32_t), nr_inner_dims_);
                    for (size_t j = 0; j < n_rows_internal_; ++j) {
                        index_dataset_nnz_len_histogram_->Observe(dataset_nnz_stats[j]);
                    }
                    for (size_t j = 0; j < nr_inner_dims_; ++j) {
                        index_posting_list_len_histogram_->Observe(posting_list_length_stats[j]);
                    }
#endif
                    break;
                }
                case DspSectionType::DSP_METADATA: {
                    reader.seekg(section_header.offset);

                    uint32_t dsp_version = 0;
                    reader.read(&dsp_version, sizeof(uint32_t));
                    if (dsp_version != 1) {
                        return Status::invalid_serialized_index_type;
                    }
                    reader.read(&n_subblocks_, sizeof(uint32_t));
                    reader.read(&n_superblocks_, sizeof(uint32_t));
                    reader.read(&n_sb_padded_, sizeof(uint32_t));

                    const uint32_t nr_dims = nr_inner_dims_;
                    dim_block_max_.resize(nr_dims);
                    for (uint32_t d = 0; d < nr_dims; ++d) {
                        auto& bm = dim_block_max_[d];
                        uint32_t n_block_ids = 0, n_logical = 0;
                        reader.read(&n_block_ids, sizeof(uint32_t));
                        reader.read(&n_logical, sizeof(uint32_t));
                        reader.read(bm.kth, 4);
                        bm.n_logical = n_logical;
                        if (n_block_ids > 0) {
                            bm.block_ids.resize(n_block_ids);
                            reader.read(bm.block_ids.data(), sizeof(uint32_t), n_block_ids);
                        }
                        uint32_t packed_size = 0;
                        reader.read(&packed_size, sizeof(uint32_t));
                        if (packed_size > 0) {
                            bm.max_scores.resize(packed_size);
                            reader.read(bm.max_scores.data(), sizeof(uint8_t), packed_size);
                        }
                    }

                    uint32_t spb_total = 0;
                    reader.read(&spb_total, sizeof(uint32_t));
                    spb_dim_offsets_.resize(nr_dims + 1);
                    reader.read(spb_dim_offsets_.data(), sizeof(uint32_t), nr_dims + 1);
                    spb_block_ids_.resize(spb_total);
                    reader.read(spb_block_ids_.data(), sizeof(uint32_t), spb_total);
                    spb_max_vals_.resize(spb_total);
                    reader.read(spb_max_vals_.data(), sizeof(float), spb_total);
                    spb_asc_vals_.resize(spb_total);
                    reader.read(spb_asc_vals_.data(), sizeof(float), spb_total);

                    uint32_t total_terms = 0, total_entries = 0;
                    reader.read(&total_terms, sizeof(uint32_t));
                    reader.read(&total_entries, sizeof(uint32_t));
                    fwd_block_term_offsets_.resize(n_subblocks_ + 1);
                    reader.read(fwd_block_term_offsets_.data(), sizeof(uint32_t), n_subblocks_ + 1);
                    fwd_term_ids_.resize(total_terms);
                    reader.read(fwd_term_ids_.data(), sizeof(uint32_t), total_terms);
                    fwd_term_entry_offsets_.resize(total_terms + 1);
                    reader.read(fwd_term_entry_offsets_.data(), sizeof(uint32_t), total_terms + 1);
                    fwd_doc_offsets_.resize(total_entries);
                    reader.read(fwd_doc_offsets_.data(), sizeof(uint8_t), total_entries);
                    fwd_scores_.resize(total_entries);
                    reader.read(fwd_scores_.data(), sizeof(float), total_entries);

                    dsp_loaded_ = true;
                    break;
                }
                default:
                    break;
            }
        }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        index_size_gauge_->Set((double)size() / 1024.0 / 1024.0);
#endif

        if (!dsp_loaded_) {
            build_dsp_metadata();
        }

        return Status::success;
    }

    void
    Search(const SparseRow<DType>& query, size_t k, float* distances, label_t* labels, const BitsetView& bitset,
           const DocValueComputer<float>& computer, DspSearchParams& approx_params) const override {
        std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
        std::fill(labels, labels + k, -1);
        if (query.size() == 0) {
            return;
        }

        auto q_vec = parse_query(query, approx_params.drop_ratio_search);
        if (q_vec.empty()) {
            return;
        }

        const size_t heap_capacity = k * approx_params.refine_factor;
        MaxMinHeap<float> heap(heap_capacity);
        search_dsp(q_vec, heap, heap_capacity, bitset, computer, approx_params.dsp_mode, approx_params.dsp_mu,
                   approx_params.dsp_eta, approx_params.dsp_gamma, approx_params.dsp_kth_init,
                   approx_params.dsp_kth_alpha);

        if (approx_params.refine_factor == 1) {
            collect_result(heap, distances, labels);
        } else {
            refine_and_collect(query, heap, k, distances, labels, computer);
        }
    }

    std::vector<float>
    GetAllDistances(const SparseRow<DType>& query, float drop_ratio_search, const BitsetView& bitset,
                    const DocValueComputer<float>& computer) const override {
        if (query.size() == 0) {
            return {};
        }
        std::vector<DType> values(query.size());
        for (size_t i = 0; i < query.size(); ++i) {
            values[i] = std::abs(query[i].val);
        }
        auto q_vec = parse_query(query, drop_ratio_search);

        auto distances = compute_all_distances(q_vec, computer);
        if (!bitset.empty()) {
            for (size_t i = 0; i < distances.size(); ++i) {
                if (bitset.test(i)) {
                    distances[i] = 0.0f;
                }
            }
        }
        return distances;
    }

    float
    GetRawDistance(const label_t vec_id, const SparseRow<DType>& query,
                   const DocValueComputer<float>& computer) const override {
        float distance = 0.0f;

        for (size_t i = 0; i < query.size(); ++i) {
            auto [dim, val] = query[i];
            auto dim_it = dim_map_.find(dim);
            if (dim_it == dim_map_.cend()) {
                continue;
            }
            auto& plist_ids = inverted_index_ids_spans_[dim_it->second];
            auto it = std::lower_bound(plist_ids.begin(), plist_ids.end(), vec_id,
                                       [](const auto& x, table_t y) { return x < y; });
            if (it != plist_ids.end() && *it == vec_id) {
                distance +=
                    val *
                    computer(inverted_index_vals_spans_[dim_it->second][it - plist_ids.begin()],
                             metric_type_ == SparseMetricType::METRIC_BM25 ? bm25_params_->row_sums_spans_[vec_id] : 0);
            }
        }

        return distance;
    }

    [[nodiscard]] size_t
    size() const override {
        size_t res = sizeof(*this);
        res += dim_map_.size() *
               (sizeof(typename decltype(dim_map_)::key_type) + sizeof(typename decltype(dim_map_)::mapped_type));

        if constexpr (mmapped) {
            return res + map_byte_size_;
        } else {
            res += sizeof(typename decltype(inverted_index_ids_spans_)::value_type) * inverted_index_ids_spans_.size();
            for (auto inverted_index_ids_span : inverted_index_ids_spans_) {
                res += sizeof(typename decltype(inverted_index_ids_spans_)::value_type::value_type) *
                       inverted_index_ids_span.size();
            }
            res +=
                sizeof(typename decltype(inverted_index_vals_spans_)::value_type) * inverted_index_vals_spans_.size();
            for (auto inverted_index_vals_span : inverted_index_vals_spans_) {
                res += sizeof(typename decltype(inverted_index_vals_spans_)::value_type::value_type) *
                       inverted_index_vals_span.size();
            }
            res += sizeof(typename decltype(max_score_in_dim_spans_)::value_type) * max_score_in_dim_spans_.size();
            return res;
        }
    }

    [[nodiscard]] size_t
    n_rows() const override {
        return n_rows_internal_;
    }

    [[nodiscard]] size_t
    n_cols() const override {
        return max_dim_;
    }

 private:
    // ========================================================================
    // Storage members (self-contained, no longer inherited from SparseInvertedStorage)
    // ========================================================================
    std::unordered_map<table_t, uint32_t> dim_map_;
    uint32_t nr_inner_dims_ = 0;

    Vector<Vector<table_t>> inverted_index_ids_;
    Vector<Vector<QType>> inverted_index_vals_;
    std::vector<boost::span<const table_t>> inverted_index_ids_spans_;
    std::vector<boost::span<const QType>> inverted_index_vals_spans_;
    Vector<float> max_score_in_dim_;
    boost::span<const float> max_score_in_dim_spans_;

    SparseMetricType metric_type_;

    size_t n_rows_internal_ = 0;
    size_t max_dim_ = 0;
    uint32_t next_dim_id_ = 0;

    char* map_ = nullptr;
    size_t map_byte_size_ = 0;
    int map_fd_ = -1;

    struct BM25Params {
        float k1;
        float b;
        Vector<float> row_sums;
        boost::span<const float> row_sums_spans_;

        DocValueComputer<float> max_score_computer;

        BM25Params(float k1, float b, float avgdl)
            : k1(k1), b(b), max_score_computer(GetDocValueBM25Computer<float>(k1, b, avgdl)) {
        }
    };

    std::unique_ptr<BM25Params> bm25_params_;

    static constexpr uint32_t index_file_v1_header_size = 32;
    static constexpr uint32_t index_file_v1_header_reserved_size = 16;

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    DspBuildStats build_stats_;

    std::string index_id_{};
    prometheus::Gauge* index_size_gauge_{nullptr};
    prometheus::Histogram* index_dataset_nnz_len_histogram_{nullptr};
    prometheus::Histogram* index_posting_list_len_histogram_{nullptr};
#endif

    // ========================================================================
    // Helper methods from SparseInvertedStorage
    // ========================================================================
    inline DType
    get_threshold(std::vector<DType>& values, float drop_ratio) const {
        auto drop_count = static_cast<size_t>(drop_ratio * values.size());
        if (drop_count == 0) {
            return 0;
        }
        auto pos = values.begin() + drop_count;
        std::nth_element(values.begin(), pos, values.end());
        return *pos;
    }

    std::vector<float>
    compute_all_distances(const std::vector<std::pair<size_t, DType>>& q_vec,
                          const DocValueComputer<float>& computer) const {
        std::vector<float> scores(n_rows_internal_, 0.0f);

        if (metric_type_ == SparseMetricType::METRIC_IP) {
            for (const auto& [dim_idx, q_weight] : q_vec) {
                const auto& plist_ids = inverted_index_ids_spans_[dim_idx];
                const auto& plist_vals = inverted_index_vals_spans_[dim_idx];

                accumulate_posting_list_contribution_ip_dispatch<QType>(
                    plist_ids.data(), plist_vals.data(), plist_ids.size(), static_cast<float>(q_weight), scores.data());
            }
        } else {
            const auto& doc_len_ratios = bm25_params_->row_sums_spans_;
            for (const auto& [dim_idx, q_weight] : q_vec) {
                const auto& plist_ids = inverted_index_ids_spans_[dim_idx];
                const auto& plist_vals = inverted_index_vals_spans_[dim_idx];
                const float q_weight_float = static_cast<float>(q_weight);
                for (size_t j = 0; j < plist_ids.size(); ++j) {
                    const auto doc_id = plist_ids[j];
                    const float doc_val = computer(plist_vals[j], doc_len_ratios[doc_id]);
                    scores[doc_id] += q_weight_float * doc_val;
                }
            }
        }

        return scores;
    }

    std::vector<std::pair<size_t, DType>>
    parse_query(const SparseRow<DType>& query, float drop_ratio_search) const {
        DType q_threshold = 0;
        if (drop_ratio_search != 0) {
            std::vector<DType> values(query.size());
            for (size_t i = 0; i < query.size(); ++i) {
                values[i] = std::abs(query[i].val);
            }
            q_threshold = get_threshold(values, drop_ratio_search);
        }

        std::vector<std::pair<size_t, DType>> filtered_query;
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

    template <typename HeapType>
    void
    collect_result(HeapType& heap, float* distances, label_t* labels) const {
        int cnt = heap.size();
        for (auto i = cnt - 1; i >= 0; --i) {
            labels[i] = heap.top().id;
            distances[i] = heap.top().val;
            heap.pop();
        }
    }

    inline void
    add_row_to_index(const SparseRow<DType>& row, table_t vec_id) {
        [[maybe_unused]] float row_sum = 0;
        for (size_t j = 0; j < row.size(); ++j) {
            auto [dim, val] = row[j];
            if (metric_type_ == SparseMetricType::METRIC_BM25) {
                row_sum += val;
            }
            if (val == 0) {
                continue;
            }
            auto dim_it = dim_map_.find(dim);
            if (dim_it == dim_map_.cend()) {
                if constexpr (mmapped) {
                    throw std::runtime_error("unexpected vector dimension in mmapped DspIndex");
                }
                dim_it = dim_map_.insert({dim, next_dim_id_++}).first;
                inverted_index_ids_.emplace_back();
                inverted_index_vals_.emplace_back();
                max_score_in_dim_.emplace_back(0.0f);
            }
            inverted_index_ids_[dim_it->second].emplace_back(vec_id);
            inverted_index_vals_[dim_it->second].emplace_back(get_quant_val(val));
        }
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        build_stats_.dataset_nnz_stats_.push_back(row.size());
#endif
        // update max_score_in_dim_
        for (size_t j = 0; j < row.size(); ++j) {
            auto [dim, val] = row[j];
            if (val == 0) {
                continue;
            }
            auto dim_it = dim_map_.find(dim);
            if (dim_it == dim_map_.cend()) {
                throw std::runtime_error("unexpected vector dimension in DspIndex");
            }
            auto score = static_cast<float>(val);
            if (metric_type_ == SparseMetricType::METRIC_BM25) {
                score = bm25_params_->max_score_computer(val, row_sum);
            }
            max_score_in_dim_[dim_it->second] = std::max(max_score_in_dim_[dim_it->second], score);
        }
        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            bm25_params_->row_sums.emplace_back(row_sum);
        }
    }

    inline QType
    get_quant_val(DType val) const {
        if constexpr (!std::is_same_v<QType, DType>) {
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

    Status
    PrepareMmap(MemoryIOReader& reader, size_t rows, int map_flags, const std::string& supplement_target_filename) {
        const auto initial_reader_location = reader.tellg();
        const auto nnz = (reader.remaining() - (rows * sizeof(size_t))) / SparseRow<DType>::element_size();

        std::unordered_map<table_t, size_t> idx_counts;
        for (size_t i = 0; i < rows; ++i) {
            size_t row_nnz;
            readBinaryPOD(reader, row_nnz);
            if (row_nnz == 0) {
                continue;
            }
            for (size_t j = 0; j < row_nnz; ++j) {
                table_t idx;
                readBinaryPOD(reader, idx);
                idx_counts[idx]++;
                reader.advance(sizeof(DType));
            }
        }
        reader.seekg(initial_reader_location);

        auto inverted_index_ids_byte_size =
            idx_counts.size() * sizeof(typename decltype(inverted_index_ids_)::value_type);
        auto inverted_index_vals_byte_size =
            idx_counts.size() * sizeof(typename decltype(inverted_index_vals_)::value_type);
        auto plists_ids_byte_size = nnz * sizeof(typename decltype(inverted_index_ids_)::value_type::value_type);
        auto plists_vals_byte_size = nnz * sizeof(typename decltype(inverted_index_vals_)::value_type::value_type);
        auto max_score_in_dim_byte_size = idx_counts.size() * sizeof(typename decltype(max_score_in_dim_)::value_type);
        size_t row_sums_byte_size = 0;

        map_byte_size_ =
            inverted_index_ids_byte_size + inverted_index_vals_byte_size + plists_ids_byte_size + plists_vals_byte_size;
        map_byte_size_ += max_score_in_dim_byte_size;
        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            row_sums_byte_size = rows * sizeof(typename decltype(bm25_params_->row_sums)::value_type);
            map_byte_size_ += row_sums_byte_size;
        }

        if (map_byte_size_ == 0) {
            return Status::success;
        }

        std::ofstream temp_file(supplement_target_filename, std::ios::binary | std::ios::trunc);
        if (!temp_file) {
            LOG_KNOWHERE_ERROR_ << "Failed to create mmap file when loading sparse DspIndex: " << strerror(errno);
            return Status::disk_file_error;
        }
        temp_file.close();

        std::filesystem::resize_file(supplement_target_filename, map_byte_size_);

        map_fd_ = open(supplement_target_filename.c_str(), O_RDWR);
        if (map_fd_ == -1) {
            LOG_KNOWHERE_ERROR_ << "Failed to open mmap file when loading sparse DspIndex: " << strerror(errno);
            return Status::disk_file_error;
        }
        std::filesystem::remove(supplement_target_filename);

        map_flags &= ~MAP_PRIVATE;
        map_flags |= MAP_SHARED;

        map_ = static_cast<char*>(mmap(nullptr, map_byte_size_, PROT_READ | PROT_WRITE, map_flags, map_fd_, 0));
        if (map_ == MAP_FAILED) {
            LOG_KNOWHERE_ERROR_ << "Failed to create mmap when loading sparse DspIndex: " << strerror(errno)
                                << ", size: " << map_byte_size_ << " on file: " << supplement_target_filename;
            return Status::disk_file_error;
        }
        if (madvise(map_, map_byte_size_, MADV_RANDOM) != 0) {
            LOG_KNOWHERE_WARNING_ << "Failed to madvise mmap when loading sparse DspIndex: " << strerror(errno);
        }

        char* ptr = map_;

        inverted_index_ids_.initialize(ptr, inverted_index_ids_byte_size);
        ptr += inverted_index_ids_byte_size;
        inverted_index_vals_.initialize(ptr, inverted_index_vals_byte_size);
        ptr += inverted_index_vals_byte_size;

        max_score_in_dim_.initialize(ptr, max_score_in_dim_byte_size);
        ptr += max_score_in_dim_byte_size;

        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            bm25_params_->row_sums.initialize(ptr, row_sums_byte_size);
            ptr += row_sums_byte_size;
        }

        for (const auto& [idx, count] : idx_counts) {
            auto& plist_ids = inverted_index_ids_.emplace_back();
            auto plist_ids_byte_size = count * sizeof(typename decltype(inverted_index_ids_)::value_type::value_type);
            plist_ids.initialize(ptr, plist_ids_byte_size);
            ptr += plist_ids_byte_size;
        }
        for (const auto& [idx, count] : idx_counts) {
            auto& plist_vals = inverted_index_vals_.emplace_back();
            auto plist_vals_byte_size = count * sizeof(typename decltype(inverted_index_vals_)::value_type::value_type);
            plist_vals.initialize(ptr, plist_vals_byte_size);
            ptr += plist_vals_byte_size;
        }
        size_t dim_id = 0;
        for (const auto& [idx, count] : idx_counts) {
            dim_map_[idx] = dim_id;
            max_score_in_dim_.emplace_back(0.0f);
            ++dim_id;
        }
        next_dim_id_ = dim_id;

        return Status::success;
    }

    // ========================================================================
    // DSP-specific members
    // ========================================================================
    void
    refine_and_collect(const SparseRow<DType>& query, MaxMinHeap<float>& inacc_heap, size_t k, float* distances,
                       label_t* labels, const DocValueComputer<float>& computer) const {
        MaxMinHeap<float> heap(k);
        while (!inacc_heap.empty()) {
            table_t doc_id = inacc_heap.pop();
            float score = GetRawDistance(doc_id, query, computer);
            heap.push(doc_id, score);
        }
        collect_result(heap, distances, labels);
    }

    struct DimBlockMax {
        std::vector<uint32_t> block_ids;
        std::vector<uint8_t> max_scores;
        uint32_t n_logical = 0;
        uint8_t kth[4] = {0, 0, 0, 0};
        bool
        is_dense() const {
            return block_ids.empty() && n_logical > 0;
        }
    };
    std::vector<DimBlockMax> dim_block_max_;

    // ========================================================================
    // Superblock max + ASC (sparse CSR format, float -- used for coarse pruning)
    // ========================================================================
    std::vector<uint32_t> spb_dim_offsets_;
    std::vector<uint32_t> spb_block_ids_;
    std::vector<float> spb_max_vals_;
    std::vector<float> spb_asc_vals_;

    // ========================================================================
    // Forward index (flat layout for cache-friendly scoring)
    // ========================================================================
    std::vector<uint32_t> fwd_block_term_offsets_;
    std::vector<uint32_t> fwd_term_ids_;
    std::vector<uint32_t> fwd_term_entry_offsets_;
    std::vector<uint8_t> fwd_doc_offsets_;
    std::vector<float> fwd_scores_;

    uint32_t n_subblocks_ = 0;
    uint32_t n_superblocks_ = 0;
    uint32_t n_sb_padded_ = 0;
    bool dsp_loaded_ = false;

    static constexpr float kDenseThreshold = 0.125f;

    static constexpr uint32_t kNumSegments = 8;
    static constexpr uint32_t kSegmentSize = kSuperblockSize / kNumSegments;

    // ========================================================================
    // Build DSP metadata from inverted index
    // ========================================================================
    void
    build_dsp_metadata() {
        if (n_rows_internal_ == 0 || nr_inner_dims_ == 0) {
            return;
        }

        n_subblocks_ = (n_rows_internal_ + kSubblockSize - 1) / kSubblockSize;
        n_superblocks_ = (n_rows_internal_ + kSuperblockSize - 1) / kSuperblockSize;
        n_sb_padded_ = (n_subblocks_ + kStride - 1) / kStride * kStride;

        const uint32_t nr_dims = nr_inner_dims_;
        const bool is_bm25 = metric_type_ == SparseMetricType::METRIC_BM25;

        // Per-doc forward index: (inner_dim, score) pairs appended per doc.
        struct DocFwdEntry {
            uint32_t inner_dim;
            float score;
        };
        std::vector<std::vector<DocFwdEntry>> per_doc_fwd(n_rows_internal_);

        std::vector<float> tmp_sb_max(n_subblocks_, 0.0f);
        std::vector<uint8_t> sb_touched(n_subblocks_, 0);
        std::vector<uint32_t> touched_list;
        touched_list.reserve(n_subblocks_);

        std::vector<float> tmp_spb_max(n_superblocks_, 0.0f);
        std::vector<uint8_t> spb_touched(n_superblocks_, 0);
        std::vector<uint32_t> spb_touched_list;
        spb_touched_list.reserve(n_superblocks_);

        std::vector<float> tmp_seg_max(n_superblocks_ * kNumSegments, 0.0f);

        struct SpbEntry {
            uint32_t block_id;
            float max_score;
            float asc;
        };
        std::vector<std::vector<SpbEntry>> per_dim_spb(nr_dims);

        dim_block_max_.resize(nr_dims);

        static constexpr uint32_t kKthSizes[4] = {10, 100, 1000, 10000};
        using KthHeap = std::priority_queue<float, std::vector<float>, std::greater<float>>;

        for (uint32_t d = 0; d < nr_dims; ++d) {
            const auto& plist_ids = inverted_index_ids_spans_[d];
            const auto& plist_vals = inverted_index_vals_spans_[d];
            const float max_score_d = max_score_in_dim_spans_[d];

            if (plist_ids.size() == 0 || max_score_d <= 0.0f) {
                continue;
            }

            const float inv_max_score_u8 = 255.0f / max_score_d;

            KthHeap kth_heaps[4];

            for (size_t i = 0; i < plist_ids.size(); ++i) {
                const uint32_t doc_id = plist_ids[i];
                const QType val = plist_vals[i];

                float score;
                if (is_bm25) {
                    score = bm25_params_->max_score_computer(val, bm25_params_->row_sums_spans_[doc_id]);
                } else {
                    score = static_cast<float>(val);
                }

                for (int h = 0; h < 4; ++h) {
                    if (kth_heaps[h].size() < kKthSizes[h]) {
                        kth_heaps[h].push(score);
                    } else if (score > kth_heaps[h].top()) {
                        kth_heaps[h].pop();
                        kth_heaps[h].push(score);
                    }
                }

                const uint32_t sb = doc_id / kSubblockSize;
                const uint32_t spb = doc_id / kSuperblockSize;

                if (!sb_touched[sb]) {
                    touched_list.push_back(sb);
                    sb_touched[sb] = 1;
                }
                tmp_sb_max[sb] = std::max(tmp_sb_max[sb], score);

                if (!spb_touched[spb]) {
                    spb_touched_list.push_back(spb);
                    spb_touched[spb] = 1;
                }
                tmp_spb_max[spb] = std::max(tmp_spb_max[spb], score);

                const uint32_t seg = doc_id / kSegmentSize;
                tmp_seg_max[seg] = std::max(tmp_seg_max[seg], score);

                per_doc_fwd[doc_id].push_back({d, score});
            }

            auto& bm = dim_block_max_[d];
            const size_t posting_len = plist_ids.size();
            for (int h = 0; h < 4; ++h) {
                if (posting_len >= kKthSizes[h] && !kth_heaps[h].empty()) {
                    float kth_f = kth_heaps[h].top();
                    bm.kth[h] = static_cast<uint8_t>(std::min(255.0f, std::floor(kth_f * inv_max_score_u8)));
                }
            }

            const uint32_t nnz_blocks = touched_list.size();
            if (nnz_blocks > static_cast<uint32_t>(n_subblocks_ * kDenseThreshold)) {
                bm.n_logical = n_sb_padded_;
                bm.max_scores.resize(n_sb_padded_, 0);
                for (uint32_t sb : touched_list) {
                    bm.max_scores[sb] =
                        static_cast<uint8_t>(std::min(255.0f, std::ceil(tmp_sb_max[sb] * inv_max_score_u8)));
                }
            } else {
                std::sort(touched_list.begin(), touched_list.end());
                bm.block_ids.resize(nnz_blocks);
                bm.n_logical = nnz_blocks;
                bm.max_scores.resize(nnz_blocks);
                for (uint32_t i = 0; i < nnz_blocks; ++i) {
                    uint32_t sb = touched_list[i];
                    bm.block_ids[i] = sb;
                    bm.max_scores[i] =
                        static_cast<uint8_t>(std::min(255.0f, std::ceil(tmp_sb_max[sb] * inv_max_score_u8)));
                }
            }

            std::sort(spb_touched_list.begin(), spb_touched_list.end());
            per_dim_spb[d].reserve(spb_touched_list.size());
            for (uint32_t spb : spb_touched_list) {
                float seg_sum = 0.0f;
                uint32_t seg_count = 0;
                for (uint32_t s = 0; s < kNumSegments; ++s) {
                    float seg_max = tmp_seg_max[spb * kNumSegments + s];
                    if (seg_max > 0.0f) {
                        seg_sum += seg_max;
                        seg_count++;
                    }
                }
                float asc = (seg_count > 0) ? (seg_sum / seg_count) : 0.0f;
                per_dim_spb[d].push_back({spb, tmp_spb_max[spb], asc});
            }

            for (uint32_t sb : touched_list) {
                tmp_sb_max[sb] = 0.0f;
                sb_touched[sb] = 0;
            }
            touched_list.clear();
            for (uint32_t spb : spb_touched_list) {
                tmp_spb_max[spb] = 0.0f;
                spb_touched[spb] = 0;
                for (uint32_t s = 0; s < kNumSegments; ++s) {
                    tmp_seg_max[spb * kNumSegments + s] = 0.0f;
                }
            }
            spb_touched_list.clear();
        }

        // ---- Phase 2: Build superblock CSR ----
        {
            uint32_t total_spb = 0;
            spb_dim_offsets_.resize(nr_dims + 1);
            for (uint32_t d = 0; d < nr_dims; ++d) {
                spb_dim_offsets_[d] = total_spb;
                total_spb += per_dim_spb[d].size();
            }
            spb_dim_offsets_[nr_dims] = total_spb;

            spb_block_ids_.resize(total_spb);
            spb_max_vals_.resize(total_spb);
            spb_asc_vals_.resize(total_spb);
            for (uint32_t d = 0; d < nr_dims; ++d) {
                uint32_t off = spb_dim_offsets_[d];
                for (const auto& e : per_dim_spb[d]) {
                    spb_block_ids_[off] = e.block_id;
                    spb_max_vals_[off] = e.max_score;
                    spb_asc_vals_[off] = e.asc;
                    ++off;
                }
            }
        }

        // ---- Phase 3: Build flat forward index from per-doc data ----
        {
            for (uint32_t doc = 0; doc < n_rows_internal_; ++doc) {
                auto& entries = per_doc_fwd[doc];
                if (entries.size() > 1) {
                    std::sort(entries.begin(), entries.end(),
                              [](const DocFwdEntry& a, const DocFwdEntry& b) { return a.inner_dim < b.inner_dim; });
                }
            }

            uint32_t total_terms = 0;
            uint32_t total_entries = 0;

            struct BlockEntry {
                uint32_t inner_dim;
                uint8_t doc_offset;
                float score;
            };
            std::vector<BlockEntry> block_buf;
            block_buf.reserve(1024);

            for (uint32_t sb = 0; sb < n_subblocks_; ++sb) {
                block_buf.clear();
                const uint32_t doc_start = sb * kSubblockSize;
                const uint32_t doc_end = std::min(doc_start + kSubblockSize, static_cast<uint32_t>(n_rows_internal_));
                for (uint32_t doc = doc_start; doc < doc_end; ++doc) {
                    const uint8_t doc_off = static_cast<uint8_t>(doc - doc_start);
                    for (const auto& e : per_doc_fwd[doc]) {
                        block_buf.push_back({e.inner_dim, doc_off, e.score});
                    }
                }
                if (block_buf.empty())
                    continue;
                std::sort(block_buf.begin(), block_buf.end(), [](const BlockEntry& a, const BlockEntry& b) {
                    return a.inner_dim < b.inner_dim || (a.inner_dim == b.inner_dim && a.doc_offset < b.doc_offset);
                });
                total_entries += block_buf.size();
                total_terms++;
                for (size_t i = 1; i < block_buf.size(); ++i) {
                    if (block_buf[i].inner_dim != block_buf[i - 1].inner_dim) {
                        total_terms++;
                    }
                }
            }

            fwd_block_term_offsets_.resize(n_subblocks_ + 1);
            fwd_term_ids_.resize(total_terms);
            fwd_term_entry_offsets_.resize(total_terms + 1);
            fwd_doc_offsets_.resize(total_entries);
            fwd_scores_.resize(total_entries);

            uint32_t term_pos = 0;
            uint32_t entry_pos = 0;

            for (uint32_t sb = 0; sb < n_subblocks_; ++sb) {
                fwd_block_term_offsets_[sb] = term_pos;
                block_buf.clear();
                const uint32_t doc_start = sb * kSubblockSize;
                const uint32_t doc_end = std::min(doc_start + kSubblockSize, static_cast<uint32_t>(n_rows_internal_));
                for (uint32_t doc = doc_start; doc < doc_end; ++doc) {
                    const uint8_t doc_off = static_cast<uint8_t>(doc - doc_start);
                    for (const auto& e : per_doc_fwd[doc]) {
                        block_buf.push_back({e.inner_dim, doc_off, e.score});
                    }
                    per_doc_fwd[doc].clear();
                    per_doc_fwd[doc].shrink_to_fit();
                }
                if (block_buf.empty())
                    continue;
                std::sort(block_buf.begin(), block_buf.end(), [](const BlockEntry& a, const BlockEntry& b) {
                    return a.inner_dim < b.inner_dim || (a.inner_dim == b.inner_dim && a.doc_offset < b.doc_offset);
                });

                fwd_term_ids_[term_pos] = block_buf[0].inner_dim;
                fwd_term_entry_offsets_[term_pos] = entry_pos;

                for (size_t i = 0; i < block_buf.size(); ++i) {
                    if (i > 0 && block_buf[i].inner_dim != block_buf[i - 1].inner_dim) {
                        term_pos++;
                        fwd_term_ids_[term_pos] = block_buf[i].inner_dim;
                        fwd_term_entry_offsets_[term_pos] = entry_pos;
                    }
                    fwd_doc_offsets_[entry_pos] = block_buf[i].doc_offset;
                    fwd_scores_[entry_pos] = block_buf[i].score;
                    entry_pos++;
                }
                term_pos++;
            }
            fwd_block_term_offsets_[n_subblocks_] = term_pos;
            fwd_term_entry_offsets_[total_terms] = entry_pos;
        }
    }

    // ========================================================================
    // DSP Search
    // ========================================================================
    template <typename DocIdFilter>
    void
    search_dsp(const std::vector<std::pair<size_t, DType>>& q_vec, MaxMinHeap<float>& heap, size_t heap_capacity,
               DocIdFilter& filter, const DocValueComputer<float>& computer, DspSearchMode mode, float mu, float eta,
               int gamma, bool kth_init = true, float kth_alpha = 1.0f) const {
        // ---- Step 0: Prepare sorted query ----
        struct QueryTerm {
            uint32_t inner_dim;
            float weight;
            uint8_t u8_weight;
        };
        std::vector<QueryTerm> query(q_vec.size());
        for (size_t i = 0; i < q_vec.size(); ++i) {
            query[i].inner_dim = static_cast<uint32_t>(q_vec[i].first);
            query[i].weight = static_cast<float>(q_vec[i].second);
        }
        std::sort(query.begin(), query.end(), [](const auto& a, const auto& b) { return a.inner_dim < b.inner_dim; });
        const size_t n_query_terms = query.size();

        // ---- Step 1: Compute u8 query weights and scale factor ----
        float S = 0.0f;
        for (const auto& qt : query) {
            S += qt.weight * max_score_in_dim_spans_[qt.inner_dim];
        }
        if (S <= 0.0f)
            return;

        const float inv_S = 255.0f / S;
        for (auto& qt : query) {
            float w = qt.weight * max_score_in_dim_spans_[qt.inner_dim] * inv_S;
            uint8_t u8w = static_cast<uint8_t>(std::min(255.0f, std::max(1.0f, std::ceil(w))));
            qt.u8_weight = u8w;
        }
        const float score_scale = 65025.0f / S;

        // ---- Step 2: Initialize thresholds from kth scores ----
        const bool has_filter = !filter.empty();
        bool bootstrap_mode = has_filter;

        float float_threshold = 0.0f;
        if (kth_init && !has_filter) {
            int kth_bucket = (heap_capacity > 10) + (heap_capacity > 100) + (heap_capacity > 1000);
            for (const auto& qt : query) {
                const auto& bm = dim_block_max_[qt.inner_dim];
                uint8_t kth_u8 = bm.kth[kth_bucket];
                if (kth_u8 == 0)
                    continue;
                float kth_float = kth_u8 / 255.0f * max_score_in_dim_spans_[qt.inner_dim];
                float term_thresh = qt.weight * kth_float;
                float_threshold = std::max(float_threshold, term_thresh);
            }
            float_threshold *= kth_alpha;
        }
        float float_block_threshold = (eta > 0.0f) ? float_threshold / eta : float_threshold;
        uint16_t u16_block_threshold = static_cast<uint16_t>(std::min(65535.0f, float_block_threshold * score_scale));

        // ---- Step 3: Superblock pruning ----
        std::vector<float> superblock_ub(n_superblocks_, 0.0f);
        std::vector<float> superblock_asc(n_superblocks_, 0.0f);
        for (const auto& qt : query) {
            const float qw = qt.weight;
            const uint32_t start = spb_dim_offsets_[qt.inner_dim];
            const uint32_t end = spb_dim_offsets_[qt.inner_dim + 1];
            for (uint32_t i = start; i < end; ++i) {
                superblock_ub[spb_block_ids_[i]] += qw * spb_max_vals_[i];
                superblock_asc[spb_block_ids_[i]] += qw * spb_asc_vals_[i];
            }
        }

        const float theta = float_threshold;
        float mu_threshold = (mu > 0.0f) ? theta / mu : theta;
        float eta_threshold = (eta > 0.0f) ? theta / eta : theta;

        std::vector<uint32_t> surviving_spb;
        surviving_spb.reserve(n_superblocks_);
        std::vector<uint8_t> spb_alive(n_superblocks_, 0);

        auto mark_alive = [&](uint32_t spb) {
            if (!spb_alive[spb]) {
                surviving_spb.push_back(spb);
                spb_alive[spb] = 1;
            }
        };

        auto add_top_gamma = [&](int g_val, float min_ub, bool inclusive = false) {
            uint32_t g = static_cast<uint32_t>(std::min(g_val, static_cast<int>(n_superblocks_)));
            std::vector<uint32_t> eligible;
            eligible.reserve(n_superblocks_);
            for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                if (inclusive ? (superblock_ub[spb] >= min_ub) : (superblock_ub[spb] > min_ub)) {
                    eligible.push_back(spb);
                }
            }
            if (eligible.size() <= g) {
                for (uint32_t spb : eligible) {
                    mark_alive(spb);
                }
            } else {
                std::nth_element(eligible.begin(), eligible.begin() + g, eligible.end(),
                                 [&](uint32_t a, uint32_t b) { return superblock_ub[a] > superblock_ub[b]; });
                for (uint32_t i = 0; i < g; ++i) {
                    mark_alive(eligible[i]);
                }
            }
        };

        // ---- Mode-driven superblock selection ----
        // DspSearchMode enum values:
        //   DSP = 0, LSP0 = 1, LSP1 = 2, LSP2 = 3
        auto select_superblocks_by_mode = [&]() {
            switch (mode) {
                case DspSearchMode::DSP: {
                    // dual-threshold (mu, eta) + optional top-gamma backstop
                    for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                        if (superblock_ub[spb] > mu_threshold || superblock_asc[spb] > eta_threshold) {
                            mark_alive(spb);
                        }
                    }
                    if (gamma > 0) {
                        add_top_gamma(gamma, 0.0f);
                    } else {
                        uint32_t top2_ids[2] = {UINT32_MAX, UINT32_MAX};
                        float top2_ub[2] = {0.0f, 0.0f};
                        for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                            if (spb_alive[spb])
                                continue;
                            float ub = superblock_ub[spb];
                            if (ub > top2_ub[0]) {
                                top2_ub[1] = top2_ub[0];
                                top2_ids[1] = top2_ids[0];
                                top2_ub[0] = ub;
                                top2_ids[0] = spb;
                            } else if (ub > top2_ub[1]) {
                                top2_ub[1] = ub;
                                top2_ids[1] = spb;
                            }
                        }
                        for (int i = 0; i < 2; ++i) {
                            if (top2_ids[i] != UINT32_MAX) {
                                mark_alive(top2_ids[i]);
                            }
                        }
                    }
                    break;
                }
                case DspSearchMode::LSP0: {
                    if (gamma <= 0) {
                        // fallback to DSP behavior
                        for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                            if (superblock_ub[spb] > mu_threshold || superblock_asc[spb] > eta_threshold) {
                                mark_alive(spb);
                            }
                        }
                        break;
                    }
                    add_top_gamma(gamma, float_threshold, true);
                    break;
                }
                case DspSearchMode::LSP1: {
                    if (gamma <= 0) {
                        // fallback to DSP behavior
                        for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                            if (superblock_ub[spb] > mu_threshold || superblock_asc[spb] > eta_threshold) {
                                mark_alive(spb);
                            }
                        }
                        break;
                    }
                    add_top_gamma(gamma, float_threshold, true);
                    for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                        if (superblock_ub[spb] > mu_threshold) {
                            mark_alive(spb);
                        }
                    }
                    break;
                }
                case DspSearchMode::LSP2: {
                    if (gamma <= 0) {
                        // fallback to DSP behavior
                        for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                            if (superblock_ub[spb] > mu_threshold || superblock_asc[spb] > eta_threshold) {
                                mark_alive(spb);
                            }
                        }
                        break;
                    }
                    add_top_gamma(gamma, float_threshold, true);
                    for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                        if (superblock_ub[spb] > mu_threshold || superblock_asc[spb] > eta_threshold) {
                            mark_alive(spb);
                        }
                    }
                    break;
                }
                default: {
                    // Default: same as DSP mode
                    for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                        if (superblock_ub[spb] > mu_threshold || superblock_asc[spb] > eta_threshold) {
                            mark_alive(spb);
                        }
                    }
                    if (gamma > 0) {
                        add_top_gamma(gamma, 0.0f);
                    }
                    break;
                }
            }
        };

        // ---- Compute block UBs for a set of superblocks ----
        std::vector<uint16_t> block_ub(n_sb_padded_, 0);
        std::vector<uint8_t> spb_in_batch(n_superblocks_, 0);

        auto compute_block_ubs = [&](const std::vector<uint32_t>& spbs) {
            for (uint32_t spb : spbs) spb_in_batch[spb] = 1;

            for (const auto& qt : query) {
                const auto& bm = dim_block_max_[qt.inner_dim];
                if (bm.n_logical == 0)
                    continue;
                if (bm.is_dense()) {
                    for (uint32_t spb : spbs) {
                        const uint32_t sb_start = spb * kStride;
                        accumulate_block_ub_dispatch(block_ub.data() + sb_start, bm.max_scores.data() + sb_start,
                                                     static_cast<uint16_t>(qt.u8_weight), kStride);
                    }
                } else {
                    const uint16_t u16w = static_cast<uint16_t>(qt.u8_weight);
                    for (size_t i = 0; i < bm.block_ids.size(); ++i) {
                        const uint32_t sb = bm.block_ids[i];
                        if (!spb_in_batch[sb / kStride])
                            continue;
                        uint32_t prod = u16w * bm.max_scores[i];
                        uint32_t sum = static_cast<uint32_t>(block_ub[sb]) + prod;
                        block_ub[sb] = static_cast<uint16_t>(sum < 65535u ? sum : 65535u);
                    }
                }
            }

            for (uint32_t spb : spbs) spb_in_batch[spb] = 0;
        };

        // ---- Collect candidates from superblocks and sort by UB descending ----
        auto collect_and_sort = [&](const std::vector<uint32_t>& spbs) -> std::vector<uint32_t> {
            std::vector<uint32_t> cands;
            cands.reserve(spbs.size() * kStride / 4);
            uint16_t local_max_ub = 0;
            for (uint32_t spb : spbs) {
                const uint32_t sb_start = spb * kStride;
                if (!scan_block_ub_any_above_dispatch(block_ub.data() + sb_start, u16_block_threshold, kStride))
                    continue;
                const uint32_t sb_end = std::min(sb_start + kStride, n_subblocks_);
                for (uint32_t sb = sb_start; sb < sb_end; ++sb) {
                    if (block_ub[sb] > u16_block_threshold) {
                        cands.push_back(sb);
                        local_max_ub = std::max(local_max_ub, block_ub[sb]);
                    }
                }
            }
            if (cands.empty())
                return {};
            const uint32_t rng = local_max_ub - u16_block_threshold;
            std::vector<uint32_t> cnt(rng + 1, 0);
            for (uint32_t sb : cands) cnt[block_ub[sb] - u16_block_threshold - 1]++;
            uint32_t p = 0;
            for (int b = static_cast<int>(rng); b >= 0; --b) {
                uint32_t c = cnt[b];
                cnt[b] = p;
                p += c;
            }
            std::vector<uint32_t> sorted(cands.size());
            for (uint32_t sb : cands) sorted[cnt[block_ub[sb] - u16_block_threshold - 1]++] = sb;
            return sorted;
        };

        // ---- Score a list of sorted blocks, updating heap and thresholds ----
        float scores[kSubblockSize];

        auto score_blocks = [&](const std::vector<uint32_t>& sorted_blocks) -> bool {
            bool bootstrap_completed = false;
            for (size_t ci = 0; ci < sorted_blocks.size(); ++ci) {
                const uint32_t sb_id = sorted_blocks[ci];
                if (block_ub[sb_id] <= u16_block_threshold)
                    break;

                const uint32_t block_term_start = fwd_block_term_offsets_[sb_id];
                const uint32_t block_term_end = fwd_block_term_offsets_[sb_id + 1];
                if (block_term_start == block_term_end)
                    continue;

                if (ci + 1 < sorted_blocks.size()) {
                    const uint32_t next_sb = sorted_blocks[ci + 1];
                    const uint32_t next_start = fwd_block_term_offsets_[next_sb];
                    __builtin_prefetch(&fwd_term_ids_[next_start], 0, 1);
                    const uint32_t next_entry_start = fwd_term_entry_offsets_[next_start];
                    __builtin_prefetch(&fwd_doc_offsets_[next_entry_start], 0, 0);
                    __builtin_prefetch(&fwd_scores_[next_entry_start], 0, 0);
                }

                std::memset(scores, 0, sizeof(scores));
                size_t qi = 0;
                uint32_t bi = block_term_start;
                while (qi < n_query_terms && bi < block_term_end) {
                    const uint32_t q_dim = query[qi].inner_dim;
                    const uint32_t b_dim = fwd_term_ids_[bi];
                    if (q_dim < b_dim) {
                        ++qi;
                    } else if (q_dim > b_dim) {
                        ++bi;
                    } else {
                        const float q_weight = query[qi].weight;
                        const uint32_t e_start = fwd_term_entry_offsets_[bi];
                        const uint32_t e_end = fwd_term_entry_offsets_[bi + 1];
                        for (uint32_t j = e_start; j < e_end; ++j) {
                            scores[fwd_doc_offsets_[j]] += q_weight * fwd_scores_[j];
                        }
                        ++qi;
                        ++bi;
                    }
                }

                const uint32_t doc_base = sb_id * kSubblockSize;
                const uint32_t doc_end = std::min(doc_base + kSubblockSize, static_cast<uint32_t>(n_rows_internal_));
                for (uint32_t i = 0; i < doc_end - doc_base; ++i) {
                    if (scores[i] > float_threshold) {
                        const uint32_t doc_id = doc_base + i;
                        if (has_filter && filter.test(doc_id))
                            continue;
                        heap.push(doc_id, scores[i]);
                        if (heap.full()) {
                            float new_thresh = heap.top().val;
                            if (new_thresh > float_threshold) {
                                float_threshold = new_thresh;
                                float_block_threshold = (eta > 0.0f) ? float_threshold / eta : float_threshold;
                                u16_block_threshold =
                                    static_cast<uint16_t>(std::min(65535.0f, float_block_threshold * score_scale));
                            }
                            if (bootstrap_mode) {
                                bootstrap_mode = false;
                                bootstrap_completed = true;
                                mu_threshold = (mu > 0.0f) ? float_threshold / mu : float_threshold;
                                eta_threshold = (eta > 0.0f) ? float_threshold / eta : float_threshold;
                            }
                        }
                    }
                }
            }
            return bootstrap_completed;
        };

        // ====================================================================
        // Two-phase filtered bootstrap
        // ====================================================================
        if (has_filter) {
            std::vector<uint32_t> spb_by_ub;
            spb_by_ub.reserve(n_superblocks_);
            for (uint32_t spb = 0; spb < n_superblocks_; ++spb) {
                if (superblock_ub[spb] > 0.0f)
                    spb_by_ub.push_back(spb);
            }
            std::sort(spb_by_ub.begin(), spb_by_ub.end(),
                      [&](uint32_t a, uint32_t b) { return superblock_ub[a] > superblock_ub[b]; });

            std::vector<uint8_t> spb_processed(n_superblocks_, 0);

            const uint32_t batch_sizes[] = {64, 256};
            const int n_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
            uint32_t cursor = 0;

            for (int batch_idx = 0; batch_idx <= n_batches && cursor < spb_by_ub.size(); ++batch_idx) {
                uint32_t batch_end;
                if (batch_idx < n_batches) {
                    batch_end = std::min(static_cast<uint32_t>(spb_by_ub.size()), batch_sizes[batch_idx]);
                } else {
                    batch_end = static_cast<uint32_t>(spb_by_ub.size());
                }
                if (batch_end <= cursor)
                    continue;

                std::vector<uint32_t> batch_spbs;
                batch_spbs.reserve(batch_end - cursor);
                for (uint32_t i = cursor; i < batch_end; ++i) {
                    mark_alive(spb_by_ub[i]);
                    spb_processed[spb_by_ub[i]] = 1;
                    batch_spbs.push_back(spb_by_ub[i]);
                }
                cursor = batch_end;

                compute_block_ubs(batch_spbs);
                auto sorted = collect_and_sort(batch_spbs);
                if (sorted.empty())
                    continue;
                bool done = score_blocks(sorted);
                if (done)
                    break;
            }

            if (!bootstrap_mode) {
                surviving_spb.clear();
                std::fill(spb_alive.begin(), spb_alive.end(), 0);
                select_superblocks_by_mode();

                std::vector<uint32_t> new_spbs;
                new_spbs.reserve(surviving_spb.size());
                for (uint32_t spb : surviving_spb) {
                    if (!spb_processed[spb])
                        new_spbs.push_back(spb);
                }

                if (!new_spbs.empty()) {
                    compute_block_ubs(new_spbs);
                    auto sorted = collect_and_sort(new_spbs);
                    if (!sorted.empty())
                        score_blocks(sorted);
                }
            }
        } else {
            // ====================================================================
            // Unfiltered path: original single-pass logic
            // ====================================================================
            select_superblocks_by_mode();
            if (surviving_spb.empty())
                return;

            compute_block_ubs(surviving_spb);
            auto sorted = collect_and_sort(surviving_spb);
            if (sorted.empty())
                return;
            score_blocks(sorted);
        }
    }
};

}  // namespace knowhere::sparse

#endif  // SPARSE_DSP_INDEX_H
