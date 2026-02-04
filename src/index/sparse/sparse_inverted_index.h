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

#ifndef SPARSE_INVERTED_INDEX_H
#define SPARSE_INVERTED_INDEX_H

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <boost/core/span.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "index/sparse/sparse_inverted_index_config.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/utils.h"
#include "simd/instruction_set.h"
#include "simd/sparse_simd.h"

namespace knowhere::sparse {

enum class InvertedIndexAlgo {
    TAAT_NAIVE,
    DAAT_WAND,
    DAAT_MAXSCORE,
    DAAT_MAXSCORE_V2,  // SIMD-optimized MaxScore with aligned posting lists
};

struct InvertedIndexBuildStats {
    std::vector<uint32_t> dataset_nnz_stats_;
    std::vector<uint32_t> posting_list_length_stats_;
};

enum class InvertedIndexSectionType : uint32_t {
    POSTING_LISTS = 0,
    METRIC_PARAMS = 1,
    DIM_MAP = 2,
    ROW_SUMS = 3,
    MAX_SCORES_PER_DIM = 4,
    PROMETHEUS_BUILD_STATS = 5
};

struct InvertedIndexSectionHeader {
    InvertedIndexSectionType type;
    uint64_t offset;
    uint64_t size;
};

struct InvertedIndexApproxSearchParams {
    int refine_factor;
    float drop_ratio_search;
    float dim_max_score_ratio;
};

template <typename T>
class BaseInvertedIndex {
 public:
    virtual ~BaseInvertedIndex() = default;

    virtual Status
    SerializeV0(MemoryIOWriter& writer) const = 0;

    // supplement_target_filename: when in mmap mode, we need an extra file to store the mmapped index data structure.
    // this file will be created during loading and deleted in the destructor.
    virtual Status
    DeserializeV0(MemoryIOReader& reader, int map_flags, const std::string& supplement_target_filename) = 0;

    virtual Status
    Serialize(MemoryIOWriter& writer) const = 0;

    virtual Status
    Deserialize(MemoryIOReader& reader) = 0;

    virtual Status
    Train(const SparseRow<T>* data, size_t rows) = 0;

    virtual Status
    Add(const SparseRow<T>* data, size_t rows, int64_t dim) = 0;

    virtual void
    Search(const SparseRow<T>& query, size_t k, float* distances, label_t* labels, const BitsetView& bitset,
           const DocValueComputer<T>& computer, InvertedIndexApproxSearchParams& approx_params) const = 0;

    virtual std::vector<float>
    GetAllDistances(const SparseRow<T>& query, float drop_ratio_search, const BitsetView& bitset,
                    const DocValueComputer<T>& computer) const = 0;

    virtual float
    GetRawDistance(const label_t vec_id, const SparseRow<T>& query, const DocValueComputer<T>& computer) const = 0;

    virtual expected<DocValueComputer<T>>
    GetDocValueComputer(const SparseInvertedIndexConfig& cfg) const = 0;

    [[nodiscard]] virtual size_t
    size() const = 0;

    [[nodiscard]] virtual size_t
    n_rows() const = 0;

    [[nodiscard]] virtual size_t
    n_cols() const = 0;
};

template <typename DType, typename QType, InvertedIndexAlgo algo, bool mmapped = false>
class InvertedIndex : public BaseInvertedIndex<DType> {
 public:
    explicit InvertedIndex(SparseMetricType metric_type) : metric_type_(metric_type) {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        // for now, use timestamp as index_id
        index_id_ = std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());
        index_size_gauge_ =
            &sparse_inverted_index_size_family.Add({{"index_id", index_id_}, {"index_type", "inverted"}});
        index_dataset_nnz_len_histogram_ =
            &sparse_dataset_nnz_len_family.Add({{"index_id", index_id_}, {"index_type", "inverted"}}, defaultBuckets);
        index_posting_list_len_histogram_ = &sparse_inverted_index_posting_list_len_family.Add(
            {{"index_id", index_id_}, {"index_type", "inverted"}}, defaultBuckets);
#endif
    }

    ~InvertedIndex() override {
        if constexpr (mmapped) {
            if (map_ != nullptr) {
                auto res = munmap(map_, map_byte_size_);
                if (res != 0) {
                    LOG_KNOWHERE_ERROR_ << "Failed to munmap when deleting sparse InvertedIndex: " << strerror(errno);
                }
                map_ = nullptr;
                map_byte_size_ = 0;
            }
            if (map_fd_ != -1) {
                // closing the file descriptor will also cause the file to be deleted.
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

    template <typename U>
    using Vector = std::conditional_t<mmapped, GrowableVectorView<U>, std::vector<U>>;

    void
    SetBM25Params(float k1, float b, float avgdl) {
        bm25_params_ = std::make_unique<BM25Params>(k1, b, avgdl);
    }

    expected<DocValueComputer<float>>
    GetDocValueComputer(const SparseInvertedIndexConfig& cfg) const override {
        // if metric_type is set in config, it must match with how the index was built.
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
        // avgdl must be supplied during search
        if (!cfg.bm25_avgdl.has_value()) {
            return expected<DocValueComputer<float>>::Err(Status::invalid_args,
                                                          "avgdl must be supplied during searching");
        }
        auto avgdl = cfg.bm25_avgdl.value();
        avgdl = std::max(avgdl, 1.0f);
        if constexpr (algo == InvertedIndexAlgo::DAAT_WAND || algo == InvertedIndexAlgo::DAAT_MAXSCORE ||
                      algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
            // daat_wand and daat_maxscore: search time k1/b must equal load time config.
            if ((cfg.bm25_k1.has_value() && cfg.bm25_k1.value() != bm25_params_->k1) ||
                ((cfg.bm25_b.has_value() && cfg.bm25_b.value() != bm25_params_->b))) {
                return expected<DocValueComputer<float>>::Err(
                    Status::invalid_args,
                    "search time k1/b must equal load time config for DAAT_WAND or DAAT_MAXSCORE algorithm.");
            }
            return GetDocValueBM25Computer<float>(bm25_params_->k1, bm25_params_->b, avgdl);
        } else {
            // inverted index: search time k1/b may override load time config.
            auto k1 = cfg.bm25_k1.has_value() ? cfg.bm25_k1.value() : bm25_params_->k1;
            auto b = cfg.bm25_b.has_value() ? cfg.bm25_b.value() : bm25_params_->b;
            return GetDocValueBM25Computer<float>(k1, b, avgdl);
        }
    }

    Status
    SerializeV0(MemoryIOWriter& writer) const override {
        /**
         * Layout:
         *
         * 1. size_t rows
         * 2. size_t cols
         * 3. DType value_threshold_ (deprecated)
         * 4. for each row:
         *     1. size_t len
         *     2. for each non-zero value:
         *        1. table_t idx
         *        2. DType val (when QType is different from DType, the QType value of val is stored as a DType with
         *           precision loss)
         *
         * inverted_index_ids_spans_, inverted_index_vals_spans_ and max_score_in_dim_spans_ are
         * not serialized, they will be constructed dynamically during
         * deserialization.
         *
         * Data are densely packed in serialized bytes and no padding is added.
         */
        DType deprecated_value_threshold = 0;
        writeBinaryPOD(writer, n_rows_internal_);
        writeBinaryPOD(writer, max_dim_);
        writeBinaryPOD(writer, deprecated_value_threshold);
        BitsetView bitset(nullptr, 0);

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
        // previous versions used the signness of rows to indicate whether to
        // use wand. now we use a template parameter to control this thus simply
        // take the absolute value of rows.
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
                LOG_KNOWHERE_INFO_ << "Sparse Inverted Index loading progress: " << (i / load_progress_interval * 10)
                                   << "%";
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
        LOG_KNOWHERE_INFO_ << "Sparse Inverted Index loading progress: 100%";

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
        // mapping data to spans
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

        return Status::success;
    }

    Status
    Serialize(MemoryIOWriter& writer) const override {
        // Serialized format:
        // 1. Index File Header (v1) (32 bytes):
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
        //    - index_encoding_type (uint32_t): Type of encoding used, only 0 is supported for now
        //    - encoded_index_data: Flattened posting lists
        //
        // 4. Dimension Map Section:
        //    - dim_map_reverse[nr_inner_dims]: Array mapping internal dimension IDs to original dimensions
        //
        // 5. Optional Row Sums Section:
        //    - row_sums[nr_rows]: Array of row sums (float)
        //
        // 6. Optional Max Scores Per Dimension Section:
        //    - max_score_per_dim[nr_inner_dims]: Array of maximum scores per dimension (float)

        // write index header data
        const uint32_t index_format_version = 1;

        // Index File Header (v1)
        writer.write(&index_format_version, sizeof(uint32_t));    // index format version
        writer.write(&this->n_rows_internal_, sizeof(uint32_t));  // number of rows
        writer.write(&this->max_dim_, sizeof(uint32_t));          // number of cols, or maximum dimension id
        writer.write(&this->nr_inner_dims_, sizeof(uint32_t));    // number of inner dimensions
        auto reserved =
            std::array<uint8_t, InvertedIndex::index_file_v1_header_reserved_size>();  // reserved for future use
        writer.write(reserved.data(), reserved.size());

        // Section Headers Table
        uint32_t nr_sections = 2;  // base sections: inverted index and dim map
        // Count additional sections based on flags in a single operation
        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            nr_sections += 1;  // row sums
        }
        if (max_score_in_dim_spans_.size() > 0) {
            nr_sections += 1;  // max scores per dim
        }
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
        // use a section to store some build stats for prometheus
        nr_sections += 1;
#endif
        writer.write(&nr_sections, sizeof(uint32_t));

        // since writer doesn't support seekp() for now, calculate all sizes of each sections first
        std::vector<InvertedIndexSectionHeader> section_headers(nr_sections);
        uint64_t used_offset = InvertedIndex::index_file_v1_header_size + sizeof(uint32_t) +
                               sizeof(InvertedIndexSectionHeader) * nr_sections;
        section_headers[0].type = InvertedIndexSectionType::POSTING_LISTS;
        section_headers[0].offset = used_offset;
        uint64_t posting_lists_size = sizeof(uint32_t);                       // used to store encoding type
        posting_lists_size += sizeof(uint64_t) * (this->nr_inner_dims_ + 1);  // used to store dim offsets
        for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
            posting_lists_size += this->inverted_index_ids_spans_[i].size() * sizeof(uint32_t) +
                                  this->inverted_index_vals_spans_[i].size() * sizeof(QType);
        }
        section_headers[0].size = posting_lists_size;
        used_offset += section_headers[0].size;

        section_headers[1].type = InvertedIndexSectionType::DIM_MAP;
        section_headers[1].offset = used_offset;
        section_headers[1].size = sizeof(uint32_t) * this->nr_inner_dims_;
        used_offset += section_headers[1].size;

        uint32_t curr_section_idx = 2;
        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            section_headers[curr_section_idx].type = InvertedIndexSectionType::ROW_SUMS;
            section_headers[curr_section_idx].offset = used_offset;
            section_headers[curr_section_idx].size = sizeof(float) * n_rows_internal_;
            used_offset += section_headers[curr_section_idx].size;
            curr_section_idx++;
        }

        if (max_score_in_dim_spans_.size() > 0) {
            section_headers[curr_section_idx].type = InvertedIndexSectionType::MAX_SCORES_PER_DIM;
            section_headers[curr_section_idx].offset = used_offset;
            section_headers[curr_section_idx].size = sizeof(float) * this->nr_inner_dims_;
            used_offset += section_headers[curr_section_idx].size;
            curr_section_idx++;
        }

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
        section_headers[curr_section_idx].type = InvertedIndexSectionType::PROMETHEUS_BUILD_STATS;
        section_headers[curr_section_idx].offset = used_offset;
        section_headers[curr_section_idx].size =
            sizeof(uint32_t) * this->n_rows_internal_ + sizeof(uint32_t) * this->nr_inner_dims_;
        used_offset += section_headers[curr_section_idx].size;
        curr_section_idx++;
#endif

        assert(curr_section_idx == nr_sections);

        // write section headers table
        writer.write(section_headers.data(), sizeof(InvertedIndexSectionHeader), nr_sections);

        // write index encoding type and index
        uint32_t index_encoding_type = 0;  // not used for now
        writer.write(&index_encoding_type, sizeof(uint32_t));
        std::vector<uint64_t> inverted_index_offsets(this->nr_inner_dims_ + 1);
        inverted_index_offsets[0] = 0;
        for (size_t i = 1; i <= this->nr_inner_dims_; ++i) {
            inverted_index_offsets[i] = inverted_index_offsets[i - 1] + this->inverted_index_ids_spans_[i - 1].size();
        }
        writer.write(inverted_index_offsets.data(), sizeof(uint64_t), inverted_index_offsets.size());
        for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
            writer.write(this->inverted_index_ids_spans_[i].data(), sizeof(uint32_t),
                         this->inverted_index_ids_spans_[i].size());
        }
        for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
            writer.write(this->inverted_index_vals_spans_[i].data(), sizeof(QType),
                         this->inverted_index_vals_spans_[i].size());
        }

        // write dim map
        auto dim_map_reverse = std::vector<uint32_t>(this->nr_inner_dims_);
        for (const auto& [dim, dim_id] : this->dim_map_) {
            dim_map_reverse[dim_id] = dim;
        }
        writer.write(dim_map_reverse.data(), sizeof(uint32_t), this->nr_inner_dims_);

        // write index meta data
        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            writer.write(bm25_params_->row_sums_spans_.data(), sizeof(float), this->n_rows_internal_);
        }

        if (max_score_in_dim_spans_.size() > 0) {
            writer.write(max_score_in_dim_spans_.data(), sizeof(float), this->nr_inner_dims_);
        }

        // write prometheus build stats
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOHWERE_WITH_LIGHT)
        writer.write(this->build_stats_.dataset_nnz_stats_.data(), sizeof(uint32_t), this->n_rows_internal_);
        writer.write(this->build_stats_.posting_list_length_stats_.data(), sizeof(uint32_t), this->nr_inner_dims_);
#endif

        return Status::success;
    }

    Status
    Deserialize(MemoryIOReader& reader) override {
        auto file_header_handler = [&]() {
            uint32_t index_format_version = 0;
            reader.read(&index_format_version, sizeof(uint32_t));
            // for now we only support version 1
            if (index_format_version != 1) {
                return Status::invalid_serialized_index_type;
            }

            reader.read(&this->n_rows_internal_, sizeof(uint32_t));
            reader.read(&this->max_dim_, sizeof(uint32_t));
            reader.read(&this->nr_inner_dims_, sizeof(uint32_t));
            // skip reserved bytes
            reader.advance(InvertedIndex::index_file_v1_header_reserved_size);

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
                        // check index encoding type, 0 is the only supported type for now
                        uint32_t index_encoding_type = 0;
                        reader.read(&index_encoding_type, sizeof(uint32_t));
                        if (index_encoding_type != 0) {
                            return Status::invalid_serialized_index_type;
                        }
                        auto inverted_index_offsets_span = boost::span<const uint64_t>(
                            reinterpret_cast<uint64_t*>(reader.data() + reader.tellg()), this->nr_inner_dims_ + 1);
                        reader.advance(sizeof(uint64_t) * (this->nr_inner_dims_ + 1));
                        inverted_index_ids_spans_.resize(this->nr_inner_dims_);
                        inverted_index_vals_spans_.resize(this->nr_inner_dims_);
                        for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
                            inverted_index_ids_spans_[i] = boost::span<const uint32_t>(
                                reinterpret_cast<uint32_t*>(reader.data() + reader.tellg()),
                                inverted_index_offsets_span[i + 1] - inverted_index_offsets_span[i]);
                            reader.advance(inverted_index_ids_spans_[i].size() * sizeof(uint32_t));
                        }
                        for (size_t i = 0; i < this->nr_inner_dims_; ++i) {
                            inverted_index_vals_spans_[i] = boost::span<const QType>(
                                reinterpret_cast<QType*>(reader.data() + reader.tellg()),
                                inverted_index_offsets_span[i + 1] - inverted_index_offsets_span[i]);
                            reader.advance(inverted_index_vals_spans_[i].size() * sizeof(QType));
                        }
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
                        bm25_params_->row_sums_spans_ = boost::span<const float>(
                            reinterpret_cast<float*>(reader.data() + section_header.offset), this->n_rows_internal_);
                        reader.advance(sizeof(float) * this->n_rows_internal_);
                        break;
                    }
                    case InvertedIndexSectionType::MAX_SCORES_PER_DIM: {
                        reader.seekg(section_header.offset);
                        max_score_in_dim_spans_ = boost::span<const float>(
                            reinterpret_cast<float*>(reader.data() + section_header.offset), this->nr_inner_dims_);
                        reader.advance(sizeof(float) * this->nr_inner_dims_);
                        break;
                    }
                    case InvertedIndexSectionType::PROMETHEUS_BUILD_STATS: {
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
                        reader.seekg(section_header.offset);
                        auto dataset_nnz_stats = std::vector<uint32_t>(this->n_rows_internal_);
                        reader.read(dataset_nnz_stats.data(), sizeof(uint32_t), this->n_rows_internal_);
                        auto posting_list_length_stats = std::vector<uint32_t>(this->nr_inner_dims_);
                        reader.read(posting_list_length_stats.data(), sizeof(uint32_t), this->nr_inner_dims_);
                        for (size_t i = 0; i < this->n_rows_internal_; ++i) {
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

    // memory in reader must be guaranteed to be valid during the lifetime of this object.
    Status
    PrepareMmap(MemoryIOReader& reader, size_t rows, int map_flags, const std::string& supplement_target_filename) {
        const auto initial_reader_location = reader.tellg();
        const auto nnz = (reader.remaining() - (rows * sizeof(size_t))) / SparseRow<DType>::element_size();

        // count raw vector idx occurrences
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
                // skip value
                reader.advance(sizeof(DType));
            }
        }
        // reset reader to the beginning
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
        if constexpr (algo == InvertedIndexAlgo::DAAT_WAND || algo == InvertedIndexAlgo::DAAT_MAXSCORE ||
                      algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
            map_byte_size_ += max_score_in_dim_byte_size;
        }
        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            row_sums_byte_size = rows * sizeof(typename decltype(bm25_params_->row_sums)::value_type);
            map_byte_size_ += row_sums_byte_size;
        }

        if (map_byte_size_ == 0) {
            // early return to avoid mmapping empty file
            return Status::success;
        }

        std::ofstream temp_file(supplement_target_filename, std::ios::binary | std::ios::trunc);
        if (!temp_file) {
            LOG_KNOWHERE_ERROR_ << "Failed to create mmap file when loading sparse InvertedIndex: " << strerror(errno);
            return Status::disk_file_error;
        }
        temp_file.close();

        std::filesystem::resize_file(supplement_target_filename, map_byte_size_);

        map_fd_ = open(supplement_target_filename.c_str(), O_RDWR);
        if (map_fd_ == -1) {
            LOG_KNOWHERE_ERROR_ << "Failed to open mmap file when loading sparse InvertedIndex: " << strerror(errno);
            return Status::disk_file_error;
        }
        // file will disappear in the filesystem immediately but the actual file will not be deleted
        // until the file descriptor is closed in the destructor.
        std::filesystem::remove(supplement_target_filename);

        // clear MAP_PRIVATE flag: we need to write to this mmapped memory/file,
        // MAP_PRIVATE triggers copy-on-write and uses extra anonymous memory.
        map_flags &= ~MAP_PRIVATE;
        map_flags |= MAP_SHARED;

        map_ = static_cast<char*>(mmap(nullptr, map_byte_size_, PROT_READ | PROT_WRITE, map_flags, map_fd_, 0));
        if (map_ == MAP_FAILED) {
            LOG_KNOWHERE_ERROR_ << "Failed to create mmap when loading sparse InvertedIndex: " << strerror(errno)
                                << ", size: " << map_byte_size_ << " on file: " << supplement_target_filename;
            return Status::disk_file_error;
        }
        if (madvise(map_, map_byte_size_, MADV_RANDOM) != 0) {
            LOG_KNOWHERE_WARNING_ << "Failed to madvise mmap when loading sparse InvertedIndex: " << strerror(errno);
        }

        char* ptr = map_;

        // initialize containers memory.
        inverted_index_ids_.initialize(ptr, inverted_index_ids_byte_size);
        ptr += inverted_index_ids_byte_size;
        inverted_index_vals_.initialize(ptr, inverted_index_vals_byte_size);
        ptr += inverted_index_vals_byte_size;

        if constexpr (algo == InvertedIndexAlgo::DAAT_WAND || algo == InvertedIndexAlgo::DAAT_MAXSCORE ||
                      algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
            max_score_in_dim_.initialize(ptr, max_score_in_dim_byte_size);
            ptr += max_score_in_dim_byte_size;
        }

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
            if constexpr (algo == InvertedIndexAlgo::DAAT_WAND || algo == InvertedIndexAlgo::DAAT_MAXSCORE ||
                          algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
                max_score_in_dim_.emplace_back(0.0f);
            }
            ++dim_id;
        }
        // in mmap mode, next_dim_id_ should never be used, but still assigning for consistency.
        next_dim_id_ = dim_id;

        return Status::success;
    }

    // Non zero drop ratio is only supported for static index, i.e. data should
    // include all rows that'll be added to the index.
    Status
    Train(const SparseRow<DType>* data, size_t rows) override {
        if constexpr (mmapped) {
            throw std::invalid_argument("mmapped InvertedIndex does not support Train");
        } else {
            return Status::success;
        }
    }

    Status
    Add(const SparseRow<DType>* data, size_t rows, int64_t dim) override {
        if constexpr (mmapped) {
            throw std::invalid_argument("mmapped InvertedIndex does not support Add");
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

            // mapping data to spans
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

            return Status::success;
        }
    }

    void
    Search(const SparseRow<DType>& query, size_t k, float* distances, label_t* labels, const BitsetView& bitset,
           const DocValueComputer<float>& computer, InvertedIndexApproxSearchParams& approx_params) const override {
        // initially set result distances to NaN and labels to -1
        std::fill(distances, distances + k, std::numeric_limits<float>::quiet_NaN());
        std::fill(labels, labels + k, -1);
        if (query.size() == 0) {
            return;
        }

        auto q_vec = parse_query(query, approx_params.drop_ratio_search);
        if (q_vec.empty()) {
            return;
        }

        MaxMinHeap<float> heap(k * approx_params.refine_factor);
        // DAAT_WAND and DAAT_MAXSCORE are based on the implementation in PISA.
        if constexpr (algo == InvertedIndexAlgo::DAAT_WAND) {
            search_daat_wand(q_vec, heap, bitset, computer, approx_params.dim_max_score_ratio);
        } else if constexpr (algo == InvertedIndexAlgo::DAAT_MAXSCORE) {
            search_daat_maxscore(q_vec, heap, bitset, computer, approx_params.dim_max_score_ratio);
        } else if constexpr (algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
            search_daat_maxscore_v2(q_vec, heap, bitset, computer, approx_params.dim_max_score_ratio);
        } else {
            search_taat_naive(q_vec, heap, bitset, computer);
        }

        if (approx_params.refine_factor == 1) {
            collect_result(heap, distances, labels);
        } else {
            refine_and_collect(query, heap, k, distances, labels, computer, approx_params);
        }
    }

    // Returned distances are inaccurate based on the drop_ratio.
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
            if constexpr (algo == InvertedIndexAlgo::DAAT_WAND || algo == InvertedIndexAlgo::DAAT_MAXSCORE ||
                          algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
                res += sizeof(typename decltype(max_score_in_dim_spans_)::value_type) * max_score_in_dim_spans_.size();
            }
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
    // Given a vector of values, returns the threshold value.
    // All values strictly smaller than the threshold will be ignored.
    // values will be modified in this function.
    inline DType
    get_threshold(std::vector<DType>& values, float drop_ratio) const {
        // drop_ratio is in [0, 1) thus drop_count is guaranteed to be less
        // than values.size().
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

    template <typename DocIdFilter>
    struct Cursor {
     public:
        Cursor(const boost::span<const table_t>& plist_ids, const boost::span<const QType>& plist_vals, size_t num_vec,
               float max_score, float q_value, DocIdFilter filter)
            : plist_ids_(plist_ids),
              plist_vals_(plist_vals),
              plist_size_(plist_ids.size()),
              total_num_vec_(num_vec),
              max_score_(max_score),
              q_value_(q_value),
              filter_(filter) {
            skip_filtered_ids();
            update_cur_vec_id();
        }
        Cursor(const Cursor& rhs) = delete;
        Cursor(Cursor&& rhs) noexcept = default;

        void
        next() {
            ++loc_;
            skip_filtered_ids();
            update_cur_vec_id();
        }

        void
        seek(table_t vec_id) {
            while (loc_ < plist_size_ && plist_ids_[loc_] < vec_id) {
                ++loc_;
            }
            skip_filtered_ids();
            update_cur_vec_id();
        }

        QType
        cur_vec_val() const {
            return plist_vals_[loc_];
        }

        const boost::span<const table_t>& plist_ids_;
        const boost::span<const QType>& plist_vals_;
        const size_t plist_size_;
        size_t loc_ = 0;
        size_t total_num_vec_ = 0;
        float max_score_ = 0.0f;
        float q_value_ = 0.0f;
        DocIdFilter filter_;
        table_t cur_vec_id_ = 0;

     private:
        inline void
        update_cur_vec_id() {
            cur_vec_id_ = (loc_ >= plist_size_) ? total_num_vec_ : plist_ids_[loc_];
        }

        inline void
        skip_filtered_ids() {
            while (loc_ < plist_size_ && !filter_.empty() && filter_.test(plist_ids_[loc_])) {
                ++loc_;
            }
        }
    };  // struct Cursor

    // SIMDCursor: Cursor with SIMD-accelerated seek for MaxScore v2
    // Uses aligned posting lists and SIMD parallel comparison for faster seek
    template <typename DocIdFilter>
    struct SIMDCursor {
     public:
        SIMDCursor(const boost::span<const table_t>& plist_ids, const boost::span<const QType>& plist_vals,
                   size_t num_vec, float max_score, float q_value, DocIdFilter filter)
            : plist_ids_(plist_ids),
              plist_vals_(plist_vals),
              plist_size_(plist_ids.size()),
              total_num_vec_(num_vec),
              max_score_(max_score),
              q_value_(q_value),
              filter_(filter) {
            skip_filtered_ids();
            update_cur_vec_id();
        }
        SIMDCursor(const SIMDCursor& rhs) = delete;
        SIMDCursor(SIMDCursor&& rhs) noexcept = default;

        void
        next() {
            ++loc_;
            skip_filtered_ids();
            update_cur_vec_id();
        }

        // SIMD-accelerated seek using parallel comparison
        void
        seek(table_t vec_id) {
            if (loc_ >= plist_size_ || plist_ids_[loc_] >= vec_id) {
                skip_filtered_ids();
                update_cur_vec_id();
                return;
            }

            // Use SIMD seek for large jumps
            loc_ = simd_seek_dispatch(plist_ids_.data(), plist_size_, loc_, vec_id);
            skip_filtered_ids();
            update_cur_vec_id();
        }

        QType
        cur_vec_val() const {
            return plist_vals_[loc_];
        }

        const boost::span<const table_t>& plist_ids_;
        const boost::span<const QType>& plist_vals_;
        const size_t plist_size_;
        size_t loc_ = 0;
        size_t total_num_vec_ = 0;
        float max_score_ = 0.0f;
        float q_value_ = 0.0f;
        DocIdFilter filter_;
        table_t cur_vec_id_ = 0;

     private:
        inline void
        update_cur_vec_id() {
            cur_vec_id_ = (loc_ >= plist_size_) ? total_num_vec_ : plist_ids_[loc_];
        }

        inline void
        skip_filtered_ids() {
            while (loc_ < plist_size_ && !filter_.empty() && filter_.test(plist_ids_[loc_])) {
                ++loc_;
            }
        }
    };  // struct SIMDCursor

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

    template <typename DocIdFilter>
    std::vector<Cursor<DocIdFilter>>
    make_cursors(const std::vector<std::pair<size_t, DType>>& q_vec, const DocValueComputer<float>& computer,
                 DocIdFilter& filter, float dim_max_score_ratio) const {
        std::vector<Cursor<DocIdFilter>> cursors;
        cursors.reserve(q_vec.size());
        for (auto q_dim : q_vec) {
            auto& plist_ids = inverted_index_ids_spans_[q_dim.first];
            auto& plist_vals = inverted_index_vals_spans_[q_dim.first];
            cursors.emplace_back(plist_ids, plist_vals, n_rows_internal_,
                                 max_score_in_dim_spans_[q_dim.first] * q_dim.second * dim_max_score_ratio,
                                 q_dim.second, filter);
        }
        return cursors;
    }

    // Create SIMD cursors with accelerated seek for MaxScore v2
    template <typename DocIdFilter>
    std::vector<SIMDCursor<DocIdFilter>>
    make_simd_cursors(const std::vector<std::pair<size_t, DType>>& q_vec, const DocValueComputer<float>& computer,
                      DocIdFilter& filter, float dim_max_score_ratio) const {
        std::vector<SIMDCursor<DocIdFilter>> cursors;
        cursors.reserve(q_vec.size());
        for (auto q_dim : q_vec) {
            auto& plist_ids = inverted_index_ids_spans_[q_dim.first];
            auto& plist_vals = inverted_index_vals_spans_[q_dim.first];
            cursors.emplace_back(plist_ids, plist_vals, n_rows_internal_,
                                 max_score_in_dim_spans_[q_dim.first] * q_dim.second * dim_max_score_ratio,
                                 q_dim.second, filter);
        }
        return cursors;
    }

    // find the top-k candidates using brute force search, k as specified by the capacity of the heap.
    // any value in q_vec that is smaller than q_threshold and any value with dimension >= n_cols() will be ignored.
    // TODO: may switch to row-wise brute force if filter rate is high. Benchmark needed.
    template <typename DocIdFilter>
    void
    search_taat_naive(const std::vector<std::pair<size_t, DType>>& q_vec, MaxMinHeap<float>& heap, DocIdFilter& filter,
                      const DocValueComputer<float>& computer) const {
        auto scores = compute_all_distances(q_vec, computer);
        for (size_t i = 0; i < n_rows_internal_; ++i) {
            if ((filter.empty() || !filter.test(i)) && scores[i] != 0) {
                heap.push(i, scores[i]);
            }
        }
    }

    template <typename DocIdFilter>
    void
    search_daat_wand(const std::vector<std::pair<size_t, DType>>& q_vec, MaxMinHeap<float>& heap, DocIdFilter& filter,
                     const DocValueComputer<float>& computer, float dim_max_score_ratio) const {
        std::vector<Cursor<DocIdFilter>> cursors = make_cursors(q_vec, computer, filter, dim_max_score_ratio);
        std::vector<Cursor<DocIdFilter>*> cursor_ptrs(cursors.size());
        for (size_t i = 0; i < cursors.size(); ++i) {
            cursor_ptrs[i] = &cursors[i];
        }

        auto sort_cursors = [&cursor_ptrs] {
            std::sort(cursor_ptrs.begin(), cursor_ptrs.end(),
                      [](auto& x, auto& y) { return x->cur_vec_id_ < y->cur_vec_id_; });
        };
        sort_cursors();

        while (true) {
            float threshold = heap.full() ? heap.top().val : 0;
            float upper_bound = 0;
            size_t pivot;

            bool found_pivot = false;
            for (pivot = 0; pivot < q_vec.size(); ++pivot) {
                if (cursor_ptrs[pivot]->cur_vec_id_ >= n_rows_internal_) {
                    break;
                }
                upper_bound += cursor_ptrs[pivot]->max_score_;
                if (upper_bound > threshold) {
                    found_pivot = true;
                    break;
                }
            }
            if (!found_pivot) {
                break;
            }

            table_t pivot_id = cursor_ptrs[pivot]->cur_vec_id_;
            if (pivot_id == cursor_ptrs[0]->cur_vec_id_) {
                float score = 0;
                float cur_vec_sum =
                    metric_type_ == SparseMetricType::METRIC_BM25 ? bm25_params_->row_sums_spans_[pivot_id] : 0;
                for (auto& cursor_ptr : cursor_ptrs) {
                    if (cursor_ptr->cur_vec_id_ != pivot_id) {
                        break;
                    }
                    score += cursor_ptr->q_value_ * computer(cursor_ptr->cur_vec_val(), cur_vec_sum);
                    cursor_ptr->next();
                }
                heap.push(pivot_id, score);
                sort_cursors();
            } else {
                size_t next_list = pivot;
                for (; cursor_ptrs[next_list]->cur_vec_id_ == pivot_id; --next_list) {
                }
                cursor_ptrs[next_list]->seek(pivot_id);
                for (size_t i = next_list + 1; i < q_vec.size(); ++i) {
                    if (cursor_ptrs[i]->cur_vec_id_ >= cursor_ptrs[i - 1]->cur_vec_id_) {
                        break;
                    }
                    std::swap(cursor_ptrs[i], cursor_ptrs[i - 1]);
                }
            }
        }
    }

    template <typename DocIdFilter>
    void
    search_daat_maxscore(std::vector<std::pair<size_t, DType>>& q_vec, MaxMinHeap<float>& heap, DocIdFilter& filter,
                         const DocValueComputer<float>& computer, float dim_max_score_ratio) const {
        std::sort(q_vec.begin(), q_vec.end(), [this](auto& a, auto& b) {
            return a.second * max_score_in_dim_spans_[a.first] > b.second * max_score_in_dim_spans_[b.first];
        });

        std::vector<Cursor<DocIdFilter>> cursors = make_cursors(q_vec, computer, filter, dim_max_score_ratio);

        float threshold = heap.full() ? heap.top().val : 0;

        std::vector<float> upper_bounds(cursors.size());
        float bound_sum = 0.0;
        for (size_t i = cursors.size() - 1; i + 1 > 0; --i) {
            bound_sum += cursors[i].max_score_;
            upper_bounds[i] = bound_sum;
        }

        table_t next_cand_vec_id = n_rows_internal_;
        for (size_t i = 0; i < cursors.size(); ++i) {
            if (cursors[i].cur_vec_id_ < next_cand_vec_id) {
                next_cand_vec_id = cursors[i].cur_vec_id_;
            }
        }

        // first_ne_idx is the index of the first non-essential cursor
        size_t first_ne_idx = cursors.size();

        while (first_ne_idx != 0 && upper_bounds[first_ne_idx - 1] <= threshold) {
            --first_ne_idx;
            if (first_ne_idx == 0) {
                return;
            }
        }

        float curr_cand_score = 0.0f;
        table_t curr_cand_vec_id = 0;

        while (curr_cand_vec_id < n_rows_internal_) {
            auto found_cand = false;
            while (found_cand == false) {
                // start find from next_vec_id
                if (next_cand_vec_id >= n_rows_internal_) {
                    return;
                }
                // get current candidate vector
                curr_cand_vec_id = next_cand_vec_id;
                curr_cand_score = 0.0f;
                // update next_cand_vec_id
                next_cand_vec_id = n_rows_internal_;
                float cur_vec_sum =
                    metric_type_ == SparseMetricType::METRIC_BM25 ? bm25_params_->row_sums_spans_[curr_cand_vec_id] : 0;

                for (size_t i = 0; i < first_ne_idx; ++i) {
                    if (cursors[i].cur_vec_id_ == curr_cand_vec_id) {
                        curr_cand_score += cursors[i].q_value_ * computer(cursors[i].cur_vec_val(), cur_vec_sum);
                        cursors[i].next();
                    }
                    if (cursors[i].cur_vec_id_ < next_cand_vec_id) {
                        next_cand_vec_id = cursors[i].cur_vec_id_;
                    }
                }

                found_cand = true;
                for (size_t i = first_ne_idx; i < cursors.size(); ++i) {
                    if (curr_cand_score + upper_bounds[i] <= threshold) {
                        found_cand = false;
                        break;
                    }
                    cursors[i].seek(curr_cand_vec_id);
                    if (cursors[i].cur_vec_id_ == curr_cand_vec_id) {
                        curr_cand_score += cursors[i].q_value_ * computer(cursors[i].cur_vec_val(), cur_vec_sum);
                    }
                }
            }

            if (curr_cand_score > threshold) {
                heap.push(curr_cand_vec_id, curr_cand_score);
                threshold = heap.full() ? heap.top().val : 0;
                while (first_ne_idx != 0 && upper_bounds[first_ne_idx - 1] <= threshold) {
                    --first_ne_idx;
                    if (first_ne_idx == 0) {
                        return;
                    }
                }
            }
        }
    }

    // MaxScore v2: Batched MaxScore with SIMD scatter-gather accumulation
    // Process each posting list's contributions in batches rather than alternating
    // between iterators. This enables:
    // - Better memory locality (prefetcher works on sequential access)
    // - Better branch prediction (consistent pattern per posting list)
    // - SIMD vectorization (AVX512 scatter-gather for batch accumulation)
    template <typename DocIdFilter>
    void
    search_daat_maxscore_v2(std::vector<std::pair<size_t, DType>>& q_vec, MaxMinHeap<float>& heap, DocIdFilter& filter,
                            const DocValueComputer<float>& computer, float dim_max_score_ratio) const {
        // Window size for batched processing (64K docs * 4 bytes = 256KB, fits in L2/L3 cache)
        constexpr size_t WINDOW_SIZE = 65536;

        // Sort query terms by contribution (max_score * query_weight) descending
        std::sort(q_vec.begin(), q_vec.end(), [this](auto& a, auto& b) {
            return a.second * max_score_in_dim_spans_[a.first] > b.second * max_score_in_dim_spans_[b.first];
        });

        // Build posting list info for each query term
        struct PostingListInfo {
            float q_weight;
            float max_score;
            boost::span<const table_t> ids;
            boost::span<const QType> vals;
        };
        std::vector<PostingListInfo> posting_lists;
        posting_lists.reserve(q_vec.size());

        for (auto& [inner_dim, q_weight] : q_vec) {
            auto plist_ids = inverted_index_ids_spans_[inner_dim];
            auto plist_vals = inverted_index_vals_spans_[inner_dim];
            if (plist_ids.empty())
                continue;

            float max_score = max_score_in_dim_spans_[inner_dim] * static_cast<float>(q_weight) * dim_max_score_ratio;
            posting_lists.push_back({static_cast<float>(q_weight), max_score, plist_ids, plist_vals});
        }

        if (posting_lists.empty())
            return;

        // Compute suffix sums for pruning (upper bounds)
        std::vector<float> upper_bounds(posting_lists.size());
        float bound_sum = 0.0f;
        for (size_t i = posting_lists.size(); i > 0; --i) {
            bound_sum += posting_lists[i - 1].max_score;
            upper_bounds[i - 1] = bound_sum;
        }

        float threshold = heap.full() ? heap.top().val : 0;

        // Allocate window score buffer (aligned for SIMD)
        std::vector<float> scores(WINDOW_SIZE);

        // Candidate indices buffer for SIMD extraction (worst case: all docs are candidates)
        std::vector<uint32_t> candidate_indices(WINDOW_SIZE);

        // Track current position in each posting list (for cursor advancement across windows)
        std::vector<size_t> cursors(posting_lists.size(), 0);

        // Process documents in windows
        for (table_t window_start = 0; window_start < n_rows_internal_; window_start += WINDOW_SIZE) {
            table_t window_end =
                std::min(static_cast<table_t>(window_start + WINDOW_SIZE), static_cast<table_t>(n_rows_internal_));
            size_t window_size = window_end - window_start;

            // Reset scores for this window
            std::memset(scores.data(), 0, window_size * sizeof(float));

            // Find essential/non-essential split based on current threshold
            size_t first_ne_idx = posting_lists.size();
            while (first_ne_idx > 0 && upper_bounds[first_ne_idx - 1] <= threshold) {
                --first_ne_idx;
            }
            if (first_ne_idx == 0)
                break;  // No more candidates possible

            // BATCH PROCESS each essential posting list
            // Process all contributions from one posting list before moving to the next,
            // enabling CPU prefetcher and SIMD
            for (size_t i = 0; i < first_ne_idx; ++i) {
                auto& pl = posting_lists[i];
                const table_t* ids = pl.ids.data();
                const QType* vals = pl.vals.data();
                size_t pl_size = pl.ids.size();

                // Advance cursor to first element >= window_start
                while (cursors[i] < pl_size && ids[cursors[i]] < window_start) {
                    cursors[i]++;
                }
                size_t list_start = cursors[i];

                // Find end position (first element >= window_end)
                size_t list_end = list_start;
                while (list_end < pl_size && ids[list_end] < window_end) {
                    list_end++;
                }

                if (list_start < list_end) {
                    if (metric_type_ == SparseMetricType::METRIC_IP) {
                        // SIMD batch accumulate for IP metric
                        // scores[doc_id - window_start] += q_weight * val
                        sparse::accumulate_window_ip_dispatch(ids, vals, list_start, list_end, pl.q_weight,
                                                              scores.data(), window_start);
                    } else {
                        // Scalar loop for BM25 metric (requires computer function)
                        const auto& doc_len_ratios = bm25_params_->row_sums_spans_;
                        for (size_t j = list_start; j < list_end; ++j) {
                            const auto doc_id = ids[j];
                            const uint32_t local_id = doc_id - window_start;
                            scores[local_id] +=
                                pl.q_weight * computer(static_cast<float>(vals[j]), doc_len_ratios[doc_id]);
                        }
                    }
                }
            }

            // Calculate max possible contribution from non-essential terms
            float ne_upper_bound = (first_ne_idx < posting_lists.size()) ? upper_bounds[first_ne_idx] : 0.0f;

            // SIMD candidate extraction: find all docs where score > effective_threshold
            // effective_threshold = max(0, threshold - ne_upper_bound)
            // This combines two checks: score > 0 AND score + ne_upper_bound > threshold
            float effective_threshold = std::max(0.0f, threshold - ne_upper_bound);
            size_t num_candidates = sparse::extract_candidates_dispatch(scores.data(), window_size, effective_threshold,
                                                                        candidate_indices.data());

            // Process extracted candidates
            for (size_t c = 0; c < num_candidates; ++c) {
                uint32_t doc_offset = candidate_indices[c];
                float score = scores[doc_offset];

                table_t doc_id = window_start + static_cast<table_t>(doc_offset);
                if (!filter.empty() && filter.test(doc_id))
                    continue;

                // Add contributions from non-essential terms
                // (These are terms whose max contribution alone can't beat threshold,
                // but might contribute when combined with essential terms)
                float full_score = score;
                float cur_vec_sum =
                    metric_type_ == SparseMetricType::METRIC_BM25 ? bm25_params_->row_sums_spans_[doc_id] : 0;

                for (size_t i = first_ne_idx; i < posting_lists.size(); ++i) {
                    // Early termination if we can't beat threshold
                    if (full_score + upper_bounds[i] <= threshold)
                        break;

                    auto& pl = posting_lists[i];
                    const table_t* ids = pl.ids.data();
                    const QType* vals = pl.vals.data();
                    size_t pl_size = pl.ids.size();

                    // Binary search for doc_id in this posting list
                    auto it = std::lower_bound(ids, ids + pl_size, doc_id);
                    if (it != ids + pl_size && *it == doc_id) {
                        size_t idx = it - ids;
                        full_score += pl.q_weight * computer(static_cast<float>(vals[idx]), cur_vec_sum);
                    }
                }

                if (full_score > threshold) {
                    heap.push(doc_id, full_score);
                    threshold = heap.full() ? heap.top().val : 0;
                }
            }
        }
    }

    void
    refine_and_collect(const SparseRow<DType>& query, MaxMinHeap<float>& inacc_heap, size_t k, float* distances,
                       label_t* labels, const DocValueComputer<float>& computer,
                       InvertedIndexApproxSearchParams& approx_params) const {
        std::vector<table_t> docids;
        MaxMinHeap<float> heap(k);

        docids.reserve(inacc_heap.size());
        while (!inacc_heap.empty()) {
            table_t u = inacc_heap.pop();
            docids.emplace_back(u);
        }

        auto q_vec = parse_query(query, 0);
        if (q_vec.empty()) {
            return;
        }

        // dim_max_score_ratio for refine process should be >= 1.0
        float dim_max_score_ratio = std::max(approx_params.dim_max_score_ratio, 1.0f);

        DocIdFilterByVector filter(std::move(docids));
        if constexpr (algo == InvertedIndexAlgo::DAAT_WAND) {
            search_daat_wand(q_vec, heap, filter, computer, dim_max_score_ratio);
        } else if constexpr (algo == InvertedIndexAlgo::DAAT_MAXSCORE) {
            search_daat_maxscore(q_vec, heap, filter, computer, dim_max_score_ratio);
        } else if constexpr (algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
            search_daat_maxscore_v2(q_vec, heap, filter, computer, dim_max_score_ratio);
        } else {
            search_taat_naive(q_vec, heap, filter, computer);
        }
        collect_result(heap, distances, labels);
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

    // Adds a single sparse row to the inverted index.
    //
    // UNIQUENESS INVARIANT: Each vec_id appears at most once per posting list.
    // This holds when:
    //   1. vec_id is assigned sequentially and is unique per document
    //   2. Each document is added exactly once via Train() or Add()
    //   3. Input sparse rows have at most one value per dimension (caller's responsibility)
    //
    // Note: SparseRow does not enforce unique dimensions - it assumes valid sorted input.
    // If a row contains duplicate dimensions, this function will append vec_id multiple
    // times to the same posting list, violating the invariant.
    //
    // This invariant is critical for AVX512 scatter operations in DAAT_MAXSCORE_V2
    // which require unique indices within each SIMD batch. Without this guarantee,
    // scatter would have undefined behavior when multiple lanes write to the same index.
    inline void
    add_row_to_index(const SparseRow<DType>& row, table_t vec_id) {
        [[maybe_unused]] float row_sum = 0;
        for (size_t j = 0; j < row.size(); ++j) {
            auto [dim, val] = row[j];
            if (metric_type_ == SparseMetricType::METRIC_BM25) {
                row_sum += val;
            }
            // Skip values equals to or close enough to zero(which contributes
            // little to the total IP score).
            if (val == 0) {
                continue;
            }
            auto dim_it = dim_map_.find(dim);
            if (dim_it == dim_map_.cend()) {
                if constexpr (mmapped) {
                    throw std::runtime_error("unexpected vector dimension in mmapped InvertedIndex");
                }
                dim_it = dim_map_.insert({dim, next_dim_id_++}).first;
                inverted_index_ids_.emplace_back();
                inverted_index_vals_.emplace_back();
                if constexpr (algo == InvertedIndexAlgo::DAAT_WAND || algo == InvertedIndexAlgo::DAAT_MAXSCORE ||
                              algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
                    max_score_in_dim_.emplace_back(0.0f);
                }
            }
            // vec_id is unique per document, so each posting list entry is unique
            inverted_index_ids_[dim_it->second].emplace_back(vec_id);
            inverted_index_vals_[dim_it->second].emplace_back(get_quant_val(val));
        }
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        build_stats_.dataset_nnz_stats_.push_back(row.size());
#endif
        // update max_score_in_dim_
        if constexpr (algo == InvertedIndexAlgo::DAAT_WAND || algo == InvertedIndexAlgo::DAAT_MAXSCORE ||
                      algo == InvertedIndexAlgo::DAAT_MAXSCORE_V2) {
            for (size_t j = 0; j < row.size(); ++j) {
                auto [dim, val] = row[j];
                if (val == 0) {
                    continue;
                }
                auto dim_it = dim_map_.find(dim);
                if (dim_it == dim_map_.cend()) {
                    throw std::runtime_error("unexpected vector dimension in InvertedIndex");
                }
                auto score = static_cast<float>(val);
                if (metric_type_ == SparseMetricType::METRIC_BM25) {
                    score = bm25_params_->max_score_computer(val, row_sum);
                }
                max_score_in_dim_[dim_it->second] = std::max(max_score_in_dim_[dim_it->second], score);
            }
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

    // key is raw sparse vector dim/idx, value is the mapped dim/idx id in the index.
    std::unordered_map<table_t, uint32_t> dim_map_;
    uint32_t nr_inner_dims_ = 0;

    // reserve, [], size, emplace_back
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
        // row_sums is used to cache the sum of values of each row, which
        // corresponds to the document length of each doc in the BM25 formula.
        Vector<float> row_sums;
        boost::span<const float> row_sums_spans_;

        DocValueComputer<float> max_score_computer;

        BM25Params(float k1, float b, float avgdl)
            : k1(k1), b(b), max_score_computer(GetDocValueBM25Computer<float>(k1, b, avgdl)) {
        }
    };  // struct BM25Params

    std::unique_ptr<BM25Params> bm25_params_;

    static constexpr uint32_t index_file_v1_header_size = 32;
    static constexpr uint32_t index_file_v1_header_reserved_size = 16;

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
    // Statistics for the build process, which will be used to generate the prometheus metrics
    InvertedIndexBuildStats build_stats_;

    std::string index_id_{};
    prometheus::Gauge* index_size_gauge_{nullptr};
    prometheus::Histogram* index_dataset_nnz_len_histogram_{nullptr};
    prometheus::Histogram* index_posting_list_len_histogram_{nullptr};
#endif
};  // class InvertedIndex

}  // namespace knowhere::sparse

#endif  // SPARSE_INVERTED_INDEX_H
