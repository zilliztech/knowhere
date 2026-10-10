// Based on the SINDI algorithm for sparse vector search.
// Reference: https://arxiv.org/abs/2509.08395

#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <span>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "index/sparse/aligned_allocator.h"
#include "index/sparse/inverted_index.h"
#include "index/sparse/inverted_index_build.h"
#include "index/sparse/inverted_index_format.h"
#include "index/sparse/parallel_build.h"
#include "index/sparse/scorer.h"
#include "index/sparse/sindi_simd.h"
#include "knowhere/bitsetview.h"
#include "knowhere/operands.h"
#include "simd/hook.h"
#include "sindi_bm25_u4.h"
#include "sindi_packed12.h"
#include "sindi_refinement.h"

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
    static constexpr bool is_ip = std::is_same_v<QuantType, knowhere::fp16>;
    static constexpr bool is_bm25 = std::is_same_v<QuantType, uint16_t> || std::is_same_v<QuantType, uint8_t>;
    static constexpr bool is_bm25_u8 = std::is_same_v<QuantType, uint8_t>;
    static_assert(is_ip || is_bm25, "QuantType must be fp16 (for IP) or u8/u16 (for BM25)");
    static_assert(!(AllowIncremental && is_bm25_u8), "growable SINDI does not support BM25 u8");

    struct Bm25U8Overflow {
        uint32_t posting_offset;
        uint16_t value;
    };

    static constexpr uint32_t min_window_size = 1024;
    static constexpr uint32_t max_window_size = 65535;

    SindiInvertedIndex(uint32_t window_size, bool refine = false, bool packed = false, bool u16_ids = false)
        : packed_(packed),
          packed_u16_ids_(u16_ids),
          refine_(refine),
          window_size_(std::clamp(window_size, min_window_size, max_window_size)) {
        if (packed_u16_ids_ && (!packed_ || !is_bm25_u8))
            throw std::invalid_argument("U16/U4 storage requires packed BM25");
        if (packed_ && ((is_ip && window_size != 4096) ||
                        (!is_ip && (!is_bm25_u8 || AllowIncremental || refine || window_size < min_window_size ||
                                    window_size > (packed_u16_ids_ ? max_window_size : 4096)))))
            throw std::invalid_argument("Invalid packed SINDI representation/window/refinement");
    }

    SindiInvertedIndex(const SindiInvertedIndex& rhs) = delete;
    SindiInvertedIndex(SindiInvertedIndex&& rhs) noexcept = default;
    SindiInvertedIndex&
    operator=(const SindiInvertedIndex& rhs) = delete;
    SindiInvertedIndex&
    operator=(SindiInvertedIndex&& rhs) noexcept = default;

    bool
    packed_storage_enabled() const noexcept {
        return packed_;
    }

    bool
    refinement_enabled() const noexcept override {
        return refine_;
    }

    [[nodiscard]] size_t
    size() const noexcept override {
        size_t res = sizeof(*this);

        res += refinement_seek_.capacity() * sizeof(sindi::RefinementSeek);
        for (const auto& seek : refinement_seek_) {
            res += (seek.windows.capacity() + seek.offsets.capacity()) * sizeof(uint32_t);
        }

        // Global posting lists
        res += plists_dim_offsets_span_.size() *
               sizeof(typename std::decay_t<decltype(plists_dim_offsets_span_)>::value_type);
        if (packed_ready_) {
            res += packed_ids_.empty() ? packed_ids_span_.size() : packed_ids_.capacity();
            res += packed_vals_.empty() ? packed_vals_span_.size() : packed_vals_.capacity();
        } else if (!total_plists_ids_flat_span_.empty()) {
            res += total_plists_ids_flat_span_.size() * sizeof(uint16_t);
            res += total_plists_vals_flat_span_.size() * sizeof(QuantType);
        } else {
            for (size_t dim_id = 0; dim_id < total_plists_ids_spans_.size(); ++dim_id) {
                res += total_plists_ids_spans_[dim_id].size() * sizeof(uint16_t);
                res += total_plists_vals_spans_[dim_id].size() * sizeof(QuantType);
            }
        }

        // Window sizes encoding (per-dim window nnz) and bitset of formats
        res += plists_wnnzs_fmts_msk_span_.size() *
               sizeof(typename std::decay_t<decltype(plists_wnnzs_fmts_msk_span_)>::value_type);
        if (!plists_window_nnzs_offsets_.empty()) {
            res += plists_window_nnzs_offsets_.size() * sizeof(size_t);
            res += plists_window_nnzs_flat_.size() * sizeof(uint8_t);
        } else {
            res += plists_window_nnzs_spans_.size() *
                   sizeof(typename std::decay_t<decltype(plists_window_nnzs_spans_)>::value_type);
            for (const auto& wspan : plists_window_nnzs_spans_) {
                res += wspan.size() * sizeof(typename std::decay_t<decltype(wspan)>::value_type);
            }
        }

        res += this->dim_map_.byte_size();

        // Row sums for BM25 support
        res += row_sums_span_.size() * sizeof(float);
        res += bm25_u8_overflow_offsets_span_.size() * sizeof(uint32_t);
        res += bm25_u8_overflow_values_span_.size() * sizeof(uint16_t);

        // Global score bounds are retained by ordinary and packed indexes alike.
        res += (max_scores_per_dim_.empty() ? max_scores_per_dim_span_.size() : max_scores_per_dim_.capacity()) *
               sizeof(float);
        return res;
    }

    void
    set_legacy_dim_map_mphf_trailer_workaround(bool enabled) {
        legacy_dim_map_mphf_trailer_workaround_ = enabled;
    }

    void
    encode_window_nnzs(bool parallel) {
        const size_t dim_count = this->nr_inner_dims_;
        plists_window_nnzs_flat_.clear();
        plists_window_nnzs_offsets_.clear();
        plists_window_nnzs_.resize(dim_count);
        plists_window_nnzs_spans_.resize(dim_count);
        std::vector<uint8_t> sparse_formats(dim_count, 0);

        // Encode window nnzs with sparse/dense format selection
        // Sparse format: (wid, wnnz) pairs when cnt_nonempty * 4 < nr_windows * 2
        // Dense format: one uint16_t per window otherwise
        auto encode_dim_nnzs = [&](size_t dim_id) {
            size_t nonempty_windows = 0;
            for (size_t wid = 0; wid < nr_windows_; ++wid) {
                nonempty_windows += window_index_plists_sz_spans_[wid][dim_id] != 0;
            }

            const bool use_sparse = nonempty_windows * sizeof(uint32_t) < nr_windows_ * sizeof(uint16_t);
            sparse_formats[dim_id] = use_sparse;
            auto& encoded = plists_window_nnzs_[dim_id];
            if (use_sparse) {
                // Sparse format: [wid | wnnz] packed into 32 bits
                encoded.resize(nonempty_windows * sizeof(uint32_t));
                auto* output = encoded.data();
                const auto wnnz_bits = 32 - __builtin_clz(window_size_);
                for (size_t wid = 0; wid < nr_windows_; ++wid) {
                    const auto wnnz = window_index_plists_sz_spans_[wid][dim_id];
                    if (wnnz != 0) {
                        const auto packed = (static_cast<uint32_t>(wid) << wnnz_bits) | wnnz;
                        std::memcpy(output, &packed, sizeof(uint32_t));
                        output += sizeof(uint32_t);
                    }
                }
            } else {
                // Dense format: one uint16_t per window
                encoded.resize(nr_windows_ * sizeof(uint16_t));
                auto* output = encoded.data();
                for (size_t wid = 0; wid < nr_windows_; ++wid) {
                    const auto wnnz = window_index_plists_sz_spans_[wid][dim_id];
                    std::memcpy(output, &wnnz, sizeof(uint16_t));
                    output += sizeof(uint16_t);
                }
            }
            plists_window_nnzs_spans_[dim_id] = std::span<const uint8_t>(encoded.data(), encoded.size());
        };

        if (parallel) {
            parallel_for(dim_count, encode_dim_nnzs);
        } else {
            for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
                encode_dim_nnzs(dim_id);
            }
        }

        plists_wnnzs_fmts_msk_.assign((dim_count + 7) / 8, 0);
        size_t sparse_dim_count = 0;
        size_t encoded_bytes = 0;
        for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
            if (sparse_formats[dim_id] != 0) {
                plists_wnnzs_fmts_msk_[dim_id >> 3] |= static_cast<uint8_t>(1u << (dim_id & 0x7));
                ++sparse_dim_count;
            }
            encoded_bytes += plists_window_nnzs_[dim_id].size();
        }
        plists_wnnzs_fmts_msk_span_ =
            std::span<const uint8_t>(plists_wnnzs_fmts_msk_.data(), plists_wnnzs_fmts_msk_.size());
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex window NNZ encoding completed: dims=" << dim_count
                           << ", windows=" << nr_windows_ << ", sparse_format_dims=" << sparse_dim_count
                           << ", dense_format_dims=" << (dim_count - sparse_dim_count)
                           << ", encoded_bytes=" << encoded_bytes << ", parallel=" << parallel;
    }

    void
    encode_window_nnzs_flat(bool parallel) {
        const size_t dim_count = this->nr_inner_dims_;
        plists_window_nnzs_.clear();
        plists_window_nnzs_spans_.clear();
        plists_window_nnzs_offsets_.assign(dim_count + 1, 0);
        std::vector<uint8_t> sparse_formats(dim_count, 0);

        auto measure_dim_nnzs = [&](size_t dim_id) {
            size_t nonempty_windows = 0;
            for (size_t wid = 0; wid < nr_windows_; ++wid) {
                nonempty_windows += window_index_plists_sz_spans_[wid][dim_id] != 0;
            }
            const bool use_sparse = nonempty_windows * sizeof(uint32_t) < nr_windows_ * sizeof(uint16_t);
            sparse_formats[dim_id] = use_sparse;
            plists_window_nnzs_offsets_[dim_id + 1] =
                use_sparse ? nonempty_windows * sizeof(uint32_t) : nr_windows_ * sizeof(uint16_t);
        };
        if (parallel) {
            parallel_for(dim_count, measure_dim_nnzs);
        } else {
            for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
                measure_dim_nnzs(dim_id);
            }
        }

        for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
            const size_t encoded_size = plists_window_nnzs_offsets_[dim_id + 1];
            if (encoded_size > std::numeric_limits<size_t>::max() - plists_window_nnzs_offsets_[dim_id]) {
                throw std::overflow_error("SindiInvertedIndex window NNZ encoding exceeds size_t capacity");
            }
            plists_window_nnzs_offsets_[dim_id + 1] = plists_window_nnzs_offsets_[dim_id] + encoded_size;
        }
        plists_window_nnzs_flat_.assign(plists_window_nnzs_offsets_.back(), 0);

        const auto encode_dim_nnzs = [&](size_t dim_id) {
            auto* output = plists_window_nnzs_flat_.data() + plists_window_nnzs_offsets_[dim_id];
            if (sparse_formats[dim_id] != 0) {
                const auto wnnz_bits = 32 - __builtin_clz(window_size_);
                for (size_t wid = 0; wid < nr_windows_; ++wid) {
                    const auto wnnz = window_index_plists_sz_spans_[wid][dim_id];
                    if (wnnz != 0) {
                        const auto packed = (static_cast<uint32_t>(wid) << wnnz_bits) | wnnz;
                        std::memcpy(output, &packed, sizeof(uint32_t));
                        output += sizeof(uint32_t);
                    }
                }
            } else {
                for (size_t wid = 0; wid < nr_windows_; ++wid) {
                    const auto wnnz = window_index_plists_sz_spans_[wid][dim_id];
                    std::memcpy(output, &wnnz, sizeof(uint16_t));
                    output += sizeof(uint16_t);
                }
            }
        };
        if (parallel) {
            parallel_for(dim_count, encode_dim_nnzs);
        } else {
            for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
                encode_dim_nnzs(dim_id);
            }
        }

        plists_wnnzs_fmts_msk_.assign((dim_count + 7) / 8, 0);
        size_t sparse_dim_count = 0;
        for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
            if (sparse_formats[dim_id] != 0) {
                plists_wnnzs_fmts_msk_[dim_id >> 3] |= static_cast<uint8_t>(1u << (dim_id & 0x7));
                ++sparse_dim_count;
            }
        }
        plists_wnnzs_fmts_msk_span_ =
            std::span<const uint8_t>(plists_wnnzs_fmts_msk_.data(), plists_wnnzs_fmts_msk_.size());
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex window NNZ encoding completed: dims=" << dim_count
                           << ", windows=" << nr_windows_ << ", sparse_format_dims=" << sparse_dim_count
                           << ", dense_format_dims=" << (dim_count - sparse_dim_count)
                           << ", encoded_bytes=" << plists_window_nnzs_flat_.size() << ", parallel=" << parallel
                           << ", storage=flat";
    }

    template <typename WindowId>
    void
    encode_window_nnzs_from_postings(std::span<const WindowId> posting_window_ids, bool parallel) {
        static_assert(std::is_unsigned_v<WindowId>);

        const size_t dim_count = this->nr_inner_dims_;
        plists_window_nnzs_.clear();
        plists_window_nnzs_spans_.clear();
        plists_window_nnzs_offsets_.assign(dim_count + 1, 0);
        std::vector<uint8_t> sparse_formats(dim_count, 0);

        // Posting lists are ordered by global row id, so their window ids are non-decreasing. Build the final
        // per-dimension encoding directly from those runs instead of materializing the dense [window][dimension]
        // count matrix used by the incremental build path.
        auto measure_dim_nnzs = [&](size_t dim_id) {
            const size_t begin = plists_dim_offsets_[dim_id];
            const size_t end = plists_dim_offsets_[dim_id + 1];
            size_t nonempty_windows = 0;
            for (size_t offset = begin; offset < end;) {
                const WindowId window_id = posting_window_ids[offset];
                ++nonempty_windows;
                do {
                    ++offset;
                } while (offset < end && posting_window_ids[offset] == window_id);
            }

            const bool use_sparse = nonempty_windows * sizeof(uint32_t) < nr_windows_ * sizeof(uint16_t);
            sparse_formats[dim_id] = use_sparse;
            plists_window_nnzs_offsets_[dim_id + 1] =
                use_sparse ? nonempty_windows * sizeof(uint32_t) : nr_windows_ * sizeof(uint16_t);
        };
        if (parallel) {
            parallel_for(dim_count, measure_dim_nnzs);
        } else {
            for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
                measure_dim_nnzs(dim_id);
            }
        }

        for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
            const size_t encoded_size = plists_window_nnzs_offsets_[dim_id + 1];
            if (encoded_size > std::numeric_limits<size_t>::max() - plists_window_nnzs_offsets_[dim_id]) {
                throw std::overflow_error("SindiInvertedIndex window NNZ encoding exceeds size_t capacity");
            }
            plists_window_nnzs_offsets_[dim_id + 1] = plists_window_nnzs_offsets_[dim_id] + encoded_size;
        }
        plists_window_nnzs_flat_.assign(plists_window_nnzs_offsets_.back(), 0);

        auto encode_dim_nnzs = [&](size_t dim_id) {
            const size_t begin = plists_dim_offsets_[dim_id];
            const size_t end = plists_dim_offsets_[dim_id + 1];
            auto* output = plists_window_nnzs_flat_.data() + plists_window_nnzs_offsets_[dim_id];
            if (sparse_formats[dim_id] != 0) {
                // Sparse format: [wid | wnnz] packed into 32 bits.
                const auto wnnz_bits = 32 - __builtin_clz(window_size_);
                for (size_t offset = begin; offset < end;) {
                    const uint32_t window_id = posting_window_ids[offset];
                    const size_t run_begin = offset;
                    do {
                        ++offset;
                    } while (offset < end && posting_window_ids[offset] == window_id);
                    const auto window_nnz = static_cast<uint32_t>(offset - run_begin);
                    const auto packed = (window_id << wnnz_bits) | window_nnz;
                    std::memcpy(output, &packed, sizeof(uint32_t));
                    output += sizeof(uint32_t);
                }
            } else {
                // Dense format: one uint16_t per window.
                for (size_t offset = begin; offset < end;) {
                    const size_t window_id = posting_window_ids[offset];
                    const size_t run_begin = offset;
                    do {
                        ++offset;
                    } while (offset < end && posting_window_ids[offset] == window_id);
                    const auto window_nnz = static_cast<uint16_t>(offset - run_begin);
                    std::memcpy(output + window_id * sizeof(uint16_t), &window_nnz, sizeof(uint16_t));
                }
            }
        };

        if (parallel) {
            parallel_for(dim_count, encode_dim_nnzs);
        } else {
            for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
                encode_dim_nnzs(dim_id);
            }
        }

        plists_wnnzs_fmts_msk_.assign((dim_count + 7) / 8, 0);
        size_t sparse_dim_count = 0;
        for (size_t dim_id = 0; dim_id < dim_count; ++dim_id) {
            if (sparse_formats[dim_id] != 0) {
                plists_wnnzs_fmts_msk_[dim_id >> 3] |= static_cast<uint8_t>(1u << (dim_id & 0x7));
                ++sparse_dim_count;
            }
        }
        plists_wnnzs_fmts_msk_span_ =
            std::span<const uint8_t>(plists_wnnzs_fmts_msk_.data(), plists_wnnzs_fmts_msk_.size());
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex window NNZ encoding completed: dims=" << dim_count
                           << ", windows=" << nr_windows_ << ", sparse_format_dims=" << sparse_dim_count
                           << ", dense_format_dims=" << (dim_count - sparse_dim_count)
                           << ", encoded_bytes=" << plists_window_nnzs_flat_.size() << ", parallel=" << parallel
                           << ", source=posting_window_ids, storage=flat";
    }

    void
    append_window_indexes(const SparseRow<DataType>* data, size_t rows) {
        const size_t dim_count = this->nr_inner_dims_;
        total_plists_ids_.resize(dim_count);
        total_plists_vals_.resize(dim_count);
        total_plists_ids_spans_.resize(dim_count);
        total_plists_vals_spans_.resize(dim_count);

        plists_dim_offsets_.resize(dim_count + 1, 0);

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
                total_plists_vals_[dim_id].push_back(quantize_value(val));
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

        encode_window_nnzs(/*parallel=*/false);
    }

    template <typename WindowId>
    void
    build_window_indexes_parallel_posting_window_ids(const SparseRow<DataType>* data, size_t rows,
                                                     PostingBuildPlan& posting_plan) {
        const size_t dim_count = this->nr_inner_dims_;
        const size_t total_postings = posting_plan.total_postings();
        std::vector<WindowId> posting_window_ids(total_postings);

        const size_t concurrency = posting_plan.cursors_by_worker.size();
        std::vector<std::vector<Bm25U8Overflow>> bm25_u8_overflows_by_worker;
        if constexpr (is_bm25_u8) {
            bm25_u8_overflows_by_worker.resize(concurrency);
        }
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex filling preallocated postings: total_postings=" << total_postings
                           << ", posting_bytes=" << (total_postings * (sizeof(uint16_t) + sizeof(QuantType)))
                           << ", posting_window_id_bytes=" << (total_postings * sizeof(WindowId))
                           << ", workers=" << concurrency;

        parallel_for_workers(concurrency, [&](size_t worker_id) {
            const auto [begin, end] = get_worker_row_range(rows, worker_id, concurrency);
            auto& worker_cursors = posting_plan.cursors_by_worker[worker_id];
            for (size_t row_id = begin; row_id < end; ++row_id) {
                const size_t window_id = row_id / window_size_;
                const auto local_id = static_cast<uint16_t>(row_id % window_size_);
                for (size_t j = 0; j < data[row_id].size(); ++j) {
                    const auto [dim, val] = data[row_id][j];
                    if (std::abs(val) < std::numeric_limits<DataType>::epsilon()) {
                        continue;
                    }
                    const auto inner_dim = this->dim_map_.lookup_trusted(dim);
                    const auto worker_offset = worker_cursors[inner_dim]++;
                    const size_t absolute_posting_offset =
                        posting_plan.cursor_mode == WorkerCursorMode::Absolute
                            ? worker_offset
                            : posting_plan.posting_offsets[inner_dim] + worker_offset;
                    total_plists_ids_flat_[absolute_posting_offset] = local_id;
                    store_sealed_value(absolute_posting_offset, val);
                    if constexpr (is_bm25_u8) {
                        const uint16_t overflow_value = bm25_u16_value(val);
                        if (overflow_value > 255) {
                            bm25_u8_overflows_by_worker[worker_id].push_back(
                                {static_cast<uint32_t>(absolute_posting_offset), overflow_value});
                        }
                    }
                    posting_window_ids[absolute_posting_offset] = static_cast<WindowId>(window_id);
                }
            }
        });
        finalize_sealed_bm25_u8_overflows(bm25_u8_overflows_by_worker);
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex preallocated postings filled: total_postings=" << total_postings;

        // The write cursors and size_t offsets are build-only. The compact uint32_t offsets retained by the index
        // are sufficient after all workers have filled their disjoint posting ranges.
        WorkerPostingCursors{}.swap(posting_plan.cursors_by_worker);
        std::vector<size_t>{}.swap(posting_plan.posting_offsets);

        max_scores_per_dim_.assign(dim_count, 0.0f);
        float bm25_p1 = 0.0f;
        float bm25_p2 = 0.0f;
        float bm25_p3 = 0.0f;
        if constexpr (is_bm25) {
            const auto& cfg = this->build_scorer_->config();
            bm25_p1 = cfg.scorer_params.bm25.k1 + 1.0f;
            bm25_p2 = cfg.scorer_params.bm25.k1 * (1.0f - cfg.scorer_params.bm25.b);
            bm25_p3 = cfg.scorer_params.bm25.k1 * cfg.scorer_params.bm25.b / cfg.scorer_params.bm25.avgdl;
        }

        parallel_for(dim_count, [&](size_t dim_id) {
            const size_t begin = plists_dim_offsets_[dim_id];
            const size_t end = plists_dim_offsets_[dim_id + 1];

            float max_score = 0.0f;
            if constexpr (is_ip) {
                for (size_t offset = begin; offset < end; ++offset) {
                    const float value = std::abs(static_cast<float>(total_plists_vals_flat_[offset]));
                    max_score = std::max(max_score, value);
                }
            } else {
                auto [overflow_cursor, overflow_end] = bm25_u8_overflow_range(begin, end);
                for (size_t offset = begin; offset < end; ++offset) {
                    const uint32_t global_id = static_cast<uint32_t>(posting_window_ids[offset]) * window_size_ +
                                               total_plists_ids_flat_[offset];
                    const float value =
                        bm25_posting_value(offset, total_plists_vals_flat_[offset], overflow_cursor, overflow_end);
                    const float dl = row_sums_[global_id];
                    max_score = std::max(max_score, bm25_p1 * value / (value + bm25_p2 + bm25_p3 * dl));
                }
            }
            max_scores_per_dim_[dim_id] = max_score;
        });
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex posting lists materialized: dims=" << dim_count
                           << ", total_postings=" << total_postings << ", windows=" << nr_windows_;

        plists_dim_offsets_span_ = std::span<const uint32_t>(plists_dim_offsets_.data(), plists_dim_offsets_.size());
        max_scores_per_dim_span_ = std::span<const float>(max_scores_per_dim_.data(), max_scores_per_dim_.size());

        encode_window_nnzs_from_postings<WindowId>(posting_window_ids, /*parallel=*/true);
    }

    void
    build_window_indexes_parallel_dense_window_counts(const SparseRow<DataType>* data, size_t rows,
                                                      PostingBuildPlan& posting_plan) {
        const size_t dim_count = this->nr_inner_dims_;
        const size_t total_postings = posting_plan.total_postings();
        window_index_plists_sz_.assign(nr_windows_, std::vector<uint16_t>(dim_count, 0));
        window_index_plists_sz_spans_.resize(nr_windows_);

        const size_t concurrency = posting_plan.cursors_by_worker.size();
        std::vector<std::vector<Bm25U8Overflow>> bm25_u8_overflows_by_worker;
        if constexpr (is_bm25_u8) {
            bm25_u8_overflows_by_worker.resize(concurrency);
        }
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex filling preallocated postings: total_postings=" << total_postings
                           << ", posting_bytes=" << (total_postings * (sizeof(uint16_t) + sizeof(QuantType)))
                           << ", dense_window_count_bytes="
                           << (static_cast<size_t>(nr_windows_) * dim_count * sizeof(uint16_t))
                           << ", workers=" << concurrency;

        parallel_for_workers(concurrency, [&](size_t worker_id) {
            const auto [begin, end] = get_worker_row_range(rows, worker_id, concurrency);
            auto& worker_cursors = posting_plan.cursors_by_worker[worker_id];
            for (size_t row_id = begin; row_id < end; ++row_id) {
                const size_t window_id = row_id / window_size_;
                const auto local_id = static_cast<uint16_t>(row_id % window_size_);
                for (size_t j = 0; j < data[row_id].size(); ++j) {
                    const auto [dim, val] = data[row_id][j];
                    if (std::abs(val) < std::numeric_limits<DataType>::epsilon()) {
                        continue;
                    }
                    const auto inner_dim = this->dim_map_.lookup_trusted(dim);
                    const auto worker_offset = worker_cursors[inner_dim]++;
                    const size_t absolute_posting_offset =
                        posting_plan.cursor_mode == WorkerCursorMode::Absolute
                            ? worker_offset
                            : posting_plan.posting_offsets[inner_dim] + worker_offset;
                    total_plists_ids_flat_[absolute_posting_offset] = local_id;
                    store_sealed_value(absolute_posting_offset, val);
                    if constexpr (is_bm25_u8) {
                        const uint16_t overflow_value = bm25_u16_value(val);
                        if (overflow_value > 255) {
                            bm25_u8_overflows_by_worker[worker_id].push_back(
                                {static_cast<uint32_t>(absolute_posting_offset), overflow_value});
                        }
                    }
                    std::atomic_ref<uint16_t>(window_index_plists_sz_[window_id][inner_dim])
                        .fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
        finalize_sealed_bm25_u8_overflows(bm25_u8_overflows_by_worker);
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex preallocated postings filled: total_postings=" << total_postings;

        WorkerPostingCursors{}.swap(posting_plan.cursors_by_worker);
        std::vector<size_t>{}.swap(posting_plan.posting_offsets);

        max_scores_per_dim_.assign(dim_count, 0.0f);
        float bm25_p1 = 0.0f;
        float bm25_p2 = 0.0f;
        float bm25_p3 = 0.0f;
        if constexpr (is_bm25) {
            const auto& cfg = this->build_scorer_->config();
            bm25_p1 = cfg.scorer_params.bm25.k1 + 1.0f;
            bm25_p2 = cfg.scorer_params.bm25.k1 * (1.0f - cfg.scorer_params.bm25.b);
            bm25_p3 = cfg.scorer_params.bm25.k1 * cfg.scorer_params.bm25.b / cfg.scorer_params.bm25.avgdl;
        }

        parallel_for(dim_count, [&](size_t dim_id) {
            const size_t begin = plists_dim_offsets_[dim_id];
            const size_t end = plists_dim_offsets_[dim_id + 1];

            float max_score = 0.0f;
            if constexpr (is_ip) {
                for (size_t offset = begin; offset < end; ++offset) {
                    const float value = std::abs(static_cast<float>(total_plists_vals_flat_[offset]));
                    max_score = std::max(max_score, value);
                }
            } else {
                auto [overflow_cursor, overflow_end] = bm25_u8_overflow_range(begin, end);
                size_t posting_offset = 0;
                for (size_t window_id = 0; window_id < nr_windows_; ++window_id) {
                    const size_t window_nnz = window_index_plists_sz_[window_id][dim_id];
                    for (size_t i = 0; i < window_nnz; ++i, ++posting_offset) {
                        const size_t offset = begin + posting_offset;
                        const uint32_t global_id = window_id * window_size_ + total_plists_ids_flat_[offset];
                        const float value =
                            bm25_posting_value(offset, total_plists_vals_flat_[offset], overflow_cursor, overflow_end);
                        const float dl = row_sums_[global_id];
                        max_score = std::max(max_score, bm25_p1 * value / (value + bm25_p2 + bm25_p3 * dl));
                    }
                }
                assert(posting_offset == end - begin);
            }
            max_scores_per_dim_[dim_id] = max_score;
        });
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex posting lists materialized: dims=" << dim_count
                           << ", total_postings=" << total_postings << ", windows=" << nr_windows_;

        plists_dim_offsets_span_ = std::span<const uint32_t>(plists_dim_offsets_.data(), plists_dim_offsets_.size());
        max_scores_per_dim_span_ = std::span<const float>(max_scores_per_dim_.data(), max_scores_per_dim_.size());

        for (size_t wid = 0; wid < nr_windows_; ++wid) {
            window_index_plists_sz_spans_[wid] =
                std::span<const uint16_t>(window_index_plists_sz_[wid].data(), window_index_plists_sz_[wid].size());
        }
        encode_window_nnzs_flat(/*parallel=*/true);

        // The sealed query and serialization paths only use the encoded counts. Keep the dense matrix solely as a
        // build-time alternative when it is smaller than one window id per posting.
        window_index_plists_sz_spans_.clear();
        window_index_plists_sz_spans_.shrink_to_fit();
        window_index_plists_sz_.clear();
        window_index_plists_sz_.shrink_to_fit();
    }

    void
    build_window_indexes_parallel(const SparseRow<DataType>* data, size_t rows, PostingBuildPlan posting_plan) {
        const size_t dim_count = this->nr_inner_dims_;
        const size_t total_postings = posting_plan.total_postings();
        if (total_postings > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("SindiInvertedIndex posting count exceeds uint32_t offset capacity");
        }

        total_plists_ids_.clear();
        total_plists_vals_.clear();
        total_plists_ids_spans_.clear();
        total_plists_vals_spans_.clear();
        total_plists_ids_flat_.resize(total_postings);
        total_plists_vals_flat_.resize(total_postings);
        total_plists_ids_flat_span_ =
            std::span<const uint16_t>(total_plists_ids_flat_.data(), total_plists_ids_flat_.size());
        total_plists_vals_flat_span_ =
            std::span<const QuantType>(total_plists_vals_flat_.data(), total_plists_vals_flat_.size());
        plists_dim_offsets_.resize(dim_count + 1);
        std::transform(posting_plan.posting_offsets.begin(), posting_plan.posting_offsets.end(),
                       plists_dim_offsets_.begin(), [](size_t offset) { return static_cast<uint32_t>(offset); });

        nr_windows_ = (rows + window_size_ - 1) / window_size_;
        size_t window_id_bytes = sizeof(uint32_t);
        if (nr_windows_ <= static_cast<size_t>(std::numeric_limits<uint8_t>::max()) + 1) {
            window_id_bytes = sizeof(uint8_t);
        } else if (nr_windows_ <= static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1) {
            window_id_bytes = sizeof(uint16_t);
        }
        const auto saturating_multiply = [](size_t lhs, size_t rhs) {
            return lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs ? std::numeric_limits<size_t>::max()
                                                                              : lhs * rhs;
        };
        const size_t dense_window_count_bytes =
            saturating_multiply(saturating_multiply(static_cast<size_t>(nr_windows_), dim_count), sizeof(uint16_t));
        const size_t posting_window_id_bytes = saturating_multiply(total_postings, window_id_bytes);
        const bool use_posting_window_ids = posting_window_id_bytes <= dense_window_count_bytes;
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex window count build strategy: strategy="
                           << (use_posting_window_ids ? "posting_window_ids" : "dense_window_counts")
                           << ", posting_window_id_bytes=" << posting_window_id_bytes
                           << ", dense_window_count_bytes=" << dense_window_count_bytes;

        if (!use_posting_window_ids) {
            build_window_indexes_parallel_dense_window_counts(data, rows, posting_plan);
        } else if (window_id_bytes == sizeof(uint8_t)) {
            build_window_indexes_parallel_posting_window_ids<uint8_t>(data, rows, posting_plan);
        } else if (window_id_bytes == sizeof(uint16_t)) {
            build_window_indexes_parallel_posting_window_ids<uint16_t>(data, rows, posting_plan);
        } else {
            build_window_indexes_parallel_posting_window_ids<uint32_t>(data, rows, posting_plan);
        }
    }

    Status
    add(const SparseRow<DataType>* data, size_t rows, int64_t dim) override {
        if constexpr (!AllowIncremental) {
            if (this->nr_rows_ != 0) {
                return Status::not_implemented;
            }
        }

        if (packed_) {
            if (rows > uint64_t(std::numeric_limits<uint32_t>::max()) - this->nr_rows_) {
                return Status::invalid_args;
            }

            uint64_t postings = plists_dim_offsets_span_.empty() ? 0 : plists_dim_offsets_span_.back();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < data[i].size(); ++j) {
                    postings += std::abs(data[i][j].val) >= std::numeric_limits<DataType>::epsilon();
                }

                if (postings > std::numeric_limits<uint32_t>::max()) {
                    return Status::invalid_args;
                }
            }
        }

        if (refine_ || packed_) {
            if constexpr (!is_ip) {
                if (!packed_ || refine_)
                    return Status::invalid_args;
            }

            for (size_t i = 0; i < rows; ++i) {
                if (!sindi::valid_refinement_row(data[i])) {
                    return Status::invalid_args;
                }

                for (size_t j = 0; j < data[i].size(); ++j) {
                    const float value = data[i][j].val;
                    if ((is_ip && !std::isfinite(static_cast<float>(knowhere::fp16(value)))) ||
                        (packed_ && std::signbit(value)) ||
                        (is_bm25 && (value > 65535 || std::floor(value) != value))) {
                        return Status::invalid_args;
                    }
                }
            }
        }

        if constexpr (AllowIncremental) {
            if (packed_ready_) {
                expand_for_add();
            }
        }

        const size_t old_nr_rows = this->nr_rows_;
        this->max_dim_ = std::max(this->max_dim_, static_cast<uint32_t>(dim));
        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex build started: rows=" << rows << ", existing_rows=" << old_nr_rows
                           << ", max_dim=" << this->max_dim_ << ", window_size=" << window_size_
                           << ", metric=" << (is_ip ? "IP" : "BM25") << ", incremental=" << AllowIncremental;

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
            auto row_scan = scan_rows_for_build(data, rows);
            LOG_KNOWHERE_INFO_ << "SindiInvertedIndex row scan completed: external_dims="
                               << row_scan.external_dims.size();
            this->dim_map_.build_from_external_dims(row_scan.external_dims);
            this->nr_inner_dims_ = this->dim_map_.size();
            DimensionSet{}.swap(row_scan.external_dims);
            auto posting_plan = prepare_posting_build_plan(std::move(row_scan.posting_counts_by_worker), this->dim_map_,
                                                           this->nr_inner_dims_);
            LOG_KNOWHERE_INFO_ << "SindiInvertedIndex posting plan completed: inner_dims=" << this->nr_inner_dims_
                               << ", total_postings=" << posting_plan.total_postings();

            if constexpr (is_bm25) {
                row_sums_.resize(rows);
                parallel_for(rows, [&](size_t i) {
                    float row_sum = 0.0f;
                    for (size_t j = 0; j < data[i].size(); ++j) {
                        const auto value = std::abs(data[i][j].val);
                        if (value >= std::numeric_limits<DataType>::epsilon()) {
                            row_sum += static_cast<float>(value);
                        }
                    }
                    row_sums_[i] = row_sum;
                });
                row_sums_span_ = std::span<const float>(row_sums_.data(), row_sums_.size());
            }

            build_window_indexes_parallel(data, rows, std::move(posting_plan));
            this->nr_rows_ = rows;
            if (packed_)
                pack_postings();

            if (refine_) {
                rebuild_refinement_seek();
            }

            LOG_KNOWHERE_INFO_ << "SindiInvertedIndex build completed: rows=" << this->nr_rows_
                               << ", inner_dims=" << this->nr_inner_dims_ << ", windows=" << nr_windows_
                               << ", index_bytes=" << size();
            return Status::success;
        }

        // Incremental mode retains the append-oriented representation and update path.
        append_window_indexes(data, rows);
        this->nr_rows_ += rows;

        // Compute row sums for BM25 support.
        if constexpr (is_bm25) {
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
        if constexpr (is_ip) {
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
                    const float tf = static_cast<float>(bm25_u16_value(abs_val));
                    const float bm25_score = p1 * tf / (tf + p2 + p3 * dl);
                    if (bm25_score > max_scores_per_dim_[dim_id]) {
                        max_scores_per_dim_[dim_id] = bm25_score;
                    }
                }
            }
        }
        max_scores_per_dim_span_ = std::span<const float>(max_scores_per_dim_.data(), max_scores_per_dim_.size());

        if (packed_) {
            pack_postings();
        }

        if (refine_) {
            rebuild_refinement_seek();
        }

        LOG_KNOWHERE_INFO_ << "SindiInvertedIndex incremental build completed: rows=" << this->nr_rows_
                           << ", inner_dims=" << this->nr_inner_dims_ << ", windows=" << nr_windows_
                           << ", index_bytes=" << size();

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
            const auto quant_type = packed_ ? packed_quant_type() : posting_quant_type<QuantType>();
            writer.write(&quant_type, sizeof(quant_type));
            const std::array<uint8_t, kInvertedIndexHeaderReservedBytes> reserved{};
            writer.write(reserved.data(), reserved.size());

            const bool has_row_sums = !row_sums_span_.empty();
            const bool has_bm25_u8_overflows = !bm25_u8_overflow_offsets_span_.empty();
            const auto dim_map_storage = legacy_dim_map_mphf_trailer_workaround_ ? DimMapMphfStorage::LegacyTrailer
                                                                                 : DimMapMphfStorage::SeparateSection;

            uint32_t nr_sections = 3;  // base sections: inverted index, dim map and max scores per dim
            if (this->dim_map_.has_mphf_section(dim_map_storage)) {
                nr_sections += 1;
            }
            if (has_row_sums) {
                nr_sections += 1;
            }
            if (has_bm25_u8_overflows) {
                nr_sections += 1;
            }
            if (refine_) {
                nr_sections += 1;
            }

            writer.write(&nr_sections, sizeof(uint32_t));

            const size_t nr_dims = this->nr_inner_dims_;

            std::vector<InvertedIndexSectionHeader> section_headers(nr_sections);
            uint64_t used_offset = first_section_offset(nr_sections);
            section_headers[0].type = InvertedIndexSectionType::POSTING_LISTS;
            section_headers[0].size = [&, this]() -> uint64_t {
                size_t res = posting_header_bytes();

                const size_t mask_sz = (nr_dims + 7) / 8;
                res += mask_sz * sizeof(uint8_t);

                for (size_t dimid = 0; dimid < nr_dims; ++dimid) {
                    const auto nnz_span = encoded_window_nnzs(dimid);
                    res += sizeof(uint32_t);
                    res += nnz_span.size();
                }

                res += sizeof(uint32_t) * (nr_dims + 1);

                if (nr_windows_ == 0) {
                    return res;
                }

                auto total_postings = plists_dim_offsets_span_[nr_dims];
                res += packed_ ? packed_payload_bytes(total_postings)
                               : total_postings * (sizeof(uint16_t) + sizeof(QuantType));

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

            if (has_bm25_u8_overflows) {
                section_headers[curr_section_idx].type = InvertedIndexSectionType::BM25_U8_OVERFLOWS;
                section_headers[curr_section_idx].size =
                    sizeof(uint32_t) + bm25_u8_overflow_offsets_span_.size() * (sizeof(uint32_t) + sizeof(uint16_t));
                assign_section_offset(section_headers[curr_section_idx], used_offset);
                curr_section_idx++;
            }

            if (has_row_sums) {
                section_headers[curr_section_idx].type = InvertedIndexSectionType::ROW_SUMS;
                section_headers[curr_section_idx].size = sizeof(float) * this->nr_rows_;
                assign_section_offset(section_headers[curr_section_idx], used_offset);
                curr_section_idx++;
            }

            if (refine_) {
                auto& section = section_headers[curr_section_idx++];
                section.type = InvertedIndexSectionType::SINDI_REFINEMENT;
                section.size = 3 * sizeof(uint32_t);
                assign_section_offset(section, used_offset);
            }

            assert(curr_section_idx == nr_sections);

            writer.write(section_headers.data(), sizeof(InvertedIndexSectionHeader), nr_sections);

            uint32_t index_encoding_type =
                static_cast<uint32_t>(packed_ ? packed_encoding() : InvertedIndexEncoding::FIXED_DOCID_WINDOWS);
            write_padding_until(writer, section_headers[0].offset);
            writer.write(&index_encoding_type, sizeof(uint32_t));
            writer.write(&this->window_size_, sizeof(uint32_t));
            writer.write(&this->nr_windows_, sizeof(uint32_t));
            if (packed_) {
                // Version 1: explicit ID layout and independent quantization identity.
                const uint32_t descriptor[] = {1, packed_id_layout(), static_cast<uint32_t>(packed_quant_type())};
                writer.write(descriptor, sizeof(descriptor));
                if constexpr (is_bm25) {
                    const auto& cfg = this->build_scorer_->config().scorer_params.bm25;
                    writer.write(lut_.decode.data(), lut_.decode.size());
                    writer.write(lut_.ends.data(), lut_.ends.size());
                    const float params[] = {cfg.k1, cfg.b, cfg.avgdl};
                    writer.write(params, sizeof(params));
                }
            }

            // write plists_woffsets_formats_mask and plists_window_nnzs
            writer.write(plists_wnnzs_fmts_msk_span_.data(), sizeof(uint8_t), plists_wnnzs_fmts_msk_span_.size());
            for (size_t dimid = 0; dimid < nr_dims; ++dimid) {
                const auto span = encoded_window_nnzs(dimid);
                uint32_t span_sz = static_cast<uint32_t>(span.size());
                writer.write(&span_sz, sizeof(uint32_t));
                writer.write(span.data(), sizeof(uint8_t), span.size());
            }

            // write plists_dim_offsets
            writer.write(plists_dim_offsets_span_.data(), sizeof(uint32_t), plists_dim_offsets_span_.size());

            // write total_plists_ids / vals per dim, concatenated
            if (packed_) {
                writer.write(packed_ids_span_.data(), packed_ids_span_.size());
                writer.write(packed_vals_span_.data(), packed_vals_span_.size());
            } else if (nr_windows_ > 0 && nr_dims > 0) {
                // ids
                for (size_t dim_id = 0; dim_id < nr_dims; ++dim_id) {
                    const auto ids = posting_ids(dim_id);
                    writer.write(ids.data(), sizeof(uint16_t), ids.size());
                }
                // vals
                for (size_t dim_id = 0; dim_id < nr_dims; ++dim_id) {
                    const auto vals = posting_vals(dim_id);
                    writer.write(vals.data(), sizeof(QuantType), vals.size());
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

            if (has_bm25_u8_overflows) {
                write_padding_until(writer, section_headers[curr_section_idx].offset);
                const auto overflow_count = static_cast<uint32_t>(bm25_u8_overflow_offsets_span_.size());
                writer.write(&overflow_count, sizeof(uint32_t));
                writer.write(bm25_u8_overflow_offsets_span_.data(), sizeof(uint32_t), overflow_count);
                writer.write(bm25_u8_overflow_values_span_.data(), sizeof(uint16_t), overflow_count);
                curr_section_idx++;
            }

            if (has_row_sums) {
                write_padding_until(writer, section_headers[curr_section_idx].offset);
                writer.write(row_sums_span_.data(), sizeof(float), row_sums_span_.size());
                curr_section_idx++;
            }

            if (refine_) {
                write_padding_until(writer, section_headers[curr_section_idx].offset);
                // Section version 1; independent ID layout (U16=1) and value codec.
                const uint32_t metadata[] = {
                    1, packed_ ? 2u : 1u,
                    static_cast<uint32_t>(packed_ ? InvertedIndexQuantType::IP_E5M7 : InvertedIndexQuantType::IP_FP16)};
                writer.write(metadata, sizeof(metadata));
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
            refine_ = false;
            refinement_seek_.clear();
            if (packed_ && !validate_packed_file(reader)) {
                return Status::invalid_serialized_index_type;
            }

            auto file_header_handler = [&, this]() {
                uint32_t index_format_version = 0;
                reader.read(&index_format_version, sizeof(uint32_t));
                if (index_format_version != kInvertedIndexFileFormatVersion) {
                    return Status::invalid_serialized_index_type;
                }

                reader.read(&this->nr_rows_, sizeof(uint32_t));
                reader.read(&this->max_dim_, sizeof(uint32_t));
                reader.read(&this->nr_inner_dims_, sizeof(uint32_t));
                InvertedIndexQuantType quant_type{};
                reader.read(&quant_type, sizeof(quant_type));
                if (packed_ ? quant_type != packed_quant_type() : !validate_posting_quant_type<QuantType>(quant_type)) {
                    return Status::invalid_serialized_index_type;
                }
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
                                static_cast<uint32_t>(packed_ ? packed_encoding()
                                                              : InvertedIndexEncoding::FIXED_DOCID_WINDOWS)) {
                                return Status::invalid_serialized_index_type;
                            }

                            // check window params
                            reader.read(&this->window_size_, sizeof(uint32_t));
                            reader.read(&this->nr_windows_, sizeof(uint32_t));
                            if (packed_) {
                                reader.advance(3 * sizeof(uint32_t));  // descriptor checked in preflight
                                if constexpr (is_bm25) {
                                    reader.read(lut_.decode.data(), lut_.decode.size());
                                    reader.read(lut_.ends.data(), lut_.ends.size());
                                    float params[3];
                                    reader.read(params, sizeof(params));
                                    lut_.k1 = params[0];
                                    lut_.validate_and_encode();
                                    this->set_build_scorer(IndexScorerConfig{
                                        .scorer_type = IndexScorerType::BM25,
                                        .scorer_params = {
                                            .bm25 = {.k1 = params[0], .b = params[1], .avgdl = params[2]}}});
                                }
                            }

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
                            const uint64_t bytes_header = static_cast<uint64_t>(posting_header_bytes());

                            window_index_plists_sz_.clear();
                            window_index_plists_sz_spans_.clear();
                            total_plists_ids_.clear();
                            total_plists_vals_.clear();
                            total_plists_ids_flat_.clear();
                            total_plists_vals_flat_.clear();
                            total_plists_ids_spans_.clear();
                            total_plists_vals_spans_.clear();
                            total_plists_ids_flat_span_ = {};
                            total_plists_vals_flat_span_ = {};
                            bm25_u8_overflow_offsets_.clear();
                            bm25_u8_overflow_values_.clear();
                            bm25_u8_overflow_offsets_span_ = {};
                            bm25_u8_overflow_values_span_ = {};
                            plists_window_nnzs_.clear();
                            plists_window_nnzs_spans_.assign(nr_dims, {});
                            plists_window_nnzs_flat_.clear();
                            plists_window_nnzs_offsets_.clear();
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
                            if (packed_) {
                                // Packed payload has byte alignment. Metadata uses memcpy, never typed byte casts.
                                plists_dim_offsets_.resize(nr_dims + 1);
                                reader.read(plists_dim_offsets_.data(), (nr_dims + 1) * sizeof(uint32_t));
                                plists_dim_offsets_span_ = plists_dim_offsets_;
                            } else {
                                plists_dim_offsets_span_ = std::span<const uint32_t>(
                                    reinterpret_cast<const uint32_t*>(reader.data() + reader.tellg()), nr_dims + 1);
                                reader.advance((nr_dims + 1) * sizeof(uint32_t));
                            }
                            total_postings = plists_dim_offsets_span_[nr_dims];

                            // Validate total_postings against section size
                            const uint64_t bytes_dim_offsets = static_cast<uint64_t>(nr_dims + 1) * sizeof(uint32_t);
                            const uint64_t bytes_postings_data = packed_ ? packed_payload_bytes(total_postings)
                                                                         : static_cast<uint64_t>(total_postings) *
                                                                               (sizeof(uint16_t) + sizeof(QuantType));
                            const uint64_t expected_section_bytes =
                                bytes_header + bytes_mask + bytes_win_nnzs + bytes_dim_offsets + bytes_postings_data;
                            if (expected_section_bytes != section_header.size) {
                                LOG_KNOWHERE_INFO_ << "SindiInvertedIndex::deserialize POSTING_LISTS size mismatch: "
                                                   << "expected=" << expected_section_bytes
                                                   << " actual=" << section_header.size;
                                return Status::invalid_serialized_index_type;
                            }

                            const uint64_t bytes_ids =
                                packed_ ? packed_id_bytes(total_postings) : total_postings * sizeof(uint16_t);
                            const uint64_t bytes_vals =
                                packed_ ? packed_value_bytes(total_postings) : total_postings * sizeof(QuantType);
                            if (packed_) {
                                packed_ids_span_ = {reader.data() + reader.tellg(), static_cast<size_t>(bytes_ids)};
                                reader.advance(bytes_ids);
                                packed_vals_span_ = {reader.data() + reader.tellg(), static_cast<size_t>(bytes_vals)};
                                reader.advance(bytes_vals);
                                packed_ready_ = true;
                            } else {
                                total_plists_ids_flat_span_ = {
                                    reinterpret_cast<const uint16_t*>(reader.data() + reader.tellg()),
                                    static_cast<size_t>(total_postings)};
                                reader.advance(bytes_ids);
                                total_plists_vals_flat_span_ = {
                                    reinterpret_cast<const QuantType*>(reader.data() + reader.tellg()),
                                    static_cast<size_t>(total_postings)};
                                reader.advance(bytes_vals);
                            }

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
                            if (packed_) {
                                max_scores_per_dim_.resize(this->nr_inner_dims_);
                                reader.read(max_scores_per_dim_.data(), sizeof(float) * this->nr_inner_dims_);
                                max_scores_per_dim_span_ = max_scores_per_dim_;
                            } else {
                                max_scores_per_dim_span_ = std::span<const float>(
                                    reinterpret_cast<const float*>(reader.data() + section_header.offset),
                                    this->nr_inner_dims_);
                                reader.advance(sizeof(float) * this->nr_inner_dims_);
                            }
                            break;
                        }
                        case InvertedIndexSectionType::BM25_U8_OVERFLOWS: {
                            if constexpr (!is_bm25_u8) {
                                return Status::invalid_serialized_index_type;
                            }
                            if (section_header.size < sizeof(uint32_t)) {
                                return Status::invalid_serialized_index_type;
                            }
                            reader.seekg(section_header.offset);
                            uint32_t overflow_count = 0;
                            reader.read(&overflow_count, sizeof(uint32_t));
                            const uint64_t expected_size = sizeof(uint32_t) + static_cast<uint64_t>(overflow_count) *
                                                                                  (sizeof(uint32_t) + sizeof(uint16_t));
                            if (section_header.size != expected_size) {
                                return Status::invalid_serialized_index_type;
                            }

                            const auto* offsets = reinterpret_cast<const uint32_t*>(reader.data() + reader.tellg());
                            bm25_u8_overflow_offsets_span_ = std::span<const uint32_t>(offsets, overflow_count);
                            reader.advance(static_cast<size_t>(overflow_count) * sizeof(uint32_t));
                            const auto* values = reinterpret_cast<const uint16_t*>(reader.data() + reader.tellg());
                            bm25_u8_overflow_values_span_ = std::span<const uint16_t>(values, overflow_count);
                            reader.advance(static_cast<size_t>(overflow_count) * sizeof(uint16_t));

                            if (!std::is_sorted(bm25_u8_overflow_offsets_span_.begin(),
                                                bm25_u8_overflow_offsets_span_.end()) ||
                                (!bm25_u8_overflow_offsets_span_.empty() &&
                                 bm25_u8_overflow_offsets_span_.back() >= total_postings)) {
                                return Status::invalid_serialized_index_type;
                            }
                            for (uint16_t value : bm25_u8_overflow_values_span_) {
                                if (value <= 255) {
                                    return Status::invalid_serialized_index_type;
                                }
                            }
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
                        case InvertedIndexSectionType::SINDI_REFINEMENT: {
                            if (!is_ip || refine_ || section_header.size != 3 * sizeof(uint32_t) ||
                                section_header.offset > reader.total_ ||
                                section_header.size > reader.total_ - section_header.offset) {
                                return Status::invalid_serialized_index_type;
                            }

                            reader.seekg(section_header.offset);
                            uint32_t metadata[3];
                            reader.read(metadata, sizeof(metadata));
                            if (metadata[0] != 1 || metadata[1] != (packed_ ? 2u : 1u) ||
                                metadata[2] != static_cast<uint32_t>(packed_ ? InvertedIndexQuantType::IP_E5M7
                                                                             : InvertedIndexQuantType::IP_FP16)) {
                                return Status::invalid_serialized_index_type;
                            }

                            refine_ = true;
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

            if (packed_ && is_bm25) {
                for (const auto dl : row_sums_span_)
                    if (!std::isfinite(dl) || dl < 0)
                        return Status::invalid_serialized_index_type;
            }
            if (refine_ || packed_) {
                try {
                    rebuild_refinement_seek();
                    if (packed_) {
                        for (size_t dim = 0; dim < this->nr_inner_dims_; ++dim) {
                            float maximum = packed_dimension_maximum(dim, refinement_seek_[dim]);

                            if (max_scores_per_dim_span_[dim] != maximum) {
                                return Status::invalid_serialized_index_type;
                            }
                        }

                        if (!refine_) {
                            std::vector<sindi::RefinementSeek>{}.swap(refinement_seek_);
                        }
                    }
                } catch (const std::exception&) {
                    return Status::invalid_serialized_index_type;
                }
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

        if (refine_) {
            search_refined(query, k, distances, labels, bitset, search_params);
            return;
        }
        auto q_vec = parse_query_with_dim_map(query, this->dim_map_, search_params.approx.drop_ratio_search);
        search_coarse(std::move(q_vec), k, distances, labels, bitset, search_params);
    }

    void
    search_coarse(std::vector<std::pair<uint32_t, float>> q_vec, size_t k, float* distances, label_t* labels,
                  const BitsetView& bitset, const InvertedIndexSearchParams& search_params) const {
        if (q_vec.empty() || k == 0) {
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

        // Skip a window when every document in it is filtered out by the bitset.
        const auto window_all_filtered = [&bitset](size_t docid_start, size_t window_doc_count) {
            if (bitset.empty()) {
                return false;
            }
            const size_t end = docid_start + window_doc_count;
            if (end > bitset.size()) {
                return false;
            }
            return bitset.range_all_filtered(docid_start, end);
        };

        // Initialize posting list cursors for each query term
        // Cursors track position in posting lists and handle window-by-window iteration
        std::vector<PostingCursor> cursors;
        cursors.reserve(sorted_q_vec.size());
        std::vector<Bm25U8QueryOverflowCursor> overflow_cursors;
        if constexpr (is_bm25_u8) {
            if (!bm25_u8_overflow_offsets_span_.empty()) {
                overflow_cursors.reserve(sorted_q_vec.size());
            }
        }
        for (auto& [qid, qval] : sorted_q_vec) {
            const auto wnnz_buf_span = encoded_window_nnzs(qid);
            const auto ids = packed_ready_ ? std::span<const uint16_t>{} : posting_ids(qid);
            const auto vals = packed_ready_ ? std::span<const QuantType>{} : posting_vals(qid);

            bool is_sparse = !plists_wnnzs_fmts_msk_span_.empty() &&
                             ((plists_wnnzs_fmts_msk_span_[qid >> 3] & static_cast<uint8_t>(0x1u << (qid & 0x7))) != 0);

            cursors.emplace_back(static_cast<uint32_t>(qid),  // dim_id
                                 qval,                        // qval
                                 ids.data(),                  // ids_base
                                 vals.data(),                 // vals_base
                                 wnnz_buf_span.data(),        // wnnz_buf
                                 wnnz_buf_span.size(),        // wnnz_buf_sz
                                 is_sparse,                   // is_sparse
                                 0,                           // cursor
                                 0,                           // offset
                                 wnnz_bits,                   // wnnz_bits
                                 wnnz_mask                    // wnnz_mask
            );
            if constexpr (is_bm25_u8) {
                if (!bm25_u8_overflow_offsets_span_.empty()) {
                    const uint32_t posting_begin = plists_dim_offsets_span_[qid];
                    const auto [overflow_begin, overflow_end] =
                        bm25_u8_overflow_range(posting_begin, plists_dim_offsets_span_[qid + 1]);
                    if (overflow_begin != overflow_end) {
                        overflow_cursors.emplace_back(cursors.size() - 1, posting_begin,
                                                      bm25_u8_overflow_offsets_span_.data(), overflow_begin,
                                                      overflow_end);
                    }
                }
            }
        }

        // Main search loop: iterate over windows and process each window
        if constexpr (is_ip) {
            const auto scatter_fn = sindi::get_ip_kernels().accumulate;
            const auto packed_fn = packed_ ? sindi::get_packed_ip_kernel() : nullptr;
            const auto batch_insert_fn = sindi::get_ip_kernels().batch_insert;

            for (size_t widx = 0; widx < nr_windows_; ++widx) {
                const size_t docid_start = window_size_ * widx;
                const uint32_t curr_window_size =
                    std::min(window_size_, static_cast<uint32_t>(this->nr_rows_ - docid_start));
                if (window_all_filtered(docid_start, curr_window_size)) {
                    // Keep posting cursors in sync even when the window is skipped, so the
                    // cumulative posting offsets stay correct for subsequent windows.
                    for (auto& cur : cursors) {
                        if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                            continue;
                        }
                        cur.advance_window(widx);
                    }
                    continue;
                }
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

                    float dispatch_max =
                        packed_ ? packed_fn(cur.qval, packed_vals_span_.data(), packed_ids_span_.data(),
                                            size_t(plists_dim_offsets_span_[cur.dim_id]) + soff, wnnz, wscores_final)
                                : scatter_fn(cur.qval, cur.vals_base + soff, cur.ids_base + soff, wnnz, wscores_final);
                    if (dispatch_max > curr_max_score) {
                        curr_max_score = dispatch_max;
                    }
                }

                if (curr_max_score > threshold) {
                    batch_insert_fn(wscores_final, docid_start, curr_window_size, topk_q, threshold, bitset);
                }
            }
        } else {
            const auto accumulate_fn = []() {
                if constexpr (is_bm25_u8) {
                    return sindi::get_bm25_u8_kernels().accumulate;
                } else {
                    return sindi::get_bm25_kernels().accumulate;
                }
            }();
            const auto batch_insert_fn = []() {
                if constexpr (is_bm25_u8) {
                    return sindi::get_bm25_u8_kernels().batch_insert;
                } else {
                    return sindi::get_bm25_kernels().batch_insert;
                }
            }();

            const auto packed_bm25_fn = packed_ ? sindi::get_packed_bm25_kernel(packed_u16_ids_) : nullptr;
            const float bm25_k1 = search_params.scorer_config.scorer_params.bm25.k1;
            const float bm25_b = search_params.scorer_config.scorer_params.bm25.b;
            const float bm25_avgdl = search_params.scorer_config.scorer_params.bm25.avgdl;
            const float* row_sums_ptr = row_sums_span_.data();

            // Compile a correction-free loop for the overwhelmingly common case where the index has no
            // overflowing TFs. This keeps BM25 U8's hot loop identical to clamp-only U8 on those datasets.
            const auto search_bm25_windows = [&]<bool RestoreOverflow>() {
                for (size_t widx = 0; widx < nr_windows_; ++widx) {
                    const size_t docid_start = window_size_ * widx;
                    const uint32_t curr_window_size =
                        std::min(window_size_, static_cast<uint32_t>(this->nr_rows_ - docid_start));
                    if (window_all_filtered(docid_start, curr_window_size)) {
                        // Keep posting cursors in sync even when the window is skipped, so the
                        // cumulative posting offsets stay correct for subsequent windows.
                        for (auto& cur : cursors) {
                            if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                                continue;
                            }
                            cur.advance_window(widx);
                        }
                        if constexpr (RestoreOverflow) {
                            for (auto& overflow_query_cur : overflow_cursors) {
                                overflow_query_cur.cursor.advance(cursors[overflow_query_cur.query_cursor].offset);
                            }
                        }
                        continue;
                    }
                    std::fill_n(wscores_final, curr_window_size, 0);

                    float curr_max_score = 0.0f;
                    bool skip_window_calc = false;
                    size_t next_overflow_cursor = 0;
                    for (size_t ci = 0; ci < cursors.size(); ++ci) {
                        auto& cur = cursors[ci];
                        if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                            continue;
                        }

                        auto [soff, wnnz] = cur.advance_window(widx);
                        Bm25U8OverflowCursor* overflow_cur = nullptr;
                        if constexpr (RestoreOverflow) {
                            if (next_overflow_cursor < overflow_cursors.size() &&
                                overflow_cursors[next_overflow_cursor].query_cursor == ci) {
                                overflow_cur = &overflow_cursors[next_overflow_cursor++].cursor;
                                overflow_cur->advance(cur.offset);
                            }
                        }

                        if (skip_window_calc || wnnz == 0) {
                            continue;
                        }

                        // Skip window calculation if current max + remaining max contributions <= threshold
                        if (curr_max_score + search_params.approx.dim_max_score_ratio * suffix_sum[ci] <= threshold) {
                            skip_window_calc = true;
                            continue;
                        }

                        const uint16_t* plist_ids = packed_ ? nullptr : cur.ids_base + soff;
                        const QuantType* plist_vals = packed_ ? nullptr : cur.vals_base + soff;

                        float dispatch_max =
                            packed_ ? packed_bm25_fn(cur.qval, packed_vals_span_.data(), packed_ids_span_.data(),
                                                     size_t(plists_dim_offsets_span_[cur.dim_id]) + soff, wnnz,
                                                     wscores_final, bm25_k1, bm25_b, bm25_avgdl,
                                                     row_sums_ptr + docid_start, lut_.decode.data())
                                    : accumulate_fn(cur.qval, plist_vals, plist_ids, wnnz, wscores_final, bm25_k1,
                                                    bm25_b, bm25_avgdl, row_sums_ptr + docid_start);
                        if constexpr (RestoreOverflow) {
                            if (overflow_cur != nullptr && overflow_cur->window_begin != overflow_cur->window_end) {
                                dispatch_max = apply_bm25_u8_overflow_corrections(
                                    overflow_cur->posting_begin, soff, plist_ids, wnnz, overflow_cur->window_begin,
                                    overflow_cur->window_end, cur.qval, wscores_final, dispatch_max, bm25_k1, bm25_b,
                                    bm25_avgdl, row_sums_ptr + docid_start);
                            }
                        }
                        if (dispatch_max > curr_max_score) {
                            curr_max_score = dispatch_max;
                        }
                    }

                    if (curr_max_score > threshold) {
                        batch_insert_fn(wscores_final, docid_start, curr_window_size, topk_q, threshold, bitset);
                    }
                }
            };

            if constexpr (is_bm25_u8) {
                if (overflow_cursors.empty()) {
                    search_bm25_windows.template operator()<false>();
                } else {
                    search_bm25_windows.template operator()<true>();
                }
            } else {
                search_bm25_windows.template operator()<false>();
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
        std::vector<Bm25U8QueryOverflowCursor> overflow_cursors;
        if constexpr (is_bm25_u8) {
            if (!bm25_u8_overflow_offsets_span_.empty()) {
                overflow_cursors.reserve(q_vec.size());
            }
        }
        for (auto& [qid, qval] : q_vec) {
            const auto nnz_span = encoded_window_nnzs(qid);
            const auto ids = packed_ready_ ? std::span<const uint16_t>{} : posting_ids(qid);
            const auto vals = packed_ready_ ? std::span<const QuantType>{} : posting_vals(qid);

            bool is_sparse = !plists_wnnzs_fmts_msk_span_.empty() &&
                             ((plists_wnnzs_fmts_msk_span_[qid >> 3] & static_cast<uint8_t>(0x1u << (qid & 0x7))) != 0);

            cursors.emplace_back(static_cast<uint32_t>(qid),  // dim_id
                                 qval,                        // qval
                                 ids.data(),                  // ids_base
                                 vals.data(),                 // vals_base
                                 nnz_span.data(),             // wnnz_buf
                                 nnz_span.size(),             // wnnz_buf_sz
                                 is_sparse,                   // is_sparse
                                 0,                           // cursor
                                 0,                           // offset
                                 wnnz_bits,                   // wnnz_bits
                                 wnnz_mask                    // wnnz_mask
            );
            if constexpr (is_bm25_u8) {
                if (!bm25_u8_overflow_offsets_span_.empty()) {
                    const uint32_t posting_begin = plists_dim_offsets_span_[qid];
                    const auto [overflow_begin, overflow_end] =
                        bm25_u8_overflow_range(posting_begin, plists_dim_offsets_span_[qid + 1]);
                    if (overflow_begin != overflow_end) {
                        overflow_cursors.emplace_back(cursors.size() - 1, posting_begin,
                                                      bm25_u8_overflow_offsets_span_.data(), overflow_begin,
                                                      overflow_end);
                    }
                }
            }
        }

        std::vector<float> distances(this->nr_rows_, 0.0f);

        // Cache SIMD kernel function pointer and process windows
        // Use compile-time dispatch to select appropriate kernel
        if constexpr (is_ip) {
            // IP scoring path
            const auto scatter_fn = sindi::get_ip_kernels().accumulate;
            const auto packed_fn = packed_ ? sindi::get_packed_ip_kernel() : nullptr;

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

                    if (packed_) {
                        packed_fn(cur.qval, packed_vals_span_.data(), packed_ids_span_.data(),
                                  size_t(plists_dim_offsets_span_[cur.dim_id]) + soff, wnnz, wscores_final);
                    } else {
                        scatter_fn(cur.qval, cur.vals_base + soff, cur.ids_base + soff, wnnz, wscores_final);
                    }
                }
            }
        } else {
            // BM25 scoring path
            const auto scatter_fn = []() {
                if constexpr (is_bm25_u8) {
                    return sindi::get_bm25_u8_kernels().accumulate;
                } else {
                    return sindi::get_bm25_kernels().accumulate;
                }
            }();

            // Extract BM25 parameters
            const auto packed_bm25_fn = packed_ ? sindi::get_packed_bm25_kernel(packed_u16_ids_) : nullptr;
            const float bm25_k1 = search_params.scorer_config.scorer_params.bm25.k1;
            const float bm25_b = search_params.scorer_config.scorer_params.bm25.b;
            const float bm25_avgdl = search_params.scorer_config.scorer_params.bm25.avgdl;
            const float* row_sums_ptr = row_sums_span_.data();

            for (size_t widx = 0; widx < nr_windows_; ++widx) {
                const size_t docid_start = window_size_ * widx;
                float* wscores_final = distances.data() + docid_start;
                size_t next_overflow_cursor = 0;

                for (size_t ci = 0; ci < cursors.size(); ++ci) {
                    auto& cur = cursors[ci];
                    if (cur.wnnz_buf == nullptr || cur.wnnz_buf_sz == 0) {
                        continue;
                    }

                    auto [soff, wnnz] = cur.advance_window(widx);
                    Bm25U8OverflowCursor* overflow_cur = nullptr;
                    if constexpr (is_bm25_u8) {
                        if (next_overflow_cursor < overflow_cursors.size() &&
                            overflow_cursors[next_overflow_cursor].query_cursor == ci) {
                            overflow_cur = &overflow_cursors[next_overflow_cursor++].cursor;
                            overflow_cur->advance(cur.offset);
                        }
                    }
                    if (wnnz == 0) {
                        continue;
                    }

                    const uint16_t* plist_ids = packed_ ? nullptr : cur.ids_base + soff;
                    const QuantType* plist_vals = packed_ ? nullptr : cur.vals_base + soff;

                    if (packed_)
                        packed_bm25_fn(cur.qval, packed_vals_span_.data(), packed_ids_span_.data(),
                                       size_t(plists_dim_offsets_span_[cur.dim_id]) + soff, wnnz, wscores_final,
                                       bm25_k1, bm25_b, bm25_avgdl, row_sums_ptr + docid_start, lut_.decode.data());
                    else
                        scatter_fn(cur.qval, plist_vals, plist_ids, wnnz, wscores_final, bm25_k1, bm25_b, bm25_avgdl,
                                   row_sums_ptr + docid_start);
                    if constexpr (is_bm25_u8) {
                        if (overflow_cur != nullptr && overflow_cur->window_begin != overflow_cur->window_end) {
                            (void)apply_bm25_u8_overflow_corrections(
                                overflow_cur->posting_begin, soff, plist_ids, wnnz, overflow_cur->window_begin,
                                overflow_cur->window_end, cur.qval, wscores_final, 0.0f, bm25_k1, bm25_b, bm25_avgdl,
                                row_sums_ptr + docid_start);
                        }
                    }
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
    void
    store_sealed_value(size_t offset, DataType value) noexcept {
        total_plists_vals_flat_[offset] = quantize_value(value);
    }

    void
    finalize_sealed_bm25_u8_overflows(const std::vector<std::vector<Bm25U8Overflow>>& overflows_by_worker) {
        if constexpr (is_bm25_u8) {
            size_t count = 0;
            for (const auto& worker_overflows : overflows_by_worker) {
                count += worker_overflows.size();
            }

            std::vector<Bm25U8Overflow> overflows;
            overflows.reserve(count);
            for (const auto& worker_overflows : overflows_by_worker) {
                overflows.insert(overflows.end(), worker_overflows.begin(), worker_overflows.end());
            }
            std::sort(overflows.begin(), overflows.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.posting_offset < rhs.posting_offset; });

            bm25_u8_overflow_offsets_.resize(count);
            bm25_u8_overflow_values_.resize(count);
            for (size_t i = 0; i < count; ++i) {
                bm25_u8_overflow_offsets_[i] = overflows[i].posting_offset;
                bm25_u8_overflow_values_[i] = overflows[i].value;
            }
            refresh_bm25_u8_overflow_spans();
            LOG_KNOWHERE_INFO_ << "SindiInvertedIndex BM25 U8 overflow sidecar built: count=" << count
                               << ", bytes=" << count * (sizeof(uint32_t) + sizeof(uint16_t));
        }
    }

    void
    refresh_bm25_u8_overflow_spans() noexcept {
        bm25_u8_overflow_offsets_span_ =
            std::span<const uint32_t>(bm25_u8_overflow_offsets_.data(), bm25_u8_overflow_offsets_.size());
        bm25_u8_overflow_values_span_ =
            std::span<const uint16_t>(bm25_u8_overflow_values_.data(), bm25_u8_overflow_values_.size());
    }

    [[nodiscard]] std::pair<uint32_t, uint32_t>
    bm25_u8_overflow_range(size_t posting_begin, size_t posting_end) const noexcept {
        if constexpr (is_bm25_u8) {
            const auto first = std::lower_bound(bm25_u8_overflow_offsets_span_.begin(),
                                                bm25_u8_overflow_offsets_span_.end(), posting_begin);
            const auto last = std::lower_bound(first, bm25_u8_overflow_offsets_span_.end(), posting_end);
            return {static_cast<uint32_t>(first - bm25_u8_overflow_offsets_span_.begin()),
                    static_cast<uint32_t>(last - bm25_u8_overflow_offsets_span_.begin())};
        } else {
            return {0, 0};
        }
    }

    [[nodiscard]] float
    bm25_posting_value(size_t posting_offset, QuantType value, uint32_t& overflow_cursor,
                       uint32_t overflow_end) const noexcept {
        if constexpr (is_bm25_u8) {
            if (overflow_cursor < overflow_end && bm25_u8_overflow_offsets_span_[overflow_cursor] == posting_offset) {
                return bm25_u8_overflow_values_span_[overflow_cursor++];
            }
        }
        return std::abs(static_cast<float>(value));
    }

    [[nodiscard]] float
    apply_bm25_u8_overflow_corrections(uint32_t posting_begin, uint32_t window_posting_offset,
                                       const uint16_t* posting_ids, uint16_t window_nnz, uint32_t overflow_begin,
                                       uint32_t overflow_end, float qval, float* scores, float current_max, float k1,
                                       float b, float avgdl, const float* row_sums) const noexcept {
        if constexpr (is_bm25_u8) {
            const float p1 = k1 + 1.0f;
            const float p2 = k1 * (1.0f - b);
            const float p3 = k1 * b / avgdl;
            const uint32_t window_begin = posting_begin + window_posting_offset;
            for (uint32_t i = overflow_begin; i < overflow_end; ++i) {
                const uint32_t local_offset = bm25_u8_overflow_offsets_span_[i] - window_begin;
                if (local_offset >= window_nnz) {
                    continue;
                }
                const uint16_t docid = posting_ids[local_offset];
                const float dl_term = p2 + p3 * row_sums[docid];
                const float true_tf = static_cast<float>(bm25_u8_overflow_values_span_[i]);
                const float correction = qval * p1 * (true_tf / (true_tf + dl_term) - 255.0f / (255.0f + dl_term));
                const float corrected_score = (scores[docid] += correction);
                current_max = std::max(current_max, corrected_score);
            }
        }
        return current_max;
    }

    [[nodiscard]] static uint16_t
    bm25_u16_value(DataType value) noexcept {
        // Match the existing BM25 u16 representation: truncate fractional TF
        // values and saturate values outside the representable range.
        const float bounded =
            std::clamp(static_cast<float>(value), 0.0f, static_cast<float>(std::numeric_limits<uint16_t>::max()));
        return static_cast<uint16_t>(bounded);
    }

    [[nodiscard]] QuantType
    quantize_value(DataType value) const {
        if constexpr (is_bm25_u8) {
            return static_cast<uint8_t>(std::min<uint16_t>(bm25_u16_value(value), 255));
        } else if constexpr (is_bm25) {
            return static_cast<QuantType>(bm25_u16_value(value));
        } else {
            const auto half = static_cast<QuantType>(static_cast<float>(value));
            return packed_ ? sindi::decode_e5m7_half(sindi::encode_e5m7(half)) : half;
        }
    }

    [[nodiscard]] std::span<const uint16_t>
    posting_ids(size_t dim_id) const {
        if (!total_plists_ids_flat_span_.empty()) {
            const size_t begin = plists_dim_offsets_span_[dim_id];
            const size_t count = plists_dim_offsets_span_[dim_id + 1] - begin;
            return total_plists_ids_flat_span_.subspan(begin, count);
        }
        return total_plists_ids_spans_[dim_id];
    }

    [[nodiscard]] std::span<const QuantType>
    posting_vals(size_t dim_id) const {
        if (!total_plists_vals_flat_span_.empty()) {
            const size_t begin = plists_dim_offsets_span_[dim_id];
            const size_t count = plists_dim_offsets_span_[dim_id + 1] - begin;
            return total_plists_vals_flat_span_.subspan(begin, count);
        }
        return total_plists_vals_spans_[dim_id];
    }

    [[nodiscard]] std::span<const uint8_t>
    encoded_window_nnzs(size_t dim_id) const {
        if (!plists_window_nnzs_offsets_.empty()) {
            const size_t begin = plists_window_nnzs_offsets_[dim_id];
            const size_t count = plists_window_nnzs_offsets_[dim_id + 1] - begin;
            return std::span<const uint8_t>(plists_window_nnzs_flat_.data(), plists_window_nnzs_flat_.size())
                .subspan(begin, count);
        }
        return plists_window_nnzs_spans_[dim_id];
    }

    // Global posting lists (per dimension, concatenated across windows)
    using aligned_u16_vec = std::vector<uint16_t, aligned_allocator<uint16_t, 64>>;
    using aligned_quant_vec = std::vector<QuantType, aligned_allocator<QuantType, 64>>;
    std::vector<aligned_u16_vec> total_plists_ids_;
    std::vector<aligned_quant_vec> total_plists_vals_;
    std::vector<std::span<const uint16_t>> total_plists_ids_spans_;
    std::vector<std::span<const QuantType>> total_plists_vals_spans_;
    // flatten posting for sealed index
    aligned_u16_vec total_plists_ids_flat_;
    aligned_quant_vec total_plists_vals_flat_;
    std::span<const uint16_t> total_plists_ids_flat_span_;
    std::span<const QuantType> total_plists_vals_flat_span_;
    // Window sizes encoding (per-dim window nnz) and bitset of formats
    // Bit=1 means sparse format (wid, wnnz pairs), Bit=0 means dense format (one entry per window)
    std::vector<uint8_t> plists_wnnzs_fmts_msk_;
    std::span<const uint8_t> plists_wnnzs_fmts_msk_span_;
    std::vector<std::vector<uint16_t>> window_index_plists_sz_;
    std::vector<std::span<const uint16_t>> window_index_plists_sz_spans_;
    std::vector<std::vector<uint8_t>> plists_window_nnzs_;
    std::vector<std::span<const uint8_t>> plists_window_nnzs_spans_;
    std::vector<uint8_t> plists_window_nnzs_flat_;
    std::vector<size_t> plists_window_nnzs_offsets_;
    std::vector<uint32_t> plists_dim_offsets_;
    std::span<const uint32_t> plists_dim_offsets_span_;
    std::vector<float> max_scores_per_dim_;
    std::span<const float> max_scores_per_dim_span_;

    // U8 keeps the hot posting array compact and stores exact TF only for values above 255.
    // Offsets are global positions in the dimension-concatenated posting array.
    std::vector<uint32_t> bm25_u8_overflow_offsets_;
    std::vector<uint16_t> bm25_u8_overflow_values_;
    std::span<const uint32_t> bm25_u8_overflow_offsets_span_;
    std::span<const uint16_t> bm25_u8_overflow_values_span_;

    // Row sums only needed for BM25
    std::vector<float> row_sums_;
    std::span<const float> row_sums_span_;

    bool
    validate_packed_file(const MemoryIOReader& reader) {
        try {
            const auto* data = reader.data();
            const size_t size = reader.total_;
            auto u32 = [&](size_t offset) {
                if (offset > size || size - offset < 4)
                    throw std::runtime_error("Truncated packed header");
                uint32_t value;
                std::memcpy(&value, data + offset, 4);
                return value;
            };

            if (size < 36 || u32(0) != kInvertedIndexFileFormatVersion ||
                u32(16) != static_cast<uint32_t>(packed_quant_type())) {
                return false;
            }

            const size_t dims = u32(12), sections = u32(32);
            if (sections < 3 || sections > (size - 36) / sizeof(InvertedIndexSectionHeader)) {
                return false;
            }

            const size_t directory_end = 36 + sections * sizeof(InvertedIndexSectionHeader);
            std::vector<InvertedIndexSectionHeader> headers(sections);
            std::memcpy(headers.data(), data + 36, sections * sizeof(InvertedIndexSectionHeader));
            std::unordered_set<uint32_t> types;
            for (const auto& h : headers) {
                switch (h.type) {
                    case InvertedIndexSectionType::POSTING_LISTS:
                    case InvertedIndexSectionType::DIM_MAP_REVERSE:
                    case InvertedIndexSectionType::DIM_MAP_MPHF:
                    case InvertedIndexSectionType::MAX_SCORES_PER_DIM:
                    case InvertedIndexSectionType::SINDI_REFINEMENT:
                        break;
                    case InvertedIndexSectionType::ROW_SUMS:
                        if constexpr (is_bm25)
                            break;
                        return false;
                    default:
                        return false;
                }

                if (!types.insert(static_cast<uint32_t>(h.type)).second || h.offset < directory_end ||
                    h.offset > size || h.size > size - h.offset) {
                    return false;
                }
            }

            auto by_offset = headers;
            std::sort(by_offset.begin(), by_offset.end(),
                      [](const auto& a, const auto& b) { return a.offset < b.offset; });
            for (size_t i = 1; i < by_offset.size(); ++i) {
                if (by_offset[i - 1].offset + by_offset[i - 1].size > by_offset[i].offset) {
                    return false;
                }
            }

            auto* h = find_section_header(headers, InvertedIndexSectionType::POSTING_LISTS);
            auto* reverse = find_section_header(headers, InvertedIndexSectionType::DIM_MAP_REVERSE);
            if (!reverse || ((reinterpret_cast<uintptr_t>(data) + reverse->offset) % alignof(uint32_t))) {
                return false;
            }

            auto* maxima = find_section_header(headers, InvertedIndexSectionType::MAX_SCORES_PER_DIM);
            if (!h || !maxima || !find_section_header(headers, InvertedIndexSectionType::DIM_MAP_REVERSE) ||
                maxima->size != dims * 4 || h->size < posting_header_bytes() ||
                u32(h->offset) != static_cast<uint32_t>(packed_encoding()) ||
                (u32(h->offset + 4) < min_window_size ||
                 u32(h->offset + 4) > (packed_u16_ids_ ? max_window_size : 4096) ||
                 (is_ip && u32(h->offset + 4) != 4096)) ||
                u32(h->offset + 12) != 1 || u32(h->offset + 16) != packed_id_layout() ||
                u32(h->offset + 20) != static_cast<uint32_t>(packed_quant_type())) {
                return false;
            }

            if (u32(h->offset + 8) != (uint64_t(u32(4)) + u32(h->offset + 4) - 1) / u32(h->offset + 4)) {
                return false;
            }

            if constexpr (is_bm25) {
                if (types.count(static_cast<uint32_t>(InvertedIndexSectionType::SINDI_REFINEMENT)))
                    return false;
                const auto* lengths = find_section_header(headers, InvertedIndexSectionType::ROW_SUMS);
                if ((u32(4) && !lengths) || (lengths && (lengths->size != uint64_t(u32(4)) * 4 || lengths->offset % 4)))
                    return false;
                sindi::Bm25U4Lut lut;
                std::memcpy(lut.decode.data(), data + h->offset + 24, 16);
                std::memcpy(lut.ends.data(), data + h->offset + 40, 16);
                float params[3];
                std::memcpy(params, data + h->offset + 56, 12);
                if (!std::isfinite(params[1]) || params[1] < 0 || params[1] > 1 || !std::isfinite(params[2]) ||
                    params[2] < 1)
                    return false;
                lut.k1 = params[0];
                lut.validate_and_encode();
            }
            size_t pos = h->offset + posting_header_bytes(), end = h->offset + h->size;
            auto advance = [&](size_t bytes) {
                if (pos > end || bytes > end - pos) {
                    throw std::runtime_error("Truncated packed postings");
                }
                pos += bytes;
            };

            advance((dims + 7) / 8);
            if (dims > (end - pos) / 4) {
                return false;
            }

            for (size_t i = 0; i < dims; ++i) {
                advance(4);
                const auto bytes = u32(pos - 4);
                advance(bytes);
            }

            const size_t offsets = pos;
            advance((dims + 1) * 4);
            if (u32(offsets) != 0) {
                return false;
            }

            for (size_t i = 0; i < dims; ++i) {
                if (u32(offsets + 4 * i) > u32(offsets + 4 * (i + 1))) {
                    return false;
                }
            }

            const auto count = u32(offsets + dims * 4);
            const auto bytes = packed_id_bytes(count);
            advance(bytes);
            const auto value_bytes = packed_value_bytes(count);
            advance(value_bytes);

            if (pos != end) {
                return false;
            }
            if ((count & 1) && ((data[end - 1] & 0xf0) || (!packed_u16_ids_ && (data[end - value_bytes - 1] & 0xf0)))) {
                return false;
            }

            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    bool packed_ = false;
    bool packed_u16_ids_ = false;
    bool packed_ready_ = false;
    std::vector<uint8_t> packed_ids_, packed_vals_;
    sindi::Bm25U4Lut lut_;

    constexpr InvertedIndexQuantType
    packed_quant_type() const {
        return is_ip ? InvertedIndexQuantType::IP_E5M7
                     : (packed_u16_ids_ ? InvertedIndexQuantType::BM25_U4_LUT_U16
                                        : InvertedIndexQuantType::BM25_U4_LUT_U12);
    }
    constexpr uint32_t
    packed_id_layout() const {
        return packed_u16_ids_ ? 1u : 2u;
    }
    constexpr InvertedIndexEncoding
    packed_encoding() const {
        return is_ip ? InvertedIndexEncoding::FIXED_DOCID_WINDOWS_U12_E5M7
                     : (packed_u16_ids_ ? InvertedIndexEncoding::FIXED_DOCID_WINDOWS_U16_U4_LUT
                                        : InvertedIndexEncoding::FIXED_DOCID_WINDOWS_U12_U4_LUT);
    }
    size_t
    packed_id_bytes(size_t n) const {
        return packed_u16_ids_ ? n * sizeof(uint16_t) : sindi::packed12_bytes(n);
    }
    size_t
    posting_header_bytes() const {
        return (packed_ ? 24 : 12) + (packed_ && is_bm25 ? 44 : 0);
    }
    static size_t
    packed_value_bytes(size_t n) {
        return is_ip ? sindi::packed12_bytes(n) : n / 2 + n % 2;
    }
    size_t
    packed_payload_bytes(size_t n) const {
        return packed_id_bytes(n) + packed_value_bytes(n);
    }
    std::span<const uint8_t> packed_ids_span_, packed_vals_span_;

    size_t
    posting_count(size_t dim) const {
        return plists_dim_offsets_span_[dim + 1] - plists_dim_offsets_span_[dim];
    }

    uint16_t
    posting_id_at(size_t dim, size_t pos) const {
        if (!packed_ready_)
            return posting_ids(dim)[pos];
        const auto posting = size_t(plists_dim_offsets_span_[dim]) + pos;
        if (packed_u16_ids_)
            return sindi::unpack_bm25_u16_id(packed_ids_span_.data(), posting);
        return sindi::unpack12(packed_ids_span_.data(), posting);
    }

    float
    posting_value_at(size_t dim, size_t pos) const {
        if (!packed_ready_)
            return static_cast<float>(posting_vals(dim)[pos]);
        const size_t posting = plists_dim_offsets_span_[dim] + pos;
        if constexpr (is_bm25)
            return lut_.decode[sindi::unpack_bm25_u4(packed_vals_span_.data(), posting)];
        else
            return sindi::decode_e5m7(sindi::unpack12(packed_vals_span_.data(), posting));
    }

    float
    packed_dimension_maximum(size_t dim, const sindi::RefinementSeek& seek) const {
        float maximum = 0;
        for (size_t w = 0; w + 1 < seek.offsets.size(); ++w) {
            const uint32_t wid = seek.sparse ? seek.windows[w] : w;
            for (size_t j = seek.offsets[w]; j < seek.offsets[w + 1]; ++j) {
                float score = posting_value_at(dim, j);
                if constexpr (is_bm25) {
                    const auto& p = this->build_scorer_->config().scorer_params.bm25;
                    const float dl = row_sums_span_[size_t(wid) * window_size_ + posting_id_at(dim, j)];
                    score = (p.k1 + 1) * score / (score + p.k1 * (1 - p.b) + p.k1 * p.b / p.avgdl * dl);
                }
                maximum = std::max(maximum, score);
            }
        }
        return maximum;
    }

    void
    pack_postings() {
        if constexpr (is_bm25_u8) {
            std::array<uint64_t, 256> histogram{};
            for (const auto tf : total_plists_vals_flat_span_) ++histogram[tf];
            lut_ = sindi::fit_bm25_u4_lut(histogram, this->build_scorer_->config().scorer_params.bm25.k1);
            const size_t count = plists_dim_offsets_span_.back();
            packed_ids_.assign(packed_id_bytes(count), 0);
            packed_vals_.assign(packed_value_bytes(count), 0);
            for (size_t i = 0; i < count; ++i) {
                if (packed_u16_ids_)
                    std::memcpy(packed_ids_.data() + i * sizeof(uint16_t), &total_plists_ids_flat_span_[i],
                                sizeof(uint16_t));
                else
                    sindi::pack12(packed_ids_.data(), i, total_plists_ids_flat_span_[i]);
                packed_vals_[i / 2] |= lut_.encode[total_plists_vals_flat_span_[i]] << (4 * (i & 1));
            }
            packed_ids_span_ = packed_ids_;
            packed_vals_span_ = packed_vals_;
            packed_ready_ = true;
            rebuild_refinement_seek();  // temporary window validation; never retained by unrefined storage
            max_scores_per_dim_.assign(this->nr_inner_dims_, 0);
            for (size_t d = 0; d < this->nr_inner_dims_; ++d)
                max_scores_per_dim_[d] = packed_dimension_maximum(d, refinement_seek_[d]);
            max_scores_per_dim_span_ = max_scores_per_dim_;
            std::vector<sindi::RefinementSeek>{}.swap(refinement_seek_);
            aligned_u16_vec{}.swap(total_plists_ids_flat_);
            aligned_quant_vec{}.swap(total_plists_vals_flat_);
            std::vector<aligned_u16_vec>{}.swap(total_plists_ids_);
            std::vector<aligned_quant_vec>{}.swap(total_plists_vals_);
            std::vector<std::span<const uint16_t>>{}.swap(total_plists_ids_spans_);
            std::vector<std::span<const QuantType>>{}.swap(total_plists_vals_spans_);
            total_plists_ids_flat_span_ = {};
            total_plists_vals_flat_span_ = {};
            std::vector<uint32_t>{}.swap(bm25_u8_overflow_offsets_);
            std::vector<uint16_t>{}.swap(bm25_u8_overflow_values_);
            bm25_u8_overflow_offsets_span_ = {};
            bm25_u8_overflow_values_span_ = {};
        }
        if constexpr (is_ip) {
            const size_t count = plists_dim_offsets_span_.back();
            std::vector<uint8_t> ids(sindi::packed12_bytes(count), 0), vals(ids.size(), 0);
            max_scores_per_dim_.assign(this->nr_inner_dims_, 0);
            // Sequential pair ownership avoids shared-nibble races at term boundaries.
            for (size_t dim = 0; dim < this->nr_inner_dims_; ++dim) {
                const auto source_ids = posting_ids(dim);
                const auto source_vals = posting_vals(dim);
                const size_t start = plists_dim_offsets_span_[dim];
                for (size_t j = 0; j < source_ids.size(); ++j) {
                    const auto code = sindi::encode_e5m7(source_vals[j]);
                    sindi::pack12(ids.data(), start + j, source_ids[j]);
                    sindi::pack12(vals.data(), start + j, code);
                    max_scores_per_dim_[dim] = std::max(max_scores_per_dim_[dim], sindi::decode_e5m7(code));
                }
            }

            packed_ids_.swap(ids);
            packed_vals_.swap(vals);

            packed_ids_span_ = packed_ids_;
            packed_vals_span_ = packed_vals_;
            max_scores_per_dim_span_ = max_scores_per_dim_;
            aligned_u16_vec{}.swap(total_plists_ids_flat_);
            aligned_quant_vec{}.swap(total_plists_vals_flat_);
            std::vector<aligned_u16_vec>{}.swap(total_plists_ids_);
            std::vector<aligned_quant_vec>{}.swap(total_plists_vals_);
            std::vector<std::span<const uint16_t>>{}.swap(total_plists_ids_spans_);
            std::vector<std::span<const QuantType>>{}.swap(total_plists_vals_spans_);
            total_plists_ids_flat_span_ = {};
            total_plists_vals_flat_span_ = {};
            packed_ready_ = true;
        }
    }

    void
    expand_for_add() {
        if constexpr (is_ip && AllowIncremental) {
            total_plists_ids_.resize(this->nr_inner_dims_);
            total_plists_vals_.resize(this->nr_inner_dims_);
            for (size_t dim = 0; dim < this->nr_inner_dims_; ++dim) {
                const size_t count = posting_count(dim);
                auto& ids = total_plists_ids_[dim];
                auto& vals = total_plists_vals_[dim];
                ids.resize(count);
                vals.resize(count);
                for (size_t j = 0; j < count; ++j) {
                    ids[j] = posting_id_at(dim, j);
                    vals[j] = static_cast<QuantType>(posting_value_at(dim, j));
                }
            }

            packed_ready_ = false;
            // append_window_indexes publishes fresh typed spans before packing again.
        }
    }

    bool legacy_dim_map_mphf_trailer_workaround_{true};
    bool refine_ = false;
    std::vector<sindi::RefinementSeek> refinement_seek_;

    void
    rebuild_refinement_seek() {
        std::vector<sindi::RefinementSeek> seeks(this->nr_inner_dims_);
        const uint32_t bits = 32 - __builtin_clz(window_size_);
        const uint32_t mask = (1u << bits) - 1;

        for (size_t dim = 0; dim < seeks.size(); ++dim) {
            auto& seek = seeks[dim];
            const auto counts = encoded_window_nnzs(dim);
            const auto count_ids = posting_count(dim);

            seek.sparse = (plists_wnnzs_fmts_msk_span_[dim >> 3] & (1u << (dim & 7))) != 0;
            const size_t entries = seek.sparse ? counts.size() / 4 : nr_windows_;
            if (counts.size() != entries * (seek.sparse ? 4 : 2)) {
                throw std::runtime_error("Invalid refinement window counts");
            }

            seek.offsets.reserve(entries + 1);
            if (seek.sparse) {
                seek.windows.reserve(entries);
            }

            size_t sum = 0;
            for (size_t i = 0; i < entries; ++i) {
                uint32_t wid = i, count = 0;
                if (seek.sparse) {
                    uint32_t packed;
                    std::memcpy(&packed, counts.data() + i * 4, 4);
                    wid = packed >> bits;
                    count = packed & mask;
                    if (wid >= nr_windows_ || (i && seek.windows.back() >= wid)) {
                        throw std::runtime_error("Invalid refinement sparse windows");
                    }

                    seek.windows.push_back(wid);
                } else {
                    uint16_t value;
                    std::memcpy(&value, counts.data() + i * 2, 2);
                    count = value;
                }

                if (count > window_size_ || sum + count > count_ids) {
                    throw std::runtime_error("Invalid refinement posting count");
                }

                seek.offsets.push_back(static_cast<uint32_t>(sum));
                // Builders emit document order; validate it also on deserialization.
                for (size_t j = sum; j < sum + count; ++j) {
                    const float represented = posting_value_at(dim, j);
                    if (!std::isfinite(represented) || represented < 0 ||
                        uint64_t(wid) * window_size_ + posting_id_at(dim, j) >= this->nr_rows_ ||
                        posting_id_at(dim, j) >= window_size_ ||
                        (j > sum && posting_id_at(dim, j - 1) >= posting_id_at(dim, j))) {
                        // no good
                        throw std::runtime_error("Invalid refinement posting IDs");
                    }
                }

                sum += count;
            }

            if (sum != count_ids || sum > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("Invalid refinement posting offsets");
            }

            seek.offsets.push_back(static_cast<uint32_t>(sum));
        }

        refinement_seek_.swap(seeks);
    }

    std::pair<uint32_t, uint32_t>
    window_posting_range(uint32_t dim, uint32_t window) const {
        return refinement_seek_.at(dim).range(window);
    }

    // Physical U16/FP16 lookup backend. Packed ID/value formats can replace this
    // operation without changing mass selection, pool sizing, or final selection.
    void
    score_candidate_ids(uint32_t dim, uint32_t begin, uint32_t end, float weight, std::span<const uint32_t> candidates,
                        std::span<float> scores) const {
        if (packed_) {
            for (size_t i = 0; i < candidates.size(); ++i) {
                const uint32_t local = candidates[i] % window_size_;
                uint32_t low = begin, high = end;
                while (low < high) {
                    const auto mid = low + (high - low) / 2;
                    if (posting_id_at(dim, mid) < local) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }

                begin = low;
                if (low < end && posting_id_at(dim, low) == local) {
                    scores[i] = std::fma(weight, posting_value_at(dim, low), scores[i]);
                }
            }

            return;
        }

        const auto ids = posting_ids(dim);
        const auto vals = posting_vals(dim);
        if (begin == end)
            return;
        auto cursor = ids.begin() + begin;
        const auto stop = ids.begin() + end;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const uint32_t local = candidates[i] % window_size_;
            cursor = std::lower_bound(cursor, stop, local);
            if (cursor != stop && *cursor == local)
                scores[i] = std::fma(weight, static_cast<float>(vals[cursor - ids.begin()]), scores[i]);
        }
    }

    void
    search_refined(const SparseRow<DataType>& query, size_t k, float* distances, label_t* labels,
                   const BitsetView& bitset, const InvertedIndexSearchParams& params) const {
        const auto count = sindi::refinement_pool_size(k, params.refine_k, this->nr_rows_);
        if (count == 0) {
            return;
        }

        const auto selected = sindi::retain_query_mass(query, params.sindi_query_mass);
        auto coarse_query = parse_query_with_dim_map(selected, this->dim_map_, 0.0f);

        std::vector<float> coarse_scores(count, std::numeric_limits<float>::quiet_NaN());
        std::vector<label_t> coarse_ids(count, -1);
        search_coarse(std::move(coarse_query), count, coarse_scores.data(), coarse_ids.data(), bitset, params);

        std::vector<uint32_t> candidates;
        candidates.reserve(count);
        for (auto id : coarse_ids) {
            if (id >= 0 && static_cast<size_t>(id) < this->nr_rows_ && (bitset.empty() || !bitset.test(id))) {
                candidates.push_back(static_cast<uint32_t>(id));
            }
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        std::vector<float> scores(candidates.size(), 0.0f);
        auto full = parse_query_with_dim_map(query, this->dim_map_, 0.0f);
        std::sort(full.begin(), full.end(), [&](const auto& a, const auto& b) {
            const float x = a.second * max_scores_per_dim_span_[a.first];
            const float y = b.second * max_scores_per_dim_span_[b.first];
            return x != y ? x > y : a.first < b.first;
        });

        for (const auto& [dim, weight] : full) {
            for (size_t start = 0; start < candidates.size();) {
                const auto window = candidates[start] / window_size_;
                size_t stop = start + 1;
                while (stop < candidates.size() && candidates[stop] / window_size_ == window) {
                    ++stop;
                }

                const auto [begin, end] = window_posting_range(dim, window);
                score_candidate_ids(dim, begin, end, weight,
                                    std::span<const uint32_t>(candidates).subspan(start, stop - start),
                                    std::span<float>(scores).subspan(start, stop - start));
                start = stop;
            }
        }

        std::vector<size_t> order(candidates.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return scores[a] != scores[b] ? (scores[a] > scores[b]) : (candidates[a] < candidates[b]);
        });

        for (size_t i = 0; i < std::min(k, order.size()); ++i) {
            labels[i] = candidates[order[i]];
            distances[i] = scores[order[i]];
        }
    }

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

    struct Bm25U8OverflowCursor {
        uint32_t posting_begin{0};
        const uint32_t* offsets{nullptr};
        uint32_t cursor{0};
        uint32_t end{0};
        uint32_t window_begin{0};
        uint32_t window_end{0};

        Bm25U8OverflowCursor(uint32_t posting_begin, const uint32_t* offsets, uint32_t cursor, uint32_t end)
            : posting_begin(posting_begin), offsets(offsets), cursor(cursor), end(end) {
        }

        inline void
        advance(uint32_t posting_offset) {
            window_begin = cursor;
            if (offsets != nullptr) {
                const uint32_t global_end = posting_begin + posting_offset;
                while (cursor < end && offsets[cursor] < global_end) {
                    ++cursor;
                }
            }
            window_end = cursor;
        }
    };

    struct Bm25U8QueryOverflowCursor {
        size_t query_cursor{0};
        Bm25U8OverflowCursor cursor;

        Bm25U8QueryOverflowCursor(size_t query_cursor, uint32_t posting_begin, const uint32_t* offsets,
                                  uint32_t cursor_begin, uint32_t cursor_end)
            : query_cursor(query_cursor), cursor(posting_begin, offsets, cursor_begin, cursor_end) {
        }
    };
};

using SindiInvertedIndexIP = SindiInvertedIndex<float, knowhere::fp16>;
using SindiInvertedIndexBM25 = SindiInvertedIndex<float, uint16_t>;
using SindiInvertedIndexBM25U8 = SindiInvertedIndex<float, uint8_t>;
using GrowableSindiInvertedIndexIP = SindiInvertedIndex<float, knowhere::fp16, true>;
using GrowableSindiInvertedIndexBM25 = SindiInvertedIndex<float, uint16_t, true>;

}  // namespace knowhere::sparse::inverted
