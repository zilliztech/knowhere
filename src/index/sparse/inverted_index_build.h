#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include "folly/container/F14Map.h"
#include "index/sparse/parallel_build.h"
#include "knowhere/sparse_utils.h"

namespace knowhere::sparse::inverted {

using DimensionSet = std::unordered_set<uint32_t>;
using WorkerPostingCounts = std::vector<folly::F14FastMap<uint32_t, uint32_t>>;

// Dense [worker][dimension] write cursors. uint32_t keeps this often-large matrix compact; global offsets remain
// size_t so the complete posting array may exceed UINT32_MAX.
using WorkerPostingCursors = std::vector<std::vector<uint32_t>>;

enum class WorkerCursorMode : uint8_t {
    Absolute,
    Relative,
};

struct RowScanResult {
    DimensionSet external_dims;
    WorkerPostingCounts posting_counts_by_worker;
};

struct PostingBuildPlan {
    WorkerPostingCursors cursors_by_worker;
    std::vector<size_t> posting_offsets;
    WorkerCursorMode cursor_mode{WorkerCursorMode::Absolute};

    [[nodiscard]] size_t
    total_postings() const {
        return posting_offsets.empty() ? 0 : posting_offsets.back();
    }

    [[nodiscard]] size_t
    posting_count(size_t inner_dim) const {
        return posting_offsets[inner_dim + 1] - posting_offsets[inner_dim];
    }
};

template <typename DType>
RowScanResult
scan_rows_for_build(const SparseRow<DType>* data, size_t rows, std::vector<uint32_t>* dataset_nnz_stats = nullptr,
                    std::vector<float>* row_sums = nullptr) {
    // Reuse the same contiguous worker partitions during counting and filling. This gives each worker a disjoint
    // posting-list range without atomics and preserves docid order.
    const auto concurrency = GetParallelBuildConcurrency(rows);
    WorkerPostingCounts posting_counts_by_worker(concurrency);

    parallel_for_workers(concurrency, [&](size_t worker_id) {
        const auto [begin, end] = get_worker_row_range(rows, worker_id, concurrency);
        auto& posting_counts = posting_counts_by_worker[worker_id];
        posting_counts.reserve(std::min<size_t>(end - begin, 65536));
        for (size_t i = begin; i < end; ++i) {
            float row_sum = 0.0f;
            for (size_t j = 0; j < data[i].size(); ++j) {
                const auto [dim, val] = data[i][j];
                if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
                    continue;
                }
                row_sum += val;
                auto [it, inserted] = posting_counts.try_emplace(dim, 1);
                if (!inserted) {
                    ++it->second;
                }
            }
            if (dataset_nnz_stats != nullptr) {
                (*dataset_nnz_stats)[i] = data[i].size();
            }
            if (row_sums != nullptr) {
                (*row_sums)[i] = row_sum;
            }
        }
    });

    RowScanResult result;
    result.posting_counts_by_worker = std::move(posting_counts_by_worker);
    size_t external_dim_capacity = 0;
    for (const auto& posting_counts : result.posting_counts_by_worker) {
        external_dim_capacity += posting_counts.size();
    }
    result.external_dims.reserve(external_dim_capacity);
    for (const auto& posting_counts : result.posting_counts_by_worker) {
        for (const auto& posting_count : posting_counts) {
            result.external_dims.insert(posting_count.first);
        }
    }
    return result;
}

template <typename DimMap>
PostingBuildPlan
prepare_posting_build_plan(WorkerPostingCounts posting_counts_by_worker, const DimMap& dim_map, size_t nr_inner_dims) {
    PostingBuildPlan plan;
    plan.cursors_by_worker.assign(posting_counts_by_worker.size(), std::vector<uint32_t>(nr_inner_dims, 0));
    plan.posting_offsets.assign(nr_inner_dims + 1, 0);

    parallel_for(posting_counts_by_worker.size(), [&](size_t worker_id) {
        auto& dense_counts = plan.cursors_by_worker[worker_id];
        for (const auto& [external_dim, count] : posting_counts_by_worker[worker_id]) {
            const auto inner_dim = dim_map.lookup_trusted(external_dim);
            dense_counts[inner_dim] = count;
        }
    });

    parallel_for(nr_inner_dims, [&](size_t inner_dim) {
        size_t posting_count = 0;
        for (const auto& dense_counts : plan.cursors_by_worker) {
            posting_count += dense_counts[inner_dim];
        }
        plan.posting_offsets[inner_dim + 1] = posting_count;
    });

    for (size_t i = 1; i < plan.posting_offsets.size(); ++i) {
        plan.posting_offsets[i] += plan.posting_offsets[i - 1];
    }

    // Prefer absolute cursors in the common case to avoid one addition per posting. Large indexes retain compact
    // uint32_t list-relative cursors and combine them with size_t list offsets in the fill loop.
    plan.cursor_mode = plan.total_postings() <= std::numeric_limits<uint32_t>::max() ? WorkerCursorMode::Absolute
                                                                                     : WorkerCursorMode::Relative;
    for (size_t inner_dim = 0; inner_dim < nr_inner_dims; ++inner_dim) {
        size_t next_offset = plan.cursor_mode == WorkerCursorMode::Absolute ? plan.posting_offsets[inner_dim] : 0;
        for (auto& worker_cursors : plan.cursors_by_worker) {
            const auto posting_count = worker_cursors[inner_dim];
            worker_cursors[inner_dim] = static_cast<uint32_t>(next_offset);
            next_offset += posting_count;
        }
    }
    return plan;
}

namespace detail {

template <WorkerCursorMode Mode, typename DType, typename QType, typename DimMap, typename Quantizer>
void
fill_postings_by_worker_impl(const SparseRow<DType>* data, size_t rows, const DimMap& dim_map,
                             std::span<uint32_t> posting_ids, std::span<QType> posting_vals, PostingBuildPlan& plan,
                             Quantizer&& quantizer) {
    const auto concurrency = plan.cursors_by_worker.size();

    parallel_for_workers(concurrency, [&](size_t worker_id) {
        const auto [begin, end] = get_worker_row_range(rows, worker_id, concurrency);
        auto& worker_cursors = plan.cursors_by_worker[worker_id];
        for (size_t i = begin; i < end; ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                const auto [dim, val] = data[i][j];
                if (std::abs(val) < std::numeric_limits<DType>::epsilon()) {
                    continue;
                }
                const auto inner_dim = dim_map.lookup_trusted(dim);
                const auto worker_offset = worker_cursors[inner_dim]++;
                size_t offset;
                if constexpr (Mode == WorkerCursorMode::Absolute) {
                    offset = worker_offset;
                } else {
                    offset = plan.posting_offsets[inner_dim] + worker_offset;
                }
                posting_ids[offset] = static_cast<uint32_t>(i);
                posting_vals[offset] = quantizer(val);
            }
        }
    });
}

}  // namespace detail

template <typename DType, typename QType, typename DimMap, typename Quantizer>
void
fill_postings_by_worker(const SparseRow<DType>* data, size_t rows, const DimMap& dim_map,
                        std::span<uint32_t> posting_ids, std::span<QType> posting_vals, PostingBuildPlan& plan,
                        Quantizer&& quantizer) {
    // Dispatch once so the hot loop contains no per-posting mode branch.
    if (plan.cursor_mode == WorkerCursorMode::Absolute) {
        detail::fill_postings_by_worker_impl<WorkerCursorMode::Absolute>(data, rows, dim_map, posting_ids, posting_vals,
                                                                         plan, std::forward<Quantizer>(quantizer));
    } else {
        detail::fill_postings_by_worker_impl<WorkerCursorMode::Relative>(data, rows, dim_map, posting_ids, posting_vals,
                                                                         plan, std::forward<Quantizer>(quantizer));
    }
}

}  // namespace knowhere::sparse::inverted
