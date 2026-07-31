#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

#include "folly/Executor.h"
#include "knowhere/comp/task.h"
#include "knowhere/thread_pool.h"

namespace knowhere::sparse::inverted {

namespace detail {

[[nodiscard]] inline bool
running_on_pool(const std::shared_ptr<ThreadPool>& pool) {
    if (pool == nullptr) {
        return false;
    }
    const auto blocking_context = folly::getExecutorBlockingContext();
    return blocking_context.has_value() && blocking_context->ex == &pool->GetPool();
}

}  // namespace detail

[[nodiscard]] inline size_t
GetParallelBuildConcurrency(size_t work_items) noexcept {
    if (work_items == 0) {
        return 1;
    }
    const auto pool = ThreadPool::GetGlobalBuildThreadPool();
    if (pool == nullptr || detail::running_on_pool(pool)) {
        return 1;
    }
    return std::max<size_t>(1, std::min(work_items / 100, GetBuildThreadPoolSize()));
}

[[nodiscard]] inline size_t
GetParallelBuildBlocks(const size_t work_items) {
    const size_t build_concurrency = GetParallelBuildConcurrency(work_items);

    constexpr size_t n_min_per_thread = 100;
    size_t n_per_thread = (work_items + build_concurrency - 1) / build_concurrency;
    if (n_per_thread < n_min_per_thread) {
        n_per_thread = n_min_per_thread;
    }

    constexpr size_t n_max_per_block = 8192;
    const size_t block_size = (n_per_thread < n_max_per_block) ? n_per_thread : n_max_per_block;
    const size_t num_blocks = (work_items + block_size - 1) / block_size;
    return num_blocks;
}

inline std::pair<size_t, size_t>
get_worker_row_range(size_t rows, size_t worker_id, size_t concurrency) {
    const auto rows_per_worker = rows / concurrency;
    const auto workers_with_extra_row = rows % concurrency;
    const auto begin = rows_per_worker * worker_id + std::min(worker_id, workers_with_extra_row);
    const auto end = begin + rows_per_worker + (worker_id < workers_with_extra_row ? 1 : 0);
    return {begin, end};
}

namespace detail {

template <typename Func>
void
parallel_for_blocks(size_t work_items, size_t num_blocks, Func&& func) {
    if (work_items == 0) {
        return;
    }

    num_blocks = std::min(work_items, std::max<size_t>(1, num_blocks));
    const auto pool = ThreadPool::GetGlobalBuildThreadPool();
    if (num_blocks == 1 || pool == nullptr || pool->size() <= 1 || running_on_pool(pool)) {
        for (size_t i = 0; i < work_items; ++i) {
            func(i);
        }
        return;
    }

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(num_blocks);
    for (size_t block_id = 0; block_id < num_blocks; ++block_id) {
        futures.emplace_back(pool->push([&, block_id] {
            const auto [begin, end] = get_worker_row_range(work_items, block_id, num_blocks);
            for (size_t i = begin; i < end; ++i) {
                func(i);
            }
        }));
    }

    WaitAllSuccess(futures);
}

}  // namespace detail

template <typename Func>
void
parallel_for(size_t work_items, size_t requested_concurrency, Func&& func) {
    auto concurrency = std::min(work_items, std::max<size_t>(1, requested_concurrency));
    const auto pool = ThreadPool::GetGlobalBuildThreadPool();
    if (pool != nullptr) {
        concurrency = std::min(concurrency, pool->size());
    }
    detail::parallel_for_blocks(work_items, concurrency, std::forward<Func>(func));
}

template <typename Func>
void
parallel_for_workers(size_t worker_count, Func&& func) {
    parallel_for(worker_count, worker_count, std::forward<Func>(func));
}

template <typename Func>
void
parallel_for(size_t work_items, Func&& func) {
    detail::parallel_for_blocks(work_items, GetParallelBuildBlocks(work_items), std::forward<Func>(func));
}

}  // namespace knowhere::sparse::inverted
