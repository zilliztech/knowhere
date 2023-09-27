// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <omp.h>
#include <sys/resource.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/futures/Future.h"
#include "knowhere/log.h"

namespace knowhere {

class ThreadPool {
 private:
    class LowPriorityThreadFactory : public folly::NamedThreadFactory {
     public:
        using folly::NamedThreadFactory::NamedThreadFactory;
        std::thread
        newThread(folly::Func&& func) override {
            return folly::NamedThreadFactory::newThread([&, func = std::move(func)]() mutable {
                if (setpriority(PRIO_PROCESS, gettid(), 19) != 0) {
                    LOG_KNOWHERE_ERROR_ << "Failed to set priority of knowhere thread. Error is: "
                                        << std::strerror(errno);
                } else {
                    LOG_KNOWHERE_INFO_ << "Successfully set priority of knowhere thread.";
                }
                func();
            });
        }
    };

 public:
    explicit ThreadPool(uint32_t num_threads, const std::string& thread_name_prefix)
        : pool_(folly::CPUThreadPoolExecutor(
              num_threads,
              std::make_unique<
                  folly::LifoSemMPMCQueue<folly::CPUThreadPoolExecutor::CPUTask, folly::QueueBehaviorIfFull::BLOCK>>(
                  num_threads * kTaskQueueFactor),
              std::make_shared<LowPriorityThreadFactory>(thread_name_prefix))) {
    }

    ThreadPool(const ThreadPool&) = delete;

    ThreadPool&
    operator=(const ThreadPool&) = delete;

    ThreadPool(ThreadPool&&) noexcept = delete;

    ThreadPool&
    operator=(ThreadPool&&) noexcept = delete;

    template <typename Func, typename... Args>
    auto
    push(Func&& func, Args&&... args) {
        return folly::makeSemiFuture().via(&pool_).then(
            [func = std::forward<Func>(func), &args...](auto&&) mutable { return func(std::forward<Args>(args)...); });
    }

    [[nodiscard]] int32_t
    size() const noexcept {
        return pool_.numThreads();
    }

    /**
     * @brief Set the threads number to the global build thread pool of knowhere
     *
     * @param num_threads
     */
    static void
    InitThreadPool(uint32_t num_threads, uint32_t& thread_pool_size) {
        if (num_threads <= 0) {
            LOG_KNOWHERE_ERROR_ << "num_threads should be bigger than 0";
            return;
        }

        if (thread_pool_size == 0) {
            std::lock_guard<std::mutex> lock(global_thread_pool_mutex_);
            if (thread_pool_size == 0) {
                thread_pool_size = num_threads;
                return;
            }
        }
    }

    static void
    InitGlobalBuildThreadPool(uint32_t num_threads) {
        InitThreadPool(num_threads, global_build_thread_pool_size_);
        LOG_KNOWHERE_WARNING_ << "Global Build ThreadPool has already been initialized with threads num: "
                              << global_build_thread_pool_size_;
    }

    /**
     * @brief Set the threads number to the global search thread pool of knowhere
     *
     * @param num_threads
     */
    static void
    InitGlobalSearchThreadPool(uint32_t num_threads) {
        InitThreadPool(num_threads, global_search_thread_pool_size_);
        LOG_KNOWHERE_WARNING_ << "Global Search ThreadPool has already been initialized with threads num: "
                              << global_search_thread_pool_size_;
    }

    /**
     * @brief Get the global thread pool of knowhere.
     *
     * @return ThreadPool&
     */

    static std::shared_ptr<ThreadPool>
    GetGlobalBuildThreadPool() {
        if (global_build_thread_pool_size_ == 0) {
            InitThreadPool(std::thread::hardware_concurrency(), global_build_thread_pool_size_);
            LOG_KNOWHERE_WARNING_ << "Global Build ThreadPool has not been initialized yet, init it with threads num: "
                                  << global_build_thread_pool_size_;
        }
        static auto pool = std::make_shared<ThreadPool>(global_build_thread_pool_size_, "Knowhere_Build");
        return pool;
    }

    static std::shared_ptr<ThreadPool>
    GetGlobalSearchThreadPool() {
        if (global_search_thread_pool_size_ == 0) {
            InitThreadPool(std::thread::hardware_concurrency(), global_search_thread_pool_size_);
            LOG_KNOWHERE_WARNING_ << "Global Search ThreadPool has not been initialized yet, init it with threads num: "
                                  << global_search_thread_pool_size_;
        }
        static auto pool = std::make_shared<ThreadPool>(global_search_thread_pool_size_, "Knowhere_Search");
        return pool;
    }

    class ScopedOmpSetter {
        int omp_before;

     public:
        explicit ScopedOmpSetter(int num_threads = 1) : omp_before(omp_get_max_threads()) {
            omp_set_num_threads(num_threads);
        }
        ~ScopedOmpSetter() {
            omp_set_num_threads(omp_before);
        }
    };

 private:
    folly::CPUThreadPoolExecutor pool_;
    inline static uint32_t global_build_thread_pool_size_ = 0;
    inline static uint32_t global_search_thread_pool_size_ = 0;
    inline static std::mutex global_thread_pool_mutex_;
    constexpr static size_t kTaskQueueFactor = 16;
};
}  // namespace knowhere
