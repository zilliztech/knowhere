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

#include <omp.h>
#include <sys/resource.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/futures/Future.h"
#include "knowhere/index_node_thread_pool_wrapper.h"
#include "knowhere/log.h"

namespace knowhere {

class ThreadPool {
#ifdef __linux__
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
#else
 public:
    explicit ThreadPool(uint32_t num_threads, const std::string& thread_name_prefix)
        : pool_(folly::CPUThreadPoolExecutor(
              num_threads,
              std::make_unique<
                  folly::LifoSemMPMCQueue<folly::CPUThreadPoolExecutor::CPUTask, folly::QueueBehaviorIfFull::BLOCK>>(
                  num_threads * kTaskQueueFactor),
              std::make_shared<folly::NamedThreadFactory>(thread_name_prefix))) {
    }
#endif

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
        explicit ScopedOmpSetter(int num_threads = 0);
        ~ScopedOmpSetter();
    };

 private:
    folly::CPUThreadPoolExecutor pool_;
    inline static uint32_t global_build_thread_pool_size_ = 0;
    inline static uint32_t global_search_thread_pool_size_ = 0;
    inline static std::mutex global_thread_pool_mutex_;
    constexpr static size_t kTaskQueueFactor = 16;
};

ThreadPool::ScopedOmpSetter::ScopedOmpSetter(int num_threads) {
    if (global_build_thread_pool_size_ == 0) {  // this should not happen in prod
        omp_before = omp_get_max_threads();
    } else {
        omp_before = global_build_thread_pool_size_;
    }

    omp_set_num_threads(num_threads <= 0 ? omp_before : num_threads);
}

ThreadPool::ScopedOmpSetter::~ScopedOmpSetter() {
    omp_set_num_threads(omp_before);
}

void
ExecOverSearchThreadPool(std::vector<std::function<void()>>& tasks) {
    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(tasks.size());
    for (auto&& t : tasks) {
        futures.emplace_back(pool->push([&t]() {
            ThreadPool::ScopedOmpSetter setter(1);
            t();
        }));
    }
    std::this_thread::yield();
    // check for exceptions. value() is {}, so either
    //   a call does nothing, or it throws an inner exception.
    for (auto& f : futures) {
        f.wait();
    }
    for (auto& f : futures) {
        f.result().value();
    }
}

void
ExecOverBuildThreadPool(std::vector<std::function<void()>>& tasks) {
    auto pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(tasks.size());
    for (auto&& t : tasks) {
        futures.emplace_back(pool->push([&t]() {
            ThreadPool::ScopedOmpSetter setter(1);
            t();
        }));
    }
    std::this_thread::yield();
    // check for exceptions. value() is {}, so either
    //   a call does nothing, or it throws an inner exception.
    for (auto& f : futures) {
        f.wait();
    }
    for (auto& f : futures) {
        f.result().value();
    }
}

void
InitBuildThreadPool(uint32_t num_threads) {
    ThreadPool::InitGlobalBuildThreadPool(num_threads);
}

void
InitSearchThreadPool(uint32_t num_threads) {
    ThreadPool::InitGlobalSearchThreadPool(num_threads);
}

size_t
GetSearchThreadPoolSize() {
    return ThreadPool::GetGlobalSearchThreadPool()->size();
}

size_t
GetBuildThreadPoolSize() {
    return ThreadPool::GetGlobalBuildThreadPool()->size();
}

std::unique_ptr<ThreadPool::ScopedOmpSetter>
CreateScopeOmpSetter(int num_threads) {
    return std::make_unique<ThreadPool::ScopedOmpSetter>(num_threads);
}

namespace {

std::shared_ptr<ThreadPool>
GlobalThreadPool(size_t pool_size) {
    static std::shared_ptr<ThreadPool> pool = std::make_shared<ThreadPool>(pool_size, "Knowhere_Global");
    return pool;
}

}  // namespace

IndexNodeThreadPoolWrapper::IndexNodeThreadPoolWrapper(std::unique_ptr<IndexNode> index_node, size_t pool_size)
    : IndexNodeThreadPoolWrapper(std::move(index_node), GlobalThreadPool(pool_size)) {
}

IndexNodeThreadPoolWrapper::IndexNodeThreadPoolWrapper(std::unique_ptr<IndexNode> index_node,
                                                       std::shared_ptr<ThreadPool> thread_pool)
    : index_node_(std::move(index_node)), thread_pool_(thread_pool) {
}

expected<DataSetPtr>
IndexNodeThreadPoolWrapper::Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    return thread_pool_->push([&]() { return this->index_node_->Search(dataset, cfg, bitset); }).get();
}

expected<DataSetPtr>
IndexNodeThreadPoolWrapper::RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const {
    return thread_pool_->push([&]() { return this->index_node_->RangeSearch(dataset, cfg, bitset); }).get();
}

}  // namespace knowhere
