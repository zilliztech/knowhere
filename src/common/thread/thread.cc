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
#include "knowhere/comp/thread_pool.h"

namespace knowhere {

void
ExecOverSearchThreadPool(std::vector<std::function<void()>>& tasks) {
    auto pool = ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(tasks.size());
    for (auto&& t : tasks) {
        futures.emplace_back(pool->push([&t]() {
            ThreadPool::ScopedSearchOmpSetter setter(1);
            t();
        }));
    }
    std::this_thread::yield();
    WaitAllSuccess(futures);
}

void
ExecOverBuildThreadPool(std::vector<std::function<void()>>& tasks) {
    auto pool = ThreadPool::GetGlobalBuildThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(tasks.size());
    for (auto&& t : tasks) {
        futures.emplace_back(pool->push([&t]() {
            ThreadPool::ScopedBuildOmpSetter setter(1);
            t();
        }));
    }
    std::this_thread::yield();
    WaitAllSuccess(futures);
}

void
InitBuildThreadPool(uint32_t num_threads) {
    ThreadPool::InitGlobalBuildThreadPool(num_threads);
}

folly::CPUThreadPoolExecutor&
GetBuildThreadPool() {
    return ThreadPool::GetGlobalBuildThreadPool()->GetPool();
}

folly::CPUThreadPoolExecutor&
GetSearchThreadPool() {
    return ThreadPool::GetGlobalSearchThreadPool()->GetPool();
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

std::unique_ptr<ThreadPool::ScopedBuildOmpSetter>
CreateScopeBuildOmpSetter(int num_threads) {
    return std::make_unique<ThreadPool::ScopedBuildOmpSetter>(num_threads);
}

std::unique_ptr<ThreadPool::ScopedSearchOmpSetter>
CreateScopeSearchOmpSetter(int num_threads) {
    return std::make_unique<ThreadPool::ScopedSearchOmpSetter>(num_threads);
}

}  // namespace knowhere
