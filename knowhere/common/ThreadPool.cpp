// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "knowhere/common/ThreadPool.h"

#include "knowhere/common/Log.h"

namespace knowhere {

namespace {
static uint32_t global_thread_pool_size_ = 0;
static std::mutex global_thread_pool_mutex_;
}  // namespace

ThreadPool::ThreadPool(uint32_t num_threads) {
    pool_ = std::make_unique<ctpl::thread_pool>(num_threads);
}

uint32_t
ThreadPool::size() const noexcept {
    return pool_->size();
}

void
ThreadPool::InitGlobalThreadPool(uint32_t num_threads) {
    if (num_threads <= 0) {
        LOG_KNOWHERE_ERROR_ << "num_threads should be bigger than 0";
        return;
    }

    if (global_thread_pool_size_ == 0) {
        std::lock_guard<std::mutex> lock(global_thread_pool_mutex_);
        if (global_thread_pool_size_ == 0) {
            global_thread_pool_size_ = num_threads;
            return;
        }
    }
    LOG_KNOWHERE_WARNING_ << "Global ThreadPool has already been inialized with threads num: "
                          << global_thread_pool_size_;
}

std::shared_ptr<ThreadPool>
ThreadPool::GetGlobalThreadPool() {
    if (global_thread_pool_size_ == 0) {
        std::lock_guard<std::mutex> lock(global_thread_pool_mutex_);
        if (global_thread_pool_size_ == 0) {
            global_thread_pool_size_ = std::thread::hardware_concurrency();
            LOG_KNOWHERE_WARNING_ << "Global ThreadPool has not been inialized yet, init it now with threads num: "
                                  << global_thread_pool_size_;
        }
    }
    static auto pool = std::make_shared<ThreadPool>(global_thread_pool_size_);
    return pool;
}

std::shared_ptr<ThreadPool>
ThreadPool::GetGlobalAsyncThreadPool() {
    if (global_thread_pool_size_ == 0) {
        std::lock_guard<std::mutex> lock(global_thread_pool_mutex_);
        if (global_thread_pool_size_ == 0) {
            global_thread_pool_size_ = std::thread::hardware_concurrency();
        }
    }
    uint32_t async_thread_pool_size = int(std::ceil(global_thread_pool_size_ / 2.0));
    LOG_KNOWHERE_WARNING_ << "async thread pool size init with thread number:"
                          << async_thread_pool_size;
    static auto async_pool = std::make_shared<ThreadPool>(async_thread_pool_size);
    return async_pool;
}
}  // namespace knowhere
