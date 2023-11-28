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

#pragma once

#include <math.h>  
#include <memory>
#include <utility>

#include "ctpl/ctpl-std.h"
namespace knowhere {

class ThreadPool {
 public:
    explicit ThreadPool(uint32_t num_threads);

    ThreadPool(const ThreadPool&) = delete;

    ThreadPool&
    operator=(const ThreadPool&) = delete;

    ThreadPool(ThreadPool&&) noexcept = delete;

    ThreadPool&
    operator=(ThreadPool&&) noexcept = delete;

    template <typename Func, typename... Args>
    auto
    push(Func&& func, Args&&... args) -> std::future<decltype(func(args...))> {
        return pool_->push([func = std::forward<Func>(func), &args...](int /* unused */) mutable {
            return func(std::forward<Args>(args)...);
        });
    }

    uint32_t
    size() const noexcept;

    /**
     * @brief Set the threads number to the global thread pool of knowhere
     *
     * @param num_threads
     */
    static void
    InitGlobalThreadPool(uint32_t num_threads);

    /**
     * @brief Get the global thread pool of knowhere.
     *
     * @return ThreadPool&
     */
    static std::shared_ptr<ThreadPool>
    GetGlobalThreadPool();

    /**
     * @brief Get the global async thread pool of knowhere.
     *
     * @return ThreadPool&
     */
    static std::shared_ptr<ThreadPool>
    GetGlobalAsyncThreadPool();

    class ScopedOmpSetter {
        int omp_before;
     public:
        explicit ScopedOmpSetter(int num_threads = 1) : omp_before(omp_get_num_threads()) {
            omp_set_num_threads(num_threads);
        }
        ~ScopedOmpSetter() {
            omp_set_num_threads(omp_before);
        }
    };

 private:
    std::unique_ptr<ctpl::thread_pool> pool_;
};
}  // namespace knowhere
