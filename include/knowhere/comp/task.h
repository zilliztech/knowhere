//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.
#ifndef KNOWHERE_COMP_TASK_H
#define KNOWHERE_COMP_TASK_H
#include <functional>
#include <vector>

#include "folly/executors/CPUThreadPoolExecutor.h"
#include "knowhere/expected.h"
#include "knowhere/thread_pool.h"

namespace knowhere {

void
ExecOverSearchThreadPool(std::vector<std::function<void()>>& tasks);
void
ExecOverBuildThreadPool(std::vector<std::function<void()>>& tasks);
void
InitBuildThreadPool(uint32_t num_threads);
void
InitSearchThreadPool(uint32_t num_threads);
size_t
GetSearchThreadPoolSize();
size_t
GetBuildThreadPoolSize();

folly::CPUThreadPoolExecutor&
GetSearchThreadPool();

folly::CPUThreadPoolExecutor&
GetBuildThreadPool();

// T is either folly::Unit or Status
template <typename T>
inline Status
WaitAllSuccess(std::vector<folly::Future<T>>& futures) {
    static_assert(std::is_same<T, folly::Unit>::value || std::is_same<T, Status>::value,
                  "WaitAllSuccess can only be used with folly::Unit or knowhere::Status");
    auto allFuts = folly::collectAll(futures.begin(), futures.end()).get();
    for (const auto& result : allFuts) {
        result.throwUnlessValue();
        if constexpr (!std::is_same_v<T, folly::Unit>) {
            if (result.value() != Status::success) {
                return result.value();
            }
        }
    }
    return Status::success;
}

}  // namespace knowhere

#endif /* TASK_H */
