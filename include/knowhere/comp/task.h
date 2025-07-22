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
#include <memory>
#include <vector>

#include "folly/executors/CPUThreadPoolExecutor.h"

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

class ThreadPool {
 public:
    class ScopedOmpSetter {
        int omp_before;

     public:
        explicit ScopedOmpSetter(int num_threads = 0);
        ~ScopedOmpSetter();
    };
};
std::unique_ptr<ThreadPool::ScopedOmpSetter>
CreateScopeOmpSetter(int num_threads = 0);

}  // namespace knowhere

#endif /* TASK_H */
