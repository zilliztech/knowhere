// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.
#ifndef KNOWHERE_INTERRUPT_H
#define KNOWHERE_INTERRUPT_H
#include <atomic>
#include <chrono>
#include <memory>

#include "knowhere/expected.h"
namespace folly {
template <typename T>
class Future;
}  // namespace folly

namespace knowhere {
class Interrupt {
 public:
#ifdef KNOWHERE_WITH_CARDINAL
    explicit Interrupt(const std::chrono::seconds& timeout);

    void
    Stop();

    bool
    Flag() const;

    bool
    IsTimeout() const;
#else
    Interrupt();
#endif

    Status
    Get();

    void
    Set(folly::Future<Status>&& future);

    ~Interrupt();

 private:
#ifdef KNOWHERE_WITH_CARDINAL
    std::chrono::steady_clock::time_point start;
    std::chrono::seconds timeout;
    mutable std::atomic_bool flag = false;
#endif
    std::unique_ptr<folly::Future<Status>> future;
};
}  // namespace knowhere
#endif /* INTERRUPT_H */
