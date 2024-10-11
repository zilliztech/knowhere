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

#include "knowhere/index/interrupt.h"

#include "folly/futures/Future.h"
namespace knowhere {
#ifdef KNOWHERE_WITH_CARDINAL
Interrupt::Interrupt(const std::chrono::seconds& timeout) : start(std::chrono::steady_clock::now()), timeout(timeout) {
}
#else
Interrupt::Interrupt() = default;
#endif

#ifdef KNOWHERE_WITH_CARDINAL
void
Interrupt::Stop() {
    this->flag.store(true);
};

bool
Interrupt::Flag() const {
    return this->flag.load();
}

bool
Interrupt::IsTimeout() const {
    auto now = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::seconds>(now - start);
    return dur.count() > timeout.count();
}
#endif
Status
Interrupt::Get() {
    future->wait();
#ifdef KNOWHERE_WITH_CARDINAL
    if (this->Flag() || this->IsTimeout())
        return Status::timeout;
#endif
    return std::move(*future).get();
}

void
Interrupt::Set(folly::Future<Status>&& future) {
    this->future = std::make_unique<folly::Future<Status>>(std::move(future));
}

}  // namespace knowhere
