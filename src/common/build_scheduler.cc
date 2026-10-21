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

#include "knowhere/build_scheduler.h"

#include <utility>

namespace knowhere {

BuildScheduler::Reservation::~Reservation() {
    Release();
}

BuildScheduler::Reservation::Reservation(Reservation&& other) noexcept
    : scheduler_(std::exchange(other.scheduler_, nullptr)) {
}

BuildScheduler::Reservation&
BuildScheduler::Reservation::operator=(Reservation&& other) noexcept {
    if (this != &other) {
        Release();
        scheduler_ = std::exchange(other.scheduler_, nullptr);
    }
    return *this;
}

BuildScheduler::Reservation::Reservation(BuildScheduler* scheduler) : scheduler_(scheduler) {
}

void
BuildScheduler::Reservation::Release() noexcept {
    if (auto* scheduler = std::exchange(scheduler_, nullptr)) {
        scheduler->build_in_use_.clear(std::memory_order_release);
    }
}

BuildScheduler&
BuildScheduler::Instance() {
    static BuildScheduler scheduler;
    return scheduler;
}

std::optional<BuildScheduler::Reservation>
BuildScheduler::TryReserve() {
    if (build_in_use_.test_and_set(std::memory_order_acquire)) {
        return std::nullopt;
    }
    return Reservation(this);
}

}  // namespace knowhere
