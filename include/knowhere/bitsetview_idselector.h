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

#pragma once

#include <faiss/impl/IDSelector.h>
#include <atomic>
#include <exception>
#include <mutex>

#include "knowhere/bitsetview.h"

namespace knowhere {

struct BitsetViewIDSelector final : faiss::IDSelector {
    const BitsetView bitset_view;

    inline BitsetViewIDSelector(BitsetView bitset_view)
        : bitset_view{bitset_view.bound() ? bitset_view : bitset_view.bind()} {
    }

    inline bool
    is_member(faiss::idx_t id) const override final {
        // it is by design that bitset_view.empty() is not tested here
        return (!bitset_view.test(id));
    }
};

// One independent FAISS task owns this selector. Exceptions cannot unwind an
// OpenMP region (even a one-thread region): retain failure here and report it
// only after the FAISS call joins. The success path takes no mutex.
class TaskBitsetViewIDSelector final : public faiss::IDSelector {
 public:
    explicit TaskBitsetViewIDSelector(const BitsetView& filter) : filter_(filter.bind()) {}

    bool is_member(faiss::idx_t id) const noexcept override {
        try {
            return !filter_.test(id);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
            failed_.store(true, std::memory_order_release);
            return false;
        }
    }

    void RethrowIfFailed() const {
        if (!failed_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_) std::rethrow_exception(failure_);
    }

 private:
    const BitsetView filter_;
    mutable std::mutex mutex_;
    mutable std::exception_ptr failure_;
    mutable std::atomic<bool> failed_{false};
};

}  // namespace knowhere
