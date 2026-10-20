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

#include <atomic>
#include <optional>

namespace knowhere {

// Serializes builds whose selected path has no backend resource admission.
// All such builds share one process-wide reservation, regardless of device.
class BuildScheduler {
 public:
    class Reservation {
     public:
        ~Reservation();
        Reservation(const Reservation&) = delete;
        Reservation&
        operator=(const Reservation&) = delete;
        Reservation(Reservation&& other) noexcept;
        Reservation&
        operator=(Reservation&& other) noexcept;

     private:
        friend class BuildScheduler;
        explicit Reservation(BuildScheduler* scheduler);

        void
        Release() noexcept;

        BuildScheduler* scheduler_;
    };

    static BuildScheduler&
    Instance();

    // Returns nullopt when the slot is occupied. Release after Build, before upload.
    // Acquisition and release may run on different worker threads.
    std::optional<Reservation>
    TryReserve();

 private:
    BuildScheduler() = default;

    std::atomic_flag build_in_use_ = ATOMIC_FLAG_INIT;
};

}  // namespace knowhere
