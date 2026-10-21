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

#include <stdexcept>
#include <thread>
#include <utility>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/build_scheduler.h"

using namespace knowhere;

TEST_CASE("Default build admission permits one owner", "[build_admission]") {
    auto& scheduler = BuildScheduler::Instance();
    auto first = scheduler.TryReserve();
    REQUIRE(first.has_value());
    REQUIRE_FALSE(scheduler.TryReserve());
    first.reset();
    REQUIRE(scheduler.TryReserve().has_value());
}

TEST_CASE("Moving a default build reservation transfers ownership", "[build_admission]") {
    auto& scheduler = BuildScheduler::Instance();
    auto first = scheduler.TryReserve();
    REQUIRE(first.has_value());
    auto moved = std::move(first);
    first.reset();
    REQUIRE_FALSE(scheduler.TryReserve());
    first = std::move(moved);
    moved.reset();
    REQUIRE_FALSE(scheduler.TryReserve());
    first.reset();
    REQUIRE(scheduler.TryReserve().has_value());
}

TEST_CASE("Failure and cancellation release default build reservations", "[build_admission]") {
    auto& scheduler = BuildScheduler::Instance();
    REQUIRE_THROWS_AS(
        [&] {
            auto reservation = scheduler.TryReserve();
            REQUIRE(reservation.has_value());
            throw std::runtime_error("build failed");
        }(),
        std::runtime_error);
    REQUIRE(scheduler.TryReserve().has_value());
    {
        auto canceled = scheduler.TryReserve();
        REQUIRE(canceled.has_value());
        REQUIRE_FALSE(scheduler.TryReserve());
    }
    REQUIRE(scheduler.TryReserve().has_value());
}

TEST_CASE("Build workers can release reservations acquired by another thread", "[build_admission]") {
    auto& scheduler = BuildScheduler::Instance();
    auto reservation = scheduler.TryReserve();
    REQUIRE(reservation.has_value());
    bool busy = false;
    std::thread worker([&] {
        busy = !scheduler.TryReserve();
        reservation.reset();
    });
    worker.join();
    REQUIRE(busy);
    REQUIRE(scheduler.TryReserve().has_value());
}
