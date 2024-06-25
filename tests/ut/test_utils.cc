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

#include <chrono>
#include <thread>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/expected.h"
#include "knowhere/heap.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"
#include "utils.h"

namespace {
const std::vector<size_t> kBitsetSizes{4, 8, 10, 64, 100, 500, 1024};
}

TEST_CASE("Test Vector Normalization", "[normalize]") {
    using Catch::Approx;

    const float floatDiff = 0.00001;
    uint64_t nb = 1000000;
    uint64_t dim = 128;
    int64_t seed = 42;

    SECTION("Test normalize") {
        auto train_ds = GenDataSet(nb, dim, seed);
        auto data = (float*)train_ds->GetTensor();

        knowhere::Normalize(train_ds);

        for (size_t i = 0; i < nb; ++i) {
            float sum = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                auto val = data[i * dim + j];
                sum += val * val;
            }
            CHECK(std::abs(1.0f - sum) <= floatDiff);
        }
    }
}

TEST_CASE("Test Bitset Generation", "[utils]") {
    SECTION("Sequential") {
        for (const auto size : kBitsetSizes) {
            for (size_t i = 0; i <= size; ++i) {
                auto bitset_data = GenerateBitsetWithFirstTbitsSet(size, i);
                knowhere::BitsetView bitset(bitset_data.data(), size);
                for (size_t j = 0; j < i; ++j) {
                    REQUIRE(bitset.test(j));
                }
                for (size_t j = i; j < size; ++j) {
                    REQUIRE(!bitset.test(j));
                }
            }
        }
    }

    SECTION("Random") {
        for (const auto size : kBitsetSizes) {
            for (size_t i = 0; i <= size; ++i) {
                auto bitset_data = GenerateBitsetWithRandomTbitsSet(size, i);
                knowhere::BitsetView bitset(bitset_data.data(), size);
                size_t cnt = 0;
                for (size_t j = 0; j < size; ++j) {
                    cnt += bitset.test(j);
                }
                REQUIRE(cnt == i);
            }
        }
    }
}

namespace {
constexpr size_t kHeapSize = 10;
constexpr size_t kElementCount = 10000;
}  // namespace

TEST_CASE("ResultMaxHeap") {
    knowhere::ResultMaxHeap<float, size_t> heap(kHeapSize);
    auto pairs = GenerateRandomDistanceIdPair(kElementCount);
    for (const auto& [dist, id] : pairs) {
        heap.Push(dist, id);
    }
    REQUIRE(heap.Size() == kHeapSize);
    std::sort(pairs.begin(), pairs.end());
    for (int i = kHeapSize - 1; i >= 0; --i) {
        auto op = heap.Pop();
        REQUIRE(op.has_value());
        REQUIRE(op.value().second == pairs[i].second);
    }
    REQUIRE(heap.Size() == 0);
}

TEST_CASE("Test Time Recorder") {
    knowhere::TimeRecorder tr("test", 2);
    int64_t sum = 0;
    for (int i = 0; i < 10000; i++) {
        sum += i * i;
    }
    auto span = tr.ElapseFromBegin("done");
    REQUIRE(span > 0);
    REQUIRE(sum > 0);
}

TEST_CASE("Test Version") {
    REQUIRE(knowhere::Version::VersionSupport(knowhere::Version::GetDefaultVersion()));
    REQUIRE(knowhere::Version::VersionSupport(knowhere::Version::GetMinimalVersion()));
    REQUIRE(knowhere::Version::VersionSupport(knowhere::Version::GetCurrentVersion()));
}

TEST_CASE("Test DiskLoad") {
    REQUIRE(knowhere::UseDiskLoad(knowhere::IndexEnum::INDEX_DISKANN,
                                  knowhere::Version::GetCurrentVersion().VersionNumber()));
#ifdef KNOWHERE_WITH_CARDINAL
    REQUIRE(
        knowhere::UseDiskLoad(knowhere::IndexEnum::INDEX_HNSW, knowhere::Version::GetCurrentVersion().VersionNumber()));
#else
    REQUIRE(!knowhere::UseDiskLoad(knowhere::IndexEnum::INDEX_HNSW,
                                   knowhere::Version::GetCurrentVersion().VersionNumber()));
#endif
}

TEST_CASE("Test WaitAllSuccess with folly::Unit futures") {
    auto pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<folly::Unit>> futures;

    SECTION("All futures succeed") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([]() { return folly::Unit(); }));
        }
        REQUIRE(knowhere::WaitAllSuccess(futures) == knowhere::Status::success);
    }

    SECTION("One future throws an exception") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([i]() {
                if (i == 5) {
                    throw std::runtime_error("Task failed");
                }
                return folly::Unit();
            }));
        }
        REQUIRE_THROWS_AS(knowhere::WaitAllSuccess(futures), std::runtime_error);
    }

    SECTION("WaitAllSuccess should wait until all tasks finish even if any throws exception") {
        std::atomic<int> externalValue{0};

        futures.emplace_back(pool->push([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            REQUIRE(externalValue.load() == 1);
            externalValue.store(2);
            return folly::Unit();
        }));

        futures.emplace_back(pool->push([&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            externalValue.store(1);
            throw std::runtime_error("Task failed");
        }));

        REQUIRE_THROWS_AS(knowhere::WaitAllSuccess(futures), std::runtime_error);
        REQUIRE(externalValue.load() == 2);
    }
}

TEST_CASE("Test WaitAllSuccess with knowhere::Status futures") {
    auto pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
    std::vector<folly::Future<knowhere::Status>> futures;

    SECTION("All futures succeed with Status::success") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([]() { return knowhere::Status::success; }));
        }
        REQUIRE(knowhere::WaitAllSuccess(futures) == knowhere::Status::success);
    }

    SECTION("One future returns Status::invalid_args") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([i]() {
                if (i == 5) {
                    return knowhere::Status::invalid_args;
                }
                return knowhere::Status::success;
            }));
        }
        REQUIRE(knowhere::WaitAllSuccess(futures) == knowhere::Status::invalid_args);
    }

    SECTION("One future throws an exception") {
        for (size_t i = 0; i < 10; ++i) {
            futures.emplace_back(pool->push([i]() {
                if (i == 5) {
                    throw std::runtime_error("Task failed");
                }
                return knowhere::Status::success;
            }));
        }
        REQUIRE_THROWS_AS(knowhere::WaitAllSuccess(futures), std::runtime_error);
    }
}
