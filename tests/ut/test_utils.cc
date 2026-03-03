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
#include "knowhere/comp/task.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/expected.h"
#include "knowhere/heap.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"
#include "utils.h"

namespace {
const std::vector<size_t> kBitsetSizes{4, 8, 10, 64, 100, 500, 1024};
}

template <typename T>
void
CheckNormalizeDataset(int rows, int dim, float diff) {
    auto ds = GenDataSet(rows, dim);
    auto type_ds = knowhere::ConvertToDataTypeIfNeeded<T>(ds);
    auto data = (T*)type_ds->GetTensor();
    knowhere::NormalizeDataset<T>(type_ds);
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            auto val = data[i * dim + j];
            sum += val * val;
        }
        CHECK(std::abs(1.0f - sum) <= diff);
    }
}

template <typename T>
void
CheckCopyAndNormalizeVecs(int rows, int dim, float diff) {
    auto ds = GenDataSet(rows, dim);
    auto type_ds = knowhere::ConvertToDataTypeIfNeeded<T>(ds);
    auto data = (T*)type_ds->GetTensor();

    auto data_copy = knowhere::CopyAndNormalizeVecs<T>(data, rows, dim);

    for (int i = 0; i < rows; ++i) {
        float sum = 0.0;
        for (int j = 0; j < dim; ++j) {
            auto val = data_copy[i * dim + j];
            sum += val * val;
        }
        CHECK(std::abs(1.0f - sum) <= diff);
    }
}

TEST_CASE("Test Vector Normalization", "[normalize]") {
    using Catch::Approx;

    uint64_t rows = 100;
    uint64_t dim = 128;

    SECTION("Test Normalize Dataset") {
        CheckNormalizeDataset<knowhere::fp32>(rows, dim, 0.00001);
        CheckNormalizeDataset<knowhere::fp16>(rows, dim, 0.001);
        CheckNormalizeDataset<knowhere::bf16>(rows, dim, 0.01);
    }

    SECTION("Test Copy and Normalize Vectors") {
        CheckCopyAndNormalizeVecs<knowhere::fp32>(rows, dim, 0.00001);
        CheckCopyAndNormalizeVecs<knowhere::fp16>(rows, dim, 0.001);
        CheckCopyAndNormalizeVecs<knowhere::bf16>(rows, dim, 0.01);
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

TEST_CASE("ResultMinHeap") {
    // ResultMinHeap keeps k largest values (min-heap, smallest on top)
    knowhere::ResultMinHeap<float, size_t> heap(kHeapSize);
    auto pairs = GenerateRandomDistanceIdPair(kElementCount);
    for (const auto& [dist, id] : pairs) {
        heap.Push(dist, id);
    }
    REQUIRE(heap.Size() == kHeapSize);
    // Sort descending to get the largest k elements
    std::sort(pairs.begin(), pairs.end(), std::greater<>());
    // Pop returns worst (smallest) first
    for (int i = kHeapSize - 1; i >= 0; --i) {
        auto op = heap.Pop();
        REQUIRE(op.has_value());
        REQUIRE(op.value().second == pairs[i].second);
    }
    REQUIRE(heap.Size() == 0);
}

TEST_CASE("ResultHeap Push returns bool") {
    knowhere::ResultMaxHeap<float, size_t> heap(3);
    // Before heap is full, Push always returns true
    REQUIRE(heap.Push(10.0f, 0) == true);
    REQUIRE(heap.Push(20.0f, 1) == true);
    REQUIRE(heap.Push(5.0f, 2) == true);
    REQUIRE(heap.Full());
    // Threshold should be the worst (largest distance for MaxHeap)
    REQUIRE(heap.Threshold() == 20.0f);
    // Pushing a worse value should return false
    REQUIRE(heap.Push(25.0f, 3) == false);
    REQUIRE(heap.Push(20.0f, 4) == false);  // equal to threshold, not strictly less
    // Pushing a better value should return true
    REQUIRE(heap.Push(3.0f, 5) == true);
    REQUIRE(heap.Size() == 3);
}

TEST_CASE("ResultHeap WouldEnter") {
    knowhere::ResultMaxHeap<float, size_t> heap(2);
    // Before full, everything would enter
    REQUIRE(heap.WouldEnter(1000.0f) == true);
    heap.Push(5.0f, 0);
    REQUIRE(heap.WouldEnter(1000.0f) == true);
    heap.Push(10.0f, 1);
    REQUIRE(heap.Full());
    // Now threshold is 10.0
    REQUIRE(heap.WouldEnter(3.0f) == true);    // better than threshold
    REQUIRE(heap.WouldEnter(10.0f) == false);  // equal, not strictly better
    REQUIRE(heap.WouldEnter(15.0f) == false);  // worse than threshold
}

TEST_CASE("ResultHeap WouldEnter MinHeap") {
    knowhere::ResultMinHeap<float, size_t> heap(2);
    heap.Push(5.0f, 0);
    heap.Push(10.0f, 1);
    REQUIRE(heap.Full());
    // MinHeap keeps largest, threshold is the smallest (worst) = 5.0
    REQUIRE(heap.Threshold() == 5.0f);
    REQUIRE(heap.WouldEnter(15.0f) == true);  // greater than threshold → enters
    REQUIRE(heap.WouldEnter(5.0f) == false);  // equal → does not enter
    REQUIRE(heap.WouldEnter(1.0f) == false);  // less → does not enter
}

TEST_CASE("ResultHeap Finalize and Results") {
    knowhere::ResultMaxHeap<float, size_t> heap(5);
    // Push values in random order
    heap.Push(30.0f, 3);
    heap.Push(10.0f, 1);
    heap.Push(50.0f, 5);
    heap.Push(20.0f, 2);
    heap.Push(40.0f, 4);
    // Finalize sorts: best first (ascending distance for MaxHeap)
    heap.Finalize();
    const auto& results = heap.Results();
    REQUIRE(results.size() == 5);
    REQUIRE(results[0].first == 10.0f);
    REQUIRE(results[1].first == 20.0f);
    REQUIRE(results[2].first == 30.0f);
    REQUIRE(results[3].first == 40.0f);
    REQUIRE(results[4].first == 50.0f);
}

TEST_CASE("ResultMinHeap Finalize and Results") {
    knowhere::ResultMinHeap<float, size_t> heap(5);
    heap.Push(30.0f, 3);
    heap.Push(10.0f, 1);
    heap.Push(50.0f, 5);
    heap.Push(20.0f, 2);
    heap.Push(40.0f, 4);
    // Finalize sorts: best first (descending score for MinHeap)
    heap.Finalize();
    const auto& results = heap.Results();
    REQUIRE(results.size() == 5);
    REQUIRE(results[0].first == 50.0f);
    REQUIRE(results[1].first == 40.0f);
    REQUIRE(results[2].first == 30.0f);
    REQUIRE(results[3].first == 20.0f);
    REQUIRE(results[4].first == 10.0f);
}

TEST_CASE("ResultHeap edge cases") {
    SECTION("Empty heap") {
        knowhere::ResultMaxHeap<float, size_t> heap(5);
        REQUIRE(heap.Size() == 0);
        REQUIRE(heap.Capacity() == 5);
        REQUIRE_FALSE(heap.Full());
        REQUIRE(heap.Pop() == std::nullopt);
    }

    SECTION("Single element") {
        knowhere::ResultMaxHeap<float, size_t> heap(1);
        REQUIRE(heap.Push(42.0f, 0) == true);
        REQUIRE(heap.Full());
        REQUIRE(heap.Size() == 1);
        REQUIRE(heap.Threshold() == 42.0f);
        // Anything worse than 42 is rejected
        REQUIRE(heap.Push(100.0f, 1) == false);
        // Better value replaces
        REQUIRE(heap.Push(10.0f, 2) == true);
        REQUIRE(heap.Size() == 1);
        auto op = heap.Pop();
        REQUIRE(op.has_value());
        REQUIRE(op.value().first == 10.0f);
        REQUIRE(op.value().second == 2);
    }

    SECTION("Clear") {
        knowhere::ResultMaxHeap<float, size_t> heap(3);
        heap.Push(1.0f, 0);
        heap.Push(2.0f, 1);
        heap.Push(3.0f, 2);
        REQUIRE(heap.Full());
        heap.Clear();
        REQUIRE(heap.Size() == 0);
        REQUIRE_FALSE(heap.Full());
        // After clear, WouldEnter should accept everything
        REQUIRE(heap.WouldEnter(1000.0f) == true);
        // Can push again
        REQUIRE(heap.Push(99.0f, 10) == true);
        REQUIRE(heap.Size() == 1);
    }

    SECTION("Fewer elements than k") {
        knowhere::ResultMaxHeap<float, size_t> heap(10);
        heap.Push(5.0f, 0);
        heap.Push(3.0f, 1);
        heap.Push(7.0f, 2);
        REQUIRE(heap.Size() == 3);
        REQUIRE_FALSE(heap.Full());
        heap.Finalize();
        const auto& results = heap.Results();
        REQUIRE(results.size() == 3);
        // Should still be sorted ascending
        REQUIRE(results[0].first == 3.0f);
        REQUIRE(results[1].first == 5.0f);
        REQUIRE(results[2].first == 7.0f);
    }
}

TEST_CASE("ResultMaxHeap correctness with large dataset") {
    // Verify that ResultMaxHeap keeps the k smallest distances
    constexpr size_t k = 100;
    constexpr size_t n = 50000;
    knowhere::ResultMaxHeap<float, size_t> heap(k);
    auto pairs = GenerateRandomDistanceIdPair(n);
    for (const auto& [dist, id] : pairs) {
        heap.Push(dist, id);
    }
    REQUIRE(heap.Size() == k);

    // Get sorted reference: k smallest distances
    std::sort(pairs.begin(), pairs.end());

    heap.Finalize();
    const auto& results = heap.Results();
    REQUIRE(results.size() == k);
    for (size_t i = 0; i < k; ++i) {
        REQUIRE(results[i].first == pairs[i].first);
        REQUIRE(results[i].second == pairs[i].second);
    }
}

TEST_CASE("ResultMinHeap correctness with large dataset") {
    // Verify that ResultMinHeap keeps the k largest scores
    constexpr size_t k = 100;
    constexpr size_t n = 50000;
    knowhere::ResultMinHeap<float, size_t> heap(k);
    auto pairs = GenerateRandomDistanceIdPair(n);
    for (const auto& [dist, id] : pairs) {
        heap.Push(dist, id);
    }
    REQUIRE(heap.Size() == k);

    // Get sorted reference: k largest scores (descending)
    std::sort(pairs.begin(), pairs.end(), std::greater<>());

    heap.Finalize();
    const auto& results = heap.Results();
    REQUIRE(results.size() == k);
    for (size_t i = 0; i < k; ++i) {
        REQUIRE(results[i].first == pairs[i].first);
        REQUIRE(results[i].second == pairs[i].second);
    }
}

TEST_CASE("ResultHeap threshold updates correctly") {
    knowhere::ResultMaxHeap<float, size_t> heap(3);
    heap.Push(10.0f, 0);
    heap.Push(20.0f, 1);
    heap.Push(30.0f, 2);
    // Threshold should be the worst (largest) = 30
    REQUIRE(heap.Threshold() == 30.0f);

    // Replace 30 with 15
    heap.Push(15.0f, 3);
    // Now heap has {10, 20, 15}, threshold should be 20
    REQUIRE(heap.Threshold() == 20.0f);

    // Replace 20 with 12
    heap.Push(12.0f, 4);
    // Now heap has {10, 15, 12}, threshold should be 15
    REQUIRE(heap.Threshold() == 15.0f);
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
    REQUIRE(knowhere::Version::VersionSupport(knowhere::Version::GetMaximumVersion()));
    REQUIRE(knowhere::Version::GetMinimalVersion() <= knowhere::Version::GetCurrentVersion());
    REQUIRE(knowhere::Version::GetCurrentVersion() <= knowhere::Version::GetMaximumVersion());
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

TEST_CASE("Test ThreadPool") {
    SECTION("Build thread pool") {
        knowhere::ThreadPool::InitGlobalBuildThreadPool(0);
        auto prev_build_thread_num = knowhere::ThreadPool::GetGlobalBuildThreadPoolSize();
        knowhere::ThreadPool::InitGlobalBuildThreadPool(prev_build_thread_num);

        knowhere::ThreadPool::SetGlobalBuildThreadPoolSize(2);
        REQUIRE(knowhere::ThreadPool::GetGlobalBuildThreadPoolSize() == 2);
        knowhere::ThreadPool::SetGlobalBuildThreadPoolSize(4);
        REQUIRE(knowhere::ThreadPool::GetGlobalBuildThreadPoolSize() == 4);
        knowhere::ThreadPool::SetGlobalBuildThreadPoolSize(0);
        REQUIRE(knowhere::ThreadPool::GetGlobalBuildThreadPoolSize() == 4);

        REQUIRE(knowhere::ThreadPool::GetBuildThreadPoolPendingTaskCount() == 0);

        if (prev_build_thread_num > 0) {
            knowhere::ThreadPool::SetGlobalBuildThreadPoolSize(prev_build_thread_num);
        }
    }

    SECTION("Search thread pool") {
        knowhere::ThreadPool::InitGlobalSearchThreadPool(0);
        auto prev_search_thread_num = knowhere::ThreadPool::GetGlobalSearchThreadPoolSize();
        knowhere::ThreadPool::InitGlobalSearchThreadPool(prev_search_thread_num);

        knowhere::ThreadPool::SetGlobalSearchThreadPoolSize(2);
        REQUIRE(knowhere::ThreadPool::GetGlobalSearchThreadPoolSize() == 2);
        knowhere::ThreadPool::SetGlobalSearchThreadPoolSize(4);
        REQUIRE(knowhere::ThreadPool::GetGlobalSearchThreadPoolSize() == 4);
        knowhere::ThreadPool::SetGlobalSearchThreadPoolSize(0);
        REQUIRE(knowhere::ThreadPool::GetGlobalSearchThreadPoolSize() == 4);

        REQUIRE(knowhere::ThreadPool::GetSearchThreadPoolPendingTaskCount() == 0);

        if (prev_search_thread_num > 0) {
            knowhere::ThreadPool::SetGlobalSearchThreadPoolSize(prev_search_thread_num);
        }
    }

    SECTION("ScopedBuildOmpSetter") {
        int prev_num_threads = knowhere::ThreadPool::GetGlobalBuildThreadPoolSize();
        {
            int target_num_threads = (prev_num_threads / 2) > 0 ? (prev_num_threads / 2) : 1;
            knowhere::ThreadPool::ScopedBuildOmpSetter setter(target_num_threads);
            auto thread_num_1 = omp_get_max_threads();
            REQUIRE(thread_num_1 == target_num_threads);
        }
        auto thread_num_2 = omp_get_max_threads();
        REQUIRE(thread_num_2 == prev_num_threads);
    }

    SECTION("ScopedSearchOmpSetter") {
        int prev_num_threads = knowhere::ThreadPool::GetGlobalSearchThreadPoolSize();
        {
            int target_num_threads = (prev_num_threads / 2) > 0 ? (prev_num_threads / 2) : 1;
            knowhere::ThreadPool::ScopedSearchOmpSetter setter(target_num_threads);
            auto thread_num_1 = omp_get_max_threads();
            REQUIRE(thread_num_1 == target_num_threads);
        }
        auto thread_num_2 = omp_get_max_threads();
        REQUIRE(thread_num_2 == prev_num_threads);
    }
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
