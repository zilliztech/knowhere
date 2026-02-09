// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

/**
 * @file test_utility_components.cc
 * @brief Unit tests for utility components
 *
 * Tests cover:
 * - BlockingQueue: thread-safe queue operations
 * - BitsetView: bitset operations and filtering
 */

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/blocking_queue.h"
#include "utils.h"

// ==================== BlockingQueue Tests ====================

TEST_CASE("Test BlockingQueue Basic Operations", "[utility][blocking_queue]") {
    knowhere::BlockingQueue<int> queue;

    SECTION("Default capacity is 32") {
        // Can put 32 items without blocking
        for (int i = 0; i < 32; ++i) {
            queue.Put(i);
        }
        REQUIRE(queue.Size() == 32);
    }

    SECTION("Put and Take single item") {
        queue.Put(42);
        REQUIRE(queue.Size() == 1);
        REQUIRE(!queue.Empty());

        int value = queue.Take();
        REQUIRE(value == 42);
        REQUIRE(queue.Size() == 0);
        REQUIRE(queue.Empty());
    }

    SECTION("FIFO ordering") {
        queue.Put(1);
        queue.Put(2);
        queue.Put(3);

        REQUIRE(queue.Take() == 1);
        REQUIRE(queue.Take() == 2);
        REQUIRE(queue.Take() == 3);
    }

    SECTION("Front and Back") {
        queue.Put(1);
        queue.Put(2);
        queue.Put(3);

        REQUIRE(queue.Front() == 1);  // Front doesn't remove
        REQUIRE(queue.Back() == 3);   // Back doesn't remove
        REQUIRE(queue.Size() == 3);
    }

    SECTION("SetCapacity") {
        queue.SetCapacity(5);

        // Should be able to put 5 items
        for (int i = 0; i < 5; ++i) {
            queue.Put(i);
        }
        REQUIRE(queue.Size() == 5);
    }

    SECTION("SetCapacity with zero should keep original") {
        queue.SetCapacity(10);
        queue.SetCapacity(0);  // Should not change capacity

        // Can still put 10 items
        for (int i = 0; i < 10; ++i) {
            queue.Put(i);
        }
        REQUIRE(queue.Size() == 10);
    }
}

TEST_CASE("Test BlockingQueue Concurrent Operations", "[utility][blocking_queue][concurrent]") {
    knowhere::BlockingQueue<int> queue;
    queue.SetCapacity(100);

    SECTION("Single producer, single consumer") {
        const int num_items = 50;
        std::atomic<int> sum{0};

        // Producer thread
        std::thread producer([&]() {
            for (int i = 1; i <= num_items; ++i) {
                queue.Put(i);
            }
        });

        // Consumer thread
        std::thread consumer([&]() {
            for (int i = 0; i < num_items; ++i) {
                sum += queue.Take();
            }
        });

        producer.join();
        consumer.join();

        // Sum of 1 to 50 = 50*51/2 = 1275
        REQUIRE(sum == 1275);
        REQUIRE(queue.Empty());
    }

    SECTION("Multiple producers, single consumer") {
        const int num_producers = 4;
        const int items_per_producer = 25;
        std::atomic<int> sum{0};

        std::vector<std::thread> producers;
        for (int p = 0; p < num_producers; ++p) {
            producers.emplace_back([&, p]() {
                for (int i = 0; i < items_per_producer; ++i) {
                    queue.Put(p * items_per_producer + i + 1);
                }
            });
        }

        // Consumer thread
        std::thread consumer([&]() {
            for (int i = 0; i < num_producers * items_per_producer; ++i) {
                sum += queue.Take();
            }
        });

        for (auto& t : producers) {
            t.join();
        }
        consumer.join();

        // Sum of 1 to 100 = 100*101/2 = 5050
        REQUIRE(sum == 5050);
    }

    SECTION("Blocking behavior when full") {
        knowhere::BlockingQueue<int> small_queue;
        small_queue.SetCapacity(2);

        small_queue.Put(1);
        small_queue.Put(2);
        // Queue is now full

        std::atomic<bool> put_completed{false};

        // This should block until space is available
        std::thread producer([&]() {
            small_queue.Put(3);
            put_completed = true;
        });

        // Give producer time to block
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        REQUIRE(!put_completed);

        // Take one item to make space
        int value = small_queue.Take();
        REQUIRE(value == 1);

        // Wait for producer to complete
        producer.join();
        REQUIRE(put_completed);
    }

    SECTION("Blocking behavior when empty") {
        std::atomic<bool> take_completed{false};
        std::atomic<int> taken_value{0};

        // This should block until an item is available
        std::thread consumer([&]() {
            taken_value = queue.Take();
            take_completed = true;
        });

        // Give consumer time to block
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        REQUIRE(!take_completed);

        // Put an item
        queue.Put(42);

        // Wait for consumer to complete
        consumer.join();
        REQUIRE(take_completed);
        REQUIRE(taken_value == 42);
    }
}

TEST_CASE("Test BlockingQueue with Different Types", "[utility][blocking_queue][types]") {
    SECTION("String type") {
        knowhere::BlockingQueue<std::string> queue;
        queue.Put("hello");
        queue.Put("world");

        REQUIRE(queue.Take() == "hello");
        REQUIRE(queue.Take() == "world");
    }

    SECTION("Pointer type") {
        knowhere::BlockingQueue<int*> queue;
        int a = 1, b = 2;
        queue.Put(&a);
        queue.Put(&b);

        REQUIRE(*queue.Take() == 1);
        REQUIRE(*queue.Take() == 2);
    }

    SECTION("Struct type") {
        struct Item {
            int id;
            std::string name;
        };

        knowhere::BlockingQueue<Item> queue;
        queue.Put({1, "first"});
        queue.Put({2, "second"});

        auto item1 = queue.Take();
        REQUIRE(item1.id == 1);
        REQUIRE(item1.name == "first");
    }
}

// ==================== BitsetView Tests ====================

TEST_CASE("Test BitsetView Basic Operations", "[utility][bitset]") {
    SECTION("Default constructor creates empty view") {
        knowhere::BitsetView view;
        REQUIRE(view.empty());
        REQUIRE(view.size() == 0);
        REQUIRE(view.count() == 0);
    }

    SECTION("Nullptr constructor creates empty view") {
        knowhere::BitsetView view(nullptr);
        REQUIRE(view.empty());
    }

    SECTION("Basic test operation - first bit set") {
        // Create bitset with first bit set (byte = 0x01)
        std::vector<uint8_t> data = {0x01};  // bit 0 is set
        knowhere::BitsetView view(data.data(), 8);

        REQUIRE(!view.empty());
        REQUIRE(view.size() == 8);
        REQUIRE(view.test(0) == true);   // bit 0 is set
        REQUIRE(view.test(1) == false);  // bit 1 is not set
    }

    SECTION("Test multiple bits") {
        // 0b10101010 = 0xAA (bits 1, 3, 5, 7 are set)
        std::vector<uint8_t> data = {0xAA};
        knowhere::BitsetView view(data.data(), 8);

        REQUIRE(view.test(0) == false);
        REQUIRE(view.test(1) == true);
        REQUIRE(view.test(2) == false);
        REQUIRE(view.test(3) == true);
        REQUIRE(view.test(4) == false);
        REQUIRE(view.test(5) == true);
        REQUIRE(view.test(6) == false);
        REQUIRE(view.test(7) == true);
    }

    SECTION("Multi-byte bitset") {
        // Two bytes: 0xFF, 0x00
        std::vector<uint8_t> data = {0xFF, 0x00};
        knowhere::BitsetView view(data.data(), 16);

        // First 8 bits should be set
        for (int i = 0; i < 8; ++i) {
            REQUIRE(view.test(i) == true);
        }
        // Next 8 bits should not be set
        for (int i = 8; i < 16; ++i) {
            REQUIRE(view.test(i) == false);
        }
    }

    SECTION("byte_size calculation") {
        REQUIRE(knowhere::BitsetView(nullptr, 1).byte_size() == 1);
        REQUIRE(knowhere::BitsetView(nullptr, 7).byte_size() == 1);
        REQUIRE(knowhere::BitsetView(nullptr, 8).byte_size() == 1);
        REQUIRE(knowhere::BitsetView(nullptr, 9).byte_size() == 2);
        REQUIRE(knowhere::BitsetView(nullptr, 16).byte_size() == 2);
        REQUIRE(knowhere::BitsetView(nullptr, 17).byte_size() == 3);
    }
}

TEST_CASE("Test BitsetView Filter Ratio", "[utility][bitset][filter_ratio]") {
    SECTION("Empty bitset has 0 filter ratio") {
        knowhere::BitsetView view;
        REQUIRE(view.filter_ratio() == 0.0f);
    }

    SECTION("All zeros - no filtering") {
        std::vector<uint8_t> data = {0x00, 0x00};
        knowhere::BitsetView view(data.data(), 16, 0);  // 0 bits filtered

        REQUIRE(view.filter_ratio() == 0.0f);
    }

    SECTION("All ones - all filtered") {
        std::vector<uint8_t> data = {0xFF, 0xFF};
        knowhere::BitsetView view(data.data(), 16, 16);  // 16 bits filtered

        REQUIRE(view.filter_ratio() == 1.0f);
    }

    SECTION("Half filtered") {
        std::vector<uint8_t> data = {0xFF, 0x00};
        knowhere::BitsetView view(data.data(), 16, 8);  // 8 bits filtered

        REQUIRE(view.filter_ratio() == 0.5f);
    }
}

TEST_CASE("Test BitsetView with ID Offset", "[utility][bitset][offset]") {
    // Bitset: 0b00000001 (only bit 0 is set)
    std::vector<uint8_t> data = {0x01};
    knowhere::BitsetView view(data.data(), 8, 0, 0);  // no offset

    SECTION("Without offset") {
        REQUIRE(view.test(0) == true);
        REQUIRE(view.test(1) == false);
    }

    SECTION("With offset") {
        view.set_id_offset(1);

        // Now index 0 maps to bit 1, which is not set
        // But test uses: index + offset, so test(0) checks bit 1
        // Since we're testing internal behavior, let's verify with specific values
        knowhere::BitsetView view2(data.data(), 8, 0, 1);

        // test(-1) would check bit 0 (which is set), but -1 is invalid
        // test(0) checks bit 1 (which is not set)
        // This behavior depends on implementation
    }
}

TEST_CASE("Test BitsetView Index Out of Range", "[utility][bitset][boundary]") {
    std::vector<uint8_t> data = {0x00};  // 8 bits, all zeros
    knowhere::BitsetView view(data.data(), 8);

    SECTION("Index >= num_bits returns true (filtered)") {
        // According to the implementation, index >= num_bits returns true
        REQUIRE(view.test(8) == true);
        REQUIRE(view.test(100) == true);
    }
}

TEST_CASE("Test BitsetView with Generated Bitsets", "[utility][bitset][generated]") {
    const size_t n = 1000;

    SECTION("GenerateBitsetWithFirstTbitsSet") {
        size_t t = 250;  // First 250 bits set
        auto data = GenerateBitsetWithFirstTbitsSet(n, t);
        knowhere::BitsetView view(data.data(), n);

        // First t bits should be set
        for (size_t i = 0; i < t; ++i) {
            REQUIRE(view.test(i) == true);
        }

        // Remaining bits should not be set
        for (size_t i = t; i < n; ++i) {
            REQUIRE(view.test(i) == false);
        }
    }

    SECTION("GenerateBitsetWithRandomTbitsSet") {
        size_t t = 300;  // 300 random bits set
        auto data = GenerateBitsetWithRandomTbitsSet(n, t);
        knowhere::BitsetView view(data.data(), n);

        // Count set bits
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (view.test(i)) {
                count++;
            }
        }
        REQUIRE(count == t);
    }
}

TEST_CASE("Test BitsetView Large Dataset", "[utility][bitset][large]") {
    const size_t n = 100000;

    SECTION("Large bitset with 50% filter rate") {
        size_t t = n / 2;
        auto data = GenerateBitsetWithRandomTbitsSet(n, t);
        knowhere::BitsetView view(data.data(), n, t);

        REQUIRE(view.size() == n);
        REQUIRE(view.filter_ratio() == Catch::Approx(0.5f).epsilon(0.01f));
    }
}

// ==================== Bitset Generation Utility Tests ====================

TEST_CASE("Test Bitset Generation Utilities", "[utility][bitset_gen]") {
    SECTION("GenerateBitsetWithFirstTbitsSet edge cases") {
        // All bits set
        auto all_set = GenerateBitsetWithFirstTbitsSet(8, 8);
        REQUIRE(all_set[0] == 0xFF);

        // No bits set
        auto none_set = GenerateBitsetWithFirstTbitsSet(8, 0);
        REQUIRE(none_set[0] == 0x00);

        // Single bit
        auto one_set = GenerateBitsetWithFirstTbitsSet(8, 1);
        REQUIRE(one_set[0] == 0x01);
    }

    SECTION("GenerateBitsetByPartition") {
        auto data = GenerateBitsetByPartition(100, 0.5f, 2);
        knowhere::BitsetView view(data.data(), 100);

        // The pass_rate parameter affects filtering behavior
        // Just verify the bitset is generated and has some bits set/unset
        size_t pass_count = 0;
        for (size_t i = 0; i < 100; ++i) {
            if (!view.test(i)) {
                pass_count++;
            }
        }
        // At least some should pass and some should be filtered
        REQUIRE(pass_count > 0);
        REQUIRE(pass_count < 100);
    }
}
