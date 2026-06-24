// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <cstdint>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "knowhere/bitsetview.h"

namespace {

std::vector<uint8_t>
MakeBitmap(size_t num_bits, const std::vector<int64_t>& ids) {
    std::vector<uint8_t> bitset((num_bits + 7) / 8, 0);
    for (auto id : ids) {
        bitset[static_cast<size_t>(id) >> 3] |= static_cast<uint8_t>(1U << (static_cast<size_t>(id) & 7));
    }
    return bitset;
}

}  // namespace

TEST_CASE("BitsetView test filters ids outside the visible bitset", "[bitset]") {
    auto bits = MakeBitmap(4, {1});
    knowhere::BitsetView bitset(bits.data(), 4);

    REQUIRE_FALSE(bitset.test(0));
    REQUIRE(bitset.test(1));
    REQUIRE_FALSE(bitset.test(3));
    REQUIRE(bitset.test(4));
    REQUIRE(bitset.test(6));
    REQUIRE(bitset.test(-1));
}

TEST_CASE("BitsetView test filters mapped ids outside out id snapshot", "[bitset][id_map]") {
    auto bits = MakeBitmap(4, {2});
    const std::vector<int32_t> out_ids = {0, 2, 4};

    knowhere::BitsetView bitset(bits.data(), 4);
    bitset.set_out_ids(out_ids.data(), out_ids.size());

    REQUIRE_FALSE(bitset.test(0));
    REQUIRE(bitset.test(1));
    REQUIRE(bitset.test(2));
    REQUIRE(bitset.test(3));
    REQUIRE(bitset.test(-1));
}

TEST_CASE("BitsetView counts filtered bits across bytes", "[bitset][count]") {
    auto bits = MakeBitmap(70, {0, 2, 7, 8, 31, 63, 69});
    knowhere::BitsetView bitset(bits.data(), 70);

    bitset.count_filtered_bits(0, 70);

    REQUIRE(bitset.size() == 70);
    REQUIRE(bitset.count() == 7);
    REQUIRE(bitset.filter_ratio() == Catch::Approx(0.1f));
    REQUIRE_FALSE(bitset.empty());
}

TEST_CASE("BitsetView counts only valid filtered bits with valid bitmap", "[bitset][count]") {
    auto bits = MakeBitmap(70, {0, 2, 7, 8, 31, 63, 69});
    auto valid_bitmap = MakeBitmap(70, {1, 2, 3, 7, 31, 32, 63, 68});
    knowhere::BitsetView bitset(bits.data(), 70);

    bitset.count_filtered_bits(0, 70, valid_bitmap.data());

    REQUIRE(bitset.size() == 8);
    REQUIRE(bitset.count() == 4);
    REQUIRE(bitset.filter_ratio() == Catch::Approx(0.5f));
    REQUIRE_FALSE(bitset.empty());
}

TEST_CASE("BitsetView counts filtered bits in non-byte-aligned valid range", "[bitset][count]") {
    auto bits = MakeBitmap(80, {5, 7, 8, 11, 14, 15, 16, 18, 23, 24, 25, 70});
    auto valid_bitmap = MakeBitmap(80, {4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 22, 23, 25, 26, 27});
    knowhere::BitsetView bitset(bits.data(), 80);

    bitset.count_filtered_bits(5, 21, valid_bitmap.data());

    REQUIRE(bitset.size() == 14);
    REQUIRE(bitset.count() == 9);
    REQUIRE(bitset.filter_ratio() == Catch::Approx(9.0f / 14.0f));
    REQUIRE_FALSE(bitset.empty());
}
