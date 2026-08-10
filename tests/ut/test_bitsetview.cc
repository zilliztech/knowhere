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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
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

knowhere::IdArray
MakeIdArray(const std::vector<int32_t>& ids, knowhere::IdArray::Type type = knowhere::IdArray::Type::ARRAY) {
    knowhere::IdArray array;
    array.SetType(type);
    if (!ids.empty()) {
        if (type == knowhere::IdArray::Type::ARRAY) {
            array.Set(ids.data(), ids.size());
        } else {
            array.Append(ids.data(), ids.size());
        }
    }
    return array;
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

TEST_CASE("BitsetView test filters mapped ids outside out id view", "[bitset][id_map]") {
    auto bits = MakeBitmap(4, {2});
    const std::vector<int32_t> out_ids = {0, 2, 4};
    auto out_id_array = MakeIdArray(out_ids);

    knowhere::BitsetView bitset(bits.data(), 4);
    bitset.set_out_ids(out_id_array, out_ids.size());

    REQUIRE_FALSE(bitset.test(0));
    REQUIRE(bitset.test(1));
    REQUIRE(bitset.test(2));
    REQUIRE(bitset.test(3));
    REQUIRE(bitset.test(-1));
}

TEST_CASE("BitsetView helpers respect mapped out ids", "[bitset][id_map]") {
    auto bits = MakeBitmap(10, {2, 6});
    const std::vector<int32_t> out_ids = {6, 2, 8, 4};
    auto out_id_array = MakeIdArray(out_ids);

    knowhere::BitsetView bitset(bits.data(), 10);
    bitset.set_out_ids(out_id_array, out_ids.size());
    bitset.set_vector_count(out_ids.size());
    bitset.set_filter_count(2);

    REQUIRE(bitset.get_out_ids().is_array());
    REQUIRE(bitset.out_ids_count() == out_ids.size());
    REQUIRE(bitset.range_all_filtered(0, 2));
    REQUIRE_FALSE(bitset.range_all_filtered(0, 3));
    REQUIRE_FALSE(bitset.range_all_filtered(2, 4));
    REQUIRE_FALSE(bitset.previous_valid_index(2).has_value());
    REQUIRE(bitset.previous_valid_index(3).value() == 2);
    REQUIRE(bitset.previous_valid_index(4).value() == 3);
    REQUIRE(bitset.get_first_valid_index() == 2);
    REQUIRE(bitset.filter_ratio() == Catch::Approx(0.5F));
}

TEST_CASE("BitsetView helpers respect append mapped out ids", "[bitset][id_map]") {
    auto bits = MakeBitmap(10, {2, 6});
    const std::vector<int32_t> values = {6, 2, 8, 4};
    auto out_id_array = MakeIdArray(values, knowhere::IdArray::Type::APPEND_ARRAY);
    knowhere::BitsetView bitset(bits.data(), 10);
    bitset.set_out_ids(out_id_array, values.size());
    bitset.set_vector_count(values.size());
    bitset.set_filter_count(2);

    REQUIRE(bitset.get_out_ids().is_append_array());
    REQUIRE(bitset.out_ids_count() == values.size());
    REQUIRE(bitset.range_all_filtered(0, 2));
    REQUIRE_FALSE(bitset.range_all_filtered(0, 3));
    REQUIRE_FALSE(bitset.range_all_filtered(2, 4));
    REQUIRE_FALSE(bitset.previous_valid_index(2).has_value());
    REQUIRE(bitset.previous_valid_index(3).value() == 2);
    REQUIRE(bitset.previous_valid_index(4).value() == 3);
    REQUIRE(bitset.get_first_valid_index() == 2);
    REQUIRE(bitset.filter_ratio() == Catch::Approx(0.5F));
}

TEST_CASE("BitsetView helpers respect mapped out id windows", "[bitset][id_map]") {
    auto bits = MakeBitmap(8, {1});
    const std::vector<int32_t> values = {0, 1, 5};
    auto out_id_array = MakeIdArray(values);

    knowhere::BitsetView bitset(bits.data(), 8);
    bitset.set_id_offset(1);
    bitset.set_out_ids(out_id_array, values.size());
    bitset.set_vector_count(2);
    bitset.set_filter_count(1);

    REQUIRE(bitset.test(0));
    REQUIRE_FALSE(bitset.test(1));
    REQUIRE(bitset.range_all_filtered(0, 1));
    REQUIRE_FALSE(bitset.range_all_filtered(0, 2));
    REQUIRE(bitset.previous_valid_index(2).value() == 1);
    REQUIRE(bitset.get_first_valid_index() == 1);
}

TEST_CASE("BitsetView helpers respect contiguous id windows", "[bitset]") {
    auto bits = MakeBitmap(8, {2});
    knowhere::BitsetView bitset(bits.data(), 8);
    bitset.set_id_offset(2);
    bitset.set_vector_count(2);
    bitset.set_filter_count(1);

    REQUIRE(bitset.test(0));
    REQUIRE_FALSE(bitset.test(1));
    REQUIRE(bitset.range_all_filtered(0, 1));
    REQUIRE_FALSE(bitset.range_all_filtered(0, 2));
    REQUIRE(bitset.previous_valid_index(2).value() == 1);
    REQUIRE(bitset.get_first_valid_index() == 1);
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

TEST_CASE("BitsetView distinguishes unknown count from known zero", "[bitset][count]") {
    auto bits = MakeBitmap(8, {1});
    knowhere::BitsetView bitset(bits.data(), 8);

    REQUIRE_FALSE(bitset.has_known_count());
    REQUIRE_FALSE(bitset.empty());
    REQUIRE_THROWS_AS(bitset.count(), std::logic_error);

    bitset.set_filter_count(0);
    REQUIRE(bitset.has_known_count());
    REQUIRE(bitset.empty());
    REQUIRE(bitset.count() == 0);
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

TEST_CASE("BitsetView counts append valid bitmap", "[bitset][count][id_map]") {
    auto bits = MakeBitmap(8, {2, 5});
    auto valid_bytes = MakeBitmap(8, {1, 2, 4, 5, 7});
    knowhere::BitmapArray valid_bitmap;
    valid_bitmap.SetType(knowhere::BitmapArray::Type::APPEND_ARRAY);
    valid_bitmap.Append(valid_bytes.data(), 8);

    knowhere::BitsetView bitset(bits.data(), 8);
    bitset.count_filtered_bits(0, 8, valid_bitmap);

    REQUIRE(bitset.size() == 5);
    REQUIRE(bitset.count() == 2);
    REQUIRE(bitset.filter_ratio() == Catch::Approx(0.4f));
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
