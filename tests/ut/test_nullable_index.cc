// Copyright (C) 2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/index/index.h"

namespace {

using namespace knowhere;

template <typename T>
concept HasIdMapGetter = requires(T& index) {
    index.GetIdMap();
};

template <typename T>
concept HasIdMapSetter = requires(T& index) {
    index.SetIdMapType({});
};

static_assert(!HasIdMapGetter<Index<IndexNode>>);
static_assert(!HasIdMapSetter<Index<IndexNode>>);

// Records backend calls and supplies errors for nullable Index API tests.
class NullableFakeIndexNode : public IndexNode {
 public:
    Status load_status = Status::success;
    Status finalize_status = Status::success;
    Status add_status = Status::success;
    int trains = 0;
    int adds = 0;
    int loads = 0;
    int file_loads = 0;
    int finalizes = 0;
    int64_t payload_count = 0;
    bool throw_on_load = false;
    bool throw_on_count = false;
    bool exact_filter_count = false;

    Status
    Train(const DataSetPtr, std::shared_ptr<Config>, bool) override {
        ++trains;
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config>, bool) override {
        ++adds;
        if (add_status == Status::success) {
            payload_count += dataset->GetRows();
        }
        return add_status;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr, std::unique_ptr<Config>, const BitsetView&, milvus::OpContext*) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "Search not implemented");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr, milvus::OpContext*) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
    }

    bool
    HasRawData(const std::string&) const override {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config>) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet&) const override {
        return Status::success;
    }

    Status
    Deserialize(const BinarySet&, std::shared_ptr<Config>) override {
        ++loads;
        if (throw_on_load) {
            throw std::bad_alloc();
        }
        return load_status;
    }

    Status
    DeserializeFromFile(const std::string&, std::shared_ptr<Config>) override {
        ++file_loads;
        return load_status;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<BaseConfig>();
    }

    int64_t
    Dim() const override {
        return 1;
    }

    int64_t
    Size() const override {
        return 0;
    }

    int64_t
    Count() const override {
        if (throw_on_count) {
            throw std::logic_error("backend count is unavailable before load");
        }
        return payload_count;
    }

    std::string
    Type() const override {
        return "NULLABLE_FAKE_INDEX";
    }

    bool
    NeedBitsetExactCount() const override {
        return exact_filter_count;
    }

    void
    CountFilterRange(BitsetView& bitset, size_t offset, size_t count) const {
        CalcBitsetCount(bitset, GetIdMap(), offset, count);
    }

    Status
    FinalizeIdMap() override {
        ++finalizes;
        return finalize_status == Status::success ? IndexNode::FinalizeIdMap() : finalize_status;
    }
};

Index<IndexNode>
CreateNullableTestIndex() {
    return Index<NullableFakeIndexNode>::Create();
}

DataSetPtr
GenNullableDataSet(int64_t payloads, const uint8_t* bits = nullptr, size_t rows = 0) {
    auto dataset = GenDataSet(payloads, 1, nullptr);
    if (bits != nullptr) {
        dataset->SetIdMapData(IdMapData::FromValidBitmap(bits, rows));
    }
    return dataset;
}

const Json config{{"metric_type", "L2"}};

}  // namespace

TEST_CASE("Nullable Index validity queries preserve identity mapping", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    REQUIRE(index.Build(GenNullableDataSet(2), config, false) == Status::success);
    REQUIRE(index.Count() == 2);
    REQUIRE(index.LogicalCount().value() == 0);
    REQUIRE(index.ValidCount().value() == 0);
    REQUIRE_FALSE(index.HasValidity().value());
    REQUIRE(index.IsRowValid(1).value());
    REQUIRE_FALSE(index.IsRowValid(-1).value());
    // No nullable map means identity validity, not an MVCC/count bound.
    REQUIRE(index.IsRowValid(100).value());
    uint8_t bits = 0xff;
    REQUIRE(index.IntersectValidity(-1, 5, {&bits, 1, 1}) == Status::success);
    REQUIRE(bits == 0xfd);
}

TEST_CASE("Nullable Index delegates Build Add Train and reload to the backend", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    auto* fake_node = static_cast<NullableFakeIndexNode*>(index.Node());
    REQUIRE(index.Build(GenNullableDataSet(2), config, false) == Status::success);
    REQUIRE(index.Add(GenNullableDataSet(1), config, false) == Status::success);
    REQUIRE(fake_node->adds == 2);
    REQUIRE(index.Count() == 3);
    REQUIRE(index.Train(GenNullableDataSet(17), config, false) == Status::success);
    REQUIRE(fake_node->trains == 2);
    REQUIRE(index.Build(GenNullableDataSet(1), config, false) == Status::success);
    REQUIRE(fake_node->trains == 3);
    REQUIRE(index.Deserialize(BinarySet{}) == Status::success);
    REQUIRE(index.Deserialize(BinarySet{}, {}) == Status::success);
    REQUIRE(fake_node->loads == 2);
}

TEST_CASE("Nullable Index consumes validity on first Add after Train", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    REQUIRE(index.Train(GenNullableDataSet(17), config, false) == Status::success);
    REQUIRE(index.Count() == 0);
    const uint8_t bits = 5;
    REQUIRE(index.Add(GenNullableDataSet(2, &bits, 3), config, false) == Status::success);
    REQUIRE(index.Add(GenNullableDataSet(2, &bits, 3), config, false) == Status::success);
    REQUIRE(index.LogicalCount().value() == 6);
    REQUIRE(index.ValidCount().value() == 4);
}

TEST_CASE("Nullable Index Add propagates backend errors", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    auto* fake_node = static_cast<NullableFakeIndexNode*>(index.Node());
    const uint8_t bits = 5;
    BuildOptions build_options;
    build_options.growing = true;
    REQUIRE(index.Build(GenNullableDataSet(2, &bits, 3), config, false, build_options) == Status::success);
    fake_node->add_status = Status::faiss_inner_error;
    REQUIRE(index.Add(GenNullableDataSet(2, &bits, 3), config, false) == Status::faiss_inner_error);
    REQUIRE(fake_node->adds == 2);
    // Mapping is appended before backend Add, so a backend error does not undo it.
    REQUIRE(index.LogicalCount().value() == 6);
    REQUIRE(index.Count() == 2);
}

TEST_CASE("Nullable Index retains the caller's query boundary", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    REQUIRE(index.Build(GenNullableDataSet(2), config, false) == Status::success);
    auto unfiltered = index.Node()->PrepareBitset(BitsetView{});
    REQUIRE(unfiltered.has_value());
    REQUIRE(unfiltered.value().bitset.num_bits() == 0);
    REQUIRE(unfiltered.value().bitset.data() == nullptr);

    const uint8_t bits = 0;
    BitsetView input(&bits, 2, 0);
    input.set_require_id_boundary(true);
    auto prepared = index.Node()->PrepareBitset(input);
    REQUIRE(prepared.has_value());
    REQUIRE_FALSE(prepared.value().bitset.empty());
    REQUIRE(index.Add(GenNullableDataSet(1), config, false) == Status::success);
    REQUIRE(prepared.value().bitset.num_bits() == 2);
    REQUIRE(prepared.value().bitset.test(2));
}

TEST_CASE("Nullable Index short validity preserves query counts and NULL tails", "[nullable][api]") {
    for (const size_t extent : {1, 7, 8, 9, 63, 64, 65}) {
        const size_t rows = extent + 129;
        std::vector<uint8_t> short_bits((extent + 7) / 8, 0);
        std::vector<uint8_t> full_bits((rows + 7) / 8, 0);
        size_t valid_count = 0;
        for (size_t row = 0; row < extent; ++row) {
            if (row % 3 == 0 || row + 1 == extent) {
                short_bits[row / 8] |= uint8_t{1} << (row % 8);
                full_bits[row / 8] |= uint8_t{1} << (row % 8);
                ++valid_count;
            }
        }
        auto compact = CreateNullableTestIndex();
        auto complete = CreateNullableTestIndex();
        REQUIRE(compact.Build(GenNullableDataSet(valid_count, short_bits.data(), extent), config, false) ==
                Status::success);
        REQUIRE(complete.Build(GenNullableDataSet(valid_count, full_bits.data(), rows), config, false) ==
                Status::success);
        auto* compact_node = static_cast<NullableFakeIndexNode*>(compact.Node());
        auto* complete_node = static_cast<NullableFakeIndexNode*>(complete.Node());
        compact_node->exact_filter_count = complete_node->exact_filter_count = true;
        REQUIRE(compact.LogicalCount().value() == extent);
        REQUIRE(compact.ValidCount().value() == complete.ValidCount().value());
        for (size_t row = 0; row < rows; ++row) {
            REQUIRE(compact.IsRowValid(row).value() == complete.IsRowValid(row).value());
        }
        std::vector<uint8_t> actual((rows + 7) / 8, 0xff);
        REQUIRE(compact.IntersectValidity(0, rows, {actual.data(), actual.size(), 0}) == Status::success);
        for (size_t row = 0; row < rows; ++row) {
            REQUIRE(((actual[row / 8] >> (row % 8)) & 1) == ((full_bits[row / 8] >> (row % 8)) & 1));
        }
        const int64_t offsets[] = {-1, 0, static_cast<int64_t>(extent - 1), static_cast<int64_t>(extent),
                                   static_cast<int64_t>(rows - 1)};
        uint8_t selected = 0xff;
        REQUIRE(compact.IntersectValidityAt(offsets, 5, {&selected, 1, 0}) == Status::success);
        REQUIRE((selected & 0x1f) == 6);
        REQUIRE(compact.Node()->PrepareBitset(BitsetView{}).has_value());
        for (const size_t query_rows : {size_t{1}, extent, extent + 1, rows}) {
            for (const uint8_t pattern : {0x00, 0xff, 0x55}) {
                CAPTURE(extent, query_rows, pattern);
                std::vector<uint8_t> filter((query_rows + 7) / 8, pattern);
                const BitsetView input(filter.data(), query_rows);
                auto a = compact.Node()->PrepareBitset(input);
                auto b = complete.Node()->PrepareBitset(input);
                REQUIRE(a.has_value());
                REQUIRE(b.has_value());
                REQUIRE(a.value().bitset.num_bits() == query_rows);
                REQUIRE(a.value().bitset.data() == filter.data());
                REQUIRE(a.value().bitset.size() == valid_count);
                REQUIRE(a.value().bitset.count() == b.value().bitset.count());
                for (const size_t offset : {size_t{0}, extent - 1, extent, rows - 1}) {
                    auto partial_a = input;
                    auto partial_b = input;
                    compact_node->CountFilterRange(partial_a, offset, 65);
                    complete_node->CountFilterRange(partial_b, offset, 65);
                    REQUIRE(partial_a.size() == partial_b.size());
                    REQUIRE(partial_a.count() == partial_b.count());
                }
            }
        }
    }
}

TEST_CASE("Nullable growing Index retains exact append positions through NULL tails", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    const uint8_t first[8] = {1};
    const uint8_t null_batch[9] = {};
    const uint8_t last = 1;
    BuildOptions options;
    options.growing = true;
    REQUIRE(index.Build(GenNullableDataSet(1, first, 64), config, false, options) == Status::success);
    REQUIRE(index.Add(GenNullableDataSet(0, null_batch, 65), config, false) == Status::success);
    REQUIRE(index.LogicalCount().value() == 129);
    REQUIRE(index.Add(GenNullableDataSet(1, &last, 1), config, false) == Status::success);
    REQUIRE(index.LogicalCount().value() == 130);
    REQUIRE(index.ValidCount().value() == 2);
    REQUIRE(index.Node()->GetIdMap().MapInToOut(1) == 129);
    REQUIRE_FALSE(index.IsRowValid(128).value());
    REQUIRE(index.IsRowValid(129).value());
}

TEST_CASE("Nullable Index all-null Build skips backend Train and Add", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    auto* fake_node = static_cast<NullableFakeIndexNode*>(index.Node());
    const uint8_t nulls = 0;
    REQUIRE(index.Build(GenNullableDataSet(0, &nulls, 3), config, false) == Status::success);
    REQUIRE(fake_node->trains == 0);
    REQUIRE(fake_node->adds == 0);
    REQUIRE(index.LogicalCount().value() == 3);
    REQUIRE(index.Count() == 0);
}

TEST_CASE("Nullable Index imports validity without loading the backend", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    auto* fake_node = static_cast<NullableFakeIndexNode*>(index.Node());
    uint8_t bits = 5;
    REQUIRE(index.ImportValidity({&bits, 1, 3}) == Status::success);
    REQUIRE(fake_node->trains == 0);
    REQUIRE(fake_node->adds == 0);
    REQUIRE(fake_node->loads == 0);
    REQUIRE(fake_node->finalizes == 1);
    REQUIRE(index.LogicalCount().value() == 3);
    REQUIRE(index.ValidCount().value() == 2);
    REQUIRE(index.Count() == 0);
    REQUIRE(index.Dim() == 1);  // backend dimension is not replaced by sidecar metadata
    auto exported = index.ExportValidity().value().value();
    bits = 0;
    REQUIRE(exported.packed_bits[0] == 5);
    REQUIRE(index.IsRowValid(2).value());
    REQUIRE_FALSE(index.IsRowValid(1).value());
}

TEST_CASE("Nullable Index validity handles bit windows and unordered duplicate IDs", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    const uint8_t bits = 5;
    REQUIRE(index.Build(GenNullableDataSet(2, &bits, 3), config, false) == Status::success);
    const int64_t ids[] = {2, 1, 0, 2, -1, 3};
    uint8_t target = 0xff;
    REQUIRE(index.IntersectValidityAt(ids, 6, {&target, 1, 1}) == Status::success);
    REQUIRE(target == 0x9b);
    REQUIRE(index.IntersectValidityAt(nullptr, 1, {&target, 1, 0}) == Status::invalid_args);
    REQUIRE(index.IntersectValidity(0, 2, {&target, 1, 7}) == Status::invalid_args);
    REQUIRE(target == 0x9b);
    REQUIRE(index.IntersectValidity(0, 0, {}) == Status::success);
}

TEST_CASE("Nullable Index load preserves backend errors and finalization order", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    auto* fake_node = static_cast<NullableFakeIndexNode*>(index.Node());
    const uint8_t bits = 5;
    LoadOptions input;
    input.validity = ValidityView{&bits, 1, 3};
    fake_node->payload_count = 2;
    SECTION("success finalizes exactly once") {
        REQUIRE(index.Deserialize(BinarySet{}, {}, input) == Status::success);
        REQUIRE(fake_node->loads == 1);
        REQUIRE(fake_node->finalizes == 1);
        REQUIRE(index.LogicalCount().value() == 3);
    }
    SECTION("backend failure does not finalize") {
        fake_node->load_status = Status::faiss_inner_error;
        REQUIRE(index.Deserialize(BinarySet{}, {}, input) == Status::faiss_inner_error);
        REQUIRE(fake_node->loads == 1);
        REQUIRE(fake_node->finalizes == 0);
    }
    SECTION("backend allocation exceptions retain the status mapping") {
        fake_node->throw_on_load = true;
        REQUIRE(index.Deserialize(BinarySet{}, {}, input) == Status::malloc_error);
        REQUIRE(fake_node->loads == 1);
        REQUIRE(fake_node->finalizes == 0);
    }
    SECTION("file load finalizes the same mapping") {
        REQUIRE(index.DeserializeFromFile("unused-index-path", {}, input) == Status::success);
        REQUIRE(fake_node->file_loads == 1);
        REQUIRE(fake_node->finalizes == 1);
        REQUIRE(index.LogicalCount().value() == 3);
        REQUIRE(index.ValidCount().value() == 2);
    }
    SECTION("finalize failure propagates") {
        fake_node->finalize_status = Status::faiss_inner_error;
        REQUIRE(index.Deserialize(BinarySet{}, {}, input) == Status::faiss_inner_error);
        REQUIRE(fake_node->finalizes == 1);
    }
    SECTION("invalid input does not reach backend") {
        input.validity->size_bytes = 0;
        REQUIRE(index.Deserialize(BinarySet{}, {}, input) == Status::invalid_args);
        REQUIRE(fake_node->loads == 0);
        REQUIRE(fake_node->finalizes == 0);
    }
}

TEST_CASE("Nullable Index validity queries do not inspect an unloaded backend", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    static_cast<NullableFakeIndexNode*>(index.Node())->throw_on_count = true;
    auto exported = index.ExportValidity();
    REQUIRE(exported.has_value());
    REQUIRE_FALSE(exported.value().has_value());
    REQUIRE(index.IsRowValid(17).value());
    REQUIRE_FALSE(index.HasValidity().value());
    REQUIRE(index.ValidCount().value() == 0);
    REQUIRE(index.LogicalCount().value() == 0);
    const uint8_t bits = 5;
    REQUIRE(index.ImportValidity({&bits, 1, 3}) == Status::success);
    REQUIRE(index.HasValidity().value());
    REQUIRE(index.ValidCount().value() == 2);
    REQUIRE(index.LogicalCount().value() == 3);
    REQUIRE(index.ExportValidity().value()->packed_bits[0] == 5);
}

TEST_CASE("Load inspection preserves the default bitmap path", "[nullable][api]") {
    auto index = CreateNullableTestIndex();
    const auto requirements = index.InspectLoadRequirements({}, config);
    REQUIRE(requirements.has_value());
    REQUIRE(requirements.value().needs_external_validity);
    const auto* node = static_cast<NullableFakeIndexNode*>(index.Node());
    REQUIRE(node->loads == 0);
    REQUIRE(node->finalizes == 0);
    REQUIRE_FALSE(Index<IndexNode>{}.InspectLoadRequirements({}, config).has_value());
}
