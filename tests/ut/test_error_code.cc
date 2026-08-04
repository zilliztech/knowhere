// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
// the License for the specific language governing permissions and limitations under the License.

#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_static.h"
#include "knowhere/segcore_error_code.h"

namespace {

class ThrowingIndexNode : public knowhere::IndexNode {
 public:
    knowhere::Status
    Train(const knowhere::DataSetPtr, std::shared_ptr<knowhere::Config>, bool) override {
        throw std::runtime_error("boom train");
    }

    knowhere::Status
    Add(const knowhere::DataSetPtr, std::shared_ptr<knowhere::Config>, bool) override {
        throw std::runtime_error("boom add");
    }

    knowhere::expected<knowhere::DataSetPtr>
    Search(const knowhere::DataSetPtr, std::unique_ptr<knowhere::Config>, const knowhere::BitsetView&,
           milvus::OpContext*) const override {
        throw std::runtime_error("boom search");
    }

    knowhere::expected<knowhere::DataSetPtr>
    RangeSearch(const knowhere::DataSetPtr, std::unique_ptr<knowhere::Config>, const knowhere::BitsetView&,
                milvus::OpContext*) const override {
        throw std::runtime_error("boom range search");
    }

    knowhere::expected<knowhere::DataSetPtr>
    GetVectorByIds(const knowhere::DataSetPtr, milvus::OpContext*) const override {
        throw std::runtime_error("boom get vector");
    }

    bool
    HasRawData(const std::string&) const override {
        throw std::runtime_error("boom raw data");
    }

    knowhere::expected<knowhere::DataSetPtr>
    GetIndexMeta(std::unique_ptr<knowhere::Config>) const override {
        throw std::runtime_error("boom meta");
    }

    knowhere::Status
    Serialize(knowhere::BinarySet&) const override {
        throw std::runtime_error("boom serialize");
    }

    knowhere::Status
    Deserialize(const knowhere::BinarySet&, std::shared_ptr<knowhere::Config>) override {
        throw std::runtime_error("boom deserialize");
    }

    knowhere::Status
    DeserializeFromFile(const std::string&, std::shared_ptr<knowhere::Config>) override {
        throw std::runtime_error("boom deserialize file");
    }

    std::unique_ptr<knowhere::BaseConfig>
    CreateConfig() const override {
        return std::make_unique<knowhere::BaseConfig>();
    }

    int64_t
    Dim() const override {
        throw std::runtime_error("boom dim");
    }

    int64_t
    Size() const override {
        throw std::runtime_error("boom size");
    }

    int64_t
    Count() const override {
        throw std::runtime_error("boom count");
    }

    std::string
    Type() const override {
        throw std::runtime_error("boom type");
    }
};

knowhere::Index<knowhere::IndexNode>
CreateThrowingIndex() {
    return knowhere::Index<ThrowingIndexNode>::Create();
}

class ThrowingStaticIndexNode {
 public:
    static std::unique_ptr<knowhere::BaseConfig>
    StaticCreateConfig() {
        throw std::runtime_error("boom static create config");
    }
};

class ThrowingIterator : public knowhere::IndexIterator {
 public:
    ThrowingIterator() : knowhere::IndexIterator(/*larger_is_closer=*/false, /*use_knowhere_search_pool=*/false) {
    }

 protected:
    void
    next_batch(std::function<void(const std::vector<knowhere::DistId>&)>) override {
        throw std::runtime_error("boom next batch");
    }
};

class StatusThrowingIterator : public knowhere::IndexIterator {
 public:
    StatusThrowingIterator() : knowhere::IndexIterator(/*larger_is_closer=*/false, /*use_knowhere_search_pool=*/false) {
    }

 protected:
    void
    next_batch(std::function<void(const std::vector<knowhere::DistId>&)>) override {
        throw knowhere::StatusException(knowhere::Status::malloc_error, "boom status alloc");
    }
};

}  // namespace

TEST_CASE("Status category separates input, transient and permanent errors", "[error_code]") {
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::invalid_args) == knowhere::StatusCategory::input_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::invalid_param_in_json) ==
                   knowhere::StatusCategory::input_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::faiss_inner_error) ==
                   knowhere::StatusCategory::permanent_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::internal_error) ==
                   knowhere::StatusCategory::permanent_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::knowhere_inner_error) ==
                   knowhere::StatusCategory::permanent_error);
    // the two transients: retry / replica-reroute may succeed
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::malloc_error) ==
                   knowhere::StatusCategory::transient_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::disk_file_error) ==
                   knowhere::StatusCategory::transient_error);
    // capability / corrupt-data statuses are the server's problem, not the
    // caller's: moved out of input_error to agree with the fine mapping
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::not_implemented) ==
                   knowhere::StatusCategory::permanent_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::invalid_instruction_set) ==
                   knowhere::StatusCategory::permanent_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::invalid_serialized_index_type) ==
                   knowhere::StatusCategory::permanent_error);
    // the deprecated alias keeps compiling and keeps its meaning
    STATIC_REQUIRE(knowhere::StatusCategory::inner_error == knowhere::StatusCategory::permanent_error);

    REQUIRE(knowhere::IsInputError(knowhere::Status::invalid_metric_type));
    REQUIRE_FALSE(knowhere::IsInputError(knowhere::Status::faiss_inner_error));
    // IsInnerError matches the deprecated alias inner_error = permanent_error:
    // transient errors are not "inner"
    REQUIRE(knowhere::IsInnerError(knowhere::Status::brute_force_inner_error));
    REQUIRE_FALSE(knowhere::IsInnerError(knowhere::Status::malloc_error));
    REQUIRE_FALSE(knowhere::IsInnerError(knowhere::Status::invalid_value_in_json));
    REQUIRE(knowhere::IsTransientError(knowhere::Status::disk_file_error));
    REQUIRE_FALSE(knowhere::IsTransientError(knowhere::Status::timeout));
}

namespace {
// every Status value, for exhaustive category<->code consistency checks
constexpr knowhere::Status kAllStatuses[] = {
    knowhere::Status::success,
    knowhere::Status::invalid_args,
    knowhere::Status::invalid_param_in_json,
    knowhere::Status::out_of_range_in_json,
    knowhere::Status::type_conflict_in_json,
    knowhere::Status::invalid_metric_type,
    knowhere::Status::empty_index,
    knowhere::Status::not_implemented,
    knowhere::Status::index_not_trained,
    knowhere::Status::index_already_trained,
    knowhere::Status::faiss_inner_error,
    knowhere::Status::hnsw_inner_error,
    knowhere::Status::malloc_error,
    knowhere::Status::diskann_inner_error,
    knowhere::Status::disk_file_error,
    knowhere::Status::invalid_value_in_json,
    knowhere::Status::arithmetic_overflow,
    knowhere::Status::cuvs_inner_error,
    knowhere::Status::invalid_binary_set,
    knowhere::Status::invalid_instruction_set,
    knowhere::Status::cardinal_inner_error,
    knowhere::Status::cuda_runtime_error,
    knowhere::Status::invalid_index_error,
    knowhere::Status::invalid_cluster_error,
    knowhere::Status::cluster_inner_error,
    knowhere::Status::timeout,
    knowhere::Status::internal_error,
    knowhere::Status::invalid_serialized_index_type,
    knowhere::Status::sparse_inner_error,
    knowhere::Status::brute_force_inner_error,
    knowhere::Status::emb_list_inner_error,
    knowhere::Status::aisaq_error,
    knowhere::Status::knowhere_inner_error,
};
}  // namespace

TEST_CASE("ToSegcoreErrorCode agrees with StatusCategoryOf for every status", "[error_code]") {
    for (auto status : kAllStatuses) {
        const auto category = knowhere::StatusCategoryOf(status);
        const auto code = knowhere::ToSegcoreErrorCode(status);
        CAPTURE(static_cast<int>(status), static_cast<int>(category), static_cast<int>(code));
        switch (category) {
            case knowhere::StatusCategory::success:
                REQUIRE(code == milvus::ErrorCode::Success);
                break;
            case knowhere::StatusCategory::input_error:
                REQUIRE(code == milvus::ErrorCode::InvalidParameter);
                break;
            case knowhere::StatusCategory::transient_error:
                // must land on a code the Go-side table marks retriable
                REQUIRE((code == milvus::ErrorCode::MemAllocateFailed || code == milvus::ErrorCode::FileReadFailed));
                break;
            case knowhere::StatusCategory::permanent_error:
                REQUIRE((code == milvus::ErrorCode::Unsupported || code == milvus::ErrorCode::DataFormatBroken ||
                         code == milvus::ErrorCode::KnowhereError));
                break;
        }
    }
}

TEST_CASE("ToSegcoreErrorCode fine mapping spot checks", "[error_code]") {
    STATIC_REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::malloc_error) ==
                   milvus::ErrorCode::MemAllocateFailed);
    STATIC_REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::disk_file_error) ==
                   milvus::ErrorCode::FileReadFailed);
    STATIC_REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::not_implemented) == milvus::ErrorCode::Unsupported);
    STATIC_REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::invalid_serialized_index_type) ==
                   milvus::ErrorCode::DataFormatBroken);
    STATIC_REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::invalid_args) == milvus::ErrorCode::InvalidParameter);
    STATIC_REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::timeout) == milvus::ErrorCode::KnowhereError);
    STATIC_REQUIRE(knowhere::ToSegcoreErrorCode(knowhere::Status::success) == milvus::ErrorCode::Success);
}

TEST_CASE("Index facade APIs are noexcept and convert exceptions to error codes", "[error_code]") {
    auto index = CreateThrowingIndex();
    auto ds = std::make_shared<knowhere::DataSet>();
    knowhere::BinarySet binset;

    STATIC_REQUIRE(noexcept(index.GetVectorByIds(ds)));
    STATIC_REQUIRE(noexcept(index.Serialize(binset)));
    STATIC_REQUIRE(noexcept(index.Count()));
    STATIC_REQUIRE(noexcept(index.Type()));
    STATIC_REQUIRE(
        noexcept(knowhere::BruteForce::Search<knowhere::fp32>(ds, ds, knowhere::Json{}, knowhere::BitsetView{})));
    STATIC_REQUIRE(noexcept(knowhere::BruteForce::SearchWithBuf<knowhere::fp32>(
        ds, ds, static_cast<int64_t*>(nullptr), static_cast<float*>(nullptr), knowhere::Json{},
        knowhere::BitsetView{})));

    const auto get_vector_result = index.GetVectorByIds(ds);
    REQUIRE(get_vector_result.error() == knowhere::Status::knowhere_inner_error);
    REQUIRE(std::string(get_vector_result.what()).find("boom get vector") != std::string::npos);

    REQUIRE(index.Serialize(binset) == knowhere::Status::knowhere_inner_error);
    REQUIRE(index.Count() == 0);
    REQUIRE(index.Type().empty());
}

TEST_CASE("Iterator APIs are noexcept and convert exceptions to error codes", "[error_code]") {
    SECTION("IndexIterator converts next_batch exceptions on first access") {
        auto it = std::make_shared<ThrowingIterator>();
        // the guarded built-in implementations are noexcept at the public boundary
        STATIC_REQUIRE(noexcept(it->Next()));
        STATIC_REQUIRE(noexcept(it->HasNext()));

        const auto has_next = it->HasNext();
        REQUIRE(!has_next.has_value());
        REQUIRE(has_next.error() == knowhere::Status::knowhere_inner_error);
        REQUIRE(has_next.what().find("boom next batch") != std::string::npos);

        const auto next = it->Next();
        REQUIRE(!next.has_value());
        REQUIRE(next.error() == knowhere::Status::knowhere_inner_error);
    }

    SECTION("GuardedCall preserves the status carried by StatusException") {
        auto it = std::make_shared<StatusThrowingIterator>();

        const auto next = it->Next();
        REQUIRE(!next.has_value());
        REQUIRE(next.error() == knowhere::Status::malloc_error);
        // message must pass through unmodified (no extra prefixing)
        REQUIRE(next.what() == "boom status alloc");
    }

    SECTION("PrecomputedDistanceIterator converts deferred compute exceptions") {
        auto it = std::make_shared<knowhere::PrecomputedDistanceIterator>(
            []() -> std::vector<knowhere::DistId> { throw std::runtime_error("boom compute dist"); },
            /*larger_is_closer=*/false, /*use_knowhere_search_pool=*/false);

        const auto next = it->Next();
        REQUIRE(!next.has_value());
        REQUIRE(next.error() == knowhere::Status::knowhere_inner_error);
        REQUIRE(next.what().find("boom compute dist") != std::string::npos);
    }

    SECTION("exhausted iterator reports an error instead of throwing") {
        auto it = std::make_shared<knowhere::PrecomputedDistanceIterator>(
            []() {
                return std::vector<knowhere::DistId>{{0, 1.0f}};
            },
            /*larger_is_closer=*/false, /*use_knowhere_search_pool=*/false);

        REQUIRE(it->HasNext().value());
        const auto first = it->Next();
        REQUIRE(first.has_value());
        REQUIRE(first.value().first == 0);

        REQUIRE(!it->HasNext().value());
        const auto next = it->Next();
        REQUIRE(!next.has_value());
        REQUIRE(next.error() == knowhere::Status::knowhere_inner_error);
    }
}

TEST_CASE("Index static facade APIs handle config creation failures", "[error_code]") {
    const knowhere::IndexType index_type = "THROWING_STATIC_CONFIG";
    constexpr knowhere::IndexVersion version = 0;
    knowhere::IndexStaticFaced<knowhere::fp32>::Instance().RegisterStaticFunc<ThrowingStaticIndexNode>(index_type);

    std::string msg;
    REQUIRE(knowhere::IndexStaticFaced<knowhere::fp32>::ConfigCheck(index_type, version, knowhere::Json{}, msg) ==
            knowhere::Status::knowhere_inner_error);
    REQUIRE(msg == "failed to create config");

    auto resource = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(index_type, version, 1024, 10, 4,
                                                                                     knowhere::Json{});
    REQUIRE(resource.error() == knowhere::Status::knowhere_inner_error);
    REQUIRE(resource.what() == "failed to create config");

    REQUIRE_FALSE(knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(index_type, version, knowhere::Json{}));
}

TEST_CASE("StatusCategoryOf classifies every status without a silent default", "[error_code]") {
    using knowhere::Status;
    using knowhere::StatusCategory;

    // success
    REQUIRE(knowhere::StatusCategoryOf(Status::success) == StatusCategory::success);

    // caller-input errors
    // clang-format off
    const Status input_errors[] = {
        Status::invalid_args,
        Status::invalid_param_in_json,
        Status::out_of_range_in_json,
        Status::type_conflict_in_json,
        Status::invalid_metric_type,
        Status::empty_index,
        Status::index_not_trained,
        Status::index_already_trained,
        Status::invalid_value_in_json,
        Status::arithmetic_overflow,
        Status::invalid_binary_set,
        Status::invalid_index_error,
        Status::invalid_cluster_error,
    };
    // clang-format on
    for (auto s : input_errors) {
        REQUIRE(knowhere::StatusCategoryOf(s) == StatusCategory::input_error);
        REQUIRE(knowhere::IsInputError(s));
    }

    // capability / corrupt-data statuses: server-side permanent, not the
    // caller's fault (aligned with the fine mapping: Unsupported /
    // DataFormatBroken)
    const Status capability_and_data_errors[] = {
        Status::not_implemented,
        Status::invalid_instruction_set,
        Status::invalid_serialized_index_type,
    };
    for (auto s : capability_and_data_errors) {
        REQUIRE(knowhere::StatusCategoryOf(s) == StatusCategory::permanent_error);
        REQUIRE_FALSE(knowhere::IsInputError(s));
    }

    // transient failures: retry / replica-reroute may succeed
    const Status transient_errors[] = {
        Status::malloc_error,
        Status::disk_file_error,
    };
    for (auto s : transient_errors) {
        REQUIRE(knowhere::StatusCategoryOf(s) == StatusCategory::transient_error);
        REQUIRE(knowhere::IsTransientError(s));
        REQUIRE_FALSE(knowhere::IsInnerError(s));  // "inner" == permanent only
    }

    // server-side permanent inner errors
    const Status inner_errors[] = {
        Status::faiss_inner_error,
        Status::hnsw_inner_error,
        Status::diskann_inner_error,
        Status::cuvs_inner_error,
        Status::cardinal_inner_error,
        Status::cuda_runtime_error,
        Status::cluster_inner_error,
        Status::timeout,
        Status::internal_error,
        Status::sparse_inner_error,
        Status::brute_force_inner_error,
        Status::emb_list_inner_error,
        Status::aisaq_error,
        Status::knowhere_inner_error,
    };
    for (auto s : inner_errors) {
        REQUIRE(knowhere::StatusCategoryOf(s) == StatusCategory::permanent_error);
        REQUIRE(knowhere::IsInnerError(s));
        REQUIRE_FALSE(knowhere::IsTransientError(s));
    }

    // Regression: cardinal_inner_error used to be caught only by the removed
    // `default:` branch; it must stay explicitly classified.
    REQUIRE(knowhere::StatusCategoryOf(Status::cardinal_inner_error) == StatusCategory::permanent_error);
}
