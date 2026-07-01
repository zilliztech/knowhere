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

}  // namespace

TEST_CASE("Status category separates input and inner errors", "[error_code]") {
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::invalid_args) == knowhere::StatusCategory::input_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::invalid_param_in_json) ==
                   knowhere::StatusCategory::input_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::faiss_inner_error) ==
                   knowhere::StatusCategory::inner_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::internal_error) ==
                   knowhere::StatusCategory::inner_error);
    STATIC_REQUIRE(knowhere::StatusCategoryOf(knowhere::Status::knowhere_inner_error) ==
                   knowhere::StatusCategory::inner_error);

    REQUIRE(knowhere::IsInputError(knowhere::Status::invalid_metric_type));
    REQUIRE_FALSE(knowhere::IsInputError(knowhere::Status::faiss_inner_error));
    REQUIRE(knowhere::IsInnerError(knowhere::Status::brute_force_inner_error));
    REQUIRE_FALSE(knowhere::IsInnerError(knowhere::Status::invalid_value_in_json));
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
    const Status input_errors[] = {
        Status::invalid_args,
        Status::invalid_param_in_json,
        Status::out_of_range_in_json,
        Status::type_conflict_in_json,
        Status::invalid_metric_type,
        Status::empty_index,
        Status::not_implemented,
        Status::index_not_trained,
        Status::index_already_trained,
        Status::invalid_value_in_json,
        Status::arithmetic_overflow,
        Status::invalid_binary_set,
        Status::invalid_instruction_set,
        Status::invalid_index_error,
        Status::invalid_cluster_error,
        Status::invalid_serialized_index_type,
    };
    for (auto s : input_errors) {
        REQUIRE(knowhere::StatusCategoryOf(s) == StatusCategory::input_error);
        REQUIRE(knowhere::IsInputError(s));
    }

    // server-side inner errors
    const Status inner_errors[] = {
        Status::faiss_inner_error,
        Status::hnsw_inner_error,
        Status::malloc_error,
        Status::diskann_inner_error,
        Status::disk_file_error,
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
        REQUIRE(knowhere::StatusCategoryOf(s) == StatusCategory::inner_error);
        REQUIRE(knowhere::IsInnerError(s));
    }

    // Regression: cardinal_inner_error used to be caught only by the removed
    // `default:` branch; it must now be classified explicitly as an inner error.
    REQUIRE(knowhere::StatusCategoryOf(Status::cardinal_inner_error) == StatusCategory::inner_error);
}
