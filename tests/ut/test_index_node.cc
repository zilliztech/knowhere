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

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/IndexFlat.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"
#include "utils.h"

using namespace knowhere;

constexpr const char* INDEX_BASE_FLAT = "BASE_FLAT";

template <typename DataType>
class BaseFlatIndexNode : public IndexNode {
 public:
    BaseFlatIndexNode(const int32_t& /*version*/, const Object& object) {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode constructor";
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::Train()";
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::Add()";
        return Status::success;
    }

    virtual expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::Search()";
        return expected<DataSetPtr>::Err(Status::not_implemented, "BaseFlatIndexNode::Search() not implemented");
    }

    virtual expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* op_context) const override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::RangeSearch()";
        return expected<DataSetPtr>::Err(Status::not_implemented, "BaseFlatIndexNode::RangeSearch() not implemented");
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool, milvus::OpContext* op_context) const override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::AnnIterator()";
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::not_implemented,
                                                                  "BaseFlatIndexNode::AnnIterator() not implemented");
    }

    virtual expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::GetVectorByIds()";
        return expected<DataSetPtr>::Err(Status::not_implemented,
                                         "BaseFlatIndexNode::GetVectorByIds() not implemented");
    }

    virtual bool
    HasRawData(const std::string& metric_type) const override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::HasRawData()";
        return true;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::GetIndexMeta()";
        return expected<DataSetPtr>::Err(Status::not_implemented, "BaseFlatIndexNode::GetIndexMeta() not implemented");
    }

    virtual Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::Serialize()";
        return Status::success;
    }

    virtual Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::Deserialize()";
        return Status::success;
    }

    virtual Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::DeserializeFromFile()";
        return Status::success;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::StaticCreateConfig()";
        return std::make_unique<BaseConfig>();
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        LOG_KNOWHERE_INFO_ << "BaseFlatIndexNode::CreateConfig()";
        return std::make_unique<BaseConfig>();
    }

    int64_t
    Dim() const override {
        return 0;
    }

    int64_t
    Size() const override {
        return 0;
    }

    int64_t
    Count() const override {
        return count_;
    }

    void
    SetCountForTest(int64_t count) {
        count_ = count;
    }

    void
    SetEmbListOffsetForTest(std::vector<size_t> offsets) {
        emb_list_offset_ = std::make_unique<EmbListOffset>(std::move(offsets));
    }

    std::string
    Type() const override {
        return INDEX_BASE_FLAT;
    }

 private:
    int64_t count_ = 0;
};

TEST_CASE("Test index node") {
    auto version = GenTestVersionList();
    DataSetPtr ds = std::make_shared<DataSet>();
    BinarySet binset;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"  // to ignore build warnings
    SECTION("Test IndexNode") {
        KNOWHERE_SIMPLE_REGISTER_GLOBAL(BASE_FLAT, BaseFlatIndexNode, fp32, knowhere::feature::FLOAT32);
        auto index = IndexFactory::Instance().Create<fp32>("BASE_FLAT", version).value();
        REQUIRE(index.Build(ds, {}) == Status::success);
        REQUIRE(index.Train(ds, {}) == Status::success);
        REQUIRE(index.Add(ds, {}) == Status::success);
        REQUIRE(index.Search(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.RangeSearch(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.AnnIterator(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.GetVectorByIds(ds).error() == Status::not_implemented);
        REQUIRE(index.HasRawData(metric::L2) == true);
        REQUIRE(index.GetIndexMeta({}).error() == Status::not_implemented);
        REQUIRE(index.Serialize(binset) == Status::success);
        REQUIRE(index.Deserialize(binset, {}) == Status::success);
        REQUIRE(index.DeserializeFromFile("", {}) == Status::success);
        REQUIRE(index.Dim() == 0);
        REQUIRE(index.Size() == 0);
        REQUIRE(index.Count() == 0);
        REQUIRE(index.Type() == INDEX_BASE_FLAT);
    }

    SECTION("Test IndexNodeDataMockWrapper") {
        KNOWHERE_MOCK_REGISTER_GLOBAL(BASE_FLAT_MOCK, BaseFlatIndexNode, fp16, knowhere::feature::FP16);
        auto index = IndexFactory::Instance().Create<fp16>("BASE_FLAT_MOCK", version).value();
        REQUIRE(index.Build(ds, {}) == Status::success);
        REQUIRE(index.Train(ds, {}) == Status::success);
        REQUIRE(index.Add(ds, {}) == Status::success);
        REQUIRE(index.Search(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.RangeSearch(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.AnnIterator(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.GetVectorByIds(ds).error() == Status::not_implemented);
        REQUIRE(index.HasRawData(metric::L2) == true);
        REQUIRE(index.GetIndexMeta({}).error() == Status::not_implemented);
        REQUIRE(index.Serialize(binset) == Status::success);
        REQUIRE(index.Deserialize(binset, {}) == Status::success);
        REQUIRE(index.DeserializeFromFile("", {}) == Status::success);
        REQUIRE(index.Dim() == 0);
        REQUIRE(index.Size() == 0);
        REQUIRE(index.Count() == 0);
        REQUIRE(index.Type() == INDEX_BASE_FLAT);
    }

    SECTION("Test IndexNodeThreadPoolWrapper") {
        KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(BASE_FLAT_THREAD, BaseFlatIndexNode, fp32, knowhere::feature::FLOAT32,
                                                  4);
        auto index = IndexFactory::Instance().Create<fp32>("BASE_FLAT_THREAD", version).value();
        REQUIRE(index.Build(ds, {}) == Status::success);
        REQUIRE(index.Train(ds, {}) == Status::success);
        REQUIRE(index.Add(ds, {}) == Status::success);
        REQUIRE(index.Search(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.RangeSearch(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.AnnIterator(ds, {}, nullptr).error() == Status::not_implemented);
        REQUIRE(index.GetVectorByIds(ds).error() == Status::not_implemented);
        REQUIRE(index.HasRawData(metric::L2) == true);
        REQUIRE(index.GetIndexMeta({}).error() == Status::not_implemented);
        REQUIRE(index.Serialize(binset) == Status::success);
        REQUIRE(index.Deserialize(binset, {}) == Status::success);
        REQUIRE(index.DeserializeFromFile("", {}) == Status::success);
        REQUIRE(index.Dim() == 0);
        REQUIRE(index.Size() == 0);
        REQUIRE(index.Count() == 0);
        REQUIRE(index.Type() == INDEX_BASE_FLAT);
    }
#pragma GCC diagnostic pop
}

TEST_CASE("Bitset size is validated in the external ID domain", "[index_node][bitset][emb_list]") {
    auto version = GenTestVersionList();
    Object object;
    auto concrete_index = Index<BaseFlatIndexNode<fp32>>::Create(version, object);
    concrete_index.Node()->SetCountForTest(1);
    concrete_index.Node()->SetEmbListOffsetForTest({0, 1, 1, 1, 1, 1});
    Index<IndexNode> index(std::move(concrete_index));

    auto dataset = std::make_shared<DataSet>();
    std::vector<uint8_t> list_bits(1, 0);
    auto list_bitset = BitsetView(list_bits.data(), 5);

    REQUIRE(index.Search(dataset, {}, list_bitset).error() == Status::not_implemented);
    REQUIRE(index.RangeSearch(dataset, {}, list_bitset).error() == Status::not_implemented);
    REQUIRE(index.AnnIterator(dataset, {}, list_bitset).error() == Status::not_implemented);

    auto oversized_bitset = BitsetView(list_bits.data(), 6);
    auto oversized_result = index.Search(dataset, {}, oversized_bitset);
    REQUIRE(oversized_result.error() == Status::invalid_args);
    REQUIRE(oversized_result.what().find("external count: 5") != std::string::npos);
}

TEST_CASE("HNSW validates embedding-list bitsets after build and load", "[index_node][bitset][emb_list][hnsw]") {
    const auto offsets = GENERATE(std::vector<size_t>{0, 0, 0, 0, 0, 1}, std::vector<size_t>{0, 4, 8});
    const auto num_lists = offsets.size() - 1;
    const auto num_vectors = offsets.back();
    CAPTURE(num_lists, num_vectors);

    constexpr int dim = 4;
    auto dataset = ::GenDataSet(num_vectors, dim);
    auto list_offsets = std::make_unique<size_t[]>(offsets.size());
    std::copy(offsets.begin(), offsets.end(), list_offsets.get());
    dataset->Set(meta::EMB_LIST_OFFSET, static_cast<const size_t*>(list_offsets.release()));

    auto query = ::GenDataSet(1, dim);
    auto query_offsets = std::make_unique<size_t[]>(2);
    query_offsets[0] = 0;
    query_offsets[1] = 1;
    query->Set(meta::EMB_LIST_OFFSET, static_cast<const size_t*>(query_offsets.release()));

    const Json config = {
        {meta::METRIC_TYPE, "MAX_SIM_IP"}, {meta::DIM, dim},   {meta::TOPK, 1}, {indexparam::HNSW_M, 4},
        {indexparam::EFCONSTRUCTION, 8},   {indexparam::EF, 8}};
    const auto version = Version::GetCurrentVersion().VersionNumber();
    auto index = IndexFactory::Instance().Create<fp32>(IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(index.Build(dataset, config, false) == Status::success);

    const auto verify = [&](const Index<IndexNode>& candidate) {
        REQUIRE(candidate.Count() == static_cast<int64_t>(num_vectors));
        REQUIRE(candidate.Node()->ExternalCount() == static_cast<int64_t>(num_lists));

        std::vector<uint8_t> bits((num_lists + 1 + 7) / 8, 0);
        auto result = candidate.Search(query, config, BitsetView(bits.data(), num_lists));
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] >= 0);
        REQUIRE(result.value()->GetIds()[0] < static_cast<int64_t>(num_lists));

        const auto oversized_bitset = BitsetView(bits.data(), num_lists + 1);
        REQUIRE(candidate.Search(query, config, oversized_bitset).error() == Status::invalid_args);
        REQUIRE(candidate.RangeSearch(query, config, oversized_bitset).error() == Status::invalid_args);
        REQUIRE(candidate.AnnIterator(query, config, oversized_bitset).error() == Status::invalid_args);
    };
    verify(index);

    BinarySet binary_set;
    REQUIRE(index.Serialize(binary_set) == Status::success);
    auto loaded_index = IndexFactory::Instance().Create<fp32>(IndexEnum::INDEX_HNSW, version).value();
    REQUIRE(loaded_index.Deserialize(binary_set, config) == Status::success);
    verify(loaded_index);
}
