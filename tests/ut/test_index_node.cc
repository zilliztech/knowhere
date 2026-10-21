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
#include <string>
#include <unordered_set>
#include <utility>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/IndexFlat.h"
#include "knowhere/build_scheduler.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"
#include "knowhere/index/mrl_index_node.h"
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
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool,
          milvus::OpContext* op_context) override {
        build_context = op_context;
        build_dataset = dataset;
        build_config = cfg;
        return IndexNode::Build(dataset, std::move(cfg), use_knowhere_build_pool, op_context);
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
        return 0;
    }

    std::string
    Type() const override {
        return INDEX_BASE_FLAT;
    }

    milvus::OpContext* build_context = nullptr;
    DataSetPtr build_dataset;
    std::shared_ptr<Config> build_config;
};

class ReservingFlatIndexNode : public BaseFlatIndexNode<fp32> {
 public:
    using BaseFlatIndexNode<fp32>::BaseFlatIndexNode;

    BuildInputOptions
    GetBuildInputOptions() const override {
        return {.use_memory = true};
    }

    expected<bool>
    TryReserveBuild(const DataSetPtr dataset, std::shared_ptr<Config> cfg) override {
        reservation_dataset = dataset;
        reservation_config = std::move(cfg);
        ++reservation_calls;
        return std::as_const(reservation_result);
    }

    expected<bool> reservation_result = false;
    DataSetPtr reservation_dataset;
    std::shared_ptr<Config> reservation_config;
    int reservation_calls = 0;
};

class DefaultBuildIndexNode : public BaseFlatIndexNode<fp32> {
 public:
    using BaseFlatIndexNode<fp32>::BaseFlatIndexNode;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        reservation_held_during_build = !BuildScheduler::Instance().TryReserve();
        return BaseFlatIndexNode<fp32>::Train(dataset, std::move(cfg), use_knowhere_build_pool);
    }

    bool reservation_held_during_build = false;
};

TEST_CASE("Default build reservations follow the build lifetime", "[build_admission]") {
    const std::string route = GENERATE("direct", "data-mock", "mrl", "thread-pool");
    CAPTURE(route);
    const auto version = Version::GetCurrentVersion().VersionNumber();
    auto& scheduler = BuildScheduler::Instance();
    {
        DefaultBuildIndexNode* node = nullptr;
        auto index = [&]() -> Index<IndexNode> {
            if (route == "data-mock" || route == "thread-pool") {
                auto base_node = std::make_unique<DefaultBuildIndexNode>(version, nullptr);
                node = base_node.get();
                if (route == "data-mock") {
                    return Index<IndexNodeDataMockWrapper<fp16>>::Create(std::move(base_node));
                }
                return Index<IndexNodeThreadPoolWrapper>::Create(std::move(base_node), size_t{1});
            }
            auto base = Index<DefaultBuildIndexNode>::Create(version, nullptr);
            node = base.Node();
            if (route == "mrl") {
                return CreateMRLIndex(std::move(base), 4, 2, DataFormatEnum::fp32, false, {});
            }
            return base;
        }();
        float values[8] = {};
        fp16 mock_values[8] = {};
        auto dataset = GenDataSet(
            2, 4, route == "data-mock" ? static_cast<const void*>(mock_values) : static_cast<const void*>(values));
        Json config = {{meta::DIM, 4}, {meta::METRIC_TYPE, metric::L2}};
        REQUIRE(index.TryReserveBuild(dataset, config).value());
        REQUIRE(index.TryReserveBuild(dataset, config).value());
        REQUIRE_FALSE(scheduler.TryReserve());

        SECTION("Synchronous completion releases before index destruction") {
            REQUIRE(index.Build(dataset, config) == Status::success);
            REQUIRE(node->reservation_held_during_build);
            REQUIRE(scheduler.TryReserve().has_value());
        }
        SECTION("Invalid configuration releases before index destruction") {
            config[meta::DIM] = "invalid";
            REQUIRE(index.Build(dataset, config) != Status::success);
            REQUIRE(scheduler.TryReserve().has_value());
        }
        SECTION("Asynchronous completion releases before index destruction") {
#ifdef KNOWHERE_WITH_CARDINAL
            auto task = index.BuildAsync(dataset, config, std::chrono::seconds(30));
#else
            auto task = index.BuildAsync(dataset, config, true);
#endif
            REQUIRE(task != nullptr);
            REQUIRE(task->Get() == Status::success);
            REQUIRE(node->reservation_held_during_build);
            REQUIRE(scheduler.TryReserve().has_value());
        }
        SECTION("Cancellation before Build releases on destruction") {
            REQUIRE_FALSE(scheduler.TryReserve());
        }
    }
    REQUIRE(scheduler.TryReserve().has_value());
}

TEST_CASE("Build input options default to files and pass through wrappers", "[build_input]") {
    const auto version = Version::GetCurrentVersion().VersionNumber();
    const auto defaults = Index<BaseFlatIndexNode<fp32>>::Create(version, nullptr);
    REQUIRE_FALSE(defaults.GetBuildInputOptions().use_memory);

    auto direct = Index<ReservingFlatIndexNode>::Create(version, nullptr);
    REQUIRE(direct.GetBuildInputOptions().use_memory);
    auto mock =
        Index<IndexNodeDataMockWrapper<fp16>>::Create(std::make_unique<ReservingFlatIndexNode>(version, nullptr));
    REQUIRE(mock.GetBuildInputOptions().use_memory);
    auto mrl = CreateMRLIndex(std::move(mock), 4, 2, DataFormatEnum::fp16, false, {});
    REQUIRE(mrl.GetBuildInputOptions().use_memory);
    auto threaded = Index<IndexNodeThreadPoolWrapper>::Create(
        std::make_unique<ReservingFlatIndexNode>(version, nullptr), size_t{1});
    REQUIRE(threaded.GetBuildInputOptions().use_memory);
}

TEST_CASE("Build forwards OpContext to the node", "[build_context]") {
    const auto route = GENERATE("sync", "async", "emb-list", "data-mock");
    CAPTURE(route);
    const auto version = Version::GetCurrentVersion().VersionNumber();
    BaseFlatIndexNode<fp32>* node = nullptr;
    auto index = [&]() -> Index<IndexNode> {
        if (std::string(route) == "data-mock") {
            auto base = std::make_unique<BaseFlatIndexNode<fp32>>(version, nullptr);
            node = base.get();
            return Index<IndexNodeDataMockWrapper<fp16>>::Create(std::move(base));
        }
        auto base = Index<BaseFlatIndexNode<fp32>>::Create(version, nullptr);
        node = base.Node();
        return base;
    }();
    float data[8] = {};
    fp16 mock_data[8] = {};
    auto dataset = knowhere::GenDataSet(
        2, 4, std::string(route) == "data-mock" ? static_cast<const void*>(mock_data) : static_cast<const void*>(data));
    Json config = {{meta::DIM, 4}, {meta::METRIC_TYPE, metric::L2}};
    size_t offsets[] = {0, 1, 2};
    if (std::string(route) == "emb-list") {
        config[meta::METRIC_TYPE] = "MAX_SIM_L2";
        dataset->Set(meta::EMB_LIST_OFFSET, static_cast<const size_t*>(offsets));
        dataset->Set(meta::EMB_LIST_COUNT, int64_t{2});
    }
    auto context = std::make_shared<milvus::OpContext>();
    if (std::string(route) == "async") {
#ifdef KNOWHERE_WITH_CARDINAL
        auto task = index.BuildAsync(dataset, config, std::chrono::seconds(30), {}, context);
#else
        auto task = index.BuildAsync(dataset, config, true, {}, context);
#endif
        REQUIRE(task != nullptr);
        REQUIRE(task->Get() == Status::success);
    } else {
        REQUIRE(index.Build(dataset, config, false, {}, context.get()) == Status::success);
    }
    REQUIRE(node->build_context == context.get());
}

TEST_CASE("Build wrappers forward reservations to the build node", "[build_admission][mrl]") {
    const bool mock = GENERATE(false, true);
    const bool mrl = GENERATE(false, true);
    CAPTURE(mock, mrl);
    const auto version = Version::GetCurrentVersion().VersionNumber();
    ReservingFlatIndexNode* node = nullptr;
    auto index = [&]() -> Index<IndexNode> {
        if (mock) {
            auto base = std::make_unique<ReservingFlatIndexNode>(version, nullptr);
            node = base.get();
            return Index<IndexNodeDataMockWrapper<fp16>>::Create(std::move(base));
        }
        auto base = Index<ReservingFlatIndexNode>::Create(version, nullptr);
        node = base.Node();
        return base;
    }();
    if (mrl) {
        index = CreateMRLIndex(std::move(index), 4, 2, mock ? DataFormatEnum::fp16 : DataFormatEnum::fp32, false, {});
    }
    auto shape = knowhere::GenDataSet(2, 4, nullptr);
    Json config = {{meta::DIM, 4}, {meta::METRIC_TYPE, metric::L2}};
    auto admitted = index.TryReserveBuild(shape, config);
    REQUIRE(admitted.has_value());
    REQUIRE_FALSE(admitted.value());
    REQUIRE(node->reservation_calls == 1);
    REQUIRE(node->reservation_dataset->GetTensor() == nullptr);
    REQUIRE(node->reservation_dataset->GetRows() == 2);
    REQUIRE(node->reservation_dataset->GetDim() == (mrl ? 2 : 4));
    REQUIRE(static_cast<BaseConfig&>(*node->reservation_config).dim.value() == (mrl ? 2 : 4));
    REQUIRE(shape->GetDim() == 4);

    node->reservation_result = expected<bool>::Err(Status::invalid_args, "reservation rejected");
    admitted = index.TryReserveBuild(shape, config);
    REQUIRE_FALSE(admitted.has_value());
    REQUIRE(admitted.error() == Status::invalid_args);
    REQUIRE(admitted.what() == "reservation rejected");

    node->reservation_result = true;
    admitted = index.TryReserveBuild(shape, config);
    REQUIRE(admitted.has_value());
    REQUIRE(admitted.value());
    REQUIRE(node->reservation_calls == 3);
    float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    fp16 mock_data[8];
    for (size_t i = 0; i < 8; ++i) {
        mock_data[i] = fp16(data[i]);
    }
    auto dataset =
        knowhere::GenDataSet(2, 4, mock ? static_cast<const void*>(mock_data) : static_cast<const void*>(data));
    milvus::OpContext context;
    REQUIRE(index.Build(dataset, config, false, {}, &context) == Status::success);
    REQUIRE(node->build_context == &context);
    REQUIRE(node->build_dataset->GetRows() == node->reservation_dataset->GetRows());
    REQUIRE(node->build_dataset->GetDim() == node->reservation_dataset->GetDim());
    REQUIRE(static_cast<BaseConfig&>(*node->build_config).dim ==
            static_cast<BaseConfig&>(*node->reservation_config).dim);
    REQUIRE(static_cast<BaseConfig&>(*node->build_config).metric_type ==
            static_cast<BaseConfig&>(*node->reservation_config).metric_type);
    const auto* built_data = static_cast<const float*>(node->build_dataset->GetTensor());
    REQUIRE(built_data[0] == 1);
    REQUIRE(built_data[mrl ? 2 : 4] == 5);

    if (mrl) {
        auto invalid = index.TryReserveBuild(knowhere::GenDataSet(2, 3, nullptr), config);
        REQUIRE_FALSE(invalid.has_value());
        REQUIRE(invalid.error() == Status::invalid_args);
        REQUIRE(node->reservation_calls == 3);
    }
}

TEST_CASE("Test index node") {
    auto version = GenTestVersionList();
    DataSetPtr ds = std::make_shared<DataSet>();
    BinarySet binset;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"  // to ignore build warnings
    SECTION("Test IndexNode") {
        KNOWHERE_SIMPLE_REGISTER_GLOBAL(BASE_FLAT, BaseFlatIndexNode, fp32, knowhere::feature::FLOAT32);
        auto index = IndexFactory::Instance().Create<fp32>("BASE_FLAT", version).value();
        auto admitted = index.TryReserveBuild(ds, {});
        REQUIRE(admitted.has_value());
        REQUIRE(admitted.value());

        Json invalid_config = {{meta::DIM, "invalid"}};
        auto cfg = index.Node()->CreateConfig();
        std::string msg;
        const auto status = Config::FormatAndCheck(*cfg, invalid_config, &msg);
        REQUIRE(status != Status::success);
        REQUIRE_FALSE(msg.empty());
        auto rejected = index.TryReserveBuild(ds, invalid_config);
        REQUIRE_FALSE(rejected.has_value());
        REQUIRE(rejected.error() == status);
        REQUIRE(rejected.what() == msg);

        REQUIRE(index.Build(ds, {}) == Status::success);
        std::shared_ptr<Config> async_cfg = index.Node()->CreateConfig();
        REQUIRE(index.Node()->BuildAsync(ds, std::move(async_cfg)) == Status::success);
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
