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

#include <cstdint>

#include "diskann/utils.h"
#include "index/minhash/minhash_index.h"
#include "index/minhash/minhash_index_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/file_manager.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/minhash_util.h"
#include "knowhere/utils.h"

// use diskann to hack
namespace knowhere {
template <typename DataType>
class MinHashIndexNode : public IndexNode {
 public:
    using DistType = float;
    MinHashIndexNode(const int32_t& version, const Object& object) : is_loaded_(false) {
        assert(typeid(object) == typeid(Pack<std::shared_ptr<FileManager>>));
        auto disk_index_pack = dynamic_cast<const Pack<std::shared_ptr<FileManager>>*>(&object);
        assert(disk_index_pack != nullptr);
        file_manager_ = disk_index_pack->GetPack();
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
    }

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "index not_implemented ");
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        return false;
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "index not_implemented ");
    }

    Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_INFO_ << "minhash index does nothing for serialize";
        return Status::success;
    }

    static expected<Resource>
    StaticEstimateLoadResource(const float file_size, const knowhere::BaseConfig& config, const IndexVersion& version) {
        return Resource{file_size * 0.25f, file_size};
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override;

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_ERROR_ << "minhash index doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<MinHashConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    Status
    SetFileManager(std::shared_ptr<FileManager> file_manager) {
        if (file_manager == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Malloc error, file_manager = nullptr.";
            return Status::malloc_error;
        }
        file_manager_ = file_manager;
        return Status::success;
    }

    int64_t
    Dim() const override {
        return minhash_index_->GetDim();
    }

    int64_t
    Size() const override {
        return minhash_index_->Size();
    }

    int64_t
    Count() const override {
        return minhash_index_->Count();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_MINHASH_INDEX;
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool) const override {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::not_implemented, "index not_implemented ");
    }

 private:
    bool
    LoadFile(const std::string& filename) {
        if (!file_manager_->LoadFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    bool
    AddFile(const std::string& filename) {
        if (!file_manager_->AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    std::string index_prefix_;
    std::shared_ptr<FileManager> file_manager_;
    std::unique_ptr<MinHashIndex> minhash_index_;
    std::shared_ptr<ThreadPool> search_pool_;
    bool is_loaded_ = false;
};
template <typename DataType>
Status
MinHashIndexNode<DataType>::Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    auto build_conf = static_cast<const MinHashConfig&>(*cfg);
    auto index_params = std::make_unique<MinHashIndexBuildParams>();
    index_params->data_path = build_conf.data_path.value();
    if (!LoadFile(build_conf.data_path.value())) {
        LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building." << std::endl;
        return Status::disk_file_error;
    }
    index_params->index_file_path = build_conf.index_prefix.value();
    index_params->block_size = build_conf.aligned_block_size.value();
    size_t dim, rows;
    diskann::get_bin_metadata(build_conf.data_path.value(), rows, dim);
    index_params->band = build_conf.band.has_value() ? build_conf.band.value() : dim;
    if (!AddFile(index_params->index_file_path)) {
        LOG_KNOWHERE_ERROR_ << "Failed to add file " << index_params->index_file_path << ".";
        return Status::disk_file_error;
    }
    return MinHashIndex::BuildAndSave(index_params.get());
}

template <typename DataType>
Status
MinHashIndexNode<DataType>::Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) {
    auto load_conf = static_cast<const MinHashConfig&>(*cfg);
    auto index_params = std::make_unique<MinHashIndexLoadParams>();
    index_params->index_file_path = load_conf.index_prefix.value();
    index_params->enable_mmap = load_conf.enable_mmap.value();
    index_params->global_bloom_filter = load_conf.shared_bloom_filter.value();
    index_params->false_positive_prob = load_conf.bloom_false_positive_prob.value();
    if (!LoadFile(load_conf.index_prefix.value())) {
        LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building." << std::endl;
        return Status::disk_file_error;
    }
    minhash_index_ = std::make_unique<MinHashIndex>();
    auto stat = minhash_index_->Load(index_params.get());
    if (stat == Status::success) {
        is_loaded_ = true;
    }
    return stat;
}

template <typename DataType>
expected<DataSetPtr>
MinHashIndexNode<DataType>::Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg,
                                   const BitsetView& bitset) const {
    if (!is_loaded_ || !minhash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load minhash index.";
        return expected<DataSetPtr>::Err(Status::empty_index, "Minhash index not loaded");
    }
    auto search_conf = static_cast<const MinHashConfig&>(*cfg);
    auto stat = MinhashConfigCheck(dataset->GetDim(), DataFormatEnum::fp32, PARAM_TYPE::SEARCH, &search_conf, &bitset);
    if (stat != Status::success) {
        std::cout << "checking MinhashConfigCheck fail" << std::endl;
        return expected<DataSetPtr>::Err(Status::invalid_args, "MinhashConfigCheck fail.");
    }
    auto nq = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto xq = static_cast<const float*>(dataset->GetTensor());
    auto p_id = std::make_unique<int64_t[]>(nq);
    auto p_dist = std::make_unique<DistType[]>(nq);
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.emplace_back(search_pool_->push([&, index = row, p_id_ptr = p_id.get(), p_dist_ptr = p_dist.get()]() {
            minhash_index_->Search(xq + (index * dim), p_dist_ptr + index, p_id_ptr + index);
        }));
    }
    WaitAllSuccess(futures);
    auto res = GenResultDataSet(nq, 1, std::move(p_id), std::move(p_dist));
    return res;
}
// hack, fp16/bf16 not work
KNOWHERE_MOCK_REGISTER_DENSE_FLOAT_ALL_GLOBAL(MinHashIndex, MinHashIndexNode, knowhere::feature::DISK)
}  // namespace knowhere
