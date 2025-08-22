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

#include "filemanager/FileManager.h"
#include "index/minhash/minhash_lsh.h"
#include "index/minhash/minhash_lsh_config.h"
#include "index/minhash/minhash_util.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"

namespace knowhere {
template <typename DataType>
class MinHashLSHNode : public IndexNode {
 public:
    using DistType = float;
    MinHashLSHNode(const int32_t& version, const Object& object) {
        if (typeid(object) == typeid(Pack<std::shared_ptr<milvus::FileManager>>)) {
            auto disk_index_pack = dynamic_cast<const Pack<std::shared_ptr<milvus::FileManager>>*>(&object);
            if (disk_index_pack != nullptr) {
                file_manager_ = disk_index_pack->GetPack();
            }
        }
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
        if (minhash_lsh_ == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index,
                                             "plz build and load index before calling GetVectorByIds.");
        }
        auto dim = this->Dim();
        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();
        if (minhash_lsh_->HasRawData()) {
            auto data = std::make_unique<char[]>(rows * ((this->Dim() + 7) / 8));
            minhash_lsh_->GetDataByIds(ids, rows, data.get());
            return GenResultDataSet(rows, dim, std::move(data));
        } else {
            return expected<DataSetPtr>::Err(Status::not_implemented, "index not_implemented ");
        }
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        const MinHashLSHConfig& mh_lsh_cfg = static_cast<const MinHashLSHConfig&>(config);
        return mh_lsh_cfg.with_raw_data.value();
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if (minhash_lsh_ == nullptr) {
            return false;
        } else {
            return minhash_lsh_->HasRawData();
        }
    }

    static expected<Resource>
    StaticEstimateLoadResource(const float file_size, const knowhere::BaseConfig& config, const IndexVersion& version) {
        const MinHashLSHConfig& mh_lsh_cfg = static_cast<const MinHashLSHConfig&>(config);
        auto code_in_mem = mh_lsh_cfg.mh_lsh_code_in_mem.value();
        if (code_in_mem) {
            // unknown dim and rows number, incorrect mem size if code_in_mem == true.
            return Resource{file_size * 1.0f, file_size * 1.0f};
        } else {
            return Resource{0.0f, file_size * 1.0f};
        }
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

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override;

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_ERROR_ << "minhash index doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<MinHashLSHConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    Status
    SetFileManager(std::shared_ptr<milvus::FileManager> file_manager) {
        if (file_manager == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Malloc error, file_manager = nullptr.";
            return Status::malloc_error;
        }
        file_manager_ = file_manager;
        return Status::success;
    }

    int64_t
    Dim() const override {
        if (minhash_lsh_ == nullptr) {
            return -1;
        } else {
            return minhash_lsh_->GetVectorSize() * 8;
        }
    }

    int64_t
    Size() const override {
        if (minhash_lsh_ == nullptr) {
            return 0;
        } else {
            return minhash_lsh_->Size();
        }
    }

    int64_t
    Count() const override {
        if (minhash_lsh_ == nullptr) {
            return 0;
        } else {
            return minhash_lsh_->Count();
        }
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_MINHASH_LSH;
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
    std::shared_ptr<milvus::FileManager> file_manager_ = nullptr;
    std::unique_ptr<minhash::MinHashLSH> minhash_lsh_ = nullptr;
    std::shared_ptr<ThreadPool> search_pool_;
    bool is_loaded_ = false;
    const std::string fname_ = "minhash_lsh_index";
};
template <typename DataType>
Status
MinHashLSHNode<DataType>::Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    auto build_conf = static_cast<const MinHashLSHConfig&>(*cfg);
    if (this->file_manager_ == nullptr) {
        LOG_KNOWHERE_ERROR_ << "Failed to use a null file manager to build.";
        return Status::internal_error;
    }
    try {
        if (!LoadFile(build_conf.data_path.value())) {
            LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building.";
            return Status::disk_file_error;
        }
        size_t dim, rows;
        load_vec_meta<DataType>(build_conf.data_path.value(), rows, dim);
        if (dim % 8 != 0 || build_conf.mh_element_bit_width.value() % 8 != 0) {
            LOG_KNOWHERE_ERROR_ << "Expecting (dim % 8 == 0) and (mh_element_bit_width % 8 == 0)";
            return Status::invalid_args;
        }
        size_t mh_vec_element_size = size_t(build_conf.mh_element_bit_width.value() / 8);
        size_t mh_vec_length = size_t(dim / build_conf.mh_element_bit_width.value());
        minhash::MinHashLSHBuildParams index_params = {
            .data_path = build_conf.data_path.value(),
            .index_file_path = build_conf.index_prefix.value() + fname_,
            .band = size_t(build_conf.mh_lsh_band.value()),
            .block_size = size_t(build_conf.mh_lsh_aligned_block_size.value()),
            .with_raw_data = build_conf.with_raw_data.value(),
            .mh_vec_element_size = mh_vec_element_size,
            .mh_vec_length = mh_vec_length};

        auto build_stat = minhash::MinHashLSH::BuildAndSave(&index_params);
        if (build_stat != Status::success) {
            return build_stat;
        }
        if (!AddFile(index_params.index_file_path)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << index_params.index_file_path << ".";
            return Status::disk_file_error;
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "minhash lsh inner error: " << e.what();
        return Status::internal_error;
    }
    return Status::success;
}

template <typename DataType>
Status
MinHashLSHNode<DataType>::Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) {
    auto load_conf = static_cast<const MinHashLSHConfig&>(*cfg);
    auto index_params_ptr = std::make_unique<minhash::MinHashLSHLoadParams>();
    index_params_ptr->index_file_path = load_conf.index_prefix.value() + fname_;
    index_params_ptr->hash_code_in_memory = load_conf.mh_lsh_code_in_mem.value();
    index_params_ptr->global_bloom_filter = load_conf.mh_lsh_shared_bloom_filter.value();
    index_params_ptr->false_positive_prob = load_conf.mh_lsh_bloom_false_positive_prob.value();
    if (!LoadFile(index_params_ptr->index_file_path)) {
        LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building.";
        return Status::disk_file_error;
    }
    Status stat = Status::success;
    try {
        minhash_lsh_ = std::make_unique<minhash::MinHashLSH>();
        stat = minhash_lsh_->Load(index_params_ptr.get());
        if (stat == Status::success) {
            is_loaded_ = true;
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "minhash lsh inner error: " << e.what();
        return Status::internal_error;
    }
    return stat;
}

template <typename DataType>
expected<DataSetPtr>
MinHashLSHNode<DataType>::Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg,
                                 const BitsetView& bitset) const {
    if (!is_loaded_ || !minhash_lsh_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load minhash index.";
        return expected<DataSetPtr>::Err(Status::empty_index, "Minhash index not loaded");
    }
    auto search_conf = static_cast<const MinHashLSHConfig&>(*cfg);
    auto stat = MinhashConfigCheck(dataset->GetDim(), DataFormatEnum::bin1, PARAM_TYPE::SEARCH, &search_conf, &bitset);
    if (stat != Status::success) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "MinhashConfigCheck() failed, please check the config.");
    }
    auto topk = search_conf.k.value();
    auto nq = dataset->GetRows();
    auto dim = dataset->GetDim() / 8;
    auto xq = static_cast<const char*>(dataset->GetTensor());
    auto p_id = std::make_unique<int64_t[]>(nq * topk);
    auto p_dist = std::make_unique<DistType[]>(nq * topk);
    minhash::MinHashLSHSearchParams search_params;
    search_params.k = topk;
    search_params.search_with_jaccard = search_conf.mh_search_with_jaccard.value();
    search_params.refine_k = search_conf.refine_k.value_or(topk);
    BitsetViewIDSelector bw_idselector(bitset);
    search_params.id_selector = (bitset.empty()) ? nullptr : &bw_idselector;
    try {
        if (search_conf.mh_lsh_batch_search.value() == true) {
            minhash_lsh_->BatchSearch(xq, nq, p_dist.get(), p_id.get(), search_pool_, &search_params);
        } else {
            std::vector<folly::Future<folly::Unit>> futures;
            constexpr size_t batch_size = 64;
            size_t run_times = (nq + batch_size - 1) / batch_size;
            futures.reserve(run_times);
            for (size_t row = 0; row < run_times; ++row) {
                futures.emplace_back(
                    search_pool_->push([&, beg = row * batch_size, end = std::min(int64_t((row + 1) * batch_size), nq),
                                        p_id_ptr = p_id.get(), p_dist_ptr = p_dist.get()]() {
                        for (size_t index = beg; index < (size_t)end; index++) {
                            minhash_lsh_->Search(xq + (index * dim), p_dist_ptr + index * topk, p_id_ptr + index * topk,
                                                 &search_params);
                        }
                    }));
            }
            WaitAllSuccess(futures);
        }
    } catch (const std::exception& e) {
        LOG_KNOWHERE_WARNING_ << "minhash lsh inner error: " << e.what();
        return expected<DataSetPtr>::Err(Status::internal_error, e.what());
    }
    auto res = GenResultDataSet(nq, topk, std::move(p_id), std::move(p_dist));
    return res;
}

KNOWHERE_MOCK_REGISTER_DENSE_BINARY_ALL_GLOBAL(MINHASH_LSH, MinHashLSHNode, knowhere::feature::DISK)
}  // namespace knowhere
