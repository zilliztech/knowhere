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

#ifndef INDEX_NODE_THREAD_POOL_WRAPPER_H
#define INDEX_NODE_THREAD_POOL_WRAPPER_H

#include "knowhere/index_node.h"

namespace knowhere {

class ThreadPool;
class IndexNodeThreadPoolWrapper : public IndexNode {
 public:
    IndexNodeThreadPoolWrapper(std::unique_ptr<IndexNode> index_node, size_t pool_size);

    IndexNodeThreadPoolWrapper(std::unique_ptr<IndexNode> index_node, std::shared_ptr<ThreadPool> thread_pool);

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        return index_node_->Train(dataset, cfg);
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        return index_node_->Add(dataset, cfg);
    }

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override {
        return index_node_->GetVectorByIds(dataset);
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return index_node_->HasRawData(metric_type);
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        return index_node_->GetIndexMeta(cfg);
    }

    Status
    Serialize(BinarySet& binset) const override {
        return index_node_->Serialize(binset);
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        return index_node_->Deserialize(binset, config);
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        return index_node_->DeserializeFromFile(filename, config);
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return index_node_->CreateConfig();
    }

    int64_t
    Dim() const override {
        return index_node_->Dim();
    }

    int64_t
    Size() const override {
        return index_node_->Size();
    }

    int64_t
    Count() const override {
        return index_node_->Count();
    }

    std::string
    Type() const override {
        return index_node_->Type();
    }

 private:
    std::unique_ptr<IndexNode> index_node_;
    std::shared_ptr<ThreadPool> thread_pool_;
};

}  // namespace knowhere

#endif /* INDEX_NODE_THREAD_POOL_WRAPPER_H */
