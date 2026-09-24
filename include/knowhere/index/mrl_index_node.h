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

#pragma once

#include <memory>

#include "knowhere/index/index.h"

namespace knowhere {

class DataViewIndexBase;

class MRLIndexNode final : public IndexNode {
 public:
    MRLIndexNode(Index<IndexNode>&& base_index, int64_t source_dim, int64_t mrl_dim, DataFormatEnum data_type,
                 bool with_mrl_refine, ViewDataOp view_data);

    ~MRLIndexNode() override;

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override;

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override;

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override;

    expected<std::vector<IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool, milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override;

    bool
    HasRawData(const std::string& metric_type) const override;

    bool
    IsAdditionalScalarSupported(bool is_mv_only) const override;

    bool
    IsIndexRefineEnabled() const override;

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override;

    Status
    Serialize(BinarySet& binset) const override;

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override;

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> cfg) override;

    std::unique_ptr<BaseConfig>
    CreateConfig() const override;

    int64_t
    Dim() const override;

    int64_t
    Size() const override;

    int64_t
    Count() const override;

    std::string
    Type() const override;

    bool
    LoadIndexWithStream() override;

    std::optional<size_t>
    GetQueryCodeSize(const DataSetPtr dataset) const override;

 private:
    Status
    ValidateDataSet(const DataSetPtr& dataset) const;

    DataSetPtr
    PreparePrefixDataSet(const DataSetPtr& dataset) const;

    Status
    BuildFromFile(const DataSetPtr& dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool);

    Status
    CreateRefineIndex(const BaseConfig& cfg);

    size_t
    ElementSize() const;

    Index<IndexNode> base_index_;
    std::shared_ptr<DataViewIndexBase> refine_index_;
    int64_t source_dim_;
    int64_t mrl_dim_;
    DataFormatEnum data_type_;
    bool with_mrl_refine_;
    ViewDataOp view_data_;
};

Index<IndexNode>
CreateMRLIndex(Index<IndexNode>&& base_index, int64_t source_dim, int64_t mrl_dim, DataFormatEnum data_type,
               bool with_mrl_refine, ViewDataOp view_data);

}  // namespace knowhere
