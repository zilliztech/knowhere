// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "knowhere/index/index_node_data_mock_wrapper.h"

#include "knowhere/index/index_node.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"

namespace knowhere {

template <typename DataType>
Status
IndexNodeDataMockWrapper<DataType>::Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                                          bool use_knowhere_build_pool) {
    auto ds_ptr = ConvertFromDataTypeIfNeeded<DataType>(dataset);
    return index_node_->Build(ds_ptr, std::move(cfg), use_knowhere_build_pool);
}

template <typename DataType>
Status
IndexNodeDataMockWrapper<DataType>::Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                                          bool use_knowhere_build_pool) {
    auto ds_ptr = ConvertFromDataTypeIfNeeded<DataType>(dataset);
    return index_node_->Train(ds_ptr, std::move(cfg), use_knowhere_build_pool);
}

template <typename DataType>
Status
IndexNodeDataMockWrapper<DataType>::Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                                        bool use_knowhere_build_pool) {
    auto ds_ptr = ConvertFromDataTypeIfNeeded<DataType>(dataset);
    return index_node_->Add(ds_ptr, std::move(cfg), use_knowhere_build_pool);
}

template <typename DataType>
expected<DataSetPtr>
IndexNodeDataMockWrapper<DataType>::Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg,
                                           const BitsetView& bitset, milvus::OpContext* op_context) const {
    auto ds_ptr = ConvertFromDataTypeIfNeeded<DataType>(dataset);
    return index_node_->Search(ds_ptr, std::move(cfg), bitset, op_context);
}

template <typename DataType>
expected<DataSetPtr>
IndexNodeDataMockWrapper<DataType>::RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg,
                                                const BitsetView& bitset, milvus::OpContext* op_context) const {
    auto ds_ptr = ConvertFromDataTypeIfNeeded<DataType>(dataset);
    return index_node_->RangeSearch(ds_ptr, std::move(cfg), bitset, op_context);
}

template <typename DataType>
expected<std::vector<IndexNode::IteratorPtr>>
IndexNodeDataMockWrapper<DataType>::AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg,
                                                const BitsetView& bitset, bool use_knowhere_search_pool,
                                                milvus::OpContext* op_context) const {
    auto ds_ptr = ConvertFromDataTypeIfNeeded<DataType>(dataset);
    return index_node_->AnnIterator(ds_ptr, std::move(cfg), bitset, use_knowhere_search_pool, op_context);
}

template <typename DataType>
expected<DataSetPtr>
IndexNodeDataMockWrapper<DataType>::GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const {
    auto res = index_node_->GetVectorByIds(dataset, op_context);
    if constexpr (!std::is_same_v<DataType, typename MockData<DataType>::type>) {
        if (res.has_value()) {
            auto res_v = data_type_conversion<typename MockData<DataType>::type, DataType>(*res.value());
            return res_v;
        } else {
            return res;
        }
    } else {
        return res;
    }
}

template class knowhere::IndexNodeDataMockWrapper<knowhere::fp16>;
template class knowhere::IndexNodeDataMockWrapper<knowhere::bf16>;
template class knowhere::IndexNodeDataMockWrapper<knowhere::int8>;

}  // namespace knowhere
