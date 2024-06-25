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

#include "knowhere/index/index_node_thread_pool_wrapper.h"

#include "knowhere/comp/thread_pool.h"
#include "knowhere/index/index_node.h"

namespace knowhere {

namespace {

std::shared_ptr<ThreadPool>
GlobalThreadPool(size_t pool_size) {
    static std::shared_ptr<ThreadPool> pool = std::make_shared<ThreadPool>(pool_size, "Knowhere_Global");
    return pool;
}

}  // namespace

IndexNodeThreadPoolWrapper::IndexNodeThreadPoolWrapper(std::unique_ptr<IndexNode> index_node, size_t pool_size)
    : IndexNodeThreadPoolWrapper(std::move(index_node), GlobalThreadPool(pool_size)) {
}

IndexNodeThreadPoolWrapper::IndexNodeThreadPoolWrapper(std::unique_ptr<IndexNode> index_node,
                                                       std::shared_ptr<ThreadPool> thread_pool)
    : index_node_(std::move(index_node)), thread_pool_(thread_pool) {
}

expected<DataSetPtr>
IndexNodeThreadPoolWrapper::Search(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const {
    return thread_pool_->push([&]() { return this->index_node_->Search(dataset, cfg, bitset); }).get();
}

expected<DataSetPtr>
IndexNodeThreadPoolWrapper::RangeSearch(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const {
    return thread_pool_->push([&]() { return this->index_node_->RangeSearch(dataset, cfg, bitset); }).get();
}

}  // namespace knowhere
