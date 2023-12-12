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

#include "knowhere/factory.h"

namespace knowhere {

template <typename DataType>
Index<IndexNode>
IndexFactory::Create(const std::string& name, const int32_t& version, const Object& object) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    auto& func_mapping_ = MapInstance();
    auto key = GetIndexKey<DataType>(name);
    assert(func_mapping_.find(key) != func_mapping_.end());
    LOG_KNOWHERE_INFO_ << "use key" << key << " to create knowhere index " << name << " with version " << version;
    auto fun_map_v = (FunMapValue<Index<IndexNode>>*)(func_mapping_[key].get());
    return fun_map_v->fun_value(version, object);
}

template <typename DataType>
const IndexFactory&
IndexFactory::Register(const std::string& name, std::function<Index<IndexNode>(const int32_t&, const Object&)> func) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    auto& func_mapping_ = MapInstance();
    auto key = GetIndexKey<DataType>(name);
    assert(func_mapping_.find(key) == func_mapping_.end());
    func_mapping_[key] = std::make_unique<FunMapValue<Index<IndexNode>>>(func);
    return *this;
}

IndexFactory&
IndexFactory::Instance() {
    static IndexFactory factory;
    return factory;
}

IndexFactory::IndexFactory() {
}
IndexFactory::FuncMap&
IndexFactory::MapInstance() {
    static FuncMap func_map;
    return func_map;
}

template class Index<IndexNode>
IndexFactory::Create<knowhere::fp32>(const std::string&, const int32_t&, const Object&);
template class Index<IndexNode>
IndexFactory::Create<knowhere::bin1>(const std::string&, const int32_t&, const Object&);
template class Index<IndexNode>
IndexFactory::Create<knowhere::fp16>(const std::string&, const int32_t&, const Object&);
template class Index<IndexNode>
IndexFactory::Create<knowhere::bf16>(const std::string&, const int32_t&, const Object&);
template const IndexFactory&
IndexFactory::Register<knowhere::fp32>(const std::string&,
                                       std::function<Index<IndexNode>(const int32_t&, const Object&)>);
template const IndexFactory&
IndexFactory::Register<knowhere::bin1>(const std::string&,
                                       std::function<Index<IndexNode>(const int32_t&, const Object&)>);
template const IndexFactory&
IndexFactory::Register<knowhere::fp16>(const std::string&,
                                       std::function<Index<IndexNode>(const int32_t&, const Object&)>);
template const IndexFactory&
IndexFactory::Register<knowhere::bf16>(const std::string&,
                                       std::function<Index<IndexNode>(const int32_t&, const Object&)>);
}  // namespace knowhere
