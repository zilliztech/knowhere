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

#include "knowhere/cluster/cluster_factory.h"

#include "knowhere/utils.h"

namespace knowhere {

template <typename DataType>
expected<Cluster<ClusterNode>>
ClusterFactory::Create(const std::string& name, const Object& object) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    auto& func_mapping_ = MapInstance();
    auto key = GetKey<DataType>(name);
    if (func_mapping_.find(key) == func_mapping_.end()) {
        LOG_KNOWHERE_ERROR_ << "failed to find cluster type " << key << " in factory";
        return expected<Cluster<ClusterNode>>::Err(Status::invalid_cluster_error, "cluster type not supported");
    }
    LOG_KNOWHERE_INFO_ << "use key " << key << " to create knowhere cluster worker " << name;
    auto fun_map_v = (FunMapValue<Cluster<ClusterNode>>*)(func_mapping_[key].get());

    return fun_map_v->fun_value(object);
}

template <typename DataType>
const ClusterFactory&
ClusterFactory::Register(const std::string& name, std::function<Cluster<ClusterNode>(const Object&)> func) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    auto& func_mapping_ = MapInstance();
    auto key = GetKey<DataType>(name);
    assert(func_mapping_.find(key) == func_mapping_.end());
    func_mapping_[key] = std::make_unique<FunMapValue<Cluster<ClusterNode>>>(func);
    return *this;
}

ClusterFactory&
ClusterFactory::Instance() {
    static ClusterFactory factory;
    return factory;
}

ClusterFactory::ClusterFactory() {
}

ClusterFactory::FuncMap&
ClusterFactory::MapInstance() {
    static FuncMap func_map;
    return func_map;
}

}  // namespace knowhere
   //
template knowhere::expected<knowhere::Cluster<knowhere::ClusterNode>>
knowhere::ClusterFactory::Create<knowhere::fp32>(const std::string&, const Object&);
template knowhere::expected<knowhere::Cluster<knowhere::ClusterNode>>
knowhere::ClusterFactory::Create<knowhere::bf16>(const std::string&, const Object&);
template knowhere::expected<knowhere::Cluster<knowhere::ClusterNode>>
knowhere::ClusterFactory::Create<knowhere::fp16>(const std::string&, const Object&);
template const knowhere::ClusterFactory&
knowhere::ClusterFactory::Register<knowhere::fp32>(
    const std::string&, std::function<knowhere::Cluster<knowhere::ClusterNode>(const Object&)>);
template const knowhere::ClusterFactory&
knowhere::ClusterFactory::Register<knowhere::bf16>(
    const std::string&, std::function<knowhere::Cluster<knowhere::ClusterNode>(const Object&)>);
template const knowhere::ClusterFactory&
knowhere::ClusterFactory::Register<knowhere::fp16>(
    const std::string&, std::function<knowhere::Cluster<knowhere::ClusterNode>(const Object&)>);
