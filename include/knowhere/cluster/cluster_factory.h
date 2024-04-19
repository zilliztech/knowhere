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

#ifndef CLUSTER_FACTORY_H
#define CLUSTER_FACTORY_H

#include <functional>
#include <string>
#include <unordered_map>

#include "knowhere/cluster/cluster.h"
#include "knowhere/utils.h"

namespace knowhere {
class ClusterFactory {
 public:
    template <typename DataType>
    expected<Cluster<ClusterNode>>
    Create(const std::string& name, const Object& object = nullptr);
    template <typename DataType>
    const ClusterFactory&
    Register(const std::string& name, std::function<Cluster<ClusterNode>(const Object&)> func);
    static ClusterFactory&
    Instance();

 private:
    struct FunMapValueBase {
        virtual ~FunMapValueBase() = default;
    };
    template <typename T1>
    struct FunMapValue : FunMapValueBase {
     public:
        FunMapValue(std::function<T1(const Object&)>& input) : fun_value(input) {
        }
        std::function<T1(const Object&)> fun_value;
    };
    typedef std::map<std::string, std::unique_ptr<FunMapValueBase>> FuncMap;
    ClusterFactory();
    static FuncMap&
    MapInstance();
};

#define KNOWHERE_CLUSTER_CONCAT(x, y) cluster_factory_ref_##x##y
#define KNOWHERE_CLUSTER_REGISTER_GLOBAL(name, func, data_type)      \
    const ClusterFactory& KNOWHERE_CLUSTER_CONCAT(name, data_type) = \
        ClusterFactory::Instance().Register<data_type>(#name, func)
#define KNOWHERE_CLUSTER_SIMPLE_REGISTER_GLOBAL(name, cluster_node, data_type, ...)                                    \
    KNOWHERE_CLUSTER_REGISTER_GLOBAL(name,                                                                             \
                                     (static_cast<Cluster<cluster_node<data_type, ##__VA_ARGS__>> (*)(const Object&)>( \
                                         &Cluster<cluster_node<data_type, ##__VA_ARGS__>>::Create)),                   \
                                     data_type)
}  // namespace knowhere

#endif /* CLUSTER_FACTORY_H */
