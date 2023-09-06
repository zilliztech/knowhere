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

Index<IndexNode>
IndexFactory::Create(const std::string& name, const Object& object) {
    auto& func_mapping_ = MapInstance();
    if (func_mapping_.find(name) == func_mapping_.end()) {
        return Index<IndexNode>();
    }
    LOG_KNOWHERE_INFO_ << "create knowhere index " << name;
    return func_mapping_[name](object);
}

const IndexFactory&
IndexFactory::Register(const std::string& name, std::function<Index<IndexNode>(const Object&)> func) {
    auto& func_mapping_ = MapInstance();
    func_mapping_[name] = func;
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
}  // namespace knowhere
