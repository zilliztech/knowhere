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

#ifndef INDEX_FACTORY_H
#define INDEX_FACTORY_H

#include <functional>
#include <string>
#include <unordered_map>

#include "knowhere/index.h"

namespace knowhere {
class IndexFactory {
 public:
    Index<IndexNode>
    Create(const std::string& name, const int32_t& version, const Object& object = nullptr);
    const IndexFactory&
    Register(const std::string& name, std::function<Index<IndexNode>(const int32_t& version, const Object&)> func);
    static IndexFactory&
    Instance();

 private:
    typedef std::map<std::string, std::function<Index<IndexNode>(const int32_t&, const Object&)>> FuncMap;
    IndexFactory();
    static FuncMap&
    MapInstance();
};

#define KNOWHERE_CONCAT(x, y) x##y
#define KNOWHERE_REGISTER_GLOBAL(name, func) \
    const IndexFactory& KNOWHERE_CONCAT(index_factory_ref_, name) = IndexFactory::Instance().Register(#name, func)
}  // namespace knowhere

#endif /* INDEX_FACTORY_H */
