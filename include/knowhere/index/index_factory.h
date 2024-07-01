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
#include <set>
#include <string>
#include <unordered_map>

#include "knowhere/index/index.h"
#include "knowhere/utils.h"

namespace knowhere {
class IndexFactory {
 public:
    template <typename DataType>
    expected<Index<IndexNode>>
    Create(const std::string& name, const int32_t& version, const Object& object = nullptr);
    template <typename DataType>
    const IndexFactory&
    Register(const std::string& name, std::function<Index<IndexNode>(const int32_t&, const Object&)> func);
    static IndexFactory&
    Instance();
    typedef std::set<std::pair<std::string, VecType>> GlobalIndexTable;
    static GlobalIndexTable&
    StaticIndexTableInstance();

 private:
    struct FunMapValueBase {
        virtual ~FunMapValueBase() = default;
    };
    template <typename T1>
    struct FunMapValue : FunMapValueBase {
     public:
        FunMapValue(std::function<T1(const int32_t&, const Object&)>& input) : fun_value(input) {
        }
        std::function<T1(const int32_t&, const Object&)> fun_value;
    };
    typedef std::map<std::string, std::unique_ptr<FunMapValueBase>> FuncMap;
    IndexFactory();
    static FuncMap&
    MapInstance();
};

#define KNOWHERE_CONCAT(x, y) index_factory_ref_##x##y
#define KNOWHERE_REGISTER_GLOBAL(name, func, data_type) \
    const IndexFactory& KNOWHERE_CONCAT(name, data_type) = IndexFactory::Instance().Register<data_type>(#name, func)
#define KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, data_type, ...)                             \
    KNOWHERE_REGISTER_GLOBAL(                                                                         \
        name,                                                                                         \
        (static_cast<Index<index_node<data_type, ##__VA_ARGS__>> (*)(const int32_t&, const Object&)>( \
            &Index<index_node<data_type, ##__VA_ARGS__>>::Create)),                                   \
        data_type)
#define KNOWHERE_MOCK_REGISTER_GLOBAL(name, index_node, data_type, ...)                                    \
    KNOWHERE_REGISTER_GLOBAL(                                                                              \
        name,                                                                                              \
        [](const int32_t& version, const Object& object) {                                                 \
            return (Index<IndexNodeDataMockWrapper<data_type>>::Create(                                    \
                std::make_unique<index_node<MockData<data_type>::type, ##__VA_ARGS__>>(version, object))); \
        },                                                                                                 \
        data_type)
#define KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(name, index_node, data_type, thread_size)              \
    KNOWHERE_REGISTER_GLOBAL(                                                                            \
        name,                                                                                            \
        [](const int32_t& version, const Object& object) {                                               \
            return (Index<IndexNodeThreadPoolWrapper>::Create(                                           \
                std::make_unique<index_node<MockData<data_type>::type>>(version, object), thread_size)); \
        },                                                                                               \
        data_type)
#define KNOWHERE_SET_STATIC_GLOBAL_INDEX_TABLE(name, index_table)            \
    static int name = []() -> int {                                          \
        auto& static_index_table = IndexFactory::StaticIndexTableInstance(); \
        static_index_table.insert(index_table.begin(), index_table.end());   \
        return 0;                                                            \
    }();
}  // namespace knowhere

#endif /* INDEX_FACTORY_H */
