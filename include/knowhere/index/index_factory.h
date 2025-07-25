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

#include "index_static.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/operands.h"
#include "knowhere/utils.h"

namespace knowhere {
class IndexFactory {
 public:
    template <typename DataType>
    expected<Index<IndexNode>>
    Create(const std::string& name, const int32_t& version, const Object& object = nullptr);

    template <typename DataType>
    const IndexFactory&
    Register(const std::string& name, std::function<Index<IndexNode>(const int32_t&, const Object&)> func,
             const uint64_t features);

    static IndexFactory&
    Instance();
    typedef std::tuple<std::set<std::pair<std::string, VecType>>, std::set<std::string>, std::set<std::string>>
        GlobalIndexTable;

    bool
    FeatureCheck(const std::string& name, uint64_t feature) const;

    static const std::map<std::string, uint64_t>&
    GetIndexFeatures();

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
    using FuncMap = std::map<std::string, std::unique_ptr<FunMapValueBase>>;
    using FeatureMap = std::map<std::string, uint64_t>;
    IndexFactory();

    static FuncMap&
    MapInstance();

    static FeatureMap&
    FeatureMapInstance();
};
// register the index adapter corresponding to indexType
#define KNOWHERE_FACTOR_CONCAT(x, y) index_factory_ref_##x##y
#define KNOWHERE_REGISTER_GLOBAL(name, func, data_type, condition, features) \
    const IndexFactory& KNOWHERE_FACTOR_CONCAT(name, data_type) =            \
        condition ? IndexFactory::Instance().Register<data_type>(#name, func, features) : IndexFactory::Instance();

// register some static methods that are bound to indexType
#define KNOWHERE_STATIC_CONCAT(x, y) index_static_ref_##x##y
#define KNOWHERE_REGISTER_STATIC(name, index_node, data_type, ...)               \
    const IndexStaticFaced<data_type>& KNOWHERE_STATIC_CONCAT(name, data_type) = \
        IndexStaticFaced<data_type>::Instance().RegisterStaticFunc<index_node<data_type, ##__VA_ARGS__>>(#name);

// register the index implementation along with its associated features. Please carefully check the types and features
// supported by the index—both need to be consistent, otherwise the registration will be skipped
#define KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, data_type, features, ...)                   \
    KNOWHERE_REGISTER_STATIC(name, index_node, data_type, ##__VA_ARGS__)                              \
    KNOWHERE_REGISTER_GLOBAL(                                                                         \
        name,                                                                                         \
        (static_cast<Index<index_node<data_type, ##__VA_ARGS__>> (*)(const int32_t&, const Object&)>( \
            &Index<index_node<data_type, ##__VA_ARGS__>>::Create)),                                   \
        data_type, typeCheck<data_type>(features), features)

#define KNOWHERE_MOCK_REGISTER_GLOBAL(name, index_node, data_type, features, ...)                          \
    KNOWHERE_REGISTER_STATIC(name, index_node, data_type, ##__VA_ARGS__)                                   \
    KNOWHERE_REGISTER_GLOBAL(                                                                              \
        name,                                                                                              \
        [](const int32_t& version, const Object& object) {                                                 \
            return (Index<IndexNodeDataMockWrapper<data_type>>::Create(                                    \
                std::make_unique<index_node<MockData<data_type>::type, ##__VA_ARGS__>>(version, object))); \
        },                                                                                                 \
        data_type, typeCheck<data_type>(features), features)

// Below are some group index registration methods for batch registration of indexes that support multiple data types.
// Please review carefully and select with caution

// register vector index supporting ALL_TYPE(binary, bf16, fp16, fp32, sparse_float32) data types
#define KNOWHERE_SIMPLE_REGISTER_ALL_GLOBAL(name, index_node, features, ...)                                        \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, bin1, (features | knowhere::feature::BINARY), ##__VA_ARGS__); \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, bf16, (features | knowhere::feature::BF16), ##__VA_ARGS__);   \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, fp16, (features | knowhere::feature::FP16), ##__VA_ARGS__);   \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, fp32,                                                         \
                                    (features | knowhere::feature::FLOAT32 | knowhere::feature::SPARSE_FLOAT32),    \
                                    ##__VA_ARGS__);

// register vector index supporting sparse_float32
#define KNOWHERE_SIMPLE_REGISTER_SPARSE_FLOAT_GLOBAL(name, index_node, features, ...)                       \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, fp32, (features | knowhere::feature::SPARSE_FLOAT32), \
                                    ##__VA_ARGS__);

// register vector index supporting ALL_DENSE_TYPE(binary, bf16, fp16, fp32) data types
#define KNOWHERE_SIMPLE_REGISTER_DENSE_ALL_GLOBAL(name, index_node, features, ...)                                  \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, bin1, (features | knowhere::feature::BINARY), ##__VA_ARGS__); \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, bf16, (features | knowhere::feature::BF16), ##__VA_ARGS__);   \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, fp16, (features | knowhere::feature::FP16), ##__VA_ARGS__);   \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, fp32, (features | knowhere::feature::FLOAT32), ##__VA_ARGS__);

// register vector index supporting binary data type
#define KNOWHERE_SIMPLE_REGISTER_DENSE_BIN_GLOBAL(name, index_node, features, ...) \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, bin1, (features | knowhere::feature::BINARY), ##__VA_ARGS__);

// register vector index supporting int8 data type
#define KNOWHERE_SIMPLE_REGISTER_DENSE_INT_GLOBAL(name, index_node, features, ...) \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, int8, (features | knowhere::feature::INT8), ##__VA_ARGS__);

// register vector index supporting ALL_DENSE_FLOAT_TYPE(float32, bf16, fp16) data types
#define KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(name, index_node, features, ...)                          \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, bf16, (features | knowhere::feature::BF16), ##__VA_ARGS__); \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, fp16, (features | knowhere::feature::FP16), ##__VA_ARGS__); \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, fp32, (features | knowhere::feature::FLOAT32), ##__VA_ARGS__);

// register vector index supporting int data type
#define KNOWHERE_MOCK_REGISTER_DENSE_INT_GLOBAL(name, index_node, features, ...) \
    KNOWHERE_MOCK_REGISTER_GLOBAL(name, index_node, int8, (features | knowhere::feature::INT8), ##__VA_ARGS__);

// register vector index supporting binary data types
#define KNOWHERE_MOCK_REGISTER_DENSE_BINARY_ALL_GLOBAL(name, index_node, features, ...) \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, bin1, (features | knowhere::feature::BINARY), ##__VA_ARGS__);

// register vector index supporting ALL_DENSE_FLOAT_TYPE(float32, bf16, fp16) data types, but mocked bf16 and fp16
#define KNOWHERE_MOCK_REGISTER_DENSE_FLOAT_ALL_GLOBAL(name, index_node, features, ...)                          \
    KNOWHERE_MOCK_REGISTER_GLOBAL(name, index_node, bf16, (features | knowhere::feature::BF16), ##__VA_ARGS__); \
    KNOWHERE_MOCK_REGISTER_GLOBAL(name, index_node, fp16, (features | knowhere::feature::FP16), ##__VA_ARGS__); \
    KNOWHERE_SIMPLE_REGISTER_GLOBAL(name, index_node, fp32, (features | knowhere::feature::FLOAT32), ##__VA_ARGS__);

#define KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(name, index_node, data_type, features, thread_size) \
    KNOWHERE_REGISTER_STATIC(name, index_node, data_type)                                             \
    KNOWHERE_REGISTER_GLOBAL(                                                                         \
        name,                                                                                         \
        [](const int32_t& version, const Object& object) {                                            \
            return (Index<IndexNodeThreadPoolWrapper>::Create(                                        \
                std::make_unique<index_node<data_type>>(version, object), thread_size));              \
        },                                                                                            \
        data_type, typeCheck<data_type>(features), features)

#define KNOWHERE_SET_STATIC_GLOBAL_INDEX_TABLE(table_index, name, index_table)                      \
    static int name = []() -> int {                                                                 \
        auto& static_index_table = std::get<table_index>(IndexFactory::StaticIndexTableInstance()); \
        static_index_table.insert(index_table.begin(), index_table.end());                          \
        return 0;                                                                                   \
    }();
}  // namespace knowhere

#endif /* INDEX_FACTORY_H */
