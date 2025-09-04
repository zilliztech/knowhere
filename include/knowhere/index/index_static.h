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

#ifndef INDEX_STATIC_H
#define INDEX_STATIC_H

#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/expected.h"

namespace knowhere {

struct Resource {
    uint64_t memoryCost;  // in bytes
    uint64_t diskCost;    // in bytes
};

#define DEFINE_HAS_STATIC_FUNC(func_name)                                               \
    template <typename, typename T>                                                     \
    struct has_static_##func_name {                                                     \
        static_assert(std::integral_constant<T, false>::value,                          \
                      "Second template parameter needs to be of function type.");       \
    };                                                                                  \
    template <typename C, typename Ret, typename... Args>                               \
    struct has_static_##func_name<C, Ret(Args...)> {                                    \
     private:                                                                           \
        template <typename T>                                                           \
        static auto                                                                     \
        test(int) -> decltype(T::func_name(std::declval<Args>()...), std::true_type{}); \
                                                                                        \
        template <typename>                                                             \
        static auto                                                                     \
        test(...) -> std::false_type;                                                   \
                                                                                        \
     public:                                                                            \
        static constexpr bool value = decltype(test<C>(0))::value;                      \
    };

DEFINE_HAS_STATIC_FUNC(StaticCreateConfig)
DEFINE_HAS_STATIC_FUNC(StaticEstimateLoadResource)
DEFINE_HAS_STATIC_FUNC(StaticHasRawData)
DEFINE_HAS_STATIC_FUNC(StaticConfigCheck)

template <typename DataType>
class IndexStaticFaced {
 public:
    /**
     * @brief static create index config by indexType and indexVersion
     *
     * @param indexType vector index type (HNSW, IVFFLAT, etc)
     * @param version vector index version (see version.h)
     * @return generate the config binding with the corresponding vector index
     */
    static std::unique_ptr<BaseConfig>
    CreateConfig(const knowhere::IndexType& indexType, const knowhere::IndexVersion& version);

    static knowhere::Status
    ConfigCheck(const knowhere::IndexType& indexType, const knowhere::IndexVersion& version,
                const knowhere::Json& params, std::string& msg);

    /**
     * @brief estimate the memory and disk resource usage before index loading by index params
     * @param indexType vector index type (HNSW, IVFFLAT, etc)
     * @param version  vector index version (see version.h)
     * @param file_size_in_bytes sum of all index file/binary sizes (in bytes)
     * @param num_rows number of rows in the index
     * @param dim dimension of the vector
     * @param params the union of the index build and load parameters
     * @return memory and disk usage for loading the index
     */
    static expected<Resource>
    EstimateLoadResource(const knowhere::IndexType& indexType, const knowhere::IndexVersion& version,
                         const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                         const knowhere::Json& params);

    /**
     * @brief determine whether the index contains the raw data before loading the index by index params
     * @param indexType vector index type (HNSW, IVFFLAT, etc)
     * @param version vector index version (see version.h)
     * @param params the union of the index build and load parameters
     * @return index contains the raw data or not
     */
    static bool
    HasRawData(const knowhere::IndexType& indexType, const knowhere::IndexVersion& version,
               const knowhere::Json& params);

    template <typename VecIndexNode>
    IndexStaticFaced&
    RegisterStaticFunc(const knowhere::IndexType& indexType) {
        static_assert(
            has_static_StaticCreateConfig<VecIndexNode,
                                          decltype(IndexStaticFaced<DataType>::InternalStaticCreateConfig)>::value,
            "VecIndexNode must implement StaticCreateConfig function");
        staticCreateConfigMap[indexType] = VecIndexNode::StaticCreateConfig;

        if constexpr (has_static_StaticEstimateLoadResource<
                          VecIndexNode, decltype(IndexStaticFaced<DataType>::InternalEstimateLoadResource)>::value) {
            staticEstimateLoadResourceMap[indexType] = VecIndexNode::StaticEstimateLoadResource;
        }

        if constexpr (has_static_StaticHasRawData<
                          VecIndexNode, decltype(IndexStaticFaced<DataType>::InternalStaticHasRawData)>::value) {
            staticHasRawDataMap[indexType] = VecIndexNode::StaticHasRawData;
        }

        if constexpr (has_static_StaticConfigCheck<VecIndexNode,
                                                   decltype(IndexStaticFaced<DataType>::InternalConfigCheck)>::value) {
            staticConfigCheckMap[indexType] = VecIndexNode::StaticConfigCheck;
        }

        return Instance();
    }

    static IndexStaticFaced&
    Instance();

 private:
    static expected<Resource>
    InternalEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                                 const knowhere::BaseConfig& config, const IndexVersion& version);

    static bool
    InternalStaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version);

    static knowhere::Status
    InternalConfigCheck(const knowhere::BaseConfig& config, const IndexVersion& version, std::string& msg);

    static std::unique_ptr<BaseConfig>
    InternalStaticCreateConfig();

    std::map<std::string, std::function<decltype(InternalStaticCreateConfig)>> staticCreateConfigMap;
    std::map<std::string, std::function<decltype(InternalStaticHasRawData)>> staticHasRawDataMap;
    std::map<std::string, std::function<decltype(InternalEstimateLoadResource)>> staticEstimateLoadResourceMap;
    std::map<std::string, std::function<decltype(InternalConfigCheck)>> staticConfigCheckMap;
};

}  // namespace knowhere

#endif /* INDEX_STATIC_H */
