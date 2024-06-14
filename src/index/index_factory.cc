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

#include "knowhere/index/index_factory.h"

#include "knowhere/index/index_table.h"

#ifdef KNOWHERE_WITH_RAFT
#include <cuda_runtime_api.h>
#endif

namespace knowhere {

#ifdef KNOWHERE_WITH_RAFT

bool
checkGpuAvailable(const std::string& name) {
    if (name == "GPU_RAFT_BRUTE_FORCE" || name == "GPU_BRUTE_FORCE" || name == "GPU_RAFT_CAGRA" ||
        name == "GPU_CAGRA" || name == "GPU_RAFT_IVF_FLAT" || name == "GPU_IVF_FLAT" || name == "GPU_RAFT_IVF_PQ" ||
        name == "GPU_IVF_PQ") {
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess) {
            LOG_KNOWHERE_INFO_ << cudaGetErrorString(status);
            return false;
        }
        if (count < 1) {
            LOG_KNOWHERE_INFO_ << "GPU not available";
            return false;
        }
    }
    return true;
}
#endif

template <typename DataType>
expected<Index<IndexNode>>
IndexFactory::Create(const std::string& name, const int32_t& version, const Object& object) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    auto& func_mapping_ = MapInstance();
    auto key = GetKey<DataType>(name);
    if (func_mapping_.find(key) == func_mapping_.end()) {
        LOG_KNOWHERE_ERROR_ << "failed to find index " << key << " in factory";
        return expected<Index<IndexNode>>::Err(Status::invalid_index_error, "index not supported");
    }
    LOG_KNOWHERE_INFO_ << "use key " << key << " to create knowhere index " << name << " with version " << version;
    auto fun_map_v = (FunMapValue<Index<IndexNode>>*)(func_mapping_[key].get());

#ifdef KNOWHERE_WITH_RAFT
    if (!checkGpuAvailable(name)) {
        return expected<Index<IndexNode>>::Err(Status::cuda_runtime_error, "gpu not available");
    }
#endif

    return fun_map_v->fun_value(version, object);
}

template <typename DataType>
const IndexFactory&
IndexFactory::Register(const std::string& name, std::function<Index<IndexNode>(const int32_t&, const Object&)> func) {
    static_assert(KnowhereDataTypeCheck<DataType>::value == true);
    auto& func_mapping_ = MapInstance();
    auto key = GetKey<DataType>(name);
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
IndexFactory::GlobalIndexTable&
IndexFactory::StaticIndexTableInstance() {
    static GlobalIndexTable static_index_table;
    return static_index_table;
}

}  // namespace knowhere
   //
template knowhere::expected<knowhere::Index<knowhere::IndexNode>>
knowhere::IndexFactory::Create<knowhere::fp32>(const std::string&, const int32_t&, const Object&);
template knowhere::expected<knowhere::Index<knowhere::IndexNode>>
knowhere::IndexFactory::Create<knowhere::bin1>(const std::string&, const int32_t&, const Object&);
template knowhere::expected<knowhere::Index<knowhere::IndexNode>>
knowhere::IndexFactory::Create<knowhere::fp16>(const std::string&, const int32_t&, const Object&);
template knowhere::expected<knowhere::Index<knowhere::IndexNode>>
knowhere::IndexFactory::Create<knowhere::bf16>(const std::string&, const int32_t&, const Object&);
template const knowhere::IndexFactory&
knowhere::IndexFactory::Register<knowhere::fp32>(
    const std::string&, std::function<knowhere::Index<knowhere::IndexNode>(const int32_t&, const Object&)>);
template const knowhere::IndexFactory&
knowhere::IndexFactory::Register<knowhere::bin1>(
    const std::string&, std::function<knowhere::Index<knowhere::IndexNode>(const int32_t&, const Object&)>);
template const knowhere::IndexFactory&
knowhere::IndexFactory::Register<knowhere::fp16>(
    const std::string&, std::function<knowhere::Index<knowhere::IndexNode>(const int32_t&, const Object&)>);
template const knowhere::IndexFactory&
knowhere::IndexFactory::Register<knowhere::bf16>(
    const std::string&, std::function<knowhere::Index<knowhere::IndexNode>(const int32_t&, const Object&)>);
