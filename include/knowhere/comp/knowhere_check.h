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

#ifndef COMP_KNOWHERE_CHECKER_H
#define COMP_KNOWHERE_CHECKER_H

#include <string>

#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
namespace knowhere {
namespace KnowhereCheck {
bool
IndexTypeAndDataTypeCheck(const std::string& index_name, VecType data_type) {
    auto& index_factory = IndexFactory::Instance();
    switch (data_type) {
        case VecType::VECTOR_BINARY:
            return index_factory.HasIndex<bin1>(index_name);
        case VecType::VECTOR_FLOAT:
            return index_factory.HasIndex<fp32>(index_name);
        case VecType::VECTOR_BFLOAT16:
            return index_factory.HasIndex<bf16>(index_name);
        case VecType::VECTOR_FLOAT16:
            return index_factory.HasIndex<fp16>(index_name);
        case VecType::VECTOR_SPARSE_FLOAT:
            if (index_name != IndexEnum::INDEX_SPARSE_INVERTED_INDEX && index_name != IndexEnum::INDEX_SPARSE_WAND) {
                return false;
            } else {
                return index_factory.HasIndex<fp32>(index_name);
            }
        default:
            return false;
    }
}
}  // namespace KnowhereCheck
}  // namespace knowhere

#endif /* COMP_KNOWHERE_CHECKER_H */
