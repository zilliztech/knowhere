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
static bool
IndexTypeAndDataTypeCheck(const std::string& index_name, VecType data_type) {
    auto& static_index_table = IndexFactory::StaticIndexTableInstance();
    auto key = std::pair<std::string, VecType>(index_name, data_type);
    if (static_index_table.find(key) != static_index_table.end()) {
        return true;
    } else {
        return false;
    }
}
}  // namespace KnowhereCheck
}  // namespace knowhere

#endif /* COMP_KNOWHERE_CHECKER_H */
