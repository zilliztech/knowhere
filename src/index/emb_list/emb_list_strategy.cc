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

#include "knowhere/index/emb_list_strategy.h"

#include "knowhere/log.h"

namespace knowhere {

EmbListStrategyPtr
CreateTokenANNEmbListStrategy();
EmbListStrategyPtr
CreateMuveraEmbListStrategy();
EmbListStrategyPtr
CreateLemurEmbListStrategy();

expected<EmbListStrategyPtr>
CreateEmbListStrategy(const std::string& strategy_type, const BaseConfig& config) {
    if (strategy_type == meta::EMB_LIST_STRATEGY_TOKENANN || strategy_type.empty()) {
        return CreateTokenANNEmbListStrategy();
    }
    if (strategy_type == meta::EMB_LIST_STRATEGY_MUVERA) {
        return CreateMuveraEmbListStrategy();
    }
    if (strategy_type == meta::EMB_LIST_STRATEGY_LEMUR) {
        return CreateLemurEmbListStrategy();
    }
    LOG_KNOWHERE_ERROR_ << "Unknown emb_list strategy: " << strategy_type;
    return expected<EmbListStrategyPtr>::Err(Status::invalid_args, "unknown emb_list strategy: " + strategy_type);
}

}  // namespace knowhere
