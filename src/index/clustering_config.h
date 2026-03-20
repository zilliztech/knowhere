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

#pragma once

#include "faiss/Clustering.h"
#include "knowhere/comp/knowhere_config.h"

namespace knowhere {

// Apply global clustering configuration to a ClusteringParameters instance.
inline void
ApplyClusteringConfig(faiss::ClusteringParameters& cp) {
    switch (KnowhereConfig::GetClusteringType()) {
        case KnowhereConfig::ClusteringType::K_MEANS:
            cp.init_method = faiss::ClusteringInitMethod::RANDOM;
            break;
        case KnowhereConfig::ClusteringType::K_MEANS_PLUS_PLUS:
            cp.init_method = faiss::ClusteringInitMethod::KMEANS_PLUS_PLUS;
            break;
        default:
            cp.init_method = faiss::ClusteringInitMethod::RANDOM;
            break;
    }
    // Knowhere API uses [0, 100] range, baseline FAISS uses [0, 1]
    cp.early_stop_threshold = KnowhereConfig::GetEarlyStopThreshold() / 100.0;
}

}  // namespace knowhere
