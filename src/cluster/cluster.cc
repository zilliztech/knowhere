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

#include "knowhere/cluster/cluster.h"

#include "knowhere/comp/time_recorder.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"

#ifdef NOT_COMPILE_FOR_SWIG
#include "knowhere/prometheus_client.h"
#include "knowhere/tracer.h"
#endif

namespace knowhere {


inline Status
LoadConfig(Config* cfg, const Json& json, knowhere::PARAM_TYPE param_type, const std::string& method,
           std::string* const msg = nullptr) {
    Json json_(json);
    auto res = Config::FormatAndCheck(*cfg, json_, msg);
    LOG_KNOWHERE_DEBUG_ << method << " config dump: " << json_.dump();
    RETURN_IF_ERROR(res);
    return Config::Load(*cfg, json_, param_type, msg);
}

template <typename T>
inline expected<DataSetPtr>
Cluster<T>::Train(const DataSet& dataset, const Json& json) {
    auto cfg = this->node->CreateConfig();
    std::string msg;
    auto status = LoadConfig(cfg.get(), json, knowhere::CLUSTER, "Train", &msg);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, msg);
    }
    return this->node->Train(dataset, *cfg);
}

template <typename T>
inline expected<DataSetPtr>
Cluster<T>::Assign(const DataSet& dataset) {
    return this->node->Assign(dataset);
}

template <typename T>
inline expected<DataSetPtr>
Cluster<T>::GetCentroids() const {
    return this->node->GetCentroids();
}

template <typename T>
inline std::string
Cluster<T>::Type() const {
    return this->node->Type();
}

template class Cluster<ClusterNode>;

}  // namespace knowhere
