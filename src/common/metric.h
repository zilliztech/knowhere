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

#ifndef COMMON_METRIC_H
#define COMMON_METRIC_H

#include <algorithm>
#include <string>
#include <unordered_map>

#include "faiss/MetricType.h"
#include "fmt/format.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/expected.h"

namespace knowhere {

inline expected<faiss::MetricType>
Str2FaissMetricType(std::string metric) {
    static const std::unordered_map<std::string, faiss::MetricType> metric_map = {
        {metric::L2, faiss::MetricType::METRIC_L2},
        {metric::IP, faiss::MetricType::METRIC_INNER_PRODUCT},
        {metric::COSINE, faiss::MetricType::METRIC_INNER_PRODUCT},
        {metric::HAMMING, faiss::MetricType::METRIC_Hamming},
        {metric::JACCARD, faiss::MetricType::METRIC_Jaccard},
        {metric::SUBSTRUCTURE, faiss::MetricType::METRIC_Substructure},
        {metric::SUPERSTRUCTURE, faiss::MetricType::METRIC_Superstructure},
        {metric::MHJACCARD, faiss::MetricType::METRIC_MinHash_Jaccard},
    };

    std::transform(metric.begin(), metric.end(), metric.begin(), toupper);
    auto it = metric_map.find(metric);
    if (it == metric_map.end()) {
        return expected<faiss::MetricType>::Err(Status::invalid_metric_type,
                                                fmt::format("unsupported metric type {}", metric));
    }
    return it->second;
}

}  // namespace knowhere

#endif /* METRIC_H */
