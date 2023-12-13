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

#include "knowhere/prometheus_client.h"

namespace knowhere {

const prometheus::Histogram::BucketBoundaries buckets = {1,   2,    4,    8,    16,   32,    64,    128,  256,
                                                         512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

const std::unique_ptr<PrometheusClient> prometheusClient = std::make_unique<PrometheusClient>();

/*******************************************************************************
 * !!! NOT use SUMMARY metrics here, because when parse SUMMARY metrics in Milvus,
 *     see following error:
 *
 *   An error has occurred while serving metrics:
 *   text format parsing error in line 50: expected float as value, got "=\"0.9\"}"
 ******************************************************************************/
DEFINE_PROMETHEUS_HISTOGRAM(knowhere_build_latency, "index build latency in knowhere (s)")
DEFINE_PROMETHEUS_HISTOGRAM(knowhere_search_latency, "search latency in knowhere (ms)")
DEFINE_PROMETHEUS_HISTOGRAM(knowhere_range_search_latency, "range search latency in knowhere (ms)")
DEFINE_PROMETHEUS_HISTOGRAM(knowhere_ann_iterator_init_latency, "ann iterator init latency in knowhere (ms)")
DEFINE_PROMETHEUS_HISTOGRAM(knowhere_search_topk, "knowhere search topk")

}  // namespace knowhere
