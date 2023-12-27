// Copyright (C) 2019-2020 Zilliz. All rights reserved.
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

#include <prometheus/collectable.h>
#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <prometheus/summary.h>
#include <prometheus/text_serializer.h>

#include <memory>
#include <string>

#include "knowhere/log.h"

namespace knowhere {

class PrometheusClient {
 public:
    PrometheusClient() = default;
    PrometheusClient(const PrometheusClient&) = delete;
    PrometheusClient&
    operator=(const PrometheusClient&) = delete;

    prometheus::Registry&
    GetRegistry() {
        return *registry_;
    }

    std::string
    GetMetrics() {
        std::ostringstream ss;
        prometheus::TextSerializer serializer;
        serializer.Serialize(ss, registry_->Collect());
        return ss.str();
    }

 private:
    std::shared_ptr<prometheus::Registry> registry_ = std::make_shared<prometheus::Registry>();
};

/*****************************************************************************/
// prometheus metrics
extern const prometheus::Histogram::BucketBoundaries defaultBuckets;
extern const std::unique_ptr<PrometheusClient> prometheusClient;

#define DEFINE_PROMETHEUS_GAUGE(name, desc)                                                                  \
    prometheus::Family<prometheus::Gauge>& name##_family =                                                   \
        prometheus::BuildGauge().Name(#name).Help(desc).Register(knowhere::prometheusClient->GetRegistry()); \
    prometheus::Gauge& name = name##_family.Add({});

#define DEFINE_PROMETHEUS_COUNTER(name, desc)                                                                  \
    prometheus::Family<prometheus::Counter>& name##_family =                                                   \
        prometheus::BuildCounter().Name(#name).Help(desc).Register(knowhere::prometheusClient->GetRegistry()); \
    prometheus::Counter& name = name##_family.Add({});

#define DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(name, desc, buckets)                                            \
    prometheus::Family<prometheus::Histogram>& name##_family =                                                   \
        prometheus::BuildHistogram().Name(#name).Help(desc).Register(knowhere::prometheusClient->GetRegistry()); \
    prometheus::Histogram& name = name##_family.Add({}, buckets);

#define DEFINE_PROMETHEUS_HISTOGRAM(name, desc) DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(name, desc, defaultBuckets)

#define DECLARE_PROMETHEUS_GAUGE(name_gauge) extern prometheus::Gauge& name_gauge;
#define DECLARE_PROMETHEUS_COUNTER(name_counter) extern prometheus::Counter& name_counter;
#define DECLARE_PROMETHEUS_HISTOGRAM(name_histogram) extern prometheus::Histogram& name_histogram;

DECLARE_PROMETHEUS_HISTOGRAM(knowhere_build_latency);
DECLARE_PROMETHEUS_HISTOGRAM(knowhere_load_latency);
DECLARE_PROMETHEUS_HISTOGRAM(knowhere_search_latency);
DECLARE_PROMETHEUS_HISTOGRAM(knowhere_range_search_latency);
DECLARE_PROMETHEUS_HISTOGRAM(knowhere_ann_iterator_init_latency);
DECLARE_PROMETHEUS_HISTOGRAM(knowhere_search_topk);

DECLARE_PROMETHEUS_HISTOGRAM(knowhere_hnsw_bitset_ratio);
DECLARE_PROMETHEUS_HISTOGRAM(knowhere_hnsw_search_hops);

DECLARE_PROMETHEUS_HISTOGRAM(knowhere_diskann_bitset_ratio);
DECLARE_PROMETHEUS_HISTOGRAM(knowhere_diskann_search_hops);
DECLARE_PROMETHEUS_HISTOGRAM(knowhere_diskann_range_search_iters);
}  // namespace knowhere
