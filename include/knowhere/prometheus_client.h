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

#define CONCATENATE(x, y) x##_##y
#define PROMETHEUS_LABEL_KNOWHERE knowhere
#define PROMETHEUS_LABEL_CARDINAL cardinal

#define DEFINE_PROMETHEUS_GAUGE_FAMILY(name, desc)                     \
    prometheus::Family<prometheus::Gauge>& CONCATENATE(name, family) = \
        prometheus::BuildGauge().Name(#name).Help(desc).Register(knowhere::prometheusClient->GetRegistry());

#define DEFINE_PROMETHEUS_GAUGE(name, module) \
    prometheus::Gauge& CONCATENATE(module, name) = CONCATENATE(name, family).Add({{"module", #module}});

#define DEFINE_PROMETHEUS_COUNTER_FAMILY(name, desc)                     \
    prometheus::Family<prometheus::Counter>& CONCATENATE(name, family) = \
        prometheus::BuildCounter().Name(#name).Help(desc).Register(knowhere::prometheusClient->GetRegistry());

#define DEFINE_PROMETHEUS_COUNTER(name, module) \
    prometheus::Counter& CONCATENATE(module, name) = CONCATENATE(name, family).Add({{"module", #module}});

#define DEFINE_PROMETHEUS_HISTOGRAM_FAMILY(name, desc)                     \
    prometheus::Family<prometheus::Histogram>& CONCATENATE(name, family) = \
        prometheus::BuildHistogram().Name(#name).Help(desc).Register(knowhere::prometheusClient->GetRegistry());

#define DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(name, module, buckets) \
    prometheus::Histogram& CONCATENATE(module, name) = CONCATENATE(name, family).Add({{"module", #module}}, buckets);

#define DEFINE_PROMETHEUS_HISTOGRAM(name, module) DEFINE_PROMETHEUS_HISTOGRAM_WITH_BUCKETS(name, module, defaultBuckets)

#define DECLARE_PROMETHEUS_GAUGE(name, module) extern prometheus::Gauge& CONCATENATE(module, name);
#define DECLARE_PROMETHEUS_COUNTER(name, module) extern prometheus::Counter& CONCATENATE(module, name);
#define DECLARE_PROMETHEUS_HISTOGRAM(name, module) extern prometheus::Histogram& CONCATENATE(module, name);

DECLARE_PROMETHEUS_HISTOGRAM(build_latency, PROMETHEUS_LABEL_KNOWHERE);
DECLARE_PROMETHEUS_HISTOGRAM(build_latency, PROMETHEUS_LABEL_CARDINAL);

DECLARE_PROMETHEUS_HISTOGRAM(load_latency, PROMETHEUS_LABEL_KNOWHERE);
DECLARE_PROMETHEUS_HISTOGRAM(load_latency, PROMETHEUS_LABEL_CARDINAL);

DECLARE_PROMETHEUS_HISTOGRAM(search_latency, PROMETHEUS_LABEL_KNOWHERE);
DECLARE_PROMETHEUS_HISTOGRAM(search_latency, PROMETHEUS_LABEL_CARDINAL);

// cardinal uses the RangeSearch function of the parent class `IndexNode` (index_node.h).
// both use the knowhere metric uniformly.
DECLARE_PROMETHEUS_HISTOGRAM(range_search_latency, PROMETHEUS_LABEL_KNOWHERE);

DECLARE_PROMETHEUS_HISTOGRAM(ann_iterator_init_latency, PROMETHEUS_LABEL_KNOWHERE);
DECLARE_PROMETHEUS_HISTOGRAM(ann_iterator_init_latency, PROMETHEUS_LABEL_CARDINAL);

DECLARE_PROMETHEUS_HISTOGRAM(search_topk, PROMETHEUS_LABEL_KNOWHERE);
DECLARE_PROMETHEUS_HISTOGRAM(search_topk, PROMETHEUS_LABEL_CARDINAL);

DECLARE_PROMETHEUS_HISTOGRAM(bitset_ratio, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(quant_compute_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(raw_compute_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(cache_hit_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(io_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(queue_latency, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(exec_latency, PROMETHEUS_LABEL_CARDINAL);

DECLARE_PROMETHEUS_HISTOGRAM(graph_search_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(ivf_search_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(bf_search_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(re_search_cnt, PROMETHEUS_LABEL_CARDINAL);

DECLARE_PROMETHEUS_HISTOGRAM(filter_connectivity_ratio, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(filter_mv_only_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(filter_mv_activated_fields_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(filter_mv_change_base_cnt, PROMETHEUS_LABEL_CARDINAL);
DECLARE_PROMETHEUS_HISTOGRAM(filter_mv_supplement_ep_bool_cnt, PROMETHEUS_LABEL_CARDINAL);

DECLARE_PROMETHEUS_HISTOGRAM(hnsw_bitset_ratio, PROMETHEUS_LABEL_KNOWHERE);
DECLARE_PROMETHEUS_HISTOGRAM(hnsw_search_hops, PROMETHEUS_LABEL_KNOWHERE);

DECLARE_PROMETHEUS_HISTOGRAM(diskann_bitset_ratio, PROMETHEUS_LABEL_KNOWHERE);
DECLARE_PROMETHEUS_HISTOGRAM(diskann_search_hops, PROMETHEUS_LABEL_KNOWHERE);
DECLARE_PROMETHEUS_HISTOGRAM(diskann_range_search_iters, PROMETHEUS_LABEL_KNOWHERE);
}  // namespace knowhere
