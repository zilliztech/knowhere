// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#include "knowhere/tracer.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include "knowhere/log.h"
#include "opentelemetry/exporters/jaeger/jaeger_exporter_factory.h"
#include "opentelemetry/exporters/ostream/span_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_grpc_exporter_factory.h"
#include "opentelemetry/sdk/resource/resource.h"
#include "opentelemetry/sdk/trace/batch_span_processor_factory.h"
#include "opentelemetry/sdk/trace/sampler.h"
#include "opentelemetry/sdk/trace/samplers/always_on.h"
#include "opentelemetry/sdk/trace/samplers/parent.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"
#include "opentelemetry/sdk/version/version.h"
#include "opentelemetry/trace/span_context.h"
#include "opentelemetry/trace/span_metadata.h"

namespace knowhere::tracer {

namespace trace = opentelemetry::trace;
namespace nostd = opentelemetry::nostd;

namespace trace_sdk = opentelemetry::sdk::trace;
namespace resource = opentelemetry::sdk::resource;
namespace jaeger = opentelemetry::exporter::jaeger;
namespace ostream = opentelemetry::exporter::trace;
namespace otlp = opentelemetry::exporter::otlp;

static bool enable_trace = true;
static std::shared_ptr<trace::TracerProvider> noop_trace_provider =
    std::make_shared<opentelemetry::trace::NoopTracerProvider>();

void
initTelemetry(const TraceConfig& cfg) {
    std::unique_ptr<opentelemetry::sdk::trace::SpanExporter> exporter;
    if (cfg.exporter == "stdout") {
        exporter = ostream::OStreamSpanExporterFactory::Create();
    } else if (cfg.exporter == "jaeger") {
        auto opts = jaeger::JaegerExporterOptions{};
        opts.transport_format = jaeger::TransportFormat::kThriftHttp;
        opts.endpoint = cfg.jaegerURL;
        exporter = jaeger::JaegerExporterFactory::Create(opts);
        LOG_KNOWHERE_INFO_ << "init jaeger exporter, endpoint: " << opts.endpoint;
        // } else if (cfg.exporter == "otlp") {
        //     auto opts = otlp::OtlpGrpcExporterOptions{};
        //     opts.endpoint = cfg.otlpEndpoint;
        //     opts.use_ssl_credentials = cfg.oltpSecure;
        //     exporter = otlp::OtlpGrpcExporterFactory::Create(opts);
        //     LOG_KNOWHERE_INFO_ << "init otlp exporter, endpoint: " << opts.endpoint;
    } else {
        LOG_KNOWHERE_INFO_ << "Empty Trace";
        enable_trace = false;
    }
    if (enable_trace) {
        auto processor = trace_sdk::BatchSpanProcessorFactory::Create(std::move(exporter), {});
        resource::ResourceAttributes attributes = {{"service.name", TRACE_SERVICE_KNOWHERE}, {"NodeID", cfg.nodeID}};
        auto resource = resource::Resource::Create(attributes);
        auto sampler = std::make_unique<trace_sdk::ParentBasedSampler>(std::make_shared<trace_sdk::AlwaysOnSampler>());
        std::shared_ptr<trace::TracerProvider> provider =
            trace_sdk::TracerProviderFactory::Create(std::move(processor), resource, std::move(sampler));
        trace::Provider::SetTracerProvider(provider);
    } else {
        trace::Provider::SetTracerProvider(noop_trace_provider);
    }
}

std::shared_ptr<trace::Tracer>
GetTracer() {
    auto provider = trace::Provider::GetTracerProvider();
    return provider->GetTracer(TRACE_SERVICE_KNOWHERE, OPENTELEMETRY_SDK_VERSION);
}

std::shared_ptr<trace::Span>
StartSpan(const std::string& name, TraceContext* ctx) {
    trace::StartSpanOptions opts;
    if (enable_trace && ctx != nullptr && ctx->traceID != nullptr && ctx->spanID != nullptr) {
        if (EmptyTraceID(ctx) || EmptySpanID(ctx)) {
            return noop_trace_provider->GetTracer("noop")->StartSpan("noop");
        }

        opts.parent = trace::SpanContext(trace::TraceId({ctx->traceID, trace::TraceId::kSize}),
                                         trace::SpanId({ctx->spanID, trace::SpanId::kSize}),
                                         trace::TraceFlags(ctx->traceFlags), true);
    }
    return GetTracer()->StartSpan(name, opts);
}

thread_local std::shared_ptr<trace::Span> local_span;
void
SetRootSpan(std::shared_ptr<trace::Span> span) {
    if (enable_trace) {
        local_span = std::move(span);
    }
}

void
CloseRootSpan() {
    if (enable_trace) {
        local_span = nullptr;
    }
}

void
AddEvent(const std::string& event_label) {
    if (enable_trace && local_span != nullptr) {
        local_span->AddEvent(event_label);
    }
}

bool
isEmptyID(const uint8_t* id, int length) {
    if (id != nullptr) {
        for (int i = 0; i < length; i++) {
            if (id[i] != 0) {
                return false;
            }
        }
    }
    return true;
}

bool
EmptyTraceID(const TraceContext* ctx) {
    return isEmptyID(ctx->traceID, trace::TraceId::kSize);
}

bool
EmptySpanID(const TraceContext* ctx) {
    return isEmptyID(ctx->spanID, trace::SpanId::kSize);
}

std::string
BytesToHexStr(const uint8_t* data, size_t len) {
    std::stringstream ss;
    for (size_t i = 0; i < len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]);
    }
    return ss.str();
}

std::string
GetIDFromHexStr(const std::string& hexStr) {
    std::stringstream ss;
    for (size_t i = 0; i < (hexStr.length() / 2) * 2; i += 2) {
        std::string byteStr = hexStr.substr(i, 2);
        char byte = static_cast<char>(std::stoi(byteStr, nullptr, 16));
        ss << byte;
    }
    return ss.str();
}

}  // namespace knowhere::tracer
