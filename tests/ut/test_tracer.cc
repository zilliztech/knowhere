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

#include <memory>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/tracer.h"

using namespace knowhere::tracer;
using namespace opentelemetry::trace;

TEST_CASE("Test Tracer init", "Init test") {
    using Catch::Approx;

    auto config = std::make_shared<TraceConfig>();
    config->exporter = "stdout";
    config->nodeID = 1;
    initTelemetry(*config);
    auto span = StartSpan("test");
    REQUIRE(span->IsRecording());

    config = std::make_shared<TraceConfig>();
    config->exporter = "jaeger";
    config->jaegerURL = "http://localhost:14268/api/traces";
    config->nodeID = 1;
    initTelemetry(*config);
    span = StartSpan("test");
    REQUIRE(span->IsRecording());
}

TEST_CASE("Test Tracer span", "Span test") {
    auto ctx = std::make_shared<TraceContext>();
    ctx->traceID =
        new uint8_t[16]{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10};
    ctx->spanID = new uint8_t[8]{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};
    ctx->traceFlags = 1;
    REQUIRE_FALSE(EmptyTraceID(ctx.get()));
    REQUIRE_FALSE(EmptySpanID(ctx.get()));

    auto span = StartSpan("test", ctx.get());
    auto spanCtx = span->GetContext();
    REQUIRE(spanCtx.trace_id() == trace::TraceId({ctx->traceID, trace::TraceId::kSize}));
    // REQUIRE(spanCtx.span_id() == trace::SpanId({ctx->spanID, trace::SpanId::kSize}));
    REQUIRE(spanCtx.trace_flags() == trace::TraceFlags(ctx->traceFlags));

    auto trace_id_hex = BytesToHexStr(ctx->traceID, TraceId::kSize);
    auto span_id_hex = BytesToHexStr(ctx->spanID, SpanId::kSize);
    auto trace_id_str = GetIDFromHexStr(trace_id_hex);
    auto span_id_str = GetIDFromHexStr(span_id_hex);

    REQUIRE(strncmp((char*)ctx->traceID, trace_id_str.c_str(), TraceId::kSize) == 0);
    REQUIRE(strncmp((char*)ctx->spanID, span_id_str.c_str(), SpanId::kSize) == 0);

    delete[] ctx->traceID;
    delete[] ctx->spanID;
}
