// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under
// the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific language
// governing permissions and limitations under the License.

#pragma once

#include "knowhere/config.h"

namespace knowhere {

class FaissConfig : public BaseConfig {
 public:
    // Required. faiss DSL understood by faiss::index_factory (fp32) or
    // faiss::index_binary_factory (bin1). Examples: "Flat", "IVF1024,PQ16x8",
    // "HNSW32,Flat", "BIVF256,Hamming".
    CFG_STRING faiss_index_name;

    // Captured subset of the incoming JSON: only keys that this config's __DICT__
    // does NOT declare (i.e. not owned by Knowhere's native config layer). Those are
    // the keys the vanilla faiss adapter forwards to faiss::ParameterSpace
    // (build) and per-family SearchParametersXxx (search). Declared keys (k,
    // metric_type, trace_id, faiss_index_name, ...) are consumed by Config::Load
    // into typed fields and therefore filtered out of raw_params at capture time.
    Json raw_params;

    KNOHWERE_DECLARE_CONFIG(FaissConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(faiss_index_name)
            .description("faiss factory string, e.g. \"IVF1024,PQ16x8\"")
            .allow_empty_without_default()
            .for_train()
            .for_deserialize()
            .for_deserialize_from_file();
    }

    void
    CaptureRawJson(const Json& json) override {
        raw_params = Json::object();
        for (auto it = json.begin(); it != json.end(); ++it) {
            // Skip any key already declared as a typed field on BaseConfig or
            // FaissConfig — those are Knowhere's own and will be consumed by
            // Config::Load. Everything else is a faiss-bound knob we forward.
            if (__DICT__.count(it.key()) == 0) {
                raw_params[it.key()] = it.value();
            }
        }
    }
};

}  // namespace knowhere
