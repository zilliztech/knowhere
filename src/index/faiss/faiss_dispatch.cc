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

#include "index/faiss/faiss_dispatch.h"

#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/cppcontrib/knowhere/SearchParamsDispatch.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>

namespace knowhere::faiss_vanilla {

namespace {

// Coerce a json value into a double for faiss consumption. Accepts:
//   - numbers:  e.g. 16, 16.0        -> 16.0
//   - booleans: true / false         -> 1.0 / 0.0
//   - stringified numbers: "16"      -> 16.0
//   - stringified booleans: "true"   -> 1.0
// Rejects arrays, objects, null, and unparseable strings. Matches the spirit of
// knowhere::Config::FormatAndCheck's string-to-typed coercion for declared fields,
// so forwarded keys behave consistently with native Knowhere keys.
Status
coerce_to_double(const Json& v, const std::string& key, double* out, std::string* err_msg) {
    if (v.is_number()) {
        *out = v.get<double>();
        return Status::success;
    }
    if (v.is_boolean()) {
        *out = v.get<bool>() ? 1.0 : 0.0;
        return Status::success;
    }
    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        if (s == "true") {
            *out = 1.0;
            return Status::success;
        }
        if (s == "false") {
            *out = 0.0;
            return Status::success;
        }
        try {
            size_t pos = 0;
            double parsed = std::stod(s, &pos);
            if (pos == s.size()) {
                *out = parsed;
                return Status::success;
            }
        } catch (const std::invalid_argument&) {
        } catch (const std::out_of_range&) {
        }
    }
    if (err_msg) {
        *err_msg = "faiss vanilla: param '" + key + "' expects a number or boolean; got " + v.dump();
    }
    return Status::invalid_args;
}

// Apply every key in raw_params to the faiss index. raw_params has already been
// filtered by FaissConfig::CaptureRawJson to exclude keys owned by Knowhere's own
// config layer (fields declared via KNOWHERE_CONFIG_DECLARE_FIELD). We pre-validate
// the remaining keys against the faiss-owned whitelist (supported_build_param_names
// + "quantizer_*" prefix handling) before calling ParameterSpace. A key that fails
// the whitelist (typo, non-faiss param) is rejected with a clear error; a key that
// passes the whitelist but is incompatible with the concrete index type (e.g.
// nprobe on an HNSW) is still caught by ParameterSpace's exception and surfaced
// as invalid_args.
template <typename IndexT>
Status
apply_impl(IndexT* index, const Json& raw_params, std::string* err_msg) {
    ::faiss::ParameterSpace ps;
    for (auto it = raw_params.begin(); it != raw_params.end(); ++it) {
        const std::string& key = it.key();
        if (!::faiss::cppcontrib::knowhere::is_supported_build_param(key)) {
            if (err_msg) {
                *err_msg = "faiss vanilla: build param '" + key + "' is not recognized";
            }
            return Status::invalid_args;
        }
        double val = 0.0;
        auto cst = coerce_to_double(it.value(), key, &val, err_msg);
        if (cst != Status::success) {
            return cst;
        }
        try {
            ps.set_index_parameter(index, key, val);
        } catch (const ::faiss::FaissException& e) {
            if (err_msg) {
                *err_msg = std::string("faiss rejected param '") + key + "': " + e.what();
            }
            return Status::invalid_args;
        }
    }
    return Status::success;
}

// Shared logic for search-param builders. `index` can be faiss::Index* or IndexBinary*.
// raw_params has already been filtered by FaissConfig::CaptureRawJson to contain only
// keys NOT declared by Knowhere's typed config. Uses the faiss-owned whitelist
// (supported_search_params) to validate remaining keys, and delegates both the
// SearchParameters-family selection and the per-name field set to the upstream
// helper. Knowhere layer only adds: (1) sel attach, (2) JSON->double conversion,
// (3) clear error wording.
template <typename IndexT>
Status
build_search_params_impl(const IndexT* index, const Json& raw_params, ::faiss::IDSelector* sel,
                         std::unique_ptr<::faiss::SearchParameters>* out, std::string* err_msg) {
    auto params = ::faiss::cppcontrib::knowhere::make_search_params(index);
    params->sel = sel;

    const auto supported = ::faiss::cppcontrib::knowhere::supported_search_params(index);
    for (auto it = raw_params.begin(); it != raw_params.end(); ++it) {
        const std::string& key = it.key();
        if (!supported.count(key)) {
            if (err_msg) {
                *err_msg = "faiss vanilla: search param '" + key + "' not supported for this index family";
            }
            return Status::invalid_args;
        }
        double val = 0.0;
        auto cst = coerce_to_double(it.value(), key, &val, err_msg);
        if (cst != Status::success) {
            return cst;
        }
        // Whitelist already guarantees try_set_search_param returns true; treat a
        // false here as an invariant breach rather than user error.
        (void)::faiss::cppcontrib::knowhere::try_set_search_param(params.get(), key, val);
    }
    *out = std::move(params);
    return Status::success;
}

}  // namespace

Status
apply_build_params(::faiss::Index* index, const Json& raw_params, std::string* err_msg) {
    return apply_impl(index, raw_params, err_msg);
}

Status
apply_build_params(::faiss::IndexBinary* index, const Json& raw_params, std::string* err_msg) {
    return apply_impl(index, raw_params, err_msg);
}

Status
build_search_params(const ::faiss::Index* index, const Json& raw_params, ::faiss::IDSelector* sel,
                    std::unique_ptr<::faiss::SearchParameters>* out, std::string* err_msg) {
    return build_search_params_impl(index, raw_params, sel, out, err_msg);
}

Status
build_search_params(const ::faiss::IndexBinary* index, const Json& raw_params, ::faiss::IDSelector* sel,
                    std::unique_ptr<::faiss::SearchParameters>* out, std::string* err_msg) {
    // IndexBinaryIVF requires SearchParametersIVF; binary side also does not honor
    // IDSelector, so attaching sel here is typically a no-op at search time.
    return build_search_params_impl(index, raw_params, sel, out, err_msg);
}

}  // namespace knowhere::faiss_vanilla
