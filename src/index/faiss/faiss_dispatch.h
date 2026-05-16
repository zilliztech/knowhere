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

#include <memory>
#include <string>

#include "knowhere/config.h"

namespace faiss {
struct Index;
struct IndexBinary;
struct IDSelector;
struct SearchParameters;
}  // namespace faiss

namespace knowhere::faiss_vanilla {

// Forwards keys from raw_params to faiss::ParameterSpace::set_index_parameter
// on the given index. Converts faiss exceptions into Status::invalid_args with the
// faiss message in *err_msg.
Status
apply_build_params(::faiss::Index* index, const Json& raw_params, std::string* err_msg);

Status
apply_build_params(::faiss::IndexBinary* index, const Json& raw_params, std::string* err_msg);

// Build a per-request SearchParameters* appropriate for the concrete faiss index
// family. The family dispatch itself lives in faiss::cppcontrib::knowhere (upstream-
// bound helper); this wrapper adds: (1) sel assignment, (2) framework-key filtering,
// (3) JSON value extraction + unknown-key error surfacing.
Status
build_search_params(const ::faiss::Index* index, const Json& raw_params, ::faiss::IDSelector* sel,
                    std::unique_ptr<::faiss::SearchParameters>* out, std::string* err_msg);

Status
build_search_params(const ::faiss::IndexBinary* index, const Json& raw_params, ::faiss::IDSelector* sel,
                    std::unique_ptr<::faiss::SearchParameters>* out, std::string* err_msg);

}  // namespace knowhere::faiss_vanilla
