// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

// Generic per-family SearchParameters helpers. Intended to be upstreamable:
// the knowhere project uses it today, but the design is index-family aware
// yet metric/config agnostic — it could live in main faiss if accepted.
//
// Motivation: a C++ caller wanting a per-request SearchParameters object for
// an arbitrary faiss::Index currently has to hand-roll a family dispatch
// (IVF -> SearchParametersIVF, HNSW -> SearchParametersHNSW, etc.) and
// recurse through wrapper indexes (PreTransform, Refine). This header
// centralizes that dispatch and exposes two primitives:
//
//   1. make_search_params(index)
//        Returns a unique_ptr<SearchParameters> of the correct concrete type
//        for the given index, including recursive inner params for wrapper
//        indexes. Ownership of nested params is held inside the returned
//        object so the caller can treat it as a single unique_ptr.
//
//   2. try_set_search_param(params, name, value)
//        Sets a named runtime knob on the given SearchParameters object,
//        walking into nested sub-params for wrapper classes. Returns whether
//        the name was recognized. Intended for loops over user-supplied
//        key/value config — caller handles "unknown key -> error".

#pragma once

#include <memory>
#include <set>
#include <string>

namespace faiss {
struct Index;
struct IndexBinary;
struct SearchParameters;
} // namespace faiss

namespace faiss::cppcontrib::knowhere {

// ---------- Search-param dispatch ----------

// Construct the appropriate SearchParameters subclass for the given index
// family. For wrapper indexes (PreTransform, Refine) this recurses into the
// inner index. Ownership of any nested SearchParameters is held by the returned
// unique_ptr.
std::unique_ptr<::faiss::SearchParameters> make_search_params(
        const ::faiss::Index* index);

// Binary variant. IndexBinaryIVF needs SearchParametersIVF; everything else
// uses base.
std::unique_ptr<::faiss::SearchParameters> make_search_params(
        const ::faiss::IndexBinary* index);

// Set a named runtime knob. Walks into PreTransform / Refine wrappers.
// Returns true if recognized and applied by some layer, false otherwise.
// double is used as the common value type (matches faiss::ParameterSpace).
bool try_set_search_param(
        ::faiss::SearchParameters* params,
        const std::string& name,
        double val);

// Returns the whitelist of search-time parameter names recognized by
// try_set_search_param for this index. Includes wrapper-level knobs
// (e.g. k_factor for IndexRefine) plus the inner family's knobs.
// Callers should use this to pre-validate user-supplied params.
std::set<std::string> supported_search_params(const ::faiss::Index* index);

std::set<std::string> supported_search_params(
        const ::faiss::IndexBinary* index);

// ---------- Build-param dispatch ----------

// Returns whether faiss::ParameterSpace::set_index_parameter would recognize
// the given name (i.e. it appears in faiss's own hardcoded if-chain, including
// the "quantizer_" prefix for recursing into coarse quantizers). Useful to pre-
// validate user-supplied build-time knobs before forwarding to ParameterSpace.
bool is_supported_build_param(const std::string& name);

// The fixed set of base names recognized by
// ParameterSpace::set_index_parameter. Note: "quantizer_<name>" where <name> is
// one of these is also supported via ParameterSpace's recursion — use
// is_supported_build_param for that case.
//
// The set is index-family agnostic, exactly like the if-chain it mirrors: a
// name that belongs to another family (nprobe on an HNSW, alpha on an IVF)
// passes this check and is then rejected by set_index_parameter itself.
std::set<std::string> supported_build_param_names();

// ---------- Per-index capability queries ----------
//
// Unlike the name list above, these answer for a concrete index, so a caller
// can apply a knob only where it exists instead of catching a rejection. All of
// them look through the wrappers ParameterSpace looks through.

// Whether `index` honors "is_static", i.e. has an immutable backend that
// consumes all of its data in the first add().
bool supports_static_index(const ::faiss::Index* index);

// Whether `index` honors "store_vectors", i.e. keeps a copy of the raw vectors
// that only backs reconstruct() and can be skipped.
bool supports_dropping_stored_vectors(const ::faiss::Index* index);

// Whether `index` implements train_with_queries(), i.e. can learn from a
// representative query sample alongside the database vectors (out-of-
// distribution training). The base method is a silent no-op, so callers need
// this to tell "supported" from "ignored".
bool supports_train_with_queries(const ::faiss::Index* index);

} // namespace faiss::cppcontrib::knowhere
