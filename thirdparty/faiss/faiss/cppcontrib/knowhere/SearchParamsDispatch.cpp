// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <faiss/cppcontrib/knowhere/SearchParamsDispatch.h>

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>

#ifdef FAISS_ENABLE_SVS
#include <faiss/svs/IndexSVSVamana.h>
#endif

namespace faiss::cppcontrib::knowhere {

namespace {

// The wrapper subclasses SearchParametersPreTransform and
// IndexRefineSearchParameters hold raw (non-owning) pointers to the nested
// sub-params (per faiss header comments: "non owning"). When we build them via
// make_search_params we want the caller to own the whole tree with a single
// unique_ptr, so we bundle the inner unique_ptr into these Owning* subclasses.
// The Owning flavors decay to the base faiss types for all consumers (faiss's
// search() implementations see them as the base class).
struct OwningSearchParametersPreTransform
        : ::faiss::SearchParametersPreTransform {
    std::unique_ptr<::faiss::SearchParameters> inner_owned;
};
struct OwningIndexRefineSearchParameters
        : ::faiss::IndexRefineSearchParameters {
    std::unique_ptr<::faiss::SearchParameters> inner_owned;
};

bool try_set_ivf(
        ::faiss::SearchParametersIVF* p,
        const std::string& name,
        double val) {
    if (name == "nprobe") {
        p->nprobe = static_cast<size_t>(val);
        return true;
    }
    if (name == "max_codes") {
        p->max_codes = static_cast<size_t>(val);
        return true;
    }
    return false;
}

bool try_set_hnsw(
        ::faiss::SearchParametersHNSW* p,
        const std::string& name,
        double val) {
    if (name == "efSearch") {
        p->efSearch = static_cast<int>(val);
        return true;
    }
    if (name == "check_relative_distance") {
        p->check_relative_distance = val != 0.0;
        return true;
    }
    if (name == "bounded_queue") {
        p->bounded_queue = val != 0.0;
        return true;
    }
    return false;
}

bool try_set_pq(
        ::faiss::SearchParametersPQ* p,
        const std::string& name,
        double val) {
    if (name == "polysemous_ht") {
        p->polysemous_ht = static_cast<int>(val);
        return true;
    }
    if (name == "search_type") {
        p->search_type = static_cast<::faiss::IndexPQ::Search_type_t>(
                static_cast<int>(val));
        return true;
    }
    return false;
}

#ifdef FAISS_ENABLE_SVS
bool try_set_svs_vamana(
        ::faiss::SearchParametersSVSVamana* p,
        const std::string& name,
        double val) {
    if (name == "search_window_size") {
        p->search_window_size = static_cast<size_t>(val);
        return true;
    }
    if (name == "search_buffer_capacity") {
        p->search_buffer_capacity = static_cast<size_t>(val);
        return true;
    }
    return false;
}
#endif

} // namespace

std::unique_ptr<::faiss::SearchParameters> make_search_params(
        const ::faiss::Index* index) {
    // Wrapper: PreTransform (OPQ, PCA, etc.). Recurse to inner index; no own
    // knobs.
    if (auto* pt = dynamic_cast<const ::faiss::IndexPreTransform*>(index)) {
        auto inner = make_search_params(pt->index);
        auto p = std::make_unique<OwningSearchParametersPreTransform>();
        p->index_params = inner.get();
        p->inner_owned = std::move(inner);
        return p;
    }
    // Wrapper: Refine (RFlat, Refine(...)). Has k_factor knob at this layer.
    if (auto* rfn = dynamic_cast<const ::faiss::IndexRefine*>(index)) {
        auto inner = make_search_params(rfn->base_index);
        auto p = std::make_unique<OwningIndexRefineSearchParameters>();
        p->base_index_params = inner.get();
        p->inner_owned = std::move(inner);
        return p;
    }
    // Leaf families (order matters only for disjoint casts; these are mutually
    // exclusive).
    if (dynamic_cast<const ::faiss::IndexHNSW*>(index)) {
        return std::make_unique<::faiss::SearchParametersHNSW>();
    }
    if (dynamic_cast<const ::faiss::IndexIVF*>(index)) {
        return std::make_unique<::faiss::SearchParametersIVF>();
    }
    if (dynamic_cast<const ::faiss::IndexPQ*>(index)) {
        return std::make_unique<::faiss::SearchParametersPQ>();
    }
#ifdef FAISS_ENABLE_SVS
    if (dynamic_cast<const ::faiss::IndexSVSVamana*>(index)) {
        // Catches IndexSVSVamana, IndexSVSVamanaLVQ, IndexSVSVamanaLeanVec.
        return std::make_unique<::faiss::SearchParametersSVSVamana>();
    }
#endif
    return std::make_unique<::faiss::SearchParameters>();
}

std::unique_ptr<::faiss::SearchParameters> make_search_params(
        const ::faiss::IndexBinary* index) {
    // IndexBinaryIVF::search uses dynamic_cast to SearchParametersIVF; giving
    // it a plain SearchParameters would fail the check. Binary IVF also does
    // not honor IDSelector, so callers should not set `sel` on the returned
    // object for BIVF.
    if (dynamic_cast<const ::faiss::IndexBinaryIVF*>(index)) {
        return std::make_unique<::faiss::SearchParametersIVF>();
    }
    return std::make_unique<::faiss::SearchParameters>();
}

// ---------- supported-name whitelists (query-only, no mutation) ----------

namespace {

// Names accepted by try_set_ivf above. Keep in sync.
const std::set<std::string>& ivf_names() {
    static const std::set<std::string> kNames = {"nprobe", "max_codes"};
    return kNames;
}

const std::set<std::string>& hnsw_names() {
    static const std::set<std::string> kNames = {
            "efSearch", "check_relative_distance", "bounded_queue"};
    return kNames;
}

const std::set<std::string>& pq_names() {
    static const std::set<std::string> kNames = {
            "polysemous_ht", "search_type"};
    return kNames;
}

#ifdef FAISS_ENABLE_SVS
const std::set<std::string>& svs_vamana_names() {
    static const std::set<std::string> kNames = {
            "search_window_size", "search_buffer_capacity"};
    return kNames;
}
#endif

} // namespace

std::set<std::string> supported_search_params(const ::faiss::Index* index) {
    // Wrappers: union of own knobs and inner supported set.
    if (auto* pt = dynamic_cast<const ::faiss::IndexPreTransform*>(index)) {
        return supported_search_params(pt->index); // no own knobs
    }
    if (auto* rfn = dynamic_cast<const ::faiss::IndexRefine*>(index)) {
        auto out = supported_search_params(rfn->base_index);
        out.insert("k_factor");
        return out;
    }
    if (dynamic_cast<const ::faiss::IndexHNSW*>(index)) {
        return hnsw_names();
    }
    if (dynamic_cast<const ::faiss::IndexIVF*>(index)) {
        return ivf_names();
    }
    if (dynamic_cast<const ::faiss::IndexPQ*>(index)) {
        return pq_names();
    }
#ifdef FAISS_ENABLE_SVS
    if (dynamic_cast<const ::faiss::IndexSVSVamana*>(index)) {
        return svs_vamana_names();
    }
#endif
    return {}; // plain Index: only sel, no named knobs
}

std::set<std::string> supported_search_params(
        const ::faiss::IndexBinary* index) {
    if (dynamic_cast<const ::faiss::IndexBinaryIVF*>(index)) {
        return ivf_names();
    }
    return {};
}

std::set<std::string> supported_build_param_names() {
    // Mirror the hardcoded if-chain in
    // faiss::ParameterSpace::set_index_parameter (AutoTune.cpp).
    // "quantizer_<name>" is handled via prefix in is_supported_build_param.
    static const std::set<std::string> kNames = {
            "nprobe",
            "ht",
            "k_factor",
            "max_codes",
            "prune_headroom",
            "efConstruction",
            "efSearch",
    };
    return kNames;
}

bool is_supported_build_param(const std::string& name) {
    if (supported_build_param_names().count(name)) {
        return true;
    }
    // ParameterSpace recursively forwards keys starting with "quantizer_" into
    // the coarse quantizer of an IVF index. Validate the suffix against the
    // same list.
    constexpr const char kQuantizerPrefix[] = "quantizer_";
    constexpr size_t kPrefixLen = sizeof(kQuantizerPrefix) - 1;
    if (name.compare(0, kPrefixLen, kQuantizerPrefix) == 0) {
        return is_supported_build_param(name.substr(kPrefixLen));
    }
    return false;
}

// ---------- runtime setter (walks into wrappers) ----------

bool try_set_search_param(
        ::faiss::SearchParameters* params,
        const std::string& name,
        double val) {
    // Wrappers first: try this layer's own knobs, then recurse to inner params.
    if (auto* pt =
                dynamic_cast<::faiss::SearchParametersPreTransform*>(params)) {
        // PreTransform has no own knobs; forward to inner.
        return pt->index_params &&
                try_set_search_param(pt->index_params, name, val);
    }
    if (auto* rfn =
                dynamic_cast<::faiss::IndexRefineSearchParameters*>(params)) {
        if (name == "k_factor") {
            rfn->k_factor = static_cast<float>(val);
            return true;
        }
        return rfn->base_index_params &&
                try_set_search_param(rfn->base_index_params, name, val);
    }
    // Leaves.
    if (auto* ivf = dynamic_cast<::faiss::SearchParametersIVF*>(params)) {
        return try_set_ivf(ivf, name, val);
    }
    if (auto* hnsw = dynamic_cast<::faiss::SearchParametersHNSW*>(params)) {
        return try_set_hnsw(hnsw, name, val);
    }
    if (auto* pq = dynamic_cast<::faiss::SearchParametersPQ*>(params)) {
        return try_set_pq(pq, name, val);
    }
#ifdef FAISS_ENABLE_SVS
    if (auto* svs = dynamic_cast<::faiss::SearchParametersSVSVamana*>(params)) {
        return try_set_svs_vamana(svs, name, val);
    }
#endif
    return false;
}

} // namespace faiss::cppcontrib::knowhere
