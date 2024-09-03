// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "IndexConditionalWrapper.h"

#include <cstddef>
#include <cstdint>

#include "faiss/IndexCosine.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexRefine.h"
#include "index/hnsw/impl/IndexBruteForceWrapper.h"
#include "index/hnsw/impl/IndexHNSWWrapper.h"
#include "index/hnsw/impl/IndexWrapperCosine.h"
#include "knowhere/utils.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

// Decides whether a brute force should be used instead of a regular HNSW search.
// This may be applicable in case of very large topk values or
//   extremely high filtering levels.
std::optional<bool>
WhetherPerformBruteForceSearch(const faiss::Index* index, const BaseConfig& cfg, const BitsetView& bitset) {
    // check if parameters have all we need
    if (!cfg.k.has_value() || index == nullptr) {
        return std::nullopt;
    }

    // decide
    const auto k = cfg.k.value();

    if (k >= (index->ntotal * HnswSearchThresholds::kHnswSearchBFTopkThreshold)) {
        return true;
    }

    if (!bitset.empty()) {
        const size_t filtered_out_num = bitset.count();
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        double ratio = ((double)filtered_out_num) / bitset.size();
        knowhere::knowhere_hnsw_bitset_ratio.Observe(ratio);
#endif
        if (filtered_out_num >= (index->ntotal * HnswSearchThresholds::kHnswSearchKnnBFFilterThreshold) ||
            k >= (index->ntotal - filtered_out_num) * HnswSearchThresholds::kHnswSearchBFTopkThreshold) {
            return true;
        }
    }

    // the default value
    return false;
}

// Decides whether a brute force should be used instead of a regular HNSW range search.
// This may be applicable in case of very large topk values or
//   extremely high filtering levels.
std::optional<bool>
WhetherPerformBruteForceRangeSearch(const faiss::Index* index, const FaissHnswConfig& cfg, const BitsetView& bitset) {
    // check if parameters have all we need
    if (!cfg.ef.has_value() || index == nullptr) {
        return std::nullopt;
    }

    // decide
    const auto ef = cfg.ef.value();

    if (ef >= (index->ntotal * HnswSearchThresholds::kHnswSearchBFTopkThreshold)) {
        return true;
    }

    if (!bitset.empty()) {
        const size_t filtered_out_num = bitset.count();
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
        double ratio = ((double)filtered_out_num) / bitset.size();
        knowhere::knowhere_hnsw_bitset_ratio.Observe(ratio);
#endif
        if (filtered_out_num >= (index->ntotal * HnswSearchThresholds::kHnswSearchRangeBFFilterThreshold) ||
            ef >= (index->ntotal - filtered_out_num) * HnswSearchThresholds::kHnswSearchRangeBFFilterThreshold) {
            return true;
        }
    }

    // the default value
    return false;
}

// returns nullptr in case of invalid index
std::tuple<std::unique_ptr<faiss::Index>, bool>
create_conditional_hnsw_wrapper(faiss::Index* index, const FaissHnswConfig& hnsw_cfg, const bool whether_bf_search) {
    const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), knowhere::metric::COSINE);

    // use knowhere-based search by default
    const bool override_faiss_search = hnsw_cfg.override_faiss_search.value_or(true);
    // const bool override_faiss_search = false;

    // check if we have a refine available.
    faiss::IndexRefine* const index_refine = dynamic_cast<faiss::IndexRefine*>(index);

    if (index_refine != nullptr) {
        // yes, it is possible to refine results.

        // cast a base index to IndexHNSW-based index
        faiss::IndexHNSW* const index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index_refine->base_index);

        if (index_hnsw == nullptr) {
            // this is unexpected
            return {nullptr, false};
        }

        // select a wrapper index, which is safe to delete without deleting
        //   an original index
        std::unique_ptr<faiss::Index> base_wrapper;

        if (!override_faiss_search) {
            // use the original index
            base_wrapper = std::make_unique<faiss::cppcontrib::knowhere::IndexWrapper>(index_hnsw);
        } else if (whether_bf_search) {
            // use brute-force wrapper
            base_wrapper = std::make_unique<knowhere::IndexBruteForceWrapper>(index_hnsw);
        } else {
            // use hnsw-search wrapper
            base_wrapper = std::make_unique<knowhere::IndexHNSWWrapper>(index_hnsw);
        }

        // check if a user wants a refined result
        if (hnsw_cfg.refine_k.has_value()) {
            // yes, a user wants to perform a refine

            // thus, we need to define a new refine index and pass
            //   wrapper_searcher into its ownership

            // is it a cosine index?
            if (index_hnsw->storage->is_cosine && is_cosine) {
                // yes, wrap both base and refine index
                std::unique_ptr<knowhere::IndexWrapperCosine> cosine_wrapper =
                    std::make_unique<knowhere::IndexWrapperCosine>(
                        index_refine->refine_index,
                        dynamic_cast<faiss::HasInverseL2Norms*>(index_hnsw->storage)->get_inverse_l2_norms());

                // create a temporary refine index
                std::unique_ptr<faiss::IndexRefine> refine_wrapper =
                    std::make_unique<faiss::IndexRefine>(base_wrapper.get(), cosine_wrapper.get());

                // transfer ownership
                refine_wrapper->own_fields = true;
                base_wrapper.release();

                refine_wrapper->own_refine_index = true;
                cosine_wrapper.release();

                // done
                return {std::move(refine_wrapper), true};
            } else {
                // no, wrap base index only.

                // create a temporary refine index
                std::unique_ptr<faiss::IndexRefine> refine_wrapper =
                    std::make_unique<faiss::IndexRefine>(base_wrapper.get(), index_refine->refine_index);

                // transfer ownership
                refine_wrapper->own_fields = true;
                base_wrapper.release();

                // done
                return {std::move(refine_wrapper), true};
            }
        } else {
            // no, a user wants to skip a refine

            // return a wrapper
            return {std::move(base_wrapper), false};
        }
    } else {
        // cast to IndexHNSW-based index
        faiss::IndexHNSW* const index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index);

        if (index_hnsw == nullptr) {
            // this is unexpected
            return {nullptr, false};
        }

        // select a wrapper index for search
        std::unique_ptr<faiss::Index> base_wrapper;

        if (!override_faiss_search) {
            // use the original index
            base_wrapper = std::make_unique<faiss::cppcontrib::knowhere::IndexWrapper>(index_hnsw);
        } else if (whether_bf_search) {
            // use brute-force wrapper
            base_wrapper = std::make_unique<knowhere::IndexBruteForceWrapper>(index_hnsw);
        } else {
            // use hnsw-search wrapper
            base_wrapper = std::make_unique<knowhere::IndexHNSWWrapper>(index_hnsw);
        }

        return {std::move(base_wrapper), false};
    }
}

}  // namespace knowhere
