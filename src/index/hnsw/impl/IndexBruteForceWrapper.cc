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

#include "index/hnsw/impl/IndexBruteForceWrapper.h"

#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <faiss/cppcontrib/knowhere/MetricType.h>
#include <faiss/cppcontrib/knowhere/impl/Bruteforce.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>

#include <algorithm>
#include <cstring>
#include <memory>

#include "knowhere/bitsetview.h"
#include "knowhere/bitsetview_idselector.h"

namespace knowhere {

using idx_t = faiss::idx_t;

// the following structure is a hack, because GCC cannot properly
//   de-virtualize a plain BitsetViewIDSelector.
struct BitsetViewIDSelectorWrapper final {
    const BitsetView bitset_view;

    inline BitsetViewIDSelectorWrapper(BitsetView bitset_view) : bitset_view{bitset_view} {
    }

    [[nodiscard]] inline bool
    is_member(faiss::idx_t id) const {
        // it is by design that bitset_view.empty() is not tested here
        return (!bitset_view.test(id));
    }
};

// Enumerate the zero bits (accepted ids) a machine word at a time. The regular
// brute-force loop checks every id, which makes bitset traversal itself dominate
// highly selective filters even though very few distances are evaluated.
template <typename C>
void
brute_force_search_sparse_bitset(const BitsetView& bitset_view, const idx_t ntotal,
                                 faiss::DistanceComputer& __restrict qdis, const idx_t k,
                                 float* __restrict distances, idx_t* __restrict labels) {
    auto heap = std::make_unique<std::pair<float, idx_t>[]>(k);
    idx_t n_added = 0;

    const size_t backend_count = std::min<size_t>(static_cast<size_t>(ntotal), bitset_view.size());
    const size_t public_begin = bitset_view.id_offset();
    const size_t public_end =
        public_begin < bitset_view.num_bits()
            ? public_begin + std::min(backend_count, bitset_view.num_bits() - public_begin)
            : public_begin;
    if (public_begin < public_end) {
        const size_t first_word = public_begin >> 6;
        const size_t last_word = (public_end - 1) >> 6;
        const size_t byte_size = bitset_view.byte_size();
        const uint8_t* const bits = bitset_view.data();

        for (size_t word_index = first_word; word_index <= last_word; ++word_index) {
            const size_t byte_offset = word_index * sizeof(uint64_t);
            const size_t available =
                byte_offset < byte_size ? std::min(sizeof(uint64_t), byte_size - byte_offset) : 0;
            uint64_t filtered = ~uint64_t{0};
            if (available != 0) {
                filtered = 0;
                std::memcpy(&filtered, bits + byte_offset, available);
                if (available < sizeof(uint64_t)) {
                    filtered |= ~uint64_t{0} << (available * 8);
                }
            }

            uint64_t valid = ~filtered;
            if (word_index == first_word) {
                valid &= ~uint64_t{0} << (public_begin & 63);
            }
            if (word_index == last_word) {
                const size_t tail_bits = ((public_end - 1) & 63) + 1;
                if (tail_bits != 64) {
                    valid &= (uint64_t{1} << tail_bits) - 1;
                }
            }

            while (valid != 0) {
                const size_t bit = static_cast<size_t>(__builtin_ctzll(valid));
                const idx_t idx = static_cast<idx_t>((word_index << 6) + bit - public_begin);
                const float distance = qdis(idx);
                if (n_added < k) {
                    ++n_added;
                    faiss::heap_push<C>(n_added, heap.get(), distance, idx);
                } else if (C::cmp(heap[0].first, distance)) {
                    faiss::heap_replace_top<C>(k, heap.get(), distance, idx);
                }
                valid &= valid - 1;
            }
        }
    }

    const idx_t len = std::min(n_added, k);
    for (idx_t i = 0; i < len; ++i) {
        labels[len - i - 1] = heap[0].second;
        distances[len - i - 1] = heap[0].first;
        faiss::heap_pop<C>(len - i, heap.get());
    }
    for (idx_t i = len; i < k; ++i) {
        labels[i] = -1;
        distances[i] = C::neutral();
    }
}

//
IndexBruteForceWrapper::IndexBruteForceWrapper(faiss::Index* underlying_index)
    : faiss::cppcontrib::knowhere::IndexWrapper{underlying_index} {
}

void
IndexBruteForceWrapper::search(faiss::idx_t n, const float* __restrict x, faiss::idx_t k, float* __restrict distances,
                               faiss::idx_t* __restrict labels,
                               const faiss::SearchParameters* __restrict params) const {
    FAISS_THROW_IF_NOT(k > 0);

    std::unique_ptr<faiss::DistanceComputer> dis(index->get_distance_computer());

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        // prepare the query
        dis->set_query(x + i * index->d);

        // allocate heap
        idx_t* const __restrict local_ids = labels + i * k;
        float* const __restrict local_distances = distances + i * k;

        // set up a filter
        faiss::IDSelector* sel = (params == nullptr) ? nullptr : params->sel;

        if (faiss::cppcontrib::knowhere::is_similarity_metric(index->metric_type)) {
            using C = faiss::CMin<float, idx_t>;

            // try knowhere-specific filter
            if (const knowhere::BitsetViewIDSelector* __restrict bw_idselector =
                    dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);
                bw_idselector && !bw_idselector->bitset_view.empty()) {
                if (!bw_idselector->bitset_view.has_out_ids()) {
                    brute_force_search_sparse_bitset<C>(
                        bw_idselector->bitset_view, index->ntotal, *dis, k, local_distances, local_ids);
                } else {
                    BitsetViewIDSelectorWrapper bw_idselector_w(bw_idselector->bitset_view);
                    faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer,
                                                                         BitsetViewIDSelectorWrapper>(
                        index->ntotal, *dis, bw_idselector_w, k, local_distances, local_ids);
                }
            } else {
                faiss::IDSelectorAll sel_all;
                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, k, local_distances, local_ids);
            }
        } else {
            using C = faiss::CMax<float, idx_t>;

            // try knowhere-specific filter
            if (const knowhere::BitsetViewIDSelector* __restrict bw_idselector =
                    dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);
                bw_idselector && !bw_idselector->bitset_view.empty()) {
                if (!bw_idselector->bitset_view.has_out_ids()) {
                    brute_force_search_sparse_bitset<C>(
                        bw_idselector->bitset_view, index->ntotal, *dis, k, local_distances, local_ids);
                } else {
                    BitsetViewIDSelectorWrapper bw_idselector_w(bw_idselector->bitset_view);
                    faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer,
                                                                         BitsetViewIDSelectorWrapper>(
                        index->ntotal, *dis, bw_idselector_w, k, local_distances, local_ids);
                }
            } else {
                faiss::IDSelectorAll sel_all;
                faiss::cppcontrib::knowhere::brute_force_search_impl<C, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, k, local_distances, local_ids);
            }
        }
    }
}

void
IndexBruteForceWrapper::range_search(faiss::idx_t n, const float* x, float radius, faiss::RangeSearchResult* result,
                                     const faiss::SearchParameters* params) const {
    using RH_min = faiss::RangeSearchBlockResultHandler<faiss::CMax<float, int64_t>>;
    using RH_max = faiss::RangeSearchBlockResultHandler<faiss::CMin<float, int64_t>>;
    RH_min bres_min(result, radius);
    RH_max bres_max(result, radius);

    std::unique_ptr<faiss::DistanceComputer> dis(index->get_distance_computer());

    // no parallelism by design
    for (idx_t i = 0; i < n; i++) {
        // prepare the query
        dis->set_query(x + i * index->d);

        // set up a filter
        faiss::IDSelector* __restrict sel = (params == nullptr) ? nullptr : params->sel;

        // If `sel` is a knowhere BitsetViewIDSelector wrapping an empty bitset, BitsetView::test
        // returns true for every id (out_id >= num_bits_=0 short-circuits), so is_member returns
        // false for every id and the BF range_search would emit zero results. Fall back to
        // IDSelectorAll in that case, mirroring the guard already present in
        // IndexBruteForceWrapper::search above.
        const knowhere::BitsetViewIDSelector* __restrict bw_idselector =
            dynamic_cast<const knowhere::BitsetViewIDSelector*>(sel);
        const bool sel_accepts_all = (sel == nullptr) || (bw_idselector && bw_idselector->bitset_view.empty());

        if (faiss::cppcontrib::knowhere::is_similarity_metric(index->metric_type)) {
            typename RH_max::SingleResultHandler res_max(bres_max);
            res_max.begin(i);

            if (sel_accepts_all) {
                // Compiler is expected to de-virtualize virtual method calls
                faiss::IDSelectorAll sel_all;

                faiss::cppcontrib::knowhere::brute_force_range_search_impl<
                    typename RH_max::SingleResultHandler, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, res_max);
            } else {
                faiss::cppcontrib::knowhere::brute_force_range_search_impl<typename RH_max::SingleResultHandler,
                                                                           faiss::DistanceComputer, faiss::IDSelector>(
                    index->ntotal, *dis, *sel, res_max);
            }

            res_max.end();
        } else {
            typename RH_min::SingleResultHandler res_min(bres_min);
            res_min.begin(i);

            if (sel_accepts_all) {
                // Compiler is expected to de-virtualize virtual method calls
                faiss::IDSelectorAll sel_all;

                faiss::cppcontrib::knowhere::brute_force_range_search_impl<
                    typename RH_min::SingleResultHandler, faiss::DistanceComputer, faiss::IDSelectorAll>(
                    index->ntotal, *dis, sel_all, res_min);
            } else {
                faiss::cppcontrib::knowhere::brute_force_range_search_impl<typename RH_min::SingleResultHandler,
                                                                           faiss::DistanceComputer, faiss::IDSelector>(
                    index->ntotal, *dis, *sel, res_min);
            }

            res_min.end();
        }
    }
}

}  // namespace knowhere
