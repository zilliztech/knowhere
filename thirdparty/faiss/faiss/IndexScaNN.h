#pragma once

#include <faiss/Index.h>
#include <faiss/IndexRefine.h>

namespace faiss {

struct IndexScaNN : IndexRefineFlat {
    explicit IndexScaNN(Index* base_index);
    IndexScaNN(Index* base_index, const float* xb);

    IndexScaNN();

    int64_t size();

    void search_thread_safe(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const size_t nprobe,
            const size_t reorder_k,
            const BitsetView bitset = nullptr) const;

    void range_search_thread_safe(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const BitsetView bitset = nullptr) const;
};

} // namespace faiss