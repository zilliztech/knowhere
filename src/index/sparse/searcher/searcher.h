#pragma once

#include <algorithm>
#include <cstdint>

#include "knowhere/bitsetview.h"
#include "knowhere/heap.h"

namespace knowhere::sparse::inverted {

using SparseHeap = knowhere::ResultMinHeap<float, uint32_t>;

struct FilterBounds {
    uint32_t lower_bound;
    uint32_t upper_bound;
};

inline FilterBounds
GetFilterBounds(const BitsetView& bitset, uint32_t max_vec_id) {
    if (bitset.empty()) {
        return {0, max_vec_id};
    }

    const auto last_valid = bitset.previous_valid_index(max_vec_id);
    if (!last_valid.has_value()) {
        return {0, 0};
    }

    const auto upper_bound = static_cast<uint32_t>(*last_valid + 1);
    const auto lower_bound = static_cast<uint32_t>(bitset.get_first_valid_index());
    return {std::min(lower_bound, upper_bound), upper_bound};
}

template <typename IndexType>
typename IndexType::posting_list_iterator
GetFilteredPostingListCursor(const IndexType& index, uint32_t dim_id, const BitsetView& bitset,
                             const FilterBounds& bounds) {
    if (!bitset.empty()) {
        if constexpr (requires {
                          index.get_dim_plist_cursor(dim_id, bitset, bounds.lower_bound, bounds.upper_bound);
                      }) {
            return index.get_dim_plist_cursor(dim_id, bitset, bounds.lower_bound, bounds.upper_bound);
        }
    }
    return index.get_dim_plist_cursor(dim_id, bitset);
}

// Base searcher interface
class Searcher {
 public:
    virtual ~Searcher() = default;
    virtual void
    search() = 0;
};

// Common base implementation for all searchers
class RankedSearcher : public Searcher {
 public:
    RankedSearcher(uint32_t k) : topk_(k) {
    }

    std::vector<typename SparseHeap::entry_type> const&
    topk() {
        topk_.Finalize();
        return topk_.Results();
    }

 protected:
    SparseHeap topk_;
};

}  // namespace knowhere::sparse::inverted
