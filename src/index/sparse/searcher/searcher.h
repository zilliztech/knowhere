#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "knowhere/bitsetview.h"
#include "knowhere/heap.h"

namespace knowhere::sparse::inverted {

inline constexpr uint32_t kBulkWindowSize = 1U << 12;

using SparseHeap = knowhere::ResultMinHeap<float, uint32_t>;

struct FilterBounds {
    uint32_t lower_bound;
    uint32_t upper_bound;
};

enum class PostingFilterMode : uint8_t {
    // The bitset is empty, so postings can advance without any filter checks.
    None,
    // Backend IDs map directly to an in-range bitset. The posting cursor can
    // use BitsetView::test_direct() without repeated mapping or bounds checks.
    Direct,
    // The non-empty bitset needs the full BitsetView::test() semantics, such
    // as output-ID mapping, an ID offset, or out-of-range handling.
    Generic,
};

inline PostingFilterMode
GetPostingFilterMode(const BitsetView& bitset, size_t required_bits) {
    // An empty BitsetView represents an unfiltered search.
    if (bitset.empty()) {
        return PostingFilterMode::None;
    }
    // Direct is safe only when supports_direct_test() has established all of
    // test_direct()'s invariants; every other non-empty filter is Generic.
    return bitset.supports_direct_test(required_bits) ? PostingFilterMode::Direct : PostingFilterMode::Generic;
}

// Dispatch posting advancement using the query's filter mode.
template <PostingFilterMode FilterMode, typename Cursor>
inline void
NextPosting(Cursor& cursor) noexcept {
    if constexpr (FilterMode == PostingFilterMode::None) {
        cursor.next_unfiltered();
    } else if constexpr (FilterMode == PostingFilterMode::Direct) {
        cursor.next_direct_filtered();
    } else {
        cursor.next();
    }
}

template <PostingFilterMode FilterMode, typename Cursor>
inline void
NextPostingGEQ(Cursor& cursor, uint32_t vec_id) noexcept {
    if (cursor.vec_id() >= vec_id) {
        return;
    }
    if constexpr (FilterMode == PostingFilterMode::None) {
        cursor.next_geq_unfiltered(vec_id);
    } else if constexpr (FilterMode == PostingFilterMode::Direct) {
        cursor.next_geq_direct_filtered(vec_id);
    } else {
        cursor.next_geq(vec_id);
    }
}

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
