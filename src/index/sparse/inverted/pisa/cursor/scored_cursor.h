#pragma once

#include <algorithm>
#include <memory>

#include "index/sparse/inverted/pisa/index_scorer.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::pisa {

template <typename IndexCursor>
class ScoredCursor {
 public:
    using base_cursor_type = IndexCursor;

    ScoredCursor(IndexCursor cursor, DimScorer scorer) : index_cursor_(std::move(cursor)), scorer_(scorer) {
    }
    ScoredCursor(ScoredCursor const&) = delete;
    ScoredCursor(ScoredCursor&&) noexcept = default;
    ScoredCursor&
    operator=(ScoredCursor const&) = delete;
    ScoredCursor&
    operator=(ScoredCursor&&) noexcept = default;
    ~ScoredCursor() = default;

    [[nodiscard]] uint32_t
    vec_id() const noexcept {
        return index_cursor_.vec_id();
    }

    [[nodiscard]] float
    val() noexcept {
        return index_cursor_.val();
    }

    [[nodiscard]] float
    score() noexcept {
        return scorer_(vec_id(), val());
    }

    void
    next() noexcept {
        index_cursor_.next();
    }

    void
    next_geq(uint32_t vec_id) noexcept {
        index_cursor_.next_geq(vec_id);
    }

    [[nodiscard]] size_t
    size() const noexcept {
        return index_cursor_.size();
    }

    [[nodiscard]] bool
    valid() const noexcept {
        return index_cursor_.valid();
    }

 private:
    IndexCursor index_cursor_;
    DimScorer scorer_;
};

template <typename IndexType>
auto
make_scored_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                    std::shared_ptr<IndexScorer> index_scorer, const BitsetView& bitset) {
    using Cursor = ScoredCursor<typename IndexType::posting_list_iterator>;
    std::vector<Cursor> cursors;
    cursors.reserve(query.size());

    std::transform(query.begin(), query.end(), std::back_inserter(cursors),
                   [&](const std::pair<uint32_t, float>& qitem) {
                       auto dim_id = qitem.first;
                       auto dim_val = qitem.second;
                       return Cursor(index.get_plist_cursor(dim_id, bitset), index_scorer->dim_scorer(dim_val));
                   });

    return cursors;
}

}  // namespace knowhere::sparse::pisa
