#pragma once

#include <memory>
#include <vector>

#include "index/sparse/inverted/pisa/cursor/scored_cursor.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::pisa {

template <typename IndexCursor>
class MaxScoredCursor : public ScoredCursor<IndexCursor> {
 public:
    using base_cursor_type = IndexCursor;

    MaxScoredCursor(IndexCursor cursor, DimScorer scorer, float max_score)
        : ScoredCursor<IndexCursor>(std::move(cursor), std::move(scorer)), max_score_(max_score) {
    }
    MaxScoredCursor(MaxScoredCursor const&) = delete;
    MaxScoredCursor(MaxScoredCursor&&) noexcept = default;
    MaxScoredCursor&
    operator=(MaxScoredCursor const&) = delete;
    MaxScoredCursor&
    operator=(MaxScoredCursor&&) noexcept = default;
    ~MaxScoredCursor() = default;

    [[nodiscard]] float
    max_score() const noexcept {
        return max_score_;
    }

 private:
    float max_score_;
};

template <typename IndexType>
auto
make_max_scored_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                        std::shared_ptr<IndexScorer> index_scorer, const BitsetView& bitset) {
    using Cursor = MaxScoredCursor<typename IndexType::posting_list_iterator>;
    std::vector<Cursor> cursors;
    cursors.reserve(query.size());

    std::transform(query.begin(), query.end(), std::back_inserter(cursors),
                   [&](const std::pair<uint32_t, float>& qitem) {
                       auto dim_id = qitem.first;
                       auto dim_val = qitem.second;
                       return Cursor(index.get_plist_cursor(dim_id, bitset), index_scorer->dim_scorer(dim_val),
                                     index.get_dim_max_score(dim_id, dim_val));
                   });

    return cursors;
}

}  // namespace knowhere::sparse::pisa
