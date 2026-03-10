// Term-at-a-Time (TAAT) naive searcher.
// Derived from the PISA search engine (Performant Indexes and Search for Academia).
//   Repository: https://github.com/pisa-engine/pisa
//   License: Apache License 2.0

#pragma once

#include <utility>
#include <vector>

#include "index/sparse/scorer.h"
#include "index/sparse/searcher/searcher.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

template <typename IndexType>
class TaatNaiveSearcher : public RankedSearcher {
 public:
    struct Cursor {
        typename IndexType::posting_list_iterator index_cursor;
        DimScorer scorer;

        [[nodiscard]] uint32_t
        vec_id() const noexcept {
            return index_cursor.vec_id();
        }

        [[nodiscard]] float
        score() noexcept {
            return scorer(index_cursor.vec_id(), index_cursor.val());
        }

        void
        next() noexcept {
            index_cursor.next();
        }

        [[nodiscard]] size_t
        size() const noexcept {
            return index_cursor.size();
        }

        [[nodiscard]] bool
        valid() const noexcept {
            return index_cursor.valid();
        }
    };

    explicit TaatNaiveSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                               std::shared_ptr<IndexScorer> search_scorer, const uint32_t k, const uint32_t max_vec_id,
                               const BitsetView& bitset)
        : RankedSearcher(k), cursors_(make_cursors(index, query, search_scorer, bitset)), max_vec_id_(max_vec_id) {
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }

        std::vector<float> distances(max_vec_id_, 0.0f);

        for (auto& en : cursors_) {
            while (en.valid()) {
                distances[en.vec_id()] += en.score();
                en.next();
            }
        }

        // Sparse vectors cannot guarantee topk results, because a value of 0.0
        // indicates that two sparse vectors have no intersection, so such cases
        // need to be excluded.
        for (size_t i = 0; i < distances.size(); ++i) {
            if (distances[i] != 0.0f) {
                topk_.Push(distances[i], i);
            }
        }
    }

 private:
    static std::vector<Cursor>
    make_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                 const std::shared_ptr<IndexScorer>& index_scorer, const BitsetView& bitset) {
        std::vector<Cursor> cursors;
        cursors.reserve(query.size());
        for (const auto& [dim_id, dim_val] : query) {
            cursors.push_back(Cursor{index.get_dim_plist_cursor(dim_id, bitset), index_scorer->dim_scorer(dim_val)});
        }
        return cursors;
    }

    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
};

}  // namespace knowhere::sparse::inverted
