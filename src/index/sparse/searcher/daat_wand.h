// Document-at-a-Time (DAAT) WAND (Weak AND) searcher.
// Derived from the PISA search engine (Performant Indexes and Search for Academia).
//   Paper: A. Broder et al., "Efficient Query Evaluation using a Two-Level Retrieval Process",
//          CIKM, 2003.
//   Repository: https://github.com/pisa-engine/pisa
//   License: Apache License 2.0

#pragma once

#include <algorithm>
#include <cassert>
#include <utility>
#include <vector>

#include "index/sparse/scorer.h"
#include "index/sparse/searcher/searcher.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

template <typename IndexType>
class DaatWandSearcher : public RankedSearcher {
 public:
    struct Cursor {
        typename IndexType::posting_list_iterator index_cursor;
        DimScorer scorer;
        float max_score;
        float qval_p1;

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

        void
        next_geq(uint32_t vec_id) noexcept {
            index_cursor.next_geq(vec_id);
        }

        [[nodiscard]] bool
        valid() const noexcept {
            return index_cursor.valid();
        }
    };

    explicit DaatWandSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                              const std::shared_ptr<IndexScorer>& search_scorer, const uint32_t k,
                              const uint32_t max_vec_id, const BitsetView& bitset, float dim_max_score_ratio)
        : RankedSearcher(k),
          filter_bounds_(GetFilterBounds(bitset, max_vec_id)),
          cursors_(make_cursors(index, query, search_scorer, bitset, dim_max_score_ratio, filter_bounds_)),
          max_vec_id_(filter_bounds_.upper_bound),
          row_sums_(index.get_row_sums()),
          scorer_type_(search_scorer->config().scorer_type) {
        if (scorer_type_ == IndexScorerType::BM25) {
            const auto* bm25_scorer = dynamic_cast<const BM25IndexScorer*>(search_scorer.get());
            assert(bm25_scorer != nullptr);
            bm25_p2_ = bm25_scorer->p2();
            bm25_p3_ = bm25_scorer->p3();
        }
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }

        if (scorer_type_ == IndexScorerType::BM25) {
            run<IndexScorerType::BM25>();
        } else {
            run<IndexScorerType::IP>();
        }
    }

 private:
    template <IndexScorerType ScorerType>
    void
    run() {
        std::vector<Cursor*> ordered_cursors;
        ordered_cursors.reserve(cursors_.size());
        for (auto& en : cursors_) {
            ordered_cursors.push_back(&en);
        }

        auto sort_cursors = [&]() {
            // sort cursors by increasing vec_id
            std::sort(ordered_cursors.begin(), ordered_cursors.end(),
                      [](Cursor* lhs, Cursor* rhs) { return lhs->vec_id() < rhs->vec_id(); });
        };

        sort_cursors();
        while (true) {
            // find pivot
            float upper_bound = 0;
            size_t pivot = 0;
            bool found_pivot = false;
            for (pivot = 0; pivot < ordered_cursors.size(); ++pivot) {
                if (ordered_cursors[pivot]->vec_id() >= max_vec_id_) {
                    break;
                }
                upper_bound += ordered_cursors[pivot]->max_score;
                if (this->topk_.WouldEnter(upper_bound)) {  // Access base class topk_
                    found_pivot = true;
                    break;
                }
            }

            // no pivot found, we can stop the search
            if (!found_pivot) {
                break;
            }

            // check if pivot is a possible match
            uint64_t pivot_id = ordered_cursors[pivot]->vec_id();
            if (pivot_id == ordered_cursors[0]->vec_id()) {
                float score = 0;
                float document_component = 0.0F;
                if constexpr (ScorerType == IndexScorerType::BM25) {
                    document_component = bm25_p3_ * row_sums_[pivot_id];
                }
                for (Cursor* en : ordered_cursors) {
                    if (en->vec_id() != pivot_id) {
                        break;
                    }
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        const float tf = static_cast<float>(en->index_cursor.val());
                        score += bm25_score_with_document_component(en->qval_p1, tf, bm25_p2_, document_component);
                    } else {
                        score += en->score();
                    }
                    en->next();
                }
                this->topk_.Push(score, pivot_id);  // Access base class topk_
                // resort by vector id
                sort_cursors();
            } else {
                // no match, move farthest list up to the pivot
                uint64_t next_list = pivot;
                for (; ordered_cursors[next_list]->vec_id() == pivot_id; --next_list) {
                }
                ordered_cursors[next_list]->next_geq(pivot_id);
                // bubble down the advanced list
                for (size_t i = next_list + 1; i < ordered_cursors.size(); ++i) {
                    if (ordered_cursors[i]->vec_id() >= ordered_cursors[i - 1]->vec_id()) {
                        break;
                    }
                    std::swap(ordered_cursors[i], ordered_cursors[i - 1]);
                }
            }
        }
    }

    static std::vector<Cursor>
    make_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                 const std::shared_ptr<IndexScorer>& index_scorer, const BitsetView& bitset, float dim_max_score_ratio,
                 const FilterBounds& filter_bounds) {
        std::vector<Cursor> cursors;
        cursors.reserve(query.size());
        const BM25IndexScorer* bm25_scorer = nullptr;
        if (index_scorer->config().scorer_type == IndexScorerType::BM25) {
            bm25_scorer = dynamic_cast<const BM25IndexScorer*>(index_scorer.get());
            assert(bm25_scorer != nullptr);
        }
        for (const auto& [dim_id, dim_val] : query) {
            cursors.push_back(Cursor{GetFilteredPostingListCursor(index, dim_id, bitset, filter_bounds),
                                     index_scorer->dim_scorer(dim_val),
                                     dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val),
                                     bm25_scorer != nullptr ? dim_val * bm25_scorer->p1() : 0.0F});
        }
        return cursors;
    }

    FilterBounds filter_bounds_;
    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
    const std::vector<float>& row_sums_;
    IndexScorerType scorer_type_;
    float bm25_p2_{0.0F};
    float bm25_p3_{0.0F};
};
}  // namespace knowhere::sparse::inverted
