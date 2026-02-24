// Document-at-a-Time (DAAT) WAND (Weak AND) searcher.
// Derived from the PISA search engine (Performant Indexes and Search for Academia).
//   Paper: A. Broder et al., "Efficient Query Evaluation using a Two-Level Retrieval Process",
//          CIKM, 2003.
//   Repository: https://github.com/pisa-engine/pisa
//   License: Apache License 2.0

#pragma once

#include <algorithm>
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
          cursors_(make_cursors(index, query, search_scorer, bitset, dim_max_score_ratio)),
          max_vec_id_(max_vec_id) {
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }

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
                for (Cursor* en : ordered_cursors) {
                    if (en->vec_id() != pivot_id) {
                        break;
                    }
                    score += en->score();
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

 private:
    static std::vector<Cursor>
    make_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                 const std::shared_ptr<IndexScorer>& index_scorer, const BitsetView& bitset,
                 float dim_max_score_ratio) {
        std::vector<Cursor> cursors;
        cursors.reserve(query.size());
        for (const auto& [dim_id, dim_val] : query) {
            cursors.push_back(Cursor{index.get_dim_plist_cursor(dim_id, bitset), index_scorer->dim_scorer(dim_val),
                                     dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val)});
        }
        return cursors;
    }

    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
};
}  // namespace knowhere::sparse::inverted
