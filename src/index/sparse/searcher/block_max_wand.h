// Block-Max WAND (BMW) searcher.
// Derived from the PISA search engine (Performant Indexes and Search for Academia).
//   Paper: S. Ding and T. Suel, "Faster Top-k Document Retrieval Using Block-Max Indexes",
//          SIGIR, 2011.
//   Repository: https://github.com/pisa-engine/pisa
//   License: Apache License 2.0

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "index/sparse/block_max_data.h"
#include "index/sparse/scorer.h"
#include "index/sparse/searcher/searcher.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

template <typename IndexType>
class BlockMaxWandSearcher : public RankedSearcher {
 public:
    struct Cursor {
        typename IndexType::posting_list_iterator index_cursor;
        DimScorer scorer;
        float max_score;
        BlockMaxDataCursor block_max_data_cursor;
        float weight;

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

        [[nodiscard]] float
        block_max_score() const noexcept {
            return block_max_data_cursor.score() * weight;
        }

        [[nodiscard]] uint32_t
        block_max_vec_id() const noexcept {
            return block_max_data_cursor.vec_id();
        }

        void
        block_max_next_geq(uint32_t vec_id) {
            block_max_data_cursor.next_geq(vec_id);
        }
    };

    explicit BlockMaxWandSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                                  const std::shared_ptr<IndexScorer>& search_scorer, const uint32_t k,
                                  const uint32_t max_vec_id, const BitsetView& bitset, float dim_max_score_ratio)
        : RankedSearcher(k),
          cursors_(make_cursors(index, query, search_scorer, bitset, dim_max_score_ratio)),
          max_vec_id_(max_vec_id) {
    }

    void
    search() override {
        std::vector<Cursor*> ordered_cursors;
        ordered_cursors.reserve(cursors_.size());
        for (auto& en : cursors_) {
            ordered_cursors.push_back(&en);
        }

        auto sort_cursors = [&]() {
            // sort enumerators by increasing vector id
            std::sort(ordered_cursors.begin(), ordered_cursors.end(),
                      [](Cursor* lhs, Cursor* rhs) { return lhs->vec_id() < rhs->vec_id(); });
        };

        sort_cursors();

        while (true) {
            // find pivot
            float upper_bound = 0.F;
            size_t pivot;
            bool found_pivot = false;
            uint32_t pivot_id = max_vec_id_;

            for (pivot = 0; pivot < ordered_cursors.size(); ++pivot) {
                if (ordered_cursors[pivot]->vec_id() >= max_vec_id_) {
                    break;
                }

                upper_bound += ordered_cursors[pivot]->max_score;
                if (topk_.WouldEnter(upper_bound)) {
                    found_pivot = true;
                    pivot_id = ordered_cursors[pivot]->vec_id();
                    for (; pivot + 1 < ordered_cursors.size() && ordered_cursors[pivot + 1]->vec_id() == pivot_id;
                         ++pivot) {
                    }
                    break;
                }
            }

            // no pivot found, we can stop the search
            if (!found_pivot) {
                break;
            }

            float block_upper_bound = 0;

            for (size_t i = 0; i < pivot + 1; ++i) {
                if (ordered_cursors[i]->block_max_vec_id() < pivot_id) {
                    ordered_cursors[i]->block_max_next_geq(pivot_id);
                }

                block_upper_bound += ordered_cursors[i]->block_max_score();
            }

            if (topk_.WouldEnter(block_upper_bound)) {
                // check if pivot is a possible match
                if (pivot_id == ordered_cursors[0]->vec_id()) {
                    float score = 0;
                    for (Cursor* en : ordered_cursors) {
                        if (en->vec_id() != pivot_id) {
                            break;
                        }
                        float part_score = en->score();
                        score += part_score;
                        block_upper_bound -= en->block_max_score() - part_score;
                        if (!topk_.WouldEnter(block_upper_bound)) {
                            break;
                        }
                    }
                    for (Cursor* en : ordered_cursors) {
                        if (en->vec_id() != pivot_id) {
                            break;
                        }
                        en->next();
                    }

                    topk_.Push(score, pivot_id);
                    // resort by vector id
                    sort_cursors();

                } else {
                    uint32_t next_list = pivot;
                    for (; ordered_cursors[next_list]->vec_id() == pivot_id; --next_list) {
                    }
                    ordered_cursors[next_list]->next_geq(pivot_id);

                    // bubble down the advanced list
                    for (size_t i = next_list + 1; i < ordered_cursors.size(); ++i) {
                        if (ordered_cursors[i]->vec_id() <= ordered_cursors[i - 1]->vec_id()) {
                            std::swap(ordered_cursors[i], ordered_cursors[i - 1]);
                        } else {
                            break;
                        }
                    }
                }

            } else {
                uint32_t next;
                uint32_t next_list = pivot;

                float max_weight = ordered_cursors[next_list]->max_score;

                for (uint32_t i = 0; i < pivot; i++) {
                    if (ordered_cursors[i]->max_score > max_weight) {
                        next_list = i;
                        max_weight = ordered_cursors[i]->max_score;
                    }
                }

                next = max_vec_id_;

                for (size_t i = 0; i <= pivot; ++i) {
                    if (ordered_cursors[i]->block_max_vec_id() < next) {
                        next = ordered_cursors[i]->block_max_vec_id();
                    }
                }

                next = next + 1;
                if (pivot + 1 < ordered_cursors.size() && ordered_cursors[pivot + 1]->vec_id() < next) {
                    next = ordered_cursors[pivot + 1]->vec_id();
                }

                if (next <= pivot_id) {
                    next = pivot_id + 1;
                }

                ordered_cursors[next_list]->next_geq(next);

                // bubble down the advanced list
                for (size_t i = next_list + 1; i < ordered_cursors.size(); ++i) {
                    if (ordered_cursors[i]->vec_id() < ordered_cursors[i - 1]->vec_id()) {
                        std::swap(ordered_cursors[i], ordered_cursors[i - 1]);
                    } else {
                        break;
                    }
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
                                     dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val),
                                     index.get_block_max_data_cursor(dim_id), dim_val});
        }
        return cursors;
    }

    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
};

}  // namespace knowhere::sparse::inverted
