// Block-Max MaxScore (BMM) searcher.
// Derived from the PISA search engine (Performant Indexes and Search for Academia).
//   Paper: S. Ding and T. Suel, "Faster Top-k Document Retrieval Using Block-Max Indexes",
//          SIGIR, 2011.
//          H. Turtle and J. Flood, "Query Evaluation: Strategies and Optimizations,"
//          Information Processing & Management, 1995.
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

template <typename IndexType, typename QueryScorer>
class BlockMaxMaxScoreSearcher : public RankedSearcher {
 public:
    using DimScorer = decltype(std::declval<const QueryScorer&>().dim_scorer(0.0f));

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

    explicit BlockMaxMaxScoreSearcher(const IndexType& index, std::vector<std::pair<uint32_t, float>>& query,
                                      const QueryScorer& search_scorer, const uint32_t k, const uint32_t max_vec_id,
                                      const BitsetView& bitset, float dim_max_score_ratio)
        : RankedSearcher(k),
          cursors_([&]() {
              std::sort(query.begin(), query.end(), [&](auto& a, auto& b) {
                  return index.get_dim_max_score(a.first, a.second) > index.get_dim_max_score(b.first, b.second);
              });
              return make_cursors(index, query, search_scorer, bitset, dim_max_score_ratio);
          }()),
          max_vec_id_(max_vec_id) {
    }

    void
    search() override {
        std::vector<float> upper_bounds(cursors_.size() + 1, 0.0f);
        float bound_sum = 0.0f;
        for (size_t i = cursors_.size() - 1; i + 1 > 0; --i) {
            bound_sum += cursors_[i].max_score;
            upper_bounds[i] = bound_sum;
        }

        float threshold = topk_.Threshold();

        uint32_t ne_start_cursor_id = cursors_.size();
        uint32_t curr_cand_vec_id = (*std::min_element(cursors_.begin(), cursors_.end(), [](auto&& lhs, auto&& rhs) {
                                        return lhs.vec_id() < rhs.vec_id();
                                    })).vec_id();

        std::vector<int64_t> search_times;
        while (ne_start_cursor_id > 0 && curr_cand_vec_id < max_vec_id_) {
            float score = 0;
            uint32_t next_cand_vec_id = max_vec_id_;

            // score essential list and find next
            for (size_t i = 0; i < ne_start_cursor_id; ++i) {
                if (cursors_[i].vec_id() == curr_cand_vec_id) {
                    score += cursors_[i].score();
                    cursors_[i].next();
                }
                if (cursors_[i].vec_id() < next_cand_vec_id) {
                    next_cand_vec_id = cursors_[i].vec_id();
                }
            }

            auto new_score = score + upper_bounds[ne_start_cursor_id];
            if (new_score > threshold) {
                // update block index for non-essential list and check block upper bound
                for (size_t i = ne_start_cursor_id; i < cursors_.size(); ++i) {
                    if (cursors_[i].block_max_vec_id() < curr_cand_vec_id) {
                        cursors_[i].block_max_next_geq(curr_cand_vec_id);
                    }
                    new_score -= cursors_[i].max_score - cursors_[i].block_max_score();
                    if (new_score <= threshold) {
                        break;
                    }
                }
                if (new_score > threshold) {
                    // try to complete evaluation with non-essential lists
                    for (size_t i = ne_start_cursor_id; i < cursors_.size(); ++i) {
                        cursors_[i].next_geq(curr_cand_vec_id);
                        if (cursors_[i].vec_id() == curr_cand_vec_id) {
                            new_score += cursors_[i].score();
                        }
                        new_score -= cursors_[i].block_max_score();

                        if (new_score <= threshold) {
                            break;
                        }
                    }
                    score = new_score;
                }
                if (score > threshold) {
                    topk_.Push(score, curr_cand_vec_id);
                    threshold = topk_.Threshold();
                    // update non-essential lists
                    while (ne_start_cursor_id != 0 && upper_bounds[ne_start_cursor_id - 1] <= threshold) {
                        --ne_start_cursor_id;
                    }
                }
            }
            curr_cand_vec_id = next_cand_vec_id;
        }
    }

 private:
    static std::vector<Cursor>
    make_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                 const QueryScorer& index_scorer, const BitsetView& bitset, float dim_max_score_ratio) {
        std::vector<Cursor> cursors;
        cursors.reserve(query.size());
        for (const auto& [dim_id, dim_val] : query) {
            cursors.push_back(Cursor{index.get_dim_plist_cursor(dim_id, bitset), index_scorer.dim_scorer(dim_val),
                                     dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val),
                                     index.get_block_max_data_cursor(dim_id), dim_val});
        }
        return cursors;
    }

    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
};
}  // namespace knowhere::sparse::inverted
