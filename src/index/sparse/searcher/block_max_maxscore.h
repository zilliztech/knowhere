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
#include <cassert>
#include <utility>
#include <vector>

#include "index/sparse/block_max_data.h"
#include "index/sparse/scorer.h"
#include "index/sparse/searcher/searcher.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

template <typename IndexType>
class BlockMaxMaxScoreSearcher : public RankedSearcher {
 public:
    struct Cursor {
        typename IndexType::posting_list_iterator index_cursor;
        DimScorer scorer;
        float max_score;
        BlockMaxDataCursor block_max_data_cursor;
        float weight;
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
                                      const std::shared_ptr<IndexScorer>& search_scorer, const uint32_t k,
                                      const uint32_t max_vec_id, const BitsetView& bitset, float dim_max_score_ratio,
                                      size_t bulk_query_nnz_threshold)
        : RankedSearcher(k),
          filter_bounds_(GetFilterBounds(bitset, max_vec_id)),
          bm25_context_(index.get_row_sums(), *search_scorer),
          cursors_([&]() {
              std::sort(query.begin(), query.end(), [&](auto& a, auto& b) {
                  return index.get_dim_max_score(a.first, a.second) > index.get_dim_max_score(b.first, b.second);
              });
              return make_cursors(index, query, search_scorer, bm25_context_, bitset, dim_max_score_ratio,
                                  filter_bounds_);
          }()),
          max_vec_id_(filter_bounds_.upper_bound),
          scorer_type_(search_scorer->config().scorer_type),
          bulk_query_nnz_threshold_(bulk_query_nnz_threshold) {
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }

        if (scorer_type_ == IndexScorerType::BM25) {
            run_sorted<IndexScorerType::BM25>();
        } else {
            run_sorted<IndexScorerType::IP>();
        }
    }

 private:
    [[nodiscard]] std::vector<float>
    calc_upper_bounds() const {
        std::vector<float> upper_bounds(cursors_.size() + 1, 0.0f);
        float bound_sum = 0.0f;
        for (size_t i = cursors_.size() - 1; i + 1 > 0; --i) {
            bound_sum += cursors_[i].max_score;
            upper_bounds[i] = bound_sum;
        }
        return upper_bounds;
    }

    template <IndexScorerType ScorerType>
    void
    run_sorted_linear() {
        auto upper_bounds = calc_upper_bounds();

        float threshold = topk_.Threshold();

        size_t ne_start_cursor_id = cursors_.size();
        uint32_t curr_cand_vec_id = (*std::min_element(cursors_.begin(), cursors_.end(), [](auto&& lhs, auto&& rhs) {
                                        return lhs.vec_id() < rhs.vec_id();
                                    })).vec_id();

        while (ne_start_cursor_id > 0 && curr_cand_vec_id < max_vec_id_) {
            float score = 0;
            uint32_t next_cand_vec_id = max_vec_id_;
            float doc_norm = 0.0F;
            if constexpr (ScorerType == IndexScorerType::BM25) {
                doc_norm = bm25_context_.doc_norm(curr_cand_vec_id);
            }

            auto score_term = [&](Cursor& cursor) -> float {
                if constexpr (ScorerType == IndexScorerType::BM25) {
                    const float tf = static_cast<float>(cursor.index_cursor.val());
                    return bm25_context_.score(cursor.qval_p1, tf, doc_norm);
                } else {
                    return cursor.score();
                }
            };

            // score essential list and find next
            for (size_t i = 0; i < ne_start_cursor_id; ++i) {
                if (cursors_[i].vec_id() == curr_cand_vec_id) {
                    score += score_term(cursors_[i]);
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
                            new_score += score_term(cursors_[i]);
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

    template <IndexScorerType ScorerType>
    void
    run_sorted_bulk() {
        // Batch essential postings in a fixed-size inner window. Non-essential block maxima are cached over their
        // common validity range, so block boundaries tighten candidate completion without fragmenting the score
        // window into many tiny ranges for long queries.
        auto upper_bounds = calc_upper_bounds();
        auto above_threshold = [&](float score) { return topk_.WouldEnter(score); };

        size_t essential_count = cursors_.size();
        uint32_t cached_block_window_max = 0;
        float cached_block_upper_bound = 0.0F;
        std::vector<float> cached_block_scores(cursors_.size(), 0.0F);

        auto update_non_essential_lists = [&]() {
            const size_t previous_essential_count = essential_count;
            while (essential_count != 0 && !above_threshold(upper_bounds[essential_count - 1])) {
                --essential_count;
            }
            if (essential_count != previous_essential_count) {
                // A newly non-essential cursor changes both the block sum and its common validity range.
                cached_block_window_max = 0;
            }
            return essential_count != 0;
        };

        if (!update_non_essential_lists()) {
            return;
        }

        auto score_term = [&](Cursor& cursor, float doc_norm) -> float {
            if constexpr (ScorerType == IndexScorerType::BM25) {
                const float tf = static_cast<float>(cursor.index_cursor.val());
                return bm25_context_.score(cursor.qval_p1, tf, doc_norm);
            } else {
                return cursor.score();
            }
        };

        auto refresh_block_upper_bound = [&](uint32_t vec_id) {
            if (vec_id < cached_block_window_max) {
                return;
            }

            cached_block_upper_bound = 0.0F;
            cached_block_window_max = max_vec_id_;
            for (size_t i = essential_count; i < cursors_.size(); ++i) {
                auto& cursor = cursors_[i];
                float block_score = 0.0F;
                if (cursor.vec_id() < max_vec_id_) {
                    // There cannot be a match before the posting cursor. Advancing the block cursor to that posting
                    // skips filtered/exhausted blocks while keeping one bound valid through the selected block end.
                    const uint32_t block_lower_bound = std::max(vec_id, cursor.vec_id());
                    if (cursor.block_max_vec_id() < block_lower_bound) {
                        cursor.block_max_next_geq(block_lower_bound);
                    }
                    block_score = cursor.block_max_score();
                    const uint32_t block_window_max = static_cast<uint32_t>(
                        std::min<uint64_t>(max_vec_id_, static_cast<uint64_t>(cursor.block_max_vec_id()) + 1));
                    cached_block_window_max = std::min(cached_block_window_max, block_window_max);
                }
                cached_block_scores[i] = block_score;
                cached_block_upper_bound += block_score;
            }
        };

        auto complete_candidate = [&](uint32_t vec_id, float score, float doc_norm) {
            // Preserve MaxScore's (possibly ratio-adjusted) global gate before paying for block metadata.
            if (!above_threshold(score + upper_bounds[essential_count])) {
                return;
            }

            refresh_block_upper_bound(vec_id);
            if (!above_threshold(score + cached_block_upper_bound)) {
                return;
            }

            float remaining_block_upper_bound = cached_block_upper_bound;
            for (size_t i = essential_count; i < cursors_.size(); ++i) {
                auto& cursor = cursors_[i];
                cursor.next_geq(vec_id);
                if (cursor.vec_id() == vec_id) {
                    score += score_term(cursor, doc_norm);
                }
                remaining_block_upper_bound = std::max(0.0F, remaining_block_upper_bound - cached_block_scores[i]);
                if (!above_threshold(score + remaining_block_upper_bound)) {
                    return;
                }
            }
            topk_.Push(score, vec_id);
        };

        // Keep these buffers uninitialized for queries that reduce to a single essential posting list.
        std::array<float, kBulkWindowSize> window_scores;
        std::array<uint64_t, kBulkWindowSize / 64> window_matches;
        bool window_buffers_initialized = false;

        while (true) {
            uint32_t window_min = max_vec_id_;
            uint32_t second_vec_id = max_vec_id_;
            size_t lead_index = 0;
            for (size_t i = 0; i < essential_count; ++i) {
                const uint32_t vec_id = cursors_[i].vec_id();
                if (vec_id < window_min) {
                    second_vec_id = window_min;
                    window_min = vec_id;
                    lead_index = i;
                } else if (vec_id < second_vec_id) {
                    second_vec_id = vec_id;
                }
            }
            if (window_min >= max_vec_id_) [[unlikely]] {
                return;
            }

            const uint32_t window_max = static_cast<uint32_t>(
                std::min<uint64_t>(max_vec_id_, static_cast<uint64_t>(window_min) + kBulkWindowSize));

            // A lone lead list needs no score buffer. The same direct path wins while all other essential lists are
            // at least half a window away.
            const bool single_lead_range =
                essential_count == 1 || static_cast<uint64_t>(window_min) + kBulkWindowSize / 2 <= second_vec_id;
            if (single_lead_range) {
                const uint32_t range_max = essential_count == 1 ? window_max : std::min(window_max, second_vec_id);
                auto& cursor = cursors_[lead_index];
                while (cursor.vec_id() < range_max) {
                    const uint32_t vec_id = cursor.vec_id();
                    float doc_norm = 0.0F;
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        doc_norm = bm25_context_.doc_norm(vec_id);
                    }
                    const float score = score_term(cursor, doc_norm);
                    cursor.next();
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        if (cursor.vec_id() < max_vec_id_) {
                            bm25_context_.prefetch_document(cursor.vec_id());
                        }
                    }
                    complete_candidate(vec_id, score, doc_norm);
                }
            } else {
                if (!window_buffers_initialized) {
                    window_scores.fill(0.0F);
                    window_matches.fill(0);
                    window_buffers_initialized = true;
                }

                for (size_t i = 0; i < essential_count; ++i) {
                    auto& cursor = cursors_[i];
                    while (cursor.vec_id() < window_max) {
                        const uint32_t vec_id = cursor.vec_id();
                        float doc_norm = 0.0F;
                        if constexpr (ScorerType == IndexScorerType::BM25) {
                            doc_norm = bm25_context_.doc_norm(vec_id);
                        }
                        const uint32_t offset = vec_id - window_min;
                        window_scores[offset] += score_term(cursor, doc_norm);
                        window_matches[offset / 64] |= uint64_t{1} << (offset % 64);
                        cursor.next();
                    }
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        if (cursor.vec_id() < max_vec_id_) {
                            bm25_context_.prefetch_document(cursor.vec_id());
                        }
                    }
                }

                const size_t window_word_count = (static_cast<size_t>(window_max - window_min) + 63) / 64;
                for (size_t word_index = 0; word_index < window_word_count; ++word_index) {
                    uint64_t matches = std::exchange(window_matches[word_index], 0);
                    while (matches != 0) {
                        const uint32_t bit_index = static_cast<uint32_t>(std::countr_zero(matches));
                        matches &= matches - 1;
                        const uint32_t offset = static_cast<uint32_t>(word_index * 64) + bit_index;
                        const uint32_t vec_id = window_min + offset;
                        const float score = std::exchange(window_scores[offset], 0.0F);
                        float doc_norm = 0.0F;
                        if constexpr (ScorerType == IndexScorerType::BM25) {
                            doc_norm = bm25_context_.doc_norm(vec_id);
                        }
                        complete_candidate(vec_id, score, doc_norm);
                    }
                }
            }

            if (!update_non_essential_lists()) {
                return;
            }
        }
    }

    template <IndexScorerType ScorerType>
    void
    run_sorted() {
        if (cursors_.size() >= bulk_query_nnz_threshold_) {
            run_sorted_bulk<ScorerType>();
        } else {
            run_sorted_linear<ScorerType>();
        }
    }

    static std::vector<Cursor>
    make_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                 const std::shared_ptr<IndexScorer>& index_scorer, const BM25ScoringContext& bm25_context,
                 const BitsetView& bitset, float dim_max_score_ratio, const FilterBounds& filter_bounds) {
        std::vector<Cursor> cursors;
        cursors.reserve(query.size());
        for (const auto& [dim_id, dim_val] : query) {
            cursors.push_back(Cursor{
                GetFilteredPostingListCursor(index, dim_id, bitset, filter_bounds), index_scorer->dim_scorer(dim_val),
                dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val), index.get_block_max_data_cursor(dim_id),
                dim_val, bm25_context.query_component(dim_val)});
        }
        return cursors;
    }

    FilterBounds filter_bounds_;
    BM25ScoringContext bm25_context_;
    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
    IndexScorerType scorer_type_;
    const size_t bulk_query_nnz_threshold_;
};
}  // namespace knowhere::sparse::inverted
