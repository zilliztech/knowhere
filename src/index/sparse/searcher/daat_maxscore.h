// Document-at-a-Time (DAAT) MaxScore searcher.
// Derived from the PISA search engine (Performant Indexes and Search for Academia).
//   Paper: H. Turtle and J. Flood, "Query Evaluation: Strategies and Optimizations",
//          Information Processing & Management, 1995.
//   Repository: https://github.com/pisa-engine/pisa
//   License: Apache License 2.0

#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "index/sparse/scorer.h"
#include "index/sparse/searcher/bulk_simd.h"
#include "index/sparse/searcher/searcher.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

template <typename IndexType>
class DaatMaxScoreSearcher : public RankedSearcher {
 public:
    struct Cursor {
        typename IndexType::posting_list_iterator index_cursor;
        float max_score;
        float qval_p1;

        [[nodiscard]] uint32_t
        vec_id() const noexcept {
            return index_cursor.vec_id();
        }

        template <PostingFilterMode FilterMode>
        inline void
        next() noexcept {
            NextPosting<FilterMode>(index_cursor);
        }

        template <PostingFilterMode FilterMode>
        void
        next_geq(uint32_t vec_id) noexcept {
            // Only non-essential lists call this method. Candidates have
            // already passed the filter through an essential list, and DAAT
            // MaxScore never promotes a non-essential list back to essential.
            if constexpr (FilterMode != PostingFilterMode::None &&
                          requires { index_cursor.seek_accepted_candidate(vec_id); }) {
                index_cursor.seek_accepted_candidate(vec_id);
            } else {
                NextPostingGEQ<FilterMode>(index_cursor, vec_id);
            }
        }
    };

    explicit DaatMaxScoreSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                                  const std::shared_ptr<IndexScorer>& search_scorer, const uint32_t k,
                                  const uint32_t max_vec_id, const BitsetView& bitset, float dim_max_score_ratio,
                                  size_t bulk_query_nnz_threshold)
        : RankedSearcher(k),
          filter_bounds_(GetFilterBounds(bitset, max_vec_id)),
          bm25_context_(index.get_row_sums(), *search_scorer),
          cursors_(
              make_cursors(index, query, search_scorer, bm25_context_, bitset, dim_max_score_ratio, filter_bounds_)),
          max_vec_id_(filter_bounds_.upper_bound),
          scorer_type_(search_scorer->config().scorer_type),
          bulk_query_nnz_threshold_(bulk_query_nnz_threshold),
          filter_mode_(GetPostingFilterMode(bitset, max_vec_id_)) {
    }

    [[nodiscard]] auto
    sorted(std::vector<Cursor>& cursors) -> std::vector<Cursor> {
        std::vector<size_t> term_positions(cursors.size());
        std::iota(term_positions.begin(), term_positions.end(), 0);
        std::sort(term_positions.begin(), term_positions.end(),
                  [&](auto&& lhs, auto&& rhs) { return cursors[lhs].max_score > cursors[rhs].max_score; });
        std::vector<Cursor> sorted;
        sorted.reserve(cursors.size());
        for (auto pos : term_positions) {
            sorted.push_back(std::move(cursors[pos]));
        }
        return sorted;
    }

    [[nodiscard]] auto
    calc_upper_bounds(std::vector<Cursor>& cursors) -> std::vector<float> {
        std::vector<float> upper_bounds(cursors.size());
        auto out = upper_bounds.rbegin();
        float bound = 0.0;
        for (auto pos = cursors.rbegin(); pos != cursors.rend(); ++pos) {
            bound += pos->max_score;
            *out++ = bound;
        }
        return upper_bounds;
    }

    [[nodiscard]] auto
    min_vec_id(std::vector<Cursor>& cursors) -> uint32_t {
        return std::min_element(cursors.begin(), cursors.end(),
                                [](auto&& lhs, auto&& rhs) { return lhs.vec_id() < rhs.vec_id(); })
            ->vec_id();
    }

    enum class UpdateResult : bool { Continue, ShortCircuit };
    enum class VectorStatus : bool { Insert, Skip };

    template <IndexScorerType ScorerType, PostingFilterMode FilterMode>
    void
    run_sorted_linear(std::vector<Cursor>& cursors, uint32_t max_vec_id) {
        auto upper_bounds = calc_upper_bounds(cursors);
        auto above_threshold = [&](auto score) { return topk_.WouldEnter(score); };

        auto first_upper_bound = upper_bounds.end();
        auto first_lookup = cursors.end();
        auto next_vec_id = min_vec_id(cursors);

        auto update_non_essential_lists = [&] {
            while (first_lookup != cursors.begin() && !above_threshold(*std::prev(first_upper_bound))) {
                --first_lookup;
                --first_upper_bound;
                if (first_lookup == cursors.begin()) {
                    return UpdateResult::ShortCircuit;
                }
            }
            return UpdateResult::Continue;
        };

        if (update_non_essential_lists() == UpdateResult::ShortCircuit) {
            return;
        }

        float current_score = 0;
        uint32_t current_vec_id = 0;

        while (current_vec_id < max_vec_id) {
            auto status = VectorStatus::Skip;
            while (status == VectorStatus::Skip) {
                if (next_vec_id >= max_vec_id) [[unlikely]] {
                    return;
                }

                current_score = 0;
                current_vec_id = std::exchange(next_vec_id, max_vec_id);
                float doc_norm = 0.0f;

                if constexpr (ScorerType == IndexScorerType::BM25) {
                    // Prefetch row_sums_ for next iterations that will be used by the BM25 scorer
                    // Experiments show this prefetch pattern is optimal vs only prefetching next_vec_id
                    bm25_context_.prefetch_document(current_vec_id);
                    doc_norm = bm25_context_.doc_norm(current_vec_id);
                }

                auto score_term = [&](auto& cursor) -> float {
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        const float tf = static_cast<float>(cursor.index_cursor.val());
                        return bm25_context_.score(cursor.qval_p1, tf, doc_norm);
                    } else {
                        return cursor.qval_p1 * static_cast<float>(cursor.index_cursor.val());
                    }
                };

                std::for_each(cursors.begin(), first_lookup, [&](auto& cursor) {
                    if (cursor.vec_id() == current_vec_id) {
                        current_score += score_term(cursor);
                        cursor.template next<FilterMode>();
                        if constexpr (ScorerType == IndexScorerType::BM25) {
                            // Prefetch row_sums_ for next iterations that will be used by the BM25 scorer
                            // Experiments show this prefetch pattern is optimal vs only prefetching next_vec_id
                            if (cursor.vec_id() < max_vec_id) {
                                bm25_context_.prefetch_document(cursor.vec_id());
                            }
                        }
                    }
                    if (auto vec_id = cursor.vec_id(); vec_id < next_vec_id) {
                        next_vec_id = vec_id;
                    }
                });

                status = VectorStatus::Insert;
                auto lookup_bound = first_upper_bound;
                for (auto pos = first_lookup; pos != cursors.end(); ++pos, ++lookup_bound) {
                    auto& cursor = *pos;
                    if (!above_threshold(current_score + *lookup_bound)) {
                        status = VectorStatus::Skip;
                        break;
                    }
                    cursor.template next_geq<FilterMode>(current_vec_id);
                    if (cursor.vec_id() == current_vec_id) {
                        current_score += score_term(cursor);
                    }
                }
            }
            if (topk_.Push(current_score, current_vec_id) &&
                update_non_essential_lists() == UpdateResult::ShortCircuit) {
                return;
            }
        }
    }

    template <IndexScorerType ScorerType, PostingFilterMode FilterMode>
    void
    run_sorted_bulk(std::vector<Cursor>& cursors, uint32_t max_vec_id) {
        // Amortize cursor dispatch over a document window. This follows the same high-level shape as
        // Lucene's MaxScoreBulkScorer: freeze the essential partition for one window, accumulate the
        // essential scores in doc-id order, then complete competitive candidates with non-essential lists.
        auto upper_bounds = calc_upper_bounds(cursors);
        auto above_threshold = [&](auto score) { return topk_.WouldEnter(score); };

        // The cursor collection is fixed during this search. Keep its bounds
        // local so decoding callbacks do not force vector metadata reloads in
        // the per-candidate loop.
        const size_t cursor_count = cursors.size();
        Cursor* const cursor_data = cursors.data();
        size_t essential_count = cursor_count;
        auto update_non_essential_lists = [&] {
            while (essential_count != 0 && !above_threshold(upper_bounds[essential_count - 1])) {
                --essential_count;
                if (essential_count == 0) {
                    return UpdateResult::ShortCircuit;
                }
            }
            return UpdateResult::Continue;
        };

        if (update_non_essential_lists() == UpdateResult::ShortCircuit) {
            return;
        }

        auto score_term = [&](auto& cursor, float doc_norm) -> float {
            if constexpr (ScorerType == IndexScorerType::BM25) {
                const float tf = static_cast<float>(cursor.index_cursor.val());
                return bm25_context_.score(cursor.qval_p1, tf, doc_norm);
            } else {
                return cursor.qval_p1 * static_cast<float>(cursor.index_cursor.val());
            }
        };

        auto complete_candidate = [&](uint32_t current_vec_id, float current_score, float doc_norm) {
            for (size_t i = essential_count; i < cursor_count; ++i) {
                if (!above_threshold(current_score + upper_bounds[i])) {
                    return;
                }
                auto& cursor = cursor_data[i];
                cursor.template next_geq<FilterMode>(current_vec_id);
                if (cursor.vec_id() == current_vec_id) {
                    current_score += score_term(cursor, doc_norm);
                }
            }
            topk_.Push(current_score, current_vec_id);
        };

        // Leave these buffers uninitialized until a window actually has multiple essential lists. Queries that
        // partition down to one essential list can then stay on the direct posting-list path entirely.
        std::array<float, kBulkWindowSize> window_scores;
        std::array<uint64_t, kBulkWindowSize / 64> window_matches;
        bool window_buffers_initialized = false;

        while (true) {
            uint32_t window_min = max_vec_id;
            uint32_t second_vec_id = max_vec_id;
            size_t lead_index = 0;
            for (size_t i = 0; i < essential_count; ++i) {
                const uint32_t vec_id = cursor_data[i].vec_id();
                if (vec_id < window_min) {
                    second_vec_id = window_min;
                    window_min = vec_id;
                    lead_index = i;
                } else if (vec_id < second_vec_id) {
                    second_vec_id = vec_id;
                }
            }
            if (window_min >= max_vec_id) [[unlikely]] {
                return;
            }
            const uint32_t window_max = static_cast<uint32_t>(
                std::min<uint64_t>(max_vec_id, static_cast<uint64_t>(window_min) + kBulkWindowSize));

            // A single essential list does not need a score buffer. The same applies to the leading list while the
            // second essential list is at least half a window away: no other essential list can contribute before it.
            const bool single_lead_range =
                essential_count == 1 || static_cast<uint64_t>(window_min) + kBulkWindowSize / 2 <= second_vec_id;
            if (single_lead_range) {
                const uint32_t range_max = essential_count == 1 ? window_max : std::min(window_max, second_vec_id);
                auto& cursor = cursor_data[lead_index];
                while (cursor.vec_id() < range_max) {
                    const uint32_t current_vec_id = cursor.vec_id();
                    float doc_norm = 0.0f;
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        doc_norm = bm25_context_.doc_norm(current_vec_id);
                    }
                    const float current_score = score_term(cursor, doc_norm);
                    cursor.template next<FilterMode>();
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        if (cursor.vec_id() < max_vec_id) {
                            bm25_context_.prefetch_document(cursor.vec_id());
                        }
                    }
                    complete_candidate(current_vec_id, current_score, doc_norm);
                }

                if (update_non_essential_lists() == UpdateResult::ShortCircuit) {
                    return;
                }
                continue;
            }

            if (!window_buffers_initialized) {
                window_scores.fill(0.0f);
                window_matches.fill(0);
                window_buffers_initialized = true;
            }

            for (size_t i = 0; i < essential_count; ++i) {
                auto& cursor = cursor_data[i];
                if constexpr (ScorerType == IndexScorerType::BM25) {
                    AccumulateBM25Bulk<FilterMode>(cursor.index_cursor, cursor.qval_p1, bm25_context_, window_min,
                                                   window_max, window_scores.data(), window_matches.data());
                } else {
                    AccumulateIPBulk<FilterMode>(cursor.index_cursor, cursor.qval_p1, window_min, window_max,
                                                 window_scores.data(), window_matches.data());
                }
                if constexpr (ScorerType == IndexScorerType::BM25) {
                    if (cursor.vec_id() < max_vec_id) {
                        bm25_context_.prefetch_document(cursor.vec_id());
                    }
                }
            }

            const size_t window_word_count = (static_cast<size_t>(window_max - window_min) + 63) / 64;
            for (size_t word_index = 0; word_index < window_word_count; ++word_index) {
                uint64_t matches = std::exchange(window_matches[word_index], 0);
                if (topk_.Full() && std::popcount(matches) >= 16) {
                    const float remaining_bound = essential_count < cursor_count ? upper_bounds[essential_count] : 0.0f;
                    matches = FilterBulkCandidates(window_scores.data() + word_index * 64, matches, remaining_bound,
                                                   topk_.Threshold());
                }
                while (matches != 0) {
                    const uint32_t bit_index = static_cast<uint32_t>(std::countr_zero(matches));
                    matches &= matches - 1;
                    const uint32_t offset = static_cast<uint32_t>(word_index * 64) + bit_index;
                    const uint32_t current_vec_id = window_min + offset;
                    float current_score = std::exchange(window_scores[offset], 0.0f);
                    float doc_norm = 0.0f;
                    if constexpr (ScorerType == IndexScorerType::BM25) {
                        doc_norm = bm25_context_.doc_norm(current_vec_id);
                    }
                    complete_candidate(current_vec_id, current_score, doc_norm);
                }
            }

            if (update_non_essential_lists() == UpdateResult::ShortCircuit) {
                return;
            }
        }
    }

    template <IndexScorerType ScorerType, PostingFilterMode FilterMode>
    void
    run_sorted(std::vector<Cursor>& cursors, uint32_t max_vec_id) {
        if (cursors.size() >= bulk_query_nnz_threshold_) {
            run_sorted_bulk<ScorerType, FilterMode>(cursors, max_vec_id);
        } else {
            run_sorted_linear<ScorerType, FilterMode>(cursors, max_vec_id);
        }
    }

    template <IndexScorerType ScorerType>
    void
    run_sorted_dispatch(std::vector<Cursor>& cursors, uint32_t max_vec_id) {
        // TODO: Check whether template-based handling of filter_mode_ is really needed, or whether passing
        // filter_mode_ directly as a field to the hot path performs equally well.
        switch (filter_mode_) {
            case PostingFilterMode::None:
                run_sorted<ScorerType, PostingFilterMode::None>(cursors, max_vec_id);
                break;
            case PostingFilterMode::Direct:
                run_sorted<ScorerType, PostingFilterMode::Direct>(cursors, max_vec_id);
                break;
            case PostingFilterMode::Generic:
                run_sorted<ScorerType, PostingFilterMode::Generic>(cursors, max_vec_id);
                break;
        }
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }
        auto cursors = sorted(cursors_);
        if (scorer_type_ == IndexScorerType::BM25) {
            run_sorted_dispatch<IndexScorerType::BM25>(cursors, max_vec_id_);
        } else {
            run_sorted_dispatch<IndexScorerType::IP>(cursors, max_vec_id_);
        }
        std::swap(cursors, cursors_);
    }

 private:
    static std::vector<Cursor>
    make_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                 const std::shared_ptr<IndexScorer>& index_scorer, const BM25ScoringContext& bm25_context,
                 const BitsetView& bitset, float dim_max_score_ratio, const FilterBounds& filter_bounds) {
        std::vector<Cursor> cursors;
        cursors.reserve(query.size());
        for (const auto& [dim_id, dim_val] : query) {
            cursors.push_back(Cursor{GetFilteredPostingListCursor(index, dim_id, bitset, filter_bounds),
                                     dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val),
                                     bm25_context.query_component(dim_val)});
        }
        return cursors;
    }

    FilterBounds filter_bounds_;
    BM25ScoringContext bm25_context_;
    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
    IndexScorerType scorer_type_;
    const size_t bulk_query_nnz_threshold_;
    const PostingFilterMode filter_mode_;
};

}  // namespace knowhere::sparse::inverted
