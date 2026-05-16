// Document-at-a-Time (DAAT) MaxScore searcher.
// Derived from the PISA search engine (Performant Indexes and Search for Academia).
//   Paper: H. Turtle and J. Flood, "Query Evaluation: Strategies and Optimizations",
//          Information Processing & Management, 1995.
//   Repository: https://github.com/pisa-engine/pisa
//   License: Apache License 2.0

#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "index/sparse/scorer.h"
#include "index/sparse/searcher/searcher.h"
#include "knowhere/bitsetview.h"

namespace knowhere::sparse::inverted {

template <typename IndexType, typename QueryScorer>
class DaatMaxScoreSearcher : public RankedSearcher {
 public:
    using DimScorer = decltype(std::declval<const QueryScorer&>().dim_scorer(0.0f));

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

    explicit DaatMaxScoreSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                                  const QueryScorer& search_scorer, const uint32_t k, const uint32_t max_vec_id,
                                  const BitsetView& bitset, float dim_max_score_ratio)
        : RankedSearcher(k),
          cursors_(make_cursors(index, query, search_scorer, bitset, dim_max_score_ratio)),
          max_vec_id_(max_vec_id),
          row_sums_(index.get_row_sums()) {
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
        };
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

    void
    run_sorted(std::vector<Cursor>& cursors, uint64_t max_vec_id) {
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

                if constexpr (QueryScorer::scorer_type == IndexScorerType::BM25) {
                    // Prefetch row_sums_ for next iterations that will be used by the BM25 scorer
                    // Experiments show this prefetch pattern is optimal vs only prefetching next_vec_id
                    __builtin_prefetch(&row_sums_[current_vec_id], 0, 3);
                }

                std::for_each(cursors.begin(), first_lookup, [&](auto& cursor) {
                    if (cursor.vec_id() == current_vec_id) {
                        current_score += cursor.score();
                        cursor.next();
                        if constexpr (QueryScorer::scorer_type == IndexScorerType::BM25) {
                            // Prefetch row_sums_ for next iterations that will be used by the BM25 scorer
                            // Experiments show this prefetch pattern is optimal vs only prefetching next_vec_id
                            __builtin_prefetch(&row_sums_[cursor.vec_id()], 0, 3);
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
                    cursor.next_geq(current_vec_id);
                    if (cursor.vec_id() == current_vec_id) {
                        current_score += cursor.score();
                    }
                }
            }
            if (topk_.Push(current_score, current_vec_id) &&
                update_non_essential_lists() == UpdateResult::ShortCircuit) {
                return;
            }
        }
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }
        auto cursors = sorted(cursors_);
        run_sorted(cursors, max_vec_id_);
        std::swap(cursors, cursors_);
    }

 private:
    static std::vector<Cursor>
    make_cursors(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                 const QueryScorer& index_scorer, const BitsetView& bitset, float dim_max_score_ratio) {
        std::vector<Cursor> cursors;
        cursors.reserve(query.size());
        for (const auto& [dim_id, dim_val] : query) {
            cursors.push_back(Cursor{index.get_dim_plist_cursor(dim_id, bitset), index_scorer.dim_scorer(dim_val),
                                     dim_max_score_ratio * index.get_dim_max_score(dim_id, dim_val)});
        }
        return cursors;
    }

    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
    // row_sums_ is only used for BM25 scorer
    const std::vector<float>& row_sums_;
};

}  // namespace knowhere::sparse::inverted
