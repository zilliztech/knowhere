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

template <typename IndexType>
class DaatMaxScoreSearcher : public RankedSearcher {
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

    explicit DaatMaxScoreSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                                  const std::shared_ptr<IndexScorer>& search_scorer, const uint32_t k,
                                  const uint32_t max_vec_id, const BitsetView& bitset, float dim_max_score_ratio)
        : RankedSearcher(k),
          cursors_(make_cursors(index, query, search_scorer, bitset, dim_max_score_ratio)),
          max_vec_id_(max_vec_id),
          row_sums_(index.get_row_sums()),
          scorer_type_(search_scorer->config().scorer_type) {
        if (scorer_type_ == IndexScorerType::BM25) {
            compute_warm_threshold(index, query, search_scorer, bitset);
        }
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

    template <IndexScorerType ScorerType>
    void
    run_sorted(std::vector<Cursor>& cursors, uint64_t max_vec_id) {
        auto upper_bounds = calc_upper_bounds(cursors);
        // Pruning threshold:
        //   - Once topk_ is full, use its real threshold (the k-th best so far).
        //   - Before topk_ fills, use warm_threshold_ if available (a safe lower bound
        //     on the true k-th best, derived from a single seed term's contributions).
        //   - If warm threshold is not valid, fall back to "always pass" (current behavior).
        // Sparse top-k uses std::greater, so the entry condition is strict `>`.
        auto above_threshold = [&](auto score) {
            if (topk_.Full()) {
                return topk_.WouldEnter(score);
            }
            if (warm_threshold_valid_) {
                return score > warm_threshold_;
            }
            return true;
        };

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

                if constexpr (ScorerType == IndexScorerType::BM25) {
                    // Prefetch row_sums_ for next iterations that will be used by the BM25 scorer
                    // Experiments show this prefetch pattern is optimal vs only prefetching next_vec_id
                    __builtin_prefetch(&row_sums_[current_vec_id], 0, 3);
                }

                std::for_each(cursors.begin(), first_lookup, [&](auto& cursor) {
                    if (cursor.vec_id() == current_vec_id) {
                        current_score += cursor.score();
                        cursor.next();
                        if constexpr (ScorerType == IndexScorerType::BM25) {
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
        if (scorer_type_ == IndexScorerType::BM25) {
            run_sorted<IndexScorerType::BM25>(cursors, max_vec_id_);
        } else {
            run_sorted<IndexScorerType::IP>(cursors, max_vec_id_);
        }
        std::swap(cursors, cursors_);
    }

 private:
    // Compute a safe warm-start pruning threshold from a single seed term, used
    // before topk_ fills.
    //
    // Math:
    //   For BM25, each per-(term, doc) contribution is non-negative. So the full
    //   query score for doc d is >= the contribution of any single matching term.
    //   If we scan a seed term's posting list, compute its single-term BM25
    //   contribution per matching doc, and take the k-th best (S_k), then at least
    //   k docs have full_score >= S_k. Therefore the true k-th best full score
    //   is also >= S_k. Using S_k as an early pruning threshold is recall-safe.
    //
    // Safety guard: only valid if the seed term has >= k matching docs. With fewer
    // hits, S_m for m < k is only a lower bound on the m-th best full score, not
    // the k-th. We skip in that case.
    //
    // Seed selection: the term with the largest per-dim max_score. High max_score
    // is correlated with low document frequency / short posting list, so this
    // gives the most discriminative bound at the lowest scan cost.
    //
    // Cost: one scan of the seed term's posting list + O(N log k) heap maintenance.
    //
    // Note: posting_list_iterator is move-only across all current sparse index
    // implementations, so the seed cursor is built fresh via get_dim_plist_cursor
    // rather than copied from the existing cursors_.
    void
    compute_warm_threshold(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                           const std::shared_ptr<IndexScorer>& search_scorer, const BitsetView& bitset) {
        const std::size_t k = topk_.Capacity();
        // Activation guards:
        //   - query.size() < 2: seed pass == full search for single-term queries.
        //   - k < 2: heap fills on first push; warm threshold serves no purpose.
        if (query.size() < 2 || k < 2) {
            return;
        }

        // Pick the query term with the largest per-dim max_score as the seed.
        // dim_max_score_ratio is a per-search scalar so it does not change the argmax.
        std::size_t seed_idx = 0;
        float best_unscaled_max = -1.0f;
        for (std::size_t i = 0; i < query.size(); ++i) {
            float ms = index.get_dim_max_score(query[i].first, query[i].second);
            if (ms > best_unscaled_max) {
                best_unscaled_max = ms;
                seed_idx = i;
            }
        }

        auto seed_iter = index.get_dim_plist_cursor(query[seed_idx].first, bitset);
        auto seed_scorer = search_scorer->dim_scorer(query[seed_idx].second);

        // Min-heap of size k holding top-k single-term contributions.
        std::vector<float> heap;
        heap.reserve(k);
        std::size_t hit_count = 0;

        while (seed_iter.valid()) {
            const float contrib = seed_scorer(seed_iter.vec_id(), seed_iter.val());
            ++hit_count;
            if (heap.size() < k) {
                heap.push_back(contrib);
                if (heap.size() == k) {
                    std::make_heap(heap.begin(), heap.end(), std::greater<>());
                }
            } else if (contrib > heap.front()) {
                std::pop_heap(heap.begin(), heap.end(), std::greater<>());
                heap.back() = contrib;
                std::push_heap(heap.begin(), heap.end(), std::greater<>());
            }
            seed_iter.next();
        }

        if (hit_count >= k && heap.size() == k) {
            warm_threshold_ = heap.front();
            warm_threshold_valid_ = true;
        }
    }

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
    // row_sums_ is only used for BM25 scorer
    const std::vector<float>& row_sums_;
    IndexScorerType scorer_type_;
    // Warm-start pruning threshold (BM25 only). Valid only after
    // compute_warm_threshold sets it; consulted by run_sorted until topk_ fills.
    float warm_threshold_ = 0.0f;
    bool warm_threshold_valid_ = false;
};

}  // namespace knowhere::sparse::inverted
