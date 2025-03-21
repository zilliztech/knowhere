#pragma once

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "index/sparse/inverted/pisa/cursor/max_scored_cursor.h"
#include "index/sparse/inverted/pisa/util/topk_queue.h"
#include "knowhere/sparse_utils.h"
#include "searcher.h"

namespace knowhere::sparse::pisa {

template <typename IndexType>
class DaatMaxScoreSearcher : public RankedSearcher {
 public:
    using Cursor = MaxScoredCursor<typename IndexType::posting_list_iterator>;
    using Cursors = std::vector<Cursor>;
    explicit DaatMaxScoreSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                                  std::shared_ptr<IndexScorer> index_scorer, const uint32_t k,
                                  const uint32_t max_vec_id, const BitsetView& bitset)
        : RankedSearcher(k),
          cursors_(std::move(make_max_scored_cursors<IndexType>(index, query, index_scorer, bitset))),
          max_vec_id_(max_vec_id),
          row_sums_(index.meta_data_.row_sums_),
          metric_type_(index.get_metric_type()) {
    }

    [[nodiscard]] inline auto
    sorted(Cursors& cursors) -> Cursors {
        std::vector<std::size_t> term_positions(cursors.size());
        std::iota(term_positions.begin(), term_positions.end(), 0);
        std::sort(term_positions.begin(), term_positions.end(),
                  [&](auto&& lhs, auto&& rhs) { return cursors[lhs].max_score() > cursors[rhs].max_score(); });
        Cursors sorted;
        for (auto pos : term_positions) {
            sorted.push_back(std::move(cursors[pos]));
        };
        return sorted;
    }

    [[nodiscard]] inline auto
    calc_upper_bounds(Cursors& cursors) -> std::vector<float> {
        std::vector<float> upper_bounds(cursors.size());
        auto out = upper_bounds.rbegin();
        float bound = 0.0;
        for (auto pos = cursors.rbegin(); pos != cursors.rend(); ++pos) {
            bound += pos->max_score();
            *out++ = bound;
        }
        return upper_bounds;
    }

    [[nodiscard]] inline auto
    min_vec_id(Cursors& cursors) -> std::uint32_t {
        return std::min_element(cursors.begin(), cursors.end(),
                                [](auto&& lhs, auto&& rhs) { return lhs.vec_id() < rhs.vec_id(); })
            ->vec_id();
    }

    enum class UpdateResult : bool { Continue, ShortCircuit };
    enum class DocumentStatus : bool { Insert, Skip };

    template <SparseMetricType MetricType>
    inline void
    run_sorted(Cursors& cursors, uint64_t max_vec_id) {
        auto upper_bounds = calc_upper_bounds(cursors);
        auto above_threshold = [&](auto score) { return topk_.would_enter(score); };

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
        std::uint32_t current_vec_id = 0;

        while (current_vec_id < max_vec_id) {
            auto status = DocumentStatus::Skip;
            while (status == DocumentStatus::Skip) {
                if (next_vec_id >= max_vec_id) [[unlikely]] {
                    return;
                }

                current_score = 0;
                current_vec_id = std::exchange(next_vec_id, max_vec_id);

                std::for_each(cursors.begin(), first_lookup, [&](auto& cursor) {
                    if (cursor.vec_id() == current_vec_id) {
                        current_score += cursor.score();
                        cursor.next();
                        if constexpr (MetricType == SparseMetricType::METRIC_BM25) {
                            // Prefetch row_sums_ for next iterations that will be used by the BM25 scorer
                            // Experiments show this prefetch pattern is optimal vs only prefetching next_vec_id
                            __builtin_prefetch(&row_sums_[cursor.vec_id()], 0, 3);
                        }
                    }
                    if (auto vec_id = cursor.vec_id(); vec_id < next_vec_id) {
                        next_vec_id = vec_id;
                    }
                });

                status = DocumentStatus::Insert;
                auto lookup_bound = first_upper_bound;
                for (auto pos = first_lookup; pos != cursors.end(); ++pos, ++lookup_bound) {
                    auto& cursor = *pos;
                    if (!above_threshold(current_score + *lookup_bound)) {
                        status = DocumentStatus::Skip;
                        break;
                    }
                    cursor.next_geq(current_vec_id);
                    if (cursor.vec_id() == current_vec_id) {
                        current_score += cursor.score();
                    }
                }
            }
            if (topk_.insert(current_score, current_vec_id) &&
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
        if (metric_type_ == SparseMetricType::METRIC_BM25) {
            run_sorted<SparseMetricType::METRIC_BM25>(cursors, max_vec_id_);
        } else {
            run_sorted<SparseMetricType::METRIC_IP>(cursors, max_vec_id_);
        }
        std::swap(cursors, cursors_);
    }

 private:
    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
    // row_sums_ is only used for BM25 metric
    const std::vector<float>& row_sums_;
    SparseMetricType metric_type_;
};

}  // namespace knowhere::sparse::pisa
