#pragma once

#include <vector>

#include "index/sparse/inverted/pisa/cursor/max_scored_cursor.h"
#include "index/sparse/inverted/pisa/util/topk_queue.h"
#include "searcher.h"

namespace knowhere::sparse::pisa {

template <typename IndexType>
class DaatWandSearcher : public RankedSearcher {
 public:
    using Cursor = MaxScoredCursor<typename IndexType::posting_list_iterator>;
    explicit DaatWandSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                              std::shared_ptr<IndexScorer> index_scorer, const uint32_t k, const uint32_t max_vec_id,
                              const BitsetView& bitset)
        : RankedSearcher(k),
          cursors_(std::move(make_max_scored_cursors<IndexType>(index, query, index_scorer, bitset))),
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

        auto sort_enums = [&]() {
            // sort enumerators by increasing vec_id
            std::sort(ordered_cursors.begin(), ordered_cursors.end(),
                      [](Cursor* lhs, Cursor* rhs) { return lhs->vec_id() < rhs->vec_id(); });
        };

        sort_enums();
        while (true) {
            // find pivot
            float upper_bound = 0;
            size_t pivot;
            bool found_pivot = false;
            for (pivot = 0; pivot < ordered_cursors.size(); ++pivot) {
                if (ordered_cursors[pivot]->vec_id() >= max_vec_id_) {
                    break;
                }
                upper_bound += ordered_cursors[pivot]->max_score();
                if (this->topk_.would_enter(upper_bound)) {  // Access base class topk_
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
                this->topk_.insert(score, pivot_id);  // Access base class topk_
                // resort by docid
                sort_enums();
            } else {
                // no match, move farthest list up to the pivot
                uint64_t next_list = pivot;
                for (; ordered_cursors[next_list]->vec_id() == pivot_id; --next_list) {
                }
                ordered_cursors[next_list]->next_geq(pivot_id);
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
    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
};

}  // namespace knowhere::sparse::pisa
