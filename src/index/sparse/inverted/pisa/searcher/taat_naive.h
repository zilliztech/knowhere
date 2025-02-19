#pragma once

#include <vector>

#include "index/sparse/inverted/pisa/cursor/scored_cursor.h"
#include "index/sparse/inverted/pisa/util/topk_queue.h"
#include "searcher.h"

namespace knowhere::sparse::pisa {

template <typename IndexType>
class TaatNaiveSearcher : public RankedSearcher {
 public:
    using Cursor = ScoredCursor<typename IndexType::posting_list_iterator>;
    explicit TaatNaiveSearcher(const IndexType& index, const std::vector<std::pair<uint32_t, float>>& query,
                               std::shared_ptr<IndexScorer> index_scorer, const uint32_t k, const uint32_t max_vec_id,
                               const BitsetView& bitset)
        : RankedSearcher(k),
          cursors_(std::move(make_scored_cursors<IndexType>(index, query, index_scorer, bitset))),
          max_vec_id_(max_vec_id) {
    }

    void
    search() override {
        if (cursors_.empty()) {
            return;
        }

        std::vector<float> distances(max_vec_id_, 0.0f);

        for (auto& en : cursors_) {
            while (en.valid()) {
                distances[en.vec_id()] += en.score();
                en.next();
            }
        }

        for (size_t i = 0; i < distances.size(); ++i) {
            topk_.insert(distances[i], i);
        }
    }

 private:
    std::vector<Cursor> cursors_;
    uint32_t max_vec_id_;
};

}  // namespace knowhere::sparse::pisa
