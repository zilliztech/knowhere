#pragma once

#include "index/sparse/inverted/pisa/util/topk_queue.h"

namespace knowhere::sparse::pisa {

// Base searcher interface
class Searcher {
 public:
    virtual ~Searcher() = default;
    virtual void
    search() = 0;
};

// Common base implementation for all searchers
class RankedSearcher : public Searcher {
 public:
    RankedSearcher(uint32_t k) : topk_(k) {
    }

    std::vector<typename topk_queue::entry_type> const&
    topk() {
        topk_.finalize();
        return topk_.topk();
    }

 protected:
    topk_queue topk_;
};

}  // namespace knowhere::sparse::pisa
