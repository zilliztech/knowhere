#pragma once

#include "knowhere/heap.h"

namespace knowhere::sparse::inverted {

using SparseHeap = knowhere::ResultMinHeap<float, uint32_t>;

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

    std::vector<typename SparseHeap::entry_type> const&
    topk() {
        topk_.Finalize();
        return topk_.Results();
    }

 protected:
    SparseHeap topk_;
};

}  // namespace knowhere::sparse::inverted
