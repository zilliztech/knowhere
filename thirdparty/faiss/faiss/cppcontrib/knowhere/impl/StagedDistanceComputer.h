#pragma once

#include <faiss/impl/DistanceComputer.h>

namespace faiss::cppcontrib::knowhere {
// Distances and thresholds follow the graph searcher's smaller-is-better convention.
struct StagedDistanceComputer : faiss::DistanceComputer {
    size_t estimate_count = 0;
    size_t refine_count = 0;
    virtual float evaluate(idx_t id, float threshold) = 0;
};
} // namespace faiss::cppcontrib::knowhere
