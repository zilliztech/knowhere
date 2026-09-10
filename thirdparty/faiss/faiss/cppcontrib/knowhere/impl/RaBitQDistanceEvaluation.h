/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the MIT license in thirdparty/faiss/LICENSE. */
#pragma once

#include <faiss/cppcontrib/knowhere/impl/HnswSearcher.h>
#include <faiss/cppcontrib/knowhere/impl/StagedDistanceComputer.h>
#include <faiss/impl/RaBitQUtils.h>
#include <limits>
#include <queue>
#include <utility>

namespace faiss::cppcontrib::knowhere::rabitq_search {

// Both graph traversal implementations compare in smaller-is-better units.
// This is a probability window, not a deterministic lower bound.
inline bool should_refine(float estimate, float f_error, float g_error,
                          float threshold, bool similarity, float scale) {
    const float error = f_error * g_error;
    return similarity ? (estimate + error) * scale > -threshold
                      : std::max(0.0f, estimate - error) < threshold;
}

// Used only by the RaBitQ wrapper's filtered/feder/RBQ1 kNN specialization.
struct DistanceEvaluation {
    size_t k = 0;
    std::priority_queue<float> results;

    void begin(size_t count) { k = count; results = {}; }
    float threshold() const {
        return results.size() < k ? std::numeric_limits<float>::infinity() : results.top();
    }
    void record(float distance, int status) {
        if (!k || status == Neighbor::kInvalid) return;
        if (results.size() < k) results.push(distance);
        else if (distance < results.top()) {
            results.pop();
            results.push(distance);
        }
    }
    template <class DC, class Emit>
    size_t compute(DC& dc, const size_t* ids, const int* statuses,
                   size_t count, int level, Emit&& emit) {
        if (k && level == 0) {
            auto& staged = static_cast<StagedDistanceComputer&>(dc);
            const auto before = staged.refine_count;
            for (size_t i = 0; i < count; ++i) {
                const float distance = staged.evaluate(ids[i], threshold());
                record(distance, statuses[i]);
                emit(i, distance);
            }
            return staged.refine_count - before;
        }
        FullDistanceEvaluation full;
        return full.compute(dc, ids, statuses, count, level, std::forward<Emit>(emit));
    }
};
} // namespace faiss::cppcontrib::knowhere::rabitq_search
