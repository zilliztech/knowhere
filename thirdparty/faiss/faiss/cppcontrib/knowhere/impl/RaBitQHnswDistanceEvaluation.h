/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the MIT license in thirdparty/faiss/LICENSE. */
#pragma once

#include <faiss/cppcontrib/knowhere/impl/HnswDistanceEvaluation.h>
#include <faiss/cppcontrib/knowhere/impl/Neighbor.h>
#include <faiss/cppcontrib/knowhere/impl/RaBitQStagedDistanceComputer.h>
#include <faiss/impl/RaBitQUtils.h>
#include <limits>
#include <vector>
#include <utility>
#include <algorithm>

namespace faiss::cppcontrib::knowhere::rabitq_search {

// Alternative to DefaultHnswDistanceEvaluation under the same policy contract.
// Requires RaBitQStagedDistanceComputer; delegates non-staged work to the
// default policy without inheriting from it or depending on the searcher.
struct RaBitQHnswDistanceEvaluation {
    size_t k = 0;
    std::vector<float> results;

    void begin(size_t count) { k = count; results.clear(); results.reserve(k); }
    float threshold() const {
        return results.size() < k ? std::numeric_limits<float>::infinity() : results.front();
    }
    void record(float distance, int status) {
        if (!k || status == Neighbor::kInvalid) return;
        if (results.size() < k) {
            results.push_back(distance);
            std::push_heap(results.begin(), results.end());
        } else if (distance < results.front()) {
            // Replace the worst retained distance with one sift-down, instead
            // of repairing the threshold heap separately for pop and push.
            size_t parent = 0;
            for (size_t child = 1; child < k; child = 2 * parent + 1) {
                if (child + 1 < k && results[child] < results[child + 1]) ++child;
                if (!(distance < results[child])) break;
                results[parent] = results[child];
                parent = child;
            }
            results[parent] = distance;
        }
    }
    template <class DC, class Emit>
    size_t compute(DC& dc, const size_t* ids, const int* statuses,
                   size_t count, int level, Emit&& emit) {
        if (k && level == 0) {
            auto& staged = static_cast<RaBitQStagedDistanceComputer&>(dc);
            const auto before = staged.refine_count;
            if (count == 4 && staged.dc->nb_bits > 1) {
                const uint8_t* codes[4];
                for (size_t i = 0; i < 4; ++i) {
                    codes[i] = staged.dc->codes + ids[i] * staged.dc->code_size;
                }
                float estimates[4];
                staged.dc->distance_to_code_1bit_batch_4(codes, estimates);
                staged.estimate_count += 4;
                for (size_t i = 0; i < 4; ++i) {
                    // Only estimates are batched. Refine and update the threshold
                    // in exactly the original candidate order.
                    const float distance = staged.evaluate_estimate(ids[i], codes[i], estimates[i], threshold());
                    record(distance, statuses[i]);
                    emit(i, distance);
                }
                return staged.refine_count - before;
            }
            for (size_t i = 0; i < count; ++i) {
                const float distance = staged.evaluate(ids[i], threshold());
                record(distance, statuses[i]);
                emit(i, distance);
            }
            return staged.refine_count - before;
        }
        DefaultHnswDistanceEvaluation default_evaluation;
        return default_evaluation.compute(dc, ids, statuses, count, level, std::forward<Emit>(emit));
    }
};
} // namespace faiss::cppcontrib::knowhere::rabitq_search
