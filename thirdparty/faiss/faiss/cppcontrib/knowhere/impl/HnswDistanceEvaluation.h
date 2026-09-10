// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace faiss::cppcontrib::knowhere {

// Compile-time HNSW distance-evaluation policy contract (no common base class):
// begin(k) initializes query-local state; record(distance, status) seeds it.
// compute receives up to four candidates and emits their distances in order.
// Its return value counts additional distance evaluations for HNSW statistics;
// the searcher already counts one evaluation per candidate.
// Policies must match the supplied DC type. They do not own traversal queues.
//
// Default policy: directly evaluate the storage distance, including batch-four.
// "Default" does not imply FP32 accuracy: the DC may use SQ/PQ or another codec.
struct DefaultHnswDistanceEvaluation {
    void begin(size_t) {}
    void record(float, int) {}

    template <class DC, class Emit>
    size_t compute(DC& dc, const size_t* ids, const int*, size_t count,
                   int, Emit&& emit) {
        if (count == 4) {
            float d[4];
            dc.distances_batch_4(ids[0], ids[1], ids[2], ids[3],
                                 d[0], d[1], d[2], d[3]);
            for (size_t i = 0; i < count; ++i) emit(i, d[i]);
        } else {
            for (size_t i = 0; i < count; ++i) emit(i, dc(ids[i]));
        }
        return 0;
    }
};

} // namespace faiss::cppcontrib::knowhere
