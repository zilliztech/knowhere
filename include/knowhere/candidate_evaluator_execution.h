// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <bit>
#include <cstddef>
#include <stdexcept>

#include "knowhere/candidate_evaluator.h"

namespace knowhere {

// Private execution state owned by a bound filter, not by a search algorithm.
// One independent search task binds once; all its test() calls reuse this state.
// Neither this adapter nor the backend knows expressions or scalar types.
// Not a DSO ABI: only CandidateEvaluatorView crosses the callback boundary.
class CandidateEvaluatorExecution {
 public:
    explicit CandidateEvaluatorExecution(const CandidateEvaluatorView& view) : view_(view) {
        if (!view_.valid()) {
            throw std::invalid_argument("invalid candidate evaluator descriptor");
        }
        if (view_.create_worker(view_.context, &worker_) != CandidateEvalStatus::Success || worker_ == nullptr) {
            if (worker_ != nullptr) {
                view_.destroy_worker(worker_);
            }
            throw std::runtime_error("candidate workspace preparation failed");
        }
    }
    ~CandidateEvaluatorExecution() {
        view_.destroy_worker(worker_);
    }
    CandidateEvaluatorExecution(const CandidateEvaluatorExecution&) = delete;
    CandidateEvaluatorExecution&
    operator=(const CandidateEvaluatorExecution&) = delete;

    static uint64_t
    LaneMask(uint32_t count) {
        if (count > 64) {
            throw std::invalid_argument("candidate batch exceeds 64 lanes");
        }
        return count == 64 ? ~uint64_t{0} : (uint64_t{1} << count) - 1;
    }

    // Public segment offsets, not primary keys. A set return bit means excluded,
    // including inactive lanes. No inactive row is evaluated by the producer.
    uint64_t
    test(const int32_t* ids, uint32_t count, uint64_t active) {
        const auto lanes = LaneMask(count);
        if ((active & ~lanes) != 0 || (count != 0 && ids == nullptr)) {
            throw std::invalid_argument("invalid candidate batch IDs/mask");
        }
        if (active == 0) {
            return lanes;
        }
        uint64_t accepted = 0;
        if (view_.eval_batch(worker_, ids, count, active, &accepted) != CandidateEvalStatus::Success) {
            throw std::runtime_error("candidate callback failed");
        }
        if ((accepted & ~active) != 0) {
            throw std::runtime_error("candidate callback returned inactive lanes");
        }
        ++batch_calls;
        rows += std::popcount(active);
        return lanes & ~accepted;
    }

    size_t batch_calls = 0;
    // Successful producer active lanes, not distinct IDs or all input lanes.
    size_t rows = 0;

 private:
    CandidateEvaluatorView view_;
    void* worker_ = nullptr;
};
}  // namespace knowhere
