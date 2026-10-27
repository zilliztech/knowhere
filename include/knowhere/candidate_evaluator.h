// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstdint>
#include <type_traits>

namespace knowhere {

enum class CandidateEvalStatus : uint32_t { Success = 0, InvalidArgument = 1, Failed = 2 };

// Creates one private mutable workspace. context is the query-owned factory;
// output is caller-owned and receives an opaque worker on success, nullptr on
// failure. The factory and query sources must outlive all workers.
using CandidateCreateWorkerFn = CandidateEvalStatus (*)(const void* context, void** output) noexcept;
// Destroys a worker through its originating library; nullptr is allowed.
using CandidateDestroyWorkerFn = void (*)(void* worker) noexcept;
// row_ids are segment row offsets (not primary keys or backend reordered IDs).
// count is 0..64; active_mask selects the lanes to evaluate. Inactive IDs are
// never read. accepted_mask is caller-owned: bit i means active row i is TRUE;
// FALSE/UNKNOWN are rejected. Errors are status values, never predicate FALSE.
using CandidateEvalBatchFn = CandidateEvalStatus (*)(void* worker, const int32_t* row_ids, uint32_t count,
                                                     uint64_t active_mask, uint64_t* accepted_mask) noexcept;

// Non-owning, fixed-layout V1 descriptor. No STL ownership crosses libraries.
// V1 size is frozen: struct_size validates this exact version, NOT append-only
// binary compatibility. A future layout requires a separate versioned type.
struct CandidateEvaluatorViewV1 {
    uint32_t abi_major = 1;
    uint32_t struct_size = sizeof(CandidateEvaluatorViewV1);
    const void* context = nullptr;
    CandidateCreateWorkerFn create_worker = nullptr;
    CandidateDestroyWorkerFn destroy_worker = nullptr;
    CandidateEvalBatchFn eval_batch = nullptr;

    bool
    valid() const noexcept {
        return abi_major == 1 && struct_size == sizeof(CandidateEvaluatorViewV1) && context != nullptr &&
               create_worker != nullptr && destroy_worker != nullptr && eval_batch != nullptr;
    }
};

static_assert(std::is_standard_layout_v<CandidateEvaluatorViewV1>);
static_assert(std::is_trivially_copyable_v<CandidateEvaluatorViewV1>);

}  // namespace knowhere
