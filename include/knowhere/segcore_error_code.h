// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef SEGCORE_ERROR_CODE_H
#define SEGCORE_ERROR_CODE_H

#include "common/EasyAssert.h"  // milvus-common: shared milvus::ErrorCode registry
#include "knowhere/expected.h"

namespace knowhere {

// Map a knowhere::Status to the shared milvus::ErrorCode that the segcore
// boundary (and ultimately the Go retry policy) consumes.
//
// "Producer owns classification": knowhere knows best whether its own status
// is the caller's fault, a transient failure, or a permanent one -- so this
// mapping lives here, next to the Status enum, instead of a hand-maintained
// copy on the milvus side (which this migrates and retires). Same convention
// as milvus_storage::ToSegcoreError.
//
// It is deliberately a switch with NO `default:` plus a post-switch fallback:
// a `default:` would suppress -Wswitch, letting a newly added Status fall
// through silently; the pragma turns the warning into an error so adding a
// Status without classifying it here breaks the build.
//
// Invariant (locked by tests): this mapping and StatusCategoryOf must agree --
//   input_error      <=> InvalidParameter
//   transient_error  <=> a code the Go-side table marks retriable
//   permanent_error  <=> a non-retriable, non-input code
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
inline constexpr milvus::ErrorCode
ToSegcoreErrorCode(knowhere::Status status) {
    switch (status) {
        case knowhere::Status::success:
            return milvus::ErrorCode::Success;
        // the request content itself is at fault
        case knowhere::Status::invalid_args:
        case knowhere::Status::invalid_param_in_json:
        case knowhere::Status::out_of_range_in_json:
        case knowhere::Status::type_conflict_in_json:
        case knowhere::Status::invalid_metric_type:
        case knowhere::Status::empty_index:
        case knowhere::Status::index_not_trained:
        case knowhere::Status::index_already_trained:
        case knowhere::Status::invalid_value_in_json:
        case knowhere::Status::arithmetic_overflow:
        case knowhere::Status::invalid_binary_set:
        case knowhere::Status::invalid_index_error:
        case knowhere::Status::invalid_cluster_error:
            return milvus::ErrorCode::InvalidParameter;
        // capability errors: the request is fine, this build/CPU cannot serve
        // it (e.g. SCANN needing AVX2) -- Unsupported, not malformed input
        case knowhere::Status::not_implemented:
        case knowhere::Status::invalid_instruction_set:
            return milvus::ErrorCode::Unsupported;
        // a serialized index that cannot be recognized is corrupt data
        case knowhere::Status::invalid_serialized_index_type:
            return milvus::ErrorCode::DataFormatBroken;
        // transient failures: retry / replica-reroute may succeed
        case knowhere::Status::malloc_error:
            return milvus::ErrorCode::MemAllocateFailed;
        case knowhere::Status::disk_file_error:
            return milvus::ErrorCode::FileReadFailed;
        // permanent server-side inner errors -> generic KnowhereError.
        // timeout stays here: it is Cardinal-only (BuildAsync
        // cancel-or-build-timeout) and conflates cancel with timeout, so it
        // must not map to a retriable code until those are separated.
        case knowhere::Status::faiss_inner_error:
        case knowhere::Status::hnsw_inner_error:
        case knowhere::Status::diskann_inner_error:
        case knowhere::Status::cuvs_inner_error:
        case knowhere::Status::cardinal_inner_error:
        case knowhere::Status::cuda_runtime_error:
        case knowhere::Status::cluster_inner_error:
        case knowhere::Status::timeout:
        case knowhere::Status::internal_error:
        case knowhere::Status::sparse_inner_error:
        case knowhere::Status::brute_force_inner_error:
        case knowhere::Status::emb_list_inner_error:
        case knowhere::Status::aisaq_error:
        case knowhere::Status::knowhere_inner_error:
            return milvus::ErrorCode::KnowhereError;
    }
    // out-of-range value: safe non-retriable fallback (does not suppress
    // -Wswitch above)
    return milvus::ErrorCode::KnowhereError;
}
#pragma GCC diagnostic pop

}  // namespace knowhere

#endif /* SEGCORE_ERROR_CODE_H */
