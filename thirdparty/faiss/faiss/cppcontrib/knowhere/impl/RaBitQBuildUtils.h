/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss::cppcontrib::knowhere::rabitq_build {

/** Populate a trained RaBitQ storage pipeline in bounded input batches.
 *
 * The caller supplies the complete add pipeline (including any pretransform,
 * cosine norms or raw-space refiner). Each original input slice is passed to
 * its existing virtual add(), preserving its encoding and ID semantics. Do
 * not pass a graph-building index: changing graph add boundaries can change
 * topology. IVF and file-backed callers retain their own training and I/O.
 *
 * This is an execution policy, not an Index subtype or serialized property.
 * It bounds rows per add call, not total RSS or training allocations. The
 * default preserves HNSW RaBitQ's original 4096-row encoding boundaries.
 * No input ownership, transformation or shared mutable state is introduced.
 * On failure, completed batches remain added; no transactional rollback is
 * promised beyond the underlying Index::add contract.
 */
inline void add_in_blocks(
        faiss::Index& storage,
        idx_t n,
        const float* x,
        idx_t block_rows = 4096) {
    FAISS_THROW_IF_NOT_MSG(n >= 0, "negative RaBitQ input count");
    FAISS_THROW_IF_NOT_MSG(block_rows > 0, "RaBitQ block size must be positive");
    FAISS_THROW_IF_NOT_MSG(storage.d > 0, "invalid RaBitQ input dimension");
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "null RaBitQ input");
    const size_t dim = static_cast<size_t>(storage.d);
    FAISS_THROW_IF_NOT_MSG(
            static_cast<uint64_t>(n) <=
                    std::numeric_limits<size_t>::max() / sizeof(float) / dim,
            "RaBitQ input size overflow");
    for (idx_t offset = 0; offset < n;) {
        const idx_t count = std::min(block_rows, n - offset);
        storage.add(count, x + static_cast<size_t>(offset) * dim);
        offset += count;
    }
}

} // namespace faiss::cppcontrib::knowhere::rabitq_build
