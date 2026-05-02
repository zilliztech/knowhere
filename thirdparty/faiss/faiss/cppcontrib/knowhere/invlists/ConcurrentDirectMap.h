/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <cstddef>
#include <deque>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

struct ConcurrentArrayInvertedLists;

/** Concurrent direct map — chunked-deque id→LO storage that supports
 * read-while-append for the CC index variants (IVF_FLAT_CC,
 * IVF_SQ_CC).
 *
 * Path-D step 10.1 extraction: this is the chunked variant previously
 * nested as `DirectMap::ConcurrentArray` inside fork::DirectMap. It is
 * now a standalone class — no inheritance or enum-branch relationship
 * with baseline or fork ::DirectMap. CC leaf classes will hold one of
 * these as a member field; non-CC paths use baseline's inherited
 * `direct_map` unchanged.
 *
 * Storage: `std::deque<std::vector<idx_t>>` with fixed chunk_size=512.
 * Existing chunks never move in memory when new chunks are appended —
 * this is what gives the CC variants safe concurrent add + search:
 * a query's direct-map lookup dereferences a pointer that stays live
 * across concurrent writes to other chunks.
 *
 * IDs are assumed sequential (0..ntotal); the map stores LO-encoded
 * `(list_no, offset)` values at index=id. `add_with_ids` is rejected
 * (check_can_add throws) because the class fundamentally relies on
 * the sequential-id invariant for O(1) lookup.
 */
struct ConcurrentDirectMap {
    constexpr static size_t chunk_size = 512;

    size_t cur_size = 0;
    std::deque<std::vector<idx_t>> offs;

    ConcurrentDirectMap() = default;

    /*********** raw storage accessors ***********/

    size_t size() const {
        return cur_size;
    }

    /// Grow (or shrink-toward) to n_total entries. New entries default
    /// to `id` (typically -1, meaning "unassigned").
    void resize(size_t n_total, idx_t id = -1);

    /// Append a single LO-encoded entry.
    void push_back(const idx_t& id);

    idx_t& operator[](idx_t idx);
    const idx_t& operator[](idx_t idx) const;

    /*********** IndexIVF-shaped direct-map API ***********/

    /// Clear all entries.
    void clear();

    /// Look up the LO-encoded (list_no, offset) for a given id. Throws
    /// if id is out of range or was never assigned (-1 entry).
    idx_t get(idx_t key) const;

    /// Throws if `ids != nullptr`; CC DirectMap only supports
    /// sequential-id add (same invariant as baseline
    /// DirectMap::Type::Array).
    void check_can_add(const idx_t* ids);

    /// Non-thread-safe single add (rarely needed; prefer
    /// ConcurrentDirectMapAdd for batch adds).
    void add_single_id(idx_t id, idx_t list_no, size_t offset);

    /// Initialize the map from an existing ConcurrentArrayInvertedLists,
    /// walking each list's segments and populating `id -> LO(list, offset)`
    /// for every entry. Assumes ids in the invlists are sequential.
    void populate_from(const ConcurrentArrayInvertedLists* cil, size_t ntotal);
};

/** Thread-safe batch-add helper for `ConcurrentDirectMap`. Mirrors the
 * shape of baseline's `::faiss::DirectMapAdd` but talks to a
 * `ConcurrentDirectMap` rather than a `DirectMap`.
 *
 * Usage: construct at the start of an `add_core` body, call `add(i,
 * list_no, offset)` per entry inside the OMP parallel loop, destruct
 * at the end. The ctor pre-reserves capacity for the batch so the
 * per-entry writes don't allocate; they only touch pre-assigned
 * chunks.
 */
struct ConcurrentDirectMapAdd {
    ConcurrentDirectMap& direct_map;
    size_t ntotal;
    size_t n;
    const idx_t* xids;

    ConcurrentDirectMapAdd(
            ConcurrentDirectMap& direct_map,
            size_t n,
            const idx_t* xids);

    /// Record that entry `i` (of this batch) landed at
    /// `(list_no, ofs)` in the invlists.
    void add(size_t i, idx_t list_no, size_t ofs);

    ~ConcurrentDirectMapAdd();
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
