/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/invlists/ConcurrentDirectMap.h>

#include <cassert>

#include <faiss/impl/FaissAssert.h>
#include <faiss/cppcontrib/knowhere/invlists/DirectMap.h>  // lo_build / lo_listno / lo_offset
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

void ConcurrentDirectMap::resize(size_t n_total, idx_t id) {
    size_t target_chunk_num =
            (n_total / chunk_size) + (n_total % chunk_size != 0);
    size_t cur_chunk_no = offs.size();
    for (size_t idx = cur_chunk_no; idx < target_chunk_num; idx++) {
        offs.emplace_back(chunk_size, id);
    }
    cur_size = n_total;
}

void ConcurrentDirectMap::push_back(const idx_t& id) {
    resize(cur_size + 1);
    this->operator[](cur_size - 1) = id;
}

idx_t& ConcurrentDirectMap::operator[](idx_t idx) {
    FAISS_THROW_IF_NOT_MSG(
            idx >= 0 && idx < static_cast<idx_t>(cur_size), "invalid key");
    return offs[idx / chunk_size][idx % chunk_size];
}

const idx_t& ConcurrentDirectMap::operator[](idx_t idx) const {
    FAISS_THROW_IF_NOT_MSG(
            idx >= 0 && idx < static_cast<idx_t>(cur_size), "invalid key");
    return offs[idx / chunk_size][idx % chunk_size];
}

void ConcurrentDirectMap::clear() {
    offs.clear();
    cur_size = 0;
}

idx_t ConcurrentDirectMap::get(idx_t key) const {
    FAISS_THROW_IF_NOT_MSG(
            key >= 0 && key < static_cast<idx_t>(cur_size), "invalid key");
    idx_t lo = (*this)[key];
    FAISS_THROW_IF_NOT_MSG(lo >= 0, "-1 entry in concurrent direct_map");
    return lo;
}

void ConcurrentDirectMap::check_can_add(const idx_t* ids) {
    if (ids) {
        FAISS_THROW_MSG(
                "ConcurrentDirectMap requires sequential ids; "
                "add_with_ids is not supported");
    }
}

void ConcurrentDirectMap::add_single_id(
        idx_t id,
        idx_t list_no,
        size_t offset) {
    if (list_no >= 0) {
        assert(static_cast<size_t>(id) == cur_size);
        push_back(lo_build(list_no, offset));
    } else {
        push_back(-1);
    }
}

void ConcurrentDirectMap::populate_from(
        const ConcurrentArrayInvertedLists* cil,
        size_t ntotal) {
    // Path-D step 10.10: call segment accessors directly on the
    // concrete ConcurrentArrayInvertedLists type rather than going
    // through the fork InvertedLists base — the base's virtual segment
    // API is being removed in this step.
    FAISS_THROW_IF_NOT_MSG(
            cil != nullptr, "populate_from needs a non-null CC invlists");

    clear();
    resize(ntotal, -1);

    for (size_t key = 0; key < cil->nlist; key++) {
        size_t segment_num = cil->get_segment_num(key);
        for (size_t segment_idx = 0; segment_idx < segment_num;
             segment_idx++) {
            size_t segment_size = cil->get_segment_size(key, segment_idx);
            size_t segment_offset =
                    cil->get_segment_offset(key, segment_idx);
            const idx_t* ids = cil->get_ids(key, segment_offset);
            for (long ofs = 0; ofs < static_cast<long>(segment_size); ofs++) {
                FAISS_THROW_IF_NOT_MSG(
                        0 <= ids[ofs] &&
                                static_cast<size_t>(ids[ofs]) < ntotal,
                        "ConcurrentDirectMap supported only for sequential ids");
                (*this)[ids[ofs]] =
                        lo_build(key, segment_offset + ofs);
            }
        }
    }
}

/********************* ConcurrentDirectMapAdd implementation */

ConcurrentDirectMapAdd::ConcurrentDirectMapAdd(
        ConcurrentDirectMap& direct_map_in,
        size_t n_in,
        const idx_t* xids_in)
        : direct_map(direct_map_in), n(n_in), xids(xids_in) {
    FAISS_THROW_IF_NOT(xids == nullptr);
    ntotal = direct_map.size();
    // Pre-allocate chunks for the upcoming batch so the parallel
    // per-entry writes only touch already-reserved storage.
    direct_map.resize(ntotal + n, -1);
}

void ConcurrentDirectMapAdd::add(size_t i, idx_t list_no, size_t ofs) {
    direct_map[ntotal + i] = lo_build(list_no, ofs);
}

ConcurrentDirectMapAdd::~ConcurrentDirectMapAdd() = default;

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
