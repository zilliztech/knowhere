/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_DIRECT_MAP_H
#define FAISS_DIRECT_MAP_H

#include <faiss/impl/IDSelector.h>
#include <faiss/invlists/InvertedLists.h>
#include <unordered_map>
#include "faiss/impl/FaissAssert.h"

namespace faiss {

// When offsets list id + offset are encoded in an uint64
// we call this LO = list-offset

inline uint64_t lo_build(uint64_t list_id, uint64_t offset) {
    return list_id << 32 | offset;
}

inline uint64_t lo_listno(uint64_t lo) {
    return lo >> 32;
}

inline uint64_t lo_offset(uint64_t lo) {
    return lo & 0xffffffff;
}

/**
 * Direct map: a way to map back from ids to inverted lists
 */
struct DirectMap {
    enum Type {
        NoMap = 0,    // default
        Array = 1,    // sequential ids (only for add, no add_with_ids)
        Hashtable = 2, // arbitrary ids
        ConcurrentArray = 3 // sequential ids (support add/get concurrently; append only)
    };
    Type type;

    struct ConcurrentArray {
        ConcurrentArray() = default;
        virtual ~ConcurrentArray() = default;
        size_t size() const {
            return cur_size;
        }
        void resize(size_t n_total, idx_t id = -1) {
            size_t target_chunk_num = (n_total / chunk_size) + (n_total % chunk_size != 0);
            size_t cur_chunk_no = offs.size();
            for (size_t idx = cur_chunk_no; idx < target_chunk_num; idx++) {
                offs.emplace_back(chunk_size, id);
            }
            cur_size = n_total;
        }
        void push_back(const idx_t& id) {
            resize(cur_size + 1);
            this->operator[](cur_size-1) = id;
        }
        idx_t& operator[](idx_t idx) {
            FAISS_THROW_IF_NOT_MSG(idx >= 0 && idx < static_cast<idx_t>(cur_size), "invalid key");
            return offs[idx / chunk_size][idx % chunk_size];
        }
        const idx_t& operator[](idx_t idx) const {
            FAISS_THROW_IF_NOT_MSG(idx >= 0 && idx < static_cast<idx_t>(cur_size), "invalid key");
            return offs[idx / chunk_size][idx % chunk_size];
        }
        constexpr static size_t chunk_size = 512;
        size_t cur_size = 0;
        std::deque<std::vector<idx_t>> offs;
    };

    /// map for direct access to the elements. Map ids to LO-encoded entries.
    std::vector<idx_t> array;
    struct ConcurrentArray concurrentArray;
    std::unordered_map<idx_t, idx_t> hashtable;

    DirectMap();

    /// set type and initialize
    void set_type(Type new_type, const InvertedLists* invlists, size_t ntotal);

    /// get an entry
    idx_t get(idx_t id) const;

    /// for quick checks
    bool no() const {
        return type == NoMap;
    }

    /**
     * update the direct_map
     */

    /// throw if Array and ids is not NULL
    void check_can_add(const idx_t* ids);

    /// non thread-safe version
    void add_single_id(idx_t id, idx_t list_no, size_t offset);

    /// remove all entries
    void clear();

    /**
     * operations on inverted lists that require translation with a DirectMap
     */

    /// remove ids from the InvertedLists, possibly using the direct map
    size_t remove_ids(const IDSelector& sel, InvertedLists* invlists);

    /// update entries, using the direct map
    void update_codes(
            InvertedLists* invlists,
            int n,
            const idx_t* ids,
            const idx_t* list_nos,
            const uint8_t* codes);
};

/// Thread-safe way of updating the direct_map
struct DirectMapAdd {
    using Type = DirectMap::Type;

    DirectMap& direct_map;
    DirectMap::Type type;
    size_t ntotal;
    size_t n;
    const idx_t* xids;

    std::vector<idx_t> all_ofs;

    DirectMapAdd(DirectMap& direct_map, size_t n, const idx_t* xids);

    /// add vector i (with id xids[i]) at list_no and offset
    void add(size_t i, idx_t list_no, size_t offset);

    ~DirectMapAdd();
};

} // namespace faiss

#endif
