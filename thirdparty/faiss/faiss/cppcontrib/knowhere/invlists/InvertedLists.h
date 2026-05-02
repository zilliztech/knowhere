/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

/**
 * Definition of inverted lists + a few common classes that implement
 * the interface.
 */

#include <atomic>
#include <cassert>
#include <deque>
#include <memory>
#include <vector>

#include <faiss/invlists/InvertedLists.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

struct PageLockMemory {
public:
    PageLockMemory() : data(nullptr), nbytes(0) {}

    PageLockMemory(size_t size);

    ~PageLockMemory();

    PageLockMemory(const PageLockMemory& other);

    PageLockMemory(PageLockMemory &&other);

    inline size_t size() {
        return nbytes;
    }

    void *data;
    size_t nbytes;
};
using PageLockMemoryPtr = std::shared_ptr<PageLockMemory>;

// Fork's InvertedListsIterator is a strict subset of baseline's
// (the three abstract virtuals is_available / next / get_id_and_codes
// are identical; baseline additionally carries has_search_callbacks_
// and two non-abstract on_* hooks that default to noops). Collapsed to
// an alias here (Path-D step 7b) so that fork InvertedListScanner (now
// subclassing baseline) can accept fork InvertedLists iterators
// without an explicit cast.
using InvertedListsIterator = ::faiss::InvertedListsIterator;

/** Fork-only capability interface: cosine per-entry norm access + the
 * fork-specific norm-bearing add path.
 *
 * Canonical fork-side extension surface over baseline's
 * ::faiss::InvertedLists. Implemented by concrete fork invlists classes
 * that actually store per-entry L2 norms: ArrayInvertedLists (when
 * with_norm=true), ConcurrentArrayInvertedLists (when save_norm=true),
 * OnDiskInvertedLists (when with_norm=true). Classes that don't store
 * norms (BlockInvertedLists, ReadOnlyArrayInvertedLists) do NOT
 * implement this interface — callers wanting norms dynamic_cast to
 * NormInvertedLists* and fall through on null.
 *
 * History: introduced in Path-D step 10.14b during the fork
 * InvertedLists hierarchy flattening (10.14c-i); took over the role of
 * the deleted fork `InvertedLists` class's norm virtuals once all
 * concrete classes had been reparented onto baseline IL + this mixin.
 */
struct NormInvertedLists {
    /// get the per-entry norm pointer for a list starting at offset;
    /// implementations return the pointer into their norm storage.
    virtual const float* get_code_norms(size_t list_no, size_t offset) const = 0;

    /// get a single norm value at (list_no, offset).
    virtual float get_norm(size_t list_no, size_t offset) const = 0;

    /// release pointer returned by get_code_norms.
    virtual void release_code_norms(size_t list_no, const float* codes) const = 0;

    /// add one entry with an optional per-entry norm. Default body
    /// delegates to the 5-arg add_entries; concrete subclasses that
    /// have a faster single-entry path may override.
    virtual size_t add_entry(
            size_t list_no,
            idx_t theid,
            const uint8_t* code,
            const float* code_norm,
            void* /*inverted_list_context*/ = nullptr) {
        return add_entries(list_no, 1, &theid, code, code_norm);
    }

    /// add n entries with optional per-entry norms.
    virtual size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code,
            const float* code_norm) = 0;

    virtual ~NormInvertedLists() = default;
};

// Generic IL type — same as baseline's. Unqualified `InvertedLists`
// in fork-namespace code resolves here. `InvertedLists::ScopedCodes`
// and `InvertedLists::ScopedIds` call sites pick up baseline's nested
// RAII wrappers automatically.
using InvertedLists = ::faiss::InvertedLists;

/// simple (default) implementation as an array of inverted lists.
/// Path-D step 10.14e: reparented from fork InvertedLists to baseline
/// ::faiss::ArrayInvertedLists + fork NormInvertedLists. Baseline
/// provides all generic-IL state + methods (`codes`, `ids`, list_size,
/// get_codes, get_ids, 4-arg add_entries, update_entries, resize,
/// permute_invlists, is_empty, dtor); fork adds per-entry norm storage
/// (`with_norm`, `code_norms`) and overrides the NormInvertedLists
/// interface methods.
struct ArrayInvertedLists : ::faiss::ArrayInvertedLists, NormInvertedLists {
    bool with_norm = false;
    std::vector<std::vector<float>> code_norms; // code norms

    ArrayInvertedLists(size_t nlist, size_t code_size, bool with_norm = false);

    // NormInvertedLists interface implementations.
    const float* get_code_norms(size_t list_no, size_t offset) const override;
    float get_norm(size_t list_no, size_t offset) const override;
    void release_code_norms(size_t list_no, const float* codes) const override;

    /// Fork's norm-bearing write path (NormInvertedLists 5-arg).
    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code,
            const float* code_norm) override;

    /// Baseline's 4-arg write path — overridden here so that the norm
    /// side-channel stays coherent with codes/ids when with_norm=true.
    /// Callers passing no norms should call the 5-arg form directly;
    /// this path is reached via inherited baseline internals
    /// (merge_from, copy_subset_to) and zero-fills norms if with_norm.
    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;
};

// A Concurrent implementation for inverted lists.
// Path-D step 10.14f: reparented from fork InvertedLists to baseline
// ::faiss::InvertedLists + fork NormInvertedLists. No baseline concrete
// equivalent exists (CC is fork-only), so the base is baseline's
// abstract InvertedLists. All CC-specific state + segment machinery
// stays unchanged; only the inheritance list changes.
struct ConcurrentArrayInvertedLists : ::faiss::InvertedLists, NormInvertedLists {
    template <typename T>
    struct Segment {
        Segment(size_t segment_size, size_t code_size) : segment_size_(segment_size), code_size_(code_size) {
            data_.reserve(segment_size_ * code_size_);
        }
        T& operator[](idx_t idx) {
            assert(idx < segment_size_);
            return data_[idx * code_size_];
        }
        const T& operator[](idx_t idx) const {
            assert(idx < segment_size_);
            return data_[idx * code_size_];
        }
        size_t segment_size_;
        size_t code_size_;
        std::vector<T> data_;
    };

    ConcurrentArrayInvertedLists(size_t nlist, size_t code_size, size_t segment_size, bool save_normal);

    size_t cal_segment_num(size_t capacity) const;
    void reserve(size_t list_no, size_t capacity);
    void shrink_to_fit(size_t list_no, size_t capacity);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    // Path-D step 10.10: segment + offset-slice accessors are now
    // plain non-virtual methods on this class (no longer overrides of
    // any base virtual). Callers that need CC-specific segment walks
    // work with `ConcurrentArrayInvertedLists*` directly.
    size_t get_segment_num(size_t list_no) const;
    size_t get_segment_size(size_t list_no, size_t segment_no) const;
    size_t get_segment_offset(size_t list_no, size_t segment_no) const;

    const uint8_t* get_codes(size_t list_no, size_t offset) const;
    const idx_t* get_ids(size_t list_no, size_t offset) const;

    const float* get_code_norms(size_t list_no, size_t offset) const override;
    float get_norm(size_t list_no, size_t offset) const override;
    void release_code_norms(size_t list_no, const float* codes)
            const override;

    idx_t get_single_id(size_t list_no, size_t offset) const override;
    const uint8_t* get_single_code(size_t list_no, size_t offset) const override;

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code,
            const float* code_norms) override;

    /// Baseline 4-arg override delegating to 5-arg with nullptr norms.
    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;

    ~ConcurrentArrayInvertedLists() override;

    size_t segment_size;
    bool save_norm;
    std::vector<std::atomic<size_t>> list_cur;
    std::vector<std::deque<Segment<uint8_t>>> codes;
    std::vector<std::deque<Segment<idx_t>>> ids;
    std::vector<std::deque<Segment<float>>> code_norms;
};

/// Read-only mmap-backed invlists used on the MMAP deserialize path.
/// Path-D step 10.14g: reparented from fork InvertedLists to baseline
/// ::faiss::InvertedLists directly. Does NOT implement NormInvertedLists
/// — this class never stores per-entry norms (no cosine variant uses
/// it). Callers that ask a `::faiss::InvertedLists*` for norms via
/// `dynamic_cast<NormInvertedLists*>(...)` get null and fall through
/// (same outcome as before — the old fork-IL default returned nullptr).
struct ReadOnlyArrayInvertedLists: ::faiss::InvertedLists {
    // for GPU
    PageLockMemoryPtr pin_readonly_codes;
    PageLockMemoryPtr pin_readonly_ids;

    // for CPU
    std::vector<uint8_t> readonly_codes;
    std::vector<idx_t> readonly_ids;

    std::vector<size_t> readonly_length;
    std::vector<size_t> readonly_offset;
    bool valid;

    ReadOnlyArrayInvertedLists(
            size_t nlist,
            size_t code_size,
            const std::vector<size_t>& list_length);

    explicit ReadOnlyArrayInvertedLists(const ArrayInvertedLists& other);
    explicit ReadOnlyArrayInvertedLists(const ArrayInvertedLists& other, bool offset);

    // Use default copy construct, just copy pointer, DON'T COPY pin_readonly_codes AND pin_readonly_ids
    // explicit ReadOnlyArrayInvertedLists(const ReadOnlyArrayInvertedLists &);
    // explicit ReadOnlyArrayInvertedLists(ReadOnlyArrayInvertedLists &&);
    virtual ~ReadOnlyArrayInvertedLists();

    size_t list_size(size_t list_no) const override;
    const uint8_t * get_codes(size_t list_no) const override;
    const idx_t * get_ids(size_t list_no) const override;

    const uint8_t * get_all_codes() const;
    const idx_t * get_all_ids() const;
    const std::vector<size_t>& get_list_length() const;

    /// Baseline 4-arg add_entries — throws (read-only).
    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t *ids,
            const uint8_t *code) override;

    void resize(size_t list_no, size_t new_size) override;

    bool is_valid();
};


}
}
} // namespace faiss
