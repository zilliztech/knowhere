/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <numeric>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

//TODO: refactor to decouple dependency between CPU and Cuda, or upgrade faiss
#ifdef USE_GPU
#include "faiss/gpu/utils/DeviceUtils.h"
#include "cuda.h"
#include "cuda_runtime.h"

namespace faiss {
namespace cppcontrib {
namespace knowhere {

/*
 * Use pin memory to build Readonly Inverted list will accelerate cuda memory copy, but it will downgrade cpu ivf search
 * performance. read only inverted list structure will also make ivf search performance not stable. ISSUE 500 mention
 * this problem. Best performance is the original inverted list with non pin memory.
 */

PageLockMemory::PageLockMemory(size_t size) : nbytes(size) {
    auto err = cudaHostAlloc(&(this->data), size, 0);
    if (err) {
        std::string msg =
            "Fail to alloc page lock memory " + std::to_string(size) + ", err code " + std::to_string((int32_t)err);
        FAISS_THROW_MSG(msg);
    }
}

PageLockMemory::~PageLockMemory() {
    CUDA_VERIFY(cudaFreeHost((void*)(this->data)));
}

PageLockMemory::PageLockMemory(const PageLockMemory& other) {
    auto err = cudaHostAlloc(&(this->data), other.nbytes, 0);
    if (err) {
        std::string msg = "Fail to alloc page lock memory " + std::to_string(other.nbytes) + ", err code " +
                          std::to_string((int32_t)err);
        FAISS_THROW_MSG(msg);
    }
    memcpy(this->data, other.data, other.nbytes);
    this->nbytes = other.nbytes;
}

PageLockMemory::PageLockMemory(PageLockMemory &&other) {
    this->data = other.data;
    this->nbytes = other.nbytes;
    other.data = nullptr;
    other.nbytes = 0;
}

}
}
}
#endif



namespace faiss::cppcontrib::knowhere {

/*****************************************
 * ArrayInvertedLists implementation
 *
 * Inherits baseline ::faiss::ArrayInvertedLists for the generic IL
 * surface (codes, ids, list_size, get_codes, get_ids, update_entries,
 * resize, permute_invlists, is_empty, dtor) and fork NormInvertedLists
 * for per-entry norm storage. The fork-local overrides below handle
 * only the norm side-channel.
 ******************************************/

ArrayInvertedLists::ArrayInvertedLists(
        size_t nlist,
        size_t code_size,
        bool _with_norm)
        : ::faiss::ArrayInvertedLists(nlist, code_size),
          with_norm(_with_norm) {
    if (with_norm) {
        code_norms.resize(nlist);
    }
}

size_t ArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code,
        const float* code_norms_in) {
    if (n_entry == 0)
        return 0;
    // Delegate code + id append to baseline, capture the pre-resize
    // offset as our return value.
    size_t o = ::faiss::ArrayInvertedLists::add_entries(
            list_no, n_entry, ids_in, code);
    if (with_norm && code_norms_in != nullptr) {
        code_norms[list_no].resize(o + n_entry);
        memcpy(&code_norms[list_no][o], code_norms_in, sizeof(float) * n_entry);
    }
    return o;
}

size_t ArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code) {
    // Baseline's 4-arg virtual path — reached via inherited
    // merge_from / copy_subset_to. Append codes/ids via baseline; when
    // with_norm=true, extend code_norms with zeros to keep the storage
    // coherent (baseline callers don't have norm data to persist).
    size_t o = ::faiss::ArrayInvertedLists::add_entries(
            list_no, n_entry, ids_in, code);
    if (with_norm) {
        code_norms[list_no].resize(o + n_entry, 0.0f);
    }
    return o;
}

const float* ArrayInvertedLists::get_code_norms(
        size_t list_no,
        size_t /*offset*/) const {
    if (with_norm) {
        assert(list_no < nlist);
        return code_norms[list_no].data();
    } else {
        return nullptr;
    }
}

float ArrayInvertedLists::get_norm(size_t list_no, size_t offset) const {
    if (with_norm) {
        assert(list_no < nlist);
        return code_norms[list_no][offset];
    } else {
        return 0.0f;
    }
}

void ArrayInvertedLists::release_code_norms(
        size_t /*list_no*/,
        const float* /*codes*/) const {
    // noop — code_norms storage is owned by this class.
}

ConcurrentArrayInvertedLists::ConcurrentArrayInvertedLists(
        size_t nlist,
        size_t code_size,
        size_t segment_size,
        bool snorm)
        : ::faiss::InvertedLists(nlist, code_size),
          segment_size(segment_size),
          save_norm(snorm),
          list_cur(nlist) {
    ids.resize(nlist);
    if (save_norm) {
        code_norms.resize(nlist);
    }
    codes.resize(nlist);
    for (int i = 0; i < nlist; i++) {
        list_cur[i].store(0);
    }
}

size_t ConcurrentArrayInvertedLists::cal_segment_num(size_t capacity) const {
    return (capacity / segment_size) + (capacity % segment_size != 0);
}

void ConcurrentArrayInvertedLists::reserve(size_t list_no, size_t capacity) {
    size_t cur_segment_no = ids[list_no].size();
    size_t target_segment_no = cal_segment_num(capacity);

    for (size_t idx = cur_segment_no; idx < target_segment_no; idx++) {
        Segment<uint8_t> segment_codes(segment_size, code_size);
        Segment<idx_t> segment_ids(segment_size, 1);
        Segment<float> segment_code_norms(segment_size, 1);
        codes[list_no].emplace_back(std::move(segment_codes));
        if (save_norm) {
            code_norms[list_no].emplace_back(std::move(segment_code_norms));
        }
        ids[list_no].emplace_back(std::move(segment_ids));
    }
}

void ConcurrentArrayInvertedLists::shrink_to_fit(size_t list_no, size_t capacity) {
    size_t cur_segment_no = ids[list_no].size();
    size_t target_segment_no = cal_segment_num(capacity);

    for (size_t idx = cur_segment_no; idx > target_segment_no; idx--) {
        ids[list_no].pop_back();
        if (save_norm) {
            code_norms[list_no].pop_back();
        }
        codes[list_no].pop_back();
    }
}

size_t ConcurrentArrayInvertedLists::list_size(size_t list_no) const {
    assert(list_no < nlist);
    return list_cur[list_no].load();
}

const uint8_t* ConcurrentArrayInvertedLists::get_codes(size_t list_no) const {
    FAISS_THROW_MSG("not implemented get_codes for non-continuous storage");
}

const idx_t* ConcurrentArrayInvertedLists::get_ids(size_t list_no) const {
    FAISS_THROW_MSG("not implemented get_ids for non-continuous storage");
}
size_t ConcurrentArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* codes_in) {
    // Baseline 4-arg override — delegate to the 5-arg form.
    return add_entries(list_no, n_entry, ids_in, codes_in, nullptr);
}

size_t ConcurrentArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* codes_in,
        const float* code_norms_in) {
    if (n_entry == 0)
        return 0;

    assert(list_no < nlist);
    size_t o = list_size(list_no);

    reserve(list_no, o + n_entry);

    int64_t first_id  = o / segment_size;
    int64_t first_cur = o % segment_size;

    if (first_cur + n_entry <= segment_size) {
        std::memcpy(&ids[list_no][first_id][first_cur], ids_in, n_entry * sizeof(ids_in[0]));
        if (save_norm) {
            std::memcpy(
                    &code_norms[list_no][first_id][first_cur],
                    code_norms_in,
                    n_entry * sizeof(float));
        }
        std::memcpy(&codes[list_no][first_id][first_cur], codes_in, n_entry * code_size);
        list_cur[list_no].fetch_add(n_entry);
        return o;
    }

    //process first segment
    int64_t first_rest = segment_size - first_cur;
    memcpy(&ids[list_no][first_id][first_cur],
           ids_in,
           first_rest * sizeof(ids_in[0]));
    if (save_norm) {
        memcpy(&code_norms[list_no][first_id][first_cur],
           code_norms_in,
           first_rest * sizeof(float));
    }
    memcpy(&codes[list_no][first_id][first_cur],
           codes_in,
           first_rest * code_size);
    list_cur[list_no].fetch_add(first_rest);

    auto rest_entry = n_entry - first_rest;
    auto entry_cur  = first_rest;
    auto segment_id = first_id;
    //process rest segment
    while (rest_entry > 0) {
        segment_id = segment_id + 1;
        if (rest_entry >= segment_size) {
            memcpy(&codes[list_no][segment_id][0],
                   codes_in + entry_cur * code_size,
                   segment_size * code_size);
            if (save_norm) {
                memcpy(&code_norms[list_no][segment_id][0],
                       code_norms_in + entry_cur,
                       segment_size * sizeof(code_norms_in[0]));
            }
            memcpy(&ids[list_no][segment_id][0],
                   ids_in + entry_cur,
                   segment_size * sizeof(ids_in[0]));
            list_cur[list_no].fetch_add(segment_size);

            entry_cur += segment_size;
            rest_entry -= segment_size;
        } else {
            memcpy(&codes[list_no][segment_id][0],
                   codes_in + entry_cur * code_size,
                   rest_entry * code_size);
            if (save_norm) {
                memcpy(&code_norms[list_no][segment_id][0],
                       code_norms_in + entry_cur,
                       rest_entry * sizeof(float));
            }
            memcpy(&ids[list_no][segment_id][0],
                   ids_in + entry_cur,
                   rest_entry * sizeof(ids_in[0]));
            list_cur[list_no].fetch_add(rest_entry);

            entry_cur += rest_entry;
            rest_entry -= rest_entry;
        }
    }
    assert(entry_cur == n_entry);
    assert(rest_entry == 0);
    return o;
}

//Not used
void ConcurrentArrayInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* codes_in) {
    if (n_entry == 0)
        return;

    assert(list_no < nlist);
    assert(n_entry + offset <= list_size(list_no));

    size_t  first_id  = offset / segment_size;
    int64_t first_cur = offset % segment_size;
    int64_t first_rest = segment_size - first_cur;

    if (n_entry <= first_rest) {
        memcpy(&ids[list_no][first_id][first_cur], ids_in, n_entry * sizeof(ids_in[0]));
        memcpy(&codes[list_no][first_id][first_cur], codes_in, n_entry * code_size);
        return ;
    }

    // update first segment
    memcpy(&ids[list_no][first_id][first_cur],
           ids_in,
           first_rest * sizeof(ids_in[0]));
    memcpy(&codes[list_no][first_id][first_cur],
           codes_in,
           first_rest * code_size);

    auto rest_entry = n_entry - first_rest;
    auto entry_cur = first_rest;
    auto segment_id = first_id;
    while (rest_entry > 0) {
        segment_id = segment_id + 1;
        if (rest_entry >= segment_size) {
            memcpy(&codes[list_no][segment_id][0],
                   codes_in + entry_cur * code_size,
                   segment_size * code_size);
            memcpy(&ids[list_no][segment_id][0],
                   ids_in + entry_cur,
                   segment_size * sizeof(ids_in[0]));

            entry_cur += segment_size;
            rest_entry -= segment_size;
        } else {
            memcpy(&codes[list_no][segment_id][0],
                   codes_in + entry_cur * code_size,
                   rest_entry * code_size);
            memcpy(&ids[list_no][segment_id][0],
                   ids_in + entry_cur,
                   rest_entry * sizeof(ids_in[0]));

            entry_cur += rest_entry;
            rest_entry -= rest_entry;
        }
    }
    assert(entry_cur == n_entry);
    assert(rest_entry == 0);
}

ConcurrentArrayInvertedLists::~ConcurrentArrayInvertedLists() {
}

void ConcurrentArrayInvertedLists::resize(size_t list_no, size_t new_size) {
    size_t o = list_size(list_no);

    if (new_size >= o) {
        reserve(list_no, new_size);
        list_cur[list_no].store(new_size);
    } else {
        list_cur[list_no].store(new_size);
        shrink_to_fit(list_no, new_size);
    }

}
size_t ConcurrentArrayInvertedLists::get_segment_num(size_t list_no) const {
    assert(list_no < nlist);
    auto o = list_cur[list_no].load();
    return (o / segment_size) + (o % segment_size != 0);
}
size_t ConcurrentArrayInvertedLists::get_segment_size(
        size_t list_no,
        size_t segment_no) const {
    assert(list_no < nlist);
    auto o = list_cur[list_no].load();
    if (segment_no == 0 && o == 0) {
        return 0;
    }
    auto seg_o = cal_segment_num(o);
    assert(segment_no < seg_o);
    auto o_last = o % segment_size > 0 ? o % segment_size : segment_size;
    if (segment_no < seg_o - 1) {
        return segment_size;
    } else {
        return o_last;
    }
}
size_t ConcurrentArrayInvertedLists::get_segment_offset(
        size_t list_no,
        size_t segment_no) const {
    assert(list_no < nlist);
    auto o = list_cur[list_no].load();
    auto seg_o = cal_segment_num(o);
    assert(segment_no < seg_o);
    return segment_size * segment_no;
}
const uint8_t* ConcurrentArrayInvertedLists::get_codes(
        size_t list_no,
        size_t offset) const {
    assert(list_no < nlist);
    assert(offset < list_size(list_no));
    auto segment_no = offset / segment_size;
    auto segment_off = offset % segment_size;
    return reinterpret_cast<const uint8_t *>(&(codes[list_no][segment_no][segment_off]));
}

const idx_t* ConcurrentArrayInvertedLists::get_ids(
        size_t list_no,
        size_t offset) const {
    assert(list_no < nlist);
    assert(offset < list_size(list_no));
    auto segment_no = offset / segment_size;
    auto segment_off = offset % segment_size;
    return reinterpret_cast<const idx_t *>(&(ids[list_no][segment_no][segment_off]));
}


const uint8_t* ConcurrentArrayInvertedLists::get_single_code(
        size_t list_no,
        size_t offset) const {
    return get_codes(list_no, offset);
}

idx_t ConcurrentArrayInvertedLists::get_single_id(size_t list_no, size_t offset)
        const {
    auto *pItem = get_ids(list_no, offset);
    return *pItem;
}

const float* ConcurrentArrayInvertedLists::get_code_norms(
        size_t list_no,
        size_t offset) const {
    if (!save_norm) {
        return nullptr;
    } else {
        assert(list_no < nlist);
        assert(offset < list_size(list_no));
        auto segment_no = offset / segment_size;
        auto segment_off = offset % segment_size;
        return &(code_norms[list_no][segment_no][segment_off]);
    }
}

float ConcurrentArrayInvertedLists::get_norm(size_t list_no, size_t offset) const {
    if (!save_norm) {
        return 0.0f;
    } else {
        assert(list_no < nlist);
        assert(offset < list_size(list_no));
        auto segment_no = offset / segment_size;
        auto segment_off = offset % segment_size;
        return code_norms[list_no][segment_no][segment_off];
    }
}

void ConcurrentArrayInvertedLists::release_code_norms(
        size_t /*list_no*/,
        const float* /*codes*/) const {
    // noop — code_norms storage is owned by this class (deque segments).
}

/*****************************************************************
 * ReadOnlyArrayInvertedLists implementations
 *****************************************************************/

ReadOnlyArrayInvertedLists::ReadOnlyArrayInvertedLists(
        size_t nlist,
        size_t code_size,
        const std::vector<size_t>& list_length)
        : ::faiss::InvertedLists(nlist, code_size),
          readonly_length(list_length) {
    valid = readonly_length.size() == nlist;
    if (!valid) {
        FAISS_THROW_MSG("Invalid list_length");
    }
    auto total_size = std::accumulate(readonly_length.begin(), readonly_length.end(), 0);
    readonly_offset.reserve(nlist);

#ifndef USE_GPU
    readonly_codes.reserve(total_size * code_size);
    readonly_ids.reserve(total_size);
#endif

    size_t offset = 0;
    for (auto i=0; i<readonly_length.size(); ++i) {
        readonly_offset.emplace_back(offset);
        offset += readonly_length[i];
    }
}

ReadOnlyArrayInvertedLists::ReadOnlyArrayInvertedLists(
        const ArrayInvertedLists& other)
        : ::faiss::InvertedLists(other.nlist, other.code_size) {
    readonly_length.resize(nlist);
    readonly_offset.resize(nlist);
    size_t offset = 0;
    for (auto i = 0; i < other.ids.size(); i++) {
        auto& list_ids = other.ids[i];
        readonly_length[i] = list_ids.size();
        readonly_offset[i] = offset;
        offset += list_ids.size();
    }

#ifdef USE_GPU
    size_t ids_size = offset * sizeof(idx_t);
    size_t codes_size = offset * (this->code_size) * sizeof(uint8_t);
    pin_readonly_codes = std::make_shared<PageLockMemory>(codes_size);
    pin_readonly_ids = std::make_shared<PageLockMemory>(ids_size);

    offset = 0;
    for (auto i = 0; i < other.ids.size(); i++) {
        auto& list_ids = other.ids[i];
        auto& list_codes = other.codes[i];

        uint8_t* ids_ptr = (uint8_t*)(pin_readonly_ids->data) + offset * sizeof(idx_t);
        memcpy(ids_ptr, list_ids.data(), list_ids.size() * sizeof(idx_t));

        uint8_t* codes_ptr = (uint8_t*)(pin_readonly_codes->data) + offset * (this->code_size) * sizeof(uint8_t);
        memcpy(codes_ptr, list_codes.data(), list_codes.size() * sizeof(uint8_t));

        offset += list_ids.size();
    }
#else
    for (auto i = 0; i < other.ids.size(); i++) {
        auto& list_ids = other.ids[i];
        readonly_ids.insert(readonly_ids.end(), list_ids.begin(), list_ids.end());

        auto& list_codes = other.codes[i];
        readonly_codes.insert(readonly_codes.end(), list_codes.begin(), list_codes.end());
    }
#endif

    valid = true;
}

ReadOnlyArrayInvertedLists::ReadOnlyArrayInvertedLists(
        const ArrayInvertedLists& other,
        bool offset_only)
        : ::faiss::InvertedLists(other.nlist, other.code_size) {
    readonly_length.resize(nlist);
    readonly_offset.resize(nlist);
    size_t offset = 0;
    for (auto i = 0; i < other.ids.size(); i++) {
        auto& list_ids = other.ids[i];
        readonly_length[i] = list_ids.size();
        readonly_offset[i] = offset;
        offset += list_ids.size();
    }

#ifdef USE_GPU
    size_t ids_size = offset * sizeof(idx_t);
    size_t codes_size = offset * (this->code_size) * sizeof(uint8_t);
    pin_readonly_codes = std::make_shared<PageLockMemory>(codes_size);
    pin_readonly_ids = std::make_shared<PageLockMemory>(ids_size);

    offset = 0;
    for (auto i = 0; i < other.ids.size(); i++) {
        auto& list_ids = other.ids[i];

        uint8_t* ids_ptr = (uint8_t*)(pin_readonly_ids->data) + offset * sizeof(idx_t);
        memcpy(ids_ptr, list_ids.data(), list_ids.size() * sizeof(idx_t));

        offset += list_ids.size();
    }
#else
    for (auto i = 0; i < other.ids.size(); i++) {
        auto& list_ids = other.ids[i];
        readonly_ids.insert(readonly_ids.end(), list_ids.begin(), list_ids.end());
    }
#endif

    valid = true;
}

ReadOnlyArrayInvertedLists::~ReadOnlyArrayInvertedLists() {
}

bool
ReadOnlyArrayInvertedLists::is_valid() {
    return valid;
}

size_t ReadOnlyArrayInvertedLists::add_entries(
        size_t,
        size_t,
        const idx_t*,
        const uint8_t*) {
    FAISS_THROW_MSG("not implemented");
}

void ReadOnlyArrayInvertedLists::update_entries(
        size_t,
        size_t,
        size_t,
        const idx_t*,
        const uint8_t*) {
    FAISS_THROW_MSG("not implemented");
}

void ReadOnlyArrayInvertedLists::resize(
        size_t,
        size_t) {
    FAISS_THROW_MSG("not implemented");
}

size_t ReadOnlyArrayInvertedLists::list_size(
        size_t list_no) const {
    FAISS_ASSERT(list_no < nlist && valid);
    return readonly_length[list_no];
}

const uint8_t* ReadOnlyArrayInvertedLists::get_codes(
        size_t list_no) const {
    FAISS_ASSERT(list_no < nlist && valid);
#ifdef USE_GPU
    uint8_t* pcodes = (uint8_t*)(pin_readonly_codes->data);
    return pcodes + readonly_offset[list_no] * code_size;
#else
    return readonly_codes.data() + readonly_offset[list_no] * code_size;
#endif
}

const idx_t* ReadOnlyArrayInvertedLists::get_ids(
        size_t list_no) const {
    FAISS_ASSERT(list_no < nlist && valid);
#ifdef USE_GPU
    idx_t* pids = (idx_t*)pin_readonly_ids->data;
    return pids + readonly_offset[list_no];
#else
    return readonly_ids.data() + readonly_offset[list_no];
#endif
}

const idx_t* ReadOnlyArrayInvertedLists::get_all_ids() const {
    FAISS_ASSERT(valid);
#ifdef USE_GPU
    return (idx_t*)(pin_readonly_ids->data);
#else
    return readonly_ids.data();
#endif
}

const uint8_t* ReadOnlyArrayInvertedLists::get_all_codes() const {
    FAISS_ASSERT(valid);
#ifdef USE_GPU
    return (uint8_t*)(pin_readonly_codes->data);
#else
    return readonly_codes.data();
#endif
}

const std::vector<size_t>& ReadOnlyArrayInvertedLists::get_list_length() const {
    FAISS_ASSERT(valid);
    return readonly_length;
}


}


