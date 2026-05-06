/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IVFIteratorWorkspace.h>

#include <cinttypes>

#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexIVF.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>  // ConcurrentArrayInvertedLists
#include <faiss/impl/FaissAssert.h>

namespace faiss::cppcontrib::knowhere {

// ---- IVFIteratorWorkspace ----

IVFIteratorWorkspace::IVFIteratorWorkspace(
        const float* query_data_in,
        const size_t d,
        const IVFSearchParameters* search_params)
        : query_data(query_data_in, query_data_in + d),
          search_params(search_params),
          dis_refine(nullptr) {}

IVFIteratorWorkspace::~IVFIteratorWorkspace() {}

// ---- IVFBaseIteratorWorkspace ----

IVFBaseIteratorWorkspace::IVFBaseIteratorWorkspace(
        const IndexIVF* ivf_index_in,
        const float* query_data_in,
        const IVFSearchParameters* params)
        : IVFIteratorWorkspace(query_data_in, ivf_index_in->d, params),
          ivf_index(ivf_index_in) {
    // Snapshot list sizes
    auto coarse_list_sizes_buf = std::make_unique<size_t[]>(ivf_index->nlist);
    size_t count = 0;
    auto max_coarse_list_size = 0;
    for (size_t list_no = 0; list_no < ivf_index->nlist; ++list_no) {
        auto list_size = ivf_index->invlists->list_size(list_no);
        coarse_list_sizes_buf[list_no] = list_size;
        count += list_size;
        if (list_size > max_coarse_list_size) {
            max_coarse_list_size = list_size;
        }
    }

    // Compute nprobe and backup_count_threshold
    size_t np = this->search_params->nprobe
            ? this->search_params->nprobe
            : ivf_index->nprobe;
    np = std::min(ivf_index->nlist, np);
    this->backup_count_threshold = count * np / ivf_index->nlist;
    auto max_backup_count =
            max_coarse_list_size + this->backup_count_threshold;

    // Coarse quantization
    auto coarse_idx_buf = std::make_unique<idx_t[]>(ivf_index->nlist);
    auto coarse_dis_buf = std::make_unique<float[]>(ivf_index->nlist);
    ivf_index->quantizer->search(
            1,
            this->query_data.data(),
            ivf_index->nlist,
            coarse_dis_buf.get(),
            coarse_idx_buf.get(),
            this->search_params
                    ? this->search_params->quantizer_params
                    : nullptr);

    this->coarse_idx = std::move(coarse_idx_buf);
    this->coarse_dis = std::move(coarse_dis_buf);
    this->coarse_list_sizes = std::move(coarse_list_sizes_buf);
    this->nprobe = np;
    this->dists.reserve(max_backup_count);
}

void IVFBaseIteratorWorkspace::next_batch(size_t current_backup_count) {
    this->dists.clear();

    while (current_backup_count + this->dists.size() <
                   this->backup_count_threshold &&
           this->next_visit_coarse_list_idx < ivf_index->nlist) {
        auto next_list_idx = this->next_visit_coarse_list_idx;
        this->next_visit_coarse_list_idx++;

        ivf_index->invlists->prefetch_lists(
                this->coarse_idx.get() + next_list_idx, 1);
        const auto list_no = this->coarse_idx[next_list_idx];
        const auto coarse_list_centroid_dist =
                this->coarse_dis[next_list_idx];

        // max_codes is the size of the list when we started the
        // iteration so that we won't search vectors added during the
        // iteration (for IVFCC).
        const auto max_codes = this->coarse_list_sizes
                                       [this->coarse_idx[next_list_idx]];
        if (list_no < 0) {
            // not enough centroids for multiprobe
            continue;
        }
        FAISS_THROW_IF_NOT_FMT(
                list_no < (idx_t)ivf_index->nlist,
                "Invalid list_no=%" PRId64 " nlist=%zd\n",
                list_no,
                ivf_index->nlist);

        // don't waste time on empty lists
        void* inverted_list_context = this->search_params
                ? this->search_params->inverted_list_context
                : nullptr;

        if (ivf_index->invlists->is_empty(list_no, inverted_list_context)) {
            continue;
        }

        // get scanner — Path-D step 11.4b: fork IndexIVF no longer
        // declares a covariant get_InvertedListScanner override, so the
        // call returns baseline's wider `::faiss::InvertedListScanner*`.
        // Take ownership in a baseline-typed unique_ptr first, then
        // dynamic_cast to the knowhere hook interface for access to
        // the fork-only virtuals (`set_list_segment`,
        // `scan_codes_and_return`) used below. All iterator-supported
        // fork-derived scanners implement these hooks, so the cast
        // succeeds for supported paths while preserving a safety net if
        // an unsupported baseline-only scanner were ever passed.
        IDSelector* sel = this->search_params
                ? this->search_params->sel
                : nullptr;
        std::unique_ptr<::faiss::InvertedListScanner> base_scanner(
                ivf_index->get_InvertedListScanner(
                        false, sel, this->search_params));
        auto* hooks =
                dynamic_cast<KnowhereInvertedListScannerHooks*>(
                        base_scanner.get());
        FAISS_THROW_IF_NOT_MSG(
                hooks != nullptr,
                "IVFIteratorWorkspace: scanner does not implement knowhere hooks");
        base_scanner->set_query(this->query_data.data());
        base_scanner->set_list(list_no, coarse_list_centroid_dist);

        // Path-D step 10.10: segment-aware walk for CC invlists (where
        // list storage is chunked across ConcurrentArrayInvertedLists
        // segments), single-scan fast path for everything else.
        // Runtime dispatch via dynamic_cast — the segment API used to
        // be a virtual on fork::InvertedLists base with a non-trivial
        // override only on the CC invlists; that virtual is going away
        // in this step, so the cast is the replacement.
        if (auto* cil =
                    dynamic_cast<const ConcurrentArrayInvertedLists*>(
                            ivf_index->invlists)) {
            // CC path: walk segments.
            size_t segment_num = cil->get_segment_num(list_no);
            size_t scan_cnt = 0;
            for (size_t segment_idx = 0; segment_idx < segment_num;
                 segment_idx++) {
                size_t segment_size =
                        cil->get_segment_size(list_no, segment_idx);
                size_t should_scan_size =
                        std::min(segment_size, max_codes - scan_cnt);
                scan_cnt += should_scan_size;
                if (should_scan_size <= 0) {
                    break;
                }
                size_t segment_offset =
                        cil->get_segment_offset(list_no, segment_idx);
                // Direct access on concrete CC invlists — the pointers
                // returned are deque-chunk interior pointers that don't
                // need RAII release.
                const uint8_t* codes =
                        cil->get_codes(list_no, segment_offset);
                const idx_t* ids = cil->get_ids(list_no, segment_offset);

                // Per-segment norms are fetched by the scanner itself in
                // set_list_segment.
                hooks->set_list_segment(segment_offset);
                hooks->scan_codes_and_return(
                        should_scan_size, codes, ids, this->dists);
            }
        } else {
            // Non-CC single-segment path: max_codes is already bounded
            // by list_size (since coarse_list_sizes snapshot is that
            // size). One scan covers the whole list.
            size_t list_size = ivf_index->invlists->list_size(list_no);
            size_t should_scan_size = std::min(list_size, max_codes);
            if (should_scan_size > 0) {
                InvertedLists::ScopedCodes scodes(
                        ivf_index->invlists, list_no);
                InvertedLists::ScopedIds sids(
                        ivf_index->invlists, list_no);
                hooks->scan_codes_and_return(
                        should_scan_size,
                        scodes.get(),
                        sids.get(),
                        this->dists);
            }
        }
    }
}

}  // namespace faiss::cppcontrib::knowhere
