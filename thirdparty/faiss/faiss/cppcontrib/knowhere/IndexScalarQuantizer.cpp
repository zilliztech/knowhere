/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IndexScalarQuantizer.h>

#include <algorithm>
#include <cstdio>
#include <memory>

#include <omp.h>

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/cppcontrib/knowhere/impl/cc_search.h>
#include <faiss/cppcontrib/knowhere/invlists/DirectMap.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>  // ConcurrentArrayInvertedLists
#include <faiss/utils/Heap.h>
#include <faiss/utils/utils.h>



namespace faiss::cppcontrib::knowhere {

/*******************************************************************
 * IndexIVFScalarQuantizer implementation
 ********************************************************************/

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        ::faiss::ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool by_residual)
        // Path-D step 11.4b: derives from baseline IndexIVF directly.
        // own_invlists_in=false suppresses baseline's auto-allocation;
        // the fork ArrayInvertedLists is installed below once the
        // SQ-derived code_size is known.
        : ::faiss::IndexIVF(quantizer, d, nlist, 0, metric,
                            /*own_invlists_in=*/false),
          sq(d, qtype) {
    code_size = sq.code_size;
    this->by_residual = by_residual;
    replace_invlists(new ArrayInvertedLists(nlist, code_size, false),
                     /*own=*/true);
    is_trained = false;
}

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer() : ::faiss::IndexIVF() {
    by_residual = true;
}

void IndexIVFScalarQuantizer::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* assign) {
    sq.train(n, x);
}

idx_t IndexIVFScalarQuantizer::train_encoder_num_vectors() const {
    return 100000;
}

void IndexIVFScalarQuantizer::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    std::unique_ptr<::faiss::ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());
    size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    memset(codes, 0, (code_size + coarse_size) * n);

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code = codes + i * (code_size + coarse_size);
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }
                if (coarse_size) {
                    encode_listno(list_no, code);
                }
                squant->encode_vector(xi, code + coarse_size);
            }
        }
    }
}

void IndexIVFScalarQuantizer::sa_decode(idx_t n, const uint8_t* codes, float* x)
        const {
    std::unique_ptr<::faiss::ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());
    size_t coarse_size = coarse_code_size();

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * (code_size + coarse_size);
            int64_t list_no = decode_listno(code);
            float* xi = x + i * d;
            squant->decode_vector(code + coarse_size, xi);
            if (by_residual) {
                quantizer->reconstruct(list_no, residual.data());
                for (size_t j = 0; j < d; j++) {
                    xi[j] += residual[j];
                }
            }
        }
    }
}

void IndexIVFScalarQuantizer::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);

    std::unique_ptr<::faiss::ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());

    DirectMapAdd dm_add(direct_map, n, xids);

#pragma omp parallel
    {
        std::vector<float> residual(d);
        std::vector<uint8_t> one_code(code_size);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                int64_t id = xids ? xids[i] : ntotal + i;

                const float* xi = x + i * d;
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }

                memset(one_code.data(), 0, code_size);
                squant->encode_vector(xi, one_code.data());

                size_t ofs = invlists->add_entry(
                        list_no, id, one_code.data(), inverted_list_context);

                dm_add.add(i, list_no, ofs);

            } else if (rank == 0 && list_no == -1) {
                dm_add.add(i, -1, 0);
            }
        }
    }

    ntotal += n;
}

namespace {

// Adapter scanners that implement the fork InvertedListScanner interface
// but delegate distance computation to a baseline SQDistanceComputer.
// Two variants are needed because the IP / L2 paths differ in how the
// coarse-centroid residual is folded into the distance:
//   IP: dis = coarse_dis + dc.query_to_code(code)
//   L2: the query is shifted into the centroid frame in set_list(), and
//       the DC already produces the final L2 distance on every code.
//
// Per-list k-NN stats (scan_cnt, nheap_updates) are reported via
// ResultHandler::stats (fork retirement, formerly a mutable scan_cnt
// member on the fork scanner). These SQ scanners now correctly report
// post-filter scan_cnt, so the IVF probe loop's max_codes /
// ensure_topk_full budget tracking works on SQ paths just as it does
// on IVFFlat.

class BaselineIVFSQScannerIP : public InvertedListScanner {
   public:
    BaselineIVFSQScannerIP(
            std::unique_ptr<::faiss::ScalarQuantizer::SQDistanceComputer> dc,
            size_t code_size_in,
            bool store_pairs_in,
            const IDSelector* sel_in,
            bool by_residual_in)
            : dc_(std::move(dc)), by_residual_(by_residual_in) {
        store_pairs = store_pairs_in;
        sel = sel_in;
        code_size = code_size_in;
        keep_max = true;
    }

    void set_query(const float* query) override {
        dc_->set_query(query);
    }

    void set_list(idx_t list_no_in, float coarse_dis) override {
        this->list_no = list_no_in;
        accu0_ = by_residual_ ? coarse_dis : 0.0f;
    }

    float distance_to_code(const uint8_t* code) const override {
        return accu0_ + dc_->query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ::faiss::ResultHandler& handler) const override {
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            if (!selector_accepts(j, ids)) {
                continue;
            }
            handler.stats.scan_cnt++;
            float dis = accu0_ + dc_->query_to_code(codes + j * code_size);
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            if (handler.add_result(dis, id)) {
                handler.stats.nheap_updates++;
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        for (size_t j = 0; j < list_size; j++) {
            if (!selector_accepts(j, ids)) {
                continue;
            }
            float dis = accu0_ + dc_->query_to_code(codes + j * code_size);
            out.emplace_back(ids[j], dis);
        }
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++) {
            if (!selector_accepts(j, ids)) {
                continue;
            }
            res.stats.scan_cnt++;
            float dis = accu0_ + dc_->query_to_code(codes + j * code_size);
            if (dis > radius) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
                res.stats.nheap_updates++;
            }
        }
    }

   private:
    bool selector_accepts(size_t j, const idx_t* ids) const {
        if (!sel) {
            return true;
        }
        return sel->is_member(store_pairs ? static_cast<idx_t>(j) : ids[j]);
    }

    std::unique_ptr<::faiss::ScalarQuantizer::SQDistanceComputer> dc_;
    bool by_residual_;
    float accu0_ = 0.0f;
};

class BaselineIVFSQScannerL2 : public InvertedListScanner {
   public:
    BaselineIVFSQScannerL2(
            std::unique_ptr<::faiss::ScalarQuantizer::SQDistanceComputer> dc,
            int d_in,
            size_t code_size_in,
            const Index* quantizer_in,
            bool store_pairs_in,
            const IDSelector* sel_in,
            bool by_residual_in)
            : dc_(std::move(dc)),
              by_residual_(by_residual_in),
              quantizer_(quantizer_in),
              tmp_(d_in) {
        store_pairs = store_pairs_in;
        sel = sel_in;
        code_size = code_size_in;
        keep_max = false;
    }

    void set_query(const float* query) override {
        x_ = query;
        if (!by_residual_) {
            dc_->set_query(query);
        }
    }

    void set_list(idx_t list_no_in, float /*coarse_dis*/) override {
        this->list_no = list_no_in;
        if (by_residual_) {
            quantizer_->compute_residual(x_, tmp_.data(), list_no_in);
            dc_->set_query(tmp_.data());
        }
    }

    float distance_to_code(const uint8_t* code) const override {
        return dc_->query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ::faiss::ResultHandler& handler) const override {
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            if (!selector_accepts(j, ids)) {
                continue;
            }
            handler.stats.scan_cnt++;
            float dis = dc_->query_to_code(codes + j * code_size);
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            if (handler.add_result(dis, id)) {
                handler.stats.nheap_updates++;
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_and_return(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            std::vector<::knowhere::DistId>& out) const override {
        for (size_t j = 0; j < list_size; j++) {
            if (!selector_accepts(j, ids)) {
                continue;
            }
            float dis = dc_->query_to_code(codes + j * code_size);
            out.emplace_back(ids[j], dis);
        }
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++) {
            if (!selector_accepts(j, ids)) {
                continue;
            }
            res.stats.scan_cnt++;
            float dis = dc_->query_to_code(codes + j * code_size);
            if (dis < radius) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
                res.stats.nheap_updates++;
            }
        }
    }

   private:
    bool selector_accepts(size_t j, const idx_t* ids) const {
        if (!sel) {
            return true;
        }
        return sel->is_member(store_pairs ? static_cast<idx_t>(j) : ids[j]);
    }

    std::unique_ptr<::faiss::ScalarQuantizer::SQDistanceComputer> dc_;
    bool by_residual_;
    const Index* quantizer_;
    const float* x_ = nullptr;
    std::vector<float> tmp_;
};

}  // namespace

InvertedListScanner* IndexIVFScalarQuantizer::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    FAISS_THROW_IF_NOT(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);
    std::unique_ptr<::faiss::ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer(metric_type));
    if (metric_type == METRIC_INNER_PRODUCT) {
        return new BaselineIVFSQScannerIP(
                std::move(dc), code_size, store_pairs, sel, by_residual);
    }
    return new BaselineIVFSQScannerL2(
            std::move(dc),
            static_cast<int>(d),
            code_size,
            quantizer,
            store_pairs,
            sel,
            by_residual);
}

void IndexIVFScalarQuantizer::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);

    if (by_residual) {
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        sq.decode(code, recons, 1);
        for (int i = 0; i < d; ++i) {
            recons[i] += centroid[i];
        }
    } else {
        sq.decode(code, recons, 1);
    }
}

// Path-D step 11.4b: CC dispatch hosted on IndexIVFScalarQuantizer
// instead of fork IndexIVF base. The qianya path
// (`src/index/ivf/ivf.cc` static_casts a deserialized
// IndexIVFScalarQuantizer to IndexIVFScalarQuantizerCC) means an
// IndexIVFScalarQuantizer instance can hold a
// ConcurrentArrayInvertedLists at runtime even though the static
// type is non-CC. Among non-CC IVF leaves, IVFSQ is the only one
// that can hit this case (IVFFlat / IVFPQ / IVFRaBitQ / IVFFastScan
// never end up with CC invlists), so the runtime dispatch lives
// here rather than on the fork base. CC-leaf classes
// (IndexIVFScalarQuantizerCC, IndexIVFFlatCC) override these
// methods themselves and delegate to cc_impl unconditionally
// (their invlists is statically known to be CC).
void IndexIVFScalarQuantizer::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) const {
    if (auto* cil =
                dynamic_cast<const ConcurrentArrayInvertedLists*>(invlists)) {
        cc_impl::search_preassigned(
                *this,
                *cil,
                n,
                x,
                k,
                keys,
                coarse_dis,
                distances,
                labels,
                store_pairs,
                params,
                ivf_stats);
        return;
    }
    ::faiss::IndexIVF::search_preassigned(
            n,
            x,
            k,
            keys,
            coarse_dis,
            distances,
            labels,
            store_pairs,
            params,
            ivf_stats);
}

void IndexIVFScalarQuantizer::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        faiss::RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    if (auto* cil =
                dynamic_cast<const ConcurrentArrayInvertedLists*>(invlists)) {
        cc_impl::range_search_preassigned(
                *this,
                *cil,
                nx,
                x,
                radius,
                keys,
                coarse_dis,
                result,
                store_pairs,
                params,
                stats);
        return;
    }
    ::faiss::IndexIVF::range_search_preassigned(
            nx,
            x,
            radius,
            keys,
            coarse_dis,
            result,
            store_pairs,
            params,
            stats);
}

}
