/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IVFFastScanIteratorWorkspace.h>

#include <cinttypes>

#include <faiss/IndexIVFFastScan.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>
#include <faiss/cppcontrib/knowhere/IndexIVFPQFastScan.h>
#include <faiss/cppcontrib/knowhere/IndexScaNN.h>
#include <faiss/impl/fast_scan/accumulate_loops.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/impl/fast_scan/LookupTableScaler.h>
#include <faiss/impl/fast_scan/simd_result_handlers.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss::cppcontrib::knowhere {

using namespace simd_result_handlers;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

// ---- IVFFastScanIteratorWorkspace::Impl ----
// All AVX2-dependent members live here, invisible to baseline TUs.

struct IVFFastScanIteratorWorkspace::Impl {
    const ::faiss::IndexIVFFastScan* index = nullptr;
    size_t dim12;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    float normalizers[2];

    void next_batch(IVFFastScanIteratorWorkspace& ws, size_t current_backup_count);

    void init(
            IVFFastScanIteratorWorkspace& ws,
            const ::faiss::IndexIVFFastScan* index,
            const IVFSearchParameters* params);

    void get_interator_next_batch_implem_10(
            const ::faiss::IndexIVFFastScan* index,
            IVFFastScanIteratorWorkspace& ws,
            SIMDResultHandlerToFloat& handler,
            size_t current_backup_count);
};

// ---- IVFFastScanIteratorWorkspace ----

IVFFastScanIteratorWorkspace::IVFFastScanIteratorWorkspace(
        const ::faiss::IndexIVFFastScan* index_in,
        const float* query_data,
        const IVFSearchParameters* params)
        : IVFIteratorWorkspace(query_data, index_in->d, params),
          impl_(std::make_unique<Impl>()) {
    impl_->index = index_in;
    impl_->init(*this, index_in, params);
}

void IVFFastScanIteratorWorkspace::Impl::init(
        IVFFastScanIteratorWorkspace& ws,
        const ::faiss::IndexIVFFastScan* index_in,
        const IVFSearchParameters* /* params */) {
    auto coarse_list_sizes_buf = std::make_unique<size_t[]>(index_in->nlist);
    size_t count = 0;
    size_t max_coarse_list_size = 0;
    for (size_t list_no = 0; list_no < index_in->nlist; ++list_no) {
        auto list_size = index_in->invlists->list_size(list_no);
        coarse_list_sizes_buf[list_no] = list_size;
        count += list_size;
        if (list_size > max_coarse_list_size) {
            max_coarse_list_size = list_size;
        }
    }

    size_t np = ws.search_params->nprobe
            ? ws.search_params->nprobe
            : index_in->nprobe;
    np = std::min(index_in->nlist, np);
    ws.backup_count_threshold = count * np / index_in->nlist;
    auto max_backup_count =
            max_coarse_list_size + ws.backup_count_threshold;

    auto coarse_idx_buf = std::make_unique<idx_t[]>(index_in->nlist);
    auto coarse_dis_buf = std::make_unique<float[]>(index_in->nlist);
    index_in->quantizer->search(
            1,
            ws.query_data.data(),
            index_in->nlist,
            coarse_dis_buf.get(),
            coarse_idx_buf.get(),
            ws.search_params
                    ? ws.search_params->quantizer_params
                    : nullptr);

    ws.coarse_idx = std::move(coarse_idx_buf);
    ws.coarse_dis = std::move(coarse_dis_buf);
    ws.coarse_list_sizes = std::move(coarse_list_sizes_buf);
    ws.nprobe = np;
    ws.dists.reserve(max_backup_count);

    dim12 = index_in->ksub * index_in->M2;
    ::faiss::IndexIVFFastScan::CoarseQuantized cq{
            ws.nprobe,
            ws.coarse_dis.get(),
            ws.coarse_idx.get()};
    faiss::FastScanDistancePostProcessing empty_context{};
    index_in->compute_LUT_uint8(
            1,
            ws.query_data.data(),
            cq,
            dis_tables,
            biases,
            normalizers,
            empty_context);
}

IVFFastScanIteratorWorkspace::~IVFFastScanIteratorWorkspace() = default;

void IVFFastScanIteratorWorkspace::next_batch(size_t current_backup_count) {
    impl_->next_batch(*this, current_backup_count);
}

void IVFFastScanIteratorWorkspace::Impl::next_batch(
        IVFFastScanIteratorWorkspace& ws,
        size_t current_backup_count) {
    ws.dists.clear();

    // Baseline SingleQueryResultCollectHandler uses std::pair<TI, T>;
    // we convert to knowhere::DistId after scanning.
    std::vector<std::pair<int64_t, float>> pairs;

    std::unique_ptr<SIMDResultHandlerToFloat> handler;
    bool is_max = !faiss::is_similarity_metric(index->metric_type);
    auto id_selector = ws.search_params->sel
            ? ws.search_params->sel
            : nullptr;

    if (is_max) {
        handler.reset(
                new SingleQueryResultCollectHandler<
                        CMax<uint16_t, int64_t>,
                        true>(pairs, index->ntotal, id_selector));
    } else {
        handler.reset(
                new SingleQueryResultCollectHandler<
                        CMin<uint16_t, int64_t>,
                        true>(pairs, index->ntotal, id_selector));
    }

    this->get_interator_next_batch_implem_10(
            index, ws, *handler.get(), current_backup_count);

    // Convert std::pair to knowhere::DistId
    ws.dists.reserve(pairs.size());
    for (auto& [id, dis] : pairs) {
        ws.dists.emplace_back(id, dis);
    }
}

void IVFFastScanIteratorWorkspace::Impl::get_interator_next_batch_implem_10(
        const ::faiss::IndexIVFFastScan* index,
        IVFFastScanIteratorWorkspace& ws,
        SIMDResultHandlerToFloat& handler,
        size_t current_backup_count) {
    bool single_LUT = !index->lookup_table_is_3d();
    handler.begin(index->skip & 16 ? nullptr : this->normalizers);
    auto dim12_local = this->dim12;
    const size_t block_stride = index->get_block_stride();
    const uint8_t* LUT = nullptr;

    if (single_LUT) {
        LUT = this->dis_tables.get();
    }
    while (current_backup_count + handler.in_range_num <
                   ws.backup_count_threshold &&
           ws.next_visit_coarse_list_idx < index->nlist) {
        auto next_list_idx = ws.next_visit_coarse_list_idx;
        ws.next_visit_coarse_list_idx++;
        if (!single_LUT) {
            LUT = this->dis_tables.get() + next_list_idx * dim12_local;
        }
        index->invlists->prefetch_lists(
                ws.coarse_idx.get() + next_list_idx, 1);
        if (this->biases.get()) {
            handler.dbias = this->biases.get() + next_list_idx;
        }
        idx_t list_no = ws.coarse_idx[next_list_idx];
        if (list_no < 0) {
            continue;
        }
        size_t ls = index->invlists->list_size(list_no);
        if (ls == 0) {
            continue;
        }

        InvertedLists::ScopedCodes codes(index->invlists, list_no);
        InvertedLists::ScopedIds ids(index->invlists, list_no);
        handler.ntotal = ls;
        handler.id_map = ids.get();
        with_SIMDResultHandler(handler, [&](auto& concrete_handler) {
            ::faiss::DummyScaler<> dummy;
            ::faiss::pq4_accumulate_loop_fixed_scaler(
                    1,
                    roundup(ls, index->bbs),
                    index->bbs,
                    index->M2,
                    codes.get(),
                    LUT,
                    concrete_handler,
                    dummy,
                    block_stride);
        });
    }
    handler.end();
}

// ---- ScaNNIteratorWorkspace ----

ScaNNIteratorWorkspace::ScaNNIteratorWorkspace(
        const IndexScaNN* scann_index,
        const float* query_data,
        const IVFSearchParameters* params)
        : inner() {
    if (auto* fast_scan_base = dynamic_cast<const ::faiss::IndexIVFPQFastScan*>(
                scann_index->base_index)) {
        inner = std::make_unique<IVFFastScanIteratorWorkspace>(
                fast_scan_base, query_data, params);
    } else {
        FAISS_THROW_MSG("IndexScaNN base index must be IndexIVFPQFastScan");
    }

    // Set up dis_refine (logic from IndexScaNN::getIteratorWorkspace)
    if (scann_index->refine_index) {
        // refine_index may be a baseline ::faiss::IndexFlat{,IP,L2} or the
        // knowhere Jaccard-aware subclass; cast to the common baseline type.
        auto refine =
                dynamic_cast<const ::faiss::IndexFlat*>(scann_index->refine_index);
        FAISS_THROW_IF_NOT(refine);
        if (auto norms = dynamic_cast<const HasInverseL2Norms*>(
                    scann_index->base_index)) {
            const float* inverse_l2_norms = norms->get_inverse_l2_norms();
            FAISS_THROW_IF_NOT(inverse_l2_norms);
            this->dis_refine = std::unique_ptr<faiss::DistanceComputer>(
                    new faiss::cppcontrib::knowhere::WithCosineNormDistanceComputer(
                            inverse_l2_norms,
                            scann_index->base_index->d,
                            std::unique_ptr<faiss::DistanceComputer>(
                                    refine->get_distance_computer())));
        } else {
            this->dis_refine = std::unique_ptr<faiss::DistanceComputer>(
                    refine->get_FlatCodesDistanceComputer());
        }
        this->dis_refine->set_query(query_data);
    } else {
        this->dis_refine = nullptr;
    }
    this->search_params = inner->search_params;
}

void ScaNNIteratorWorkspace::next_batch(size_t current_backup_count) {
    inner->next_batch(current_backup_count);
    this->dists = std::move(inner->dists);
}

}  // namespace faiss::cppcontrib::knowhere
