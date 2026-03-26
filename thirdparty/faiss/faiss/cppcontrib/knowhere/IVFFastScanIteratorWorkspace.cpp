/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/cppcontrib/knowhere/IVFFastScanIteratorWorkspace.h>

#include <cinttypes>

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
    const IndexIVFFastScan* index;
    size_t dim12;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    float normalizers[2];

    void next_batch(IVFFastScanIteratorWorkspace& ws, size_t current_backup_count);

    void get_interator_next_batch_implem_10(
            IVFFastScanIteratorWorkspace& ws,
            SIMDResultHandlerToFloat& handler,
            size_t current_backup_count);
};

// ---- IVFFastScanIteratorWorkspace ----

IVFFastScanIteratorWorkspace::IVFFastScanIteratorWorkspace(
        const IndexIVFFastScan* index_in,
        const float* query_data,
        const IVFSearchParameters* params)
        : IVFIteratorWorkspace(query_data, index_in->d, params),
          impl_(std::make_unique<Impl>()) {
    impl_->index = index_in;

    auto coarse_list_sizes_buf = std::make_unique<size_t[]>(index_in->nlist);
    size_t count = 0;
    auto max_coarse_list_size = 0;
    for (size_t list_no = 0; list_no < index_in->nlist; ++list_no) {
        auto list_size = index_in->invlists->list_size(list_no);
        coarse_list_sizes_buf[list_no] = list_size;
        count += list_size;
        if (list_size > max_coarse_list_size) {
            max_coarse_list_size = list_size;
        }
    }

    size_t np = this->search_params->nprobe
            ? this->search_params->nprobe
            : index_in->nprobe;
    np = std::min(index_in->nlist, np);
    this->backup_count_threshold = count * np / index_in->nlist;
    auto max_backup_count =
            max_coarse_list_size + this->backup_count_threshold;

    auto coarse_idx_buf = std::make_unique<idx_t[]>(index_in->nlist);
    auto coarse_dis_buf = std::make_unique<float[]>(index_in->nlist);
    index_in->quantizer->search(
            1,
            this->query_data.data(),
            index_in->nlist,
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

    impl_->dim12 = index_in->ksub * index_in->M2;
    IndexIVFFastScan::CoarseQuantized cq{
            this->nprobe,
            this->coarse_dis.get(),
            this->coarse_idx.get()};
    faiss::FastScanDistancePostProcessing empty_context{};
    index_in->compute_LUT_uint8(
            1,
            this->query_data.data(),
            cq,
            impl_->dis_tables,
            impl_->biases,
            impl_->normalizers,
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

    this->get_interator_next_batch_implem_10(ws, *handler.get(), current_backup_count);

    // Convert std::pair to knowhere::DistId
    ws.dists.reserve(pairs.size());
    for (auto& [id, dis] : pairs) {
        ws.dists.emplace_back(id, dis);
    }
}

void IVFFastScanIteratorWorkspace::Impl::get_interator_next_batch_implem_10(
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
        size_t ls = index->invlists->list_size(list_no);
        if (list_no < 0 || ls == 0)
            continue;

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
        : inner(std::make_unique<IVFFastScanIteratorWorkspace>(
                  dynamic_cast<const IndexIVFPQFastScan*>(scann_index->base_index),
                  query_data,
                  params)) {
    auto* fast_scan_base =
            dynamic_cast<const IndexIVFPQFastScan*>(scann_index->base_index);
    // Set up dis_refine (logic from IndexScaNN::getIteratorWorkspace)
    if (scann_index->refine_index) {
        auto refine = dynamic_cast<const IndexFlat*>(scann_index->refine_index);
        if (auto base_cosine =
                    dynamic_cast<const IndexIVFPQFastScanCosine*>(fast_scan_base)) {
            this->dis_refine = std::unique_ptr<faiss::DistanceComputer>(
                    new faiss::cppcontrib::knowhere::WithCosineNormDistanceComputer(
                            base_cosine->inverse_norms_storage.inverse_l2_norms.data(),
                            base_cosine->d,
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
