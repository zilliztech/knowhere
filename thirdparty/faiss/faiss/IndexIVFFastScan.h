/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <faiss/IndexIVF.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct NormTableScaler;
struct SIMDResultHandlerToFloat;

/** Fast scan version of IVFPQ and IVFAQ. Works for 4-bit PQ/AQ for now.
 *
 * The codes in the inverted lists are not stored sequentially but
 * grouped in blocks of size bbs. This makes it possible to very quickly
 * compute distances with SIMD instructions.
 *
 * Implementations (implem):
 * 0: auto-select implementation (default)
 * 1: orig's search, re-implemented
 * 2: orig's search, re-ordered by invlist
 * 10: optimizer int16 search, collect results in heap, no qbs
 * 11: idem, collect results in reservoir
 * 12: optimizer int16 search, collect results in heap, uses qbs
 * 13: idem, collect results in reservoir
 * 14: internally multithreaded implem over nq * nprobe
 * 15: same with reservoir
 *
 * For range search, only 10 and 12 are supported.
 * add 100 to the implem to force single-thread scanning (the coarse quantizer
 * may still use multiple threads).
 *
 * For search interator, only 10 are supported, one query, no qbs
 */

struct IVFFastScanIteratorWorkspace : IVFIteratorWorkspace {
    IVFFastScanIteratorWorkspace() = default;
    IVFFastScanIteratorWorkspace(
            const float* query_data,
            const size_t d,
            const IVFSearchParameters* search_params)
            : IVFIteratorWorkspace(query_data, d, search_params){};
    IVFFastScanIteratorWorkspace(
            std::unique_ptr<IVFIteratorWorkspace>&& base_workspace) {
        this->query_data = base_workspace->query_data;
        this->search_params = base_workspace->search_params;
        this->nprobe = base_workspace->nprobe;
        this->backup_count_threshold = base_workspace->backup_count_threshold;
        this->coarse_dis = std::move(base_workspace->coarse_dis);
        this->coarse_idx = std::move(base_workspace->coarse_idx);
        this->coarse_list_sizes = std::move(base_workspace->coarse_list_sizes);
        base_workspace = nullptr;
        return;
    }
    size_t dim12;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    float normalizers[2];
};

struct IndexIVFFastScan : IndexIVF {
    // size of the kernel
    int bbs; // set at build time

    size_t M;
    size_t nbits;
    size_t ksub;

    // M rounded up to a multiple of 2
    size_t M2;

    // search-time implementation
    int implem = 0;
    // skip some parts of the computation (for timing)
    int skip = 0;

    // batching factors at search time (0 = default)
    int qbs = 0;
    size_t qbs2 = 0;

    // // todo aguzhva: get rid of this
    std::vector<float> norms;

    IndexIVFFastScan(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t code_size,
            MetricType metric = METRIC_L2,
            bool is_cosine = false);

    IndexIVFFastScan();

    void init_fastscan(
            size_t M,
            size_t nbits,
            size_t nlist,
            MetricType metric,
            int bbs);

    // initialize the CodePacker in the InvertedLists
    void init_code_packer();

    ~IndexIVFFastScan() override;

    /// orig's inverted lists (for debugging)
    InvertedLists* orig_invlists = nullptr;

    // Knowhere-specific function, needed for norms, introduced in PR #1
    // final is needed because 'x' can be renormalized inside it,
    //   so a derived class is not allowed to override this function.
    void add_with_ids(idx_t n, const float* x, const idx_t* xids)
            override final;

    // This matches Faiss baseline.
    void add_with_ids_impl(idx_t n, const float* x, const idx_t* xids);

    // Knowhere-specific override.
    // final is needed because 'x' can be renormalized inside it,
    //   so a derived class is not allowed to override this function.
    void train(idx_t n, const float* x) override final;

    // prepare look-up tables

    virtual bool lookup_table_is_3d() const = 0;

    // compact way of conveying coarse quantization results
    struct CoarseQuantized {
        size_t nprobe;
        const float* dis = nullptr;
        const idx_t* ids = nullptr;
    };

    virtual void compute_LUT(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
            AlignedTable<float>& dis_tables,
            AlignedTable<float>& biases) const = 0;

    void compute_LUT_uint8(
            size_t n,
            const float* x,
            const CoarseQuantized& cq,
            AlignedTable<uint8_t>& dis_tables,
            AlignedTable<uint16_t>& biases,
            float* normalizers) const;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const idx_t* assign,
            const float* centroid_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* stats = nullptr) const override;

    std::unique_ptr<IVFIteratorWorkspace> getIteratorWorkspace(
            const float* query_data,
            const IVFSearchParameters* ivfsearchParams) const override;

    void getIteratorNextBatch(
            IVFIteratorWorkspace* workspace,
            size_t current_backup_count) const override;

    // range_search implementation was introduced in Knowhere,
    //   diff 73f03354568b4bf5a370df6f37e8d56dfc3a9c85
    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    // internal search funcs

    // dispatch to implementations and parallelize
    void search_dispatch_implem(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    void range_search_dispatch_implem(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult& rres,
            const CoarseQuantized& cq_in,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    // impl 1 and 2 are just for verification
    template <class C>
    void search_implem_1(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    template <class C>
    void search_implem_2(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    // implem 10 and 12 are not multithreaded internally, so
    // export search stats
    void search_implem_10(
            idx_t n,
            const float* x,
            idx_t k,
            SIMDResultHandlerToFloat& handler,
            const CoarseQuantized& cq,
            size_t* ndis_out,
            size_t* nlist_out,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    void range_search_implem_10(
            idx_t n,
            const float* x,
            SIMDResultHandlerToFloat& handler,
            const CoarseQuantized& cq,
            size_t* ndis_out,
            size_t* nlist_out,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    void search_implem_12(
            idx_t n,
            const float* x,
            SIMDResultHandlerToFloat& handler,
            const CoarseQuantized& cq,
            size_t* ndis_out,
            size_t* nlist_out,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    void range_search_implem_12(
            idx_t n,
            const float* x,
            SIMDResultHandlerToFloat& handler,
            const CoarseQuantized& cq,
            size_t* ndis_out,
            size_t* nlist_out,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    // one query call, no qbs
    void get_interator_next_batch_implem_10(
            SIMDResultHandlerToFloat& handler,
            IVFFastScanIteratorWorkspace* workspace,
            size_t current_backup_count) const;

    // implem 14 is multithreaded internally across nprobes and queries
    void search_implem_14(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const CoarseQuantized& cq,
            int impl,
            const NormTableScaler* scaler,
            const IVFSearchParameters* params = nullptr) const;

    // reconstruct vectors from packed invlists
    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    CodePacker* get_CodePacker() const override;

    // reconstruct orig invlists (for debugging)
    void reconstruct_orig_invlists();
};

// // todo aguzhva: removed in https://github.com/zilliztech/knowhere/pull/180,
// //   but commented out here
// struct IVFFastScanStats {
//     uint64_t times[10];
//     uint64_t t_compute_distance_tables, t_round;
//     uint64_t t_copy_pack, t_scan, t_to_flat;
//     uint64_t reservoir_times[4];
//     double t_aq_encode;
//     double t_aq_norm_encode;
//
//     double Mcy_at(int i) {
//         return times[i] / (1000 * 1000.0);
//     }
//
//     double Mcy_reservoir_at(int i) {
//         return reservoir_times[i] / (1000 * 1000.0);
//     }
//     IVFFastScanStats() {
//         reset();
//     }
//     void reset() {
//         memset(this, 0, sizeof(*this));
//     }
// };
//
// FAISS_API extern IVFFastScanStats IVFFastScan_stats;

} // namespace faiss
