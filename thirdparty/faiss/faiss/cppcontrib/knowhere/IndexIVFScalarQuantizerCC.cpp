#include <faiss/cppcontrib/knowhere/IndexIVFScalarQuantizerCC.h>

#include <omp.h>

#include <faiss/cppcontrib/knowhere/impl/cc_search.h>
#include <faiss/cppcontrib/knowhere/invlists/InvertedLists.h>
#include <faiss/cppcontrib/knowhere/IndexScalarQuantizer.h>



namespace faiss::cppcontrib::knowhere {

IndexIVFScalarQuantizerCC::IndexIVFScalarQuantizerCC(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t ssize,
        ::faiss::ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool by_residual,
        std::optional<std::string> raw_data_prefix_path)
        : IndexIVFScalarQuantizer(
                  quantizer,
                  d,
                  nlist,
                  qtype,
                  metric,
                  by_residual) {
    if (raw_data_prefix_path.has_value()) {
        raw_data_backup_ = std::make_unique<DataBackFileHandler>(
                raw_data_prefix_path.value(), d * sizeof(float));
    }
    replace_invlists(
            new ConcurrentArrayInvertedLists(nlist, code_size, ssize, false),
            true);
}

IndexIVFScalarQuantizerCC::IndexIVFScalarQuantizerCC() {
    this->by_residual = false;
}

// Path-D step 10.6: `train` and `add_with_ids` overrides deleted. The
// previous bodies just called up the inheritance chain without adding
// behavior. Removing them lets the inherited virtuals dispatch directly
// (train goes through IndexIVF::train; add_with_ids through baseline's
// inherited version which itself calls virtual add_core → our override
// below).

// Path-D step 10.9: CC add_core reimplements the SQ encode loop against
// cc_direct_map (ConcurrentDirectMapAdd) rather than inheriting fork's
// direct_map path. Structurally mirrors IndexIVFScalarQuantizer::add_core
// (its parent), with DirectMapAdd → ConcurrentDirectMapAdd and
// direct_map → cc_direct_map. The raw-data sidecar append (fork-only
// extension) runs sequentially after the OMP loop — see the same
// ordering rationale in step 10.6.
void IndexIVFScalarQuantizerCC::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    cc_direct_map.check_can_add(xids);

    std::unique_ptr<::faiss::ScalarQuantizer::SQuantizer> squant(
            sq.select_quantizer());
    ConcurrentDirectMapAdd dm_adder(cc_direct_map, n, xids);

#pragma omp parallel
    {
        std::vector<float> residual(d);
        std::vector<uint8_t> one_code(code_size);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

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
                        list_no,
                        id,
                        one_code.data(),
                        inverted_list_context);

                dm_adder.add(i, list_no, ofs);
            }
        }
    }

    if (raw_data_backup_ != nullptr) {
        // Raw-data sidecar is indexed by insertion order: block `id` of
        // the sidecar must contain the original (un-quantized) float
        // vector for entry `id`. Appending sequentially in `i` order
        // guarantees sidecar block N == x[N*d:(N+1)*d].
        for (idx_t i = 0; i < n; i++) {
            if (coarse_idx[i] >= 0) {
                raw_data_backup_->AppendDataBlock((char*)(x + i * d));
            }
        }
    }

    ntotal += n;
}

void IndexIVFScalarQuantizerCC::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    const auto* cil = static_cast<const ConcurrentArrayInvertedLists*>(
            this->invlists);
    cc_impl::search_preassigned(
            *this,
            *cil,
            n,
            x,
            k,
            assign,
            centroid_dis,
            distances,
            labels,
            store_pairs,
            params,
            stats);
}

void IndexIVFScalarQuantizerCC::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        faiss::RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    const auto* cil = static_cast<const ConcurrentArrayInvertedLists*>(
            this->invlists);
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
}

void IndexIVFScalarQuantizerCC::reconstruct(idx_t key, float* recons) const {
    FAISS_THROW_IF_NOT_MSG(
            raw_data_backup_ != nullptr,
            "IndexIVFScalarQuantizerCC can't get raw data if raw_data_backup_ not set.");
    raw_data_backup_->ReadDataBlock((char*)recons, key);
}

bool IndexIVFScalarQuantizerCC::with_raw_data() {
    return (raw_data_backup_ != nullptr);
}

void IndexIVFScalarQuantizerCC::reconstruct_n(idx_t i0, idx_t ni, float* recons)
        const {
    FAISS_THROW_MSG("IndexIVFScalarQuantizerCC not support reconstruct_n");
}

IndexIVFScalarQuantizerCCCosine::IndexIVFScalarQuantizerCCCosine(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t ssize,
        ::faiss::ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool by_residual,
        std::optional<std::string> raw_data_prefix_path)
        : IndexIVFScalarQuantizerCC(
                  quantizer, d, nlist, ssize, qtype, metric, by_residual, raw_data_prefix_path) {
}

IndexIVFScalarQuantizerCCCosine::IndexIVFScalarQuantizerCCCosine() {
    this->by_residual = false;
}

void IndexIVFScalarQuantizerCCCosine::train(idx_t n, const float* x) {
    auto x_normalized = ::knowhere::CopyAndNormalizeVecs(x, n, d);
    IndexIVF::train(n, x_normalized.get());
}

void IndexIVFScalarQuantizerCCCosine::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    // Cosine CC add path: encode NORMALIZED vectors (so the stored SQ
    // codes + inner-product search produce correct cosine distances),
    // but write ORIGINAL (un-normalized) vectors to the raw-data
    // sidecar so `GetVectorByIds` returns un-normalized data per the
    // knowhere contract.
    //
    // We bypass IndexIVFScalarQuantizerCC::add_core (our direct parent)
    // and delegate to the grandparent IndexIVFScalarQuantizer::add_core
    // with normalized data — that handles encoding + invlists insert +
    // direct_map bookkeeping. CC's sidecar append is skipped via the
    // bypass, and we append original data to the sidecar sequentially
    // here ourselves. Sequential append guarantees sidecar block `id`
    // matches insertion order (same ordering correctness as step 10.6).
    FAISS_THROW_IF_NOT(is_trained);
    cc_direct_map.check_can_add(xids);

    // Cosine CC add path: encode NORMALIZED vectors (so stored SQ
    // codes + inner-product search produce correct cosine distances),
    // but write ORIGINAL vectors to the raw-data sidecar. Same as
    // step 10.7 but now routed through cc_direct_map + the full
    // encode loop implemented here (no grandparent delegation).
    auto x_normalized = ::knowhere::CopyAndNormalizeVecs(x, n, d);
    const float* base_x = x_normalized.get();

    std::unique_ptr<::faiss::ScalarQuantizer::SQuantizer> squant(
            sq.select_quantizer());
    ConcurrentDirectMapAdd dm_adder(cc_direct_map, n, xids);

#pragma omp parallel
    {
        std::vector<float> residual(d);
        std::vector<uint8_t> one_code(code_size);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        for (size_t i = 0; i < n; i++) {
            int64_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                int64_t id = xids ? xids[i] : ntotal + i;

                const float* xi = base_x + i * d; // NORMALIZED for encoding
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }

                memset(one_code.data(), 0, code_size);
                squant->encode_vector(xi, one_code.data());

                size_t ofs = invlists->add_entry(
                        list_no,
                        id,
                        one_code.data(),
                        inverted_list_context);

                dm_adder.add(i, list_no, ofs);
            }
        }
    }

    // Sequential sidecar append with ORIGINAL (un-normalized) data.
    if (raw_data_backup_ != nullptr) {
        for (idx_t i = 0; i < n; i++) {
            if (coarse_idx[i] >= 0) {
                raw_data_backup_->AppendDataBlock((char*)(x + i * d));
            }
        }
    }

    ntotal += n;
}

void IndexIVFScalarQuantizerCCCosine::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    {
        auto x_normalized = ::knowhere::CopyAndNormalizeVecs(x, n, d);
        quantizer->assign(n, x_normalized.get(), coarse_idx.get());
    }
    add_core(n, x, xids, coarse_idx.get());
}

}


