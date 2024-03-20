#include <faiss/IndexIVFScalarQuantizerCC.h>
#include <omp.h>

namespace faiss {
IndexIVFScalarQuantizerCC::IndexIVFScalarQuantizerCC(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t ssize,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool is_cosine,
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
    this->is_cosine = is_cosine;
    // add code into ConcurrentArrayInvertedLists, no need to normalize ig
    // metric == cosine
    replace_invlists(
            new ConcurrentArrayInvertedLists(nlist, code_size, ssize, false),
            true);
}

IndexIVFScalarQuantizerCC::IndexIVFScalarQuantizerCC() {
    this->by_residual = false;
}

void IndexIVFScalarQuantizerCC::train(idx_t n, const float* x) {
    if (is_cosine) {
        auto x_normalized = knowhere::CopyAndNormalizeVecs(x, n, d);
        IndexIVF::train(n, x_normalized.get());
    } else {
        IndexIVF::train(n, x);
    }
}

void IndexIVFScalarQuantizerCC::add_core(
        idx_t n,
        const float* x,
        const float* x_norms,
        const idx_t* xids,
        const idx_t* coarse_idx) {
    FAISS_THROW_IF_NOT(is_trained);
    std::unique_ptr<float[]> x_normalized = nullptr;
    const float* base_x = x;
    if (is_cosine) {
        x_normalized = knowhere::CopyAndNormalizeVecs(x, n, d);
        base_x = x_normalized.get();
    }

    size_t nadd = 0;
    std::unique_ptr<ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());

    DirectMapAdd dm_add(direct_map, n, xids);

#pragma omp parallel reduction(+ : nadd)
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

                const float* xi = base_x + i * d;
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }

                memset(one_code.data(), 0, code_size);
                squant->encode_vector(xi, one_code.data());

                size_t ofs = invlists->add_entry(list_no, id, one_code.data());

                dm_add.add(i, list_no, ofs);
                if (raw_data_backup_ != nullptr) {
                    raw_data_backup_->AppendDataBlock((char*)(x + i * d));
                }
                nadd++;

            } else if (rank == 0 && list_no == -1) {
                dm_add.add(i, -1, 0);
            }
        }
    }
    ntotal += n;
}

void IndexIVFScalarQuantizerCC::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    if (is_cosine) {
        std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
        {
            auto x_normalized = knowhere::CopyAndNormalizeVecs(x, n, d);
            quantizer->assign(n, x_normalized.get(), coarse_idx.get());
        }
        add_core(n, x, nullptr, xids, coarse_idx.get());
    } else {
        IndexIVFScalarQuantizer::add_with_ids(n, x, xids);
    }
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
} // namespace faiss