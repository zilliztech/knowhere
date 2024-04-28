#pragma once
#include <faiss/IndexScalarQuantizer.h>
#include <cstdio>
#include <fstream>
#include <optional>
#include "knowhere/utils.h"
#include "utils/data_backup_file.h"
namespace faiss {

/***************************************************
 *IndexIVFScalarQuantizerCC
 ***************************************************/
struct IndexIVFScalarQuantizerCC : IndexIVFScalarQuantizer {
    std::unique_ptr<DataBackFileHandler> raw_data_backup_ = nullptr;

    IndexIVFScalarQuantizerCC(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t ssize,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool is_cosine = false,
            bool by_residual = false,
            std::optional<std::string> raw_data_prefix_path = std::nullopt);

    IndexIVFScalarQuantizerCC();

    void train(idx_t n, const float* x) override;

    void add_core(
            idx_t n,
            const float* x,
            const float* x_norms,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;

    void reconstruct(idx_t key, float* recons) const override;

    bool with_raw_data();

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;
};
} // namespace faiss