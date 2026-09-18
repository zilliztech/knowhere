// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "index/diskann/rabitq_store.h"

#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/VectorTransform.h>
#include <faiss/cppcontrib/knowhere/impl/RaBitQBuildUtils.h>
#include <faiss/cppcontrib/knowhere/index_io.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "diskann/utils.h"
#include "index/diskann/diskann_config.h"

namespace knowhere {
namespace {

constexpr size_t kBuildBlockBytes = 32UL * 1024 * 1024;

size_t
BlockRows(size_t dim) {
    if (dim == 0) {
        throw std::invalid_argument("RaBitQ sidecar dimension must be positive");
    }
    return std::max<size_t>(1, kBuildBlockBytes / (dim * sizeof(float)));
}

template <typename Fn>
void
ForEachFloatBinBlock(const std::string& data_path, size_t rows, size_t dim, Fn&& fn) {
    std::ifstream input(data_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open RaBitQ source data: " + data_path);
    }

    input.seekg(2 * sizeof(uint32_t), std::ios::beg);
    const size_t block_rows = BlockRows(dim);
    std::vector<float> block(block_rows * dim);
    size_t row_offset = 0;
    while (row_offset < rows) {
        const size_t current_rows = std::min(block_rows, rows - row_offset);
        const size_t current_values = current_rows * dim;
        input.read(reinterpret_cast<char*>(block.data()), current_values * sizeof(float));
        if (!input) {
            throw std::runtime_error("short read while building RaBitQ sidecar from: " + data_path);
        }
        fn(block.data(), current_rows);
        row_offset += current_rows;
    }
}

// Single-query rotation application.
//
// RandomRotationMatrix::apply_noalloc() routes through OpenBLAS sgemm_, which
// for a skinny n=1 GEMV spawns an internal worker pool whose launch/packing
// overhead dominates the actual multiply (observed as a serial-p50 regression
// when the search worker thread already saturates the CPU). Since DiskANN
// searches one query at a time, apply the rotation with a plain single-threaded
// GEMV instead.
//
// The stored matrix A is d_in x d_out in column-major order (d_in == d_out == d
// here), and apply_noalloc computes xt = A^T * x, i.e.
//   xt[j] = sum_k A[k + j * d] * x[k].
void
apply_rotation_single_query(const faiss::RandomRotationMatrix* rotation, const float* x, float* xt) {
    const int d = rotation->d_in;
    const float* a = rotation->A.data();
    for (int j = 0; j < d; ++j) {
        const float* col = a + j * d;
        float acc = rotation->have_bias ? rotation->b[j] : 0.0f;
#pragma omp simd reduction(+ : acc)
        for (int k = 0; k < d; ++k) {
            acc += col[k] * x[k];
        }
        xt[j] = acc;
    }
}

class RaBitQNavigationDistanceComputer final : public diskann::NavigationDistanceComputer {
 public:
    RaBitQNavigationDistanceComputer(const faiss::RandomRotationMatrix* rotation, const faiss::IndexRaBitQ* rabitq,
                                     bool probabilistic_refinement, uint8_t query_bits)
        : rotation_(rotation),
          rabitq_(rabitq),
          probabilistic_refinement_(probabilistic_refinement),
          distance_computer_(rabitq->get_quantized_distance_computer(query_bits, false)),
          rabitq_distance_computer_(dynamic_cast<faiss::RaBitQDistanceComputer*>(distance_computer_.get())) {
        if (rabitq_distance_computer_ == nullptr) {
            throw std::runtime_error("RaBitQ sidecar returned an incompatible distance computer");
        }
    }

    void
    set_query(const float* query) override {
        const int d = rotation_->d_in;
        transformed_query_ = std::make_unique<float[]>(d);
        apply_rotation_single_query(rotation_, query, transformed_query_.get());
        distance_computer_->set_query(transformed_query_.get());
    }

    void
    compute_distances(const unsigned* ids, _u64 n_ids, float* distances, float threshold, bool threshold_valid,
                      diskann::QueryStats* stats) override {
        const bool can_prune = probabilistic_refinement_ && threshold_valid && rabitq_->rabitq.nb_bits > 1;
        if (!can_prune) {
            _u64 i = 0;
            for (; i + 4 <= n_ids; i += 4) {
                distance_computer_->distances_batch_4(ids[i], ids[i + 1], ids[i + 2], ids[i + 3], distances[i],
                                                      distances[i + 1], distances[i + 2], distances[i + 3]);
            }
            for (; i < n_ids; ++i) {
                distances[i] = (*distance_computer_)(ids[i]);
            }
            if (stats != nullptr && rabitq_->rabitq.nb_bits > 1) {
                stats->n_approx_refinements += n_ids;
            }
            return;
        }

        std::array<faiss::idx_t, 4> refine_ids{};
        std::array<_u64, 4> refine_positions{};
        size_t pending_refinements = 0;
        const auto flush_refinements = [&]() {
            distance_computer_->distances_batch_4(refine_ids[0], refine_ids[1], refine_ids[2], refine_ids[3],
                                                  distances[refine_positions[0]], distances[refine_positions[1]],
                                                  distances[refine_positions[2]], distances[refine_positions[3]]);
            pending_refinements = 0;
        };

        const auto process_estimate = [&](_u64 i, const uint8_t* code, float estimate) {
            if (stats != nullptr) {
                ++stats->n_approx_estimates;
            }
            if (!rabitq_distance_computer_->should_refine(code, estimate, threshold, false)) {
                distances[i] = std::numeric_limits<float>::infinity();
                if (stats != nullptr) {
                    ++stats->n_approx_pruned;
                    ++stats->n_cmps_saved;
                }
                return;
            }
            refine_ids[pending_refinements] = ids[i];
            refine_positions[pending_refinements] = i;
            ++pending_refinements;
            if (stats != nullptr) {
                ++stats->n_approx_refinements;
            }
            if (pending_refinements == 4) {
                flush_refinements();
            }
        };

        // Batch independent estimates, then retain the original neighbor order
        // and the caller's threshold snapshot when deciding which codes to refine.
        _u64 i = 0;
        for (; i + 4 <= n_ids; i += 4) {
            std::array<const uint8_t*, 4> codes{};
            std::array<float, 4> estimates{};
            for (size_t j = 0; j < 4; ++j) {
                codes[j] = rabitq_->codes.data() + static_cast<size_t>(ids[i + j]) * rabitq_->code_size;
            }
            rabitq_distance_computer_->distance_to_code_1bit_batch_4(codes.data(), estimates.data());
            for (size_t j = 0; j < 4; ++j) {
                process_estimate(i + j, codes[j], estimates[j]);
            }
        }
        for (; i < n_ids; ++i) {
            const uint8_t* code = rabitq_->codes.data() + static_cast<size_t>(ids[i]) * rabitq_->code_size;
            process_estimate(i, code, rabitq_distance_computer_->distance_to_code_1bit(code));
        }
        for (size_t i = 0; i < pending_refinements; ++i) {
            const auto id = refine_ids[i];
            const uint8_t* code = rabitq_->codes.data() + static_cast<size_t>(id) * rabitq_->code_size;
            distances[refine_positions[i]] = rabitq_distance_computer_->distance_to_code_full(code);
        }
    }

 private:
    const faiss::RandomRotationMatrix* rotation_;
    const faiss::IndexRaBitQ* rabitq_;
    const bool probabilistic_refinement_;
    std::unique_ptr<faiss::FlatCodesDistanceComputer> distance_computer_;
    faiss::RaBitQDistanceComputer* rabitq_distance_computer_;
    std::unique_ptr<float[]> transformed_query_;
};

}  // namespace

std::string
RaBitQStore::SidecarFilename(const std::string& index_prefix) {
    return index_prefix + "_rabitq.index";
}

void
RaBitQStore::BuildFromFloatBin(const std::string& data_path, const std::string& sidecar_path, uint8_t rbq_bits) {
    if (rbq_bits < 1 || rbq_bits > 9) {
        throw std::invalid_argument("RaBitQ database bits must be in [1, 9]");
    }

    size_t rows = 0;
    size_t dim = 0;
    diskann::get_bin_metadata(data_path, rows, dim);
    if (rows == 0 || dim == 0 || dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid RaBitQ source metadata");
    }

    constexpr size_t header_size = 2 * sizeof(uint32_t);
    if (dim > (std::numeric_limits<size_t>::max() - header_size) / sizeof(float) / rows) {
        throw std::invalid_argument("RaBitQ source metadata overflows file size");
    }
    const auto expected_size = header_size + rows * dim * sizeof(float);
    if (std::filesystem::file_size(data_path) != expected_size) {
        throw std::runtime_error("RaBitQ source file size does not match float32 metadata");
    }

    auto rotation = std::make_unique<faiss::RandomRotationMatrix>(static_cast<int>(dim), static_cast<int>(dim));
    rotation->init(12345);

    std::vector<double> sums(dim, 0.0);
    ForEachFloatBinBlock(data_path, rows, dim, [&](const float* block, size_t block_rows) {
        for (size_t i = 0; i < block_rows; ++i) {
            const float* row = block + i * dim;
            for (size_t j = 0; j < dim; ++j) {
                sums[j] += row[j];
            }
        }
    });

    std::vector<float> mean(dim);
    for (size_t j = 0; j < dim; ++j) {
        mean[j] = static_cast<float>(sums[j] / static_cast<double>(rows));
    }
    std::vector<float> rotated_center(dim);
    rotation->apply_noalloc(1, mean.data(), rotated_center.data());

    auto rabitq = std::make_unique<faiss::IndexRaBitQ>(static_cast<faiss::idx_t>(dim), faiss::METRIC_L2, rbq_bits);
    rabitq->center = std::move(rotated_center);
    rabitq->qb = 4;
    rabitq->centered = false;
    rabitq->is_trained = true;
    auto pretransform = std::make_unique<faiss::IndexPreTransform>(rotation.get(), rabitq.get());
    pretransform->own_fields = true;
    rotation.release();
    rabitq.release();

    ForEachFloatBinBlock(data_path, rows, dim, [&](const float* block, size_t block_rows) {
        // Preserve DiskANN's existing input-block boundaries while sharing
        // the same bounded storage population utility as HNSW.
        faiss::cppcontrib::knowhere::rabitq_build::add_in_blocks(*pretransform, static_cast<faiss::idx_t>(block_rows),
                                                                 block, static_cast<faiss::idx_t>(BlockRows(dim)));
    });
    if (pretransform->ntotal != static_cast<faiss::idx_t>(rows)) {
        throw std::runtime_error("RaBitQ sidecar point count mismatch after encoding");
    }

    const std::string temporary_path = sidecar_path + ".tmp";
    std::error_code error;
    std::filesystem::remove(temporary_path, error);
    try {
        faiss::cppcontrib::knowhere::write_index(pretransform.get(), temporary_path.c_str());
        std::filesystem::rename(temporary_path, sidecar_path);
    } catch (...) {
        std::filesystem::remove(temporary_path, error);
        throw;
    }
}

RaBitQStore::RaBitQStore(const std::string& sidecar_path)
    : index_(faiss::cppcontrib::knowhere::read_index(sidecar_path.c_str())) {
    Validate();
}

RaBitQStore::~RaBitQStore() = default;

void
RaBitQStore::Validate() {
    pretransform_ = dynamic_cast<const faiss::IndexPreTransform*>(index_.get());
    if (pretransform_ == nullptr || pretransform_->chain.size() != 1) {
        throw std::runtime_error("DiskANN RaBitQ sidecar must be an IndexPreTransform with one transform");
    }
    rotation_ = dynamic_cast<const faiss::RandomRotationMatrix*>(pretransform_->chain[0]);
    if (rotation_ == nullptr || !rotation_->is_trained || rotation_->d_in <= 0 || rotation_->d_in != rotation_->d_out) {
        throw std::runtime_error("DiskANN RaBitQ sidecar has an invalid random rotation");
    }
    const auto rotation_dim = static_cast<size_t>(rotation_->d_in);
    if (rotation_->A.size() != rotation_dim * rotation_dim ||
        (rotation_->have_bias ? rotation_->b.size() != rotation_dim : !rotation_->b.empty())) {
        throw std::runtime_error("DiskANN RaBitQ sidecar rotation storage is inconsistent");
    }
    rabitq_ = dynamic_cast<const faiss::IndexRaBitQ*>(pretransform_->index);
    if (rabitq_ == nullptr || !pretransform_->is_trained || !rabitq_->is_trained) {
        throw std::runtime_error("DiskANN RaBitQ sidecar has an invalid RaBitQ leaf");
    }
    if (pretransform_->metric_type != faiss::METRIC_L2 || rabitq_->metric_type != faiss::METRIC_L2 ||
        rabitq_->rabitq.metric_type != faiss::METRIC_L2 || pretransform_->d != rotation_->d_in ||
        rabitq_->d != rotation_->d_out || pretransform_->ntotal != rabitq_->ntotal) {
        throw std::runtime_error("DiskANN RaBitQ sidecar metadata is inconsistent");
    }
    if (rabitq_->ntotal < 0 || rabitq_->rabitq.nb_bits < 1 || rabitq_->rabitq.nb_bits > 9 || rabitq_->qb > 8 ||
        rabitq_->centered) {
        throw std::runtime_error("DiskANN RaBitQ sidecar quantizer metadata is inconsistent");
    }
    const auto expected_code_size =
        rabitq_->rabitq.compute_code_size(static_cast<size_t>(rabitq_->d), rabitq_->rabitq.nb_bits);
    const auto point_count = static_cast<size_t>(rabitq_->ntotal);
    if (rabitq_->code_size != expected_code_size || rabitq_->rabitq.code_size != expected_code_size ||
        expected_code_size == 0 || point_count > std::numeric_limits<size_t>::max() / expected_code_size ||
        rabitq_->center.size() != static_cast<size_t>(rabitq_->d) ||
        rabitq_->codes.size() != point_count * expected_code_size) {
        throw std::runtime_error("DiskANN RaBitQ sidecar code storage is inconsistent");
    }
}

std::unique_ptr<diskann::NavigationDistanceComputer>
RaBitQStore::CreateDistanceComputer(const DiskANNConfig& config) const {
    const auto* navigation = dynamic_cast<const DiskANNNavigationConfig*>(&config);
    if (!navigation) {
        throw std::invalid_argument("RaBitQ navigation requires query configuration");
    }
    const auto query_metric = config.metric_type.value_or(metric::L2);
    if (query_metric != metric::L2 && query_metric != metric::IP) {
        throw std::invalid_argument("RaBitQ navigation currently supports L2 and IP");
    }
    const auto mode = navigation->rbq_refine_mode.value_or("probabilistic");
    if (mode != "probabilistic" && mode != "full") {
        throw std::invalid_argument("invalid RaBitQ refinement mode");
    }
    const auto qb = navigation->rbq_bits_query.value_or(4);
    if (qb < 0 || qb > 8) {
        throw std::invalid_argument("RaBitQ query bits must be in [0, 8]");
    }
    return CreateDistanceComputer(mode == "probabilistic", static_cast<uint8_t>(qb));
}

std::unique_ptr<diskann::NavigationDistanceComputer>
RaBitQStore::CreateDistanceComputer(bool probabilistic_refinement, uint8_t query_bits) const {
    if (query_bits > 8) {
        throw std::invalid_argument("RaBitQ query bits must be in [0, 8]");
    }
    return std::make_unique<RaBitQNavigationDistanceComputer>(rotation_, rabitq_, probabilistic_refinement, query_bits);
}

int64_t
RaBitQStore::Count() const {
    return pretransform_->ntotal;
}

int64_t
RaBitQStore::Dimension() const {
    return pretransform_->d;
}

uint8_t
RaBitQStore::Bits() const {
    return static_cast<uint8_t>(rabitq_->rabitq.nb_bits);
}

size_t
RaBitQStore::CodeSize() const {
    return rabitq_->code_size;
}

size_t
RaBitQStore::MemorySize() const {
    return rabitq_->codes.size() * sizeof(uint8_t) + rabitq_->center.size() * sizeof(float) +
           rotation_->A.size() * sizeof(float) + rotation_->b.size() * sizeof(float);
}

uint64_t
RaBitQStore::EstimateMemorySize(int64_t rows, int64_t prepared_dim, uint8_t rbq_bits) {
    if (rows <= 0 || prepared_dim <= 0 || prepared_dim > std::numeric_limits<int>::max() || rbq_bits < 1 ||
        rbq_bits > 9) {
        throw std::invalid_argument("invalid RaBitQ resource estimate dimensions or bits");
    }
    const uint64_t d = static_cast<uint64_t>(prepared_dim);
    const uint64_t code_size = faiss::RaBitQuantizer().compute_code_size(d, rbq_bits);
    // d is bounded by the transform's int dimension; d*(d+1)*sizeof(float)
    // fits uint64_t. Includes the square rotation and the global centroid.
    const uint64_t model_bytes = d * (d + 1) * sizeof(float);
    if (static_cast<uint64_t>(rows) > (std::numeric_limits<uint64_t>::max() - model_bytes) / code_size) {
        throw std::overflow_error("RaBitQ resource estimate overflows uint64_t");
    }
    return static_cast<uint64_t>(rows) * code_size + model_bytes;
}

}  // namespace knowhere
