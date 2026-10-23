// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "diskann/rabitq_navigation.h"

#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/VectorTransform.h>
#include <faiss/cppcontrib/knowhere/index_io.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "diskann/utils.h"

namespace diskann {
  namespace {

    constexpr size_t kBuildBlockBytes = 32UL * 1024 * 1024;

    size_t BlockRows(size_t dim) {
      return std::max<size_t>(1, kBuildBlockBytes / (dim * sizeof(float)));
    }

    template<typename Fn>
    void ForEachFloatBinBlock(const std::string& data_path, size_t rows,
                              size_t dim, Fn&& fn) {
      std::ifstream input(data_path, std::ios::binary);
      if (!input) {
        throw std::runtime_error("failed to open RaBitQ source data: " +
                                 data_path);
      }

      input.seekg(2 * sizeof(uint32_t), std::ios::beg);
      const size_t       block_rows = BlockRows(dim);
      std::vector<float> block(block_rows * dim);
      size_t             row_offset = 0;
      while (row_offset < rows) {
        const size_t current_rows = std::min(block_rows, rows - row_offset);
        const size_t current_values = current_rows * dim;
        input.read(reinterpret_cast<char*>(block.data()),
                   current_values * sizeof(float));
        if (!input) {
          throw std::runtime_error(
              "short read while building RaBitQ sidecar from: " + data_path);
        }
        fn(block.data(), current_rows);
        row_offset += current_rows;
      }
    }

    class RaBitQNavigationDistanceComputer final
        : public NavigationDistanceComputer {
     public:
      RaBitQNavigationDistanceComputer(
          std::shared_ptr<faiss::Index>      model,
          const faiss::RandomRotationMatrix& rotation,
          const faiss::IndexRaBitQ&          rabitq)
          : model_(std::move(model)), rotation_(rotation), rabitq_(rabitq),
            transformed_query_(rotation.d_out) {
      }
      void ResetQueryParameters() override {
        query_bits_ = 4;
      }
      void Configure(uint8_t bits) {
        if (bits > 8)
          throw std::invalid_argument("RaBitQ query bits must be in [0, 8]");
        query_bits_ = bits;
      }
      void set_query(const float* query) override {
        if (!computer_ || active_bits_ != query_bits_) {
          computer_.reset(
              rabitq_.get_quantized_distance_computer(query_bits_, false));
          active_bits_ = query_bits_;
        }
        rotation_.apply_noalloc(1, query, transformed_query_.data());
        computer_->set_query(transformed_query_.data());
      }
      void compute_distances(const unsigned* ids, uint64_t count,
                             float* distances) override {
        uint64_t i = 0;
        for (; i + 4 <= count; i += 4)
          computer_->distances_batch_4(
              ids[i], ids[i + 1], ids[i + 2], ids[i + 3], distances[i],
              distances[i + 1], distances[i + 2], distances[i + 3]);
        for (; i < count; ++i)
          distances[i] = (*computer_)(ids[i]);
      }

     private:
      std::shared_ptr<faiss::Index>                     model_;
      const faiss::RandomRotationMatrix&                rotation_;
      const faiss::IndexRaBitQ&                         rabitq_;
      std::vector<float>                                transformed_query_;
      std::unique_ptr<faiss::FlatCodesDistanceComputer> computer_;
      uint8_t                        query_bits_ = 4, active_bits_ = 255;
    };
  }  // namespace

  RaBitQNavigationStore::RaBitQNavigationStore(
      RaBitQNavigationBuildOptions options)
      : options_(options) {
  }
  RaBitQNavigationStore::~RaBitQNavigationStore() = default;
  int RaBitQNavigationStore::ValidateBuild() const {
    if (options_.bits < 1 || options_.bits > 9)
      throw std::invalid_argument("RaBitQ database bits must be in [1, 9]");
    return 0;
  }
  int RaBitQNavigationStore::Build(const NavigationBuildContext& context) {
    ValidateBuild();
    if (!IsCompatible(context.data_type, false))
      throw std::invalid_argument("RaBitQ navigation requires FP32 data");
    size_t rows, dim;
    get_bin_metadata(context.data_path, rows, dim);
    if (rows != context.count || dim != context.dimension)
      throw std::invalid_argument(
          "RaBitQ source metadata does not match build context");
    const auto& data_path = context.data_path;
    if (rows == 0 || dim == 0 ||
        dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("invalid RaBitQ source metadata");
    }

    constexpr size_t header_size = 2 * sizeof(uint32_t);
    if (dim > (std::numeric_limits<size_t>::max() - header_size) /
                  sizeof(float) / rows) {
      throw std::invalid_argument("RaBitQ source metadata overflows file size");
    }
    const auto expected_size = header_size + rows * dim * sizeof(float);
    if (std::filesystem::file_size(data_path) != expected_size) {
      throw std::runtime_error(
          "RaBitQ source file size does not match float32 metadata");
    }

    auto rotation = std::make_unique<faiss::RandomRotationMatrix>(
        static_cast<int>(dim), static_cast<int>(dim));
    rotation->init(12345);

    std::vector<double> sums(dim, 0.0);
    ForEachFloatBinBlock(data_path, rows, dim,
                         [&](const float* block, size_t block_rows) {
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

    auto rabitq = std::make_unique<faiss::IndexRaBitQ>(
        static_cast<faiss::idx_t>(dim), faiss::METRIC_L2, options_.bits);
    rabitq->center = std::move(rotated_center);
    rabitq->qb = 4;
    rabitq->centered = false;
    rabitq->is_trained = true;
    auto pretransform = std::make_unique<faiss::IndexPreTransform>(
        rotation.get(), rabitq.get());
    pretransform->own_fields = true;
    rotation.release();
    rabitq.release();

    ForEachFloatBinBlock(
        data_path, rows, dim, [&](const float* block, size_t block_rows) {
          pretransform->add(static_cast<faiss::idx_t>(block_rows), block);
        });
    if (pretransform->ntotal != static_cast<faiss::idx_t>(rows)) {
      throw std::runtime_error(
          "RaBitQ sidecar point count mismatch after encoding");
    }

    faiss::cppcontrib::knowhere::write_index(
        pretransform.get(), SidecarFilename(context.prefix).c_str());
    return 0;
  }
  int RaBitQNavigationStore::Load(const std::string& prefix) {
    index_.reset(faiss::cppcontrib::knowhere::read_index(
        SidecarFilename(prefix).c_str()));
    Validate();
    return 0;
  }
  void RaBitQNavigationStore::GetNecessaryFilenames(
      const std::string& prefix, std::vector<std::string>& files,
      NavigationFilePurpose) const {
    files.push_back(SidecarFilename(prefix));
  }
  std::optional<size_t> RaBitQNavigationStore::BuildCacheMemorySize(
      const std::string& prefix) const {
    const auto file = SidecarFilename(prefix);
    if (!file_exists(file))
      return std::nullopt;
    return get_file_size(file);
  }
  std::unique_ptr<NavigationDistanceComputer>
  RaBitQNavigationStore::CreateDistanceComputer(uint64_t, uint64_t) const {
    if (!index_)
      throw std::logic_error("RaBitQ navigation is not loaded");
    return std::make_unique<RaBitQNavigationDistanceComputer>(
        index_, *rotation_, *rabitq_);
  }
  void RaBitQNavigationStore::ConfigureQuery(
      NavigationDistanceComputer& computer, uint8_t bits) {
    dynamic_cast<RaBitQNavigationDistanceComputer&>(computer).Configure(bits);
  }
  size_t RaBitQNavigationStore::QueryMemorySize(uint64_t) const {
    // Upper bound for qb=0..8: rotated input and Faiss rotated_q, plus padded
    // scalar-quantized query and eight bit planes. Faiss owns these scratch
    // arrays.
    return sizeof(RaBitQNavigationDistanceComputer) +
           sizeof(faiss::RaBitQDistanceComputer) +
           3 * sizeof(std::vector<float>) +
           sizeof(faiss::rabitq_utils::QueryFactorsData) + 32 +
           8 * Dimension() + 2 * ROUND_UP(Dimension(), 8);
  }
  std::string RaBitQNavigationStore::SidecarFilename(
      const std::string& index_prefix) {
    return index_prefix + "_rabitq.index";
  }

  void RaBitQNavigationStore::Validate() {
    pretransform_ = dynamic_cast<const faiss::IndexPreTransform*>(index_.get());
    if (pretransform_ == nullptr || pretransform_->chain.size() != 1) {
      throw std::runtime_error(
          "DiskANN RaBitQ sidecar must be an IndexPreTransform with one "
          "transform");
    }
    rotation_ = dynamic_cast<const faiss::RandomRotationMatrix*>(
        pretransform_->chain[0]);
    if (rotation_ == nullptr || !rotation_->is_trained ||
        rotation_->d_in <= 0 || rotation_->d_in != rotation_->d_out ||
        !rotation_->is_orthonormal || rotation_->have_bias ||
        !rotation_->b.empty()) {
      throw std::runtime_error(
          "DiskANN RaBitQ sidecar has an invalid random rotation");
    }
    const auto rotation_dim = static_cast<size_t>(rotation_->d_in);
    if (rotation_->A.size() != rotation_dim * rotation_dim) {
      throw std::runtime_error(
          "DiskANN RaBitQ sidecar rotation storage is inconsistent");
    }
    rabitq_ = dynamic_cast<const faiss::IndexRaBitQ*>(pretransform_->index);
    if (rabitq_ == nullptr || !pretransform_->is_trained ||
        !rabitq_->is_trained) {
      throw std::runtime_error(
          "DiskANN RaBitQ sidecar has an invalid RaBitQ leaf");
    }
    if (pretransform_->metric_type != faiss::METRIC_L2 ||
        rabitq_->metric_type != faiss::METRIC_L2 ||
        pretransform_->d != rotation_->d_in || rabitq_->d != rotation_->d_out ||
        pretransform_->ntotal != rabitq_->ntotal) {
      throw std::runtime_error(
          "DiskANN RaBitQ sidecar metadata is inconsistent");
    }
    // The reader validates the RaBitQ leaf. Retain the nonempty model and
    // size-overflow constraints required by this navigation consumer.
    if (rabitq_->ntotal <= 0 ||
        static_cast<uint64_t>(rabitq_->ntotal) >
            std::numeric_limits<size_t>::max() / rabitq_->code_size) {
      throw std::runtime_error(
          "DiskANN RaBitQ sidecar has an invalid point count");
    }
  }

  uint64_t RaBitQNavigationStore::Count() const {
    return pretransform_->ntotal;
  }

  uint64_t RaBitQNavigationStore::Dimension() const {
    return pretransform_->d;
  }

  size_t RaBitQNavigationStore::MemorySize() const {
    return sizeof(*this) + sizeof(faiss::IndexPreTransform) +
           sizeof(faiss::RandomRotationMatrix) + sizeof(faiss::IndexRaBitQ) +
           rabitq_->codes.owned_data.capacity() * sizeof(uint8_t) +
           rabitq_->center.capacity() * sizeof(float) +
           rotation_->A.capacity() * sizeof(float) +
           rotation_->b.capacity() * sizeof(float);
  }

}  // namespace diskann
