// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "diskann/distance.h"
#include "diskann/pq_table.h"
#include "diskann/utils.h"

namespace diskann {
  // Internal SSD scores are smaller-is-better. IP uses the prepared L2 domain
  // for raw and PQ payloads; COSINE uses negative similarity.
  inline float RestoreSSDScore(Metric metric, float distance, float query_norm,
                               float max_base_norm) {
    if (metric == Metric::INNER_PRODUCT) {
      distance = 1.0 - distance / 2.0;
      if (max_base_norm != 0)
        distance *= max_base_norm * query_norm;
    } else if (metric == Metric::COSINE) {
      distance = -distance;
    }
    return distance;
  }

  // Knowhere iterators apply their own similarity sign transform at the API
  // boundary.
  inline float SSDIteratorScore(Metric metric, float distance, float query_norm,
                                float max_base_norm) {
    if (metric == Metric::INNER_PRODUCT) {
      distance = distance / 2.0f - 1.0f;
      if (max_base_norm != 0)
        distance *= max_base_norm * query_norm;
    }
    return distance;
  }

  inline bool StorageHasRawData(bool compressed, Metric metric) {
    return !compressed && (metric == Metric::L2 || metric == Metric::COSINE);
  }

  // Storage encoding is independent of navigation encoding and graph layout.
  // Codecs own codebooks/normalization data; callers own IO and payload
  // buffers.
  template<typename T>
  class DiskStorageCodec {
   public:
    DiskStorageCodec(Metric metric, uint64_t dimension, uint64_t count,
                     const std::string& disk_file)
        : metric_(metric), dimension_(dimension),
          aligned_dimension_(ROUND_UP(dimension, 8)), count_(count),
          raw_distance_(get_distance_function<T>(
              metric == Metric::INNER_PRODUCT ? Metric::L2 : metric)),
          float_distance_(get_distance_function<float>(
              metric == Metric::INNER_PRODUCT ? Metric::L2 : metric)) {
      const auto norm_file = disk_file + "_max_base_norm.bin";
      if (file_exists(norm_file) &&
          (metric == Metric::INNER_PRODUCT || metric == Metric::COSINE)) {
        size_t                   rows, dims;
        std::unique_ptr<float[]> values;
        load_bin<float>(norm_file, values, rows, dims);
        if (dims != 1 || rows != (metric == Metric::INNER_PRODUCT ? 1 : count))
          throw std::runtime_error(
              "SSD normalization metadata does not match the stored space");
        if (metric == Metric::INNER_PRODUCT)
          max_base_norm_ = values[0];
        else
          base_norms_ = std::move(values);
      }
      if (metric == Metric::COSINE && !base_norms_)
        throw std::runtime_error("Missing SSD COSINE normalization data");
    }
    virtual ~DiskStorageCodec() = default;
    virtual bool   IsCompressed() const = 0;
    virtual size_t PayloadBytes() const = 0;
    bool           SupportsReorder() const {
      return IsCompressed();
    }
    virtual float  Score(const T* query, const float* query_float,
                         const void* payload, uint32_t id) const = 0;
    virtual void   Decode(const void* payload, float* output) const = 0;
    virtual size_t MemorySize() const = 0;
    bool           HasRawData() const {
      return StorageHasRawData(IsCompressed(), metric_);
    }
    float MaxBaseNorm() const {
      return max_base_norm_;
    }
    float Restore(float distance, float query_norm) const {
      return RestoreSSDScore(metric_, distance, query_norm, max_base_norm_);
    }
    float RawDistance(const T* query, const T* vector, size_t dimensions,
                      uint32_t id) const {
      const auto distance = raw_distance_(query, vector, dimensions);
      return metric_ == Metric::COSINE ? distance / base_norms_[id] : distance;
    }
    // Preserve the legacy centroid-selection convention on already-decoded
    // vectors independently of the payload encoding.
    float DecodedDistance(const float* query, const float* vector,
                          size_t dimensions, uint32_t id) const {
      const auto distance = float_distance_(query, vector, dimensions);
      return metric_ == Metric::COSINE ? distance / base_norms_[id] : distance;
    }
    void CopyOriginal(const void* payload, T* output) const {
      if (!HasRawData())
        throw std::runtime_error(
            "SSD encoding does not retain original vectors for this metric");
      std::memcpy(output, payload, dimension_ * sizeof(T));
    }

   protected:
    size_t NormBytes() const {
      return base_norms_ ? count_ * sizeof(float) : 0;
    }
    Metric                   metric_;
    uint64_t                 dimension_, aligned_dimension_, count_;
    float                    max_base_norm_ = 0;
    std::unique_ptr<float[]> base_norms_;
    DISTFUN<T>               raw_distance_;
    DISTFUN<float>           float_distance_;
  };

  template<typename T>
  class RawDiskStorageCodec final : public DiskStorageCodec<T> {
   public:
    using DiskStorageCodec<T>::DiskStorageCodec;
    bool IsCompressed() const override {
      return false;
    }
    size_t PayloadBytes() const override {
      return this->dimension_ * sizeof(T);
    }
    float Score(const T* query, const float*, const void* payload,
                uint32_t id) const override {
      return this->RawDistance(query, static_cast<const T*>(payload),
                               this->aligned_dimension_, id);
    }
    void Decode(const void* payload, float* output) const override {
      const auto* vector = static_cast<const T*>(payload);
      for (uint64_t i = 0; i < this->dimension_; ++i)
        output[i] = vector[i];
    }
    size_t MemorySize() const override {
      return sizeof(*this) + this->NormBytes();
    }
  };

  template<typename T>
  class PQDiskStorageCodec final : public DiskStorageCodec<T> {
   public:
    PQDiskStorageCodec(Metric metric, uint64_t dimension, uint64_t count,
                       const std::string& disk_file)
        : DiskStorageCodec<T>(metric, dimension, count, disk_file) {
      table_.load_pq_centroid_bin((disk_file + "_pq_pivots.bin").c_str(), 0);
      if (table_.get_total_dims() != dimension ||
          table_.get_num_chunks() == 0 || table_.get_num_chunks() > dimension)
        throw std::runtime_error(
            "SSD PQ metadata does not match the prepared space");
    }
    bool IsCompressed() const override {
      return true;
    }
    size_t PayloadBytes() const override {
      return table_.get_num_chunks();
    }
    float Score(const T*, const float* query, const void* payload,
                uint32_t) const override {
      auto* code = static_cast<uint8_t*>(const_cast<void*>(payload));
      if (this->metric_ == Metric::INNER_PRODUCT)
        return 2.0f + 2.0f * table_.inner_product(query, code);
      if (this->metric_ == Metric::COSINE)
        return table_.inner_product(query, code);
      return table_.l2_distance(query, code);
    }
    void Decode(const void* payload, float* output) const override {
      table_.inflate_vector(static_cast<uint8_t*>(const_cast<void*>(payload)),
                            output);
    }
    size_t MemorySize() const override {
      return sizeof(*this) + this->NormBytes() +
             this->dimension_ *
                 (256 * sizeof(float) * 2 + sizeof(uint32_t) + sizeof(float)) +
             (table_.get_num_chunks() + 1) * sizeof(uint32_t);
    }

   private:
    // The legacy PQ API lacks const qualifiers on its read-only kernels.
    mutable FixedChunkPQTable table_;
  };

  template<typename T>
  std::unique_ptr<DiskStorageCodec<T>> LoadDiskStorageCodec(
      const std::string& disk_file, Metric metric, uint64_t dimension,
      uint64_t count) {
    if (file_exists(disk_file + "_pq_pivots.bin"))
      return std::make_unique<PQDiskStorageCodec<T>>(metric, dimension, count,
                                                     disk_file);
    return std::make_unique<RawDiskStorageCodec<T>>(metric, dimension, count,
                                                    disk_file);
  }
}  // namespace diskann
