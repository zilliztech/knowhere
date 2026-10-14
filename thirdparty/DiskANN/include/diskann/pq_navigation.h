// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>
#include "diskann/navigation_runtime.h"
#include "diskann/pq_table.h"

namespace diskann {
  // AiSAQ retains its existing reader and layout-specific code source.
  class PQDataGetter {
   public:
    virtual _u8* get_pq_data() = 0;
    virtual _u64 get_origin_id(_u64 id) = 0;
    virtual void release_pq_data(size_t offset = 0, size_t size = 0) = 0;
    virtual void aggregate_pq_coords(const unsigned* ids, _u64 count,
                                     _u64 chunks, _u8* output) = 0;
    virtual ~PQDataGetter() = default;
  };

  class PQNavigationStore;

  class PQNavigationDistanceComputer final : public NavigationDistanceComputer {
   public:
    PQNavigationDistanceComputer(const PQNavigationStore& model,
                                 uint64_t                 aligned_dimension);
    ~PQNavigationDistanceComputer() override;
    void set_query(const float* query) override;
    void compute_distances(const unsigned* ids, uint64_t count,
                           float* distances) override;
    void compute_distances(const unsigned* ids, uint64_t count,
                           float* distances, PQDataGetter& source);
    // Inline, cached and reader-completed codes use the same PQ lookup.
    void EvaluateCodes(const uint8_t* codes, uint64_t count, float* distances);

   private:
    const PQNavigationStore& model_;
    uint8_t*                 gathered_codes_ = nullptr;
    float*                   lookup_table_ = nullptr;
  };

  // A stack adapter for AiSAQ's filtered scan. It shares the borrowed slot's
  // LUT and gather buffer and does not allocate another query workspace.
  class PQCodeDistanceComputer final : public NavigationDistanceComputer {
   public:
    PQCodeDistanceComputer(PQNavigationDistanceComputer& computer,
                           PQDataGetter&                 source)
        : computer_(computer), source_(source) {
    }
    void set_query(const float* query) override {
      computer_.set_query(query);
    }
    void compute_distances(const unsigned* ids, uint64_t count,
                           float* distances) override {
      computer_.compute_distances(ids, count, distances, source_);
    }

   private:
    PQNavigationDistanceComputer& computer_;
    PQDataGetter&                 source_;
  };

  class PQNavigationStore final : public NavigationStore {
   public:
    explicit PQNavigationStore(bool        resident_codes = true,
                               bool        rearranged = false,
                               std::string pivots_path = {});
    const char* CodecName() const override {
      return "PQ";
    }
    int      Load(const std::string& prefix) override;
    void     GetNecessaryFilenames(const std::string&        prefix,
                                   std::vector<std::string>& filenames,
                                   NavigationFilePurpose     purpose =
                                       NavigationFilePurpose::Build) const override;
    uint64_t Count() const override {
      return count_;
    }
    uint64_t Dimension() const override {
      return dimension_;
    }
    size_t MemorySize() const override;
    size_t QueryMemorySize(uint64_t aligned_dimension) const override;
    std::unique_ptr<NavigationDistanceComputer> CreateDistanceComputer(
        uint64_t aligned_dimension) const override;
    uint64_t ChunkCount() const {
      return chunks_;
    }

   private:
    friend class PQNavigationDistanceComputer;
    bool                       resident_codes_;
    bool                       rearranged_;
    std::string                pivots_path_;
    uint64_t                   count_ = 0, dimension_ = 0, chunks_ = 0;
    std::unique_ptr<uint8_t[]> codes_;
    // The existing table API is non-const; query preparation only reads it.
    mutable FixedChunkPQTable pq_table_;
  };
}  // namespace diskann
