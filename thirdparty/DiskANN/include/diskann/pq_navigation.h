// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>

#include "diskann/navigation_runtime.h"

namespace diskann {
  struct PQNavigationState;
  struct PQNavigationQueryTable;

  // PQ-only encoding source. AiSAQ supplies reader-backed codes; ordinary PQ
  // owns resident codes. This interface is not part of the common graph engine.
  class PQCodeSource {
   public:
    virtual ~PQCodeSource() = default;
    virtual void gather(const unsigned* ids, uint64_t count, uint64_t chunks,
                        uint8_t* output) = 0;
  };

  class PQNavigationDistanceComputer final : public NavigationDistanceComputer {
   public:
    PQNavigationDistanceComputer(std::shared_ptr<PQNavigationState> state,
                                 PQCodeSource* source = nullptr);
    ~PQNavigationDistanceComputer() override;
    void set_query(const float* query) override;
    void compute_distances(const unsigned* ids, uint64_t count,
                           float* distances, float threshold,
                           bool threshold_valid, QueryStats* stats) override;
    // AiSAQ scores inline/cache/read-completion codes without changing its IO
    // scheduling. These extensions are deliberately confined to PQ.
    void         evaluate_encoded(const uint8_t* codes, uint64_t count,
                                  float* distances);
    const float* lookup_table() const;

   private:
    std::shared_ptr<PQNavigationState>      state_;
    std::unique_ptr<PQNavigationQueryTable> workspace_;
    PQCodeSource*                           source_;
  };

  class PQNavigationStore final : public NavigationStore {
   public:
    explicit PQNavigationStore(const std::string& prefix);
    PQNavigationStore(const std::string& pivots, const std::string& codes,
                      bool resident);
    ~PQNavigationStore() override;
    int64_t  Count() const override;
    int64_t  Dimension() const override;
    size_t   MemorySize() const override;
    uint64_t ChunkCount() const;
    void     PrepareWorkspaces(size_t count) const;
    std::unique_ptr<NavigationDistanceComputer> CreateDistanceComputer(
        const NavigationQueryOptions& options) const override;
    std::unique_ptr<PQNavigationDistanceComputer> CreatePQDistanceComputer(
        PQCodeSource* source = nullptr) const;

   private:
    std::shared_ptr<PQNavigationState> state_;
  };
}  // namespace diskann
