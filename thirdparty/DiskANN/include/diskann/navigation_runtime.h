// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace diskann {
  struct QueryStats;

  struct NavigationSpace {
    uint64_t count;
    uint64_t
        dimension;  // Metric-prepared space, including IP's extra coordinate.
    uint64_t input_dimension;
  };

  // Query-local state. Inputs and batch threshold snapshots belong to the
  // navigation score domain (smaller is better), never the SSD score domain.
  // A scorer may explicitly reject a candidate with +infinity under its
  // approximate policy. It must not modify the graph's candidate queues.
  class NavigationDistanceComputer {
   public:
    virtual ~NavigationDistanceComputer() = default;
    virtual void set_query(const float* query) = 0;
    virtual void compute_distances(const unsigned* ids, uint64_t count,
                                   float* distances, float threshold,
                                   bool threshold_valid, QueryStats* stats) = 0;
  };

  // Immutable model data; each computer owns independent mutable query state.
  // Implementations keep their model data alive for outstanding computers.
  class NavigationStore {
   public:
    virtual ~NavigationStore() = default;
    virtual int64_t                                     Count() const = 0;
    virtual int64_t                                     Dimension() const = 0;
    virtual size_t                                      MemorySize() const = 0;
    virtual std::unique_ptr<NavigationDistanceComputer> CreateDistanceComputer()
        const = 0;
  };

  // User ID mapping belongs to the index layout, not the navigation encoding.
  class IndexIdMapping {
   public:
    virtual ~IndexIdMapping() = default;
    virtual uint64_t get_origin_id(uint64_t id) const {
      return id;
    }
  };
}  // namespace diskann
