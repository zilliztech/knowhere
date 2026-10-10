// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace diskann {
  struct BuildConfig;
  class PreparedBuildContext;

  // Build-time adapter only. The graph search hot path is deliberately
  // separate.
  struct NavigationTrainingData {
    const float *data;
    size_t       rows;
    size_t       dimension;
  };

  class NavigationBuilder {
   public:
    virtual ~NavigationBuilder() = default;
    virtual bool validate(const BuildConfig &) const {
      return true;
    }
    virtual bool needs_training_sample() const {
      return false;
    }
    virtual bool supports_aisaq() const {
      return false;
    }
    virtual void build(const BuildConfig &, const PreparedBuildContext &,
                       const NavigationTrainingData &) const = 0;
    virtual void build_cache(const BuildConfig &, const PreparedBuildContext &,
                             const std::vector<std::vector<unsigned>> &,
                             unsigned) const {
    }
  };

  // Shared file declaration for native PQ build, Knowhere registration/load and
  // AiSAQ's rearranged-code layout. The on-disk format is unchanged.
  std::vector<std::string> pq_navigation_files(const std::string &prefix,
                                               bool rearranged = false);
  template<typename T>
  std::unique_ptr<NavigationBuilder> make_pq_navigation_builder();
}  // namespace diskann
