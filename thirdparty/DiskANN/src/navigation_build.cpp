// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "diskann/navigation_build.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "diskann/aux_utils.h"
#include "diskann/defaults.h"
#include "diskann/partition_and_pq.h"
#include "diskann/utils.h"

namespace diskann {
  std::vector<std::string> pq_navigation_files(const std::string &prefix,
                                               bool               rearranged) {
    const auto pivots = get_pq_pivots_filename(prefix);
    return {pivots, get_pq_rearrangement_perm_filename(pivots),
            get_pq_chunk_offsets_filename(pivots),
            get_pq_centroid_filename(pivots),
            rearranged ? get_pq_compressed_rearranged_filename(prefix)
                       : get_pq_compressed_filename(prefix)};
  }

  template<typename T>
  class PQNavigationBuilder final : public NavigationBuilder {
   public:
    bool validate(const BuildConfig &config) const override {
      if (get_memory_budget(config.pq_code_size_gb) <= 0) {
        LOG_KNOWHERE_ERROR_ << "PQ navigation requires a positive code budget";
        return false;
      }
      return true;
    }
    bool needs_training_sample() const override {
      return true;
    }
    bool supports_aisaq() const override {
      return true;
    }
    void build(const BuildConfig &config, const PreparedBuildContext &context,
               const NavigationTrainingData &sample) const override {
      const auto start = std::chrono::high_resolution_clock::now();
      const auto budget = get_memory_budget(config.pq_code_size_gb);
      size_t     chunks =
          static_cast<size_t>(std::floor(_u64(budget / context.rows)));
      chunks = std::max<size_t>(1, chunks);
      chunks = std::min(chunks, context.prepared_dim);
      chunks = std::min<size_t>(chunks, diskann::defaults::MAX_PQ_CHUNKS);
      const auto pivots = get_pq_pivots_filename(context.prefix);
      const auto codes = get_pq_compressed_filename(context.prefix);
      LOG_KNOWHERE_INFO_ << "Compressing " << context.prepared_dim
                         << "-dimensional data into " << chunks
                         << " bytes per vector.";
      generate_pq_pivots(sample.data, sample.rows,
                         (uint32_t) context.prepared_dim, 256,
                         (uint32_t) chunks, NUM_KMEANS_REPS, pivots,
                         context.metric == diskann::Metric::L2);
      generate_pq_data_from_pivots<T>(context.prepared_source.c_str(), 256,
                                      (uint32_t) chunks, pivots, codes);
      const std::chrono::duration<double> elapsed =
          std::chrono::high_resolution_clock::now() - start;
      LOG_KNOWHERE_INFO_ << "Training PQ codes cost: " << elapsed.count()
                         << "s";
    }
    void build_cache(const BuildConfig                        &config,
                     const PreparedBuildContext               &context,
                     const std::vector<std::vector<unsigned>> &graph,
                     unsigned entry_point) const override {
      const auto sample_file = get_sample_data_filename(context.prefix);
      const auto pivots = get_pq_pivots_filename(context.prefix);
      const auto codes = get_pq_compressed_filename(context.prefix);
      // Keep the native cache-generation policy and its memory allowance.
      constexpr float cache_mem_factor = 1.1;
      const auto      usage = cache_mem_factor *
                         (get_file_size(context.graph_index_path) +
                          get_file_size(sample_file) + get_file_size(codes) +
                          get_file_size(pivots)) /
                         (1024 * 1024 * 1024);
      if (config.num_nodes_to_cache > 0 && !graph.empty() &&
          usage < config.index_mem_gb) {
        generate_cache_list_from_graph_with_pq<T>(
            config.num_nodes_to_cache, config.max_degree, context.metric,
            sample_file, pivots, codes, entry_point, graph,
            get_cached_nodes_file(context.prefix));
      }
    }
  };

  template<typename T>
  std::unique_ptr<NavigationBuilder> make_pq_navigation_builder() {
    return std::make_unique<PQNavigationBuilder<T>>();
  }
  template std::unique_ptr<NavigationBuilder>
  make_pq_navigation_builder<float>();
  template std::unique_ptr<NavigationBuilder>
  make_pq_navigation_builder<knowhere::fp16>();
  template std::unique_ptr<NavigationBuilder>
  make_pq_navigation_builder<knowhere::bf16>();
}  // namespace diskann
