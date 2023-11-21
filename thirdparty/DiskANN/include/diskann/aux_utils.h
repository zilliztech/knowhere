// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <algorithm>
#include <fcntl.h>
#include <future>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#include <unistd.h>

#include "cached_io.h"
#include "common_includes.h"
#include "tsl/robin_set.h"

#include "utils.h"
#include "knowhere/comp/thread_pool.h"
#include "diskann/index.h"

typedef int FileHandle;

namespace diskann {
  constexpr size_t MAX_N_SECTOR_READS = 256;
  const size_t     MAX_PQ_TRAINING_SET_SIZE = 256000;
  const size_t     MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
  const double     PQ_TRAINING_SET_FRACTION = 0.1;
  const double     SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
  const double     THRESHOLD_FOR_CACHING_IN_GB = 1.0;
  const uint32_t   NUM_NODES_TO_CACHE = 250000;
  const uint32_t   WARMUP_L = 20;
  const uint32_t   NUM_KMEANS_REPS = 12;

  template<typename T>
  class PQFlashIndex;

  double get_memory_budget(const std::string &mem_budget_str);
  double get_memory_budget(double search_ram_budget_in_gb);
  void   add_new_file_to_single_index(std::string index_file,
                                      std::string new_file);

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned recall_at);

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned                        recall_at,
                          const tsl::robin_set<unsigned> &active_tags);

  double calculate_range_search_recall(
      unsigned num_queries, std::vector<std::vector<_u32>> &groundtruth,
      std::vector<std::vector<_u32>> &our_results);

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs);

  template<typename T>
  T *load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num,
                 uint64_t warmup_dim, uint64_t warmup_aligned_dim);

  int merge_shards(const std::string &vamana_prefix,
                   const std::string &vamana_suffix,
                   const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards,
                   unsigned max_degree, const std::string &output_vamana,
                   const std::string &medoids_file);

  template<typename T>
  std::string preprocess_base_file(const std::string &infile,
                                   const std::string &indexPrefix,
                                   diskann::Metric   &distMetric);

  /* The entry point of the graph is used as the return value. If the graph
   * parameter cannot be generated successfully, it is set to -1.*/
  template<typename T>
  std::unique_ptr<diskann::Index<T>> build_merged_vamana_index(
      std::string base_file, diskann::Metric _compareMetric, unsigned L,
      unsigned R, bool accelerate_build, double sampling_rate,
      double ram_budget, std::string mem_index_path, std::string medoids_file,
      std::string centroids_file);

  template<typename T>
  void generate_cache_list_from_graph_with_pq(
      _u64 num_nodes_to_cache, unsigned R, const diskann::Metric compare_metric,
      const std::string &sample_file, const std::string &pq_pivots_path,
      const std::string &pq_compressed_code_path, const unsigned entry_point,
      const std::vector<std::vector<unsigned>> &graph,
      const std::string                        &cache_file);

  template<typename T>
  uint32_t optimize_beamwidth(
      std::unique_ptr<diskann::PQFlashIndex<T>> &_pFlashIndex, T *tuning_sample,
      _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim, uint32_t L,
      uint32_t nthreads, uint32_t start_bw = 2);

  struct BuildConfig {
    std::string     data_file_path = "";
    std::string     index_file_path = "";
    diskann::Metric compare_metric = diskann::Metric::L2;
    // R (max degree)
    unsigned max_degree = 0;
    // L (indexing list size, better if >= R)
    unsigned search_list_size = 0;
    // B (PQ code size in GB)
    double pq_code_size_gb = 0.0;
    // M (memory limit while indexing)
    double index_mem_gb = 0.0;
    // B' (PQ dim for disk index: optional parameter for very
    // large dimensional data)
    uint32_t disk_pq_dims = 0;
    // reorder (set true to include full precision in data file:
    // optional paramter, use only when using disk PQ
    bool reorder = false;
    // acclerate the index build ~30% and lose ~1% recall
    bool accelerate_build = false;
    // the cached nodes number
    uint32_t num_nodes_to_cache = 0;
  };

  template<typename T>
  int build_disk_index(const BuildConfig &config);

  template<typename T>
  void create_disk_layout(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file,
      const std::string reorder_data_file = std::string(""));

}  // namespace diskann
