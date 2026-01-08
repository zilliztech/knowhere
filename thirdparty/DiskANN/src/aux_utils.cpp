// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#include "boost/dynamic_bitset.hpp"
#include "diskann/aisaq_utils.h"
#include "diskann/aux_utils.h"
#include "diskann/cached_io.h"
#include "diskann/index.h"
#include "diskann/logger.h"
#include "diskann/partition_and_pq.h"
#include "diskann/percentile_stats.h"
#include "diskann/pq_flash_index.h"
#include "knowhere/comp/task.h"
#include "tsl/robin_set.h"
#ifdef KNOWHERE_WITH_CUVS
#include "diskann/diskann_gpu.h"
#endif


namespace diskann {
  namespace {
    static constexpr uint32_t kSearchLForCache = 15;
    static constexpr float    kCacheMemFactor = 1.1;
  };  // namespace

  void add_new_file_to_single_index(std::string index_file,
                                    std::string new_file) {
    std::unique_ptr<_u64[]> metadata;
    _u64                    nr, nc;
    diskann::load_bin<_u64>(index_file, metadata, nr, nc);
    if (nc != 1) {
      std::stringstream stream;
      stream << "Error, index file specified does not have correct metadata. "
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }
    size_t          index_ending_offset = metadata[nr - 1];
    _u64            read_blk_size = 64 * 1024 * 1024;
    cached_ofstream writer(index_file, read_blk_size);
    _u64            check_file_size = get_file_size(index_file);
    if (check_file_size != index_ending_offset) {
      std::stringstream stream;
      stream << "Error, index file specified does not have correct metadata "
                "(last entry must match the filesize). "
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }

    cached_ifstream reader(new_file, read_blk_size);
    size_t          fsize = reader.get_file_size();
    if (fsize == 0) {
      std::stringstream stream;
      stream << "Error, new file specified is empty. Not appending. "
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }

    size_t num_blocks = DIV_ROUND_UP(fsize, read_blk_size);
    auto dump = std::make_unique<char[]>(read_blk_size);
    for (_u64 i = 0; i < num_blocks; i++) {
      size_t cur_block_size = read_blk_size > fsize - (i * read_blk_size)
                                  ? fsize - (i * read_blk_size)
                                  : read_blk_size;
      reader.read(dump.get(), cur_block_size);
      writer.write(dump.get(), cur_block_size);
    }
    dump.reset();
    //    reader.close();
    //    writer.close();

    std::vector<_u64> new_meta;
    for (_u64 i = 0; i < nr; i++)
      new_meta.push_back(metadata[i]);
    new_meta.push_back(metadata[nr - 1] + fsize);

    diskann::save_bin<_u64>(index_file, new_meta.data(), new_meta.size(), 1);
  }

  double get_memory_budget(double pq_code_size) {
    double final_pq_code_limit = pq_code_size;
    return final_pq_code_limit * 1024 * 1024 * 1024;
  }

  double get_memory_budget(const std::string &mem_budget_str) {
    double search_ram_budget = atof(mem_budget_str.c_str());
    return get_memory_budget(search_ram_budget);
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned recall_at) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t    tie_breaker = recall_at;
      if (gs_dist != nullptr) {
        tie_breaker = recall_at - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[recall_at - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec,
                 res_vec + recall_at);  // change to recall_at for recall k@k or
                                        // dim_or for k@dim_or
      unsigned cur_recall = 0;
      for (auto &v : gt) {
        if (res.find(v) != res.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return total_recall / (num_queries) * (100.0 / recall_at);
  }

  double calculate_recall(unsigned num_queries, unsigned *gold_std,
                          float *gs_dist, unsigned dim_gs,
                          unsigned *our_results, unsigned dim_or,
                          unsigned                        recall_at,
                          const tsl::robin_set<unsigned> &active_tags) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;
    bool               printed = false;
    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      size_t    tie_breaker = recall_at;
      unsigned  active_points_count = 0;
      unsigned  cur_counter = 0;
      while (active_points_count < recall_at && cur_counter < dim_gs) {
        if (active_tags.find(*(gt_vec + cur_counter)) != active_tags.end()) {
          active_points_count++;
        }
        cur_counter++;
      }
      if (active_tags.empty())
        cur_counter = recall_at;

      if ((active_points_count < recall_at && !active_tags.empty()) &&
          !printed) {
        LOG_KNOWHERE_INFO_ << "Warning: Couldn't find enough closest neighbors "
                      << active_points_count << "/" << recall_at
                      << " from truthset for query # "
                      << i << ". Will result in under-reported value of recall.";
        printed = true;
      }
      if (gs_dist != nullptr) {
        tie_breaker = cur_counter - 1;
        float *gt_dist_vec = gs_dist + dim_gs * i;
        while (tie_breaker < dim_gs &&
               gt_dist_vec[tie_breaker] == gt_dist_vec[cur_counter - 1])
          tie_breaker++;
      }

      gt.insert(gt_vec, gt_vec + tie_breaker);
      res.insert(res_vec, res_vec + recall_at);
      unsigned cur_recall = 0;
      for (auto &v : res) {
        if (gt.find(v) != gt.end()) {
          cur_recall++;
        }
      }
      total_recall += cur_recall;
    }
    return ((double) (total_recall / (num_queries))) *
           ((double) (100.0 / recall_at));
  }

  double calculate_range_search_recall(
      unsigned num_queries, std::vector<std::vector<_u32>> &groundtruth,
      std::vector<std::vector<_u32>> &our_results) {
    double             total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();

      gt.insert(groundtruth[i].begin(), groundtruth[i].end());
      res.insert(our_results[i].begin(), our_results[i].end());
      unsigned cur_recall = 0;
      for (auto &v : gt) {
        if (res.find(v) != res.end()) {
          cur_recall++;
        }
      }
      if (gt.size() != 0)
        total_recall += ((100.0 * cur_recall) / gt.size());
      else
        total_recall += 100;
    }
    return total_recall / (num_queries);
  }

  template<typename T>
  T *generateRandomWarmup(uint64_t warmup_num, uint64_t warmup_dim,
                          uint64_t warmup_aligned_dim) {
    T *warmup = nullptr;
    warmup_num = 100000;
    diskann::cout << "Generating random warmup file with dim " << warmup_dim
                  << " and aligned dim " << warmup_aligned_dim << std::flush;
    diskann::alloc_aligned(((void **) &warmup),
                           warmup_num * warmup_aligned_dim * sizeof(T),
                           8 * sizeof(T));
    std::memset((void *) warmup, 0,
                warmup_num * warmup_aligned_dim * sizeof(T));
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);
    for (uint32_t i = 0; i < warmup_num; i++) {
      for (uint32_t d = 0; d < warmup_dim; d++) {
        warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
      }
    }
    LOG_KNOWHERE_INFO_ << "..done";
    return warmup;
  }

  template<typename T>
  T *load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num,
                 uint64_t warmup_dim, uint64_t warmup_aligned_dim) {
    T       *warmup = nullptr;
    uint64_t file_dim, file_aligned_dim;

    if (file_exists(cache_warmup_file)) {
      diskann::load_aligned_bin<T>(cache_warmup_file, warmup, warmup_num,
                                   file_dim, file_aligned_dim);
      if (file_dim != warmup_dim || file_aligned_dim != warmup_aligned_dim) {
        std::stringstream stream;
        stream << "Mismatched dimensions in sample file. file_dim = "
               << file_dim << " file_aligned_dim: " << file_aligned_dim
               << " index_dim: " << warmup_dim
               << " index_aligned_dim: " << warmup_aligned_dim << std::endl;
        throw diskann::ANNException(stream.str(), -1);
      }
    } else {
      warmup =
          generateRandomWarmup<T>(warmup_num, warmup_dim, warmup_aligned_dim);
    }
    return warmup;
  }

  /***************************************************
      Support for Merging Many Vamana Indices
   ***************************************************/

  void read_idmap(const std::string &fname, std::vector<unsigned> &ivecs) {
    uint32_t      npts32, dim;
    size_t        actual_file_size = get_file_size(fname);
    std::ifstream reader(fname.c_str(), std::ios::binary);
    reader.read((char *) &npts32, sizeof(uint32_t));
    reader.read((char *) &dim, sizeof(uint32_t));
    if (dim != 1 || actual_file_size != ((size_t) npts32) * sizeof(uint32_t) +
                                            2 * sizeof(uint32_t)) {
      std::stringstream stream;
      stream << "Error reading idmap file. Check if the file is bin file with "
                "1 dimensional data. Actual: "
             << actual_file_size
             << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t)
             << std::endl;

      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    ivecs.resize(npts32);
    reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
    reader.close();
  }

  int merge_shards(const std::string &vamana_prefix,
                   const std::string &vamana_suffix,
                   const std::string &idmaps_prefix,
                   const std::string &idmaps_suffix, const _u64 nshards,
                   unsigned max_degree, const std::string &output_vamana,
                   const std::string &medoids_file) {
    // Read ID maps
    std::vector<std::string>           vamana_names(nshards);
    std::vector<std::vector<unsigned>> idmaps(nshards);
    for (_u64 shard = 0; shard < nshards; shard++) {
      vamana_names[shard] =
          vamana_prefix + std::to_string(shard) + vamana_suffix;
      read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix,
                 idmaps[shard]);
    }

    // find max node id
    _u64 nnodes = 0;
    _u64 nelems = 0;
    for (auto &idmap : idmaps) {
      for (auto &id : idmap) {
        nnodes = std::max(nnodes, (_u64) id);
      }
      nelems += idmap.size();
    }
    nnodes++;
    LOG_KNOWHERE_DEBUG_ << "# nodes: " << nnodes
                        << ", max. degree: " << max_degree;

    // compute inverse map: node -> shards
    std::vector<std::pair<unsigned, unsigned>> node_shard;
    node_shard.reserve(nelems);
    for (_u64 shard = 0; shard < nshards; shard++) {
      LOG_KNOWHERE_INFO_ << "Creating inverse map -- shard #" << shard;
      for (_u64 idx = 0; idx < idmaps[shard].size(); idx++) {
        _u64 node_id = idmaps[shard][idx];
        node_shard.push_back(std::make_pair((_u32) node_id, (_u32) shard));
      }
    }
    std::sort(node_shard.begin(), node_shard.end(),
              [](const auto &left, const auto &right) {
                return left.first < right.first || (left.first == right.first &&
                                                    left.second < right.second);
              });
    LOG_KNOWHERE_INFO_ << "Finished computing node -> shards map";

    // create cached vamana readers
    std::vector<cached_ifstream> vamana_readers(nshards);
    for (_u64 i = 0; i < nshards; i++) {
      vamana_readers[i].open(vamana_names[i], BUFFER_SIZE_FOR_CACHED_IO);
      size_t expected_file_size;
      vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
    }

    size_t vamana_metadata_size =
        sizeof(_u64) + sizeof(_u32) + sizeof(_u32) +
        sizeof(_u64);  // expected file size + max degree + medoid_id +
                       // frozen_point info

    // create cached vamana writers
    cached_ofstream merged_vamana_writer(output_vamana,
                                         BUFFER_SIZE_FOR_CACHED_IO);

    size_t merged_index_size =
        vamana_metadata_size;  // we initialize the size of the merged index to
                               // the metadata size
    size_t merged_index_frozen = 0;
    merged_vamana_writer.write(
        (char *) &merged_index_size,
        sizeof(uint64_t));  // we will overwrite the index size at the end

    unsigned output_width = max_degree;
    unsigned max_input_width = 0;
    // read width from each vamana to advance buffer by sizeof(unsigned) bytes
    for (auto &reader : vamana_readers) {
      unsigned input_width;
      reader.read((char *) &input_width, sizeof(unsigned));
      max_input_width =
          input_width > max_input_width ? input_width : max_input_width;
    }

    LOG_KNOWHERE_INFO_ << "Max input width: " << max_input_width
                       << ", output width: " << output_width;

    merged_vamana_writer.write((char *) &output_width, sizeof(unsigned));
    std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
    _u32          nshards_u32 = (_u32) nshards;
    _u32          one_val = 1;
    medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
    medoid_writer.write((char *) &one_val, sizeof(uint32_t));

    _u64 vamana_index_frozen =
        0;  // as of now the functionality to merge many overlapping vamana
            // indices is supported only for bulk indices without frozen point.
            // Hence the final index will also not have any frozen points.
    for (_u64 shard = 0; shard < nshards; shard++) {
      unsigned medoid;
      // read medoid
      vamana_readers[shard].read((char *) &medoid, sizeof(unsigned));
      vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(_u64));
      assert(vamana_index_frozen == false);
      // rename medoid
      medoid = idmaps[shard][medoid];

      medoid_writer.write((char *) &medoid, sizeof(uint32_t));
      // write renamed medoid
      if (shard == (nshards - 1))  //--> uncomment if running hierarchical
        merged_vamana_writer.write((char *) &medoid, sizeof(unsigned));
    }
    merged_vamana_writer.write((char *) &merged_index_frozen, sizeof(_u64));
    medoid_writer.close();

    LOG_KNOWHERE_INFO_ << "Starting merge";

    // Gopal. random_shuffle() is deprecated.
    std::random_device rng;
    std::mt19937       urng(rng());

    std::vector<bool>     nhood_set(nnodes, 0);
    std::vector<unsigned> final_nhood;

    unsigned nnbrs = 0, shard_nnbrs = 0;
    unsigned cur_id = 0;
    for (const auto &id_shard : node_shard) {
      unsigned node_id = id_shard.first;
      unsigned shard_id = id_shard.second;
      if (cur_id < node_id) {
        // Gopal. random_shuffle() is deprecated.
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs =
            (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        merged_vamana_writer.write((char *) &nnbrs, sizeof(unsigned));
        merged_vamana_writer.write((char *) final_nhood.data(),
                                   nnbrs * sizeof(unsigned));
        merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
        if (cur_id % 499999 == 1) {
          LOG_KNOWHERE_DEBUG_ << ".";
        }
        cur_id = node_id;
        nnbrs = 0;
        for (auto &p : final_nhood)
          nhood_set[p] = 0;
        final_nhood.clear();
      }
      // read from shard_id ifstream
      vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(unsigned));
      std::vector<unsigned> shard_nhood(shard_nnbrs);
      vamana_readers[shard_id].read((char *) shard_nhood.data(),
                                    shard_nnbrs * sizeof(unsigned));

      // rename nodes
      for (_u64 j = 0; j < shard_nnbrs; j++) {
        if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
          nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
          final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
        }
      }
    }

    // Gopal. random_shuffle() is deprecated.
    std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
    nnbrs = (unsigned) (std::min)(final_nhood.size(), (uint64_t) max_degree);
    // write into merged ofstream
    merged_vamana_writer.write((char *) &nnbrs, sizeof(unsigned));
    merged_vamana_writer.write((char *) final_nhood.data(),
                               nnbrs * sizeof(unsigned));
    merged_index_size += (sizeof(unsigned) + nnbrs * sizeof(unsigned));
    for (auto &p : final_nhood)
      nhood_set[p] = 0;
    final_nhood.clear();

    LOG_KNOWHERE_DEBUG_ << "Expected size: " << merged_index_size;

    merged_vamana_writer.reset();
    merged_vamana_writer.write((char *) &merged_index_size, sizeof(uint64_t));

    LOG_KNOWHERE_INFO_ << "Finished merge";
    return 0;
  }

  template<typename T>
  std::unique_ptr<diskann::Index<T>> build_merged_vamana_index(
      std::string base_file, bool ip_prepared, diskann::Metric compareMetric,
      unsigned L, unsigned R, bool accelerate_build, bool shuffle_build, double sampling_rate,
      double ram_budget, std::string mem_index_path, std::string medoids_file,
      std::string centroids_file) {
    size_t base_num, base_dim;
    uint32_t shard_r = 2*R/3;
    diskann::get_bin_metadata(base_file, base_num, base_dim);
#ifdef KNOWHERE_WITH_CUVS
    raft::device_resources dev_resources;
    if(is_gpu_available()) {
      size_t gpu_free_mem, gpu_total_mem;
      gpu_get_mem_info(dev_resources, gpu_free_mem, gpu_total_mem);
      LOG_KNOWHERE_INFO_ << "GPU has " <<  gpu_free_mem/(1024*1024*1024L) <<
              " Gib free memory out of " <<  gpu_total_mem/(1024*1024*1024L) << " Gib total";
      ram_budget = std::min<double>(ram_budget,(double)0.9*(gpu_free_mem/(1024*1024*1024)));
      shard_r = std::max<uint32_t>((uint32_t)32,(uint32_t)R/2);
    }
#endif

    double full_index_ram =
        estimate_ram_usage(base_num, base_dim, sizeof(T), R);
    if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
      LOG_KNOWHERE_INFO_
          << "Full index fits in RAM budget, should consume at most "
          << full_index_ram / (1024 * 1024 * 1024)
          << "GiBs, so building in one shot";
      diskann::Parameters paras;
      paras.Set<unsigned>("L", (unsigned) L);
      paras.Set<unsigned>("R", (unsigned) R);
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<unsigned>("num_rnds", 2);
      paras.Set<bool>("saturate_graph", 1);
      paras.Set<std::string>("save_path", mem_index_path);
      paras.Set<bool>("accelerate_build", accelerate_build);
      paras.Set<bool>("shuffle_build", shuffle_build);

      std::unique_ptr<diskann::Index<T>> _pvamanaIndex =
          std::unique_ptr<diskann::Index<T>>(new diskann::Index<T>(
          compareMetric, ip_prepared, base_dim, base_num, false, false));
      bool built_with_gpu=false;
#ifdef KNOWHERE_WITH_CUVS
      //currently cuvs vamana build only supports L2Expanded metric
      if (compareMetric == diskann::L2 && is_gpu_available () &&
              (std::is_same_v<T, float> || std::is_same_v<T, uint8_t>) ) {
        LOG_KNOWHERE_INFO_ << "Building with GPU!" << " R= "<< R<<" L=" << L;

        if (std::is_same_v<T, float>) {
          auto dataset = read_bin_dataset<float, uint64_t>(dev_resources, base_file);
          vamana_build_and_write<float>(dev_resources,
                                           raft::make_const_mdspan(dataset.view()),
                                           mem_index_path,
                                           R,
                                           L,
                                           0.06,
                                           1);
        }else {
          auto dataset = read_bin_dataset<uint8_t, uint64_t>(dev_resources, base_file);
          vamana_build_and_write<uint8_t>(dev_resources,
                                           raft::make_const_mdspan(dataset.view()),
                                           mem_index_path,
                                           R,
                                           L,
                                           0.06,
                                           1);
        }
        built_with_gpu=true;
      }
#endif
      if(!built_with_gpu) {
        _pvamanaIndex->build(base_file.c_str(), base_num, paras);
        _pvamanaIndex->save(mem_index_path.c_str(), true);
      }else {
        _pvamanaIndex->load_graph(mem_index_path, base_num);
      }

      std::remove(medoids_file.c_str());
      std::remove(centroids_file.c_str());
      return _pvamanaIndex;
    }
    std::string merged_index_prefix = mem_index_path + "_tempFiles";
    int         num_parts =
        partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget,
                                     shard_r, merged_index_prefix, 2);

    std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
    std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";

      std::string shard_ids_file = merged_index_prefix + "_subshard-" +
                                   std::to_string(p) + "_ids_uint32.bin";

      retrieve_shard_data_from_ids<T>(base_file, shard_ids_file,
                                      shard_base_file);

      std::string shard_index_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

      diskann::Parameters paras;
      paras.Set<unsigned>("L", L);
      paras.Set<unsigned>("R", shard_r);
      paras.Set<unsigned>("C", 750);
      paras.Set<float>("alpha", 1.2f);
      paras.Set<unsigned>("num_rnds", 2);
      paras.Set<bool>("saturate_graph", 0);
      paras.Set<std::string>("save_path", shard_index_file);
      paras.Set<bool>("accelerate_build", accelerate_build);
      paras.Set<bool>("shuffle_build", shuffle_build);

      _u64 shard_base_dim, shard_base_pts;
      bool built_with_gpu=false;
      get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
      std::unique_ptr<diskann::Index<T>> _pvamanaIndex =
          std::unique_ptr<diskann::Index<T>>(
              new diskann::Index<T>(compareMetric, ip_prepared, shard_base_dim,
              shard_base_pts, false));  // TODO: Single?
#ifdef KNOWHERE_WITH_CUVS
      //currently cuvs vamana build only supports L2Expanded metric
      if (compareMetric == diskann::L2 && is_gpu_available () &&
              (std::is_same_v<T, float> || std::is_same_v<T, uint8_t>)) {
        LOG_KNOWHERE_INFO_ << "Building with GPU!" << " R= "<< shard_r <<" L=" << L;
        if (std::is_same_v<T, float> ) {
          auto dataset = read_bin_dataset<float, uint64_t>(dev_resources, shard_base_file);
          vamana_build_and_write<float>(dev_resources,
                                       raft::make_const_mdspan(dataset.view()),
                                       shard_index_file,
                                       shard_r,
                                       L,
                                       0.06,
                                       1);
        }else {
          auto dataset = read_bin_dataset<uint8_t, uint64_t>(dev_resources, shard_base_file);
          vamana_build_and_write<uint8_t>(dev_resources,
                                       raft::make_const_mdspan(dataset.view()),
                                       shard_index_file,
                                       shard_r,
                                       L,
                                       0.06,
                                       1);
        }
        built_with_gpu = true;
      }
#endif
      if(!built_with_gpu){
        _pvamanaIndex->build(shard_base_file.c_str(), shard_base_pts, paras);
        _pvamanaIndex->save(shard_index_file.c_str());
      }
      std::remove(shard_base_file.c_str());
    }

    diskann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index",
                          merged_index_prefix + "_subshard-", "_ids_uint32.bin",
                          num_parts, R, mem_index_path, medoids_file);

    // delete tempFiles
    for (int p = 0; p < num_parts; p++) {
      std::string shard_base_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
      std::string shard_id_file = merged_index_prefix + "_subshard-" +
                                  std::to_string(p) + "_ids_uint32.bin";
      std::string shard_index_file =
          merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
      std::string shard_index_file_data = shard_index_file + ".data";

      std::remove(shard_base_file.c_str());
      std::remove(shard_id_file.c_str());
      std::remove(shard_index_file.c_str());
      std::remove(shard_index_file_data.c_str());
    }
    std::remove(medoids_file.c_str());
    std::remove(centroids_file.c_str());
    if (get_file_size(mem_index_path) < ram_budget * 1024 * 1024 * 1024) {
      auto total_vamana_index =
          std::unique_ptr<diskann::Index<T>>(new diskann::Index<T>(
              compareMetric, base_dim, base_num, false, false));
      total_vamana_index->load_graph(mem_index_path, base_num);
      return total_vamana_index;
    }
    return nullptr;
  }

  template<typename T>
  void generate_cache_list_from_graph_with_pq(
      _u64 num_nodes_to_cache, unsigned R, const diskann::Metric compare_metric,
      const std::string &sample_file, const std::string &pq_pivots_path,
      const std::string &pq_compressed_code_path, const unsigned entry_point,
      const std::vector<std::vector<unsigned>> &graph,
      const std::string                        &cache_file) {
    if (num_nodes_to_cache <= 0) {
      LOG_KNOWHERE_INFO_
          << "The number of cache nodes <= 0, no need to generate cache files";
      return;
    }
    if (compare_metric == diskann::Metric::INNER_PRODUCT &&
        !std::is_same_v<T, float>) {
      LOG_KNOWHERE_ERROR_ << "Inner product only support float type in diskann";
      return;
    }

    _u64 sample_num, sample_dim;
    std::unique_ptr<T[]> samples;
    if (file_exists(sample_file)) {
      diskann::load_bin<T>(sample_file, samples, sample_num, sample_dim);
    } else {
      LOG_KNOWHERE_ERROR_ << "Sample bin file not found. Not generating cache.";
      return;
    }

    auto thread_pool = knowhere::ThreadPool::GetGlobalBuildThreadPool();

    auto points_num = graph.size();
    if (num_nodes_to_cache >= points_num) {
      LOG_KNOWHERE_INFO_
          << "The number of cache nodes is greater than the total number of "
             "nodes, adjust the number of cache nodes from "
          << num_nodes_to_cache << " to " << points_num;
      num_nodes_to_cache = points_num;
    }

    std::unique_ptr<uint8_t[]> pq_code;
    diskann::FixedChunkPQTable pq_table;
    uint64_t                   pq_chunks, pq_npts = 0;
    if (file_exists(pq_pivots_path) && file_exists(pq_compressed_code_path)) {
      diskann::load_bin<_u8>(pq_compressed_code_path, pq_code, pq_npts,
                             pq_chunks);
      pq_table.load_pq_centroid_bin(pq_pivots_path.c_str(), pq_chunks);
    } else {
      LOG_KNOWHERE_ERROR_
          << "PQ pivots and compressed code not found. Not generating cache.";
      return;
    }
    LOG_KNOWHERE_INFO_ << "Use " << sample_num << " sampled quries to generate "
                       << num_nodes_to_cache << " cached nodes.";
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(sample_num);

    std::vector<std::pair<uint32_t, uint32_t>> node_count_list(points_num);
    for (size_t node_id = 0; node_id < points_num; node_id++) {
      node_count_list[node_id] = std::pair<uint32_t, uint32_t>(node_id, 0);
    }

    for (_s64 i = 0; i < (int64_t) sample_num; i++) {
      futures.push_back(thread_pool->push([&, index = i]() {
        // search params
        auto search_l = kSearchLForCache;

        // preprocess queries
        auto query_dim = sample_dim;
        auto old_dim = query_dim;
        if (compare_metric == diskann::INNER_PRODUCT) {
          query_dim++;
        }
        auto aligned_dim = ROUND_UP(query_dim, 8);

        auto   query_float = std::make_unique<float[]>(aligned_dim);
        double query_norm_dw = 0.0;
        for (uint32_t d = 0; d < old_dim; d++) {
          query_float[d] = static_cast<float>(samples[index * old_dim + d]);
          query_norm_dw += query_float[d] * query_float[d];
        }

        if (compare_metric == diskann::INNER_PRODUCT) {
          if (query_norm_dw == 0)
            return;
          query_float[query_dim - 1] = 0;
          auto query_norm = float(std::sqrt(query_norm_dw));
          for (uint32_t d = 0; d < old_dim; d++) {
            query_float[d] /= query_norm;
          }
        }

        // prepare pq table and pq code
        auto pq_table_dists =
            std::shared_ptr<float[]>(new float[256 * aligned_dim]);
        auto scratch_dists = std::shared_ptr<float[]>(new float[R]);
        auto scratch_ids = std::shared_ptr<_u8[]>(new _u8[R * aligned_dim]);
        pq_table.populate_chunk_distances(query_float.get(),
                                          pq_table_dists.get());

        auto compute_dists = [&, scratch_ids, pq_table_dists](
                                 const unsigned *ids, const _u64 n_ids,
                                 float *dists_out) {
          aggregate_coords(ids, n_ids, pq_code.get(), pq_chunks,
                           scratch_ids.get());
          pq_dist_lookup(scratch_ids.get(), n_ids, pq_chunks,
                         pq_table_dists.get(), dists_out);
        };

        // init search list and search graph
        auto retset = std::vector<Neighbor>(search_l * 2);
        auto visited = boost::dynamic_bitset<>{points_num, 0};

        compute_dists(&entry_point, 1, scratch_dists.get());
        retset[0].id = entry_point;
        retset[0].flag = true;
        retset[0].distance = scratch_dists[0];
        visited[entry_point] = true;
        unsigned cur_list_size = 1;
        unsigned k = 0;

        while (k < cur_list_size) {
          auto nk = cur_list_size;

          if (retset[k].flag) {
            auto target_id = retset[k].id;
            if (node_count_list.size() != 0) {
              reinterpret_cast<std::atomic<_u32> &>(
                  node_count_list[target_id].second)
                  .fetch_add(1);
            }
            _u64 neighbor_num = graph[target_id].size();
            compute_dists(graph[target_id].data(), neighbor_num,
                          scratch_dists.get());

            for (size_t m = 0; m < neighbor_num; m++) {
              auto id = graph[target_id][m];
              if (visited[id]) {
                continue;
              } else {
                visited[id] = true;
                float dist = scratch_dists[m];
                if (cur_list_size > 0 &&
                    dist >= retset[cur_list_size - 1].distance &&
                    (cur_list_size == L_SET))
                  continue;
                Neighbor nn(id, dist, true);
                auto     r = InsertIntoPool(retset.data(), cur_list_size, nn);
                if (cur_list_size < search_l)
                  ++cur_list_size;
                if (r < nk)
                  nk = r;
              }
            }
            if (nk <= k)
              k = nk;
            else
              ++k;
          } else {
            ++k;
          }
        }
      }));
    }

    knowhere::WaitAllSuccess(futures);

    std::sort(node_count_list.begin(), node_count_list.end(),
              [](std::pair<_u32, _u32> &a, std::pair<_u32, _u32> &b) {
                return a.second > b.second;
              });

    std::vector<uint32_t> node_list(num_nodes_to_cache);
    for (_u64 node_i = 0; node_i < num_nodes_to_cache; node_i++) {
      node_list[node_i] = node_count_list[node_i].first;
    }

    save_bin<uint32_t>(cache_file, node_list.data(), num_nodes_to_cache, 1);
  }

  // General purpose support for DiskANN interface

  // optimizes the beamwidth to maximize QPS for a given L_search subject to
  // 99.9 latency not blowing up
  template<typename T>
  uint32_t optimize_beamwidth(
      std::unique_ptr<diskann::PQFlashIndex<T>> &pFlashIndex, T *tuning_sample,
      _u64 tuning_sample_num, _u64 tuning_sample_aligned_dim, uint32_t L,
      uint32_t nthreads, uint32_t start_bw) {
    uint32_t cur_bw = start_bw;
    double   max_qps = 0;
    uint32_t best_bw = start_bw;
    bool     stop_flag = false;

    auto thread_pool = knowhere::ThreadPool::GetGlobalBuildThreadPool();

    while (!stop_flag) {
      std::vector<int64_t> tuning_sample_result_ids_64(tuning_sample_num, 0);
      std::vector<float>   tuning_sample_result_dists(tuning_sample_num, 0);
      auto stats = std::make_unique<diskann::QueryStats[]>(tuning_sample_num);

      std::vector<folly::Future<folly::Unit>> futures;
      futures.reserve(tuning_sample_num);
      auto s = std::chrono::high_resolution_clock::now();
      for (_s64 i = 0; i < (int64_t) tuning_sample_num; i++) {
        futures.emplace_back(thread_pool->push([&, index = i]() {
          pFlashIndex->cached_beam_search(
              tuning_sample + (index * tuning_sample_aligned_dim), 1, L,
              tuning_sample_result_ids_64.data() + (index * 1),
              tuning_sample_result_dists.data() + (index * 1), cur_bw, false,
              stats.get() + index);
        }));
      }
      knowhere::WaitAllSuccess(futures);
      auto e = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = e - s;
      double                        qps =
          (1.0f * (float) tuning_sample_num) / (1.0f * (float) diff.count());

      double lat_999 = diskann::get_percentile_stats<float>(
          stats.get(), tuning_sample_num, 0.999f,
          [](const diskann::QueryStats &stats) { return stats.total_us; });

      double mean_latency = diskann::get_mean_stats<float>(
          stats.get(), tuning_sample_num,
          [](const diskann::QueryStats &stats) { return stats.total_us; });

      if (qps > max_qps && lat_999 < (15000) + mean_latency * 2) {
        max_qps = qps;
        best_bw = cur_bw;
        cur_bw = (uint32_t) (std::ceil)((float) cur_bw * 1.1f);
      } else {
        stop_flag = true;
      }
      if (cur_bw > 64)
        stop_flag = true;
    }
    return best_bw;
  }

  template<typename T>
  void create_disk_layout(const std::string base_file,
                          const std::string mem_index_file,
                          const std::string output_file,
                          const std::string reorder_data_file) {
    unsigned npts, ndims;

    // amount to read or write in one shot
    _u64            read_blk_size = 64 * 1024 * 1024;
    _u64            write_blk_size = read_blk_size;
    cached_ifstream base_reader(base_file, read_blk_size);
    base_reader.read((char *) &npts, sizeof(uint32_t));
    base_reader.read((char *) &ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // Check if we need to append data for re-ordering
    bool          append_reorder_data = false;
    std::ifstream reorder_data_reader;

    unsigned npts_reorder_file = 0, ndims_reorder_file = 0;
    if (reorder_data_file != std::string("")) {
      append_reorder_data = true;
      size_t reorder_data_file_size = get_file_size(reorder_data_file);
      reorder_data_reader.exceptions(std::ofstream::failbit |
                                     std::ofstream::badbit);

      try {
        reorder_data_reader.open(reorder_data_file, std::ios::binary);
        reorder_data_reader.read((char *) &npts_reorder_file, sizeof(unsigned));
        reorder_data_reader.read((char *) &ndims_reorder_file,
                                 sizeof(unsigned));
        if (npts_reorder_file != npts)
          throw ANNException(
              "Mismatch in num_points between reorder data file and base file",
              -1, __FUNCSIG__, __FILE__, __LINE__);
        if (reorder_data_file_size != 8 + sizeof(float) *
                                              (size_t) npts_reorder_file *
                                              (size_t) ndims_reorder_file)
          throw ANNException("Discrepancy in reorder data file size ", -1,
                             __FUNCSIG__, __FILE__, __LINE__);
      } catch (std::system_error &e) {
        throw FileException(reorder_data_file, e, __FUNCSIG__, __FILE__,
                            __LINE__);
      }
    }

    // create cached reader + writer
    size_t actual_file_size = get_file_size(mem_index_file);
    LOG_KNOWHERE_INFO_ << "Vamana index file size: " << actual_file_size;
    std::ifstream   vamana_reader(mem_index_file, std::ios::binary);
    cached_ofstream diskann_writer(output_file, write_blk_size);

    // metadata: width, medoid
    unsigned width_u32, medoid_u32;
    size_t   index_file_size;

    vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));
    if (index_file_size != actual_file_size) {
      std::stringstream stream;
      stream << "Vamana Index file size does not match expected size per "
                "meta-data."
             << " file size from file: " << index_file_size
             << " actual file size: " << actual_file_size << std::endl;

      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    _u64 vamana_frozen_num = false, vamana_frozen_loc = 0;

    vamana_reader.read((char *) &width_u32, sizeof(unsigned));
    vamana_reader.read((char *) &medoid_u32, sizeof(unsigned));
    vamana_reader.read((char *) &vamana_frozen_num, sizeof(_u64));
    // compute
    _u64 medoid, max_node_len;
    _u64 nsector_per_node;
    _u64 nnodes_per_sector;
    npts_64 = (_u64) npts;
    medoid = (_u64) medoid_u32;
    if (vamana_frozen_num == 1)
      vamana_frozen_loc = medoid;
    max_node_len =
        (((_u64) width_u32 + 1) * sizeof(unsigned)) + (ndims_64 * sizeof(T));

    bool long_node = max_node_len > diskann::defaults::SECTOR_LEN;
    if (long_node) {
      if (append_reorder_data) {
        throw diskann::ANNException(
            "Reorder data for long node is not supported.", -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }
      nsector_per_node = ROUND_UP(max_node_len, diskann::defaults::SECTOR_LEN) / diskann::defaults::SECTOR_LEN;
      nnodes_per_sector = -1;
      LOG_KNOWHERE_DEBUG_ << "medoid: " << medoid << "B"
                          << "max_node_len: " << max_node_len << "B"
                          << "nsector_per_node: " << nsector_per_node << "B";
    } else {
      nnodes_per_sector = diskann::defaults::SECTOR_LEN / max_node_len;
      nsector_per_node = -1;
      LOG_KNOWHERE_DEBUG_ << "medoid: " << medoid << "B"
                          << "max_node_len: " << max_node_len << "B"
                          << "nnodes_per_sector: " << nnodes_per_sector << "B";
    }

    // number of sectors (1 for meta data)
    _u64 n_sectors =
        long_node ? nsector_per_node * npts_64
                  : ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector;
    _u64 n_reorder_sectors = 0;
    _u64 n_data_nodes_per_sector = 0;

    if (append_reorder_data) {
      n_data_nodes_per_sector =
          diskann::defaults::SECTOR_LEN / (ndims_reorder_file * sizeof(float));
      n_reorder_sectors =
          ROUND_UP(npts_64, n_data_nodes_per_sector) / n_data_nodes_per_sector;
    }
    _u64 disk_index_file_size =
        (n_sectors + n_reorder_sectors + 1) * diskann::defaults::SECTOR_LEN;

    // SECTOR_LEN buffer for each sector
    _u64 sector_buf_size =
        long_node ? nsector_per_node * diskann::defaults::SECTOR_LEN : diskann::defaults::SECTOR_LEN;
    std::unique_ptr<char[]> sector_buf =
        std::make_unique<char[]>(sector_buf_size);

    // write first sector with metadata
    *(_u64 *) (sector_buf.get() + 0 * sizeof(_u64)) = disk_index_file_size;
    *(_u64 *) (sector_buf.get() + 1 * sizeof(_u64)) = npts_64;
    *(_u64 *) (sector_buf.get() + 2 * sizeof(_u64)) = medoid;
    *(_u64 *) (sector_buf.get() + 3 * sizeof(_u64)) = max_node_len;
    *(_u64 *) (sector_buf.get() + 4 * sizeof(_u64)) = nnodes_per_sector;
    *(_u64 *) (sector_buf.get() + 5 * sizeof(_u64)) = vamana_frozen_num;
    *(_u64 *) (sector_buf.get() + 6 * sizeof(_u64)) = vamana_frozen_loc;
    *(_u64 *) (sector_buf.get() + 7 * sizeof(_u64)) = append_reorder_data;
    if (append_reorder_data) {
      *(_u64 *) (sector_buf.get() + 8 * sizeof(_u64)) = n_sectors + 1;
      *(_u64 *) (sector_buf.get() + 9 * sizeof(_u64)) = ndims_reorder_file;
      *(_u64 *) (sector_buf.get() + 10 * sizeof(_u64)) =
          n_data_nodes_per_sector;
    }

    diskann_writer.write(sector_buf.get(), diskann::defaults::SECTOR_LEN);

    if (long_node) {
      for (_u64 node_id = 0; node_id < npts_64; ++node_id) {
        memset(sector_buf.get(), 0, sector_buf_size);
        char *nnbrs = sector_buf.get() + ndims_64 * sizeof(T);
        char *nhood_buf =
            sector_buf.get() + (ndims_64 * sizeof(T)) + sizeof(unsigned);

        // read cur node's nnbrs
        vamana_reader.read(nnbrs, sizeof(unsigned));

        // sanity checks on nnbrs
        assert(static_cast<uint32_t>(*nnbrs) > 0);
        assert(static_cast<uint32_t>(*nnbrs) <= width_u32);

        // read node's nhood
        vamana_reader.read(nhood_buf, *((unsigned *) nnbrs) * sizeof(unsigned));

        // write coords of node first
        base_reader.read((char *) sector_buf.get(), sizeof(T) * ndims_64);

        diskann_writer.write(sector_buf.get(), sector_buf_size);
      }
      LOG_KNOWHERE_DEBUG_ << "Output file written.";
      return;
    }

    LOG_KNOWHERE_DEBUG_ << "# sectors: " << n_sectors;
    _u64 cur_node_id = 0;
    for (_u64 sector = 0; sector < n_sectors; sector++) {
      if (sector % 100000 == 0) {
        LOG_KNOWHERE_DEBUG_ << "Sector #" << sector << "written";
      }
      memset(sector_buf.get(), 0, diskann::defaults::SECTOR_LEN);
      for (_u64 sector_node_id = 0;
           sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
           sector_node_id++) {
        char *sector_node_buf =
            sector_buf.get() + (sector_node_id * max_node_len);
        char *nnbrs = sector_node_buf + ndims_64 * sizeof(T);
        char *nhood_buf =
            sector_node_buf + (ndims_64 * sizeof(T)) + sizeof(unsigned);

        // read cur node's nnbrs
        vamana_reader.read(nnbrs, sizeof(unsigned));

        // sanity checks on nnbrs
        assert(static_cast<uint32_t>(*nnbrs) > 0);
        assert(static_cast<uint32_t>(*nnbrs) <= width_u32);

        // read node's nhood
        vamana_reader.read(nhood_buf, *((unsigned *) nnbrs) * sizeof(unsigned));

        // write coords of node first
        base_reader.read(sector_node_buf, sizeof(T) * ndims_64);

        cur_node_id++;
      }
      // flush sector to disk
      diskann_writer.write(sector_buf.get(), diskann::defaults::SECTOR_LEN);
    }
    if (append_reorder_data) {
      LOG_KNOWHERE_INFO_ << "Index written. Appending reorder data...";

      auto                    vec_len = ndims_reorder_file * sizeof(float);
      std::unique_ptr<char[]> vec_buf = std::make_unique<char[]>(vec_len);

      for (_u64 sector = 0; sector < n_reorder_sectors; sector++) {
        if (sector % 100000 == 0) {
          LOG_KNOWHERE_INFO_ << "Reorder data Sector #" << sector << "written";
        }

        memset(sector_buf.get(), 0, diskann::defaults::SECTOR_LEN);

        for (_u64 sector_node_id = 0;
             sector_node_id < n_data_nodes_per_sector &&
             sector_node_id < npts_64;
             sector_node_id++) {
          memset(vec_buf.get(), 0, vec_len);
          reorder_data_reader.read(vec_buf.get(), vec_len);

          // copy node buf into sector_node_buf
          memcpy(sector_buf.get() + (sector_node_id * vec_len), vec_buf.get(),
                 vec_len);
        }
        // flush sector to disk
        diskann_writer.write(sector_buf.get(), diskann::defaults::SECTOR_LEN);
      }
    }
    LOG_KNOWHERE_DEBUG_ << "Output file written.";
  }

struct vamana_read_context {
    vamana_read_context(std::ifstream &vamana_reader, uint64_t *node_to_pos_map)
        : vamana_reader(vamana_reader), node_to_pos_map(node_to_pos_map) {
    }
    std::ifstream &vamana_reader;
    uint64_t *node_to_pos_map;
};

template <typename T, typename LabelT>
static std::vector<bool> read_node_nbrs_from_vamana(void *context, const std::vector<uint32_t> &node_ids,
                        std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers)
{
    struct vamana_read_context *ctx = reinterpret_cast<struct vamana_read_context *>(context);
    std::vector<bool> retval(node_ids.size(), false);
    for (uint32_t i = 0; i < node_ids.size(); i++) {
        auto node_id = node_ids[i];
        ctx->vamana_reader.seekg(ctx->node_to_pos_map[node_id], ctx->vamana_reader.beg);
        ctx->vamana_reader.read((char *)&nbr_buffers[i].first, sizeof(uint32_t));
        ctx->vamana_reader.read((char *)nbr_buffers[i].second, nbr_buffers[i].first * sizeof(uint32_t));
        retval[i] = true;
    }
    return retval;
}

template std::vector<bool> read_node_nbrs_from_vamana<float, uint32_t>(void *, const std::vector<uint32_t> &,
                            std::vector<std::pair<uint32_t, uint32_t *>> &);
template std::vector<bool> read_node_nbrs_from_vamana<int8_t, uint32_t>(void *, const std::vector<uint32_t> &,
                            std::vector<std::pair<uint32_t, uint32_t *>> &);
template std::vector<bool> read_node_nbrs_from_vamana<uint8_t, uint32_t>(void *, const std::vector<uint32_t> &,
                            std::vector<std::pair<uint32_t, uint32_t *>> &);

template <typename T>
void aisaq_calc_inline_layout(int inline_pq, uint32_t pq_compressed_nbytes, uint32_t max_degree, bool &rearrange,
                              uint32_t &inline_pq_vectors, uint64_t &max_node_len)
{
    if (inline_pq >= 0) {
        if (inline_pq > 0) {
            assert((uint32_t)inline_pq <= max_degree);
            inline_pq_vectors = inline_pq;
            if (rearrange && inline_pq_vectors < max_degree) {
                max_node_len+= sizeof(uint32_t);
            }
        } else {
            /* calculate the number of compressed vectors that can be appended to the node without increasing the index file size */
            uint32_t _n_inline = aisaq_calc_max_inline_pq_vectors(max_node_len, pq_compressed_nbytes, max_degree);
            if (rearrange && _n_inline < max_degree) {
                /* calc with rearrange */
                max_node_len+= sizeof(uint32_t);
                _n_inline = aisaq_calc_max_inline_pq_vectors(max_node_len, pq_compressed_nbytes, max_degree);
            }
            inline_pq_vectors = _n_inline;
        }
        max_node_len+= inline_pq_vectors * pq_compressed_nbytes;
        if (inline_pq_vectors == max_degree) {
            if (rearrange) {
                /* ignore reaarange */
                LOG_KNOWHERE_INFO_ << "all pq vectors will be stored inline, ignoring rearrange";
                rearrange = false;
            } else {
                LOG_KNOWHERE_INFO_ << "all pq vectors will be stored inline";
            }
        } else {
            LOG_KNOWHERE_INFO_ << inline_pq_vectors << " (" << (inline_pq_vectors * 100) / max_degree
                          << "%) PQ vectors will be stored inline";
        }
    } else {
        inline_pq_vectors = 0;
        if (rearrange) {
            max_node_len+= sizeof(uint32_t);
        }
    }
}

template <typename T>
void create_aisaq_layout(const std::string base_file, const std::string mem_index_file, const std::string output_file,
                        const std::string reorder_data_file,
                        const std::string &index_prefix_path,
                        const diskann::Metric metric,
                        int inline_pq /* control num of inline pq: -1=none, 0=auto, others: num of pq vectors <= R */,
                        bool &rearrange /* enable vectors rearangement */)
{
    uint32_t npts, ndims;

    // amount to read or write in one shot
    size_t read_blk_size = 64 * 1024 * 1024;
    size_t write_blk_size = read_blk_size;
    cached_ifstream base_reader(base_file, read_blk_size);
    base_reader.read((char *)&npts, sizeof(uint32_t));
    base_reader.read((char *)&ndims, sizeof(uint32_t));

    size_t npts_64, ndims_64;
    npts_64 = npts;
    ndims_64 = ndims;

    // Check if we need to append data for re-ordering
    bool append_reorder_data = false;
    std::ifstream reorder_data_reader;

    uint32_t npts_reorder_file = 0, ndims_reorder_file = 0;
    if (reorder_data_file != std::string(""))
    {
        append_reorder_data = true;
        size_t reorder_data_file_size = get_file_size(reorder_data_file);
        reorder_data_reader.exceptions(std::ofstream::failbit | std::ofstream::badbit);

        try
        {
            reorder_data_reader.open(reorder_data_file, std::ios::binary);
            reorder_data_reader.read((char *)&npts_reorder_file, sizeof(uint32_t));
            reorder_data_reader.read((char *)&ndims_reorder_file, sizeof(uint32_t));
            if (npts_reorder_file != npts)
                throw ANNException("Mismatch in num_points between reorder "
                                   "data file and base file",
                                   -1, __FUNCSIG__, __FILE__, __LINE__);
            if (reorder_data_file_size != 8 + sizeof(float) * (size_t)npts_reorder_file * (size_t)ndims_reorder_file)
                throw ANNException("Discrepancy in reorder data file size ", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        catch (std::system_error &e)
        {
            throw FileException(reorder_data_file, e, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    // create cached reader + writer
    size_t actual_file_size = get_file_size(mem_index_file);
    LOG_KNOWHERE_INFO_ << "Vamana index file size=" << actual_file_size;
    std::ifstream vamana_reader(mem_index_file, std::ios::binary);
    cached_ofstream diskann_writer(output_file, write_blk_size);

    // metadata: width, medoid
    uint32_t width_u32, medoid_u32;
    size_t index_file_size;

    vamana_reader.read((char *)&index_file_size, sizeof(uint64_t));
    if (index_file_size != actual_file_size)
    {
        std::stringstream stream;
        stream << "Vamana Index file size does not match expected size per "
                  "meta-data."
               << " file size from file: " << index_file_size << " actual file size: " << actual_file_size << std::endl;

        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    uint64_t vamana_frozen_num = false, vamana_frozen_loc = 0;

    vamana_reader.read((char *)&width_u32, sizeof(uint32_t));
    vamana_reader.read((char *)&medoid_u32, sizeof(uint32_t));
    vamana_reader.read((char *)&vamana_frozen_num, sizeof(uint64_t));
    // compute
    uint64_t medoid, max_node_len, nnodes_per_sector;
    npts_64 = (uint64_t)npts;
    max_node_len = (((uint64_t)width_u32 + 1) * sizeof(uint32_t)) + (ndims_64 * sizeof(T));

    /* open and validate compressed vectors file */
    std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";
    uint32_t pq_compressed_nbytes, pq_compressed_vectors_npts;
    std::ifstream pq_compressed_vectors_reader;
    size_t pq_compressed_vectors_file_size = get_file_size(pq_compressed_vectors_path);
    pq_compressed_vectors_reader.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    try {
        pq_compressed_vectors_reader.open(pq_compressed_vectors_path, std::ios::binary);
        pq_compressed_vectors_reader.read((char *)&pq_compressed_vectors_npts, sizeof(uint32_t));
        pq_compressed_vectors_reader.read((char *)&pq_compressed_nbytes, sizeof(uint32_t));
        if (pq_compressed_vectors_npts != npts) {
            throw ANNException("Mismatch in num_points between pq compressed vectors file and base file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        if (pq_compressed_vectors_file_size != 8 + (size_t)pq_compressed_nbytes * (size_t)pq_compressed_vectors_npts) {
            throw ANNException("Discrepancy in pq compressed vectors file size", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
    } catch (std::system_error &e) {
        throw FileException(pq_compressed_vectors_path, e, __FUNCSIG__, __FILE__, __LINE__);
    }

    /* calculate num of inline vectors */
    uint32_t inline_pq_vectors;
    aisaq_calc_inline_layout<T>(inline_pq, pq_compressed_nbytes, width_u32, rearrange,
                                inline_pq_vectors, max_node_len);
    

    uint32_t __nnodes, __nsectors, __remainder;
    if (max_node_len >= defaults::SECTOR_LEN) {
        __nnodes = 1;
        __nsectors = DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
    } else {
        __nnodes = defaults::SECTOR_LEN / max_node_len;
        __nsectors = 1;
    }
    __remainder = (defaults::SECTOR_LEN * __nsectors) - (__nnodes * max_node_len);
    LOG_KNOWHERE_DEBUG_ << "[ node: " << max_node_len << "B ] x " << __nnodes
                  << " + [ remainder: " << __remainder << "B ] --> [ "
                  << (defaults::SECTOR_LEN >> 10) << "KiB ] x " << __nsectors;
    double wasted_disk_space_pcnt = (__remainder * 100) / (defaults::SECTOR_LEN * __nsectors);
    if (wasted_disk_space_pcnt > 0) {
        LOG_KNOWHERE_INFO_ << wasted_disk_space_pcnt << "% wasted disk space, optimal node size may reduce wasted disk space, "
                         "node size is derived from disk-pq-bytes, max-degree and pq-inline parameters";
    }

    uint64_t *vamana_reader_node_to_pos_map = nullptr;
    uint32_t *rearranged_vectors_map = nullptr;
    uint32_t *reversed_rearranged_vectors_map = nullptr;
    if (rearrange) {
        /* with rearrange enabled, base reader is not reading the vectors sequentially.
           reduce the cache size to a single vector data aligned to sector size */
        base_reader.set_cache_size(ROUND_UP(sizeof(T) * ndims_64, defaults::SECTOR_LEN));
        /* generate vamana graph helper
           map node_id to its offset within vamana reader */
        vamana_reader_node_to_pos_map = new uint64_t[npts_64];
        if (vamana_reader_node_to_pos_map == nullptr) {
            throw ANNException("memory allocation failed"
                   , -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        /* save current position */
        auto pos = vamana_reader.tellg();
        uint64_t offset = pos;
        uint32_t nnbrs;
        vamana_reader_node_to_pos_map[0] = offset;
        for (uint64_t i = 1; i < npts_64; i++) {
            vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));
            assert(nnbrs > 0);
            assert(nnbrs <= width_u32);
            /* skip nhood */
            vamana_reader.seekg(nnbrs * sizeof(uint32_t), vamana_reader.cur);
            offset+= (nnbrs + 1) * sizeof(uint32_t);
            vamana_reader_node_to_pos_map[i] = offset;
        }
        
        std::string medoids_path = index_prefix_path + "_disk.index_medoids.bin";
        std::string entry_points_path = index_prefix_path + "_disk.index_entry_points.bin";
        std::unique_ptr <uint32_t[]> entry_points = nullptr;
        size_t n_entry_points, __dim;
        if (file_exists(entry_points_path)) {
            diskann::load_bin<uint32_t>(entry_points_path, entry_points, n_entry_points, __dim);
        } else {
            if (file_exists(medoids_path)) {
                diskann::load_bin<uint32_t>(medoids_path, entry_points, n_entry_points, __dim);
            } else {
                entry_points = std::make_unique < uint32_t[]>(1);
                entry_points[0] = medoid_u32;
                n_entry_points = 1;
            }
        }
        
        std::unordered_map<uint32_t, std::vector<uint32_t>> filter_to_medoid_ids;
        struct vamana_read_context context(vamana_reader, vamana_reader_node_to_pos_map);
        if (aisaq_generate_vectors_rearrange_map<T, uint32_t>(aisaq_rearrange_sorter_default, rearranged_vectors_map, (uint32_t)npts_64,
            pq_compressed_nbytes , width_u32, entry_points.get(), n_entry_points, filter_to_medoid_ids,
            read_node_nbrs_from_vamana<T, uint32_t>, &context) != 0) {
            throw ANNException("failed to generate rearranged vectors data"
                   , -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        /* restore last position */
        vamana_reader.seekg(pos, vamana_reader.beg);
        /* create reversed vectors map */
        if (aisaq_create_reversed_vectors_map(reversed_rearranged_vectors_map, rearranged_vectors_map, (uint32_t)npts_64) != 0) {
            throw ANNException("failed to generate reversed rearranged vectors map"
                   , -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        /* generate rearranged pq compressed vectors file instead of existing non-rearranged
           (unaligned for DiskANN to load into DRAM) */
        std::string pq_compressed_rearranged_vectors_path_tmp = pq_compressed_vectors_path + ".tmp";
        LOG_KNOWHERE_INFO_ << "generating pq compressed vectors file " << pq_compressed_vectors_path;
        cached_ofstream pq_compressed_rearranged_vectors_writer(pq_compressed_rearranged_vectors_path_tmp, write_blk_size);
        std::unique_ptr<char[]> pq_vector_buf = std::make_unique<char[]>(pq_compressed_nbytes);
        /* write the header */
        uint32_t num_points_u32 = (uint32_t)npts_64;
        pq_compressed_rearranged_vectors_writer.write((char *)&num_points_u32, sizeof(uint32_t));
        pq_compressed_rearranged_vectors_writer.write((char *)&pq_compressed_nbytes, sizeof(uint32_t));
        uint32_t progress_step = npts_64 / 100;
        for (uint32_t i = 0; i < npts_64; i++) {
            if ((i % progress_step) == 0) {
                diskann::cout << "." << std::flush;
            }
            uint32_t rid = reversed_rearranged_vectors_map[i];
            assert(rid < npts_64);
            pq_compressed_vectors_reader.seekg((sizeof(uint32_t) * 2) + ((uint64_t)rid * pq_compressed_nbytes),
                                               pq_compressed_vectors_reader.beg);
            pq_compressed_vectors_reader.read(pq_vector_buf.get(), pq_compressed_nbytes);
            pq_compressed_rearranged_vectors_writer.write(pq_vector_buf.get(), pq_compressed_nbytes);
        }
        LOG_KNOWHERE_INFO_ << "done";
        pq_compressed_rearranged_vectors_writer.close();
        pq_compressed_vectors_reader.close();
        delete_file(pq_compressed_vectors_path);
        rename(pq_compressed_rearranged_vectors_path_tmp.c_str(), pq_compressed_vectors_path.c_str());
        pq_compressed_vectors_reader.open(pq_compressed_vectors_path, std::ios::binary);

        /* create aligned pq compressed rearranged file */
        std::string rearranged_pq_compressed_vectors_path = index_prefix_path + "_pq_compressed_rearranged.bin";
        if (aisaq_create_aligned_rearranged_pq_compressed_vectors_file(pq_compressed_vectors_reader,
            rearranged_pq_compressed_vectors_path, AISAQ_REARRANGED_PQ_FILE_PAGE_SIZE_DEFAULT,
            nullptr /*reversed_rearranged_vectors_map*/, npts_64, pq_compressed_nbytes) != 0) {
            throw ANNException("failed to create aligned rearranged pq vectors file"
                   , -1, __FUNCSIG__, __FILE__, __LINE__);
        };
        
        /* create rearrange map that can be used by filter search
           rearrange map contains mapping from new_id --> origin_id */
        std::string rearrange_map_path = index_prefix_path + "_disk.index_rearrange.bin";
        diskann::save_bin<uint32_t>(rearrange_map_path, reversed_rearranged_vectors_map, npts_64, 1, 0);
        
        /* translate medoid */
        medoid_u32 = rearranged_vectors_map[medoid_u32];
        /* generate rearranged medoid file - translated inline */
        if (file_exists(medoids_path)) {
            LOG_KNOWHERE_INFO_ << "rearranging medoids file " << medoids_path;
            aisaq_rearrange_vectors_file(medoids_path, rearranged_vectors_map, npts_64);
        }
        /* generate rearranged entry points file - translated inline */
        if (file_exists(entry_points_path)) {
            LOG_KNOWHERE_INFO_ << "rearranging entry points file " << entry_points_path;
            aisaq_rearrange_vectors_file(entry_points_path, rearranged_vectors_map, npts_64);
        }
        /* rearrange norm file */
        if (metric == diskann::Metric::COSINE) {
            std::string norm_path =
                get_disk_index_max_base_norm_file(std::string(index_prefix_path + "_disk.index"));
            if (file_exists(norm_path)) {
                LOG_KNOWHERE_INFO_ << "rearranging normalization file " << norm_path;
                std::unique_ptr<float[]> norm_data = nullptr;
                size_t __npts, __sz;
                diskann::load_bin<float>(norm_path, norm_data, __npts, __sz);
                assert(npts_64 == __npts);
                float *rearranged_norm_data = nullptr;
                diskann::alloc_aligned(((void **) &rearranged_norm_data),
                           __npts * sizeof(float),
                           sizeof(float));
                for (unsigned int i = 0; i < __npts; i++) {
                    rearranged_norm_data[i] = norm_data[rearranged_vectors_map[i]];
                }
                diskann::save_bin<float>(norm_path, rearranged_norm_data, __npts, 1);  
                aligned_free((void*)rearranged_norm_data);
            }
        }
    }

    medoid = (uint64_t)medoid_u32;
    if (vamana_frozen_num == 1)
        vamana_frozen_loc = medoid;
    nnodes_per_sector = defaults::SECTOR_LEN / max_node_len; // 0 if max_node_len > SECTOR_LEN

    LOG_KNOWHERE_INFO_ << "medoid: " << medoid
                       << " max_node_len: " << max_node_len << "B"
                       << " nnodes_per_sector: " << nnodes_per_sector
                       << " inline_pq: " << inline_pq
                       << " rearrange: " << rearrange;

    // defaults::SECTOR_LEN buffer for each sector
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(defaults::SECTOR_LEN);
    std::unique_ptr<char[]> multisector_buf = std::make_unique<char[]>(ROUND_UP(max_node_len, defaults::SECTOR_LEN));
    std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
    uint32_t &nnbrs = *(uint32_t *)(node_buf.get() + (ndims_64 * sizeof(T)));
    uint32_t *nhood_buf = (uint32_t *)(node_buf.get() + (ndims_64 * sizeof(T)) + sizeof(uint32_t));

    // number of sectors (1 for meta data)
    uint64_t n_sectors = nnodes_per_sector > 0 ? ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector
                                               : npts_64 * DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
    uint64_t n_reorder_sectors = 0;
    uint64_t n_data_nodes_per_sector = 0;

    if (append_reorder_data)
    {
        n_data_nodes_per_sector = defaults::SECTOR_LEN / (ndims_reorder_file * sizeof(float));
        n_reorder_sectors = ROUND_UP(npts_64, n_data_nodes_per_sector) / n_data_nodes_per_sector;
    }
    uint64_t disk_index_file_size = (n_sectors + n_reorder_sectors + 1) * defaults::SECTOR_LEN;

    std::vector<uint64_t> output_file_meta;
    output_file_meta.push_back(npts_64);
    output_file_meta.push_back(ndims_64);
    output_file_meta.push_back(medoid);
    output_file_meta.push_back(max_node_len);
    output_file_meta.push_back(nnodes_per_sector);
    output_file_meta.push_back(vamana_frozen_num);
    output_file_meta.push_back(vamana_frozen_loc);
    output_file_meta.push_back((uint64_t)append_reorder_data);
    if (append_reorder_data)
    {
        output_file_meta.push_back(n_sectors + 1);
        output_file_meta.push_back(ndims_reorder_file);
        output_file_meta.push_back(n_data_nodes_per_sector);
    }
    output_file_meta.push_back(disk_index_file_size);

    /* update metadata with backward compatibility */
    uint64_t val = 0;
    if (inline_pq_vectors > 0) {
        val = width_u32;
    }
    output_file_meta.push_back(val);
    val = (uint64_t)rearrange;
    output_file_meta.push_back(val);

    diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);

    std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
    LOG_KNOWHERE_INFO_ << "# sectors: " << n_sectors;
    uint64_t cur_node_id = 0;

    if (nnodes_per_sector > 0)
    { // Write multiple nodes per sector
        for (uint64_t sector = 0; sector < n_sectors; sector++)
        {
            if (sector % 100000 == 0)
            {
                LOG_KNOWHERE_ERROR_ << "Sector #" << sector << "written";
            }
            memset(sector_buf.get(), 0, defaults::SECTOR_LEN);
            for (uint64_t sector_node_id = 0; sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
                 sector_node_id++)
            {
                memset(node_buf.get(), 0, max_node_len);
                uint32_t rid;
                if (rearrange) {
                    rid = reversed_rearranged_vectors_map[cur_node_id];
                    assert(rid < npts_64);
                    vamana_reader.seekg(vamana_reader_node_to_pos_map[rid],vamana_reader.beg);
                    vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));
                    assert(nnbrs > 0);
                    assert(nnbrs <= width_u32);
                    vamana_reader.read((char *)nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
                    for (uint32_t j = 0; j < nnbrs; j++) {
                        assert(nhood_buf[j] < npts_64);
                        nhood_buf[j] = rearranged_vectors_map[nhood_buf[j]];
                    }
                    base_reader.seek((sizeof(uint32_t) * 2) + (rid * sizeof(T) * ndims_64));
                    base_reader.read((char *)cur_node_coords.get(), sizeof(T) * ndims_64);
                } else {
                    // read cur node's nnbrs
                    vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));

                    // sanity checks on nnbrs
                    assert(nnbrs > 0);
                    assert(nnbrs <= width_u32);

                    // read node's nhood
                    vamana_reader.read((char *)nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
                    if (nnbrs > width_u32)
                    {
                        vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
                    }

                    // write coords of node first
                    //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
                    base_reader.read((char *)cur_node_coords.get(), sizeof(T) * ndims_64);
                }
                memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

                // write nnbrs
                *(uint32_t *)(node_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

                // write nhood next
                memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
                       (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

                if (rearrange) {
                    memcpy(node_buf.get() + (ndims_64 * sizeof(T)) + ((width_u32 + 1) * sizeof(uint32_t)),
                                &rid, sizeof(uint32_t));
                }
                // write compressed vectors
                if (inline_pq_vectors > 0) {
                    char *comp_vec_buf = (char *)(node_buf.get() + (ndims_64 * sizeof(T)) +
                                                         ((width_u32 + 1) * sizeof(uint32_t)));
                    if (rearrange) {
                        comp_vec_buf+= sizeof(uint32_t);
                    }
                    for (uint32_t cv = 0; cv < inline_pq_vectors && cv < nnbrs; cv++) {
                        /* read compressed vector nhood_buf[i] into comp_vec_buf */
                        /* note that when rearrange is enabled, pq_compressed_vectors_reader is already rearranged */
                        pq_compressed_vectors_reader.seekg((sizeof(uint32_t) * 2) + ((uint64_t)nhood_buf[cv] * pq_compressed_nbytes),
                                                           pq_compressed_vectors_reader.beg);
                        pq_compressed_vectors_reader.read(comp_vec_buf, pq_compressed_nbytes);
                        comp_vec_buf+= pq_compressed_nbytes;
                    }
                }
                // get offset into sector_buf
                char *sector_node_buf = sector_buf.get() + (sector_node_id * max_node_len);

                // copy node buf into sector_node_buf
                memcpy(sector_node_buf, node_buf.get(), max_node_len);
                cur_node_id++;
            }
            // flush sector to disk
            diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);
        }
    }
    else
    { // Write multi-sector nodes
        uint64_t nsectors_per_node = DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
        for (uint64_t i = 0; i < npts_64; i++)
        {
            if ((i * nsectors_per_node) % 100000 == 0)
            {
                LOG_KNOWHERE_ERROR_ << "Sector #" << i * nsectors_per_node << "written";
            }
            memset(multisector_buf.get(), 0, nsectors_per_node * defaults::SECTOR_LEN);

            memset(node_buf.get(), 0, max_node_len);
            uint32_t rid;
            if (rearrange) {
                rid = reversed_rearranged_vectors_map[i];
                assert(rid < npts_64);
                vamana_reader.seekg(vamana_reader_node_to_pos_map[rid],vamana_reader.beg);
                vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));
                assert(nnbrs > 0);
                assert(nnbrs <= width_u32);
                vamana_reader.read((char *)nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
                for (uint32_t j = 0; j < nnbrs; j++) {
                    assert(nhood_buf[j] < npts_64);
                    nhood_buf[j] = rearranged_vectors_map[nhood_buf[j]];
                }
                base_reader.seek((sizeof(uint32_t) * 2) + (rid * sizeof(T) * ndims_64));
                base_reader.read((char *)cur_node_coords.get(), sizeof(T) * ndims_64);
            } else {
                // read cur node's nnbrs
                vamana_reader.read((char *)&nnbrs, sizeof(uint32_t));

                // sanity checks on nnbrs
                assert(nnbrs > 0);
                assert(nnbrs <= width_u32);

                // read node's nhood
                vamana_reader.read((char *)nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
                if (nnbrs > width_u32)
                {
                    vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
                }

                // write coords of node first
                //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
                base_reader.read((char *)cur_node_coords.get(), sizeof(T) * ndims_64);
            }
            memcpy(multisector_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

            // write nnbrs
            *(uint32_t *)(multisector_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

            // write nhood next
            memcpy(multisector_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
                   (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
            if (rearrange) {
                memcpy(multisector_buf.get() + (ndims_64 * sizeof(T)) + ((width_u32 + 1) * sizeof(uint32_t)),
                            &rid, sizeof(uint32_t));
            }
            if (inline_pq_vectors > 0) {
                char *comp_vec_buf = (char *) (multisector_buf.get() + (ndims_64 * sizeof(T)) +
                                               ((width_u32 + 1) * sizeof(uint32_t)));
                if (rearrange) {
                    comp_vec_buf+= sizeof(uint32_t);
                }
                for (uint32_t cv = 0; cv < inline_pq_vectors && cv < nnbrs; cv++) {
                    /* copy compressed vector nhood_buf[cv] into comp_vec_buf */
                    /* note that when rearrange is enabled, pq_compressed_vectors_reader is already rearranged */
                    pq_compressed_vectors_reader.seekg(
                            (sizeof(uint32_t) * 2) + ((uint64_t)nhood_buf[cv] * pq_compressed_nbytes),
                            pq_compressed_vectors_reader.beg);
                    pq_compressed_vectors_reader.read(comp_vec_buf, pq_compressed_nbytes);
                    comp_vec_buf += pq_compressed_nbytes;
                }
            }
            // flush sector to disk
            diskann_writer.write(multisector_buf.get(), nsectors_per_node * defaults::SECTOR_LEN);
        }
    }

    if (vamana_reader_node_to_pos_map != nullptr) {
        delete [] vamana_reader_node_to_pos_map;
        vamana_reader_node_to_pos_map = nullptr;
    }
    if (rearranged_vectors_map != nullptr) {
        delete [] rearranged_vectors_map;
        rearranged_vectors_map = nullptr;
    }

    if (append_reorder_data)
    {
        LOG_KNOWHERE_INFO_ << "Index written. Appending reorder data...";

        auto vec_len = ndims_reorder_file * sizeof(float);
        std::unique_ptr<char[]> vec_buf = std::make_unique<char[]>(vec_len);
        cur_node_id = 0;
        for (uint64_t sector = 0; sector < n_reorder_sectors; sector++)
        {
            if (sector % 100000 == 0)
            {
                LOG_KNOWHERE_ERROR_ << "Reorder data Sector #" << sector << "written";
            }

            memset(sector_buf.get(), 0, defaults::SECTOR_LEN);

            for (uint64_t sector_node_id = 0; sector_node_id < n_data_nodes_per_sector && cur_node_id < npts_64;
                 sector_node_id++)
            {
                memset(vec_buf.get(), 0, vec_len);
                if (rearrange) {
                    uint32_t rid = reversed_rearranged_vectors_map[cur_node_id];
                    assert(rid < npts_64);
                    reorder_data_reader.seekg((sizeof(uint32_t) * 2) + (rid * vec_len), reorder_data_reader.beg);
                }
                reorder_data_reader.read(vec_buf.get(), vec_len);

                // copy node buf into sector_node_buf
                memcpy(sector_buf.get() + (sector_node_id * vec_len), vec_buf.get(), vec_len);
                cur_node_id++;
            }
            // flush sector to disk
            diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);
        }
    }
    diskann_writer.close();
    if (reversed_rearranged_vectors_map != nullptr) {
        delete [] reversed_rearranged_vectors_map;
        reversed_rearranged_vectors_map = nullptr;
    }
    diskann::save_bin<uint64_t>(output_file, output_file_meta.data(), output_file_meta.size(), 1);
    LOG_KNOWHERE_ERROR_ << "Output disk index file written to " << output_file;
}

template<typename T>
  int build_disk_index(BuildConfig &config) {
    if (!knowhere::KnowhereFloatTypeCheck<T>::value &&
        (config.compare_metric == diskann::Metric::INNER_PRODUCT ||
         config.compare_metric == diskann::Metric::COSINE)) {
      std::stringstream stream;
      stream << "DiskANN currently only supports floating point data for Max "
                "Inner Product Search and Min Cosine Search."
             << std::endl;
      throw diskann::ANNException(stream.str(), -1);
    }

    _u32 disk_pq_dims = config.disk_pq_dims;
    bool use_disk_pq = disk_pq_dims != 0;

    bool reorder_data = config.reorder;
    bool ip_prepared = false;

    std::string base_file = config.data_file_path;
    std::string data_file_to_use = base_file;
    std::string data_file_to_save = base_file;
    std::string index_prefix_path = config.index_file_path;
    std::string pq_pivots_path = get_pq_pivots_filename(index_prefix_path);
    std::string pq_compressed_vectors_path =
        get_pq_compressed_filename(index_prefix_path);
    std::string mem_index_path = index_prefix_path + "_mem.index";
    std::string disk_index_path = get_disk_index_filename(index_prefix_path);
    std::string medoids_path = get_disk_index_medoids_filename(disk_index_path);
    std::string centroids_path =
        get_disk_index_centroids_filename(disk_index_path);
    std::string sample_data_file = get_sample_data_filename(index_prefix_path);
    // optional, used if disk index file must store pq data
    std::string disk_pq_pivots_path =
        index_prefix_path + "_disk.index_pq_pivots.bin";
    // optional, used if disk index must store pq data
    std::string disk_pq_compressed_vectors_path =
        index_prefix_path + "_disk.index_pq_compressed.bin";
    // optional, used if build mem usage is enough to generate cached nodes
    std::string cached_nodes_file = get_cached_nodes_file(index_prefix_path);

    // output a new base file which contains extra dimension with sqrt(1 -
    // ||x||^2/M^2) for every x, M is max norm of all points. Extra space on
    // disk needed!
    if (config.compare_metric == diskann::Metric::INNER_PRODUCT) {
      LOG_KNOWHERE_INFO_
          << "Using Inner Product search, so need to pre-process base "
             "data into temp file. Please ensure there is additional "
             "(n*(d+1)*4) bytes for storing pre-processed base vectors, "
             "apart from the intermin indices and final index.";
      std::string prepped_base = index_prefix_path + "_prepped_base.bin";
      data_file_to_use = prepped_base;
      data_file_to_save = prepped_base;
      float max_norm_of_base =
          diskann::prepare_base_for_inner_products<T>(base_file, prepped_base);
      std::string norm_file =
          get_disk_index_max_base_norm_file(disk_index_path);
      diskann::save_bin<float>(norm_file, &max_norm_of_base, 1, 1);
      ip_prepared = true;
    }
    if (config.compare_metric == diskann::Metric::COSINE) {
      LOG_KNOWHERE_INFO_
          << "Using Cosine search, so need to pre-process base "
             "data into temp file. Please ensure there is additional "
             "(n*d*4) bytes for storing pre-processed base vectors, "
             "apart from the intermin indices and final index.";
      std::string prepped_base = index_prefix_path + "_prepped_base.bin";
      data_file_to_use = prepped_base;
      auto norms_of_base =
          diskann::prepare_base_for_cosine<T>(base_file, prepped_base);
      std::string norm_file =
          get_disk_index_max_base_norm_file(disk_index_path);
      diskann::save_bin<float>(norm_file, norms_of_base.data(),
                               norms_of_base.size(), 1);
    }

    unsigned R = config.max_degree;
    unsigned L = config.search_list_size;

    double pq_code_size_limit = get_memory_budget(config.pq_code_size_gb);
    if (pq_code_size_limit <= 0) {
      LOG(ERROR) << "Insufficient memory budget (or string was not in right "
                    "format). Should be > 0.";
      return -1;
    }
    double indexing_ram_budget = config.index_mem_gb;
    if (indexing_ram_budget <= 0) {
      LOG(ERROR) << "Not building index. Please provide more RAM budget";
      return -1;
    }
#ifdef KNOWHERE_WITH_CUVS
    if(is_gpu_available()) {
      if (R != 32 && R != 64 && R != 128) {
        LOG_KNOWHERE_ERROR_ << "Invalid R value for cuvs - should be only 32 or 64 or 128";
        return -1;
      }
      if (L != 32 && L != 64 && L != 128 && L != 256) {
        LOG_KNOWHERE_ERROR_ << "Invalid L value for cuvs - should be only 32, 64, 128 or 256";
        return -1;
      }
      if (R >= L) {
        LOG_KNOWHERE_ERROR_ << "Invalid L value for cuvs - L must be > R";
        return -1;
      }
    }
#endif
    LOG_KNOWHERE_INFO_ << "Starting index build: R=" << R << " L=" << L
                       << " Query RAM budget: "
                       << pq_code_size_limit / (1024 * 1024 * 1024) << "(GiB)"
                       << " Indexing ram budget: " << indexing_ram_budget
                       << "(GiB)";

    auto s = std::chrono::high_resolution_clock::now();

    size_t points_num, dim;

    diskann::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);

    size_t num_pq_chunks =
        (size_t) (std::floor)(_u64(pq_code_size_limit / points_num));

    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > diskann::defaults::MAX_PQ_CHUNKS ? diskann::defaults::MAX_PQ_CHUNKS : num_pq_chunks;

    LOG_KNOWHERE_INFO_ << "Compressing " << dim << "-dimensional data into "
                       << num_pq_chunks << " bytes per vector.";

    size_t train_size, train_dim;
    std::unique_ptr<float[]> train_data = nullptr;

    double p_val = ((double) MAX_PQ_TRAINING_SET_SIZE / (double) points_num);
    // generates random sample and sets it to train_data and updates
    // train_size
    gen_random_slice<T>(data_file_to_use.c_str(), p_val, train_data, train_size,
                        train_dim);

    if (use_disk_pq) {
      if (disk_pq_dims > dim)
        disk_pq_dims = dim;

      LOG_KNOWHERE_DEBUG_ << "Compressing base for disk-PQ into "
                          << disk_pq_dims << " chunks ";
      generate_pq_pivots(train_data.get(), train_size, (uint32_t) dim, 256,
                         (uint32_t) disk_pq_dims, NUM_KMEANS_REPS,
                         disk_pq_pivots_path, false);
      if (config.compare_metric == diskann::Metric::INNER_PRODUCT ||
          config.compare_metric == diskann::Metric::COSINE)
        generate_pq_data_from_pivots<float>(
            data_file_to_use.c_str(), 256, (uint32_t) disk_pq_dims,
            disk_pq_pivots_path, disk_pq_compressed_vectors_path);
      else
        generate_pq_data_from_pivots<T>(
            data_file_to_use.c_str(), 256, (uint32_t) disk_pq_dims,
            disk_pq_pivots_path, disk_pq_compressed_vectors_path);
    }
    LOG_KNOWHERE_DEBUG_ << "Training data loaded of size " << train_size;

    // don't translate data to make zero mean for PQ compression. We must not
    // translate for inner product search.
    bool make_zero_mean = true;
    if (config.compare_metric != diskann::Metric::L2)
      make_zero_mean = false;

    auto pq_s = std::chrono::high_resolution_clock::now();

    LOG_KNOWHERE_INFO_ << "Generating PQ pivots";
    generate_pq_pivots(train_data.get(), train_size, (uint32_t) dim, 256,
                       (uint32_t) num_pq_chunks, NUM_KMEANS_REPS,
                       pq_pivots_path, make_zero_mean);

    LOG_KNOWHERE_INFO_ << "Encoding PQ data";
    generate_pq_data_from_pivots<T>(data_file_to_use.c_str(), 256,
                                    (uint32_t) num_pq_chunks, pq_pivots_path,
                                    pq_compressed_vectors_path);
    auto pq_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> pq_diff = pq_e - pq_s;
    LOG_KNOWHERE_INFO_ << "Training PQ codes cost: " << pq_diff.count() << "s";
// Gopal. Splitting diskann_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
    MallocExtension::instance()->ReleaseFreeMemory();
#endif

    auto graph_s = std::chrono::high_resolution_clock::now();
    auto vamana_index = diskann::build_merged_vamana_index<T>(
        data_file_to_use.c_str(), ip_prepared, diskann::Metric::L2, L, R,
        config.accelerate_build, config.shuffle_build, p_val, indexing_ram_budget, mem_index_path,
        medoids_path, centroids_path);
    auto graph_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> graph_diff = graph_e - graph_s;
    LOG_KNOWHERE_INFO_ << "Training graph cost: " << graph_diff.count() << "s";
    if (config.aisaq_mode) {
        bool rearrange = config.rearrange;
        int inline_pq = config.inline_pq;
        int num_entry_points = config.num_entry_points;
        LOG_KNOWHERE_INFO_ << "AiSAQ build mode enabled, inline pq is " << inline_pq
                << ", rearrange is " << rearrange << ", entry points: " << num_entry_points;
        if (num_entry_points > 0) {
            std::string entry_points_path = index_prefix_path + "_disk.index_entry_points.bin";
            LOG_KNOWHERE_INFO_ << "generating entry points file: " << entry_points_path;
            if (file_exists(entry_points_path)) {
                delete_file(entry_points_path);
            }
            if (partition_calc_kmeans<T>(data_file_to_use, entry_points_path, num_entry_points) != 0) {
                LOG_KNOWHERE_ERROR_ << "failed to generate entry points file";
                return -1;
            }
        }
        LOG_KNOWHERE_INFO_ << "Call create_aisaq_layout";
        if (!use_disk_pq) {
            diskann::create_aisaq_layout<T>(data_file_to_use.c_str(), mem_index_path, disk_index_path
                                       , std::string("")
                                       , index_prefix_path
                                       , config.compare_metric
                                       , inline_pq
                                       , rearrange
           );
        } else {
              LOG_KNOWHERE_INFO_ << "create_disk_layout use_disk_pq: " << use_disk_pq << " reorder_data: " << reorder_data;
          if (!reorder_data)
                diskann::create_aisaq_layout<_u8>(disk_pq_compressed_vectors_path
                                        , mem_index_path, disk_index_path, std::string("")
                                        , index_prefix_path
                                        , config.compare_metric
                                        , inline_pq
                                        , rearrange
                );
         else
                diskann::create_aisaq_layout<_u8>(disk_pq_compressed_vectors_path,
                                             mem_index_path, disk_index_path,
											 data_file_to_save.c_str()
                                            , index_prefix_path
                                            , config.compare_metric
                                            , inline_pq
                                            , rearrange
                );
        }
        config.rearrange = rearrange;
    }
    else
    {
        if (!use_disk_pq) {
          diskann::create_disk_layout<T>(data_file_to_save.c_str(), mem_index_path,
                                         disk_index_path);
        } else {
          if (!reorder_data)
            diskann::create_disk_layout<_u8>(disk_pq_compressed_vectors_path,
                                             mem_index_path, disk_index_path);
          else
            diskann::create_disk_layout<_u8>(disk_pq_compressed_vectors_path,
                                             mem_index_path, disk_index_path,
                                             data_file_to_save.c_str());
        }
    }
    double ten_percent_points = std::ceil(points_num * 0.1);
    double num_sample_points = ten_percent_points > MAX_SAMPLE_POINTS_FOR_WARMUP
                                   ? MAX_SAMPLE_POINTS_FOR_WARMUP
                                   : ten_percent_points;
    double sample_sampling_rate = num_sample_points / points_num;
    gen_random_slice<T>(base_file.c_str(), sample_data_file,
                        sample_sampling_rate);

    if (vamana_index != nullptr) {
      auto final_graph = vamana_index->get_graph();
      auto entry_point = vamana_index->get_entry_point();

      auto generate_cache_mem_usage =
          kCacheMemFactor *
          (get_file_size(mem_index_path) + get_file_size(sample_data_file) +
           get_file_size(pq_compressed_vectors_path) +
           get_file_size(pq_pivots_path)) /
          (1024 * 1024 * 1024);

      if (config.num_nodes_to_cache > 0 && final_graph->size() != 0 &&
          generate_cache_mem_usage < config.index_mem_gb) {
        generate_cache_list_from_graph_with_pq<T>(
            config.num_nodes_to_cache, config.max_degree, config.compare_metric,
            sample_data_file, pq_pivots_path, pq_compressed_vectors_path,
            entry_point, *final_graph, cached_nodes_file);
      }
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    LOG_KNOWHERE_INFO_ << "Indexing time: " << diff.count();

    if (config.compare_metric == diskann::Metric::INNER_PRODUCT) {
      std::remove(data_file_to_use.c_str());
    }
    std::remove(mem_index_path.c_str());
    if (use_disk_pq)
      std::remove(disk_pq_compressed_vectors_path.c_str());
    return 0;
  }

  template void create_disk_layout<int8_t>(const std::string base_file,
                                           const std::string mem_index_file,
                                           const std::string output_file,
                                           const std::string reorder_data_file);
  template void create_disk_layout<uint8_t>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file, const std::string reorder_data_file);
  template void create_disk_layout<float>(const std::string base_file,
                                          const std::string mem_index_file,
                                          const std::string output_file,
                                          const std::string reorder_data_file);
  template void create_disk_layout<knowhere::fp16>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file, const std::string reorder_data_file);
  template void create_disk_layout<knowhere::bf16>(
      const std::string base_file, const std::string mem_index_file,
      const std::string output_file, const std::string reorder_data_file);

  template int8_t  *load_warmup<int8_t>(const std::string &cache_warmup_file,
                                       uint64_t          &warmup_num,
                                       uint64_t           warmup_dim,
                                       uint64_t           warmup_aligned_dim);
  template uint8_t *load_warmup<uint8_t>(const std::string &cache_warmup_file,
                                         uint64_t          &warmup_num,
                                         uint64_t           warmup_dim,
                                         uint64_t           warmup_aligned_dim);
  template float   *load_warmup<float>(const std::string &cache_warmup_file,
                                     uint64_t &warmup_num, uint64_t warmup_dim,
                                     uint64_t warmup_aligned_dim);
  template knowhere::fp16 *load_warmup<knowhere::fp16>(
      const std::string &cache_warmup_file, uint64_t &warmup_num,
      uint64_t warmup_dim, uint64_t warmup_aligned_dim);
  template knowhere::bf16 *load_warmup<knowhere::bf16>(
      const std::string &cache_warmup_file, uint64_t &warmup_num,
      uint64_t warmup_dim, uint64_t warmup_aligned_dim);

  // knowhere not support uint8/int8 diskann
  // template uint32_t optimize_beamwidth<int8_t>(
  //     std::unique_ptr<diskann::PQFlashIndex<int8_t>> &pFlashIndex,
  //     int8_t *tuning_sample, _u64 tuning_sample_num,
  //     _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
  //     uint32_t start_bw);
  // template uint32_t optimize_beamwidth<uint8_t>(
  //     std::unique_ptr<diskann::PQFlashIndex<uint8_t>> &pFlashIndex,
  //     uint8_t *tuning_sample, _u64 tuning_sample_num,
  //     _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
  //     uint32_t start_bw);
  template uint32_t optimize_beamwidth<float>(
      std::unique_ptr<diskann::PQFlashIndex<float>> &pFlashIndex,
      float *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);
  template uint32_t optimize_beamwidth<knowhere::fp16>(
      std::unique_ptr<diskann::PQFlashIndex<knowhere::fp16>> &pFlashIndex,
      knowhere::fp16 *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);
  template uint32_t optimize_beamwidth<knowhere::bf16>(
      std::unique_ptr<diskann::PQFlashIndex<knowhere::bf16>> &pFlashIndex,
      knowhere::bf16 *tuning_sample, _u64 tuning_sample_num,
      _u64 tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
      uint32_t start_bw);

  // not support build uint8/int8 diskindex in knowhere
  // template int build_disk_index<int8_t>(BuildConfig &config);
  // template int build_disk_index<uint8_t>(BuildConfig &config);
  template int build_disk_index<float>(BuildConfig &config);
  template int build_disk_index<knowhere::fp16>(BuildConfig &config);
  template int build_disk_index<knowhere::bf16>(BuildConfig &config);

  template std::unique_ptr<diskann::Index<int8_t>>
  build_merged_vamana_index<int8_t>(
      std::string base_file, bool ip_prepared, diskann::Metric compareMetric,
      unsigned L, unsigned R, bool accelerate_build, bool shuffle_build,
      double sampling_rate, double ram_budget, std::string mem_index_path,
      std::string medoids_path, std::string centroids_file);
  template std::unique_ptr<diskann::Index<float>>
  build_merged_vamana_index<float>(
      std::string base_file, bool ip_prepared, diskann::Metric compareMetric,
      unsigned L, unsigned R, bool accelerate_build, bool shuffle_build,
      double sampling_rate, double ram_budget, std::string mem_index_path,
      std::string medoids_path, std::string centroids_file);
  template std::unique_ptr<diskann::Index<uint8_t>>
  build_merged_vamana_index<uint8_t>(
      std::string base_file, bool ip_prepared, diskann::Metric compareMetric,
      unsigned L, unsigned R, bool accelerate_build, bool shuffle_build,
      double sampling_rate, double ram_budget, std::string mem_index_path,
      std::string medoids_path, std::string centroids_file);
  template std::unique_ptr<diskann::Index<knowhere::fp16>>
  build_merged_vamana_index<knowhere::fp16>(
      std::string base_file, bool ip_prepared, diskann::Metric compareMetric,
      unsigned L, unsigned R, bool accelerate_build, bool shuffle_build,
      double sampling_rate, double ram_budget, std::string mem_index_path,
      std::string medoids_path, std::string centroids_file);
  template std::unique_ptr<diskann::Index<knowhere::bf16>>
  build_merged_vamana_index<knowhere::bf16>(
      std::string base_file, bool ip_prepared, diskann::Metric compareMetric,
      unsigned L, unsigned R, bool accelerate_build, bool shuffle_build,
      double sampling_rate, double ram_budget, std::string mem_index_path,
      std::string medoids_path, std::string centroids_file);

  template void generate_cache_list_from_graph_with_pq<int8_t>(
      _u64 num_nodes_to_cache, unsigned R, const diskann::Metric compare_metric,
      const std::string &sample_file, const std::string &pq_pivots_path,
      const std::string &pq_compressed_code_path, const unsigned entry_point,
      const std::vector<std::vector<unsigned>> &graph,
      const std::string                        &cache_file);
  template void generate_cache_list_from_graph_with_pq<float>(
      _u64 num_nodes_to_cache, unsigned R, const diskann::Metric compare_metric,
      const std::string &sample_file, const std::string &pq_pivots_path,
      const std::string &pq_compressed_code_path, const unsigned entry_point,
      const std::vector<std::vector<unsigned>> &graph,
      const std::string                        &cache_file);
  template void generate_cache_list_from_graph_with_pq<uint8_t>(
      _u64 num_nodes_to_cache, unsigned R, const diskann::Metric compare_metric,
      const std::string &sample_file, const std::string &pq_pivots_path,
      const std::string &pq_compressed_code_path, const unsigned entry_point,
      const std::vector<std::vector<unsigned>> &graph,
      const std::string                        &cache_file);
  template void generate_cache_list_from_graph_with_pq<knowhere::fp16>(
      _u64 num_nodes_to_cache, unsigned R, const diskann::Metric compare_metric,
      const std::string &sample_file, const std::string &pq_pivots_path,
      const std::string &pq_compressed_code_path, const unsigned entry_point,
      const std::vector<std::vector<unsigned>> &graph,
      const std::string                        &cache_file);
  template void generate_cache_list_from_graph_with_pq<knowhere::bf16>(
      _u64 num_nodes_to_cache, unsigned R, const diskann::Metric compare_metric,
      const std::string &sample_file, const std::string &pq_pivots_path,
      const std::string &pq_compressed_code_path, const unsigned entry_point,
      const std::vector<std::vector<unsigned>> &graph,
      const std::string                        &cache_file);
  template void create_aisaq_layout<float>(
            const std::string base_file,
            const std::string mem_index_file,
            const std::string output_file,
            const std::string reorder_data_file,
            const std::string &index_prefix_path,
            const diskann::Metric metric,  
            int inline_pq,
            bool &rearrange
    );
  template void create_aisaq_layout<int8_t>(
            const std::string base_file,
            const std::string mem_index_file,
            const std::string output_file,
            const std::string reorder_data_file,
            const std::string &index_prefix_path,
            const diskann::Metric metric,  
            int inline_pq,
            bool &rearrange
    );
  template void create_aisaq_layout<uint8_t>(
            const std::string base_file,
            const std::string mem_index_file,
            const std::string output_file,
            const std::string reorder_data_file,
            const std::string &index_prefix_path,
            const diskann::Metric metric,  
            int inline_pq,
            bool &rearrange
    );
  template void create_aisaq_layout<knowhere::bf16>(
            const std::string base_file,
            const std::string mem_index_file,
            const std::string output_file,
            const std::string reorder_data_file,
            const std::string &index_prefix_path,
            const diskann::Metric metric,  
            int inline_pq,
            bool &rearrange
    );
  template void create_aisaq_layout<knowhere::fp16>(
            const std::string base_file,
            const std::string mem_index_file,
            const std::string output_file,
            const std::string reorder_data_file,
            const std::string &index_prefix_path,
            const diskann::Metric metric,  
            int inline_pq,
            bool &rearrange
    );
};  // namespace diskann
