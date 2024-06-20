// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "diskann/aligned_file_reader.h"
#include "diskann/logger.h"
#include "diskann/pq_flash_index.h"
#include <malloc.h>
#include "diskann/percentile_stats.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <thread>
#include <unordered_map>
#include "diskann/distance.h"
#include "diskann/exceptions.h"
#include "diskann/aux_utils.h"
#include "diskann/timer.h"
#include "diskann/utils.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/heap.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/utils.h"
#include "tsl/robin_set.h"

#include "diskann/linux_aligned_file_reader.h"

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_U32(stream, val) stream.read((char *) &val, sizeof(_u32))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + disk_bytes_per_point)

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) \
  (((_u64) (id)) / nvecs_per_sector + reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) \
  ((((_u64) (id)) % nvecs_per_sector) * data_dim * sizeof(float))

namespace {
  static auto async_pool = knowhere::ThreadPool::CreateFIFO(1, "DiskANN_Async_Cache_Making");

  constexpr _u64  kRefineBeamWidthFactor = 2;
  constexpr _u64  kBruteForceTopkRefineExpansionFactor = 2;
  constexpr float kFilterThreshold = 0.93f;
  constexpr float kAlpha = 0.15f;
}  // namespace

namespace diskann {
  template<typename T>
  PQFlashIndex<T>::PQFlashIndex(std::shared_ptr<AlignedFileReader> fileReader,
                                diskann::Metric                    m)
      : reader(fileReader), metric(m) {
    if (m == diskann::Metric::INNER_PRODUCT || m == diskann::Metric::COSINE) {
      if (!knowhere::KnowhereFloatTypeCheck<T>::value) {
        LOG(WARNING) << "Cannot normalize integral data types."
                     << " This may result in erroneous results or poor recall."
                     << " Consider using L2 distance with integral data types.";
      }
      if (m == diskann::Metric::INNER_PRODUCT) {
        LOG(INFO) << "Cosine metric chosen for (normalized) float data."
                     "Changing distance to L2 to boost accuracy.";
        m = diskann::Metric::L2;
      }
    }

    this->dist_cmp = diskann::get_distance_function<T>(m);
    this->dist_cmp_float = diskann::get_distance_function<float>(m);
  }

  template<typename T>
  PQFlashIndex<T>::~PQFlashIndex() {
    destroy_cache_async_task();

    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    // delete backing bufs for nhood and coord cache
    if (nhood_cache_buf != nullptr) {
      nhood_cache_buf.reset();
      diskann::aligned_free(coord_cache_buf);
    }

    if (load_flag) {
      this->destroy_thread_data();
      reader->close();
    }
  }

  template<typename T>
  void PQFlashIndex<T>::setup_thread_data(_u64 nthreads) {
    LOG(INFO) << "Setting up thread-specific contexts for nthreads: "
              << nthreads;
    for (_s64 thread = 0; thread < (_s64) nthreads; thread++) {
      QueryScratch<T> scratch;
      _u64 coord_alloc_size = ROUND_UP(sizeof(T) * this->aligned_dim, 256);
      diskann::alloc_aligned((void **) &scratch.coord_scratch, coord_alloc_size,
                             256);
      diskann::alloc_aligned((void **) &scratch.sector_scratch,
                             (_u64) MAX_N_SECTOR_READS * read_len_for_node,
                             SECTOR_LEN);
      diskann::alloc_aligned(
          (void **) &scratch.aligned_pq_coord_scratch,
          (_u64) MAX_GRAPH_DEGREE * (_u64) this->aligned_dim * sizeof(_u8),
          256);
      diskann::alloc_aligned((void **) &scratch.aligned_pqtable_dist_scratch,
                             256 * (_u64) this->aligned_dim * sizeof(float),
                             256);
      diskann::alloc_aligned((void **) &scratch.aligned_dist_scratch,
                             (_u64) MAX_GRAPH_DEGREE * sizeof(float), 256);
      diskann::alloc_aligned((void **) &scratch.aligned_query_T,
                             this->aligned_dim * sizeof(T), 8 * sizeof(T));
      diskann::alloc_aligned((void **) &scratch.aligned_query_float,
                             this->aligned_dim * sizeof(float),
                             8 * sizeof(float));
      scratch.visited = new tsl::robin_set<_u64>(4096);

      memset((void *) scratch.coord_scratch, 0, sizeof(T) * this->aligned_dim);
      memset((void *) scratch.aligned_query_T, 0,
             this->aligned_dim * sizeof(T));
      memset(scratch.aligned_query_float, 0, this->aligned_dim * sizeof(float));

      ThreadData<T> data;
      data.scratch = scratch;
      this->thread_data.push(data);
    }
    load_flag = true;
  }

  template<typename T>
  _u64 PQFlashIndex<T>::get_thread_data_size() {
    _u64 thread_data_size = 0;
    thread_data_size += ROUND_UP(sizeof(T) * this->aligned_dim, 256);
    thread_data_size +=
        ROUND_UP((_u64) MAX_N_SECTOR_READS * read_len_for_node, SECTOR_LEN);
    thread_data_size += ROUND_UP(
        (_u64) MAX_GRAPH_DEGREE * (_u64) this->aligned_dim * sizeof(_u8), 256);
    thread_data_size +=
        ROUND_UP(256 * (_u64) this->aligned_dim * sizeof(float), 256);
    thread_data_size += ROUND_UP((_u64) MAX_GRAPH_DEGREE * sizeof(float), 256);
    thread_data_size += ROUND_UP(this->aligned_dim * sizeof(T), 8 * sizeof(T));
    thread_data_size +=
        ROUND_UP(this->aligned_dim * sizeof(float), 8 * sizeof(float));
    return thread_data_size;
  }

  template<typename T>
  void PQFlashIndex<T>::destroy_thread_data() {
    LOG_KNOWHERE_DEBUG_ << "Clearing scratch";
    assert(this->thread_data.size() == this->max_nthreads);
    while (this->thread_data.size() > 0) {
      ThreadData<T> data = this->thread_data.pop();
      while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
      }
      auto &scratch = data.scratch;
      diskann::aligned_free((void *) scratch.coord_scratch);
      diskann::aligned_free((void *) scratch.sector_scratch);
      diskann::aligned_free((void *) scratch.aligned_pq_coord_scratch);
      diskann::aligned_free((void *) scratch.aligned_pqtable_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_dist_scratch);
      diskann::aligned_free((void *) scratch.aligned_query_float);
      diskann::aligned_free((void *) scratch.aligned_query_T);

      delete scratch.visited;
    }
  }

  template<typename T>
  void PQFlashIndex<T>::load_cache_list(std::vector<uint32_t> &node_list) {
    _u64 num_cached_nodes = node_list.size();
    LOG_KNOWHERE_DEBUG_ << "Loading the cache list(" << num_cached_nodes
                        << " points) into memory...";

    auto ctx = this->reader->get_ctx();

    if (nhood_cache_buf == nullptr) {
      nhood_cache_buf =
          std::make_unique<unsigned[]>(num_cached_nodes * (max_degree + 1));
      memset(nhood_cache_buf.get(), 0, num_cached_nodes * (max_degree + 1) * sizeof(unsigned));
    }

    _u64 coord_cache_buf_len = num_cached_nodes * aligned_dim;
    if (coord_cache_buf == nullptr) {
        diskann::alloc_aligned((void **) &coord_cache_buf,
                             coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
        std::fill_n(coord_cache_buf, coord_cache_buf_len, T());
    }

    size_t BLOCK_SIZE = 32;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);

    for (_u64 block = 0; block < num_blocks; block++) {
      _u64 start_idx = block * BLOCK_SIZE;
      _u64 end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);
      std::vector<AlignedRead>             read_reqs;
      std::vector<std::pair<_u32, char *>> nhoods;
      for (_u64 node_idx = start_idx; node_idx < end_idx; node_idx++) {
        AlignedRead read;
        char       *buf = nullptr;
        alloc_aligned((void **) &buf, read_len_for_node, SECTOR_LEN);
        nhoods.push_back(std::make_pair(node_list[node_idx], buf));
        read.len = read_len_for_node;
        read.buf = buf;
        read.offset = get_node_sector_offset(node_list[node_idx]);
        read_reqs.push_back(read);
      }

      reader->read(read_reqs, ctx);

      _u64 node_idx = start_idx;
      for (_u32 i = 0; i < read_reqs.size(); i++) {
        auto &nhood = nhoods[i];
        char *node_buf = get_offset_to_node(nhood.second, nhood.first);
        T    *node_coords = OFFSET_TO_NODE_COORDS(node_buf);
        T    *cached_coords = coord_cache_buf + node_idx * aligned_dim;
        memcpy(cached_coords, node_coords, disk_bytes_per_point);

        // insert node nhood into nhood_cache
        unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);

        auto                        nnbrs = *node_nhood;
        unsigned                   *nbrs = node_nhood + 1;
        std::pair<_u32, unsigned *> cnhood;
        cnhood.first = nnbrs;
        cnhood.second = nhood_cache_buf.get() + node_idx * (max_degree + 1);
        memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
        {
          std::unique_lock<std::shared_mutex> lock(this->cache_mtx);
          coord_cache.insert(std::make_pair(nhood.first, cached_coords));
          nhood_cache.insert(std::make_pair(nhood.first, cnhood));
        }
        aligned_free(nhood.second);
        node_idx++;
      }
    }
    this->reader->put_ctx(ctx);
    LOG_KNOWHERE_DEBUG_ << "done.";
  }

  template<typename T>
  void PQFlashIndex<T>::async_generate_cache_list_from_sample_queries(
      std::string sample_bin, _u64 l_search, _u64 beamwidth,
      _u64 num_nodes_to_cache) {
    this->search_counter.store(0);
    node_visit_counter.clear();
    node_visit_counter.resize(this->num_points);
    for (_u32 i = 0; i < this->node_visit_counter.size(); i++) {
      this->node_visit_counter[i].first = i;
      this->node_visit_counter[i].second =
          std::make_unique<std::atomic<_u32>>(0);
    }
    this->count_visited_nodes.store(true);

    // sync allocate memory
    if (nhood_cache_buf == nullptr) {
      nhood_cache_buf =
          std::make_unique<unsigned[]>(num_nodes_to_cache * (max_degree + 1));
      memset(nhood_cache_buf.get(), 0, num_nodes_to_cache * (max_degree + 1) * sizeof(unsigned));
    }

    _u64 coord_cache_buf_len = num_nodes_to_cache * aligned_dim;
    if (coord_cache_buf == nullptr) {
        diskann::alloc_aligned((void **) &coord_cache_buf,
                             coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
        std::fill_n(coord_cache_buf, coord_cache_buf_len, T());
    }

    async_pool.push([&, state_controller = this->state_controller, sample_bin,
                     l_search, beamwidth, num_nodes_to_cache]() {
      {
        std::unique_lock<std::mutex> guard(state_controller->status_mtx);
        if (state_controller->status.load() ==
            ThreadSafeStateController::Status::KILLED) {
          state_controller->status.store(
              ThreadSafeStateController::Status::DONE);
          return;
        }
        state_controller->status.store(
            ThreadSafeStateController::Status::DOING);
      }
      T *samples;
      try {
        auto s = std::chrono::high_resolution_clock::now();
        _u64 sample_num, sample_dim, sample_aligned_dim;

        std::stringstream stream;

        if (file_exists(sample_bin)) {
          diskann::load_aligned_bin<T>(sample_bin, samples, sample_num,
                                       sample_dim, sample_aligned_dim);
        } else {
          stream << "Sample bin file not found. Not generating cache."
                 << std::endl;
          throw diskann::ANNException(stream.str(), -1);
        }

        int64_t tmp_result_id_64 = 0;
        float   tmp_result_dist = 0.0;

        _u64 id = 0;
        while (this->search_counter.load() < sample_num && id < sample_num &&
               state_controller->status.load() !=
                   ThreadSafeStateController::Status::STOPPING) {
          cached_beam_search(samples + (id * sample_aligned_dim), 1, l_search,
                             &tmp_result_id_64, &tmp_result_dist, beamwidth);
          id++;
        }

        if (state_controller->status.load() ==
            ThreadSafeStateController::Status::STOPPING) {
          stream << "pq_flash_index is destoried, async thread should be exit."
                 << std::endl;
          throw diskann::ANNException(stream.str(), -1);
        }

        {
          std::unique_lock<std::shared_mutex> lock(
              this->node_visit_counter_mtx);
          this->count_visited_nodes.store(false);

          std::sort(this->node_visit_counter.begin(),
                    this->node_visit_counter.end(),
                    [](auto &left, auto &right) {
                      return *(left.second) > *(right.second);
                    });
        }

        std::vector<uint32_t> node_list;
        node_list.clear();
        node_list.shrink_to_fit();
        node_list.reserve(num_nodes_to_cache);
        for (_u64 i = 0; i < num_nodes_to_cache; i++) {
          node_list.push_back(this->node_visit_counter[i].first);
        }

        this->load_cache_list(node_list);
        auto e = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = e - s;
        LOG(INFO) << "Using sample queries to generate cache, cost: "
                  << diff.count() << "s";
      } catch (std::exception &e) {
        LOG(INFO) << "Can't generate Diskann cache: " << e.what();
      }

      // clear up
      {
        std::unique_lock<std::shared_mutex> lock(this->node_visit_counter_mtx);
        if (this->count_visited_nodes.load() == true) {
          this->count_visited_nodes.store(false);
        }
        node_visit_counter.clear();
        node_visit_counter.shrink_to_fit();
      }

      this->search_counter.store(0);
      // free samples
      if (samples != nullptr) {
        diskann::aligned_free(samples);
      }

      {
        std::unique_lock<std::mutex> guard(state_controller->status_mtx);
        state_controller->status.store(ThreadSafeStateController::Status::DONE);
        state_controller->cond.notify_one();
      }
    });

    return;
  }

  template<typename T>
  void PQFlashIndex<T>::cache_bfs_levels(_u64 num_nodes_to_cache,
                                         std::vector<uint32_t> &node_list) {
    std::random_device rng;
    std::mt19937       urng(rng());

    node_list.clear();

    // borrow thread data
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      this_thread_data = this->thread_data.pop();
    }

    auto ctx = this->reader->get_ctx();

    std::unique_ptr<tsl::robin_set<unsigned>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<unsigned>>();
    prev_level = std::make_unique<tsl::robin_set<unsigned>>();

    for (_u64 miter = 0; miter < num_medoids; miter++) {
      cur_level->insert(medoids[miter]);
    }

    _u64     lvl = 1;
    uint64_t prev_node_list_size = 0;
    while ((node_list.size() + cur_level->size() < num_nodes_to_cache) &&
           cur_level->size() != 0) {
      // swap prev_level and cur_level
      std::swap(prev_level, cur_level);
      // clear cur_level
      cur_level->clear();

      std::vector<unsigned> nodes_to_expand;

      for (const unsigned &id : *prev_level) {
        if (std::find(node_list.begin(), node_list.end(), id) !=
            node_list.end()) {
          continue;
        }
        node_list.push_back(id);
        nodes_to_expand.push_back(id);
      }

      std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);

      LOG_KNOWHERE_DEBUG_ << "Level: " << lvl;
      bool finish_flag = false;

      uint64_t BLOCK_SIZE = 1024;
      uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
      for (size_t block = 0; block < nblocks && !finish_flag; block++) {
        size_t start = block * BLOCK_SIZE;
        size_t end =
            (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());
        std::vector<AlignedRead>             read_reqs;
        std::vector<std::pair<_u32, char *>> nhoods;
        for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
          char *buf = nullptr;
          alloc_aligned((void **) &buf, read_len_for_node, SECTOR_LEN);
          nhoods.push_back(std::make_pair(nodes_to_expand[cur_pt], buf));
          AlignedRead read;
          read.len = read_len_for_node;
          read.buf = buf;
          read.offset = get_node_sector_offset(nodes_to_expand[cur_pt]);
          read_reqs.push_back(read);
        }

        // issue read requests
        reader->read(read_reqs, ctx);

        // process each nhood buf
        for (_u32 i = 0; i < read_reqs.size(); i++) {
          auto &nhood = nhoods[i];

          // insert node coord into coord_cache
          char     *node_buf = get_offset_to_node(nhood.second, nhood.first);
          unsigned *node_nhood = OFFSET_TO_NODE_NHOOD(node_buf);
          _u64      nnbrs = (_u64) *node_nhood;
          unsigned *nbrs = node_nhood + 1;
          // explore next level
          for (_u64 j = 0; j < nnbrs && !finish_flag; j++) {
            if (std::find(node_list.begin(), node_list.end(), nbrs[j]) ==
                node_list.end()) {
              cur_level->insert(nbrs[j]);
            }
            if (cur_level->size() + node_list.size() >= num_nodes_to_cache) {
              finish_flag = true;
            }
          }
          aligned_free(nhood.second);
        }
      }

      LOG_KNOWHERE_DEBUG_ << ". #nodes: "
                          << node_list.size() - prev_node_list_size
                          << ", #nodes thus far: " << node_list.size();
      prev_node_list_size = node_list.size();
      lvl++;
    }

    std::vector<uint32_t> cur_level_node_list;
    for (const unsigned &p : *cur_level)
      cur_level_node_list.push_back(p);

    std::shuffle(cur_level_node_list.begin(), cur_level_node_list.end(), urng);
    size_t residual = num_nodes_to_cache - node_list.size();

    for (size_t i = 0; i < (std::min)(residual, cur_level_node_list.size());
         i++)
      node_list.push_back(cur_level_node_list[i]);

    LOG_KNOWHERE_DEBUG_ << "Level: " << lvl << ". #nodes: "
                        << node_list.size() - prev_node_list_size
                        << ", #nodes thus far: " << node_list.size();

    // return thread data
    this->thread_data.push(this_thread_data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);

    LOG(INFO) << "done";
  }

  template<typename T>
  void PQFlashIndex<T>::use_medoids_data_as_centroids() {
    if (centroid_data != nullptr)
      aligned_free(centroid_data);
    alloc_aligned(((void **) &centroid_data),
                  num_medoids * aligned_dim * sizeof(float), 32);
    std::memset(centroid_data, 0, num_medoids * aligned_dim * sizeof(float));

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    auto ctx = this->reader->get_ctx();
    // borrow buf
    auto scratch = &(data.scratch);
    scratch->reset();
    char *sector_scratch = scratch->sector_scratch;
    T    *medoid_coords = scratch->coord_scratch;

    LOG(INFO) << "Loading centroid data from medoids vector data of "
              << num_medoids << " medoid(s)";
    for (uint64_t cur_m = 0; cur_m < num_medoids; cur_m++) {
      auto medoid = medoids[cur_m];
      // read medoid nhood
      std::vector<AlignedRead> medoid_read(1);
      medoid_read[0].len = read_len_for_node;
      medoid_read[0].buf = sector_scratch;
      medoid_read[0].offset = get_node_sector_offset(medoid);
      reader->read(medoid_read, ctx);

      // all data about medoid
      char *medoid_node_buf = get_offset_to_node(sector_scratch, medoid);

      // add medoid coords to `coord_cache`
      T *medoid_disk_coords = OFFSET_TO_NODE_COORDS(medoid_node_buf);
      memcpy(medoid_coords, medoid_disk_coords, disk_bytes_per_point);

      if (!use_disk_index_pq) {
        for (uint32_t i = 0; i < data_dim; i++) {
          centroid_data[cur_m * aligned_dim + i] = medoid_coords[i];
        }
      } else {
        disk_pq_table.inflate_vector((_u8 *) medoid_coords,
                                     (centroid_data + cur_m * aligned_dim));
      }
    }

    // return ctx
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);
  }

  template<typename T>
  int PQFlashIndex<T>::load(uint32_t num_threads, const char *index_prefix) {
    std::string pq_table_bin =
        get_pq_pivots_filename(std::string(index_prefix));
    std::string pq_compressed_vectors =
        get_pq_compressed_filename(std::string(index_prefix));
    std::string disk_index_file =
        get_disk_index_filename(std::string(index_prefix));
    std::string medoids_file =
        get_disk_index_medoids_filename(std::string(disk_index_file));
    std::string centroids_file =
        get_disk_index_centroids_filename(std::string(disk_index_file));

    size_t pq_file_dim, pq_file_num_centroids;
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim);

    this->disk_index_file = disk_index_file;
    if (pq_file_num_centroids != 256) {
      LOG(ERROR) << "Error. Number of PQ centroids is not 256. Exitting.";
      return -1;
    }

    this->data_dim = pq_file_dim;
    // will reset later if we use PQ on disk
    this->disk_data_dim = this->data_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->disk_bytes_per_point = this->data_dim * sizeof(T);
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
    diskann::load_bin<_u8>(pq_compressed_vectors, this->data, npts_u64,
                           nchunks_u64);

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

    pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);

    LOG(INFO)
        << "Loaded PQ centroids and in-memory compressed vectors. #points: "
        << num_points << " #dim: " << data_dim
        << " #aligned_dim: " << aligned_dim << " #chunks: " << n_chunks;

    std::string disk_pq_pivots_path = this->disk_index_file + "_pq_pivots.bin";
    if (file_exists(disk_pq_pivots_path)) {
      use_disk_index_pq = true;
      // giving 0 chunks to make the pq_table infer from the
      // chunk_offsets file the correct value
      disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
      disk_pq_n_chunks = disk_pq_table.get_num_chunks();
      disk_bytes_per_point =
          disk_pq_n_chunks *
          sizeof(_u8);  // revising disk_bytes_per_point since DISK PQ is used.
      std::cout << "Disk index uses PQ data compressed down to "
                << disk_pq_n_chunks << " bytes per point." << std::endl;
    }

    // read index metadata
    std::ifstream index_metadata(disk_index_file, std::ios::binary);

    size_t actual_index_size = get_file_size(disk_index_file);
    size_t expected_file_size;
    READ_U64(index_metadata, expected_file_size);
    if (actual_index_size != expected_file_size) {
      LOG(ERROR) << "File size mismatch for " << disk_index_file
                 << " (size: " << actual_index_size << ")"
                 << " with meta-data size: " << expected_file_size;
      return -1;
    }

    _u64 disk_nnodes;
    READ_U64(index_metadata, disk_nnodes);
    if (disk_nnodes != num_points) {
      LOG(ERROR) << "Mismatch in #points for compressed data file and disk "
                    "index file: "
                 << disk_nnodes << " vs " << num_points;
      return -1;
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, max_node_len);
    READ_U64(index_metadata, nnodes_per_sector);

    if (max_node_len > SECTOR_LEN) {
      long_node = true;
      nsectors_per_node = ROUND_UP(max_node_len, SECTOR_LEN) / SECTOR_LEN;
      read_len_for_node = SECTOR_LEN * nsectors_per_node;
    }

    max_degree = ((max_node_len - disk_bytes_per_point) / sizeof(unsigned)) - 1;

    if (max_degree > MAX_GRAPH_DEGREE) {
      std::stringstream stream;
      stream << "Error loading index. Ensure that max graph degree (R) does "
                "not exceed "
             << MAX_GRAPH_DEGREE << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->num_frozen_points);
    _u64 file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->num_frozen_points == 1)
      this->frozen_location = file_frozen_id;
    if (this->num_frozen_points == 1) {
      diskann::cout << " Detected frozen point in index at location "
                    << this->frozen_location
                    << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->reorder_data_exists);
    if (this->reorder_data_exists) {
      if (this->use_disk_index_pq == false) {
        throw ANNException(
            "Reordering is designed for used with disk PQ compression option",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      READ_U64(index_metadata, this->reorder_data_start_sector);
      READ_U64(index_metadata, this->ndims_reorder_vecs);
      READ_U64(index_metadata, this->nvecs_per_sector);
    }
    LOG(INFO) << "Disk-Index File Meta-data: "
              << "# nodes per sector: " << nnodes_per_sector
              << ", max node len (bytes): " << max_node_len
              << ", max node degree: " << max_degree;

    index_metadata.close();

    // open AlignedFileReader handle to index_file
    std::string index_fname(disk_index_file);
    reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

    if (file_exists(medoids_file)) {
      size_t tmp_dim;
      diskann::load_bin<uint32_t>(medoids_file, medoids, num_medoids, tmp_dim);

      if (tmp_dim != 1) {
        std::stringstream stream;
        stream << "Error loading medoids file. Expected bin format of m times "
                  "1 vector of uint32_t."
               << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
      if (!file_exists(centroids_file)) {
        LOG(INFO)
            << "Centroid data file not found. Using corresponding vectors "
               "for the medoids ";
        use_medoids_data_as_centroids();
      } else {
        size_t num_centroids, aligned_tmp_dim;
        diskann::load_aligned_bin<float>(centroids_file, centroid_data,
                                         num_centroids, tmp_dim,
                                         aligned_tmp_dim);
        if (aligned_tmp_dim != aligned_dim || num_centroids != num_medoids) {
          std::stringstream stream;
          stream << "Error loading centroids data file. Expected bin format of "
                    "m times data_dim vector of float, where m is number of "
                    "medoids "
                    "in medoids file.";
          LOG(ERROR) << stream.str();
          throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                      __LINE__);
        }
      }
    } else {
      num_medoids = 1;
      medoids =  std::make_unique<uint32_t[]>(1);
      medoids[0] = (_u32) (medoid_id_on_file);
      use_medoids_data_as_centroids();
    }

    std::string norm_file =
        get_disk_index_max_base_norm_file(std::string(disk_index_file));

    if (file_exists(norm_file) && metric == diskann::Metric::INNER_PRODUCT) {
      _u64   dumr, dumc;
      std::unique_ptr<float[]> norm_val = nullptr;
      diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
      this->max_base_norm = norm_val[0];
      LOG_KNOWHERE_DEBUG_ << "Setting re-scaling factor of base vectors to "
                          << this->max_base_norm;
    }

    if (file_exists(norm_file) && metric == diskann::Metric::COSINE) {
      _u64 dumr, dumc;
      diskann::load_bin<float>(norm_file, base_norms, dumr, dumc);
      LOG_KNOWHERE_DEBUG_ << "Setting base vector norms";
    }

    return 0;
  }

  template<typename T>
  std::optional<float> PQFlashIndex<T>::init_thread_data(ThreadData<T> &data,
                                                         const T *query1) {
    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float query_norm = 0;
    auto  q_dim = this->data_dim;
    if (metric == diskann::Metric::INNER_PRODUCT) {
      // query_dim need to be specially treated when using IP
      q_dim--;
    }
    for (uint32_t i = 0; i < q_dim; i++) {
      data.scratch.aligned_query_float[i] = (float) query1[i];
      data.scratch.aligned_query_T[i] = query1[i];
      query_norm += (float) query1[i] * (float) query1[i];
    }

    // if inner product, we also normalize the query and set the last coordinate
    // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
    if (metric == diskann::Metric::INNER_PRODUCT ||
        metric == diskann::Metric::COSINE) {
      if (query_norm == 0) {
        return std::nullopt;
      }
      query_norm = std::sqrt(query_norm);
      if (metric == diskann::Metric::INNER_PRODUCT) {
        data.scratch.aligned_query_T[this->data_dim - 1] = 0;
        data.scratch.aligned_query_float[this->data_dim - 1] = 0;
      }
      for (uint32_t i = 0; i < q_dim; i++) {
        data.scratch.aligned_query_T[i] =
            (T) ((float) data.scratch.aligned_query_T[i] / query_norm);
        data.scratch.aligned_query_float[i] /= query_norm;
      }
    }

    data.scratch.reset();
    return query_norm;
  }

  template<typename T>
  void PQFlashIndex<T>::brute_force_beam_search(
      ThreadData<T> &data, const float query_norm, const _u64 k_search,
      _s64 *indices, float *distances, const _u64 beam_width_param,
      IOContext &ctx, QueryStats *stats,
      const knowhere::feder::diskann::FederResultUniq &feder,
      knowhere::BitsetView                             bitset_view) {
    auto         query_scratch = &(data.scratch);
    const T     *query = data.scratch.aligned_query_T;
    auto         beam_width = beam_width_param * kRefineBeamWidthFactor;
    const float *query_float = data.scratch.aligned_query_float;
    float       *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);
    float         *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8           *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
    constexpr _u32 pq_batch_size = MAX_GRAPH_DEGREE;
    std::vector<unsigned> pq_batch_ids;
    pq_batch_ids.reserve(pq_batch_size);
    const _u64 pq_topk = k_search * kBruteForceTopkRefineExpansionFactor;
    knowhere::ResultMaxHeap<float, int64_t> pq_max_heap(pq_topk);
    T *data_buf = query_scratch->coord_scratch;
    std::unordered_map<_u64, std::vector<_u64>> nodes_in_sectors_to_visit;
    std::vector<AlignedRead>                    frontier_read_reqs;
    frontier_read_reqs.reserve(beam_width);
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;
    knowhere::ResultMaxHeap<float, _u64> max_heap(k_search);
    Timer                                io_timer, query_timer;

    // scan un-marked points and calculate pq dists
    for (_u64 id = 0; id < num_points; ++id) {
      if (bitset_view.empty() || !bitset_view.test(id)) {
        pq_batch_ids.push_back(id);
      }

      if (pq_batch_ids.size() == pq_batch_size || id == num_points - 1) {
        const size_t sz = pq_batch_ids.size();
        aggregate_coords(pq_batch_ids.data(), sz, this->data.get(), this->n_chunks,
                         pq_coord_scratch);
        pq_dist_lookup(pq_coord_scratch, sz, this->n_chunks, pq_dists,
                       dist_scratch);
        for (size_t i = 0; i < sz; ++i) {
          pq_max_heap.Push(dist_scratch[i], pq_batch_ids[i]);
        }
        pq_batch_ids.clear();
      }
    }

    // deduplicate sectors by ids
    while (const auto opt = pq_max_heap.Pop()) {
      const auto [dist, id] = opt.value();

      // check if in cache
      {
        std::shared_lock<std::shared_mutex> lock(this->cache_mtx);
        if (coord_cache.find(id) != coord_cache.end()) {
          float dist = dist_cmp_wrap(query, coord_cache.at(id),
                                     (size_t) aligned_dim, id);
          max_heap.Push(dist, id);
          continue;
        }
      }

      // deduplicate and prepare for I/O
      const _u64 sector_offset = get_node_sector_offset(id);
      nodes_in_sectors_to_visit[sector_offset].push_back(id);
    }

    for (auto it = nodes_in_sectors_to_visit.cbegin();
         it != nodes_in_sectors_to_visit.cend();) {
      const auto sector_offset = it->first;
      frontier_read_reqs.emplace_back(
          sector_offset, read_len_for_node,
          sector_scratch + sector_scratch_idx * read_len_for_node);
      ++sector_scratch_idx, ++it;
      if (stats != nullptr) {
        stats->n_4k++;
        stats->n_ios++;
      }

      // perform I/Os and calculate exact distances
      if (frontier_read_reqs.size() == beam_width ||
          it == nodes_in_sectors_to_visit.cend()) {
        io_timer.reset();
        reader->read(frontier_read_reqs, ctx);  // synchronous IO linux
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }

        T *node_fp_coords_copy = data_buf;
        for (const auto &req : frontier_read_reqs) {
          const auto offset = req.offset;
          char      *sector_buf = reinterpret_cast<char *>(req.buf);
          for (const auto cur_id : nodes_in_sectors_to_visit[offset]) {
            char *node_buf = get_offset_to_node(sector_buf, cur_id);
            memcpy(node_fp_coords_copy, node_buf,
                   disk_bytes_per_point);  // Do we really need memcpy here?
            float dist = dist_cmp_wrap(query, node_fp_coords_copy,
                                       (size_t) aligned_dim, cur_id);
            max_heap.Push(dist, cur_id);
            if (feder != nullptr) {
              feder->visit_info_.AddTopCandidateInfo(cur_id, dist);
              feder->id_set_.insert(cur_id);
            }
          }
        }
        frontier_read_reqs.clear();
        sector_scratch_idx = 0;
      }
    }

    for (_s64 i = k_search - 1; i >= 0; --i) {
      if ((_u64) i >= max_heap.Size()) {
        indices[i] = -1;
        if (distances != nullptr) {
          distances[i] = -1;
        }
        continue;
      }
      if (const auto op = max_heap.Pop()) {
        const auto [dis, id] = op.value();
        indices[i] = id;
        if (distances != nullptr) {
          distances[i] = dis;
          if (metric == diskann::Metric::INNER_PRODUCT) {
            distances[i] = 1.0 - distances[i] / 2.0;
            if (max_base_norm != 0) {
              distances[i] *= (max_base_norm * query_norm);
            }
          } else if (metric == diskann::Metric::COSINE) {
            distances[i] = -distances[i];
          }
        }
      } else {
        LOG(ERROR) << "Size is incorrect";
      }
    }
    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return;
  }

  template<typename T>
  void PQFlashIndex<T>::cached_beam_search(
      const T *query1, const _u64 k_search, const _u64 l_search, _s64 *indices,
      float *distances, const _u64 beam_width, const bool use_reorder_data,
      QueryStats *stats, const knowhere::feder::diskann::FederResultUniq &feder,
      knowhere::BitsetView bitset_view, const float filter_ratio_in,
      const bool for_tuning) {
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }
    auto query_norm_opt = init_thread_data(data, query1);
    if (!query_norm_opt.has_value()) {
      // return an empty answer when calcu a zero point
      this->thread_data.push(data);
      this->thread_data.push_notify_all();
      return;
    }
    float query_norm = query_norm_opt.value();
    auto  ctx = this->reader->get_ctx();

    size_t bv_cnt = 0;

    if (!bitset_view.empty()) {
      const auto filter_threshold =
          filter_ratio_in < 0 ? kFilterThreshold : filter_ratio_in;
      bv_cnt = bitset_view.count();
#ifdef NOT_COMPILE_FOR_SWIG
      double ratio = ((double) bv_cnt) / bitset_view.size();
      knowhere::knowhere_diskann_bitset_ratio.Observe(ratio);
#endif
      if (bitset_view.size() == bv_cnt) {
        for (_u64 i = 0; i < k_search; i++) {
          indices[i] = -1;
          if (distances != nullptr) {
            distances[i] = -1;
          }
        }
        return;
      }

      if (bv_cnt >= bitset_view.size() * filter_threshold) {
        brute_force_beam_search(data, query_norm, k_search, indices, distances,
                                beam_width, ctx, stats, feder, bitset_view);
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
        this->reader->put_ctx(ctx);
        return;
      }
    }

    // Turn to BF is k_search is too large
    if (k_search > 0.5 * (num_points - bv_cnt)) {
      brute_force_beam_search(data, query_norm, k_search, indices, distances,
                              beam_width, ctx, stats, feder, bitset_view);
      this->thread_data.push(data);
      this->thread_data.push_notify_all();
      this->reader->put_ctx(ctx);
      return;
    }

    auto         query_scratch = &(data.scratch);
    const T     *query = data.scratch.aligned_query_T;
    const float *query_float = data.scratch.aligned_query_float;

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    Timer io_timer, query_timer;
    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    // query <-> PQ chunk centers distances
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    pq_table.populate_chunk_distances(query_float, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    _u8   *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      aggregate_coords(ids, n_ids, this->data.get(), this->n_chunks,
                       pq_coord_scratch);
      pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                     dists_out);
    };
    Timer                 cpu_timer;
    std::vector<Neighbor> retset(l_search + 1);
    tsl::robin_set<_u64> &visited = *(query_scratch->visited);

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);
    auto vec_hash = knowhere::hash_vec(query_float, data_dim);
    _u32 best_medoid = 0;
    // for tuning, do not use cache
    if (for_tuning || !lru_cache.try_get(vec_hash, best_medoid)) {
      float best_dist = (std::numeric_limits<float>::max)();
      std::vector<SimpleNeighbor> medoid_dists;
      for (_u64 cur_m = 0; cur_m < num_medoids; cur_m++) {
        float cur_expanded_dist = dist_cmp_float_wrap(
            query_float, centroid_data + aligned_dim * cur_m,
            (size_t) aligned_dim, medoids[cur_m]);
        if (cur_expanded_dist < best_dist) {
          best_medoid = medoids[cur_m];
          best_dist = cur_expanded_dist;
        }
      }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset[0].id = best_medoid;
    retset[0].flag = true;
    retset[0].distance = dist_scratch[0];
    visited.insert(best_medoid);

    unsigned cur_list_size = 1;

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    float                 accumulative_alpha = 0;
    std::vector<unsigned> filtered_nbrs;
    filtered_nbrs.reserve(this->max_degree);
    auto filter_nbrs = [&](_u64      nnbrs,
                           unsigned *node_nbrs) -> std::pair<_u64, unsigned *> {
      filtered_nbrs.clear();
      for (_u64 m = 0; m < nnbrs; ++m) {
        unsigned id = node_nbrs[m];
        if (visited.find(id) != visited.end()) {
          continue;
        }
        visited.insert(id);
        if (!bitset_view.empty() && bitset_view.test(id)) {
          accumulative_alpha += kAlpha;
          if (accumulative_alpha < 1.0f) {
            continue;
          }
          accumulative_alpha -= 1.0f;
        }
        cmps++;
        filtered_nbrs.push_back(id);
      }
      return {filtered_nbrs.size(), filtered_nbrs.data()};
    };

    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;
      // find new beam
      _u32 marker = k;
      _u32 num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width &&
             num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          {
            std::shared_lock<std::shared_mutex> lock(this->cache_mtx);
            auto iter = nhood_cache.find(retset[marker].id);
            if (iter != nhood_cache.end()) {
              cached_nhoods.push_back(
                  std::make_pair(retset[marker].id, iter->second));
              if (stats != nullptr) {
                stats->n_cache_hits++;
              }
            } else {
              frontier.push_back(retset[marker].id);
            }
          }
          retset[marker].flag = false;
          {
            std::shared_lock<std::shared_mutex> lock(
                this->node_visit_counter_mtx);
            if (this->count_visited_nodes) {
              this->node_visit_counter[retset[marker].id].second->fetch_add(1);
            }
          }
          if (!bitset_view.empty() && bitset_view.test(retset[marker].id)) {
            std::memmove(&retset[marker], &retset[marker + 1],
                         (cur_list_size - marker - 1) * sizeof(Neighbor));
            cur_list_size--;
          } else {
            marker++;
          }
        } else {
          marker++;
        }
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto                    id = frontier[i];
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second =
              sector_scratch + sector_scratch_idx * read_len_for_node;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(get_node_sector_offset(((size_t) id)),
                                          read_len_for_node, fnhood.second);
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
        reader->read(frontier_read_reqs, ctx);  // synchronous IO linux
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
      }

      auto process_node = [&](T *node_fp_coords_copy, auto node_id, auto n_nbr,
                              auto *nbrs) {
        if (bitset_view.empty() || !bitset_view.test(node_id)) {
          float cur_expanded_dist;
          if (!use_disk_index_pq) {
            cur_expanded_dist = dist_cmp_wrap(query, node_fp_coords_copy,
                                              (size_t) aligned_dim, node_id);
          } else {
            if (metric == diskann::Metric::INNER_PRODUCT ||
                metric == diskann::Metric::COSINE)
              cur_expanded_dist = disk_pq_table.inner_product(
                  query_float, (_u8 *) node_fp_coords_copy);
            else
              cur_expanded_dist = disk_pq_table.l2_distance(
                  query_float, (_u8 *) node_fp_coords_copy);
          }
          full_retset.push_back(
              Neighbor((unsigned) node_id, cur_expanded_dist, true));

          // add top candidate info into feder result
          if (feder != nullptr) {
            feder->visit_info_.AddTopCandidateInfo(node_id, cur_expanded_dist);
            feder->id_set_.insert(node_id);
          }
        }
        auto [nnbrs, node_nbrs] = filter_nbrs(n_nbr, nbrs);

        // compute node_nbrs <-> query dists in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetched nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];

          // add neighbor info into feder result
          if (feder != nullptr) {
            feder->visit_info_.AddTopCandidateNeighbor(node_id, id,
                                                       dist_scratch[m]);
            feder->id_set_.insert(id);
          }

          float dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (cur_list_size > 0 && dist >= retset[cur_list_size - 1].distance &&
              (cur_list_size == l_search))
            continue;
          Neighbor nn(id, dist, true);
          // Return position in sorted list where nn inserted.
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search)
            ++cur_list_size;
          if (r < nk)
            // nk logs the best position in the retset that was
            // updated due to neighbors of n.
            nk = r;
        }
        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      };

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        if (stats != nullptr) {
          stats->n_hops++;
        }
        T *node_fp_coords_copy;
        {
          std::shared_lock<std::shared_mutex> lock(this->cache_mtx);
          auto global_cache_iter = coord_cache.find(cached_nhood.first);
          node_fp_coords_copy = global_cache_iter->second;
        }
        process_node(node_fp_coords_copy, cached_nhood.first,
                     cached_nhood.second.first, cached_nhood.second.second);
      }

      for (auto &frontier_nhood : frontier_nhoods) {
        char *node_disk_buf =
            get_offset_to_node(frontier_nhood.second, frontier_nhood.first);
        unsigned *node_buf = OFFSET_TO_NODE_NHOOD(node_disk_buf);
        T        *node_fp_coords = OFFSET_TO_NODE_COORDS(node_disk_buf);
        T        *node_fp_coords_copy = data_buf;
        memcpy(node_fp_coords_copy, node_fp_coords, disk_bytes_per_point);
        process_node(node_fp_coords_copy, frontier_nhood.first, *node_buf,
                     node_buf + 1);
      }

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) {
                return left.distance < right.distance;
              });

    if (use_reorder_data) {
      if (!(this->reorder_data_exists)) {
        throw ANNException(
            "Requested use of reordering data which does not exist in index "
            "file",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }

      std::vector<AlignedRead> vec_read_reqs;

      if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
        full_retset.erase(
            full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
            full_retset.end());

      for (size_t i = 0; i < full_retset.size(); ++i) {
        vec_read_reqs.emplace_back(
            VECTOR_SECTOR_NO(((size_t) full_retset[i].id)) * SECTOR_LEN,
            SECTOR_LEN, sector_scratch + i * SECTOR_LEN);

        if (stats != nullptr) {
          stats->n_4k++;
          stats->n_ios++;
        }
      }

      io_timer.reset();
      reader->read(vec_read_reqs, ctx);  // synchronous IO linux
      if (stats != nullptr) {
        stats->io_us += io_timer.elapsed();
      }

      for (size_t i = 0; i < full_retset.size(); ++i) {
        auto id = full_retset[i].id;
        auto location =
            (sector_scratch + i * SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
        full_retset[i].distance =
            dist_cmp_wrap(query, (T *) location, this->data_dim, id);
      }

      std::sort(full_retset.begin(), full_retset.end(),
                [](const Neighbor &left, const Neighbor &right) {
                  return left.distance < right.distance;
                });
    }
    // copy k_search values
    for (_u64 i = 0; i < k_search; i++) {
      if (i >= full_retset.size()) {
        indices[i] = -1;
        if (distances != nullptr) {
          distances[i] = -1;
        }
        continue;
      }
      indices[i] = full_retset[i].id;
      if (distances != nullptr) {
        distances[i] = full_retset[i].distance;
        if (metric == diskann::Metric::INNER_PRODUCT) {
          // convert l2 distance to ip distance
          distances[i] = 1.0 - distances[i] / 2.0;
          // rescale to revert back to original norms (cancelling the effect of
          // base and query pre-processing)
          if (max_base_norm != 0)
            distances[i] *= (max_base_norm * query_norm);
        } else if (metric == diskann::Metric::COSINE) {
          distances[i] = -distances[i];
        }
      }
    }
    if (k_search > 0 && indices[0] != -1) {
      lru_cache.put(vec_hash, indices[0]);
    }

    this->thread_data.push(data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);
    // std::cout << num_ios << " " <<stats << std::endl;

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    if (this->count_visited_nodes) {
      this->search_counter.fetch_add(1);
    }
  }

  // range search returns results of all neighbors within distance of range.
  // indices and distances need to be pre-allocated of size l_search and the
  // return value is the number of matching hits.
  template<typename T>
  _u32 PQFlashIndex<T>::range_search(
      const T *query1, const double range, const _u64 min_l_search,
      const _u64 max_l_search, std::vector<_s64> &indices,
      std::vector<float> &distances, const _u64 beam_width,
      knowhere::BitsetView bitset_view, QueryStats *stats) {
    _u32 res_count = 0;

    bool stop_flag = false;

    _u32 l_search = min_l_search;  // starting size of the candidate list
    while (!stop_flag) {
      if (stats != nullptr) {
        stats->n_iters++;
      }
      indices.resize(l_search);
      distances.resize(l_search);
      for (auto &x : distances)
        x = std::numeric_limits<float>::max();
      this->cached_beam_search(query1, l_search, l_search, indices.data(),
                               distances.data(), beam_width, false, stats,
                               nullptr, bitset_view);
      for (_u32 i = 0; i < l_search; i++) {
        if (indices[i] == -1) {
          res_count = i;
          break;
        }
        bool in_range = (metric == diskann::Metric::INNER_PRODUCT ||
                         metric == diskann::Metric::COSINE)
                            ? distances[i] > (float) range
                            : distances[i] < (float) range;
        if (!in_range) {
          res_count = i;
          break;
        }
        if (i == l_search - 1) {
          res_count = l_search;
        }
      }
      if (res_count < (_u32) (l_search / 2.0))
        stop_flag = true;
      l_search = l_search * 2;
      if (l_search > max_l_search)
        stop_flag = true;
    }
    indices.resize(res_count);
    distances.resize(res_count);
    return res_count;
  }

  template<typename T>
  inline void PQFlashIndex<T>::copy_vec_base_data(T *des, const int64_t des_idx,
                                                  void *src) {
    if (metric == Metric::INNER_PRODUCT) {
      assert(max_base_norm != 0);
      const auto original_dim = data_dim - 1;
      memcpy(des + des_idx * original_dim, src, original_dim * sizeof(T));
      for (size_t i = 0; i < original_dim; ++i) {
        des[des_idx * original_dim + i] =
            (T) (max_base_norm * (float) des[des_idx * original_dim + i]);
      }
    } else {
      memcpy(des + des_idx * data_dim, src, data_dim * sizeof(T));
    }
  }

  template<typename T>
  std::unordered_map<_u64, std::vector<_u64>>
  PQFlashIndex<T>::get_sectors_layout_and_write_data_from_cache(
      const int64_t *ids, int64_t n, T *output_data) {
    std::unordered_map<_u64, std::vector<_u64>> sectors_to_visit;
    for (int64_t i = 0; i < n; ++i) {
      _u64 id = ids[i];
      {
        std::shared_lock<std::shared_mutex> lock(this->cache_mtx);
        if (coord_cache.find(id) != coord_cache.end()) {
          copy_vec_base_data(output_data, i, coord_cache.at(id));
        } else {
          const _u64 sector_offset = get_node_sector_offset(id);
          sectors_to_visit[sector_offset].push_back(i);
        }
      }
    }
    return sectors_to_visit;
  }

  template<typename T>
  void PQFlashIndex<T>::get_vector_by_ids(const int64_t *ids, const int64_t n,
                                          T *output_data) {
    auto sectors_to_visit =
        get_sectors_layout_and_write_data_from_cache(ids, n, output_data);
    if (0 == sectors_to_visit.size()) {
      return;
    }

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
      this->thread_data.wait_for_push_notify();
      data = this->thread_data.pop();
    }

    const size_t batch_size =
        std::min(AioContextPool::GetGlobalAioPool()->max_events_per_ctx(),
                 std::min(MAX_N_SECTOR_READS / 2UL, sectors_to_visit.size()));
    const size_t half_buf_idx = MAX_N_SECTOR_READS / 2 * read_len_for_node;
    char        *sector_scratch = data.scratch.sector_scratch;
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(batch_size);

    std::vector<_u64> sector_offsets;
    sector_offsets.reserve(sectors_to_visit.size());
    for (const auto &it : sectors_to_visit) {
      sector_offsets.emplace_back(it.first);
    }

    auto                     ctx = this->reader->get_ctx();
    const auto               sector_num = sector_offsets.size();
    const _u64               num_blocks = DIV_ROUND_UP(sector_num, batch_size);
    std::vector<AlignedRead> last_reqs;
    bool                     rotate = false;

    for (_u64 i = 0; i < num_blocks; ++i) {
      _u64 start_idx = i * batch_size;
      _u64 idx_len = std::min(batch_size, sector_num - start_idx);
      last_reqs = frontier_read_reqs;
      frontier_read_reqs.clear();
      for (_u64 j = 0; j < idx_len; ++j) {
        char *sector_buf =
            sector_scratch + rotate * half_buf_idx + j * read_len_for_node;
        frontier_read_reqs.emplace_back(sector_offsets[start_idx + j],
                                        read_len_for_node, sector_buf);
      }
      rotate ^= 0x1;
      reader->submit_req(ctx, frontier_read_reqs);
      for (const auto &req : last_reqs) {
        auto  offset = req.offset;
        char *sector_buf = static_cast<char *>(req.buf);
        for (auto idx : sectors_to_visit.at(offset)) {
          char *node_buf = get_offset_to_node(sector_buf, ids[idx]);
          copy_vec_base_data(output_data, idx, node_buf);
        }
      }
      reader->get_submitted_req(ctx, frontier_read_reqs.size());
    }

    // if any remaining
    for (const auto &req : frontier_read_reqs) {
      auto  offset = req.offset;
      char *sector_buf = static_cast<char *>(req.buf);
      for (auto idx : sectors_to_visit.at(offset)) {
        char *node_buf = get_offset_to_node(sector_buf, ids[idx]);
        copy_vec_base_data(output_data, idx, node_buf);
      }
    }

    this->reader->put_ctx(ctx);
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
  }

  template<typename T>
  _u64 PQFlashIndex<T>::get_num_points() const noexcept {
    return num_points;
  }

  template<typename T>
  _u64 PQFlashIndex<T>::get_data_dim() const noexcept {
    return data_dim;
  }

  template<typename T>
  _u64 PQFlashIndex<T>::get_max_degree() const noexcept {
    return max_degree;
  }

  template<typename T>
  _u32 *PQFlashIndex<T>::get_medoids() const noexcept {
    return medoids.get();
  }

  template<typename T>
  size_t PQFlashIndex<T>::get_num_medoids() const noexcept {
    return num_medoids;
  }

  template<typename T>
  diskann::Metric PQFlashIndex<T>::get_metric() const noexcept {
    return metric;
  }

  template<typename T>
  _u64 PQFlashIndex<T>::cal_size() {
    _u64 index_mem_size = 0;
    index_mem_size += sizeof(*this);
    // thread data size:
    index_mem_size += (_u64) this->thread_data.size() * get_thread_data_size();
    // get cache size:
    auto num_cached_nodes = coord_cache.size();
    index_mem_size +=
        ROUND_UP(num_cached_nodes * aligned_dim * sizeof(T), 8 * sizeof(T));
    index_mem_size += num_cached_nodes * (max_degree + 1) * sizeof(unsigned);
    index_mem_size += coord_cache.size() * sizeof(std::pair<_u32, T *>);
    index_mem_size +=
        nhood_cache.size() * sizeof(std::pair<_u32, std::pair<_u32, _u32 *>>);
    // get entry points:
    index_mem_size += ROUND_UP(num_medoids * aligned_dim * sizeof(float), 32);
    index_mem_size += num_medoids * aligned_dim * sizeof(uint32_t);
    // get pq data and pq_table:
    index_mem_size += this->num_points * this->n_chunks * sizeof(uint8_t);
    index_mem_size += this->pq_table.get_total_dims() * 256 * sizeof(float) * 2;
    index_mem_size +=
        this->pq_table.get_total_dims() * (sizeof(uint32_t) + sizeof(float));
    index_mem_size += (this->pq_table.get_num_chunks() + 1) * sizeof(uint32_t);
    // base norms:
    if (this->metric == diskann::Metric::COSINE) {
      index_mem_size += sizeof(float) * this->num_points;
    }

    return index_mem_size;
  }

  template<typename T>
  void PQFlashIndex<T>::destroy_cache_async_task() {
    std::unique_lock<std::mutex> guard(state_controller->status_mtx);
    if (this->state_controller->status.load() ==
        ThreadSafeStateController::Status::DONE) {
      return;
    }
    if (this->state_controller->status.load() ==
        ThreadSafeStateController::Status::NONE) {
      this->state_controller->status.store(
          ThreadSafeStateController::Status::KILLED);
      return;
    }
    this->state_controller->status.store(
        ThreadSafeStateController::Status::STOPPING);
    if (this->state_controller->status.load() !=
        ThreadSafeStateController::Status::DONE) {
      this->state_controller->cond.wait(guard);
    }
  }

  // knowhere not support uint8/int8 diskann
  // template class PQFlashIndex<_u8>;
  // template class PQFlashIndex<_s8>;
  template class PQFlashIndex<float>;
  template class PQFlashIndex<knowhere::fp16>;
  template class PQFlashIndex<knowhere::bf16>;

}  // namespace diskann
