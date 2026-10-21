// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cassert>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <sstream>
#include <stack>
#include <string>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include "knowhere/bitsetview.h"
#include "knowhere/feder/DiskANN.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "navigation_runtime.h"
#include "utils.h"
#include "diskann/distance.h"
#include "knowhere/thread_pool.h"

#include "defaults.h"

//#define MAX_GRAPH_DEGREE defaults::MAX_GRAPH_DEGREE
//#define SECTOR_LEN defaults::SECTOR_LEN

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann {
  class ThreadSafeStateController {
   public:
    enum class Status {
      NONE,
      DOING,
      STOPPING,
      DONE,
      KILLED,
    };

    std::atomic<Status>     status;
    std::condition_variable cond;
    std::mutex              status_mtx;
  };
  template<typename T>
  struct QueryScratch {
    T *coord_scratch = nullptr;  // MUST BE AT LEAST [sizeof(T) * data_dim]

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    NavigationDistanceComputer *navigation = nullptr;
    float                      *aligned_dist_scratch = nullptr;
    T     *aligned_query_T = nullptr;
    float *aligned_query_float = nullptr;

    tsl::robin_set<_u64> *visited = nullptr;

    void reset() {
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
    }
    
    };

  template<typename T>
  struct ThreadData {
    QueryScratch<T> scratch;
  };

  /** Algorithm Introduction for diskann-iterator
   * First, two unbounded min-heaps are maintained: `retset` and `candidates`,
   * sorted by *pq_dist*. (Similar to the navigation search path of hnswlib-hnsw
   * with ef=1.)
   *
   * When visiting candidates, another unbounded minimum heap, `full_retset`, is
   * maintained to collect all *full_dist* along the paths.
   * Each `iterator->next` is called, `candidates` are continually visited until
   * the optimal candidates are farther than the optimal retset.
   * "candidates.top.pq_dist < retset.top.pq_dist"
   *
   * Once the condition is met, we consider the `retset` to have reached its
   * current optimal state, and we stop visiting new candidates. The top element
   * of `full_retset` is returned as the final result. At this point, the top of
   * `retset` has also completed its task and will be directly popped.
   *
   * It is important to note that "ef=1" is clearly insufficient in terms of
   * accuracy. To address this, the `lsearch` parameter is provided, which
   * allows the `workspace` to effectively make an *additional* `lsearch`
   * iterations each time `iterator->next` is called. Specifically, this means
   * "good_pq_res_count - next_count >= lsearch".
   * The additional results will be sorted and saved by the upper-level
   * `iterator.res_` through `next_batch` func with `backup_res`.
   */
  template<typename T>
  struct IteratorWorkspace {
    IteratorWorkspace(const T *query_data, const diskann::Metric metric,
                      const uint64_t aligned_dim, const uint64_t data_dim,
                      const float alpha, const uint64_t lsearch,
                      const uint64_t beam_width, const float filter_ratio,
                      const float                 max_base_norm,
                      const knowhere::BitsetView &bitset);

    ~IteratorWorkspace();

    bool is_good_pq_enough();

    bool has_candidates();

    bool should_visit_next_candidate();

    void insert_to_pq(unsigned id, float dist, bool valid);

    void insert_to_full(unsigned id, float dist);

    void pop_pq_retset();

    void move_full_retset_to_backup();

    void move_last_full_retset_to_backup();

    NavigationQuerySetup navigation_setup;

    // convert a full-precision distance to the value stored in `backup_res`
    float to_backup_dist(float dist) const;

    uint64_t q_dim = 0;
    uint64_t lsearch = 0;
    uint64_t beam_width = 0;
    float    filter_ratio = 0;
    Metric   metric = Metric::L2;
    float    alpha = 0;
    float    acc_alpha = 0;
    bool     initialized = false;

    T                    *aligned_query_T = nullptr;
    float                *aligned_query_float = nullptr;
    tsl::robin_set<_u64> *visited = nullptr;
    float                 query_norm = 0.0f;
    float                 max_base_norm = 0.0f;
    bool not_l2_but_zero = false;  // (cosine or ip) and query_norm == 0.
    std::vector<unsigned>      frontier;
    std::vector<AlignedRead>   frontier_read_reqs;
    const knowhere::BitsetView bitset;

    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;

    struct MinHeapCompareForSimpleNeighbor {
      bool operator()(const SimpleNeighbor &a, const SimpleNeighbor &b) {
        return a.distance > b.distance;
      }
    };
    std::priority_queue<SimpleNeighbor, std::vector<SimpleNeighbor>,
                        MinHeapCompareForSimpleNeighbor>
        full_retset;
    std::priority_queue<SimpleNeighbor, std::vector<SimpleNeighbor>,
                        MinHeapCompareForSimpleNeighbor>
        retset;
    std::priority_queue<SimpleNeighbor, std::vector<SimpleNeighbor>,
                        MinHeapCompareForSimpleNeighbor>
        candidates;

    size_t good_pq_res_count = 0;
    size_t next_count = 0;

    std::vector<knowhere::DistId> backup_res;

    // Brute-force mode, decided once on the first `getIteratorNextBatch` call
    // when the bitset filters out at least the iterator filter threshold.
    // `bf_candidates` is a min-heap of (navigation dist, id) over all
    // unfiltered points; each refine pops the next window of it and computes
    // the full-precision distances.
    bool                                    bf_mode = false;
    std::vector<std::pair<float, unsigned>> bf_candidates;
    size_t                                  bf_next_batch_calls = 0;
  };

  template<typename T>
  class SSDGraphIndex {
   public:
    SSDGraphIndex(std::shared_ptr<AlignedFileReader> fileReader,
                 diskann::Metric metric = diskann::Metric::L2);
    virtual ~SSDGraphIndex();

    // load compressed data, and obtains the handle to the disk-resident index
    int load(uint32_t num_threads, const char *index_prefix,
             std::unique_ptr<NavigationStore> navigation);

    virtual void load_cache_list(std::vector<uint32_t> &node_list);

    // asynchronously collect the access frequency of each node in the graph
    void async_generate_cache_list_from_sample_queries(std::string sample_bin,
                                                       _u64        l_search,
                                                       _u64        beamwidth,
                                                       _u64 num_nodes_to_cache);

    virtual void cache_bfs_levels(_u64                   num_nodes_to_cache,
                          std::vector<uint32_t> &node_list);

    void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _s64 *res_ids,
        float *res_dists, const _u64 beam_width,
        const bool use_reorder_data = false, QueryStats *stats = nullptr,
        const knowhere::feder::diskann::FederResultUniq &feder = nullptr,
        knowhere::BitsetView                             bitset_view = nullptr,
        const float                                      filter_ratio = -1.0f,
        const NavigationQuerySetup                      &navigation_setup = {});

    void calc_dist_by_ids(const T *query, const int64_t *ids, const int64_t n,
                          float *const output_dists);

    void get_vector_by_ids(const int64_t *ids, const int64_t n,
                           T *const output_data);

    std::shared_ptr<AlignedFileReader> reader;

    _u64 get_num_points() const noexcept;

    _u64 get_data_dim() const noexcept;

    _u64 get_max_degree() const noexcept;

    _u32 *get_medoids() const noexcept;

    size_t get_num_medoids() const noexcept;

    diskann::Metric get_metric() const noexcept;
    const char* get_navigation_codec() const noexcept { return navigation_->CodecName(); }
    bool is_navigation_compatible(knowhere::DataFormatEnum data_type,
                                  bool embedding_list) const {
      return navigation_->IsCompatible(data_type, embedding_list);
    }

    void getIteratorNextBatch(IteratorWorkspace<T> *workspace);

    std::unique_ptr<IteratorWorkspace<T>> getIteratorWorkspace(
        const T *query_data, const uint64_t lsearch, const uint64_t beam_width,
        const float filter_ratio, const knowhere::BitsetView &bitset);

    _u64 cal_size();

    // for async cache making task
    void destroy_cache_async_task();

  protected:
    void use_medoids_data_as_centroids();
    void setup_thread_data(_u64 nthreads);
    void destroy_thread_data();
    _u64 get_thread_data_size();

   //private:
    // sector # on disk where node_id is present with in the graph part
    virtual _u64 get_node_sector_offset(_u64 node_id) {
      return long_node ? (node_id * nsectors_per_node + 1) * diskann::defaults::SECTOR_LEN
                       : (node_id / nnodes_per_sector + 1) * diskann::defaults::SECTOR_LEN;
    }

    // obtains region of sector containing node
    char *get_offset_to_node(char *sector_buf, _u64 node_id) {
      return long_node
                 ? sector_buf
                 : sector_buf + (node_id % nnodes_per_sector) * max_node_len;
    }

    void copy_vec_base_data(T *des, const int64_t des_idx, void *src);

    // Init thread data and returns query norm if avaialble.
    // If there is no value, there is nothing to do with the given query
    std::optional<float> init_thread_data(ThreadData<T> &data, const T *query1);

    // Brute force search for the given query. Use beam search rather than
    // sending whole bunch of requests at once to avoid all threads sending I/O
    // requests and the time overlaps.
    // The beam width is adjusted in the function.
    void brute_force_beam_search(
        ThreadData<T> &data, const float query_norm, const _u64 k_search,
        _s64 *indices, float *distances, const _u64 beam_width_param,
        IOContext &ctx, QueryStats *stats,
        const knowhere::feder::diskann::FederResultUniq &feder,
        knowhere::BitsetView                             bitset_view,
        NavigationDistanceComputer                      &navigation,
        const std::function<_u64(_u64)>                 &origin_id = {},
        bool query_initialized = false);

    // Brute-force counterpart of `getIteratorNextBatch` for highly filtered
    // queries. The first call ranks the unfiltered points by navigation
    // distance (memory only); each call then refines the next window from disk
    // if fewer than `lsearch` refined results are left unconsumed.
    void iterator_brute_force_next_batch(IteratorWorkspace<T> *workspace,
                                         ThreadData<T> &data, IOContext &ctx);

    // Assign the index of ids to its corresponding sector and if it is in
    // cache, write to the output_data
    std::unordered_map<_u64, std::vector<_u64>>
    get_sectors_layout_and_write_data_from_cache(const int64_t *ids, int64_t n,
                                                 T *output_data);

    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // Data used for searching with re-order vectors
    _u64 ndims_reorder_vecs = 0, reorder_data_start_sector = 0,
         nvecs_per_sector = 0;

    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float max_base_norm = 0.0f;

    // used only for cosine search to re-scale the caculated distance.
    std::unique_ptr<float[]> base_norms = nullptr;

    // data info
    bool long_node = false;
    _u64 nsectors_per_node = 0;
    _u64 read_len_for_node = diskann::defaults::SECTOR_LEN;
    _u64 num_points = 0;
    _u64 num_frozen_points = 0;
    _u64 frozen_location = 0;
    _u64 data_dim = 0;
    _u64 disk_data_dim = 0;  // will be different from data_dim only if we use
                             // PQ for disk data (very large dimensionality)
    _u64 aligned_dim = 0;
    _u64 disk_bytes_per_point = 0;

    std::string       disk_index_file;
    std::shared_mutex node_visit_counter_mtx;
    std::vector<std::pair<_u32, std::unique_ptr<std::atomic<_u32>>>>
                      node_visit_counter;
    std::atomic<_u32> search_counter = 0;

    std::shared_ptr<ThreadSafeStateController> state_controller =
        std::make_shared<ThreadSafeStateController>();

    // Runtime navigation is independent of the SSD payload encoding.
    std::unique_ptr<NavigationStore> navigation_;

    // distance comparator
    DISTFUN<T>     dist_cmp;
    DISTFUN<float> dist_cmp_float;

    float dist_cmp_wrap(const T *x, const T *y, size_t d, int32_t u) {
      if (metric == Metric::COSINE) {
        return dist_cmp(x, y, d) / base_norms[u];
      } else {
        return dist_cmp(x, y, d);
      }
    }

    float dist_cmp_float_wrap(const float *x, const float *y, size_t d,
                              int32_t u) {
      if (metric == Metric::COSINE) {
        return dist_cmp_float(x, y, d) / base_norms[u];
      } else {
        return dist_cmp_float(x, y, d);
      }
    }

    // for very large datasets: we use PQ even for the disk resident index
    bool              use_disk_index_pq = false;
    _u64              disk_pq_n_chunks = 0;
    FixedChunkPQTable disk_pq_table;

    // medoid/start info

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    std::unique_ptr<uint32_t[]> medoids = nullptr;
    // defaults to 1
    size_t num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *centroid_data = nullptr;

    // cache
    std::shared_mutex cache_mtx;

    // nhood_cache
    std::unique_ptr<unsigned[]> nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>>
        nhood_cache;  // <id, <neihbors_num, neihbors>>

    // coord_cache
    T                        *coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> coord_cache;

    // thread-specific scratch
    ConcurrentQueue<ThreadData<T>> thread_data;
    _u64                           max_nthreads;
    bool                           load_flag = false;
    std::atomic<bool>              count_visited_nodes = false;
    bool                           reorder_data_exists = false;
    _u64                           reoreder_data_offset = 0;
  };
}  // namespace diskann
