// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <cassert>
#include <functional>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <stack>
#include <string>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include <unordered_map>

#include "concurrent_queue.h"
#include "defaults.h"
#include "distance.h"
#include "knowhere/comp/thread_pool.h"
#include "neighbor.h"
#include "parameters.h"
#include "utils.h"

#define OVERHEAD_FACTOR 1.1

namespace boost {
#ifndef BOOST_DYNAMIC_BITSET_FWD_HPP
  template<typename Block = unsigned long,
           typename Allocator = std::allocator<Block>>
  class dynamic_bitset;
#endif
}  // namespace boost

namespace diskann {
  inline double estimate_ram_usage(_u64 size, _u32 dim, _u32 datasize,
                                   _u32 degree) {
    double size_of_data = ((double) size) * ROUND_UP(dim, 8) * datasize;
    double size_of_graph =
        ((double) size) * degree * sizeof(unsigned) * diskann::defaults::GRAPH_SLACK_FACTOR;
    double size_of_locks = ((double) size) * sizeof(std::mutex);
    double size_of_outer_vector = ((double) size) * sizeof(ptrdiff_t);

    return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks +
                              size_of_outer_vector);
  }

  template<typename T>
  struct InMemQueryScratch {
    std::vector<Neighbor>    *_pool = nullptr;
    tsl::robin_set<unsigned> *_visited = nullptr;
    std::vector<unsigned>    *_des = nullptr;
    std::vector<Neighbor>    *_best_l_nodes = nullptr;
    tsl::robin_set<unsigned> *_inserted_into_pool_rs = nullptr;
    boost::dynamic_bitset<>  *_inserted_into_pool_bs = nullptr;

    T        *aligned_query = nullptr;
    uint32_t *indices = nullptr;
    float    *interim_dists = nullptr;

    uint32_t search_l;
    uint32_t indexing_l;
    uint32_t r;

    InMemQueryScratch();
    void setup(uint32_t search_l, uint32_t indexing_l, uint32_t r, size_t dim);
    void clear();
    void resize_for_query(uint32_t new_search_l);
    void destroy();

    std::vector<Neighbor> &pool() {
      return *_pool;
    }
    std::vector<unsigned> &des() {
      return *_des;
    }
    tsl::robin_set<unsigned> &visited() {
      return *_visited;
    }
    std::vector<Neighbor> &best_l_nodes() {
      return *_best_l_nodes;
    }
    tsl::robin_set<unsigned> &inserted_into_pool_rs() {
      return *_inserted_into_pool_rs;
    }
    boost::dynamic_bitset<> &inserted_into_pool_bs() {
      return *_inserted_into_pool_bs;
    }
  };

  template<typename T, typename TagT = uint32_t>
  class Index {
   public:
    // Constructor for Bulk operations and for creating the index object solely
    // for loading a prexisting index.
    Index(Metric m, bool ip_prepared, const size_t dim, const size_t max_points,
          const bool dynamic_index, const bool enable_tags = false,
          const bool support_eager_delete = false);

    // Constructor for incremental index
    Index(Metric m, bool ip_prepared, const size_t dim, const size_t max_points,
          const bool dynamic_index, const Parameters &indexParameters,
          const Parameters &searchParameters, const bool enable_tags = false,
          const bool support_eager_delete = false);

    ~Index();

    // Public Functions for Static Support

    // checks if data is consolidated, saves graph, metadata and associated
    // tags.
    void save(const char *filename, const bool disk_index = false);
    _u64 save_graph(std::string filename);
    _u64 save_data(std::string filename);
    _u64 save_tags(std::string filename);
    _u64 save_delete_list(const std::string &filename);

    void load(const char *index_file, uint32_t num_threads, uint32_t search_l);

    size_t load_graph(const std::string filename, size_t expected_num_points);

    size_t load_data(std::string filename0);

    size_t load_tags(const std::string tag_file_name);
    size_t load_delete_set(const std::string &filename);

    size_t get_num_points();

    size_t return_max_points();

    void build(const char *filename, const size_t num_points_to_load,
               Parameters              &parameters,
               const std::vector<TagT> &tags = std::vector<TagT>());

    void build(const char *filename, const size_t num_points_to_load,
               Parameters &parameters, const char *tag_filename);

    // Added search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    template<typename IDType>
    std::pair<uint32_t, uint32_t> search(const T *query, const size_t K,
                                         const unsigned L, IDType *indices,
                                         float *distances = nullptr);

    size_t search_with_tags(const T *query, const uint64_t K, const unsigned L,
                            TagT *tags, float *distances,
                            std::vector<T *> &res_vectors);

    void clear_index();

    // Public Functions for Incremental Support

    // insertions possible only when id corresponding to tag does not already
    // exist in the graph
    int insert_point(const T *point, const TagT tag);

    // call before triggering deleteions - sets important flags required for
    // deletion related operations
    int enable_delete();

    // call after all delete requests have been served, checks if deletions were
    // executed correctly, rearranges metadata in case of lazy deletes
    int disable_delete(const Parameters &parameters,
                       const bool        consolidate = false);

    // Record deleted point now and restructure graph later. Return -1 if tag
    // not found, 0 if OK. Do not call if _eager_delete was called earlier and
    // data was not consolidated
    int lazy_delete(const TagT &tag);

    // Record deleted points now and restructure graph later. Add to failed_tags
    // if tag not found. Do not call if _eager_delete was called earlier and
    // data was not consolidated. Return -1 if
    int lazy_delete(const tsl::robin_set<TagT> &tags,
                    std::vector<TagT>          &failed_tags);

    // Delete point from graph and restructure it immediately. Do not call if
    // _lazy_delete was called earlier and data was not consolidated
    int eager_delete(const TagT tag, const Parameters &parameters,
                     int delete_mode = 1);
    // return _data and tag_to_location offset
    int extract_data(T                                  *ret_data,
                     std::unordered_map<TagT, unsigned> &tag_to_location);

    void get_location_to_tag(
        std::unordered_map<unsigned, TagT> &ret_loc_to_tag);

    void prune_all_nbrs(const Parameters &parameters);

    void compact_data_for_insert();

    bool                                      hasIndexBeenSaved();
    const std::vector<std::vector<unsigned>> *get_graph() const {
      return &this->_final_graph;
    }

    unsigned get_entry_point() {
      return _ep;
    }

    T                                        *get_data();
    const std::unordered_map<unsigned, TagT> *get_tags() const {
      return &this->_location_to_tag;
    };
    // repositions frozen points to the end of _data - if they have been moved
    // during deletion
    void reposition_frozen_point_to_end();
    void reposition_point(unsigned old_location, unsigned new_location);

    void compact_frozen_point();
    void compact_data_for_search();

    void consolidate(Parameters &parameters);

    // void save_index_as_one_file(bool flag);

    void get_active_tags(tsl::robin_set<TagT> &active_tags);

    int      get_vector_by_tag(TagT &tag, T *vec);
    const T *get_vector_by_tag(const TagT &tag);

    void print_status() const;

    // This variable MUST be updated if the number of entries in the metadata
    // change.
    static const int METADATA_ROWS = 5;

    // For Bulk Index FastL2 search, we interleave the data with graph
    void optimize_index_layout();

    // For FastL2 search on optimized layout
    void search_with_optimized_layout(const T *query, size_t K, size_t L,
                                      unsigned *indices);

    /*  Internals of the library */
   protected:
    // No copy/assign.
    Index(const Index<T, TagT> &) = delete;
    Index<T, TagT> &operator=(const Index<T, TagT> &) = delete;

    std::vector<std::vector<unsigned>> _final_graph;
    std::vector<std::vector<unsigned>> _in_graph;

    // generates one frozen point that will never get deleted from the
    // graph
    int generate_frozen_point();

    // determines navigating node of the graph by calculating medoid of data
    unsigned calculate_entry_point();
    // called only when _eager_delete is to be supported
    void update_in_graph();

    template<typename IDType>
    std::pair<uint32_t, uint32_t> search_impl(const T *query, const size_t K,
                                              const unsigned L, IDType *indices,
                                              float                *distances,
                                              InMemQueryScratch<T> &scratch);

    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(
        const T *node_coords, const unsigned Lindex,
        const std::vector<unsigned> &init_ids,
        std::vector<Neighbor>       &expanded_nodes_info,
        tsl::robin_set<unsigned>    &expanded_nodes_ids,
        std::vector<Neighbor> &best_L_nodes, std::vector<unsigned> &des,
        tsl::robin_set<unsigned> &inserted_into_pool_rs,
        boost::dynamic_bitset<> &inserted_into_pool_bs, bool ret_frozen = true,
        bool search_invocation = false);
    void get_expanded_nodes(const size_t node, const unsigned Lindex,
                            std::vector<unsigned>     init_ids,
                            std::vector<Neighbor>    &expanded_nodes_info,
                            tsl::robin_set<unsigned> &expanded_nodes_ids,
                            std::vector<unsigned>    &des,
                            std::vector<Neighbor>    &best_L_nodes,
                            tsl::robin_set<unsigned> &inserted_into_pool_rs,
                            boost::dynamic_bitset<>  &inserted_into_pool_bs);

    // get_expanded_nodes for insertion. Must investigate to see if perf can
    // be improved here as well using the same technique as above.
    void get_expanded_nodes(const size_t node_id, const unsigned Lindex,
                            std::vector<unsigned>     init_ids,
                            std::vector<Neighbor>    &expanded_nodes_info,
                            tsl::robin_set<unsigned> &expanded_nodes_ids);

    void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                         std::vector<unsigned> &pruned_list);

    void prune_neighbors(const unsigned location, std::vector<Neighbor> &pool,
                         const _u32 range, const _u32 max_candidate_size,
                         const float alpha, std::vector<unsigned> &pruned_list);

    void occlude_list(std::vector<Neighbor> &pool, const float alpha,
                      const unsigned degree, const unsigned maxc,
                      std::vector<Neighbor> &result);

    void occlude_list(std::vector<Neighbor> &pool, const float alpha,
                      const unsigned degree, const unsigned maxc,
                      std::vector<Neighbor> &result,
                      std::vector<float>    &occlude_factor);

    void batch_inter_insert(unsigned                     n,
                            const std::vector<unsigned> &pruned_list,
                            const _u32                   range,
                            std::vector<unsigned>       &need_to_sync);

    void batch_inter_insert(unsigned                     n,
                            const std::vector<unsigned> &pruned_list,
                            std::vector<unsigned>       &need_to_sync);

    void inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                      const _u32 range, bool update_in_graph);

    void inter_insert(unsigned n, std::vector<unsigned> &pruned_list,
                      bool update_in_graph);

    void link(Parameters &parameters);

    // WARNING: Do not call reserve_location() without acquiring change_lock_
    int  reserve_location();
    void release_location();

    // Support for resizing the index
    // This function must be called ONLY after taking the _change_lock and
    // _update_lock. Anything else in a MT environment will lead to an
    // inconsistent index.
    void resize(size_t new_max_points);

    /*    // get new location corresponding to each undeleted tag after
       deletions std::vector<unsigned> get_new_location(unsigned &active);*/

    // renumber nodes, update tag and location maps and compact the graph, mode
    // = _consolidated_order in case of lazy deletion and _compacted_order in
    // case of eager deletion
    void compact_data();

    // WARNING: Do not call consolidate_deletes without acquiring change_lock_
    // Returns number of live points left after consolidation
    size_t consolidate_deletes(const Parameters &parameters);

    void initialize_query_scratch(uint32_t num_threads, uint32_t search_l,
                                  uint32_t indexing_l, uint32_t r, size_t dim);

   private:
    Metric     _dist_metric = diskann::L2;
    size_t     _dim = 0;
    size_t     _padding_id = 0;
    size_t     _aligned_dim = 0;
    T         *_data = nullptr;
    size_t     _nd = 0;  // number of active points i.e. existing in the graph
    size_t     _max_points = 0;  // total number of points in given data set
    size_t     _num_frozen_pts = 0;
    bool       _has_built = false;
    DISTFUN<T> _func = nullptr;
    DISTFUN<T> _distance;
    unsigned                                       _width = 0;
    unsigned                                       _ep = 0;
    size_t _max_range_of_loaded_graph = 0;
    bool   _saturate_graph = false;
    bool _save_as_one_file = false;  // plan to support in next version to avoid
                                     // too many files per index
    bool _dynamic_index = false;
    bool _enable_tags = false;
    // Using normalied L2 for cosine.
    bool _normalize_vecs = false;

    // Indexing parameters
    uint32_t _indexingQueueSize, _indexingRange, _indexingMaxC;
    float    _indexingAlpha;
    uint32_t _search_queue_size;

    // Query scratch data structures
    ConcurrentQueue<InMemQueryScratch<T>> _query_scratch;

    // flags for dynamic indexing
    std::unordered_map<TagT, unsigned> _tag_to_location;
    std::unordered_map<unsigned, TagT> _location_to_tag;

    tsl::robin_set<unsigned> _delete_set;
    tsl::robin_set<unsigned> _empty_slots;

    bool _support_eager_delete =
        false;  //_support_eager_delete = activates extra data

    bool _eager_done = false;     // true if eager deletions have been made
    bool _lazy_done = false;      // true if lazy deletions have been made
    bool _data_compacted = true;  // true if data has been consolidated
    bool _is_saved = false;  // Gopal. Checking if the index is already saved.

    std::vector<std::mutex> _locks;  // Per node lock, cardinality=max_points_
    std::shared_timed_mutex _tag_lock;  // reader-writer lock on
                                        // _tag_to_location and
    std::mutex _change_lock;  // Lock taken to synchronously modify _nd
    std::vector<std::mutex> _locks_in;     // Per node lock
    std::shared_timed_mutex _delete_lock;  // Lock on _delete_set and
                                           // _empty_slots when reading and
                                           // writing to them

    // _location_to_tag, has a shared lock
    // and exclusive lock associated with
    // it.
    std::shared_timed_mutex _update_lock;  // coordinate save() and any change
                                           // being done to the graph.
    static const float INDEX_GROWTH_FACTOR;

    char  *_opt_graph;
    size_t _node_size;
    size_t _data_len;
    size_t _neighbor_len;

    std::shared_ptr<knowhere::ThreadPool> _build_thread_pool;
    std::shared_ptr<knowhere::ThreadPool> _search_thread_pool;
  };
}  // namespace diskann
