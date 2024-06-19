// TODOS
// ADD THE COPYRIGHT NOTICE
// ADD FAST L2 DISTANCE FUNCTION
// CHECK ALL DISTANCE FUNCTIONS ARE VALID (INNER PRODUCT, COSINE, L2, FAST L2)
// CHECK FOR BULK AND FRESH
// CHECK FOR FLOAT, INT8 and UINT8

#include <fcntl.h>
#include <unordered_map>
#include <set>
#include <shared_mutex>
#include <string>
#include <type_traits>

#include "boost/dynamic_bitset.hpp"
#include "tsl/robin_set.h"

#include "diskann/ann_exception.h"
#include "diskann/common_includes.h"
#include "diskann/index.h"
#include "diskann/logger.h"
#include "diskann/parameters.h"
#include "diskann/partition_and_pq.h"
#include "diskann/timer.h"
#include "diskann/utils.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/log.h"

#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#if !defined(__ARM_NEON) || !defined(__aarch64__)
#include <xmmintrin.h>
#endif

#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace diskann {

  template<typename T>
  inline T diskann_max(T left, T right) {
    return left > right ? left : right;
  }
  // QueryScratch functions
  template<typename T>
  InMemQueryScratch<T>::InMemQueryScratch() {
    search_l = indexing_l = r = 0;
    // pointers are initialized in the header itself.
  }
  template<typename T>
  void InMemQueryScratch<T>::setup(uint32_t search_l, uint32_t indexing_l,
                                   uint32_t r, size_t dim) {
    if (search_l == 0 || indexing_l == 0 || r == 0 || dim == 0) {
      std::stringstream ss;
      ss << "In InMemQueryScratch, one of search_l = " << search_l
         << ", indexing_l = " << indexing_l << ", dim = " << dim
         << " or r = " << r << " is zero." << std::endl;
      throw diskann::ANNException(ss.str(), -1);
    }
    indices = new uint32_t[search_l];     // only used by search
    interim_dists = new float[search_l];  // only used by search
    memset(indices, 0, sizeof(uint32_t) * search_l);
    memset(interim_dists, 0, sizeof(float) * search_l);
    this->search_l = search_l;
    this->indexing_l = indexing_l;
    this->r = r;

    auto   aligned_dim = ROUND_UP(dim, 8);
    size_t allocSize = aligned_dim * sizeof(T);
    alloc_aligned(((void **) &aligned_query), allocSize, 8 * sizeof(T));
    memset((void *) aligned_query, 0, aligned_dim * sizeof(T));

    auto l_to_use = diskann_max(search_l, indexing_l);

    _des = new std::vector<unsigned>();
    _des->reserve(2 * r);
    _pool = new std::vector<Neighbor>();
    _pool->reserve(l_to_use * 10);
    _visited = new tsl::robin_set<unsigned>();
    _visited->reserve(l_to_use * 2);
    _best_l_nodes = new std::vector<Neighbor>();
    _best_l_nodes->resize(l_to_use + 1);
    _inserted_into_pool_rs = new tsl::robin_set<unsigned>();
    _inserted_into_pool_rs->reserve(l_to_use * 20);
    _inserted_into_pool_bs = new boost::dynamic_bitset<>();
  }

  template<typename T>
  void InMemQueryScratch<T>::clear() {
    memset(indices, 0, sizeof(uint32_t) * search_l);
    memset(interim_dists, 0, sizeof(float) * search_l);
    _pool->clear();
    _visited->clear();
    _des->clear();
    _inserted_into_pool_rs->clear();
    _inserted_into_pool_bs->reset();
    // Should not clear best_l_nodes because code expects its size > 0.
    //_best_l_nodes->clear();
    // aligned_query gets overwritten every time, so no need to memset it.
  }

  template<typename T>
  void InMemQueryScratch<T>::resize_for_query(uint32_t new_search_l) {
    if (search_l < new_search_l) {
      if (indices != nullptr) {
        delete[] indices;
      }
      indices = new uint32_t[new_search_l];

      if (interim_dists != nullptr) {
        delete[] interim_dists;
      }
      interim_dists = new float[new_search_l];
      search_l = new_search_l;
    }
  }

  template<typename T>
  void InMemQueryScratch<T>::destroy() {
    if (indices != nullptr) {
      delete[] indices;
      indices = nullptr;
    }
    if (interim_dists != nullptr) {
      delete[] interim_dists;
      interim_dists = nullptr;
    }
    if (_pool != nullptr) {
      delete _pool;
      _pool = nullptr;
    }
    if (_visited != nullptr) {
      delete _visited;
      _visited = nullptr;
    }
    if (_des != nullptr) {
      delete _des;
      _des = nullptr;
    }
    if (_best_l_nodes != nullptr) {
      delete _best_l_nodes;
      _best_l_nodes = nullptr;
    }
    if (aligned_query != nullptr) {
      aligned_free(aligned_query);
      aligned_query = nullptr;
    }

    if (_inserted_into_pool_rs != nullptr) {
      delete _inserted_into_pool_rs;
      _inserted_into_pool_rs = nullptr;
    }
    if (_inserted_into_pool_bs != nullptr) {
      delete _inserted_into_pool_bs;
      _inserted_into_pool_bs = nullptr;
    }

    search_l = indexing_l = r = 0;
  }

  // Class to avoid the hassle of pushing and popping the query scratch.
  template<typename T>
  class ScratchStoreManager {
   public:
    diskann::InMemQueryScratch<T>          _scratch;
    ConcurrentQueue<InMemQueryScratch<T>> &_query_scratch;
    ScratchStoreManager(ConcurrentQueue<InMemQueryScratch<T>> &query_scratch)
        : _query_scratch(query_scratch) {
      _scratch = query_scratch.pop();
      while (_scratch.indices == nullptr) {
        query_scratch.wait_for_push_notify();
        _scratch = query_scratch.pop();
      }
    }
    InMemQueryScratch<T> scratch_space() {
      return _scratch;
    }

    ~ScratchStoreManager() {
      // std::stringstream ss;
      // ss << "Thread id:" << std::this_thread::get_id()
      //   << ": Clearing scratch.pool with ptr: " << std::hex
      //   << (_u64) _scratch._pool << std::dec << std::endl;
      // std::cout << ss.str();
      _scratch.clear();
      _query_scratch.push(_scratch);
      _query_scratch.push_notify_all();
    }

   private:
    ScratchStoreManager(const ScratchStoreManager<T> &);
    ScratchStoreManager<T> &operator=(const ScratchStoreManager<T> &);
  };

  // Initialize an index with metric m, load the data of type T with filename
  // (bin), and initialize max_points
  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, bool ip_prepared, const size_t dim, const size_t max_points,
                        const bool dynamic_index, const Parameters &indexParams,
                        const Parameters &searchParams, const bool enable_tags,
                        const bool support_eager_delete)
      : Index(m, ip_prepared, dim, max_points, dynamic_index, enable_tags,
              support_eager_delete) {  // Thank you C++ 11!
    _indexingQueueSize = indexParams.Get<uint32_t>("L");
    _indexingRange = indexParams.Get<uint32_t>("R");
    _indexingMaxC = indexParams.Get<uint32_t>("C");
    _indexingAlpha = indexParams.Get<float>("alpha");

    _build_thread_pool = knowhere::ThreadPool::GetGlobalBuildThreadPool();
    _search_thread_pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();
    uint32_t search_l = searchParams.Get<uint32_t>("L");

    initialize_query_scratch(_search_thread_pool->size(), search_l, _indexingQueueSize,
                             _indexingRange, dim);
  }

  template<typename T, typename TagT>
  Index<T, TagT>::Index(Metric m, bool ip_prepared, const size_t dim, const size_t max_points,
                        const bool dynamic_index, const bool enable_tags,
                        const bool support_eager_delete)
      : _dist_metric(m), _dim(dim), _max_points(max_points),
        _dynamic_index(dynamic_index), _enable_tags(enable_tags),
        _support_eager_delete(support_eager_delete) {
    if (dynamic_index && !enable_tags) {
      throw diskann::ANNException(
          "ERROR: Eager Deletes must have Dynamic Indexing enabled.", -1,
          __FUNCSIG__, __FILE__, __LINE__);
      diskann::cerr
          << "WARNING: Dynamic Indices must have tags enabled. Auto-enabling."
          << std::endl;
      _enable_tags = true;
    }
    if (support_eager_delete && !dynamic_index) {
      diskann::cerr << "ERROR: Eager Deletes must have Dynamic Indexing "
                       "enabled. Exitting."
                    << std::endl;
      throw diskann::ANNException(
          "ERROR: Eager deletes are possible only if dynamic indexing is "
          "enabled. Exiting.",
          -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    // data is stored to _nd * aligned_dim matrix with necessary
    // zero-padding
    _aligned_dim = ROUND_UP(_dim, 8);

    if (dynamic_index) {
      _num_frozen_pts = 1;
    }
    // Sanity check. While logically it is correct, max_points ==0 causes
    // downstream problems.
    if (_max_points == 0) {
      _max_points = 1;
    }

    alloc_aligned(((void **) &_data),
                  (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T),
                  8 * sizeof(T));
    std::memset((void *) _data, 0,
                (_max_points + _num_frozen_pts) * _aligned_dim * sizeof(T));

    _ep = (unsigned) _max_points;

    _build_thread_pool = knowhere::ThreadPool::GetGlobalBuildThreadPool();
    _search_thread_pool = knowhere::ThreadPool::GetGlobalSearchThreadPool();

    //_final_graph.reserve(_max_points + _num_frozen_pts);
    _final_graph.resize(_max_points + _num_frozen_pts);

    if (_support_eager_delete) {
      _in_graph.reserve(_max_points + _num_frozen_pts);
      _in_graph.resize(_max_points + _num_frozen_pts);
    }

    this->_func = get_distance_function<T>(m);
    if (ip_prepared) {
        _padding_id = _dim - 1;
        this->_distance = [this](const T* x, const T* y, size_t n) -> float {
            auto ret = _func(x, y, n);
            return ret + 2*x[_padding_id]*y[_padding_id];
        };
    } else {
        this->_distance = _func;
    }

    _locks = std::vector<std::mutex>(_max_points + _num_frozen_pts);

    if (_support_eager_delete)
      _locks_in = std::vector<std::mutex>(_max_points + _num_frozen_pts);

    _width = 0;
  }

  template<typename T, typename TagT>
  Index<T, TagT>::~Index() {
    // Ensure that no other activity is happening before dtor()
    std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
    std::unique_lock<std::shared_timed_mutex> tul(_tag_lock);
    std::unique_lock<std::shared_timed_mutex> tdl(_delete_lock);

    for (auto &lock : _locks) {
      LockGuard lg(lock);
    }
    for (auto &lock : _locks_in) {
      LockGuard lg(lock);
    }

    if (this->_data != nullptr) {
      aligned_free(this->_data);
      this->_data = nullptr;
    }

    while (!_query_scratch.empty()) {
      auto val = _query_scratch.pop();
      while (val.indices == nullptr) {
        _query_scratch.wait_for_push_notify();
        val = _query_scratch.pop();
      }
      val.destroy();
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::initialize_query_scratch(uint32_t num_threads,
                                                uint32_t search_l,
                                                uint32_t indexing_l, uint32_t r,
                                                size_t dim) {
    for (uint32_t i = 0; i < num_threads; i++) {
      InMemQueryScratch<T> scratch;
      scratch.setup(search_l, indexing_l, r, dim);
      _query_scratch.push(scratch);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::clear_index() {
    memset((void*) _data, 0,
           _aligned_dim * (_max_points + _num_frozen_pts) * sizeof(T));
    _nd = 0;
    for (size_t i = 0; i < _final_graph.size(); i++)
      _final_graph[i].clear();

    _tag_to_location.clear();
    _location_to_tag.clear();

    // What about empty slots, delete_set, etc?
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_tags(std::string tags_file) {
    if (!_enable_tags) {
      LOG_KNOWHERE_DEBUG_ << "Not saving tags as they are not enabled.";
      return 0;
    }
    size_t tag_bytes_written;
    auto   tag_data = std::make_unique<TagT[]>(_nd + _num_frozen_pts);
    for (_u32 i = 0; i < _nd; i++) {
      if (_location_to_tag.find(i) != _location_to_tag.end()) {
        tag_data[i] = _location_to_tag[i];
      } else {
        // catering to future when tagT can be any type.
        std::memset((char *) &tag_data[i], 0, sizeof(TagT));
      }
    }
    if (_num_frozen_pts > 0) {
      std::memset((char *) &tag_data[_ep], 0, sizeof(TagT));
    }
    try {
      tag_bytes_written =
          save_bin<TagT>(tags_file, tag_data.get(), _nd + _num_frozen_pts, 1);
    } catch (std::system_error &e) {
      throw FileException(tags_file, e, __FUNCSIG__, __FILE__, __LINE__);
    }
    return tag_bytes_written;
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_data(std::string data_file) {
    return save_data_in_base_dimensions(data_file, _data, _nd + _num_frozen_pts,
                                        _dim, _aligned_dim);
  }

  // save the graph index on a file as an adjacency list. For each point,
  // first store the number of neighbors, and then the neighbor list (each as
  // 4 byte unsigned)
  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_graph(std::string graph_file) {
    std::ofstream out;
    open_file_to_write(out, graph_file);

    _u64 file_offset = 0;  // we will use this if we want
    out.seekp(file_offset, out.beg);
    _u64 index_size = 24;
    _u32 max_degree = 0;
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &_width, sizeof(unsigned));
    unsigned ep_u32 = _ep;
    out.write((char *) &ep_u32, sizeof(unsigned));
    out.write((char *) &_num_frozen_pts, sizeof(_u64));
    for (unsigned i = 0; i < _nd + _num_frozen_pts; i++) {
      unsigned GK = (unsigned) _final_graph[i].size();
      out.write((char *) &GK, sizeof(unsigned));
      out.write((char *) _final_graph[i].data(), GK * sizeof(unsigned));
      max_degree = _final_graph[i].size() > max_degree
                       ? (_u32) _final_graph[i].size()
                       : max_degree;
      index_size += (_u64) (sizeof(unsigned) * (GK + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char *) &index_size, sizeof(uint64_t));
    out.write((char *) &max_degree, sizeof(_u32));
    out.close();
    return index_size;  // number of bytes written
  }

  template<typename T, typename TagT>
  _u64 Index<T, TagT>::save_delete_list(const std::string &filename) {
    if (_delete_set.size() == 0) {
      return 0;
    }
    std::unique_ptr<_u32[]> delete_list =
        std::make_unique<_u32[]>(_delete_set.size());
    _u32 i = 0;
    for (auto &del : _delete_set) {
      delete_list[i++] = del;
    }
    return save_bin<_u32>(filename, delete_list.get(), _delete_set.size(), 1);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::save(const char *filename, const bool disk_index) {
    // first check if no thread is inserting
    auto start = std::chrono::high_resolution_clock::now();
    std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
    _change_lock.lock();

    // compact_data();
    compact_frozen_point();
    if (!_save_as_one_file) {
      std::string graph_file = std::string(filename);
      delete_file(graph_file);
      save_graph(graph_file);
      if (!disk_index) {
        std::string tags_file = std::string(filename) + ".tags";
        std::string data_file = std::string(filename) + ".data";
        std::string delete_list_file = std::string(filename) + ".del";

        // Because the save_* functions use append mode, ensure that
        // the files are deleted before save. Ideally, we should check
        // the error code for delete_file, but will ignore now because
        // delete should succeed if save will succeed.

        delete_file(data_file);
        save_data(data_file);
        delete_file(tags_file);
        save_tags(tags_file);
        delete_file(delete_list_file);
        save_delete_list(delete_list_file);
      }
    } else {
      LOG(WARNING) << "Save index in a single file currently not supported. "
                      "Not saving the index.";
    }
    if (disk_index) {
      if (_data != nullptr) {
        aligned_free(_data);
        _data = nullptr;
      }
    }

    reposition_frozen_point_to_end();

    _change_lock.unlock();
    auto stop = std::chrono::high_resolution_clock::now();
    auto timespan =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    LOG_KNOWHERE_DEBUG_ << "Time taken for save: " << timespan.count() << "s."
                        << std::endl;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_tags(const std::string tag_filename) {
    if (_enable_tags && !file_exists(tag_filename)) {
      diskann::cerr << "Tag file provided does not exist!" << std::endl;
      throw diskann::ANNException("Tag file provided does not exist!", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }
    if (!_enable_tags) {
      diskann::cout << "Tags not loaded as tags not enabled." << std::endl;
      return 0;
    }

    size_t file_dim, file_num_points;
    std::unique_ptr<TagT[]> tag_data = nullptr;
    load_bin<TagT>(std::string(tag_filename), tag_data, file_num_points,
                   file_dim);

    if (file_dim != 1) {
      std::stringstream stream;
      stream << "ERROR: Found " << file_dim << " dimensions for tags,"
             << "but tag file must have 1 dimension.";
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t num_data_points =
        _num_frozen_pts > 0 ? file_num_points - 1 : file_num_points;
    for (_u32 i = 0; i < (_u32) num_data_points; i++) {
      TagT tag = tag_data[i];
      if (_delete_set.find(i) == _delete_set.end()) {
        _location_to_tag[i] = tag;
        _tag_to_location[tag] = (_u32) i;
      }
    }
    diskann::cout << "Tags loaded." << std::endl;
    return file_num_points;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_data(std::string filename) {
    size_t file_dim, file_num_points;
    if (!file_exists(filename)) {
      std::stringstream stream;
      stream << "ERROR: data file " << filename << " does not exist."
             << std::endl;
      LOG(ERROR) << stream.str();
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);

    // since we are loading a new dataset, _empty_slots must be cleared
    _empty_slots.clear();

    if (file_dim != _dim) {
      std::stringstream stream;
      stream << "ERROR: Driver requests loading " << _dim << " dimension,"
             << "but file has " << file_dim << " dimension." << std::endl;
      LOG(ERROR) << stream.str();
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    if (file_num_points > _max_points) {
      //_change_lock is already locked in load()
      std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      resize(file_num_points);
    }

    copy_aligned_data_from_file<T>(std::string(filename), _data,
                                   file_num_points, file_dim, _aligned_dim);
    return file_num_points;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_delete_set(const std::string &filename) {
    std::unique_ptr<_u32[]> delete_list;
    _u64                    npts, ndim;

    diskann::load_bin<_u32>(filename, delete_list, npts, ndim);
    assert(ndim == 1);
    for (uint32_t i = 0; i < npts; i++) {
      _delete_set.insert(delete_list[i]);
    }
    return npts;
  }

  // load the index from file and update the width (max_degree), ep (navigating
  // node id), and _final_graph (adjacency list)
  template<typename T, typename TagT>
  void Index<T, TagT>::load(const char *filename, uint32_t num_threads,
                            uint32_t search_l) {
    _change_lock.lock();

    size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;

    if (!_save_as_one_file) {
      // For DLVS Store, we will not support saving the index in multiple files.
      std::string data_file = std::string(filename) + ".data";
      std::string tags_file = std::string(filename) + ".tags";
      std::string delete_set_file = std::string(filename) + ".del";
      std::string graph_file = std::string(filename);
      data_file_num_pts = load_data(data_file);
      if (file_exists(delete_set_file)) {
        load_delete_set(delete_set_file);
      }
      if (_enable_tags) {
        tags_file_num_pts = load_tags(tags_file);
      }
      graph_num_pts = load_graph(graph_file, data_file_num_pts);
    } else {
      diskann::cout << "Single index file saving/loading support not yet "
                       "enabled. Not loading the index."
                    << std::endl;
      return;
    }

    if (data_file_num_pts != graph_num_pts ||
        (data_file_num_pts != tags_file_num_pts && _enable_tags)) {
      std::stringstream stream;
      stream << "ERROR: When loading index, loaded " << data_file_num_pts
             << " points from datafile, " << graph_num_pts
             << " from graph, and " << tags_file_num_pts
             << " tags, with num_frozen_pts being set to " << _num_frozen_pts
             << " in constructor." << std::endl;
      LOG(ERROR) << stream.str();
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    _nd = data_file_num_pts - _num_frozen_pts;
    _empty_slots.clear();
    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

    _lazy_done = _delete_set.size() != 0;

    reposition_frozen_point_to_end();
    diskann::cout << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd
                  << " _ep: " << _ep
                  << " size(_location_to_tag): " << _location_to_tag.size()
                  << " size(_tag_to_location):" << _tag_to_location.size()
                  << " Max points: " << _max_points << std::endl;

    _search_queue_size = search_l;
    // For incremental index, _query_scratch is initialized in the constructor.
    // For the bulk index, the parameters required to initialize _query_scratch
    // are known only at load time, hence this check and the call to
    // initialize_q_s().
    if (_query_scratch.size() == 0) {
      initialize_query_scratch(num_threads, search_l, search_l,
                               (uint32_t) _max_range_of_loaded_graph, _dim);
    }

    _change_lock.unlock();
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::load_graph(std::string filename,
                                    size_t      expected_num_points) {
    size_t expected_file_size;
    _u64   file_frozen_pts;
    _u64 file_offset = 0;  // will need this for single file format support
    std::ifstream in(filename, std::ios::binary);
    in.seekg(file_offset, in.beg);
    in.read((char *) &expected_file_size, sizeof(_u64));
    in.read((char *) &_width, sizeof(unsigned));
    in.read((char *) &_ep, sizeof(unsigned));
    in.read((char *) &file_frozen_pts, sizeof(_u64));
    _u64 vamana_metadata_size =
        sizeof(_u64) + sizeof(_u32) + sizeof(_u32) + sizeof(_u64);

    diskann::cout << "From graph header, expected_file_size: "
                  << expected_file_size << ", _width: " << _width
                  << ", _ep: " << _ep
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    if (file_frozen_pts != _num_frozen_pts) {
      std::stringstream stream;
      if (file_frozen_pts == 1) {
        stream << "ERROR: When loading index, detected dynamic index, but "
                  "constructor asks for static index. Exitting."
               << std::endl;
      } else {
        stream << "ERROR: When loading index, detected static index, but "
                  "constructor asks for dynamic index. Exitting."
               << std::endl;
      }
      LOG(ERROR) << stream.str();
      aligned_free(_data);
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    diskann::cout << "Loading vamana graph " << filename << "..." << std::flush;

    // Sanity check. In case the user gave us fewer points as max_points than
    // the number
    // of points in the dataset, resize the _final_graph to the larger size.
    if (_max_points < expected_num_points) {
      diskann::cout << "Number of points in data: " << expected_num_points
                    << " is greater than max_points: " << _max_points
                    << " Setting max points to: " << expected_num_points
                    << std::endl;
      _final_graph.resize(expected_num_points + _num_frozen_pts);
      _max_points = expected_num_points;
    }

    size_t   bytes_read = vamana_metadata_size;
    size_t   cc = 0;
    unsigned nodes_read = 0;
    while (bytes_read != expected_file_size) {
      unsigned k;
      in.read((char *) &k, sizeof(unsigned));
      if (k == 0) {
        diskann::cerr << "ERROR: Point found with no out-neighbors, point#"
                      << nodes_read << std::endl;
      }
      //      if (in.eof())
      //        break;
      cc += k;
      ++nodes_read;
      std::vector<unsigned> tmp(k);
      tmp.reserve(k);
      in.read((char *) tmp.data(), k * sizeof(unsigned));
      _final_graph[nodes_read - 1].swap(tmp);
      bytes_read += sizeof(uint32_t) * ((_u64) k + 1);
      if (nodes_read % 10000000 == 0)
        diskann::cout << "." << std::flush;
      if (k > _max_range_of_loaded_graph) {
        _max_range_of_loaded_graph = k;
      }
    }

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc
                  << " out-edges, _ep is set to " << _ep << std::endl;
    return nodes_read;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::get_vector_by_tag(TagT &tag, T *vec) {
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      diskann::cout << "Tag " << tag << " does not exist" << std::endl;
      return -1;
    }

    size_t location = _tag_to_location[tag];  // must be
    // memory should be allocated for vec before calling this function
    memcpy((void *) vec, (void *) (_data + location * _aligned_dim),
           (size_t) _dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  const T *Index<T, TagT>::get_vector_by_tag(const TagT &tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    if (_tag_to_location.find(tag) == _tag_to_location.end()) {
      diskann::cout << "Tag " << tag << " does not exist in the index."
                    << std::endl;
      return nullptr;
    } else {
      unsigned location = _tag_to_location[tag];
      return _data + (((size_t) location) * _aligned_dim);
    }
  }

  /**************************************************************
   *      Support for Static Index Building and Searching
   **************************************************************/

  /* This function finds out the navigating node, which is the medoid node
   * in the graph.
   */
  template<typename T, typename TagT>
  unsigned Index<T, TagT>::calculate_entry_point() {
    // allocate and init centroid
    auto center = std::make_unique<float[]>(_aligned_dim);
    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] = 0;

    for (size_t i = 0; i < _nd; i++)
      for (size_t j = 0; j < _aligned_dim; j++)
        center[j] += (float) _data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
      center[j] /= (float) _nd;

    // compute all to one distance
    auto distances = std::make_unique<float[]>(_nd);
    auto l2_distance_fun = get_distance_function<float>(diskann::Metric::L2);

    auto num_threads = _build_thread_pool->size();
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(num_threads);
    auto future_task_size = DIV_ROUND_UP(_nd, num_threads);
    for (_s64 i = 0; i < (_s64) _nd; i += future_task_size) {
      futures.emplace_back(_build_thread_pool->push(
          [&, beg_id = i, end_id = std::min(_nd, i + future_task_size),
           center_ptr = center.get(), distances_ptr = distances.get()]() {
            for (auto node_id = beg_id; node_id < (_s64) end_id; node_id++) {
              // extract point and distance reference
              float   &dist = distances_ptr[node_id];
              const T *cur_vec = _data + (node_id * (size_t) _aligned_dim);
              if constexpr (std::is_same<T, float>::value) {
                dist =
                    l2_distance_fun(center_ptr, cur_vec, (size_t) _aligned_dim);
              } else {
                dist = 0;
                float diff = 0;
                for (size_t j = 0; j < _aligned_dim; j++) {
                  diff = (center_ptr[j] - (float) cur_vec[j]) *
                         (center_ptr[j] - (float) cur_vec[j]);
                  dist += diff;
                }
              }
            }
          }));
    }
    knowhere::WaitAllSuccess(futures);
    // find imin
    unsigned min_idx = 0;
    float    min_dist = distances[0];
    for (unsigned i = 1; i < _nd; i++) {
      if (distances[i] < min_dist) {
        min_idx = i;
        min_dist = distances[i];
      }
    }

    return min_idx;
  }

  /* iterate_to_fixed_point():
   * node_coords : point whose neighbors to be found.
   * init_ids : ids of initial search list.
   * Lsize : size of list.
   * beam_width: beam_width when performing indexing
   * expanded_nodes_info: will contain all the node ids and distances from
   * query that are expanded
   * expanded_nodes_ids : will contain all the nodes that are expanded during
   * search.
   * best_L_nodes: ids of closest L nodes in list
   */
  template<typename T, typename TagT>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::iterate_to_fixed_point(
      const T *node_coords, const unsigned Lsize,
      const std::vector<unsigned> &init_ids,
      std::vector<Neighbor>       &expanded_nodes_info,
      tsl::robin_set<unsigned>    &expanded_nodes_ids,
      std::vector<Neighbor> &best_L_nodes, std::vector<unsigned> &des,
      tsl::robin_set<unsigned> &inserted_into_pool_rs,
      boost::dynamic_bitset<> &inserted_into_pool_bs, bool ret_frozen,
      bool search_invocation) {
    for (unsigned i = 0; i < Lsize + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }
    if (!search_invocation) {
      expanded_nodes_ids.clear();
      expanded_nodes_info.clear();
      // expanded_nodes_info.reserve(10 * Lsize);
      // expanded_nodes_ids.reserve((int) (1.5 * Lsize));
      des.clear();
    }

    unsigned l = 0;
    Neighbor nn;

    bool fast_iterate =
        (_max_points + _num_frozen_pts) <= MAX_POINTS_FOR_USING_BITSET;
    // tsl::robin_set<unsigned> inserted_into_pool_rs;
    // boost::dynamic_bitset<> inserted_into_pool_bs;

    if (fast_iterate) {
      auto total_num_points = _max_points + _num_frozen_pts;
      if (inserted_into_pool_bs.size() < total_num_points) {
        // hopefully using 2X will reduce the number of allocations.
        auto resize_size = 2 * total_num_points > MAX_POINTS_FOR_USING_BITSET
                               ? MAX_POINTS_FOR_USING_BITSET
                               : 2 * total_num_points;
        //        inserted_into_pool_bs.resize(2 * total_num_points);
        inserted_into_pool_bs.resize(resize_size);
      }
    }

    for (auto id : init_ids) {
      if (id >= _max_points + _num_frozen_pts) {
        diskann::cerr << "Wrong id : " << id << std::endl;
        throw diskann::ANNException(
            std::string("Wrong id") + std::to_string(id), -1, __FUNCSIG__,
            __FILE__, __LINE__);
      }
      nn = Neighbor(id,
                    _distance(_data + _aligned_dim * (size_t) id, node_coords,
                              (unsigned) _aligned_dim),
                    true);
      if (fast_iterate) {
        if (inserted_into_pool_bs[id] == 0) {
          inserted_into_pool_bs[id] = 1;
          best_L_nodes[l++] = nn;
        }
      } else {
        if (inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end()) {
          inserted_into_pool_rs.insert(id);
          best_L_nodes[l++] = nn;
        }
      }
      if (l == Lsize)
        break;
    }

    /* sort best_L_nodes based on distance of each point to node_coords */
    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;
    uint32_t hops = 0;
    uint32_t cmps = 0;

    while (k < l) {
      unsigned nk = l;

      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        auto n = best_L_nodes[k].id;
        if (!(best_L_nodes[k].id == _ep && _num_frozen_pts > 0 &&
              !ret_frozen)) {
          if (!search_invocation) {
            expanded_nodes_info.emplace_back(best_L_nodes[k]);
            expanded_nodes_ids.insert(n);
          }
        }
        // std::vector<unsigned> des;
        des.clear();
        // des.reserve(2 * _final_graph[n].size());
        if (_dynamic_index) {
          LockGuard guard(_locks[n]);
          for (unsigned m = 0; m < _final_graph[n].size(); m++) {
            if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
              diskann::cerr << "Wrong id found: " << _final_graph[n][m]
                            << std::endl;
              throw diskann::ANNException(
                  std::string("Wrong id found") +
                      std::to_string(_final_graph[n][m]),
                  -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            des.emplace_back(_final_graph[n][m]);
          }
        } else {
          for (unsigned m = 0; m < _final_graph[n].size(); m++) {
            if (_final_graph[n][m] >= _max_points + _num_frozen_pts) {
              diskann::cerr << "Wrong id found: " << _final_graph[n][m]
                            << std::endl;
              throw diskann::ANNException(
                  std::string("Wrong id found") +
                      std::to_string(_final_graph[n][m]),
                  -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            des.emplace_back(_final_graph[n][m]);
          }
        }

        for (unsigned m = 0; m < des.size(); ++m) {
          unsigned id = des[m];
          bool     id_is_missing = fast_iterate ? inserted_into_pool_bs[id] == 0
                                                : inserted_into_pool_rs.find(id) ==
                                                  inserted_into_pool_rs.end();
          if (id_is_missing) {
            if (fast_iterate) {
              inserted_into_pool_bs[id] = 1;
            } else {
              inserted_into_pool_rs.insert(id);
            }
            if ((m + 1) < des.size()) {
              auto nextn = des[m + 1];
              diskann::prefetch_vector(
                  (const char *) _data + _aligned_dim * (size_t) nextn,
                  sizeof(T) * _aligned_dim);
            }

            cmps++;
            float dist =
                _distance(node_coords, _data + _aligned_dim * (size_t) id,
                          (unsigned) _aligned_dim);

            if (dist >= best_L_nodes[l - 1].distance && (l == Lsize))
              continue;

            Neighbor nn(id, dist, true);
            unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < Lsize)
              ++l;
            if (r < nk)
              nk = r;
          }
        }

        if (nk <= k)
          k = nk;
        else
          ++k;
      } else
        k++;
    }
    return std::make_pair(hops, cmps);
  }

  // template<typename T, typename TagT>
  // void Index<T, TagT>::iterate_to_fixed_point(
  //    const T *node_coords, const unsigned Lindex,
  //    std::vector<Neighbor> &expanded_nodes_info,
  //    tsl::robin_map<uint32_t, T *> &coord_map, bool return_frozen_pt) {
  //  std::vector<uint32_t> init_ids;
  //  init_ids.push_back(this->_ep);
  //  std::vector<Neighbor>    best_L_nodes;
  //  tsl::robin_set<uint32_t> expanded_nodes_ids;
  //  this->iterate_to_fixed_point(node_coords, Lindex, init_ids,
  //                               expanded_nodes_info, expanded_nodes_ids,
  //                               best_L_nodes, return_frozen_pt);
  //  for (Neighbor &einf : expanded_nodes_info) {
  //    T *coords =
  //        this->_data + (uint64_t) einf.id * (uint64_t) this->_aligned_dim;
  //    coord_map.insert(std::make_pair(einf.id, coords));
  //  }
  //}

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(
      const size_t node_id, const unsigned Lindex,
      std::vector<unsigned>     init_ids,
      std::vector<Neighbor>    &expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids, std::vector<unsigned> &des,
      std::vector<Neighbor>    &best_L_nodes,
      tsl::robin_set<unsigned> &inserted_into_pool_rs,
      boost::dynamic_bitset<>  &inserted_into_pool_bs) {
    const T *node_coords = _data + _aligned_dim * node_id;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_ep);

    iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info,
                           expanded_nodes_ids, best_L_nodes, des,
                           inserted_into_pool_rs, inserted_into_pool_bs);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_expanded_nodes(
      const size_t node_id, const unsigned Lindex,
      std::vector<unsigned>     init_ids,
      std::vector<Neighbor>    &expanded_nodes_info,
      tsl::robin_set<unsigned> &expanded_nodes_ids) {
    const T *node_coords = _data + _aligned_dim * node_id;

    if (init_ids.size() == 0)
      init_ids.emplace_back(_ep);

    std::vector<unsigned> des;
    std::vector<Neighbor> best_L_nodes;
    best_L_nodes.resize(Lindex + 1);
    tsl::robin_set<unsigned> inserted_into_pool_rs;
    boost::dynamic_bitset<>  inserted_into_pool_bs;

    iterate_to_fixed_point(node_coords, Lindex, init_ids, expanded_nodes_info,
                           expanded_nodes_ids, best_L_nodes, des,
                           inserted_into_pool_rs, inserted_into_pool_bs);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result) {
    auto               pool_size = (_u32) pool.size();
    std::vector<float> occlude_factor(pool_size, 0);
    occlude_list(pool, alpha, degree, maxc, result, occlude_factor);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::occlude_list(std::vector<Neighbor> &pool,
                                    const float alpha, const unsigned degree,
                                    const unsigned         maxc,
                                    std::vector<Neighbor> &result,
                                    std::vector<float>    &occlude_factor) {
    if (pool.empty())
      return;
    assert(std::is_sorted(pool.begin(), pool.end()));
    assert(!pool.empty());

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
      unsigned start = 0;
      float eps = cur_alpha + 0.01f;  // used for MIPS, where we store a value
                                      // of eps in cur_alpha to
      // denote pruned out entries which we can skip in later rounds.
      while (result.size() < degree && (start) < pool.size() && start < maxc) {
        auto &p = pool[start];
        if (occlude_factor[start] > cur_alpha) {
          start++;
          continue;
        }
        occlude_factor[start] = std::numeric_limits<float>::max();
        result.push_back(p);
        for (unsigned t = start + 1; t < pool.size() && t < maxc; t++) {
          if (occlude_factor[t] > alpha)
            continue;
          float djk = _distance(_data + _aligned_dim * (size_t) pool[t].id,
                                _data + _aligned_dim * (size_t) p.id,
                                (size_t) _aligned_dim);
          if (_dist_metric == diskann::Metric::L2 ||
              _dist_metric == diskann::Metric::COSINE) {
            occlude_factor[t] =
                //(std::max)(occlude_factor[t], pool[t].distance / djk);
                diskann_max(occlude_factor[t], pool[t].distance / djk);
          } else if (_dist_metric ==
                     diskann::Metric::INNER_PRODUCT) {  // stylized rules for
                                                        // inner product since
                                                        // we want max instead
                                                        // of min distance
            float x = -pool[t].distance;
            float y = -djk;
            if (y > cur_alpha * x) {
              occlude_factor[t] =
                  /* (std::max)*/ diskann_max(occlude_factor[t], eps);
            }
          }
        }
        start++;
      }
      cur_alpha *= 1.2;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned         location,
                                       std::vector<Neighbor> &pool,
                                       std::vector<unsigned> &pruned_list) {
    prune_neighbors(location, pool, _indexingRange, _indexingMaxC,
                    _indexingAlpha, pruned_list);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_neighbors(const unsigned         location,
                                       std::vector<Neighbor> &pool,
                                       const _u32             range,
                                       const _u32  max_candidate_size,
                                       const float alpha,
                                       std::vector<unsigned> &pruned_list) {
    if (pool.size() == 0) {
      std::stringstream ss;
      ss << "Thread id:" << std::this_thread::get_id()
         << " Pool address: " << &pool << std::endl;
      std::cout << ss.str();
      throw diskann::ANNException("Pool passed to prune_neighbors is empty",
                                  -1);
    }

    _width = (std::max)(_width, range);

    // sort the pool based on distance to query
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(range);
    std::vector<float> occlude_factor(pool.size(), 0);

    occlude_list(pool, alpha, range, max_candidate_size, result,
                 occlude_factor);

    /* Add all the nodes in result into a variable called cut_graph
     * So this contains all the neighbors of id location
     */
    pruned_list.clear();
    assert(result.size() <= range);
    for (auto iter : result) {
      if (iter.id != location)
        pruned_list.emplace_back(iter.id);
    }

    if (_saturate_graph && alpha > 1) {
      for (uint32_t i = 0; i < pool.size() && pruned_list.size() < range; i++) {
        if ((std::find(pruned_list.begin(), pruned_list.end(), pool[i].id) ==
             pruned_list.end()) &&
            pool[i].id != location)
          pruned_list.emplace_back(pool[i].id);
      }
    }
  }

  /* batch_inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::batch_inter_insert(
      unsigned n, const std::vector<unsigned> &pruned_list, const _u32 range,
      std::vector<unsigned> &need_to_sync) {
    // assert(!src_pool.empty());

    for (auto des : pruned_list) {
      if (des == n)
        continue;
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      if (des > _max_points)
        diskann::cout << "error. " << des << " exceeds max_pts" << std::endl;
      /* des_pool contains the neighbors of the neighbors of n */

      {
        LockGuard guard(_locks[des]);
        if (std::find(_final_graph[des].begin(), _final_graph[des].end(), n) ==
            _final_graph[des].end()) {
          _final_graph[des].push_back(n);
          if (_final_graph[des].size() >
              (unsigned) (range * GRAPH_SLACK_FACTOR))
            need_to_sync[des] = 1;
        }
      }  // des lock is released by this point
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::batch_inter_insert(
      unsigned n, const std::vector<unsigned> &pruned_list,
      std::vector<unsigned> &need_to_sync) {
    batch_inter_insert(n, pruned_list, _indexingRange, need_to_sync);
  }

  /* inter_insert():
   * This function tries to add reverse links from all the visited nodes to
   * the current node n.
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::inter_insert(unsigned               n,
                                    std::vector<unsigned> &pruned_list,
                                    const _u32 range, bool update_in_graph) {
    //    const auto range = _indexingRange;  // parameter.Get<unsigned>("R");
    assert(n >= 0 && n < _nd + _num_frozen_pts);
    const auto &src_pool = pruned_list;

    assert(!src_pool.empty());

    for (auto des : src_pool) {
      /* des.id is the id of the neighbors of n */
      assert(des >= 0 && des < _max_points + _num_frozen_pts);
      /* des_pool contains the neighbors of the neighbors of n */
      auto                 &des_pool = _final_graph[des];
      std::vector<unsigned> copy_of_neighbors;
      bool                  prune_needed = false;
      {
        LockGuard guard(_locks[des]);
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < (_u64) (GRAPH_SLACK_FACTOR * range)) {
            des_pool.emplace_back(n);
            if (update_in_graph) {
              LockGuard guard(_locks_in[n]);
              _in_graph[n].emplace_back(des);
            }
            prune_needed = false;
          } else {
            copy_of_neighbors = des_pool;
            prune_needed = true;
          }
        }
      }  // des lock is released by this point

      if (prune_needed) {
        copy_of_neighbors.push_back(n);
        tsl::robin_set<unsigned> dummy_visited(0);
        std::vector<Neighbor>    dummy_pool(0);

        size_t reserveSize =
            (size_t) (std::ceil(1.05 * GRAPH_SLACK_FACTOR * range));
        dummy_visited.reserve(reserveSize);
        dummy_pool.reserve(reserveSize);

        for (auto cur_nbr : copy_of_neighbors) {
          if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
              cur_nbr != des) {
            float dist = _distance(_data + _aligned_dim * (size_t) des,
                                   _data + _aligned_dim * (size_t) cur_nbr,
                                   (size_t) _aligned_dim);
            dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
            dummy_visited.insert(cur_nbr);
          }
        }
        std::vector<unsigned> new_out_neighbors;
        prune_neighbors(des, dummy_pool, new_out_neighbors);
        {
          LockGuard guard(_locks[des]);
          // updating in_graph of out-neighbors of des
          if (update_in_graph) {
            for (auto out_nbr : _final_graph[des]) {
              {
                LockGuard guard(_locks_in[out_nbr]);
                for (unsigned i = 0; i < _in_graph[out_nbr].size(); i++) {
                  if (_in_graph[out_nbr][i] == des) {
                    _in_graph[out_nbr].erase(_in_graph[out_nbr].begin() + i);
                    break;
                  }
                }
              }
            }
          }

          _final_graph[des].clear();
          for (auto new_nbr : new_out_neighbors) {
            _final_graph[des].emplace_back(new_nbr);
            if (update_in_graph) {
              LockGuard guard(_locks_in[new_nbr]);
              _in_graph[new_nbr].emplace_back(des);
            }
          }
        }
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::inter_insert(unsigned               n,
                                    std::vector<unsigned> &pruned_list,
                                    bool                   update_in_graph) {
    inter_insert(n, pruned_list, _indexingRange, update_in_graph);
  }

  /* Link():
   * The graph creation function.
   *    The graph will be updated periodically in NUM_SYNCS batches
   */
  template<typename T, typename TagT>
  void Index<T, TagT>::link(Parameters &parameters) {
    uint32_t num_syncs = (unsigned) DIV_ROUND_UP(_nd + _num_frozen_pts, 8192);
    if (num_syncs < 40)
      num_syncs = 40;
    LOG_KNOWHERE_DEBUG_ << "Number of syncs: " << num_syncs;

    _saturate_graph = parameters.Get<bool>("saturate_graph");

    auto                                    num_threads = _build_thread_pool->size();
    std::vector<folly::Future<folly::Unit>> futures;

    _indexingQueueSize = parameters.Get<unsigned>("L");  // Search list size
    _indexingRange = parameters.Get<unsigned>("R");
    _indexingMaxC = parameters.Get<unsigned>("C");
    const bool  accelerate_build = parameters.Get<bool>("accelerate_build");
    const bool  shuffle_build    = parameters.Get<bool>("shuffle_build");
    const float last_round_alpha = parameters.Get<float>("alpha");
    unsigned    L = _indexingQueueSize;

    std::vector<unsigned> Lvec;
    Lvec.push_back(unsigned(0.8 * L));
    Lvec.push_back(L);
    const unsigned NUM_RNDS = 2;

    // Max degree of graph
    // Pruning parameter
    // Set alpha=1 for the first pass; use specified alpha for last pass
    // parameters.Set<float>("alpha", 1);
    _indexingAlpha = 1.0f;

    /* visit_order is a vector that is initialized to the entire graph */
    std::vector<unsigned>          visit_order;
    std::vector<diskann::Neighbor> pool, tmp;
    tsl::robin_set<unsigned>       visited;
    visit_order.reserve(_nd + _num_frozen_pts);
    for (unsigned i = 0; i < (unsigned) _nd; i++) {
      visit_order.emplace_back(i);
    }

    if (_num_frozen_pts > 0)
      visit_order.emplace_back((unsigned) _max_points);

    // if there are frozen points, the first such one is set to be the _ep
    if (_num_frozen_pts > 0)
      _ep = (unsigned) _max_points;
    else
      _ep = calculate_entry_point();

    /*    _final_graph.reserve(_max_points + _num_frozen_pts);
        _final_graph.resize(_max_points + _num_frozen_pts); */

    if (_support_eager_delete) {
      _in_graph.reserve(_max_points + _num_frozen_pts);
      _in_graph.resize(_max_points + _num_frozen_pts);
    }

    for (uint64_t p = 0; p < _nd; p++) {
      _final_graph[p].reserve(
          (size_t) (std::ceil(_indexingRange * GRAPH_SLACK_FACTOR * 1.05)));
    }

    std::random_device               rd;
    std::mt19937                     gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    // creating a initial list to begin the search process. it has _ep and
    // random other nodes
    std::set<unsigned> unique_start_points;
    unique_start_points.insert(_ep);

    if (accelerate_build) {
      while (unique_start_points.size() < 128) {
        unique_start_points.insert(_nd * dis(gen));
      }
    }
    std::vector<unsigned> init_ids;
    for (auto pt : unique_start_points)
      init_ids.emplace_back(pt);

    diskann::Timer link_timer;
    for (uint32_t rnd_no = 0; rnd_no < NUM_RNDS; rnd_no++) {
      L = Lvec[rnd_no];

      if (rnd_no == NUM_RNDS - 1) {
        if (last_round_alpha > 1)
          // parameters.Set<float>("alpha", last_round_alpha);
          _indexingAlpha = last_round_alpha;
      }

      double   sync_time = 0, total_sync_time = 0;
      double   inter_time = 0, total_inter_time = 0;
      size_t   inter_count = 0, total_inter_count = 0;
      unsigned progress_counter = 0;

      size_t round_size = DIV_ROUND_UP(_nd, num_syncs);  // size of each batch
      std::vector<unsigned> need_to_sync(_max_points + _num_frozen_pts, 0);

      std::vector<std::vector<unsigned>> pruned_list_vector(round_size);

      auto round_num_syncs = num_syncs;
      if (accelerate_build && rnd_no == 0) {
        round_num_syncs = num_syncs * 0.05;
      }

      futures.reserve(round_size);
      std::vector<unsigned> shuffle_batch_ids;
      if (shuffle_build) {
        shuffle_batch_ids.reserve(round_num_syncs);
        for (unsigned i = 0; i < (unsigned) round_num_syncs; i++) {
          shuffle_batch_ids.emplace_back(i);
        }
        std::random_device rng;
        std::mt19937       urng(rng());
        std::shuffle(shuffle_batch_ids.begin(), shuffle_batch_ids.end(), urng);
      }
      for (uint32_t sync_num = 0; sync_num < round_num_syncs; sync_num++) {
        size_t start_id = (shuffle_build ? shuffle_batch_ids[sync_num] : sync_num) * round_size;
        size_t end_id =
            (std::min)(_nd + _num_frozen_pts, ((shuffle_build ? shuffle_batch_ids[sync_num] : sync_num) + 1) * round_size);

        auto s = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff;
        futures.clear();
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             ++node_ctr) {
          futures.emplace_back(_build_thread_pool->push([&, node_id = node_ctr]() {
            auto                     node = visit_order[node_id];
            size_t                   node_offset = node_id - start_id;
            tsl::robin_set<unsigned> visited;
            std::vector<unsigned>   &pruned_list =
                pruned_list_vector[node_offset];
            // get nearest neighbors of n in tmp. pool contains all the
            // points that were checked along with their distance from
            // n. visited contains all the points visited, just the ids
            std::vector<Neighbor> pool;
            pool.reserve(L * 2);
            visited.reserve(L * 2);
            get_expanded_nodes(node, L, init_ids, pool, visited);
            /* check the neighbors of the query that are not part of
             * visited, check their distance to the query, and add it to
             * pool.
             */
            if (!_final_graph[node].empty())
              for (auto id : _final_graph[node]) {
                if (visited.find(id) == visited.end() && id != node) {
                  float dist = _distance(_data + _aligned_dim * (size_t) node,
                                         _data + _aligned_dim * (size_t) id,
                                         (size_t) _aligned_dim);
                  pool.emplace_back(Neighbor(id, dist, true));
                  visited.insert(id);
                }
              }
            prune_neighbors(node, pool, pruned_list);
          }));
        }
        knowhere::WaitAllSuccess(futures);
        diff = std::chrono::high_resolution_clock::now() - s;
        sync_time += diff.count();

        // prune_neighbors will check pool, and remove some of the points and
        // create a cut_graph, which contains neighbors for point n
        size_t batch_size = DIV_ROUND_UP(end_id - start_id, num_threads);
        futures.clear();
        for (_s64 node_ctr = (_s64) start_id; node_ctr < (_s64) end_id;
             node_ctr += batch_size) {
          futures.emplace_back(_build_thread_pool->push(
              [&, batch_begin_id = node_ctr,
               batch_end_id = std::min(end_id, node_ctr + batch_size)]() {
                for (auto id = batch_begin_id; id < (_s64) batch_end_id; id++) {
                  _u64                   node = visit_order[id];
                  size_t                 node_offset = id - start_id;
                  std::vector<unsigned> &pruned_list =
                      pruned_list_vector[node_offset];
                  _final_graph[node].clear();
                  for (auto id : pruned_list)
                    _final_graph[node].emplace_back(id);
                }
              }));
        }
        knowhere::WaitAllSuccess(futures);
        s = std::chrono::high_resolution_clock::now();

        futures.clear();
        for (_s64 node_ctr = start_id; node_ctr < (_s64) end_id;
             node_ctr += batch_size) {
          futures.emplace_back(_build_thread_pool->push(
              [&, batch_begin_id = node_ctr,
               batch_end_id = std::min(end_id, node_ctr + batch_size)]() {
                for (auto id = batch_begin_id; id < (_s64) batch_end_id; id++) {
                  auto                   node = visit_order[id];
                  _u64                   node_offset = id - start_id;
                  std::vector<unsigned> &pruned_list =
                      pruned_list_vector[node_offset];
                  batch_inter_insert(node, pruned_list, need_to_sync);
                  //          inter_insert(node, pruned_list, parameters, 0);
                  pruned_list.clear();
                  pruned_list.shrink_to_fit();
                }
              }));
        }
        knowhere::WaitAllSuccess(futures);

        futures.clear();
        for (_s64 node_ctr = 0; node_ctr < (_s64) (visit_order.size());
             node_ctr++) {
          auto cur_node = visit_order[node_ctr];
          if (need_to_sync[cur_node] != 0) {
            futures.emplace_back(_build_thread_pool->push([&, node_id = cur_node]() {
              need_to_sync[node_id] = 0;
              inter_count++;
              tsl::robin_set<unsigned> dummy_visited(0);
              std::vector<Neighbor>    dummy_pool(0);
              std::vector<unsigned>    new_out_neighbors;

              for (auto cur_nbr : _final_graph[node_id]) {
                if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                    cur_nbr != node_id) {
                  float dist =
                      _distance(_data + _aligned_dim * (size_t) node_id,
                                _data + _aligned_dim * (size_t) cur_nbr,
                                (size_t) _aligned_dim);
                  dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                  dummy_visited.insert(cur_nbr);
                }
              }
              prune_neighbors(node_id, dummy_pool, new_out_neighbors);

              _final_graph[node_id].clear();
              for (auto id : new_out_neighbors)
                _final_graph[node_id].emplace_back(id);
            }));
          }
        }
        knowhere::WaitAllSuccess(futures);
        futures.clear();

        diff = std::chrono::high_resolution_clock::now() - s;
        inter_time += diff.count();

        if ((sync_num * 100) / num_syncs > progress_counter) {
          std::stringstream stream;
          stream.precision(4);
          stream << "Completed  (round: " << rnd_no << ", sync: " << sync_num
                 << "/" << num_syncs << " with L " << L << ")"
                 << " sync_time: " << sync_time << "s"
                 << "; inter_time: " << inter_time << "s" << std::endl;
          LOG_KNOWHERE_INFO_ << stream.str();
          total_sync_time += sync_time;
          total_inter_time += inter_time;
          total_inter_count += inter_count;
          sync_time = 0;
          inter_time = 0;
          inter_count = 0;
          progress_counter += 20;
        }
      }
// Gopal. Splitting nsg_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#if defined(RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && \
    defined(DISKANN_BUILD)
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
      if (_nd > 0) {
        LOG_KNOWHERE_INFO_ << "Completed Pass " << rnd_no
                           << " of data using L=" << L
                           << " and alpha=" << _indexingAlpha << ". Stats: "
                           << "search+prune_time=" << total_sync_time
                           << "s, inter_time=" << total_inter_time
                           << "s, inter_count=" << total_inter_count;
      }
    }

    if (_nd > 0) {
      LOG_KNOWHERE_DEBUG_ << "Starting final cleanup..";
    }
    futures.clear();
    for (_s64 node_ctr = 0; node_ctr < (_s64) (visit_order.size());
         node_ctr++) {
      auto cur_node = visit_order[node_ctr];
      if (_final_graph[cur_node].size() > _indexingRange) {
        futures.emplace_back(_build_thread_pool->push([&, node_id = cur_node]() {
          tsl::robin_set<unsigned> dummy_visited(0);
          std::vector<Neighbor>    dummy_pool(0);
          std::vector<unsigned>    new_out_neighbors;

          for (auto cur_nbr : _final_graph[node_id]) {
            if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                cur_nbr != node_id) {
              float dist = _distance(_data + _aligned_dim * (size_t) node_id,
                                     _data + _aligned_dim * (size_t) cur_nbr,
                                     (size_t) _aligned_dim);
              dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
              dummy_visited.insert(cur_nbr);
            }
          }
          prune_neighbors(node_id, dummy_pool, new_out_neighbors);

          _final_graph[node_id].clear();
          for (auto id : new_out_neighbors)
            _final_graph[node_id].emplace_back(id);
        }));
      }
    }
    knowhere::WaitAllSuccess(futures);
    if (_nd > 0) {
      LOG_KNOWHERE_DEBUG_ << "final cleanup done. Link time: "
                          << ((double) link_timer.elapsed() / (double) 1000000)
                          << "s";
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::prune_all_nbrs(const Parameters &parameters) {
    const unsigned range = parameters.Get<unsigned>("R");

    diskann::Timer                 timer;
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(16384);
    for (_s64 cur_node = 0; cur_node < (_s64) (_max_points + _num_frozen_pts);
         cur_node++) {
      if ((size_t) cur_node < _nd || (size_t) cur_node == _max_points) {
        if (_final_graph[cur_node].size() > range) {
          futures.emplace_back(_build_thread_pool->push([&, node_id = cur_node]() {
            tsl::robin_set<unsigned> dummy_visited(0);
            std::vector<Neighbor>    dummy_pool(0);
            std::vector<unsigned>    new_out_neighbors;

            for (auto cur_nbr : _final_graph[node_id]) {
              if (dummy_visited.find(cur_nbr) == dummy_visited.end() &&
                  cur_nbr != node_id) {
                float dist = _distance(_data + _aligned_dim * (size_t) node_id,
                                       _data + _aligned_dim * (size_t) cur_nbr,
                                       (size_t) _aligned_dim);
                dummy_pool.emplace_back(Neighbor(cur_nbr, dist, true));
                dummy_visited.insert(cur_nbr);
              }
            }
            prune_neighbors((_u32) node_id, dummy_pool, new_out_neighbors);

            _final_graph[node_id].clear();
            for (auto id : new_out_neighbors)
              _final_graph[node_id].emplace_back(id);
          }));
        }
      }
    }
    knowhere::WaitAllSuccess(futures);

    diskann::cout << "Prune time : " << timer.elapsed() / 1000 << "ms"
                  << std::endl;
    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < (_nd + _num_frozen_pts); i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG_KNOWHERE_INFO_ << "Index built with degree: max:" << max << "  avg:"
                         << (float) total / (float) (_nd + _num_frozen_pts)
                         << "  min:" << min << "  count(deg<2):" << cnt;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char              *filename,
                             const size_t             num_points_to_load,
                             Parameters              &parameters,
                             const std::vector<TagT> &tags) {
    if (!file_exists(filename)) {
      std::stringstream stream;
      stream << "Data file " << filename << " does not exist." << std::endl;
      LOG(ERROR) << stream.str();
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      LOG(WARNING) << "Starting with an empty index.";
      _nd = 0;
    } else {
      diskann::get_bin_metadata(filename, file_num_points, file_dim);
      if (file_num_points > _max_points ||
          num_points_to_load > file_num_points) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load
               << " points and file has " << file_num_points << " points, but "
               << "index can support only " << _max_points
               << " points as specified in constructor." << std::endl;
        LOG(ERROR) << stream.str();
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
      if (file_dim != _dim) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        LOG(ERROR) << stream.str();
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      copy_aligned_data_from_file<T>(std::string(filename), _data,
                                     file_num_points, file_dim, _aligned_dim);
      if (_normalize_vecs) {
        for (uint64_t i = 0; i < file_num_points; i++) {
          normalize(_data + _aligned_dim * i, _aligned_dim);
        }
      }

      LOG_KNOWHERE_INFO_ << "Building start. Loading only first "
                         << num_points_to_load << " from file.. ";
      _nd = num_points_to_load;

      if (_enable_tags && tags.size() != num_points_to_load) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load
               << " points from file,"
               << "but tags vector is of size " << tags.size() << "."
               << std::endl;
        LOG(ERROR) << stream.str();
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
      if (_enable_tags) {
        for (size_t i = 0; i < tags.size(); ++i) {
          _tag_to_location[tags[i]] = (unsigned) i;
          _location_to_tag[(unsigned) i] = tags[i];
        }
      }
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    if (_support_eager_delete) {
      update_in_graph();  // copying values to in_graph
    }

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG_KNOWHERE_INFO_ << "Index built with degree: max:" << max << "  avg:"
                         << (float) total / (float) (_nd + _num_frozen_pts)
                         << "  min:" << min << "  count(deg<2):" << cnt;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::build(const char  *filename,
                             const size_t num_points_to_load,
                             Parameters &parameters, const char *tag_filename) {
    if (!file_exists(filename)) {
      std::stringstream stream;
      stream << "Data file provided " << filename << " does not exist."
             << std::endl;
      LOG(ERROR) << stream.str() << std::endl;
      throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                  __LINE__);
    }

    size_t file_num_points, file_dim;
    if (filename == nullptr) {
      LOG_KNOWHERE_INFO_ << "Starting with an empty index.";
      _nd = 0;
    } else {
      diskann::get_bin_metadata(filename, file_num_points, file_dim);
      if (file_num_points > _max_points ||
          num_points_to_load > file_num_points) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << num_points_to_load
               << " points and file has " << file_num_points << " points, but "
               << "index can support only " << _max_points
               << " points as specified in constructor." << std::endl;
        LOG(ERROR) << stream.str();
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }
      if (file_dim != _dim) {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << _dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        LOG(ERROR) << stream.str();
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
      }

      copy_aligned_data_from_file<T>(std::string(filename), _data,
                                     file_num_points, file_dim, _aligned_dim);

      LOG_KNOWHERE_DEBUG_ << "Loading only first " << num_points_to_load
                          << " from file.. ";
      _nd = num_points_to_load;
      if (_enable_tags) {
        if (tag_filename == nullptr) {
          for (unsigned i = 0; i < num_points_to_load; i++) {
            _tag_to_location[i] = i;
            _location_to_tag[i] = i;
          }
        } else {
          if (file_exists(tag_filename)) {
            LOG_KNOWHERE_DEBUG_ << "Loading tags from " << tag_filename
                                << " for vamana index build";
            std::unique_ptr<TagT[]> tag_data = nullptr;
            size_t npts, ndim;
            diskann::load_bin(tag_filename, tag_data, npts, ndim);
            if (npts != num_points_to_load) {
              std::stringstream sstream;
              sstream << "Loaded " << npts
                      << " tags instead of expected number: "
                      << num_points_to_load;
              LOG(ERROR) << sstream.str();
              throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__,
                                          __FILE__, __LINE__);
            }
            for (size_t i = 0; i < npts; i++) {
              _tag_to_location[tag_data[i]] = (unsigned) i;
              _location_to_tag[(unsigned) i] = tag_data[i];
            }
          } else {
            LOG(ERROR) << "Tag file " << tag_filename
                       << " does not exist. Exiting...";
            throw diskann::ANNException(
                std::string("Tag file") + tag_filename + " does not exist", -1,
                __FUNCSIG__, __FILE__, __LINE__);
          }
        }
      }
    }

    generate_frozen_point();
    link(parameters);  // Primary func for creating nsg graph

    if (_support_eager_delete) {
      update_in_graph();  // copying values to in_graph
    }

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < _nd; i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG_KNOWHERE_INFO_ << "Index built with degree: max:" << max << "  avg:"
                         << (float) total / (float) (_nd + _num_frozen_pts)
                         << "  min:" << min << "  count(deg<2):" << cnt;
    }
    _width = (std::max)((unsigned) max, _width);
    _has_built = true;
  }

  template<typename T, typename TagT>
  template<typename IdType>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search(const T       *query,
                                                       const size_t   K,
                                                       const unsigned L,
                                                       IdType        *indices,
                                                       float *distances) {
    ScratchStoreManager<T> manager(_query_scratch);
    auto                   scratch = manager.scratch_space();

    return search_impl(query, K, L, indices, distances, scratch);
  }

  template<typename T, typename TagT>
  template<typename IdType>
  std::pair<uint32_t, uint32_t> Index<T, TagT>::search_impl(
      const T *query, const size_t K, const unsigned L, IdType *indices,
      float *distances, InMemQueryScratch<T> &scratch) {
    std::vector<Neighbor>    &expanded_nodes_info = scratch.pool();
    tsl::robin_set<unsigned> &expanded_nodes_ids = scratch.visited();
    std::vector<unsigned>    &des = scratch.des();
    std::vector<Neighbor>     best_L_nodes = scratch.best_l_nodes();
    tsl::robin_set<unsigned> &inserted_into_pool_rs =
        scratch.inserted_into_pool_rs();
    boost::dynamic_bitset<> &inserted_into_pool_bs =
        scratch.inserted_into_pool_bs();

    std::vector<unsigned> init_ids;

    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

    if (init_ids.size() == 0) {
      init_ids.emplace_back(_ep);
    }
    T *aligned_query = scratch.aligned_query;
    memcpy(aligned_query, query, _dim * sizeof(T));

    if (_normalize_vecs) {
      normalize((float *) aligned_query, _dim);
    }

    auto retval = iterate_to_fixed_point(
        aligned_query, L, init_ids, expanded_nodes_info, expanded_nodes_ids,
        best_L_nodes, des, inserted_into_pool_rs, inserted_into_pool_bs, true,
        true);

    size_t pos = 0;
    for (auto it : best_L_nodes) {
      if (it.id < _max_points) {
        indices[pos] =
            (IdType) it.id;  // safe because our indices are always uint32_t and
                             // IDType will be uint32_t or uint64_t
        if (distances != nullptr) {
          distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT
                               ? -1 * it.distance
                               : it.distance;
        }
        pos++;
      }
      if (pos == K)
        break;
    }
    return retval;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::search_with_tags(const T *query, const uint64_t K,
                                          const unsigned L, TagT *tags,
                                          float            *distances,
                                          std::vector<T *> &res_vectors) {
    ScratchStoreManager<T> manager(_query_scratch);
    auto                   scratch = manager.scratch_space();

    if (L > scratch.search_l) {
      scratch.resize_for_query(L);
      LOG_KNOWHERE_DEBUG_
          << "Expanding query scratch_space. Was created with Lsize: "
          << scratch.search_l << " but search L is: " << L;
    }
    _u32  *indices = scratch.indices;             // new unsigned[L];
    float *dist_interim = scratch.interim_dists;  // new float[L];
    search_impl(query, L, L, indices, dist_interim, scratch);

    std::shared_lock<std::shared_timed_mutex> ulock(_update_lock);
    // std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
    size_t                                    pos = 0;

    for (int i = 0; i < (int) L; ++i)
      if (_location_to_tag.find(indices[i]) != _location_to_tag.end()) {
        tags[pos] = _location_to_tag[indices[i]];

        // assuming that client has already initialized enough space if they
        // want the result vectors returned.
        if (res_vectors.size() > 0) {
          memcpy(res_vectors[pos], _data + ((size_t) indices[i]) * _aligned_dim,
                 _dim * sizeof(T));
          // res_vectors[i] = _data + indices[i] * _aligned_dim;
        }

        if (distances != nullptr) {
          distances[pos] = _dist_metric == INNER_PRODUCT ? -1 * dist_interim[i]
                                                         : dist_interim[i];
        }
        pos++;
        // in the weird case that res_vectors.size() != k, terminate at the
        // lower value.
        if (pos == K || pos == res_vectors.size())
          break;
      }

    return pos;
  }

  template<typename T, typename TagT>
  size_t Index<T, TagT>::get_num_points() {
    return _nd;
  }

  template<typename T, typename TagT>
  T *Index<T, TagT>::get_data() {
    if (_num_frozen_pts > 0) {
      T     *ret_data = nullptr;
      size_t allocSize = ((size_t) _nd) * _aligned_dim * sizeof(T);
      alloc_aligned(((void **) &ret_data), allocSize, 8 * sizeof(T));
      memset((void*) ret_data, 0, allocSize);
      memcpy(ret_data, _data, allocSize);
      return ret_data;
    }
    return _data;
  }
  template<typename T, typename TagT>
  size_t Index<T, TagT>::return_max_points() {
    return _max_points;
  }

  /*************************************************
   *      Support for Incremental Update
   *************************************************/

  // in case we add ''frozen'' auxiliary points to the dataset, these are not
  // visible to external world, we generate them here and update our dataset
  template<typename T, typename TagT>
  int Index<T, TagT>::generate_frozen_point() {
    if (_num_frozen_pts == 0)
      return 0;

    if (_nd == 0) {
      memset((void *) (_data + (_max_points) *_aligned_dim), 0,
             _aligned_dim * sizeof(T));
      return 1;
    }
    size_t res = calculate_entry_point();
    memcpy(_data + _max_points * _aligned_dim, _data + res * _aligned_dim,
           _aligned_dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::enable_delete() {
    assert(_enable_tags);

    if (!_enable_tags) {
      diskann::cerr << "Tags must be instantiated for deletions" << std::endl;
      return -2;
    }

    if (_data_compacted) {
      for (unsigned slot = (unsigned) _nd; slot < _max_points; ++slot) {
        _empty_slots.insert(slot);
      }
    }

    _lazy_done = false;
    _eager_done = false;

    if (_support_eager_delete) {
      _in_graph.resize(_max_points + _num_frozen_pts);
      _in_graph.reserve(_max_points + _num_frozen_pts);
      update_in_graph();
    }
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::release_location() {
    LockGuard guard(_change_lock);
    _nd--;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::eager_delete(const TagT tag, const Parameters &parameters,
                                   int delete_mode) {
    if (_lazy_done && (!_data_compacted)) {
      LOG(ERROR) << "Lazy delete requests issued but data not consolidated, "
                    "cannot proceed with eager deletes.";
      return -1;
    }

    unsigned id;  // since we will return if tag is not found, ok to leave it
                  // uninitialized.
    {
      // std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
      std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        diskann::cerr << "Delete tag " << tag << " not found" << std::endl;
        return -1;
      }
      id = _tag_to_location[tag];
    }

    {
      // std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);
      _location_to_tag.erase(_tag_to_location[tag]);
      _tag_to_location.erase(tag);
    }

    {
      // id will be valid because if not, it'll return in the {} above.
      std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
      _delete_set.insert(id);
      _empty_slots.insert(id);
    }

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    // delete point from out-neighbors' in-neighbor list
    {
      LockGuard guard(_locks[id]);
      for (size_t i = 0; i < _final_graph[id].size(); i++) {
        unsigned j = _final_graph[id][i];
        {
          LockGuard guard(_locks_in[j]);
          for (unsigned k = 0; k < _in_graph[j].size(); k++) {
            if (_in_graph[j][k] == id) {
              _in_graph[j].erase(_in_graph[j].begin() + k);
              break;
            }
          }
        }
      }
    }

    tsl::robin_set<unsigned> in_nbr;
    {
      LockGuard guard(_locks_in[id]);
      for (unsigned i = 0; i < _in_graph[id].size(); i++)
        in_nbr.insert(_in_graph[id][i]);
    }
    assert(_in_graph[id].size() == in_nbr.size());

    std::vector<Neighbor>    pool, tmp;
    tsl::robin_set<unsigned> visited;
    std::vector<unsigned>    intersection;
    unsigned                 Lindex = parameters.Get<unsigned>("L");
    std::vector<unsigned>    init_ids;

    if (delete_mode == 2) {
      // constructing list of in-neighbors to be processed
      get_expanded_nodes(id, Lindex, init_ids, pool, visited);

      for (auto node : visited) {
        if (in_nbr.find(node) != in_nbr.end()) {
          intersection.push_back(node);
        }
      }
    }

    // deleting deleted point from all in-neighbors' out-neighbor list
    for (auto it : in_nbr) {
      LockGuard guard(_locks[it]);
      _final_graph[it].erase(
          std::remove(_final_graph[it].begin(), _final_graph[it].end(), id),
          _final_graph[it].end());
    }

    tsl::robin_set<unsigned> candidate_set;
    std::vector<Neighbor>    expanded_nghrs;
    std::vector<Neighbor>    result;

    for (size_t i = 0; i < intersection.size(); i++) {
      auto ngh = intersection[i];

      candidate_set.clear();
      expanded_nghrs.clear();
      result.clear();

      {
        std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
        if (_delete_set.find(ngh) != _delete_set.end())
          continue;
      }

      {
        LockGuard guard(_locks[ngh]);

        // constructing candidate set from out-neighbors and out-neighbors of
        // ngh and id
        {  // should a shared reader lock on delete_lock be held here at the
           // beginning of the two for loops or should it be held and release
           // for ech iteration of the for loops? Which is faster?

          std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
          for (auto j : _final_graph[id]) {
            if ((j != id) && (j != ngh) &&
                (_delete_set.find(j) == _delete_set.end()))
              candidate_set.insert(j);
          }

          for (auto j : _final_graph[ngh]) {
            if ((j != id) && (j != ngh) &&
                (_delete_set.find(j) == _delete_set.end()))
              candidate_set.insert(j);
          }
        }

        for (auto j : candidate_set)
          expanded_nghrs.push_back(
              Neighbor(j,
                       _distance(_data + _aligned_dim * (size_t) ngh,
                                 _data + _aligned_dim * (size_t) j,
                                 (size_t) _aligned_dim),
                       true));
        std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
        occlude_list(expanded_nghrs, alpha, range, maxc, result);

        // deleting ngh from its old out-neighbors' in-neighbor list
        for (auto iter : _final_graph[ngh]) {
          {
            LockGuard guard(_locks_in[iter]);
            for (unsigned k = 0; k < _in_graph[iter].size(); k++) {
              if (_in_graph[iter][k] == ngh) {
                _in_graph[iter].erase(_in_graph[iter].begin() + k);
                break;
              }
            }
          }
        }

        _final_graph[ngh].clear();

        // updating out-neighbors and in-neighbors of ngh
        {
          std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
          for (size_t i = 0; i < result.size(); i++) {
            auto j = result[i];
            if (_delete_set.find(j.id) == _delete_set.end()) {
              _final_graph[ngh].push_back(j.id);
              {
                LockGuard guard(_locks_in[j.id]);
                if (std::find(_in_graph[j.id].begin(), _in_graph[j.id].end(),
                              ngh) == _in_graph[j.id].end()) {
                  _in_graph[j.id].emplace_back(ngh);
                }
              }
            }
          }
        }
      }
    }

    _final_graph[id].clear();
    _in_graph[id].clear();

    release_location();

    _eager_done = true;
    _data_compacted = false;
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::update_in_graph() {
    //  diskann::cout << "Updating in_graph.....";
    for (unsigned i = 0; i < _in_graph.size(); i++)
      _in_graph[i].clear();

    for (size_t i = 0; i < _final_graph.size();
         i++)  // copying to in-neighbor graph
      for (size_t j = 0; j < _final_graph[i].size(); j++)
        _in_graph[_final_graph[i][j]].emplace_back((_u32) i);
  }

  // Do not call consolidate_deletes() if you have not locked _change_lock.
  // Returns number of live points left after consolidation
  // proxy inserts all nghrs of deleted points
  // original approach
  template<typename T, typename TagT>
  size_t Index<T, TagT>::consolidate_deletes(const Parameters &parameters) {
    if (_eager_done) {
      diskann::cout
          << "In consolidate_deletes(), _eager_done is true. So exiting."
          << std::endl;
      return 0;
    }

    assert(_enable_tags);
    assert(_delete_set.size() <= _nd);
    assert(_empty_slots.size() + _nd == _max_points);

    const unsigned range = parameters.Get<unsigned>("R");
    const unsigned maxc = parameters.Get<unsigned>("C");
    const float    alpha = parameters.Get<float>("alpha");

    _u64     total_pts = _max_points + _num_frozen_pts;
    unsigned block_size = 1 << 10;
    _s64     total_blocks = DIV_ROUND_UP(total_pts, block_size);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(total_blocks);
    for (_s64 block_i = 0; block_i < total_blocks; ++block_i) {
      futures.emplace_back(_build_thread_pool->push([&, block = block_i]() {
        tsl::robin_set<unsigned> candidate_set;
        std::vector<Neighbor>    expanded_nghrs;
        std::vector<Neighbor>    result;

        for (_s64 i = block * block_size;
             i < (_s64) ((block + 1) * block_size) &&
             i < (_s64) (_max_points + _num_frozen_pts);
             i++) {
          if ((_delete_set.find((_u32) i) == _delete_set.end()) &&
              (_empty_slots.find((_u32) i) == _empty_slots.end())) {
            candidate_set.clear();
            expanded_nghrs.clear();
            result.clear();

            bool modify = false;
            for (auto ngh : _final_graph[(_u32) i]) {
              if (_delete_set.find(ngh) != _delete_set.end()) {
                modify = true;

                // Add outgoing links from
                for (auto j : _final_graph[ngh])
                  if (_delete_set.find(j) == _delete_set.end())
                    candidate_set.insert(j);
              } else {
                candidate_set.insert(ngh);
              }
            }
            if (modify) {
              for (auto j : candidate_set) {
                expanded_nghrs.push_back(
                    Neighbor(j,
                             _distance(_data + _aligned_dim * i,
                                       _data + _aligned_dim * (size_t) j,
                                       (unsigned) _aligned_dim),
                             true));
              }

              std::sort(expanded_nghrs.begin(), expanded_nghrs.end());
              occlude_list(expanded_nghrs, alpha, range, maxc, result);

              _final_graph[(_u32) i].clear();
              for (auto j : result) {
                if (j.id != (_u32) i &&
                    (_delete_set.find(j.id) == _delete_set.end()))
                  _final_graph[(_u32) i].push_back(j.id);
              }
            }
          }
        }
      }));
    }

    knowhere::WaitAllSuccess(futures);

    if (_support_eager_delete)
      update_in_graph();

    for (auto iter : _delete_set) {
      _empty_slots.insert(iter);
    }
    _nd -= _delete_set.size();

    _data_compacted = _delete_set.size() == 0;

    auto stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Time taken for consolidate_deletes() "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(
                         stop - start)
                         .count()
                  << "s." << std::endl;

    return _nd;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::consolidate(Parameters &parameters) {
    consolidate_deletes(parameters);
    compact_data();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_frozen_point() {
    if (_nd < _max_points) {
      if (_num_frozen_pts > 0) {
        // set new _ep to be frozen point
        _ep = (_u32) _nd;
        if (!_final_graph[_max_points].empty()) {
          for (unsigned i = 0; i < _nd; i++)
            for (unsigned j = 0; j < _final_graph[i].size(); j++)
              if (_final_graph[i][j] == _max_points)
                _final_graph[i][j] = (_u32) _nd;

          _final_graph[_nd].clear();
          for (unsigned k = 0; k < _final_graph[_max_points].size(); k++)
            _final_graph[_nd].emplace_back(_final_graph[_max_points][k]);

          _final_graph[_max_points].clear();
          if (_support_eager_delete)
            update_in_graph();

          memcpy((void *) (_data + _aligned_dim * _nd),
                 _data + (size_t) _aligned_dim * _max_points, sizeof(T) * _dim);
          memset((void *) (_data + (size_t) _aligned_dim * _max_points), 0,
                 sizeof(T) * _aligned_dim);
        }
      }
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data_for_search() {
    compact_data();
    compact_frozen_point();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data_for_insert() {
    compact_data();

    size_t max = 0, min = 1 << 30, total = 0, cnt = 0;
    for (size_t i = 0; i < (_nd + _num_frozen_pts); i++) {
      auto &pool = _final_graph[i];
      max = (std::max)(max, pool.size());
      min = (std::min)(min, pool.size());
      total += pool.size();
      if (pool.size() < 2)
        cnt++;
    }
    if (min > max)
      min = max;
    if (_nd > 0) {
      LOG_KNOWHERE_INFO_ << "Index built with degree: max:" << max << "  avg:"
                         << (float) total / (float) (_nd + _num_frozen_pts)
                         << "  min:" << min << "  count(deg<2):" << cnt;
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::compact_data() {
    if (!_dynamic_index)
      return;

    if (!_lazy_done && !_eager_done)
      return;

    if (_data_compacted) {
      diskann::cerr
          << "Warning! Calling compact_data() when _data_compacted is true!"
          << std::endl;
      return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto fnstart = start;

    std::vector<unsigned> new_location = std::vector<unsigned>(
        _max_points + _num_frozen_pts, (_u32) _max_points);

    _u32 new_counter = 0;

    for (_u32 old_counter = 0; old_counter < _max_points + _num_frozen_pts;
         old_counter++) {
      if (_location_to_tag.find(old_counter) != _location_to_tag.end()) {
        new_location[old_counter] = new_counter;
        new_counter++;
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    // If start node is removed, replace it.
    if (_delete_set.find(_ep) != _delete_set.end()) {
      diskann::cerr << "Replacing start node which has been deleted... "
                    << std::flush;
      auto old_ep = _ep;
      // First active neighbor of old start node is new start node
      for (auto iter : _final_graph[_ep])
        if (_delete_set.find(iter) != _delete_set.end()) {
          _ep = iter;
          break;
        }
      if (_ep == old_ep) {
        throw diskann::ANNException(
            "ERROR: Did not find a replacement for start node.", -1,
            __FUNCSIG__, __FILE__, __LINE__);
      } else {
        assert(_delete_set.find(_ep) == _delete_set.end());
        //        diskann::cout << "New start node is " << _ep << std::endl;
      }
    }

    start = std::chrono::high_resolution_clock::now();
    double copy_time = 0;
    for (unsigned old = 0; old <= _max_points; ++old) {
      if ((new_location[old] < _max_points) ||
          (old == _max_points)) {  // If point continues to exist

        // Renumber nodes to compact the order
        for (size_t i = 0; i < _final_graph[old].size(); ++i) {
          if (new_location[_final_graph[old][i]] > _final_graph[old][i]) {
            std::stringstream sstream;
            sstream << "Error in compact_data(). Found point: " << old
                    << " whose " << i << "th neighbor has new location "
                    << new_location[_final_graph[old][i]]
                    << " that is greater than its old location: "
                    << _final_graph[old][i];
            if (_delete_set.find(_final_graph[old][i]) != _delete_set.end()) {
              sstream << std::endl
                      << " Point: " << old << " index: " << i
                      << " neighbor: " << _final_graph[old][i]
                      << " found in delete set of size: " << _delete_set.size()
                      << std::endl;
            } else {
              sstream << " Point: " << old
                      << " neighbor: " << _final_graph[old][i]
                      << " NOT found in delete set of size: "
                      << _delete_set.size() << std::endl;
            }
            throw diskann::ANNException(sstream.str(), -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
          }
          _final_graph[old][i] = new_location[_final_graph[old][i]];
        }

        if (_support_eager_delete)
          for (size_t i = 0; i < _in_graph[old].size(); ++i) {
            if (new_location[_in_graph[old][i]] <= _in_graph[old][i])
              _in_graph[old][i] = new_location[_in_graph[old][i]];
          }

        // Move the data and adj list to the correct position
        auto c_start = std::chrono::high_resolution_clock::now();
        if (new_location[old] != old) {
          assert(new_location[old] < old);
          _final_graph[new_location[old]].swap(_final_graph[old]);
          if (_support_eager_delete)
            _in_graph[new_location[old]].swap(_in_graph[old]);
          memcpy((void *) (_data + _aligned_dim * (size_t) new_location[old]),
                 (void *) (_data + _aligned_dim * (size_t) old),
                 _aligned_dim * sizeof(T));
        }
        auto c_stop = std::chrono::high_resolution_clock::now();
        copy_time += std::chrono::duration_cast<std::chrono::duration<double>>(
                         c_stop - c_start)
                         .count();

      } else {
        _final_graph[old].clear();
      }
    }
    stop = std::chrono::high_resolution_clock::now();

    start = std::chrono::high_resolution_clock::now();
    _tag_to_location.clear();
    for (auto iter : _location_to_tag) {
      _tag_to_location[iter.second] = new_location[iter.first];
    }
    _location_to_tag.clear();
    for (auto iter : _tag_to_location) {
      _location_to_tag[iter.second] = iter.first;
    }

    for (_u64 old = _nd; old < _max_points; ++old) {
      _final_graph[old].clear();
    }
    _delete_set.clear();
    _empty_slots.clear();
    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

    _lazy_done = false;
    _eager_done = false;
    _data_compacted = true;
    stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Time taken for compact_data(): "
                  << std::chrono::duration_cast<std::chrono::duration<double>>(
                         stop - fnstart)
                         .count()
                  << "s." << std::endl;
  }

  // Do not call reserve_location() if you have not locked _change_lock.
  // It is not thread safe.
  template<typename T, typename TagT>
  int Index<T, TagT>::reserve_location() {
    LockGuard guard(_change_lock);
    if (_nd >= _max_points) {
      return -1;
    }
    unsigned location;
    if (_data_compacted) {
      location = (unsigned) _nd;
      _empty_slots.erase(location);
    } else {
      // no need of delete_lock here, _change_lock will ensure no other thread
      // executes this block of code
      assert(_empty_slots.size() != 0);
      assert(_empty_slots.size() + _nd == _max_points);

      auto iter = _empty_slots.begin();
      location = *iter;
      _empty_slots.erase(iter);
      _delete_set.erase(location);
    }

    ++_nd;
    return location;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_point(unsigned old_location,
                                        unsigned new_location) {
    for (unsigned i = 0; i < _nd; i++)
      for (unsigned j = 0; j < _final_graph[i].size(); j++)
        if (_final_graph[i][j] == old_location)
          _final_graph[i][j] = (unsigned) new_location;

    _final_graph[new_location].clear();
    for (unsigned k = 0; k < _final_graph[_nd].size(); k++)
      _final_graph[new_location].emplace_back(_final_graph[old_location][k]);

    _final_graph[old_location].clear();

    if (_support_eager_delete) {
      update_in_graph();
    }
    memcpy((void *) (_data + (size_t) _aligned_dim * new_location),
           _data + (size_t) _aligned_dim * old_location,
           sizeof(T) * _aligned_dim);
    memset((void *) (_data + (size_t) _aligned_dim * old_location), 0,
           sizeof(T) * _aligned_dim);
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::reposition_frozen_point_to_end() {
    if (_num_frozen_pts == 0)
      return;

    if (_nd == _max_points) {
      diskann::cout
          << "Not repositioning frozen point as it is already at the end."
          << std::endl;
      return;
    }
    reposition_point((_u32) _nd, (_u32) _max_points);
    _ep = (_u32) _max_points;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::resize(size_t new_max_points) {
    auto start = std::chrono::high_resolution_clock::now();
    assert(_empty_slots.size() ==
           0);  // should not resize if there are empty slots.
    T *new_data;
    alloc_aligned((void **) &new_data,
                  (new_max_points + 1) * _aligned_dim * sizeof(T),
                  8 * sizeof(T));
    memcpy(new_data, _data, (_max_points + 1) * _aligned_dim * sizeof(T));
    aligned_free(_data);
    _data = new_data;
    _final_graph.resize(new_max_points + 1);
    _locks = std::vector<std::mutex>(new_max_points + 1);
    if (_support_eager_delete) {
      _in_graph.resize(new_max_points + 1);
      _locks_in = std::vector<std::mutex>(new_max_points + 1);
    }

    reposition_point((_u32) _max_points, (_u32) new_max_points);
    _max_points = new_max_points;
    _ep = (_u32) new_max_points;

    for (auto i = _nd; i < _max_points; i++) {
      _empty_slots.insert((uint32_t) i);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    diskann::cout << "Resizing took: "
                  << std::chrono::duration<double>(stop - start).count() << "s"
                  << std::endl;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::insert_point(const T *point, const TagT tag) {
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    unsigned range = _indexingRange;  // parameters.Get<unsigned>("R");
    //    assert(_has_built);

    // Avoiding the use of temporary containers.
    ScratchStoreManager<T> manager(_query_scratch);
    auto                   scratch = manager.scratch_space();

    std::vector<Neighbor>    &pool = scratch.pool();
    tsl::robin_set<unsigned> &visited = scratch.visited();

    {
      // std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
      std::shared_lock<std::shared_timed_mutex> tsl(_tag_lock);
      if (_enable_tags &&
          (_tag_to_location.find(tag) != _tag_to_location.end())) {
        // This is a repeat of lazy_delete, but we can't call
        // that function because we are taking many locks here. Hence
        // the repeated code.
        tsl.unlock();
        std::unique_lock<std::shared_timed_mutex> tul(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> tdl(_delete_lock);
        _lazy_done = true;
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    auto location = reserve_location();
    if (location == -1) {
      lock.unlock();
      std::unique_lock<std::shared_timed_mutex> growth_lock(_update_lock);

      if (_nd >= _max_points) {
        auto new_max_points = (size_t) (_max_points * INDEX_GROWTH_FACTOR);
        resize(new_max_points);
      }
      growth_lock.unlock();
      lock.lock();
      location = reserve_location();
      // TODO: Consider making this a while/do_while loop so that we retry
      // instead of terminating.
      if (location == -1) {
        throw diskann::ANNException(
            "Cannot reserve location even after expanding graph. Terminating.",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }
    }

    {
      std::unique_lock<std::shared_timed_mutex> lock(_tag_lock);

      _tag_to_location[tag] = location;
      _location_to_tag[location] = tag;
    }

    auto offset_data = _data + (size_t) _aligned_dim * location;
    memset((void *) offset_data, 0, sizeof(T) * _aligned_dim);
    memcpy((void *) offset_data, point, sizeof(T) * _dim);

    if (_normalize_vecs) {
      normalize((float *) offset_data, _dim);
    }

    pool.clear();
    visited.clear();
    std::vector<unsigned> pruned_list;
    unsigned Lindex = _indexingQueueSize;  // parameters.Get<unsigned>("L");

    std::vector<unsigned> init_ids;
    get_expanded_nodes(location, Lindex, init_ids, pool, visited, scratch.des(),
                       scratch.best_l_nodes(), scratch.inserted_into_pool_rs(),
                       scratch.inserted_into_pool_bs());

    for (unsigned i = 0; i < pool.size(); i++)
      if (pool[i].id == (unsigned) location) {
        pool.erase(pool.begin() + i);
        visited.erase((unsigned) location);
        break;
      }

    prune_neighbors(location, pool, pruned_list);
    assert(_final_graph.size() == _max_points + _num_frozen_pts);

    if (_support_eager_delete) {
      for (unsigned i = 0; i < _final_graph[location].size(); i++) {
        {
          LockGuard guard(_locks_in[_final_graph[location][i]]);
          _in_graph[_final_graph[location][i]].erase(
              std::remove(_in_graph[_final_graph[location][i]].begin(),
                          _in_graph[_final_graph[location][i]].end(), location),
              _in_graph[_final_graph[location][i]].end());
        }
      }
    }

    _final_graph[location].clear();
    _final_graph[location].shrink_to_fit();
    _final_graph[location].reserve((_u64) (range * GRAPH_SLACK_FACTOR * 1.05));

    assert(!pruned_list.empty());
    {
      LockGuard guard(_locks[location]);
      for (auto link : pruned_list) {
        _final_graph[location].emplace_back(link);
        if (_support_eager_delete) {
          LockGuard guard(_locks_in[link]);
          _in_graph[link].emplace_back(location);
        }
      }
    }

    assert(_final_graph[location].size() <= range);
    if (_support_eager_delete) {
      inter_insert(location, pruned_list, 1);
    } else {
      inter_insert(location, pruned_list, 0);
    }
    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::disable_delete(const Parameters &parameters,
                                     const bool        consolidate) {
    if (!_enable_tags) {
      diskann::cerr << "Point tag array not instantiated" << std::endl;
      throw diskann::ANNException("Point tag array not instantiated", -1,
                                  __FUNCSIG__, __FILE__, __LINE__);
    }
    if (_eager_done) {
      if (_tag_to_location.size() != _nd) {
        diskann::cerr << "Tags to points array wrong sized : "
                      << _tag_to_location.size() << std::endl;
        return -2;
      }
    } else if (_tag_to_location.size() + _delete_set.size() != _nd) {
      diskann::cerr << "Tags to points array wrong sized"
                    << "\n_tag_to_location.size():  " << _tag_to_location.size()
                    << "\n_delete_set.size():  " << _delete_set.size()
                    << "\n_nd:  " << _nd << std::endl;
      return -2;
    }

    if (_eager_done) {
      if (_location_to_tag.size() != _nd) {
        diskann::cerr << "Points to tags array wrong sized" << std::endl;
        return -3;
      }
    } else if (_location_to_tag.size() + _delete_set.size() != _nd) {
      diskann::cerr << "Points to tags array wrong sized "
                    << _location_to_tag.size() << " " << _delete_set.size()
                    << std::endl;
      return -3;
    }

    if (consolidate) {
      std::unique_lock<std::shared_timed_mutex> lock(_update_lock);
      consolidate_deletes(parameters);
    }

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const TagT &tag) {
    if ((_eager_done) && (!_data_compacted)) {
      diskann::cerr << "Eager delete requests were issued but data was not "
                       "compacted, cannot proceed with lazy_deletes"
                    << std::endl;
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    {
      // std::shared_lock<std::shared_timed_mutex> l(_tag_lock);
      std::shared_lock<std::shared_timed_mutex> l(_tag_lock);

      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        diskann::cerr << "Delete tag not found" << std::endl;
        return -1;
      }
      assert(_tag_to_location[tag] < _max_points);
    }

    {
      std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> l(_delete_lock);
      _delete_set.insert(_tag_to_location[tag]);
    }

    {
      // std::unique_lock<std::shared_timed_mutex> l(_tag_lock);
      std::unique_lock<std::shared_timed_mutex> l(_tag_lock);
      _location_to_tag.erase(_tag_to_location[tag]);
      _tag_to_location.erase(tag);
    }

    return 0;
  }

  // TODO: Check if this function needs a shared_lock on _tag_lock.
  template<typename T, typename TagT>
  int Index<T, TagT>::lazy_delete(const tsl::robin_set<TagT> &tags,
                                  std::vector<TagT>          &failed_tags) {
    if (failed_tags.size() > 0) {
      diskann::cerr << "failed_tags should be passed as an empty list"
                    << std::endl;
      return -3;
    }
    if ((_eager_done) && (!_data_compacted)) {
      diskann::cout << "Eager delete requests were issued but data was not "
                       "compacted, cannot proceed with lazy_deletes"
                    << std::endl;
      return -2;
    }
    std::shared_lock<std::shared_timed_mutex> lock(_update_lock);
    _lazy_done = true;

    for (auto tag : tags) {
      //      assert(_tag_to_location[tag] < _max_points);
      if (_tag_to_location.find(tag) == _tag_to_location.end()) {
        failed_tags.push_back(tag);
      } else {
        _delete_set.insert(_tag_to_location[tag]);
        _location_to_tag.erase(_tag_to_location[tag]);
        _tag_to_location.erase(tag);
      }
    }

    return 0;
  }

  template<typename T, typename TagT>
  int Index<T, TagT>::extract_data(
      T *ret_data, std::unordered_map<TagT, unsigned> &tag_to_location) {
    if (!_data_compacted) {
      diskann::cerr
          << "Error! Data not compacted. Cannot give access to private data."
          << std::endl;
      return -1;
    }
    std::memset((void*) ret_data, 0, (size_t) _aligned_dim * _nd * sizeof(T));
    std::memcpy(ret_data, _data, (size_t) (_aligned_dim) *_nd * sizeof(T));
    tag_to_location = _tag_to_location;
    return 0;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_location_to_tag(
      std::unordered_map<unsigned, TagT> &ret_loc_to_tag) {
    ret_loc_to_tag = _location_to_tag;
  }

  template<typename T, typename TagT>
  bool Index<T, TagT>::hasIndexBeenSaved() {
    return _is_saved;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::get_active_tags(tsl::robin_set<TagT> &active_tags) {
    active_tags.clear();
    for (auto iter : _tag_to_location) {
      active_tags.insert(iter.first);
    }
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::print_status() const {
    diskann::cout << "------------------- Index object: " << (uint64_t) this
                  << " -------------------" << std::endl;
    diskann::cout << "Number of points: " << _nd << std::endl;
    diskann::cout << "Graph size: " << _final_graph.size() << std::endl;
    diskann::cout << "Location to tag size: " << _location_to_tag.size()
                  << std::endl;
    diskann::cout << "Tag to location size: " << _tag_to_location.size()
                  << std::endl;
    diskann::cout << "Number of empty slots: " << _empty_slots.size()
                  << std::endl;
    // print_delete_set();
    // are_deleted_points_in_graph();
    diskann::cout << std::boolalpha
                  << "Data compacted: " << this->_data_compacted
                  << " Lazy done: " << this->_lazy_done
                  << " Eager done: " << this->_eager_done << std::endl;
    diskann::cout << "---------------------------------------------------------"
                     "------------"
                  << std::endl;
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::optimize_index_layout() {  // use after build or load
    _data_len = (_aligned_dim + 1) * sizeof(float);
    _neighbor_len = (_width + 1) * sizeof(unsigned);
    _node_size = _data_len + _neighbor_len;
    _opt_graph = (char *) malloc(_node_size * _nd);
    for (unsigned i = 0; i < _nd; i++) {
      char *cur_node_offset = _opt_graph + i * _node_size;
      float cur_norm = norm_l2sqr(_data + i * _aligned_dim, _aligned_dim);
      std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
      std::memcpy(cur_node_offset + sizeof(float), _data + i * _aligned_dim,
                  _data_len - sizeof(float));

      cur_node_offset += _data_len;
      unsigned k = _final_graph[i].size();
      std::memcpy(cur_node_offset, &k, sizeof(unsigned));
      std::memcpy(cur_node_offset + sizeof(unsigned), _final_graph[i].data(),
                  k * sizeof(unsigned));
      std::vector<unsigned>().swap(_final_graph[i]);
    }
    _final_graph.clear();
    _final_graph.shrink_to_fit();
  }

  template<typename T, typename TagT>
  void Index<T, TagT>::search_with_optimized_layout(const T *query, size_t K,
                                                    size_t    L,
                                                    unsigned *indices) {
    auto fast_distance =
        get_distance_function<T>(diskann::Metric::INNER_PRODUCT);
    std::vector<Neighbor> retset(L + 1);
    std::vector<unsigned> init_ids(L);
    // std::mt19937 rng(rand());
    // GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

    boost::dynamic_bitset<> flags{_nd, 0};
    unsigned                tmp_l = 0;
    unsigned               *neighbors =
        (unsigned *) (_opt_graph + _node_size * _ep + _data_len);
    unsigned MaxM_ep = *neighbors;
    neighbors++;

    for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
      init_ids[tmp_l] = neighbors[tmp_l];
      flags[init_ids[tmp_l]] = true;
    }

    while (tmp_l < L) {
      unsigned id = rand() % _nd;
      if (flags[id])
        continue;
      flags[id] = true;
      init_ids[tmp_l] = id;
      tmp_l++;
    }

    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      if (id >= _nd)
        continue;
#if defined(__ARM_NEON) && defined(__aarch64__)
      __builtin_prefetch(_opt_graph + _node_size * id, 0, 3);
#else
      _mm_prefetch(_opt_graph + _node_size * id, _MM_HINT_T0);
#endif
    }
    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
      unsigned id = init_ids[i];
      if (id >= _nd)
        continue;
      T    *x = (T *) (_opt_graph + _node_size * id);
      float norm_x = *x;
      x++;
      float dist = norm_x - 2 * fast_distance(x, query, (size_t) _aligned_dim);
      retset[i] = Neighbor(id, dist, true);
      flags[id] = true;
      L++;
    }
    // std::cout<<L<<std::endl;

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int) L) {
      int nk = L;

      if (retset[k].flag) {
        retset[k].flag = false;
        unsigned n = retset[k].id;

#if defined(__ARM_NEON) && defined(__aarch64__)
        __builtin_prefetch(_opt_graph + _node_size * n + _data_len, 0, 3);
#else
        _mm_prefetch(_opt_graph + _node_size * n + _data_len, _MM_HINT_T0);
#endif
        unsigned *neighbors =
            (unsigned *) (_opt_graph + _node_size * n + _data_len);
        unsigned MaxM = *neighbors;
        neighbors++;
        for (unsigned m = 0; m < MaxM; ++m) {
#if defined(__ARM_NEON) && defined(__aarch64__)
          __builtin_prefetch(_opt_graph + _node_size * neighbors[m], 0, 3);
#else
          _mm_prefetch(_opt_graph + _node_size * neighbors[m], _MM_HINT_T0);
#endif
        }
        for (unsigned m = 0; m < MaxM; ++m) {
          unsigned id = neighbors[m];
          if (flags[id])
            continue;
          flags[id] = 1;
          T    *data = (T *) (_opt_graph + _node_size * id);
          float norm = *data;
          data++;
          float dist =
              norm - 2 * fast_distance(query, data, (size_t) _aligned_dim);
          if (dist >= retset[L - 1].distance)
            continue;
          Neighbor nn(id, dist, true);
          int      r = InsertIntoPool(retset.data(), L, nn);

          // if(L+1 < retset.size()) ++L;
          if (r < nk)
            nk = r;
        }
      }
      if (nk <= k)
        k = nk;
      else
        ++k;
    }
    for (size_t i = 0; i < K; i++) {
      indices[i] = retset[i].id;
    }
  }

  /*  Internals of the library */
  template<typename T, typename TagT>
  const float Index<T, TagT>::INDEX_GROWTH_FACTOR = 1.5f;

  // EXPORTS
  template class Index<float, int32_t>;
  template class Index<int8_t, int32_t>;
  template class Index<uint8_t, int32_t>;
  template class Index<knowhere::fp16, int32_t>;
  template class Index<knowhere::bf16, int32_t>;
  template class Index<float, uint32_t>;
  template class Index<int8_t, uint32_t>;
  template class Index<uint8_t, uint32_t>;
  template class Index<knowhere::fp16, uint32_t>;
  template class Index<knowhere::bf16, uint32_t>;
  template class Index<float, int64_t>;
  template class Index<int8_t, int64_t>;
  template class Index<uint8_t, int64_t>;
  template class Index<knowhere::fp16, int64_t>;
  template class Index<knowhere::bf16, int64_t>;
  template class Index<float, uint64_t>;
  template class Index<int8_t, uint64_t>;
  template class Index<uint8_t, uint64_t>;
  template class Index<knowhere::fp16, uint64_t>;
  template class Index<knowhere::bf16, uint64_t>;

  template std::pair<uint32_t, uint32_t>
  Index<float, uint64_t>::search<uint64_t>(const float *query, const size_t K,
                                           const unsigned L, uint64_t *indices,
                                           float *distances);
  template std::pair<uint32_t, uint32_t>
  Index<float, uint64_t>::search<uint32_t>(const float *query, const size_t K,
                                           const unsigned L, uint32_t *indices,
                                           float *distances);
  template std::pair<uint32_t, uint32_t>
  Index<uint8_t, uint64_t>::search<uint64_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint64_t *indices,
                                             float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<uint8_t, uint64_t>::search<uint32_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint32_t *indices,
                                             float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<int8_t, uint64_t>::search<uint64_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint64_t *indices,
                                            float *distances);
  template std::pair<uint32_t, uint32_t>
  Index<int8_t, uint64_t>::search<uint32_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint32_t *indices,
                                            float *distances);
  template std::pair<uint32_t, uint32_t>
  Index<knowhere::fp16, uint64_t>::search<uint64_t>(const knowhere::fp16 *query,
                                                    const size_t          K,
                                                    const unsigned        L,
                                                    uint64_t *indices,
                                                    float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<knowhere::fp16, uint64_t>::search<uint32_t>(const knowhere::fp16 *query,
                                                    const size_t          K,
                                                    const unsigned        L,
                                                    uint32_t *indices,
                                                    float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<knowhere::bf16, uint64_t>::search<uint64_t>(const knowhere::bf16 *query,
                                                    const size_t          K,
                                                    const unsigned        L,
                                                    uint64_t *indices,
                                                    float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<knowhere::bf16, uint64_t>::search<uint32_t>(const knowhere::bf16 *query,
                                                    const size_t          K,
                                                    const unsigned        L,
                                                    uint32_t *indices,
                                                    float    *distances);
  // TagT==uint32_t
  template std::pair<uint32_t, uint32_t>
  Index<float, uint32_t>::search<uint64_t>(const float *query, const size_t K,
                                           const unsigned L, uint64_t *indices,
                                           float *distances);
  template std::pair<uint32_t, uint32_t>
  Index<float, uint32_t>::search<uint32_t>(const float *query, const size_t K,
                                           const unsigned L, uint32_t *indices,
                                           float *distances);
  template std::pair<uint32_t, uint32_t>
  Index<uint8_t, uint32_t>::search<uint64_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint64_t *indices,
                                             float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<uint8_t, uint32_t>::search<uint32_t>(const uint8_t *query,
                                             const size_t K, const unsigned L,
                                             uint32_t *indices,
                                             float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<int8_t, uint32_t>::search<uint64_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint64_t *indices,
                                            float *distances);
  template std::pair<uint32_t, uint32_t>
  Index<int8_t, uint32_t>::search<uint32_t>(const int8_t *query, const size_t K,
                                            const unsigned L, uint32_t *indices,
                                            float *distances);

  template std::pair<uint32_t, uint32_t>
  Index<knowhere::fp16, uint32_t>::search<uint64_t>(const knowhere::fp16 *query,
                                                    const size_t          K,
                                                    const unsigned        L,
                                                    uint64_t *indices,
                                                    float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<knowhere::fp16, uint32_t>::search<uint32_t>(const knowhere::fp16 *query,
                                                    const size_t          K,
                                                    const unsigned        L,
                                                    uint32_t *indices,
                                                    float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<knowhere::bf16, uint32_t>::search<uint64_t>(const knowhere::bf16 *query,
                                                    const size_t          K,
                                                    const unsigned        L,
                                                    uint64_t *indices,
                                                    float    *distances);
  template std::pair<uint32_t, uint32_t>
  Index<knowhere::bf16, uint32_t>::search<uint32_t>(const knowhere::bf16 *query,
                                                    const size_t          K,
                                                    const unsigned        L,
                                                    uint32_t *indices,
                                                    float    *distances);

}  // namespace diskann
