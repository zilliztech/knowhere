// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#include <ctime>
#include <ratio>
#include <chrono>

#include "diskann/aisaq.h"
#include "diskann/aisaq_pq_reader.h"
#include "diskann/aisaq_utils.h"
#include "diskann/defaults.h"
#include "diskann/pq_flash_aisaq_index.h"
#include "diskann/utils.h"
#include "knowhere/log.h"
#include "knowhere/prometheus_client.h"
#include "diskann/memory_mapper.h"
#include "diskann/aio_context_pool.h"

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id)                                                   \
    (((uint64_t)(id)) / this->nvecs_per_sector +                               \
     this->reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id)                                               \
    ((((uint64_t)(id)) % this->nvecs_per_sector) * this->data_dim *            \
     sizeof(float))

#define PQ_CACHE_MAX_VECTORS_PCNT 100
#define PQ_CACHE_MAX_DRAM_GB 8.0f /* 8GB */

#define OFFSET_TO_NODE_NHOOD(node_buf)                                         \
    (unsigned *)((char *)node_buf + this->disk_bytes_per_point)

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *)(node_buf)

#define OFFSET_TO_NODE_PQ_NBRS(node_buf)                                       \
    (uint8_t *)(node_buf + this->disk_bytes_per_point +                        \
                sizeof(uint32_t) * (1 + this->max_degree))

namespace {
static auto async_pool =
    knowhere::ThreadPool::CreateFIFO(1, "DiskANN_Async_Cache_Making");

constexpr uint64_t kRefineBeamWidthFactor = 2;
constexpr uint64_t kBruteForceTopkRefineExpansionFactor = 2;
constexpr float kFilterThreshold = 0.93f;

constexpr uint64_t kMaxLSearch = 400;
} // namespace

namespace diskann {
using namespace diskann::defaults;

//
// Base Class Implementatons
//

template <typename T>
inline uint64_t PQFlashAisaqIndex<T>::get_node_sector(uint64_t node_id) {
    return 1 + (this->nnodes_per_sector > 0
                    ? node_id / this->nnodes_per_sector
                    : node_id * DIV_ROUND_UP(this->max_node_len,
                                             diskann::defaults::SECTOR_LEN));
}

template <typename T>
inline char *PQFlashAisaqIndex<T>::offset_to_node(char *sector_buf,
                                                  uint64_t node_id) {
    return sector_buf +
           (this->nnodes_per_sector == 0
                ? 0
                : (node_id % this->nnodes_per_sector) * this->max_node_len);
}

template <typename T>
inline char *
PQFlashAisaqIndex<T>::aisaq_offset_to_node_aisaq_data(char *node_buf) {
    return node_buf + this->disk_bytes_per_point +
           ((this->max_degree + 1) * sizeof(uint32_t));
}

template <typename T>
inline uint32_t *PQFlashAisaqIndex<T>::offset_to_node_nhood(char *node_buf) {
    return (unsigned *)(node_buf + this->disk_bytes_per_point);
}

template <typename T>
inline T *PQFlashAisaqIndex<T>::offset_to_node_coords(char *node_buf) {
    return (T *)(node_buf);
}

template <typename T>
void PQFlashAisaqIndex<T>::setup_thread_data(uint64_t nthreads) {
    PQFlashIndex<T>::setup_thread_data(nthreads);
    LOG(INFO) << "Setting up thread-specific contexts for AiSAQ nthreads: "
            << nthreads;
    for (_s64 thread = 0; thread < (_s64)nthreads; thread++) {
#pragma omp critical
        {
            AisaqThreadData<T> aisaq_data;
            aisaq_data.full_retset.reserve(4096);
            aisaq_thread_data.push(aisaq_data);
        }
    }
}

template<typename T>
void PQFlashAisaqIndex<T>::aisaq_async_generate_cache_list_from_sample_queries(
                           std::string sample_bin, uint64_t l_search,
                           uint64_t beamwidth,
                           uint64_t num_nodes_to_cache) {
    this->search_counter.store(0);
    this->node_visit_counter.clear();
    this->node_visit_counter.resize(this->num_points);
    for (uint32_t i = 0; i < this->node_visit_counter.size(); i++) {
        this->node_visit_counter[i].first = i;
        this->node_visit_counter[i].second =
                std::make_unique<std::atomic < _u32 >> (0);
    }
    this->count_visited_nodes.store(true);

    // sync allocate memory
    if (this->nhood_cache_buf == nullptr) {
        this->nhood_cache_buf =
                std::make_unique<unsigned[]>(num_nodes_to_cache * (this->max_degree + 1));
        memset(this->nhood_cache_buf.get(), 0,
               num_nodes_to_cache * (this->max_degree + 1) * sizeof(unsigned));
    }

    uint64_t coord_cache_buf_len = num_nodes_to_cache * this->aligned_dim;
    if (this->coord_cache_buf == nullptr) {
        diskann::alloc_aligned((void **) & this->coord_cache_buf,
                               coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
        std::fill_n(this->coord_cache_buf, coord_cache_buf_len, T());
    }

    async_pool.push([&, state_controller = this->state_controller, sample_bin,
                     l_search, beamwidth, num_nodes_to_cache]() {
        {
            std::unique_lock<std::mutex> guard(state_controller->status_mtx);
            if (state_controller->status.load() ==
                    ThreadSafeStateController::Status::KILLED ||
                state_controller->status.load() ==
                    ThreadSafeStateController::Status::STOPPING) {
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
            uint64_t sample_num, sample_dim, sample_aligned_dim;

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
            float tmp_result_dist = 0.0;

            uint64_t id = 0;
            knowhere::BitsetView bitset;
            struct diskann::aisaq_search_config aisaq_search_config;
            aisaq_search_config.pq_io_engine = aisaq_pq_io_engine_aio;
            aisaq_search_config.pq_cache_size = 0;
            aisaq_search_config.pq_read_page_cache_size = 0;
            aisaq_search_config.vector_beamwidth = 1;
            while (this->search_counter.load() < sample_num && id < sample_num) {
                {
                    std::unique_lock<std::mutex> guard(state_controller->status_mtx);
                    if (state_controller->status.load() == ThreadSafeStateController::Status::DOING) {
                        aisaq_cached_beam_search(samples + (id * sample_aligned_dim), 1, l_search,
                                         &tmp_result_id_64, &tmp_result_dist, beamwidth,
                                         false, nullptr, nullptr,
                                         bitset, -1.0f,
                                         &aisaq_search_config);
                        id++;
                    }
                }
            }

            if (state_controller->status.load() ==
                ThreadSafeStateController::Status::STOPPING) {
                    stream << "pq_flash_index is destoried, async thread should be exit."
                           << std::endl;
                throw diskann::ANNException(stream.str(), -1);
            }
            {
                std::unique_lock<std::shared_mutex> lock(this->node_visit_counter_mtx);
                this->count_visited_nodes.store(false);
                std::sort(this->node_visit_counter.begin(),
                          this->node_visit_counter.end(),
                          [](auto &left, auto &right){
                              return *(left.second) > *(right.second);
                          });
            }

            std::vector<uint32_t> node_list;
            node_list.clear();
            node_list.shrink_to_fit();
            node_list.reserve(num_nodes_to_cache);
            for (uint64_t i = 0; i < num_nodes_to_cache; i++) {
                node_list.push_back(this->node_visit_counter[i].first);
            }

            load_cache_list(node_list);
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
            this->node_visit_counter.clear();
            this->node_visit_counter.shrink_to_fit();
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

template <typename T>
std::vector<bool> PQFlashAisaqIndex<T>::read_nodes(
    const std::vector<uint32_t> &node_ids, std::vector<T *> &coord_buffers,
    std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers,
    std::vector<uint8_t *> *aisaq_buffers) {
    std::vector<AlignedRead> read_reqs;
    std::vector<bool> retval(node_ids.size(), true);
    uint32_t aisaq_data_size;

    if (aisaq_buffers != nullptr) {
        aisaq_data_size =
            _aisaq_inline_pq_vectors * this->n_chunks * sizeof(uint8_t);
        if (_aisaq_rearranged_vectors) {
            aisaq_data_size += sizeof(uint32_t);
        }
    }
    char *buf = nullptr;
    auto num_sectors =
        this->nnodes_per_sector > 0
            ? 1
            : DIV_ROUND_UP(this->max_node_len, defaults::SECTOR_LEN);
    alloc_aligned((void **)&buf,
                  node_ids.size() * num_sectors * defaults::SECTOR_LEN,
                  defaults::SECTOR_LEN);

    // create read requests
    for (size_t i = 0; i < node_ids.size(); ++i) {
        auto node_id = node_ids[i];

        AlignedRead read;
        read.len = num_sectors * defaults::SECTOR_LEN;
        read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
        read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
        read_reqs.push_back(read);
    }

    // borrow thread data and issue reads
    ThreadData<T> this_thread_data = this->thread_data.pop();
    while (this_thread_data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        this_thread_data = this->thread_data.pop();
    }

    auto ctx = this->reader->get_ctx();
    this->reader->read(read_reqs, ctx);
    // copy reads into buffers
    for (uint32_t i = 0; i < read_reqs.size(); i++) {
        char *node_buf = offset_to_node((char *)read_reqs[i].buf, node_ids[i]);
        if (coord_buffers[i] != nullptr) {
            T *node_coords = offset_to_node_coords(node_buf);
            memcpy(coord_buffers[i], node_coords, this->disk_bytes_per_point);
        }

        if (nbr_buffers[i].second != nullptr) {
            uint32_t *node_nhood = offset_to_node_nhood(node_buf);
            auto num_nbrs = *node_nhood;
            nbr_buffers[i].first = num_nbrs;
            memcpy(nbr_buffers[i].second, node_nhood + 1,
                   num_nbrs * sizeof(uint32_t));
        }
        if (aisaq_buffers != nullptr) {
            char *aisaq_data = aisaq_offset_to_node_aisaq_data(node_buf);
            memcpy((*aisaq_buffers)[i], aisaq_data, aisaq_data_size);
        }
    }
    aligned_free(buf);

    // return thread data
    this->thread_data.push(this_thread_data);
    this->thread_data.push_notify_all();
    this->reader->put_ctx(ctx);
    return retval;
}

static int aisaq_read_pq_vectors(class AisaqPQReader &aisaq_pq_vectors_reader,
                                 AisaqPQReaderContext &ctx, uint32_t max_ios,
                                 uint32_t *ids, uint32_t n_vectors,
                                 uint32_t vector_size, uint8_t *pq_vectors) {
    uint32_t offset = 0, count, tmp;
    while (offset < n_vectors) {
        count = std::min((uint32_t)n_vectors - offset, max_ios);
        if (aisaq_pq_vectors_reader.read_pq_vectors_submit(ctx, ids + offset,
                                                           count, tmp) != 0) {
            LOG_KNOWHERE_ERROR_ << "failed to read pq vectors";
            return -1;
        }
        uint32_t read_vec[count]; /* index array */
        uint8_t
            *pq_read_buffers[count]; /* pointers of where the vectors read to */
        if (aisaq_pq_vectors_reader.read_pq_vectors_wait_completion(
                ctx, read_vec, pq_read_buffers, count, count, tmp) != 0) {
            LOG_KNOWHERE_ERROR_ << "failed to read pq vectors";
            return -1;
        }
        for (uint32_t i = 0; i < count; i++) {
            memcpy(pq_vectors + ((offset + read_vec[i]) * vector_size),
                   pq_read_buffers[i], vector_size);
        }
        aisaq_pq_vectors_reader.read_pq_vectors_done(ctx);
        offset += count;
    }
    return 0;
}

  template<typename T>
  uint64_t PQFlashAisaqIndex<T>::aisaq_get_thread_data_size() {
    _u64 thread_data_size = PQFlashIndex<T>::get_thread_data_size();
    thread_data_size += 4096 * sizeof(Neighbor);
    thread_data_size += this->aligned_dim * sizeof(float);
    AisaqThreadData<T> aisaq_data = aisaq_thread_data.pop();
    if (aisaq_data.aisaq_pq_reader_ctx != nullptr) {
        thread_data_size += _aisaq_pq_vectors_reader->get_context_size(*aisaq_data.aisaq_pq_reader_ctx);
        aisaq_thread_data.push(aisaq_data);
    }
    return thread_data_size;
  }

  template<typename T>
uint64_t PQFlashAisaqIndex<T>::aisaq_cal_size() {
    _u64 index_mem_size = 0;
    index_mem_size += sizeof(*this);
    // thread data size:
    index_mem_size += (_u64) this->thread_data.size() * aisaq_get_thread_data_size();
    // get cache size:
    auto num_cached_nodes = this->coord_cache.size();
    index_mem_size +=
        ROUND_UP(num_cached_nodes * this->aligned_dim * sizeof(T), 8 * sizeof(T));
    index_mem_size += num_cached_nodes * (this->max_degree + 1) * sizeof(unsigned);
    index_mem_size += this->coord_cache.size() * sizeof(std::pair<_u32, T *>);
    index_mem_size +=
        this->nhood_cache.size() * sizeof(std::pair<_u32, std::pair<_u32, _u32 *>>);
    
    
    // AiSAQ node data cache
    uint32_t aisaq_data_len_u32 = DIV_ROUND_UP(
        _aisaq_inline_pq_vectors * this->n_chunks * sizeof(uint8_t),
        sizeof(uint32_t));
    if (_aisaq_rearranged_vectors) {
        aisaq_data_len_u32++;
    }
    index_mem_size += num_cached_nodes * aisaq_data_len_u32 * sizeof(uint32_t);
    // medoids pq vectors 
    index_mem_size +=
        this->num_medoids * this->n_chunks * sizeof(uint8_t);
    // AiSAQ multiple entry points pq vectors
    index_mem_size +=
        _aisaq_num_entry_points * this->n_chunks * sizeof(uint8_t);
    // AiSAQ static cache pq vectors
    index_mem_size +=
        _aisaq_pq_vectors_cache_count * this->n_chunks * sizeof(uint8_t);
    // AiSAQ rearrange map
    if (_aisaq_rearranged_vectors) {
        index_mem_size +=
                this->num_points * sizeof(uint32_t);
    }
    // get entry points:
    index_mem_size += ROUND_UP(this->num_medoids * this->aligned_dim * sizeof(float), 32);
    index_mem_size += this->num_medoids * this->aligned_dim * sizeof(uint32_t);
    // get pq data and pq_table:
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

template <typename T>
int PQFlashAisaqIndex<T>::aisaq_init(
    const diskann::aisaq_pq_io_engine pq_io_engine,
    const char *index_prefix) {

    if (this->_aisaq_rearranged_vectors) {
        if (aisaq_load_rearrange_data(index_prefix) != 0) {
            return -1;
        }
    }
    auto ctx_pool = AioContextPool::GetGlobalAioPool();
    std::string pq_file_path =
        _aisaq_rearranged_vectors
            ? (std::string(index_prefix) + "_pq_compressed_rearranged.bin")
            : (std::string(index_prefix) + "_pq_compressed.bin");
    _aisaq_pq_vectors_reader = AisaqPQReader::create_reader(
        pq_io_engine, pq_file_path.c_str(), _aisaq_rearranged_vectors);
    if (_aisaq_pq_vectors_reader == nullptr) {
        LOG_KNOWHERE_ERROR_ << "failed to create_reader";
        return -1;
    }
    /* create local read context */
    uint32_t max_ios = ctx_pool->max_events_per_ctx();
    AisaqPQReaderContext *ctx =
        _aisaq_pq_vectors_reader->create_context(max_ios);
    if (ctx == nullptr) {
        LOG_KNOWHERE_ERROR_ << "failed to initialize temp pq reader context";
        return -1;
    }
    auto io_ctx = ctx_pool->pop();
    _aisaq_pq_vectors_reader->set_io_ctx(*ctx, io_ctx);
    /* allocate memory for medoids pq vectors */
    _aisaq_medoids_pq_vectors_buff =
        new uint8_t[this->num_medoids * this->n_chunks * sizeof(uint8_t)];
    if (_aisaq_medoids_pq_vectors_buff == nullptr) {
        LOG_KNOWHERE_ERROR_
            << "failed to allocate memory for medoids pq vectors";
        _aisaq_pq_vectors_reader->destroy_context(*ctx);
        ctx_pool->push(io_ctx);
        return -1;
    }
    /* load medoids pq vectors from media */
    if (aisaq_read_pq_vectors(*_aisaq_pq_vectors_reader, *ctx, max_ios,
                              this->medoids.get(), this->num_medoids,
                              this->n_chunks * sizeof(uint8_t),
                              _aisaq_medoids_pq_vectors_buff) != 0) {
        LOG_KNOWHERE_ERROR_ << "failed to read medoids pq vectors";
        _aisaq_pq_vectors_reader->destroy_context(*ctx);
        ctx_pool->push(io_ctx);
        return -1;
    }
    /* handle multiple entry points */
    std::string entry_points_path =
        std::string(index_prefix) + "_disk.index_entry_points.bin";
    if (file_exists(entry_points_path)) {
        /* load entry points pq vectors */
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(entry_points_path, _aisaq_entry_points,
                                    _aisaq_num_entry_points, tmp_dim);
        assert(tmp_dim == 1);
        LOG_KNOWHERE_DEBUG_ << "aisaq search using " << _aisaq_num_entry_points
                           << " entry points";
        /* allocate memory for entry points pq vectors */
        _aisaq_entry_points_pq_vectors_buff =
            new uint8_t[_aisaq_num_entry_points * this->n_chunks *
                        sizeof(uint8_t)];
        if (_aisaq_entry_points_pq_vectors_buff == nullptr) {
            LOG_KNOWHERE_ERROR_
                << "failed to allocate memory for entry points pq vectors";
            _aisaq_pq_vectors_reader->destroy_context(*ctx);
            ctx_pool->push(io_ctx);
            return -1;
        }
        if (aisaq_read_pq_vectors(*_aisaq_pq_vectors_reader, *ctx, max_ios,
                                  _aisaq_entry_points.get(),
                                  _aisaq_num_entry_points,
                                  this->n_chunks * sizeof(uint8_t),
                                  _aisaq_entry_points_pq_vectors_buff) != 0) {
            LOG_KNOWHERE_ERROR_ << "failed to read entry points pq vectors";
            _aisaq_pq_vectors_reader->destroy_context(*ctx);
            ctx_pool->push(io_ctx);
            return -1;
        }
    }
    ctx_pool->push(io_ctx);
    _aisaq_pq_vectors_reader->destroy_context(*ctx);

    if (_aisaq_inline_pq_vectors <= this->max_degree) {
        /* some or none inline, minimal number of 16 ios is needed */
        max_ios = (this->max_degree - _aisaq_inline_pq_vectors) *
                  diskann::defaults::MAX_AISAQ_VECTORS_BEAMWIDTH;
        if(max_ios > ctx_pool->max_events_per_ctx()){
        	max_ios = ctx_pool->max_events_per_ctx();
        }
    } else {
        max_ios = 0;
    }

    ConcurrentQueue<AisaqThreadData <T>> thread_data_list;
    while (!aisaq_thread_data.empty()) {
        AisaqThreadData<T> data = aisaq_thread_data.pop();
        data.aisaq_max_read_nodes = defaults::MAX_N_SECTOR_READS;
        for (uint32_t i = 0; i < data.aisaq_max_read_nodes; i++) {
            data.aisaq_scratch_mem_offset.push_back(this->read_len_for_node * i);
        }
        if (max_ios > 0) {
            data.aisaq_pq_reader_ctx = _aisaq_pq_vectors_reader->create_context(
                max_ios);
            if (data.aisaq_pq_reader_ctx == nullptr) {
                LOG_KNOWHERE_ERROR_ << "failed to initialize pq reader context";
                aisaq_thread_data.push(data);
                while (!thread_data_list.empty()) {
                    data = thread_data_list.pop();
                    _aisaq_pq_vectors_reader->destroy_context(*data.aisaq_pq_reader_ctx);
                    data.aisaq_pq_reader_ctx = nullptr;
                    aisaq_thread_data.push(data);
                }
                return -1;
            }
        }
        else
            data.aisaq_pq_reader_ctx = nullptr;
        thread_data_list.push(data);
    }
    while (!thread_data_list.empty()) {
        AisaqThreadData data = thread_data_list.pop();
        aisaq_thread_data.push(data);
    }
    return 0;
}

// obtains region of sector containing node
template <typename T>
void PQFlashAisaqIndex<T>::aisaq_get_vector_by_ids(const int64_t *ids,
                                             const int64_t n, T *output_data) {
    auto sectors_to_visit =
        this->get_sectors_layout_and_write_data_from_cache(ids, n, output_data);
    if (0 == sectors_to_visit.size()) {
        return;
    }

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
    }
    uint32_t aio_max_events = 8;

    const size_t batch_size =
        std::min((size_t)aio_max_events,
                 std::min(MAX_N_SECTOR_READS / 2UL, sectors_to_visit.size()));
    const size_t half_buf_idx =
        MAX_N_SECTOR_READS / 2 * this->read_len_for_node;
    char *sector_scratch = data.scratch.sector_scratch;
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(batch_size);

    std::vector<uint64_t> sector_offsets;
    sector_offsets.reserve(sectors_to_visit.size());
    for (const auto &it : sectors_to_visit) {
        sector_offsets.emplace_back(it.first);
    }

    auto ctx = this->reader->get_ctx();
    const auto sector_num = sector_offsets.size();
    const uint64_t num_blocks = DIV_ROUND_UP(sector_num, batch_size);
    std::vector<AlignedRead> last_reqs;
    bool rotate = false;

    for (uint64_t i = 0; i < num_blocks; ++i) {
        uint64_t start_idx = i * batch_size;
        uint64_t idx_len = std::min(batch_size, sector_num - start_idx);
        last_reqs = frontier_read_reqs;
        frontier_read_reqs.clear();
        for (uint64_t j = 0; j < idx_len; ++j) {
            char *sector_buf = sector_scratch + rotate * half_buf_idx +
                               j * this->read_len_for_node;
            frontier_read_reqs.emplace_back(sector_offsets[start_idx + j],
                                            this->read_len_for_node,
                                            sector_buf);
        }
        rotate ^= 0x1;
        this->reader->submit_req(ctx, frontier_read_reqs);
        for (const auto &req : last_reqs) {
            auto offset = req.offset;
            char *sector_buf = static_cast<char *>(req.buf);
            for (auto idx : sectors_to_visit.at(offset)) {
                // char *node_buf = offset_to_node(sector_buf, ids[idx]);
                char *node_buf = this->get_offset_to_node(sector_buf, ids[idx]);
                this->copy_vec_base_data(output_data, idx, node_buf);
            }
        }
        this->reader->get_submitted_req(ctx, frontier_read_reqs.size());
    }

    // if any remaining
    for (const auto &req : frontier_read_reqs) {
        auto offset = req.offset;
        char *sector_buf = static_cast<char *>(req.buf);
        for (auto idx : sectors_to_visit.at(offset)) {
            // char *node_buf = offset_to_node(sector_buf, ids[idx]);
            char *node_buf = this->get_offset_to_node(sector_buf, ids[idx]);
            this->copy_vec_base_data(output_data, idx, node_buf);
        }
    }

    this->reader->put_ctx(ctx);
    this->thread_data.push(data);
    this->thread_data.push_notify_all();
}

template <typename T> bool PQFlashAisaqIndex<T>::get_rearranged_index() {
    return _aisaq_rearranged_vectors;
}

template <typename T> std::uint64_t PQFlashAisaqIndex<T>::get_n_chunks() {
    return this->n_chunks;
}

/* load pq cache according to specified policy
   0: use bfs (breadth-first search) logic to
      enumerate the nodes until the desired number of vectors is reached.
      inline vectors are skipped.
   1: direct-mapping, it will be forced when the number of cached vectors
      is higher than a predefined thresholds */
template <typename T>
void PQFlashAisaqIndex<T>::aisaq_load_pq_cache(
    const std::string pq_compressed_vectors_path, uint64_t pq_cache_size_bytes,
    uint32_t policy) {
    if (_aisaq_inline_pq_vectors >= this->max_degree) {
        LOG_KNOWHERE_DEBUG_
            << "all pq vectors are stored inline, pq cache will not be used";
        return;
    }
    uint64_t pq_cache_max_bytes_limit =
        (uint64_t)(double)(AISAQ_SEARCH_PQ_CACHE_MAX_DRAM_GB * (1 << 30));
    if (pq_cache_size_bytes > pq_cache_max_bytes_limit) {
        LOG_KNOWHERE_DEBUG_ << "pq cache DRAM size will be limited to "
                           << AISAQ_SEARCH_PQ_CACHE_MAX_DRAM_GB << "GB";
        pq_cache_size_bytes = pq_cache_max_bytes_limit;
    }
    uint32_t pq_vec_size = this->n_chunks * sizeof(uint8_t);
    uint64_t pq_cache_max_vec = pq_cache_size_bytes / pq_vec_size;
    uint64_t pq_cache_max_vec_limit =
        (this->num_points * AISAQ_SEARCH_PQ_CACHE_MAX_VECTORS_PCNT) / 100;
    if (pq_cache_max_vec > pq_cache_max_vec_limit) {
        LOG_KNOWHERE_DEBUG_ << "pq cache will be limited to "
                           << AISAQ_SEARCH_PQ_CACHE_MAX_VECTORS_PCNT
                           << "% of total vectors (" << pq_cache_max_vec_limit
                           << ")";
        pq_cache_max_vec = pq_cache_max_vec_limit;
    }

    std::ifstream pq_compressed_vectors_reader;
    size_t pq_compressed_vectors_file_size =
        get_file_size(pq_compressed_vectors_path);
    pq_compressed_vectors_reader.exceptions(std::ofstream::failbit |
                                            std::ofstream::badbit);
    try {
        uint32_t pq_compressed_vectors_npts, pq_compressed_nbytes;
        pq_compressed_vectors_reader.open(pq_compressed_vectors_path,
                                          std::ios::binary);
        pq_compressed_vectors_reader.read((char *)&pq_compressed_vectors_npts,
                                          sizeof(uint32_t));
        pq_compressed_vectors_reader.read((char *)&pq_compressed_nbytes,
                                          sizeof(uint32_t));
        if (pq_compressed_vectors_npts != this->num_points) {
            throw ANNException(
                "Mismatch in num_points between pq compressed vectors file and "
                "base file",
                -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        if (pq_compressed_nbytes != pq_vec_size) {
            throw ANNException("Mismatch in pq vector size between pq "
                               "compressed vectors file and "
                               "base file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        if (pq_compressed_vectors_file_size !=
            8 + (size_t)pq_compressed_nbytes *
                    (size_t)pq_compressed_vectors_npts) {
            throw ANNException(
                "Discrepancy in pq compressed vectors file size ", -1,
                __FUNCSIG__, __FILE__, __LINE__);
        }
    } catch (std::system_error &e) {
        throw FileException(pq_compressed_vectors_path, e, __FUNCSIG__,
                            __FILE__, __LINE__);
    }

    /* allocate memory */
    _aisaq_pq_vectors_cache_buf = new uint8_t[pq_cache_max_vec * pq_vec_size];
    if (_aisaq_pq_vectors_cache_buf == nullptr) {
        return;
    }
    LOG_KNOWHERE_DEBUG_ << "allocated " << std::setprecision(4)
                       << (float)(pq_cache_max_vec * pq_vec_size) / (1 << 20)
                       << " MiB for pq cache";
    _aisaq_pq_vectors_cache_count = pq_cache_max_vec;
    _aisaq_pq_vectors_cache_direct =
        policy == aisaq_pq_cache_policy_direct ||
        (policy == aisaq_pq_cache_policy_auto &&
         (_aisaq_rearranged_vectors ||
          (_aisaq_pq_vectors_cache_count == this->num_points ||
           _aisaq_pq_vectors_cache_count >
               AISAQ_SEARCH_PQ_CACHE_DIRECT_THRESHOLD_N ||
           _aisaq_pq_vectors_cache_count >
               ((this->num_points *
                 AISAQ_SEARCH_PQ_CACHE_DIRECT_THRESHOLD_PCNT) /
                100))));
    if (_aisaq_pq_vectors_cache_direct) {
        LOG_KNOWHERE_DEBUG_ << "loading pq cache with " << pq_cache_max_vec
                           << " compressed vectors using direct policy";
        LOG_KNOWHERE_DEBUG_ << "populating pq cache...";
        /* skip header */
        size_t tocopy = pq_cache_max_vec * pq_vec_size;
        size_t offset = 0;
        pq_compressed_vectors_reader.seekg((sizeof(uint32_t) * 2),
                                           pq_compressed_vectors_reader.beg);
        while (tocopy > 0) {
            size_t count = std::min(tocopy, (size_t)(1 << 20));
            pq_compressed_vectors_reader.read(
                (char *)_aisaq_pq_vectors_cache_buf + offset, count);
            tocopy -= count;
            offset += count;
        }
        pq_compressed_vectors_reader.close();
        LOG_KNOWHERE_DEBUG_ << "...done";
        return;
    }
    if (policy != aisaq_pq_cache_policy_bfs &&
        policy != aisaq_pq_cache_policy_auto) {
        LOG_KNOWHERE_ERROR_
            << "unknown pq cache load policy, bfs policy will be used instead.";
    }
    /* bfs */
    bool shuffle = false;
    std::random_device rng;
    std::mt19937 urng(rng());

    LOG_KNOWHERE_DEBUG_ << "loading pq cache with " << pq_cache_max_vec
                       << " compressed vectors using bfs policy";
    LOG_KNOWHERE_DEBUG_ << "preparing pq cache map...";
    tsl::robin_set<uint32_t> node_set;
    std::unique_ptr<tsl::robin_set<uint32_t>> cur_level, prev_level;
    cur_level = std::make_unique<tsl::robin_set<uint32_t>>();
    prev_level = std::make_unique<tsl::robin_set<uint32_t>>();

    uint8_t *pq_cache_buf_it = _aisaq_pq_vectors_cache_buf;
    uint32_t pq_cache_vec_count = 0;
    uint64_t prev_pq_cache_vec_count = 0;
    uint64_t prev_nodes_count = 0;

    diskann::cout << "medoids:..." << std::flush;
    for (uint64_t miter = 0;
         miter < this->num_medoids && pq_cache_vec_count < pq_cache_max_vec;
         miter++) {
        cur_level->insert(this->medoids[miter]);
        if (_aisaq_pq_vectors_cache_map.find(this->medoids[miter]) ==
            _aisaq_pq_vectors_cache_map.end()) {
            _aisaq_pq_vectors_cache_map[this->medoids[miter]] = pq_cache_buf_it;
            pq_cache_buf_it += pq_vec_size;
            pq_cache_vec_count++;
        }
    }

    uint64_t block_size = 1024;
    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> coord_buffers(block_size, nullptr);
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
    /* allocate nnbrs memory once */
    for (uint32_t i = 0; i < block_size; i++) {
        nbr_buffers.emplace_back(0, new uint32_t[this->max_degree]);
    }
    LOG_KNOWHERE_DEBUG_ << "... +"
                       << pq_cache_vec_count - prev_pq_cache_vec_count
                       << " vectors";
    prev_pq_cache_vec_count = pq_cache_vec_count;
    uint64_t lvl = 1;
    while (pq_cache_vec_count < pq_cache_max_vec && cur_level->size() != 0) {
        std::swap(prev_level, cur_level);
        cur_level->clear();
        std::vector<uint32_t> nodes_to_expand;
        for (const uint32_t &id : *prev_level) {
            if (node_set.find(id) != node_set.end()) {
                continue;
            }
            node_set.insert(id);
            nodes_to_expand.push_back(id);
        }

        if (shuffle) {
            std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
        } else {
            std::sort(nodes_to_expand.begin(), nodes_to_expand.end());
        }

        diskann::cout << "level: " << lvl << "..." << std::flush;
        bool finish_flag = false;
        uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), block_size);
        uint32_t progress_step = std::max(1lu, nblocks / 20);
        for (size_t block = 0; block < nblocks && !finish_flag; block++) {
            if ((block % progress_step) == 0) {
                diskann::cout << "." << std::flush;
            }
            size_t start = block * block_size;
            size_t end =
                (std::min)((block + 1) * block_size, nodes_to_expand.size());
            for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
                nodes_to_read.push_back(nodes_to_expand[cur_pt]);
            }

            /* issue read requests */
            auto read_status =
                read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

            for (uint32_t i = 0; i < read_status.size(); i++) {
                if (read_status[i] == false) {
                    continue;
                }
                uint32_t nnbrs = nbr_buffers[i].first;
                uint32_t *nbrs = nbr_buffers[i].second;
                if (pq_cache_vec_count < pq_cache_max_vec) {
                    for (uint32_t j = _aisaq_inline_pq_vectors; j < nnbrs;
                         j++) {
                        if (_aisaq_pq_vectors_cache_map.find(nbrs[j]) ==
                            _aisaq_pq_vectors_cache_map.end()) {
                            _aisaq_pq_vectors_cache_map[nbrs[j]] =
                                pq_cache_buf_it;
                            pq_cache_buf_it += pq_vec_size;
                            pq_cache_vec_count++;
                            if (pq_cache_vec_count == pq_cache_max_vec) {
                                cur_level->clear();
                                break;
                            }
                        }
                    }
                }
                if (pq_cache_vec_count < pq_cache_max_vec) {
                    /* next level */
                    for (uint32_t j = 0; j < nnbrs; j++) {
                        if (node_set.find(nbrs[j]) == node_set.end()) {
                            cur_level->insert(nbrs[j]);
                        }
                    }
                }
            }
            nodes_to_read.clear();
        }
        LOG_KNOWHERE_DEBUG_ << "... +" << node_set.size() - prev_nodes_count
                           << " nodes +"
                           << pq_cache_vec_count - prev_pq_cache_vec_count
                           << " vectors"
                           << " --> " << node_set.size() << " nodes "
                           << pq_cache_vec_count << " vectors";
        prev_pq_cache_vec_count = pq_cache_vec_count;
        prev_nodes_count = node_set.size();
        lvl++;
    }
    /* free nnbrs memory */
    for (uint32_t i = 0; i < block_size; i++) {
        delete[] nbr_buffers[i].second;
    }
    nbr_buffers.clear();

    diskann::cout << "populating pq cache..." << std::flush;
    for (auto iter = _aisaq_pq_vectors_cache_map.begin();
         iter != _aisaq_pq_vectors_cache_map.end(); iter++) {
        pq_compressed_vectors_reader.seekg(
            (sizeof(uint32_t) * 2) + ((uint64_t)iter->first * pq_vec_size),
            pq_compressed_vectors_reader.beg);
        pq_compressed_vectors_reader.read((char *)iter->second, pq_vec_size);
    }
    pq_compressed_vectors_reader.close();
    LOG_KNOWHERE_DEBUG_ << "...done";
}

template <typename T>
void PQFlashAisaqIndex<T>::use_medoids_data_as_centroids() {
    if (this->centroid_data != nullptr)
        aligned_free(this->centroid_data);
    alloc_aligned(((void **)&this->centroid_data),
                  this->num_medoids * this->aligned_dim * sizeof(float), 32);
    std::memset(this->centroid_data, 0,
                this->num_medoids * this->aligned_dim * sizeof(float));

    LOG_KNOWHERE_DEBUG_ << "Loading centroid data from medoids vector data of "
                       << this->num_medoids << " medoid(s)";

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < this->num_medoids; cur_m++) {
        nodes_to_read.push_back(this->medoids[cur_m]);
        medoid_bufs.push_back(new T[this->data_dim]);
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < this->num_medoids; cur_m++) {
        if (read_status[cur_m] == true) {
            if (!this->use_disk_index_pq) {
                for (uint32_t i = 0; i < this->data_dim; i++)
                    this->centroid_data[cur_m * this->aligned_dim + i] =
                        medoid_bufs[cur_m][i];
            } else {
                this->disk_pq_table.inflate_vector(
                    (uint8_t *)medoid_bufs[cur_m],
                    (this->centroid_data + cur_m * this->aligned_dim));
            }
        } else {
            throw ANNException("Unable to read a medoid", -1, __FUNCSIG__,
                               __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];
    }
}

template <typename T>
int PQFlashAisaqIndex<T>::aisaq_load(uint32_t num_threads,
                                     const char *index_prefix) {
	this->index_prefix = std::string(index_prefix);
    std::string pq_table_bin = this->index_prefix + "_pq_pivots.bin";
    std::string pq_compressed_vectors =
    		this->index_prefix + "_pq_compressed.bin";
    std::string disk_index_file = this->index_prefix + "_disk.index";
    return aisaq_load_from_separate_paths(num_threads, disk_index_file.c_str(),
                                          pq_table_bin.c_str(),
                                          pq_compressed_vectors.c_str());
}

template <typename T>
int PQFlashAisaqIndex<T>::aisaq_load_from_separate_paths(
    uint32_t num_threads, const char *index_filepath,
    const char *pivots_filepath, const char *compressed_filepath) {
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string medoids_file = std::string(index_filepath) + "_medoids.bin";
    std::string centroids_file = std::string(index_filepath) + "_centroids.bin";

    std::string labels_file = std::string(index_filepath) + "_labels.txt";
    std::string labels_to_medoids =
        std::string(index_filepath) + "_labels_to_medoids.txt";
    std::string labels_map_file =
        std::string(index_filepath) + "_labels_map.txt";

    size_t pq_file_dim, pq_file_num_centroids;

    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim);

    this->disk_index_file = index_filepath;

    if (pq_file_num_centroids != 256) {
        LOG_KNOWHERE_ERROR_ << "Number of PQ centroids is not 256. Exiting.";
        return -1;
    }

    this->data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->disk_bytes_per_point = this->data_dim * sizeof(T);
    this->aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
    std::ifstream freader;
    freader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
        freader.open(pq_compressed_vectors, std::ios::binary);
        uint32_t val;
        freader.read((char *)&val, sizeof(uint32_t));
        npts_u64 = (size_t)val;
        freader.read((char *)&val, sizeof(uint32_t));
        nchunks_u64 = (size_t)val;
        freader.close();
    } catch (std::system_error &e) {
        throw FileException(pq_compressed_vectors, e, __FUNCSIG__, __FILE__,
                            __LINE__);
    }

    this->num_points = npts_u64;
    this->n_chunks = nchunks_u64;

    this->pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);

    LOG_KNOWHERE_DEBUG_
        << "Loaded PQ centroids and in-memory compressed vectors. #points: "
        << this->num_points << " #dim: " << this->data_dim
        << " #aligned_dim: " << this->aligned_dim
        << " #chunks: " << this->n_chunks;

    if (this->n_chunks > diskann::defaults::MAX_PQ_CHUNKS) {
        std::stringstream stream;
        stream
            << "Error: Loading index. Ensure that max PQ bytes for in-memory "
               "PQ data does not exceed "
            << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    }

    std::string disk_pq_pivots_path =
        std::string(index_filepath) + "_pq_pivots.bin";
    if (file_exists(disk_pq_pivots_path)) {
        this->use_disk_index_pq = true;
        // giving 0 chunks to make the _pq_table infer from the
        // chunk_offsets file the correct value
        this->disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(),
                                                 0);
        this->disk_pq_n_chunks = this->disk_pq_table.get_num_chunks();
        this->disk_bytes_per_point =
            this->disk_pq_n_chunks *
            sizeof(uint8_t); // revising disk_bytes_per_point since DISK PQ is
                             // used.
        LOG_KNOWHERE_DEBUG_ << "Disk index uses PQ data compressed down to "
                           << this->disk_pq_n_chunks << " bytes per point.";
    }

    // read index metadata
    std::ifstream index_metadata(this->disk_index_file, std::ios::binary);

    uint32_t nr,
        nc; // metadata itself is stored as bin format (nr is number of
    // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    if (disk_nnodes != this->num_points) {
        std::stringstream stream;
        stream << "Error: Mismatch in #points for compressed data file and disk  "
                  "index file: "
               << disk_nnodes << " vs " << this->num_points << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    }

    size_t medoid_id_on_file;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, this->max_node_len);
    READ_U64(index_metadata, this->nnodes_per_sector);

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->num_frozen_points);
    uint64_t file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->num_frozen_points == 1)
        this->frozen_location = file_frozen_id;
    if (this->num_frozen_points == 1) {
        LOG_KNOWHERE_DEBUG_ << " Detected frozen point in index at location "
                           << this->frozen_location
                           << ". Will not output it at search time.";
    }

    READ_U64(index_metadata, this->_reorder_data_exists);
    if (this->_reorder_data_exists) {
        if (this->use_disk_index_pq == false) {
            throw ANNException("Reordering is designed for used with disk PQ "
                               "compression option",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        READ_U64(index_metadata, this->reorder_data_start_sector);
        READ_U64(index_metadata, this->ndims_reorder_vecs);
        READ_U64(index_metadata, this->nvecs_per_sector);
    }

    uint64_t __md_file_size, __md_max_degree, __md_rearranged_index;
    READ_U64(index_metadata, __md_file_size);        /* file_size */
    READ_U64(index_metadata, __md_max_degree);       /* max_degree */
    READ_U64(index_metadata, __md_rearranged_index); /* rearranged_index */
    if (get_file_size(this->disk_index_file) != __md_file_size) {
        std::stringstream stream;
        stream << "Error: Loading index. Incorrect file size, file '"
               << this->disk_index_file
               << "' may be corrupted. expected size is: " << __md_file_size
               << " Bytes" << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    }
    uint64_t __max_max_degree;
    _aisaq_rearranged_vectors = __md_rearranged_index != 0;
    __max_max_degree = ((this->max_node_len - this->disk_bytes_per_point -
                         (_aisaq_rearranged_vectors ? sizeof(uint32_t) : 0)) /
                        sizeof(uint32_t)) -
                       1;
    if (__md_max_degree != 0) {
        /* aisaq */
        if (__md_max_degree > __max_max_degree) {
            std::stringstream stream;
            stream << "Error: Loading index. Incorrect max graph degree (R) in "
                      "metadata "
                   << __md_max_degree << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }
        this->max_degree = __md_max_degree;
    } else {
        this->max_degree = __max_max_degree;
    }

    if (this->max_degree > defaults::MAX_GRAPH_DEGREE) {
        std::stringstream stream;
        stream << "Error: Loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << defaults::MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    }
    _aisaq_inline_pq_vectors =
        (this->max_node_len -
         (this->disk_bytes_per_point +
          ((this->max_degree + 1) * sizeof(uint32_t)) +
          (_aisaq_rearranged_vectors ? sizeof(uint32_t) : 0))) /
        (this->n_chunks * sizeof(uint8_t));
    LOG_KNOWHERE_DEBUG_ << "Disk-Index File Meta-data: "
                        << "# nodes per sector: " << this->nnodes_per_sector
                        << ", max node len (bytes): " << this->max_node_len
                        << ", max node degree: " << this->max_degree
                        << ", inline vectors: " << _aisaq_inline_pq_vectors
                        << ", rearranged vectors: "
                        << _aisaq_rearranged_vectors;

    index_metadata.close();
    if (this->max_node_len > defaults::SECTOR_LEN) {
        this->long_node = true;
        this->nsectors_per_node =
            ROUND_UP(this->max_node_len, diskann::defaults::SECTOR_LEN) /
            diskann::defaults::SECTOR_LEN;
        this->read_len_for_node =
            diskann::defaults::SECTOR_LEN * this->nsectors_per_node;
    }
    // open AlignedFileReader handle to index_file
    std::string index_fname(this->disk_index_file);
    this->reader->open(index_fname);
    setup_thread_data(num_threads);
    this->max_nthreads = num_threads;

    if (file_exists(medoids_file)) {
        size_t tmp_dim;
        diskann::load_bin<uint32_t>(medoids_file, this->medoids,
                                    this->num_medoids, tmp_dim);

        if (tmp_dim != 1) {
            std::stringstream stream;
            stream << "Error: Loading medoids file. Expected bin format of m "
                      "times 1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }
        if (!file_exists(centroids_file)) {
            LOG_KNOWHERE_DEBUG_ << "Centroid data file not found. Using "
                                  "corresponding vectors for the medoids ";
            use_medoids_data_as_centroids();
        } else {
            size_t num_centroids, aligned_tmp_dim;
            diskann::load_aligned_bin<float>(centroids_file,
                                             this->centroid_data, num_centroids,
                                             tmp_dim, aligned_tmp_dim);
            if (aligned_tmp_dim != this->aligned_dim ||
                num_centroids != this->num_medoids) {
                std::stringstream stream;
                stream
                    << "Error: Loading centroids data file. Expected bin format of "
                       "m times data_dim vector of float, where m is number of "
                       "medoids in medoids file."
                    << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
            }
        }
    } else {
        this->num_medoids = 1;
        this->medoids = std::make_unique<uint32_t[]>(1);
        this->medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(index_filepath) + "_max_base_norm.bin";

    if (file_exists(norm_file) &&
        this->metric == diskann::Metric::INNER_PRODUCT) {
        uint64_t dumr, dumc;
        std::unique_ptr<float[]> norm_val = nullptr;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
        this->max_base_norm = norm_val[0];
        LOG_KNOWHERE_DEBUG_ << "Setting re-scaling factor of base vectors to "
                           << this->max_base_norm;
    }
    if (file_exists(norm_file) && this->metric == diskann::Metric::COSINE) {
        _u64 dumr, dumc;
        diskann::load_bin<float>(norm_file, this->base_norms, dumr, dumc);
        LOG_KNOWHERE_DEBUG_ << "Setting base vector norms";
    }
    LOG_KNOWHERE_DEBUG_ << "done..";
    return 0;
}

template <typename T>
uint8_t *PQFlashAisaqIndex<T>::aisaq_pq_cache_lookup(uint32_t id) {
    if (_aisaq_pq_vectors_cache_direct) {
        if (id < _aisaq_pq_vectors_cache_count) {
            /* vector is in cache */
            return _aisaq_pq_vectors_cache_buf +
                   (id * this->n_chunks * sizeof(uint8_t));
        }
    } else {
        auto _pq_cache_iter = _aisaq_pq_vectors_cache_map.find(id);
        if (_pq_cache_iter != _aisaq_pq_vectors_cache_map.end()) {
            /* vector is in cache */
            return _pq_cache_iter->second;
        }
    }
    return nullptr;
}

template <typename T>
void PQFlashAisaqIndex<T>::get_entry_point_medoid(uint32_t &best_medoid, float &best_dist,
                                                  float *pq_dists, float *dist_scratch,
                                                  float *query_float)
{
    best_medoid = 0;
    best_dist = (std::numeric_limits<float>::max)();
    if (_aisaq_num_entry_points > 0) {
        /* in this case, best_medoid is determined in pq space */
        uint32_t offset = 0;
        while (offset < _aisaq_num_entry_points) {
            /* dist_scratch size is limited to MAX_GRAPH_DEGREE */
            uint32_t count =
                std::min((uint32_t)defaults::MAX_GRAPH_DEGREE,
                         (uint32_t)_aisaq_num_entry_points - offset);
            diskann::pq_dist_lookup(
                _aisaq_entry_points_pq_vectors_buff +
                    (offset * this->n_chunks * sizeof(uint8_t)),
                count, this->n_chunks, pq_dists, dist_scratch);
            for (uint32_t i = 0; i < count; i++) {
                if (dist_scratch[i] < best_dist) {
                    best_dist = dist_scratch[i];
                    best_medoid = _aisaq_entry_points[offset + i];
                }
            }
            offset += count;
        }
    }else {
        uint32_t best_medoid_index;
        for (uint64_t cur_m = 0; cur_m < this->num_medoids; cur_m++) {
            float cur_expanded_dist = this->dist_cmp_float_wrap(
                query_float, this->centroid_data + this->aligned_dim * cur_m,
                (size_t)this->aligned_dim, this->medoids[cur_m]);
            if (cur_expanded_dist < best_dist) {
                best_medoid_index = cur_m;
                best_dist = cur_expanded_dist;
            }
        }
        /* now calc best_medoid distance in pq space */
        best_medoid = this->medoids[best_medoid_index];
        diskann::pq_dist_lookup(
            _aisaq_medoids_pq_vectors_buff +
                (best_medoid_index * this->n_chunks * sizeof(uint8_t)),
            1, this->n_chunks, pq_dists, &best_dist);
    }
   
}

template <typename T>
void PQFlashAisaqIndex<T>::updata_io_stats(QueryStats &stats, size_t count, uint64_t size_in_sectors){
    switch (size_in_sectors) {
    case 1:
        stats.n_4k += count;
        break;
    case 2:
        stats.n_8k += count;
        break;
    default:
        stats.n_12k += count;
        break;
    }
    stats.n_ios += count;    
}

template <typename T>
void PQFlashAisaqIndex<T>::aisaq_cached_beam_search(
    const T *query1, const uint64_t k_search, const uint64_t l_search,
    int64_t *indices, float *distances, const uint64_t beam_width,
    const bool use_reorder_data, QueryStats *stats,
    const knowhere::feder::diskann::FederResultUniq &feder,
    const knowhere::BitsetView bitset, const float filter_ratio_in,
    const struct diskann::aisaq_search_config *aisaq_search_config){
    uint64_t num_sectors_per_node =
        this->nnodes_per_sector > 0
            ? 1
            : DIV_ROUND_UP(this->max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sectors_per_node * defaults::MAX_N_SECTOR_READS)
        throw ANNException(
            "Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1,
            __FUNCSIG__, __FILE__, __LINE__);

    ThreadData<T> data = this->thread_data.pop();
    while (data.scratch.sector_scratch == nullptr) {
        this->thread_data.wait_for_push_notify();
        data = this->thread_data.pop();
    }
    auto query_norm_opt = this->init_thread_data(data, query1);
    if (!query_norm_opt.has_value()) {
        // return an empty answer when calcu a zero point
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
        return;
    }
    float query_norm = query_norm_opt.value();
    AisaqThreadData aisaq_data = aisaq_thread_data.pop();
    auto ctx = this->reader->get_ctx();
    if(aisaq_data.aisaq_pq_reader_ctx) {
    	_aisaq_pq_vectors_reader->set_io_ctx(*aisaq_data.aisaq_pq_reader_ctx, ctx);
    }
    auto release_data = [this, data, aisaq_data, ctx]() mutable {
        this->thread_data.push(data);
        this->thread_data.push_notify_all();
        this->aisaq_thread_data.push(aisaq_data);
        this->aisaq_thread_data.push_notify_all();
        if(aisaq_data.aisaq_pq_reader_ctx){
        	_aisaq_pq_vectors_reader->set_io_ctx(*aisaq_data.aisaq_pq_reader_ctx, nullptr);
        }
        this->reader->put_ctx(ctx);
    };
    size_t bv_cnt = 0;
    uint64_t local_l_search = l_search;
    float alpha = 0.15;
    if (!bitset.empty()) {
        const auto filter_threshold =
            filter_ratio_in < 0 ? kFilterThreshold : filter_ratio_in;
        bv_cnt = bitset.count();
#ifdef NOT_COMPILE_FOR_SWIG
        double ratio = ((double)bv_cnt) / bitset.size();
        knowhere::knowhere_diskann_bitset_ratio.Observe(ratio);
#endif
        if (bitset.size() == bv_cnt) {
            for (uint64_t i = 0; i < k_search; i++) {
                indices[i] = -1;
                if (distances != nullptr) {
                    distances[i] = -1;
                }
            }
            release_data();
            return;
        }
        if (bv_cnt >= bitset.size() * filter_threshold ||
        		(k_search > 0.5 * (this->num_points - bv_cnt))) {
            std::string pq_compressed_vectors =
            		this->index_prefix + "_pq_compressed.bin";
            try{
                MemoryMapper mapper(pq_compressed_vectors);
                size_t pq_size = 8+this->num_points*this->n_chunks;
                madvise(mapper.getBuf(), pq_size, MADV_SEQUENTIAL);
                _u8* pq_data = (_u8*)mapper.getBuf()+8;
                AisaqPQDataGetter pq_getter(pq_data, this->_aisaq_rearranged_vectors, this->_aisaq_rearranged_vectors_map.get(), pq_size);

                PQFlashIndex<T>::brute_force_beam_search(data, query_norm, k_search, indices, distances,
						beam_width, ctx, stats, nullptr, bitset, &pq_getter);
            }
            catch(...){
                release_data();
                throw ANNException("Failed pq mmap",
                                   -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        	release_data();
            return;
        }
    }
    if (local_l_search < l_search) {
        local_l_search = l_search;
    }
    auto query_scratch = &(data.scratch);
    // reset query scratch
    query_scratch->reset();
    aisaq_data.retset.clear();
    aisaq_data.full_retset.clear();
    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    T *aligned_query_T = query_scratch->aligned_query_T;
    float *query_float = query_scratch->aligned_query_float;

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;
#if defined(__ARM_NEON) && defined(__aarch64__)
    __builtin_prefetch(data_buf, 1, 3);
#else
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);
#endif

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    // query <-> PQ chunk centers distances
    // we have a rotation matrix
    float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
    this->pq_table.populate_chunk_distances(query_float, pq_dists);

    uint32_t bv = diskann::defaults::DEFAULT_AISAQ_VECTORS_BEAMWIDTH;
    uint64_t pq_read_page_cache_size = 0;
    if (aisaq_search_config != nullptr) {
        bv = aisaq_search_config->vector_beamwidth;
        pq_read_page_cache_size = aisaq_search_config->pq_read_page_cache_size;
    }
    // query <-> neighbor list
    float *dist_scratch = query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;


    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, release_data, pq_coord_scratch,
                          pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                    float *dists_out, AisaqPQReaderContext &ctx,
                                    QueryStats *stats) mutable{
        Timer timer;
        timer.reset();
        uint32_t io_count;
        /* submit */
        if (_aisaq_pq_vectors_reader->read_pq_vectors_submit(ctx, ids, n_ids,
                                                             io_count) != 0) {
        	release_data();
            throw ANNException("Failed read PQ vectors submit",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        uint32_t read_vec[n_ids]; /* index array */
        uint8_t *read_coords[n_ids];
        uint32_t rcount, min_events = 8, read_remain = n_ids, i;
        /* wait completion */
        do {
            if (min_events > read_remain) {
                min_events = read_remain;
            }
            if (_aisaq_pq_vectors_reader->read_pq_vectors_wait_completion(
                    ctx, read_vec, read_coords, min_events, read_remain,
                    rcount) != 0) {
            	release_data();
                throw ANNException("Failed read PQ wait",
                                   -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            /* handle rcount vectors in read_vec */
            for (i = 0; i < rcount; i++) {
                diskann::pq_dist_lookup(read_coords[i], 1, this->n_chunks,
                                        pq_dists, dists_out + read_vec[i]);
            }
            read_remain -= rcount;
        } while (read_remain > 0);
        if (stats != nullptr) {
            stats->io_us += (float)timer.elapsed();
            stats->n_ios += io_count;
            stats->n_4k += io_count;
            stats->n_cmps += (uint32_t)n_ids;
        }

        /* done */
        _aisaq_pq_vectors_reader->read_pq_vectors_done(ctx);
    };
    auto ctx_pool = AioContextPool::GetGlobalAioPool();
    auto max_ios = ctx_pool->max_events_per_ctx();

    Timer query_timer, io_timer, cpu_timer;
    if (aisaq_data.aisaq_pq_reader_ctx != nullptr) 
        _aisaq_pq_vectors_reader->set_page_cache_size(*aisaq_data.aisaq_pq_reader_ctx, pq_read_page_cache_size);

    tsl::robin_set<uint64_t> *visited = query_scratch->visited;
    NeighborPriorityQueue &retset = aisaq_data.retset;
    retset.reserve(local_l_search);
    std::vector<Neighbor> &full_retset = aisaq_data.full_retset;
    
    uint32_t best_medoid;
    float best_dist;
    get_entry_point_medoid(best_medoid, best_dist, pq_dists, dist_scratch, query_float);
    
    retset.insert(Neighbor(best_medoid, best_dist));
    visited->insert(best_medoid);

    // cleared every iteration
    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    struct aisaq_node_placement np[bv];
    uint32_t bv_count;
    tsl::robin_map<uint32_t, char *> frontier_items;
    T *node_fp_coords;
    uint32_t nnbrs;
    uint32_t *node_nbrs;
    uint32_t agg_nnbrs;
    uint32_t agg_nnbrs_inline;
    uint32_t agg_node_nbrs[bv * this->max_degree];
    uint32_t agg_node_nbrs_inline[bv * this->max_degree];
    float agg_dist_scratch[bv * this->max_degree];
    float agg_dist_scratch_inline[bv * this->max_degree];

    float cur_expanded_dist;
    std::vector<uint32_t> free_ids;
    size_t position;
    uint32_t id;
    char *buf;

    struct {
        uint32_t *nbrs_list;
        float *dist_list;
        const uint32_t &size;
    } agg_nbrs_lists[] = {
        {agg_node_nbrs, agg_dist_scratch, agg_nnbrs},
        {agg_node_nbrs_inline, agg_dist_scratch_inline, agg_nnbrs_inline},
    };
    /* initialize free nodes pool */
    cpu_timer.reset();
    while (retset.has_unexpanded_node()) {
        bv_count = 0;
        /* #select Bi neighbors */
        /* #phase 1, iterate over closest unexpanded neighbours in retset
         * (without expanding) */
        /* #check if the first Bv nodes are available either in cache or in
         * pre-fetched, if not, fetch Bi items */
        /* Get first/next closest unexpanded Neighbour from retset (without
         * expanding) */
        if (retset.get_first_unexpanded_position(position, true)) {
            do {
                std::shared_lock<std::shared_mutex> lock(this->cache_mtx);
                id = retset[position].id;
                auto iter = this->nhood_cache.find(id);
                if (iter != this->nhood_cache.end()) {
                    if (bv_count < bv) {
                        /* Add entry to Bv_list with _nhood_cache placement &
                         * node */
                        np[bv_count].id = id;
                        np[bv_count].is_in_cache = true;
                        np[bv_count].ptr = (char *)(&(iter.value()));
                        bv_count++;
                    }
                    if (stats != nullptr) {
                        stats->n_cache_hits++;
                    }
                } else {
                    /* Otherwise, Add Neighbour’s vector ID to frontier_read_req
                     * list */
                    auto frontier_iter = frontier_items.find(id);
                    if (frontier_iter == frontier_items.end()) {
                        /* Not in frontier map. Needs to be read */
                        if (aisaq_data.aisaq_scratch_mem_offset.empty()) {
                        	release_data();
                            throw ANNException("No free nodes, increase "
                                               "defaults::MAX_N_SECTOR_READS.",
                                               -1, __FUNCSIG__, __FILE__,
                                               __LINE__);
                        }
                        buf = sector_scratch +
                              aisaq_data.aisaq_scratch_mem_offset.back();
                        frontier_read_reqs.emplace_back(
                            get_node_sector((size_t)id) * defaults::SECTOR_LEN,
                            num_sectors_per_node * defaults::SECTOR_LEN, buf);
                        /* Add to frontier buffer */
                        frontier_items[id] = buf;
                        aisaq_data.aisaq_scratch_mem_offset.pop_back();
                    } else {
                        /* In frontier map */
                        buf = (char *)(frontier_iter->second);
                    }
                    if (bv_count < bv) {
                        /* Add entry to Bv_list with frontier_nhood placement
                         * data ptr = null*/
                        np[bv_count].id = id;
                        np[bv_count].is_in_cache = false;
                        np[bv_count].ptr = buf;
                        bv_count++;
                    }
                }
                if (bv_count <= bv) {
                    std::shared_lock<std::shared_mutex> lock(this->node_visit_counter_mtx);
                    if (this->count_visited_nodes) {
                        this->node_visit_counter[id].second->fetch_add(1);
                    }
                }
                float alpha_temp=0;
                //After expanding point that should ignore, remove it
                if(should_ignore_point(id, alpha, alpha_temp, bitset)){
                	retset.remove(position);
               		free_ids.push_back(id);
                	position--;
                }
                /* #if Bv items are available without the need to read, skip
                 * read this time */
                /* If node_index < Bv */
                /* #assure that not more than Bi items will be read at once
                 */
                /* If frontier_read_req list size is equal Bi */
                if ((bv_count == bv && frontier_read_reqs.size() == 0)
                		|| (frontier_read_reqs.size() == beam_width)) {
                    break;
                }
            } while (
                retset.get_next_unexpanded_position(position, bv_count < bv));
        }
        /* #read frontier from disk */
        /* If frontier_read_req is not empty */
        if (!frontier_read_reqs.empty()) {
            io_timer.reset();
            this->reader->read(frontier_read_reqs, ctx); // synchronous IO linux

            if (stats != nullptr) {
                stats->io_us += (float)io_timer.elapsed();
                stats->n_hops++;
                updata_io_stats(*stats, frontier_read_reqs.size(), num_sectors_per_node);
            }

            frontier_read_reqs.clear();
        }

        cpu_timer.reset();
        /* For each item in Bv_list */
        agg_nnbrs = 0, agg_nnbrs_inline = 0;
        float accumlative_alpha=0;
        for (uint32_t i = 0; i < bv_count; i++) {
            /* #Expand the node depending on its placement */
            /* Calculate node memory location according to its placement
             * (cached_nhood, frontier_nhoods, prefetched_nhood) */
            /* Calculate node distance (with precision saved as part of the
             * index - full or disk_pq) */
            /* Insert node to full_retset */
            id = np[i].id;
            uint32_t rid;
            if (np[i].is_in_cache) {
                std::shared_lock<std::shared_mutex> lock(this->cache_mtx);
                auto global_cache_iter = this->coord_cache.find(id);
                node_fp_coords = global_cache_iter->second;
                std::pair<uint32_t, uint32_t *> *cache_item =
                    (std::pair<uint32_t, uint32_t *> *)(np[i].ptr);
                nnbrs = cache_item->first;
                node_nbrs = cache_item->second;
                if (_aisaq_rearranged_vectors) {
                    rid = *((uint32_t *)_aisaq_node_cache.find(id)->second);
                }
            } else {
                char *node_disk_buf = offset_to_node(np[i].ptr, id);
                uint32_t *nhood_buf = offset_to_node_nhood(node_disk_buf);
                node_fp_coords = offset_to_node_coords(node_disk_buf);
                nnbrs = *nhood_buf;
                node_nbrs = (nhood_buf + 1);
                if ((uint64_t)node_fp_coords & 0x1fllu) {
                    /* Copy only if not aligned to 32 */
                    memcpy(data_buf, node_fp_coords,
                           this->disk_bytes_per_point);
                    node_fp_coords = data_buf;
                }
                if (_aisaq_rearranged_vectors) {
                    rid = *(nhood_buf + this->max_degree + 1);
                }
                auto iter = std::find(free_ids.begin(),free_ids.end(),id);
                if(iter == free_ids.end()){
                	free_ids.push_back(id);
                }
            }
            if (!this->use_disk_index_pq) {
                cur_expanded_dist =
                    this->dist_cmp_wrap(aligned_query_T, node_fp_coords,
                                        (size_t)this->aligned_dim, id);
            } else {
                if (this->metric == diskann::Metric::INNER_PRODUCT ||
                        this->metric == diskann::Metric::COSINE) {
                    cur_expanded_dist = this->disk_pq_table.inner_product(
                        query_float, (uint8_t *)node_fp_coords);
                } else {
                    cur_expanded_dist =
                        this->disk_pq_table.l2_distance( // disk_pq does not
                                                         // support OPQ yet
                            query_float, (uint8_t *)node_fp_coords);
                }
 
            }
            if (stats != nullptr) {
                 stats->n_cmps++;
            }
            /* Insert node to full_retset */
            float temp_alpha=0;
            //After expanding point that should ignore, remove it
            if (!should_ignore_point(id, alpha, temp_alpha,bitset)) {
            	full_retset.push_back(Neighbor(id, cur_expanded_dist, rid));
                // add top candidate info into feder result
                if (feder != nullptr) {
                    feder->visit_info_.AddTopCandidateInfo(_aisaq_rearranged_vectors ? rid : id, cur_expanded_dist);
                    feder->id_set_.insert(_aisaq_rearranged_vectors ? rid : id);
                }
            }
            uint32_t m = 0, idn;
            /* note that node cache does not hold inline vectors
               _aisaq_inline_pq_vectors > 0 */
            if (_aisaq_inline_pq_vectors > 0) {
                /* handle inline vectors */
                uint32_t inline_limit = std::min(_aisaq_inline_pq_vectors, nnbrs);
                char *inline_pq_vectors_buff;
                if (np[i].is_in_cache) {
                    inline_pq_vectors_buff =
                        (char *)_aisaq_node_cache.find(id)->second;
                } else {
                    char *node_buf = offset_to_node(np[i].ptr, id);
                    inline_pq_vectors_buff =
                        aisaq_offset_to_node_aisaq_data(node_buf);
                }
                if (_aisaq_rearranged_vectors) {
                    inline_pq_vectors_buff += sizeof(uint32_t);
                }
                diskann::pq_dist_lookup(
                    (uint8_t *)inline_pq_vectors_buff, inline_limit,
                    this->n_chunks, pq_dists,
                    agg_dist_scratch_inline + agg_nnbrs_inline);
                uint32_t __iv_count = 0;
                for (; m < inline_limit; m++) {
                    idn = node_nbrs[m];
                    if (!visited->insert(idn).second) {
                        /* already visited */
                        continue;
                    }
                    if (should_ignore_point(idn, alpha, accumlative_alpha, bitset)) {
                        continue;
                    }
                    agg_node_nbrs_inline[agg_nnbrs_inline + __iv_count] = idn;
                    agg_dist_scratch_inline[agg_nnbrs_inline + __iv_count] =
                        agg_dist_scratch_inline[agg_nnbrs_inline + m];
                    __iv_count++;
                }
                agg_nnbrs_inline += __iv_count;
            }
            for (; m < nnbrs; m++) {
                idn = node_nbrs[m];
                if (!visited->insert(idn).second) {
                    /* already visited */
                    continue;
                }
                if (should_ignore_point(idn, alpha, accumlative_alpha, bitset)) {
                    continue;
                }
                /* nhood non-inline vectors */
                uint8_t *pqvb = aisaq_pq_cache_lookup(idn);
                if (pqvb != nullptr) {
                    /* vector is in cache */
                    diskann::pq_dist_lookup(pqvb, 1, this->n_chunks, pq_dists,
                                            agg_dist_scratch_inline +
                                                agg_nnbrs_inline);
                    agg_node_nbrs_inline[agg_nnbrs_inline] = idn;
                    agg_nnbrs_inline++;
                    continue;
                }
                agg_node_nbrs[agg_nnbrs] = idn;
                agg_nnbrs++;
            }
        }
        if (stats != nullptr) {
            stats->cpu_us += (float)cpu_timer.elapsed();
        }
        if (agg_nnbrs > 0) {
               uint32_t computed_count = 0;
               do {
				  uint32_t _nids = std::min(agg_nnbrs - computed_count, (uint32_t)max_ios);
				  compute_dists(agg_node_nbrs + computed_count, _nids, agg_dist_scratch + computed_count, *aisaq_data.aisaq_pq_reader_ctx, stats);
				  computed_count +=  _nids;
               } while (computed_count < agg_nnbrs);
        }
        //}
        cpu_timer.reset();
        // process prefetched nhood
        for (uint32_t nl = 0;
            nl < sizeof(agg_nbrs_lists) / sizeof(agg_nbrs_lists[0]); nl++) {
            for (uint32_t m = 0; m < agg_nbrs_lists[nl].size; m++) {
                uint32_t idn = agg_nbrs_lists[nl].nbrs_list[m];
                float dist = agg_nbrs_lists[nl].dist_list[m];
                Neighbor nn(idn, dist);
                uint32_t removed_vec_id;
                bool is_removed;
                retset.insert_with_rem_info(nn, is_removed, removed_vec_id);
                if (is_removed) {
                    free_ids.push_back(removed_vec_id);
                }
            }
        }
        for (uint32_t free_it = 0; free_it < free_ids.size(); free_it++) {
            auto it = frontier_items.find(free_ids[free_it]);
            if (it != frontier_items.end()) {
                aisaq_data.aisaq_scratch_mem_offset.push_back(it.value() -
                                                        sector_scratch);
                frontier_items.erase(it);
            }
        }
        free_ids.clear();
        if (stats != nullptr) {
            stats->cpu_us += (float)cpu_timer.elapsed();
        }
        // hops++;
    }
    /* clear page cache between queries */
    // PqVectorsOnDisk::getInstance()->clear_page_cache(&data->pq_ctx);

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data) {
        if (!(this->_reorder_data_exists)) {
        	release_data();
            throw ANNException("Requested use of reordering data which does "
                               "not exist in index "
                               "file",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        rerank_candidate_list(full_retset, k_search,
                              sector_scratch, stats,
                              io_timer, ctx, aligned_query_T);
        
    }

    prepare_search_results(full_retset, k_search, distances, indices, query_norm);

    if (stats != nullptr) {
        stats->total_us = (float)query_timer.elapsed();
    }
    if (aisaq_data.aisaq_pq_reader_ctx != nullptr)
        _aisaq_pq_vectors_reader->hibernate(*aisaq_data.aisaq_pq_reader_ctx);

    release_data();
}

template <typename T>
void PQFlashAisaqIndex<T>::prepare_search_results(std::vector<Neighbor> &full_retset,
                                                  const uint64_t k_search,
                                                  float *distances, int64_t *indices,
                                                  float query_norm)
{
    for (uint64_t i = 0; i < k_search; i++) {
        if (i >= full_retset.size()) {
            indices[i] = -1;
            if (distances != nullptr) {
                distances[i] = -1;
            }
            continue;
        }
        indices[i] =
            _aisaq_rearranged_vectors ? full_retset[i].rid : full_retset[i].id;

        if (distances != nullptr) {
            distances[i] = full_retset[i].distance;
            if (this->metric == diskann::Metric::INNER_PRODUCT) {
                // convert l2 distance to ip distance
                distances[i] = 1.0 - distances[i] / 2.0;
                // rescale to revert back to original norms (cancelling the effect of
                // base and query pre-processing)
                if (this->max_base_norm != 0)
                    distances[i] *= (this->max_base_norm * query_norm);
            } else if (this->metric == diskann::Metric::COSINE) {
                distances[i] = -distances[i];
            }
        }
    }
}

template <typename T>
void PQFlashAisaqIndex<T>::rerank_candidate_list(std::vector<Neighbor> &full_retset, const uint64_t k_search,
                                                 char *sector_scratch, QueryStats *stats,
                                                 Timer &io_timer, IOContext ctx, T *aligned_query_T){
    std::vector<AlignedRead> vec_read_reqs;

    if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
        full_retset.erase(full_retset.begin() +
                              k_search * FULL_PRECISION_REORDER_MULTIPLIER,
                          full_retset.end());

    for (size_t i = 0; i < full_retset.size(); ++i) {
        // MULTISECTORFIX
        vec_read_reqs.emplace_back(
            VECTOR_SECTOR_NO(((size_t)full_retset[i].id)) *
                defaults::SECTOR_LEN,
            defaults::SECTOR_LEN,
            sector_scratch + i * defaults::SECTOR_LEN);

        if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
        }
    }

    io_timer.reset();
    this->reader->read(vec_read_reqs, ctx); // synchronous IO linux
    if (stats != nullptr) {
        stats->io_us += io_timer.elapsed();
    }

    for (size_t i = 0; i < full_retset.size(); ++i) {
        auto id = full_retset[i].id;
        // MULTISECTORFIX
        auto location = (sector_scratch + i * defaults::SECTOR_LEN) +
                        VECTOR_SECTOR_OFFSET(id);
        full_retset[i].distance = this->dist_cmp_wrap(
            aligned_query_T, (T *)location, (size_t)this->aligned_dim, id);
    }

    std::sort(full_retset.begin(), full_retset.end());
}

template <typename T>
void PQFlashAisaqIndex<T>::load_cache_list(std::vector<uint32_t> &node_list) {
    LOG_KNOWHERE_DEBUG_ << "Loading the cache list into memory..";
    size_t num_cached_nodes = node_list.size();

    // Allocate space for neighborhood cache
    this->nhood_cache_buf =
        std::make_unique<unsigned[]>(num_cached_nodes * (this->max_degree + 1));
    memset(this->nhood_cache_buf.get(), 0,
           num_cached_nodes * (this->max_degree + 1));

    // Allocate space for coordinate cache
    size_t coord_cache_buf_len = num_cached_nodes * this->aligned_dim;
    diskann::alloc_aligned((void **)&this->coord_cache_buf,
                           coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset((void*)this->coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    // Allocate space for AiSAQ node data cache
    uint32_t aisaq_data_len_u32 = DIV_ROUND_UP(
        _aisaq_inline_pq_vectors * this->n_chunks * sizeof(uint8_t),
        sizeof(uint32_t));
    if (_aisaq_rearranged_vectors) {
        aisaq_data_len_u32++;
    }
    if (aisaq_data_len_u32 > 0) {
        _aisaq_node_cache_buf =
            (uint8_t *)(new uint32_t[num_cached_nodes * aisaq_data_len_u32]);
    }

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
    for (size_t block = 0; block < num_blocks; block++) {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

        // Copy offset into buffers to read into
        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> coord_buffers;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        std::vector<uint8_t *> aisaq_buffers;
        for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++) {
            nodes_to_read.push_back(node_list[node_idx]);
            coord_buffers.push_back(this->coord_cache_buf +
                                    node_idx * this->aligned_dim);
            nbr_buffers.emplace_back(0, this->nhood_cache_buf.get() +
                                            node_idx * this->max_degree);
            if (_aisaq_node_cache_buf != nullptr) {
                aisaq_buffers.push_back(
                    _aisaq_node_cache_buf +
                    (node_idx * aisaq_data_len_u32 * sizeof(uint32_t)));
            }
        }

        // issue the reads
        auto read_status = read_nodes(
            nodes_to_read, coord_buffers, nbr_buffers,
            _aisaq_node_cache_buf != nullptr ? &aisaq_buffers : nullptr);

        // check for success and insert into the cache.
        for (size_t i = 0; i < read_status.size(); i++) {
            if (read_status[i] == true) {
                {
                    std::unique_lock<std::shared_mutex> lock(this->cache_mtx);
                    this->coord_cache.insert(
                        std::make_pair(nodes_to_read[i], coord_buffers[i]));
                    this->nhood_cache.insert(
                        std::make_pair(nodes_to_read[i], nbr_buffers[i]));
                    if (_aisaq_node_cache_buf != nullptr) {
                        _aisaq_node_cache.insert(
                            std::make_pair(nodes_to_read[i], aisaq_buffers[i]));
                    }
                }
            }
        }
    }
    LOG_KNOWHERE_DEBUG_ << "..done.";
}

template <typename T> uint32_t PQFlashAisaqIndex<T>::get_max_node_len() {
    return this->max_node_len;
}

template <typename T>
int PQFlashAisaqIndex<T>::aisaq_load_rearrange_data(const char *index_prefix) {
    std::string rearrange_map_path =
        std::string(index_prefix) + "_disk.index_rearrange.bin";
    size_t npts, dim;
    diskann::load_bin<uint32_t>(rearrange_map_path,
                                _aisaq_rearranged_vectors_map, npts, dim);
    if (dim != 1 || npts != this->num_points) {
        LOG_KNOWHERE_ERROR_ << "rearrange map size mismatch in file "
                            << rearrange_map_path;
        auto ptr = _aisaq_rearranged_vectors_map.release();
        delete[] ptr;
        _aisaq_rearranged_vectors_map = nullptr;
        return -1;
    }

    return 0;
}

template <typename T>
bool PQFlashAisaqIndex<T>::should_ignore_point(
    uint32_t id, float alpha, float& accumulative_alpha, const knowhere::BitsetView &bitset) {
    uint32_t __idn = _aisaq_rearranged_vectors
                         ? _aisaq_rearranged_vectors_map.get()[id]
                         : id;
    bool ignore = !bitset.empty() && bitset.test(__idn);
    if(ignore) {
    	accumulative_alpha += alpha;
 	    if (accumulative_alpha < 1.0f) {
		  return true;
	    }
 	   accumulative_alpha -= 1.0f;
    }
    return false;
}

template <typename T>
PQFlashAisaqIndex<T>::PQFlashAisaqIndex(
    std::shared_ptr<AlignedFileReader> fileReader, diskann::Metric m)
    : PQFlashIndex<T>(fileReader, m) {
}

template <typename T> PQFlashAisaqIndex<T>::~PQFlashAisaqIndex() {
    this->destroy_cache_async_task();
    {
        std::unique_lock<std::mutex> guard(this->state_controller->status_mtx);
        this->state_controller->status.store(ThreadSafeStateController::Status::DONE);
    }
    if (this->data != nullptr) {
        auto data_ptr = this->data.release();
        delete[] data_ptr;
        this->data = nullptr;
    }

    if (this->centroid_data != nullptr) {
        aligned_free(this->centroid_data);
        this->centroid_data = nullptr;
    }
    // delete backing bufs for nhood and coord cache
    if (this->nhood_cache_buf != nullptr) {
        auto ptr = this->nhood_cache_buf.release();
        delete[] ptr;
        diskann::aligned_free(this->coord_cache_buf);
        this->nhood_cache_buf = nullptr;
    }

    if (this->medoids != nullptr) {
        auto ptr = this->medoids.release();
        delete[] ptr;
        this->medoids = nullptr;
    }
    if (_aisaq_entry_points_pq_vectors_buff != nullptr) {
        delete[] _aisaq_entry_points_pq_vectors_buff;
        _aisaq_entry_points_pq_vectors_buff = nullptr;
    }
    if (_aisaq_entry_points != nullptr) {
        auto ptr = _aisaq_entry_points.release();
        delete[] ptr;
        _aisaq_entry_points = nullptr;
    }
    /* aisaq related */
    if (_aisaq_node_cache_buf != nullptr) {
        delete[] _aisaq_node_cache_buf;
        _aisaq_node_cache_buf = nullptr;
    }
    if (_aisaq_pq_vectors_cache_buf != nullptr) {
        delete[] _aisaq_pq_vectors_cache_buf;
        _aisaq_pq_vectors_cache_buf = nullptr;
    }
    if (_aisaq_rearranged_vectors_map != nullptr) {
        auto ptr = _aisaq_rearranged_vectors_map.release();
        delete[] ptr;
        _aisaq_rearranged_vectors_map = nullptr;
    }
    if (_aisaq_medoids_pq_vectors_buff != nullptr) {
        delete[] _aisaq_medoids_pq_vectors_buff;
	_aisaq_medoids_pq_vectors_buff = nullptr;
    }
    if (this->load_flag) {
        while (aisaq_thread_data.size() > 0) {
            AisaqThreadData<T> aisaq_data = aisaq_thread_data.pop();
            if (aisaq_data.aisaq_pq_reader_ctx != nullptr) {
                _aisaq_pq_vectors_reader->destroy_context(*aisaq_data.aisaq_pq_reader_ctx);
            }
         }
    }
    if (_aisaq_pq_vectors_reader != nullptr) {
        delete _aisaq_pq_vectors_reader;
        _aisaq_pq_vectors_reader = nullptr;
    }
}


template class PQFlashAisaqIndex<float>;
template class PQFlashAisaqIndex<knowhere::fp16>;
template class PQFlashAisaqIndex<knowhere::bf16>;

} // namespace diskann
