// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#include "diskann/common_includes.h"

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#ifdef _WINDOWS
#error "windows is not supported"
#endif

#include <fcntl.h>
#include <libaio.h>
#include <linux/fs.h>
#include <list>
#include <map>
#include <mntent.h>
#include <sys/ioctl.h>
#include <sys/sysmacros.h>
#include <tsl/robin_map.h>

#include "diskann/defaults.h"
#include "diskann/utils.h"
#include "diskann/aisaq_pq_reader.h"

#define SECTOR_SIZE 512
namespace diskann {

/* buffers pool manager */
class AisaqPQReaderBuffersPoolMgr {
public:
    static AisaqPQReaderBuffersPoolMgr *get_instance(bool create = false);
    void put_instance();
    uint32_t get_buffers(uint32_t count, uint32_t size /* in sectors */, std::vector<uint8_t *> &buffers);
    uint32_t put_buffers(uint32_t count, uint32_t size /* in sectors */, std::vector<uint8_t *> &buffers);
    void get_info(uint64_t &total_allocated /* in bytes */, uint64_t &total_in_pool /* in bytes */) {
        total_allocated = m_allocated * SECTOR_SIZE;
        total_in_pool = m_in_pool * SECTOR_SIZE;
    }
private:
    AisaqPQReaderBuffersPoolMgr();
    ~AisaqPQReaderBuffersPoolMgr();
    std::pair<uint32_t, std::vector<uint8_t *>> *get_pool(uint32_t size /* in sectors */) {
        for (uint32_t i = 0; i < m_pools.size(); i++) {
            if (m_pools[i].first == size) {
                return &m_pools[i];
            }
        }
        return nullptr;
    }
    uint32_t move_elements(std::vector<uint8_t *> &to, std::vector<uint8_t *> &from, uint32_t max_count) {
        uint32_t count = std::min(max_count, (uint32_t)from.size());
        to.insert(to.end(),
                std::make_move_iterator(from.begin()),
                std::make_move_iterator(from.begin() + count));
        from.erase(from.begin(), from.begin() + count);
        return count;
    }
    std::vector<std::pair<uint32_t, std::vector<uint8_t *>>> m_pools; /* <size-sectors, pool> */
    std::atomic<uint64_t> m_allocated; /* in sectors */
    std::atomic<uint64_t> m_in_pool;   /* in sectors */
    uint32_t m_ref_count;
    static std::mutex m_lock;
    static AisaqPQReaderBuffersPoolMgr *m_instance;
};

std::mutex AisaqPQReaderBuffersPoolMgr::m_lock;
AisaqPQReaderBuffersPoolMgr *AisaqPQReaderBuffersPoolMgr::m_instance = NULL;

AisaqPQReaderBuffersPoolMgr::AisaqPQReaderBuffersPoolMgr()
    : m_allocated(0), m_in_pool(0), m_ref_count(1)
{
}

AisaqPQReaderBuffersPoolMgr::~AisaqPQReaderBuffersPoolMgr()
{
    for (unsigned int i = 0; i < m_pools.size(); i++) {
        std::vector<uint8_t *> &pool_buffers = m_pools[i].second;
        while (pool_buffers.size() > 0) {
            std::free(pool_buffers.back());
            pool_buffers.pop_back();
        }
    }
}

/* static, public */
AisaqPQReaderBuffersPoolMgr *AisaqPQReaderBuffersPoolMgr::get_instance(bool create /* = false */)
{
    AisaqPQReaderBuffersPoolMgr::m_lock.lock();
    if (AisaqPQReaderBuffersPoolMgr::m_instance != NULL) {
        AisaqPQReaderBuffersPoolMgr::m_instance->m_ref_count++;
    } else {
        if (create) {
            AisaqPQReaderBuffersPoolMgr::m_instance = new AisaqPQReaderBuffersPoolMgr;
        }
    }
    AisaqPQReaderBuffersPoolMgr::m_lock.unlock();
    return AisaqPQReaderBuffersPoolMgr::m_instance;
}

/* public */
void AisaqPQReaderBuffersPoolMgr::put_instance()
{
    AisaqPQReaderBuffersPoolMgr::m_lock.lock();
    if (--AisaqPQReaderBuffersPoolMgr::m_instance->m_ref_count == 0) {
        delete AisaqPQReaderBuffersPoolMgr::m_instance;
        AisaqPQReaderBuffersPoolMgr::m_instance = NULL;
    }
    AisaqPQReaderBuffersPoolMgr::m_lock.unlock();
}

/* return the number of buffers successfully transferred from pool */
uint32_t AisaqPQReaderBuffersPoolMgr::get_buffers(uint32_t count,
        uint32_t size /* in sectors */, std::vector<uint8_t *> &buffers)
{
    uint32_t from_pool = 0, allocated = 0;
    std::pair<uint32_t, std::vector<uint8_t *>> *pool;

    AisaqPQReaderBuffersPoolMgr::m_lock.lock();
    if ((pool = get_pool(size)) != nullptr) {
        from_pool = move_elements(buffers, pool->second, count);
        m_in_pool.fetch_sub(from_pool * size);
    }
    AisaqPQReaderBuffersPoolMgr::m_lock.unlock();
    if (from_pool < count) {
        /* not enough buffers in pool, allocate */
        count-= from_pool;
        do {
            uint8_t *buff = (uint8_t *)std::aligned_alloc(SECTOR_SIZE, size * SECTOR_SIZE);
            if (buff == nullptr) {
                break;
            }
            buffers.push_back(buff);
            allocated++;
        } while (allocated < count);
        m_allocated.fetch_add(allocated * size);
    }
    return from_pool + allocated;
}

/* return the number of buffers successfully transferred back in pool */
uint32_t AisaqPQReaderBuffersPoolMgr::put_buffers(uint32_t count,
        uint32_t size /* in sectors */, std::vector<uint8_t *> &buffers)
{
    uint32_t ret = 0;
    std::pair<uint32_t, std::vector<uint8_t *>> *pool;

    AisaqPQReaderBuffersPoolMgr::m_lock.lock();
    if ((pool = get_pool(size)) == nullptr) {
        /* create a new pool */
        m_pools.push_back(std::pair<uint32_t, std::vector<uint8_t *>>(size, std::vector<uint8_t *>()));
        pool = &m_pools.back();
    }
    ret = move_elements(pool->second, buffers, count);
    m_in_pool.fetch_add(ret * size);
    AisaqPQReaderBuffersPoolMgr::m_lock.unlock();
    return ret;
}

/**************************************/
/* reader context */
class AisaqPQReaderContext {
public:
    virtual ~AisaqPQReaderContext();
    virtual int init_context(const char *pq_file_path, uint32_t max_ios, uint32_t max_io_size_sectors) = 0;
    virtual void cleanup_context() = 0;
private:
    void clear_page_cache();
    bool set_page_cache_size(uint64_t page_cache_size_bytes);
    void hibernate();
    bool wakeup();
    bool set_current_max_ios(uint32_t current_max_ios);
    bool reserve_release_read_buffers(bool reserve_only = false);
    virtual uint32_t get_size() = 0;
protected:
    AisaqPQReaderContext();
    int init_context_common(const char *pq_file_path, int oflags, uint32_t max_ios, uint32_t max_io_size_sectors);
    void cleanup_context_common();
    friend class AisaqPQReader;
    friend class AisaqPQReader_aio;
    struct page_cache_node {
        page_cache_node(uint64_t page_id, uint8_t *buff)
            : m_page_id(page_id), m_buff(buff) {
        }
        uint64_t m_page_id;
        uint8_t *m_buff;
        std::list<page_cache_node>::iterator self;
    };
    int m_fd;
    bool m_hibernated;
    uint32_t m_max_ios;
    uint32_t m_current_max_ios; /* <= m_max_ios */
    uint32_t m_max_io_size_sectors;
    uint32_t m_pending_io_count;
    uint32_t *m_pending_io_completion_events;
    uint32_t m_pending_io_completion_events_count;
    std::vector<uint8_t *> m_free_data_buffers;
    uint32_t m_pq_read_page_cache_n_buffers;
    uint32_t m_allocated_buffers;
    tsl::robin_map<uint64_t, page_cache_node *> m_cached_data_buffers; /* page id to cache node */
    std::list<page_cache_node> m_cached_data_buffers_lru_list;
    AisaqPQReaderBuffersPoolMgr *m_buffers_pool_mgr;
};

AisaqPQReaderContext::AisaqPQReaderContext()
	: m_fd(-1), m_pending_io_count(0)
    , m_pending_io_completion_events(nullptr), m_pending_io_completion_events_count(0)
    , m_pq_read_page_cache_n_buffers(0), m_buffers_pool_mgr(nullptr)
{
}

AisaqPQReaderContext::~AisaqPQReaderContext()
{
}

int AisaqPQReaderContext::init_context_common(const char *pq_file_path, int oflags, uint32_t max_ios, uint32_t max_io_size_sectors)
{
    oflags|= O_RDONLY | O_LARGEFILE;
    m_fd = open(pq_file_path, oflags);
	if (m_fd <= 0) {
        LOG_KNOWHERE_ERROR_ << "failed to open PQ compressed vectors file " << pq_file_path;
        return -1;
    }
    posix_fadvise(m_fd, 0, 0, POSIX_FADV_DONTNEED);

    m_hibernated = true;
    m_max_ios = max_ios;
    m_current_max_ios = m_max_ios;
    m_allocated_buffers = 0;
    m_max_io_size_sectors = max_io_size_sectors;
    m_pq_read_page_cache_n_buffers = 0;

    m_pending_io_completion_events = new uint32_t[max_ios];
    if (m_pending_io_completion_events == nullptr) {
        cleanup_context_common();
        return -1;
    }
    m_buffers_pool_mgr = AisaqPQReaderBuffersPoolMgr::get_instance(true);
    if (m_buffers_pool_mgr == nullptr) {
        cleanup_context_common();
        return -1;
    }
    return 0;
}

void AisaqPQReaderContext::cleanup_context_common()
{
    if (m_buffers_pool_mgr != nullptr) {
        m_buffers_pool_mgr->put_buffers(m_free_data_buffers.size(), m_max_io_size_sectors, m_free_data_buffers);
        m_buffers_pool_mgr->put_instance();
        m_buffers_pool_mgr = nullptr;
    }
    if (m_pending_io_completion_events != nullptr) {
        delete [] m_pending_io_completion_events;
        m_pending_io_completion_events = nullptr;
    }
    if (m_fd > 0) {
        close(m_fd);
        m_fd = -1;
    }
}

void AisaqPQReaderContext::clear_page_cache()
{
    assert(m_pending_io_count == 0 && m_pending_io_completion_events_count == 0);
    while (!m_cached_data_buffers_lru_list.empty()) {
        struct AisaqPQReaderContext::page_cache_node &cache_node = m_cached_data_buffers_lru_list.back();
        m_free_data_buffers.push_back(cache_node.m_buff);
        m_cached_data_buffers_lru_list.pop_back();
    }
    m_cached_data_buffers.clear();
}

bool AisaqPQReaderContext::reserve_release_read_buffers(bool reserve_only)
{
	assert(!m_hibernated);
	uint32_t desired_buffers = std::max(m_pq_read_page_cache_n_buffers, m_current_max_ios);
	if (desired_buffers == m_allocated_buffers) {
		return true;
	}
	uint32_t nbuffers, act_nbuffers;
	if (desired_buffers > m_allocated_buffers) {
		nbuffers = desired_buffers - m_allocated_buffers;
		/* reserve additional nbuffers buffers */
		act_nbuffers = m_buffers_pool_mgr->get_buffers(nbuffers, m_max_io_size_sectors, m_free_data_buffers);
		m_allocated_buffers+= act_nbuffers;
	} else {
		if (!reserve_only) {
			clear_page_cache();
			nbuffers = m_allocated_buffers - desired_buffers;
			/* release nbuffers buffers */
			act_nbuffers = m_buffers_pool_mgr->put_buffers(nbuffers, m_max_io_size_sectors, m_free_data_buffers);
			m_allocated_buffers-= act_nbuffers;
		}
	}
	return desired_buffers == m_allocated_buffers;
}

bool AisaqPQReaderContext::set_page_cache_size(uint64_t page_cache_size_bytes)
{
	m_pq_read_page_cache_n_buffers = page_cache_size_bytes / (m_max_io_size_sectors * SECTOR_SIZE);
	if (!m_hibernated) {
		return reserve_release_read_buffers();
	}
	return true;
}

bool AisaqPQReaderContext::set_current_max_ios(uint32_t current_max_ios)
{
	if (current_max_ios > m_max_ios || current_max_ios == 0) {
		return false;
	}
	m_current_max_ios = current_max_ios;
	if (!m_hibernated) {
		return reserve_release_read_buffers(true);
	}
	return true;
}

/* enter hibernation state.
   in this state all buffers are released to the pool */
void AisaqPQReaderContext::hibernate()
{
    if (!m_hibernated) {
		clear_page_cache();
		m_buffers_pool_mgr->put_buffers(m_free_data_buffers.size(), m_max_io_size_sectors, m_free_data_buffers);
		m_allocated_buffers = 0;
		m_hibernated = true;
    }
}

/* exit hibernation state.
   try to restore all buffers including cache buffers from the pool */
bool AisaqPQReaderContext::wakeup()
{
    if (m_hibernated) {
		m_hibernated = false;
		return reserve_release_read_buffers();
    }
    return true;
}

/**************************************/

class AisaqPQReaderContext_aio : public AisaqPQReaderContext {
public:
    AisaqPQReaderContext_aio();
protected:
    virtual ~AisaqPQReaderContext_aio();
    virtual int init_context(const char *pq_file_path, uint32_t max_ios, uint32_t max_io_size_sectors);
    virtual void cleanup_context();
    virtual uint32_t get_size();
    friend class AisaqPQReader_aio;
    struct io_data {
        struct iocb iocb;
        bool root_io;
        bool in_page_cache;
        int32_t hooked; /* index of hooked io_data index, -1 for none */
        uint64_t from_sector;
        uint64_t to_sector;
        uint32_t vector_id;
        uint32_t vector_index;
        uint32_t buff_offset;
        uint8_t *buff;
    };
private:
    io_context_t m_aio_ctx;
    struct io_data *m_io_data;
    uint32_t m_io_data_count;
    struct iocb **m_iocbs_ptr;
};

AisaqPQReaderContext_aio::AisaqPQReaderContext_aio()
	: m_aio_ctx(nullptr), m_io_data(nullptr), m_io_data_count(0), m_iocbs_ptr(nullptr)
{
}

AisaqPQReaderContext_aio::~AisaqPQReaderContext_aio()
{
}

int AisaqPQReaderContext_aio::init_context(const char *pq_file_path, uint32_t max_ios, uint32_t max_io_size_sectors)
{
	if (init_context_common(pq_file_path, O_RDONLY | O_LARGEFILE | O_NONBLOCK | O_DIRECT,
                    max_ios, max_io_size_sectors) != 0) {
        return -1;
    }
    if (io_setup(max_ios, &m_aio_ctx) != 0) {
        cleanup_context();
        LOG_KNOWHERE_ERROR_ << "failed to setup aio context";
        return -1;
    }
    m_io_data = new struct AisaqPQReaderContext_aio::io_data[max_ios];
    if (m_io_data == nullptr) {
		cleanup_context();
        return -1;
    }
    m_iocbs_ptr = new struct iocb *[max_ios];
    if (m_iocbs_ptr == nullptr) {
        cleanup_context();
        return -1;
    }
    return 0;
}

void AisaqPQReaderContext_aio::cleanup_context()
{
	if (m_iocbs_ptr != nullptr) {
		delete [] m_iocbs_ptr;
        m_iocbs_ptr = nullptr;
	}
    if (m_io_data != nullptr) {
        delete [] m_io_data;
        m_io_data = nullptr;
    }
    if (m_aio_ctx != nullptr) {
    	io_destroy(m_aio_ctx);
        m_aio_ctx = nullptr;
    }
    cleanup_context_common();
}

uint32_t AisaqPQReaderContext_aio::get_size()
{
	return sizeof(*this) +
		(sizeof(struct AisaqPQReaderContext_aio::io_data) * m_max_ios) +
		(sizeof(struct iocb *) * m_max_ios);
}

/**************************************/

/* readers */
class AisaqPQReader_aio : public AisaqPQReader {
public:
    AisaqPQReader_aio();
protected:
    virtual ~AisaqPQReader_aio();
    virtual const char *get_io_engine_name();
    virtual int init_reader(const char *pq_file_path, bool rearranged);
    virtual void cleanup_reader();
    virtual AisaqPQReaderContext *create_context(uint32_t max_ios);
    virtual void destroy_context(AisaqPQReaderContext &ctx);
    virtual int read_pq_vectors_submit(AisaqPQReaderContext &ctx,
						const uint32_t *ids, const uint32_t n_ids, uint32_t &io_count);
    virtual int read_pq_vectors_wait_completion(AisaqPQReaderContext &ctx, uint32_t *read_vec,
                        uint8_t **pq_vectors, uint32_t nr_events, uint32_t max_events, uint32_t &rcount);
    virtual void read_pq_vectors_done(AisaqPQReaderContext &ctx);
private:
    void drain_ios(AisaqPQReaderContext_aio &aio_ctx);
};

AisaqPQReader_aio::AisaqPQReader_aio()
{
}

AisaqPQReader_aio::~AisaqPQReader_aio()
{
}

const char *AisaqPQReader_aio::get_io_engine_name()
{
	return "aio";
}

int AisaqPQReader_aio::init_reader(const char *pq_file_path, bool rearranged)
{
    int rc = init_reader_common(pq_file_path, rearranged);
    if (rc != 0) {
        return rc;
    }
    /* add engine specific initializations */
    return 0;
}

void AisaqPQReader_aio::cleanup_reader()
{
    cleanup_reader_common();
}

AisaqPQReaderContext *AisaqPQReader_aio::create_context(uint32_t max_ios)
{
    AisaqPQReaderContext *ctx = new AisaqPQReaderContext_aio();
    if (ctx != nullptr) {
        if (ctx->init_context(m_pq_file_path.c_str(), max_ios, m_max_io_size_sectors) != 0) {
            delete ctx;
        	ctx = nullptr;
        }
    }
    return ctx;
}

void AisaqPQReader_aio::destroy_context(AisaqPQReaderContext &ctx)
{
    ctx.cleanup_context();
    delete &ctx;
}

int AisaqPQReader_aio::read_pq_vectors_submit(AisaqPQReaderContext &ctx,
		const uint32_t *ids, const uint32_t n_ids, uint32_t &io_count)
{
	ctx.set_current_max_ios(n_ids);
	if (ctx.m_hibernated) {
		ctx.wakeup();
	}
	if (n_ids > ctx.m_allocated_buffers) {
        LOG_KNOWHERE_ERROR_ << "id list size is greater than max allowed or there is not enough memory to handle the request";
        return -1;
    }
    AisaqPQReaderContext_aio &aio_ctx = reinterpret_cast<AisaqPQReaderContext_aio &>(ctx);
    std::map<uint64_t, uint32_t> sectors_map; /* start sector -> index map */
    uint32_t read_sector_count;
    struct AisaqPQReaderContext_aio::io_data *io_data;
    assert(aio_ctx.m_pending_io_count == 0);
    aio_ctx.m_pending_io_completion_events_count = 0;
    io_count = 0;
    aio_ctx.m_io_data_count = 0;
    if (m_rearranged) {
        /* handle items in cache first, this ensures best cache utilization
           by not dropping cache items that might be needed in this iteration */
        for (uint32_t i = 0; i < n_ids; i++) {
            io_data = aio_ctx.m_io_data + i;
            calc_pq_vector_read_params(ids[i], io_data->from_sector, io_data->to_sector, io_data->buff_offset);
            read_sector_count = io_data->to_sector - io_data->from_sector + 1;
            assert(read_sector_count <= m_max_io_size_sectors);
            io_data->vector_id = ids[i];
            io_data->vector_index = i;
            uint64_t page_id = io_data->from_sector / m_rearranged_pq_sectors_per_page;
            auto iter = aio_ctx.m_cached_data_buffers.find(page_id);
            if ((io_data->in_page_cache = (iter != aio_ctx.m_cached_data_buffers.end()))) {
                /* in cache */
                struct AisaqPQReaderContext::page_cache_node &cache_node = *iter->second;
                /* move to lru back, this also ensures it will not be removed during this read sequence */
                aio_ctx.m_cached_data_buffers_lru_list.splice(aio_ctx.m_cached_data_buffers_lru_list.end(),
                                                              aio_ctx.m_cached_data_buffers_lru_list,
                                                              cache_node.self);
                io_data->hooked = -1;
                io_data->root_io = false;
                io_data->buff = cache_node.m_buff;
                /* mark as completed */
                add_pending_io_completion_event(aio_ctx, io_data->vector_index);
            }
        }
    }

    for (uint32_t i = 0; i < n_ids; i++) {
        io_data = aio_ctx.m_io_data + i;
        if (m_rearranged) {
            if (io_data->in_page_cache) {
                aio_ctx.m_io_data_count++;
                continue;
            }
        } else {
            calc_pq_vector_read_params(ids[i], io_data->from_sector, io_data->to_sector,
                                       io_data->buff_offset);
            read_sector_count = io_data->to_sector - io_data->from_sector + 1;
            assert(read_sector_count <= m_max_io_size_sectors);
            io_data->vector_id = ids[i];
            io_data->vector_index = i;
        }
        auto sectors_map_it = sectors_map.find(io_data->from_sector);
        if (sectors_map_it != sectors_map.end() &&
            aio_ctx.m_io_data[sectors_map_it->second].to_sector >= io_data->to_sector) {
            /* overlapping io, hook it */
            io_data->hooked = aio_ctx.m_io_data[sectors_map_it->second].hooked;
            aio_ctx.m_io_data[sectors_map_it->second].hooked = i;
            io_data->root_io = false;
            io_data->buff = aio_ctx.m_io_data[sectors_map_it->second].buff;
        } else {
            io_data->hooked = -1;
            if (sectors_map_it == sectors_map.end()) {
                sectors_map[io_data->from_sector] = i;
            }
            io_data->buff = get_free_data_buffer(aio_ctx);
            if (io_data->buff == nullptr) {
                LOG_KNOWHERE_ERROR_ << "No available data buffers to read PQ vectors";
                drain_ios(aio_ctx);
                return -1;
            }
            io_data->root_io = true;
            io_prep_pread(&io_data->iocb, aio_ctx.m_fd, io_data->buff,
                          read_sector_count * SECTOR_SIZE, io_data->from_sector * SECTOR_SIZE);
            /* must be set after io_prep_pread */
            io_data->iocb.data = io_data;
            aio_ctx.m_iocbs_ptr[io_count] = &io_data->iocb;
            io_count++;
        }
        aio_ctx.m_io_data_count++;
    }
    int ret = io_submit(aio_ctx.m_aio_ctx, (int64_t)io_count, aio_ctx.m_iocbs_ptr);
    if (ret != (int)io_count) {
        LOG_KNOWHERE_ERROR_ << "io_submit() failed; returned " << ret << ", expected=" << io_count << ", ernno=" << errno
                  << "=" << ::strerror(-ret);
        if (ret > 0) {
            aio_ctx.m_pending_io_count = ret;
        }
        drain_ios(aio_ctx);
        return -1;
    }
    aio_ctx.m_pending_io_count = io_count;
    return 0;
}

int AisaqPQReader_aio::read_pq_vectors_wait_completion(AisaqPQReaderContext &ctx,
		uint32_t *read_vec, uint8_t **pq_vectors, uint32_t nr_events, uint32_t max_events, uint32_t &rcount)
{
    AisaqPQReaderContext_aio &aio_ctx = reinterpret_cast<AisaqPQReaderContext_aio &>(ctx);
    struct AisaqPQReaderContext_aio::io_data *io_data;
    rcount = 0;
    while (aio_ctx.m_pending_io_completion_events_count > 0 && rcount < max_events) {
        aio_ctx.m_pending_io_completion_events_count--;
        io_data = &aio_ctx.m_io_data[aio_ctx.m_pending_io_completion_events[aio_ctx.m_pending_io_completion_events_count]];
        read_vec[rcount] =  io_data->vector_index;
        pq_vectors[rcount] = io_data->buff + io_data->buff_offset;
        rcount++;
    }
    if (rcount >= nr_events) {
        return 0;
    }
    uint32_t __max_events = max_events - rcount;
    if (__max_events > 0 && aio_ctx.m_pending_io_count > 0) {
        struct io_event evts[__max_events];
        if (nr_events > __max_events) {
            nr_events = __max_events;
        }
        if (nr_events > aio_ctx.m_pending_io_count) {
            nr_events = aio_ctx.m_pending_io_count;
        }
        int ret = io_getevents(aio_ctx.m_aio_ctx, (int64_t)nr_events, (int64_t)__max_events, evts, nullptr);
        if (ret <= 0) {
            LOG_KNOWHERE_ERROR_ << "io_getevents() failed; returned " << ret
                      << ", ernno=" << errno << "=" << ::strerror(-ret);
            return -1;
        }
        for (uint32_t i = 0; i < ret; i++) {
            io_data = (struct AisaqPQReaderContext_aio::io_data *) (evts[i].data);
            do {
                if (rcount < max_events) {
                    read_vec[rcount] = io_data->vector_index;
                    pq_vectors[rcount] = io_data->buff + io_data->buff_offset;
                    rcount++;
                } else {
                    add_pending_io_completion_event(aio_ctx, io_data->vector_index);
                }
                if (io_data->hooked == -1) {
                    break;
                }
                io_data = &aio_ctx.m_io_data[io_data->hooked];
             } while (true);
        }
        aio_ctx.m_pending_io_count-= ret;
    }
    return 0;
}

void AisaqPQReader_aio::read_pq_vectors_done(AisaqPQReaderContext &ctx)
{
    AisaqPQReaderContext_aio &aio_ctx = reinterpret_cast<AisaqPQReaderContext_aio &>(ctx);
    struct AisaqPQReaderContext_aio::io_data *io_data;
    assert(aio_ctx.m_pending_io_count == 0);
    for (uint32_t i = 0; i < aio_ctx.m_io_data_count; i++) {
        io_data = aio_ctx.m_io_data + i;
        if (io_data->root_io) {
            uint8_t *buff = io_data->buff;
            if (m_rearranged) {
                uint64_t page_id = io_data->from_sector / m_rearranged_pq_sectors_per_page;
                //assert(aio_ctx.m_cached_data_buffers.find(page_id) == aio_ctx.m_cached_data_buffers.end());
                struct AisaqPQReaderContext::page_cache_node &cache_node =
                                aio_ctx.m_cached_data_buffers_lru_list.emplace_back(page_id, buff);
                cache_node.self = std::prev(aio_ctx.m_cached_data_buffers_lru_list.end());
                /* add to cache */
                aio_ctx.m_cached_data_buffers[page_id] = &cache_node;
            } else {
                aio_ctx.m_free_data_buffers.push_back(buff);
            }
        }
    }
    aio_ctx.m_io_data_count = 0;
}

void AisaqPQReader_aio::drain_ios(AisaqPQReaderContext_aio &aio_ctx)
{
    if (aio_ctx.m_pending_io_count > 0) {
        int retries = 5;
        struct io_event evts[aio_ctx.m_pending_io_count];
        do {
            int ret = io_getevents(aio_ctx.m_aio_ctx, (int64_t) aio_ctx.m_pending_io_count,
                                   (int64_t) aio_ctx.m_pending_io_count, evts, nullptr);
            if (ret > 0) {
                aio_ctx.m_pending_io_count-= ret;
                continue;
            }
            if (retries > 0) {
                retries--;
                continue;
            }
            LOG_KNOWHERE_ERROR_ << "io_getevents() failed; returned " << ret
                      << ", ernno=" << errno << "=" << ::strerror(-ret);
            aio_ctx.m_pending_io_count = 0;
        } while (aio_ctx.m_pending_io_count > 0);
    }
    struct AisaqPQReaderContext_aio::io_data *io_data;
    while (aio_ctx.m_io_data_count > 0) {
        aio_ctx.m_io_data_count--;
        io_data = aio_ctx.m_io_data + aio_ctx.m_io_data_count;
        if (io_data->root_io) {
            aio_ctx.m_free_data_buffers.push_back(io_data->buff);
        }
    }
}

/**************************************/

AisaqPQReader::AisaqPQReader()
{
}

AisaqPQReader::~AisaqPQReader()
{
}

static bool __get_device_logical_block_size(int major, int minor, uint32_t &block_size)
{
    /* should support running from container, this means only sysfs can be used
       for partition, travel one level up to the parent device */
    char block_size_attr_path[PATH_MAX];
    snprintf(block_size_attr_path, sizeof(block_size_attr_path),
             "/sys/dev/block/%d:%d/queue/logical_block_size", major, minor);
    std::ifstream file(block_size_attr_path);
    if (!file) {
        char device_path[40];
        snprintf(device_path, sizeof(device_path), "/sys/dev/block/%d:%d", major, minor);
        strcpy(block_size_attr_path, "/sys/dev/block/");
        ssize_t llen = strlen(block_size_attr_path);
        ssize_t len = readlink(device_path, block_size_attr_path + llen, sizeof(block_size_attr_path) - llen - 1);
        if (len <= 0) {
            return false;
        }
        strcat(block_size_attr_path, "/../queue/logical_block_size");
        file.open(block_size_attr_path);
        if (!file) {
            return false;
        }
    }
    file >> block_size;
    file.close();
    return true;
}

int AisaqPQReader::init_reader_common(const char *pq_file_path, bool rearranged)
{
    struct stat file_stat;
    if (stat(pq_file_path, &file_stat) != 0) {
        LOG_KNOWHERE_ERROR_ << "failed to stat PQ vectors file";
    	return -1;
    }
    /* init m_block_size */
    if (!__get_device_logical_block_size(major(file_stat.st_dev), minor(file_stat.st_dev), m_block_size)) {
        m_block_size = diskann::defaults::SECTOR_LEN;
        LOG_KNOWHERE_DEBUG_ << "unable to detect PQ vectors file block size, using default " << m_block_size;
    }
    /* init m_num_vectors, m_pq_vector_size, m_rearranged_pq_page_size */
    int fd = open(pq_file_path, O_RDONLY);
    if (fd <= 0) {
        LOG_KNOWHERE_ERROR_ << "failed to open PQ vectors file " << pq_file_path;
        return -1;
    }
    size_t res;
	if (rearranged) {
    	struct aisaq_rearranged_pq_compressed_vectors_file_header file_header;
    	res = read(fd, &file_header, sizeof(file_header));
        assert(res == sizeof(file_header));
    	m_num_vectors = file_header.num_vectors;
    	m_pq_vector_size = file_header.vector_size;
    	m_rearranged_pq_page_size = file_header.page_size;
	} else {
    	res = read(fd, &m_num_vectors, sizeof(uint32_t));
        assert(res == sizeof(uint32_t));
    	res = read(fd, &m_pq_vector_size, sizeof(uint32_t));
        assert(res == sizeof(uint32_t));
        m_rearranged_pq_page_size = 0;  /* invalid */
	}
	close(fd);

    /* init m_rearranged_pq_vectors_per_page, m_rearranged_pq_sectors_per_page, m_max_io_size_sectors */
    uint64_t expected_file_size;
    if (rearranged) {
    	m_rearranged_pq_vectors_per_page = m_rearranged_pq_page_size / m_pq_vector_size;
        m_rearranged_pq_sectors_per_page = m_rearranged_pq_page_size / SECTOR_SIZE;
        if (m_rearranged_pq_page_size == 0 ||
            (m_rearranged_pq_page_size % m_block_size) != 0 ||
            (m_rearranged_pq_page_size % diskann::defaults::SECTOR_LEN) != 0) {
            LOG_KNOWHERE_ERROR_ << "invalid/unsupported page size " << m_rearranged_pq_page_size;
            return -1;
        }
        m_max_io_size_sectors = m_rearranged_pq_sectors_per_page;
        expected_file_size =
                (DIV_ROUND_UP(m_num_vectors, m_rearranged_pq_vectors_per_page) * m_rearranged_pq_page_size)
            			+ diskann::defaults::SECTOR_LEN;
    } else {
        m_rearranged_pq_vectors_per_page = m_rearranged_pq_sectors_per_page = 0; /* invalid */
        m_max_io_size_sectors = (((m_pq_vector_size - 1) / m_block_size) + 2) * (m_block_size / SECTOR_SIZE);
        expected_file_size = (sizeof(uint32_t) * 2) + ((uint64_t) m_num_vectors * m_pq_vector_size);
    }
	/* validate file size */
    if (file_stat.st_size != expected_file_size) {
        LOG_KNOWHERE_ERROR_ << "pq vectors file " << pq_file_path << " does not match meta data";
        return -1;
    }
    /* init m_pq_file_path, m_rearranged */
    m_pq_file_path = pq_file_path;
    m_rearranged = rearranged;
    return 0;
}

/* static public */
AisaqPQReader *AisaqPQReader::create_reader(enum aisaq_pq_io_engine io_engine,
	const char *pq_file_path, bool rearranged)
{
    AisaqPQReader *aisaq_reader;
    switch (io_engine) {
        case aisaq_pq_io_engine_aio:
            aisaq_reader = new AisaqPQReader_aio();
            break;
        default:
            return nullptr;
    }
    if (aisaq_reader != nullptr) {
        if (aisaq_reader->init_reader(pq_file_path, rearranged) != 0) {
            delete aisaq_reader;
            aisaq_reader = nullptr;
        }
    }
    return aisaq_reader;
}

void AisaqPQReader::cleanup_reader_common()
{
}

/* public */
void AisaqPQReader::clear_page_cache(AisaqPQReaderContext &ctx)
{
    ctx.clear_page_cache();
}

/* public */
bool AisaqPQReader::set_page_cache_size(AisaqPQReaderContext &ctx, uint64_t page_cache_size_bytes)
{
    uint64_t page_cache_size_bytes_local = page_cache_size_bytes;
    if (!m_rearranged) {
        page_cache_size_bytes_local = 0;
    }
    return ctx.set_page_cache_size(page_cache_size_bytes_local);
}

/* public */
void AisaqPQReader::hibernate(AisaqPQReaderContext &ctx)
{
    ctx.hibernate();
}

/* public */
uint32_t AisaqPQReader::get_context_size(AisaqPQReaderContext &ctx)
{
	return ctx.get_size();
}

/* helpers */

void AisaqPQReader::calc_pq_vector_offset_bytes(uint32_t id, uint64_t &offset_from_header, uint32_t &header_size)
{
    if (m_rearranged) {
        header_size = diskann::defaults::SECTOR_LEN;
        offset_from_header = (((uint64_t)id / m_rearranged_pq_vectors_per_page) * m_rearranged_pq_page_size) +
                             ((id % m_rearranged_pq_vectors_per_page) * m_pq_vector_size);
        return;
    }
    header_size = sizeof(uint32_t) * 2;
    offset_from_header = (uint64_t)id * m_pq_vector_size;
}

void AisaqPQReader::calc_pq_vector_read_params(uint32_t id, uint64_t &from_sector, uint64_t &to_sector, uint32_t &buff_offset)
{
    uint64_t vector_offset_from_header;
    uint32_t header_size;

    calc_pq_vector_offset_bytes(id, vector_offset_from_header, header_size);
    if (m_rearranged) {
        from_sector = (header_size / SECTOR_SIZE) +
                      (vector_offset_from_header / m_rearranged_pq_page_size) * m_rearranged_pq_sectors_per_page;
        to_sector = from_sector + m_rearranged_pq_sectors_per_page - 1;
        buff_offset = vector_offset_from_header % m_rearranged_pq_page_size;
    } else {
        uint32_t sectors_per_block = m_block_size / SECTOR_SIZE;
        from_sector = ((header_size + vector_offset_from_header) / m_block_size) * sectors_per_block;
        to_sector = ((((header_size + vector_offset_from_header + m_pq_vector_size - 1) / m_block_size) + 1) *
                     sectors_per_block) - 1;
        buff_offset = header_size + vector_offset_from_header - (from_sector * SECTOR_SIZE);
    }
}

uint8_t *AisaqPQReader::get_free_data_buffer(AisaqPQReaderContext &ctx)
{
    if (!ctx.m_free_data_buffers.empty()) {
        /* pop from free */
        uint8_t *buff = ctx.m_free_data_buffers.back();
        ctx.m_free_data_buffers.pop_back();
        return buff;
    }
    if (!ctx.m_cached_data_buffers_lru_list.empty()) {
        /* pop from cache */
        struct AisaqPQReaderContext::page_cache_node &cache_node = ctx.m_cached_data_buffers_lru_list.front();
        ctx.m_cached_data_buffers.erase(cache_node.m_page_id);
        ctx.m_cached_data_buffers_lru_list.pop_front();
        return cache_node.m_buff;
    }
    return nullptr;
}

void AisaqPQReader::add_pending_io_completion_event(AisaqPQReaderContext &ctx, uint32_t completed_index)
{
    ctx.m_pending_io_completion_events[ctx.m_pending_io_completion_events_count] = completed_index;
    ctx.m_pending_io_completion_events_count++;
}

/* static, public */
void AisaqPQReader::get_buffers_pool_info(uint64_t &total_allocated /* in bytes */, uint64_t &total_in_pool /* in bytes */)
{
    AisaqPQReaderBuffersPoolMgr *pm = AisaqPQReaderBuffersPoolMgr::get_instance();
    if (pm != nullptr) {
        pm->get_info(total_allocated, total_in_pool);
        pm->put_instance();
    } else {
        total_allocated = 0;
        total_in_pool = 0;
    }
}

}
