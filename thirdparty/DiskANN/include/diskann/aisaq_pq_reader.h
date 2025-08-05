// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <stdint.h>
#include <string>
#include <tsl/robin_map.h>

#include "aisaq.h"

namespace diskann {
    class AisaqPQReaderContext;
    
    class AisaqPQReader {
    public:
        static AisaqPQReader *create_reader(enum aisaq_pq_io_engine io_engine, const char *pq_file_path, bool rearranged);
        virtual ~AisaqPQReader();
        virtual const char *get_io_engine_name() = 0;
        virtual AisaqPQReaderContext *create_context(uint32_t max_ios) = 0;
        uint32_t get_context_size(AisaqPQReaderContext &ctx);
        virtual void destroy_context(AisaqPQReaderContext &ctx) = 0;
        virtual int read_pq_vectors_submit(AisaqPQReaderContext &ctx, const uint32_t *ids, const uint32_t n_ids, uint32_t &io_count) = 0;
        virtual int read_pq_vectors_wait_completion(AisaqPQReaderContext &ctx, uint32_t *read_vec, uint8_t **pq_vectors, uint32_t nr_events,
                uint32_t max_events, uint32_t &rcount) = 0;
        virtual void read_pq_vectors_done(AisaqPQReaderContext &ctx) = 0;
        void clear_page_cache(AisaqPQReaderContext &ctx);
        bool set_page_cache_size(AisaqPQReaderContext &ctx, uint64_t page_cache_size_bytes);
        void hibernate(AisaqPQReaderContext &ctx);
        static void get_buffers_pool_info(uint64_t &total_allocated /* in bytes */, uint64_t &total_in_pool /* in bytes */);
    protected:
        AisaqPQReader();
        virtual int init_reader(const char *pq_file_path, bool rearranged) = 0;
        virtual void cleanup_reader() = 0;
        int init_reader_common(const char *pq_file_path, bool rearranged);
        void cleanup_reader_common();
        /* helpers */
        void calc_pq_vector_offset_bytes(uint32_t id, uint64_t &offset_from_header, uint32_t &header_size);
        void calc_pq_vector_read_params(uint32_t id, uint64_t &from_sector, uint64_t &to_sector, uint32_t &buff_offset);
        uint8_t *get_free_data_buffer(AisaqPQReaderContext &ctx);
        void add_pending_io_completion_event(AisaqPQReaderContext &ctx, uint32_t completed_index);

        std::string m_pq_file_path;
        bool m_rearranged;
        uint32_t m_rearranged_pq_page_size; /* applicable with rearranged only */
        uint32_t m_rearranged_pq_vectors_per_page; /* applicable with rearranged only */
        uint32_t m_rearranged_pq_sectors_per_page; /* applicable with rearranged only */
        uint32_t m_block_size; /* file block size */
        uint32_t m_num_vectors;
        uint32_t m_pq_vector_size; /* in bytes */
        uint32_t m_max_io_size_sectors;
    };
}
