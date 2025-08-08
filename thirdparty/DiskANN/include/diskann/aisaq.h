// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once
#include <stdint.h>

#include "diskann/defaults.h"
#include "diskann/neighbor.h"

namespace diskann {

#define AISAQ_VERSION "0.2.0"
#define AISAQ_SEARCH_PQ_CACHE_MAX_VECTORS_PCNT 100
#define AISAQ_SEARCH_PQ_CACHE_MAX_DRAM_GB             8.0f /* 8GB */
#define AISAQ_SEARCH_PQ_CACHE_DIRECT_THRESHOLD_PCNT   100
#define AISAQ_SEARCH_PQ_CACHE_DIRECT_THRESHOLD_N      (10 << 20) /* 10m */
#define AISAQ_SEARCH_PQ_READ_PAGE_CACHE_MAX_DRAM_MB   32.0f /* 32MB per thread */
#define AISAQ_REARRANGED_PQ_FILE_PAGE_SIZE_DEFAULT    diskann::defaults::SECTOR_LEN
#define AISAQ_INVALID_VID                             0xffffffff


    enum aisaq_rearrange_sorter {
        __rearrange_sorter_opt_nhops = 1 << 8,
        __rearrange_sorter_opt_score = 1 << 9,
        __rearrange_sorter_opt_nnbrs = 1 << 10,

        aisaq_rearrange_sorter_nhops = 1 | __rearrange_sorter_opt_nhops,
        aisaq_rearrange_sorter_random = 2,
        aisaq_rearrange_sorter_nhops_score = 3 | __rearrange_sorter_opt_nhops | __rearrange_sorter_opt_score,
        aisaq_rearrange_sorter_nhops_nnbrs = 4 | __rearrange_sorter_opt_nhops | __rearrange_sorter_opt_nnbrs,
        aisaq_rearrange_sorter_nhops_nnbrs_score = 5 | __rearrange_sorter_opt_nhops | __rearrange_sorter_opt_nnbrs | __rearrange_sorter_opt_score,
        aisaq_rearrange_sorter_nhops_score_nnbrs = 6 | __rearrange_sorter_opt_nhops | __rearrange_sorter_opt_nnbrs | __rearrange_sorter_opt_score,
        aisaq_rearrange_sorter_default = aisaq_rearrange_sorter_nhops,
    };

    enum aisaq_pq_cache_policy {
        aisaq_pq_cache_policy_bfs = 0,
        aisaq_pq_cache_policy_direct = 1,
        aisaq_pq_cache_policy_auto = 100,

        aisaq_pq_cache_policy_default = aisaq_pq_cache_policy_auto,
    };

    enum aisaq_pq_io_engine {
        aisaq_pq_io_engine_aio = 0,

        aisaq_pq_io_engine_default = aisaq_pq_io_engine_aio,
    };

    struct aisaq_node_placement {
        uint32_t id;
        bool is_in_cache;
        char *ptr;
    };

    struct aisaq_search_config {
        uint32_t vector_beamwidth;
        enum aisaq_pq_io_engine pq_io_engine;
        uint64_t pq_cache_size;
        uint64_t pq_read_page_cache_size;
    };

    struct aisaq_rearranged_pq_compressed_vectors_file_header {
        uint32_t num_vectors;
        uint32_t vector_size;
        uint32_t page_size;
    };

}
