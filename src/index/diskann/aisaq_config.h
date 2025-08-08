// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef AISAQ_CONFIG_H
#define AISAQ_CONFIG_H

#include "../../../thirdparty/DiskANN/include/diskann/aisaq.h"
#include "diskann_config.h"

namespace knowhere {

class AisaqConfig : public DiskANNConfig {
 public:
    // Block AiSAQ parameters
    // PQ vector beam width
    CFG_INT vectors_beamwidth;
    // enable compressed vectors to be stored in-line within the node, the number of in-line vectors is limited by max
    // degree
    CFG_INT inline_pq;
    // compressed vectors cache DRAM size in bytes, default 0
    CFG_INT pq_cache_size;
    // enable compressed vectors reordering search optimization, default false
    CFG_BOOL rearrange;
    // enable compressed vectors read-page cache DRAM size per thread, default 0
    CFG_INT pq_read_page_cache_size;
    // pq vectors read io engine to use, one of {aio, uring} valid only with aisaq option.
    CFG_STRING pq_read_io_engine;
    // number of entry points valid only with aisaq option.
    CFG_INT num_entry_points;

    KNOHWERE_DECLARE_CONFIG(AisaqConfig) {
        // Block AiSAQ parameters

        KNOWHERE_CONFIG_DECLARE_FIELD(vectors_beamwidth)
            .set_default(1)
            .set_range(1, diskann::defaults::MAX_AISAQ_VECTORS_BEAMWIDTH)
            .description("Beam width of the compressed vectors")
            .for_search()
            .for_range_search()
            .for_iterator();

        KNOWHERE_CONFIG_DECLARE_FIELD(inline_pq)
            .set_default(diskann::defaults::DEFAULT_INLINE_PQ)
            .set_range(diskann::defaults::DEFAULT_INLINE_PQ, 2048)
            .description(
                "Enable compressed vectors to be stored in-line within the node, the number of in-line vectors is "
                "limited by max degree")
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(pq_cache_size)
            .set_default(diskann::defaults::DEFAULT_PQ_CACHE_SIZE)
            .set_range(diskann::defaults::MIN_PQ_CACHE_SIZE, diskann::defaults::MAX_PQ_CACHE_SIZE)
            .description("Compressed vectors cache DRAM size in bytes")
            .for_train()
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(pq_read_page_cache_size)
            .set_default(diskann::defaults::DEFAULT_PQ_READ_PAGE_CACHE_SIZE)
            .set_range(diskann::defaults::MIN_PQ_READ_PAGE_CACHE_SIZE, diskann::defaults::MAX_PQ_READ_PAGE_CACHE_SIZE)
            .description("compressed vectors read-page cache DRAM size per thread")
            .for_train()
            .for_deserialize()
            .for_search()
            .for_range_search()
            .for_iterator();

        KNOWHERE_CONFIG_DECLARE_FIELD(rearrange)
            .set_default(diskann::defaults::DEFAULT_REARRANGE)
            .description("Enable compressed vectors reordering search optimization")
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(pq_read_io_engine)
            .set_default("aio")
            .description("pq vectors read io engine to use, one of {aio, uring}")
            .for_train()
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(num_entry_points)
            .set_default(diskann::defaults::DEFAULT_NUM_ENTRY_POINTS)
            .set_range(diskann::defaults::MIN_NUM_ENTRY_POINTS, diskann::defaults::MAX_NUM_ENTRY_POINTS)
            .description("number of entry points")
            .for_train();
    }
};
}  // namespace knowhere
#endif /* AISAQ_CONFIG_H */
