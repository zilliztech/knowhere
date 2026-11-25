// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef AISAQ_CONFIG_H
#define AISAQ_CONFIG_H

#include "../../../thirdparty/DiskANN/include/diskann/aisaq.h"
#include "diskann_config.h"

namespace knowhere {

class AisaqConfig : public DiskANNConfig {
 public:
    Status
    LoadSubConfig(const Json& json, PARAM_TYPE type, std::string* error) override {
        if (type == PARAM_TYPE::TRAIN && quant_type && *quant_type != "PQ")
            return HandleError(error, "AiSAQ requires PQ navigation", Status::invalid_args);
        return DiskANNConfig::LoadSubConfig(json, type, error);
    }
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

    KNOWHERE_DECLARE_CONFIG(AisaqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(search_cache_budget_gb_ratio)
            .description("the ratio of the size reserved for the search cache to the size of the raw data.")
            .set_default(0)
            .set_range(0, std::numeric_limits<CFG_FLOAT::value_type>::max())
            .for_train()
            .for_deserialize();
        // Block AiSAQ parameters

        KNOWHERE_CONFIG_DECLARE_FIELD(beamwidth)
            .description("the maximum number of IO requests each query will issue per iteration of search code.")
            .set_default(diskann::defaults::DEFAULT_AISAQ_BEAMWIDTH)
            .set_range(1, diskann::defaults::MAX_AISAQ_BEAMWIDTH)
            .for_search()
            .for_range_search()
            .for_iterator();
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

        // Search inherits the value established by Deserialize unless the request supplies one.
        KNOWHERE_CONFIG_DECLARE_FIELD(pq_read_page_cache_size)
            .allow_empty_without_default()
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

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        auto status = DiskANNConfig::CheckAndAdjust(param_type, err_msg);
        if (status != Status::success) {
            return status;
        }
        if ((param_type == PARAM_TYPE::TRAIN || param_type == PARAM_TYPE::DESERIALIZE) &&
            !pq_read_page_cache_size.has_value()) {
            pq_read_page_cache_size = diskann::defaults::DEFAULT_PQ_READ_PAGE_CACHE_SIZE;
        }
        if (param_type == PARAM_TYPE::TRAIN &&
            (inline_pq.value() > max_degree.value() ||
             max_degree.value() > static_cast<int>(diskann::defaults::MAX_AISAQ_MAX_DEGREE) ||
             search_list_size.value() > static_cast<int>(diskann::defaults::MAX_AISAQ_SEARCH_LIST_SIZE))) {
            return HandleError(err_msg, "AiSAQ graph parameters exceed supported limits", Status::out_of_range_in_json);
        }
        // AiSAQ always rejected a disk PQ wider than the vector; DiskANN's builder clamps it instead.
        if (param_type == PARAM_TYPE::TRAIN && dim.has_value() && disk_pq_dims.value_or(0) > dim.value()) {
            return HandleError(err_msg, "disk_pq_dims exceeds vector dimension", Status::out_of_range_in_json);
        }
        return Status::success;
    }
};
}  // namespace knowhere
#endif /* AISAQ_CONFIG_H */
