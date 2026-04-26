// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef PQ_FLASH_AISAQ_INDEX_H
#define PQ_FLASH_AISAQ_INDEX_H

#include "diskann/aisaq_neighbor_priority_queue.h"
#include "diskann/pq_flash_index.h"
#include "diskann/timer.h"

#pragma once
namespace diskann {
    template<typename T>
    struct AisaqThreadData{
        class AisaqPQReaderContext *aisaq_pq_reader_ctx = nullptr;
        uint32_t aisaq_max_read_nodes;
        std::vector<uint64_t> aisaq_scratch_mem_offset;
        NeighborPriorityQueue retset;
        std::vector<Neighbor> full_retset;
    };

class AisaqPQDataGetter: public PQDataGetter {
	uint32_t* _aisaq_rearranged_vectors_map;
	bool _rearranged_index;
    bool _rearrange_during_search;
    uint32_t _max_ios;
    class AisaqPQReader &_aisaq_pq_vectors_reader;
    class AisaqPQReaderContext &_ctx;

public:
	AisaqPQDataGetter(bool rearranged_index, bool rearrange_during_search, uint32_t* aisaq_rearranged_vectors_map, class AisaqPQReader &aisaq_pq_vectors_reader,
        class AisaqPQReaderContext &ctx, uint32_t max_ios) : _aisaq_pq_vectors_reader(aisaq_pq_vectors_reader), _ctx(ctx) { 
		_aisaq_rearranged_vectors_map = aisaq_rearranged_vectors_map;
		_rearranged_index = rearranged_index;
        _rearrange_during_search = rearrange_during_search;
        _max_ios = max_ios;
	}
    	virtual _u8* get_pq_data() {
		return nullptr;
	}
	virtual _u64 get_origin_id(_u64 id){
		return !_rearranged_index  ? id :  _aisaq_rearranged_vectors_map[(_u32)id];
	}
	virtual void release_pq_data(size_t offset=0, size_t size=0) {
	}

    virtual void aggregate_pq_coords(const unsigned* ids, const _u64 n_ids, const _u64 ndims, _u8* out);
};

template<typename T>
class PQFlashAisaqIndex: public PQFlashIndex<T> {
public:
    PQFlashAisaqIndex(std::shared_ptr<AlignedFileReader> fileReader,
                diskann::Metric metric = diskann::Metric::L2);
    // sector # on disk where node_id is present with in the graph part
    virtual ~PQFlashAisaqIndex();
    uint64_t get_node_sector(uint64_t node_id);

    // ptr to start of the node
    char *offset_to_node(char *sector_buf, uint64_t node_id);

    // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
    uint32_t *offset_to_node_nhood(char *node_buf);

    // returns region of `node_buf` containing [COORD(T)]
    T *offset_to_node_coords(char *node_buf);

    char *aisaq_offset_to_node_aisaq_data(char *node_buf);
    uint64_t aisaq_get_thread_data_size();
    uint64_t aisaq_cal_size();
    int aisaq_init(const diskann::aisaq_pq_io_engine pq_io_engine, const char *index_prefix);
    std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                                   std::vector<T *> &coord_buffers,
                                                   std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers,
                                                   std::vector<uint8_t *> *aisaq_buffers = nullptr);

    void aisaq_get_vector_by_ids(const int64_t *ids, const int64_t n,
                                 T *const output_data);
    bool aisaq_load_pq_cache(const std::string index_path,
                             uint64_t pq_cache_size_bytes, uint32_t policy, bool is_rearranged);
    bool get_rearranged_index();
    bool get_rearrange_during_search();
    int aisaq_load(uint32_t num_threads, const char *index_prefix);
    int aisaq_load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                       const char *pivots_filepath, const char *compressed_filepath);
    void use_medoids_data_as_centroids();
    
    uint8_t *aisaq_pq_cache_lookup(uint32_t id);

    void get_entry_point_medoid(uint32_t &best_medoid, float &best_dist,
                                float *pq_dists, float *dist_scratch,
                                float *query_float);
    
    void updata_io_stats(QueryStats &stats, size_t count, uint64_t size_in_sectors);
    
    void prepare_search_results(std::vector<Neighbor> &full_retset,
                                const uint64_t k_search,
                                float *distances, int64_t *indices,
                                float query_norm);
    
    void aisaq_cached_beam_search(
        const T *query, const uint64_t k_search, const uint64_t l_search, int64_t *res_ids,
        float *res_dists, const uint64_t beam_width,
        const bool use_reorder_data = false, QueryStats *stats = nullptr,
        const knowhere::feder::diskann::FederResultUniq &feder = nullptr,
        const knowhere::BitsetView bitset_view = nullptr,
        const float filter_ratio_in = -1.0f,
        const struct diskann::aisaq_search_config *aisaq_search_config = nullptr);
    
    void load_cache_list(std::vector<uint32_t> &node_list);
    
    void setup_thread_data(uint64_t nthreads);
    
    uint32_t get_max_node_len();
    
    bool should_ignore_point(uint32_t id, float alpha, float& accumulative_alpha, const knowhere::BitsetView& bitset);
    
    int aisaq_load_rearrange_data(const char *index_prefix);
    // asynchronously collect the access frequency of each node in the graph
    void aisaq_async_generate_cache_list_from_sample_queries(std::string sample_bin,
                                                       uint64_t        l_search,
                                                       uint64_t        beamwidth,
                                                       uint64_t num_nodes_to_cache);
    int aisaq_load_mediods_pq(uint32_t max_ios, AisaqPQReaderContext &ctx);
    int aisaq_load_entry_points_pq(uint32_t max_ios, std::string &entry_points_path, AisaqPQReaderContext &ctx);
    /* AiSAQ node cache addition */
    uint8_t *_aisaq_node_cache_buf = nullptr;
    tsl::robin_map<uint32_t, uint8_t *> _aisaq_node_cache;
    /* AiSAQ pq vectors reader */
    class AisaqPQReader *_aisaq_pq_vectors_reader = nullptr;
    /* AiSAQ static pq cache */
    uint8_t *_aisaq_pq_vectors_cache_buf = nullptr;
    tsl::robin_map<uint32_t, uint8_t *> _aisaq_pq_vectors_cache_map;
    uint64_t _aisaq_pq_vectors_cache_count = 0;
    bool _aisaq_pq_vectors_cache_direct = true;
    /* AiSAQ rearranged index */
    bool _aisaq_rearranged_index = false; /* indicate whether index is reordered */
    bool _aisaq_rearrange_vectors_during_search = false; /* indicate whether rearranged should be performed during search */
    std::unique_ptr<uint32_t[]> _aisaq_rearranged_vectors_map = nullptr; /* if _aisaq_rearranged_index is true the map will map new id --> origin id */
                                                                         /*if _aisaq_rearrange_vectors_during_search is true the map will map origin id --> new id */
    /* AiSAQ num of inline pq vectors */
    uint32_t _aisaq_inline_pq_vectors = 0;
    /* AiSAQ reader context size */
    uint32_t _aisaq_reader_context_size = 0;
    /* AiSAQ multiple entry points */
    std::unique_ptr<uint32_t[]> _aisaq_entry_points = nullptr;
    size_t _aisaq_num_entry_points = 0;
    uint8_t *_aisaq_entry_points_pq_vectors_buff = nullptr;
    /* medoids pq vectors */
    uint8_t *_aisaq_medoids_pq_vectors_buff = nullptr; 
    
    bool _reorder_data_exists = false;
    
    ConcurrentQueue<AisaqThreadData <T>> aisaq_thread_data;
    std::string index_prefix;
};
} // namespace diskann
#endif /* PQ_FLASH_AISAQ_INDEX_H */
