// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>    // For std::vector

#include "aisaq.h"

namespace diskann {

    template <typename T, typename LabelT>
    using aisaq_read_nodes_nbrs_func_t = std::vector<bool>(*)(void *param, const std::vector<uint32_t> &node_ids,
            std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers);
    /*
      generate vectors rearranging map using a specified sorter
      caller must implement node_nbrs reading function (read_nodes_nbrs_func)
      on success, rearranged_vectors_map is allocated, the caller is responsible on deleting it
      return 0 on success
     */
    template <typename T, typename LabelT>
    int aisaq_generate_vectors_rearrange_map(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<LabelT, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<T, LabelT> read_nodes_nbrs_func, void *context);

    /*
     create a reversed map
     on success, reversed_vectors_map is allocated, the caller is responsible on deleting it
     return 0 on success
     */
    int aisaq_create_reversed_vectors_map(uint32_t *&reversed_vectors_map,
            const uint32_t *vectors_map, uint32_t num_points);

    /*
     calculate the maximal number of pq vectors that can be stored inline as part of the index node
     */
    uint32_t aisaq_calc_max_inline_pq_vectors(uint32_t max_node_len, uint32_t pq_nbytes, uint32_t max_degree);

    int aisaq_rearrange_vectors_file(const std::string &file_path, const uint32_t *rearrange_map, uint32_t map_size);

    const char *aisaq_get_io_engine_string(enum aisaq_pq_io_engine io_engine);

    /*
     read pq vectors from unaligned pq compressed vectors file pq_compressed_vectors_path
     and write them into a new aligned pq compressed vectors file aligned_rearranged_pq_compressed_vectors_path
     the pq vectors are written with page alignment (page size) and are optionally rearranged
     (if rearrange_map is supplied)
     num_points and pq_vector_size are verified against the input file.
     rearrange_map if supplied, must be in size of num_points.
     return 0 on success
     */
    int aisaq_create_aligned_rearranged_pq_compressed_vectors_file(std::ifstream &pq_compressed_vectors_reader,
            const std::string &aligned_rearranged_pq_compressed_vectors_path,
            uint32_t page_size, uint32_t *rearrange_map, uint32_t num_points, uint32_t pq_vector_size);

    int aisaq_create_aligned_rearranged_pq_compressed_vectors_file(const std::string &pq_compressed_vectors_path,
            const std::string &aligned_rearranged_pq_compressed_vectors_path,
            uint32_t page_size, uint32_t *rearrange_map, uint32_t num_points, uint32_t pq_vector_size);

}
