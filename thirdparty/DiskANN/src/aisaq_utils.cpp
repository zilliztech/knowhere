// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#include "diskann/common_includes.h"

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include "gperftools/malloc_extension.h"
#endif

#ifdef _WINDOWS
#error "windows is not supported"
#endif

#include <algorithm> // For std::sort, std::find, etc.
#include "diskann/logger.h"
#include "diskann/utils.h"
#include "tsl/robin_set.h"

#include "diskann/defaults.h"
#include "diskann/aisaq_utils.h"

namespace diskann {

    /* todo: control rearrange logic at runtime */
    template <typename T, typename LabelT>
    int aisaq_generate_vectors_rearrange_map(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<LabelT, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<T, LabelT> read_nodes_nbrs_func, void *context) {
        if (num_points == 0 || num_medoids == 0) {
            return -1;
        }

        struct rearrange_node {
            uint32_t id;
            uint32_t score;
            std::vector<uint32_t> nbrs;
        };

        struct rearrange_nodes_sorter {

            static bool compare_by_nhops_score(rearrange_node &n1, rearrange_node &n2) {
                /* "nhops->score" */
                return n1.score < n2.score;
            }

            static bool compare_by_nhops_nnbrs(rearrange_node &n1, rearrange_node &n2) {
                /* "nhops->nnbrs" */
                return n1.nbrs.size() < n2.nbrs.size();
            }

            static bool compare_by_nhops_nnbrs_score(rearrange_node &n1, rearrange_node &n2) {
                /* "nhops->nnbrs->score" */
                return n1.nbrs.size() != n2.nbrs.size() ? n1.nbrs.size() < n2.nbrs.size() : n1.score < n2.score;
            }

            static bool compare_by_nhops_score_nnbrs(rearrange_node &n1, rearrange_node &n2) {
                /* "nhops->score->nnbrs" */
                return n1.score != n2.score ? n1.score < n2.score : n1.nbrs.size() < n2.nbrs.size();
            }

            void init(bool (*_compare_function)(rearrange_node &, rearrange_node &), const char *_name) {
                compare_function = _compare_function;
                name = _name;
            }
            bool (*compare_function)(rearrange_node &, rearrange_node &) = nullptr;
            const char *name = "";
        };
        struct rearrange_nodes_sorter nodes_sorter;
        switch (rearrange_sorter) {
            case aisaq_rearrange_sorter_nhops:
                nodes_sorter.init(nullptr, "nhops");
                break;
            case aisaq_rearrange_sorter_random:
                nodes_sorter.init(nullptr, "nhops->random");
                break;
            case aisaq_rearrange_sorter_nhops_score:
                nodes_sorter.init(rearrange_nodes_sorter::compare_by_nhops_score, "nhops->score");
                break;
            case aisaq_rearrange_sorter_nhops_nnbrs:
                nodes_sorter.init(rearrange_nodes_sorter::compare_by_nhops_nnbrs, "nhops->nnbrs");
                break;
            case aisaq_rearrange_sorter_nhops_nnbrs_score:
                nodes_sorter.init(rearrange_nodes_sorter::compare_by_nhops_nnbrs_score, "nhops->nnbrs->score");
                break;
            case aisaq_rearrange_sorter_nhops_score_nnbrs:
                nodes_sorter.init(rearrange_nodes_sorter::compare_by_nhops_score_nnbrs, "nhops->score_nnbrs");
                break;
            default:
                LOG_KNOWHERE_ERROR_ << "invalid rearrange sorter";
                return -1;
        }

        rearranged_vectors_map = new uint32_t[num_points];
        if (rearranged_vectors_map == nullptr) {
            LOG_KNOWHERE_ERROR_ << "memory allocation failure";
            return -1;
        }
        /* init */
        for (uint32_t i = 0; i < num_points; i++) {
            rearranged_vectors_map[i] = AISAQ_INVALID_VID;
        }

        LOG_KNOWHERE_INFO_ << "generating vectors rearrange mapping data (using sorter: "
                << nodes_sorter.name << ")...";
        std::unique_ptr<tsl::robin_set < uint32_t>> cur_level;
        cur_level = std::make_unique<tsl::robin_set < uint32_t >> ();
        uint32_t vid_hover = 0, prev_vid_hover = 0, nhops = 1;
        for (uint32_t i = 0; i < num_medoids; i++) {
            cur_level->insert(medoids[i]);
        }
        if (filter_to_medoid_ids.size() > 0) {
            for (auto &x : filter_to_medoid_ids) {
                for (auto &y : x.second) {
                    cur_level->insert(y);
                }
            }
        }
        for (const uint32_t &id : *cur_level) {
            if (rearranged_vectors_map[id] == AISAQ_INVALID_VID) {
                rearranged_vectors_map[id] = vid_hover++;
            }
        }
        uint64_t block_size = 1024;
        std::vector<uint32_t> nodes_to_read;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
        for (uint32_t i = 0; i < block_size; i++) {
            nbr_buffers.emplace_back(0, new uint32_t[max_degree]);
        }
        while (cur_level->size() != 0) {
            diskann::cout << "nhops: " << nhops << "..." << std::flush;
            /* expand cur_level */
            std::vector<struct rearrange_node> rnodes;
            uint64_t ncount = cur_level->size();
            uint64_t nblocks = DIV_ROUND_UP(ncount, block_size);
            auto vector_it = cur_level->begin();
            uint32_t progress_step = std::max(1lu, nblocks / 20);
            for (size_t block = 0; block < nblocks; block++) {
                if ((block % progress_step) == 0) {
                    diskann::cout << "." << std::flush;
                }
                size_t start = block * block_size;
                size_t end = (std::min)((block + 1) * block_size, ncount);
                for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
                    nodes_to_read.push_back(*vector_it);
                    vector_it++;
                }
                /* issue read requests */
                auto read_status = read_nodes_nbrs_func(context, nodes_to_read, nbr_buffers);
                for (uint32_t i = 0; i < read_status.size(); i++) {
                    if (read_status[i] == false) {
                        LOG_KNOWHERE_ERROR_ << "read failed";
                        continue;
                    }
                    uint32_t nnbrs = nbr_buffers[i].first;
                    uint32_t *nbrs = nbr_buffers[i].second;
                    struct rearrange_node rnode;
                    rnode.id = nodes_to_read[i];
                    /* next level */
                    for (uint32_t j = 0; j < nnbrs; j++) {
                        rnode.nbrs.push_back(nbrs[j]);
                    }
                    //std::sort(rnode.nbrs.begin(), rnode.nbrs.end());
                    rnode.score = 0;
                    if ((rearrange_sorter & __rearrange_sorter_opt_score) != 0) {
                        assert(pq_vector_bytes > 0);
                        /* calculate score */
                        uint32_t __sector_x1, __sector_x2, __sector_x4, last_sector_x1, last_sector_x2, last_sector_x4, last_score = 0;
                        uint32_t __vectors_per_io_x1 = defaults::SECTOR_LEN / (pq_vector_bytes * sizeof (uint8_t)),
                                __vectors_per_io_x2 = (defaults::SECTOR_LEN * 2) / (pq_vector_bytes * sizeof (uint8_t)),
                                __vectors_per_io_x4 = (defaults::SECTOR_LEN * 4) / (pq_vector_bytes * sizeof (uint8_t));
                        for (uint32_t j = 0; j < rnode.nbrs.size(); j++) {
                            __sector_x1 = rnode.nbrs[j] / __vectors_per_io_x1;
                            __sector_x2 = rnode.nbrs[j] / __vectors_per_io_x2;
                            __sector_x4 = rnode.nbrs[j] / __vectors_per_io_x4;
                            if (j == 0 || (last_sector_x1 != __sector_x1 && last_sector_x2 != __sector_x2 && last_sector_x4 != __sector_x4)) {
                                last_sector_x1 = __sector_x1;
                                last_sector_x2 = __sector_x2;
                                last_sector_x4 = __sector_x4;
                                last_score = 0;
                            } else {
                                if (last_sector_x1 == __sector_x1) {
                                    last_score += 4;
                                } else if (last_sector_x2 == __sector_x2) {
                                    last_score += 2;
                                } else {
                                    /* last_sector_x4 == __sector_x4 */
                                    last_score++;
                                }
                            }
                            if (last_score > rnode.score) {
                                rnode.score = last_score;
                            }
                        }
                    }
                    rnodes.push_back(rnode);
                }
                nodes_to_read.clear();
            }
            if (nodes_sorter.compare_function != nullptr) {
                /* we wish to keep rnodes ordered by distance_score, highest first
                   sort in descending order.
                   it is much faster to push back all items and then sort. */
                std::sort(rnodes.rbegin(), rnodes.rend(), nodes_sorter.compare_function);
            } else if (rearrange_sorter == aisaq_rearrange_sorter_random) {
                std::random_device rng;
                std::mt19937 urng(rng());
                std::shuffle(rnodes.begin(), rnodes.end(), urng);
            }
            cur_level->clear();
            /* assign new ids and update next level */
            for (uint32_t i = 0; i < rnodes.size(); i++) {
                const struct rearrange_node &rnode = rnodes[i];
                for (const uint32_t &id : rnode.nbrs) {
                    if (rearranged_vectors_map[id] == AISAQ_INVALID_VID) {
                        rearranged_vectors_map[id] = vid_hover++;
                        cur_level->insert(id);
                    }
                }
            }
            rnodes.clear();
            LOG_KNOWHERE_INFO_ << "... +" << vid_hover - prev_vid_hover << " vectors"
                    << " --> " << vid_hover << " vectors";
            prev_vid_hover = vid_hover;
            nhops++;
        }
        for (uint32_t i = 0; i < block_size; i++) {
            delete [] nbr_buffers[i].second;
        }
        if (vid_hover < num_points) {
            LOG_KNOWHERE_INFO_ << num_points - vid_hover << " unreferenced vectors";
            for (uint32_t i = 0; i < num_points; i++) {
                if (rearranged_vectors_map[i] == AISAQ_INVALID_VID) {
                    rearranged_vectors_map[i] = vid_hover++;
                }
            }
        }
        return 0;
    }

    int aisaq_create_reversed_vectors_map(uint32_t *&reversed_vectors_map, const uint32_t *vectors_map, uint32_t num_points) {
        reversed_vectors_map = new uint32_t[num_points];
        if (reversed_vectors_map == nullptr) {
            LOG_KNOWHERE_ERROR_ << "failed to allocate memory.";
            return -1;
        }
        uint32_t i;
        for (i = 0; i < num_points; i++) {
            reversed_vectors_map[i] = AISAQ_INVALID_VID;
        }
        for (i = 0; i < num_points; i++) {
            uint32_t rid = vectors_map[i];
            if (reversed_vectors_map[rid] != AISAQ_INVALID_VID) {
                break;
            }
            reversed_vectors_map[rid] = i;
        }
        if (i < num_points) {
            delete [] reversed_vectors_map;
            reversed_vectors_map = nullptr;
            LOG_KNOWHERE_ERROR_ << "rearranged vectors map error";
            return -1;
        }
        return 0;
    }

    uint32_t aisaq_calc_max_inline_pq_vectors(uint32_t max_node_len, uint32_t pq_nbytes, uint32_t max_degree) {
        if (max_node_len >= defaults::SECTOR_LEN) {
            /* node size >= 4KiB */
            uint32_t _disk_space = ROUND_UP(max_node_len, defaults::SECTOR_LEN);
            return std::min(max_degree, (uint32_t) ((_disk_space - max_node_len) / pq_nbytes));
        }
        /* node size < 4KiB */
        uint32_t _nnodes = defaults::SECTOR_LEN / max_node_len;
        return std::min(max_degree, (uint32_t) ((defaults::SECTOR_LEN - (_nnodes * max_node_len)) / (_nnodes * pq_nbytes)));
    }

    int aisaq_create_aligned_rearranged_pq_compressed_vectors_file(std::ifstream &pq_compressed_vectors_reader,
            const std::string &aligned_rearranged_pq_compressed_vectors_path,
            uint32_t page_size, uint32_t *rearrange_map, uint32_t num_points, uint32_t pq_vector_size) {
        diskann::cout << "generating aligned rearranged pq compressed vectors file (page size is "
                << (page_size >> 10) << "KiB)..." << std::flush;
        do {
            /* generate pq compressed rearranged vectors file */
            uint8_t *page_buff = nullptr;
            diskann::alloc_aligned((void **) &page_buff,
                    page_size,
                    defaults::SECTOR_LEN);
            if (page_buff == nullptr) {
                LOG_KNOWHERE_ERROR_ << "failed to allocate memory.";
                break;
            }
            bool file_error = false;
            /* use open api since the file must be open in direct mode with no fs cache */
            do {
                int writer_fd = open(aligned_rearranged_pq_compressed_vectors_path.c_str(),
                        O_CREAT | O_WRONLY | O_DIRECT,
                        S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
                if (writer_fd <= 0) {
                    file_error = true;
                    break;
                }
                /* write the header */
                memset(page_buff, 0, page_size);
                struct aisaq_rearranged_pq_compressed_vectors_file_header *file_header =
                        (struct aisaq_rearranged_pq_compressed_vectors_file_header *) page_buff;
                file_header->num_vectors = num_points;
                file_header->vector_size = pq_vector_size;
                file_header->page_size = page_size;
                if (write(writer_fd, file_header, defaults::SECTOR_LEN) != defaults::SECTOR_LEN) {
                    close(writer_fd);
                    file_error = true;
                    break;
                }
                /* prepare and write one page at a time */
                uint32_t pq_vectors_per_page = page_size / pq_vector_size;
                uint32_t id = 0, nid, vec_in_page = 0;
                uint32_t progress_step = std::max((uint32_t) 1, num_points / 100);
                while (!file_error && id < num_points) {
                    if ((id % progress_step) == 0) {
                        diskann::cout << "." << std::flush;
                    }
                    /* read original vector id and write it in nid */
                    nid = rearrange_map == nullptr ? id : rearrange_map[id];
                    if (nid >= num_points) {
                        memset((char *) page_buff + (vec_in_page * pq_vector_size), 0, pq_vector_size);
                    } else {
                        pq_compressed_vectors_reader.seekg((sizeof (uint32_t) * 2) + ((uint64_t) nid * pq_vector_size),
                                pq_compressed_vectors_reader.beg);
                        pq_compressed_vectors_reader.read((char *) page_buff + (vec_in_page * pq_vector_size),
                                pq_vector_size);
                    }
                    vec_in_page++;
                    id++;
                    if (vec_in_page == pq_vectors_per_page || id == num_points) {
                        uint32_t remain = page_size - (vec_in_page * pq_vector_size);
                        if (remain > 0) {
                            memset((char *) page_buff + (vec_in_page * pq_vector_size), 0, remain);
                        }
                        if (write(writer_fd, (char *) page_buff, page_size) != page_size) {
                            file_error = true;
                            break;
                        }
                        vec_in_page = 0;
                    }
                };
                close(writer_fd);
            } while (false);
            diskann::aligned_free((void *) page_buff);
            if (file_error) {
                LOG_KNOWHERE_ERROR_ << "failed to open/write pq compressed rearranged vectors file";
                break;
            }
            LOG_KNOWHERE_INFO_ << "...done";
            return 0;
        } while (false);
        return -1;

    }

    int aisaq_create_aligned_rearranged_pq_compressed_vectors_file(const std::string &pq_compressed_vectors_path,
            const std::string &aligned_rearranged_pq_compressed_vectors_path,
            uint32_t page_size, uint32_t *rearrange_map, uint32_t num_points, uint32_t pq_vector_size) {
        uint64_t file_size = get_file_size(pq_compressed_vectors_path);
        std::ifstream pq_compressed_vectors_reader;
        pq_compressed_vectors_reader.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        uint32_t _num_points, _pq_vector_size;
        try {
            pq_compressed_vectors_reader.open(pq_compressed_vectors_path, std::ios::binary);
            pq_compressed_vectors_reader.read((char *) &_num_points, sizeof (uint32_t));
            pq_compressed_vectors_reader.read((char *) &_pq_vector_size, sizeof (uint32_t));
        } catch (std::system_error &e) {
            LOG_KNOWHERE_ERROR_ << "failed to open/read header of pq compressed vectors file " << pq_compressed_vectors_path;
            return -1;
        }
        uint64_t expected_file_size = ((uint64_t) _num_points * _pq_vector_size) + (sizeof (uint32_t) * 2);
        if (num_points != _num_points || pq_vector_size != _pq_vector_size || file_size != expected_file_size) {
            LOG_KNOWHERE_ERROR_ << "corrupted pq compressed vectors file " << pq_compressed_vectors_path;
            return -1;
        }
        int ret = aisaq_create_aligned_rearranged_pq_compressed_vectors_file(pq_compressed_vectors_reader,
                aligned_rearranged_pq_compressed_vectors_path,
                page_size, rearrange_map, num_points, pq_vector_size);
        //pq_compressed_vectors_reader.close();
        return ret;
    }

    int aisaq_rearrange_vectors_file(const std::string &file_path, const uint32_t *rearrange_map, uint32_t map_size) {
        if (!file_exists(file_path)) {
            return -1;
        }
        std::unique_ptr < uint32_t[] > data = nullptr;
        size_t npts, dim;
        diskann::load_bin<uint32_t>(file_path, data, npts, dim);
        assert(dim == 1);
        for (unsigned int i = 0; i < npts; i++) {
            assert(data[i] < map_size);
            data[i] = rearrange_map[data[i]];
        }
        diskann::save_bin<uint32_t>(file_path, data.get(), npts, dim);
        return 0;
    }

    const char *aisaq_get_io_engine_string(enum aisaq_pq_io_engine io_engine) {
        switch (io_engine) {
            case aisaq_pq_io_engine_aio:
                return "aio";
            default:
                break;
        }
        return "unknown";
    }

    /* instatiations */
    template int aisaq_generate_vectors_rearrange_map<float, uint32_t>(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<uint32_t, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<float, uint32_t> read_nodes_nbrs_func, void *context);
    template int aisaq_generate_vectors_rearrange_map<int8_t, uint32_t>(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<uint32_t, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<int8_t, uint32_t> read_nodes_nbrs_func, void *context);
    template int aisaq_generate_vectors_rearrange_map<uint8_t, uint32_t>(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<uint32_t, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<uint8_t, uint32_t> read_nodes_nbrs_func, void *context);
    template int aisaq_generate_vectors_rearrange_map<float, uint16_t>(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<uint16_t, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<float, uint16_t> read_nodes_nbrs_func, void *context);
    template int aisaq_generate_vectors_rearrange_map<int8_t, uint16_t>(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<uint16_t, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<int8_t, uint16_t> read_nodes_nbrs_func, void *context);
    template int aisaq_generate_vectors_rearrange_map<uint8_t, uint16_t>(enum aisaq_rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<uint16_t, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<uint8_t, uint16_t> read_nodes_nbrs_func, void *context);
    template int aisaq_generate_vectors_rearrange_map<knowhere::bf16, uint32_t>(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<uint32_t, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<float, uint32_t> read_nodes_nbrs_func, void *context);
    template int aisaq_generate_vectors_rearrange_map<knowhere::fp16, uint32_t>(enum aisaq_rearrange_sorter rearrange_sorter, uint32_t *&rearranged_vectors_map,
            uint32_t num_points, uint32_t pq_vector_bytes, uint32_t max_degree, const uint32_t *medoids, uint32_t num_medoids,
            std::unordered_map<uint32_t, std::vector<uint32_t>> &filter_to_medoid_ids,
            aisaq_read_nodes_nbrs_func_t<int8_t, uint32_t> read_nodes_nbrs_func, void *context);

}
