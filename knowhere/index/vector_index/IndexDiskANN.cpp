// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "index/vector_index/IndexDiskANN.h"

#include <math.h>
#include <omp.h>

#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DiskANN/include/aux_utils.h"
#include "DiskANN/include/utils.h"
#ifndef _WINDOWS
#include "DiskANN/include/linux_aligned_file_reader.h"
#else
#include "DiskANN/include/windows_aligned_file_reader.h"
#endif
#include "knowhere/common/Exception.h"
#include "knowhere/feder/DiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/RangeUtil.h"

namespace knowhere {

template <typename T>
IndexDiskANN<T>::IndexDiskANN(std::string index_prefix, MetricType metric_type,
                              std::shared_ptr<FileManager> file_manager)
    : index_prefix_(index_prefix), file_manager_(file_manager) {
    index_type_ = IndexEnum::INDEX_DISKANN;

    if (metric_type == metric::L2) {
        metric_ = diskann::L2;
    } else if (metric_type == metric::IP) {
        if (!std::is_same_v<T, float>) {
            KNOWHERE_THROW_MSG(
                "DiskANN currently only supports floating point data for Max "
                "Inner Product Search. ");
        }
        metric_ = diskann::INNER_PRODUCT;
    } else {
        KNOWHERE_THROW_MSG("DiskANN only support L2 and IP distance.");
    }
}

namespace {
static constexpr float kCacheExpansionRate = 1.2;
static constexpr uint32_t kLinuxAioMaxnrLimit = 65536;
void
CheckPreparation(bool is_prepared) {
    if (!is_prepared) {
        KNOWHERE_THROW_MSG("DiskANN is not prepared yet, plz call Prepare() to make it ready for queries.");
    }
}

std::vector<std::string>
GetNecessaryFilenames(const std::string& prefix, const bool is_inner_product, const bool use_sample_cache,
                      const bool use_sample_warmup) {
    std::vector<std::string> filenames;
    auto pq_pivots_filename = diskann::get_pq_pivots_filename(prefix);
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);

    filenames.push_back(pq_pivots_filename);
    filenames.push_back(diskann::get_pq_rearrangement_perm_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_chunk_offsets_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_centroid_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_compressed_filename(prefix));
    filenames.push_back(disk_index_filename);
    if (is_inner_product) {
        filenames.push_back(diskann::get_disk_index_max_base_norm_file(disk_index_filename));
    }
    if (use_sample_cache || use_sample_warmup) {
        filenames.push_back(diskann::get_sample_data_filename(prefix));
    }
    return filenames;
}

std::vector<std::string>
GetOptionalFilenames(const std::string& prefix) {
    std::vector<std::string> filenames;
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);
    filenames.push_back(diskann::get_disk_index_centroids_filename(disk_index_filename));
    filenames.push_back(diskann::get_disk_index_medoids_filename(disk_index_filename));
    return filenames;
}

template <typename T>
void
CheckDataFile(const std::string& data_path, const size_t num, const size_t dim) {
    std::ifstream file(data_path.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        KNOWHERE_THROW_MSG("Raw data path is not found");
    }
    uint64_t autual_file_size = static_cast<uint64_t>(file.tellg());
    file.close();
    uint64_t expected_file_size = num * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (autual_file_size != expected_file_size) {
        KNOWHERE_THROW_FORMAT("Actual file size (%ld bytes) not equal to expected size (%ld bytes)", autual_file_size,
                              expected_file_size);
    }
}

bool
AnyIndexFileExist(const std::string& index_prefix) {
    auto file_exist = [&index_prefix](std::vector<std::string> filenames) -> bool {
        for (auto& filename : filenames) {
            if (file_exists(filename)) {
                return true;
            }
        }
        return false;
    };
    auto is_exist = file_exist(GetNecessaryFilenames(index_prefix, diskann::INNER_PRODUCT, true, true));
    is_exist = is_exist | file_exist(GetOptionalFilenames(index_prefix));
    return is_exist;
}

template <typename T>
std::optional<T>
TryDiskANNCall(std::function<T()>&& diskann_call) {
    try {
        return std::make_optional<T>(diskann_call());
    } catch (const diskann::FileException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN File Exception: " << e.what();
    } catch (const diskann::ANNException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Exception: " << e.what();
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Other Exception: " << e.what();
    }
    return std::nullopt;
}

template <typename T>
T
TryDiskANNCallAndThrow(std::function<T()>&& diskann_call) {
    try {
        return diskann_call();
    } catch (const diskann::FileException& e) {
        KNOWHERE_THROW_MSG("DiskANN File Exception: " + std::string(e.what()));
    } catch (const diskann::ANNException& e) {
        KNOWHERE_THROW_MSG("DiskANN Exception: " + std::string(e.what()));
    } catch (const std::exception& e) {
        KNOWHERE_THROW_MSG("DiskANN Other Exception: " + std::string(e.what()));
    }
}

}  // namespace

template <typename T>
void
IndexDiskANN<T>::AddWithoutIds(const DatasetPtr& data_set, const Config& config) {
    std::lock_guard<std::mutex> lock(preparation_lock_);
    auto build_conf = DiskANNBuildConfig::Get(config);
    auto& data_path = build_conf.data_path;
    // Load raw data
    if (!LoadFile(data_path)) {
        KNOWHERE_THROW_MSG("Failed load the raw data before building.");
    }

    size_t count;
    size_t dim;
    diskann::get_bin_metadata(build_conf.data_path, count, dim);
    CheckDataFile<T>(data_path, count, dim);

    count_.store(count);
    dim_.store(dim);

    if (AnyIndexFileExist(index_prefix_)) {
        KNOWHERE_THROW_MSG("This index prefix already has index files.");
    }

    diskann::BuildConfig diskann_internal_build_config{data_path,
                                                       index_prefix_,
                                                       metric_,
                                                       build_conf.max_degree,
                                                       build_conf.search_list_size,
                                                       build_conf.pq_code_budget_gb,
                                                       build_conf.build_dram_budget_gb,
                                                       build_conf.num_threads,
                                                       build_conf.disk_pq_dims,
                                                       false,
                                                       build_conf.accelerate_build};

    auto build_successful = TryDiskANNCallAndThrow<int>(
        [&]() -> int { return diskann::build_disk_index<T>(diskann_internal_build_config); });

    if (build_successful != 0) {
        KNOWHERE_THROW_MSG("Failed to build DiskANN.");
    }

    // Add file to the file manager
    for (auto& filename : GetNecessaryFilenames(index_prefix_, metric_ == diskann::INNER_PRODUCT, true, true)) {
        if (!AddFile(filename)) {
            KNOWHERE_THROW_MSG("Failed to add file " + filename + ".");
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        if (file_exists(filename) && !AddFile(filename)) {
            KNOWHERE_THROW_MSG("Failed to add file " + filename + ".");
        }
    }

    is_prepared_.store(false);
}

template <typename T>
bool
IndexDiskANN<T>::Prepare(const Config& config) {
    std::lock_guard<std::mutex> lock(preparation_lock_);

    auto prep_conf = DiskANNPrepareConfig::Get(config);
    if (is_prepared_.load()) {
        return true;
    }

    // Load file from file manager.
    for (auto& filename :
         GetNecessaryFilenames(index_prefix_, metric_ == diskann::INNER_PRODUCT,
                               prep_conf.search_cache_budget_gb > 0 && !prep_conf.use_bfs_cache, prep_conf.warm_up)) {
        if (!LoadFile(filename)) {
            return false;
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        auto is_exist_op = file_manager_->IsExisted(filename);
        if (!is_exist_op.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to check existence of file " << filename << ".";
            return false;
        }
        if (is_exist_op.value() && !LoadFile(filename)) {
            return false;
        }
    }

    // set thread pool
    auto num_thread_max_value = kLinuxAioMaxnrLimit / prep_conf.aio_maxnr;
    pool_ = ThreadPool::GetGlobalThreadPool();

    // The number of threads used for preparing and searching. Threads run in parallel and one thread handles one query
    // at a time. More threads will result in higher aggregate query throughput, but will also use more IOs/second
    // across the system, which may lead to higher per-query latency. So find the balance depending on the maximum
    // number of IOPs supported by the SSD.
    if (num_thread_max_value < pool_->size()) {
        LOG_KNOWHERE_ERROR_ << "The global thread pool is to large for DiskANN. Expected max: " << num_thread_max_value
                            << ", actual: " << pool_->size();
        return false;
    }

    // load PQ file
    LOG_KNOWHERE_INFO_ << "Loading index from disk.";
    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new LinuxAlignedFileReader(prep_conf.aio_maxnr));
#endif

    pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<T>>(reader, metric_);

    auto load_successful = TryDiskANNCall<int>(
        [&]() -> int { return pq_flash_index_->load(prep_conf.num_threads, index_prefix_.c_str()); });

    if (!load_successful.has_value() || load_successful.value() != 0) {
        LOG_KNOWHERE_ERROR_ << "Failed to load DiskANN.";
        return false;
    }

    count_.store(pq_flash_index_->get_num_points());
    // DiskANN will add one more dim for IP type.
    if (metric_ == diskann::Metric::INNER_PRODUCT) {
        dim_.store(pq_flash_index_->get_data_dim() - 1);
    } else {
        dim_.store(pq_flash_index_->get_data_dim());
    }

    std::string warmup_query_file = diskann::get_sample_data_filename(index_prefix_);
    // load cache
    auto num_nodes_to_cache = GetCachedNodeNum(prep_conf.search_cache_budget_gb, pq_flash_index_->get_data_dim(),
                                               pq_flash_index_->get_max_degree());
    if (num_nodes_to_cache > pq_flash_index_->get_num_points() / 3) {
        KNOWHERE_THROW_MSG("Failed to generate cache, num_nodes_to_cache is larger than 1/3 of the total data number.");
    }
    if (num_nodes_to_cache > 0) {
        LOG_KNOWHERE_INFO_ << "Caching " << num_nodes_to_cache << " sample nodes around medoid(s).";
        if (prep_conf.use_bfs_cache) {
            std::vector<uint32_t> node_list;
            auto gen_cache_successful = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->cache_bfs_levels(num_nodes_to_cache, node_list);
                return true;
            });

            if (!gen_cache_successful.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to generate bfs cache for DiskANN.";
                return false;
            }
            auto load_cache_successful = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->load_cache_list(node_list);
                return true;
            });

            if (!load_cache_successful.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to load cache for DiskANN.";
                return false;
            }
        } else {
            // init the statistical object
            pq_flash_index_->async_generate_cache_list_from_sample_queries(warmup_query_file, 15, 6,
                                                                           num_nodes_to_cache);
        }
    }
    // warmup
    if (prep_conf.warm_up) {
        LOG_KNOWHERE_DEBUG_ << "Warming up.";
        uint64_t warmup_L = 20;
        uint64_t warmup_num = 0;
        uint64_t warmup_dim = 0;
        uint64_t warmup_aligned_dim = 0;
        T* warmup = nullptr;
        auto load_successful = TryDiskANNCall<bool>([&]() -> bool {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
            return true;
        });
        if (!load_successful.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to load warmup file for DiskANN.";
            return false;
        }
        std::vector<int64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

        bool all_searches_are_good = true;
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t)warmup_num; ++i) {
            auto search_successful = TryDiskANNCall<bool>([&]() -> bool {
                pq_flash_index_->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                                    warmup_result_ids_64.data() + (i * 1),
                                                    warmup_result_dists.data() + (i * 1), 4);
                return true;
            });
            if (!search_successful.has_value()) {
                all_searches_are_good = false;
            }
        }

        if (warmup != nullptr) {
            diskann::aligned_free(warmup);
        }

        if (!all_searches_are_good) {
            LOG_KNOWHERE_ERROR_ << "Failed to do search on warmup file for DiskANN.";
            return false;
        }
    }

    is_prepared_.store(true);
    LOG_KNOWHERE_INFO_ << "End of diskann preparation.";
    return true;
}

template <typename T>

DatasetPtr
IndexDiskANN<T>::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    CheckPreparation(is_prepared_.load());

    auto query_conf = DiskANNQueryConfig::Get(config);
    auto& k = query_conf.k;

    GET_TENSOR_DATA_DIM(dataset_ptr);
    auto query = static_cast<const T*>(p_data);
    auto p_id = new int64_t[k * rows];
    auto p_dist = new float[k * rows];

    feder::diskann::FederResultUniq feder_result;
    if (CheckKeyInConfig(config, meta::TRACE_VISIT) && GetMetaTraceVisit(config)) {
        if (rows != 1) {
            delete[] p_id;
            delete[] p_dist;
            KNOWHERE_THROW_MSG("NQ must be 1 when Feder tracing");
        }
        feder_result = std::make_unique<feder::diskann::FederResult>();
        feder_result->visit_info_.SetQueryConfig(query_conf);
    }

    std::vector<std::future<void>> futures;
    futures.reserve(rows);
    for (int64_t row = 0; row < rows; ++row) {
        futures.push_back(pool_->push([&, index = row]() {
            pq_flash_index_->cached_beam_search(query + (index * dim), k, query_conf.search_list_size,
                                                p_id + (index * k), p_dist + (index * k), query_conf.beamwidth, false,
                                                nullptr, nullptr, bitset, query_conf.filter_threshold);
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    // set visit_info json string into result dataset
    if (feder_result != nullptr) {
        Config json_visit_info, json_id_set;
        nlohmann::to_json(json_visit_info, feder_result->visit_info_);
        nlohmann::to_json(json_id_set, feder_result->id_set_);
        return GenResultDataset(p_id, p_dist, json_visit_info.dump(), json_id_set.dump());
    }

    return GenResultDataset(p_id, p_dist);
}

template <typename T>
DatasetPtr
IndexDiskANN<T>::QueryByRange(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    CheckPreparation(is_prepared_.load());

    auto query_conf = DiskANNQueryByRangeConfig::Get(config);
    auto radius = query_conf.radius;
    auto range_filter_exist = query_conf.range_filter_exist;
    auto range_filter = query_conf.range_filter;
    bool is_ip = (pq_flash_index_->get_metric() == diskann::Metric::INNER_PRODUCT);

    GET_TENSOR_DATA_DIM(dataset_ptr);
    auto query = static_cast<const T*>(p_data);

    std::vector<std::vector<int64_t>> result_id_array(rows);
    std::vector<std::vector<float>> result_dist_array(rows);

    std::vector<std::future<void>> futures;
    futures.reserve(rows);
    for (int64_t row = 0; row < rows; ++row) {
        futures.push_back(pool_->push([&, index = row]() {
            std::vector<int64_t> indices;
            std::vector<float> distances;

            auto res_count = pq_flash_index_->range_search(
                query + (index * dim), radius, query_conf.min_k, query_conf.max_k, result_id_array[index],
                result_dist_array[index], query_conf.beamwidth, query_conf.search_list_and_k_ratio, bitset);

            // filter range search result
            if (range_filter_exist) {
                FilterRangeSearchResultForOneNq(result_dist_array[index], result_id_array[index], is_ip, radius,
                                                range_filter);
            }
        }));
    }

    for (auto& future : futures) {
        future.get();
    }

    size_t* p_lims = nullptr;
    int64_t* p_id = nullptr;
    float* p_dist = nullptr;

    GetRangeSearchResult(result_dist_array, result_id_array, is_ip, rows, radius, range_filter, p_dist, p_id, p_lims);

    return GenResultDataset(p_id, p_dist, p_lims);
}

template <typename T>
DatasetPtr
IndexDiskANN<T>::GetIndexMeta(const Config& config) {
    std::vector<int64_t> entry_points;
    for (size_t i = 0; i < pq_flash_index_->get_num_medoids(); i++) {
        entry_points.push_back(pq_flash_index_->get_medoids()[i]);
    }
    feder::diskann::DiskANNMeta meta(DiskANNBuildConfig::Get(config), Count(), entry_points);
    std::unordered_set<int64_t> id_set(entry_points.begin(), entry_points.end());

    Config json_meta, json_id_set;
    nlohmann::to_json(json_meta, meta);
    nlohmann::to_json(json_id_set, id_set);
    return GenResultDataset(json_meta.dump(), json_id_set.dump());
}

template <typename T>
int64_t
IndexDiskANN<T>::Count() {
    if (count_.load() == -1) {
        KNOWHERE_THROW_MSG("index is not ready yet.");
    }
    return count_.load();
}

template <typename T>
int64_t
IndexDiskANN<T>::Dim() {
    if (dim_.load() == -1) {
        KNOWHERE_THROW_MSG("index is not ready yet.");
    }
    return dim_.load();
}

template <typename T>
bool
IndexDiskANN<T>::LoadFile(const std::string& filename) {
    if (!file_manager_->LoadFile(filename)) {
        LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
        return false;
    }
    return true;
}

template <typename T>
bool
IndexDiskANN<T>::AddFile(const std::string& filename) {
    if (!file_manager_->AddFile(filename)) {
        LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
        return false;
    }
    return true;
}

template <typename T>
uint64_t
IndexDiskANN<T>::GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim, const uint64_t max_degree) {
    uint32_t one_cached_node_budget = (max_degree + 1) * sizeof(unsigned) + sizeof(T) * data_dim;
    auto num_nodes_to_cache =
        static_cast<uint64_t>(1024 * 1024 * 1024 * cache_dram_budget) / (one_cached_node_budget * kCacheExpansionRate);
    return num_nodes_to_cache;
}

// Explicit template instantiation
template class IndexDiskANN<float>;
}  // namespace knowhere
