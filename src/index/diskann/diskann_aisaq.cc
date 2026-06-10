// Copyright (c) KIOXIA Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstdint>
#include <memory>

#include "diskann/aisaq.h"
#include "diskann/aux_utils.h"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/pq_flash_aisaq_index.h"
#include "diskann/pq_flash_index.h"
#include "filemanager/FileManager.h"
#include "fmt/core.h"
#include "index/diskann/aisaq_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/context.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feder/DiskANN.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/range_util.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"

namespace knowhere {

template <typename DataType>
class AisaqIndexNode : public IndexNode {
    static_assert(KnowhereFloatTypeCheck<DataType>::value,
                  "AiSAQ only support floating point data type(float32, float16, bfloat16)");

 public:
    using DistType = float;

    AisaqIndexNode(const int32_t& version, const Object& object) : is_prepared_(false), dim_(-1), count_(-1) {
        assert(typeid(object) == typeid(Pack<std::shared_ptr<milvus::FileManager>>));
        auto diskann_index_pack = dynamic_cast<const Pack<std::shared_ptr<milvus::FileManager>>*>(&object);
        assert(diskann_index_pack != nullptr);
        file_manager_ = diskann_index_pack->GetPack();
    }

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override;

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        knowhere::MetricType metric_type = config.metric_type.has_value() ? config.metric_type.value() : "";
        return IsMetricType(metric_type, metric::L2) || IsMetricType(metric_type, metric::COSINE);
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return IsMetricType(metric_type, metric::L2) || IsMetricType(metric_type, metric::COSINE);
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override;

    Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_INFO_ << "AiSAQ does nothing for serialize";
        if (emb_list_strategy_ == nullptr) {
            RETURN_IF_ERROR(AppendExternalIdMapToBinarySet(binset, meta::EXTERNAL_ID_MAP));
        }
        return Status::success;
    }

    static expected<Resource>
    StaticEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                               const knowhere::BaseConfig& config, const IndexVersion& version) {
        uint64_t s = file_size_in_bytes / 1024;
        return Resource{.memoryCost = s, .diskCost = file_size_in_bytes};
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override;

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_ERROR_ << "AiSAQ doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<AisaqConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    Status
    SetFileManager(std::shared_ptr<milvus::FileManager> file_manager) {
        if (file_manager == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Malloc error, file_manager = nullptr.";
            return Status::malloc_error;
        }
        file_manager_ = file_manager;
        return Status::success;
    }

    int64_t
    Dim() const override {
        if (dim_.load() == -1) {
            LOG_KNOWHERE_ERROR_ << "Dim() function is not supported when index is not ready yet.";
            return 0;
        }
        return dim_.load();
    }

    int64_t
    Size() const override {
        if (!is_prepared_.load() || !pq_flash_index_) {
            LOG_KNOWHERE_ERROR_ << "AiSAQ not loaded.";
            return 0;
        }
        return pq_flash_index_->aisaq_cal_size();
    }

    int64_t
    Count() const override {
        if (count_.load() == -1) {
            LOG_KNOWHERE_ERROR_ << "Count() function is not supported when index is not ready yet.";
            return 0;
        }
        return count_.load();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_AISAQ;
    }

 private:
    bool
    LoadFile(const std::string& filename) {
        if (!file_manager_->LoadFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    bool
    AddFile(const std::string& filename) {
        if (!file_manager_->AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    uint64_t
    GetCachedNodeNum(const float cache_dram_budget, const uint32_t max_node_len = 0);

    std::string index_prefix_;
    mutable std::mutex preparation_lock_;
    std::atomic_bool is_prepared_;
    std::shared_ptr<milvus::FileManager> file_manager_;
    std::unique_ptr<diskann::PQFlashAisaqIndex<DataType>> pq_flash_index_;
    std::atomic_int64_t dim_;
    std::atomic_int64_t count_;
    std::shared_ptr<ThreadPool> search_pool_;
};

}  // namespace knowhere

namespace knowhere {
namespace {
static constexpr float kCacheExpansionRate = 1.2;

Status
TryDiskANNCall(std::function<void()>&& diskann_call) {
    try {
        diskann_call();
        return Status::success;
    } catch (const diskann::FileException& e) {
        LOG_KNOWHERE_ERROR_ << "AiSAQ File Exception: " << e.what();
        return Status::disk_file_error;
    } catch (const diskann::ANNException& e) {
        LOG_KNOWHERE_ERROR_ << "AiSAQ Exception: " << e.what();
        return Status::diskann_inner_error;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "AiSAQ Other Exception: " << e.what();
        return Status::diskann_inner_error;
    }
}

std::vector<std::string>
GetNecessaryFilenames(const std::string& prefix, const bool need_norm, const bool use_sample_cache,
                      const bool use_sample_warmup, const bool rearrange, const bool entry_points) {
    std::vector<std::string> filenames;
    auto pq_pivots_filename = diskann::get_pq_pivots_filename(prefix);
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);

    filenames.push_back(pq_pivots_filename);
    filenames.push_back(diskann::get_pq_rearrangement_perm_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_chunk_offsets_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_centroid_filename(pq_pivots_filename));
    filenames.push_back(disk_index_filename);
    if (need_norm) {
        filenames.push_back(diskann::get_disk_index_max_base_norm_file(disk_index_filename));
    }
    if (use_sample_cache || use_sample_warmup) {
        filenames.push_back(diskann::get_sample_data_filename(prefix));
    }
    if (rearrange) {
        filenames.push_back(diskann::get_index_rearranged_filename(prefix));
        filenames.push_back(diskann::get_pq_compressed_rearranged_filename(prefix));
    } else {
        filenames.push_back(diskann::get_pq_compressed_filename(prefix));
    }
    if (entry_points) {
        filenames.push_back(diskann::get_index_entry_points_filename(prefix));
    }
    return filenames;
}

std::vector<std::string>
GetOptionalFilenames(const std::string& prefix) {
    std::vector<std::string> filenames;
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);
    auto disk_pq_pivots_file_name = diskann::get_disk_index_pq_pivots_filename(disk_index_filename);
    filenames.push_back(diskann::get_disk_index_centroids_filename(disk_index_filename));
    filenames.push_back(diskann::get_disk_index_medoids_filename(disk_index_filename));
    filenames.push_back(disk_pq_pivots_file_name);
    filenames.push_back(diskann::get_pq_rearrangement_perm_filename(disk_pq_pivots_file_name));
    filenames.push_back(diskann::get_pq_chunk_offsets_filename(disk_pq_pivots_file_name));
    filenames.push_back(diskann::get_pq_centroid_filename(disk_pq_pivots_file_name));
    filenames.push_back(diskann::get_cached_nodes_file(prefix));
    return filenames;
}

inline bool
AnyIndexFileExist(const std::string& index_prefix) {
    auto file_exist = [](std::vector<std::string> filenames) -> bool {
        for (auto& filename : filenames) {
            if (file_exists(filename)) {
                return true;
            }
        }
        return false;
    };
    return file_exist(GetNecessaryFilenames(index_prefix, diskann::INNER_PRODUCT, true, true, false, false)) ||
           file_exist(GetOptionalFilenames(index_prefix));
}

inline bool
CheckMetric(const std::string& diskann_metric) {
    if (diskann_metric != knowhere::metric::L2 && diskann_metric != knowhere::metric::IP &&
        diskann_metric != knowhere::metric::COSINE) {
        LOG_KNOWHERE_ERROR_ << "AiSAQ currently only supports floating point "
                               "data for Minimum Euclidean "
                               "distance(L2), Max Inner Product Search(IP) "
                               "and Minimum Cosine Search(COSINE).";
        return false;
    } else {
        return true;
    }
}
}  // namespace

static size_t
get_pq_size(size_t points_num, size_t dim, double pq_code_size_gb) {
    double pq_code_size_limit = diskann::get_memory_budget(pq_code_size_gb);
    size_t num_pq_chunks = static_cast<size_t>((std::floor)(static_cast<_u64>(pq_code_size_limit / points_num)));
    num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
    num_pq_chunks = num_pq_chunks > diskann::defaults::MAX_PQ_CHUNKS ? diskann::defaults::MAX_PQ_CHUNKS : num_pq_chunks;
    return num_pq_chunks * sizeof(_u8);
}

template <typename DataType>
Status
AisaqIndexNode<DataType>::Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    assert(file_manager_ != nullptr);
    auto build_conf = static_cast<const AisaqConfig&>(*cfg);

    assert(build_conf.inline_pq.has_value());
    assert(build_conf.rearrange.has_value());
    assert(build_conf.num_entry_points.has_value());
    if (build_conf.inline_pq.value() > build_conf.max_degree.value()) {
        LOG_KNOWHERE_ERROR_ << "inline pq more than max degree value";
        return Status::aisaq_error;
    }

    const int32_t max_degree = static_cast<int32_t>(diskann::defaults::MAX_AISAQ_MAX_DEGREE);
    if (build_conf.max_degree.value() > max_degree) {
        LOG_KNOWHERE_ERROR_ << "max degree more than maximum allowed max degree value";
        return Status::aisaq_error;
    }

    if (build_conf.disk_pq_dims.value() < 0 || build_conf.disk_pq_dims.value() > build_conf.dim.value()) {
        LOG_KNOWHERE_ERROR_ << "disk PQ badget more than dimension value";
        return Status::aisaq_error;
    }

    const int32_t max_aisaq_search_list_size = static_cast<int32_t>(diskann::defaults::MAX_AISAQ_SEARCH_LIST_SIZE);
    if (build_conf.search_list_size.value() > max_aisaq_search_list_size) {
        LOG_KNOWHERE_ERROR_ << "search list size value more than maximum allowed value";
        return Status::aisaq_error;
    }

    LOG_KNOWHERE_INFO_ << "AiSAQ build Configuration:"
                       << " metric type: " << build_conf.metric_type.value()
                       << " inline pq: " << build_conf.inline_pq.value()
                       << " rearrange: " << build_conf.rearrange.value()
                       << " number of entry points: " << build_conf.num_entry_points.value()
                       << " max degree: " << build_conf.max_degree.value()
                       << " search list size: " << build_conf.search_list_size.value()
                       << " pq_code_budget_gb: " << build_conf.pq_code_budget_gb.value()
                       << " build_dram_budget_gb: " << build_conf.build_dram_budget_gb.value()
                       << " search_cache_budget_gb: " << build_conf.search_cache_budget_gb.value()
                       << " disk PQ budget: " << build_conf.disk_pq_dims.value();

    std::scoped_lock lock(preparation_lock_);
    if (!CheckMetric(build_conf.metric_type.value())) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << build_conf.metric_type.value();
        return Status::invalid_metric_type;
    }
    if (!(build_conf.index_prefix.has_value() && build_conf.data_path.has_value())) {
        LOG_KNOWHERE_ERROR_ << "AiSAQ file path for build is empty.";
        return Status::invalid_param_in_json;
    }
    if (AnyIndexFileExist(build_conf.index_prefix.value())) {
        LOG_KNOWHERE_ERROR_ << "This index prefix already has index files.";
        return Status::disk_file_error;
    }
    if (!LoadFile(build_conf.data_path.value())) {
        LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building.";
        return Status::disk_file_error;
    }
    auto data_path = build_conf.data_path.value();

    index_prefix_ = build_conf.index_prefix.value();
    if (emb_list_strategy_ == nullptr) {
        RETURN_IF_ERROR(SetExternalIdMapFromDataset(dataset));
    }

    size_t count;
    size_t dim;
    diskann::get_bin_metadata(build_conf.data_path.value(), count, dim);
    count_.store(count);
    dim_.store(dim);
    if (count == 0) {
        return Status::empty_index;
    }

    bool need_norm = IsMetricType(build_conf.metric_type.value(), knowhere::metric::IP) ||
                     IsMetricType(build_conf.metric_type.value(), knowhere::metric::COSINE);
    auto diskann_metric = [m = build_conf.metric_type.value()] {
        if (IsMetricType(m, knowhere::metric::L2)) {
            return diskann::Metric::L2;
        } else if (IsMetricType(m, knowhere::metric::COSINE)) {
            return diskann::Metric::COSINE;
        } else {
            return diskann::Metric::INNER_PRODUCT;
        }
    }();

    uint32_t inline_pq_vectors;
    uint32_t pq_compressed_nbytes = get_pq_size(count, dim, build_conf.pq_code_budget_gb.value());
    uint64_t max_node_len;
    // calc max_node_len in order to estimate number of nodes to cache
    if (build_conf.disk_pq_dims.value() > 0) {
        uint32_t disk_pq_nchunks = dim;
        if (std::cmp_less(build_conf.disk_pq_dims.value(), dim)) {
            disk_pq_nchunks = build_conf.disk_pq_dims.value();
        }
        max_node_len = (static_cast<uint64_t>(build_conf.max_degree.value()) + 1) * sizeof(uint32_t) +
                       disk_pq_nchunks * sizeof(_u8);
        diskann::aisaq_calc_inline_layout<_u8>(build_conf.inline_pq.value(), pq_compressed_nbytes,
                                               build_conf.max_degree.value(), build_conf.rearrange.value(),
                                               inline_pq_vectors, max_node_len);
    } else {
        max_node_len =
            (static_cast<uint64_t>(build_conf.max_degree.value()) + 1) * sizeof(uint32_t) + dim * sizeof(DataType);
        diskann::aisaq_calc_inline_layout<DataType>(build_conf.inline_pq.value(), pq_compressed_nbytes,
                                                    build_conf.max_degree.value(), build_conf.rearrange.value(),
                                                    inline_pq_vectors, max_node_len);
    }

    auto num_nodes_to_cache = GetCachedNodeNum(build_conf.search_cache_budget_gb.value(), max_node_len);
    diskann::BuildConfig aisaq_internal_build_config{data_path,
                                                     index_prefix_,
                                                     diskann_metric,
                                                     static_cast<unsigned>(build_conf.max_degree.value()),
                                                     static_cast<unsigned>(build_conf.search_list_size.value()),
                                                     static_cast<double>(build_conf.pq_code_budget_gb.value()),
                                                     static_cast<double>(build_conf.build_dram_budget_gb.value()),
                                                     static_cast<uint32_t>(build_conf.disk_pq_dims.value()),
                                                     false,
                                                     build_conf.accelerate_build.value(),
                                                     static_cast<uint32_t>(num_nodes_to_cache), /* num_nodes_to_cache */
                                                     build_conf.shuffle_build.value(),
                                                     true,
                                                     static_cast<uint32_t>(build_conf.inline_pq.value()),
                                                     build_conf.rearrange.value(),
                                                     build_conf.num_entry_points.value()};
    RETURN_IF_ERROR(TryDiskANNCall([&]() {
        int res = diskann::build_disk_index<DataType>(aisaq_internal_build_config);
        if (res != 0)
            throw diskann::ANNException("AiSAQ::build_disk_index returned non-zero value: " + std::to_string(res), -1);
    }));

    // Add file to the file manager
    for (auto& filename :
         GetNecessaryFilenames(index_prefix_, need_norm, true, true, aisaq_internal_build_config.rearrange,
                               build_conf.num_entry_points.value() > 0)) {
        if (!AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return Status::disk_file_error;
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        if (file_exists(filename) && !AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return Status::disk_file_error;
        }
    }
    if (emb_list_strategy_ == nullptr) {
        RETURN_IF_ERROR(SaveExternalIdMapToFileManager(file_manager_, index_prefix_ + "_id_map"));
    }

    is_prepared_.store(false);
    return Status::success;
}

template <typename DataType>
Status
AisaqIndexNode<DataType>::Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) {
    auto prep_conf = static_cast<const AisaqConfig&>(*cfg);
    bool rearrange = prep_conf.rearrange.value();
    bool entry_points = prep_conf.num_entry_points.value() > 0;
    assert(prep_conf.pq_cache_size.has_value());
    LOG_KNOWHERE_INFO_ << "AiSAQ deserialize configuration:"
                       << " max vectors_beam_width: " << diskann::defaults::MAX_AISAQ_VECTORS_BEAMWIDTH
                       << " pq cache size: " << prep_conf.pq_cache_size.value() << " bytes";

    diskann::aisaq_pq_io_engine pq_read_io_engine = diskann::aisaq_pq_io_engine_default;

    std::scoped_lock lock(preparation_lock_);
    if (!CheckMetric(prep_conf.metric_type.value())) {
        return Status::invalid_metric_type;
    }
    if (is_prepared_.load()) {
        return Status::success;
    }
    if (!(prep_conf.index_prefix.has_value())) {
        LOG_KNOWHERE_ERROR_ << "AiSAQ file path for deserialize is empty.";
        return Status::invalid_param_in_json;
    }
    index_prefix_ = prep_conf.index_prefix.value();
    bool is_ip = IsMetricType(prep_conf.metric_type.value(), knowhere::metric::IP);
    bool need_norm = IsMetricType(prep_conf.metric_type.value(), knowhere::metric::IP) ||
                     IsMetricType(prep_conf.metric_type.value(), knowhere::metric::COSINE);
    auto diskann_metric = [m = prep_conf.metric_type.value()] {
        if (IsMetricType(m, knowhere::metric::L2)) {
            return diskann::Metric::L2;
        } else if (IsMetricType(m, knowhere::metric::COSINE)) {
            return diskann::Metric::COSINE;
        } else {
            return diskann::Metric::INNER_PRODUCT;
        }
    }();

    // Load file from file manager.
    bool use_bfs_cache = prep_conf.use_bfs_cache.value();
    for (auto& filename :
         GetNecessaryFilenames(index_prefix_, need_norm, prep_conf.search_cache_budget_gb.value() > 0 && !use_bfs_cache,
                               prep_conf.warm_up.value(), rearrange, entry_points)) {
        if (filename == diskann::get_pq_compressed_filename(index_prefix_)) {
            LOG_KNOWHERE_DEBUG_ << "File load " << filename << " skipped";
            continue;
        }
        if (!LoadFile(filename)) {
            return Status::disk_file_error;
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        auto is_exist_op = file_manager_->IsExisted(filename);
        if (!is_exist_op.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to check existence of file " << filename << ".";
            return Status::disk_file_error;
        }
        if (is_exist_op.value() && !LoadFile(filename)) {
            return Status::disk_file_error;
        }
    }
    if (emb_list_strategy_ == nullptr) {
        if (binset.GetByName(meta::EXTERNAL_ID_MAP) != nullptr) {
            RETURN_IF_ERROR(LoadExternalIdMapFromBinarySet(binset, meta::EXTERNAL_ID_MAP));
        } else {
            RETURN_IF_ERROR(LoadExternalIdMapFromFileManager(file_manager_, index_prefix_ + "_id_map"));
        }
    }

    // set thread pool
    search_pool_ = ThreadPool::GetGlobalSearchThreadPool();

    // load diskann pq code and meta info
    std::shared_ptr<AlignedFileReader> reader = nullptr;

    reader = std::make_shared<LinuxAlignedFileReader>();

    pq_flash_index_ = std::make_unique<diskann::PQFlashAisaqIndex<DataType>>(reader, diskann_metric);

    auto disk_ann_call = [&]() {
        int res = pq_flash_index_->aisaq_load(search_pool_->size(), index_prefix_.c_str());
        if (res != 0) {
            throw diskann::ANNException("AiSAQ load returned non-zero value: " + std::to_string(res), -1);
        }
    };
    if (TryDiskANNCall(disk_ann_call) != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Failed to load AiSAQ.";
        return Status::aisaq_error;
    }
    if (!pq_flash_index_->get_rearranged_index() && !pq_flash_index_->get_rearrange_during_search() &&
        prep_conf.pq_read_page_cache_size.value() > 0) {
        LOG_KNOWHERE_WARNING_
            << "Dynamic cache can only be used when vectors rearrangement is enabled. dynamic cache will be disabled";
        prep_conf.pq_read_page_cache_size.value() = 0;
    }

    if (pq_flash_index_->aisaq_init(pq_read_io_engine, index_prefix_.c_str()) != 0) {
        return Status::aisaq_error;
    }

    if (prep_conf.pq_cache_size.value() > 0) {
        if (pq_flash_index_->aisaq_load_pq_cache(index_prefix_, prep_conf.pq_cache_size.value(),
                                                 diskann::aisaq_pq_cache_policy_auto,
                                                 pq_flash_index_->get_rearrange_during_search()) != true) {
            LOG_KNOWHERE_ERROR_ << "Failed to load aisaq cache";
            return Status::aisaq_error;
        }
    }

    count_.store(pq_flash_index_->get_num_points());
    // AiSAQ will add one more dim for IP type.
    if (is_ip) {
        dim_.store(pq_flash_index_->get_data_dim() - 1);
    } else {
        dim_.store(pq_flash_index_->get_data_dim());
    }

    std::string warmup_query_file = diskann::get_sample_data_filename(index_prefix_);
    // load cache
    auto cached_nodes_file = diskann::get_cached_nodes_file(index_prefix_);
    std::vector<uint32_t> node_list;
    if (file_exists(cached_nodes_file)) {
        LOG_KNOWHERE_INFO_ << "Reading cached nodes from file.";
        size_t num_nodes, nodes_id_dim;
        std::unique_ptr<uint32_t[]> cached_nodes_ids = nullptr;
        diskann::load_bin<uint32_t>(cached_nodes_file, cached_nodes_ids, num_nodes, nodes_id_dim);
        node_list.assign(cached_nodes_ids.get(), cached_nodes_ids.get() + num_nodes);
    } else {
        auto num_nodes_to_cache =
            GetCachedNodeNum(prep_conf.search_cache_budget_gb.value(), pq_flash_index_->get_max_node_len());
        if (num_nodes_to_cache > pq_flash_index_->get_num_points() / 3) {
            LOG_KNOWHERE_ERROR_ << "Failed to generate cache, num_nodes_to_cache(" << num_nodes_to_cache
                                << ") is larger than 1/3 of the total data number.";
            return Status::invalid_args;
        }
        if (num_nodes_to_cache > 0) {
            if (use_bfs_cache) {
                LOG_KNOWHERE_INFO_ << "Use bfs to generate cache list";
                if (TryDiskANNCall([&]() { pq_flash_index_->cache_bfs_levels(num_nodes_to_cache, node_list); }) !=
                    Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate bfs cache for AiSAQ.";
                    return Status::aisaq_error;
                }
            } else {
                LOG_KNOWHERE_INFO_ << "Use sample_queries to generate cache list";
                if (TryDiskANNCall([&]() {
                        pq_flash_index_->aisaq_async_generate_cache_list_from_sample_queries(warmup_query_file, 15, 6,
                                                                                             num_nodes_to_cache);
                    }) != Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate cache from sample queries for AiSAQ.";
                    return Status::aisaq_error;
                }
            }
        }
        LOG_KNOWHERE_INFO_ << "End of preparing AiSAQ index.";
    }

    if (node_list.size() > 0) {
        if (TryDiskANNCall([&]() { pq_flash_index_->aisaq_load_cache_list(node_list); }) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load cache for AiSAQ.";
            return Status::aisaq_error;
        }
    }

    // warmup
    if (prep_conf.warm_up.value()) {
        LOG_KNOWHERE_INFO_ << "Warming up.";
        uint64_t warmup_L = 20;
        uint64_t warmup_num = 0;
        uint64_t warmup_dim = 0;
        uint64_t warmup_aligned_dim = 0;
        DataType* warmup = nullptr;
        if (TryDiskANNCall([&]() {
                diskann::load_aligned_bin<DataType>(warmup_query_file, warmup, warmup_num, warmup_dim,
                                                    warmup_aligned_dim);
            }) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load warmup file for AiSAQ.";
            return Status::disk_file_error;
        }
        std::vector<int64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<DistType> warmup_result_dists(warmup_num, 0);

        std::vector<folly::Future<folly::Unit>> futures;
        futures.reserve(warmup_num);
        for (uint64_t i = 0; i < warmup_num; ++i) {
            futures.emplace_back(search_pool_->push([&, index = i]() {
                pq_flash_index_->aisaq_cached_beam_search(warmup + (index * warmup_aligned_dim), 1, warmup_L,
                                                          warmup_result_ids_64.data() + (index * 1),
                                                          warmup_result_dists.data() + (index * 1), 4);
            }));
        }

        bool failed = TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success;

        if (warmup != nullptr) {
            diskann::aligned_free(warmup);
        }

        if (failed) {
            LOG_KNOWHERE_ERROR_ << "Failed to do search on warmup file for AiSAQ.";
            return Status::aisaq_error;
        }
    }

    is_prepared_.store(true);
    LOG_KNOWHERE_INFO_ << "End of AiSAQ loading.";
    return Status::success;
}

template <typename DataType>
expected<DataSetPtr>
AisaqIndexNode<DataType>::Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                                 milvus::OpContext* op_context) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load AiSAQ.";
        return expected<DataSetPtr>::Err(Status::empty_index, "AiSAQ not loaded");
    }

    auto search_conf = static_cast<const AisaqConfig&>(*cfg);
    if (!CheckMetric(search_conf.metric_type.value())) {
        return expected<DataSetPtr>::Err(Status::invalid_metric_type, "unsupported metric type");
    }

    auto k = static_cast<uint64_t>(search_conf.k.value());
    auto lsearch = static_cast<uint64_t>(search_conf.search_list_size.value());
    auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth.value());
    auto filter_ratio = static_cast<float>(search_conf.filter_threshold.value());

    struct diskann::aisaq_search_config aisaq_search_config;
    assert(search_conf.beamwidth.has_value());
    assert(search_conf.vectors_beamwidth.has_value());
    assert(search_conf.pq_read_page_cache_size.has_value());

    if (std::cmp_greater(search_conf.beamwidth.value(), diskann::defaults::MAX_AISAQ_BEAMWIDTH)) {
        LOG_KNOWHERE_ERROR_ << "Error. Beam width more than max value";
        return expected<DataSetPtr>::Err(Status::aisaq_error, "beam width more than maximal");
    }

    if (search_conf.vectors_beamwidth.value() > search_conf.beamwidth.value()) {
        LOG_KNOWHERE_ERROR_ << "Error. Vector beam width more than beam width";
        return expected<DataSetPtr>::Err(Status::aisaq_error, "vector beam width more than beam width");
    }
    aisaq_search_config.vector_beamwidth = search_conf.vectors_beamwidth.value();

    aisaq_search_config.pq_read_page_cache_size = search_conf.pq_read_page_cache_size.value();

    auto nq = static_cast<uint64_t>(dataset->GetRows());
    auto dim = dataset->GetDim();
    auto xq = static_cast<const DataType*>(dataset->GetTensor());

    LOG_KNOWHERE_DEBUG_ << "AiSAQ search configuration :"
                        << " index beam width: " << beamwidth
                        << " vectors beam width: " << aisaq_search_config.vector_beamwidth
                        << " pq-read-page-cache-size: " << aisaq_search_config.pq_read_page_cache_size << " bytes"
                        << " search list size: " << search_conf.search_list_size.value();

    feder::diskann::FederResultUniq feder_result;
    if (search_conf.trace_visit.value()) {
        if (nq != 1) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "nq must be 1");
        }
        feder_result = std::make_unique<feder::diskann::FederResult>();
        feder_result->visit_info_.SetQueryConfig(search_conf.k.value(), search_conf.search_list_size.value(),
                                                 search_conf.beamwidth.value(), aisaq_search_config.vector_beamwidth);
    }

    auto p_id = std::make_unique<int64_t[]>(k * nq);
    auto p_dist = std::make_unique<DistType[]>(k * nq);
    BitsetView mapped_bitset(bitset);
    external_id_map_.SetOutIdsToBitset(mapped_bitset);

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(nq);
    for (uint64_t row = 0; row < nq; ++row) {
        futures.emplace_back(search_pool_->push([&, index = row, p_id_ptr = p_id.get(), p_dist_ptr = p_dist.get()]() {
            knowhere::checkCancellation(op_context);
            diskann::QueryStats stats;
            pq_flash_index_->aisaq_cached_beam_search(xq + (index * dim), k, lsearch, p_id_ptr + (index * k),
                                                      p_dist_ptr + (index * k), beamwidth, false, &stats, feder_result,
                                                      mapped_bitset, filter_ratio, &aisaq_search_config);
#ifdef NOT_COMPILE_FOR_SWIG
            knowhere_diskann_search_hops.Observe(stats.n_hops);
#endif
        }));
    }

    if (TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success) {
        return expected<DataSetPtr>::Err(Status::aisaq_error, "some search failed");
    }

    external_id_map_.MapResultIds(p_id.get(), k * nq);
    auto res = GenResultDataSet(nq, k, std::move(p_id), std::move(p_dist));

    // set visit_info json string into result dataset
    if (feder_result != nullptr) {
        Json json_visit_info, json_id_set;
        nlohmann::to_json(json_visit_info, feder_result->visit_info_);
        nlohmann::to_json(json_id_set, feder_result->id_set_);
        res->SetJsonInfo(json_visit_info.dump());
        res->SetJsonIdSet(json_id_set.dump());
    }
    return res;
}

/*
 * Get raw vector data given their ids.
 * It first tries to get data from cache, if failed, it will try to get data from disk.
 * It reads as much as possible and it is thread-pool free, it totally depends on the outside to control concurrency.
 */
template <typename DataType>
expected<DataSetPtr>
AisaqIndexNode<DataType>::GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load AiSAQ.";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
    }
    auto dim = Dim();
    auto rows = dataset->GetRows();
    auto ids = dataset->GetIds();
    std::vector<int64_t> internal_ids;
    if (emb_list_strategy_ == nullptr) {
        ids = external_id_map_.ToInternalIds(ids, rows, internal_ids);
    }
    for (int64_t i = 0; i < rows; ++i) {
        if (ids[i] < 0 || ids[i] >= count_.load()) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "invalid vector id");
        }
    }
    auto* data = new DataType[dim * rows];
    if (data == nullptr) {
        LOG_KNOWHERE_ERROR_ << "Failed to allocate memory for data.";
        return expected<DataSetPtr>::Err(Status::malloc_error, "failed to allocate memory for data");
    }

    if (TryDiskANNCall([&]() { pq_flash_index_->aisaq_get_vector_by_ids(ids, rows, data); }) != Status::success) {
        delete[] data;
        return expected<DataSetPtr>::Err(Status::aisaq_error, "failed to get vector");
    };

    return GenResultDataSet(rows, dim, data);
}

template <typename DataType>
expected<DataSetPtr>
AisaqIndexNode<DataType>::GetIndexMeta(std::unique_ptr<Config> cfg) const {
    std::vector<int64_t> entry_points;
    auto num_medoids = pq_flash_index_->get_num_medoids();
    entry_points.reserve(num_medoids);

    for (size_t i = 0; i < num_medoids; i++) {
        entry_points.push_back(pq_flash_index_->get_medoids()[i]);
    }
    auto aisaq_conf = static_cast<const DiskANNConfig&>(*cfg);
    LOG_KNOWHERE_INFO_ << "Count " << Count();
    LOG_KNOWHERE_INFO_ << "max_degree " << aisaq_conf.max_degree.value();
    LOG_KNOWHERE_INFO_ << "search_list_size " << aisaq_conf.search_list_size.value();
    LOG_KNOWHERE_INFO_ << "pq_code_budget_gb " << aisaq_conf.pq_code_budget_gb.value();
    LOG_KNOWHERE_INFO_ << "build_dram_budget_gb " << aisaq_conf.build_dram_budget_gb.value();
    LOG_KNOWHERE_INFO_ << "disk_pq_dims " << aisaq_conf.disk_pq_dims.value();
    LOG_KNOWHERE_INFO_ << "accelerate_build " << aisaq_conf.accelerate_build.value();
    LOG_KNOWHERE_INFO_ << "data_path " << aisaq_conf.data_path.value();
    feder::diskann::DiskANNMeta meta(aisaq_conf.data_path.value(), aisaq_conf.max_degree.value(),
                                     aisaq_conf.search_list_size.value(), aisaq_conf.pq_code_budget_gb.value(),
                                     aisaq_conf.build_dram_budget_gb.value(), aisaq_conf.disk_pq_dims.value(),
                                     aisaq_conf.accelerate_build.value(), Count(), entry_points);
    std::unordered_set<int64_t> id_set(entry_points.begin(), entry_points.end());

    Json json_meta, json_id_set;
    nlohmann::to_json(json_meta, meta);
    nlohmann::to_json(json_id_set, id_set);
    return GenResultDataSet(json_meta.dump(), json_id_set.dump());
}

template <typename DataType>
uint64_t
AisaqIndexNode<DataType>::GetCachedNodeNum(const float cache_dram_budget, const uint32_t max_node_len) {
    auto num_nodes_to_cache =
        static_cast<uint64_t>(1024 * 1024 * 1024 * cache_dram_budget) / (max_node_len * kCacheExpansionRate);
    return num_nodes_to_cache;
}

KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(AISAQ, AisaqIndexNode, knowhere::feature::DISK)

}  // namespace knowhere
