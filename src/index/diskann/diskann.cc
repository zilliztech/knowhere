// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "knowhere/feder/DiskANN.h"

#include <folly/ScopeGuard.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <unordered_set>

#include "diskann/aux_utils.h"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/pq_flash_index.h"
#include "filemanager/FileManager.h"
#include "fmt/core.h"
#include "index/diskann/build_files.h"
#include "index/diskann/diskann_config.h"
#include "index/diskann/navigation_store.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/context.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/range_util.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"

namespace knowhere {
template <typename DataType>
class DiskANNIndexNode : public IndexNode {
    static_assert(KnowhereFloatTypeCheck<DataType>::value,
                  "DiskANN only support floating point data type(float32, float16, bfloat16)");

 public:
    using DistType = float;
    DiskANNIndexNode(const int32_t& version, const Object& object)
        : IndexNode(version), is_prepared_(false), dim_(-1), count_(-1) {
        assert(typeid(object) == typeid(Pack<std::shared_ptr<milvus::FileManager>>));
        auto diskann_index_pack = dynamic_cast<const Pack<std::shared_ptr<milvus::FileManager>>*>(&object);
        assert(diskann_index_pack != nullptr);
        file_manager_ = diskann_index_pack->GetPack();
    }

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override;

    Status
    BuildEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    bool
    NeedBitsetExactCount() const override {
        return true;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override;

    std::optional<size_t>
    GetQueryCodeSize(const DataSetPtr dataset) const override {
        if (dataset == nullptr) {
            LOG_KNOWHERE_ERROR_ << "GetQueryCodeSize: dataset is nullptr";
            return std::nullopt;
        }
        const auto dim = dataset->GetDim();
        if constexpr (std::is_same_v<DataType, knowhere::fp32>) {
            return sizeof(float) * dim;
        } else if constexpr (std::is_same_v<DataType, knowhere::fp16>) {
            return sizeof(uint16_t) * dim;
        } else if constexpr (std::is_same_v<DataType, knowhere::bf16>) {
            return sizeof(uint16_t) * dim;
        }
        LOG_KNOWHERE_ERROR_ << "Invalid data type: " << typeid(DataType).name();
        return std::nullopt;
    }

    expected<DataSetPtr>
    CalcDistByStorageIds(const DataSetPtr dataset, const BitsetView& bitset, const int64_t* labels,
                         const size_t labels_len, const bool is_cosine, milvus::OpContext* op_context) const override;

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        const auto* disk_config = dynamic_cast<const DiskANNConfig*>(&config);
        if (disk_config && disk_config->disk_pq_dims.value_or(0) > 0) {
            return false;
        }
        knowhere::MetricType metric_type = config.metric_type.has_value() ? config.metric_type.value() : "";
        const auto& base_metric = get_sub_metric_type(metric_type).value_or(metric_type);
        return IsMetricType(base_metric, metric::L2) || IsMetricType(base_metric, metric::COSINE);
    }

    static Status
    StaticConfigCheck(const Config& cfg, PARAM_TYPE paramType, std::string& msg) {
        auto& base_cfg = static_cast<const BaseConfig&>(cfg);
        if (UsesExternalNavigation(static_cast<const DiskANNConfig&>(cfg))) {
            if (!std::is_same_v<DataType, float>) {
                msg = "external DiskANN navigation currently requires FP32 data";
                return Status::invalid_args;
            }
            // The strategy has a default even for ordinary vectors. Only
            // actual embedding-list inputs select embedding-list mode.
            if (get_el_metric_type(base_cfg.metric_type.value_or(metric::L2)).has_value() ||
                base_cfg.emb_list_offset_file_path.has_value()) {
                msg = "external DiskANN navigation does not support embedding-list mode";
                return Status::not_implemented;
            }
        }
        auto strategy = base_cfg.emb_list_strategy.value_or("");
        if (strategy == meta::EMB_LIST_STRATEGY_MUVERA || strategy == meta::EMB_LIST_STRATEGY_LEMUR) {
            msg = "DiskANN only supports TokenANN strategy, got '" + strategy + "'";
            return Status::invalid_args;
        }
        return Status::success;
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if (pq_flash_index_ && pq_flash_index_->uses_disk_pq()) {
            return false;
        }
        const auto& base_metric = get_sub_metric_type(metric_type).value_or(metric_type);
        return IsMetricType(base_metric, metric::L2) || IsMetricType(base_metric, metric::COSINE);
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override;

    Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_INFO_ << "DiskANN does nothing for serialize";
        return Status::success;
    }

    Status
    SerializeEmbListIfNeed(BinarySet& binset) const override {
        LOG_KNOWHERE_INFO_ << "DiskANN does nothing for serialize (with emb list if needed)";
        return Status::success;
    }

    static expected<Resource>
    StaticEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                               const knowhere::BaseConfig& config, const IndexVersion& version) {
        const auto& disk_config = static_cast<const DiskANNConfig&>(config);
        if (num_rows < 0 || dim <= 0) {
            return expected<Resource>::Err(Status::invalid_args, "Invalid DiskANN resource estimate dimensions");
        }
        try {
            const auto navigation_bytes = EstimateNavigationMemory(disk_config, num_rows, dim);
            const long double raw_bytes = static_cast<long double>(num_rows) * dim * sizeof(float);
            const long double cache_bytes =
                std::max(static_cast<long double>(disk_config.search_cache_budget_gb.value_or(0)) * (1ULL << 30),
                         static_cast<long double>(disk_config.search_cache_budget_gb_ratio.value_or(0)) * raw_bytes);
            // Preserve the legacy engine allowance. Without complete model
            // parameters, budget the entire serialized payload as resident
            // rather than guessing PQ or applying the RBQ build defaults.
            // Cache is additional in every branch, including explicit PQ.
            // This estimates admission costs, not exact or peak RSS.
            const long double memory_bytes = file_size_in_bytes / 4 +
                                             static_cast<long double>(navigation_bytes.value_or(file_size_in_bytes)) +
                                             std::ceil(cache_bytes);
            if (!std::isfinite(memory_bytes) || memory_bytes < 0 ||
                memory_bytes >= static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
                return expected<Resource>::Err(Status::invalid_args, "DiskANN resource estimate overflows");
            }
            return Resource{.memoryCost = static_cast<uint64_t>(memory_bytes), .diskCost = file_size_in_bytes};
        } catch (const std::exception& e) {
            return expected<Resource>::Err(Status::invalid_args, e.what());
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override;

    Status
    DeserializeEmbListIfNeed(const BinarySet& binset, std::shared_ptr<Config> config) override;

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_ERROR_ << "DiskANN doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    Status
    DeserializeFromFileIfNeed(const std::string& filename, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_INFO_ << "DiskANN doesn't support deserialize from file (with emb list if needed)";
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<DiskANNNavigationConfig>();
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
            LOG_KNOWHERE_ERROR_ << "Diskann not loaded.";
            return 0;
        }
        auto size = pq_flash_index_->cal_size();
        if (navigation_store_ != nullptr) {
            size += navigation_store_->MemorySize();
        }
        return size;
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
        return knowhere::IndexEnum::INDEX_DISKANN;
    }

    expected<std::vector<IndexNode::IteratorPtr>>
    AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                bool use_knowhere_search_pool, milvus::OpContext* op_context) const override;

 protected:
    expected<DataSetPtr>
    GetVectorByStorageIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override;

 private:
    class iterator : public IndexIterator {
     public:
        iterator(const bool transform, const DataType* query_data, const uint64_t lsearch, const uint64_t beam_width,
                 const float filter_ratio, const knowhere::BitsetView& bitset, diskann::PQFlashIndex<DataType>* index,
                 bool use_knowhere_search_pool = true)
            : IndexIterator(transform, use_knowhere_search_pool),
              index_(index),
              transform_(transform),
              workspace_(index_->getIteratorWorkspace(query_data, lsearch, beam_width, filter_ratio, bitset)) {
        }

     protected:
        void
        next_batch(std::function<void(const std::vector<DistId>&)> batch_handler) override {
            index_->getIteratorNextBatch(workspace_.get());
            if (transform_) {
                for (auto& p : workspace_->backup_res) {
                    p.val = -p.val;
                }
            }
            batch_handler(workspace_->backup_res);
            workspace_->backup_res.clear();
        }

     private:
        diskann::PQFlashIndex<DataType>* index_;
        const bool transform_;
        std::unique_ptr<diskann::IteratorWorkspace<DataType>> workspace_;
    };

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
    GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim, size_t chunk_size,
                     const uint64_t max_degree);

    std::string index_prefix_;
    mutable std::mutex preparation_lock_;
    std::atomic_bool is_prepared_;
    std::shared_ptr<milvus::FileManager> file_manager_;
    std::unique_ptr<diskann::PQFlashIndex<DataType>> pq_flash_index_;
    std::unique_ptr<NavigationStore> navigation_store_;
    std::string loaded_navigation_codec_;
    std::atomic_int64_t dim_;
    std::atomic_int64_t count_;
    std::shared_ptr<ThreadPool> search_pool_;
};

}  // namespace knowhere

namespace knowhere {
namespace {
static constexpr float kCacheExpansionRate = 1.2;

Status
ReadEmbListOffsetFromFile(const std::string& file_path, std::vector<size_t>& offsets) {
    std::ifstream in_file(file_path, std::ios::binary);
    if (!in_file) {
        LOG_KNOWHERE_ERROR_ << "Failed to open emb_list offset file for reading: " << file_path;
        return Status::emb_list_inner_error;
    }

    size_t size = 0;
    in_file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    if (!in_file || in_file.gcount() != sizeof(size_t)) {
        LOG_KNOWHERE_ERROR_ << "Failed to read size from emb_list offset file: " << file_path;
        return Status::emb_list_inner_error;
    }
    if (size == 0) {
        LOG_KNOWHERE_ERROR_ << "Emb_list offset file is empty: " << file_path;
        return Status::emb_list_inner_error;
    }

    offsets.resize(size);
    in_file.read(reinterpret_cast<char*>(offsets.data()), size * sizeof(size_t));
    if (!in_file || static_cast<size_t>(in_file.gcount()) != size * sizeof(size_t)) {
        LOG_KNOWHERE_ERROR_ << "Failed to read offset data from emb_list offset file: " << file_path;
        return Status::emb_list_inner_error;
    }

    return Status::success;
}

Status
WriteEmbListOffsetToFile(const std::string& file_path, const std::vector<size_t>& offsets) {
    std::ofstream out_file(file_path, std::ios::binary);
    if (!out_file) {
        LOG_KNOWHERE_ERROR_ << "Failed to open emb_list offset file for writing: " << file_path;
        return Status::emb_list_inner_error;
    }

    const size_t size = offsets.size();
    out_file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    out_file.write(reinterpret_cast<const char*>(offsets.data()), size * sizeof(size_t));
    if (!out_file) {
        LOG_KNOWHERE_ERROR_ << "Failed to write emb_list offset data to file: " << file_path;
        return Status::emb_list_inner_error;
    }
    return Status::success;
}

Status
TryDiskANNCall(std::function<void()>&& diskann_call) {
    try {
        diskann_call();
        return Status::success;
    } catch (const diskann::FileException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN File Exception: " << e.what();
        return Status::disk_file_error;
    } catch (const diskann::ANNException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Exception: " << e.what();
        return Status::diskann_inner_error;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Other Exception: " << e.what();
        return Status::diskann_inner_error;
    }
}

std::vector<std::string>
GetNecessaryFilenames(const DiskANNConfig& config, const std::string& prefix, const bool need_norm,
                      const bool use_sample_cache, const bool use_sample_warmup) {
    auto filenames = NavigationFiles(config, prefix).required;
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);

    filenames.push_back(disk_index_filename);
    if (need_norm) {
        filenames.push_back(diskann::get_disk_index_max_base_norm_file(disk_index_filename));
    }
    if (use_sample_cache || use_sample_warmup) {
        filenames.push_back(diskann::get_sample_data_filename(prefix));
    }
    return filenames;
}

std::vector<std::string>
GetOptionalFilenames(const DiskANNConfig& config, const std::string& prefix) {
    auto filenames = NavigationFiles(config, prefix).optional;
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);
    auto disk_pq_pivots_file_name = diskann::get_disk_index_pq_pivots_filename(disk_index_filename);
    filenames.push_back(diskann::get_disk_index_centroids_filename(disk_index_filename));
    filenames.push_back(diskann::get_disk_index_medoids_filename(disk_index_filename));
    filenames.push_back(disk_pq_pivots_file_name);
    filenames.push_back(diskann::get_pq_rearrangement_perm_filename(disk_pq_pivots_file_name));
    filenames.push_back(diskann::get_pq_chunk_offsets_filename(disk_pq_pivots_file_name));
    filenames.push_back(diskann::get_pq_centroid_filename(disk_pq_pivots_file_name));

    filenames.push_back(diskann::get_cached_nodes_file(prefix));
    filenames.push_back(diskann::get_emb_list_offset_file(prefix));
    return filenames;
}

inline bool
AnyIndexFileExist(const std::string& index_prefix, const DiskANNConfig& config) {
    auto file_exist = [](std::vector<std::string> filenames) -> bool {
        for (auto& filename : filenames) {
            if (file_exists(filename)) {
                return true;
            }
        }
        return false;
    };
    return file_exist(GetNecessaryFilenames(config, index_prefix, true, true, true)) ||
           file_exist(GetOptionalFilenames(config, index_prefix)) || file_exist(AllNavigationFiles(index_prefix));
}

inline bool
CheckMetric(const std::string& diskann_metric) {
    if (diskann_metric != knowhere::metric::L2 && diskann_metric != knowhere::metric::IP &&
        diskann_metric != knowhere::metric::COSINE) {
        LOG_KNOWHERE_ERROR_ << "DiskANN currently only supports floating point "
                               "data for Minimum Euclidean "
                               "distance(L2), Max Inner Product Search(IP) "
                               "and Minimum Cosine Search(COSINE)."
                            << std::endl;
        return false;
    } else {
        return true;
    }
}
}  // namespace

template <typename DataType>
Status
DiskANNIndexNode<DataType>::Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    assert(file_manager_ != nullptr);
    const auto& build_conf = static_cast<const DiskANNConfig&>(*cfg);
    const bool external_navigation = UsesExternalNavigation(build_conf);
    if (external_navigation && !std::is_same_v<DataType, float>) {
        return Status::invalid_args;
    }
    if (!CheckMetric(build_conf.metric_type.value())) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << build_conf.metric_type.value();
        return Status::invalid_metric_type;
    }
    if (!(build_conf.index_prefix.has_value() && build_conf.data_path.has_value())) {
        LOG_KNOWHERE_ERROR_ << "DiskANN file path for build is empty." << std::endl;
        return Status::invalid_param_in_json;
    }
    if (AnyIndexFileExist(build_conf.index_prefix.value(), build_conf)) {
        LOG_KNOWHERE_ERROR_ << "This index prefix already has index files." << std::endl;
        return Status::disk_file_error;
    }
    if (!LoadFile(build_conf.data_path.value())) {
        LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building." << std::endl;
        return Status::disk_file_error;
    }
    auto data_path = build_conf.data_path.value();

    index_prefix_ = build_conf.index_prefix.value();

    size_t count;
    size_t dim;
    diskann::get_bin_metadata(build_conf.data_path.value(), count, dim);

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
    // The coordinate cache uses aligned T slots even when the SSD payload is PQ.
    const auto cache_dim = ROUND_UP(dim + (diskann_metric == diskann::Metric::INNER_PRODUCT ? 1 : 0), 8);
    const auto num_nodes_to_cache = GetCachedNodeNum(build_conf.search_cache_budget_gb.value(), cache_dim,
                                                     sizeof(DataType), build_conf.max_degree.value());
    diskann::BuildConfig diskann_internal_build_config{data_path,
                                                       index_prefix_,
                                                       diskann_metric,
                                                       static_cast<unsigned>(build_conf.max_degree.value()),
                                                       static_cast<unsigned>(build_conf.search_list_size.value()),
                                                       static_cast<double>(build_conf.pq_code_budget_gb.value()),
                                                       static_cast<double>(build_conf.build_dram_budget_gb.value()),
                                                       static_cast<uint32_t>(build_conf.disk_pq_dims.value()),
                                                       false,
                                                       build_conf.accelerate_build.value(),
                                                       static_cast<uint32_t>(num_nodes_to_cache),
                                                       build_conf.shuffle_build.value()};
    std::unique_ptr<diskann::PreparedBuildContext> context;
    DiskANNBuildRegistration registration(*file_manager_);
    for (const auto& path : GetNecessaryFilenames(build_conf, index_prefix_, need_norm, true, true)) {
        if (!registration.Reserve(path))
            return Status::disk_file_error;
    }
    for (const auto& path : GetOptionalFilenames(build_conf, index_prefix_)) {
        if (!registration.Reserve(path))
            return Status::disk_file_error;
    }
    RETURN_IF_ERROR(TryDiskANNCall([&]() {
        context = diskann::prepare_build_context<DataType>(diskann_internal_build_config);
        for (const auto& path : GetNecessaryFilenames(build_conf, index_prefix_, need_norm, true, true)) {
            context->own_output(path);
        }
        for (const auto& path : GetOptionalFilenames(build_conf, index_prefix_)) {
            context->own_output(path);
        }
        auto navigation = CreateNavigationBuilder(build_conf, diskann::make_pq_navigation_builder<DataType>());
        const int res = diskann::build_disk_index<DataType>(diskann_internal_build_config, *context, *navigation);
        if (res != 0) {
            throw diskann::ANNException("diskann::build_disk_index returned non-zero value: " + std::to_string(res),
                                        -1);
        }
    }));

    // Add file to the file manager
    for (auto& filename : GetNecessaryFilenames(build_conf, index_prefix_, need_norm, true, true)) {
        if (!registration.Add(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return Status::disk_file_error;
        }
    }
    for (auto& filename : GetOptionalFilenames(build_conf, index_prefix_)) {
        if (file_exists(filename) && !registration.Add(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return Status::disk_file_error;
        }
    }

    registration.Commit();
    context->commit_outputs();
    count_.store(count);
    dim_.store(dim);
    is_prepared_.store(false);
    return Status::success;
}

template <typename DataType>
Status
DiskANNIndexNode<DataType>::BuildEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                                               bool use_knowhere_build_pool) {
    assert(file_manager_ != nullptr);
    std::scoped_lock lock(preparation_lock_);
    auto& config = static_cast<BaseConfig&>(*cfg);
    auto el_metric_type_or = get_el_metric_type(config.metric_type.value());
    if (!el_metric_type_or.has_value()) {
        // If not emb_list metric type, use the default build method
        return Build(dataset, std::move(cfg), use_knowhere_build_pool);
    }
    if (UsesExternalNavigation(static_cast<const DiskANNConfig&>(*cfg))) {
        LOG_KNOWHERE_ERROR_ << "DISKANN_RABITQ does not support embedding-list mode";
        return Status::not_implemented;
    }

    // DiskANN only supports TokenANN strategy
    auto strategy_type = config.emb_list_strategy.value_or(meta::EMB_LIST_STRATEGY_TOKENANN);
    if (strategy_type != meta::EMB_LIST_STRATEGY_TOKENANN) {
        LOG_KNOWHERE_ERROR_ << "DiskANN only supports TokenANN strategy, got: " << strategy_type;
        return Status::invalid_args;
    }

    LOG_KNOWHERE_INFO_ << "Build emb_list index and read emb_list offset from file.";

    // Validate and get the emb_list offset file path
    if (!config.emb_list_offset_file_path.has_value()) {
        LOG_KNOWHERE_ERROR_ << "Emb_list offset file path is not set";
        return Status::emb_list_inner_error;
    }
    const auto& input_file_path = config.emb_list_offset_file_path.value();
    if (!file_exists(input_file_path)) {
        LOG_KNOWHERE_ERROR_ << "Emb_list offset file does not exist: " << input_file_path;
        return Status::emb_list_inner_error;
    }

    // Read emb_list offset data from the input file
    std::vector<size_t> offset;
    RETURN_IF_ERROR(ReadEmbListOffsetFromFile(input_file_path, offset));
    if (offset.empty() || offset.front() != 0) {
        LOG_KNOWHERE_ERROR_ << "Invalid emb_list offset data (expect first offset = 0), file: " << input_file_path;
        return Status::emb_list_inner_error;
    }

    LOG_KNOWHERE_INFO_ << "Read emb_list offset from file: " << input_file_path << ", size: " << offset.size()
                       << ", first offset: " << offset.front() << ", last offset: " << offset.back();

    auto build_status =
        BuildEmbList(dataset, std::move(cfg), offset.data(), offset.back(), offset.size() - 1, use_knowhere_build_pool);
    if (build_status != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Failed to build base index.";
        return build_status;
    }

    // Save the emb_list offset information to the index file
    {
        const auto output_file_path = diskann::get_emb_list_offset_file(index_prefix_);
        RETURN_IF_ERROR(WriteEmbListOffsetToFile(output_file_path, offset));
        // Add file to the file manager
        if (!AddFile(output_file_path)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << output_file_path << ".";
            return Status::disk_file_error;
        }
    }

    return Status::success;
}

template <typename DataType>
Status
DiskANNIndexNode<DataType>::Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) {
    auto prep_conf = static_cast<const DiskANNNavigationConfig&>(*cfg);
    if (!CheckMetric(prep_conf.metric_type.value())) {
        return Status::invalid_metric_type;
    }
    if (is_prepared_.load()) {
        if (prep_conf.index_prefix.value_or("") != index_prefix_ ||
            (prep_conf.navigation_codec.has_value() &&
             prep_conf.navigation_codec.value() != loaded_navigation_codec_)) {
            return Status::invalid_serialized_index_type;
        }
        return Status::success;
    }
    const auto rollback = folly::makeGuard([this]() {
        if (!is_prepared_.load()) {
            navigation_store_.reset();
            loaded_navigation_codec_.clear();
            pq_flash_index_.reset();
            count_.store(-1);
            dim_.store(-1);
        }
    });
    if (!(prep_conf.index_prefix.has_value())) {
        LOG_KNOWHERE_ERROR_ << "DiskANN file path for deserialize is empty." << std::endl;
        return Status::invalid_param_in_json;
    }
    index_prefix_ = prep_conf.index_prefix.value();
    const auto detected = DetectNavigationCodec(prep_conf, index_prefix_, *file_manager_);
    if (!detected.has_value()) {
        LOG_KNOWHERE_ERROR_ << detected.what();
        return detected.error();
    }
    prep_conf.navigation_codec = detected.value();
    const bool external_navigation = UsesExternalNavigation(prep_conf);
    if (external_navigation && !std::is_same_v<DataType, float>)
        return Status::invalid_args;
    if (external_navigation && (!el_metric_type_.empty() || prep_conf.emb_list_offset_file_path.has_value())) {
        return Status::not_implemented;
    }
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
    for (auto& filename : GetNecessaryFilenames(
             prep_conf, index_prefix_, need_norm,
             prep_conf.search_cache_budget_gb.value() > 0 && !prep_conf.use_bfs_cache.value() && !external_navigation,
             prep_conf.warm_up.value())) {
        if (!LoadFile(filename)) {
            return Status::disk_file_error;
        }
    }
    for (auto& filename : GetOptionalFilenames(prep_conf, index_prefix_)) {
        auto is_exist_op = file_manager_->IsExisted(filename);
        if (!is_exist_op.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to check existence of file " << filename << ".";
            return Status::disk_file_error;
        }
        if (is_exist_op.value() && !LoadFile(filename)) {
            return Status::disk_file_error;
        }
    }

    navigation_store_.reset();
    if (external_navigation) {
        try {
            navigation_store_ = LoadNavigationStore(prep_conf, index_prefix_);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "Failed to initialize DiskANN navigation sidecar: " << e.what();
            return Status::invalid_index_error;
        }
    }

    // set thread pool
    search_pool_ = ThreadPool::GetGlobalSearchThreadPool();

    // load diskann pq code and meta info
    std::shared_ptr<AlignedFileReader> reader = nullptr;

    reader = std::make_shared<LinuxAlignedFileReader>();

    pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<DataType>>(reader, diskann_metric);
    auto disk_ann_call = [&]() {
        typename diskann::PQFlashIndex<DataType>::NavigationMetadata metadata{};
        if (navigation_store_) {
            metadata.count = navigation_store_->Count();
            metadata.dimension = navigation_store_->Dimension();
        }
        int res = pq_flash_index_->load(search_pool_->size(), index_prefix_.c_str(), !external_navigation,
                                        navigation_store_ ? &metadata : nullptr);
        if (res != 0) {
            throw diskann::ANNException("pq_flash_index_->load returned non-zero value: " + std::to_string(res), -1);
        }
    };
    if (TryDiskANNCall(disk_ann_call) != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Failed to load DiskANN.";
        return Status::diskann_inner_error;
    }

    count_.store(pq_flash_index_->get_num_points());
    // DiskANN will add one more dim for IP type.
    if (is_ip) {
        dim_.store(pq_flash_index_->get_data_dim() - 1);
    } else {
        dim_.store(pq_flash_index_->get_data_dim());
    }

    if (external_navigation) {
        try {
            if (navigation_store_->Count() != static_cast<int64_t>(pq_flash_index_->get_num_points()) ||
                navigation_store_->Dimension() != static_cast<int64_t>(pq_flash_index_->get_data_dim())) {
                LOG_KNOWHERE_ERROR_ << "DiskANN graph and navigation sidecar metadata do not match";
                navigation_store_.reset();
                return Status::invalid_index_error;
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "Failed to initialize DiskANN navigation sidecar: " << e.what();
            navigation_store_.reset();
            return Status::invalid_index_error;
        }
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
        const auto num_nodes_to_cache =
            GetCachedNodeNum(prep_conf.search_cache_budget_gb.value(), ROUND_UP(pq_flash_index_->get_data_dim(), 8),
                             sizeof(DataType), pq_flash_index_->get_max_degree());
        if (num_nodes_to_cache > pq_flash_index_->get_num_points() / 3) {
            LOG_KNOWHERE_ERROR_ << "Failed to generate cache, num_nodes_to_cache(" << num_nodes_to_cache
                                << ") is larger than 1/3 of the total data number.";
            return Status::invalid_args;
        }
        if (num_nodes_to_cache > 0) {
            LOG_KNOWHERE_INFO_ << "Caching " << num_nodes_to_cache << " sample nodes around medoid(s).";
            if (prep_conf.use_bfs_cache.value() || external_navigation) {
                if (external_navigation && !prep_conf.use_bfs_cache.value()) {
                    LOG_KNOWHERE_INFO_ << "External navigation uses BFS cache generation because navigation PQ is not "
                                          "resident";
                }
                LOG_KNOWHERE_INFO_ << "Use bfs to generate cache list";
                if (TryDiskANNCall([&]() { pq_flash_index_->cache_bfs_levels(num_nodes_to_cache, node_list); }) !=
                    Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate bfs cache for DiskANN.";
                    return Status::diskann_inner_error;
                }
            } else {
                LOG_KNOWHERE_INFO_ << "Use sample_queries to generate cache list";
                if (TryDiskANNCall([&]() {
                        pq_flash_index_->async_generate_cache_list_from_sample_queries(warmup_query_file, 15, 6,
                                                                                       num_nodes_to_cache);
                    }) != Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate cache from sample queries for DiskANN.";
                    return Status::diskann_inner_error;
                }
            }
        }
        LOG_KNOWHERE_INFO_ << "End of preparing diskann index.";
    }

    if (node_list.size() > 0) {
        const auto size_before_cache = pq_flash_index_->cal_size();
        if (TryDiskANNCall([&]() { pq_flash_index_->load_cache_list(node_list); }) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load cache for DiskANN.";
            return Status::diskann_inner_error;
        }
        const auto size_after_cache = pq_flash_index_->cal_size();
        uint64_t cache_list_fingerprint = 1469598103934665603ULL;
        for (const auto node_id : node_list) {
            cache_list_fingerprint ^= node_id;
            cache_list_fingerprint *= 1099511628211ULL;
        }
        LOG_KNOWHERE_INFO_ << "DiskANN node cache: nodes=" << node_list.size()
                           << ", bytes=" << size_after_cache - size_before_cache
                           << ", list_fingerprint=" << cache_list_fingerprint;
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
            LOG_KNOWHERE_ERROR_ << "Failed to load warmup file for DiskANN.";
            return Status::disk_file_error;
        }
        std::vector<int64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<DistType> warmup_result_dists(warmup_num, 0);

        std::vector<folly::Future<folly::Unit>> futures;
        futures.reserve(warmup_num);
        for (uint64_t i = 0; i < warmup_num; ++i) {
            futures.emplace_back(search_pool_->push([&, index = i]() {
                auto navigation = navigation_store_ ? navigation_store_->CreateDistanceComputer(prep_conf) : nullptr;
                pq_flash_index_->cached_beam_search(
                    warmup + (index * warmup_aligned_dim), 1, warmup_L, warmup_result_ids_64.data() + (index * 1),
                    warmup_result_dists.data() + (index * 1), 4, false, nullptr, nullptr, {}, -1.0f, navigation.get());
            }));
        }

        bool failed = TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success;

        if (warmup != nullptr) {
            diskann::aligned_free(warmup);
        }

        if (failed) {
            LOG_KNOWHERE_ERROR_ << "Failed to do search on warmup file for DiskANN.";
            return Status::diskann_inner_error;
        }
    }

    loaded_navigation_codec_ = detected.value();
    is_prepared_.store(true);
    LOG_KNOWHERE_INFO_ << "End of diskann loading.";
    return Status::success;
}

template <typename DataType>
Status
DiskANNIndexNode<DataType>::DeserializeEmbListIfNeed(const BinarySet& binset, std::shared_ptr<Config> cfg) {
    std::scoped_lock lock(preparation_lock_);
    auto& config = static_cast<BaseConfig&>(*cfg);
    auto el_metric_type_or = get_el_metric_type(config.metric_type.value());
    if (!el_metric_type_or.has_value()) {
        // If not emb_list metric type, use the default deserialize method
        return Deserialize(binset, std::move(cfg));
    }
    if (UsesExternalNavigation(static_cast<const DiskANNConfig&>(*cfg))) {
        LOG_KNOWHERE_ERROR_ << "DISKANN_RABITQ does not support embedding-list mode";
        return Status::not_implemented;
    }

    LOG_KNOWHERE_INFO_ << "Deserialize emb_list index and read emb_list offset from file.";

    // Step 1: Split metric_type into el_metric_type and sub_metric_type
    el_metric_type_ = el_metric_type_or.value();
    auto sub_metric_type_or = get_sub_metric_type(config.metric_type.value());
    if (!sub_metric_type_or.has_value()) {
        LOG_KNOWHERE_ERROR_ << "Invalid sub metric type: " << config.metric_type.value();
        return Status::emb_list_inner_error;
    }
    config.metric_type = sub_metric_type_or.value();

    // Step 2: Deserialize base index with sub_metric_type
    RETURN_IF_ERROR(Deserialize(binset, cfg));

    // Step 3: Deserialize emb_list offset from file
    // Note: emb_list_offset_file is in optional files list, but for emb_list metric type it should exist
    const auto emb_list_offset_file = diskann::get_emb_list_offset_file(index_prefix_);
    auto is_exist_op = file_manager_->IsExisted(emb_list_offset_file);
    if (!is_exist_op.has_value()) {
        LOG_KNOWHERE_ERROR_ << "Failed to check existence of emb_list offset file: " << emb_list_offset_file;
        return Status::emb_list_inner_error;
    }
    if (!is_exist_op.value()) {
        LOG_KNOWHERE_ERROR_ << "Emb_list offset file does not exist: " << emb_list_offset_file;
        return Status::emb_list_inner_error;
    }
    if (!LoadFile(emb_list_offset_file)) {
        LOG_KNOWHERE_ERROR_ << "Failed to load emb_list offset file: " << emb_list_offset_file;
        return Status::disk_file_error;
    }

    std::vector<size_t> offset;
    RETURN_IF_ERROR(ReadEmbListOffsetFromFile(emb_list_offset_file, offset));
    if (offset.empty() || offset.front() != 0) {
        LOG_KNOWHERE_ERROR_ << "Invalid emb_list offset data (expect first offset = 0), file: " << emb_list_offset_file;
        return Status::emb_list_inner_error;
    }
    LOG_KNOWHERE_INFO_ << "Read emb_list offset from file: " << emb_list_offset_file << ", size: " << offset.size()
                       << ", first offset: " << offset.front() << ", last offset: " << offset.back();

    // Step 4: Create strategy and set offset directly
    auto strategy_or = CreateEmbListStrategy(meta::EMB_LIST_STRATEGY_TOKENANN, config);
    if (!strategy_or.has_value()) {
        LOG_KNOWHERE_ERROR_ << "Failed to create emb_list strategy";
        return strategy_or.error();
    }
    emb_list_strategy_ = std::move(strategy_or.value());

    emb_list_offset_ = std::make_shared<EmbListOffset>(std::move(offset));
    RETURN_IF_ERROR(emb_list_strategy_->SetEmbListOffset(emb_list_offset_));
    LOG_KNOWHERE_INFO_ << "Created emb_list strategy: " << emb_list_strategy_->Type()
                       << ", doc_count=" << emb_list_strategy_->GetDocCount();

    return Status::success;
}

template <typename DataType>
expected<std::vector<IndexNode::IteratorPtr>>
DiskANNIndexNode<DataType>::AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                                        bool use_knowhere_search_pool, milvus::OpContext* op_context) const {
    if (navigation_store_) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::not_implemented,
                                                                  "DISKANN_RABITQ does not support iterator search");
    }
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::empty_index, "DiskANN not loaded");
    }

    const auto& search_conf = static_cast<const DiskANNConfig&>(*cfg);
    if (!CheckMetric(search_conf.metric_type.value())) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::invalid_metric_type,
                                                                  "unsupported metric type");
    }

    constexpr uint64_t k_lsearch_iterator = 32;
    auto lsearch = static_cast<uint64_t>(search_conf.search_list_size.value_or(k_lsearch_iterator));
    auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth.value());
    auto filter_ratio = static_cast<float>(search_conf.filter_threshold.value());

    auto nq = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto xq = dataset->GetTensor();

    auto vec = std::vector<IndexNode::IteratorPtr>(nq, nullptr);
    auto metric = search_conf.metric_type.value();
    bool transform = metric != knowhere::metric::L2;
    const auto& id_map = this->GetIdMap();
    const auto* result_id_map = this->SearchResultIdMap(id_map);

    try {
        for (int i = 0; i < nq; i++) {
            auto single_query = static_cast<const DataType*>(xq) + i * dim;
            auto it = std::make_shared<iterator>(transform, single_query, lsearch, beamwidth, filter_ratio, bitset,
                                                 pq_flash_index_.get(), use_knowhere_search_pool);
            it->SetResultIdMap(result_id_map);
            vec[i] = it;
        }
    } catch (const std::exception& e) {
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::diskann_inner_error, e.what());
    }

    return vec;
}

template <typename DataType>
expected<DataSetPtr>
DiskANNIndexNode<DataType>::Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset_,
                                   milvus::OpContext* op_context) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return expected<DataSetPtr>::Err(Status::empty_index, "DiskANN not loaded");
    }

    const auto& search_conf = static_cast<const DiskANNConfig&>(*cfg);
    if (!CheckMetric(search_conf.metric_type.value())) {
        return expected<DataSetPtr>::Err(Status::invalid_metric_type, "unsupported metric type");
    }
    auto k = static_cast<uint64_t>(search_conf.k.value());
    auto lsearch = static_cast<uint64_t>(search_conf.search_list_size.value());
    auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth.value());
    auto filter_ratio = static_cast<float>(search_conf.filter_threshold.value());
    auto nq = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto xq = static_cast<const DataType*>(dataset->GetTensor());

    feder::diskann::FederResultUniq feder_result;
    if (search_conf.trace_visit.value()) {
        if (nq != 1) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "nq must be 1");
        }
        feder_result = std::make_unique<feder::diskann::FederResult>();
        feder_result->visit_info_.SetQueryConfig(search_conf.k.value(), search_conf.beamwidth.value(),
                                                 search_conf.search_list_size.value(), search_conf.beamwidth.value());
    }

    auto p_id = std::make_unique<int64_t[]>(k * nq);
    auto p_dist = std::make_unique<DistType[]>(k * nq);
    std::vector<diskann::QueryStats> query_stats(nq);

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.emplace_back(search_pool_->push([&, index = row, p_id_ptr = p_id.get(), p_dist_ptr = p_dist.get()]() {
            knowhere::checkCancellation(op_context);
            auto& stats = query_stats[index];
            auto navigation = navigation_store_ ? navigation_store_->CreateDistanceComputer(search_conf) : nullptr;
            pq_flash_index_->cached_beam_search(xq + (index * dim), k, lsearch, p_id_ptr + (index * k),
                                                p_dist_ptr + (index * k), beamwidth, false, &stats, feder_result,
                                                bitset_, filter_ratio, navigation.get());
#ifdef NOT_COMPILE_FOR_SWIG
            knowhere_diskann_search_hops.Observe(stats.n_hops);
#endif
        }));
    }

    if (TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success) {
        return expected<DataSetPtr>::Err(Status::diskann_inner_error, "some search failed");
    }

    if (navigation_store_) {
        uint64_t estimates = 0;
        uint64_t refinements = 0;
        uint64_t pruned = 0;
        for (const auto& stats : query_stats) {
            estimates += stats.n_approx_estimates;
            refinements += stats.n_approx_refinements;
            pruned += stats.n_approx_pruned;
        }
        const double prune_ratio = estimates == 0 ? 0.0 : static_cast<double>(pruned) / estimates;
        LOG_KNOWHERE_DEBUG_ << "DiskANN navigation refinement stats: queries=" << nq << ", estimates=" << estimates
                            << ", full_distances=" << refinements << ", pruned=" << pruned
                            << ", prune_ratio=" << prune_ratio;
    }

    {
        double total_us = 0.0;
        double cpu_us = 0.0;
        double io_us = 0.0;
        uint64_t n_ios = 0;
        uint64_t n_cache_hits = 0;
        for (const auto& stats : query_stats) {
            total_us += stats.total_us;
            cpu_us += stats.cpu_us;
            io_us += stats.io_us;
            n_ios += stats.n_ios;
            n_cache_hits += stats.n_cache_hits;
        }
        const double n = static_cast<double>(nq);
        LOG_KNOWHERE_DEBUG_ << "DiskANN search stats: queries=" << nq << ", avg_total_us=" << total_us / n
                            << ", avg_cpu_us=" << cpu_us / n << ", avg_io_us=" << io_us / n
                            << ", avg_n_ios=" << n_ios / n << ", avg_cache_hits=" << n_cache_hits / n;
    }

    auto res = GenResultDataSet(nq, k, std::move(p_id), std::move(p_dist));
    MapSearchResultIdsToOutIds(res);

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

template <typename DataType>
expected<DataSetPtr>
DiskANNIndexNode<DataType>::CalcDistByStorageIds(const DataSetPtr dataset, const BitsetView& bitset,
                                                 const int64_t* labels, const size_t labels_len, const bool is_cosine,
                                                 milvus::OpContext* op_context) const {
    (void)bitset;
    (void)is_cosine;
    if (dataset == nullptr || dataset->GetTensor() == nullptr) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "empty query dataset");
    }
    if (labels == nullptr && labels_len != 0) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "labels is nullptr");
    }
    if (labels_len > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "labels_len overflow");
    }

    auto nq = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto xq = static_cast<const DataType*>(dataset->GetTensor());
    auto p_dist = std::make_unique<DistType[]>(nq * labels_len);
    // CalcDistByStorageIds is a refine/rerank primitive: labels are already in the
    // backend compact id domain. Do not apply nullable public-id mapping here.
    const int64_t* labels_to_calc = labels;
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return expected<DataSetPtr>::Err(Status::empty_index, "DiskANN not loaded");
    }
    if (!search_pool_) {
        LOG_KNOWHERE_ERROR_ << "Search thread pool is not initialized.";
        return expected<DataSetPtr>::Err(Status::internal_error, "search pool not initialized");
    }

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.emplace_back(search_pool_->push([&, index = row, p_dist_ptr = p_dist.get()]() {
            knowhere::checkCancellation(op_context);
            pq_flash_index_->calc_dist_by_ids(xq + (index * dim), labels_to_calc, static_cast<int64_t>(labels_len),
                                              p_dist_ptr + index * labels_len);
        }));
    }
    if (TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success) {
        return expected<DataSetPtr>::Err(Status::diskann_inner_error, "some calc dist by ids failed");
    }

    std::unique_ptr<int64_t[]> ids = nullptr;
    return GenResultDataSet(nq, labels_len, std::move(ids), std::move(p_dist));
}

/*
 * Get raw vector data given their ids.
 * It first tries to get data from cache, if failed, it will try to get data from disk.
 * It reads as much as possible and it is thread-pool free, it totally depends on the outside to control concurrency.
 */
template <typename DataType>
expected<DataSetPtr>
DiskANNIndexNode<DataType>::GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const {
    if (dataset == nullptr) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "GetVectorByIds dataset is null");
    }
    auto rows = dataset->GetRows();
    auto ids = dataset->GetIds();
    std::vector<int64_t> in_ids;
    auto status = this->CompactOutToIn(ids, rows, in_ids);
    if (status != Status::success) {
        return expected<DataSetPtr>::Err(status, "GetVectorByIds failed to map ids");
    }
    ids = in_ids.data();
    rows = static_cast<int64_t>(in_ids.size());
    if (rows == 0) {
        // Public-id retrieve may compact missing nullable ids to an empty
        // storage-id set. Return an empty vector result without touching
        // backend storage.
        return GenResultDataSet(0, Dim(), nullptr);
    }
    auto storage_ds = GenIdsDataSet(rows, ids);
    return GetVectorByStorageIds(storage_ds, op_context);
}

template <typename DataType>
expected<DataSetPtr>
DiskANNIndexNode<DataType>::GetVectorByStorageIds(const DataSetPtr dataset, milvus::OpContext* op_context) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
    }
    if (pq_flash_index_->uses_disk_pq()) {
        return expected<DataSetPtr>::Err(Status::not_implemented, "SSD PQ does not retain original vectors");
    }
    auto dim = Dim();
    auto rows = dataset->GetRows();
    auto ids = dataset->GetIds();
    auto* data = new DataType[dim * rows];
    if (data == nullptr) {
        LOG_KNOWHERE_ERROR_ << "Failed to allocate memory for data.";
        return expected<DataSetPtr>::Err(Status::malloc_error, "failed to allocate memory for data");
    }

    if (TryDiskANNCall([&]() { pq_flash_index_->get_vector_by_ids(ids, rows, data); }) != Status::success) {
        delete[] data;
        return expected<DataSetPtr>::Err(Status::diskann_inner_error, "failed to get vector");
    };

    return GenResultDataSet(rows, dim, data);
}

template <typename DataType>
expected<DataSetPtr>
DiskANNIndexNode<DataType>::GetIndexMeta(std::unique_ptr<Config> cfg) const {
    std::vector<int64_t> entry_points;
    entry_points.reserve(pq_flash_index_->get_num_medoids());
    for (size_t i = 0; i < pq_flash_index_->get_num_medoids(); i++) {
        entry_points.push_back(pq_flash_index_->get_medoids()[i]);
    }
    auto diskann_conf = static_cast<const DiskANNConfig&>(*cfg);
    feder::diskann::DiskANNMeta meta(diskann_conf.data_path.value(), diskann_conf.max_degree.value(),
                                     diskann_conf.search_list_size.value(), diskann_conf.pq_code_budget_gb.value(),
                                     diskann_conf.build_dram_budget_gb.value(), diskann_conf.disk_pq_dims.value(),
                                     diskann_conf.accelerate_build.value(), Count(), entry_points);
    std::unordered_set<int64_t> id_set(entry_points.begin(), entry_points.end());

    Json json_meta, json_id_set;
    nlohmann::to_json(json_meta, meta);
    nlohmann::to_json(json_id_set, id_set);
    return GenResultDataSet(json_meta.dump(), json_id_set.dump());
}

template <typename DataType>
uint64_t
DiskANNIndexNode<DataType>::GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim, size_t chunk_size,
                                             const uint64_t max_degree) {
    uint32_t one_cached_node_budget = (max_degree + 1) * sizeof(unsigned) + chunk_size * data_dim;
    auto num_nodes_to_cache =
        static_cast<uint64_t>(1024 * 1024 * 1024 * cache_dram_budget) / (one_cached_node_budget * kCacheExpansionRate);
    return num_nodes_to_cache;
}

template <typename DataType>
class DiskANNRaBitQIndexNode : public DiskANNIndexNode<DataType> {
 public:
    using DiskANNIndexNode<DataType>::DiskANNIndexNode;

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<DiskANNRaBitQConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_DISKANN_RABITQ;
    }
};

#ifdef KNOWHERE_WITH_CARDINAL
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(DISKANN_DEPRECATED, DiskANNIndexNode,
                                                knowhere::feature::DISK | knowhere::feature::EMB_LIST)
#else
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(DISKANN, DiskANNIndexNode,
                                                knowhere::feature::DISK | knowhere::feature::EMB_LIST)
#endif
KNOWHERE_SIMPLE_REGISTER_GLOBAL(DISKANN_RABITQ, DiskANNRaBitQIndexNode, fp32,
                                knowhere::feature::DISK | knowhere::feature::FLOAT32)
}  // namespace knowhere
