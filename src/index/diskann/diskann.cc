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

#include <cstdint>
#include <fstream>
#include <limits>

#include "diskann/aux_utils.h"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/file_index_reader.h"
#include "diskann/pq_flash_index.h"
#include "filemanager/FileManager.h"
#include "fmt/core.h"
#include "index/diskann/diskann_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/context.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
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

    // Friend helper for test access to pq_flash_index_
    template <typename T>
    friend uint64_t GetDiskANNNodeSectorOffsetForTest(knowhere::Index<knowhere::IndexNode>& index, uint64_t node_id);
    template <typename T>
    friend char* GetDiskANNOffsetToNodeForTest(knowhere::Index<knowhere::IndexNode>& index, char* sector_buf, uint64_t node_id);
    template <typename T>
    friend uint64_t GetDiskANNMaxNodeLenForTest(knowhere::Index<knowhere::IndexNode>& index);
    template <typename T>
    friend size_t GetDiskANNReadLenForNodeForTest(knowhere::Index<knowhere::IndexNode>& index);

 public:
    using DistType = float;
    DiskANNIndexNode(const int32_t& version, const Object& object) : is_prepared_(false), dim_(-1), count_(-1) {
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

    Status
    SetInternalIdToMostExternalIdMap(std::vector<uint32_t>&& map) override {
        internal_id_to_most_external_id_map_ = std::move(map);
        return Status::success;
    }

    expected<DataSetPtr>
    CalcDistByIDs(const DataSetPtr dataset, const BitsetView& bitset, const int64_t* labels, const size_t labels_len,
                  const bool is_cosine, milvus::OpContext* op_context) const override;

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
        if(config.ncs_enable.value()){
            LOG_KNOWHERE_DEBUG_ << "DiskANN configured to use NCS, disk cost is estimated as zero.";
            return Resource{file_size_in_bytes / 4, 0};
        }
        else{
            return Resource{file_size_in_bytes / 4, file_size_in_bytes};
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

    std::vector<std::string>
    ListFilesForNcsUpload() const override{
        return std::vector<std::string>{diskann::get_disk_index_data_filename("")};
    }

    milvus::NcsStatus
    NcsUpload(std::shared_ptr<Config> cfg) override{
        auto conf = static_cast<const DiskANNConfig&>(*cfg);
        size_t count;
        size_t dim;
        diskann::get_bin_metadata(conf.data_path.value(), count, dim);

        assert(conf.ncs_enable.value());
        const milvus::NcsDescriptor* descriptor = &(conf.ncs_descriptor.value());

        auto connector = std::unique_ptr<milvus::NcsConnector>(
            milvus::NcsConnectorFactory::Instance().createConnector(descriptor));
        
        if(!connector){
            LOG_KNOWHERE_ERROR_ << "Failed to create NcsConnector for NCS upload.";
            return milvus::NcsStatus::ERROR;
        }

        std::string index_prefix = std::string(conf.index_prefix.value().c_str());
        std::string disk_index_data_filename = diskann::get_disk_index_data_filename(index_prefix);
        
        // Initialize pq_flash_index_ if not already initialized
        if (!pq_flash_index_) {
            auto diskann_metric = GetDiskANNMetric(conf.metric_type.value());
            std::shared_ptr<IndexReader> reader = nullptr;
            pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<DataType>>(reader, diskann_metric);
        }
        
        std::string disk_index_metadata_filename = diskann::get_disk_index_metadata_filename(index_prefix);
        pq_flash_index_->load_metadata(disk_index_metadata_filename, disk_index_data_filename, true, false);

        // Use actual node size instead of sector-aligned size since FileIndexReader handles alignment
        uint64_t max_node_len = pq_flash_index_->get_max_node_len();
        
        // Use data file instead of combined index file
        std::unique_ptr<IndexReader> reader = std::make_unique<FileIndexReader>(
            disk_index_data_filename, 
            [this](size_t n) { return pq_flash_index_->get_node_sector_offset(n); },
            [this](char* sector_buf, uint64_t node_id) { return pq_flash_index_->get_offset_to_node(sector_buf, node_id); },
            pq_flash_index_->get_read_len_for_node()
        );

        if(!reader){
            LOG_KNOWHERE_ERROR_ << "Failed to create IndexReader for NCS upload.";
            return milvus::NcsStatus::ERROR;
        }

        int batch_size = diskann::defaults::MAX_GRAPH_DEGREE;

        std::vector<std::vector<char>> buffers(batch_size, std::vector<char>(max_node_len));

        for(uint64_t i=0 ; i < count ; i+=batch_size){
            std::vector<ReadReq> reqs;
            int batch_size_i;
            if(i+batch_size > count){
                batch_size_i = count % batch_size;
            } else {
                batch_size_i = batch_size;
            }

            for(int j=0 ; j < batch_size_i ; j++){
                reqs.emplace_back((uint64_t)(i+j), max_node_len, (void*)buffers[j].data());
            }
            reader->read(reqs);

            std::vector<uint32_t> keys;
            std::vector<milvus::SpanBytes> valueSpans;
            for(uint32_t j=0 ; j < batch_size_i ; j++){
                auto key = i + j;
                keys.push_back(key);
                valueSpans.emplace_back(reqs[j].buf, max_node_len);
            }

            auto putResults = connector->multiPut(keys, valueSpans);
        }


        return milvus::NcsStatus::OK;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<DiskANNConfig>();
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
        return pq_flash_index_->cal_size();
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
    GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim, const uint64_t max_degree);

    diskann::Metric
    GetDiskANNMetric(const std::string& metric_type) const;

    std::string index_prefix_;
    mutable std::mutex preparation_lock_;
    std::atomic_bool is_prepared_;
    std::shared_ptr<milvus::FileManager> file_manager_;
    std::unique_ptr<diskann::PQFlashIndex<DataType>> pq_flash_index_;
    std::atomic_int64_t dim_;
    std::atomic_int64_t count_;
    std::shared_ptr<ThreadPool> search_pool_;
    std::vector<uint32_t> internal_id_to_most_external_id_map_;  // for 1-hop bitset check
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
GetNecessaryFilenames(const std::string& prefix, const bool need_norm, const bool use_sample_cache,
                      const bool use_sample_warmup) {
    std::vector<std::string> filenames;
    auto pq_pivots_filename = diskann::get_pq_pivots_filename(prefix);
    auto disk_index_metadata_filename = diskann::get_disk_index_metadata_filename(prefix);
    auto disk_index_data_filename = diskann::get_disk_index_data_filename(prefix);

    filenames.push_back(pq_pivots_filename);
    filenames.push_back(diskann::get_pq_rearrangement_perm_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_chunk_offsets_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_centroid_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_compressed_filename(prefix));
    filenames.push_back(disk_index_metadata_filename);
    filenames.push_back(disk_index_data_filename);
    if (need_norm) {
        filenames.push_back(diskann::get_disk_index_max_base_norm_file(prefix));
    }
    if (use_sample_cache || use_sample_warmup) {
        filenames.push_back(diskann::get_sample_data_filename(prefix));
    }
    return filenames;
}

std::vector<std::string>
GetOptionalFilenames(const std::string& prefix) {
    std::vector<std::string> filenames;
    filenames.push_back(diskann::get_disk_index_centroids_filename(prefix));
    filenames.push_back(diskann::get_disk_index_medoids_filename(prefix));
    filenames.push_back(diskann::get_cached_nodes_file(prefix));
    filenames.push_back(diskann::get_emb_list_offset_file(prefix));
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
    return file_exist(GetNecessaryFilenames(index_prefix, diskann::INNER_PRODUCT, true, true)) ||
           file_exist(GetOptionalFilenames(index_prefix));
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
diskann::Metric
DiskANNIndexNode<DataType>::GetDiskANNMetric(const std::string& metric_type) const {
    if (IsMetricType(metric_type, knowhere::metric::L2)) {
        return diskann::Metric::L2;
    } else if (IsMetricType(metric_type, knowhere::metric::COSINE)) {
        return diskann::Metric::COSINE;
    } else {
        return diskann::Metric::INNER_PRODUCT;
    }
}

template <typename DataType>
Status
DiskANNIndexNode<DataType>::Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    assert(file_manager_ != nullptr);
    auto build_conf = static_cast<const DiskANNConfig&>(*cfg);
    if (!CheckMetric(build_conf.metric_type.value())) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << build_conf.metric_type.value();
        return Status::invalid_metric_type;
    }
    if (!(build_conf.index_prefix.has_value() && build_conf.data_path.has_value())) {
        LOG_KNOWHERE_ERROR_ << "DiskANN file path for build is empty." << std::endl;
        return Status::invalid_param_in_json;
    }
    if (AnyIndexFileExist(build_conf.index_prefix.value())) {
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
    count_.store(count);
    dim_.store(dim);

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
    auto num_nodes_to_cache =
        GetCachedNodeNum(build_conf.search_cache_budget_gb.value(), dim, build_conf.max_degree.value());
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
    RETURN_IF_ERROR(TryDiskANNCall([&]() {
        int res = diskann::build_disk_index<DataType>(diskann_internal_build_config);
        if (res != 0)
            throw diskann::ANNException("diskann::build_disk_index returned non-zero value: " + std::to_string(res),
                                        -1);
    }));

    // Add file to the file manager
    for (auto& filename : GetNecessaryFilenames(index_prefix_, need_norm, true, true)) {
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

    is_prepared_.store(false);
    return Status::success;
}

template <typename DataType>
Status
DiskANNIndexNode<DataType>::BuildEmbListIfNeed(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                                               bool use_knowhere_build_pool) {
    assert(file_manager_ != nullptr);
    std::lock_guard<std::mutex> lock(preparation_lock_);
    auto& config = static_cast<BaseConfig&>(*cfg);
    auto el_metric_type_or = get_el_metric_type(config.metric_type.value());
    if (!el_metric_type_or.has_value()) {
        // If not emb_list metric type, use the default build method
        return Build(dataset, std::move(cfg), use_knowhere_build_pool);
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

    auto build_status = BuildEmbList(dataset, std::move(cfg), offset.data(), offset.back(), use_knowhere_build_pool);
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
    auto prep_conf = static_cast<const DiskANNConfig&>(*cfg);
    if (!CheckMetric(prep_conf.metric_type.value())) {
        return Status::invalid_metric_type;
    }
    if (is_prepared_.load()) {
        return Status::success;
    }
    if (!(prep_conf.index_prefix.has_value())) {
        LOG_KNOWHERE_ERROR_ << "DiskANN file path for deserialize is empty." << std::endl;
        return Status::invalid_param_in_json;
    }
    index_prefix_ = prep_conf.index_prefix.value();
    bool is_ip = IsMetricType(prep_conf.metric_type.value(), knowhere::metric::IP);
    bool need_norm = IsMetricType(prep_conf.metric_type.value(), knowhere::metric::IP) ||
                    IsMetricType(prep_conf.metric_type.value(), knowhere::metric::COSINE);
    auto diskann_metric = GetDiskANNMetric(prep_conf.metric_type.value());

    // Load file from file manager.
    for (auto& filename : GetNecessaryFilenames(
             index_prefix_, need_norm, prep_conf.search_cache_budget_gb.value() > 0 && !prep_conf.use_bfs_cache.value(),
             prep_conf.warm_up.value())) {
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

    // set thread pool
    search_pool_ = ThreadPool::GetGlobalSearchThreadPool();

    const milvus::NcsDescriptor* descriptor = nullptr;
    bool use_ncs = prep_conf.ncs_enable.value();
    if(use_ncs){
        descriptor = &(prep_conf.ncs_descriptor.value());
    }

    // load diskann pq code and meta info
    std::shared_ptr<IndexReader> reader = nullptr;

    pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<DataType>>(reader, diskann_metric);
    auto disk_ann_call = [&]() {
        int res = pq_flash_index_->load(search_pool_->size(), index_prefix_.c_str(), use_ncs, descriptor);
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
        auto num_nodes_to_cache = GetCachedNodeNum(prep_conf.search_cache_budget_gb.value(),
                                                   pq_flash_index_->get_data_dim(), pq_flash_index_->get_max_degree());
        if (num_nodes_to_cache > pq_flash_index_->get_num_points() / 3) {
            LOG_KNOWHERE_ERROR_ << "Failed to generate cache, num_nodes_to_cache(" << num_nodes_to_cache
                                << ") is larger than 1/3 of the total data number.";
            return Status::invalid_args;
        }
        if (num_nodes_to_cache > 0) {
            LOG_KNOWHERE_INFO_ << "Caching " << num_nodes_to_cache << " sample nodes around medoid(s).";
            if (prep_conf.use_bfs_cache.value()) {
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
        if (TryDiskANNCall([&]() { pq_flash_index_->load_cache_list(node_list); }) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load cache for DiskANN.";
            return Status::diskann_inner_error;
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
            LOG_KNOWHERE_ERROR_ << "Failed to load warmup file for DiskANN.";
            return Status::disk_file_error;
        }
        std::vector<int64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<DistType> warmup_result_dists(warmup_num, 0);

        std::vector<folly::Future<folly::Unit>> futures;
        futures.reserve(warmup_num);
        for (_s64 i = 0; i < (int64_t)warmup_num; ++i) {
            futures.emplace_back(search_pool_->push([&, index = i]() {
                pq_flash_index_->cached_beam_search(warmup + (index * warmup_aligned_dim), 1, warmup_L,
                                                    warmup_result_ids_64.data() + (index * 1),
                                                    warmup_result_dists.data() + (index * 1), 4);
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

    is_prepared_.store(true);
    LOG_KNOWHERE_INFO_ << "End of diskann loading.";
    return Status::success;
}

template <typename DataType>
Status
DiskANNIndexNode<DataType>::DeserializeEmbListIfNeed(const BinarySet& binset, std::shared_ptr<Config> cfg) {
    std::lock_guard<std::mutex> lock(preparation_lock_);
    auto& config = static_cast<BaseConfig&>(*cfg);
    auto el_metric_type_or = get_el_metric_type(config.metric_type.value());
    if (!el_metric_type_or.has_value()) {
        // If not emb_list metric type, use the default deserialize method
        return Deserialize(binset, std::move(cfg));
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

    emb_list_offset_ = std::make_unique<EmbListOffset>(std::move(offset));

    // Step 4: Set base index id map for 1-hop bitset check
    return SetBaseIndexIDMap();
}

template <typename DataType>
expected<std::vector<IndexNode::IteratorPtr>>
DiskANNIndexNode<DataType>::AnnIterator(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                                        bool use_knowhere_search_pool, milvus::OpContext* op_context) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return expected<std::vector<IndexNode::IteratorPtr>>::Err(Status::empty_index, "DiskANN not loaded");
    }

    auto search_conf = static_cast<const DiskANNConfig&>(*cfg);
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

    try {
        for (int i = 0; i < nq; i++) {
            auto single_query = (DataType*)xq + i * dim;
            auto it = std::make_shared<iterator>(transform, single_query, lsearch, beamwidth, filter_ratio, bitset,
                                                 pq_flash_index_.get(), use_knowhere_search_pool);
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

    auto search_conf = static_cast<const DiskANNConfig&>(*cfg);
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

    BitsetView bitset(bitset_);
    if (!internal_id_to_most_external_id_map_.empty()) {
        if (emb_list_offset_ != nullptr) {
            // if emb list, manually calculate the number of filtered out ids
            size_t num_filtered_out_ids = 0;
            const auto num_el = std::min(
                bitset.size(), emb_list_offset_->offset.empty() ? size_t{0} : emb_list_offset_->offset.size() - 1);
            if (emb_list_offset_->offset.size() < bitset.size() + 1) {
                LOG_KNOWHERE_WARNING_ << "Bitset size(" << bitset.size() << ") doesn't match emb_list offset size("
                                      << emb_list_offset_->offset.size()
                                      << "), will compute filtered ids using min size(" << num_el << ")";
            }
            for (size_t i = 0; i < num_el; i++) {
                if (bitset.test(i)) {
                    num_filtered_out_ids += emb_list_offset_->offset[i + 1] - emb_list_offset_->offset[i];
                }
            }
            bitset.set_out_ids(internal_id_to_most_external_id_map_.data(), internal_id_to_most_external_id_map_.size(),
                               num_filtered_out_ids);
        } else {
            bitset.set_out_ids(internal_id_to_most_external_id_map_.data(),
                               internal_id_to_most_external_id_map_.size());
        }
    }

    auto p_id = std::make_unique<int64_t[]>(k * nq);
    auto p_dist = std::make_unique<DistType[]>(k * nq);

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.emplace_back(search_pool_->push([&, index = row, p_id_ptr = p_id.get(), p_dist_ptr = p_dist.get()]() {
            knowhere::checkCancellation(op_context);
            diskann::QueryStats stats;
            pq_flash_index_->cached_beam_search(xq + (index * dim), k, lsearch, p_id_ptr + (index * k),
                                                p_dist_ptr + (index * k), beamwidth, false, &stats, feder_result,
                                                bitset, filter_ratio);
#ifdef NOT_COMPILE_FOR_SWIG
            knowhere_diskann_search_hops.Observe(stats.n_hops);
#endif
        }));
    }

    if (TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success) {
        return expected<DataSetPtr>::Err(Status::diskann_inner_error, "some search failed");
    }

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

template <typename DataType>
expected<DataSetPtr>
DiskANNIndexNode<DataType>::CalcDistByIDs(const DataSetPtr dataset, const BitsetView& bitset, const int64_t* labels,
                                          const size_t labels_len, const bool is_cosine,
                                          milvus::OpContext* op_context) const {
    (void)bitset;
    (void)is_cosine;
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return expected<DataSetPtr>::Err(Status::empty_index, "DiskANN not loaded");
    }
    if (!search_pool_) {
        LOG_KNOWHERE_ERROR_ << "Search thread pool is not initialized.";
        return expected<DataSetPtr>::Err(Status::internal_error, "search pool not initialized");
    }
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

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.emplace_back(search_pool_->push([&, index = row, p_dist_ptr = p_dist.get()]() {
            knowhere::checkCancellation(op_context);
            pq_flash_index_->calc_dist_by_ids(xq + (index * dim), labels, static_cast<int64_t>(labels_len),
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
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load diskann.";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
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
DiskANNIndexNode<DataType>::GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim,
                                             const uint64_t max_degree) {
    uint32_t one_cached_node_budget = (max_degree + 1) * sizeof(unsigned) + sizeof(DataType) * data_dim;
    auto num_nodes_to_cache =
        static_cast<uint64_t>(1024 * 1024 * 1024 * cache_dram_budget) / (one_cached_node_budget * kCacheExpansionRate);
    return num_nodes_to_cache;
}

#ifdef KNOWHERE_WITH_CARDINAL
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(DISKANN_DEPRECATED, DiskANNIndexNode,
                                                knowhere::feature::DISK | knowhere::feature::EMB_LIST)
#else
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(DISKANN, DiskANNIndexNode,
                                                knowhere::feature::DISK | knowhere::feature::EMB_LIST)
#endif

// Helper function for test access to get_node_sector_offset
template <typename DataType>
uint64_t GetDiskANNNodeSectorOffsetForTest(knowhere::Index<knowhere::IndexNode>& index, uint64_t node_id) {
    auto* diskann_node = dynamic_cast<DiskANNIndexNode<DataType>*>(index.Node());
    if (!diskann_node || !diskann_node->pq_flash_index_) {
        return 0;
    }
    return diskann_node->pq_flash_index_->get_node_sector_offset(node_id);
}

// Helper function for test access to get_offset_to_node
template <typename DataType>
char* GetDiskANNOffsetToNodeForTest(knowhere::Index<knowhere::IndexNode>& index, char* sector_buf, uint64_t node_id) {
    auto* diskann_node = dynamic_cast<DiskANNIndexNode<DataType>*>(index.Node());
    if (!diskann_node || !diskann_node->pq_flash_index_) {
        return nullptr;
    }
    return diskann_node->pq_flash_index_->get_offset_to_node(sector_buf, node_id);
}

// Helper function for test access to get_max_node_len
template <typename DataType>
uint64_t GetDiskANNMaxNodeLenForTest(knowhere::Index<knowhere::IndexNode>& index) {
    auto* diskann_node = dynamic_cast<DiskANNIndexNode<DataType>*>(index.Node());
    if (!diskann_node || !diskann_node->pq_flash_index_) {
        return 0;
    }
    return diskann_node->pq_flash_index_->get_max_node_len();
}

// Helper function for test access to get_read_len_for_node
template <typename T>
size_t GetDiskANNReadLenForNodeForTest(knowhere::Index<knowhere::IndexNode>& index){
    auto* diskann_node = dynamic_cast<DiskANNIndexNode<T>*>(index.Node());
    if (!diskann_node || !diskann_node->pq_flash_index_) {
        return 0;
    }
    return diskann_node->pq_flash_index_->get_read_len_for_node();
}

// Explicit instantiation for fp32
template uint64_t GetDiskANNNodeSectorOffsetForTest<knowhere::fp32>(knowhere::Index<knowhere::IndexNode>&, uint64_t);
template char* GetDiskANNOffsetToNodeForTest<knowhere::fp32>(knowhere::Index<knowhere::IndexNode>&, char*, uint64_t);
template uint64_t GetDiskANNMaxNodeLenForTest<knowhere::fp32>(knowhere::Index<knowhere::IndexNode>&);
template size_t GetDiskANNReadLenForNodeForTest<knowhere::fp32>(knowhere::Index<knowhere::IndexNode>&);

}  // namespace knowhere
