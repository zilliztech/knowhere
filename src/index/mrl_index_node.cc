// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "knowhere/index/mrl_index_node.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "index/data_view_dense_index/data_view_dense_index.h"
#include "knowhere/log.h"

namespace knowhere {
namespace {
constexpr const char* kMRLMeta = "MRL_META";
constexpr const char* kBasePrefix = "MRL_BASE/";
constexpr const char* kRefinePrefix = "MRL_REFINE/";
constexpr uint32_t kMRLMagic = 0x4d524c31;
constexpr uint32_t kMRLVersion = 1;
constexpr size_t kFileBatchRows = 4096;

struct MRLMeta {
    uint32_t magic;
    uint32_t version;
    int64_t source_dim;
    int64_t mrl_dim;
    int32_t data_type;
    uint8_t with_mrl_refine;
};

void
AppendWithPrefix(BinarySet& output, const char* prefix, const BinarySet& input) {
    for (const auto& [name, binary] : input.binary_map_) {
        output.Append(std::string(prefix) + name, binary);
    }
}

BinarySet
ExtractWithPrefix(const BinarySet& input, const char* prefix) {
    BinarySet output;
    const std::string prefix_string(prefix);
    for (const auto& [name, binary] : input.binary_map_) {
        if (name.starts_with(prefix_string)) {
            output.Append(name.substr(prefix_string.size()), binary);
        }
    }
    return output;
}

class TemporaryFile {
 public:
    explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path)) {
    }

    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path&
    Path() const {
        return path_;
    }

 private:
    std::filesystem::path path_;
};

std::filesystem::path
PrefixFilePath(const std::filesystem::path& source) {
    static std::atomic<uint64_t> sequence{0};
    return source.string() + ".mrl." + std::to_string(sequence.fetch_add(1));
}

Status
SetConfigDim(Config* cfg, int64_t dim) {
    auto base_cfg = dynamic_cast<BaseConfig*>(cfg);
    if (base_cfg == nullptr) {
        return Status::invalid_args;
    }
    base_cfg->dim = dim;
    return Status::success;
}
}  // namespace

MRLIndexNode::MRLIndexNode(Index<IndexNode>&& base_index, int64_t source_dim, int64_t mrl_dim, DataFormatEnum data_type,
                           bool with_mrl_refine, ViewDataOp view_data)
    : base_index_(std::move(base_index)),
      source_dim_(source_dim),
      mrl_dim_(mrl_dim),
      data_type_(data_type),
      with_mrl_refine_(with_mrl_refine),
      view_data_(std::move(view_data)) {
    if (base_index_.Node() == nullptr || source_dim_ <= 0 || mrl_dim_ <= 0 || mrl_dim_ >= source_dim_) {
        throw std::invalid_argument("invalid MRL index dimensions or base index");
    }
    if (data_type_ != DataFormatEnum::fp32 && data_type_ != DataFormatEnum::fp16 &&
        data_type_ != DataFormatEnum::bf16) {
        throw std::invalid_argument("MRL index supports only float vector data types");
    }
}

MRLIndexNode::~MRLIndexNode() = default;

size_t
MRLIndexNode::ElementSize() const {
    return data_type_ == DataFormatEnum::fp32 ? sizeof(fp32) : sizeof(fp16);
}

Status
MRLIndexNode::ValidateDataSet(const DataSetPtr& dataset) const {
    if (dataset == nullptr || dataset->GetDim() != source_dim_ ||
        (dataset->GetRows() > 0 && dataset->GetTensor() == nullptr)) {
        return Status::invalid_args;
    }
    return Status::success;
}

DataSetPtr
MRLIndexNode::PreparePrefixDataSet(const DataSetPtr& dataset) const {
    const auto rows = dataset->GetRows();
    const auto source_row_size = source_dim_ * ElementSize();
    const auto prefix_row_size = mrl_dim_ * ElementSize();
    auto prefix = std::make_unique<uint8_t[]>(rows * prefix_row_size);
    auto source = static_cast<const uint8_t*>(dataset->GetTensor());
    for (int64_t row = 0; row < rows; ++row) {
        std::memcpy(prefix.get() + row * prefix_row_size, source + row * source_row_size, prefix_row_size);
    }

    auto output = std::make_shared<DataSet>();
    output->SetRows(rows);
    output->SetDim(mrl_dim_);
    output->SetTensor(std::move(prefix));
    output->SetTensorBeginId(dataset->GetTensorBeginId());
    return output;
}

Status
MRLIndexNode::CreateRefineIndex(const BaseConfig& cfg) {
    if (!with_mrl_refine_) {
        return Status::success;
    }
    if (!cfg.metric_type.has_value()) {
        return Status::invalid_args;
    }

    const auto& source_metric = cfg.metric_type.value();
    const bool is_cosine = IsMetricType(source_metric, metric::COSINE);
    if (!is_cosine && !IsMetricType(source_metric, metric::L2) && !IsMetricType(source_metric, metric::IP)) {
        return Status::invalid_metric_type;
    }
    const auto refine_metric = is_cosine ? metric::IP : source_metric;
    refine_index_ = std::make_shared<DataViewIndexFlat>(source_dim_, data_type_, refine_metric, view_data_, is_cosine,
                                                        RefineType::DATA_VIEW, cfg.num_build_thread);
    return Status::success;
}

Status
MRLIndexNode::Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    auto base_cfg = dynamic_cast<BaseConfig*>(cfg.get());
    if (base_cfg == nullptr) {
        return Status::invalid_args;
    }
    RETURN_IF_ERROR(SetConfigDim(cfg.get(), mrl_dim_));
    if (base_cfg->data_path.has_value()) {
        return BuildFromFile(dataset, std::move(cfg), use_knowhere_build_pool);
    }

    RETURN_IF_ERROR(ValidateDataSet(dataset));
    if (with_mrl_refine_) {
        RETURN_IF_ERROR(CreateRefineIndex(*base_cfg));
    }
    auto prefix_dataset = PreparePrefixDataSet(dataset);
    RETURN_IF_ERROR(base_index_.Node()->Build(prefix_dataset, cfg, use_knowhere_build_pool));
    if (refine_index_ != nullptr) {
        refine_index_->Add(dataset->GetRows(), dataset->GetTensor(), nullptr, use_knowhere_build_pool);
    }
    return Status::success;
}

Status
MRLIndexNode::BuildFromFile(const DataSetPtr& dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    auto& base_cfg = static_cast<BaseConfig&>(*cfg);
    const auto source_path = std::filesystem::path(base_cfg.data_path.value());
    TemporaryFile prefix_file(PrefixFilePath(source_path));
    std::ifstream input(source_path, std::ios::binary);
    std::ofstream output(prefix_file.Path(), std::ios::binary | std::ios::trunc);
    if (!input || !output) {
        return Status::disk_file_error;
    }

    uint32_t rows = 0;
    uint32_t dim = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!input || dim != static_cast<uint32_t>(source_dim_) || mrl_dim_ > std::numeric_limits<uint32_t>::max()) {
        return Status::invalid_args;
    }
    const auto prefix_dim = static_cast<uint32_t>(mrl_dim_);
    output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    output.write(reinterpret_cast<const char*>(&prefix_dim), sizeof(prefix_dim));

    if (with_mrl_refine_) {
        RETURN_IF_ERROR(CreateRefineIndex(base_cfg));
    }
    const auto source_row_size = source_dim_ * ElementSize();
    const auto prefix_row_size = mrl_dim_ * ElementSize();
    std::vector<uint8_t> source_buffer(kFileBatchRows * source_row_size);
    std::vector<uint8_t> prefix_buffer(kFileBatchRows * prefix_row_size);
    for (uint32_t offset = 0; offset < rows;) {
        const auto batch_rows = std::min<uint32_t>(kFileBatchRows, rows - offset);
        input.read(reinterpret_cast<char*>(source_buffer.data()), batch_rows * source_row_size);
        if (!input) {
            return Status::disk_file_error;
        }
        for (uint32_t row = 0; row < batch_rows; ++row) {
            std::memcpy(prefix_buffer.data() + row * prefix_row_size, source_buffer.data() + row * source_row_size,
                        prefix_row_size);
        }
        output.write(reinterpret_cast<const char*>(prefix_buffer.data()), batch_rows * prefix_row_size);
        if (!output) {
            return Status::disk_file_error;
        }
        if (refine_index_ != nullptr) {
            refine_index_->Add(batch_rows, source_buffer.data(), nullptr, use_knowhere_build_pool);
        }
        offset += batch_rows;
    }
    output.close();

    base_cfg.data_path = prefix_file.Path().string();
    return base_index_.Node()->Build(dataset, std::move(cfg), use_knowhere_build_pool);
}

Status
MRLIndexNode::Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    RETURN_IF_ERROR(ValidateDataSet(dataset));
    RETURN_IF_ERROR(SetConfigDim(cfg.get(), mrl_dim_));
    if (with_mrl_refine_) {
        RETURN_IF_ERROR(CreateRefineIndex(static_cast<const BaseConfig&>(*cfg)));
    }
    return base_index_.Node()->Train(PreparePrefixDataSet(dataset), std::move(cfg), use_knowhere_build_pool);
}

Status
MRLIndexNode::Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) {
    RETURN_IF_ERROR(ValidateDataSet(dataset));
    RETURN_IF_ERROR(SetConfigDim(cfg.get(), mrl_dim_));
    if (with_mrl_refine_ && refine_index_ == nullptr) {
        RETURN_IF_ERROR(CreateRefineIndex(static_cast<const BaseConfig&>(*cfg)));
    }
    auto prefix_dataset = PreparePrefixDataSet(dataset);
    RETURN_IF_ERROR(base_index_.Node()->Add(prefix_dataset, std::move(cfg), use_knowhere_build_pool));
    if (refine_index_ != nullptr) {
        refine_index_->Add(dataset->GetRows(), dataset->GetTensor(), nullptr, use_knowhere_build_pool);
    }
    return Status::success;
}

expected<DataSetPtr>
MRLIndexNode::Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                     milvus::OpContext* op_context) const {
    if (ValidateDataSet(dataset) != Status::success) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "MRL query must use the source dimension");
    }
    if (with_mrl_refine_ && (refine_index_ == nullptr || !view_data_)) {
        return expected<DataSetPtr>::Err(Status::empty_index, "MRL raw-data refiner is not bound");
    }
    if (SetConfigDim(cfg.get(), mrl_dim_) != Status::success) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "invalid MRL search config");
    }

    const auto nq = dataset->GetRows();
    const auto topk = static_cast<const BaseConfig&>(*cfg).k.value();
    auto result = base_index_.Node()->Search(PreparePrefixDataSet(dataset), std::move(cfg), bitset, op_context);
    if (!result.has_value() || !with_mrl_refine_) {
        return result;
    }

    const auto candidate_count = result.value()->GetDim();
    std::vector<idx_t> query_limits(nq + 1);
    for (int64_t i = 0; i <= nq; ++i) {
        query_limits[i] = candidate_count * i;
    }
    auto labels = std::make_unique<int64_t[]>(nq * topk);
    auto distances = std::make_unique<float[]>(nq * topk);
    refine_index_->SearchWithIds(nq, dataset->GetTensor(), query_limits.data(), result.value()->GetIds(), topk,
                                 distances.get(), labels.get(), false);
    return GenResultDataSet(nq, topk, std::move(labels), std::move(distances));
}

expected<std::vector<IndexNode::IteratorPtr>>
MRLIndexNode::AnnIterator(const DataSetPtr, std::unique_ptr<Config>, const BitsetView&, bool,
                          milvus::OpContext*) const {
    return expected<std::vector<IteratorPtr>>::Err(Status::not_implemented, "MRL iterator search is not supported");
}

expected<DataSetPtr>
MRLIndexNode::RangeSearch(const DataSetPtr, std::unique_ptr<Config>, const BitsetView&, milvus::OpContext*) const {
    return expected<DataSetPtr>::Err(Status::not_implemented, "MRL range search is not supported");
}

expected<DataSetPtr>
MRLIndexNode::GetVectorByIds(const DataSetPtr, milvus::OpContext*) const {
    return expected<DataSetPtr>::Err(Status::not_implemented, "MRL index does not own source vectors");
}

bool
MRLIndexNode::HasRawData(const std::string&) const {
    return false;
}

bool
MRLIndexNode::IsAdditionalScalarSupported(bool is_mv_only) const {
    return base_index_.Node()->IsAdditionalScalarSupported(is_mv_only);
}

bool
MRLIndexNode::IsIndexRefineEnabled() const {
    return base_index_.Node()->IsIndexRefineEnabled();
}

expected<DataSetPtr>
MRLIndexNode::GetIndexMeta(std::unique_ptr<Config> cfg) const {
    if (SetConfigDim(cfg.get(), mrl_dim_) != Status::success) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "invalid MRL index metadata config");
    }
    return base_index_.Node()->GetIndexMeta(std::move(cfg));
}

Status
MRLIndexNode::Serialize(BinarySet& binset) const {
    MRLMeta meta{};
    meta.magic = kMRLMagic;
    meta.version = kMRLVersion;
    meta.source_dim = source_dim_;
    meta.mrl_dim = mrl_dim_;
    meta.data_type = static_cast<int32_t>(data_type_);
    meta.with_mrl_refine = static_cast<uint8_t>(with_mrl_refine_);
    auto meta_data = std::shared_ptr<uint8_t[]>(new uint8_t[sizeof(meta)]);
    std::memcpy(meta_data.get(), &meta, sizeof(meta));
    binset.Append(kMRLMeta, std::move(meta_data), sizeof(meta));

    BinarySet base_binary;
    RETURN_IF_ERROR(base_index_.Node()->Serialize(base_binary));
    AppendWithPrefix(binset, kBasePrefix, base_binary);
    if (with_mrl_refine_) {
        if (refine_index_ == nullptr) {
            return Status::empty_index;
        }
        BinarySet refine_binary;
        RETURN_IF_ERROR(refine_index_->SerializeState(refine_binary));
        AppendWithPrefix(binset, kRefinePrefix, refine_binary);
    }
    return Status::success;
}

Status
MRLIndexNode::Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) {
    auto meta_binary = binset.GetByName(kMRLMeta);
    if (meta_binary == nullptr || meta_binary->size != static_cast<int64_t>(sizeof(MRLMeta))) {
        return Status::invalid_binary_set;
    }
    MRLMeta meta{};
    std::memcpy(&meta, meta_binary->data.get(), sizeof(meta));
    if (meta.magic != kMRLMagic || meta.version != kMRLVersion || meta.source_dim != source_dim_ ||
        meta.mrl_dim != mrl_dim_ || meta.data_type != static_cast<int32_t>(data_type_) ||
        meta.with_mrl_refine != static_cast<uint8_t>(with_mrl_refine_)) {
        return Status::invalid_binary_set;
    }
    if (with_mrl_refine_ && !view_data_) {
        return Status::invalid_args;
    }

    RETURN_IF_ERROR(SetConfigDim(cfg.get(), mrl_dim_));
    auto base_binary = ExtractWithPrefix(binset, kBasePrefix);
    RETURN_IF_ERROR(base_index_.Node()->Deserialize(base_binary, cfg));
    if (with_mrl_refine_) {
        RETURN_IF_ERROR(CreateRefineIndex(static_cast<const BaseConfig&>(*cfg)));
        auto refine_binary = ExtractWithPrefix(binset, kRefinePrefix);
        RETURN_IF_ERROR(refine_index_->DeserializeState(refine_binary));
    }
    return Status::success;
}

Status
MRLIndexNode::DeserializeFromFile(const std::string&, std::shared_ptr<Config>) {
    return Status::not_implemented;
}

std::unique_ptr<BaseConfig>
MRLIndexNode::CreateConfig() const {
    return base_index_.Node()->CreateConfig();
}

int64_t
MRLIndexNode::Dim() const {
    return source_dim_;
}

int64_t
MRLIndexNode::Size() const {
    return base_index_.Size();
}

int64_t
MRLIndexNode::Count() const {
    return base_index_.Count();
}

std::string
MRLIndexNode::Type() const {
    return base_index_.Type() + "_MRL";
}

bool
MRLIndexNode::LoadIndexWithStream() {
    return base_index_.Node()->LoadIndexWithStream();
}

std::optional<size_t>
MRLIndexNode::GetQueryCodeSize(const DataSetPtr) const {
    return source_dim_ * ElementSize();
}

}  // namespace knowhere
