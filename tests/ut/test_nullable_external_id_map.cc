// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#ifdef KNOWHERE_WITH_CARDINAL
#include "cachinglayer/Manager.h"
#endif
#include "filemanager/impl/LocalFileManager.h"
#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/external_id_map.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"
#include "knowhere/object.h"
#include "knowhere/version.h"
#include "utils.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "Missing the <filesystem> header."
#endif

namespace {

using knowhere::Status;

class NullableWrapperFakeIndexNode : public knowhere::IndexNode {
 public:
    Status
    Train(const knowhere::DataSetPtr dataset, std::shared_ptr<knowhere::Config> cfg,
          bool use_knowhere_build_pool) override {
        return Status::success;
    }

    Status
    Add(const knowhere::DataSetPtr dataset, std::shared_ptr<knowhere::Config> cfg,
        bool use_knowhere_build_pool) override {
        return Status::success;
    }

    knowhere::expected<knowhere::DataSetPtr>
    Search(const knowhere::DataSetPtr dataset, std::unique_ptr<knowhere::Config> cfg,
           const knowhere::BitsetView& bitset, milvus::OpContext* op_context) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented,
                                                             "Search not implemented");
    }

    knowhere::expected<knowhere::DataSetPtr>
    RangeSearch(const knowhere::DataSetPtr dataset, std::unique_ptr<knowhere::Config> cfg,
                const knowhere::BitsetView& bitset, milvus::OpContext* op_context) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented,
                                                             "RangeSearch not implemented");
    }

    knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
    AnnIterator(const knowhere::DataSetPtr dataset, std::unique_ptr<knowhere::Config> cfg,
                const knowhere::BitsetView& bitset, bool use_knowhere_search_pool,
                milvus::OpContext* op_context) const override {
        return knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>::Err(
            Status::not_implemented, "AnnIterator not implemented");
    }

    knowhere::expected<knowhere::DataSetPtr>
    GetVectorByIds(const knowhere::DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented,
                                                             "GetVectorByIds not implemented");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return true;
    }

    knowhere::expected<knowhere::DataSetPtr>
    GetIndexMeta(std::unique_ptr<knowhere::Config> cfg) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented,
                                                             "GetIndexMeta not implemented");
    }

    Status
    Serialize(knowhere::BinarySet& binset) const override {
        return Status::success;
    }

    Status
    Deserialize(const knowhere::BinarySet& binset, std::shared_ptr<knowhere::Config> config) override {
        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<knowhere::Config> config) override {
        return Status::success;
    }

    std::unique_ptr<knowhere::BaseConfig>
    CreateConfig() const override {
        return std::make_unique<knowhere::BaseConfig>();
    }

    int64_t
    Dim() const override {
        return 0;
    }

    int64_t
    Size() const override {
        return 0;
    }

    int64_t
    Count() const override {
        return 0;
    }

    std::string
    Type() const override {
        return "NULLABLE_WRAPPER_FAKE";
    }
};

constexpr int64_t kTotalRows = 32;
constexpr int64_t kDenseDim = 16;
constexpr int64_t kGpuRows = 64;
constexpr int64_t kGpuDim = 32;
constexpr int64_t kFileRows = 64;
constexpr int64_t kFileDim = 32;
constexpr int64_t kEmbDocs = 16;
constexpr int64_t kEmbVectorsPerDoc = 2;
constexpr int64_t kTopK = 3;
constexpr int64_t kEmbTopK = 2;
constexpr int64_t kQueryCount = 3;

enum class DataKind {
    DenseFp32,
    DenseBin,
    Sparse,
    MinHash,
    DiskAnn,
    Aisaq,
    GpuFp32,
};

enum class Mode {
    Vector,
    EmbList,
    MultiIndex,
};

enum class Operation {
    Build,
    Search,
    Range,
    Iterator,
    SearchFilter,
    RangeFilter,
    IteratorFilter,
    BinarySetSerialize,
    BinarySetDeserialize,
    FileSerialize,
    FileDeserialize,
};

enum class IndexSource {
    Fresh,
    BinarySet,
    File,
};

enum class NullableRatio {
    R0,
    R50,
    R100,
};

enum class FilterRatio {
    None,
    R0,
    R50,
    R100,
    Collapsed,
};

struct Capabilities {
    bool build = false;
    bool search = false;
    bool range = false;
    bool iterator = false;
    bool search_filter = false;
    bool range_filter = false;
    bool iterator_filter = false;
    bool binaryset_serialize = false;
    bool binaryset_deserialize = false;
    bool file_serialize = false;
    bool file_deserialize = false;
    bool emb_build = false;
    bool emb_search = false;
    bool emb_range = false;
    bool emb_iterator = false;
    bool emb_search_filter = false;
    bool emb_range_filter = false;
    bool emb_iterator_filter = false;
    bool multi_build = false;
    bool multi_search = false;
    bool multi_range = false;
    bool multi_iterator = false;
    bool multi_search_filter = false;
    bool multi_range_filter = false;
    bool multi_iterator_filter = false;
};

struct IndexRow {
    std::string label;
    std::string index_type;
    DataKind data_kind = DataKind::DenseFp32;
    Capabilities caps;
    bool maybe_unavailable = false;
    bool requires_cardinal = false;
    bool requires_svs = false;
    bool requires_gpu = false;
    bool exact = false;
};

struct Scenario {
    std::string name;
    NullableRatio nullable_ratio = NullableRatio::R0;
    FilterRatio filter_ratio = FilterRatio::None;
    Mode mode = Mode::Vector;
    Operation op = Operation::Build;
    IndexSource source = IndexSource::Fresh;
};

struct MatrixData {
    int64_t total_count = 0;
    int64_t dim = 0;
    std::vector<int32_t> valid_ids;
    std::vector<int32_t> query_ids;
    std::vector<int32_t> selected_ids;
    knowhere::DataSetPtr full_ds;
    knowhere::DataSetPtr train_ds;
    knowhere::DataSetPtr query_ds;
};

struct BuiltArtifact {
    IndexRow row;
    Scenario scenario;
    MatrixData data;
    knowhere::Json json;
    knowhere::Index<knowhere::IndexNode> index;
    knowhere::Index<knowhere::IndexNode> binary_loaded;
    knowhere::Index<knowhere::IndexNode> file_loaded;
    knowhere::BinarySet binset;
    std::shared_ptr<milvus::FileManager> file_manager;
    knowhere::Status create_status = knowhere::Status::success;
    knowhere::Status build_status = knowhere::Status::success;
    knowhere::Status serialize_status = knowhere::Status::success;
    knowhere::Status binary_deserialize_status = knowhere::Status::success;
    knowhere::Status file_serialize_status = knowhere::Status::success;
    knowhere::Status file_deserialize_status = knowhere::Status::success;
    bool create_ok = false;
    bool serialized = false;
    bool binary_loaded_ready = false;
    bool file_serialized = false;
    bool file_loaded_ready = false;
    fs::path work_dir;
    fs::path main_file;
    fs::path id_map_file;
    fs::path emb_meta_file;
    fs::path emb_raw_file;
    fs::path emb_offset_file;
    fs::path raw_data_file;
    fs::path file_index_prefix;
};

int32_t
VersionForRow(const IndexRow& row) {
#ifdef KNOWHERE_WITH_CARDINAL
    if (!row.requires_cardinal &&
        (row.index_type == knowhere::IndexEnum::INDEX_HNSW || row.index_type == knowhere::IndexEnum::INDEX_DISKANN)) {
        return knowhere::Version::GetDefaultVersion().VersionNumber();
    }
    if (row.requires_cardinal || row.index_type == knowhere::IndexEnum::INDEX_CARDINAL_TIERED) {
        return std::max(knowhere::Version::GetCurrentVersion().VersionNumber(), 9);
    }
#endif
    return knowhere::Version::GetCurrentVersion().VersionNumber();
}

std::string
NullableName(NullableRatio ratio) {
    switch (ratio) {
        case NullableRatio::R0:
            return "nullable0";
        case NullableRatio::R50:
            return "nullable50";
        case NullableRatio::R100:
            return "nullable100";
    }
    return "nullable_unknown";
}

std::string
FilterName(FilterRatio ratio) {
    switch (ratio) {
        case FilterRatio::None:
            return "filter_none";
        case FilterRatio::R0:
            return "filter0";
        case FilterRatio::R50:
            return "filter50";
        case FilterRatio::R100:
            return "filter100";
        case FilterRatio::Collapsed:
            return "filter_collapsed";
    }
    return "filter_unknown";
}

std::string
ModeName(Mode mode) {
    switch (mode) {
        case Mode::Vector:
            return "vector";
        case Mode::EmbList:
            return "emblist";
        case Mode::MultiIndex:
            return "multi";
    }
    return "unknown_mode";
}

std::string
SourceName(IndexSource source) {
    switch (source) {
        case IndexSource::Fresh:
            return "fresh";
        case IndexSource::BinarySet:
            return "binary";
        case IndexSource::File:
            return "file";
    }
    return "unknown_source";
}

std::string
OperationName(Operation op) {
    switch (op) {
        case Operation::Build:
            return "build";
        case Operation::Search:
            return "search";
        case Operation::Range:
            return "range";
        case Operation::Iterator:
            return "iterator";
        case Operation::SearchFilter:
            return "search_filter";
        case Operation::RangeFilter:
            return "range_filter";
        case Operation::IteratorFilter:
            return "iterator_filter";
        case Operation::BinarySetSerialize:
            return "binaryset_serialize";
        case Operation::BinarySetDeserialize:
            return "binaryset_deserialize";
        case Operation::FileSerialize:
            return "file_serialize";
        case Operation::FileDeserialize:
            return "file_deserialize";
    }
    return "unknown_operation";
}

std::vector<int32_t>
AllIds(int64_t count) {
    std::vector<int32_t> ids(count);
    std::iota(ids.begin(), ids.end(), 0);
    return ids;
}

std::vector<int32_t>
SwappedEvenIds(int64_t count) {
    std::vector<int32_t> ids;
    ids.reserve((count + 1) / 2);
    for (int32_t id = 0; id < count; id += 4) {
        if (id + 2 < count) {
            ids.push_back(id + 2);
        }
        ids.push_back(id);
    }
    return ids;
}

std::vector<int32_t>
SwappedPairedIds(int64_t count) {
    std::vector<int32_t> ids;
    ids.reserve((count + 3) / 4 * 2);
    for (int32_t id = 0; id < count; id += 8) {
        if (id + 4 < count) {
            ids.push_back(id + 4);
        }
        if (id + 5 < count) {
            ids.push_back(id + 5);
        }
        ids.push_back(id);
        if (id + 1 < count) {
            ids.push_back(id + 1);
        }
    }
    return ids;
}

std::vector<int32_t>
ValidIdsFor(NullableRatio ratio, int64_t count, bool paired_for_multi = false) {
    switch (ratio) {
        case NullableRatio::R0:
            return AllIds(count);
        case NullableRatio::R50:
            return paired_for_multi ? SwappedPairedIds(count) : SwappedEvenIds(count);
        case NullableRatio::R100:
            return {};
    }
    return {};
}

bool
ContainsId(const std::vector<int32_t>& ids, int64_t id) {
    return std::find(ids.begin(), ids.end(), static_cast<int32_t>(id)) != ids.end();
}

int64_t
FirstMissingExternalId(const std::vector<int32_t>& valid_ids, int64_t total_count) {
    for (int64_t id = 0; id < total_count; ++id) {
        if (!ContainsId(valid_ids, id)) {
            return id;
        }
    }
    return total_count;
}

std::vector<int32_t>
FirstQueryIds(const std::vector<int32_t>& valid_ids, int64_t total_count, int64_t query_count = kQueryCount) {
    std::vector<int32_t> ids;
    ids.reserve(query_count);
    for (auto id : valid_ids) {
        ids.push_back(id);
        if (static_cast<int64_t>(ids.size()) == query_count) {
            return ids;
        }
    }
    for (int32_t id = 0; static_cast<int64_t>(ids.size()) < query_count && id < total_count; ++id) {
        ids.push_back(id);
    }
    while (static_cast<int64_t>(ids.size()) < query_count) {
        ids.push_back(0);
    }
    return ids;
}

std::vector<int32_t>
PartitionIds(const std::vector<int32_t>& valid_ids, int partition) {
    std::vector<int32_t> ids;
    for (auto id : valid_ids) {
        if ((id & 1) == partition) {
            ids.push_back(id);
        }
    }
    return ids;
}

void
FillDenseVector(float* data, int64_t row, int64_t dim, int32_t logical_id) {
    auto* dst = data + row * dim;
    std::fill(dst, dst + dim, 0.0f);
    dst[0] = static_cast<float>(logical_id * 100 + 1);
    dst[logical_id % dim] += 1.0f;
}

knowhere::DataSetPtr
GenFullDenseDataSet(int64_t total_count, int64_t dim) {
    auto data = std::make_unique<float[]>(std::max<int64_t>(total_count * dim, 1));
    for (int64_t logical_id = 0; logical_id < total_count; ++logical_id) {
        FillDenseVector(data.get(), logical_id, dim, static_cast<int32_t>(logical_id));
    }
    auto ds = knowhere::GenDataSet(total_count, dim, data.release());
    ds->SetIsOwner(true);
    return ds;
}

knowhere::DataSetPtr
GenNullableDenseDataSet(int64_t total_count, const std::vector<int32_t>& valid_ids, int64_t dim) {
    const auto rows = static_cast<int64_t>(valid_ids.size());
    auto data = std::make_unique<float[]>(std::max<int64_t>(rows * dim, 1));
    for (int64_t row = 0; row < rows; ++row) {
        FillDenseVector(data.get(), row, dim, valid_ids[row]);
    }
    auto ds = knowhere::GenDataSet(rows, dim, data.release());
    ds->SetIsOwner(true);
    if (rows != total_count) {
        ds->SetInternalToExternalIds(valid_ids, static_cast<size_t>(total_count));
    }
    return ds;
}

knowhere::DataSetPtr
GenDenseQueryDataSet(const std::vector<int32_t>& logical_ids, int64_t dim) {
    auto data = std::make_unique<float[]>(std::max<int64_t>(static_cast<int64_t>(logical_ids.size()) * dim, 1));
    for (int64_t row = 0; row < static_cast<int64_t>(logical_ids.size()); ++row) {
        FillDenseVector(data.get(), row, dim, logical_ids[row]);
    }
    auto ds = knowhere::GenDataSet(static_cast<int64_t>(logical_ids.size()), dim, data.release());
    ds->SetIsOwner(true);
    return ds;
}

knowhere::DataSetPtr
GenNullableBinaryDataSet(int64_t total_count, const std::vector<int32_t>& valid_ids, int64_t dim,
                         const knowhere::DataSetPtr& full_ds, std::vector<uint8_t>& owned_data) {
    const auto bytes_per_row = dim / 8;
    const auto* full_data = static_cast<const uint8_t*>(full_ds->GetTensor());
    owned_data.assign(std::max<int64_t>(static_cast<int64_t>(valid_ids.size()) * bytes_per_row, 1), 0);
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        std::memcpy(owned_data.data() + i * bytes_per_row, full_data + valid_ids[i] * bytes_per_row, bytes_per_row);
    }
    auto ds = knowhere::GenDataSet(static_cast<int64_t>(valid_ids.size()), dim, owned_data.data());
    ds->SetIsOwner(false);
    if (static_cast<int64_t>(valid_ids.size()) != total_count) {
        ds->SetInternalToExternalIds(valid_ids, static_cast<size_t>(total_count));
    }
    return ds;
}

knowhere::DataSetPtr
GenBinaryQueryDataSet(const std::vector<int32_t>& query_ids, int64_t dim, const knowhere::DataSetPtr& full_ds,
                      std::vector<uint8_t>& owned_data) {
    const auto bytes_per_row = dim / 8;
    const auto* full_data = static_cast<const uint8_t*>(full_ds->GetTensor());
    owned_data.assign(std::max<int64_t>(static_cast<int64_t>(query_ids.size()) * bytes_per_row, 1), 0);
    for (size_t i = 0; i < query_ids.size(); ++i) {
        std::memcpy(owned_data.data() + i * bytes_per_row, full_data + query_ids[i] * bytes_per_row, bytes_per_row);
    }
    auto ds = knowhere::GenDataSet(static_cast<int64_t>(query_ids.size()), dim, owned_data.data());
    ds->SetIsOwner(false);
    return ds;
}

knowhere::DataSetPtr
GenNullableSparseDataSet(int64_t total_count, const std::vector<int32_t>& valid_ids) {
    std::vector<std::map<int32_t, float>> rows;
    rows.reserve(valid_ids.size());
    for (auto logical_id : valid_ids) {
        rows.push_back({{logical_id + 1, 1.0f}});
    }
    auto ds = GenSparseDataSet(rows, static_cast<int32_t>(total_count + 2));
    if (static_cast<int64_t>(valid_ids.size()) != total_count) {
        ds->SetInternalToExternalIds(valid_ids, static_cast<size_t>(total_count));
    }
    return ds;
}

knowhere::DataSetPtr
GenSparseQueryDataSet(const std::vector<int32_t>& logical_ids, int64_t total_count) {
    std::vector<std::map<int32_t, float>> rows;
    rows.reserve(logical_ids.size());
    for (auto logical_id : logical_ids) {
        rows.push_back({{logical_id + 1, 1.0f}});
    }
    return GenSparseDataSet(rows, static_cast<int32_t>(total_count + 2));
}

knowhere::DataSetPtr
GenNullableEmbListDataSet(int64_t total_docs, const std::vector<int32_t>& valid_doc_ids, int64_t dim,
                          int64_t vectors_per_doc) {
    const auto rows = static_cast<int64_t>(valid_doc_ids.size()) * vectors_per_doc;
    auto data = std::make_unique<float[]>(std::max<int64_t>(rows * dim, 1));
    std::vector<size_t> offsets(valid_doc_ids.size() + 1, 0);
    for (size_t doc = 0; doc < valid_doc_ids.size(); ++doc) {
        offsets[doc] = doc * vectors_per_doc;
        for (int64_t v = 0; v < vectors_per_doc; ++v) {
            FillDenseVector(data.get(), static_cast<int64_t>(doc) * vectors_per_doc + v, dim, valid_doc_ids[doc]);
        }
    }
    offsets[valid_doc_ids.size()] = static_cast<size_t>(rows);
    auto ds = knowhere::GenDataSet(rows, dim, data.release());
    ds->SetIsOwner(true);
    auto offset_data = std::make_unique<size_t[]>(offsets.size());
    std::copy(offsets.begin(), offsets.end(), offset_data.get());
    ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(offset_data.release()));
    if (static_cast<int64_t>(valid_doc_ids.size()) != total_docs) {
        ds->SetInternalToExternalIds(valid_doc_ids, static_cast<size_t>(total_docs));
    }
    return ds;
}

knowhere::DataSetPtr
GenEmbListQueryDataSet(const std::vector<int32_t>& logical_doc_ids, int64_t dim) {
    auto ds = GenDenseQueryDataSet(logical_doc_ids, dim);
    auto offsets = std::make_unique<size_t[]>(logical_doc_ids.size() + 1);
    for (size_t i = 0; i <= logical_doc_ids.size(); ++i) {
        offsets[i] = i;
    }
    ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(offsets.release()));
    return ds;
}

std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>>
ScalarInfoForEvenOddPartitions(const std::vector<int32_t>& valid_ids) {
    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info;
    scalar_info[0].resize(2);
    for (size_t row = 0; row < valid_ids.size(); ++row) {
        scalar_info[0][valid_ids[row] & 1].push_back(static_cast<uint32_t>(row));
    }
    return scalar_info;
}

std::vector<uint8_t>
MakeBitmap(size_t total_count, const std::vector<int32_t>& filtered_ids) {
    std::vector<uint8_t> bitmap((total_count + 7) / 8, 0);
    for (auto id : filtered_ids) {
        if (id >= 0 && static_cast<size_t>(id) < total_count) {
            bitmap[id >> 3] |= static_cast<uint8_t>(1U << (id & 7));
        }
    }
    return bitmap;
}

std::vector<int32_t>
HalfFilteredIds(size_t total_count, const std::vector<int32_t>& candidate_ids) {
    std::vector<int32_t> ids;
    const auto target = candidate_ids.size() / 2;
    ids.reserve(target);
    for (size_t i = 0; i < candidate_ids.size() && ids.size() < target; i += 2) {
        ids.push_back(candidate_ids[i]);
    }
    return ids;
}

std::vector<int32_t>
FilterIdsFor(FilterRatio filter_ratio, int64_t total_count, const std::vector<int32_t>& valid_ids, Mode mode,
             const std::vector<int32_t>& selected_ids) {
    if (filter_ratio == FilterRatio::None || filter_ratio == FilterRatio::R0) {
        if (mode != Mode::MultiIndex) {
            return {};
        }
    }
    if (filter_ratio == FilterRatio::R100 || filter_ratio == FilterRatio::Collapsed) {
        return AllIds(total_count);
    }

    std::vector<int32_t> filtered_ids;
    if (mode == Mode::MultiIndex) {
        std::set<int32_t> selected(selected_ids.begin(), selected_ids.end());
        for (int32_t id = 0; id < total_count; ++id) {
            if (selected.find(id) == selected.end()) {
                filtered_ids.push_back(id);
            }
        }
        if (filter_ratio == FilterRatio::R50) {
            auto half_selected = HalfFilteredIds(total_count, selected_ids);
            filtered_ids.insert(filtered_ids.end(), half_selected.begin(), half_selected.end());
        }
        return filtered_ids;
    }

    if (filter_ratio == FilterRatio::R50) {
        return HalfFilteredIds(total_count, valid_ids);
    }
    return {};
}

std::vector<int32_t>
AllowedIdsAfterFilter(const std::vector<int32_t>& valid_ids, const std::vector<int32_t>& filtered_ids) {
    std::vector<int32_t> allowed;
    for (auto id : valid_ids) {
        if (!ContainsId(filtered_ids, id)) {
            allowed.push_back(id);
        }
    }
    return allowed;
}

knowhere::BitsetView
BitsetViewFrom(std::vector<uint8_t>& bitmap, int64_t total_count, const std::vector<int32_t>& filtered_ids) {
    bitmap = MakeBitmap(total_count, filtered_ids);
    return knowhere::BitsetView(bitmap.data(), total_count);
}

knowhere::Json
BaseDenseConfig(const std::string& metric = knowhere::metric::L2, int64_t dim = kDenseDim, int64_t topk = kTopK) {
    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = topk;
    json[knowhere::meta::RADIUS] = 100000000.0f;
    json[knowhere::meta::RANGE_FILTER] = 0.0f;
    json[knowhere::meta::RANGE_SEARCH_K] = 64;
    return json;
}

knowhere::Json
DenseConfigFor(const IndexRow& row, Mode mode) {
    const auto dim = row.data_kind == DataKind::GpuFp32 ? kGpuDim : kDenseDim;
    auto json = BaseDenseConfig(mode == Mode::EmbList ? "MAX_SIM_L2" : knowhere::metric::L2, dim,
                                mode == Mode::EmbList ? kEmbTopK : kTopK);
    if (row.index_type == knowhere::IndexEnum::INDEX_FAISS) {
        json["faiss_index_name"] = "Flat";
        return json;
    }
    json[knowhere::meta::INDEX_TYPE] = row.index_type;
    json[knowhere::indexparam::NLIST] = 4;
    json[knowhere::indexparam::NPROBE] = 4;
    json[knowhere::indexparam::HNSW_M] = 8;
    json[knowhere::indexparam::EFCONSTRUCTION] = 40;
    json[knowhere::indexparam::EF] = 64;
    json[knowhere::indexparam::M] = 4;
    json[knowhere::indexparam::NBITS] = 4;
    json[knowhere::indexparam::PRQ_NUM] = 2;
    json[knowhere::indexparam::SQ_TYPE] = "SQ8";
    json[knowhere::indexparam::REORDER_K] = 8;
    json[knowhere::indexparam::SUB_DIM] = 2;
    json[knowhere::indexparam::WITH_RAW_DATA] = true;
    json[knowhere::indexparam::ENSURE_TOPK_FULL] = true;
    json[knowhere::indexparam::REFINE_RATIO] = 2.0f;
    json[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 2.0f;
    if (mode == Mode::EmbList) {
        json["emb_list_strategy"] = knowhere::meta::EMB_LIST_STRATEGY_TOKENANN;
    }

    if (row.index_type == "IVF_SQ") {
        json[knowhere::indexparam::SQ_TYPE] = "SQ8";
    }
    if (row.index_type == knowhere::IndexEnum::INDEX_SVS_VAMANA ||
        row.index_type == knowhere::IndexEnum::INDEX_SVS_VAMANA_LVQ ||
        row.index_type == knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC) {
        json[knowhere::indexparam::SVS_GRAPH_MAX_DEGREE] = 16;
        json[knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE] = 40;
        json[knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE] = 40;
        json[knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY] = 40;
        json[knowhere::indexparam::SVS_ALPHA] = 1.2f;
        json[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("fp32");
        if (row.index_type == knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC) {
            json[knowhere::indexparam::SVS_LEANVEC_DIM] = 8;
        }
    }
    if (row.data_kind == DataKind::GpuFp32) {
        json[knowhere::meta::DIM] = kGpuDim;
        json[knowhere::meta::RADIUS] = 100000000.0f;
        if (row.index_type == knowhere::IndexEnum::INDEX_CUVS_CAGRA ||
            row.index_type == knowhere::IndexEnum::INDEX_GPU_CAGRA) {
            json[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = 32;
            json[knowhere::indexparam::GRAPH_DEGREE] = 16;
            json[knowhere::indexparam::ITOPK_SIZE] = 32;
        }
    }
    return json;
}

knowhere::Json
BinaryConfigFor(const IndexRow& row) {
    knowhere::Json json;
    const auto dim = row.index_type == knowhere::IndexEnum::INDEX_MINHASH_LSH ? 1024 : 128;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = row.index_type == knowhere::IndexEnum::INDEX_MINHASH_LSH
                                            ? knowhere::metric::MHJACCARD
                                            : knowhere::metric::HAMMING;
    json[knowhere::meta::TOPK] = kTopK;
    json[knowhere::meta::RADIUS] = 1024.0f;
    json[knowhere::meta::RANGE_FILTER] = 0.0f;
    if (row.index_type == knowhere::IndexEnum::INDEX_FAISS) {
        json["faiss_index_name"] = "BFlat";
        return json;
    }
    json[knowhere::indexparam::NLIST] = 4;
    json[knowhere::indexparam::NPROBE] = 4;
    json["faiss_index_name"] = "Flat";
    json["refine_k"] = 12;
    json["mh_lsh_band"] = 4;
    json["mh_element_bit_width"] = 32;
    json["mh_search_with_jaccard"] = true;
    json["mh_lsh_batch_search"] = true;
    json["mh_lsh_aligned_block_size"] = 4096;
    json["mh_lsh_shared_bloom_filter"] = true;
    json["mh_lsh_bloom_false_positive_prob"] = 0.01;
    json["with_raw_data"] = true;
    json["hash_code_in_memory"] = true;
    return json;
}

knowhere::Json
SparseConfig() {
    knowhere::Json json;
    json[knowhere::meta::DIM] = kTotalRows + 2;
    json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
    json[knowhere::meta::TOPK] = kTopK;
    json[knowhere::meta::RADIUS] = 0.5f;
    json[knowhere::meta::RANGE_FILTER] = 2.0f;
    json[knowhere::indexparam::DROP_RATIO_BUILD] = 0.0f;
    json[knowhere::indexparam::DROP_RATIO_SEARCH] = 0.0f;
    json[knowhere::indexparam::INVERTED_INDEX_ALGO] = "DAAT_MAXSCORE";
    return json;
}

knowhere::Json
DiskAnnConfig(const IndexRow& row, const fs::path& raw_data_file, const fs::path& index_prefix) {
    auto json = BaseDenseConfig(knowhere::metric::L2, kFileDim, kTopK);
    json[knowhere::meta::INDEX_TYPE] = row.index_type;
    json[knowhere::meta::DATA_PATH] = raw_data_file.string();
    json[knowhere::meta::INDEX_PREFIX] = index_prefix.string();
    json[knowhere::indexparam::MAX_DEGREE] = 16;
    json[knowhere::indexparam::SEARCH_LIST_SIZE] = 32;
    json[knowhere::indexparam::PQ_CODE_BUDGET_GB] = 0.01;
    json[knowhere::indexparam::SEARCH_CACHE_BUDGET_GB] = 0.0;
    json[knowhere::indexparam::BUILD_DRAM_BUDGET_GB] = 1.0;
    json[knowhere::indexparam::BEAMWIDTH] = 8;
    json["min_k"] = 3;
    json["max_k"] = 64;
    if (row.data_kind == DataKind::Aisaq) {
        json[knowhere::indexparam::REARRANGE] = false;
        json[knowhere::indexparam::INLINE_PQ] = 0;
        json[knowhere::indexparam::PQ_CACHE_SIZE] = 0;
        json[knowhere::indexparam::PQ_READ_PAGE_CACHE_SIZE] = 0;
        json[knowhere::indexparam::VECTORS_BEAMWIDTH] = 4;
    }
    return json;
}

Capabilities
VectorCaps(bool range, bool iterator, bool search_filter, bool range_filter, bool iterator_filter,
           bool binaryset_io, bool file_io) {
    Capabilities caps;
    caps.build = true;
    caps.search = true;
    caps.range = range;
    caps.iterator = iterator;
    caps.search_filter = search_filter;
    caps.range_filter = range_filter;
    caps.iterator_filter = iterator_filter;
    caps.binaryset_serialize = binaryset_io;
    caps.binaryset_deserialize = binaryset_io;
    caps.file_serialize = file_io;
    caps.file_deserialize = file_io;
    return caps;
}

void
AddEmbCaps(Capabilities& caps, bool search, bool range, bool iterator, bool search_filter, bool range_filter,
           bool iterator_filter) {
    caps.emb_build = search || range || iterator || search_filter || range_filter || iterator_filter;
    caps.emb_search = search;
    caps.emb_range = range;
    caps.emb_iterator = iterator;
    caps.emb_search_filter = search_filter;
    caps.emb_range_filter = range_filter;
    caps.emb_iterator_filter = iterator_filter;
}

void
AddMultiCaps(Capabilities& caps, bool search, bool range, bool iterator, bool search_filter, bool range_filter,
             bool iterator_filter) {
    caps.multi_build = search || range || iterator || search_filter || range_filter || iterator_filter;
    caps.multi_search = search;
    caps.multi_range = range;
    caps.multi_iterator = iterator;
    caps.multi_search_filter = search_filter;
    caps.multi_range_filter = range_filter;
    caps.multi_iterator_filter = iterator_filter;
}

std::vector<IndexRow>
IndexRows() {
    std::vector<IndexRow> rows;
    auto add = [&](IndexRow row) {
        rows.push_back(std::move(row));
    };

    auto flat = VectorCaps(true, false, true, true, false, true, true);
    add({"FLAT", knowhere::IndexEnum::INDEX_FAISS_IDMAP, DataKind::DenseFp32, flat, false, false, false, false, true});
    add({"BIN_FLAT / BINFLAT", knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, DataKind::DenseBin,
         flat, false, false, false, false, false});
    add({"FAISS (fp32)", knowhere::IndexEnum::INDEX_FAISS, DataKind::DenseFp32, flat, false, false, false,
         false, true});
    auto faiss_bin = VectorCaps(false, false, true, false, false, true, true);
    add({"FAISS (bin1)", knowhere::IndexEnum::INDEX_FAISS, DataKind::DenseBin, faiss_bin});

    add({"BIN_IVF_FLAT / IVFBIN", knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, DataKind::DenseBin,
         VectorCaps(true, false, true, true, false, true, true)});
    auto ivf_flat = VectorCaps(true, true, true, true, true, true, true);
    AddEmbCaps(ivf_flat, true, false, false, true, false, false);
    add({"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32, ivf_flat});
    add({"IVF_FLAT_CC / IVFFLATCC", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, DataKind::DenseFp32, ivf_flat});
    add({"IVF_PQ / IVFPQ", knowhere::IndexEnum::INDEX_FAISS_IVFPQ, DataKind::DenseFp32,
         VectorCaps(true, false, true, true, false, true, true)});
    add({"IVF_SQ8", knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, DataKind::DenseFp32,
         VectorCaps(true, true, true, true, true, true, true)});
    add({"IVF_SQ / IVFSQ", "IVF_SQ", DataKind::DenseFp32,
         VectorCaps(true, true, true, true, true, true, true)});
    add({"IVF_SQ_CC", knowhere::IndexEnum::INDEX_FAISS_IVFSQ_CC, DataKind::DenseFp32,
         VectorCaps(true, true, true, true, true, true, true)});
    add({"IVF_RABITQ / IVFRABITQ", knowhere::IndexEnum::INDEX_FAISS_IVFRABITQ, DataKind::DenseFp32,
         VectorCaps(true, true, true, true, true, true, true)});
    add({"IVF_RABITQ_FASTSCAN", knowhere::IndexEnum::INDEX_FAISS_IVFRABITQ_FASTSCAN, DataKind::DenseFp32,
         VectorCaps(true, false, true, true, false, true, true)});
    add({"SCANN", knowhere::IndexEnum::INDEX_FAISS_SCANN, DataKind::DenseFp32,
         VectorCaps(true, true, true, true, true, true, true), true});
    auto scann_dvr = VectorCaps(true, true, true, true, true, false, false);
    AddEmbCaps(scann_dvr, true, false, false, true, false, false);
    add({"SCANN_DVR", knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, DataKind::DenseFp32, scann_dvr, true});

    auto hnsw_native = VectorCaps(true, true, true, true, true, true, true);
    AddMultiCaps(hnsw_native, true, true, true, true, true, true);
    add({"HNSW (Knowhere/Faiss)", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32, hnsw_native});
    auto hnsw = hnsw_native;
    AddEmbCaps(hnsw, true, false, false, true, false, false);
    add({"HNSW_SQ", knowhere::IndexEnum::INDEX_HNSW_SQ, DataKind::DenseFp32, hnsw});
    auto hnsw_quantized = VectorCaps(true, true, true, true, true, true, true);
    AddEmbCaps(hnsw_quantized, true, false, false, true, false, false);
    add({"HNSW_PQ", knowhere::IndexEnum::INDEX_HNSW_PQ, DataKind::DenseFp32, hnsw_quantized});
    add({"HNSW_PRQ", knowhere::IndexEnum::INDEX_HNSW_PRQ, DataKind::DenseFp32, hnsw_quantized});
    add({"HNSW_DEPRECATED", "HNSW_DEPRECATED", DataKind::DenseFp32, hnsw, true, true});
    add({"HNSWLIB_DEPRECATED", "HNSWLIB_DEPRECATED", DataKind::DenseFp32,
         VectorCaps(true, true, true, true, true, true, true)});
    add({"HNSW_V1 sparse (Cardinal v1)", "HNSW_V1", DataKind::Sparse,
         VectorCaps(true, true, true, true, true, true, true), true, true, false, false, true});

    auto diskann = VectorCaps(true, true, true, true, true, false, true);
    add({"DISKANN (Knowhere native)", knowhere::IndexEnum::INDEX_DISKANN, DataKind::DiskAnn, diskann});
    add({"AISAQ", knowhere::IndexEnum::INDEX_AISAQ, DataKind::Aisaq,
         VectorCaps(false, false, true, false, false, false, true)});
    add({"DISKANN_DEPRECATED", "DISKANN_DEPRECATED", DataKind::DiskAnn, diskann, true, true});

    add({"SPARSE_INVERTED_INDEX (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
         DataKind::Sparse, VectorCaps(true, true, true, true, true, true, true), false, false, false, false, true});
    add({"SPARSE_WAND (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_WAND, DataKind::Sparse,
         VectorCaps(true, true, true, true, true, true, true), false, false, false, false, true});
    add({"SPARSE_INVERTED_INDEX_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
         DataKind::Sparse, VectorCaps(true, true, true, true, true, false, false), false, false, false, false, true});
    add({"SPARSE_WAND_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, DataKind::Sparse,
         VectorCaps(true, true, true, true, true, false, false), false, false, false, false, true});
    add({"SPARSE_INVERTED_INDEX (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
         DataKind::Sparse, VectorCaps(true, true, true, true, true, true, true), true, true, false, false, true});
    add({"SPARSE_WAND (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_WAND, DataKind::Sparse,
         VectorCaps(true, true, true, true, true, true, true), true, true, false, false, true});
    add({"SPARSE_INVERTED_INDEX_CC (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
         DataKind::Sparse, VectorCaps(true, true, true, true, true, false, false), true, true, false, false, true});
    add({"SPARSE_WAND_CC (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, DataKind::Sparse,
         VectorCaps(true, true, true, true, true, false, false), true, true, false, false, true});
    add({"MINHASH_LSH", knowhere::IndexEnum::INDEX_MINHASH_LSH, DataKind::MinHash,
         VectorCaps(false, false, true, false, false, false, true)});

    add({"GPU_CUVS_BRUTE_FORCE", knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_BRUTE_FORCE", knowhere::IndexEnum::INDEX_GPU_BRUTEFORCE, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_CUVS_IVF_FLAT", knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_IVF_FLAT", knowhere::IndexEnum::INDEX_GPU_IVFFLAT, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_CUVS_IVF_PQ", knowhere::IndexEnum::INDEX_CUVS_IVFPQ, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_IVF_PQ", knowhere::IndexEnum::INDEX_GPU_IVFPQ, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_CUVS_CAGRA", knowhere::IndexEnum::INDEX_CUVS_CAGRA, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_CAGRA", knowhere::IndexEnum::INDEX_GPU_CAGRA, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_FAISS_FLAT", knowhere::IndexEnum::INDEX_FAISS_GPU_IDMAP, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_FAISS_IVF_FLAT", knowhere::IndexEnum::INDEX_FAISS_GPU_IVFFLAT, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_FAISS_IVF_PQ", knowhere::IndexEnum::INDEX_FAISS_GPU_IVFPQ, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});
    add({"GPU_FAISS_IVF_SQ8", knowhere::IndexEnum::INDEX_FAISS_GPU_IVFSQ8, DataKind::GpuFp32,
         VectorCaps(false, false, true, false, false, true, false), true, false, false, true});

    add({"SVS_FLAT", knowhere::IndexEnum::INDEX_SVS_FLAT, DataKind::DenseFp32,
         VectorCaps(false, false, false, false, false, true, true), true, false, true});
    add({"SVS_VAMANA", knowhere::IndexEnum::INDEX_SVS_VAMANA, DataKind::DenseFp32,
         VectorCaps(true, false, true, true, false, true, true), true, false, true});
    add({"SVS_VAMANA_LVQ", knowhere::IndexEnum::INDEX_SVS_VAMANA_LVQ, DataKind::DenseFp32,
         VectorCaps(true, false, true, true, false, true, true), true, false, true});
    add({"SVS_VAMANA_LEANVEC", knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC, DataKind::DenseFp32,
         VectorCaps(true, false, true, true, false, true, true), true, false, true});

    auto cardinal_v2 = hnsw;
    add({"HNSW (Cardinal v2)", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32, cardinal_v2, true, true});
    auto cardinal_v2_disk = VectorCaps(false, false, true, false, false, false, false);
    AddEmbCaps(cardinal_v2_disk, true, false, false, true, false, false);
    AddMultiCaps(cardinal_v2_disk, true, false, false, true, false, false);
    add({"DISKANN (Cardinal v2)", knowhere::IndexEnum::INDEX_DISKANN, DataKind::DiskAnn, cardinal_v2_disk, true, true});
    add({"CARDINAL_TIERED", knowhere::IndexEnum::INDEX_CARDINAL_TIERED, DataKind::DenseFp32, cardinal_v2, true, true});

    return rows;
}

void
AddScenario(std::vector<Scenario>& scenarios, NullableRatio nullable_ratio, Mode mode, Operation op,
            IndexSource source, FilterRatio filter_ratio = FilterRatio::None) {
    Scenario scenario;
    scenario.nullable_ratio = nullable_ratio;
    scenario.mode = mode;
    scenario.op = op;
    scenario.source = source;
    scenario.filter_ratio = filter_ratio;
    scenario.name = NullableName(nullable_ratio) + "/" + ModeName(mode) + "/" + OperationName(op) + "/" +
                    SourceName(source) + "/" + FilterName(filter_ratio);
    scenarios.push_back(std::move(scenario));
}

void
AddNonFilterScenarios(std::vector<Scenario>& scenarios, NullableRatio ratio, Mode mode) {
    AddScenario(scenarios, ratio, mode, Operation::Build, IndexSource::Fresh);
    AddScenario(scenarios, ratio, mode, Operation::Search, IndexSource::Fresh);
    AddScenario(scenarios, ratio, mode, Operation::Range, IndexSource::Fresh);
    AddScenario(scenarios, ratio, mode, Operation::Iterator, IndexSource::Fresh);
    AddScenario(scenarios, ratio, mode, Operation::BinarySetSerialize, IndexSource::Fresh);
    AddScenario(scenarios, ratio, mode, Operation::BinarySetDeserialize, IndexSource::BinarySet);
    AddScenario(scenarios, ratio, mode, Operation::Search, IndexSource::BinarySet);
    AddScenario(scenarios, ratio, mode, Operation::Range, IndexSource::BinarySet);
    AddScenario(scenarios, ratio, mode, Operation::Iterator, IndexSource::BinarySet);
    AddScenario(scenarios, ratio, mode, Operation::FileSerialize, IndexSource::Fresh);
    AddScenario(scenarios, ratio, mode, Operation::FileDeserialize, IndexSource::File);
    AddScenario(scenarios, ratio, mode, Operation::Search, IndexSource::File);
    AddScenario(scenarios, ratio, mode, Operation::Iterator, IndexSource::File);
}

std::vector<Scenario>
BuildScenarios() {
    std::vector<Scenario> scenarios;
    for (auto ratio : {NullableRatio::R0, NullableRatio::R50, NullableRatio::R100}) {
        for (auto mode : {Mode::Vector, Mode::EmbList, Mode::MultiIndex}) {
            AddNonFilterScenarios(scenarios, ratio, mode);
        }
        const std::vector<FilterRatio> filter_ratios =
            ratio == NullableRatio::R100 ? std::vector<FilterRatio>{FilterRatio::Collapsed}
                                         : std::vector<FilterRatio>{FilterRatio::R0, FilterRatio::R50,
                                                                    FilterRatio::R100};
        for (auto mode : {Mode::Vector, Mode::EmbList, Mode::MultiIndex}) {
            for (auto filter_ratio : filter_ratios) {
                AddScenario(scenarios, ratio, mode, Operation::SearchFilter, IndexSource::Fresh, filter_ratio);
                AddScenario(scenarios, ratio, mode, Operation::RangeFilter, IndexSource::Fresh, filter_ratio);
                AddScenario(scenarios, ratio, mode, Operation::IteratorFilter, IndexSource::Fresh, filter_ratio);
            }
        }
    }
    return scenarios;
}

bool
SupportsModeBuild(const IndexRow& row, Mode mode) {
    switch (mode) {
        case Mode::Vector:
            return row.caps.build;
        case Mode::EmbList:
            return row.caps.emb_build;
        case Mode::MultiIndex:
            return row.caps.multi_build;
    }
    return false;
}

bool
SupportsScenario(const IndexRow& row, const Scenario& scenario) {
    const auto& caps = row.caps;
    if (row.index_type == knowhere::IndexEnum::INDEX_HNSW && !row.requires_cardinal &&
        VersionForRow(row) < 6 && scenario.nullable_ratio != NullableRatio::R0) {
        return false;
    }
#ifdef KNOWHERE_WITH_CARDINAL
    if (row.data_kind == DataKind::Sparse && !row.requires_cardinal &&
        scenario.nullable_ratio != NullableRatio::R0) {
        return false;
    }
#endif
    if (row.requires_cardinal && row.index_type == knowhere::IndexEnum::INDEX_DISKANN &&
        scenario.mode != Mode::Vector &&
        (scenario.op == Operation::FileSerialize || scenario.op == Operation::FileDeserialize ||
         scenario.source == IndexSource::File)) {
        return false;
    }

    const bool reads_index = scenario.op == Operation::Search || scenario.op == Operation::Range ||
                             scenario.op == Operation::Iterator;
    if (reads_index && scenario.source == IndexSource::BinarySet && !caps.binaryset_deserialize) {
        return false;
    }
    if (reads_index && scenario.source == IndexSource::File && !caps.file_deserialize) {
        return false;
    }
    if (row.data_kind == DataKind::MinHash &&
        (scenario.op == Operation::Search || scenario.op == Operation::SearchFilter ||
         scenario.op == Operation::Range || scenario.op == Operation::RangeFilter ||
         scenario.op == Operation::Iterator || scenario.op == Operation::IteratorFilter) &&
        scenario.source == IndexSource::Fresh) {
        return false;
    }
    switch (scenario.op) {
        case Operation::Build:
            return SupportsModeBuild(row, scenario.mode);
        case Operation::Search:
            return scenario.mode == Mode::Vector    ? caps.search
                   : scenario.mode == Mode::EmbList ? caps.emb_search
                                                    : caps.multi_search;
        case Operation::Range:
            return scenario.mode == Mode::Vector    ? caps.range
                   : scenario.mode == Mode::EmbList ? caps.emb_range
                                                    : caps.multi_range;
        case Operation::Iterator:
            return scenario.mode == Mode::Vector    ? caps.iterator
                   : scenario.mode == Mode::EmbList ? caps.emb_iterator
                                                    : caps.multi_iterator;
        case Operation::SearchFilter:
            return scenario.mode == Mode::Vector    ? caps.search_filter
                   : scenario.mode == Mode::EmbList ? caps.emb_search_filter
                                                    : caps.multi_search_filter;
        case Operation::RangeFilter:
            return scenario.mode == Mode::Vector    ? caps.range_filter
                   : scenario.mode == Mode::EmbList ? caps.emb_range_filter
                                                    : caps.multi_range_filter;
        case Operation::IteratorFilter:
            return scenario.mode == Mode::Vector    ? caps.iterator_filter
                   : scenario.mode == Mode::EmbList ? caps.emb_iterator_filter
                                                    : caps.multi_iterator_filter;
        case Operation::BinarySetSerialize:
            return SupportsModeBuild(row, scenario.mode) && caps.binaryset_serialize;
        case Operation::BinarySetDeserialize:
            return SupportsModeBuild(row, scenario.mode) && caps.binaryset_deserialize;
        case Operation::FileSerialize:
            return SupportsModeBuild(row, scenario.mode) && caps.file_serialize;
        case Operation::FileDeserialize:
            return SupportsModeBuild(row, scenario.mode) && caps.file_deserialize;
    }
    return false;
}

bool
IsReadOperation(Operation op) {
    return op == Operation::Search || op == Operation::Range || op == Operation::Iterator ||
           op == Operation::SearchFilter || op == Operation::RangeFilter || op == Operation::IteratorFilter;
}

bool
IsFreshFileBackedRead(const IndexRow& row, const Scenario& scenario) {
    return scenario.source == IndexSource::Fresh && IsReadOperation(scenario.op) &&
           (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq);
}

bool
IsExpectedEmptyBuildStatus(knowhere::Status status) {
    return status == knowhere::Status::success || status == knowhere::Status::empty_index ||
           status == knowhere::Status::invalid_args || status == knowhere::Status::faiss_inner_error ||
           status == knowhere::Status::hnsw_inner_error || status == knowhere::Status::diskann_inner_error ||
           status == knowhere::Status::disk_file_error || status == knowhere::Status::aisaq_error ||
           status == knowhere::Status::cardinal_inner_error || status == knowhere::Status::sparse_inner_error ||
           status == knowhere::Status::emb_list_inner_error || status == knowhere::Status::cuda_runtime_error ||
           status == knowhere::Status::not_implemented;
}

bool
IsExpectedEmptySearchError(knowhere::Status status) {
    return status == knowhere::Status::empty_index || status == knowhere::Status::invalid_args ||
           status == knowhere::Status::not_implemented || status == knowhere::Status::faiss_inner_error ||
           status == knowhere::Status::hnsw_inner_error || status == knowhere::Status::diskann_inner_error ||
           status == knowhere::Status::aisaq_error || status == knowhere::Status::cardinal_inner_error ||
           status == knowhere::Status::sparse_inner_error || status == knowhere::Status::emb_list_inner_error ||
           status == knowhere::Status::cuda_runtime_error;
}

void
RequireAllResultIdsNegative(const knowhere::DataSet& result) {
    const auto nq = result.GetRows();
    const auto k = result.GetDim();
    const auto* lims = result.GetLims();
    const auto* ids = result.GetIds();
    const auto length = k == 0 ? lims[nq] : nq * k;
    for (int64_t i = 0; i < length; ++i) {
        REQUIRE(ids[i] < 0);
    }
}

void
RequireRangeResultEmpty(const knowhere::DataSet& result) {
    REQUIRE(result.GetLims()[result.GetRows()] == 0);
}

void
RequireResultIdsIn(const knowhere::DataSet& result, const std::vector<int32_t>& allowed_ids,
                   const std::vector<int32_t>& filtered_ids, bool expect_empty) {
    if (expect_empty) {
        RequireAllResultIdsNegative(result);
        return;
    }

    const auto nq = result.GetRows();
    const auto k = result.GetDim();
    const auto* lims = result.GetLims();
    const auto* ids = result.GetIds();
    const auto length = k == 0 ? lims[nq] : nq * k;
    bool has_result = false;
    for (int64_t i = 0; i < length; ++i) {
        if (ids[i] < 0) {
            continue;
        }
        has_result = true;
        REQUIRE(ContainsId(allowed_ids, ids[i]));
        REQUIRE(!ContainsId(filtered_ids, ids[i]));
    }
    REQUIRE(has_result);
}

void
RequireRangeIdsIn(const knowhere::DataSet& result, const std::vector<int32_t>& allowed_ids,
                  const std::vector<int32_t>& filtered_ids, bool expect_empty) {
    if (expect_empty) {
        RequireRangeResultEmpty(result);
        return;
    }
    RequireResultIdsIn(result, allowed_ids, filtered_ids, false);
}

void
RequireIteratorResults(const std::vector<knowhere::IndexNode::IteratorPtr>& iterators,
                       const std::vector<int32_t>& allowed_ids, const std::vector<int32_t>& filtered_ids,
                       bool expect_empty, const std::vector<int32_t>* query_ids = nullptr) {
    if (expect_empty) {
        for (const auto& iterator : iterators) {
            REQUIRE(!iterator->HasNext());
        }
        return;
    }

    bool checked = false;
    for (size_t i = 0; i < iterators.size(); ++i) {
        const auto& iterator = iterators[i];
        if (!iterator->HasNext()) {
            continue;
        }
        auto [id, dist] = iterator->Next();
        (void)dist;
        REQUIRE(ContainsId(allowed_ids, id));
        REQUIRE(!ContainsId(filtered_ids, id));
        if (query_ids != nullptr && i < query_ids->size() && ContainsId(allowed_ids, (*query_ids)[i])) {
            REQUIRE(id == (*query_ids)[i]);
        }
        checked = true;
    }
    REQUIRE(checked);
}

void
RequireExpectedVectorResult(const knowhere::expected<knowhere::DataSetPtr>& result,
                            const std::vector<int32_t>& allowed_ids, const std::vector<int32_t>& filtered_ids,
                            bool expect_empty) {
    if (!result.has_value()) {
        REQUIRE(expect_empty);
        REQUIRE(IsExpectedEmptySearchError(result.error()));
        return;
    }
    RequireResultIdsIn(*result.value(), allowed_ids, filtered_ids, expect_empty);
}

void
RequireExactFirstHits(const knowhere::DataSet& result, const std::vector<int32_t>& query_ids,
                      const std::vector<int32_t>& allowed_ids) {
    const auto nq = result.GetRows();
    const auto k = result.GetDim();
    const auto* ids = result.GetIds();
    for (int64_t i = 0; i < nq && i < static_cast<int64_t>(query_ids.size()); ++i) {
        if (!ContainsId(allowed_ids, query_ids[i])) {
            continue;
        }
        REQUIRE(k > 0);
        REQUIRE(ids[i * k] == query_ids[i]);
    }
}

void
RequireExactRangeHits(const knowhere::DataSet& result, const std::vector<int32_t>& query_ids,
                      const std::vector<int32_t>& allowed_ids) {
    const auto nq = result.GetRows();
    const auto* lims = result.GetLims();
    const auto* ids = result.GetIds();
    for (int64_t i = 0; i < nq && i < static_cast<int64_t>(query_ids.size()); ++i) {
        if (!ContainsId(allowed_ids, query_ids[i])) {
            continue;
        }
        bool found = false;
        for (size_t j = lims[i]; j < lims[i + 1]; ++j) {
            found = found || ids[j] == query_ids[i];
        }
        REQUIRE(found);
    }
}

void
RequireBufferedExactFirstHits(const std::vector<int64_t>& ids, int64_t topk, const std::vector<int32_t>& query_ids,
                              const std::vector<int32_t>& allowed_ids) {
    for (int64_t i = 0; i < static_cast<int64_t>(query_ids.size()); ++i) {
        if (!ContainsId(allowed_ids, query_ids[i])) {
            continue;
        }
        REQUIRE(topk > 0);
        REQUIRE(ids[i * topk] == query_ids[i]);
    }
}

void
RequireExpectedRangeResult(const knowhere::expected<knowhere::DataSetPtr>& result,
                           const std::vector<int32_t>& allowed_ids, const std::vector<int32_t>& filtered_ids,
                           bool expect_empty) {
    if (!result.has_value()) {
        REQUIRE(expect_empty);
        REQUIRE(IsExpectedEmptySearchError(result.error()));
        return;
    }
    RequireRangeIdsIn(*result.value(), allowed_ids, filtered_ids, expect_empty);
}

void
RequireExpectedIteratorResult(const knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>& result,
                              const std::vector<int32_t>& allowed_ids, const std::vector<int32_t>& filtered_ids,
                              bool expect_empty, const std::vector<int32_t>* query_ids = nullptr) {
    if (!result.has_value()) {
        REQUIRE(expect_empty);
        REQUIRE(IsExpectedEmptySearchError(result.error()));
        return;
    }
    RequireIteratorResults(result.value(), allowed_ids, filtered_ids, expect_empty, query_ids);
}

void
RequireBufferedIdsIn(const std::vector<int64_t>& ids, const std::vector<int32_t>& allowed_ids,
                     const std::vector<int32_t>& filtered_ids, bool expect_empty) {
    bool has_result = false;
    for (auto id : ids) {
        if (id < 0) {
            continue;
        }
        has_result = true;
        REQUIRE(ContainsId(allowed_ids, id));
        REQUIRE(!ContainsId(filtered_ids, id));
    }
    REQUIRE(has_result != expect_empty);
}

void
RequireBufferedStatus(knowhere::Status status, const std::vector<int64_t>& ids,
                      const std::vector<int32_t>& allowed_ids, const std::vector<int32_t>& filtered_ids,
                      bool expect_empty) {
    if (status != knowhere::Status::success) {
        REQUIRE(expect_empty);
        REQUIRE(IsExpectedEmptySearchError(status));
        return;
    }
    RequireBufferedIdsIn(ids, allowed_ids, filtered_ids, expect_empty);
}

void
RequireExternalIdMapContent(const knowhere::Index<knowhere::IndexNode>& index, const std::vector<int32_t>& valid_ids,
                            int64_t external_count) {
    REQUIRE(index.ExternalCount() == external_count);
    REQUIRE(index.Node() != nullptr);
    const auto& map = index.Node()->GetExternalIdMap();
    REQUIRE(map.ExternalCount(0) == external_count);
    REQUIRE(map.GetInternalToExternalIds() == valid_ids);
    auto valid = map.GetValidBitmapView();
    REQUIRE(valid.size == static_cast<size_t>(external_count));
    REQUIRE(valid.data != nullptr);
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        CAPTURE(i, valid_ids[i]);
        REQUIRE(map.ToExternalId(i) == valid_ids[i]);
        REQUIRE(map.ToInternalId(valid_ids[i]) == static_cast<int64_t>(i));
    }
    if (static_cast<int64_t>(valid_ids.size()) != external_count) {
        const auto missing_id = FirstMissingExternalId(valid_ids, external_count);
        CAPTURE(missing_id);
        REQUIRE(map.ToInternalId(missing_id) == -1);
    }
}

std::vector<float>
DenseVectorForLogicalId(int32_t logical_id, int64_t dim) {
    std::vector<float> data(dim);
    FillDenseVector(data.data(), 0, dim, logical_id);
    return data;
}

float
DenseL2ForLogicalIds(int32_t lhs, int32_t rhs, int64_t dim) {
    auto lhs_vec = DenseVectorForLogicalId(lhs, dim);
    auto rhs_vec = DenseVectorForLogicalId(rhs, dim);
    float dist = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        const auto diff = lhs_vec[i] - rhs_vec[i];
        dist += diff * diff;
    }
    return dist;
}

void
RequireDenseVectorsMatchLogicalIds(const knowhere::DataSet& vectors, const std::vector<int32_t>& logical_ids,
                                   int64_t dim) {
    REQUIRE(vectors.GetRows() == static_cast<int64_t>(logical_ids.size()));
    REQUIRE(vectors.GetDim() == dim);
    const auto* tensor = static_cast<const float*>(vectors.GetTensor());
    REQUIRE(tensor != nullptr);
    for (size_t i = 0; i < logical_ids.size(); ++i) {
        auto expected = DenseVectorForLogicalId(logical_ids[i], dim);
        for (int64_t j = 0; j < dim; ++j) {
            CAPTURE(i, j, logical_ids[i]);
            REQUIRE(tensor[i * dim + j] == Catch::Approx(expected[j]));
        }
    }
}

void
RequireDenseCalcDistByIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data) {
    std::vector<int64_t> labels(data.query_ids.begin(), data.query_ids.end());
    auto result = index.CalcDistByIDs(data.query_ds, knowhere::BitsetView{}, labels.data(), labels.size(), false);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetRows() == data.query_ds->GetRows());
    REQUIRE(result.value()->GetDim() == static_cast<int64_t>(labels.size()));
    const auto* distances = result.value()->GetDistance();
    REQUIRE(distances != nullptr);
    for (size_t i = 0; i < data.query_ids.size(); ++i) {
        for (size_t j = 0; j < labels.size(); ++j) {
            const auto expected = DenseL2ForLogicalIds(data.query_ids[i], static_cast<int32_t>(labels[j]), data.dim);
            CAPTURE(i, j, data.query_ids[i], labels[j]);
            REQUIRE(distances[i * labels.size() + j] == Catch::Approx(expected));
        }
    }
}

void
RequireDenseCalcDistRejectsInvalidExternalIds(const knowhere::Index<knowhere::IndexNode>& index,
                                              const MatrixData& data) {
    const int64_t labels[] = {FirstMissingExternalId(data.valid_ids, data.total_count), data.total_count};
    for (auto label : labels) {
        CAPTURE(label);
        auto result = index.CalcDistByIDs(data.query_ds, knowhere::BitsetView{}, &label, 1, false);
        REQUIRE(!result.has_value());
        REQUIRE(result.error() == Status::invalid_args);
    }
}

void
RequireGetVectorRejectsInvalidExternalIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data) {
    const int64_t ids[] = {FirstMissingExternalId(data.valid_ids, data.total_count), data.total_count};
    for (auto id : ids) {
        CAPTURE(id);
        auto result = index.GetVectorByIds(knowhere::GenIdsDataSet(1, &id));
        REQUIRE(!result.has_value());
        REQUIRE(result.error() == Status::invalid_args);
    }
}

void
RequireDenseVectorApisUseExternalIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data,
                                     const knowhere::Json& json) {
    RequireExternalIdMapContent(index, data.valid_ids, data.total_count);

    auto search = index.Search(data.query_ds, json, knowhere::BitsetView{});
    REQUIRE(search.has_value());
    RequireExactFirstHits(*search.value(), data.query_ids, data.valid_ids);

    std::vector<int64_t> ids(data.query_ids.begin(), data.query_ids.end());
    auto retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()));
    REQUIRE(retrieve.has_value());
    RequireDenseVectorsMatchLogicalIds(*retrieve.value(), data.query_ids, data.dim);
    RequireGetVectorRejectsInvalidExternalIds(index, data);

    RequireDenseCalcDistByIds(index, data);
    RequireDenseCalcDistRejectsInvalidExternalIds(index, data);
}

void
RequireEmbListVectorsMatchLogicalIds(const knowhere::DataSet& vectors, const std::vector<int32_t>& logical_ids,
                                     int64_t dim, int64_t vectors_per_doc) {
    REQUIRE(vectors.GetRows() == static_cast<int64_t>(logical_ids.size()));
    REQUIRE(vectors.GetDim() == dim);
    const auto* offsets = vectors.Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
    REQUIRE(offsets != nullptr);
    const auto* tensor = static_cast<const float*>(vectors.GetTensor());
    REQUIRE(tensor != nullptr);
    for (size_t i = 0; i < logical_ids.size(); ++i) {
        REQUIRE(offsets[i + 1] - offsets[i] == static_cast<size_t>(vectors_per_doc));
        auto expected = DenseVectorForLogicalId(logical_ids[i], dim);
        for (int64_t v = 0; v < vectors_per_doc; ++v) {
            for (int64_t j = 0; j < dim; ++j) {
                CAPTURE(i, v, j, logical_ids[i]);
                REQUIRE(tensor[(offsets[i] + v) * dim + j] == Catch::Approx(expected[j]));
            }
        }
    }
}

void
RequireEmbListApisUseExternalIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data,
                                 const knowhere::Json& json) {
    RequireExternalIdMapContent(index, data.valid_ids, data.total_count);

    auto search = index.Search(data.query_ds, json, knowhere::BitsetView{});
    REQUIRE(search.has_value());
    RequireExactFirstHits(*search.value(), data.query_ids, data.valid_ids);

    std::vector<int64_t> ids(data.query_ids.begin(), data.query_ids.end());
    auto retrieve =
        index.GetEmbListByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()), "MAX_SIM_L2");
    REQUIRE(retrieve.has_value());
    RequireEmbListVectorsMatchLogicalIds(*retrieve.value(), data.query_ids, data.dim, kEmbVectorsPerDoc);

    const int64_t invalid_ids[] = {FirstMissingExternalId(data.valid_ids, data.total_count), data.total_count};
    for (auto invalid_id : invalid_ids) {
        CAPTURE(invalid_id);
        auto invalid =
            index.GetEmbListByIds(knowhere::GenIdsDataSet(1, &invalid_id), "MAX_SIM_L2");
        REQUIRE(!invalid.has_value());
        REQUIRE(invalid.error() == Status::invalid_args);
    }
}

std::vector<std::vector<float>>
BuildChunkStorage(const knowhere::DataSetPtr& dense_ds, int64_t chunks, std::vector<size_t>& chunk_lims) {
    const auto rows = dense_ds->GetRows();
    const auto dim = dense_ds->GetDim();
    auto* dense = static_cast<const float*>(dense_ds->GetTensor());
    chunk_lims.resize(chunks + 1);
    std::vector<std::vector<float>> chunk_storage(chunks);
    int64_t offset = 0;
    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        chunk_lims[chunk] = static_cast<size_t>(offset);
        const auto chunk_rows = (rows * (chunk + 1)) / chunks - (rows * chunk) / chunks;
        chunk_storage[chunk].resize(chunk_rows * dim);
        if (chunk_rows > 0) {
            std::memcpy(chunk_storage[chunk].data(), dense + offset * dim, chunk_rows * dim * sizeof(float));
        }
        offset += chunk_rows;
    }
    chunk_lims[chunks] = static_cast<size_t>(rows);
    return chunk_storage;
}

knowhere::DataSetPtr
GenNullableDenseChunkDataSet(int64_t total_count, const std::vector<int32_t>& valid_ids, int64_t dim,
                             std::vector<std::vector<float>>& chunk_storage,
                             std::vector<const float*>& chunk_ptrs,
                             std::vector<size_t>& chunk_lims) {
    auto dense_ds = GenNullableDenseDataSet(total_count, valid_ids, dim);
    chunk_storage = BuildChunkStorage(dense_ds, 2, chunk_lims);
    chunk_ptrs.resize(chunk_storage.size());
    for (size_t i = 0; i < chunk_storage.size(); ++i) {
        chunk_ptrs[i] = chunk_storage[i].data();
    }
    auto ds = knowhere::GenDataSet(dense_ds->GetRows(), dim, chunk_ptrs.data());
    ds->SetIsChunk(true);
    ds->SetIsOwner(false);
    ds->SetNumChunk(static_cast<int64_t>(chunk_ptrs.size()));
    ds->SetTensorBeginId(dense_ds->GetTensorBeginId());
    if (static_cast<int64_t>(valid_ids.size()) != total_count) {
        ds->SetInternalToExternalIds(valid_ids, static_cast<size_t>(total_count));
    }
    auto lims_data = std::make_unique<size_t[]>(chunk_lims.size());
    std::copy(chunk_lims.begin(), chunk_lims.end(), lims_data.get());
    ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(lims_data.release()));
    return ds;
}

knowhere::DataSetPtr
GenNullableEmbListChunkDataSet(int64_t total_docs, const std::vector<int32_t>& valid_doc_ids, int64_t dim,
                               int64_t vectors_per_doc, std::vector<std::vector<float>>& chunk_storage,
                               std::vector<const float*>& chunk_ptrs, std::vector<size_t>& chunk_lims,
                               std::vector<size_t>& emb_offsets) {
    auto dense_ds = GenNullableEmbListDataSet(total_docs, valid_doc_ids, dim, vectors_per_doc);
    chunk_storage = BuildChunkStorage(dense_ds, std::max<int64_t>(static_cast<int64_t>(valid_doc_ids.size()), 1),
                                      chunk_lims);
    chunk_ptrs.resize(chunk_storage.size());
    for (size_t i = 0; i < chunk_storage.size(); ++i) {
        chunk_ptrs[i] = chunk_storage[i].data();
    }
    emb_offsets.resize(valid_doc_ids.size() + 1);
    for (size_t i = 0; i <= valid_doc_ids.size(); ++i) {
        emb_offsets[i] = i * vectors_per_doc;
    }
    auto ds = knowhere::GenDataSet(dense_ds->GetRows(), dim, chunk_ptrs.data());
    ds->SetIsChunk(true);
    ds->SetIsOwner(false);
    ds->SetNumChunk(static_cast<int64_t>(chunk_ptrs.size()));
    if (static_cast<int64_t>(valid_doc_ids.size()) != total_docs) {
        ds->SetInternalToExternalIds(valid_doc_ids, static_cast<size_t>(total_docs));
    }
    auto lims_data = std::make_unique<size_t[]>(chunk_lims.size());
    std::copy(chunk_lims.begin(), chunk_lims.end(), lims_data.get());
    ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(lims_data.release()));
    return ds;
}

std::shared_ptr<milvus::FileManager>
LocalFileManager() {
    return std::make_shared<milvus::LocalFileManager>();
}

MatrixData
BuildMatrixData(const IndexRow& row, const Scenario& scenario, const fs::path& work_dir) {
    MatrixData data;
    const bool multi = scenario.mode == Mode::MultiIndex;
    if (scenario.mode == Mode::EmbList) {
        data.total_count = kEmbDocs;
        data.dim = kDenseDim;
        data.valid_ids = ValidIdsFor(scenario.nullable_ratio, kEmbDocs);
        data.selected_ids = data.valid_ids;
        data.query_ids = FirstQueryIds(data.valid_ids, kEmbDocs, 2);
        data.train_ds = GenNullableEmbListDataSet(kEmbDocs, data.valid_ids, data.dim, kEmbVectorsPerDoc);
        data.full_ds = row.index_type == knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR
                           ? data.train_ds
                           : GenFullDenseDataSet(kEmbDocs, data.dim);
        data.query_ds = GenEmbListQueryDataSet(data.query_ids, data.dim);
        return data;
    }

    if (row.data_kind == DataKind::GpuFp32) {
        data.total_count = kGpuRows;
        data.dim = kGpuDim;
    } else if (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq ||
               row.data_kind == DataKind::MinHash) {
        data.total_count = kFileRows;
        data.dim = row.data_kind == DataKind::MinHash ? 1024 : kFileDim;
    } else if (row.data_kind == DataKind::DenseBin) {
        data.total_count = kTotalRows;
        data.dim = 128;
    } else {
        data.total_count = kTotalRows;
        data.dim = kDenseDim;
    }
    data.valid_ids = ValidIdsFor(scenario.nullable_ratio, data.total_count, multi);
    data.selected_ids = multi ? PartitionIds(data.valid_ids, 0) : data.valid_ids;
    if (multi && data.selected_ids.empty() && !data.valid_ids.empty()) {
        data.selected_ids = PartitionIds(data.valid_ids, 1);
    }
    data.query_ids = FirstQueryIds(multi ? data.selected_ids : data.valid_ids, data.total_count);

    if (row.data_kind == DataKind::DenseFp32 || row.data_kind == DataKind::GpuFp32 ||
        row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq) {
        data.full_ds = GenFullDenseDataSet(data.total_count, data.dim);
        data.train_ds = GenNullableDenseDataSet(data.total_count, data.valid_ids, data.dim);
        data.query_ds = GenDenseQueryDataSet(data.query_ids, data.dim);
    } else if (row.data_kind == DataKind::DenseBin || row.data_kind == DataKind::MinHash) {
        data.full_ds = GenBinDataSet(static_cast<int>(data.total_count), static_cast<int>(data.dim), 22);
        static std::map<std::string, std::vector<uint8_t>> owned_binary;
        const auto key = work_dir.string() + "/bin_train";
        const auto query_key = work_dir.string() + "/bin_query";
        data.train_ds = GenNullableBinaryDataSet(data.total_count, data.valid_ids, data.dim, data.full_ds,
                                                 owned_binary[key]);
        data.query_ds = GenBinaryQueryDataSet(data.query_ids, data.dim, data.full_ds, owned_binary[query_key]);
    } else {
        data.full_ds = nullptr;
        data.train_ds = GenNullableSparseDataSet(data.total_count, data.valid_ids);
        data.query_ds = GenSparseQueryDataSet(data.query_ids, data.total_count);
    }

    if (multi) {
        data.train_ds->Set(knowhere::meta::SCALAR_INFO, ScalarInfoForEvenOddPartitions(data.valid_ids));
    }
    return data;
}

knowhere::Json
BuildJson(const IndexRow& row, Mode mode, const MatrixData& data, const fs::path& raw_data_file,
          const fs::path& file_index_prefix) {
    if (row.index_type == "HNSW_V1") {
        auto json = DenseConfigFor(row, mode);
        json[knowhere::meta::DIM] = data.train_ds->GetDim();
        json[knowhere::meta::METRIC_TYPE] = knowhere::metric::IP;
        json[knowhere::meta::RADIUS] = 0.5f;
        json[knowhere::meta::RANGE_FILTER] = 2.0f;
        json[knowhere::meta::RETRIEVE_FRIENDLY] = true;
        return json;
    }
    if (row.data_kind == DataKind::Sparse) {
        return SparseConfig();
    }
    if (row.data_kind == DataKind::DenseBin || row.data_kind == DataKind::MinHash) {
        auto json = BinaryConfigFor(row);
        if (row.data_kind == DataKind::MinHash) {
            json[knowhere::meta::DATA_PATH] = raw_data_file.string();
            json[knowhere::meta::INDEX_PREFIX] = file_index_prefix.string();
        }
        return json;
    }
    if (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq) {
        auto json = DiskAnnConfig(row, raw_data_file, file_index_prefix);
        if (mode == Mode::EmbList) {
            json[knowhere::meta::METRIC_TYPE] = "MAX_SIM_L2";
            json[knowhere::meta::TOPK] = kEmbTopK;
            json["emb_list_strategy"] = knowhere::meta::EMB_LIST_STRATEGY_TOKENANN;
        }
        return json;
    }
    return DenseConfigFor(row, mode);
}

void
WriteRawDataIfNeeded(const IndexRow& row, const MatrixData& data, const fs::path& raw_data_file) {
    if (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq) {
        auto* tensor = static_cast<const float*>(data.train_ds->GetTensor());
        WriteRawDataToDisk<float>(raw_data_file.string(), tensor, static_cast<uint32_t>(data.train_ds->GetRows()),
                                  static_cast<uint32_t>(data.dim));
    } else if (row.data_kind == DataKind::MinHash) {
        auto* tensor = static_cast<const knowhere::bin1*>(data.train_ds->GetTensor());
        WriteRawDataToDisk<knowhere::bin1>(raw_data_file.string(), tensor,
                                           static_cast<uint32_t>(data.train_ds->GetRows()),
                                           static_cast<uint32_t>(data.dim));
    }
}

struct CreateResult {
    knowhere::Index<knowhere::IndexNode> index;
    knowhere::Status status = knowhere::Status::success;
    bool ok = false;
};

CreateResult
CreateIndex(const IndexRow& row, const MatrixData& data, std::shared_ptr<milvus::FileManager> file_manager = nullptr) {
    CreateResult result;

#ifndef KNOWHERE_WITH_CARDINAL
    if (row.requires_cardinal) {
        result.status = knowhere::Status::invalid_index_error;
        return result;
    }
#endif
#ifndef KNOWHERE_WITH_SVS
    if (row.requires_svs) {
        result.status = knowhere::Status::invalid_index_error;
        return result;
    }
#endif
#ifndef KNOWHERE_WITH_CUVS
    if (row.requires_gpu) {
        result.status = knowhere::Status::invalid_index_error;
        return result;
    }
#endif

    const auto version = VersionForRow(row);
    auto create_with_object = [&](const knowhere::Object& object) {
        if (row.data_kind == DataKind::DenseBin || row.data_kind == DataKind::MinHash) {
            auto created = knowhere::IndexFactory::Instance().Create<knowhere::bin1>(row.index_type, version, object);
            if (!created.has_value()) {
                result.status = created.error();
                return result;
            }
            result.index = std::move(created.value());
        } else if (row.data_kind == DataKind::Sparse) {
            auto created =
                knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(row.index_type, version, object);
            if (!created.has_value()) {
                result.status = created.error();
                return result;
            }
            result.index = std::move(created.value());
        } else {
            auto created = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(row.index_type, version, object);
            if (!created.has_value()) {
                result.status = created.error();
                return result;
            }
            result.index = std::move(created.value());
        }
        result.ok = true;
        return result;
    };

    if (row.index_type == knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR && data.full_ds != nullptr) {
        knowhere::ViewDataOp view_data = [full = data.full_ds, dim = data.full_ds->GetDim()](size_t logical_id) {
            auto* tensor = static_cast<const float*>(full->GetTensor());
            return tensor + logical_id * dim;
        };
        auto object = knowhere::Pack(view_data);
        return create_with_object(object);
    } else if (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq ||
               row.data_kind == DataKind::MinHash) {
        auto object = knowhere::Pack(file_manager != nullptr ? file_manager : LocalFileManager());
        return create_with_object(object);
    }

    return create_with_object(knowhere::Object(nullptr));
}

std::map<std::string, std::shared_ptr<BuiltArtifact>>&
ArtifactCache() {
    static std::map<std::string, std::shared_ptr<BuiltArtifact>> cache;
    return cache;
}

std::string
ArtifactKey(const IndexRow& row, const Scenario& scenario) {
    return row.label + "|" + row.index_type + "|" + ModeName(scenario.mode) + "|" +
           NullableName(scenario.nullable_ratio);
}

std::string
PathSafe(std::string value) {
    for (auto& c : value) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
            c = '_';
        }
    }
    return value;
}

knowhere::Status
WriteEmbListOffsetToFile(const knowhere::DataSetPtr& dataset, size_t offset_count, const fs::path& path);

std::shared_ptr<BuiltArtifact>
GetArtifact(const IndexRow& row, const Scenario& scenario) {
    auto key = ArtifactKey(row, scenario);
    auto& cache = ArtifactCache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    auto artifact = std::make_shared<BuiltArtifact>();
    artifact->row = row;
    artifact->scenario = scenario;
    const auto work_dir_name = row.label + "_" + ModeName(scenario.mode) + "_" +
                               NullableName(scenario.nullable_ratio);
    artifact->work_dir = fs::current_path() / "nullable_external_id_map_matrix" / PathSafe(work_dir_name);
    fs::remove_all(artifact->work_dir);
    fs::create_directories(artifact->work_dir);
    artifact->main_file = artifact->work_dir / "main.index";
    artifact->id_map_file = artifact->work_dir / "main.index_id_map";
    artifact->emb_meta_file = artifact->work_dir / "emb_list_meta.bin";
    artifact->emb_raw_file = artifact->work_dir / "emb_list_raw.index";
    artifact->emb_offset_file = artifact->work_dir / "emb_list_offset.bin";
    artifact->raw_data_file = artifact->work_dir / "raw_data.bin";
    artifact->file_index_prefix = artifact->work_dir / "file_index";
    fs::create_directories(artifact->file_index_prefix.parent_path());

    artifact->data = BuildMatrixData(row, scenario, artifact->work_dir);
    if (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq ||
        row.data_kind == DataKind::MinHash) {
        artifact->file_manager = LocalFileManager();
    }
    WriteRawDataIfNeeded(row, artifact->data, artifact->raw_data_file);
    artifact->json =
        BuildJson(row, scenario.mode, artifact->data, artifact->raw_data_file, artifact->file_index_prefix);
    if (row.data_kind == DataKind::DiskAnn && scenario.mode == Mode::EmbList) {
        auto status = WriteEmbListOffsetToFile(
            artifact->data.train_ds, artifact->data.valid_ids.size() + 1, artifact->emb_offset_file);
        if (status == knowhere::Status::success) {
            artifact->json["emb_list_offset_file_path"] = artifact->emb_offset_file.string();
        } else {
            artifact->build_status = status;
        }
    }

    auto created = CreateIndex(row, artifact->data, artifact->file_manager);
    artifact->create_ok = created.ok;
    artifact->create_status = created.status;
    if (!created.ok) {
        cache[key] = artifact;
        return artifact;
    }

    artifact->index = std::move(created.index);
    artifact->build_status = artifact->index.Build(artifact->data.train_ds, artifact->json);
    cache[key] = artifact;
    return artifact;
}

knowhere::Status
EnsureSerialized(BuiltArtifact& artifact) {
    if (artifact.serialized) {
        return artifact.serialize_status;
    }
    artifact.serialize_status = artifact.index.Serialize(artifact.binset);
    artifact.serialized = true;
    return artifact.serialize_status;
}

knowhere::Status
EnsureBinaryLoaded(BuiltArtifact& artifact) {
    if (artifact.binary_loaded_ready || artifact.binary_deserialize_status != knowhere::Status::success) {
        return artifact.binary_deserialize_status;
    }
    RETURN_IF_ERROR(EnsureSerialized(artifact));
    auto created = CreateIndex(artifact.row, artifact.data, artifact.file_manager);
    if (!created.ok) {
        artifact.binary_deserialize_status = created.status;
        return artifact.binary_deserialize_status;
    }
    artifact.binary_loaded = std::move(created.index);
    artifact.binary_deserialize_status = artifact.binary_loaded.Deserialize(artifact.binset, artifact.json);
    artifact.binary_loaded_ready = artifact.binary_deserialize_status == knowhere::Status::success;
    return artifact.binary_deserialize_status;
}

knowhere::Status
WriteBinaryToFile(const knowhere::BinaryPtr& binary, const fs::path& path) {
    if (binary == nullptr) {
        return knowhere::Status::invalid_binary_set;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return knowhere::Status::disk_file_error;
    }
    out.write(reinterpret_cast<const char*>(binary->data.get()), binary->size);
    if (!out) {
        return knowhere::Status::disk_file_error;
    }
    return knowhere::Status::success;
}

knowhere::Status
WriteBytesToFile(const fs::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return knowhere::Status::disk_file_error;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        return knowhere::Status::disk_file_error;
    }
    return knowhere::Status::success;
}

knowhere::BinaryPtr
MakeBinary(const std::vector<uint8_t>& bytes) {
    auto binary = std::make_shared<knowhere::Binary>();
    binary->size = static_cast<int64_t>(bytes.size());
    binary->data = std::shared_ptr<uint8_t[]>(new uint8_t[std::max<size_t>(bytes.size(), 1)]);
    std::memcpy(binary->data.get(), bytes.data(), bytes.size());
    return binary;
}

knowhere::Status
WriteEmbListOffsetToFile(const knowhere::DataSetPtr& dataset, size_t offset_count, const fs::path& path) {
    auto offsets = dataset->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
    if (offsets == nullptr) {
        return knowhere::Status::emb_list_inner_error;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return knowhere::Status::disk_file_error;
    }
    out.write(reinterpret_cast<const char*>(&offset_count), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(offsets), offset_count * sizeof(size_t));
    if (!out) {
        return knowhere::Status::disk_file_error;
    }
    return knowhere::Status::success;
}

knowhere::BinaryPtr
MainIndexBinary(const knowhere::BinarySet& binset) {
    for (const auto& [key, binary] : binset.binary_map_) {
        if (key == knowhere::meta::EXTERNAL_ID_MAP || key == knowhere::meta::EMB_LIST_META ||
            key == knowhere::meta::EMB_LIST_RAW_INDEX) {
            continue;
        }
        return binary;
    }
    return nullptr;
}

knowhere::Json
FileLoadJson(const BuiltArtifact& artifact) {
    auto json = artifact.json;
    if (fs::exists(artifact.id_map_file) && fs::file_size(artifact.id_map_file) > 0) {
        json["external_id_map_file_path"] = artifact.id_map_file.string();
    }
    if (artifact.scenario.mode == Mode::EmbList) {
        json["emb_list_meta_file_path"] = artifact.emb_meta_file.string();
        json["emb_list_raw_index_file_path"] = artifact.emb_raw_file.string();
    }
    return json;
}

knowhere::Status
EnsureFileSerialized(BuiltArtifact& artifact) {
    if (artifact.file_serialized) {
        return artifact.file_serialize_status;
    }
    if (artifact.row.data_kind == DataKind::DiskAnn || artifact.row.data_kind == DataKind::Aisaq ||
        artifact.row.data_kind == DataKind::MinHash) {
        artifact.file_serialize_status = artifact.index.Serialize(artifact.binset);
        artifact.file_serialized = true;
        return artifact.file_serialize_status;
    }

    RETURN_IF_ERROR(EnsureSerialized(artifact));
    RETURN_IF_ERROR(WriteBinaryToFile(MainIndexBinary(artifact.binset), artifact.main_file));
    auto id_map_binary = artifact.binset.GetByName(knowhere::meta::EXTERNAL_ID_MAP);
    if (id_map_binary != nullptr) {
        RETURN_IF_ERROR(WriteBinaryToFile(id_map_binary, artifact.id_map_file));
    } else {
        std::ofstream out(artifact.id_map_file, std::ios::binary | std::ios::trunc);
    }

    if (artifact.scenario.mode == Mode::EmbList) {
        RETURN_IF_ERROR(WriteBinaryToFile(artifact.binset.GetByName(knowhere::meta::EMB_LIST_META),
                                          artifact.emb_meta_file));
        auto raw_binary = artifact.binset.GetByName(knowhere::meta::EMB_LIST_RAW_INDEX);
        if (raw_binary != nullptr) {
            RETURN_IF_ERROR(WriteBinaryToFile(raw_binary, artifact.emb_raw_file));
        }
    }

    artifact.file_serialize_status = knowhere::Status::success;
    artifact.file_serialized = true;
    return artifact.file_serialize_status;
}

knowhere::Status
EnsureFileLoaded(BuiltArtifact& artifact) {
    if (artifact.file_loaded_ready || artifact.file_deserialize_status != knowhere::Status::success) {
        return artifact.file_deserialize_status;
    }
    RETURN_IF_ERROR(EnsureFileSerialized(artifact));
    auto created = CreateIndex(artifact.row, artifact.data, artifact.file_manager);
    if (!created.ok) {
        artifact.file_deserialize_status = created.status;
        return artifact.file_deserialize_status;
    }
    artifact.file_loaded = std::move(created.index);
    if (artifact.row.data_kind == DataKind::DiskAnn || artifact.row.data_kind == DataKind::Aisaq ||
        artifact.row.data_kind == DataKind::MinHash) {
        artifact.file_deserialize_status = artifact.file_loaded.Deserialize(artifact.binset, artifact.json);
    } else {
        artifact.file_deserialize_status = artifact.file_loaded.DeserializeFromFile(artifact.main_file.string(),
                                                                                   FileLoadJson(artifact));
    }
    artifact.file_loaded_ready = artifact.file_deserialize_status == knowhere::Status::success;
    return artifact.file_deserialize_status;
}

const knowhere::Index<knowhere::IndexNode>&
IndexForSource(BuiltArtifact& artifact, IndexSource source) {
    switch (source) {
        case IndexSource::Fresh:
            return artifact.index;
        case IndexSource::BinarySet:
            REQUIRE(EnsureBinaryLoaded(artifact) == knowhere::Status::success);
            return artifact.binary_loaded;
        case IndexSource::File:
            REQUIRE(EnsureFileLoaded(artifact) == knowhere::Status::success);
            return artifact.file_loaded;
    }
    return artifact.index;
}

void
RequireBuildReady(const BuiltArtifact& artifact) {
    if (artifact.scenario.nullable_ratio == NullableRatio::R100) {
        REQUIRE(IsExpectedEmptyBuildStatus(artifact.build_status));
    } else {
        REQUIRE(artifact.build_status == knowhere::Status::success);
    }
}

void
ExecuteSearchLike(BuiltArtifact& artifact, const Scenario& scenario, Operation op) {
    const auto& index = IndexForSource(artifact, scenario.source);
    auto filtered_ids = FilterIdsFor(scenario.filter_ratio, artifact.data.total_count, artifact.data.valid_ids,
                                    scenario.mode, artifact.data.selected_ids);
    auto candidates = scenario.mode == Mode::MultiIndex ? artifact.data.selected_ids : artifact.data.valid_ids;
    auto allowed_ids = AllowedIdsAfterFilter(candidates, filtered_ids);
    const bool expect_empty = allowed_ids.empty();

    std::vector<uint8_t> bitset_data;
    knowhere::BitsetView bitset;
    const bool needs_bitset = scenario.filter_ratio != FilterRatio::None || scenario.mode == Mode::MultiIndex;
    if (needs_bitset) {
        bitset = BitsetViewFrom(bitset_data, artifact.data.total_count, filtered_ids);
    }
    milvus::OpContext op_context;

    if (op == Operation::Search || op == Operation::SearchFilter) {
        auto result = index.Search(artifact.data.query_ds, artifact.json, needs_bitset ? bitset : nullptr, &op_context);
        if (IsFreshFileBackedRead(artifact.row, scenario) && !result.has_value()) {
            REQUIRE(IsExpectedEmptySearchError(result.error()));
            return;
        }
        RequireExpectedVectorResult(result, allowed_ids, filtered_ids, expect_empty);
        if (artifact.row.exact && result.has_value() && !expect_empty) {
            RequireExactFirstHits(*result.value(), artifact.data.query_ids, allowed_ids);
        }
    } else if (op == Operation::Range || op == Operation::RangeFilter) {
        auto result =
            index.RangeSearch(artifact.data.query_ds, artifact.json, needs_bitset ? bitset : nullptr, &op_context);
        if (IsFreshFileBackedRead(artifact.row, scenario) && !result.has_value()) {
            REQUIRE(IsExpectedEmptySearchError(result.error()));
            return;
        }
        RequireExpectedRangeResult(result, allowed_ids, filtered_ids, expect_empty);
        if (artifact.row.exact && result.has_value() && !expect_empty) {
            RequireExactRangeHits(*result.value(), artifact.data.query_ids, allowed_ids);
        }
    } else if (op == Operation::Iterator || op == Operation::IteratorFilter) {
        auto result =
            index.AnnIterator(artifact.data.query_ds, artifact.json, needs_bitset ? bitset : nullptr, true, &op_context);
        if (IsFreshFileBackedRead(artifact.row, scenario) && !result.has_value()) {
            REQUIRE(IsExpectedEmptySearchError(result.error()));
            return;
        }
        RequireExpectedIteratorResult(result, allowed_ids, filtered_ids, expect_empty,
                                      artifact.row.exact ? &artifact.data.query_ids : nullptr);
    }
}

void
ExecuteSupportedScenario(const IndexRow& row, const Scenario& scenario) {
    auto artifact = GetArtifact(row, scenario);
    CAPTURE(row.label, row.index_type, scenario.name, artifact->json.dump());

    if (!artifact->create_ok) {
        REQUIRE(row.maybe_unavailable);
        REQUIRE(artifact->create_status != knowhere::Status::success);
        return;
    }

    if (scenario.op == Operation::Build) {
        RequireBuildReady(*artifact);
        return;
    }

    if (artifact->build_status != knowhere::Status::success) {
        REQUIRE(scenario.nullable_ratio == NullableRatio::R100);
        REQUIRE(IsExpectedEmptyBuildStatus(artifact->build_status));
        return;
    }

    switch (scenario.op) {
        case Operation::Search:
        case Operation::Range:
        case Operation::Iterator:
        case Operation::SearchFilter:
        case Operation::RangeFilter:
        case Operation::IteratorFilter:
            ExecuteSearchLike(*artifact, scenario, scenario.op);
            return;
        case Operation::BinarySetSerialize:
            REQUIRE(EnsureSerialized(*artifact) == knowhere::Status::success);
            REQUIRE(artifact->binset.Size() > 0);
            return;
        case Operation::BinarySetDeserialize:
            REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
            REQUIRE(artifact->binary_loaded.ExternalCount() == artifact->index.ExternalCount());
            return;
        case Operation::FileSerialize:
            REQUIRE(EnsureFileSerialized(*artifact) == knowhere::Status::success);
            return;
        case Operation::FileDeserialize:
            REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
            if (artifact->row.data_kind == DataKind::MinHash && scenario.nullable_ratio == NullableRatio::R0) {
                REQUIRE(artifact->file_loaded.ExternalCount() == artifact->data.total_count);
            } else {
                REQUIRE(artifact->file_loaded.ExternalCount() == artifact->index.ExternalCount());
            }
            return;
        case Operation::Build:
            return;
    }
}

bool
IsExpectedUnsupportedStatus(knowhere::Status status) {
    return status != knowhere::Status::success;
}

void
RequireExpectedUnsupportedStatus(knowhere::Status status) {
    REQUIRE(IsExpectedUnsupportedStatus(status));
}

void
ExecuteUnsupportedScenario(const IndexRow& row, const Scenario& scenario) {
    if (!SupportsModeBuild(row, scenario.mode)) {
        REQUIRE_FALSE(SupportsScenario(row, scenario));
        return;
    }

    auto artifact = GetArtifact(row, scenario);
    CAPTURE(row.label, row.index_type, scenario.name, artifact->json.dump());

    if (!artifact->create_ok) {
        RequireExpectedUnsupportedStatus(artifact->create_status);
        return;
    }

    if (scenario.op == Operation::Build) {
        return;
    }

    if (artifact->build_status != knowhere::Status::success) {
        RequireExpectedUnsupportedStatus(artifact->build_status);
        return;
    }

    switch (scenario.op) {
        case Operation::Search:
        case Operation::Range:
        case Operation::Iterator:
        case Operation::SearchFilter:
        case Operation::RangeFilter:
        case Operation::IteratorFilter:
            SUCCEED("unsupported read operation is covered by capability matrix");
            return;
        case Operation::BinarySetSerialize:
            (void)EnsureSerialized(*artifact);
            return;
        case Operation::BinarySetDeserialize: {
            auto serialize_status = EnsureSerialized(*artifact);
            if (serialize_status != knowhere::Status::success) {
                RequireExpectedUnsupportedStatus(serialize_status);
                return;
            }
            (void)EnsureBinaryLoaded(*artifact);
            return;
        }
        case Operation::FileSerialize:
            (void)EnsureFileSerialized(*artifact);
            return;
        case Operation::FileDeserialize: {
            auto serialize_status = EnsureFileSerialized(*artifact);
            if (serialize_status != knowhere::Status::success) {
                RequireExpectedUnsupportedStatus(serialize_status);
                return;
            }
            (void)EnsureFileLoaded(*artifact);
            return;
        }
        case Operation::Build:
            return;
    }
}

void
ExecuteScenario(const IndexRow& row, const Scenario& scenario) {
    const bool supported = SupportsScenario(row, scenario);
    if (!supported) {
        ExecuteUnsupportedScenario(row, scenario);
        return;
    }
    ExecuteSupportedScenario(row, scenario);
}

std::vector<uint8_t>
MakeValidBitmap(size_t total_count, const std::vector<int32_t>& valid_ids) {
    return MakeBitmap(total_count, valid_ids);
}

#ifdef KNOWHERE_WITH_CARDINAL
void
ConfigureTieredStorageForNullableMatrix() {
    static const bool configured = [] {
        static const int64_t mb = 1024 * 1024;
        milvus::cachinglayer::Manager::ConfigureTieredStorage(
            {CacheWarmupPolicy::CacheWarmupPolicy_Disable, CacheWarmupPolicy::CacheWarmupPolicy_Disable,
             CacheWarmupPolicy::CacheWarmupPolicy_Disable, CacheWarmupPolicy::CacheWarmupPolicy_Disable},
            {1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb, 1024 * mb}, true, true, {10, false, 30},
            std::chrono::milliseconds(1000));
        return true;
    }();
    (void)configured;
}
#endif

void
RequireExternalIdMapForwarding(knowhere::IndexNode& node) {
    knowhere::ExternalIdMap map;
    map.SetInternalToExternalIds(std::vector<int32_t>{1, 3}, 4);
    REQUIRE(node.SetExternalIdMap(std::move(map)) == Status::success);
    REQUIRE(node.ExternalCount() == 4);
    REQUIRE(node.GetValidBitmapView().size == 4);
    REQUIRE(node.GetExternalIdMap().ToExternalId(1) == 3);
}

void
RequireReadApisReturnError(knowhere::IndexNode& node) {
    auto dataset = GenNullableDenseDataSet(4, {0, 2}, kDenseDim);
    auto query = GenDenseQueryDataSet({0}, kDenseDim);
    int64_t id = 0;
    auto ids = knowhere::GenIdsDataSet(1, &id);
    auto new_config = [] { return std::make_unique<knowhere::BaseConfig>(); };

    auto search = node.Search(query, new_config(), knowhere::BitsetView{});
    REQUIRE(!search.has_value());
    REQUIRE(search.error() == Status::not_implemented);

    auto range = node.RangeSearch(query, new_config(), knowhere::BitsetView{});
    REQUIRE(!range.has_value());
    REQUIRE(range.error() == Status::not_implemented);

    auto iterator = node.AnnIterator(query, new_config(), knowhere::BitsetView{});
    REQUIRE(!iterator.has_value());
    REQUIRE(iterator.error() == Status::not_implemented);

    auto vector = node.GetVectorByIds(ids);
    REQUIRE(!vector.has_value());
    REQUIRE(vector.error() == Status::not_implemented);

    auto calc_dist = node.CalcDistByIDs(dataset, knowhere::BitsetView{}, &id, 1, false);
    REQUIRE(!calc_dist.has_value());
    REQUIRE(calc_dist.error() == Status::not_implemented);

    auto emb_list = node.GetEmbListByIds(ids, knowhere::metric::L2);
    REQUIRE(!emb_list.has_value());
    REQUIRE(emb_list.error() == Status::emb_list_inner_error);
}

struct RequiredIndexRow {
    const char* label;
    const char* index_type;
    DataKind data_kind;
};

bool
HasIndexRow(const std::vector<IndexRow>& rows, const RequiredIndexRow& required) {
    return std::any_of(rows.begin(), rows.end(), [&](const IndexRow& row) {
        return row.label == required.label && row.index_type == required.index_type &&
               row.data_kind == required.data_kind;
    });
}

}  // namespace

TEST_CASE("Nullable ExternalIdMap and DataSet primitives", "[nullable][external_id_map]") {
    SECTION("DataSet valid bitmap builds 0 50 100 percent mappings") {
        auto full_ds = GenNullableDenseDataSet(8, AllIds(8), 4);
        auto full_bitmap = MakeValidBitmap(8, AllIds(8));
        full_ds->SetValidBitmap(full_bitmap.data(), 8);
        REQUIRE(full_ds->GetInternalToExternalIds().empty());
        REQUIRE(full_ds->GetExternalCount() == 0);

        auto half_ds = GenNullableDenseDataSet(8, {0, 2, 4, 6}, 4);
        auto half_bitmap = MakeValidBitmap(8, {0, 2, 4, 6});
        half_ds->SetValidBitmap(half_bitmap.data(), 8);
        REQUIRE(half_ds->GetInternalToExternalIds() == std::vector<int32_t>{0, 2, 4, 6});
        REQUIRE(half_ds->GetExternalCount() == 8);

        auto null_ds = GenNullableDenseDataSet(8, {}, 4);
        auto empty_bitmap = MakeValidBitmap(8, {});
        null_ds->SetValidBitmap(empty_bitmap.data(), 8);
        REQUIRE(null_ds->GetInternalToExternalIds().empty());
        REQUIRE(null_ds->GetExternalCount() == 8);
    }

    SECTION("ExternalIdMap serializes vector and emb-list mappings") {
        knowhere::ExternalIdMap map;
        map.SetInternalToExternalIds(std::vector<int32_t>{0, 2, 4, 6}, 8);
        REQUIRE(map.HasInternalToExternalIds());
        REQUIRE(map.ExternalCount(0) == 8);
        REQUIRE(map.ToExternalId(2) == 4);
        REQUIRE(map.ToInternalId(4) == 2);
        REQUIRE(map.ToInternalId(5) == -1);

        std::vector<int64_t> result_ids = {0, 1, 2, 3, -1};
        map.MapResultIds(result_ids);
        REQUIRE(result_ids == std::vector<int64_t>{0, 2, 4, 6, -1});

        map.SetExternalIdOffset(100);
        REQUIRE(map.ToExternalId(1) == 2);
        REQUIRE(map.ToInternalId(2) == 1);

        map.AppendInternalToEmbListIds(0, 2, 1);
        map.AppendInternalToEmbListIds(2, 4, 2);
        REQUIRE(map.GetInternalToEmbListIds() == std::vector<int32_t>{2, 2, 4, 4});

        std::vector<uint8_t> binary(map.BinarySize());
        map.Serialize(binary.data(), binary.size());
        knowhere::ExternalIdMap loaded;
        loaded.Deserialize(binary.data(), static_cast<int64_t>(binary.size()));
        REQUIRE(loaded.ExternalCount(0) == 8);
        REQUIRE(loaded.GetInternalToExternalIds() == std::vector<int32_t>{0, 2, 4, 6});
    }

    SECTION("ExternalIdMap preserves all-null external count") {
        knowhere::ExternalIdMap map;
        map.SetInternalToExternalIds(nullptr, 0, 8);
        REQUIRE(!map.HasInternalToExternalIds());
        REQUIRE(map.ExternalCount(0) == 8);
        auto valid = map.GetValidBitmapView();
        REQUIRE(valid.size == 8);
        REQUIRE(valid.data != nullptr);
        REQUIRE(valid.data[0] == 0);

        std::vector<uint8_t> binary(map.BinarySize());
        map.Serialize(binary.data(), binary.size());
        knowhere::ExternalIdMap loaded;
        loaded.Deserialize(binary.data(), static_cast<int64_t>(binary.size()));
        REQUIRE(loaded.ExternalCount(0) == 8);
        REQUIRE(loaded.GetInternalToExternalIds().empty());
        REQUIRE(loaded.GetValidBitmapView().size == 8);
    }

    SECTION("ExternalIdMap rejects invalid mappings") {
        knowhere::ExternalIdMap map;
        REQUIRE_THROWS(map.SetInternalToExternalIds(std::vector<int32_t>{0, 0}, 4));
        REQUIRE_THROWS(map.SetInternalToExternalIds(std::vector<int32_t>{0, 4}, 4));
        REQUIRE_THROWS(map.SetInternalToExternalIds(std::vector<int32_t>{0, 1, 2}, 2));
        REQUIRE_THROWS(map.SetInternalToExternalIds(nullptr, 1, 4));

        const int32_t ids[] = {0, 2};
        REQUIRE_THROWS(map.SetInternalToExternalIds(ids, 2, 1));
        REQUIRE_THROWS(map.SetInternalToExternalIds(ids, -1, 4));
    }

    SECTION("ExternalIdMap rejects malformed binary payloads") {
        knowhere::ExternalIdMap map;

        std::vector<uint8_t> too_small(sizeof(uint64_t), 0);
        REQUIRE_THROWS(map.Deserialize(too_small.data(), static_cast<int64_t>(too_small.size())));

        std::vector<uint8_t> truncated(2 * sizeof(uint64_t), 0);
        uint64_t external_count = 8;
        uint64_t map_size = 2;
        std::memcpy(truncated.data(), &external_count, sizeof(uint64_t));
        std::memcpy(truncated.data() + sizeof(uint64_t), &map_size, sizeof(uint64_t));
        REQUIRE_THROWS(map.Deserialize(truncated.data(), static_cast<int64_t>(truncated.size())));

        std::vector<uint8_t> duplicate(2 * sizeof(uint64_t) + 2 * sizeof(int32_t), 0);
        int32_t duplicate_ids[] = {2, 2};
        std::memcpy(duplicate.data(), &external_count, sizeof(uint64_t));
        std::memcpy(duplicate.data() + sizeof(uint64_t), &map_size, sizeof(uint64_t));
        std::memcpy(duplicate.data() + 2 * sizeof(uint64_t), duplicate_ids, sizeof(duplicate_ids));
        REQUIRE_THROWS(map.Deserialize(duplicate.data(), static_cast<int64_t>(duplicate.size())));
    }

    SECTION("ExternalIdMap applies runtime identity offset") {
        knowhere::ExternalIdMap map;
        map.SetExternalIdOffset(10);
        REQUIRE(map.HasResultIdMap());
        REQUIRE(!map.HasInternalToExternalIds());
        REQUIRE(map.ToExternalId(2) == 12);
        REQUIRE(map.ToInternalId(12) == 2);
        REQUIRE(map.ToInternalId(9) == -1);

        std::vector<int64_t> result_ids = {0, 2, -1};
        map.MapResultIds(result_ids);
        REQUIRE(result_ids == std::vector<int64_t>{10, 12, -1});

        std::vector<int64_t> external_ids = {10, 12, 9};
        std::vector<int64_t> internal_ids;
        const auto* mapped_ids = map.ToInternalIds(external_ids.data(), external_ids.size(), internal_ids);
        REQUIRE(mapped_ids == internal_ids.data());
        REQUIRE(internal_ids == std::vector<int64_t>{0, 2, -1});

        std::vector<uint8_t> filter_bits(2, 0);
        filter_bits[12 >> 3] |= static_cast<uint8_t>(1U << (12 & 7));
        knowhere::BitsetView bitset(filter_bits.data(), 16, 1);
        map.SetOutIdsToBitset(bitset);
        REQUIRE(!bitset.has_out_ids());
        REQUIRE(!bitset.test(1));
        REQUIRE(bitset.test(2));

        knowhere::BitsetView chunk_bitset(filter_bits.data(), 16, 1);
        map.SetOutIdsToBitset(chunk_bitset, 0, std::nullopt, 2);
        REQUIRE(chunk_bitset.test(0));
    }

    SECTION("ExternalIdMap appends build-add batches") {
        knowhere::ExternalIdMap map;
        map.SetInternalToExternalIds(std::vector<int32_t>{0, 2, 4}, 6);
        const int32_t second_batch[] = {5};
        map.AddInternalToExternalIds(second_batch, 1, 7, 3);
        REQUIRE(map.GetInternalToExternalIds() == std::vector<int32_t>{0, 2, 4, 5});
        REQUIRE(map.ExternalCount(0) == 7);
        REQUIRE(map.ToInternalId(5) == 3);

        const int32_t nullable_after_full[] = {4, 6};
        knowhere::ExternalIdMap map_from_full;
        map_from_full.AddInternalToExternalIds(nullable_after_full, 2, 8, 3);
        REQUIRE(map_from_full.GetInternalToExternalIds() == std::vector<int32_t>{0, 1, 2, 4, 6});
        REQUIRE(map_from_full.ExternalCount(0) == 8);

        map_from_full.AddInternalToExternalIds(nullptr, 2, 0, 5);
        REQUIRE(map_from_full.GetInternalToExternalIds() == std::vector<int32_t>{0, 1, 2, 4, 6, 8, 9});
        REQUIRE(map_from_full.ExternalCount(0) == 10);
    }

    SECTION("Data type conversion preserves vector and emb-list nullable metadata") {
        auto vector_ds = GenNullableDenseDataSet(8, {0, 2, 4, 6}, 4);
        auto converted_vector = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(vector_ds, 1, 2);
        REQUIRE(converted_vector->GetRows() == 2);
        REQUIRE(converted_vector->GetInternalToExternalIds() == std::vector<int32_t>{2, 4});
        REQUIRE(converted_vector->GetExternalCount() == 8);

        auto emb_ds = GenNullableEmbListDataSet(8, {1, 3, 5}, 4, 2);
        auto converted_emb = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(emb_ds);
        REQUIRE(converted_emb->GetRows() == emb_ds->GetRows());
        REQUIRE(converted_emb->GetInternalToExternalIds() == std::vector<int32_t>{1, 3, 5});
        REQUIRE(converted_emb->GetExternalCount() == 8);
        auto offsets = converted_emb->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(offsets != nullptr);
        REQUIRE(offsets[0] == 0);
        REQUIRE(offsets[3] == 6);
    }

    SECTION("IndexNode wrappers forward nullable external id map state") {
        auto data_mock = knowhere::IndexNodeDataMockWrapper<knowhere::fp16>(
            std::make_unique<NullableWrapperFakeIndexNode>());
        RequireExternalIdMapForwarding(data_mock);
        RequireReadApisReturnError(data_mock);

        auto thread_pool = knowhere::IndexNodeThreadPoolWrapper(
            std::make_unique<NullableWrapperFakeIndexNode>(), 1);
        RequireExternalIdMapForwarding(thread_pool);
        RequireReadApisReturnError(thread_pool);
    }
}

TEST_CASE("Nullable Flat Add appends external id map", "[nullable][flat][add]") {
    auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, 1);
    auto first_batch = GenNullableDenseDataSet(8, {0, 2}, kDenseDim);
    auto second_batch = GenNullableDenseDataSet(8, {4, 6}, kDenseDim);
    auto query = GenDenseQueryDataSet({0, 6}, kDenseDim);

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version)
                     .value();
    REQUIRE(index.Train(first_batch, json) == Status::success);
    REQUIRE(index.Add(first_batch, json) == Status::success);
    REQUIRE(index.Add(second_batch, json) == Status::success);

    auto result = index.Search(query, json, knowhere::BitsetView{});
    REQUIRE(result.has_value());
    const auto* ids = result.value()->GetIds();
    REQUIRE(ids[0] == 0);
    REQUIRE(ids[1] == 6);

    int64_t retrieve_ids[] = {0, 6};
    auto retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(2, retrieve_ids));
    REQUIRE(retrieve.has_value());
    const auto* retrieved_data = static_cast<const float*>(retrieve.value()->GetTensor());
    REQUIRE(retrieved_data[0] == 2.0f);
    REQUIRE(retrieved_data[kDenseDim] == 601.0f);
}

TEST_CASE("Nullable SCANN_DVR emb-list cosine rerank maps calc-dist ids", "[nullable][scann_dvr][emblist]") {
    IndexRow row{"SCANN_DVR", knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::EmbList;
    scenario.nullable_ratio = NullableRatio::R50;

    auto work_dir = fs::temp_directory_path() / "knowhere_nullable_scann_dvr_calc_dist";
    auto data = BuildMatrixData(row, scenario, work_dir);
    auto json = DenseConfigFor(row, Mode::EmbList);
    json[knowhere::meta::METRIC_TYPE] = "MAX_SIM_COSINE";

    auto created = CreateIndex(row, data);
    REQUIRE(created.ok);
    REQUIRE(created.index.Build(data.train_ds, json) == Status::success);

    auto result = created.index.Search(data.query_ds, json, knowhere::BitsetView{});
    REQUIRE(result.has_value());
    RequireExpectedVectorResult(result, data.valid_ids, {}, false);
}

TEST_CASE("Nullable IVF_FLAT vector APIs preserve logical ids across reload", "[nullable][ivf][api]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    auto artifact = GetArtifact(row, scenario);
    CAPTURE(artifact->json.dump());
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);

    RequireDenseVectorApisUseExternalIds(artifact->index, artifact->data, artifact->json);

    REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
    RequireDenseVectorApisUseExternalIds(artifact->binary_loaded, artifact->data, artifact->json);

    REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
    RequireDenseVectorApisUseExternalIds(artifact->file_loaded, artifact->data, artifact->json);
}

TEST_CASE("Nullable IVF_FLAT emb-list APIs preserve list ids across reload", "[nullable][ivf][emblist][api]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::EmbList;
    scenario.nullable_ratio = NullableRatio::R50;

    auto artifact = GetArtifact(row, scenario);
    CAPTURE(artifact->json.dump());
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);

    RequireEmbListApisUseExternalIds(artifact->index, artifact->data, artifact->json);

    REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
    RequireEmbListApisUseExternalIds(artifact->binary_loaded, artifact->data, artifact->json);

    REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
    RequireEmbListApisUseExternalIds(artifact->file_loaded, artifact->data, artifact->json);
}

TEST_CASE("Nullable retrieval APIs reject invalid logical ids", "[nullable][api][invalid]") {
    const std::vector<IndexRow> rows = {
        {"FLAT", knowhere::IndexEnum::INDEX_FAISS_IDMAP, DataKind::DenseFp32},
        {"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32},
        {"HNSWLIB_DEPRECATED", "HNSWLIB_DEPRECATED", DataKind::DenseFp32},
        {"DISKANN (Knowhere native)", knowhere::IndexEnum::INDEX_DISKANN, DataKind::DiskAnn},
        {"AISAQ", knowhere::IndexEnum::INDEX_AISAQ, DataKind::Aisaq},
        {"SPARSE_INVERTED_INDEX_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
         DataKind::Sparse},
#ifdef KNOWHERE_WITH_CARDINAL
        {"HNSW (Cardinal v2)", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32, Capabilities{}, false, true},
        {"SPARSE_INVERTED_INDEX_CC (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
         DataKind::Sparse, Capabilities{}, false, true, false, false, true},
#endif
        {"MINHASH_LSH", knowhere::IndexEnum::INDEX_MINHASH_LSH, DataKind::MinHash},
    };

    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;
    for (const auto& row : rows) {
        CAPTURE(row.label, row.index_type);
        auto artifact = GetArtifact(row, scenario);
        REQUIRE(artifact->create_ok);
        REQUIRE(artifact->build_status == knowhere::Status::success);
        RequireExternalIdMapContent(artifact->index, artifact->data.valid_ids, artifact->data.total_count);
        const auto& index = [&]() -> const knowhere::Index<knowhere::IndexNode>& {
            if (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq ||
                row.data_kind == DataKind::MinHash) {
                REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
                return artifact->file_loaded;
            }
            return artifact->index;
        }();
        RequireExternalIdMapContent(index, artifact->data.valid_ids, artifact->data.total_count);
        RequireGetVectorRejectsInvalidExternalIds(index, artifact->data);
    }
}

TEST_CASE("Nullable CalcDistByIDs rejects invalid logical ids", "[nullable][api][invalid]") {
    const std::vector<IndexRow> rows = {
        {"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32},
#ifdef KNOWHERE_WITH_CARDINAL
        {"HNSW (Cardinal v2)", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32, Capabilities{}, false, true},
#endif
    };

    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;
    for (const auto& row : rows) {
        CAPTURE(row.label, row.index_type);
        auto artifact = GetArtifact(row, scenario);
        REQUIRE(artifact->create_ok);
        REQUIRE(artifact->build_status == knowhere::Status::success);
        const auto& index = [&]() -> const knowhere::Index<knowhere::IndexNode>& {
            if (row.data_kind == DataKind::DiskAnn) {
                REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
                return artifact->file_loaded;
            }
            return artifact->index;
        }();
        RequireDenseCalcDistRejectsInvalidExternalIds(index, artifact->data);
    }
}

TEST_CASE("Nullable loading rejects corrupted external id maps", "[nullable][file][negative]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;
    auto artifact = GetArtifact(row, scenario);
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);

    SECTION("binaryset external id map") {
        REQUIRE(EnsureSerialized(*artifact) == knowhere::Status::success);
        auto corrupt_binset = artifact->binset;
        corrupt_binset.Append(knowhere::meta::EXTERNAL_ID_MAP, MakeBinary({0x01, 0x02, 0x03}));
        auto created = CreateIndex(row, artifact->data, artifact->file_manager);
        REQUIRE(created.ok);
        REQUIRE(created.index.Deserialize(corrupt_binset, artifact->json) == knowhere::Status::invalid_binary_set);
    }

    SECTION("file external id map") {
        REQUIRE(EnsureFileSerialized(*artifact) == knowhere::Status::success);
        auto corrupt_id_map = artifact->work_dir / "corrupt_id_map.bin";
        REQUIRE(WriteBytesToFile(corrupt_id_map, {0x01, 0x02, 0x03}) == knowhere::Status::success);
        auto json = FileLoadJson(*artifact);
        json["external_id_map_file_path"] = corrupt_id_map.string();
        auto created = CreateIndex(row, artifact->data, artifact->file_manager);
        REQUIRE(created.ok);
        REQUIRE(created.index.DeserializeFromFile(artifact->main_file.string(), json) ==
                knowhere::Status::invalid_binary_set);
    }
}

TEST_CASE("Nullable file loading rejects corrupted persisted data", "[nullable][file][negative]") {
    SECTION("main index file") {
        IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
        Scenario scenario;
        scenario.mode = Mode::Vector;
        scenario.nullable_ratio = NullableRatio::R50;
        auto artifact = GetArtifact(row, scenario);
        REQUIRE(artifact->create_ok);
        REQUIRE(artifact->build_status == knowhere::Status::success);
        REQUIRE(EnsureFileSerialized(*artifact) == knowhere::Status::success);

        auto corrupt_main = artifact->work_dir / "corrupt_main.index";
        REQUIRE(WriteBytesToFile(corrupt_main, {0x01, 0x02, 0x03}) == knowhere::Status::success);
        auto created = CreateIndex(row, artifact->data, artifact->file_manager);
        REQUIRE(created.ok);
        auto status = created.index.DeserializeFromFile(corrupt_main.string(), FileLoadJson(*artifact));
        REQUIRE(status != knowhere::Status::success);
    }

    SECTION("emb-list metadata file") {
        IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
        Scenario scenario;
        scenario.mode = Mode::EmbList;
        scenario.nullable_ratio = NullableRatio::R50;
        auto artifact = GetArtifact(row, scenario);
        REQUIRE(artifact->create_ok);
        REQUIRE(artifact->build_status == knowhere::Status::success);
        REQUIRE(EnsureFileSerialized(*artifact) == knowhere::Status::success);

        auto corrupt_meta = artifact->work_dir / "corrupt_emb_list_meta.bin";
        REQUIRE(WriteBytesToFile(corrupt_meta, {0x01, 0x02, 0x03}) == knowhere::Status::success);
        auto json = FileLoadJson(*artifact);
        json["emb_list_meta_file_path"] = corrupt_meta.string();
        auto created = CreateIndex(row, artifact->data, artifact->file_manager);
        REQUIRE(created.ok);
        auto status = created.index.DeserializeFromFile(artifact->main_file.string(), json);
        REQUIRE(status != knowhere::Status::success);
    }

#ifdef KNOWHERE_WITH_CARDINAL
    SECTION("cardinal metadata file") {
        IndexRow row{"HNSW (Cardinal v2)", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32,
                     Capabilities{}, false, true};
        Scenario scenario;
        scenario.mode = Mode::Vector;
        scenario.nullable_ratio = NullableRatio::R50;
        auto artifact = GetArtifact(row, scenario);
        REQUIRE(artifact->create_ok);
        REQUIRE(artifact->build_status == knowhere::Status::success);
        REQUIRE(EnsureFileSerialized(*artifact) == knowhere::Status::success);

        auto corrupt_cardinal = artifact->work_dir / "corrupt_cardinal.index";
        REQUIRE(WriteBytesToFile(corrupt_cardinal, {0x01, 0x02, 0x03}) == knowhere::Status::success);
        auto created = CreateIndex(row, artifact->data, artifact->file_manager);
        REQUIRE(created.ok);
        auto status = created.index.DeserializeFromFile(corrupt_cardinal.string(), FileLoadJson(*artifact));
        REQUIRE(status != knowhere::Status::success);
    }
#endif
}

TEST_CASE("Nullable mmap file load preserves large non-identity mapping", "[nullable][mmap][api]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    BuiltArtifact artifact;
    artifact.row = row;
    artifact.scenario = scenario;
    artifact.work_dir = fs::current_path() / "nullable_external_id_map_matrix" / "ivf_mmap_large_mapping";
    fs::remove_all(artifact.work_dir);
    fs::create_directories(artifact.work_dir);
    artifact.main_file = artifact.work_dir / "main.index";
    artifact.id_map_file = artifact.work_dir / "main.index_id_map";
    artifact.data.total_count = 160;
    artifact.data.dim = kDenseDim;
    artifact.data.valid_ids = SwappedEvenIds(artifact.data.total_count);
    artifact.data.query_ids = FirstQueryIds(artifact.data.valid_ids, artifact.data.total_count);
    artifact.data.train_ds =
        GenNullableDenseDataSet(artifact.data.total_count, artifact.data.valid_ids, artifact.data.dim);
    artifact.data.query_ds = GenDenseQueryDataSet(artifact.data.query_ids, artifact.data.dim);
    artifact.json = DenseConfigFor(row, Mode::Vector);

    auto created = CreateIndex(row, artifact.data);
    REQUIRE(created.ok);
    artifact.index = std::move(created.index);
    artifact.build_status = artifact.index.Build(artifact.data.train_ds, artifact.json);
    REQUIRE(artifact.build_status == knowhere::Status::success);
    REQUIRE(EnsureFileSerialized(artifact) == knowhere::Status::success);

    auto load_json = FileLoadJson(artifact);
    load_json["enable_mmap"] = true;
    auto loaded = CreateIndex(row, artifact.data);
    REQUIRE(loaded.ok);
    REQUIRE(loaded.index.DeserializeFromFile(artifact.main_file.string(), load_json) == knowhere::Status::success);
    RequireDenseVectorApisUseExternalIds(loaded.index, artifact.data, artifact.json);
}

TEST_CASE("Nullable BruteForce component maps logical ids", "[nullable][bruteforce]") {
    const auto nullable_ratios = {NullableRatio::R0, NullableRatio::R50, NullableRatio::R100};
    const auto filter_ratios = {FilterRatio::R0, FilterRatio::R50, FilterRatio::R100};

    SECTION("dense vector") {
        auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, kTopK);
        for (auto nullable_ratio : nullable_ratios) {
            auto valid_ids = ValidIdsFor(nullable_ratio, kTotalRows);
            auto query_ids = FirstQueryIds(valid_ids, kTotalRows);
            auto base_ds = GenNullableDenseDataSet(kTotalRows, valid_ids, kDenseDim);
            auto query_ds = GenDenseQueryDataSet(query_ids, kDenseDim);

            std::vector<std::vector<float>> chunk_storage;
            std::vector<const float*> chunk_ptrs;
            std::vector<size_t> chunk_lims;
            auto chunk_base_ds =
                GenNullableDenseChunkDataSet(kTotalRows, valid_ids, kDenseDim, chunk_storage, chunk_ptrs, chunk_lims);

            for (auto filter_ratio : filter_ratios) {
                CAPTURE(NullableName(nullable_ratio), FilterName(filter_ratio), "dense vector");
                auto filtered_ids = FilterIdsFor(filter_ratio, kTotalRows, valid_ids, Mode::Vector, valid_ids);
                auto allowed_ids = AllowedIdsAfterFilter(valid_ids, filtered_ids);
                const bool expect_empty = allowed_ids.empty();
                std::vector<uint8_t> bitset_data;
                auto bitset = BitsetViewFrom(bitset_data, kTotalRows, filtered_ids);

                auto search = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, json, bitset);
                RequireExpectedVectorResult(search, allowed_ids, filtered_ids, expect_empty);
                if (search.has_value() && !expect_empty) {
                    RequireExactFirstHits(*search.value(), query_ids, allowed_ids);
                }

                std::vector<int64_t> ids(query_ds->GetRows() * kTopK, -1);
                std::vector<float> dists(query_ds->GetRows() * kTopK, 0.0f);
                auto status = knowhere::BruteForce::SearchWithBuf<knowhere::fp32>(
                    base_ds, query_ds, ids.data(), dists.data(), json, bitset);
                RequireBufferedStatus(status, ids, allowed_ids, filtered_ids, expect_empty);
                if (status == knowhere::Status::success && !expect_empty) {
                    RequireBufferedExactFirstHits(ids, kTopK, query_ids, allowed_ids);
                }

                auto range = knowhere::BruteForce::RangeSearch<knowhere::fp32>(base_ds, query_ds, json, bitset);
                RequireExpectedRangeResult(range, allowed_ids, filtered_ids, expect_empty);
                if (range.has_value() && !expect_empty) {
                    RequireExactRangeHits(*range.value(), query_ids, allowed_ids);
                }

                auto iterators = knowhere::BruteForce::AnnIterator<knowhere::fp32>(base_ds, query_ds, json, bitset);
                RequireExpectedIteratorResult(iterators, allowed_ids, filtered_ids, expect_empty, &query_ids);

                auto chunk_search =
                    knowhere::BruteForce::Search<knowhere::fp32>(chunk_base_ds, query_ds, json, bitset);
                RequireExpectedVectorResult(chunk_search, allowed_ids, filtered_ids, expect_empty);
                if (chunk_search.has_value() && !expect_empty) {
                    RequireExactFirstHits(*chunk_search.value(), query_ids, allowed_ids);
                }

            }
        }
    }

    SECTION("sparse vector") {
        auto json = SparseConfig();
        for (auto nullable_ratio : nullable_ratios) {
            auto valid_ids = ValidIdsFor(nullable_ratio, kTotalRows);
            auto query_ids = FirstQueryIds(valid_ids, kTotalRows);
            auto base_ds = GenNullableSparseDataSet(kTotalRows, valid_ids);
            auto query_ds = GenSparseQueryDataSet(query_ids, kTotalRows);

            for (auto filter_ratio : filter_ratios) {
                CAPTURE(NullableName(nullable_ratio), FilterName(filter_ratio), "sparse vector");
                auto filtered_ids = FilterIdsFor(filter_ratio, kTotalRows, valid_ids, Mode::Vector, valid_ids);
                auto allowed_ids = AllowedIdsAfterFilter(valid_ids, filtered_ids);
                const bool expect_empty = allowed_ids.empty();
                std::vector<uint8_t> bitset_data;
                auto bitset = BitsetViewFrom(bitset_data, kTotalRows, filtered_ids);

                auto search = knowhere::BruteForce::SearchSparse(base_ds, query_ds, json, bitset);
                RequireExpectedVectorResult(search, allowed_ids, filtered_ids, expect_empty);
                if (search.has_value() && !expect_empty) {
                    RequireExactFirstHits(*search.value(), query_ids, allowed_ids);
                }

                std::vector<knowhere::sparse::label_t> sparse_ids(query_ds->GetRows() * kTopK, -1);
                std::vector<float> dists(query_ds->GetRows() * kTopK, 0.0f);
                auto status = knowhere::BruteForce::SearchSparseWithBuf(
                    base_ds, query_ds, sparse_ids.data(), dists.data(), json, bitset);
                std::vector<int64_t> ids(sparse_ids.begin(), sparse_ids.end());
                RequireBufferedStatus(status, ids, allowed_ids, filtered_ids, expect_empty);
                if (status == knowhere::Status::success && !expect_empty) {
                    RequireBufferedExactFirstHits(ids, kTopK, query_ids, allowed_ids);
                }

                auto range = knowhere::BruteForce::RangeSearch<knowhere::sparse::SparseRow<float>>(
                    base_ds, query_ds, json, bitset);
                RequireExpectedRangeResult(range, allowed_ids, filtered_ids, expect_empty);
                if (range.has_value() && !expect_empty) {
                    RequireExactRangeHits(*range.value(), query_ids, allowed_ids);
                }

                auto iterators = knowhere::BruteForce::AnnIterator<knowhere::sparse::SparseRow<float>>(
                    base_ds, query_ds, json, bitset);
                RequireExpectedIteratorResult(iterators, allowed_ids, filtered_ids, expect_empty, &query_ids);
            }
        }
    }

    SECTION("emb-list") {
        auto json = BaseDenseConfig("MAX_SIM_L2", kDenseDim, kEmbTopK);
        for (auto nullable_ratio : nullable_ratios) {
            auto valid_ids = ValidIdsFor(nullable_ratio, kEmbDocs);
            auto query_ids = FirstQueryIds(valid_ids, kEmbDocs, 2);
            auto base_ds = GenNullableEmbListDataSet(kEmbDocs, valid_ids, kDenseDim, kEmbVectorsPerDoc);
            auto query_ds = GenEmbListQueryDataSet(query_ids, kDenseDim);

            std::vector<std::vector<float>> chunk_storage;
            std::vector<const float*> chunk_ptrs;
            std::vector<size_t> chunk_lims;
            std::vector<size_t> emb_offsets;
            auto chunk_base_ds = GenNullableEmbListChunkDataSet(kEmbDocs, valid_ids, kDenseDim, kEmbVectorsPerDoc,
                                                                chunk_storage, chunk_ptrs, chunk_lims, emb_offsets);

            for (auto filter_ratio : filter_ratios) {
                CAPTURE(NullableName(nullable_ratio), FilterName(filter_ratio), "emb-list");
                auto filtered_ids = FilterIdsFor(filter_ratio, kEmbDocs, valid_ids, Mode::EmbList, valid_ids);
                auto allowed_ids = AllowedIdsAfterFilter(valid_ids, filtered_ids);
                const bool expect_empty = allowed_ids.empty();
                std::vector<uint8_t> bitset_data;
                auto bitset = BitsetViewFrom(bitset_data, kEmbDocs, filtered_ids);

                auto search = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, json, bitset);
                RequireExpectedVectorResult(search, allowed_ids, filtered_ids, expect_empty);
                if (search.has_value() && !expect_empty) {
                    RequireExactFirstHits(*search.value(), query_ids, allowed_ids);
                }

                std::vector<int64_t> ids(query_ids.size() * kEmbTopK, -1);
                std::vector<float> dists(query_ids.size() * kEmbTopK, 0.0f);
                auto status = knowhere::BruteForce::SearchWithBuf<knowhere::fp32>(
                    base_ds, query_ds, ids.data(), dists.data(), json, bitset);
                RequireBufferedStatus(status, ids, allowed_ids, filtered_ids, expect_empty);
                if (status == knowhere::Status::success && !expect_empty) {
                    RequireBufferedExactFirstHits(ids, kEmbTopK, query_ids, allowed_ids);
                }

                auto chunk_search =
                    knowhere::BruteForce::Search<knowhere::fp32>(chunk_base_ds, query_ds, json, bitset);
                RequireExpectedVectorResult(chunk_search, allowed_ids, filtered_ids, expect_empty);
                if (chunk_search.has_value() && !expect_empty) {
                    RequireExactFirstHits(*chunk_search.value(), query_ids, allowed_ids);
                }
            }
        }
    }
}

TEST_CASE("Nullable matrix has 53 index rows and 180 scenarios per row", "[nullable][matrix][count]") {
    const auto rows = IndexRows();
    const auto scenarios = BuildScenarios();
    REQUIRE(rows.size() == 53);
    REQUIRE(scenarios.size() == 180);
    REQUIRE(rows.size() * scenarios.size() == 9540);

    const std::vector<RequiredIndexRow> required_rows = {
        {"FLAT", knowhere::IndexEnum::INDEX_FAISS_IDMAP, DataKind::DenseFp32},
        {"BIN_FLAT / BINFLAT", knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, DataKind::DenseBin},
        {"FAISS (fp32)", knowhere::IndexEnum::INDEX_FAISS, DataKind::DenseFp32},
        {"FAISS (bin1)", knowhere::IndexEnum::INDEX_FAISS, DataKind::DenseBin},
        {"BIN_IVF_FLAT / IVFBIN", knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT, DataKind::DenseBin},
        {"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32},
        {"IVF_FLAT_CC / IVFFLATCC", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, DataKind::DenseFp32},
        {"IVF_PQ / IVFPQ", knowhere::IndexEnum::INDEX_FAISS_IVFPQ, DataKind::DenseFp32},
        {"IVF_SQ8", knowhere::IndexEnum::INDEX_FAISS_IVFSQ8, DataKind::DenseFp32},
        {"IVF_SQ / IVFSQ", "IVF_SQ", DataKind::DenseFp32},
        {"IVF_SQ_CC", knowhere::IndexEnum::INDEX_FAISS_IVFSQ_CC, DataKind::DenseFp32},
        {"IVF_RABITQ / IVFRABITQ", knowhere::IndexEnum::INDEX_FAISS_IVFRABITQ, DataKind::DenseFp32},
        {"IVF_RABITQ_FASTSCAN", knowhere::IndexEnum::INDEX_FAISS_IVFRABITQ_FASTSCAN, DataKind::DenseFp32},
        {"SCANN", knowhere::IndexEnum::INDEX_FAISS_SCANN, DataKind::DenseFp32},
        {"SCANN_DVR", knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, DataKind::DenseFp32},
        {"HNSW (Knowhere/Faiss)", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32},
        {"HNSW_SQ", knowhere::IndexEnum::INDEX_HNSW_SQ, DataKind::DenseFp32},
        {"HNSW_PQ", knowhere::IndexEnum::INDEX_HNSW_PQ, DataKind::DenseFp32},
        {"HNSW_PRQ", knowhere::IndexEnum::INDEX_HNSW_PRQ, DataKind::DenseFp32},
        {"HNSW_DEPRECATED", "HNSW_DEPRECATED", DataKind::DenseFp32},
        {"HNSWLIB_DEPRECATED", "HNSWLIB_DEPRECATED", DataKind::DenseFp32},
        {"HNSW_V1 sparse (Cardinal v1)", "HNSW_V1", DataKind::Sparse},
        {"DISKANN (Knowhere native)", knowhere::IndexEnum::INDEX_DISKANN, DataKind::DiskAnn},
        {"AISAQ", knowhere::IndexEnum::INDEX_AISAQ, DataKind::Aisaq},
        {"DISKANN_DEPRECATED", "DISKANN_DEPRECATED", DataKind::DiskAnn},
        {"SPARSE_INVERTED_INDEX (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
         DataKind::Sparse},
        {"SPARSE_WAND (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_WAND, DataKind::Sparse},
        {"SPARSE_INVERTED_INDEX_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
         DataKind::Sparse},
        {"SPARSE_WAND_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, DataKind::Sparse},
        {"SPARSE_INVERTED_INDEX (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX,
         DataKind::Sparse},
        {"SPARSE_WAND (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_WAND, DataKind::Sparse},
        {"SPARSE_INVERTED_INDEX_CC (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
         DataKind::Sparse},
        {"SPARSE_WAND_CC (Cardinal v1)", knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, DataKind::Sparse},
        {"MINHASH_LSH", knowhere::IndexEnum::INDEX_MINHASH_LSH, DataKind::MinHash},
        {"GPU_CUVS_BRUTE_FORCE", knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE, DataKind::GpuFp32},
        {"GPU_BRUTE_FORCE", knowhere::IndexEnum::INDEX_GPU_BRUTEFORCE, DataKind::GpuFp32},
        {"GPU_CUVS_IVF_FLAT", knowhere::IndexEnum::INDEX_CUVS_IVFFLAT, DataKind::GpuFp32},
        {"GPU_IVF_FLAT", knowhere::IndexEnum::INDEX_GPU_IVFFLAT, DataKind::GpuFp32},
        {"GPU_CUVS_IVF_PQ", knowhere::IndexEnum::INDEX_CUVS_IVFPQ, DataKind::GpuFp32},
        {"GPU_IVF_PQ", knowhere::IndexEnum::INDEX_GPU_IVFPQ, DataKind::GpuFp32},
        {"GPU_CUVS_CAGRA", knowhere::IndexEnum::INDEX_CUVS_CAGRA, DataKind::GpuFp32},
        {"GPU_CAGRA", knowhere::IndexEnum::INDEX_GPU_CAGRA, DataKind::GpuFp32},
        {"GPU_FAISS_FLAT", knowhere::IndexEnum::INDEX_FAISS_GPU_IDMAP, DataKind::GpuFp32},
        {"GPU_FAISS_IVF_FLAT", knowhere::IndexEnum::INDEX_FAISS_GPU_IVFFLAT, DataKind::GpuFp32},
        {"GPU_FAISS_IVF_PQ", knowhere::IndexEnum::INDEX_FAISS_GPU_IVFPQ, DataKind::GpuFp32},
        {"GPU_FAISS_IVF_SQ8", knowhere::IndexEnum::INDEX_FAISS_GPU_IVFSQ8, DataKind::GpuFp32},
        {"SVS_FLAT", knowhere::IndexEnum::INDEX_SVS_FLAT, DataKind::DenseFp32},
        {"SVS_VAMANA", knowhere::IndexEnum::INDEX_SVS_VAMANA, DataKind::DenseFp32},
        {"SVS_VAMANA_LVQ", knowhere::IndexEnum::INDEX_SVS_VAMANA_LVQ, DataKind::DenseFp32},
        {"SVS_VAMANA_LEANVEC", knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC, DataKind::DenseFp32},
        {"HNSW (Cardinal v2)", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32},
        {"DISKANN (Cardinal v2)", knowhere::IndexEnum::INDEX_DISKANN, DataKind::DiskAnn},
        {"CARDINAL_TIERED", knowhere::IndexEnum::INDEX_CARDINAL_TIERED, DataKind::DenseFp32},
    };
    REQUIRE(required_rows.size() == rows.size());
    for (const auto& required : required_rows) {
        CAPTURE(required.label, required.index_type);
        REQUIRE(HasIndexRow(rows, required));
    }
}

TEST_CASE("Nullable matrix covers every IndexNode operation", "[nullable][matrix]") {
#ifdef KNOWHERE_WITH_CARDINAL
    ConfigureTieredStorageForNullableMatrix();
#endif
    const auto rows = IndexRows();
    const auto scenarios = BuildScenarios();

    for (const auto& row : rows) {
        SECTION(row.label) {
            for (const auto& scenario : scenarios) {
                SECTION(scenario.name) {
                    ExecuteScenario(row, scenario);
                }
            }
        }
    }
}
