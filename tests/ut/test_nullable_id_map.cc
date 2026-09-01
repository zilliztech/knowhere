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
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "filemanager/impl/LocalFileManager.h"
#include "index/data_view_dense_index/data_view_dense_index.h"
#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/id_map.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"
#include "knowhere/object.h"
#include "knowhere/thread_pool.h"
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
    NullableWrapperFakeIndexNode() = default;

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
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented, "Search not implemented");
    }

    knowhere::expected<knowhere::DataSetPtr>
    RangeSearch(const knowhere::DataSetPtr dataset, std::unique_ptr<knowhere::Config> cfg,
                const knowhere::BitsetView& bitset, milvus::OpContext* op_context) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented, "RangeSearch not implemented");
    }

    knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
    AnnIterator(const knowhere::DataSetPtr dataset, std::unique_ptr<knowhere::Config> cfg,
                const knowhere::BitsetView& bitset, bool use_knowhere_search_pool,
                milvus::OpContext* op_context) const override {
        return knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>::Err(Status::not_implemented,
                                                                                      "AnnIterator not implemented");
    }

    knowhere::expected<knowhere::DataSetPtr>
    GetVectorByIds(const knowhere::DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return true;
    }

    bool
    NeedBitsetExactCount() const override {
        return true;
    }

    knowhere::expected<knowhere::DataSetPtr>
    GetIndexMeta(std::unique_ptr<knowhere::Config> cfg) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
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

class FailingAddFakeIndexNode : public NullableWrapperFakeIndexNode {
 public:
    Status
    Add(const knowhere::DataSetPtr dataset, std::shared_ptr<knowhere::Config> cfg,
        bool use_knowhere_build_pool) override {
        return Status::invalid_args;
    }

    std::string
    Type() const override {
        return "FAILING_ADD_FAKE";
    }
};

struct CapturedBitsetState {
    bool empty = true;
    bool need_filter = false;
    bool has_out_ids = false;
    size_t size = 0;
    size_t num_bits = 0;
    size_t count = 0;
    std::vector<int> filtered_in_ids;
};

class EmptyIterator : public knowhere::IndexNode::iterator {
 public:
    knowhere::expected<std::pair<int64_t, float>>
    Next() override {
        return {-1, 0.0f};
    }

    [[nodiscard]] knowhere::expected<bool>
    HasNext() override {
        return false;
    }
};

class CapturingBitsetFakeIndexNode : public knowhere::IndexNode {
 public:
    explicit CapturingBitsetFakeIndexNode(int64_t count, int64_t dim) : count_(count), dim_(dim) {
    }

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
        last_search_bitset = Capture(bitset);
        auto ids = std::make_unique<int64_t[]>(1);
        auto distances = std::make_unique<float[]>(1);
        ids[0] = -1;
        distances[0] = 0.0f;
        return knowhere::GenResultDataSet(1, 1, std::move(ids), std::move(distances));
    }

    knowhere::expected<knowhere::DataSetPtr>
    RangeSearch(const knowhere::DataSetPtr dataset, std::unique_ptr<knowhere::Config> cfg,
                const knowhere::BitsetView& bitset, milvus::OpContext* op_context) const override {
        last_range_bitset = Capture(bitset);
        knowhere::RangeSearchResult range;
        range.lims = std::make_unique<size_t[]>(2);
        range.lims[0] = 0;
        range.lims[1] = 0;
        return knowhere::GenResultDataSet(1, std::move(range));
    }

    knowhere::expected<std::vector<knowhere::IndexNode::IteratorPtr>>
    AnnIterator(const knowhere::DataSetPtr dataset, std::unique_ptr<knowhere::Config> cfg,
                const knowhere::BitsetView& bitset, bool use_knowhere_search_pool,
                milvus::OpContext* op_context) const override {
        last_iterator_bitset = Capture(bitset);
        return std::vector<knowhere::IndexNode::IteratorPtr>{std::make_shared<EmptyIterator>()};
    }

    knowhere::expected<knowhere::DataSetPtr>
    GetVectorByIds(const knowhere::DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented, "not implemented");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return true;
    }

    bool
    NeedBitsetExactCount() const override {
        return true;
    }

    knowhere::expected<knowhere::DataSetPtr>
    GetIndexMeta(std::unique_ptr<knowhere::Config> cfg) const override {
        return knowhere::expected<knowhere::DataSetPtr>::Err(Status::not_implemented, "not implemented");
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
        return dim_;
    }

    int64_t
    Size() const override {
        return count_ * dim_ * static_cast<int64_t>(sizeof(float));
    }

    int64_t
    Count() const override {
        return count_;
    }

    std::string
    Type() const override {
        return "CAPTURING_BITSET_FAKE";
    }

    void
    SetEmbListOffsetForTest(std::vector<size_t> offsets) {
        emb_list_offset_ = std::make_shared<knowhere::EmbListOffset>(std::move(offsets));
    }

    void
    BuildInToOutEblIdsForTest(const std::vector<size_t>& offsets) {
        id_map_.FinalizeEmbListIds(offsets.data(), offsets.size() - 1);
    }

    mutable CapturedBitsetState last_search_bitset;
    mutable CapturedBitsetState last_range_bitset;
    mutable CapturedBitsetState last_iterator_bitset;

 private:
    static CapturedBitsetState
    Capture(const knowhere::BitsetView& bitset) {
        CapturedBitsetState state;
        state.empty = bitset.empty();
        state.need_filter = !bitset.empty();
        state.has_out_ids = bitset.has_out_ids();
        state.size = bitset.size();
        state.num_bits = bitset.num_bits();
        state.count = bitset.count();
        const auto test_size = state.has_out_ids ? state.size : std::min(state.size, state.num_bits);
        for (size_t i = 0; i < test_size; ++i) {
            if (bitset.test(static_cast<int64_t>(i))) {
                state.filtered_in_ids.push_back(static_cast<int>(i));
            }
        }
        return state;
    }

    int64_t count_;
    int64_t dim_;
};

constexpr int64_t kTotalRows = 32;
constexpr int64_t kDenseDim = 16;
constexpr int64_t kGpuRows = 64;
constexpr int64_t kGpuDim = 32;
constexpr int64_t kFileRows = 64;
constexpr int64_t kFileDim = 32;
constexpr int64_t kEmbDocs = 32;
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
    bool requires_current_version = false;
    bool requires_svs = false;
    bool requires_gpu = false;
    bool exact = false;
};

IndexRow
NativeFaissHnswRow() {
#ifdef KNOWHERE_WITH_CARDINAL
    return {"HNSW_DEPRECATED", "HNSW_DEPRECATED", DataKind::DenseFp32};
#else
    return {"HNSW Knowhere Faiss", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32};
#endif
}

IndexRow
NativeDiskAnnRow() {
#ifdef KNOWHERE_WITH_CARDINAL
    return {"DISKANN_DEPRECATED", "DISKANN_DEPRECATED", DataKind::DiskAnn};
#else
    return {"DISKANN (Knowhere native)", knowhere::IndexEnum::INDEX_DISKANN, DataKind::DiskAnn};
#endif
}

struct Scenario {
    std::string name;
    NullableRatio nullable_ratio = NullableRatio::R0;
    FilterRatio filter_ratio = FilterRatio::None;
    Mode mode = Mode::Vector;
    Operation op = Operation::Build;
    IndexSource source = IndexSource::Fresh;
    std::string emb_list_strategy;
    bool force_raw_data = false;
    int64_t total_count_override = 0;
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

void
RemoveAllNoThrow(const fs::path& path) {
    if (path.empty()) {
        return;
    }
    std::error_code ec;
    fs::remove_all(path, ec);
}

class ScopedWorkDir {
 public:
    explicit ScopedWorkDir(fs::path path) : path_(std::move(path)) {
        RemoveAllNoThrow(path_);
        fs::create_directories(path_);
    }

    ~ScopedWorkDir() {
        RemoveAllNoThrow(path_);
    }

 private:
    fs::path path_;
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
    fs::path emb_meta_file;
    fs::path emb_raw_file;
    fs::path emb_offset_file;
    fs::path raw_data_file;
    fs::path file_index_prefix;

    ~BuiltArtifact() {
        RemoveAllNoThrow(work_dir);
    }
};

knowhere::Status
BuildRuntimeIdMap(const std::vector<int32_t>& valid_ids, int64_t total_count, knowhere::IdMap& id_map);

knowhere::Status
AttachRuntimeIdMapData(const knowhere::DataSetPtr& dataset, const std::vector<int32_t>& valid_ids, int64_t total_count);

knowhere::Status
BuildRuntimeValidBitmap(const std::vector<int32_t>& valid_ids, int64_t total_count, knowhere::IdMap& id_map);

int32_t
VersionForRow(const IndexRow& row) {
#ifdef KNOWHERE_WITH_CARDINAL
    if (!row.requires_current_version &&
        (row.index_type == knowhere::IndexEnum::INDEX_HNSW || row.index_type == knowhere::IndexEnum::INDEX_DISKANN)) {
        return knowhere::Version::GetDefaultVersion().VersionNumber();
    }
    if (row.requires_current_version) {
        return knowhere::Version::GetMaximumVersion().VersionNumber();
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
EvenIds(int64_t count) {
    std::vector<int32_t> ids;
    ids.reserve((count + 1) / 2);
    for (int32_t id = 0; id < count; id += 2) {
        ids.push_back(id);
    }
    return ids;
}

std::vector<int32_t>
PairedIds(int64_t count) {
    std::vector<int32_t> ids;
    ids.reserve((count + 3) / 4 * 2);
    for (int32_t id = 0; id < count; id += 8) {
        ids.push_back(id);
        if (id + 1 < count) {
            ids.push_back(id + 1);
        }
        if (id + 4 < count) {
            ids.push_back(id + 4);
        }
        if (id + 5 < count) {
            ids.push_back(id + 5);
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
            return paired_for_multi ? PairedIds(count) : EvenIds(count);
    }
    return {};
}

bool
ContainsId(const std::vector<int32_t>& ids, int64_t id) {
    return std::find(ids.begin(), ids.end(), static_cast<int32_t>(id)) != ids.end();
}

int64_t
CompactIdForLogicalId(const std::vector<int32_t>& valid_ids, int64_t logical_id) {
    auto it = std::find(valid_ids.begin(), valid_ids.end(), static_cast<int32_t>(logical_id));
    REQUIRE(it != valid_ids.end());
    return static_cast<int64_t>(std::distance(valid_ids.begin(), it));
}

int64_t
FirstMissingOutId(const std::vector<int32_t>& valid_ids, int64_t total_count) {
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
    return GenSparseDataSet(rows, static_cast<int32_t>(total_count + 2));
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
    ds->Set(knowhere::meta::EMB_LIST_COUNT, static_cast<int64_t>(valid_doc_ids.size()));
    ds->Set(knowhere::meta::NQ, static_cast<int64_t>(valid_doc_ids.size()));
    return ds;
}

knowhere::DataSetPtr
GenEmbListBatchDataSet(const std::vector<int32_t>& public_doc_ids, const std::vector<size_t>& absolute_offsets,
                       int64_t dim) {
    REQUIRE(absolute_offsets.size() == public_doc_ids.size() + 1);
    REQUIRE(!absolute_offsets.empty());
    REQUIRE(absolute_offsets.back() >= absolute_offsets.front());
    const auto vector_begin = absolute_offsets.front();
    const auto rows = static_cast<int64_t>(absolute_offsets.back() - vector_begin);
    auto data = std::make_unique<float[]>(std::max<int64_t>(rows * dim, 1));
    for (size_t doc = 0; doc < public_doc_ids.size(); ++doc) {
        REQUIRE(absolute_offsets[doc + 1] >= absolute_offsets[doc]);
        for (size_t vector_id = absolute_offsets[doc]; vector_id < absolute_offsets[doc + 1]; ++vector_id) {
            FillDenseVector(data.get(), static_cast<int64_t>(vector_id - vector_begin), dim, public_doc_ids[doc]);
        }
    }
    auto ds = knowhere::GenDataSet(rows, dim, rows == 0 ? nullptr : data.release());
    ds->SetIsOwner(rows != 0);
    auto offset_data = std::make_unique<size_t[]>(absolute_offsets.size());
    std::copy(absolute_offsets.begin(), absolute_offsets.end(), offset_data.get());
    ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(offset_data.release()));
    ds->Set(knowhere::meta::EMB_LIST_COUNT, static_cast<int64_t>(public_doc_ids.size()));
    ds->Set(knowhere::meta::NQ, static_cast<int64_t>(public_doc_ids.size()));
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
    ds->Set(knowhere::meta::EMB_LIST_COUNT, static_cast<int64_t>(logical_doc_ids.size()));
    ds->Set(knowhere::meta::NQ, static_cast<int64_t>(logical_doc_ids.size()));
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
    knowhere::BitsetView bitset(bitmap.data(), total_count);
    bitset.count_filtered_bits(0, total_count);
    return bitset;
}

knowhere::IndexNode::PreparedBitset
NormalizedBitsetFrom(std::vector<uint8_t>& bitmap, const knowhere::IdMap& id_map, int64_t total_count,
                     const std::vector<int32_t>& filtered_ids) {
    knowhere::IndexNode::PreparedBitset prepared(BitsetViewFrom(bitmap, total_count, filtered_ids));
    const auto& out_ids = id_map.InToOutIds();
    if (!out_ids.empty()) {
        prepared.bitset.set_id_offset(0);
        prepared.bitset.set_out_ids(out_ids, out_ids.size());
        const auto& valid_bitmap = id_map.ValidBitmap();
        prepared.bitset.count_filtered_bits(0, prepared.bitset.num_bits(), valid_bitmap.data());
    }
    return prepared;
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

std::string
EmbListStrategyFor(const Scenario& scenario) {
    if (scenario.emb_list_strategy.empty()) {
        return knowhere::meta::EMB_LIST_STRATEGY_TOKENANN;
    }
    return scenario.emb_list_strategy;
}

bool
RequiresEmbListMetaV2(const Scenario& scenario) {
    if (scenario.mode != Mode::EmbList) {
        return false;
    }
    const auto strategy = EmbListStrategyFor(scenario);
    return strategy == knowhere::meta::EMB_LIST_STRATEGY_MUVERA || strategy == knowhere::meta::EMB_LIST_STRATEGY_LEMUR;
}

int32_t
VersionForScenario(const IndexRow& row, const Scenario& scenario) {
    if (RequiresEmbListMetaV2(scenario)) {
        return knowhere::kEmbListMetaV2MinVersion;
    }
    return VersionForRow(row);
}

std::vector<std::string>
EmbListStrategies() {
    return {knowhere::meta::EMB_LIST_STRATEGY_TOKENANN, knowhere::meta::EMB_LIST_STRATEGY_MUVERA,
            knowhere::meta::EMB_LIST_STRATEGY_LEMUR};
}

void
ApplyEmbListStrategyConfig(knowhere::Json& json, const std::string& strategy) {
    json["emb_list_strategy"] = strategy;
    if (strategy == knowhere::meta::EMB_LIST_STRATEGY_MUVERA) {
        json["muvera_num_projections"] = 2;
        json["muvera_num_repeats"] = 2;
        json["muvera_seed"] = 42;
    } else if (strategy == knowhere::meta::EMB_LIST_STRATEGY_LEMUR) {
        json["lemur_hidden_dim"] = 8;
        json["lemur_num_train_samples"] = 1000;
        json["lemur_num_epochs"] = 1;
        json["lemur_batch_size"] = 8;
        json["lemur_learning_rate"] = 0.001f;
        json["lemur_seed"] = 42;
        json["lemur_num_layers"] = 1;
    }
}

knowhere::Json
DenseConfigFor(const IndexRow& row, Mode mode,
               const std::string& emb_list_strategy = knowhere::meta::EMB_LIST_STRATEGY_TOKENANN) {
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
        ApplyEmbListStrategyConfig(json, emb_list_strategy);
    }
    if (mode == Mode::EmbList && row.index_type == knowhere::IndexEnum::INDEX_HNSW_PRQ &&
        emb_list_strategy == knowhere::meta::EMB_LIST_STRATEGY_MUVERA) {
        json[knowhere::indexparam::M] = 16;
        json[knowhere::indexparam::PRQ_NUM] = 1;
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

void
UseRawDataMode(knowhere::Json& json) {
    json[knowhere::meta::RETRIEVE_FRIENDLY] = true;
    json["build_quant_type"] = "NONE";
    json["search_quant_type"] = "NONE";
    json["refine_quant_type"] = "NONE";
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
VectorCaps(bool range, bool iterator, bool search_filter, bool range_filter, bool iterator_filter, bool binaryset_io,
           bool file_io) {
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
    auto add = [&](IndexRow row) { rows.push_back(std::move(row)); };

    auto flat = VectorCaps(true, false, true, true, false, true, true);
    add({"FLAT", knowhere::IndexEnum::INDEX_FAISS_IDMAP, DataKind::DenseFp32, flat, false, false, false, false, true});
    add({"BIN_FLAT / BINFLAT", knowhere::IndexEnum::INDEX_FAISS_BIN_IDMAP, DataKind::DenseBin, flat, false, false,
         false, false, false});
    add({"FAISS (fp32)", knowhere::IndexEnum::INDEX_FAISS, DataKind::DenseFp32, flat, false, false, false, false,
         true});
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
    add({"IVF_SQ / IVFSQ", "IVF_SQ", DataKind::DenseFp32, VectorCaps(true, true, true, true, true, true, true)});
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
    add({"HNSWLIB_DEPRECATED", "HNSWLIB_DEPRECATED", DataKind::DenseFp32,
         VectorCaps(true, true, true, true, true, true, true)});

    auto diskann = VectorCaps(true, true, true, true, true, false, true);
    add({"DISKANN (Knowhere native)", knowhere::IndexEnum::INDEX_DISKANN, DataKind::DiskAnn, diskann, true});
    add({"AISAQ", knowhere::IndexEnum::INDEX_AISAQ, DataKind::Aisaq,
         VectorCaps(false, false, true, false, false, false, true), true});

    add({"SPARSE_INVERTED_INDEX (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, DataKind::Sparse,
         VectorCaps(true, true, true, true, true, true, true), false, false, false, false, true});
    add({"SPARSE_WAND (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_WAND, DataKind::Sparse,
         VectorCaps(true, true, true, true, true, true, true), false, false, false, false, true});
    add({"SPARSE_INVERTED_INDEX_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
         DataKind::Sparse, VectorCaps(true, true, true, true, true, false, false), false, false, false, false, true});
    add({"SPARSE_WAND_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, DataKind::Sparse,
         VectorCaps(true, true, true, true, true, false, false), false, false, false, false, true});
    add({"MINHASH_LSH", knowhere::IndexEnum::INDEX_MINHASH_LSH, DataKind::MinHash,
         VectorCaps(false, false, true, false, false, false, false)});

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

    return rows;
}

void
AddScenario(std::vector<Scenario>& scenarios, NullableRatio nullable_ratio, Mode mode, Operation op, IndexSource source,
            FilterRatio filter_ratio = FilterRatio::None, const std::string& emb_list_strategy = "") {
    Scenario scenario;
    scenario.nullable_ratio = nullable_ratio;
    scenario.mode = mode;
    scenario.op = op;
    scenario.source = source;
    scenario.filter_ratio = filter_ratio;
    scenario.emb_list_strategy = mode == Mode::EmbList ? emb_list_strategy : "";
    scenario.name = NullableName(nullable_ratio) + "/" + ModeName(mode) + "/" + OperationName(op) + "/" +
                    SourceName(source) + "/" + FilterName(filter_ratio);
    if (mode == Mode::EmbList) {
        scenario.name += "/" + EmbListStrategyFor(scenario);
    }
    scenarios.push_back(std::move(scenario));
}

void
AddNonFilterScenarios(std::vector<Scenario>& scenarios, NullableRatio ratio, Mode mode,
                      const std::string& emb_list_strategy = "") {
    AddScenario(scenarios, ratio, mode, Operation::Build, IndexSource::Fresh, FilterRatio::None, emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::Search, IndexSource::Fresh, FilterRatio::None, emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::Range, IndexSource::Fresh, FilterRatio::None, emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::Iterator, IndexSource::Fresh, FilterRatio::None, emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::BinarySetSerialize, IndexSource::Fresh, FilterRatio::None,
                emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::BinarySetDeserialize, IndexSource::BinarySet, FilterRatio::None,
                emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::Search, IndexSource::BinarySet, FilterRatio::None,
                emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::Range, IndexSource::BinarySet, FilterRatio::None, emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::Iterator, IndexSource::BinarySet, FilterRatio::None,
                emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::FileSerialize, IndexSource::Fresh, FilterRatio::None,
                emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::FileDeserialize, IndexSource::File, FilterRatio::None,
                emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::Search, IndexSource::File, FilterRatio::None, emb_list_strategy);
    AddScenario(scenarios, ratio, mode, Operation::Iterator, IndexSource::File, FilterRatio::None, emb_list_strategy);
}

std::vector<Scenario>
BuildScenarios() {
    std::vector<Scenario> scenarios;
    for (auto ratio : {NullableRatio::R0, NullableRatio::R50}) {
        AddNonFilterScenarios(scenarios, ratio, Mode::Vector);
        for (const auto& strategy : EmbListStrategies()) {
            AddNonFilterScenarios(scenarios, ratio, Mode::EmbList, strategy);
        }
        AddNonFilterScenarios(scenarios, ratio, Mode::MultiIndex);
        const std::vector<FilterRatio> filter_ratios = {FilterRatio::R0, FilterRatio::R50, FilterRatio::R100};
        for (auto filter_ratio : filter_ratios) {
            AddScenario(scenarios, ratio, Mode::Vector, Operation::SearchFilter, IndexSource::Fresh, filter_ratio);
            AddScenario(scenarios, ratio, Mode::Vector, Operation::RangeFilter, IndexSource::Fresh, filter_ratio);
            AddScenario(scenarios, ratio, Mode::Vector, Operation::IteratorFilter, IndexSource::Fresh, filter_ratio);
            for (const auto& strategy : EmbListStrategies()) {
                AddScenario(scenarios, ratio, Mode::EmbList, Operation::SearchFilter, IndexSource::Fresh, filter_ratio,
                            strategy);
                AddScenario(scenarios, ratio, Mode::EmbList, Operation::RangeFilter, IndexSource::Fresh, filter_ratio,
                            strategy);
                AddScenario(scenarios, ratio, Mode::EmbList, Operation::IteratorFilter, IndexSource::Fresh,
                            filter_ratio, strategy);
            }
            AddScenario(scenarios, ratio, Mode::MultiIndex, Operation::SearchFilter, IndexSource::Fresh, filter_ratio);
            AddScenario(scenarios, ratio, Mode::MultiIndex, Operation::RangeFilter, IndexSource::Fresh, filter_ratio);
            AddScenario(scenarios, ratio, Mode::MultiIndex, Operation::IteratorFilter, IndexSource::Fresh,
                        filter_ratio);
        }
    }
    return scenarios;
}

bool
SupportsEmbListStrategy(const IndexRow& row, const Scenario& scenario) {
    if (scenario.mode != Mode::EmbList) {
        return true;
    }
    const auto strategy = EmbListStrategyFor(scenario);
    if (strategy == knowhere::meta::EMB_LIST_STRATEGY_TOKENANN) {
        return row.index_type != knowhere::IndexEnum::INDEX_FAISS_IVFFLAT;
    }
    if (row.requires_current_version) {
        return false;
    }
    if (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq || row.data_kind == DataKind::DenseBin ||
        row.data_kind == DataKind::Sparse || row.data_kind == DataKind::MinHash || row.data_kind == DataKind::GpuFp32) {
        return false;
    }
    return strategy == knowhere::meta::EMB_LIST_STRATEGY_MUVERA || strategy == knowhere::meta::EMB_LIST_STRATEGY_LEMUR;
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
    if (row.index_type == knowhere::IndexEnum::INDEX_HNSW && !row.requires_current_version && VersionForRow(row) < 6 &&
        scenario.nullable_ratio != NullableRatio::R0) {
        return false;
    }
    if (!SupportsEmbListStrategy(row, scenario)) {
        return false;
    }
#ifdef KNOWHERE_WITH_CARDINAL
    const bool current_version_reads_index =
        scenario.op == Operation::Search || scenario.op == Operation::Range || scenario.op == Operation::Iterator ||
        scenario.op == Operation::SearchFilter || scenario.op == Operation::RangeFilter ||
        scenario.op == Operation::IteratorFilter;
    if (row.requires_current_version && current_version_reads_index && scenario.source == IndexSource::Fresh) {
        return false;
    }
#endif
    if (row.requires_current_version && row.index_type == knowhere::IndexEnum::INDEX_DISKANN &&
        scenario.mode != Mode::Vector &&
        (scenario.op == Operation::FileSerialize || scenario.op == Operation::FileDeserialize ||
         scenario.source == IndexSource::File)) {
        return false;
    }
    const bool reads_index =
        scenario.op == Operation::Search || scenario.op == Operation::Range || scenario.op == Operation::Iterator;
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
    const auto length = k == 0 ? lims[nq] : static_cast<size_t>(nq * k);
    for (size_t i = 0; i < length; ++i) {
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
    const auto length = k == 0 ? lims[nq] : static_cast<size_t>(nq * k);
    bool has_result = false;
    for (size_t i = 0; i < length; ++i) {
        if (ids[i] < 0) {
            continue;
        }
        CAPTURE(i, ids[i], allowed_ids, filtered_ids);
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
            auto has_next = iterator->HasNext();
            REQUIRE(has_next.has_value());
            REQUIRE(!has_next.value());
        }
        return;
    }

    bool checked = false;
    for (size_t i = 0; i < iterators.size(); ++i) {
        const auto& iterator = iterators[i];
        auto has_next = iterator->HasNext();
        REQUIRE(has_next.has_value());
        if (!has_next.value()) {
            continue;
        }
        auto next = iterator->Next();
        REQUIRE(next.has_value());
        auto [id, dist] = next.value();
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
RequireBufferedStatus(knowhere::Status status, const std::vector<int64_t>& ids, const std::vector<int32_t>& allowed_ids,
                      const std::vector<int32_t>& filtered_ids, bool expect_empty) {
    if (status != knowhere::Status::success) {
        REQUIRE(expect_empty);
        REQUIRE(IsExpectedEmptySearchError(status));
        return;
    }
    RequireBufferedIdsIn(ids, allowed_ids, filtered_ids, expect_empty);
}

void
RequireIdArrayEquals(const knowhere::IdArray& view, const std::vector<int32_t>& expected) {
    REQUIRE(view.size() == expected.size());
    if (!expected.empty()) {
        REQUIRE_FALSE(view.empty());
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE(view[i] == expected[i]);
    }
}

int64_t
RequireCompactOutToIn(const knowhere::IdMap& map, int64_t out_id) {
    auto compact_id = map.MapOutToIn(out_id);
    REQUIRE(compact_id >= 0);
    return compact_id;
}

int64_t
RequireCompactOutToIn(const knowhere::IndexNode& index_node, int64_t out_id) {
    auto compact_id = index_node.MapOutToIn(out_id);
    REQUIRE(compact_id >= 0);
    return compact_id;
}

void
RequireInvalidCompactOutToIn(const knowhere::IdMap& map, int64_t out_id) {
    REQUIRE(map.MapOutToIn(out_id) < 0);
}

void
RequireInvalidCompactOutToIn(const knowhere::IndexNode& index_node, int64_t out_id) {
    REQUIRE(index_node.MapOutToIn(out_id) < 0);
}

knowhere::IdArray
RequireIdMapArrayContent(const knowhere::Index<knowhere::IndexNode>& index, const std::vector<int32_t>& valid_ids,
                         int64_t count) {
    REQUIRE(index.Node() != nullptr);
    const auto& map = index.Node()->GetIdMap();
    REQUIRE(map.OutCount() == static_cast<size_t>(count));
    auto valid = map.ValidBitmap();
    REQUIRE(valid.size() == static_cast<size_t>(count));
    REQUIRE(valid.data() != nullptr);
    auto in_to_out_ids = map.InToOutIds();
    RequireIdArrayEquals(in_to_out_ids, valid_ids);
    return in_to_out_ids;
}

void
RequireIdMapContent(const knowhere::Index<knowhere::IndexNode>& index, const std::vector<int32_t>& valid_ids,
                    int64_t count) {
    auto in_to_out_ids = RequireIdMapArrayContent(index, valid_ids, count);
    const auto& map = index.Node()->GetIdMap();
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        CAPTURE(i, valid_ids[i]);
        REQUIRE(map.MapInToOut(static_cast<int64_t>(i)) == valid_ids[i]);
        REQUIRE(RequireCompactOutToIn(*index.Node(), valid_ids[i]) == static_cast<int64_t>(i));
    }
}

void
RequireIdMapBitmap(const knowhere::Index<knowhere::IndexNode>& index, int64_t count) {
    REQUIRE(index.Node() != nullptr);
    const auto& map = index.Node()->GetIdMap();
    REQUIRE(map.OutCount() == static_cast<size_t>(count));
    auto valid = map.ValidBitmap();
    REQUIRE(valid.size() == static_cast<size_t>(count));
    REQUIRE(valid.data() != nullptr);
}

void
RequireVectorIdMapCompacted(const knowhere::Index<knowhere::IndexNode>& index, const std::vector<int32_t>& valid_ids,
                            int64_t count) {
    REQUIRE(index.Node() != nullptr);
    const auto& map = index.Node()->GetIdMap();
    REQUIRE(map.OutCount() == static_cast<size_t>(count));
    REQUIRE(map.InCount() == valid_ids.size());
    auto valid = map.ValidBitmap();
    REQUIRE(valid.size() == static_cast<size_t>(count));
    REQUIRE(valid.data() != nullptr);
    REQUIRE(map.InToOutIds().empty());
    REQUIRE(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST).empty());
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        CAPTURE(i, valid_ids[i]);
        RequireInvalidCompactOutToIn(map, valid_ids[i]);
        REQUIRE(RequireCompactOutToIn(*index.Node(), valid_ids[i]) == static_cast<int64_t>(i));
    }
    const auto missing_id = FirstMissingOutId(valid_ids, count);
    if (missing_id < count) {
        RequireInvalidCompactOutToIn(map, missing_id);
        RequireInvalidCompactOutToIn(*index.Node(), missing_id);
    }
}

void
RequireEmbListIdMapCompacted(const knowhere::Index<knowhere::IndexNode>& index, const std::vector<int32_t>& valid_ids,
                             int64_t count) {
    REQUIRE(index.Node() != nullptr);
    const auto& map = index.Node()->GetIdMap();
    REQUIRE(map.OutCount() == static_cast<size_t>(count));
    REQUIRE(map.InCount() == valid_ids.size());
    auto valid = map.ValidBitmap();
    REQUIRE(valid.size() == static_cast<size_t>(count));
    REQUIRE(valid.data() != nullptr);
    RequireIdArrayEquals(map.InToOutIds(), valid_ids);
    REQUIRE(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST).empty());
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        CAPTURE(i, valid_ids[i]);
        REQUIRE(RequireCompactOutToIn(index.Node()->GetIdMap(), valid_ids[i]) == static_cast<int64_t>(i));
        REQUIRE(RequireCompactOutToIn(*index.Node(), valid_ids[i]) == static_cast<int64_t>(i));
    }
    const auto missing_id = FirstMissingOutId(valid_ids, count);
    if (missing_id < count) {
        RequireInvalidCompactOutToIn(map, missing_id);
        RequireInvalidCompactOutToIn(*index.Node(), missing_id);
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
    if (logical_ids.empty()) {
        return;
    }
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
RequireDenseCalcDistByPublicIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data) {
    std::vector<int64_t> labels(data.query_ids.begin(), data.query_ids.end());
    auto result = index.CalcDistByIDs(data.query_ds, knowhere::BitsetView{}, labels.data(), labels.size(), false);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetRows() == data.query_ds->GetRows());
    REQUIRE(result.value()->GetDim() == static_cast<int64_t>(labels.size()));
    const auto* distances = result.value()->GetDistance();
    REQUIRE(distances != nullptr);
    for (size_t i = 0; i < data.query_ids.size(); ++i) {
        for (size_t j = 0; j < labels.size(); ++j) {
            const auto label_logical_id = labels[j];
            const auto expected = DenseL2ForLogicalIds(data.query_ids[i], label_logical_id, data.dim);
            CAPTURE(i, j, data.query_ids[i], labels[j]);
            REQUIRE(distances[i * labels.size() + j] == Catch::Approx(expected));
        }
    }
}

void
RequireDenseCalcDistByStorageIds(const knowhere::IndexNode& node, const MatrixData& data) {
    std::vector<int64_t> labels;
    labels.reserve(data.query_ids.size());
    for (auto logical_id : data.query_ids) {
        labels.push_back(CompactIdForLogicalId(data.valid_ids, logical_id));
    }
    auto result = node.CalcDistByStorageIds(data.query_ds, knowhere::BitsetView{}, labels.data(), labels.size(), false);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetRows() == data.query_ds->GetRows());
    REQUIRE(result.value()->GetDim() == static_cast<int64_t>(labels.size()));
    const auto* distances = result.value()->GetDistance();
    REQUIRE(distances != nullptr);
    for (size_t i = 0; i < data.query_ids.size(); ++i) {
        for (size_t j = 0; j < labels.size(); ++j) {
            const auto label_logical_id = data.valid_ids[static_cast<size_t>(labels[j])];
            const auto expected = DenseL2ForLogicalIds(data.query_ids[i], label_logical_id, data.dim);
            CAPTURE(i, j, data.query_ids[i], labels[j]);
            REQUIRE(distances[i * labels.size() + j] == Catch::Approx(expected));
        }
    }
}

void
RequireDenseRetrieveCompactsMissingOutIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data) {
    const auto missing_id = FirstMissingOutId(data.valid_ids, data.total_count);
    const std::vector<int64_t> rejected_ids = {-1, data.total_count};

    for (auto rejected_id : rejected_ids) {
        CAPTURE(rejected_id);
        int64_t retrieve_id = rejected_id;
        auto retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(1, &retrieve_id));
        REQUIRE(retrieve.has_value());
        RequireDenseVectorsMatchLogicalIds(*retrieve.value(), {}, data.dim);
    }

    auto null_retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(1, static_cast<const int64_t*>(nullptr)));
    REQUIRE_FALSE(null_retrieve.has_value());
    REQUIRE(null_retrieve.error() == knowhere::Status::invalid_args);

    if (missing_id < data.total_count) {
        auto retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(1, &missing_id));
        REQUIRE(retrieve.has_value());
        RequireDenseVectorsMatchLogicalIds(*retrieve.value(), {}, data.dim);
    }

    if (missing_id < data.total_count && data.query_ids.size() >= 2) {
        std::vector<int32_t> expected_ids = {data.query_ids[0], data.query_ids[1]};
        std::vector<int64_t> ids = {data.query_ids[0], missing_id, data.query_ids[1]};
        auto retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()));
        REQUIRE(retrieve.has_value());
        RequireDenseVectorsMatchLogicalIds(*retrieve.value(), expected_ids, data.dim);
    }
}

void
RequireDenseVectorPublicApisUseOutIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data,
                                      const knowhere::Json& json, bool require_exact_first_hit) {
    auto search = index.Search(data.query_ds, json, knowhere::BitsetView{});
    REQUIRE(search.has_value());
    if (require_exact_first_hit) {
        RequireExactFirstHits(*search.value(), data.query_ids, data.valid_ids);
    } else {
        RequireExpectedVectorResult(search, data.valid_ids, {}, false);
    }

    std::vector<int64_t> ids(data.query_ids.begin(), data.query_ids.end());
    auto retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()));
    REQUIRE(retrieve.has_value());
    RequireDenseVectorsMatchLogicalIds(*retrieve.value(), data.query_ids, data.dim);

    RequireDenseCalcDistByPublicIds(index, data);
    RequireDenseRetrieveCompactsMissingOutIds(index, data);
}

void
RequireDenseVectorApisUseOutIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data,
                                const knowhere::Json& json) {
    RequireIdMapContent(index, data.valid_ids, data.total_count);
    RequireDenseVectorPublicApisUseOutIds(index, data, json, true);
}

void
RequireEmbListVectorsMatchLogicalIds(const knowhere::DataSet& vectors, const std::vector<int32_t>& logical_ids,
                                     int64_t dim, int64_t vectors_per_doc) {
    REQUIRE(vectors.GetRows() == static_cast<int64_t>(logical_ids.size()));
    REQUIRE(vectors.GetDim() == dim);
    const auto* offsets = vectors.Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
    REQUIRE(offsets != nullptr);
    if (logical_ids.empty()) {
        REQUIRE(offsets[0] == 0);
        return;
    }
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
RequireEmbListRetrieveCompactsMissingOutId(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data);

void
RequireEmbListApisUseOutIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data,
                            const knowhere::Json& json) {
    RequireIdMapContent(index, data.valid_ids, data.total_count);

    auto search = index.Search(data.query_ds, json, knowhere::BitsetView{});
    REQUIRE(search.has_value());
    RequireExactFirstHits(*search.value(), data.query_ids, data.valid_ids);

    std::vector<int64_t> ids(data.query_ids.begin(), data.query_ids.end());
    auto retrieve =
        index.GetEmbListByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()), "MAX_SIM_L2");
    REQUIRE(retrieve.has_value());
    RequireEmbListVectorsMatchLogicalIds(*retrieve.value(), data.query_ids, data.dim, kEmbVectorsPerDoc);
}

void
RequireEmbListRetrieveUsesOutIds(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data) {
    REQUIRE(data.query_ids.size() >= 2);
    std::vector<int32_t> logical_ids = {data.query_ids[1], data.query_ids[0], data.query_ids[1]};
    std::vector<int64_t> ids(logical_ids.begin(), logical_ids.end());
    auto retrieve =
        index.GetEmbListByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()), "MAX_SIM_L2");
    REQUIRE(retrieve.has_value());
    RequireEmbListVectorsMatchLogicalIds(*retrieve.value(), logical_ids, data.dim, kEmbVectorsPerDoc);
    RequireEmbListRetrieveCompactsMissingOutId(index, data);
}

void
RequireEmbListRetrieveCompactsMissingOutId(const knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data) {
    const auto missing_id = FirstMissingOutId(data.valid_ids, data.total_count);
    const std::vector<int64_t> rejected_ids = {-1, data.total_count};

    for (auto rejected_id : rejected_ids) {
        CAPTURE(rejected_id);
        int64_t id = rejected_id;
        auto retrieve = index.GetEmbListByIds(knowhere::GenIdsDataSet(1, &id), "MAX_SIM_L2");
        REQUIRE(retrieve.has_value());
        RequireEmbListVectorsMatchLogicalIds(*retrieve.value(), {}, data.dim, kEmbVectorsPerDoc);
    }

    auto null_retrieve =
        index.GetEmbListByIds(knowhere::GenIdsDataSet(1, static_cast<const int64_t*>(nullptr)), "MAX_SIM_L2");
    REQUIRE_FALSE(null_retrieve.has_value());
    REQUIRE(null_retrieve.error() == knowhere::Status::invalid_args);

    if (missing_id < data.total_count) {
        auto retrieve = index.GetEmbListByIds(knowhere::GenIdsDataSet(1, &missing_id), "MAX_SIM_L2");
        REQUIRE(retrieve.has_value());
        RequireEmbListVectorsMatchLogicalIds(*retrieve.value(), {}, data.dim, kEmbVectorsPerDoc);
    }

    if (missing_id < data.total_count && data.query_ids.size() >= 2) {
        std::vector<int32_t> logical_ids = {data.query_ids[0], data.query_ids[1]};
        std::vector<int64_t> ids = {data.query_ids[0], missing_id, data.query_ids[1]};
        auto retrieve =
            index.GetEmbListByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()), "MAX_SIM_L2");
        REQUIRE(retrieve.has_value());
        RequireEmbListVectorsMatchLogicalIds(*retrieve.value(), logical_ids, data.dim, kEmbVectorsPerDoc);
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
                             std::vector<std::vector<float>>& chunk_storage, std::vector<const float*>& chunk_ptrs,
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
    ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(chunk_lims.data()));
    return ds;
}

knowhere::DataSetPtr
GenNullableEmbListChunkDataSet(int64_t total_docs, const std::vector<int32_t>& valid_doc_ids, int64_t dim,
                               int64_t vectors_per_doc, std::vector<std::vector<float>>& chunk_storage,
                               std::vector<const float*>& chunk_ptrs, std::vector<size_t>& chunk_lims,
                               std::vector<size_t>& emb_offsets) {
    auto dense_ds = GenNullableEmbListDataSet(total_docs, valid_doc_ids, dim, vectors_per_doc);
    chunk_storage =
        BuildChunkStorage(dense_ds, std::max<int64_t>(static_cast<int64_t>(valid_doc_ids.size()), 1), chunk_lims);
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
    ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(chunk_lims.data()));
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
        data.total_count = scenario.total_count_override > 0 ? scenario.total_count_override : kEmbDocs;
        data.dim = kDenseDim;
        data.valid_ids = ValidIdsFor(scenario.nullable_ratio, data.total_count);
        data.selected_ids = data.valid_ids;
        data.query_ids = FirstQueryIds(data.valid_ids, data.total_count, 2);
        data.train_ds = GenNullableEmbListDataSet(data.total_count, data.valid_ids, data.dim, kEmbVectorsPerDoc);
        data.full_ds = row.index_type == knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR
                           ? data.train_ds
                           : GenFullDenseDataSet(data.total_count, data.dim);
        data.query_ds = GenEmbListQueryDataSet(data.query_ids, data.dim);
        return data;
    }

    if (scenario.total_count_override > 0) {
        data.total_count = scenario.total_count_override;
        data.dim = row.data_kind == DataKind::DenseBin ? 128 : kDenseDim;
    } else if (row.data_kind == DataKind::GpuFp32) {
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
        data.train_ds =
            GenNullableBinaryDataSet(data.total_count, data.valid_ids, data.dim, data.full_ds, owned_binary[key]);
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
BuildJson(const IndexRow& row, const Scenario& scenario, const MatrixData& data, const fs::path& raw_data_file,
          const fs::path& file_index_prefix) {
    auto mode = scenario.mode;
    auto emb_list_strategy = EmbListStrategyFor(scenario);
    if (row.index_type == "HNSW_V1") {
        auto json = DenseConfigFor(row, mode, emb_list_strategy);
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
            ApplyEmbListStrategyConfig(json, emb_list_strategy);
        }
        return json;
    }
    return DenseConfigFor(row, mode, emb_list_strategy);
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
CreateIndex(const IndexRow& row, const MatrixData& data, std::shared_ptr<milvus::FileManager> file_manager = nullptr,
            std::optional<int32_t> version_override = std::nullopt) {
    CreateResult result;

#ifndef KNOWHERE_WITH_CARDINAL
    if (row.requires_current_version) {
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

    const auto version = version_override.value_or(VersionForRow(row));
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

    if (row.index_type == knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR && data.train_ds != nullptr) {
        knowhere::ViewDataOp view_data = [train = data.train_ds, dim = data.train_ds->GetDim()](size_t in_id) {
            auto* tensor = static_cast<const float*>(train->GetTensor());
            return tensor + in_id * dim;
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

std::string
PathSafe(std::string value) {
    for (auto& c : value) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) {
            c = '_';
        }
    }
    return value;
}

fs::path
NullableIdMapWorkDir(std::string name) {
    return fs::temp_directory_path() / ("knowhere_nullable_id_map_matrix_" + PathSafe(std::move(name)));
}

knowhere::Status
WriteEmbListOffsetToFile(const knowhere::DataSetPtr& dataset, size_t offset_count, const fs::path& path);

std::shared_ptr<BuiltArtifact>
BuildArtifactForScenario(const IndexRow& row, const Scenario& scenario) {
    auto artifact = std::make_shared<BuiltArtifact>();
    artifact->row = row;
    artifact->scenario = scenario;
    auto work_dir_name = row.label + "_" + ModeName(scenario.mode) + "_" + NullableName(scenario.nullable_ratio);
    if (scenario.mode == Mode::EmbList) {
        work_dir_name += "_" + EmbListStrategyFor(scenario);
    }
    artifact->work_dir = NullableIdMapWorkDir(work_dir_name);
    RemoveAllNoThrow(artifact->work_dir);
    fs::create_directories(artifact->work_dir);
    artifact->main_file = artifact->work_dir / "main.index";
    artifact->emb_meta_file = artifact->work_dir / "emb_list_meta.bin";
    artifact->emb_raw_file = artifact->work_dir / "emb_list_raw.index";
    artifact->emb_offset_file = artifact->work_dir / "emb_list_offset.bin";
    artifact->raw_data_file = artifact->work_dir / "raw_data.bin";
    artifact->file_index_prefix = artifact->work_dir / "file_index";
    fs::create_directories(artifact->file_index_prefix.parent_path());

    artifact->data = BuildMatrixData(row, scenario, artifact->work_dir);
    if (row.data_kind == DataKind::DiskAnn || row.data_kind == DataKind::Aisaq || row.data_kind == DataKind::MinHash) {
        artifact->file_manager = LocalFileManager();
    }
    WriteRawDataIfNeeded(row, artifact->data, artifact->raw_data_file);
    artifact->json = BuildJson(row, scenario, artifact->data, artifact->raw_data_file, artifact->file_index_prefix);
    if (scenario.force_raw_data) {
        UseRawDataMode(artifact->json);
    }
    if (row.data_kind == DataKind::DiskAnn && scenario.mode == Mode::EmbList) {
        auto status = WriteEmbListOffsetToFile(artifact->data.train_ds, artifact->data.valid_ids.size() + 1,
                                               artifact->emb_offset_file);
        if (status == knowhere::Status::success) {
            artifact->json["emb_list_offset_file_path"] = artifact->emb_offset_file.string();
        } else {
            artifact->build_status = status;
        }
    }

    auto created = CreateIndex(row, artifact->data, artifact->file_manager, VersionForScenario(row, scenario));
    artifact->create_ok = created.ok;
    artifact->create_status = created.status;
    if (!created.ok) {
        return artifact;
    }

    artifact->index = std::move(created.index);
    artifact->index.SetIdMapType(knowhere::IdMap::Type::SEALED);
    artifact->build_status =
        AttachRuntimeIdMapData(artifact->data.train_ds, artifact->data.valid_ids, artifact->data.total_count);
    if (artifact->build_status == knowhere::Status::success) {
        artifact->build_status = artifact->index.Build(artifact->data.train_ds, artifact->json);
    }
    return artifact;
}

knowhere::Status
EnsureSerialized(BuiltArtifact& artifact) {
    if (artifact.serialized) {
        return artifact.serialize_status;
    }
    artifact.serialize_status = artifact.index.Serialize(artifact.binset);
    // Serialized index payloads do not carry validity input.
    if (artifact.serialize_status == knowhere::Status::success &&
        artifact.binset.GetByName("EXTERNAL_ID_MAP") != nullptr) {
        artifact.serialize_status = knowhere::Status::invalid_binary_set;
    }
    artifact.serialized = true;
    return artifact.serialize_status;
}

knowhere::Status
AttachRuntimeIdMapData(const knowhere::DataSetPtr& dataset, const std::vector<int32_t>& valid_ids,
                       int64_t total_count) {
    if (dataset == nullptr || total_count < 0 || static_cast<int64_t>(valid_ids.size()) > total_count) {
        return knowhere::Status::invalid_args;
    }
    if (total_count > std::numeric_limits<int32_t>::max()) {
        return knowhere::Status::invalid_args;
    }
    for (auto id : valid_ids) {
        if (id < 0 || id >= total_count) {
            return knowhere::Status::invalid_args;
        }
    }
    dataset->SetIdMapData(knowhere::IdMapData::FromIds(valid_ids.empty() ? nullptr : valid_ids.data(), valid_ids.size(),
                                                       static_cast<size_t>(total_count)));
    return knowhere::Status::success;
}

knowhere::Status
BuildRuntimeIdMap(const std::vector<int32_t>& valid_ids, int64_t total_count, knowhere::IdMap& id_map) {
    if (total_count < 0 || static_cast<int64_t>(valid_ids.size()) > total_count) {
        return knowhere::Status::invalid_args;
    }
    try {
        id_map.SetType(knowhere::IdMap::Type::SEALED);
        if (total_count > std::numeric_limits<int32_t>::max()) {
            return knowhere::Status::invalid_args;
        }
        for (auto id : valid_ids) {
            if (id < 0 || id >= total_count) {
                return knowhere::Status::invalid_args;
            }
        }
        id_map.AddFromData(knowhere::IdMapData::FromIds(valid_ids.empty() ? nullptr : valid_ids.data(),
                                                        valid_ids.size(), static_cast<size_t>(total_count)));
        return knowhere::Status::success;
    } catch (const std::exception&) {
        return knowhere::Status::invalid_args;
    }
}

knowhere::Status
BuildRuntimeValidBitmap(const std::vector<int32_t>& valid_ids, int64_t total_count, knowhere::IdMap& id_map) {
    if (total_count < 0 || static_cast<int64_t>(valid_ids.size()) > total_count) {
        return knowhere::Status::invalid_args;
    }
    try {
        id_map.SetType(knowhere::IdMap::Type::SEALED);
        auto valid_data = std::make_unique<bool[]>(static_cast<size_t>(total_count));
        std::fill_n(valid_data.get(), static_cast<size_t>(total_count), false);
        for (auto id : valid_ids) {
            if (id < 0 || id >= total_count) {
                return knowhere::Status::invalid_args;
            }
            valid_data[static_cast<size_t>(id)] = true;
        }
        id_map.AddFromData(knowhere::IdMapData::FromValidData(valid_data.get(), static_cast<size_t>(total_count)));
        return knowhere::Status::success;
    } catch (const std::exception&) {
        return knowhere::Status::invalid_args;
    }
}

knowhere::Status
SetRuntimeIdMap(knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data) {
    if (index.Node() == nullptr) {
        return knowhere::Status::invalid_args;
    }
    RETURN_IF_ERROR(BuildRuntimeIdMap(data.valid_ids, data.total_count, index.GetIdMap()));
    return knowhere::Status::success;
}

knowhere::Status
SetRuntimeValidBitmap(knowhere::Index<knowhere::IndexNode>& index, const MatrixData& data) {
    if (index.Node() == nullptr) {
        return knowhere::Status::invalid_args;
    }
    RETURN_IF_ERROR(BuildRuntimeValidBitmap(data.valid_ids, data.total_count, index.GetIdMap()));
    return knowhere::Status::success;
}

knowhere::Status
EnsureBinaryLoaded(BuiltArtifact& artifact) {
    if (artifact.binary_loaded_ready || artifact.binary_deserialize_status != knowhere::Status::success) {
        return artifact.binary_deserialize_status;
    }
    RETURN_IF_ERROR(EnsureSerialized(artifact));
    auto created = CreateIndex(artifact.row, artifact.data, artifact.file_manager,
                               VersionForScenario(artifact.row, artifact.scenario));
    if (!created.ok) {
        artifact.binary_deserialize_status = created.status;
        return artifact.binary_deserialize_status;
    }
    artifact.binary_loaded = std::move(created.index);
    artifact.binary_deserialize_status = SetRuntimeValidBitmap(artifact.binary_loaded, artifact.data);
    if (artifact.binary_deserialize_status == knowhere::Status::success) {
        artifact.binary_deserialize_status = artifact.binary_loaded.Deserialize(artifact.binset, artifact.json);
    }
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
        if (key == knowhere::meta::EMB_LIST_META || key == knowhere::meta::EMB_LIST_RAW_INDEX) {
            continue;
        }
        return binary;
    }
    return nullptr;
}

knowhere::Json
FileLoadJson(const BuiltArtifact& artifact) {
    auto json = artifact.json;
    if (artifact.scenario.mode == Mode::EmbList) {
        if (fs::exists(artifact.emb_meta_file)) {
            json["emb_list_meta_file_path"] = artifact.emb_meta_file.string();
        }
        if (fs::exists(artifact.emb_raw_file)) {
            json["emb_list_raw_index_file_path"] = artifact.emb_raw_file.string();
        }
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

    if (artifact.scenario.mode == Mode::EmbList) {
        auto meta_binary = artifact.binset.GetByName(knowhere::meta::EMB_LIST_META);
        if (meta_binary != nullptr) {
            RETURN_IF_ERROR(WriteBinaryToFile(meta_binary, artifact.emb_meta_file));
        }
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
    auto created = CreateIndex(artifact.row, artifact.data, artifact.file_manager,
                               VersionForScenario(artifact.row, artifact.scenario));
    if (!created.ok) {
        artifact.file_deserialize_status = created.status;
        return artifact.file_deserialize_status;
    }
    artifact.file_loaded = std::move(created.index);
    artifact.file_deserialize_status = SetRuntimeValidBitmap(artifact.file_loaded, artifact.data);
    if (artifact.file_deserialize_status == knowhere::Status::success) {
        if (artifact.row.data_kind == DataKind::DiskAnn || artifact.row.data_kind == DataKind::Aisaq ||
            artifact.row.data_kind == DataKind::MinHash) {
            artifact.file_deserialize_status = artifact.file_loaded.Deserialize(artifact.binset, artifact.json);
        } else {
            artifact.file_deserialize_status =
                artifact.file_loaded.DeserializeFromFile(artifact.main_file.string(), FileLoadJson(artifact));
        }
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
    REQUIRE(artifact.build_status == knowhere::Status::success);
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
        auto result = index.AnnIterator(artifact.data.query_ds, artifact.json, needs_bitset ? bitset : nullptr, true,
                                        &op_context);
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
    auto artifact = BuildArtifactForScenario(row, scenario);
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

    REQUIRE(artifact->build_status == knowhere::Status::success);

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
            {
                const auto& loaded_id_map = artifact->binary_loaded.Node()->GetIdMap();
                const auto& index_id_map = artifact->index.Node()->GetIdMap();
                REQUIRE(loaded_id_map.OutCount() == index_id_map.OutCount());
            }
            return;
        case Operation::FileSerialize:
            REQUIRE(EnsureFileSerialized(*artifact) == knowhere::Status::success);
            return;
        case Operation::FileDeserialize:
            REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
            {
                const auto& loaded_id_map = artifact->file_loaded.Node()->GetIdMap();
                if (artifact->row.data_kind == DataKind::MinHash && scenario.nullable_ratio == NullableRatio::R0) {
                    REQUIRE(loaded_id_map.OutCount() == static_cast<size_t>(artifact->data.total_count));
                } else {
                    const auto& index_id_map = artifact->index.Node()->GetIdMap();
                    REQUIRE(loaded_id_map.OutCount() == index_id_map.OutCount());
                }
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

    auto artifact = BuildArtifactForScenario(row, scenario);
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

std::unique_ptr<bool[]>
MakeValidData(size_t total_count, const std::vector<int32_t>& valid_ids) {
    auto valid_data = std::make_unique<bool[]>(total_count);
    std::fill_n(valid_data.get(), total_count, false);
    for (auto id : valid_ids) {
        valid_data[static_cast<size_t>(id)] = true;
    }
    return valid_data;
}

void
SetIdMapIds(knowhere::IdMap& map, const std::vector<int32_t>& ids, int64_t count) {
    map.SetType(knowhere::IdMap::Type::SEALED);
    map.AddFromData(
        knowhere::IdMapData::FromIds(ids.empty() ? nullptr : ids.data(), ids.size(), static_cast<size_t>(count)));
}

void
RequireIdMapForwarding(knowhere::IndexNode& node) {
    knowhere::IdMap map;
    map.SetType(knowhere::IdMap::Type::SEALED);
    const std::vector<int32_t> ids = {1, 3};
    map.AddFromData(knowhere::IdMapData::FromIds(ids.data(), ids.size(), 4));
    node.GetIdMap() = std::move(map);
    const auto& id_map = node.GetIdMap();
    REQUIRE(id_map.OutCount() == 4);
    REQUIRE(id_map.ValidBitmap().size() == 4);
    REQUIRE(id_map.MapInToOut(1) == 3);
}

void
RequireReadApisReturnError(knowhere::IndexNode& node) {
    auto dataset = GenNullableDenseDataSet(4, {0, 2}, kDenseDim);
    auto query = GenDenseQueryDataSet({0}, kDenseDim);
    int64_t storage_id = 0;
    auto ids = knowhere::GenIdsDataSet(1, &storage_id);
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

    // Storage-id APIs should surface the default not_implemented boundary directly.
    auto calc_dist = node.CalcDistByStorageIds(dataset, knowhere::BitsetView{}, &storage_id, 1, false);
    REQUIRE(!calc_dist.has_value());
    REQUIRE(calc_dist.error() == Status::not_implemented);

    auto emb_list = node.GetEmbListByIds(ids, knowhere::metric::L2);
    REQUIRE(!emb_list.has_value());
    REQUIRE(emb_list.error() == Status::emb_list_inner_error);
}

struct RequiredIndexRow {
    RequiredIndexRow(const char* label, const char* index_type, DataKind data_kind)
        : label(label), index_type(index_type), data_kind(data_kind) {
    }

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

void
RequireIdMapVectorState(const knowhere::IdMap& map, size_t out_count, const std::vector<int32_t>& in_to_out_ids) {
    REQUIRE(map.OutCount() == out_count);
    REQUIRE(map.InCount() == in_to_out_ids.size());
    RequireIdArrayEquals(map.InToOutIds(), in_to_out_ids);
    for (size_t in_id = 0; in_id < in_to_out_ids.size(); ++in_id) {
        CAPTURE(in_id);
        REQUIRE(map.MapInToOut(static_cast<int64_t>(in_id)) == in_to_out_ids[in_id]);
        REQUIRE(RequireCompactOutToIn(map, in_to_out_ids[in_id]) == static_cast<int64_t>(in_id));
    }
}

void
RequireIdMapVectorStateOnly(const knowhere::IdMap& map, size_t out_count, const std::vector<int32_t>& in_to_out_ids) {
    RequireIdMapVectorState(map, out_count, in_to_out_ids);
    REQUIRE(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST).empty());
}

void
RequireDatasetsEqual(const knowhere::DataSet& lhs, const knowhere::DataSet& rhs) {
    REQUIRE(lhs.GetRows() == rhs.GetRows());
    REQUIRE(lhs.GetDim() == rhs.GetDim());
    const auto rows = lhs.GetRows();
    const auto dim = lhs.GetDim();

    const auto* lhs_lims = lhs.GetLims();
    const auto* rhs_lims = rhs.GetLims();
    size_t value_count = dim == 0 && lhs_lims != nullptr ? lhs_lims[rows] : static_cast<size_t>(rows * dim);
    if (lhs_lims != nullptr || rhs_lims != nullptr) {
        REQUIRE(lhs_lims != nullptr);
        REQUIRE(rhs_lims != nullptr);
        for (int64_t i = 0; i <= rows; ++i) {
            REQUIRE(lhs_lims[i] == rhs_lims[i]);
        }
    }

    const auto* lhs_ids = lhs.GetIds();
    const auto* rhs_ids = rhs.GetIds();
    if (lhs_ids != nullptr || rhs_ids != nullptr) {
        REQUIRE(lhs_ids != nullptr);
        REQUIRE(rhs_ids != nullptr);
        for (size_t i = 0; i < value_count; ++i) {
            CAPTURE(i);
            REQUIRE(lhs_ids[i] == rhs_ids[i]);
        }
    }

    const auto* lhs_distances = lhs.GetDistance();
    const auto* rhs_distances = rhs.GetDistance();
    if (lhs_distances != nullptr || rhs_distances != nullptr) {
        REQUIRE(lhs_distances != nullptr);
        REQUIRE(rhs_distances != nullptr);
        for (size_t i = 0; i < value_count; ++i) {
            CAPTURE(i);
            REQUIRE(lhs_distances[i] == Catch::Approx(rhs_distances[i]).epsilon(0.0001f));
        }
    }
}

void
RequireIteratorFirstResultsEqual(const knowhere::Index<knowhere::IndexNode>& lhs,
                                 const knowhere::Index<knowhere::IndexNode>& rhs, const knowhere::DataSetPtr& query_ds,
                                 const knowhere::Json& json, const knowhere::BitsetView& bitset) {
    auto lhs_iterators = lhs.AnnIterator(query_ds, json, bitset);
    auto rhs_iterators = rhs.AnnIterator(query_ds, json, bitset);
    REQUIRE(lhs_iterators.has_value());
    REQUIRE(rhs_iterators.has_value());
    REQUIRE(lhs_iterators.value().size() == rhs_iterators.value().size());

    for (size_t i = 0; i < lhs_iterators.value().size(); ++i) {
        auto lhs_has_next = lhs_iterators.value()[i]->HasNext();
        auto rhs_has_next = rhs_iterators.value()[i]->HasNext();
        REQUIRE(lhs_has_next.has_value());
        REQUIRE(rhs_has_next.has_value());
        REQUIRE(lhs_has_next.value() == rhs_has_next.value());
        if (!lhs_has_next.value()) {
            continue;
        }

        auto lhs_next = lhs_iterators.value()[i]->Next();
        auto rhs_next = rhs_iterators.value()[i]->Next();
        REQUIRE(lhs_next.has_value());
        REQUIRE(rhs_next.has_value());
        REQUIRE(lhs_next.value().first == rhs_next.value().first);
        REQUIRE(lhs_next.value().second == Catch::Approx(rhs_next.value().second).epsilon(0.0001f));
    }
}

void
RequireEmbListRetrieveMatches(const knowhere::DataSet& result, const std::vector<int32_t>& public_ids,
                              const std::vector<size_t>& vector_counts, int64_t dim) {
    REQUIRE(public_ids.size() == vector_counts.size());
    REQUIRE(result.GetRows() == static_cast<int64_t>(public_ids.size()));
    REQUIRE(result.GetDim() == dim);

    const auto* offsets = result.Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
    REQUIRE(offsets != nullptr);
    size_t total_vectors = 0;
    for (size_t i = 0; i < vector_counts.size(); ++i) {
        REQUIRE(offsets[i] == total_vectors);
        total_vectors += vector_counts[i];
        REQUIRE(offsets[i + 1] == total_vectors);
    }

    if (total_vectors == 0) {
        return;
    }
    const auto* tensor = static_cast<const float*>(result.GetTensor());
    REQUIRE(tensor != nullptr);
    size_t vector_offset = 0;
    for (size_t i = 0; i < public_ids.size(); ++i) {
        auto expected = DenseVectorForLogicalId(public_ids[i], dim);
        for (size_t v = 0; v < vector_counts[i]; ++v) {
            for (int64_t j = 0; j < dim; ++j) {
                CAPTURE(i, v, j, public_ids[i]);
                REQUIRE(tensor[(vector_offset + v) * dim + j] == Catch::Approx(expected[j]));
            }
        }
        vector_offset += vector_counts[i];
    }
}

}  // namespace

TEST_CASE("Nullable IdMap primitives", "[nullable][id_map]") {
    SECTION("IdMap maps compact vector ids") {
        knowhere::IdMap map;
        const std::vector<int32_t> ids = {0, 2, 4, 6};
        SetIdMapIds(map, ids, 8);
        REQUIRE(!map.InToOutIds().empty());
        REQUIRE(map.OutCount() == 8);
        REQUIRE(map.MapInToOut(2) == 4);
        REQUIRE(map.MapInToOut(static_cast<int64_t>(ids.size())) == -1);
        REQUIRE(RequireCompactOutToIn(map, 4) == 2);
        RequireInvalidCompactOutToIn(map, 5);

        std::vector<int64_t> result_ids = {0, 1, 2, 3, 4, -1};
        map.MapInToOut(result_ids.data(), result_ids.size());
        REQUIRE(result_ids == std::vector<int64_t>{0, 2, 4, 6, -1, -1});
    }

    SECTION("IdMap requires an enabled storage mode before nullable input") {
        knowhere::IdMap map;
        const int32_t ids[] = {0};
        REQUIRE_THROWS(map.AddFromData(knowhere::IdMapData::FromIds(ids, 1, 1)));
        REQUIRE_FALSE(map.IsEnabled());
        REQUIRE_FALSE(map.HasVectorMap());
    }

    SECTION("IdMap rejects malformed ids before publishing state") {
        auto require_rejected_on_empty_map = [](knowhere::IdMapData data) {
            knowhere::IdMap map;
            map.SetType(knowhere::IdMap::Type::SEALED);
            REQUIRE_THROWS(map.AddFromData(data));
            REQUIRE(map.OutCount() == 0);
            REQUIRE(map.InCount() == 0);
            REQUIRE(map.ValidBitmap().empty());
            REQUIRE(map.InToOutIds().empty());
        };
        auto require_rejected_on_growing_map = [](knowhere::IdMapData data) {
            knowhere::IdMap map;
            map.SetType(knowhere::IdMap::Type::GROWING);
            const int32_t base_ids[] = {1, 3};
            map.AddFromData(knowhere::IdMapData::FromIds(base_ids, 2, 4));

            REQUIRE_THROWS(map.AddFromData(data));
            RequireIdMapVectorStateOnly(map, 4, {1, 3});
            RequireInvalidCompactOutToIn(map, 0);
            RequireInvalidCompactOutToIn(map, 2);
            RequireInvalidCompactOutToIn(map, 4);
        };

        int32_t negative_ids[] = {0, -1};
        require_rejected_on_empty_map(knowhere::IdMapData::FromIds(negative_ids, 2, 4));
        require_rejected_on_growing_map(knowhere::IdMapData::FromIds(negative_ids, 2, 4));

        int32_t out_of_range_ids[] = {0, 4};
        require_rejected_on_empty_map(knowhere::IdMapData::FromIds(out_of_range_ids, 2, 4));
        require_rejected_on_growing_map(knowhere::IdMapData::FromIds(out_of_range_ids, 2, 4));

        int32_t duplicate_ids[] = {1, 1};
        require_rejected_on_empty_map(knowhere::IdMapData::FromIds(duplicate_ids, 2, 4));
        require_rejected_on_growing_map(knowhere::IdMapData::FromIds(duplicate_ids, 2, 4));

        require_rejected_on_empty_map(knowhere::IdMapData::FromIds(nullptr, 1, 4));
        require_rejected_on_growing_map(knowhere::IdMapData::FromIds(nullptr, 1, 4));

        int32_t id_with_empty_domain[] = {0};
        require_rejected_on_empty_map(knowhere::IdMapData::FromIds(id_with_empty_domain, 1, 0));
        require_rejected_on_growing_map(knowhere::IdMapData::FromIds(id_with_empty_domain, 1, 0));
    }

    SECTION("IdMap rejects malformed validity inputs") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::SEALED);
        REQUIRE_THROWS(map.AddFromData(knowhere::IdMapData::FromValidData(nullptr, 4)));
        REQUIRE_THROWS(map.AddFromData(knowhere::IdMapData::FromValidBitmap(nullptr, 4)));
        REQUIRE(map.OutCount() == 0);
        REQUIRE(map.InCount() == 0);
        REQUIRE(map.ValidBitmap().empty());
        REQUIRE(map.InToOutIds().empty());
    }

    SECTION("IdMap type is fixed after data is published") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::GROWING);
        const int32_t ids[] = {0, 2};
        map.AddFromData(knowhere::IdMapData::FromIds(ids, 2, 4));
        REQUIRE_THROWS(map.SetType(knowhere::IdMap::Type::SEALED));
        REQUIRE(map.type() == knowhere::IdMap::Type::GROWING);
        REQUIRE(map.OutCount() == 4);
        RequireIdArrayEquals(map.InToOutIds(), {0, 2});
    }

    SECTION("IdMap rejects malformed emb-list offsets") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::GROWING);
        const int32_t ids[] = {1, 4};
        map.AddFromData(knowhere::IdMapData::FromIds(ids, 2, 6));

        const size_t decreasing_offsets[] = {0, 3, 2};
        REQUIRE_THROWS(map.AppendEmbListIds(0, decreasing_offsets, 2));

        const size_t out_of_map_offsets[] = {0, 1};
        REQUIRE_THROWS(map.AppendEmbListIds(2, out_of_map_offsets, 1));

        REQUIRE(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST).empty());
        RequireIdArrayEquals(map.InToOutIds(), {1, 4});
    }

    SECTION("IdMap can be built directly from valid data") {
        auto valid_data = MakeValidData(10, {1, 4, 9});
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::SEALED);
        map.AddFromData(knowhere::IdMapData::FromValidData(valid_data.get(), 10));
        REQUIRE(map.OutCount() == 10);
        RequireIdArrayEquals(map.InToOutIds(), {});

        map.FinalizeVectorIds();
        REQUIRE(!map.InToOutIds().empty());
        RequireIdArrayEquals(map.InToOutIds(), {1, 4, 9});
        REQUIRE(RequireCompactOutToIn(map, 1) == 0);
        REQUIRE(RequireCompactOutToIn(map, 4) == 1);
        REQUIRE(RequireCompactOutToIn(map, 9) == 2);
        RequireInvalidCompactOutToIn(map, 8);
    }

    SECTION("IdMap views keep array buffers alive") {
        auto bitmap = MakeValidBitmap(10, {1, 4, 9});
        knowhere::IdArray ids_array;
        knowhere::BitmapArray valid_array;
        {
            knowhere::IdMap map;
            map.SetType(knowhere::IdMap::Type::SEALED);
            map.AddFromData(knowhere::IdMapData::FromValidBitmap(bitmap.data(), 10));
            REQUIRE(map.InToOutIds().empty());
            map.FinalizeVectorIds();

            ids_array = map.InToOutIds();
            valid_array = map.ValidBitmap();
        }

        RequireIdArrayEquals(ids_array, {1, 4, 9});
        REQUIRE(valid_array.size() == 10);
        REQUIRE((valid_array.data()[1 >> 3] & (1U << (1 & 7))) != 0);
        REQUIRE((valid_array.data()[4 >> 3] & (1U << (4 & 7))) != 0);
        REQUIRE((valid_array.data()[9 >> 3] & (1U << (9 & 7))) != 0);
    }

    SECTION("IdMap appends ids and bitmap together") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::GROWING);
        const int32_t ids[] = {0, 2};
        map.AddFromData(knowhere::IdMapData::FromIds(ids, 2, 4));
        REQUIRE(map.OutCount() == 4);
        REQUIRE(map.MapInToOut(1) == 2);
        REQUIRE(RequireCompactOutToIn(map, 2) == 1);
        RequireInvalidCompactOutToIn(map, 1);

        std::vector<int64_t> result_ids = {0, 1, -1};
        map.MapInToOut(result_ids.data(), result_ids.size());
        REQUIRE(result_ids == std::vector<int64_t>{0, 2, -1});
    }

    SECTION("BitsetView keeps a stable id window over a growing IdMap") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::GROWING);
        const int32_t base_ids[] = {0, 2};
        map.AddFromData(knowhere::IdMapData::FromIds(base_ids, 2, 4));

        std::vector<uint8_t> filter_bits(1, 0);
        knowhere::BitsetView bitset(filter_bits.data(), 4);
        const auto& out_ids = map.InToOutIds();
        REQUIRE(out_ids.is_append_array());
        bitset.set_out_ids(out_ids, out_ids.size());

        const int32_t append_ids[] = {0, 1};
        map.AddFromData(knowhere::IdMapData::FromIds(append_ids, 2, 2));

        REQUIRE(bitset.out_ids_count() == 2);
        REQUIRE_FALSE(bitset.test(0));
        REQUIRE_FALSE(bitset.test(1));
        REQUIRE(bitset.test(2));
        RequireIdArrayEquals(map.InToOutIds(), {0, 2, 4, 5});
    }

    SECTION("Growing IdMap appends non-byte-aligned bitmap without changing visible prefix") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::GROWING);

        const bool first_batch[] = {true, false, true, true, false};
        map.AddFromData(knowhere::IdMapData::FromValidData(first_batch, sizeof(first_batch) / sizeof(bool)));

        const auto first_out_count = map.OutCount();
        const auto first_in_count = map.InCount();
        auto first_valid_bitmap = map.ValidBitmap();
        auto first_out_ids = map.InToOutIds();

        const bool second_batch[] = {true, true, false, true, false};
        map.AddFromData(knowhere::IdMapData::FromValidData(second_batch, sizeof(second_batch) / sizeof(bool)));

        REQUIRE(first_out_count == 5);
        REQUIRE(first_in_count == 3);
        REQUIRE(first_valid_bitmap.size() == 5);
        REQUIRE((first_valid_bitmap[0] & 0x1F) == 0x0D);
        RequireIdArrayEquals(first_out_ids, {0, 2, 3});

        REQUIRE(map.OutCount() == 10);
        REQUIRE(map.InCount() == 6);
        REQUIRE((map.ValidBitmap()[0] & 0xFF) == 0x6D);
        REQUIRE((map.ValidBitmap()[1] & 0x03) == 0x01);
        RequireIdArrayEquals(map.InToOutIds(), {0, 2, 3, 5, 6, 8});
    }

    SECTION("Growing IdMap keeps search-side prefixes stable across appends") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::GROWING);

        const bool first_batch[] = {true, false, true, true, false};
        map.AddFromData(knowhere::IdMapData::FromValidData(first_batch, sizeof(first_batch) / sizeof(bool)));

        auto search_valid_bitmap = map.ValidBitmap();
        auto search_out_ids = map.InToOutIds();
        std::vector<uint8_t> filter_bits(1, 0);
        knowhere::BitsetView search_bitset(filter_bits.data(), search_valid_bitmap.size());
        search_bitset.set_out_ids(search_out_ids, search_out_ids.size());
        search_bitset.set_vector_count(search_out_ids.size());
        search_bitset.set_filter_count(0);

        auto second_batch = MakeValidBitmap(7, {0, 3, 6});
        map.AddFromData(knowhere::IdMapData::FromValidBitmap(second_batch.data(), 7));

        REQUIRE(search_valid_bitmap.size() == 5);
        REQUIRE((search_valid_bitmap[0] & 0x1F) == 0x0D);
        RequireIdArrayEquals(search_out_ids, {0, 2, 3});
        REQUIRE(search_bitset.out_ids_count() == 3);
        REQUIRE_FALSE(search_bitset.test(0));
        REQUIRE_FALSE(search_bitset.test(1));
        REQUIRE_FALSE(search_bitset.test(2));
        REQUIRE(search_bitset.test(3));

        REQUIRE(map.OutCount() == 12);
        REQUIRE(map.InCount() == 6);
        RequireIdArrayEquals(map.InToOutIds(), {0, 2, 3, 5, 8, 11});
        REQUIRE(RequireCompactOutToIn(map, 11) == 5);
    }

    SECTION("Sealed IdMap uses array storage") {
        const bool base_valid[] = {true, true, true, true};
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::SEALED);
        map.AddFromData(knowhere::IdMapData::FromValidData(base_valid, 4));
        map.FinalizeVectorIds();
        RequireIdArrayEquals(map.InToOutIds(), {0, 1, 2, 3});
        REQUIRE(RequireCompactOutToIn(map, 3) == 3);

        const int32_t append_ids[] = {0, 2};
        map.AddFromData(knowhere::IdMapData::FromIds(append_ids, 2, 4));
        REQUIRE(map.OutCount() == 4);
        RequireIdArrayEquals(map.InToOutIds(), {0, 2});
        REQUIRE(RequireCompactOutToIn(map, 2) == 1);
        REQUIRE(map.InToOutIds().is_array());
    }

    SECTION("IdMap appends emb-list ids without bitmap") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::GROWING);
        const size_t first_offsets[] = {0, 2, 5};
        map.AppendEmbListIds(0, first_offsets, 2);

        REQUIRE(map.OutCount() == 0);
        REQUIRE(map.InToOutIds().empty());
        REQUIRE(map.ValidBitmap().empty());
        auto first_ebl_ids = map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST);
        RequireIdArrayEquals(first_ebl_ids, {0, 0, 1, 1, 1});

        const size_t second_offsets[] = {0, 2};
        map.AppendEmbListIds(2, second_offsets, 1);

        RequireIdArrayEquals(first_ebl_ids, {0, 0, 1, 1, 1});
        REQUIRE(map.OutCount() == 0);
        REQUIRE(map.InToOutIds().empty());
        REQUIRE(map.ValidBitmap().empty());
        RequireIdArrayEquals(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {0, 0, 1, 1, 1, 2, 2});
    }

    SECTION("IdMap appends emb-list ids from the list-domain public map") {
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::GROWING);

        const int32_t base_list_ids[] = {1, 4};
        map.AddFromData(knowhere::IdMapData::FromIds(base_list_ids, 2, 6));
        const size_t base_offsets[] = {0, 2, 5};
        map.AppendEmbListIds(0, base_offsets, 2);
        RequireIdArrayEquals(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {1, 1, 4, 4, 4});

        const int32_t append_list_ids[] = {1, 3};
        map.AddFromData(knowhere::IdMapData::FromIds(append_list_ids, 2, 4));
        const size_t append_offsets[] = {0, 0, 2};
        map.AppendEmbListIds(2, append_offsets, 2);

        RequireIdArrayEquals(map.InToOutIds(), {1, 4, 7, 9});
        RequireIdArrayEquals(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {1, 1, 4, 4, 4, 9, 9});
    }

    SECTION("IdMap builds derived emb-list ids without changing bitmap") {
        auto bitmap = MakeValidBitmap(10, {1, 4, 9});
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::SEALED);
        map.AddFromData(knowhere::IdMapData::FromValidBitmap(bitmap.data(), 10));
        map.FinalizeVectorIds();

        const size_t offsets[] = {0, 2, 3, 6};
        map.FinalizeEmbListIds(offsets, 3);
        RequireIdArrayEquals(map.InToOutIds(), {1, 4, 9});
        RequireIdArrayEquals(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {1, 1, 4, 9, 9, 9});
        REQUIRE(RequireCompactOutToIn(map, 4) == 1);

        auto valid_before_noop = map.ValidBitmap();
        REQUIRE(valid_before_noop.size() == 10);
        REQUIRE((valid_before_noop.data()[1 >> 3] & (1U << (1 & 7))) != 0);
        REQUIRE((valid_before_noop.data()[4 >> 3] & (1U << (4 & 7))) != 0);
        REQUIRE((valid_before_noop.data()[9 >> 3] & (1U << (9 & 7))) != 0);

        map.FinalizeEmbListIds(nullptr, 0);
        RequireIdArrayEquals(map.InToOutIds(), {1, 4, 9});
        RequireIdArrayEquals(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {1, 1, 4, 9, 9, 9});
        REQUIRE(RequireCompactOutToIn(map, 4) == 1);

        auto valid_after_noop = map.ValidBitmap();
        REQUIRE(valid_after_noop.size() == 10);
        REQUIRE((valid_after_noop.data()[1 >> 3] & (1U << (1 & 7))) != 0);
        REQUIRE((valid_after_noop.data()[4 >> 3] & (1U << (4 & 7))) != 0);
        REQUIRE((valid_after_noop.data()[9 >> 3] & (1U << (9 & 7))) != 0);
    }

    SECTION("IdMap clears handed-off vector-domain maps in both directions") {
        auto bitmap = MakeValidBitmap(10, {1, 4, 9});
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::SEALED);
        map.AddFromData(knowhere::IdMapData::FromValidBitmap(bitmap.data(), 10));
        map.FinalizeVectorIds();

        const size_t offsets[] = {0, 2, 3, 6};
        map.FinalizeEmbListIds(offsets, 3);
        RequireIdArrayEquals(map.InToOutIds(), {1, 4, 9});
        RequireIdArrayEquals(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {1, 1, 4, 9, 9, 9});
        REQUIRE(map.OutCount() == 10);
        REQUIRE(map.InCount() == 3);
        REQUIRE(RequireCompactOutToIn(map, 9) == 2);

        map.ClearEblIds();
        RequireIdArrayEquals(map.InToOutIds(), {1, 4, 9});
        REQUIRE(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST).empty());
        REQUIRE(map.OutCount() == 10);
        REQUIRE(map.InCount() == 3);
        REQUIRE(RequireCompactOutToIn(map, 4) == 1);

        map.ClearIds();
        REQUIRE(map.InToOutIds().empty());
        REQUIRE(map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST).empty());
        REQUIRE(map.OutCount() == 10);
        REQUIRE(map.InCount() == 3);
        REQUIRE(!map.ValidBitmap().empty());
        RequireInvalidCompactOutToIn(map, 1);
        RequireInvalidCompactOutToIn(map, 4);
        RequireInvalidCompactOutToIn(map, 9);
    }

    SECTION("IdMap supports mmap-backed id arrays") {
        auto work_dir = NullableIdMapWorkDir("id_map_mmap");
        RemoveAllNoThrow(work_dir);

        knowhere::IdMapMmapOptions mmap_options;
        mmap_options.enable_in_to_out_ids = true;
        mmap_options.enable_out_to_in_ids = true;
        mmap_options.mmap_dir_path = work_dir.string();

        knowhere::IdArray ids_array;
        knowhere::IdArray ebl_ids_array;
        knowhere::BitmapArray valid_array;
        {
            auto valid_data = MakeValidData(10, {1, 4, 6, 9});
            knowhere::IdMap map;
            map.SetType(knowhere::IdMap::Type::SEALED);
            map.ConfigureMmap(mmap_options);
            map.AddFromData(knowhere::IdMapData::FromValidData(valid_data.get(), 10));
            map.FinalizeVectorIds();

            const size_t offsets[] = {0, 1, 3, 3, 4};
            map.FinalizeEmbListIds(offsets, 4);

            ids_array = map.InToOutIds();
            ebl_ids_array = map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST);
            valid_array = map.ValidBitmap();
            RequireIdArrayEquals(ids_array, {1, 4, 6, 9});
            RequireIdArrayEquals(ebl_ids_array, {1, 4, 4, 9});
            REQUIRE(RequireCompactOutToIn(map, 6) == 2);
            RequireInvalidCompactOutToIn(map, 5);
            REQUIRE(valid_array.size() == 10);

            size_t mmap_file_count = 0;
            bool has_in_to_out_ids_file = false;
            bool has_in_to_out_ebl_ids_file = false;
            bool has_out_to_in_ids_file = false;
            bool has_valid_bitmap_file = false;
            for (const auto& entry : fs::directory_iterator(work_dir)) {
                if (entry.is_regular_file()) {
                    ++mmap_file_count;
                    const auto filename = entry.path().filename().string();
                    has_in_to_out_ids_file = has_in_to_out_ids_file || filename.find("in_to_out_ids.") == 0;
                    has_in_to_out_ebl_ids_file = has_in_to_out_ebl_ids_file || filename.find("in_to_out_ebl_ids.") == 0;
                    has_out_to_in_ids_file = has_out_to_in_ids_file || filename.find("out_to_in_ids.") == 0;
                    has_valid_bitmap_file = has_valid_bitmap_file || filename.find("valid_bitmap.") == 0;
                }
            }
            REQUIRE(mmap_file_count > 0);
            REQUIRE(has_in_to_out_ids_file);
            REQUIRE(has_in_to_out_ebl_ids_file);
            REQUIRE(has_out_to_in_ids_file);
            REQUIRE_FALSE(has_valid_bitmap_file);
        }

        RequireIdArrayEquals(ids_array, {1, 4, 6, 9});
        RequireIdArrayEquals(ebl_ids_array, {1, 4, 4, 9});
        REQUIRE(valid_array.size() == 10);
    }

    SECTION("Sealed all-null IdMap does not materialize mmap id arrays") {
        auto work_dir = NullableIdMapWorkDir("all_null_id_map_mmap");
        RemoveAllNoThrow(work_dir);

        knowhere::IdMapMmapOptions mmap_options;
        mmap_options.enable_in_to_out_ids = true;
        mmap_options.enable_out_to_in_ids = true;
        mmap_options.mmap_dir_path = work_dir.string();

        auto valid_data = MakeValidData(10, {});
        knowhere::IdMap map;
        map.SetType(knowhere::IdMap::Type::SEALED);
        map.ConfigureMmap(mmap_options);
        map.AddFromData(knowhere::IdMapData::FromValidData(valid_data.get(), 10));
        map.FinalizeVectorIds();

        REQUIRE(map.OutCount() == 10);
        REQUIRE(map.InCount() == 0);
        REQUIRE(map.InToOutIds().empty());
        RequireInvalidCompactOutToIn(map, 0);
        REQUIRE_FALSE(map.IsValidOutId(0));

        size_t mmap_file_count = 0;
        for (const auto& entry : fs::directory_iterator(work_dir)) {
            if (entry.is_regular_file()) {
                ++mmap_file_count;
            }
        }
        REQUIRE(mmap_file_count == 0);
    }

    SECTION("BitsetView preserves zero mapped count") {
        knowhere::IdMap map;
        const std::vector<int32_t> ids = {0, 2, 4, 6};
        SetIdMapIds(map, ids, 8);

        std::vector<uint8_t> filter_bits(1, 0);
        filter_bits[1 >> 3] |= static_cast<uint8_t>(1U << (1 & 7));
        knowhere::BitsetView bitset(filter_bits.data(), 8);
        const auto& out_ids = map.InToOutIds();
        bitset.set_out_ids(out_ids, out_ids.size());
        auto valid_bitmap = map.ValidBitmap();
        bitset.count_filtered_bits(0, bitset.num_bits(), valid_bitmap.data());

        REQUIRE(bitset.size() == 4);
        REQUIRE(bitset.count() == 0);
        REQUIRE(bitset.filter_ratio() == Catch::Approx(0.0f));
        REQUIRE(bitset.empty());
        for (size_t i = 0; i < bitset.size(); ++i) {
            REQUIRE_FALSE(bitset.test(i));
        }
    }

    SECTION("BitsetView counts unaligned buffers safely") {
        constexpr size_t nbits = 73;
        std::vector<uint8_t> bit_storage((nbits + 7) / 8 + 3, 0);
        std::vector<uint8_t> valid_storage((nbits + 7) / 8 + 5, 0);
        auto* bits = bit_storage.data() + 1;
        auto* valid = valid_storage.data() + 3;

        size_t expected = 0;
        for (size_t i = 0; i < nbits; ++i) {
            const bool filtered = i % 3 == 0 || i == 71;
            const bool present = i % 5 != 0;
            if (filtered) {
                bits[i >> 3] |= static_cast<uint8_t>(1U << (i & 7));
            }
            if (present) {
                valid[i >> 3] |= static_cast<uint8_t>(1U << (i & 7));
            }
            if (filtered && present) {
                ++expected;
            }
        }

        knowhere::BitsetView bitset(bits, nbits);
        bitset.count_filtered_bits(0, bitset.num_bits(), valid);
        REQUIRE(bitset.count() == expected);

        constexpr size_t bit_offset = 9;
        constexpr size_t bit_count = 51;
        size_t expected_range = 0;
        for (size_t i = bit_offset; i < bit_offset + bit_count; ++i) {
            const bool filtered = i % 3 == 0 || i == 71;
            const bool present = i % 5 != 0;
            if (filtered && present) {
                ++expected_range;
            }
        }
        bitset.count_filtered_bits(bit_offset, bit_count, valid);
        REQUIRE(bitset.count() == expected_range);
    }

    SECTION("BitsetView first valid index ignores padding bits") {
        {
            constexpr size_t nbits = 62;
            std::vector<uint8_t> bits((nbits + 7) / 8, 0xFF);
            bits.back() = 0x7F;
            knowhere::BitsetView bitset(bits.data(), nbits, nbits);
            REQUIRE(bitset.get_first_valid_index() == nbits);
        }
        {
            constexpr size_t nbits = 65;
            std::vector<uint8_t> bits((nbits + 7) / 8, 0xFF);
            bits.back() = 0x03;
            knowhere::BitsetView bitset(bits.data(), nbits, nbits);
            REQUIRE(bitset.get_first_valid_index() == nbits);
        }
    }

    SECTION("Data type conversion preserves row slicing and emb-list layout") {
        auto vector_ds = GenFullDenseDataSet(8, 4);
        vector_ds->SetTensorBeginId(10);
        auto converted_vector = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(vector_ds, 1, 2);
        REQUIRE(converted_vector->GetRows() == 2);
        REQUIRE(converted_vector->GetTensorBeginId() == 11);
        REQUIRE(converted_vector->GetDim() == 4);

        auto emb_ds = GenNullableEmbListDataSet(8, {1, 3, 5}, 4, 2);
        emb_ds->SetTensorBeginId(20);
        auto converted_emb = knowhere::ConvertToDataTypeIfNeeded<knowhere::fp16>(emb_ds);
        REQUIRE(converted_emb->GetRows() == emb_ds->GetRows());
        REQUIRE(converted_emb->GetTensorBeginId() == 20);
        auto offsets = converted_emb->Get<const size_t*>(knowhere::meta::EMB_LIST_OFFSET);
        REQUIRE(offsets != nullptr);
        REQUIRE(offsets[0] == 0);
        REQUIRE(offsets[3] == 6);
    }

    SECTION("IndexNode wrappers forward nullable out id map state") {
        auto data_mock =
            knowhere::IndexNodeDataMockWrapper<knowhere::fp16>(std::make_unique<NullableWrapperFakeIndexNode>());
        RequireIdMapForwarding(data_mock);
        RequireReadApisReturnError(data_mock);

        auto local_pool = std::make_shared<knowhere::ThreadPool>(1, "Knowhere_Nullable_Test");
        {
            auto thread_pool =
                knowhere::IndexNodeThreadPoolWrapper(std::make_unique<NullableWrapperFakeIndexNode>(), local_pool);
            RequireIdMapForwarding(thread_pool);
        }
    }
}

TEST_CASE("Nullable Index normalizes bitset before calling IndexNode", "[nullable][bitset][api]") {
    auto concrete_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(4, kDenseDim);
    auto* node = concrete_index.Node();
    knowhere::IdMap map;
    SetIdMapIds(map, {0, 2, 4, 6}, 8);
    node->GetIdMap() = std::move(map);

    knowhere::Index<knowhere::IndexNode> index(std::move(concrete_index));
    auto query_ds = GenDenseQueryDataSet({0}, kDenseDim);
    auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, 1);

    std::vector<uint8_t> missing_only_bits;
    auto missing_only_bitset = BitsetViewFrom(missing_only_bits, 8, {1});
    REQUIRE_FALSE(missing_only_bitset.empty());
    auto search = index.Search(query_ds, json, missing_only_bitset);
    REQUIRE(search.has_value());
    REQUIRE(node->last_search_bitset.has_out_ids);
    REQUIRE(node->last_search_bitset.size == 4);
    REQUIRE(node->last_search_bitset.num_bits == 8);
    REQUIRE(node->last_search_bitset.count == 0);
    REQUIRE_FALSE(node->last_search_bitset.need_filter);
    REQUIRE(node->last_search_bitset.filtered_in_ids.empty());

    std::vector<uint8_t> valid_bits;
    auto valid_bitset = BitsetViewFrom(valid_bits, 8, {2});
    auto range = index.RangeSearch(query_ds, json, valid_bitset);
    REQUIRE(range.has_value());
    REQUIRE(node->last_range_bitset.has_out_ids);
    REQUIRE(node->last_range_bitset.size == 4);
    REQUIRE(node->last_range_bitset.count == 1);
    REQUIRE(node->last_range_bitset.need_filter);
    REQUIRE(node->last_range_bitset.filtered_in_ids == std::vector<int>{1});

    auto iterator = index.AnnIterator(query_ds, json, valid_bitset);
    REQUIRE(iterator.has_value());
    REQUIRE(node->last_iterator_bitset.has_out_ids);
    REQUIRE(node->last_iterator_bitset.size == 4);
    REQUIRE(node->last_iterator_bitset.count == 1);
    REQUIRE(node->last_iterator_bitset.need_filter);
    REQUIRE(node->last_iterator_bitset.filtered_in_ids == std::vector<int>{1});

    auto emb_concrete_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(6, kDenseDim);
    auto* emb_node = emb_concrete_index.Node();
    emb_node->SetEmbListOffsetForTest({0, 2, 5, 6});
    knowhere::IdMap emb_map;
    SetIdMapIds(emb_map, {0, 1, 2}, 3);
    emb_node->GetIdMap() = std::move(emb_map);
    emb_node->BuildInToOutEblIdsForTest({0, 2, 5, 6});
    knowhere::Index<knowhere::IndexNode> emb_index(std::move(emb_concrete_index));

    std::vector<uint8_t> emb_bits;
    auto emb_bitset = BitsetViewFrom(emb_bits, 3, {1});
    auto emb_search = emb_index.Search(query_ds, json, emb_bitset);
    REQUIRE(emb_search.has_value());
    REQUIRE(emb_node->last_search_bitset.has_out_ids);
    REQUIRE(emb_node->last_search_bitset.size == 6);
    REQUIRE(emb_node->last_search_bitset.num_bits == 3);
    REQUIRE(emb_node->last_search_bitset.count == 3);
    REQUIRE(emb_node->last_search_bitset.need_filter);
    REQUIRE(emb_node->last_search_bitset.filtered_in_ids == std::vector<int>{2, 3, 4});

    auto doc_level_concrete_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(3, kDenseDim);
    auto* doc_level_node = doc_level_concrete_index.Node();
    doc_level_node->SetEmbListOffsetForTest({0, 2, 5, 6});
    knowhere::Index<knowhere::IndexNode> doc_level_index(std::move(doc_level_concrete_index));

    auto doc_level_search = doc_level_index.Search(query_ds, json, emb_bitset);
    REQUIRE(doc_level_search.has_value());
    REQUIRE_FALSE(doc_level_node->last_search_bitset.has_out_ids);
    REQUIRE(doc_level_node->last_search_bitset.size == 6);
    REQUIRE(doc_level_node->last_search_bitset.num_bits == 3);
    REQUIRE(doc_level_node->last_search_bitset.count == 3);
    REQUIRE(doc_level_node->last_search_bitset.need_filter);
    REQUIRE(doc_level_node->last_search_bitset.filtered_in_ids == std::vector<int>{1});
}

TEST_CASE("Nullable growing Index keeps visible prefix as a filter boundary", "[nullable][bitset][growing]") {
    bool first_batch_valid[] = {true, true, true, true};
    bool second_batch_valid[] = {true, false, true, true};

    knowhere::IdMap map;
    map.SetType(knowhere::IdMap::Type::GROWING);
    map.AddFromData(knowhere::IdMapData::FromValidData(first_batch_valid, 4));
    map.AddFromData(knowhere::IdMapData::FromValidData(second_batch_valid, 4));
    map.FinalizeVectorIds();

    auto concrete_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(7, kDenseDim);
    auto* node = concrete_index.Node();
    node->GetIdMap() = std::move(map);
    knowhere::Index<knowhere::IndexNode> index(std::move(concrete_index));

    auto query_ds = GenDenseQueryDataSet({0}, kDenseDim);
    auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, 1);
    std::vector<uint8_t> visible_prefix_bits(1, 0);
    knowhere::BitsetView visible_prefix(visible_prefix_bits.data(), 4);

    auto search = index.Search(query_ds, json, visible_prefix);
    REQUIRE(search.has_value());
    REQUIRE(node->last_search_bitset.has_out_ids);
    REQUIRE(node->last_search_bitset.size == 7);
    REQUIRE(node->last_search_bitset.num_bits == 4);
    REQUIRE(node->last_search_bitset.count == 3);
    REQUIRE(node->last_search_bitset.need_filter);
    REQUIRE(node->last_search_bitset.filtered_in_ids == std::vector<int>{4, 5, 6});
}

TEST_CASE("Nullable Index exact bitset count filters rows outside visible bitset", "[nullable][bitset][api]") {
    auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, 1);
    auto query_ds = GenDenseQueryDataSet({0}, kDenseDim);

    auto non_nullable_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(6, kDenseDim);
    auto* non_nullable_node = non_nullable_index.Node();
    knowhere::Index<knowhere::IndexNode> non_nullable(std::move(non_nullable_index));
    std::vector<uint8_t> non_nullable_bits;
    auto non_nullable_bitset = BitsetViewFrom(non_nullable_bits, 4, {1});
    auto non_nullable_search = non_nullable.Search(query_ds, json, non_nullable_bitset);
    REQUIRE(non_nullable_search.has_value());
    REQUIRE_FALSE(non_nullable_node->last_search_bitset.has_out_ids);
    REQUIRE(non_nullable_node->last_search_bitset.size == 6);
    REQUIRE(non_nullable_node->last_search_bitset.num_bits == 4);
    REQUIRE(non_nullable_node->last_search_bitset.count == 3);

    auto nullable_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(4, kDenseDim);
    auto* nullable_node = nullable_index.Node();
    knowhere::IdMap nullable_map;
    SetIdMapIds(nullable_map, {0, 2, 4, 6}, 7);
    nullable_node->GetIdMap() = std::move(nullable_map);
    knowhere::Index<knowhere::IndexNode> nullable(std::move(nullable_index));
    std::vector<uint8_t> nullable_bits;
    auto nullable_bitset = BitsetViewFrom(nullable_bits, 4, {2});
    auto nullable_search = nullable.Search(query_ds, json, nullable_bitset);
    REQUIRE(nullable_search.has_value());
    REQUIRE(nullable_node->last_search_bitset.has_out_ids);
    REQUIRE(nullable_node->last_search_bitset.size == 4);
    REQUIRE(nullable_node->last_search_bitset.num_bits == 4);
    REQUIRE(nullable_node->last_search_bitset.count == 3);
    REQUIRE(nullable_node->last_search_bitset.filtered_in_ids == std::vector<int>{1, 2, 3});

    const std::vector<size_t> emb_offsets = {0, 2, 5, 6, 8};
    auto emb_non_nullable_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(8, kDenseDim);
    auto* emb_non_nullable_node = emb_non_nullable_index.Node();
    emb_non_nullable_node->SetEmbListOffsetForTest(emb_offsets);
    knowhere::Index<knowhere::IndexNode> emb_non_nullable(std::move(emb_non_nullable_index));
    std::vector<uint8_t> emb_non_nullable_bits;
    auto emb_non_nullable_bitset = BitsetViewFrom(emb_non_nullable_bits, 3, {1});
    auto emb_non_nullable_search = emb_non_nullable.Search(query_ds, json, emb_non_nullable_bitset);
    REQUIRE(emb_non_nullable_search.has_value());
    REQUIRE_FALSE(emb_non_nullable_node->last_search_bitset.has_out_ids);
    REQUIRE(emb_non_nullable_node->last_search_bitset.size == 8);
    REQUIRE(emb_non_nullable_node->last_search_bitset.num_bits == 3);
    REQUIRE(emb_non_nullable_node->last_search_bitset.count == 5);

    const std::vector<size_t> compact_emb_offsets = {0, 2, 5, 6};
    auto emb_nullable_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(6, kDenseDim);
    auto* emb_nullable_node = emb_nullable_index.Node();
    emb_nullable_node->SetEmbListOffsetForTest(compact_emb_offsets);
    knowhere::IdMap emb_nullable_map;
    SetIdMapIds(emb_nullable_map, {0, 2, 4}, 5);
    emb_nullable_node->GetIdMap() = std::move(emb_nullable_map);
    knowhere::Index<knowhere::IndexNode> emb_nullable(std::move(emb_nullable_index));
    std::vector<uint8_t> emb_nullable_bits;
    auto emb_nullable_bitset = BitsetViewFrom(emb_nullable_bits, 3, {2});
    auto emb_nullable_search = emb_nullable.Search(query_ds, json, emb_nullable_bitset);
    REQUIRE(emb_nullable_search.has_value());
    REQUIRE(emb_nullable_node->last_search_bitset.has_out_ids);
    REQUIRE(emb_nullable_node->last_search_bitset.size == 6);
    REQUIRE(emb_nullable_node->last_search_bitset.num_bits == 3);
    REQUIRE(emb_nullable_node->last_search_bitset.count == 4);

    auto tokenann_non_nullable_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(8, kDenseDim);
    auto* tokenann_non_nullable_node = tokenann_non_nullable_index.Node();
    tokenann_non_nullable_node->SetEmbListOffsetForTest(emb_offsets);
    tokenann_non_nullable_node->BuildInToOutEblIdsForTest(emb_offsets);
    knowhere::Index<knowhere::IndexNode> tokenann_non_nullable(std::move(tokenann_non_nullable_index));
    std::vector<uint8_t> tokenann_non_nullable_bits;
    auto tokenann_non_nullable_bitset = BitsetViewFrom(tokenann_non_nullable_bits, 3, {1});
    auto tokenann_non_nullable_search = tokenann_non_nullable.Search(query_ds, json, tokenann_non_nullable_bitset);
    REQUIRE(tokenann_non_nullable_search.has_value());
    REQUIRE(tokenann_non_nullable_node->last_search_bitset.has_out_ids);
    REQUIRE(tokenann_non_nullable_node->last_search_bitset.size == 8);
    REQUIRE(tokenann_non_nullable_node->last_search_bitset.num_bits == 3);
    REQUIRE(tokenann_non_nullable_node->last_search_bitset.count == 5);

    auto tokenann_nullable_index = knowhere::Index<CapturingBitsetFakeIndexNode>::Create(6, kDenseDim);
    auto* tokenann_nullable_node = tokenann_nullable_index.Node();
    tokenann_nullable_node->SetEmbListOffsetForTest(compact_emb_offsets);
    knowhere::IdMap tokenann_nullable_map;
    SetIdMapIds(tokenann_nullable_map, {0, 2, 4}, 5);
    tokenann_nullable_node->GetIdMap() = std::move(tokenann_nullable_map);
    tokenann_nullable_node->BuildInToOutEblIdsForTest(compact_emb_offsets);
    knowhere::Index<knowhere::IndexNode> tokenann_nullable(std::move(tokenann_nullable_index));
    std::vector<uint8_t> tokenann_nullable_bits;
    auto tokenann_nullable_bitset = BitsetViewFrom(tokenann_nullable_bits, 3, {2});
    auto tokenann_nullable_search = tokenann_nullable.Search(query_ds, json, tokenann_nullable_bitset);
    REQUIRE(tokenann_nullable_search.has_value());
    REQUIRE(tokenann_nullable_node->last_search_bitset.has_out_ids);
    REQUIRE(tokenann_nullable_node->last_search_bitset.size == 6);
    REQUIRE(tokenann_nullable_node->last_search_bitset.num_bits == 3);
    REQUIRE(tokenann_nullable_node->last_search_bitset.count == 4);
}

TEST_CASE("Nullable Flat Add adds out id map", "[nullable][flat][add]") {
    auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, 1);
    auto first_batch = GenNullableDenseDataSet(4, {0, 2}, kDenseDim);
    auto second_batch = GenNullableDenseDataSet(4, {4, 6}, kDenseDim);
    auto query = GenDenseQueryDataSet({0, 6}, kDenseDim);
    const int32_t batch_ids[] = {0, 2};
    first_batch->SetIdMapData(knowhere::IdMapData::FromIds(batch_ids, 2, 4));
    second_batch->SetIdMapData(knowhere::IdMapData::FromIds(batch_ids, 2, 4));

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version)
                     .value();
    index.SetIdMapType(knowhere::IdMap::Type::GROWING);
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

TEST_CASE("Nullable IVF_FLAT_CC emb-list Add preserves append public list ids", "[nullable][ivf][emblist][add]") {
    auto json = BaseDenseConfig("MAX_SIM_L2", kDenseDim, 1);
    json[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC;
    json[knowhere::indexparam::NLIST] = 1;
    json[knowhere::indexparam::NPROBE] = 1;
    ApplyEmbListStrategyConfig(json, knowhere::meta::EMB_LIST_STRATEGY_TOKENANN);

    auto first_batch = GenEmbListBatchDataSet({1, 4}, {0, 2, 5}, kDenseDim);
    const int32_t first_ids[] = {1, 4};
    first_batch->SetIdMapData(knowhere::IdMapData::FromIds(first_ids, 2, 6));

    auto append_batch = GenEmbListBatchDataSet({7, 9, 10}, {5, 5, 7, 7}, kDenseDim);
    const int32_t append_ids[] = {1, 3, 4};
    append_batch->SetIdMapData(knowhere::IdMapData::FromIds(append_ids, 3, 5));

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index = knowhere::IndexFactory::Instance()
                     .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, version)
                     .value();
    index.SetIdMapType(knowhere::IdMap::Type::GROWING);
    REQUIRE(index.Build(first_batch, json, false) == Status::success);
    REQUIRE(index.Add(append_batch, json, false) == Status::success);

    RequireIdArrayEquals(index.GetIdMap().InToOutIds(), {1, 4, 7, 9, 10});
    RequireIdArrayEquals(index.GetIdMap().InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {1, 1, 4, 4, 4, 9, 9});

    auto query = GenEmbListQueryDataSet({9}, kDenseDim);
    auto search = index.Search(query, json, knowhere::BitsetView{});
    REQUIRE(search.has_value());
    RequireExactFirstHits(*search.value(), {9}, {1, 4, 9});

    auto filter_bits = MakeBitmap(11, {9});
    auto prepared = index.Node()->PrepareBitset(knowhere::BitsetView(filter_bits.data(), 11));
    REQUIRE(prepared.has_value());
    REQUIRE(prepared.value().bitset.has_out_ids());
    REQUIRE(prepared.value().bitset.size() == 7);
    REQUIRE(prepared.value().bitset.count() == 2);
    REQUIRE(prepared.value().bitset.test(5));
    REQUIRE(prepared.value().bitset.test(6));
    REQUIRE_FALSE(prepared.value().bitset.test(0));

    auto filtered_search = index.Search(query, json, knowhere::BitsetView(filter_bits.data(), 11));
    REQUIRE(filtered_search.has_value());
    RequireExpectedVectorResult(filtered_search, {1, 4}, {9}, false);

    auto tail_empty_filter_bits = MakeBitmap(11, {10});
    auto tail_empty_prepared = index.Node()->PrepareBitset(knowhere::BitsetView(tail_empty_filter_bits.data(), 11));
    REQUIRE(tail_empty_prepared.has_value());
    REQUIRE(tail_empty_prepared.value().bitset.has_out_ids());
    REQUIRE(tail_empty_prepared.value().bitset.count() == 0);

    int64_t retrieve_ids[] = {1, 7, 9, 10};
    auto retrieved = index.GetEmbListByIds(knowhere::GenIdsDataSet(4, retrieve_ids), "MAX_SIM_L2");
    REQUIRE(retrieved.has_value());
    RequireEmbListRetrieveMatches(*retrieved.value(), {1, 7, 9, 10}, {2, 0, 2, 0}, kDenseDim);

    int64_t missing_id = 3;
    auto missing = index.GetEmbListByIds(knowhere::GenIdsDataSet(1, &missing_id), "MAX_SIM_L2");
    REQUIRE(missing.has_value());
    RequireEmbListRetrieveMatches(*missing.value(), {}, {}, kDenseDim);
}

TEST_CASE("Nullable Index rejects plain and mapped Add mixing", "[nullable][id_map][add]") {
    auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, 1);
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    const int32_t batch_ids[] = {0, 2};
    auto create_flat_index = [&] {
        return knowhere::IndexFactory::Instance()
            .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IDMAP, version)
            .value();
    };

    SECTION("DISABLED rejects IdMapData") {
        auto batch = GenNullableDenseDataSet(4, {0, 2}, kDenseDim);
        batch->SetIdMapData(knowhere::IdMapData::FromIds(batch_ids, 2, 4));

        auto created = create_flat_index();
        REQUIRE(created.Build(batch, json) == Status::invalid_args);

        auto add_index = create_flat_index();
        REQUIRE(add_index.Train(batch, json) == Status::success);
        REQUIRE(add_index.Add(batch, json) == Status::invalid_args);
    }

    SECTION("enabled IdMap requires IdMapData") {
        auto plain_batch = GenNullableDenseDataSet(4, {0, 1, 2, 3}, kDenseDim);
        for (auto type : {knowhere::IdMap::Type::SEALED, knowhere::IdMap::Type::GROWING}) {
            CAPTURE(static_cast<int>(type));
            auto build_index = create_flat_index();
            build_index.SetIdMapType(type);
            REQUIRE(build_index.Build(plain_batch, json) == Status::invalid_args);
        }

        auto add_index = create_flat_index();
        add_index.SetIdMapType(knowhere::IdMap::Type::GROWING);
        REQUIRE(add_index.Train(plain_batch, json) == Status::success);
        REQUIRE(add_index.Add(plain_batch, json) == Status::invalid_args);
    }

    SECTION("mapped index rejects a plain append batch") {
        auto first_batch = GenNullableDenseDataSet(4, {0, 2}, kDenseDim);
        first_batch->SetIdMapData(knowhere::IdMapData::FromIds(batch_ids, 2, 4));
        auto plain_second_batch = GenNullableDenseDataSet(4, {4, 6}, kDenseDim);

        auto index = create_flat_index();
        index.SetIdMapType(knowhere::IdMap::Type::GROWING);
        REQUIRE(index.Train(first_batch, json) == Status::success);
        REQUIRE(index.Add(first_batch, json) == Status::success);
        REQUIRE(index.Add(plain_second_batch, json) == Status::invalid_args);
    }

    SECTION("all-valid nullable append still requires IdMapData") {
        const int32_t all_valid_ids[] = {0, 1};
        auto first_batch = GenNullableDenseDataSet(2, {0, 1}, kDenseDim);
        first_batch->SetIdMapData(knowhere::IdMapData::FromIds(all_valid_ids, 2, 2));
        auto all_valid_plain_append = GenNullableDenseDataSet(2, {2, 3}, kDenseDim);

        auto index = create_flat_index();
        index.SetIdMapType(knowhere::IdMap::Type::GROWING);
        REQUIRE(index.Train(first_batch, json) == Status::success);
        REQUIRE(index.Add(first_batch, json) == Status::success);
        REQUIRE(index.Add(all_valid_plain_append, json) == Status::invalid_args);
        RequireIdMapVectorStateOnly(index.GetIdMap(), 2, {0, 1});
    }

    SECTION("plain index rejects a mapped append batch") {
        auto plain_first_batch = GenNullableDenseDataSet(4, {0, 2}, kDenseDim);
        auto second_batch = GenNullableDenseDataSet(4, {4, 6}, kDenseDim);
        second_batch->SetIdMapData(knowhere::IdMapData::FromIds(batch_ids, 2, 4));

        auto index = create_flat_index();
        REQUIRE(index.Train(plain_first_batch, json) == Status::success);
        REQUIRE(index.Add(plain_first_batch, json) == Status::success);
        index.SetIdMapType(knowhere::IdMap::Type::GROWING);
        REQUIRE(index.Add(second_batch, json) == Status::invalid_args);
    }
}

TEST_CASE("Nullable vector Build plus Add matches full Build", "[nullable][ivf][add]") {
    auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, 2);
    json[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC;
    json[knowhere::indexparam::NLIST] = 1;
    json[knowhere::indexparam::NPROBE] = 1;

    const std::vector<int32_t> full_ids = {1, 4, 7, 9};
    const int32_t first_ids[] = {1, 4};
    const int32_t append_ids[] = {1, 3};

    auto full_ds = GenNullableDenseDataSet(11, full_ids, kDenseDim);
    full_ds->SetIdMapData(knowhere::IdMapData::FromIds(full_ids.data(), full_ids.size(), 11));

    auto first_ds = GenNullableDenseDataSet(6, {1, 4}, kDenseDim);
    first_ds->SetIdMapData(knowhere::IdMapData::FromIds(first_ids, 2, 6));
    auto append_ds = GenNullableDenseDataSet(5, {7, 9}, kDenseDim);
    append_ds->SetIdMapData(knowhere::IdMapData::FromIds(append_ids, 2, 5));

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto full_index = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, version)
                          .value();
    full_index.SetIdMapType(knowhere::IdMap::Type::SEALED);
    REQUIRE(full_index.Build(full_ds, json, false) == Status::success);

    auto split_index = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, version)
                           .value();
    split_index.SetIdMapType(knowhere::IdMap::Type::GROWING);
    REQUIRE(split_index.Build(first_ds, json, false) == Status::success);
    REQUIRE(split_index.Add(append_ds, json, false) == Status::success);

    RequireIdMapVectorStateOnly(full_index.GetIdMap(), 11, full_ids);
    RequireIdMapVectorStateOnly(split_index.GetIdMap(), 11, full_ids);

    auto query = GenDenseQueryDataSet({1, 9}, kDenseDim);
    auto filter_bits = MakeBitmap(11, {4});
    knowhere::BitsetView bitset(filter_bits.data(), 11);

    auto full_search = full_index.Search(query, json, bitset);
    auto split_search = split_index.Search(query, json, bitset);
    REQUIRE(full_search.has_value());
    REQUIRE(split_search.has_value());
    RequireDatasetsEqual(*full_search.value(), *split_search.value());

    auto range_json = json;
    range_json[knowhere::meta::RADIUS] = 100000000.0f;
    auto full_range = full_index.RangeSearch(query, range_json, bitset);
    auto split_range = split_index.RangeSearch(query, range_json, bitset);
    REQUIRE(full_range.has_value());
    REQUIRE(split_range.has_value());
    RequireDatasetsEqual(*full_range.value(), *split_range.value());

    RequireIteratorFirstResultsEqual(full_index, split_index, query, json, bitset);

    int64_t selected_ids[] = {1, 9};
    auto full_vectors = full_index.GetVectorByIds(knowhere::GenIdsDataSet(2, selected_ids));
    auto split_vectors = split_index.GetVectorByIds(knowhere::GenIdsDataSet(2, selected_ids));
    REQUIRE(full_vectors.has_value());
    REQUIRE(split_vectors.has_value());
    RequireDenseVectorsMatchLogicalIds(*full_vectors.value(), {1, 9}, kDenseDim);
    RequireDenseVectorsMatchLogicalIds(*split_vectors.value(), {1, 9}, kDenseDim);

    int64_t selected_public_ids[] = {1, 9};
    auto full_dist = full_index.CalcDistByIDs(query, knowhere::BitsetView{}, selected_public_ids, 2, false);
    auto split_dist = split_index.CalcDistByIDs(query, knowhere::BitsetView{}, selected_public_ids, 2, false);
    REQUIRE(full_dist.has_value());
    REQUIRE(split_dist.has_value());
    RequireDatasetsEqual(*full_dist.value(), *split_dist.value());
}

TEST_CASE("Nullable emb-list Build plus Add matches full Build with empty lists", "[nullable][ivf][emblist][add]") {
    auto json = BaseDenseConfig("MAX_SIM_L2", kDenseDim, 2);
    json[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC;
    json[knowhere::indexparam::NLIST] = 1;
    json[knowhere::indexparam::NPROBE] = 1;
    ApplyEmbListStrategyConfig(json, knowhere::meta::EMB_LIST_STRATEGY_TOKENANN);

    const int32_t full_ids[] = {1, 4, 7, 9, 10};
    const int32_t first_ids[] = {1, 4};
    const int32_t append_ids[] = {1, 3, 4};

    auto full_ds = GenEmbListBatchDataSet({1, 4, 7, 9, 10}, {0, 2, 5, 5, 7, 7}, kDenseDim);
    full_ds->SetIdMapData(knowhere::IdMapData::FromIds(full_ids, 5, 11));
    auto first_ds = GenEmbListBatchDataSet({1, 4}, {0, 2, 5}, kDenseDim);
    first_ds->SetIdMapData(knowhere::IdMapData::FromIds(first_ids, 2, 6));
    auto append_ds = GenEmbListBatchDataSet({7, 9, 10}, {5, 5, 7, 7}, kDenseDim);
    append_ds->SetIdMapData(knowhere::IdMapData::FromIds(append_ids, 3, 5));

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto full_index = knowhere::IndexFactory::Instance()
                          .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, version)
                          .value();
    full_index.SetIdMapType(knowhere::IdMap::Type::SEALED);
    REQUIRE(full_index.Build(full_ds, json, false) == Status::success);

    auto split_index = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, version)
                           .value();
    split_index.SetIdMapType(knowhere::IdMap::Type::GROWING);
    REQUIRE(split_index.Build(first_ds, json, false) == Status::success);
    REQUIRE(split_index.Add(append_ds, json, false) == Status::success);

    RequireIdMapVectorState(full_index.GetIdMap(), 11, {1, 4, 7, 9, 10});
    RequireIdMapVectorState(split_index.GetIdMap(), 11, {1, 4, 7, 9, 10});
    RequireIdArrayEquals(full_index.GetIdMap().InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {1, 1, 4, 4, 4, 9, 9});
    RequireIdArrayEquals(split_index.GetIdMap().InToOutIds(knowhere::IdMap::Domain::EMB_LIST), {1, 1, 4, 4, 4, 9, 9});

    auto query = GenEmbListQueryDataSet({9, 1}, kDenseDim);
    auto filter_bits = MakeBitmap(11, {9});
    knowhere::BitsetView bitset(filter_bits.data(), 11);
    auto full_search = full_index.Search(query, json, bitset);
    auto split_search = split_index.Search(query, json, bitset);
    REQUIRE(full_search.has_value());
    REQUIRE(split_search.has_value());
    RequireDatasetsEqual(*full_search.value(), *split_search.value());
    RequireExpectedVectorResult(split_search, {1, 4}, {9}, false);

    int64_t retrieve_ids[] = {1, 7, 9, 10};
    auto full_retrieve = full_index.GetEmbListByIds(knowhere::GenIdsDataSet(4, retrieve_ids), "MAX_SIM_L2");
    auto split_retrieve = split_index.GetEmbListByIds(knowhere::GenIdsDataSet(4, retrieve_ids), "MAX_SIM_L2");
    REQUIRE(full_retrieve.has_value());
    REQUIRE(split_retrieve.has_value());
    RequireEmbListRetrieveMatches(*full_retrieve.value(), {1, 7, 9, 10}, {2, 0, 2, 0}, kDenseDim);
    RequireEmbListRetrieveMatches(*split_retrieve.value(), {1, 7, 9, 10}, {2, 0, 2, 0}, kDenseDim);
}

TEST_CASE("Nullable retrieve compacts missing public ids", "[nullable][id_map][api]") {
    struct SelectedIdCase {
        std::string label;
        IndexRow row;
        bool maybe_unavailable = false;
    };

    auto diskann = NativeDiskAnnRow();
    diskann.maybe_unavailable = true;

    std::vector<SelectedIdCase> cases = {
        {"FLAT", {"FLAT", knowhere::IndexEnum::INDEX_FAISS_IDMAP, DataKind::DenseFp32}, false},
        {"IVF_FLAT_CC",
         {"IVF_FLAT_CC / IVFFLATCC", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, DataKind::DenseFp32},
         false},
        {"HNSW", NativeFaissHnswRow(), true},
        {"DISKANN", diskann, true},
        {"SPARSE_CC",
         {"SPARSE_INVERTED_INDEX_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
          DataKind::Sparse},
         false},
    };

    for (const auto& test_case : cases) {
        CAPTURE(test_case.label);
        Scenario scenario;
        scenario.mode = Mode::Vector;
        scenario.nullable_ratio = NullableRatio::R50;

        auto artifact = BuildArtifactForScenario(test_case.row, scenario);
        if (!artifact->create_ok || artifact->build_status != Status::success) {
            const bool allowed_unavailable = test_case.maybe_unavailable || test_case.row.maybe_unavailable;
            REQUIRE(allowed_unavailable);
            continue;
        }

        const auto missing_id = FirstMissingOutId(artifact->data.valid_ids, artifact->data.total_count);
        std::vector<int64_t> rejected_ids = {-1, artifact->data.total_count};
        if (missing_id < artifact->data.total_count) {
            rejected_ids.push_back(missing_id);
        }

        for (auto rejected_id : rejected_ids) {
            CAPTURE(rejected_id);
            auto get_vector = artifact->index.GetVectorByIds(knowhere::GenIdsDataSet(1, &rejected_id));
            REQUIRE(get_vector.has_value());
            REQUIRE(get_vector.value()->GetRows() == 0);
        }

        auto null_get_vector =
            artifact->index.GetVectorByIds(knowhere::GenIdsDataSet(1, static_cast<const int64_t*>(nullptr)));
        REQUIRE_FALSE(null_get_vector.has_value());
        REQUIRE(null_get_vector.error() == Status::invalid_args);

        if (missing_id < artifact->data.total_count && artifact->data.query_ids.size() >= 2) {
            std::vector<int64_t> retrieve_ids = {artifact->data.query_ids[0], missing_id, artifact->data.query_ids[1]};
            auto get_vector = artifact->index.GetVectorByIds(
                knowhere::GenIdsDataSet(static_cast<int64_t>(retrieve_ids.size()), retrieve_ids.data()));
            if (!get_vector.has_value()) {
                REQUIRE((test_case.maybe_unavailable || test_case.row.maybe_unavailable));
                continue;
            }
            REQUIRE(get_vector.has_value());
            REQUIRE(get_vector.value()->GetRows() == 2);
        }
    }
}

TEST_CASE("Nullable Index Add keeps id map append when backend add fails", "[nullable][id_map][add]") {
    auto concrete_index = knowhere::Index<FailingAddFakeIndexNode>::Create();
    auto* node = concrete_index.Node();
    knowhere::IdMap map;
    map.SetType(knowhere::IdMap::Type::GROWING);
    const int32_t base_ids[] = {1, 3};
    map.AddFromData(knowhere::IdMapData::FromIds(base_ids, 2, 4));
    node->GetIdMap() = std::move(map);

    knowhere::Index<knowhere::IndexNode> index(std::move(concrete_index));
    auto append_batch = GenNullableDenseDataSet(4, {4, 6}, kDenseDim);
    const int32_t append_ids[] = {0, 2};
    append_batch->SetIdMapData(knowhere::IdMapData::FromIds(append_ids, 2, 4));

    auto json = BaseDenseConfig(knowhere::metric::L2, kDenseDim, 1);
    REQUIRE(index.Add(append_batch, json) == Status::invalid_args);

    REQUIRE(index.GetIdMap().OutCount() == 8);
    RequireIdArrayEquals(index.GetIdMap().InToOutIds(), {1, 3, 4, 6});
    REQUIRE(RequireCompactOutToIn(index.GetIdMap(), 1) == 0);
    REQUIRE(RequireCompactOutToIn(index.GetIdMap(), 3) == 1);
    REQUIRE(RequireCompactOutToIn(index.GetIdMap(), 4) == 2);
    REQUIRE(RequireCompactOutToIn(index.GetIdMap(), 6) == 3);
}

TEST_CASE("Nullable DataView full-scan uses internal source ids", "[nullable][data_view][api]") {
    constexpr int64_t total_count = 64;
    const std::vector<int32_t> valid_ids = {20, 22, 24, 26, 28, 30, 32, 34};
    const std::vector<int32_t> query_ids = {20, 26};
    const std::vector<knowhere::idx_t> labels = {0, 2, 7};

    std::vector<float> source(valid_ids.size() * kDenseDim, -1000.0f);
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        FillDenseVector(source.data(), static_cast<int64_t>(i), kDenseDim, valid_ids[i]);
    }

    auto train_ds = GenNullableDenseDataSet(total_count, valid_ids, kDenseDim);
    auto query_ds = GenDenseQueryDataSet(query_ids, kDenseDim);

    knowhere::IdMap map;
    SetIdMapIds(map, valid_ids, total_count);
    knowhere::ViewDataOp view_data = [&](size_t in_id) -> const void* {
        if (in_id >= valid_ids.size()) {
            return source.data();
        }
        return source.data() + in_id * kDenseDim;
    };

    knowhere::DataViewIndexFlat refiner(kDenseDim, knowhere::DataFormatEnum::fp32, knowhere::metric::L2, view_data,
                                        false, knowhere::DATA_VIEW, std::nullopt);
    refiner.Train(train_ds->GetRows(), train_ds->GetTensor(), false);
    refiner.Add(train_ds->GetRows(), train_ds->GetTensor(), nullptr, false);

    std::vector<float> calc_dist(query_ids.size() * labels.size());
    refiner.CalcDistByIDs(query_ids.size(), query_ds->GetTensor(), labels.size(), labels.data(), calc_dist.data(),
                          false);
    for (size_t i = 0; i < query_ids.size(); ++i) {
        for (size_t j = 0; j < labels.size(); ++j) {
            CAPTURE(i, j, query_ids[i], labels[j]);
            REQUIRE(calc_dist[i * labels.size() + j] ==
                    Catch::Approx(DenseL2ForLogicalIds(query_ids[i], valid_ids[labels[j]], kDenseDim)));
        }
    }

    std::vector<float> distances(query_ids.size() * kTopK, -1.0f);
    std::vector<knowhere::idx_t> result_ids(query_ids.size() * kTopK, -1);
    refiner.Search(query_ids.size(), query_ds->GetTensor(), kTopK, distances.data(), result_ids.data(),
                   knowhere::BitsetView{}, nullptr, false);
    for (size_t i = 0; i < query_ids.size(); ++i) {
        CAPTURE(i, query_ids[i]);
        REQUIRE(result_ids[i * kTopK] == static_cast<knowhere::idx_t>(i == 0 ? 0 : 3));
        REQUIRE(distances[i * kTopK] == Catch::Approx(0.0f));
    }

    std::vector<uint8_t> filtered_bits;
    auto prepared_bitset = NormalizedBitsetFrom(filtered_bits, map, total_count, {query_ids[0]});
    auto& bitset = prepared_bitset.bitset;
    REQUIRE(bitset.has_out_ids());
    REQUIRE_FALSE(bitset.empty());
    REQUIRE(bitset.count() == 1);
    std::fill(distances.begin(), distances.end(), -1.0f);
    std::fill(result_ids.begin(), result_ids.end(), -1);
    refiner.Search(query_ids.size(), query_ds->GetTensor(), kTopK, distances.data(), result_ids.data(), bitset, nullptr,
                   false);
    REQUIRE(result_ids[0] != 0);
    REQUIRE(result_ids[kTopK] == 3);

    auto range = refiner.RangeSearch(query_ids.size(), query_ds->GetTensor(), 0.5f, 0.0f, knowhere::BitsetView{},
                                     nullptr, false);
    REQUIRE(range.lims != nullptr);
    REQUIRE(range.labels != nullptr);
    REQUIRE(range.distances != nullptr);
    for (size_t i = 0; i < query_ids.size(); ++i) {
        CAPTURE(i, query_ids[i]);
        REQUIRE(range.lims[i + 1] > range.lims[i]);
        REQUIRE(range.labels[range.lims[i]] == static_cast<knowhere::idx_t>(i == 0 ? 0 : 3));
        REQUIRE(range.distances[range.lims[i]] == Catch::Approx(0.0f));
    }

    auto filtered_range =
        refiner.RangeSearch(query_ids.size(), query_ds->GetTensor(), 0.5f, 0.0f, bitset, nullptr, false);
    REQUIRE(filtered_range.lims[1] == 0);
    REQUIRE(filtered_range.lims[2] > filtered_range.lims[1]);
    REQUIRE(filtered_range.labels[filtered_range.lims[1]] == 3);
    REQUIRE(filtered_range.distances[filtered_range.lims[1]] == Catch::Approx(0.0f));
}

TEST_CASE("Nullable DataView selected-id APIs use internal source ids", "[nullable][data_view][api]") {
    constexpr int64_t total_count = 64;
    const std::vector<int32_t> valid_ids = {20, 22, 24, 26, 28, 30, 32, 34};
    const std::vector<int32_t> query_ids = {20, 26};

    std::vector<float> source(valid_ids.size() * kDenseDim, -1000.0f);
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        FillDenseVector(source.data(), static_cast<int64_t>(i), kDenseDim, valid_ids[i]);
    }

    auto train_ds = GenNullableDenseDataSet(total_count, valid_ids, kDenseDim);
    auto query_ds = GenDenseQueryDataSet(query_ids, kDenseDim);

    knowhere::IdMap map;
    SetIdMapIds(map, valid_ids, total_count);
    knowhere::ViewDataOp view_data = [&](size_t in_id) -> const void* {
        if (in_id >= valid_ids.size()) {
            return source.data();
        }
        return source.data() + in_id * kDenseDim;
    };

    knowhere::DataViewIndexFlat refiner(kDenseDim, knowhere::DataFormatEnum::fp32, knowhere::metric::L2, view_data,
                                        false, knowhere::DATA_VIEW, std::nullopt);
    refiner.Train(train_ds->GetRows(), train_ds->GetTensor(), false);
    refiner.Add(train_ds->GetRows(), train_ds->GetTensor(), nullptr, false);

    const std::vector<knowhere::idx_t> ids_num_lims = {0, 4, 8};
    const std::vector<knowhere::idx_t> ids = {7, 2, 0, 5, 0, 1, 3, 7};
    constexpr knowhere::idx_t selected_topk = 2;
    std::vector<float> distances(query_ids.size() * selected_topk, -1.0f);
    std::vector<knowhere::idx_t> result_ids(query_ids.size() * selected_topk, -1);

    refiner.SearchWithIds(query_ids.size(), query_ds->GetTensor(), ids_num_lims.data(), ids.data(), selected_topk,
                          distances.data(), result_ids.data(), false);
    REQUIRE(result_ids[0] == 0);
    REQUIRE(distances[0] == Catch::Approx(0.0f));
    REQUIRE(result_ids[selected_topk] == 3);
    REQUIRE(distances[selected_topk] == Catch::Approx(0.0f));

    auto range = refiner.RangeSearchWithIds(query_ids.size(), query_ds->GetTensor(), ids_num_lims.data(), ids.data(),
                                            0.5f, 0.0f, false);
    REQUIRE(range.lims != nullptr);
    REQUIRE(range.labels != nullptr);
    REQUIRE(range.distances != nullptr);
    for (size_t i = 0; i < query_ids.size(); ++i) {
        CAPTURE(i, query_ids[i]);
        REQUIRE(range.lims[i + 1] == range.lims[i] + 1);
        REQUIRE(range.labels[range.lims[i]] == static_cast<knowhere::idx_t>(i == 0 ? 0 : 3));
        REQUIRE(range.distances[range.lims[i]] == Catch::Approx(0.0f));
    }
}

TEST_CASE("Nullable DataView cosine uses internal source ids and internal norms", "[nullable][data_view][api]") {
    constexpr int64_t total_count = 64;
    const std::vector<int32_t> valid_ids = {20, 22, 24, 26, 28, 30, 32, 34};
    const std::vector<int32_t> query_ids = {20, 26};
    const std::vector<knowhere::idx_t> labels = {0, 2, 3};

    auto fill_cosine_vector = [](float* dst, int32_t logical_id) {
        std::fill(dst, dst + kDenseDim, 0.0f);
        dst[logical_id % kDenseDim] = 1.0f;
    };

    std::vector<float> source(valid_ids.size() * kDenseDim, 0.0f);
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        fill_cosine_vector(source.data() + i * kDenseDim, valid_ids[i]);
    }

    auto train_data = std::make_unique<float[]>(valid_ids.size() * kDenseDim);
    for (size_t i = 0; i < valid_ids.size(); ++i) {
        fill_cosine_vector(train_data.get() + i * kDenseDim, valid_ids[i]);
    }
    auto train_ds = knowhere::GenDataSet(static_cast<int64_t>(valid_ids.size()), kDenseDim, train_data.release());
    train_ds->SetIsOwner(true);

    auto query_data = std::make_unique<float[]>(query_ids.size() * kDenseDim);
    for (size_t i = 0; i < query_ids.size(); ++i) {
        fill_cosine_vector(query_data.get() + i * kDenseDim, query_ids[i]);
    }
    auto query_ds = knowhere::GenDataSet(static_cast<int64_t>(query_ids.size()), kDenseDim, query_data.release());
    query_ds->SetIsOwner(true);

    knowhere::IdMap map;
    SetIdMapIds(map, valid_ids, total_count);
    knowhere::ViewDataOp view_data = [&](size_t in_id) -> const void* {
        if (in_id >= valid_ids.size()) {
            return source.data();
        }
        return source.data() + in_id * kDenseDim;
    };

    knowhere::DataViewIndexFlat refiner(kDenseDim, knowhere::DataFormatEnum::fp32, knowhere::metric::IP, view_data,
                                        true, knowhere::DATA_VIEW, std::nullopt);
    refiner.Train(train_ds->GetRows(), train_ds->GetTensor(), false);
    refiner.Add(train_ds->GetRows(), train_ds->GetTensor(), nullptr, false);

    std::vector<float> calc_dist(query_ids.size() * labels.size(), 0.0f);
    refiner.CalcDistByIDs(query_ids.size(), query_ds->GetTensor(), labels.size(), labels.data(), calc_dist.data(),
                          false);
    REQUIRE(calc_dist[0] == Catch::Approx(1.0f).epsilon(0.0001f));
    REQUIRE(calc_dist[labels.size() + 2] == Catch::Approx(1.0f).epsilon(0.0001f));

    std::vector<float> distances(query_ids.size(), -1.0f);
    std::vector<knowhere::idx_t> result_ids(query_ids.size(), -1);
    refiner.Search(query_ids.size(), query_ds->GetTensor(), 1, distances.data(), result_ids.data(),
                   knowhere::BitsetView{}, nullptr, false);
    REQUIRE(result_ids[0] == 0);
    REQUIRE(distances[0] == Catch::Approx(1.0f).epsilon(0.0001f));
    REQUIRE(result_ids[1] == 3);
    REQUIRE(distances[1] == Catch::Approx(1.0f).epsilon(0.0001f));
}

TEST_CASE("Nullable SCANN_DVR emb-list cosine rerank maps calc-dist ids", "[nullable][scann_dvr][emblist]") {
    IndexRow row{"SCANN_DVR", knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::EmbList;
    scenario.nullable_ratio = NullableRatio::R50;

    auto work_dir = NullableIdMapWorkDir("scann_dvr_calc_dist");
    ScopedWorkDir work_dir_guard(work_dir);
    auto data = BuildMatrixData(row, scenario, work_dir);
    auto json = DenseConfigFor(row, Mode::EmbList);
    json[knowhere::meta::METRIC_TYPE] = "MAX_SIM_COSINE";

    auto created = CreateIndex(row, data);
    REQUIRE(created.ok);
    created.index.SetIdMapType(knowhere::IdMap::Type::SEALED);
    REQUIRE(AttachRuntimeIdMapData(data.train_ds, data.valid_ids, data.total_count) == Status::success);
    REQUIRE(created.index.Build(data.train_ds, json) == Status::success);

    auto result = created.index.Search(data.query_ds, json, knowhere::BitsetView{});
    REQUIRE(result.has_value());
    RequireExpectedVectorResult(result, data.valid_ids, {}, false);

    const auto* ids = result.value()->GetIds();
    const auto* distances = result.value()->GetDistance();
    REQUIRE(ids != nullptr);
    REQUIRE(distances != nullptr);
    const auto rows = result.value()->GetRows();
    const auto k = result.value()->GetDim();
    std::vector<int64_t> labels;
    labels.reserve(static_cast<size_t>(rows * k));
    for (int64_t i = 0; i < rows * k; ++i) {
        if (ids[i] >= 0) {
            REQUIRE(ContainsId(data.valid_ids, ids[i]));
            labels.push_back(ids[i]);
        }
    }
    REQUIRE_FALSE(labels.empty());
    auto calc_dist =
        created.index.CalcDistByIDs(data.query_ds, knowhere::BitsetView{}, labels.data(), labels.size(), true);
    REQUIRE(calc_dist.has_value());
    const auto* calc_distances = calc_dist.value()->GetDistance();
    REQUIRE(calc_distances != nullptr);
    size_t label_pos = 0;
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            if (ids[i * k + j] < 0) {
                continue;
            }
            CAPTURE(i, j, ids[i * k + j]);
            REQUIRE(distances[i * k + j] ==
                    Catch::Approx(calc_distances[i * labels.size() + label_pos]).epsilon(0.0001f));
            ++label_pos;
        }
    }

    std::vector<int64_t> storage_labels;
    storage_labels.reserve(labels.size());
    for (auto public_id : labels) {
        storage_labels.push_back(CompactIdForLogicalId(data.valid_ids, public_id));
    }
    REQUIRE(created.index.Node() != nullptr);
    auto storage_calc = created.index.Node()->CalcDistByStorageIds(data.query_ds, knowhere::BitsetView{},
                                                                   storage_labels.data(), storage_labels.size(), true);
    REQUIRE(storage_calc.has_value());
}

TEST_CASE("Nullable SCANN_DVR calc-dist APIs keep public and storage ids separate", "[nullable][scann_dvr][api]") {
    IndexRow row{"SCANN_DVR", knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    auto artifact = BuildArtifactForScenario(row, scenario);
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);
    RequireDenseCalcDistByPublicIds(artifact->index, artifact->data);
    REQUIRE(artifact->index.Node() != nullptr);
    RequireDenseCalcDistByStorageIds(*artifact->index.Node(), artifact->data);
}

TEST_CASE("Nullable IVF_FLAT vector APIs use restored logical-id map after reload", "[nullable][ivf][api]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    auto artifact = BuildArtifactForScenario(row, scenario);
    CAPTURE(artifact->json.dump());
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);

    REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
    RequireDenseVectorApisUseOutIds(artifact->binary_loaded, artifact->data, artifact->json);

    REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
    RequireDenseVectorApisUseOutIds(artifact->file_loaded, artifact->data, artifact->json);
}

TEST_CASE("Nullable FAISS HNSW vector APIs use restored logical-id map after reload", "[nullable][hnsw][api]") {
    IndexRow row = NativeFaissHnswRow();
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    auto artifact = BuildArtifactForScenario(row, scenario);
    CAPTURE(artifact->json.dump());
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);

    auto require_vector_apis = [&](const knowhere::Index<knowhere::IndexNode>& index, bool require_ordered_ids) {
        if (require_ordered_ids) {
            RequireIdMapContent(index, artifact->data.valid_ids, artifact->data.total_count);
        } else {
            RequireIdMapBitmap(index, artifact->data.total_count);
        }

        std::vector<int64_t> ids(artifact->data.query_ids.begin(), artifact->data.query_ids.end());
        auto retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()));
        REQUIRE(retrieve.has_value());
        RequireDenseVectorsMatchLogicalIds(*retrieve.value(), artifact->data.query_ids, artifact->data.dim);

        RequireDenseCalcDistByPublicIds(index, artifact->data);
    };

    require_vector_apis(artifact->index, true);

    REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
    require_vector_apis(artifact->binary_loaded, false);

    REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
    require_vector_apis(artifact->file_loaded, false);
}

TEST_CASE("Nullable IdMap state is reconstructed across serialize reload", "[nullable][ivf][api]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    auto artifact = BuildArtifactForScenario(row, scenario);
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);
    RequireIdMapContent(artifact->index, artifact->data.valid_ids, artifact->data.total_count);

    REQUIRE(EnsureSerialized(*artifact) == knowhere::Status::success);
    REQUIRE(artifact->binset.GetByName("EXTERNAL_ID_MAP") == nullptr);

    auto loaded_created = CreateIndex(row, artifact->data, artifact->file_manager);
    REQUIRE(loaded_created.ok);
    REQUIRE(SetRuntimeValidBitmap(loaded_created.index, artifact->data) == knowhere::Status::success);
    REQUIRE(loaded_created.index.Deserialize(artifact->binset, artifact->json) == knowhere::Status::success);
    RequireIdMapContent(loaded_created.index, artifact->data.valid_ids, artifact->data.total_count);
}

TEST_CASE("Nullable loaded indexes use restored external-id maps for filters", "[nullable][filter][reload]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    auto artifact = BuildArtifactForScenario(row, scenario);
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);
    REQUIRE(artifact->data.query_ids.size() >= 2);

    auto filtered_ids = std::vector<int32_t>{artifact->data.query_ids[0]};
    auto allowed_ids = AllowedIdsAfterFilter(artifact->data.valid_ids, filtered_ids);
    std::vector<uint8_t> bitset_data;
    auto bitset = BitsetViewFrom(bitset_data, artifact->data.total_count, filtered_ids);
    auto exact_range_json = artifact->json;
    exact_range_json[knowhere::meta::RADIUS] = 0.5f;

    auto require_filtered_reads = [&](const knowhere::Index<knowhere::IndexNode>& index) {
        auto search = index.Search(artifact->data.query_ds, artifact->json, bitset);
        REQUIRE(search.has_value());
        RequireExpectedVectorResult(search, allowed_ids, filtered_ids, false);
        REQUIRE(search.value()->GetIds()[0] != artifact->data.query_ids[0]);
        REQUIRE(search.value()->GetIds()[kTopK] == artifact->data.query_ids[1]);
        REQUIRE(search.value()->GetDistance()[kTopK] == Catch::Approx(0.0f));

        auto range = index.RangeSearch(artifact->data.query_ds, exact_range_json, bitset);
        REQUIRE(range.has_value());
        RequireExpectedRangeResult(range, allowed_ids, filtered_ids, false);
        REQUIRE(range.value()->GetLims()[1] == 0);
        REQUIRE(range.value()->GetLims()[2] > range.value()->GetLims()[1]);
        REQUIRE(range.value()->GetIds()[range.value()->GetLims()[1]] == artifact->data.query_ids[1]);

        auto iterators = index.AnnIterator(artifact->data.query_ds, artifact->json, bitset);
        REQUIRE(iterators.has_value());
        REQUIRE(iterators.value().size() == static_cast<size_t>(artifact->data.query_ds->GetRows()));
        auto first_has_next = iterators.value()[0]->HasNext();
        REQUIRE(first_has_next.has_value());
        REQUIRE(first_has_next.value());
        auto first = iterators.value()[0]->Next();
        REQUIRE(first.has_value());
        auto [first_id, first_dist] = first.value();
        (void)first_dist;
        REQUIRE(first_id != artifact->data.query_ids[0]);
        auto second_has_next = iterators.value()[1]->HasNext();
        REQUIRE(second_has_next.has_value());
        REQUIRE(second_has_next.value());
        auto second = iterators.value()[1]->Next();
        REQUIRE(second.has_value());
        auto [second_id, second_dist] = second.value();
        REQUIRE(second_id == artifact->data.query_ids[1]);
        REQUIRE(second_dist == Catch::Approx(0.0f));
    };

    require_filtered_reads(artifact->index);

    REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
    require_filtered_reads(artifact->binary_loaded);

    REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
    require_filtered_reads(artifact->file_loaded);
}

TEST_CASE("Nullable HNSW multi-index keeps id map from valid bitmap", "[nullable][hnsw][multi]") {
    IndexRow row{"HNSW_SQ", knowhere::IndexEnum::INDEX_HNSW_SQ, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::MultiIndex;
    scenario.nullable_ratio = NullableRatio::R0;
    scenario.total_count_override = 512;

    auto artifact = BuildArtifactForScenario(row, scenario);
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);
    RequireIdMapContent(artifact->index, artifact->data.valid_ids, artifact->data.total_count);
    {
        const auto& id_map = artifact->index.Node()->GetIdMap();
        REQUIRE(id_map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST).empty());
    }

    auto filtered_ids = FilterIdsFor(FilterRatio::R0, artifact->data.total_count, artifact->data.valid_ids,
                                     Mode::MultiIndex, artifact->data.selected_ids);
    auto allowed_ids = AllowedIdsAfterFilter(artifact->data.selected_ids, filtered_ids);
    REQUIRE(!allowed_ids.empty());
    std::vector<uint8_t> bitset_data;
    auto bitset = BitsetViewFrom(bitset_data, artifact->data.total_count, filtered_ids);

    auto search = artifact->index.Search(artifact->data.query_ds, artifact->json, bitset);
    REQUIRE(search.has_value());
    RequireExpectedVectorResult(search, allowed_ids, filtered_ids, false);
}

TEST_CASE("Nullable HNSW multi-index filters use out ids with local partitions", "[nullable][hnsw][multi]") {
    IndexRow row = NativeFaissHnswRow();
    Scenario scenario;
    scenario.mode = Mode::MultiIndex;
    scenario.nullable_ratio = NullableRatio::R50;
    scenario.total_count_override = 512;

    auto artifact = BuildArtifactForScenario(row, scenario);
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);
    REQUIRE(!artifact->data.selected_ids.empty());

    for (auto filter_ratio : {FilterRatio::R0, FilterRatio::R50}) {
        CAPTURE(FilterName(filter_ratio));
        auto filtered_ids = FilterIdsFor(filter_ratio, artifact->data.total_count, artifact->data.valid_ids,
                                         Mode::MultiIndex, artifact->data.selected_ids);
        auto allowed_ids = AllowedIdsAfterFilter(artifact->data.selected_ids, filtered_ids);
        REQUIRE(!allowed_ids.empty());
        std::vector<uint8_t> bitset_data;
        auto bitset = BitsetViewFrom(bitset_data, artifact->data.total_count, filtered_ids);

        auto search = artifact->index.Search(artifact->data.query_ds, artifact->json, bitset);
        REQUIRE(search.has_value());
        RequireExpectedVectorResult(search, allowed_ids, filtered_ids, false);

        auto range_json = artifact->json;
        range_json[knowhere::meta::RADIUS] = 0.5f;
        auto range = artifact->index.RangeSearch(artifact->data.query_ds, range_json, bitset);
        REQUIRE(range.has_value());
        RequireExpectedRangeResult(range, allowed_ids, filtered_ids, false);

        auto iterator = artifact->index.AnnIterator(artifact->data.query_ds, artifact->json, bitset);
        REQUIRE(iterator.has_value());
        RequireExpectedIteratorResult(iterator, allowed_ids, filtered_ids, false);
    }
}

TEST_CASE("Nullable HNSW multi-index vector APIs use out ids after reload", "[nullable][hnsw][multi][api]") {
    IndexRow row = NativeFaissHnswRow();
    Scenario scenario;
    scenario.mode = Mode::MultiIndex;
    scenario.nullable_ratio = NullableRatio::R50;
    scenario.total_count_override = 512;

    auto artifact = BuildArtifactForScenario(row, scenario);
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);
    REQUIRE(!artifact->data.selected_ids.empty());

    auto require_vector_apis = [&](const knowhere::Index<knowhere::IndexNode>& index, bool require_ordered_ids) {
        if (require_ordered_ids) {
            RequireIdMapContent(index, artifact->data.valid_ids, artifact->data.total_count);
        } else {
            RequireIdMapBitmap(index, artifact->data.total_count);
        }

        for (auto filter_ratio : {FilterRatio::R0, FilterRatio::R50}) {
            CAPTURE(FilterName(filter_ratio));
            auto filtered_ids = FilterIdsFor(filter_ratio, artifact->data.total_count, artifact->data.valid_ids,
                                             Mode::MultiIndex, artifact->data.selected_ids);
            auto allowed_ids = AllowedIdsAfterFilter(artifact->data.selected_ids, filtered_ids);
            REQUIRE(!allowed_ids.empty());
            std::vector<uint8_t> bitset_data;
            auto bitset = BitsetViewFrom(bitset_data, artifact->data.total_count, filtered_ids);

            auto search = index.Search(artifact->data.query_ds, artifact->json, bitset);
            REQUIRE(search.has_value());
            RequireExpectedVectorResult(search, allowed_ids, filtered_ids, false);
        }

        std::vector<int64_t> ids(artifact->data.query_ids.begin(), artifact->data.query_ids.end());
        auto retrieve = index.GetVectorByIds(knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()));
        REQUIRE(retrieve.has_value());
        RequireDenseVectorsMatchLogicalIds(*retrieve.value(), artifact->data.query_ids, artifact->data.dim);
    };

    require_vector_apis(artifact->index, true);

    REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
    require_vector_apis(artifact->binary_loaded, false);

    REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
    require_vector_apis(artifact->file_loaded, false);
}

TEST_CASE("Nullable HNSW emb-list multi-index filters use external doc ids", "[nullable][hnsw][emblist][multi]") {
    IndexRow row{"HNSW_SQ", knowhere::IndexEnum::INDEX_HNSW_SQ, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::EmbList;
    scenario.nullable_ratio = NullableRatio::R50;
    scenario.emb_list_strategy = knowhere::meta::EMB_LIST_STRATEGY_TOKENANN;
    scenario.total_count_override = 256;

    auto work_dir = NullableIdMapWorkDir("hnsw_emblist_multi");
    ScopedWorkDir work_dir_guard(work_dir);
    auto data = BuildMatrixData(row, scenario, work_dir);
    REQUIRE(data.valid_ids.size() >= 4);

    std::vector<int32_t> selected_ids(data.valid_ids.begin(), data.valid_ids.begin() + data.valid_ids.size() / 2);
    data.selected_ids = selected_ids;
    data.query_ids = FirstQueryIds(selected_ids, data.total_count, 2);
    data.query_ds = GenEmbListQueryDataSet(data.query_ids, data.dim);

    std::unordered_map<int64_t, std::vector<std::vector<uint32_t>>> scalar_info;
    scalar_info[0].resize(2);
    for (size_t doc_id = 0; doc_id < data.valid_ids.size(); ++doc_id) {
        const auto partition_id = doc_id < selected_ids.size() ? 0 : 1;
        for (int64_t vector_id = 0; vector_id < kEmbVectorsPerDoc; ++vector_id) {
            scalar_info[0][partition_id].push_back(static_cast<uint32_t>(doc_id * kEmbVectorsPerDoc + vector_id));
        }
    }
    data.train_ds->Set(knowhere::meta::SCALAR_INFO, scalar_info);

    auto json = BuildJson(row, scenario, data, work_dir / "raw_data.bin", work_dir / "file_index");
    auto created = CreateIndex(row, data);
    REQUIRE(created.ok);

    created.index.SetIdMapType(knowhere::IdMap::Type::SEALED);
    REQUIRE(AttachRuntimeIdMapData(data.train_ds, data.valid_ids, data.total_count) == knowhere::Status::success);
    REQUIRE(created.index.Build(data.train_ds, json) == knowhere::Status::success);
    RequireIdMapContent(created.index, data.valid_ids, data.total_count);
    {
        const auto& created_id_map = created.index.Node()->GetIdMap();
        REQUIRE(!created_id_map.InToOutIds(knowhere::IdMap::Domain::EMB_LIST).empty());
    }

    std::set<int32_t> selected(selected_ids.begin(), selected_ids.end());
    std::vector<int32_t> filtered_ids;
    for (int32_t id = 0; id < data.total_count; ++id) {
        if (selected.find(id) == selected.end()) {
            filtered_ids.push_back(id);
        }
    }
    filtered_ids.push_back(selected_ids.front());
    auto allowed_ids = AllowedIdsAfterFilter(selected_ids, filtered_ids);
    REQUIRE(!allowed_ids.empty());

    std::vector<uint8_t> bitset_data;
    auto bitset = BitsetViewFrom(bitset_data, data.total_count, filtered_ids);
    auto search = created.index.Search(data.query_ds, json, bitset);
    REQUIRE(search.has_value());
    RequireExpectedVectorResult(search, allowed_ids, filtered_ids, false);
}

TEST_CASE("Nullable TokenANN GetEmbListByIds uses logical ids", "[nullable][emblist][api]") {
    struct RetrieveCase {
        std::string label;
        IndexRow row;
        bool check_built = true;
        bool check_binary_load = false;
        bool check_file_load = false;
    };

    auto diskann = NativeDiskAnnRow();
    diskann.maybe_unavailable = true;

    std::vector<RetrieveCase> cases = {
        {"HNSW", NativeFaissHnswRow(), true, true, true},
        {"IVF_FLAT",
         {"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32},
         false,
         true,
         true},
        {"IVF_FLAT_CC",
         {"IVF_FLAT_CC / IVFFLATCC", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC, DataKind::DenseFp32},
         true,
         true,
         true},
        {"DISKANN", diskann, false, false, false},
    };

    for (const auto& test_case : cases) {
        CAPTURE(test_case.label);
        Scenario scenario;
        scenario.mode = Mode::EmbList;
        scenario.nullable_ratio = NullableRatio::R50;
        scenario.emb_list_strategy = knowhere::meta::EMB_LIST_STRATEGY_TOKENANN;

        auto artifact = BuildArtifactForScenario(test_case.row, scenario);
        CAPTURE(artifact->json.dump());
        if (!artifact->create_ok) {
            REQUIRE(test_case.row.maybe_unavailable);
            REQUIRE(artifact->create_status != knowhere::Status::success);
            continue;
        }
        REQUIRE(artifact->create_ok);
        REQUIRE(artifact->build_status == knowhere::Status::success);

        if (test_case.check_built) {
            RequireEmbListRetrieveUsesOutIds(artifact->index, artifact->data);
        }

        if (test_case.check_binary_load) {
            REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
            RequireEmbListRetrieveUsesOutIds(artifact->binary_loaded, artifact->data);
        }
        if (test_case.check_file_load) {
            REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
            RequireEmbListRetrieveUsesOutIds(artifact->file_loaded, artifact->data);
        }
    }
}

TEST_CASE("Nullable SCANN_DVR TokenANN GetEmbListByIds reports unsupported raw retrieve",
          "[nullable][scann_dvr][emblist][api]") {
    IndexRow row{"SCANN_DVR", knowhere::IndexEnum::INDEX_FAISS_SCANN_DVR, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::EmbList;
    scenario.nullable_ratio = NullableRatio::R50;
    scenario.emb_list_strategy = knowhere::meta::EMB_LIST_STRATEGY_TOKENANN;

    auto artifact = BuildArtifactForScenario(row, scenario);
    CAPTURE(artifact->json.dump());
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);

    std::vector<int64_t> ids(artifact->data.query_ids.begin(), artifact->data.query_ids.end());
    auto retrieve = artifact->index.GetEmbListByIds(
        knowhere::GenIdsDataSet(static_cast<int64_t>(ids.size()), ids.data()), "MAX_SIM_L2");
    REQUIRE_FALSE(retrieve.has_value());
    REQUIRE(retrieve.error() == knowhere::Status::not_implemented);
}

#ifdef KNOWHERE_WITH_CARDINAL
TEST_CASE("Nullable raw-data vector APIs map compact serialized ids after reload", "[nullable][id_map][api]") {
    IndexRow row{"HNSW current", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32, Capabilities{}, false, true};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;
    scenario.force_raw_data = true;

    auto artifact = BuildArtifactForScenario(row, scenario);
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);

    RequireVectorIdMapCompacted(artifact->index, artifact->data.valid_ids, artifact->data.total_count);
    RequireDenseVectorPublicApisUseOutIds(artifact->index, artifact->data, artifact->json, false);

    REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
    RequireVectorIdMapCompacted(artifact->binary_loaded, artifact->data.valid_ids, artifact->data.total_count);
    RequireDenseVectorPublicApisUseOutIds(artifact->binary_loaded, artifact->data, artifact->json, false);

    REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
    RequireVectorIdMapCompacted(artifact->file_loaded, artifact->data.valid_ids, artifact->data.total_count);
    RequireDenseVectorPublicApisUseOutIds(artifact->file_loaded, artifact->data, artifact->json, false);
}

TEST_CASE("Nullable TokenANN keeps only required list id map state", "[nullable][id_map][emblist][memory]") {
    IndexRow row{"HNSW current", knowhere::IndexEnum::INDEX_HNSW, DataKind::DenseFp32, Capabilities{}, false, true};
    Scenario scenario;
    scenario.mode = Mode::EmbList;
    scenario.nullable_ratio = NullableRatio::R50;
    scenario.emb_list_strategy = knowhere::meta::EMB_LIST_STRATEGY_TOKENANN;
    scenario.force_raw_data = true;

    auto artifact = BuildArtifactForScenario(row, scenario);
    CAPTURE(artifact->json.dump());
    REQUIRE(artifact->create_ok);
    REQUIRE(artifact->build_status == knowhere::Status::success);

    RequireEmbListIdMapCompacted(artifact->index, artifact->data.valid_ids, artifact->data.total_count);
    RequireEmbListRetrieveUsesOutIds(artifact->index, artifact->data);

    REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
    RequireEmbListIdMapCompacted(artifact->binary_loaded, artifact->data.valid_ids, artifact->data.total_count);
    RequireEmbListRetrieveUsesOutIds(artifact->binary_loaded, artifact->data);

    REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
    RequireEmbListIdMapCompacted(artifact->file_loaded, artifact->data.valid_ids, artifact->data.total_count);
    RequireEmbListRetrieveUsesOutIds(artifact->file_loaded, artifact->data);
}
#endif

TEST_CASE("Nullable IVF_FLAT MUVERA/LEMUR emb-list APIs rebuild list ids after reload",
          "[nullable][ivf][emblist][api]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    for (const auto& strategy : {knowhere::meta::EMB_LIST_STRATEGY_MUVERA, knowhere::meta::EMB_LIST_STRATEGY_LEMUR}) {
        CAPTURE(strategy);
        Scenario scenario;
        scenario.mode = Mode::EmbList;
        scenario.nullable_ratio = NullableRatio::R50;
        scenario.emb_list_strategy = strategy;

        auto artifact = BuildArtifactForScenario(row, scenario);
        CAPTURE(artifact->json.dump());
        REQUIRE(artifact->create_ok);
        REQUIRE(artifact->build_status == knowhere::Status::success);

        RequireIdMapContent(artifact->index, artifact->data.valid_ids, artifact->data.total_count);
        RequireEmbListApisUseOutIds(artifact->index, artifact->data, artifact->json);

        REQUIRE(EnsureBinaryLoaded(*artifact) == knowhere::Status::success);
        RequireIdMapContent(artifact->binary_loaded, artifact->data.valid_ids, artifact->data.total_count);
        RequireEmbListApisUseOutIds(artifact->binary_loaded, artifact->data, artifact->json);

        REQUIRE(EnsureFileLoaded(*artifact) == knowhere::Status::success);
        RequireIdMapContent(artifact->file_loaded, artifact->data.valid_ids, artifact->data.total_count);
        RequireEmbListApisUseOutIds(artifact->file_loaded, artifact->data, artifact->json);
    }
}

TEST_CASE("Nullable mmap file load preserves large non-identity mapping", "[nullable][mmap][api]") {
    IndexRow row{"IVF_FLAT / IVFFLAT", knowhere::IndexEnum::INDEX_FAISS_IVFFLAT, DataKind::DenseFp32};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    BuiltArtifact artifact;
    artifact.row = row;
    artifact.scenario = scenario;
    artifact.work_dir = NullableIdMapWorkDir("ivf_mmap_large_mapping");
    RemoveAllNoThrow(artifact.work_dir);
    fs::create_directories(artifact.work_dir);
    artifact.main_file = artifact.work_dir / "main.index";
    artifact.data.total_count = 160;
    artifact.data.dim = kDenseDim;
    artifact.data.valid_ids = EvenIds(artifact.data.total_count);
    artifact.data.query_ids = FirstQueryIds(artifact.data.valid_ids, artifact.data.total_count);
    artifact.data.train_ds =
        GenNullableDenseDataSet(artifact.data.total_count, artifact.data.valid_ids, artifact.data.dim);
    artifact.data.query_ds = GenDenseQueryDataSet(artifact.data.query_ids, artifact.data.dim);
    artifact.json = DenseConfigFor(row, Mode::Vector);

    auto created = CreateIndex(row, artifact.data);
    REQUIRE(created.ok);
    artifact.index = std::move(created.index);
    artifact.index.SetIdMapType(knowhere::IdMap::Type::SEALED);
    REQUIRE(AttachRuntimeIdMapData(artifact.data.train_ds, artifact.data.valid_ids, artifact.data.total_count) ==
            Status::success);
    artifact.build_status = artifact.index.Build(artifact.data.train_ds, artifact.json);
    REQUIRE(artifact.build_status == knowhere::Status::success);
    REQUIRE(EnsureFileSerialized(artifact) == knowhere::Status::success);

    auto load_json = FileLoadJson(artifact);
    load_json["enable_mmap"] = true;
    auto loaded = CreateIndex(row, artifact.data);
    REQUIRE(loaded.ok);
    REQUIRE(SetRuntimeValidBitmap(loaded.index, artifact.data) == knowhere::Status::success);
    REQUIRE(loaded.index.DeserializeFromFile(artifact.main_file.string(), load_json) == knowhere::Status::success);
    RequireDenseVectorApisUseOutIds(loaded.index, artifact.data, artifact.json);
}

TEST_CASE("Nullable cuVS search materializes mapped filters to internal bitmap", "[nullable][cuvs][gpu]") {
#ifndef KNOWHERE_WITH_CUVS
    SUCCEED("cuVS is not enabled in this build");
#else
    IndexRow row{"GPU_CUVS_BRUTE_FORCE",
                 knowhere::IndexEnum::INDEX_CUVS_BRUTEFORCE,
                 DataKind::GpuFp32,
                 Capabilities{},
                 true,
                 false,
                 false,
                 true};
    Scenario scenario;
    scenario.mode = Mode::Vector;
    scenario.nullable_ratio = NullableRatio::R50;

    auto artifact = BuildArtifactForScenario(row, scenario);
    if (!artifact->create_ok || artifact->build_status != knowhere::Status::success) {
        SUCCEED("cuVS is unavailable in this test environment");
        return;
    }
    REQUIRE(artifact->data.query_ids.size() >= 2);

    std::vector<uint8_t> missing_filter_bits;
    auto missing_id = FirstMissingOutId(artifact->data.valid_ids, artifact->data.total_count);
    auto missing_filter =
        BitsetViewFrom(missing_filter_bits, artifact->data.total_count, {static_cast<int32_t>(missing_id)});
    auto missing_filter_result = artifact->index.Search(artifact->data.query_ds, artifact->json, missing_filter);
    if (!missing_filter_result.has_value()) {
        SUCCEED("cuVS search is unavailable in this test environment");
        return;
    }
    RequireExactFirstHits(*missing_filter_result.value(), artifact->data.query_ids, artifact->data.valid_ids);

    std::vector<uint8_t> valid_filter_bits;
    auto valid_filter = BitsetViewFrom(valid_filter_bits, artifact->data.total_count, {artifact->data.query_ids[0]});
    auto valid_filter_result = artifact->index.Search(artifact->data.query_ds, artifact->json, valid_filter);
    REQUIRE(valid_filter_result.has_value());
    RequireExpectedVectorResult(valid_filter_result, artifact->data.valid_ids, {artifact->data.query_ids[0]}, false);
    REQUIRE(valid_filter_result.value()->GetIds()[0] != artifact->data.query_ids[0]);
    REQUIRE(valid_filter_result.value()->GetIds()[kTopK] == artifact->data.query_ids[1]);
#endif
}

TEST_CASE("Nullable matrix has 43 index rows and 220 scenarios per row", "[nullable][matrix][count]") {
    const auto rows = IndexRows();
    const auto scenarios = BuildScenarios();
    REQUIRE(rows.size() == 43);
    REQUIRE(scenarios.size() == 220);
    REQUIRE(rows.size() * scenarios.size() == 9460);

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
        {"HNSWLIB_DEPRECATED", "HNSWLIB_DEPRECATED", DataKind::DenseFp32},
        {"DISKANN (Knowhere native)", knowhere::IndexEnum::INDEX_DISKANN, DataKind::DiskAnn},
        {"AISAQ", knowhere::IndexEnum::INDEX_AISAQ, DataKind::Aisaq},
        {"SPARSE_INVERTED_INDEX (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, DataKind::Sparse},
        {"SPARSE_WAND (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_WAND, DataKind::Sparse},
        {"SPARSE_INVERTED_INDEX_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX_CC,
         DataKind::Sparse},
        {"SPARSE_WAND_CC (Knowhere native)", knowhere::IndexEnum::INDEX_SPARSE_WAND_CC, DataKind::Sparse},
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
    };
    REQUIRE(required_rows.size() == rows.size());
    for (const auto& required : required_rows) {
        CAPTURE(required.label, required.index_type);
        REQUIRE(HasIndexRow(rows, required));
    }
}

TEST_CASE("Nullable native matrix covers every IndexNode operation", "[nullable][matrix]") {
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
