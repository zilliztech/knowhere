// Copyright (C) 2019-2026 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// Knowhere owns DiskANN/AiSAQ parameter defaults, validation and budget resolution. These tests pin the defaults
// Milvus used to inject, check that they reach the native consumers, and cover disk-PQ artifacts.

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "../DiskANN/include/diskann/defaults.h"
#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "diskann/pq_table.h"
#include "diskann/ssd_graph_index.h"
#include "diskann/utils.h"
#include "filemanager/FileManager.h"
#include "filemanager/impl/LocalFileManager.h"
#include "index/diskann/aisaq_config.h"
#include "index/diskann/diskann_config.h"
#include "index/diskann/pq_navigation_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_static.h"
#include "knowhere/utils.h"
#include "utils.h"

namespace fs = std::filesystem;

namespace {
// Select the native implementation even when Cardinal registers its own under the public DISKANN name.
#ifdef KNOWHERE_WITH_CARDINAL
constexpr const char* kNativeDiskANNType = "DISKANN_DEPRECATED";
#else
constexpr const char* kNativeDiskANNType = "DISKANN";
#endif

const knowhere::PQNavigationConfig&
PQConfig(const knowhere::BaseConfig& cfg) {
    return static_cast<const knowhere::PQNavigationConfig&>(
        *static_cast<const knowhere::DiskANNConfig&>(cfg).navigation);
}

double
RawGB(size_t rows, size_t dimension, size_t element_bytes) {
    return static_cast<double>(rows) * dimension * element_bytes / (1024.0 * 1024 * 1024);
}
}  // namespace

TEST_CASE("Disk index native defaults and validation", "[diskann][disk-parameters]") {
    using namespace knowhere;
    const auto type = GENERATE(as<std::string>{}, kNativeDiskANNType, "AISAQ");
    const bool aisaq = type == "AISAQ";
    const auto version = Version::GetMaximumVersion().VersionNumber();
    std::string message;
    REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(type, version, Json::object(), message) == Status::success);
    auto cfg = IndexStaticFaced<fp32>::CreateConfig(type, version);
    REQUIRE(Config::Load(*cfg, Json::object(), TRAIN, &message) == Status::success);
    auto& disk = static_cast<DiskANNConfig&>(*cfg);
    // The values Milvus used to inject for DISKANN, and AISAQ's shipped Milvus defaults.
    REQUIRE(disk.max_degree == 56);
    REQUIRE(disk.search_list_size == 100);
    REQUIRE(PQConfig(*cfg).pq_code_budget_gb_ratio == 0.125f);
    REQUIRE(disk.search_cache_budget_gb_ratio == (aisaq ? 0.0f : 0.10f));
    REQUIRE_FALSE(disk.disk_pq_dims.has_value());
    REQUIRE(disk.disk_pq_code_budget_gb_ratio == 0);
    REQUIRE_FALSE(disk.UsesDiskPQ());
    if (aisaq) {
        auto& aisaq_cfg = static_cast<AisaqConfig&>(*cfg);
        REQUIRE(aisaq_cfg.inline_pq == -1);
        REQUIRE(aisaq_cfg.num_entry_points == 100);
        REQUIRE(aisaq_cfg.rearrange == true);
        REQUIRE(aisaq_cfg.pq_cache_size == 0);
        REQUIRE(aisaq_cfg.pq_read_page_cache_size == diskann::defaults::DEFAULT_PQ_READ_PAGE_CACHE_SIZE);
        auto load = IndexStaticFaced<fp32>::CreateConfig(type, version);
        REQUIRE(Config::Load(*load, Json::object(), DESERIALIZE) == Status::success);
        REQUIRE(static_cast<AisaqConfig&>(*load).pq_read_page_cache_size ==
                diskann::defaults::DEFAULT_PQ_READ_PAGE_CACHE_SIZE);
        // An omitted search value inherits the policy that Deserialize established.
        auto search = IndexStaticFaced<fp32>::CreateConfig(type, version);
        REQUIRE(Config::Load(*search, Json::object(), SEARCH) == Status::success);
        REQUIRE_FALSE(static_cast<AisaqConfig&>(*search).pq_read_page_cache_size.has_value());
        // The IO engine was never consumed; persisted values must keep loading.
        REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(type, version, {{"pq_read_io_engine", "uring"}}, message) ==
                Status::success);
    }
    for (const auto& invalid : std::vector<Json>{{{"max_degree", 0}},
                                                 {{"max_degree", diskann::defaults::MAX_GRAPH_DEGREE + 1}},
                                                 {{"disk_pq_dims", -1}},
                                                 {{"disk_pq_code_budget_gb_ratio", -0.1}},
                                                 {{"disk_pq_code_budget_gb_ratio", 1.1}},
                                                 {{"pq_code_budget_gb_ratio", -1}},
                                                 {{"search_cache_budget_gb_ratio", -1}},
                                                 {{"pq_code_budget_gb", "bad"}},
                                                 {{"accelerate_build", "bad"}}}) {
        CAPTURE(type, invalid.dump());
        REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(type, version, invalid, message) != Status::success);
    }
    REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(type, version, {{"max_degree", diskann::defaults::MAX_GRAPH_DEGREE}},
                                                message) == Status::success);
    // DiskANN's builder clamps a disk PQ wider than the vector, as before; AISAQ rejects it below.
    REQUIRE((IndexStaticFaced<fp32>::ConfigCheck(type, version, {{"dim", 16}, {"disk_pq_dims", 17}}, message) ==
             Status::success) == !aisaq);
    if (aisaq) {
        for (const auto& invalid : std::vector<Json>{{{"inline_pq", 57}},
                                                     {{"dim", 16}, {"disk_pq_dims", 17}},
                                                     {{"search_list_size", 513}},
                                                     {{"num_entry_points", -1}},
                                                     {{"pq_cache_size", -1}},
                                                     {{"pq_read_page_cache_size", -1}}}) {
            CAPTURE(invalid.dump());
            REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(type, version, invalid, message) != Status::success);
        }
    }
}

TEST_CASE("Disk index defaults leave the Cardinal-shared configuration unchanged", "[diskann][disk-parameters]") {
    knowhere::DiskANNSSDConfig shared;
    REQUIRE(knowhere::Config::Load(shared, knowhere::Json::object(), knowhere::PARAM_TYPE::TRAIN) ==
            knowhere::Status::success);
    REQUIRE(shared.max_degree == 48);
    REQUIRE(shared.search_list_size == 128);
    REQUIRE(shared.search_cache_budget_gb_ratio == 0);
    REQUIRE(shared.disk_pq_dims == 0);
    knowhere::DiskANNSSDConfig search;
    REQUIRE(knowhere::Config::Load(search, knowhere::Json::object(), knowhere::PARAM_TYPE::SEARCH) ==
            knowhere::Status::success);
    REQUIRE(search.beamwidth == static_cast<int>(diskann::defaults::DEFAULT_DISKANN_BEAMWIDTH));
}

TEST_CASE("Disk index budgets resolve against the raw data", "[diskann][disk-parameters]") {
    using namespace knowhere;
    const auto type = GENERATE(as<std::string>{}, kNativeDiskANNType, "AISAQ");
    const auto element_bytes = GENERATE(2, 4);
    const bool aisaq = type == "AISAQ";
    const auto version = Version::GetMaximumVersion().VersionNumber();
    auto load = [&](const Json& params) {
        auto cfg = IndexStaticFaced<fp32>::CreateConfig(type, version);
        REQUIRE(Config::Load(*cfg, params, TRAIN) == Status::success);
        return cfg;
    };
    // Defaults: ratios apply to the raw bytes being indexed, and disk PQ stays off.
    {
        auto cfg = load(Json::object());
        auto& disk = static_cast<DiskANNConfig&>(*cfg);
        disk.ResolveBuildBudgets(512, 32, element_bytes);
        REQUIRE(PQConfig(*cfg).pq_code_budget_gb.value() == Catch::Approx(RawGB(512, 32, element_bytes) * .125));
        REQUIRE(disk.search_cache_budget_gb.value() ==
                Catch::Approx(aisaq ? 0.0 : RawGB(512, 32, element_bytes) * .10));
        REQUIRE(disk.disk_pq_dims == 0);
    }
    // A disk-PQ ratio sizes the codes from the element width; an explicit dimension wins.
    {
        auto cfg = load({{"disk_pq_code_budget_gb_ratio", .25}});
        auto& disk = static_cast<DiskANNConfig&>(*cfg);
        REQUIRE(disk.UsesDiskPQ());
        disk.ResolveBuildBudgets(512, 32, element_bytes);
        REQUIRE(disk.disk_pq_dims == 32 * element_bytes / 4);
        auto explicit_zero = load({{"disk_pq_code_budget_gb_ratio", .25}, {"disk_pq_dims", 0}});
        REQUIRE_FALSE(static_cast<DiskANNConfig&>(*explicit_zero).UsesDiskPQ());
        static_cast<DiskANNConfig&>(*explicit_zero).ResolveBuildBudgets(512, 32, element_bytes);
        REQUIRE(static_cast<DiskANNConfig&>(*explicit_zero).disk_pq_dims == 0);
    }
    // Absolute budgets remain floors: max(absolute, ratio * raw size), as before the move.
    {
        const Json absolute = {{"pq_code_budget_gb", .001},
                               {"pq_code_budget_gb_ratio", 1},
                               {"search_cache_budget_gb", .002},
                               {"search_cache_budget_gb_ratio", 1}};
        auto small = load(absolute);
        static_cast<DiskANNConfig&>(*small).ResolveBuildBudgets(512, 32, element_bytes);
        REQUIRE(PQConfig(*small).pq_code_budget_gb.value() == Catch::Approx(.001));
        REQUIRE(static_cast<DiskANNConfig&>(*small).search_cache_budget_gb.value() == Catch::Approx(.002));
        auto large = load(absolute);
        static_cast<DiskANNConfig&>(*large).ResolveBuildBudgets(1000000, 128, element_bytes);
        REQUIRE(PQConfig(*large).pq_code_budget_gb.value() == Catch::Approx(RawGB(1000000, 128, element_bytes)));
        REQUIRE(static_cast<DiskANNConfig&>(*large).search_cache_budget_gb.value() ==
                Catch::Approx(RawGB(1000000, 128, element_bytes)));
    }
    // Index definitions created by older Milvus may persist an explicit zero; the ratio still applies.
    {
        std::string message;
        const Json persisted = {{"pq_code_budget_gb", 0}, {"pq_code_budget_gb_ratio", .125}};
        REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(type, version, persisted, message) == Status::success);
        auto cfg = load(persisted);
        static_cast<DiskANNConfig&>(*cfg).ResolveBuildBudgets(512, 32, element_bytes);
        REQUIRE(PQConfig(*cfg).pq_code_budget_gb.value() == Catch::Approx(RawGB(512, 32, element_bytes) * .125));
    }
    // Deserialize resolves the cache budget against the loaded vectors.
    {
        auto cfg = IndexStaticFaced<fp32>::CreateConfig(type, version);
        REQUIRE(Config::Load(*cfg, {{"search_cache_budget_gb_ratio", .2}}, DESERIALIZE) == Status::success);
        auto& disk = static_cast<DiskANNConfig&>(*cfg);
        disk.ResolveCacheBudget(512, 32, element_bytes);
        REQUIRE(disk.search_cache_budget_gb.value() == Catch::Approx(RawGB(512, 32, element_bytes) * .2));
    }
}

TEST_CASE("Disk raw-data capability reflects effective compression", "[diskann][disk-parameters]") {
    using namespace knowhere;
    const auto type = GENERATE(kNativeDiskANNType, "AISAQ");
    const auto metric_type = GENERATE("L2", "COSINE", "IP");
    const auto version = Version::GetMaximumVersion().VersionNumber();
    const bool raw_metric = std::string(metric_type) != "IP";
    Json params = {{"metric_type", metric_type}};
    REQUIRE(IndexStaticFaced<fp32>::HasRawData(type, version, params) == raw_metric);
    params.update({{"disk_pq_dims", 0}, {"disk_pq_code_budget_gb_ratio", .25}});
    REQUIRE(IndexStaticFaced<fp32>::HasRawData(type, version, params) == raw_metric);
    params.update({{"disk_pq_dims", 8}, {"disk_pq_code_budget_gb_ratio", 0}});
    REQUIRE_FALSE(IndexStaticFaced<fp32>::HasRawData(type, version, params));
    params.erase("disk_pq_dims");
    REQUIRE(IndexStaticFaced<fp32>::HasRawData(type, version, params) == raw_metric);
    params["disk_pq_code_budget_gb_ratio"] = .25;
    REQUIRE_FALSE(IndexStaticFaced<fp32>::HasRawData(type, version, params));
}

TEST_CASE("Disk arrays reject compression before building or reranking", "[diskann][disk-parameters]") {
    using namespace knowhere;
    const auto version = Version::GetMaximumVersion().VersionNumber();
    const auto metric_type = GENERATE("MAX_SIM_L2", "MAX_SIM_COSINE", "L2");
    Json params = {{"metric_type", metric_type}};
    if (std::string(metric_type) == "L2") {
        // Offset metadata also identifies arrays when building an element-space index.
        params["emb_list_offset_file_path"] = "unused-offsets";
    }
    std::shared_ptr<milvus::FileManager> manager = std::make_shared<milvus::LocalFileManager>();
    auto index = IndexFactory::Instance().Create<fp32>(kNativeDiskANNType, version, Pack(manager)).value();
    std::string message;
    for (const auto& compression : std::vector<Json>{{{"disk_pq_code_budget_gb_ratio", .25}},
                                                     {{"disk_pq_dims", 8}, {"disk_pq_code_budget_gb_ratio", 0}}}) {
        auto invalid = params;
        invalid.update(compression);
        REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(kNativeDiskANNType, version, invalid, message) ==
                Status::invalid_param_in_json);
        REQUIRE(message.find("disk PQ") != std::string::npos);
        REQUIRE(message.find("embedding-list") != std::string::npos);
        // No paths or raw files are needed to reject the unsupported combination.
        REQUIRE(index.Build(nullptr, invalid) == Status::invalid_param_in_json);
    }
    params.update({{"disk_pq_dims", 0}, {"disk_pq_code_budget_gb_ratio", .25}});
    REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(kNativeDiskANNType, version, params, message) == Status::success);
    params.erase("disk_pq_dims");
    params["disk_pq_code_budget_gb_ratio"] = 0;
    REQUIRE(IndexStaticFaced<fp32>::ConfigCheck(kNativeDiskANNType, version, params, message) == Status::success);
}

TEST_CASE("Disk index search parameters at each stage", "[diskann][disk-parameters]") {
    using namespace knowhere;
    const auto stage = GENERATE(SEARCH, RANGE_SEARCH, ITERATOR);
    const auto type = GENERATE(as<std::string>{}, kNativeDiskANNType, "AISAQ");
    const auto version = Version::GetMaximumVersion().VersionNumber();
    auto load = [&](const Json& params) {
        auto cfg = IndexStaticFaced<fp32>::CreateConfig(type, version);
        REQUIRE(Config::Load(*cfg, params, stage) == Status::success);
        return cfg;
    };
    {
        auto cfg = load({{"search_list", 67}, {"beamwidth", 4}, {"vectors_beamwidth", 2}});
        auto& disk = static_cast<DiskANNConfig&>(*cfg);
        REQUIRE(disk.search_list_size == 67);
        REQUIRE(disk.beamwidth == 4);
        if (type == "AISAQ") {
            REQUIRE(static_cast<AisaqConfig&>(*cfg).vectors_beamwidth == 2);
        }
    }
    // Milvus translated search_list over an explicit search_list_size; keep that precedence.
    REQUIRE(static_cast<DiskANNConfig&>(*load({{"search_list", 67}, {"search_list_size", 68}})).search_list_size == 67);
    // Milvus forced beamwidth 16 for DISKANN search and range search; iterators and AISAQ kept their defaults.
    const int expected_beamwidth = type == "AISAQ"     ? static_cast<int>(diskann::defaults::DEFAULT_AISAQ_BEAMWIDTH)
                                   : stage == ITERATOR ? static_cast<int>(diskann::defaults::DEFAULT_DISKANN_BEAMWIDTH)
                                                       : 16;
    REQUIRE(static_cast<DiskANNConfig&>(*load(Json::object())).beamwidth == expected_beamwidth);
}

TEST_CASE("Disk iterator final batch preserves metric conversion", "[diskann][disk-parameters]") {
    const auto metric = GENERATE(diskann::Metric::L2, diskann::Metric::INNER_PRODUCT, diskann::Metric::COSINE);
    const float query[] = {3, 4, 0};
    diskann::IteratorWorkspace<float> workspace(query, metric, 8, 3, .15f, 0, 4, -1, 10, knowhere::BitsetView());
    const float internal_distance = metric == diskann::Metric::COSINE ? -.2f : 1.6f;
    // IP's normalized dot .2 becomes score 10 after ||q||=5 and M=10;
    // the workspace exposes negative IP for the min-heap iterator adapter.
    const float expected = metric == diskann::Metric::INNER_PRODUCT ? -10 : internal_distance;
    workspace.insert_to_full(0, internal_distance);
    workspace.move_full_retset_to_backup();
    workspace.insert_to_full(1, internal_distance);
    workspace.move_last_full_retset_to_backup();
    REQUIRE(workspace.backup_res.size() == 2);
    REQUIRE(workspace.backup_res[0].val == Catch::Approx(expected));
    REQUIRE(workspace.backup_res[1].val == Catch::Approx(expected));
}

TEST_CASE("Disk build rejects invalid raw artifacts before resolving budgets", "[diskann][disk-parameters-native]") {
    using namespace knowhere;
    const auto type = GENERATE(kNativeDiskANNType, "AISAQ");
    const auto fault = GENERATE("missing", "empty", "short-header", "zero-rows", "zero-dim", "negative-rows",
                                "short-payload", "directory");
    CAPTURE(type, fault);
    const auto directory = fs::temp_directory_path() / "knowhere_disk_raw_metadata";
    fs::remove_all(directory);
    fs::create_directories(directory);
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            std::error_code error;
            fs::remove_all(path, error);
        }
    } cleanup{directory};
    const auto raw = directory / "raw.bin";
    const std::string failure = fault;
    if (failure == "directory") {
        fs::create_directory(raw);
    } else if (failure != "missing") {
        std::ofstream output(raw, std::ios::binary);
        int32_t shape[] = {1, 1};
        if (failure == "zero-rows") {
            shape[0] = 0;
        } else if (failure == "zero-dim") {
            shape[1] = 0;
        } else if (failure == "negative-rows") {
            shape[0] = -1;
        }
        if (failure != "empty") {
            output.write(reinterpret_cast<const char*>(shape), failure == "short-header" ? 4 : sizeof(shape));
        }
    }
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto pack = Pack(file_manager);
    auto index = IndexFactory::Instance().Create<fp32>(type, Version::GetMaximumVersion().VersionNumber(), pack);
    REQUIRE(index.has_value());
    Json config = {{"metric_type", "L2"},
                   {"index_prefix", (directory / "index").string()},
                   {"data_path", raw.string()},
                   {"build_dram_budget_gb", .25}};
    REQUIRE(index.value().Build(nullptr, config) == Status::disk_file_error);
}

namespace {
void
ReadDiskTestBinMetadata(const std::string& path, size_t& rows, size_t& columns) {
    CAPTURE(path);
    std::array<uint32_t, 2> shape{};
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.read(reinterpret_cast<char*>(shape.data()), sizeof(shape)).good());
    rows = shape[0];
    columns = shape[1];
}

std::array<uint64_t, 11>
ReadDiskNodeMetadata(const std::string& prefix, const std::string& type) {
    std::array<uint64_t, 11> metadata{};
    std::ifstream input(diskann::get_disk_index_filename(prefix), std::ios::binary);
    // Only AISAQ prefixes its metadata with the two uint32 bin dimensions.
    input.seekg(type == "AISAQ" ? 2 * sizeof(uint32_t) : 0);
    input.read(reinterpret_cast<char*>(metadata.data()), sizeof(metadata));
    REQUIRE(input.good());
    return metadata;
}

// Independent metric oracle: read actual stored codes, inflate their prototypes,
// then calculate vector arithmetic without the backend candidate-distance helper.
std::vector<float>
ReadDecodedDiskPQScores(const std::string& prefix, const std::string& type, const knowhere::DataSetPtr& query,
                        size_t rows, const std::string& metric_type) {
    const auto disk_file = diskann::get_disk_index_filename(prefix);
    const auto metadata = ReadDiskNodeMetadata(prefix, type);
    REQUIRE(metadata[4] > 0);  // This small fixture packs multiple nodes per sector.
    diskann::FixedChunkPQTable table;
    table.load_pq_centroid_bin(diskann::get_disk_index_pq_pivots_filename(disk_file).c_str(), 0);
    const auto dimension = query->GetDim();
    REQUIRE(table.get_total_dims() == dimension + (metric_type == "IP" ? 1 : 0));
    const auto* original_query = static_cast<const float*>(query->GetTensor());
    float query_norm = 0;
    for (int64_t col = 0; col < dimension; ++col) {
        query_norm += original_query[col] * original_query[col];
    }
    query_norm = std::sqrt(query_norm);
    REQUIRE(query_norm > 0);
    float max_base_norm = 1;
    if (metric_type == "IP") {
        std::unique_ptr<float[]> norms;
        size_t norm_rows, norm_columns;
        diskann::load_bin<float>(diskann::get_disk_index_max_base_norm_file(disk_file), norms, norm_rows, norm_columns);
        REQUIRE(norm_rows == 1);
        REQUIRE(norm_columns == 1);
        max_base_norm = norms[0];
        REQUIRE(max_base_norm > 0);
    }
    std::ifstream input(disk_file, std::ios::binary);
    std::vector<uint8_t> codes(table.get_num_chunks());
    std::vector<float> decoded(table.get_total_dims());
    std::vector<float> scores(rows);
    for (size_t id = 0; id < rows; ++id) {
        const auto offset = (id / metadata[4] + 1) * diskann::defaults::SECTOR_LEN + (id % metadata[4]) * metadata[3];
        input.seekg(offset);
        input.read(reinterpret_cast<char*>(codes.data()), codes.size());
        REQUIRE(input.good());
        table.inflate_vector(codes.data(), decoded.data());
        double value = 0;
        for (int64_t col = 0; col < dimension; ++col) {
            if (metric_type == "L2") {
                const double delta = original_query[col] - decoded[col];
                value += delta * delta;
            } else {
                const float normalized_query = original_query[col] / query_norm;
                value += static_cast<double>(normalized_query) * decoded[col];
            }
        }
        // The augmented IP coordinate has query value zero. Undo only the
        // actual query normalization and stored maximum-base-norm scaling.
        scores[id] = metric_type == "IP" ? value * max_base_norm * query_norm : value;
    }
    return scores;
}

void
CheckDiskPQScore(const std::vector<float>& scores, int64_t id, float score) {
    REQUIRE(id >= 0);
    REQUIRE(id < static_cast<int64_t>(scores.size()));
    REQUIRE(score == Catch::Approx(scores[id]).margin(.001));
}

void
CheckDiskPQIPIteratorAndRange(knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSetPtr& query,
                              const std::vector<float>& scores) {
    using namespace knowhere;
    auto iterators =
        index.AnnIterator(query, {{"metric_type", "IP"}, {"search_list", scores.size()}, {"beamwidth", 4}}, nullptr);
    REQUIRE(iterators.has_value());
    size_t count = 0;
    while (true) {
        auto has_next = iterators.value()[0]->HasNext();
        REQUIRE(has_next.has_value());
        if (!has_next.value()) {
            break;
        }
        auto next = iterators.value()[0]->Next();
        REQUIRE(next.has_value());
        CheckDiskPQScore(scores, next.value().first, next.value().second);
        REQUIRE(++count <= scores.size());
    }
    REQUIRE(count >= 16);
    auto ordered_scores = scores;
    std::sort(ordered_scores.begin(), ordered_scores.end());
    // Use a real gap so floating-point roundoff cannot change threshold membership.
    size_t split = 1;
    for (size_t i = 2; i < ordered_scores.size(); ++i) {
        if (ordered_scores[i] - ordered_scores[i - 1] > ordered_scores[split] - ordered_scores[split - 1]) {
            split = i;
        }
    }
    REQUIRE(ordered_scores[split] - ordered_scores[split - 1] > .01);
    const float radius = (ordered_scores[split - 1] + ordered_scores[split]) / 2;
    auto range = index.RangeSearch(
        query, {{"metric_type", "IP"}, {"search_list", scores.size()}, {"beamwidth", 4}, {"radius", radius}}, nullptr);
    REQUIRE(range.has_value());
    const auto found = range.value()->GetLims()[1];
    REQUIRE(found > 0);
    for (size_t i = 0; i < found; ++i) {
        const auto id = range.value()->GetIds()[i];
        CheckDiskPQScore(scores, id, range.value()->GetDistance()[i]);
        REQUIRE(scores[id] > radius);
    }
}

// Restrict brute force to exactly the graph result's candidates, so PQ distances and ranking
// can be compared without treating lossy PQ as an exact raw-vector distance.
void
CheckForcedDiskPQSearch(knowhere::Index<knowhere::IndexNode>& index, const knowhere::DataSetPtr& query,
                        const knowhere::DataSetPtr& graph, size_t rows, const std::string& metric_type,
                        const std::vector<float>* decoded_scores = nullptr) {
    using namespace knowhere;
    const auto k = graph->GetDim();
    std::vector<uint8_t> bits((rows + 7) / 8, 0xff);
    std::unordered_map<int64_t, float> expected;
    for (int64_t i = 0; i < k; ++i) {
        const auto id = graph->GetIds()[i];
        REQUIRE(id >= 0);
        REQUIRE(id < static_cast<int64_t>(rows));
        if (decoded_scores != nullptr) {
            CheckDiskPQScore(*decoded_scores, id, graph->GetDistance()[i]);
        }
        expected.emplace(id, graph->GetDistance()[i]);
        bits[id / 8] &= ~(uint8_t(1) << (id % 8));
    }
    REQUIRE(expected.size() == static_cast<size_t>(k));
    auto actual = index.Search(
        query,
        {{"metric_type", metric_type}, {"k", k}, {"search_list", rows}, {"beamwidth", 4}, {"filter_threshold", 0}},
        BitsetView(bits.data(), rows));
    REQUIRE(actual.has_value());
    for (int64_t i = 0; i < k; ++i) {
        const auto id = actual.value()->GetIds()[i];
        REQUIRE(expected.count(id) == 1);
        if (decoded_scores != nullptr) {
            CheckDiskPQScore(*decoded_scores, id, actual.value()->GetDistance()[i]);
        }
        // Ties may reorder IDs, but values and sorted ranks must agree.
        REQUIRE(actual.value()->GetDistance()[i] == Catch::Approx(expected.at(id)).margin(.0001));
        REQUIRE(actual.value()->GetDistance()[i] == Catch::Approx(graph->GetDistance()[i]).margin(.0001));
        expected.erase(id);
    }
    REQUIRE(expected.empty());
}

// Reads the metadata of a loaded node through its prepared build configuration, as the existing
// metadata tests do; graph degree and disk PQ come from the artifact.
knowhere::Json
LoadedBuildMetadata(knowhere::Index<knowhere::IndexNode>& index, const knowhere::Json& build) {
    auto config = index.Node()->CreateConfig();
    REQUIRE(knowhere::Config::Load(*config, build, knowhere::PARAM_TYPE::TRAIN) == knowhere::Status::success);
    auto meta = index.Node()->GetIndexMeta(std::move(config));
    REQUIRE(meta.has_value());
    return knowhere::Json::parse(meta.value()->GetJsonInfo())["build_params_"];
}

template <typename DataType>
void
CheckDiskParameterConsumers(const std::string& type, bool explicit_settings, bool rearranged = false, int inline_pq = 0,
                            bool compressed_diskann = false) {
    using namespace knowhere;
    constexpr size_t rows = 512;
    constexpr size_t dimension = 32;
    const auto directory = fs::current_path() / "disk_parameter_ownership";
    fs::remove_all(directory);
    fs::create_directories(directory);
    const std::string prefix = (directory / "index").string();
    const std::string raw = (directory / "raw").string();
    auto base = ConvertToDataTypeIfNeeded<DataType>(GenDataSet(rows, dimension, 30));
    auto query = ConvertToDataTypeIfNeeded<DataType>(GenDataSet(1, dimension, 30));
    WriteRawDataToDisk<DataType>(raw, static_cast<const DataType*>(base->GetTensor()), rows, dimension);
    std::shared_ptr<milvus::FileManager> manager = std::make_shared<milvus::LocalFileManager>();
    auto pack = Pack(manager);
    const auto version = Version::GetMaximumVersion().VersionNumber();
    Json build = {{"metric_type", "L2"},
                  {"data_path", raw},
                  {"index_prefix", prefix},
                  {"build_dram_budget_gb", .25},
                  {"num_build_thread", 2}};
    if (explicit_settings) {
        build.update({{"max_degree", 24},
                      {"search_list_size", 48},
                      {"pq_code_budget_gb_ratio", .0625},
                      {"disk_pq_code_budget_gb_ratio", .125},
                      {"disk_pq_dims", 0},
                      {"search_cache_budget_gb", .00001},
                      {"inline_pq", inline_pq},
                      {"rearrange", rearranged},
                      {"num_entry_points", 0}});
    }
    if (compressed_diskann) {
        build["disk_pq_dims"] = 8;
        build["disk_pq_code_budget_gb_ratio"] = 0;  // Positive absolute wins.
    }
    // No default enables disk PQ; only the explicit compressed DiskANN case stores codes on disk.
    const size_t disk_pq_dims = compressed_diskann ? 8 : 0;
    BinarySet binary;
    {
        auto index = IndexFactory::Instance().Create<DataType>(type, version, pack).value();
        REQUIRE(index.Build(nullptr, build) == Status::success);
        REQUIRE(index.Serialize(binary) == Status::success);
    }
    // Rearrangement replaces the ordinary PQ artifact. Its first two uint32 fields
    // still encode the vector count and code width (followed by the page size).
    const bool uses_rearrangement = type == "AISAQ" && (!explicit_settings || (rearranged && inline_pq < 24));
    const auto code_file = uses_rearrangement ? diskann::get_pq_compressed_rearranged_filename(prefix)
                                              : diskann::get_pq_compressed_filename(prefix);
    REQUIRE(fs::exists(diskann::get_pq_compressed_filename(prefix)) == !uses_rearrangement);
    // The serialized PQ codes prove that the real builder consumed the ratio and element width.
    size_t code_rows, code_bytes;
    ReadDiskTestBinMetadata(code_file, code_rows, code_bytes);
    REQUIRE(code_rows == rows);
    REQUIRE(code_bytes == dimension * sizeof(DataType) * (explicit_settings ? .0625 : .125));
    const auto disk_pivots = diskann::get_disk_index_pq_pivots_filename(diskann::get_disk_index_filename(prefix));
    REQUIRE(fs::exists(disk_pivots) == (disk_pq_dims != 0));
    if (fs::exists(disk_pivots)) {
        size_t chunks, columns;
        ReadDiskTestBinMetadata(diskann::get_pq_chunk_offsets_filename(disk_pivots), chunks, columns);
        REQUIRE(chunks == disk_pq_dims + 1);
    }
    if (type == "AISAQ") {
        REQUIRE(fs::exists(diskann::get_index_entry_points_filename(prefix)) == !explicit_settings);
        REQUIRE(fs::exists(diskann::get_index_rearranged_filename(prefix)) == uses_rearrangement);
    }
    const auto metadata = ReadDiskNodeMetadata(prefix, type);
    const size_t degree = explicit_settings ? 24 : 56;
    const size_t disk_vector_bytes = disk_pq_dims != 0 ? disk_pq_dims : dimension * sizeof(DataType);
    const size_t node_without_inline = (degree + 1) * sizeof(uint32_t) + disk_vector_bytes;
    if (type != "AISAQ" || !explicit_settings) {
        // -1 disables inline PQ, independently of the enabled rearrangement default.
        REQUIRE(metadata[3] == node_without_inline);
        if (type == "AISAQ") {
            REQUIRE(metadata[9] == 0);
        }
    } else {
        REQUIRE(metadata[9] == degree);
        if (inline_pq == 0) {
            // Automatic placement consumes otherwise unused sector space, not the full neighborhood.
            REQUIRE(metadata[3] > node_without_inline);
            REQUIRE(diskann::defaults::SECTOR_LEN / metadata[3] == diskann::defaults::SECTOR_LEN / node_without_inline);
        } else if (inline_pq == static_cast<int>(degree)) {
            REQUIRE(metadata[3] == node_without_inline + degree * code_bytes);
        } else {
            REQUIRE(metadata[3] >= node_without_inline + inline_pq * code_bytes);
            REQUIRE(metadata[3] < node_without_inline + degree * code_bytes);
        }
    }
    // The build budget decides whether Build saves a candidate list. Rearrangement removes the
    // ordinary PQ file that the build-time generator needs, so those layouts generate on load.
    const auto cache_file = diskann::get_cached_nodes_file(prefix);
    const bool expect_saved_list = !uses_rearrangement && (type != "AISAQ" || explicit_settings);
    REQUIRE(fs::exists(cache_file) == expect_saved_list);
    size_t saved_nodes = 0, saved_columns = 0;
    if (expect_saved_list) {
        ReadDiskTestBinMetadata(cache_file, saved_nodes, saved_columns);
        REQUIRE(saved_nodes > 1);
    }
    auto load_and_search = [&](Json load) {
        load["metric_type"] = "L2";
        load["index_prefix"] = prefix;
        // BFS is synchronous, so measured memory reflects fully populated cache nodes.
        load["use_bfs_cache"] = true;
        if (uses_rearrangement) {
            load["pq_read_page_cache_size"] = 8192;
        }
        auto index = IndexFactory::Instance().Create<DataType>(type, version, pack).value();
        REQUIRE(index.Deserialize(binary, load) == Status::success);
        auto build_meta = LoadedBuildMetadata(index, build);
        REQUIRE(build_meta["max_degree"] == degree);
        REQUIRE(build_meta["disk_pq_dims"] == disk_pq_dims);
        REQUIRE(index.HasRawData(metric::L2) == (disk_pq_dims == 0));
        REQUIRE(index.HasRawData(metric::COSINE) == (disk_pq_dims == 0));
        REQUIRE_FALSE(index.HasRawData(metric::IP));
        // Exercise all ids on every load: populated-cache and uncached nodes both remain safe.
        std::vector<int64_t> raw_ids(rows);
        std::iota(raw_ids.begin(), raw_ids.end(), 0);
        auto empty_vectors = index.GetVectorByIds(GenIdsDataSet(0, raw_ids.data()));
        REQUIRE(empty_vectors.has_value());
        REQUIRE(empty_vectors.value()->GetRows() == 0);
        auto vectors = index.GetVectorByIds(GenIdsDataSet(rows, raw_ids.data()));
        if (disk_pq_dims != 0) {
            REQUIRE_FALSE(vectors.has_value());
            REQUIRE(vectors.error() == Status::not_implemented);
        } else {
            REQUIRE(vectors.has_value());
            const auto* actual = static_cast<const DataType*>(vectors.value()->GetTensor());
            const auto* original = static_cast<const DataType*>(base->GetTensor());
            for (size_t i = 0; i < rows * dimension; ++i) {
                REQUIRE(static_cast<float>(actual[i]) == static_cast<float>(original[i]));
            }
        }
        if (type != "AISAQ") {
            auto distances = index.CalcDistByIDs(query, nullptr, raw_ids.data(), raw_ids.size(), false);
            if (disk_pq_dims != 0) {
                REQUIRE_FALSE(distances.has_value());
                REQUIRE(distances.error() == Status::not_implemented);
            } else {
                REQUIRE(distances.has_value());
                const auto* original = static_cast<const DataType*>(base->GetTensor());
                const auto* query_data = static_cast<const DataType*>(query->GetTensor());
                for (size_t row = 0; row < rows; ++row) {
                    float distance = 0;
                    for (size_t col = 0; col < dimension; ++col) {
                        const float delta =
                            static_cast<float>(original[row * dimension + col]) - static_cast<float>(query_data[col]);
                        distance += delta * delta;
                    }
                    REQUIRE(distances.value()->GetDistance()[row] == Catch::Approx(distance).margin(.001));
                }
            }
        }
        Json search = {{"metric_type", "L2"},         {"k", 1},
                       {"search_list", 40},           {"beamwidth", 4},
                       {"vectors_beamwidth", 2},      {"trace_visit", true},
                       {"pq_read_page_cache_size", 0}};
        auto result = index.Search(query, search, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] >= 0);
        if (disk_pq_dims != 0) {
            auto graph = index.Search(
                query, {{"metric_type", "L2"}, {"k", 16}, {"search_list", rows}, {"beamwidth", 4}}, nullptr);
            REQUIRE(graph.has_value());
            CheckForcedDiskPQSearch(index, query, graph.value(), rows, metric::L2);
        }
        const auto trace = Json::parse(result.value()->GetJsonInfo())["query_params_"];
        REQUIRE(trace["search_list_size"] == 40);
        REQUIRE(trace["beamwidth"] == 4);
        if (type == "AISAQ") {
            REQUIRE(trace["vectors_beamwidth"] == 2);
        } else {
            auto iterators =
                index.AnnIterator(query, {{"metric_type", "L2"}, {"search_list", 19}, {"beamwidth", 2}}, nullptr);
            REQUIRE(iterators.has_value());
            auto has_next = iterators.value()[0]->HasNext();
            REQUIRE(has_next.has_value());
            REQUIRE(has_next.value());
            auto ranges = index.RangeSearch(
                query, {{"metric_type", "L2"}, {"search_list", 19}, {"beamwidth", 2}, {"radius", 1000000}}, nullptr);
            REQUIRE(ranges.has_value());
        }
        if (uses_rearrangement) {
            // Missing search value inherits the load policy; an explicit zero disables it again.
            search.erase("pq_read_page_cache_size");
            REQUIRE(index.Search(query, search, nullptr).has_value());
            search["pq_read_page_cache_size"] = 4096;
            REQUIRE(index.Search(query, search, nullptr).has_value());
            search["pq_read_page_cache_size"] = 0;
            REQUIRE(index.Search(query, search, nullptr).has_value());
        }
        return index.Size();
    };
    // Load-time parameters must never change the capability of an existing artifact.
    load_and_search({{"search_cache_budget_gb", 0},
                     {"disk_pq_dims", disk_pq_dims != 0 ? 0 : 8},
                     {"disk_pq_code_budget_gb_ratio", disk_pq_dims != 0 ? 0 : .25}});
    const auto default_cache = load_and_search(Json::object());
    if (expect_saved_list) {
        // As before, a saved build-time list is loaded as saved, whatever the load budget.
        REQUIRE(saved_nodes < rows / 3);
        REQUIRE(load_and_search({{"search_cache_budget_gb", 0}, {"search_cache_budget_gb_ratio", 0}}) == default_cache);
        REQUIRE(load_and_search({{"search_cache_budget_gb", 1}}) == default_cache);
    }
    // Without a saved list, Deserialize generates exactly the resolved budget.
    fs::remove(cache_file);
    const auto zero = load_and_search({{"search_cache_budget_gb", 0}, {"search_cache_budget_gb_ratio", 0}});
    const auto generated_default = load_and_search(Json::object());
    const auto generated_small = load_and_search({{"search_cache_budget_gb_ratio", .1}});
    const auto generated_large = load_and_search({{"search_cache_budget_gb_ratio", .2}});
    REQUIRE(generated_small > zero);
    REQUIRE(generated_large > generated_small);
    // A budget beyond a third of the points is capped instead of failing the load.
    REQUIRE(load_and_search({{"search_cache_budget_gb", 1}}) > generated_large);
    if (expect_saved_list) {
        REQUIRE(default_cache > zero);
    }
    if (type == "AISAQ") {
        // AISAQ ships without a search cache; only an explicit build budget saves a list.
        if (!expect_saved_list) {
            REQUIRE(default_cache == zero);
        }
        REQUIRE(generated_default == zero);
    } else {
        REQUIRE(generated_default == generated_small);
    }
    REQUIRE_FALSE(fs::exists(cache_file));
    fs::remove_all(directory);
}

template <typename DataType>
void
CheckDiskArrayBudgets() {
    using namespace knowhere;
    constexpr size_t inner_rows = 512;
    constexpr size_t dimension = 32;
    constexpr size_t vectors_per_array = 8;
    auto base = ConvertToDataTypeIfNeeded<DataType>(GenEmbListDataSet(inner_rows, dimension, 42, vectors_per_array));
    auto query = ConvertToDataTypeIfNeeded<DataType>(GenQueryEmbListDataSet(1, dimension, 42));
    const auto directory = fs::current_path() / "disk_array_parameter_ownership";
    fs::remove_all(directory);
    fs::create_directories(directory);
    const auto raw = (directory / "raw").string();
    const auto prefix = (directory / "index").string();
    const auto offsets_path = (directory / "offsets").string();
    auto offsets = GenEmbListOffset(inner_rows, vectors_per_array);
    WriteRawDataToDisk<DataType>(raw, static_cast<const DataType*>(base->GetTensor()), inner_rows, dimension);
    WriteEmbListOffsetToDisk(offsets_path, offsets.data(), offsets.size());
    Json build = {{"metric_type", "MAX_SIM_L2"}, {"data_path", raw},
                  {"index_prefix", prefix},      {"emb_list_offset_file_path", offsets_path},
                  {"build_dram_budget_gb", .25}, {"num_build_thread", 2},
                  {"vec_field_size_gb", 0}};
    std::shared_ptr<milvus::FileManager> manager = std::make_shared<milvus::LocalFileManager>();
    auto pack = Pack(manager);
    const auto version = Version::GetMaximumVersion().VersionNumber();
    BinarySet binary;
    {
        auto index = IndexFactory::Instance().Create<DataType>(kNativeDiskANNType, version, pack).value();
        REQUIRE(index.Build(nullptr, build) == Status::success);
        REQUIRE(index.Serialize(binary) == Status::success);
    }
    // The PQ budget counts the inner vectors in the raw file, not the outer rows.
    size_t code_rows, code_bytes;
    ReadDiskTestBinMetadata(diskann::get_pq_compressed_filename(prefix), code_rows, code_bytes);
    REQUIRE(code_rows == inner_rows);
    REQUIRE(code_bytes == dimension * sizeof(DataType) / 8);
    {
        auto index = IndexFactory::Instance().Create<DataType>(kNativeDiskANNType, version, pack).value();
        REQUIRE(index.Deserialize(binary, {{"metric_type", "MAX_SIM_L2"}, {"index_prefix", prefix}}) ==
                Status::success);
        auto result = index.Search(query, {{"metric_type", "MAX_SIM_L2"}, {"k", 1}, {"search_list", 40}}, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] >= 0);
        REQUIRE(result.value()->GetIds()[0] < static_cast<int64_t>(inner_rows / vectors_per_array));
    }
    fs::remove_all(directory);
}
}  // namespace

TEST_CASE("Disk parameter defaults reach native consumers", "[diskann][disk-parameters-native]") {
    auto type = GENERATE(as<std::string>{}, kNativeDiskANNType, "AISAQ");
    SECTION("fp32") {
        CheckDiskParameterConsumers<knowhere::fp32>(type, false);
    }
    SECTION("fp16") {
        CheckDiskParameterConsumers<knowhere::fp16>(type, false);
    }
    SECTION("bf16") {
        CheckDiskParameterConsumers<knowhere::bf16>(type, false);
    }
}

TEST_CASE("Disk parameter overrides reach native consumers", "[diskann][disk-parameters-native]") {
    auto type = GENERATE(as<std::string>{}, kNativeDiskANNType, "AISAQ");
    CheckDiskParameterConsumers<knowhere::fp32>(type, true);
}

TEST_CASE("AiSAQ rearrangement and page-cache load policy", "[diskann][disk-parameters-native]") {
    CheckDiskParameterConsumers<knowhere::fp32>("AISAQ", true, true);
}

TEST_CASE("AiSAQ partial and full inline layouts reach disk artifacts", "[diskann][disk-parameters-native]") {
    auto inline_pq = GENERATE(4, 24);
    CheckDiskParameterConsumers<knowhere::fp32>("AISAQ", true, true, inline_pq);
}

TEST_CASE("Compressed DiskANN rejects raw vectors and exact distances", "[diskann][disk-parameters-native]") {
    CheckDiskParameterConsumers<knowhere::fp32>(kNativeDiskANNType, true, false, 0, true);
}

TEST_CASE("Disk parameter defaults count inner array vectors", "[diskann][disk-parameters-native]") {
    SECTION("fp32") {
        CheckDiskArrayBudgets<knowhere::fp32>();
    }
    SECTION("fp16") {
        CheckDiskArrayBudgets<knowhere::fp16>();
    }
    SECTION("bf16") {
        CheckDiskArrayBudgets<knowhere::bf16>();
    }
}

TEST_CASE("Compressed disk filtered refinement matches graph distances", "[diskann][disk-parameters-native]") {
    using namespace knowhere;
    const auto type = GENERATE(kNativeDiskANNType, "AISAQ");
    const auto metric_type = GENERATE("L2", "COSINE", "IP");
    constexpr size_t rows = 512, dimension = 32, k = 16;
    const auto directory = fs::current_path() / "disk_pq_filtered_refinement";
    fs::remove_all(directory);
    fs::create_directories(directory);
    const auto prefix = (directory / "index").string();
    const auto raw = (directory / "raw").string();
    auto base = GenDataSet(rows, dimension, 30);
    auto query = GenDataSet(1, dimension, 42);
    WriteRawDataToDisk<float>(raw, static_cast<const float*>(base->GetTensor()), rows, dimension);
    std::shared_ptr<milvus::FileManager> manager = std::make_shared<milvus::LocalFileManager>();
    auto pack = Pack(manager);
    const auto version = Version::GetMaximumVersion().VersionNumber();
    Json build = {
        {"metric_type", metric_type}, {"data_path", raw}, {"index_prefix", prefix}, {"build_dram_budget_gb", .25},
        {"num_build_thread", 2},      {"max_degree", 24}, {"search_list_size", 48}, {"disk_pq_dims", 8},
        {"rearrange", false},         {"inline_pq", -1},  {"num_entry_points", 0}};
    BinarySet binary;
    {
        auto index = IndexFactory::Instance().Create<fp32>(type, version, pack).value();
        REQUIRE(index.Build(nullptr, build) == Status::success);
        REQUIRE(index.Serialize(binary) == Status::success);
    }
    const auto decoded_scores = ReadDecodedDiskPQScores(prefix, type, query, rows, metric_type);
    // A saved build-time list is always loaded, so remove it to start uncached.
    fs::remove(diskann::get_cached_nodes_file(prefix));
    Json load = {{"metric_type", metric_type},
                 {"index_prefix", prefix},
                 {"search_cache_budget_gb", 0},
                 {"search_cache_budget_gb_ratio", 0}};
    DataSetPtr graph;
    uint64_t uncached_size;
    {
        auto index = IndexFactory::Instance().Create<fp32>(type, version, pack).value();
        REQUIRE(index.Deserialize(binary, load) == Status::success);
        REQUIRE(LoadedBuildMetadata(index, build)["disk_pq_dims"] == 8);
        // Empty bitset and k <= rows/2 keep this reference on graph search.
        auto reference = index.Search(
            query, {{"metric_type", metric_type}, {"k", k}, {"search_list", rows}, {"beamwidth", 4}}, nullptr);
        REQUIRE(reference.has_value());
        graph = reference.value();
        CheckForcedDiskPQSearch(index, query, graph, rows, metric_type, &decoded_scores);
        if (std::string(type) != "AISAQ" && std::string(metric_type) == "IP") {
            CheckDiskPQIPIteratorAndRange(index, query, decoded_scores);
        }
        uncached_size = index.Size();
    }
    // Control cache membership: the front half of the graph candidates is cached, the rest is not, which
    // exercises cached and uncached brute-force refinement together.
    std::vector<uint32_t> cache_order;
    std::unordered_set<uint32_t> candidates;
    for (size_t i = 0; i < k; ++i) {
        candidates.insert(graph->GetIds()[i]);
        if (i < k / 2) {
            cache_order.push_back(graph->GetIds()[i]);
        }
    }
    for (uint32_t id = 0; id < rows && cache_order.size() < rows / 3; ++id) {
        if (candidates.count(id) == 0) {
            cache_order.push_back(id);
        }
    }
    diskann::save_bin<uint32_t>(diskann::get_cached_nodes_file(prefix), cache_order.data(), cache_order.size(), 1);
    {
        // With a zero budget nothing is generated, so any cache comes from the saved list.
        auto index = IndexFactory::Instance().Create<fp32>(type, version, pack).value();
        REQUIRE(index.Deserialize(binary, load) == Status::success);
        REQUIRE(index.Size() > uncached_size);
        auto cached_graph = index.Search(
            query, {{"metric_type", metric_type}, {"k", k}, {"search_list", rows}, {"beamwidth", 4}}, nullptr);
        REQUIRE(cached_graph.has_value());
        for (size_t i = 0; i < k; ++i) {
            CheckDiskPQScore(decoded_scores, cached_graph.value()->GetIds()[i], cached_graph.value()->GetDistance()[i]);
        }
        CheckForcedDiskPQSearch(index, query, graph, rows, metric_type, &decoded_scores);
        if (std::string(type) != "AISAQ" && std::string(metric_type) == "IP") {
            CheckDiskPQIPIteratorAndRange(index, query, decoded_scores);
        }
    }
    fs::remove_all(directory);
}
