// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifdef KNOWHERE_WITH_DISKANN
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define KNOWHERE_TEST_ASAN_ALLOCATOR
#endif
#endif
#if defined(__GLIBC__)
#include <malloc.h>
#if __GLIBC_PREREQ(2, 33) && !defined(__SANITIZE_ADDRESS__) && !defined(KNOWHERE_TEST_ASAN_ALLOCATOR)
#define KNOWHERE_TEST_HAS_MALLINFO2
#endif
#endif

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/pq_flash_index.h"
#include "diskann/pq_navigation.h"
#include "diskann/pq_table.h"
#include "filemanager/impl/LocalFileManager.h"
#include "index/diskann/diskann_config.h"
#include "index/diskann/navigation_store.h"
#include "index/diskann/rabitq_store.h"
#include "knowhere/feder/DiskANN.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_static.h"
#include "utils.h"

namespace {
#ifdef KNOWHERE_WITH_CARDINAL
constexpr const char* kDiskIndexType = "DISKANN_DEPRECATED";
#else
constexpr const char* kDiskIndexType = "DISKANN";
#endif

// Decode the actual SSD payload for an independent scalar score reference.
class DiskPQReference : public diskann::SSDGraphIndex<float> {
 public:
    explicit DiskPQReference(diskann::Metric metric)
        : SSDGraphIndex(std::make_shared<LinuxAlignedFileReader>(), metric) {
    }

    void
    CheckGraphResources() {
        auto slot = thread_data.pop();
        REQUIRE(slot.scratch.coord_scratch != nullptr);
        thread_data.push(slot);
    }

    float
    Score(const std::string& prefix, unsigned id, const float* query, size_t dim) {
        std::ifstream file(prefix + "_disk.index", std::ios::binary);
        file.seekg(get_node_sector_offset(id) + (long_node ? 0 : (id % nnodes_per_sector) * max_node_len));
        if (!reference_table_) {
            reference_table_ = std::make_unique<diskann::FixedChunkPQTable>();
            reference_table_->load_pq_centroid_bin((prefix + "_disk.index_pq_pivots.bin").c_str(), 0);
        }
        std::vector<uint8_t> code(reference_table_->get_num_chunks());
        file.read(reinterpret_cast<char*>(code.data()), code.size());
        REQUIRE(file.good());
        std::vector<float> decoded(data_dim);
        reference_table_->inflate_vector(code.data(), decoded.data());
        double dot = 0, norm = 0, l2 = 0;
        for (size_t j = 0; j < dim; ++j) {
            dot += double(query[j]) * decoded[j];
            norm += double(query[j]) * query[j];
            l2 += (double(query[j]) - decoded[j]) * (double(query[j]) - decoded[j]);
        }
        if (metric == diskann::Metric::INNER_PRODUCT)
            return dot * storage_codec_->MaxBaseNorm();
        if (metric == diskann::Metric::COSINE)
            return dot / std::sqrt(norm);
        return l2;
    }

 private:
    std::unique_ptr<diskann::FixedChunkPQTable> reference_table_;
};
// Read the persisted layout independently of either graph searcher's scorer.
class AisaqPayloadReference {
 public:
    explicit AisaqPayloadReference(const std::string& prefix) : file_(prefix + "_disk.index", std::ios::binary) {
        // AiSAQ starts with two uint32 shape fields, then count/dim/medoid/layout.
        uint64_t header[12];
        file_.read(reinterpret_cast<char*>(header), sizeof(header));
        REQUIRE(file_.good());
        node_bytes_ = header[4];
        nodes_per_sector_ = header[5];
        table_.load_pq_centroid_bin((prefix + "_disk.index_pq_pivots.bin").c_str(), 0);
        const auto map_file = prefix + "_disk.index_rearrange.bin";
        if (header[11] && std::filesystem::exists(map_file)) {
            size_t rows, dim;
            std::unique_ptr<uint32_t[]> mapping;
            diskann::load_bin<uint32_t>(map_file, mapping, rows, dim);
            REQUIRE(dim == 1);
            physical_ids_.resize(rows);
            for (size_t i = 0; i < rows; ++i) physical_ids_[mapping[i]] = i;
        }
        const auto norm_file = prefix + "_disk.index_max_base_norm.bin";
        if (std::filesystem::exists(norm_file)) {
            size_t rows, dim;
            std::unique_ptr<float[]> norms;
            diskann::load_bin<float>(norm_file, norms, rows, dim);
            if (rows == 1)
                max_norm_ = norms[0];
        }
    }

    float
    Score(unsigned id, const float* query, size_t dim, const std::string& metric) {
        const uint64_t physical = physical_ids_.empty() ? id : physical_ids_.at(id);
        const uint64_t offset =
            nodes_per_sector_
                ? (1 + physical / nodes_per_sector_) * diskann::defaults::SECTOR_LEN +
                      (physical % nodes_per_sector_) * node_bytes_
                : (1 + physical * ((node_bytes_ + diskann::defaults::SECTOR_LEN - 1) / diskann::defaults::SECTOR_LEN)) *
                      diskann::defaults::SECTOR_LEN;
        file_.seekg(offset);
        std::vector<uint8_t> code(table_.get_num_chunks());
        file_.read(reinterpret_cast<char*>(code.data()), code.size());
        REQUIRE(file_.good());
        std::vector<float> decoded(table_.get_total_dims());
        table_.inflate_vector(code.data(), decoded.data());
        double dot = 0, query_norm = 0, l2 = 0;
        for (size_t j = 0; j < dim; ++j) {
            dot += double(query[j]) * decoded[j];
            query_norm += double(query[j]) * query[j];
            l2 += (double(query[j]) - decoded[j]) * (double(query[j]) - decoded[j]);
        }
        if (metric == "IP")
            return dot * max_norm_;
        if (metric == "COSINE")
            return dot / std::sqrt(query_norm);
        return l2;
    }

 private:
    std::ifstream file_;
    diskann::FixedChunkPQTable table_;
    uint64_t node_bytes_, nodes_per_sector_;
    float max_norm_ = 1;
    std::vector<uint32_t> physical_ids_;
};

}  // namespace

TEST_CASE("DiskANN public static configuration and raw data capabilities", "[diskann][ssd_pq][diskann_review]") {
    const auto version = GenTestVersionList();
    const auto metric = GENERATE(std::string("L2"), std::string("IP"), std::string("COSINE"));
    // Native PQ, codec-selected RBQ, and the fixed RBQ alias.
    const int codec = GENERATE(0, 1, 2);
    const auto* type = codec == 2 ? "DISKANN_RABITQ" : kDiskIndexType;
    knowhere::Json config = {{"dim", 16}, {"metric_type", metric}};
    if (codec == 1)
        config["navigation_codec"] = "RABITQ";
    std::string error;
    using Static = knowhere::IndexStaticFaced<knowhere::fp32>;
    REQUIRE(Static::ConfigCheck(type, version, config, error) == knowhere::Status::success);
    REQUIRE(Static::HasRawData(type, version, config) == (metric != "IP"));
    config["emb_list_strategy"] = "tokenann";
    REQUIRE(Static::ConfigCheck(type, version, config, error) == knowhere::Status::success);
    for (int disk_bits : {0, 4}) {
        config["disk_pq_dims"] = disk_bits;
        REQUIRE(Static::HasRawData(type, version, config) == (disk_bits == 0 && metric != "IP"));
    }
    if (codec != 0) {
        config["emb_list_offset_file_path"] = "unused_offsets.bin";
        REQUIRE(Static::ConfigCheck(type, version, config, error) == knowhere::Status::not_implemented);
        config.erase("emb_list_offset_file_path");
        config["metric_type"] = "MAX_SIM_" + metric;
        REQUIRE(Static::ConfigCheck(type, version, config, error) != knowhere::Status::success);
    }
}

TEST_CASE("DiskANN iterator uses the same score conversion for normal and final batches",
          "[diskann][ssd_pq][diskann_review]") {
    const auto metric = GENERATE(diskann::Metric::L2, diskann::Metric::INNER_PRODUCT, diskann::Metric::COSINE);
    const float base_norm = GENERATE(0.0f, 1.0f, 7.0f);
    const float query[] = {3, 4, 0, 0};
    diskann::IteratorWorkspace<float> workspace(query, metric, 8, metric == diskann::Metric::INNER_PRODUCT ? 5 : 4, 1,
                                                1, 1, 0, base_norm, knowhere::BitsetView());
    for (float score : {0.4f, 2.8f}) {
        CAPTURE(metric, base_norm, score);
        const float expected =
            metric == diskann::Metric::INNER_PRODUCT ? (score / 2 - 1) * (base_norm != 0 ? base_norm * 5 : 1) : score;
        workspace.good_pq_res_count = workspace.next_count + workspace.lsearch;
        workspace.insert_to_full(10, score);
        workspace.move_full_retset_to_backup();
        REQUIRE(workspace.backup_res.size() == 1);
        REQUIRE(workspace.backup_res[0].val == Catch::Approx(expected));
        workspace.backup_res.clear();
        workspace.insert_to_full(10, score);
        workspace.move_last_full_retset_to_backup();
        REQUIRE(workspace.backup_res.size() == 1);
        REQUIRE(workspace.backup_res[0].val == Catch::Approx(expected));
        REQUIRE(workspace.full_retset.empty());
        workspace.backup_res.clear();
    }
}

TEST_CASE("DiskANN SSD scores are independent of navigation and cache", "[diskann][ssd_pq][diskann_review]") {
    const auto metric = GENERATE(std::string("L2"), std::string("IP"), std::string("COSINE"));
    const int codec = GENERATE(0, 1, 2);
    const bool external = codec != 0;
    const int disk_pq_dims = GENERATE(0, 4);
    const auto version = GenTestVersionList();
    constexpr size_t rows = 300, dim = 16, nq = 3, k = 10;
    const auto dir = std::filesystem::current_path() /
                     ("ssd_pq_regression_" + metric + "_" + std::to_string(codec) + "_" + std::to_string(disk_pq_dims));
    std::filesystem::create_directories(dir);
    const auto prefix = (dir / "index").string();
    const auto raw = (dir / "base.bin").string();
    auto base = GenDataSet(rows, dim, 73);
    auto queries = GenDataSet(nq, dim, 29);
    auto* xb = const_cast<float*>(static_cast<const float*>(base->GetTensor()));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < dim; ++j) xb[i * dim + j] *= 0.2f + float(i % 7);
    {
        std::ofstream file(raw, std::ios::binary);
        uint32_t header[] = {rows, dim};
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
        file.write(reinterpret_cast<const char*>(xb), rows * dim * sizeof(float));
    }
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(std::make_shared<milvus::LocalFileManager>()));
    const char* index_type = codec == 2 ? "DISKANN_RABITQ" : kDiskIndexType;
    knowhere::Json config = {{"dim", dim},
                             {"metric_type", metric},
                             {"index_prefix", prefix},
                             {"data_path", raw},
                             {"max_degree", 24},
                             {"search_list_size", 100},
                             {"pq_code_budget_gb", 0.001},
                             {"build_dram_budget_gb", 1.0},
                             {"disk_pq_dims", disk_pq_dims},
                             {"search_cache_budget_gb", 0},
                             {"search_cache_budget_gb_ratio", 0}};
    if (external) {
        config["navigation_codec"] = "RABITQ";
        config["rbq_bits"] = 4;
        // Omitted and explicit-zero navigation budgets must both work,
        // regardless of whether SSD PQ is enabled.
        config.erase("pq_code_budget_gb");
        if (codec == 2) {
            config["pq_code_budget_gb"] = 0;
            config["pq_code_budget_gb_ratio"] = 0;
        }
    }
    auto built = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, pack).value();
    if (metric == "L2" && disk_pq_dims == 0) {
        auto invalid = config;
        invalid["build_dram_budget_gb"] = 0;
        REQUIRE(built.Build(nullptr, invalid) != knowhere::Status::success);
        if (!external) {
            invalid = config;
            invalid["pq_code_budget_gb"] = 0;
            REQUIRE(built.Build(nullptr, invalid) != knowhere::Status::success);
        }
    }
    REQUIRE(built.Build(nullptr, config) == knowhere::Status::success);
    knowhere::BinarySet binary;
    REQUIRE(built.Serialize(binary) == knowhere::Status::success);
    REQUIRE(std::filesystem::exists(prefix + "_pq_compressed.bin") == !external);
    REQUIRE(std::filesystem::exists(prefix + "_pq_pivots.bin") == !external);
    REQUIRE(std::filesystem::exists(prefix + "_disk.index_pq_pivots.bin") == (disk_pq_dims > 0));
    const auto dm = metric == "L2"   ? diskann::Metric::L2
                    : metric == "IP" ? diskann::Metric::INNER_PRODUCT
                                     : diskann::Metric::COSINE;
    const auto* xq = static_cast<const float*>(queries->GetTensor());
    for (bool cached : {false, true}) {
        auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, pack).value();
        config["search_cache_budget_gb"] = cached ? 0.000005 : 0.0;
        config["use_bfs_cache"] = true;
        config["warm_up"] = true;
        REQUIRE(index.Deserialize(binary, config) == knowhere::Status::success);
        const bool has_raw = disk_pq_dims == 0 && metric != "IP";
        REQUIRE(index.HasRawData(metric) == has_raw);
        REQUIRE(knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData(index_type, version, config) == has_raw);
        int64_t id = 0;
        if (!has_raw) {
            auto original = index.GetVectorByIds(knowhere::GenIdsDataSet(1, &id));
            REQUIRE_FALSE(original.has_value());
            REQUIRE(original.error() == knowhere::Status::not_implemented);
        } else if (has_raw)
            REQUIRE(index.GetVectorByIds(knowhere::GenIdsDataSet(1, &id)).has_value());
        DiskPQReference reference(dm);
        const diskann::NavigationSpace metadata{rows, dim + (metric == "IP" ? 1 : 0), dim};
        REQUIRE(reference.load(1, prefix.c_str(), metadata) == 0);
        reference.CheckGraphResources();
        std::unique_ptr<diskann::NavigationStore> navigation;
        if (external)
            navigation = std::make_unique<knowhere::RaBitQStore>(prefix + "_rabitq.index");
        else
            navigation = std::make_unique<diskann::PQNavigationStore>(prefix);
        REQUIRE(navigation != nullptr);
        REQUIRE(navigation->Count() == rows);
        REQUIRE(navigation->Dimension() == metadata.dimension);
        if (cached) {
            std::vector<uint32_t> ids(60);
            std::iota(ids.begin(), ids.end(), 0);
            reference.load_cache_list(ids);
        }
        const auto expected_score = [&](unsigned id, const float* query) {
            if (disk_pq_dims > 0)
                return reference.Score(prefix, id, query, dim);
            double dot = 0, nq = 0, nb = 0, l2 = 0;
            for (size_t j = 0; j < dim; ++j) {
                const double b = xb[id * dim + j], q = query[j];
                dot += b * q;
                nq += q * q;
                nb += b * b;
                l2 += (q - b) * (q - b);
            }
            return float(metric == "L2" ? l2 : metric == "IP" ? dot : dot / std::sqrt(nq * nb));
        };
        knowhere::Json search = {{"metric_type", metric}, {"k", k}, {"search_list_size", 128}, {"beamwidth", 4}};
        for (size_t filtered : {size_t(0), size_t(285)}) {
            std::vector<uint8_t> mask((rows + 7) / 8, 0);
            for (size_t i = 0; i < filtered; ++i) mask[i / 8] |= 1u << (i % 8);
            auto result = index.Search(queries, search, knowhere::BitsetView(mask.data(), rows));
            REQUIRE(result.has_value());
            for (size_t q = 0; q < nq; ++q) {
                for (size_t j = 0; j < k; ++j) {
                    const auto offset = q * k + j;
                    const auto label = result.value()->GetIds()[offset];
                    REQUIRE(label >= static_cast<int64_t>(filtered));
                    REQUIRE(label < int64_t(rows));
                    const auto expected = expected_score(label, xq + q * dim);
                    REQUIRE(result.value()->GetDistance()[offset] ==
                            Catch::Approx(expected).epsilon(0.0002).margin(0.0002));
                }
            }
        }
        // Exercise both cached and uncached IDs in the independent rerank API.
        int64_t ids[] = {1, 12, 61, 299};
        float distances[4];
        reference.calc_dist_by_ids(xq, ids, 4, distances);
        for (size_t j = 0; j < 4; ++j)
            REQUIRE(distances[j] == Catch::Approx(expected_score(ids[j], xq)).epsilon(0.0002).margin(0.0002));
        if (!external && metric == "IP") {
            // Small L exercises ordinary batches and the tail; L > rows forces
            // exhaustion before the ordinary batch threshold is reached.
            for (int iterator_l : {2, int(rows + 1)}) {
                auto iterator_config = search;
                iterator_config["search_list_size"] = iterator_l;
                auto result = index.AnnIterator(queries, iterator_config, knowhere::BitsetView(), false);
                REQUIRE(result.has_value());
                for (size_t q = 0; q < nq; ++q) {
                    auto& iterator = result.value()[q];
                    std::vector<bool> seen(rows, false);
                    size_t count = 0;
                    while (iterator->HasNext().value()) {
                        const auto [id, score] = iterator->Next().value();
                        REQUIRE(id >= 0);
                        REQUIRE(id < int64_t(rows));
                        REQUIRE_FALSE(seen[id]);
                        seen[id] = true;
                        ++count;
                        REQUIRE(score ==
                                Catch::Approx(expected_score(id, xq + q * dim)).epsilon(0.0002).margin(0.0002));
                    }
                    // Exhausting an approximate graph traversal need not
                    // enumerate disconnected/unreachable base vectors.
                    REQUIRE(count > k);
                    REQUIRE(count <= rows);
                    REQUIRE_FALSE(iterator->HasNext().value());
                }
            }
            // Explicitly observe the low-level exhausted-candidate path,
            // rather than relying only on the public iterator's buffering.
            auto workspace = reference.getIteratorWorkspace(xq, rows + 1, 4, 0, knowhere::BitsetView(),
                                                            navigation->CreateDistanceComputer());
            reference.getIteratorNextBatch(workspace.get());
            REQUIRE_FALSE(workspace->has_candidates());
            REQUIRE(workspace->full_retset.empty());
            REQUIRE(workspace->next_count == 0);
            REQUIRE(workspace->backup_res.size() > k);
            for (const auto& item : workspace->backup_res) {
                REQUIRE(-item.val == Catch::Approx(expected_score(item.id, xq)).epsilon(0.0002).margin(0.0002));
            }
            struct ThrowOnceNavigation final : diskann::NavigationDistanceComputer {
                std::unique_ptr<diskann::NavigationDistanceComputer> delegate;
                bool fail = true;
                explicit ThrowOnceNavigation(std::unique_ptr<diskann::NavigationDistanceComputer> scorer)
                    : delegate(std::move(scorer)) {
                }
                void
                set_query(const float* query) override {
                    delegate->set_query(query);
                }
                void
                compute_distances(const unsigned* ids, uint64_t count, float* output, float threshold, bool valid,
                                  diskann::QueryStats* stats) override {
                    if (fail) {
                        fail = false;
                        throw std::runtime_error("iterator navigation failure");
                    }
                    delegate->compute_distances(ids, count, output, threshold, valid, stats);
                }
            };
            auto retry = reference.getIteratorWorkspace(
                xq, rows + 1, 4, 0, knowhere::BitsetView(),
                std::make_unique<ThrowOnceNavigation>(navigation->CreateDistanceComputer()));
            REQUIRE_THROWS(reference.getIteratorNextBatch(retry.get()));
            // There is only one IO scratch slot. Retrying also verifies that
            // the failed batch returned both the slot and the read context.
            REQUIRE_NOTHROW(reference.getIteratorNextBatch(retry.get()));
            REQUIRE(retry->backup_res.size() > k);
        }
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("AiSAQ SSD PQ scores match decoded payloads across search paths", "[diskann][aisaq_review][aisaq_score]") {
    const auto metric = GENERATE(std::string("IP"), std::string("L2"), std::string("COSINE"));
    const bool rearranged = GENERATE(false, true);
    const int inline_pq = GENERATE(0, 4);
    constexpr size_t rows = 300, dim = 16, nq = 3, k = 10;
    const auto dir = std::filesystem::current_path() /
                     ("aisaq_scores_" + metric + "_" + std::to_string(rearranged) + "_" + std::to_string(inline_pq));
    std::filesystem::create_directories(dir);
    const auto prefix = (dir / "index").string();
    auto base = GenDataSet(rows, dim, 73);
    auto queries = GenDataSet(nq, dim, 29);
    auto* xb = const_cast<float*>(static_cast<const float*>(base->GetTensor()));
    auto* xq = const_cast<float*>(static_cast<const float*>(queries->GetTensor()));
    for (size_t i = 0; i < rows * dim; ++i) xb[i] *= 0.2f + float((i / dim) % 7);
    for (size_t i = 0; i < nq * dim; ++i) xq[i] *= 0.1f + 3 * (i / dim);
    const auto raw = (dir / "base.bin").string();
    WriteRawDataToDisk<float>(raw, xb, rows, dim);
    auto manager = std::make_shared<milvus::LocalFileManager>();
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(manager));
    const auto version = GenTestVersionList();
    knowhere::Json config = {{"dim", dim},
                             {"metric_type", metric},
                             {"index_prefix", prefix},
                             {"data_path", raw},
                             {"max_degree", 24},
                             {"search_list_size", 100},
                             {"pq_code_budget_gb", 0.000002},
                             {"build_dram_budget_gb", 1},
                             {"disk_pq_dims", 4},
                             {"search_cache_budget_gb", 0},
                             {"search_cache_budget_gb_ratio", 0},
                             {"rearrange", rearranged},
                             {"inline_pq", inline_pq},
                             {"pq_cache_size", 0},
                             {"pq_read_page_cache_size", 0},
                             {"num_entry_points", 0},
                             {"warm_up", false},
                             {"use_bfs_cache", true}};
    auto built = knowhere::IndexFactory::Instance().Create<knowhere::fp32>("AISAQ", version, pack).value();
    REQUIRE(built.Build(nullptr, config) == knowhere::Status::success);
    AisaqPayloadReference reference(prefix);
    for (int cache_mode : {0, 1, 2}) {
        if (cache_mode) {
            std::vector<uint32_t> nodes;
            for (uint32_t i = 0; i < rows; ++i)
                if (cache_mode == 1 || i % 2 == 0)
                    nodes.push_back(i);
            const auto cached = diskann::get_cached_nodes_file(prefix);
            std::filesystem::remove(cached);
            diskann::save_bin<uint32_t>(cached, nodes.data(), nodes.size(), 1);
            REQUIRE(manager->AddFile(cached));
        }
        auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>("AISAQ", version, pack).value();
        knowhere::BinarySet empty;
        REQUIRE(index.Deserialize(empty, config) == knowhere::Status::success);
        knowhere::Json search = {{"metric_type", metric},   {"k", k},
                                 {"search_list_size", 128}, {"beamwidth", 4},
                                 {"vectors_beamwidth", 1},  {"filter_threshold", 0.5}};
        for (size_t filtered : {size_t(0), rows - 15}) {
            std::vector<uint8_t> mask((rows + 7) / 8, 0);
            for (size_t i = 0; i < filtered; ++i) mask[i / 8] |= 1u << (i % 8);
            auto result = index.Search(queries, search, knowhere::BitsetView(mask.data(), rows));
            REQUIRE(result.has_value());
            for (size_t q = 0; q < nq; ++q)
                for (size_t j = 0; j < k; ++j) {
                    const auto at = q * k + j;
                    const auto id = result.value()->GetIds()[at];
                    REQUIRE(id >= int64_t(filtered));
                    REQUIRE(id < int64_t(rows));
                    CAPTURE(metric, rearranged, inline_pq, cache_mode, filtered, q, id);
                    REQUIRE(
                        result.value()->GetDistance()[at] ==
                        Catch::Approx(reference.Score(id, xq + q * dim, dim, metric)).epsilon(0.0002).margin(0.0002));
                }
        }
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("AiSAQ raw-vector declarations and calls agree with SSD encoding", "[diskann][aisaq_review][aisaq_raw]") {
    const auto metric = GENERATE(std::string("L2"), std::string("COSINE"));
    const int disk_pq_dims = GENERATE(0, 4);
    constexpr uint32_t rows = 300, dim = 16;
    const auto dir = std::filesystem::current_path() / ("aisaq_raw_" + metric + "_" + std::to_string(disk_pq_dims));
    std::filesystem::create_directories(dir);
    const auto prefix = (dir / "index").string();
    auto base = GenDataSet(rows, dim, 51);
    const auto* xb = static_cast<const float*>(base->GetTensor());
    const auto raw = (dir / "base.bin").string();
    WriteRawDataToDisk<float>(raw, xb, rows, dim);
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(std::make_shared<milvus::LocalFileManager>()));
    const auto version = GenTestVersionList();
    knowhere::Json config = {{"dim", dim},
                             {"metric_type", metric},
                             {"index_prefix", prefix},
                             {"data_path", raw},
                             {"max_degree", 24},
                             {"search_list_size", 64},
                             {"pq_code_budget_gb", 0.000002},
                             {"build_dram_budget_gb", 1},
                             {"disk_pq_dims", disk_pq_dims},
                             {"search_cache_budget_gb", 0},
                             {"search_cache_budget_gb_ratio", 0},
                             {"inline_pq", 0},
                             {"pq_cache_size", 0},
                             {"num_entry_points", 0},
                             {"warm_up", false}};
    REQUIRE(knowhere::IndexStaticFaced<knowhere::fp32>::HasRawData("AISAQ", version, config) == (disk_pq_dims == 0));
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>("AISAQ", version, pack).value();
    REQUIRE(index.Build(nullptr, config) == knowhere::Status::success);
    auto load = config;
    load.erase("disk_pq_dims");  // Capability is determined by the stored model, not a load default.
    knowhere::BinarySet empty;
    REQUIRE(index.Deserialize(empty, load) == knowhere::Status::success);
    REQUIRE(index.HasRawData(metric) == (disk_pq_dims == 0));
    int64_t id = 17;
    auto retrieved = index.GetVectorByIds(knowhere::GenIdsDataSet(1, &id));
    if (disk_pq_dims) {
        REQUIRE_FALSE(retrieved.has_value());
        REQUIRE(retrieved.error() == knowhere::Status::not_implemented);
    } else {
        REQUIRE(retrieved.has_value());
        const auto* vector = static_cast<const float*>(retrieved.value()->GetTensor());
        for (size_t j = 0; j < dim; ++j) REQUIRE(vector[j] == xb[id * dim + j]);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("PQ iterators retain chunk-sized LUTs and share bounded gather buffers", "[diskann][pq_memory]") {
    constexpr uint32_t rows = 300, dim = 960, chunks = 64, count = 64;
    const auto dir = std::filesystem::current_path() / "pq_iterator_memory";
    std::filesystem::create_directories(dir);
    const auto prefix = (dir / "index").string();
    const auto raw = (dir / "base.bin").string();
    auto base = GenDataSet(rows, dim, 61);
    auto query = GenDataSet(2, dim, 31);
    WriteRawDataToDisk<float>(raw, static_cast<const float*>(base->GetTensor()), rows, dim);
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(std::make_shared<milvus::LocalFileManager>()));
    auto built =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(kDiskIndexType, GenTestVersionList(), pack).value();
    knowhere::Json config = {{"dim", dim},
                             {"metric_type", "L2"},
                             {"index_prefix", prefix},
                             {"data_path", raw},
                             {"max_degree", 16},
                             {"search_list_size", 32},
                             {"pq_code_budget_gb", rows * 64.25 / (1ULL << 30)},
                             {"build_dram_budget_gb", 1},
                             {"search_cache_budget_gb", 0},
                             {"search_cache_budget_gb_ratio", 0}};
    REQUIRE(built.Build(nullptr, config) == knowhere::Status::success);
    diskann::PQNavigationStore model(prefix);
    REQUIRE(model.ChunkCount() == chunks);
    model.PrepareWorkspaces(2);
    diskann::SSDGraphIndex<float> graph(std::make_shared<LinuxAlignedFileReader>(), diskann::Metric::L2);
    REQUIRE(graph.load(1, prefix.c_str(), {rows, dim, dim}) == 0);
    const auto reserved = model.MemorySize();
    constexpr size_t lut_bytes = 256 * chunks * sizeof(float);
    std::vector<std::unique_ptr<diskann::IteratorWorkspace<float>>> iterators;
    iterators.reserve(count);
    std::set<const float*> tables;
#ifdef KNOWHERE_TEST_HAS_MALLINFO2
    const auto before = mallinfo2();
#endif
    for (size_t i = 0; i < count; ++i) {
        const auto* q = static_cast<const float*>(query->GetTensor()) + (i % 2) * dim;
        auto workspace = graph.getIteratorWorkspace(q, 16, 4, 0, {}, model.CreatePQDistanceComputer());
        auto* scorer = dynamic_cast<diskann::PQNavigationDistanceComputer*>(workspace->navigation.get());
        REQUIRE(scorer != nullptr);
        tables.insert(scorer->lookup_table());
#if defined(__GLIBC__)
        // Inspect the allocator's actual block, independently of MemorySize().
        REQUIRE(malloc_usable_size(const_cast<float*>(scorer->lookup_table())) >= lut_bytes);
        REQUIRE(malloc_usable_size(const_cast<float*>(scorer->lookup_table())) <= lut_bytes + 256);
#endif
        iterators.push_back(std::move(workspace));
    }
    REQUIRE(tables.size() == count);
    REQUIRE(model.MemorySize() == reserved + (count - 2) * lut_bytes);
#ifdef KNOWHERE_TEST_HAS_MALLINFO2
    const auto active = mallinfo2();
    REQUIRE(active.uordblks + active.hblkhd >= before.uordblks + before.hblkhd + count * lut_bytes / 2);
#endif
    const auto active_size = model.MemorySize();
    for (auto& workspace : iterators) {
        graph.getIteratorNextBatch(workspace.get());
        REQUIRE_FALSE(workspace->backup_res.empty());
        REQUIRE(model.MemorySize() == active_size);  // Serial consumption never adds per-iterator gathers.
    }
    iterators.clear();
    REQUIRE(model.MemorySize() == reserved);  // Both idle pools are bounded by the prepared capacity.
#ifdef KNOWHERE_TEST_HAS_MALLINFO2
    const auto released = mallinfo2();
    REQUIRE(released.uordblks + released.hblkhd <= before.uordblks + before.hblkhd + (1ULL << 20));
#endif
    struct ThrowOnceSource : diskann::PQCodeSource {
        std::unique_ptr<uint8_t[]> codes;
        bool fail = true;
        void
        gather(const unsigned* ids, uint64_t n, uint64_t c, uint8_t* output) override {
            if (fail) {
                fail = false;
                throw std::runtime_error("gather failure");
            }
            diskann::aggregate_coords(ids, n, codes.get(), c, output);
        }
    } source;
    size_t n, c;
    diskann::load_bin<uint8_t>(diskann::get_pq_compressed_filename(prefix), source.codes, n, c);
    auto scorer = model.CreatePQDistanceComputer(&source);
    scorer->set_query(static_cast<const float*>(query->GetTensor()));
    const unsigned id = 0;
    float distance;
    REQUIRE_THROWS_AS(scorer->compute_distances(&id, 1, &distance, 0, false, nullptr), std::runtime_error);
    REQUIRE(model.MemorySize() == reserved);
    REQUIRE_NOTHROW(scorer->compute_distances(&id, 1, &distance, 0, false, nullptr));
    REQUIRE(model.MemorySize() == reserved);
    std::filesystem::remove_all(dir);
}

TEST_CASE("DiskANN public metadata follows the loaded navigation codec", "[diskann][navigation_metadata]") {
    constexpr uint32_t rows = 300, dim = 32;
    const auto codec = GENERATE(std::string("PQ"), std::string("RABITQ"));
    const auto dir = std::filesystem::current_path() / ("navigation_metadata_" + codec);
    std::filesystem::create_directories(dir);
    const auto prefix = (dir / "index").string();
    const auto raw = (dir / "base.bin").string();
    auto base = GenDataSet(rows, dim, 39);
    WriteRawDataToDisk<float>(raw, static_cast<const float*>(base->GetTensor()), rows, dim);
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(std::make_shared<milvus::LocalFileManager>()));
    auto built =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(kDiskIndexType, GenTestVersionList(), pack).value();
    knowhere::Json config = {{"dim", dim},
                             {"metric_type", "L2"},
                             {"index_prefix", prefix},
                             {"data_path", raw},
                             {"max_degree", 16},
                             {"search_list_size", 48},
                             {"build_dram_budget_gb", 1.5},
                             {"search_cache_budget_gb", 0},
                             {"search_cache_budget_gb_ratio", 0},
                             {"navigation_codec", codec},
                             {"rbq_bits", 4},
                             {"pq_code_budget_gb", 0.000002}};
    REQUIRE(built.Build(nullptr, config) == knowhere::Status::success);
    auto loaded =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(kDiskIndexType, GenTestVersionList(), pack).value();
    knowhere::BinarySet binary;
    // Generic DISKANN, no codec or build settings: use the actual stored model.
    REQUIRE(
        loaded.Deserialize(
            binary, {{"dim", dim}, {"metric_type", "L2"}, {"index_prefix", prefix}, {"search_cache_budget_gb", 0}}) ==
        knowhere::Status::success);
    auto result = loaded.GetIndexMeta(config);
    REQUIRE(result.has_value());
    auto meta = knowhere::Json::parse(result.value()->GetJsonInfo());
    REQUIRE(meta["num_elem_"] == rows);
    REQUIRE_FALSE(meta["entry_points_"].empty());
    REQUIRE(meta["entry_points_"] == knowhere::Json::parse(result.value()->GetJsonIdSet()));
    REQUIRE(meta["build_params_"]["data_path"] == raw);
    REQUIRE(meta["build_params_"]["max_degree"] == 16);
    REQUIRE(meta["build_params_"]["search_list_size"] == 48);
    REQUIRE(meta["build_params_"]["build_dram_budget_gb"] == 1.5);
    REQUIRE(meta["navigation_"]["codec"] == codec);
    if (codec == "PQ") {
        REQUIRE(meta["build_params_"]["pq_code_budget_gb"].get<double>() == Catch::Approx(0.000002));
        REQUIRE(meta["navigation_"]["pq_code_budget_gb"].get<double>() == Catch::Approx(0.000002));
        diskann::PQNavigationStore model(prefix);
        REQUIRE(meta["navigation_"]["chunks"] == model.ChunkCount());
        // Full legacy PQ consumers can still deserialize the old build keys.
        REQUIRE_NOTHROW(meta.get<knowhere::feder::diskann::DiskANNMeta>());
        auto ratio = config;
        ratio["pq_code_budget_gb"] = "0.25";
        ratio["pq_code_budget_gb_ratio"] = "0.5";
        ratio["vec_field_size_gb"] = 1;
        auto ratio_result = loaded.GetIndexMeta(ratio);
        REQUIRE(ratio_result.has_value());
        auto ratio_meta = knowhere::Json::parse(ratio_result.value()->GetJsonInfo());
        REQUIRE(ratio_meta["build_params_"]["pq_code_budget_gb"] == 0.5);
        // Missing field size leaves an adjusted ratio budget unknown.
        ratio.erase("vec_field_size_gb");
        auto incomplete = loaded.GetIndexMeta(ratio);
        REQUIRE(incomplete.has_value());
        REQUIRE(knowhere::Json::parse(incomplete.value()->GetJsonInfo())["navigation_"]["pq_code_budget_gb"].is_null());
        auto invalid = config;
        invalid["pq_code_budget_gb"] = -1;
        REQUIRE_FALSE(loaded.GetIndexMeta(invalid).has_value());
        config["pq_code_budget_gb"] = 0;
        auto zero = loaded.GetIndexMeta(config);
        REQUIRE(zero.has_value());
        REQUIRE(knowhere::Json::parse(zero.value()->GetJsonInfo())["navigation_"]["pq_code_budget_gb"] == 0);
    } else {
        REQUIRE(meta["navigation_"]["rbq_bits"] == 4);
        REQUIRE_FALSE(meta["build_params_"].contains("pq_code_budget_gb"));
        REQUIRE_FALSE(meta["navigation_"].contains("pq_code_budget_gb"));
        // A request's old/default selector and unrelated PQ knobs cannot relabel
        // a loaded RBQ model or make it parse PQ parameters.
        auto stale = config;
        stale["navigation_codec"] = "PQ";
        stale["rbq_bits"] = 1;
        stale["pq_code_budget_gb"] = "not-a-PQ-budget";
        auto stale_result = loaded.GetIndexMeta(stale);
        REQUIRE(stale_result.has_value());
        auto stale_meta = knowhere::Json::parse(stale_result.value()->GetJsonInfo());
        REQUIRE(stale_meta["navigation_"]["codec"] == "RABITQ");
        REQUIRE(stale_meta["navigation_"]["rbq_bits"] == 4);
        REQUIRE_FALSE(stale_meta["build_params_"].contains("pq_code_budget_gb"));
    }
    auto absent = loaded.GetIndexMeta({});
    REQUIRE(absent.has_value());
    auto absent_meta = knowhere::Json::parse(absent.value()->GetJsonInfo());
    REQUIRE(absent_meta["navigation_"]["codec"] == codec);
    REQUIRE_FALSE(absent_meta["build_params_"].contains("pq_code_budget_gb"));
    if (codec == "PQ")
        REQUIRE(absent_meta["navigation_"]["pq_code_budget_gb"].is_null());
    else
        REQUIRE(absent_meta["navigation_"]["rbq_bits"] == 4);
    std::filesystem::remove_all(dir);
}

TEST_CASE("DiskANN metadata rejects unloaded and mismatched models", "[diskann][navigation_metadata]") {
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(std::make_shared<milvus::LocalFileManager>()));
    auto index =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(kDiskIndexType, GenTestVersionList(), pack).value();
    auto result = index.GetIndexMeta({});
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error() == knowhere::Status::empty_index);
    struct Model : knowhere::NavigationStore {
        int64_t
        Count() const override {
            return 1;
        }
        int64_t
        Dimension() const override {
            return 1;
        }
        size_t
        MemorySize() const override {
            return 0;
        }
        std::unique_ptr<diskann::NavigationDistanceComputer>
        CreateDistanceComputer() const override {
            return {};
        }
    } model;
    knowhere::DiskANNConfig config;
    for (const auto* codec : {"PQ", "RABITQ", "unknown"}) {
        auto metadata = knowhere::GetNavigationIndexMetadata(codec, model, config);
        REQUIRE_FALSE(metadata.has_value());
        REQUIRE(metadata.error() == knowhere::Status::invalid_index_error);
    }
}

#endif
