// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifdef KNOWHERE_WITH_DISKANN
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/pq_flash_index.h"
#include "filemanager/impl/LocalFileManager.h"
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
class DiskPQReference : public diskann::PQFlashIndex<float> {
 public:
    explicit DiskPQReference(diskann::Metric metric)
        : PQFlashIndex(std::make_shared<LinuxAlignedFileReader>(), metric) {
    }

    void
    CheckExternalResources() {
        REQUIRE(data == nullptr);
        REQUIRE(pq_table.get_total_dims() == 0);
        auto slot = thread_data.pop();
        REQUIRE(slot.scratch.aligned_pq_coord_scratch == nullptr);
        REQUIRE(slot.scratch.aligned_pqtable_dist_scratch == nullptr);
        REQUIRE(slot.scratch.coord_scratch != nullptr);
        thread_data.push(slot);
    }

    float
    Score(const std::string& prefix, unsigned id, const float* query, size_t dim) {
        std::ifstream file(prefix + "_disk.index", std::ios::binary);
        file.seekg(get_node_sector_offset(id) + (long_node ? 0 : (id % nnodes_per_sector) * max_node_len));
        std::vector<uint8_t> code(disk_pq_n_chunks);
        file.read(reinterpret_cast<char*>(code.data()), code.size());
        REQUIRE(file.good());
        std::vector<float> decoded(data_dim);
        disk_pq_table.inflate_vector(code.data(), decoded.data());
        double dot = 0, norm = 0, l2 = 0;
        for (size_t j = 0; j < dim; ++j) {
            dot += double(query[j]) * decoded[j];
            norm += double(query[j]) * query[j];
            l2 += (double(query[j]) - decoded[j]) * (double(query[j]) - decoded[j]);
        }
        if (metric == diskann::Metric::INNER_PRODUCT)
            return dot * max_base_norm;
        if (metric == diskann::Metric::COSINE)
            return dot / std::sqrt(norm);
        return l2;
    }
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
        if (disk_pq_dims > 0)
            REQUIRE_FALSE(index.GetVectorByIds(knowhere::GenIdsDataSet(1, &id)).has_value());
        else if (has_raw)
            REQUIRE(index.GetVectorByIds(knowhere::GenIdsDataSet(1, &id)).has_value());
        DiskPQReference reference(dm);
        const DiskPQReference::NavigationMetadata metadata{rows, dim + (metric == "IP" ? 1 : 0)};
        REQUIRE(reference.load(1, prefix.c_str(), !external, external ? &metadata : nullptr) == 0);
        if (external)
            reference.CheckExternalResources();
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
            auto workspace = reference.getIteratorWorkspace(xq, rows + 1, 4, 0, knowhere::BitsetView());
            reference.getIteratorNextBatch(workspace.get());
            REQUIRE_FALSE(workspace->has_candidates());
            REQUIRE(workspace->full_retset.empty());
            REQUIRE(workspace->next_count == 0);
            REQUIRE(workspace->backup_res.size() > k);
            for (const auto& item : workspace->backup_res) {
                REQUIRE(-item.val == Catch::Approx(expected_score(item.id, xq)).epsilon(0.0002).margin(0.0002));
            }
        }
    }
    std::filesystem::remove_all(dir);
}
#endif
