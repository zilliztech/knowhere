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
#include "utils.h"

namespace {
// Decode the actual SSD payload for an independent scalar score reference.
class DiskPQReference : public diskann::PQFlashIndex<float> {
 public:
    explicit DiskPQReference(diskann::Metric metric)
        : PQFlashIndex(std::make_shared<LinuxAlignedFileReader>(), metric) {
    }

    float Score(const std::string& prefix, unsigned id, const float* query, size_t dim) {
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
        if (metric == diskann::Metric::INNER_PRODUCT) return dot * max_base_norm;
        if (metric == diskann::Metric::COSINE) return dot / std::sqrt(norm);
        return l2;
    }
};
}  // namespace

TEST_CASE("DiskANN SSD PQ scores are independent of navigation and cache", "[diskann][ssd_pq]") {
    const auto metric = GENERATE(std::string("L2"), std::string("IP"), std::string("COSINE"));
    const auto version = GenTestVersionList();
    constexpr size_t rows = 300, dim = 16, nq = 3, k = 10;
    const auto dir = std::filesystem::current_path() / ("ssd_pq_regression_" + metric);
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
#ifdef KNOWHERE_WITH_CARDINAL
    const char* index_type = "DISKANN_DEPRECATED";
#else
    const char* index_type = "DISKANN";
#endif
    knowhere::Json config = {{"dim", dim}, {"metric_type", metric}, {"index_prefix", prefix}, {"data_path", raw},
                             {"max_degree", 24}, {"search_list_size", 100}, {"pq_code_budget_gb", 0.001},
                             {"build_dram_budget_gb", 1.0}, {"disk_pq_dims", 4},
                             {"search_cache_budget_gb", 0}, {"search_cache_budget_gb_ratio", 0}};
    auto built = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, pack).value();
    REQUIRE(built.Build(nullptr, config) == knowhere::Status::success);
    knowhere::BinarySet binary;
    REQUIRE(built.Serialize(binary) == knowhere::Status::success);
    const auto dm = metric == "L2" ? diskann::Metric::L2
                                   : metric == "IP" ? diskann::Metric::INNER_PRODUCT : diskann::Metric::COSINE;
    const auto* xq = static_cast<const float*>(queries->GetTensor());
    for (bool cached : {false, true}) {
        auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, pack).value();
        config["search_cache_budget_gb"] = cached ? 0.000005 : 0.0;
        config["use_bfs_cache"] = true;
        config["warm_up"] = true;
        REQUIRE(index.Deserialize(binary, config) == knowhere::Status::success);
        REQUIRE_FALSE(index.HasRawData(metric));
        int64_t id = 0;
        REQUIRE_FALSE(index.GetVectorByIds(knowhere::GenIdsDataSet(1, &id)).has_value());
        DiskPQReference reference(dm);
        REQUIRE(reference.load(1, prefix.c_str()) == 0);
        if (cached) {
            std::vector<uint32_t> ids(60);
            std::iota(ids.begin(), ids.end(), 0);
            reference.load_cache_list(ids);
        }
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
                    REQUIRE(label < rows);
                    const auto expected = reference.Score(prefix, label, xq + q * dim, dim);
                    REQUIRE(result.value()->GetDistance()[offset] == Catch::Approx(expected).epsilon(0.0002).margin(0.0002));
                }
            }
        }
        // Exercise both cached and uncached IDs in the independent rerank API.
        int64_t ids[] = {1, 12, 61, 299};
        float distances[4];
        reference.calc_dist_by_ids(xq, ids, 4, distances);
        for (size_t j = 0; j < 4; ++j)
            REQUIRE(distances[j] == Catch::Approx(reference.Score(prefix, ids[j], xq, dim)).epsilon(0.0002).margin(0.0002));
    }
    std::filesystem::remove_all(dir);
}
#endif
