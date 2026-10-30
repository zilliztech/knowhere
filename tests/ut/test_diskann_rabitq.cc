// Copyright (C) 2026 Zilliz. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <faiss/IndexFlat.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/VectorTransform.h>
#include <faiss/cppcontrib/knowhere/index_io.h>
#include <faiss/impl/RaBitQuantizer.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "diskann/aux_utils.h"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/rabitq_navigation.h"
#include "diskann/ssd_graph_index.h"
#include "filemanager/impl/LocalFileManager.h"
#include "index/diskann/aisaq_config.h"
#include "index/diskann/rabitq_navigation_config.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/context.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {
#ifdef KNOWHERE_WITH_CARDINAL
constexpr const char* kNativeDiskANNType = "DISKANN_DEPRECATED";
#else
constexpr const char* kNativeDiskANNType = "DISKANN";
#endif
namespace fs = std::filesystem;
struct RaBitQFixture {
    static constexpr size_t rows = 512, nq = 3;
    const size_t dim;
    fs::path directory = fs::current_path() / "diskann_rabitq_test";
    std::string prefix = (directory / "index").string();
    std::string source = (directory / "base.bin").string();
    std::vector<float> base, query;
    explicit RaBitQFixture(size_t dimension = 16) : dim(dimension), base(rows * dim), query(nq * dim) {
        fs::remove_all(directory);
        fs::create_directories(directory);
        std::mt19937 rng(20261015);
        for (auto& x : base) x = float(rng() % 65536) / 32768 - 1;
        for (auto& x : query) x = float(rng() % 65536) / 32768 - 1;
        diskann::save_bin<float>(source, base.data(), rows, dim);
    }
    ~RaBitQFixture() {
        fs::remove_all(directory);
    }
    knowhere::Json
    BuildConfig(const std::string& metric, int ssd_pq = 0) const {
        return {{"metric_type", metric},
                {"quant_type", "RABITQ"},
                {"dim", dim},
                {"data_path", source},
                {"index_prefix", prefix},
                {"max_degree", 16},
                {"search_list_size", 64},
                {"build_dram_budget_gb", 1.0},
                {"disk_pq_dims", ssd_pq},
                {"vec_field_size_gb", double(rows * dim * sizeof(float)) / (1 << 30)}};
    }
    diskann::NavigationBuildContext
    Context() const {
        return {prefix, source, knowhere::DataFormatEnum::fp32, diskann::Metric::L2, rows, dim, nullptr, 0};
    }
};

auto
CreateDiskANN(const std::string& type, int version,
              std::shared_ptr<milvus::FileManager> manager = std::make_shared<milvus::LocalFileManager>()) {
    auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(manager));
    return knowhere::IndexFactory::Instance()
        .Create<knowhere::fp32>(type == "DISKANN" ? kNativeDiskANNType : type, version, pack)
        .value();
}

float
ExactScore(const RaBitQFixture& f, size_t q, int64_t id, const std::string& metric) {
    float ip = 0, l2 = 0, qnorm = 0, bnorm = 0;
    for (size_t d = 0; d < f.dim; ++d) {
        const float a = f.query[q * f.dim + d], b = f.base[id * f.dim + d];
        ip += a * b;
        l2 += (a - b) * (a - b);
        qnorm += a * a;
        bnorm += b * b;
    }
    return metric == "L2" ? l2 : metric == "IP" ? ip : ip / std::sqrt(qnorm * bnorm);
}
// Reference the ordinary SSD-PQ path directly from the stored codes.
float
LegacySSDPQScore(const RaBitQFixture& f, size_t q, int64_t id, const std::string& metric) {
    std::ifstream graph(diskann::get_disk_index_filename(f.prefix), std::ios::binary);
    uint64_t node_bytes, per_sector;
    graph.seekg(3 * sizeof(uint64_t));
    graph.read(reinterpret_cast<char*>(&node_bytes), sizeof(node_bytes));
    graph.read(reinterpret_cast<char*>(&per_sector), sizeof(per_sector));
    std::vector<uint8_t> code(4);
    graph.seekg((id / per_sector + 1) * diskann::defaults::SECTOR_LEN + (id % per_sector) * node_bytes);
    graph.read(reinterpret_cast<char*>(code.data()), code.size());
    REQUIRE(bool(graph));
    diskann::FixedChunkPQTable table;
    table.load_pq_centroid_bin((f.prefix + "_disk.index_pq_pivots.bin").c_str(), code.size());
    std::vector<float> query(f.query.begin() + q * f.dim, f.query.begin() + (q + 1) * f.dim);
    float norm = std::sqrt(std::inner_product(query.begin(), query.end(), query.begin(), 0.0f));
    if (metric != "L2")
        for (auto& x : query) x /= norm;
    if (metric == "IP")
        query.push_back(0);
    if (metric == "L2")
        return table.l2_distance(query.data(), code.data());
    const float negative_ip = table.inner_product(query.data(), code.data());
    if (metric == "COSINE")
        return -negative_ip;
    std::ifstream norms(diskann::get_disk_index_max_base_norm_file(diskann::get_disk_index_filename(f.prefix)),
                        std::ios::binary);
    float max_norm;
    norms.seekg(2 * sizeof(uint32_t));
    norms.read(reinterpret_cast<char*>(&max_norm), sizeof(max_norm));
    REQUIRE(bool(norms));
    return -negative_ip * max_norm * norm;
}
}  // namespace

TEST_CASE("RaBitQ navigation scores match Faiss and retain model ownership", "[diskann][diskann-rabitq]") {
    const auto bits = GENERATE(1, 3, 9);
    const auto dimension = GENERATE(16, 17, 128);
    RaBitQFixture f(dimension);
    diskann::RaBitQNavigationStore writer({uint8_t(bits)});
    REQUIRE(writer.Build(f.Context()) == 0);
    auto model = diskann::CreateNavigationStore("RABITQ");
    REQUIRE(model->Load(f.prefix) == 0);
    REQUIRE(model->Count() == f.rows);
    REQUIRE(model->Dimension() == f.dim);
    REQUIRE(model->MemorySize() > f.rows * (f.dim * bits / 8));
    std::unique_ptr<faiss::Index> reference(
        faiss::cppcontrib::knowhere::read_index(diskann::RaBitQNavigationStore::SidecarFilename(f.prefix).c_str()));
    auto& transform = dynamic_cast<faiss::IndexPreTransform&>(*reference);
    auto& leaf = dynamic_cast<faiss::IndexRaBitQ&>(*transform.index);
    auto first = model->CreateDistanceComputer(f.dim);
    auto second = model->CreateDistanceComputer(f.dim);
    // Computers keep their immutable model alive, including the transform.
    model.reset();
    std::vector<float> rotated(f.dim);
    for (uint8_t query_bits : {0, 4, 8, 1, 4}) {
        diskann::RaBitQNavigationStore::ConfigureQuery(*first, query_bits);
        first->set_query(f.query.data());
        second->ResetQueryParameters();
        second->set_query(f.query.data() + f.dim);
        transform.chain[0]->apply_noalloc(1, f.query.data(), rotated.data());
        std::unique_ptr<faiss::FlatCodesDistanceComputer> native(
            leaf.get_quantized_distance_computer(query_bits, false));
        native->set_query(rotated.data());
        for (size_t count : {0, 1, 3, 4, 7, 32, 513}) {
            std::vector<unsigned> ids(count);
            std::vector<float> scores(count);
            for (size_t i = 0; i < count; ++i) ids[i] = (i * 97) % f.rows;
            first->compute_distances(ids.data(), count, scores.data());
            for (size_t i = 0; i < count; ++i) REQUIRE(scores[i] == Catch::Approx((*native)(ids[i])).margin(1e-5));
        }
    }
    REQUIRE_THROWS_AS(diskann::RaBitQNavigationStore::ConfigureQuery(*first, 9), std::invalid_argument);
}

TEST_CASE("RaBitQ navigation build validates its file and context", "[diskann][diskann-rabitq]") {
    RaBitQFixture f;
    auto context = f.Context();
    uint8_t bits = 1;
    SECTION("Invalid bit width") {
        bits = 10;
    }
    SECTION("Mismatched context") {
        ++context.count;
    }
    SECTION("Empty input") {
        context.count = 0;
        diskann::save_bin<float>(f.source, f.base.data(), 0, f.dim);
    }
    SECTION("Zero dimension") {
        context.dimension = 0;
        diskann::save_bin<float>(f.source, f.base.data(), f.rows, 0);
    }
    SECTION("Truncated input") {
        fs::resize_file(f.source, fs::file_size(f.source) - sizeof(float));
    }
    SECTION("Trailing input") {
        fs::resize_file(f.source, fs::file_size(f.source) + sizeof(float));
    }
    diskann::RaBitQNavigationStore writer({bits});
    REQUIRE_THROWS(writer.Build(context));
    REQUIRE_FALSE(fs::exists(diskann::RaBitQNavigationStore::SidecarFilename(f.prefix)));
}

TEST_CASE("DiskANN RaBitQ registration and codec parameters", "[diskann][diskann-rabitq]") {
    const auto version = GenTestVersionList();
    REQUIRE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck("DISKANN", knowhere::VecType::VECTOR_FLOAT, true));
    for (bool embedding : {false, true})
        REQUIRE_FALSE(knowhere::KnowhereCheck::IndexTypeAndDataTypeCheck("DISKANN_RABITQ",
                                                                         knowhere::VecType::VECTOR_FLOAT, embedding));
    REQUIRE_FALSE(knowhere::IndexFactory::Instance().FeatureCheck("DISKANN_RABITQ", knowhere::feature::EMB_LIST));
    RaBitQFixture f;
    auto index = CreateDiskANN("DISKANN", version);
    REQUIRE(index.Type() == "DISKANN");
    auto json = f.BuildConfig("L2");
    auto config = knowhere::IndexStaticFaced<knowhere::fp32>::CreateConfig(kNativeDiskANNType, version);
    REQUIRE(knowhere::Config::Load(*config, json, knowhere::PARAM_TYPE::TRAIN) == knowhere::Status::success);
    auto& navigation = static_cast<knowhere::DiskANNConfig&>(*config).navigation;
    REQUIRE(static_cast<const knowhere::RaBitQNavigationConfig&>(*navigation).BuildOptions().bits == 1);
    knowhere::Json metadata;
    navigation->AppendBuildMetadata(metadata);
    // Loading without the original build config must not invent stored bits.
    REQUIRE_FALSE(metadata.contains("rbq_bits"));
    REQUIRE(std::string(navigation->CreateForBuild()->CodecName()) == "RABITQ");
    for (int bits : {-1, 9}) {
        auto search = knowhere::Json{{"metric_type", "L2"}, {"k", 8}, {"rbq_bits_query", bits}};
        REQUIRE(knowhere::Config::Load(*config, search, knowhere::PARAM_TYPE::SEARCH) ==
                knowhere::Status::out_of_range_in_json);
    }
    for (int bits : {0, 10}) {
        json["rbq_bits"] = bits;
        REQUIRE(knowhere::Config::Load(*config, json, knowhere::PARAM_TYPE::TRAIN) ==
                knowhere::Status::out_of_range_in_json);
    }
    knowhere::AisaqConfig aisaq;
    json.erase("rbq_bits");
    REQUIRE(knowhere::Config::Load(aisaq, json, knowhere::PARAM_TYPE::TRAIN) == knowhere::Status::invalid_args);
}

TEST_CASE("DiskANN navigation validation matches input capabilities", "[diskann][diskann-rabitq]") {
    RaBitQFixture f;
    const auto version = GenTestVersionList();
    auto request = f.BuildConfig(GENERATE("L2", "MAX_SIM_L2"));
    const auto codec = GENERATE("", "PQ", "RABITQ");
    if (std::string(codec).empty()) {
        request.erase("quant_type");
    } else {
        request["quant_type"] = codec;
    }
    auto check = [&](auto value) {
        using T = decltype(value);
        std::string error;
        const auto expected = std::string(codec) == "RABITQ" && !std::is_same_v<T, knowhere::fp32>
                                  ? knowhere::Status::invalid_args
                                  : knowhere::Status::success;
        REQUIRE(knowhere::IndexStaticFaced<T>::ConfigCheck(kNativeDiskANNType, version, request, error) == expected);
    };
    check(knowhere::fp32{});
    check(knowhere::fp16{});
    check(knowhere::bf16{});
    for (const auto* strategy : {knowhere::meta::EMB_LIST_STRATEGY_MUVERA, knowhere::meta::EMB_LIST_STRATEGY_LEMUR}) {
        request["emb_list_strategy"] = strategy;
        std::string error;
        REQUIRE(knowhere::IndexStaticFaced<knowhere::fp32>::ConfigCheck(kNativeDiskANNType, version, request, error) ==
                knowhere::Status::invalid_args);
    }
}

TEST_CASE("RaBitQ TokenANN preserves document scores and filtering", "[diskann][diskann-rabitq]") {
    const auto metric = GENERATE("MAX_SIM_L2", "MAX_SIM_IP", "MAX_SIM_COSINE");
    const bool cache = GENERATE(false, true);
    RaBitQFixture f;
    std::vector<size_t> offsets{0};
    while (offsets.back() < f.rows) {
        offsets.push_back(std::min(f.rows, offsets.back() + 1 + offsets.size() % 5));
    }
    const auto docs = offsets.size() - 1;
    const std::vector<size_t> query_offsets{0, 1, f.nq};
    auto base = knowhere::GenDataSet(f.rows, f.dim, f.base.data());
    base->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(offsets.data()));
    base->Set(knowhere::meta::EMB_LIST_COUNT, int64_t(docs));
    auto query = knowhere::GenDataSet(f.nq, f.dim, f.query.data());
    query->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(query_offsets.data()));
    query->Set(knowhere::meta::EMB_LIST_COUNT, int64_t(query_offsets.size() - 1));
    const auto offset_file = (f.directory / "offsets.bin").string();
    WriteEmbListOffsetToDisk(offset_file, offsets.data(), offsets.size());
    auto build = f.BuildConfig(metric);
    build["emb_list_offset_file_path"] = offset_file;
    build["rbq_bits"] = cache ? 4 : 1;
    const double cache_budget = cache ? 0.000001 : 0.0;
    build["search_cache_budget_gb"] = cache_budget;
    const auto version = GenTestVersionList();
    auto manager = std::make_shared<milvus::LocalFileManager>();
    {
        auto index = CreateDiskANN("DISKANN", version, manager);
        REQUIRE(index.Build(nullptr, build) == knowhere::Status::success);
    }
    auto index = CreateDiskANN("DISKANN", version, manager);
    knowhere::Json load = {{"metric_type", metric},
                           {"index_prefix", f.prefix},
                           {"search_cache_budget_gb", cache_budget},
                           {"use_bfs_cache", true}};
    REQUIRE(index.Deserialize({}, load) == knowhere::Status::success);
    // A prepared node keeps its effective prefix when asked to load again.
    load["index_prefix"] = (f.directory / "unused").string();
    REQUIRE(index.Deserialize({}, load) == knowhere::Status::success);
    knowhere::Json search = {{"metric_type", metric},
                             {"k", 5},
                             {"search_list_size", f.rows},
                             {"beamwidth", 4},
                             {"retrieval_ann_ratio", double(f.rows)},
                             {"rbq_bits_query", cache ? 8 : 4}};
    for (size_t excluded : {size_t(0), docs / 3}) {
        std::vector<uint8_t> bits((docs + 7) / 8, 0);
        for (size_t i = 0; i < excluded; ++i) {
            bits[i / 8] |= uint8_t(1u << (i % 8));
        }
        knowhere::BitsetView filter(bits.data(), docs);
        auto actual = index.Search(query, search, filter);
        REQUIRE(actual.has_value());
        auto reference_config = search;
        reference_config["k"] = docs;
        auto reference = knowhere::BruteForce::Search<knowhere::fp32>(base, query, reference_config, filter);
        REQUIRE(reference.has_value());
        for (size_t q = 0; q < query_offsets.size() - 1; ++q) {
            std::unordered_set<int64_t> expected_ids;
            for (size_t j = 0; j < 5; ++j) {
                expected_ids.insert(reference.value()->GetIds()[q * docs + j]);
            }
            for (size_t j = 0; j < 5; ++j) {
                const auto id = actual.value()->GetIds()[q * 5 + j];
                REQUIRE(expected_ids.erase(id) == 1);
                const auto* begin = reference.value()->GetIds() + q * docs;
                const auto* found = std::find(begin, begin + docs, id);
                REQUIRE(found != begin + docs);
                REQUIRE(actual.value()->GetDistance()[q * 5 + j] ==
                        Catch::Approx(reference.value()->GetDistance()[q * docs + (found - begin)])
                            .epsilon(1e-4)
                            .margin(1e-5));
            }
        }
    }
    if (std::string(metric) == "MAX_SIM_L2" && !cache) {
        // Embedding-list reranking needs exact distances, which a disk-PQ artifact cannot provide.
        build["index_prefix"] = f.prefix + "_ssd_pq";
        build["disk_pq_dims"] = 4;
        auto compressed = CreateDiskANN("DISKANN", version, manager);
        REQUIRE(compressed.Build(nullptr, build) == knowhere::Status::invalid_param_in_json);
    }
}

TEST_CASE("DiskANN RaBitQ searches independent SSD encodings and metrics", "[diskann][diskann-rabitq]") {
    const auto metric = GENERATE(std::string("L2"), std::string("IP"), std::string("COSINE"));
    const auto ssd_pq = GENERATE(0, 4);
    const auto version = GenTestVersionList();
    RaBitQFixture f;
    auto build = f.BuildConfig(metric, ssd_pq);
    build["rbq_bits"] = ssd_pq ? 3 : 1;
    build["search_cache_budget_gb"] = 0.000001;
    {
        auto index = CreateDiskANN("DISKANN", version);
        REQUIRE(index.Build(nullptr, build) == knowhere::Status::success);
    }
    REQUIRE(diskann::ReadNavigationCodec({f.prefix, {}, {}}) == "RABITQ");
    REQUIRE(fs::exists(diskann::RaBitQNavigationStore::SidecarFilename(f.prefix)));
    REQUIRE_FALSE(fs::exists(diskann::get_pq_pivots_filename(f.prefix)));
    REQUIRE_FALSE(fs::exists(diskann::get_pq_compressed_filename(f.prefix)));
    REQUIRE(fs::exists(diskann::get_cached_nodes_file(f.prefix)));
    // Fresh ordinary DISKANN and no navigation parameter: metadata chooses RaBitQ.
    auto index = CreateDiskANN("DISKANN", version);
    REQUIRE(index.Deserialize({}, {{"metric_type", metric},
                                   {"index_prefix", f.prefix},
                                   {"search_cache_budget_gb", ssd_pq ? 0.0 : 0.000001},
                                   {"warm_up", true}}) == knowhere::Status::success);
    REQUIRE(index.Count() == f.rows);
    REQUIRE(index.Dim() == f.dim);
    REQUIRE(index.Size() > 0);
    auto query = knowhere::GenDataSet(f.nq, f.dim, f.query.data());
    auto check_scores = [&](const knowhere::DataSetPtr& result, const knowhere::BitsetView& filter, size_t k) {
        for (size_t q = 0; q < f.nq; ++q) {
            std::unordered_set<int64_t> seen;
            for (size_t i = 0; i < k; ++i) {
                const auto id = result->GetIds()[q * k + i];
                REQUIRE(id >= 0);
                REQUIRE(id < int64_t(f.rows));
                REQUIRE(seen.insert(id).second);
                REQUIRE((filter.empty() || !filter.test(id)));
                if (!ssd_pq)
                    REQUIRE(result->GetDistance()[q * k + i] ==
                            Catch::Approx(ExactScore(f, q, id, metric)).margin(1e-4));
                else {
                    REQUIRE(result->GetDistance()[q * k + i] ==
                            Catch::Approx(LegacySSDPQScore(f, q, id, metric)).margin(1e-4));
                }
            }
        }
    };
    knowhere::Json search = {{"metric_type", metric}, {"k", 8}, {"search_list_size", 64}};
    for (int query_bits : {0, 8, 4}) {
        search["rbq_bits_query"] = query_bits;
        auto result = index.Search(query, search, nullptr);
        REQUIRE(result.has_value());
        check_scores(result.value(), nullptr, 8);
    }
    std::vector<uint8_t> filtered(f.rows / 8, 0x55);
    knowhere::BitsetView bitset(filtered.data(), f.rows);
    // Force both graph filtering and the shared brute-force fallback.
    for (float threshold : ssd_pq ? std::vector<float>{1.0f} : std::vector<float>{1.0f, 0.0f}) {
        search["filter_threshold"] = threshold;
        auto result = index.Search(query, search, bitset);
        REQUIRE(result.has_value());
        check_scores(result.value(), bitset, 8);
    }
    // Complete filtering must return every borrowed slot; repeat beyond pool size.
    std::fill(filtered.begin(), filtered.end(), 0xff);
    for (int i = 0; i < 8; ++i) {
        auto result = index.Search(query, search, bitset);
        REQUIRE(result.has_value());
        for (size_t j = 0; j < f.nq * 8; ++j) REQUIRE(result.value()->GetIds()[j] == -1);
    }
    auto after = index.Search(query, search, nullptr);
    REQUIRE(after.has_value());
    folly::CancellationSource cancellation;
    milvus::OpContext context(cancellation.getToken());
    cancellation.requestCancellation();
    auto cancelled = index.Search(query, search, nullptr, &context);
    REQUIRE_FALSE(cancelled.has_value());
    REQUIRE(cancelled.error() == knowhere::Status::cancelled);
    REQUIRE(index.Search(query, search, nullptr).has_value());
    auto config = index.Node()->CreateConfig();
    REQUIRE(knowhere::Config::Load(*config, build, knowhere::PARAM_TYPE::TRAIN) == knowhere::Status::success);
    auto meta = index.Node()->GetIndexMeta(std::move(config));
    REQUIRE(meta.has_value());
    const auto params = knowhere::Json::parse(meta.value()->GetJsonInfo()).at("build_params_");
    REQUIRE(params.at("rbq_bits") == build.at("rbq_bits"));
    REQUIRE_FALSE(params.contains("pq_code_budget_gb"));
    auto fresh = index.Node()->CreateConfig();
    REQUIRE(knowhere::Config::Load(*fresh, {}, knowhere::PARAM_TYPE::FEDER) == knowhere::Status::success);
    auto& fresh_navigation = static_cast<knowhere::DiskANNConfig&>(*fresh).navigation;
    REQUIRE(dynamic_cast<knowhere::RaBitQNavigationConfig*>(fresh_navigation.get()) != nullptr);
    knowhere::Json unloaded_params;
    fresh_navigation->AppendBuildMetadata(unloaded_params);
    REQUIRE_FALSE(unloaded_params.contains("rbq_bits"));
    // A loaded RBQ model must not change the default for a new build.
    auto rebuild = build;
    rebuild.erase("quant_type");
    rebuild["rbq_bits"] = 10;
    REQUIRE(knowhere::Config::Load(*fresh, rebuild, knowhere::PARAM_TYPE::TRAIN) == knowhere::Status::success);
    REQUIRE(std::string(static_cast<knowhere::DiskANNConfig&>(*fresh).navigation->CreateForBuild()->CodecName()) ==
            "PQ");
}

TEST_CASE("RaBitQ iterators interleave query parameters and searches", "[diskann][diskann-rabitq]") {
    RaBitQFixture f;
    const auto version = GenTestVersionList();
    auto index = CreateDiskANN("DISKANN", version);
    REQUIRE(index.Build(nullptr, f.BuildConfig("L2")) == knowhere::Status::success);
    REQUIRE(index.Deserialize({}, {{"metric_type", "L2"}, {"index_prefix", f.prefix}}) == knowhere::Status::success);
    auto query = knowhere::GenDataSet(f.nq, f.dim, f.query.data());
    knowhere::Json search = {{"metric_type", "L2"}, {"k", 8}, {"search_list_size", 16}, {"rbq_bits_query", 0}};
    auto first = index.AnnIterator(query, search, nullptr, false);
    REQUIRE(first.has_value());
    search["rbq_bits_query"] = 8;
    auto second = index.AnnIterator(query, search, nullptr, false);
    REQUIRE(second.has_value());
    for (size_t q = 0; q < f.nq; ++q) {
        std::unordered_set<int64_t> seen[2];
        for (size_t i = 0; i < 80; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                auto& it = (j ? second : first).value()[q];
                REQUIRE(it->HasNext().value());
                auto next = it->Next();
                REQUIRE(next.has_value());
                auto [id, score] = next.value();
                REQUIRE(seen[j].insert(id).second);
                REQUIRE(score == Catch::Approx(ExactScore(f, q, id, "L2")).margin(1e-4));
            }
        }
        auto result = index.Search(query, search, nullptr);
        REQUIRE(result.has_value());
    }
    auto range = index.RangeSearch(
        query, {{"metric_type", "L2"}, {"radius", 100.0}, {"range_search_k", -1}, {"rbq_bits_query", 0}}, nullptr);
    REQUIRE(range.has_value());
    for (size_t q = 0; q < f.nq; ++q) {
        REQUIRE(range.value()->GetLims()[q + 1] - range.value()->GetLims()[q] == f.rows);
        for (size_t i = range.value()->GetLims()[q]; i < range.value()->GetLims()[q + 1]; ++i)
            REQUIRE(range.value()->GetDistance()[i] ==
                    Catch::Approx(ExactScore(f, q, range.value()->GetIds()[i], "L2")).margin(1e-4));
    }
    // Final SSD scores alone cannot show whether query-bit settings reached
    // navigation. Compare the actual visited-neighbor scores against Faiss.
    std::unique_ptr<faiss::Index> reference(
        faiss::cppcontrib::knowhere::read_index(diskann::RaBitQNavigationStore::SidecarFilename(f.prefix).c_str()));
    auto& transform = dynamic_cast<faiss::IndexPreTransform&>(*reference);
    auto& leaf = dynamic_cast<faiss::IndexRaBitQ&>(*transform.index);
    std::vector<float> rotated(f.dim);
    transform.chain[0]->apply_noalloc(1, f.query.data(), rotated.data());
    auto single_query = knowhere::GenDataSet(1, f.dim, f.query.data());
    for (uint8_t bits : {0, 8, 4}) {
        auto traced = search;
        traced["rbq_bits_query"] = bits;
        traced["trace_visit"] = true;
        auto result = index.Search(single_query, traced, nullptr);
        REQUIRE(result.has_value());
        std::unique_ptr<faiss::FlatCodesDistanceComputer> native(leaf.get_quantized_distance_computer(bits, false));
        native->set_query(rotated.data());
        const auto infos = knowhere::Json::parse(result.value()->GetJsonInfo()).at("infos_");
        size_t checked = 0;
        for (const auto& candidate : infos)
            for (const auto& neighbor : candidate.at("neighbors_")) {
                const auto id = neighbor.at(0).get<int64_t>();
                REQUIRE(neighbor.at(1).get<float>() == Catch::Approx((*native)(id)).margin(1e-5));
                ++checked;
            }
        REQUIRE(checked > 0);
    }
    search.erase("rbq_bits_query");
    auto default_result = index.Search(query, search, nullptr);
    search["rbq_bits_query"] = 4;
    auto explicit_result = index.Search(query, search, nullptr);
    REQUIRE(default_result.has_value());
    REQUIRE(explicit_result.has_value());
    for (size_t i = 0; i < f.nq * 8; ++i) {
        REQUIRE(default_result.value()->GetIds()[i] == explicit_result.value()->GetIds()[i]);
        REQUIRE(default_result.value()->GetDistance()[i] == explicit_result.value()->GetDistance()[i]);
    }
}

TEST_CASE("RaBitQ loading rejects wrong metadata and unsupported consumers", "[diskann][diskann-rabitq]") {
    RaBitQFixture f;
    const auto version = GenTestVersionList();
    {
        auto index = CreateDiskANN("DISKANN", version);
        REQUIRE(index.Build(nullptr, f.BuildConfig("L2")) == knowhere::Status::success);
    }
    const knowhere::Json load = {{"metric_type", "L2"}, {"index_prefix", f.prefix}};
    SECTION("Expected PQ conflicts") {
        auto index = CreateDiskANN("DISKANN", version);
        auto request = load;
        request["quant_type"] = "PQ";
        REQUIRE(index.Deserialize({}, request) == knowhere::Status::invalid_serialized_index_type);
    }
    SECTION("Missing metadata means PQ only") {
        fs::remove(diskann::GetNavigationMetadataFilename(f.prefix));
        auto index = CreateDiskANN("DISKANN", version);
        REQUIRE(index.Deserialize({}, load) != knowhere::Status::success);
    }
    SECTION("Missing RaBitQ model") {
        fs::remove(diskann::RaBitQNavigationStore::SidecarFilename(f.prefix));
        auto index = CreateDiskANN("DISKANN", version);
        REQUIRE(index.Deserialize({}, load) != knowhere::Status::success);
    }
    SECTION("Wrong Faiss model") {
        faiss::IndexFlatL2 wrong(f.dim);
        faiss::cppcontrib::knowhere::write_index(&wrong,
                                                 diskann::RaBitQNavigationStore::SidecarFilename(f.prefix).c_str());
        auto index = CreateDiskANN("DISKANN", version);
        REQUIRE(index.Deserialize({}, load) != knowhere::Status::success);
    }
    SECTION("Truncated RaBitQ leaf") {
        const auto file = diskann::RaBitQNavigationStore::SidecarFilename(f.prefix);
        fs::resize_file(file, fs::file_size(file) - 1);
        diskann::RaBitQNavigationStore navigation;
        REQUIRE_THROWS(navigation.Load(f.prefix));
    }
    SECTION("Invalid serialized leaf query width") {
        // The leaf's final serialized field is qb; the shared reader validates it.
        std::fstream file(diskann::RaBitQNavigationStore::SidecarFilename(f.prefix),
                          std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(-1, std::ios::end);
        file.put(char(9));
        file.close();
        diskann::RaBitQNavigationStore navigation;
        REQUIRE_THROWS(navigation.Load(f.prefix));
    }
    SECTION("Inconsistent outer model") {
        const auto file = diskann::RaBitQNavigationStore::SidecarFilename(f.prefix);
        std::unique_ptr<faiss::Index> model(faiss::cppcontrib::knowhere::read_index(file.c_str()));
        ++model->d;
        faiss::cppcontrib::knowhere::write_index(model.get(), file.c_str());
        diskann::RaBitQNavigationStore navigation;
        REQUIRE_THROWS(navigation.Load(f.prefix));
    }
    SECTION("Ordinary DISKANN low precision autoload") {
        auto check = [&](auto value) {
            using T = decltype(value);
            auto manager = std::make_shared<milvus::LocalFileManager>();
            auto pack = knowhere::Pack(std::shared_ptr<milvus::FileManager>(manager));
            auto index = knowhere::IndexFactory::Instance().Create<T>(kNativeDiskANNType, version, pack).value();
            REQUIRE(index.Deserialize({}, load) == knowhere::Status::invalid_serialized_index_type);
            auto request = f.BuildConfig("L2");
            REQUIRE(index.Build(nullptr, request) == knowhere::Status::invalid_args);
        };
        check(knowhere::fp16{});
        check(knowhere::bf16{});
    }
    SECTION("AiSAQ autoload") {
        auto index = CreateDiskANN("AISAQ", version);
        REQUIRE(index.Deserialize({}, load) == knowhere::Status::invalid_serialized_index_type);
    }
    SECTION("Embedding list build requires offsets") {
        auto index = CreateDiskANN("DISKANN", version);
        auto request = f.BuildConfig("MAX_SIM_L2");
        REQUIRE(index.Build(nullptr, request) == knowhere::Status::emb_list_inner_error);
    }
    SECTION("Embedding list autoload") {
        auto index = CreateDiskANN("DISKANN", version);
        auto request = load;
        request["metric_type"] = "MAX_SIM_L2";
        REQUIRE(index.Deserialize({}, request) == knowhere::Status::emb_list_inner_error);
        REQUIRE(index.Deserialize({}, load) == knowhere::Status::success);
    }
    SECTION("Invalid offsets preserve a loaded ordinary node") {
        auto manager = std::make_shared<milvus::LocalFileManager>();
        auto index = CreateDiskANN("DISKANN", version, manager);
        REQUIRE(index.Deserialize({}, load) == knowhere::Status::success);
        if (GENERATE(false, true)) {
            const auto path = diskann::get_emb_list_offset_file(f.prefix);
            const std::vector<size_t> offsets{0, f.rows};
            WriteEmbListOffsetToDisk(path, offsets.data(), offsets.size());
            fs::resize_file(path, sizeof(size_t) + 1);
            REQUIRE(manager->AddFile(path));
        }
        auto query = knowhere::GenDataSet(f.nq, f.dim, f.query.data());
        const knowhere::Json search = {{"metric_type", "L2"}, {"k", 8}, {"search_list_size", 64}};
        auto before = index.Search(query, search, nullptr);
        REQUIRE(before.has_value());
        auto request = load;
        request["metric_type"] = "MAX_SIM_L2";
        REQUIRE(index.Deserialize({}, request) == knowhere::Status::emb_list_inner_error);
        auto after = index.Search(query, search, nullptr);
        REQUIRE(after.has_value());
        for (size_t i = 0; i < f.nq * 8; ++i) {
            REQUIRE(before.value()->GetIds()[i] == after.value()->GetIds()[i]);
            REQUIRE(before.value()->GetDistance()[i] == after.value()->GetDistance()[i]);
        }
    }
}

TEST_CASE("RaBitQ query setup failures return the borrowed thread slot", "[diskann][diskann-rabitq]") {
    RaBitQFixture f;
    {
        auto index = CreateDiskANN("DISKANN", GenTestVersionList());
        REQUIRE(index.Build(nullptr, f.BuildConfig("L2")) == knowhere::Status::success);
    }
    auto reader = std::make_shared<LinuxAlignedFileReader>();
    diskann::SSDGraphIndex<float> index(reader, diskann::Metric::L2);
    auto navigation = diskann::CreateNavigationStore("RABITQ");
    REQUIRE(navigation->Load(f.prefix) == 0);
    REQUIRE(index.load(1, f.prefix.c_str(), std::move(navigation)) == 0);
    int64_t ids[8];
    float scores[8];
    diskann::NavigationQuerySetup fail = [](auto&) { throw std::runtime_error("injected setup failure"); };
    for (int i = 0; i < 3; ++i)
        REQUIRE_THROWS(index.cached_beam_search(f.query.data(), 8, 64, ids, scores, 4, false, nullptr, nullptr, nullptr,
                                                -1, fail));
    index.cached_beam_search(f.query.data(), 8, 64, ids, scores, 4);
    for (size_t i = 0; i < 8; ++i) REQUIRE(ids[i] >= 0);
    auto workspace = index.getIteratorWorkspace(f.query.data(), 16, 4, -1, nullptr);
    workspace->navigation_setup = fail;
    for (int i = 0; i < 3; ++i) REQUIRE_THROWS(index.getIteratorNextBatch(workspace.get()));
    workspace->navigation_setup = {};
    index.getIteratorNextBatch(workspace.get());
    REQUIRE_FALSE(workspace->backup_res.empty());
}

TEST_CASE("DiskANN load admission preserves legacy PQ estimates", "[diskann][diskann-rabitq]") {
    const auto version = GenTestVersionList();
    constexpr uint64_t file_size = uint64_t(8) << 30;
    const knowhere::Json request = {{"metric_type", "L2"}};
    auto estimate = [&](knowhere::Json config) {
        return knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(kNativeDiskANNType, version, file_size,
                                                                                512, 16, config);
    };
    auto omitted = estimate(request);
    auto pq = request;
    pq["quant_type"] = "PQ";
    auto explicit_pq = estimate(pq);
    REQUIRE(omitted.has_value());
    REQUIRE(explicit_pq.has_value());
    REQUIRE(omitted.value().memoryCost == file_size / 4);
    REQUIRE(omitted.value().memoryCost == explicit_pq.value().memoryCost);
    auto rabitq = request;
    rabitq["quant_type"] = "RABITQ";
    auto explicit_rabitq = estimate(rabitq);
    REQUIRE(explicit_rabitq.has_value());
    REQUIRE(explicit_rabitq.value().memoryCost == file_size + file_size / 4);
    auto overflow = knowhere::IndexStaticFaced<knowhere::fp32>::EstimateLoadResource(
        kNativeDiskANNType, version, std::numeric_limits<uint64_t>::max(), 512, 16, rabitq);
    REQUIRE_FALSE(overflow.has_value());
    auto unsupported = request;
    unsupported["quant_type"] = "TQ";
    auto rejected = estimate(unsupported);
    REQUIRE_FALSE(rejected.has_value());
    REQUIRE(rejected.error() == knowhere::Status::invalid_args);
}
