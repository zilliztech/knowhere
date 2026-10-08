// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <dlfcn.h>
#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#if defined(__linux__)
#include <sched.h>
#if defined(__aarch64__)
#include <sys/prctl.h>
#endif
#endif

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/version.h"
#include "src/index/sparse/inverted_index_format.h"
#include "src/index/sparse/sindi_simd.h"

namespace {
using Clock = std::chrono::steady_clock;
using Row = knowhere::sparse::SparseRow<float>;
using knowhere::Json;

void
Check(bool ok, std::string_view message) {
    if (!ok) {
        throw std::runtime_error(std::string(message));
    }
}

double
Seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// The challenge files are little-endian, structure-of-arrays CSR / k-NN files.
class MappedFile {
 public:
    explicit MappedFile(const std::string& path) {
        Check(std::endian::native == std::endian::little, "Only little-endian hosts are supported");
        int fd = open(path.c_str(), O_RDONLY);
        Check(fd >= 0, "Cannot open " + path);
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            close(fd);
            throw std::runtime_error("Cannot stat or empty file: " + path);
        }
        size = st.st_size;
        void* ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        Check(ptr != MAP_FAILED, "Cannot mmap " + path);
        data = static_cast<const char*>(ptr);
    }
    ~MappedFile() {
        munmap(const_cast<char*>(data), size);
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile&
    operator=(const MappedFile&) = delete;

    template <typename T>
    T
    Read(uint64_t offset) const {
        Check(offset <= size && sizeof(T) <= size - offset, "Truncated binary file");
        T value;
        std::memcpy(&value, data + offset, sizeof(T));
        return value;
    }
    const char* data = nullptr;
    uint64_t size = 0;
};

struct SparseData {
    uint64_t file_rows = 0, dim = 0, nnz = 0;
    std::vector<Row> rows;

    knowhere::DataSetPtr
    Dataset() const {
        auto ds = knowhere::GenDataSet(rows.size(), dim, rows.data());
        ds->SetIsSparse(true);
        return ds;
    }
};

SparseData
LoadCsr(const std::string& path, const std::vector<uint64_t>* selection, uint64_t limit = 0) {
    MappedFile f(path);
    SparseData out;
    out.file_rows = f.Read<uint64_t>(0);
    out.dim = f.Read<uint64_t>(8);
    const auto nnz = f.Read<uint64_t>(16);
    Check(f.size >= 32 && out.file_rows > 0 && out.file_rows <= (f.size - 32) / 8 && out.dim > 0 &&
              out.dim <= std::numeric_limits<uint32_t>::max(),
          "Invalid CSR dimensions: " + path);
    const uint64_t indices_offset = 24 + 8 * (out.file_rows + 1);
    Check(nnz <= (f.size - indices_offset) / 8 && indices_offset + 8 * nnz == f.size,
          "CSR file size does not match header: " + path);
    const uint64_t values_offset = indices_offset + 4 * nnz;
    Check(f.Read<uint64_t>(24) == 0 && f.Read<uint64_t>(24 + 8 * out.file_rows) == nnz, "Invalid CSR endpoint offsets");
    uint64_t previous = 0;
    for (uint64_t i = 1; i <= out.file_rows; ++i) {
        const auto next = f.Read<uint64_t>(24 + 8 * i);
        Check(next >= previous && next <= nnz, "Invalid CSR row offsets");
        previous = next;
    }
    const auto count = selection ? selection->size() : (limit ? std::min(limit, out.file_rows) : out.file_rows);
    out.rows.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        const auto id = selection ? selection->at(i) : i;
        Check(id < out.file_rows, "Query ID outside CSR file");
        const auto start = f.Read<uint64_t>(24 + 8 * id);
        const auto end = f.Read<uint64_t>(24 + 8 * (id + 1));
        out.rows.emplace_back(end - start);
        uint32_t last = 0;
        for (auto j = start; j < end; ++j) {
            const auto col = f.Read<uint32_t>(indices_offset + 4 * j);
            const auto value = f.Read<float>(values_offset + 4 * j);
            Check(col < out.dim && (j == start || col > last), "CSR columns must be sorted and unique");
            Check(std::isfinite(value) && value >= 0, "Expected finite nonnegative sparse values");
            out.rows.back().set_at(j - start, col, value);
            last = col;
        }
        out.nnz += end - start;
    }
    return out;
}

struct Truth {
    std::vector<int64_t> ids;
    std::vector<float> scores;
};

Truth
LoadTruth(const std::string& path, const std::vector<uint64_t>& selection, uint64_t nq, uint64_t nb, int k) {
    MappedFile f(path);
    const uint64_t rows = f.Read<uint32_t>(0), width = f.Read<uint32_t>(4);
    Check(rows == nq && width >= static_cast<uint64_t>(k) && f.size >= 8 && rows > 0 &&
              width <= (f.size - 8) / 8 / rows && 8 + rows * width * 8 == f.size,
          "Invalid ground-truth header or unsupported k");
    Truth out;
    for (auto q : selection) {
        std::set<int64_t> unique;
        float previous = std::numeric_limits<float>::infinity();
        for (int j = 0; j < k; ++j) {
            auto id = f.Read<uint32_t>(8 + 4 * (q * width + j));
            auto score = f.Read<float>(8 + 4 * rows * width + 4 * (q * width + j));
            Check(
                id < nb && unique.insert(id).second && std::isfinite(score) &&
                    score <= previous + 4 * std::numeric_limits<float>::epsilon() * std::max(1.0f, std::abs(previous)),
                "Invalid ground-truth IDs or score order");
            out.ids.push_back(id);
            out.scores.push_back(score);
            previous = score;
        }
    }
    return out;
}

// ID recall deliberately exposes ties and FP16 rounding relative to FP32 ground truth.
double
Recall(const knowhere::DataSetPtr& result, const Truth& truth, size_t nq, int k, size_t nb) {
    size_t hits = 0, total = 0;
    const auto ids = result->GetIds();
    const auto scores = result->GetDistance();
    for (size_t q = 0; q < nq; ++q) {
        std::set<int64_t> seen;
        float previous = std::numeric_limits<float>::infinity();
        bool padding = false;
        for (int j = 0; j < k; ++j) {
            const auto pos = q * k + j;
            if (truth.ids[pos] >= 0)
                ++total;
            if (ids[pos] == -1) {
                padding = true;
                continue;
            }
            Check(!padding && ids[pos] >= 0 && static_cast<size_t>(ids[pos]) < nb && seen.insert(ids[pos]).second &&
                      std::isfinite(scores[pos]) && scores[pos] <= previous,
                  "Invalid search result IDs or score order");
            previous = scores[pos];
            auto begin = truth.ids.begin() + q * k;
            hits += std::find(begin, begin + k, ids[pos]) != begin + k;
        }
    }
    return total ? double(hits) / total : 1.0;
}

Truth
Capture(const knowhere::DataSetPtr& r, size_t n) {
    Truth out{{r->GetIds(), r->GetIds() + n}, {r->GetDistance(), r->GetDistance() + n}};
    // Scores for missing neighbors are NaN/unspecified; compare only meaningful
    // scores.
    for (size_t i = 0; i < n; ++i)
        if (out.ids[i] < 0)
            out.scores[i] = 0;
    return out;
}
Truth
ReadBmTruth(const std::string& path, const Json& sig) {
    std::ifstream meta(path + ".json");
    Json stored;
    meta >> stored;
    Check(stored == sig, "Truth signature mismatch");
    MappedFile f(path);
    auto n = f.Read<uint64_t>(0);
    Check(n <= f.size / 12 && f.size == 8 + n * 12, "Invalid truth");
    Truth t;
    t.ids.resize(n);
    t.scores.resize(n);
    std::memcpy(t.ids.data(), f.data + 8, n * 8);
    std::memcpy(t.scores.data(), f.data + 8 + n * 8, n * 4);
    return t;
}

// Ordinary-search harness adapted from the ARM Knowhere donor. No graph,
// refinement, quantization experiments, or changes to the search implementation.
struct Options {
    std::string data, metric = "IP", split = "dev", quant, truth, output = "sparse_results.csv";
    std::string save, load, generate_truth;
    uint64_t queries = 0, offset = 0, base_limit = 0;
    int threads = std::max(2u, std::thread::hardware_concurrency()), repeats = 3, warmup = 1, k = 10;
    int window = 4096, truth_audit_queries = 0;
    float ratio = 1.05f;
    bool brute = false, reload = false, h2 = false;
};

Options
Parse(int argc, char** argv) {
    Options o;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout << "benchmark_sparse --data-dir DIR --metric IP|BM25 [--split dev|test|hidden]\n"
                         "  --quant-type fp16|u8|u16|auto --method SINDI --h2 0|1\n"
                         "  --queries 0 (all) --query-offset 0 --base-limit 0 (all)\n"
                         "  --threads N --warmup 1 --repeats 3 --k 10 --window 4096 --ratio 1.05\n"
                         "  --truth FILE | --brute-force 1 --output FILE.csv\n"
                         "  --truth-audit-queries N (sample full-corpus exhaustive queries before timing)\n"
                         "  --index-save FILE --index-load FILE --reload-check 0|1\n"
                         "  --generate-ip-truth FILE (exact FP32 TAAT generation only)\n"
                         "Prefix runs require --brute-force 1.\n";
            std::exit(0);
        }
        Check(seen.insert(key).second, "Duplicate option: " + key);
        Check(i + 1 < argc, "Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--data-dir")
            o.data = value;
        else if (key == "--metric")
            o.metric = value;
        else if (key == "--split")
            o.split = value;
        else if (key == "--quant-type")
            o.quant = value;
        else if (key == "--truth")
            o.truth = value;
        else if (key == "--output")
            o.output = value;
        else if (key == "--index-save")
            o.save = value;
        else if (key == "--index-load")
            o.load = value;
        else if (key == "--generate-ip-truth")
            o.generate_truth = value;
        else if (key == "--method")
            Check(value == "SINDI", "Only ordinary SINDI is supported");
        else if (key == "--h2") {
            Check(value == "0" || value == "1", "--h2 expects 0 or 1");
            o.h2 = value == "1";
        } else if (key == "--ratio") {
            size_t end = 0;
            o.ratio = std::stof(value, &end);
            Check(end == value.size() && std::isfinite(o.ratio) && o.ratio >= .5f && o.ratio <= 1.3f,
                  "Invalid bound ratio");
        } else {
            Check(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                  "Expected unsigned integer for " + key);
            const uint64_t n = std::stoull(value);
            if (key == "--queries")
                o.queries = n;
            else if (key == "--query-offset")
                o.offset = n;
            else if (key == "--base-limit")
                o.base_limit = n;
            else if (key == "--brute-force" || key == "--reload-check") {
                Check(n <= 1, "Expected 0 or 1 for " + key);
                (key == "--brute-force" ? o.brute : o.reload) = n;
            } else {
                Check(n <= INT32_MAX && n > 0, "Expected positive int for " + key);
                if (key == "--threads")
                    o.threads = n;
                else if (key == "--warmup")
                    o.warmup = n;
                else if (key == "--repeats")
                    o.repeats = n;
                else if (key == "--k")
                    o.k = n;
                else if (key == "--window")
                    o.window = n;
                else if (key == "--truth-audit-queries")
                    o.truth_audit_queries = n;
                else
                    throw std::runtime_error("Unknown option: " + key);
            }
        }
    }
    Check(!o.data.empty(), "--data-dir is required");
    Check(o.metric == "IP" || o.metric == "BM25", "Metric must be IP or BM25");
    const bool bm = o.metric == "BM25";
    Check(o.split == "dev" || (bm ? o.split == "test" : o.split == "hidden"), "Invalid metric/split");
    if (o.quant.empty())
        o.quant = bm ? "auto" : "fp16";
    Check(bm ? (o.quant == "u8" || o.quant == "u16" || o.quant == "auto") : o.quant == "fp16",
          "Invalid metric/quantization combination");
    Check(o.window >= 1024 && o.window <= 65535, "Window must be 1024..65535");
    Check(o.threads >= 2, "Knowhere thread pools require at least two workers");
    Check(!o.brute || o.truth.empty(), "Choose --truth or --brute-force, not both");
    Check(!o.base_limit || o.brute, "Prefix corpus requires fresh brute-force truth");
    Check(!bm || o.brute || !o.truth.empty(), "BM25 requires --truth or --brute-force 1");
    Check(o.save.empty() || o.load.empty(), "Choose index-save or index-load");
    Check(o.generate_truth.empty() || (!bm && !o.h2 && o.save.empty() && o.load.empty()),
          "Truth generation requires IP with H2 disabled and no index save/load");
    return o;
}

Json
ReadJson(const std::string& path) {
    std::ifstream f(path);
    Check(f.good(), "Cannot open " + path);
    Json j;
    f >> j;
    return j;
}

// Hashing is outside search timing and verifies manifests against actual bytes.
std::string
Sha256(const std::string& path) {
    MappedFile f(path);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    Check(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "SHA256 initialization failed");
    for (uint64_t offset = 0; offset < f.size;) {
        const auto length = std::min<uint64_t>(8 * 1024 * 1024, f.size - offset);
        Check(EVP_DigestUpdate(ctx.get(), f.data + offset, length) == 1, "SHA256 update failed");
        offset += length;
    }
    unsigned char bytes[EVP_MAX_MD_SIZE];
    unsigned int count = 0;
    Check(EVP_DigestFinal_ex(ctx.get(), bytes, &count) == 1 && count == 32, "SHA256 finalization failed");
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < count; ++i) out << std::setw(2) << unsigned(bytes[i]);
    return out.str();
}

void
ValidateTruth(const Truth& t, size_t nq, int k, size_t nb) {
    Check(t.ids.size() == nq * k && t.scores.size() == t.ids.size(), "Wrong truth count");
    for (size_t q = 0; q < nq; ++q) {
        std::set<int64_t> ids;
        double previous = INFINITY;
        bool padding = false;
        for (int j = 0; j < k; ++j) {
            auto p = q * k + j;
            if (t.ids[p] == -1) {
                padding = true;
                continue;
            }
            Check(!padding && t.ids[p] >= 0 && size_t(t.ids[p]) < nb && ids.insert(t.ids[p]).second &&
                      std::isfinite(t.scores[p]) && t.scores[p] <= previous,
                  "Invalid truth ID, ordering, or score");
            previous = t.scores[p];
        }
    }
}

void
CheckHitScores(const Truth& hits, const SparseData& base, const SparseData& queries, int k, const Json& search,
               bool represented_ip = true) {
    const bool bm = search["metric_type"] == "BM25";
    const double k1 = bm ? search["bm25_k1"].get<double>() : 0;
    const double b = bm ? search["bm25_b"].get<double>() : 0;
    const double avgdl = bm ? search["bm25_avgdl"].get<double>() : 1;
    for (size_t p = 0; p < hits.ids.size(); ++p) {
        if (hits.ids[p] < 0)
            continue;
        const auto& doc = base.rows.at(hits.ids[p]);
        const auto& query = queries.rows.at(p / k);
        // Document lengths follow the index's original FP32 row-sum contract.
        float dl = 0;
        for (size_t i = 0; i < doc.size(); ++i) dl += doc[i].val;
        size_t x = 0, y = 0;
        double score = 0;
        while (x < doc.size() && y < query.size()) {
            auto dv = doc[x], qv = query[y];
            if (dv.id < qv.id)
                ++x;
            else if (qv.id < dv.id)
                ++y;
            else {
                const double value = bm ? std::min(65535.0, std::floor(double(dv.val)))
                                        : (represented_ip ? float(knowhere::fp16(dv.val)) : dv.val);
                score += bm ? qv.val * value * (k1 + 1) / (value + k1 * (1 - b + b * dl / avgdl)) : qv.val * value;
                ++x;
                ++y;
            }
        }
        Check(std::isfinite(score) && std::abs(hits.scores[p] - score) <= 1e-4 + 1e-5 * std::abs(score),
              "Independent represented-score check failed at hit " + std::to_string(p));
    }
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        auto o = Parse(argc, argv);
        const bool bm = o.metric == "BM25";
        knowhere::KnowhereConfig::SetBuildThreadPoolSize(o.threads);
        knowhere::KnowhereConfig::SetSearchThreadPoolSize(o.threads);
        const auto base_path = o.data + (bm ? "/sparse/base_tf.csr" : "/base_full.csr");
        const auto query_path = o.data + (bm ? "/sparse/queries." : "/queries.") + o.split + (bm ? ".idf.csr" : ".csr");
        MappedFile query_file(query_path);
        const auto file_queries = query_file.Read<uint64_t>(0);
        Check(o.offset < file_queries, "Invalid query offset");
        if (!o.queries)
            o.queries = file_queries - o.offset;
        Check(o.queries <= file_queries - o.offset, "Invalid query count");
        std::vector<uint64_t> selected(o.queries);
        std::iota(selected.begin(), selected.end(), o.offset);
        auto start = Clock::now();
        auto queries = LoadCsr(query_path, &selected);
        auto base = LoadCsr(base_path, nullptr, o.base_limit);
        Check(base.rows.size() >= size_t(o.k), "Base smaller than k");
        if (bm) {
            for (const auto& row : base.rows)
                for (size_t i = 0; i < row.size(); ++i)
                    Check(row[i].val <= 65535 && row[i].val == std::floor(row[i].val), "BM25 expects integer u16 TF");
        }
        base.dim = queries.dim = std::max(base.dim, queries.dim);
        const double input_seconds = Seconds(start);
        auto bd = base.Dataset(), qd = queries.Dataset();
        Json search = {
            {"metric_type", o.metric}, {"k", o.k}, {"drop_ratio_search", 0.0}, {"dim_max_score_ratio", o.ratio}};
        Json manifest;
        if (bm) {
            manifest = ReadJson(o.data + "/manifest.json");
            double k1 = manifest.at("bm25_k1"), b = manifest.at("bm25_b"), avgdl = manifest.at("avgdl");
            Check(std::isfinite(k1) && k1 >= 0 && std::isfinite(b) && b >= 0 && b <= 1 && std::isfinite(avgdl) &&
                      avgdl > 0,
                  "Invalid BM25 parameters");
            search["bm25_k1"] = k1;
            search["bm25_b"] = b;
            search["bm25_avgdl"] = avgdl;
        }
        start = Clock::now();
        const auto base_sha = Sha256(base_path), query_sha = Sha256(query_path);
        const double input_hash_seconds = Seconds(start);
        start = Clock::now();
        if (bm) {
            Check(manifest.at("sha256").at("base_tf.csr") == base_sha &&
                      manifest.at("sha256").at("queries." + o.split + ".idf.csr") == query_sha,
                  "Input SHA256 does not match dataset manifest");
        }
        if (!o.generate_truth.empty()) {
            Check(o.offset == 0 && o.queries == file_queries, "Truth generation requires the complete query split");
            Check(!std::filesystem::exists(o.generate_truth) && !std::filesystem::exists(o.generate_truth + ".json"),
                  "Truth output already exists");
            auto exact_config = search;
            exact_config["dim"] = base.dim;
            exact_config["inverted_index_algo"] = "TAAT_NAIVE";
            exact_config["search_algo"] = "TAAT_NAIVE";
            exact_config["quant_type"] = "fp32";
            auto exact_index =
                knowhere::IndexFactory::Instance()
                    .Create<knowhere::sparse_u32_f32>(knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, 11)
                    .value();
            start = Clock::now();
            Check(exact_index.Build(bd, exact_config) == knowhere::Status::success, "TAAT truth build failed");
            const double build_seconds = Seconds(start);
            start = Clock::now();
            auto result = exact_index.Search(qd, exact_config, nullptr);
            Check(result.has_value(), "TAAT truth search failed");
            const double search_seconds = Seconds(start);
            auto truth = Capture(result.value(), o.queries * o.k);
            ValidateTruth(truth, o.queries, o.k, base.rows.size());
            CheckHitScores(truth, base, queries, o.k, search, false);
            Json audit = Json::array();
            for (size_t i = 0, n = std::min<size_t>(std::max(1, o.truth_audit_queries), o.queries); i < n; ++i) {
                size_t qi = i * o.queries / n;
                auto query = knowhere::GenDataSet(1, base.dim, &queries.rows[qi]);
                query->SetIsSparse(true);
                auto brute = knowhere::BruteForce::SearchSparse(bd, query, search, nullptr);
                Check(brute.has_value(), "Truth generator exhaustive audit failed");
                auto oracle = Capture(brute.value(), o.k);
                size_t shared = 0;
                for (int j = 0; j < o.k; ++j) {
                    Check(std::abs(double(oracle.scores[j]) - truth.scores[qi * o.k + j]) <=
                              1e-4 + 2e-5 * std::abs(double(oracle.scores[j])),
                          "TAAT/exhaustive oracle score disagreement");
                    auto begin = truth.ids.begin() + qi * o.k;
                    shared += std::find(begin, begin + o.k, oracle.ids[j]) != begin + o.k;
                }
                audit.push_back({{"query_id", qi}, {"id_recall", double(shared) / o.k}, {"rank_scores_passed", true}});
            }
            Json legacy = nullptr;
            if (!o.truth.empty()) {
                const int compare_k = std::min(o.k, 10);
                auto previous = LoadTruth(o.truth, selected, file_queries, base.file_rows, compare_k);
                size_t shared = 0;
                for (size_t q = 0; q < o.queries; ++q)
                    for (int j = 0; j < compare_k; ++j) {
                        Check(std::abs(double(previous.scores[q * compare_k + j]) - truth.scores[q * o.k + j]) <=
                                  1e-4 + 2e-5 * std::abs(double(truth.scores[q * o.k + j])),
                              "Generated truth disagrees with supplied top-10 scores");
                        auto begin = truth.ids.begin() + q * o.k;
                        shared +=
                            std::find(begin, begin + compare_k, previous.ids[q * compare_k + j]) != begin + compare_k;
                    }
                legacy = {{"path", o.truth},
                          {"sha256", Sha256(o.truth)},
                          {"rank_scores_passed", true},
                          {"id_recall", double(shared) / (o.queries * compare_k)}};
            }
            std::vector<uint32_t> ids;
            ids.reserve(truth.ids.size());
            for (auto id : truth.ids) {
                Check(id >= 0 && uint64_t(id) < base.rows.size(), "Legacy IP truth format requires k positive hits");
                ids.push_back(id);
            }
            std::ofstream output(o.generate_truth, std::ios::binary);
            const uint32_t rows = o.queries, width = o.k;
            output.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            output.write(reinterpret_cast<const char*>(&width), sizeof(width));
            output.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(uint32_t));
            output.write(reinterpret_cast<const char*>(truth.scores.data()), truth.scores.size() * sizeof(float));
            output.close();
            Check(output.good(), "Truth write failed");
            Json provenance = {{"method", "TAAT_NAIVE FP32, no pruning"},
                               {"base_csr_sha256", base_sha},
                               {"query_csr_sha256", query_sha},
                               {"base_rows", base.rows.size()},
                               {"queries", o.queries},
                               {"split", o.split},
                               {"k", o.k},
                               {"config", exact_config},
                               {"threads", o.threads},
                               {"build_seconds", build_seconds},
                               {"search_seconds", search_seconds},
                               {"independent_scores_passed", true},
                               {"exhaustive_samples", audit},
                               {"supplied_top10", legacy},
                               {"truth_sha256", Sha256(o.generate_truth)}};
            struct rusage usage{};
            getrusage(RUSAGE_SELF, &usage);
            provenance["peak_rss_kib"] = usage.ru_maxrss;
            std::ofstream metadata(o.generate_truth + ".json");
            metadata << provenance.dump(2) << '\n';
            metadata.close();
            Check(metadata.good(), "Truth provenance write failed");
            std::cout << "Exact truth generated and independently checked: " << o.generate_truth << '\n';
            return 0;
        }
        Truth truth;
        if (o.brute) {
            auto gt = knowhere::BruteForce::SearchSparse(bd, qd, search, nullptr);
            Check(gt.has_value(), "Brute force failed: " + gt.what());
            truth = Capture(gt.value(), o.queries * o.k);
        } else if (bm) {
            // Existing truth's ratio is provenance of its exact generator, not
            // a constraint on the approximate index's bound ratio.
            auto truth_search = search;
            truth_search["dim_max_score_ratio"] = 1.05;
            Json sig = {{"sha256", manifest.at("sha256")},
                        {"base_rows", base.rows.size()},
                        {"query_ids", selected},
                        {"search", truth_search}};
            if (manifest.value("schema_version", 0) >= 1) {
                sig["query_split"] = o.split;
                sig["query_csr_sha256"] = manifest.at("sha256").at("queries." + o.split + ".idf.csr");
            }
            truth = ReadBmTruth(o.truth, sig);
        } else {
            if (o.truth.empty())
                o.truth = o.data + "/base_full." + o.split + ".gt";
            truth = LoadTruth(o.truth, selected, file_queries, base.file_rows, o.k);
        }
        ValidateTruth(truth, o.queries, o.k, base.rows.size());
        const double truth_seconds = Seconds(start);
        Json truth_audit = {{"queries", 0}};
        if (o.truth_audit_queries && !o.brute) {
            start = Clock::now();
            std::vector<uint64_t> audit_ids;
            for (size_t i = 0, n = std::min<size_t>(o.truth_audit_queries, o.queries); i < n; ++i)
                audit_ids.push_back(o.offset + i * o.queries / n);
            auto audit_queries = LoadCsr(query_path, &audit_ids);
            auto audit = knowhere::BruteForce::SearchSparse(bd, audit_queries.Dataset(), search, nullptr);
            Check(audit.has_value(), "Exhaustive truth audit failed");
            auto exact = Capture(audit.value(), audit_ids.size() * o.k);
            size_t shared = 0, valid = 0;
            for (size_t q = 0; q < audit_ids.size(); ++q) {
                const size_t cached = (audit_ids[q] - o.offset) * o.k;
                for (int rank = 0; rank < o.k; ++rank) {
                    const size_t p = q * o.k + rank;
                    Check((exact.ids[p] < 0) == (truth.ids[cached + rank] < 0), "Truth audit padding mismatch");
                    if (exact.ids[p] < 0)
                        continue;
                    ++valid;
                    auto begin = truth.ids.begin() + cached;
                    shared += std::find(begin, begin + o.k, exact.ids[p]) != begin + o.k;
                    Check(std::abs(double(exact.scores[p]) - truth.scores[cached + rank]) <=
                              1e-4 + 2e-5 * std::abs(double(exact.scores[p])),
                          "Truth audit rank score mismatch at query " + std::to_string(audit_ids[q]));
                }
            }
            truth_audit = {{"queries", audit_ids.size()},
                           {"query_ids", audit_ids},
                           {"seconds", Seconds(start)},
                           {"id_recall", valid ? double(shared) / valid : 1.0},
                           {"rank_scores_passed", true}};
        }
        Json build = search;
        build["dim"] = base.dim;
        build["inverted_index_algo"] = "SINDI";
        build["quant_type"] = o.quant;
        build["sindi_window_size"] = o.window;
        build["sindi_h2"] = o.h2;
        const auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
        auto create = [&]() {
            auto result = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
                knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version);
            Check(result.has_value(), "Index creation failed: " + result.what());
            return std::move(result.value());
        };
        Json contract = {{"base_csr_sha256", base_sha}, {"base_rows", base.rows.size()},
                         {"index_version", version},    {"metric", o.metric},
                         {"window", o.window},          {"quant", o.quant}};
        if (bm)
            for (auto key : {"bm25_k1", "bm25_b", "bm25_avgdl"}) contract[key] = search[key];
        auto index = create();
        start = Clock::now();
        if (o.load.empty())
            Check(index.Build(bd, build) == knowhere::Status::success, "Build failed");
        else {
            Check(ReadJson(o.load + ".json") == contract, "Index dataset/configuration mismatch");
            Check(index.DeserializeFromFile(o.load, build) == knowhere::Status::success, "mmap load failed");
        }
        const double operation_seconds = Seconds(start);
        start = Clock::now();
        Json storage = {{"resolved_quant_type", "fp16"}};
        if (!o.save.empty() || o.reload || bm) {
            knowhere::BinarySet bs;
            Check(index.Serialize(bs) == knowhere::Status::success, "Serialize failed");
            auto binary = bs.GetByName(index.Type());
            Check(binary != nullptr, "Missing serialized index");
            storage["serialized_bytes"] = binary->size;
            if (bm) {
                auto type =
                    knowhere::sparse::inverted::peek_quant_type_from_index_data(binary->data.get(), binary->size);
                storage["resolved_quant_type"] =
                    type == knowhere::sparse::inverted::InvertedIndexQuantType::BM25_U8 ? "u8" : "u16";
            } else
                storage["resolved_quant_type"] = "fp16";
            if (!o.save.empty()) {
                Check(!std::filesystem::exists(o.save) && !std::filesystem::exists(o.save + ".json"),
                      "Index output already exists");
                std::ofstream f(o.save, std::ios::binary);
                f.write(reinterpret_cast<const char*>(binary->data.get()), binary->size);
                f.close();
                Check(f.good(), "Index write failed");
                std::ofstream m(o.save + ".json");
                m << contract.dump(2);
                m.close();
                Check(m.good(), "Index metadata write failed");
            }
            if (o.reload) {
                auto loaded = create();
                Check(loaded.Deserialize(bs, build) == knowhere::Status::success, "BinarySet reload failed");
                auto original = index.Search(qd, search, nullptr), restored = loaded.Search(qd, search, nullptr);
                Check(original.has_value() && restored.has_value(), "Reload search failed");
                auto a = Capture(original.value(), o.queries * o.k), b = Capture(restored.value(), o.queries * o.k);
                Check(a.ids == b.ids && a.scores == b.scores, "BinarySet reload changed results");
                storage["binaryset_reload_passed"] = true;
            }
        }
        const double serialization_seconds = Seconds(start);
        start = Clock::now();
        for (int i = 0; i < o.warmup; ++i) {
            auto r = index.Search(qd, search, nullptr);
            Check(r.has_value(), "Warmup failed: " + r.what());
        }
        const double warmup_seconds = Seconds(start);
        Json meta = {{"metric", o.metric},
                     {"input_hash_seconds", input_hash_seconds},
                     {"truth_load_or_generate_seconds", truth_seconds},
                     {"truth_audit", truth_audit},
                     {"serialization_reload_seconds", serialization_seconds},
                     {"warmup_seconds", warmup_seconds},
                     {"split", o.split},
                     {"build_config", build},
                     {"search_config", search},
                     {"threads", o.threads},
                     {"warmup_batches", o.warmup},
                     {"repeats", o.repeats},
                     {"h2", o.h2},
                     {"input_load_seconds", input_seconds},
                     {"index_operation", o.load.empty() ? "build" : "mmap_load"},
                     {"index_operation_seconds", operation_seconds},
                     {"index_bytes", index.Size()},
                     {"storage", storage},
                     {"base_path", base_path},
                     {"query_path", query_path},
                     {"query_ids", selected},
                     {"contract", contract},
                     {"query_csr_sha256", query_sha},
                     {"truth_path", o.truth},
                     {"brute_force", o.brute},
                     {"compiler", __VERSION__},
                     {"source_revision", KNOWHERE_SPARSE_BENCHMARK_REVISION},
                     {"measurements", Json::array()}};
#if defined(__linux__) && defined(__aarch64__) && defined(PR_SVE_GET_VL)
        const int vl = prctl(PR_SVE_GET_VL);
        meta["sve_vector_bytes"] = vl > 0 ? vl & PR_SVE_VL_LEN_MASK : 0;
#endif
        meta["truth_sha256"] = o.truth.empty() ? Json(nullptr) : Json(Sha256(o.truth));
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        Check(sched_getaffinity(0, sizeof(affinity), &affinity) == 0, "Cannot read CPU affinity");
        meta["cpu_affinity"] = Json::array();
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &affinity))
                meta["cpu_affinity"].push_back(i);
        Dl_info info{};
        const auto fn =
            bm ? (o.quant == "u16" || storage.value("resolved_quant_type", "") == "u16"
                      ? reinterpret_cast<void*>(knowhere::sparse::inverted::sindi::get_bm25_kernels().accumulate)
                      : reinterpret_cast<void*>(knowhere::sparse::inverted::sindi::get_bm25_u8_kernels().accumulate))
               : reinterpret_cast<void*>(knowhere::sparse::inverted::sindi::get_ip_kernels().accumulate);
        dladdr(fn, &info);
        meta["effective_kernel"] = info.dli_sname ? info.dli_sname : "unavailable";
        std::ofstream csv(o.output);
        Check(csv.good(), "Cannot write results");
        csv << "metric,quant_type,h2,window,k,threads,queries,base_rows,index_bytes,index_operation_seconds,repeat,"
               "search_seconds,qps,id_recall\n"
            << std::setprecision(12);
        Truth first;
        double total_seconds = 0;
        for (int rep = 0; rep < o.repeats; ++rep) {
            start = Clock::now();
            auto result = index.Search(qd, search, nullptr);
            const double seconds = Seconds(start);
            Check(result.has_value(), "Search failed: " + result.what());
            auto hits = Capture(result.value(), o.queries * o.k);
            const double recall = Recall(result.value(), truth, o.queries, o.k, base.rows.size());
            if (!rep)
                first = hits;
            else
                Check(first.ids == hits.ids && first.scores == hits.scores, "Results changed between repeats");
            total_seconds += seconds;
            meta["measurements"].push_back({{"seconds", seconds}, {"qps", o.queries / seconds}, {"id_recall", recall}});
            csv << o.metric << ',' << o.quant << ',' << o.h2 << ',' << o.window << ',' << o.k << ',' << o.threads << ','
                << o.queries << ',' << base.rows.size() << ',' << index.Size() << ',' << operation_seconds << ','
                << rep + 1 << ',' << seconds << ',' << o.queries / seconds << ',' << recall << '\n';
        }
        start = Clock::now();
        CheckHitScores(first, base, queries, o.k, search);
        meta["independent_score_check_seconds"] = Seconds(start);
        meta["independent_hit_scores_passed"] = true;
        meta["repeat_identity_passed"] = true;
        meta["pooled_qps"] = double(o.queries) * o.repeats / total_seconds;
        struct rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        meta["peak_rss_kib"] = usage.ru_maxrss;
        start = Clock::now();
        std::ofstream hits(o.output + ".hits.csv");
        hits << "query_id,rank,document_id,score\n" << std::setprecision(9);
        for (size_t i = 0; i < first.ids.size(); ++i)
            hits << selected[i / o.k] << ',' << i % o.k + 1 << ',' << first.ids[i] << ',' << first.scores[i] << '\n';
        hits.close();
        Check(hits.good(), "Hit output failed");
        meta["hit_write_seconds"] = Seconds(start);
        csv.close();
        Check(csv.good(), "CSV output failed");
        std::ofstream json(o.output + ".json");
        json << meta.dump(2) << '\n';
        json.close();
        Check(json.good(), "JSON output failed");
        std::cout << o.metric << " " << o.quant << " pooled QPS=" << meta["pooled_qps"]
                  << " independent scores passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "benchmark_sparse: " << e.what() << '\n';
        return 1;
    }
}
