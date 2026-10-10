// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <dlfcn.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

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
#include <numeric>
#include <optional>
#include <random>
#include <set>
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
#include "src/index/sparse/sindi_bm25_u4.h"
#include "src/index/sparse/sindi_refinement.h"
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
        struct stat st {};
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
            Check(std::isfinite(value) && value >= 0, "Expected finite nonnegative SPLADE weights");
            out.rows.back().set_at(j - start, col, value);
            last = col;
        }
        out.nnz += end - start;
    }
    return out;
}

Json
ReadJson(const std::string& path) {
    std::ifstream f(path);
    Check(f.good(), "Cannot read " + path);
    Json j;
    f >> j;
    return j;
}

double
Score(const Row& query, const Row& document, double dl, double k1, double b, double avgdl,
      const knowhere::sparse::inverted::sindi::Bm25U4Lut* lut) {
    double score = 0;
    size_t i = 0, j = 0;
    while (i < query.size() && j < document.size()) {
        if (query[i].id < document[j].id)
            ++i;
        else if (query[i].id > document[j].id)
            ++j;
        else {
            double tf = document[j].val;
            if (lut)
                tf = lut->decode[lut->encode[std::min<unsigned>(tf, 255)]];
            score += query[i].val * (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl / avgdl));
            ++i;
            ++j;
        }
    }
    return score;
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
        auto get = [&](std::string name, std::string fallback) { return args.count(name) ? args.at(name) : fallback; };
        const std::string data = get("--data-dir", "/volume/data_sparse/nq"), split = get("--split", "test"),
                          quant = get("--quant-type", "u8"), output = get("--output", "bm25_compact.json");
        const bool u4 = quant == "u4_lut" || quant == "u4_lut_u12" || quant == "u4_lut_u16";
        const bool u16_ids = quant == "u4_lut_u16";
        const size_t k = std::stoul(get("--k", "10")), limit = std::stoul(get("--queries", "0"));
        const int threads = std::stoi(get("--threads", "8")), requested = std::stoi(get("--repeats", "3"));
        Check(k > 0 && (k == 10 || k == 100) && threads > 0 && requested >= 3, "Invalid benchmark arguments");
        const auto simd = get("--simd", "auto");
        auto simd_type = knowhere::KnowhereConfig::SimdType::AUTO;
        if (simd == "avx2")
            simd_type = knowhere::KnowhereConfig::SimdType::AVX2;
        else if (simd == "scalar")
            simd_type = knowhere::KnowhereConfig::SimdType::GENERIC;
        else
            Check(simd == "auto", "Unsupported --simd mode (auto/avx2/scalar)");
        const auto selected_simd = knowhere::KnowhereConfig::SetSimdType(simd_type);
        knowhere::KnowhereConfig::SetBuildThreadPoolSize(threads);
        knowhere::KnowhereConfig::SetSearchThreadPoolSize(threads);
        const auto manifest = ReadJson(data + "/manifest.json");
        auto base = LoadCsr(data + "/sparse/base_tf.csr", nullptr);
        auto queries = LoadCsr(data + "/sparse/queries." + split + ".idf.csr", nullptr, limit);
        const float k1 = manifest.at("bm25_k1"), b = manifest.at("bm25_b"), avgdl = manifest.at("avgdl");
        const std::string truth_path =
            get("--truth", data + "/ground_truth/" + split + ".k" + std::to_string(k) + ".bm25");
        auto signature = ReadJson(truth_path + ".json");
        Check(signature.at("base_rows") == base.rows.size(), "Truth corpus mismatch");
        Check(signature.at("query_ids").size() == queries.file_rows, "Truth query count mismatch");
        for (size_t q = 0; q < queries.file_rows; ++q)
            Check(signature["query_ids"][q] == q, "Truth query order mismatch");
        for (auto key : {"bm25_k1", "bm25_b", "bm25_avgdl"}) {
            const float value = signature.at("search").at(key);
            const float expected = std::string(key) == "bm25_k1" ? k1 : (std::string(key) == "bm25_b" ? b : avgdl);
            Check(std::abs(value - expected) < 1e-5 * std::max(1.f, std::abs(expected)), "Truth scorer mismatch");
        }
        Check(signature.at("search").at("k") == k, "Truth k mismatch");
        if (signature.contains("query_csr_sha256")) {
            Check(signature["query_csr_sha256"] == manifest.at("sha256").at("queries." + split + ".idf.csr"),
                  "Truth query hash mismatch");
        }
        Check(signature.at("sha256").at("base_tf.csr") == manifest.at("sha256").at("base_tf.csr"),
              "Truth base hash mismatch");
        MappedFile truth(truth_path);
        const auto truth_n = truth.Read<uint64_t>(0);
        Check(truth_n == queries.file_rows * k && truth.size == 8 + 12 * truth_n, "Truth payload mismatch");
        size_t valid_truth = 0;
        for (size_t q = 0; q < queries.rows.size(); ++q) {
            std::set<int64_t> seen;
            bool padding = false;
            for (size_t j = 0; j < k; ++j) {
                const auto id = truth.Read<int64_t>(8 + 8 * (q * k + j));
                if (id < 0) {
                    padding = true;
                    continue;
                }
                Check(!padding && uint64_t(id) < base.rows.size() && seen.insert(id).second,
                      "Invalid truth ordering/ID");
                ++valid_truth;
            }
        }
        Check(valid_truth > 0, "No valid ground-truth neighbors");
        const auto type = knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
        auto index = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, 11).value();
        Json build{{"metric_type", "BM25"}, {"inverted_index_algo", "SINDI"},
                   {"quant_type", quant},   {"sindi_window_size", 4096},
                   {"refine", false},       {"bm25_k1", k1},
                   {"bm25_b", b},           {"bm25_avgdl", avgdl}};
        auto t = Clock::now();
        Check(index.Build(base.Dataset(), build) == knowhere::Status::success, "Build failed");
        const auto build_seconds = Seconds(t);
        const auto index_bytes = index.Size();
        knowhere::BinarySet bytes;
        Check(index.Serialize(bytes) == knowhere::Status::success, "Serialize failed");
        auto blob = bytes.GetByName(type);
        knowhere::sparse::inverted::sindi::Bm25U4Lut lut;
        Json meta{{"dataset", data},
                  {"split", split},
                  {"quant_type", quant},
                  {"threads", threads},
                  {"k", k},
                  {"queries", queries.rows.size()},
                  {"base_rows", base.rows.size()},
                  {"postings", base.nnz},
                  {"build_seconds", build_seconds},
                  {"index_bytes", index_bytes},
                  {"serialized_bytes", blob->size},
                  {"compiler", __VERSION__},
                  {"requested_simd", simd},
                  {"selected_simd", selected_simd},
                  {"build", build},
                  {"runs", Json::array()}};
        using namespace knowhere::sparse::inverted::sindi;
        void* kernel = u4              ? reinterpret_cast<void*>(get_packed_bm25_kernel(u16_ids))
                       : quant == "u8" ? reinterpret_cast<void*>(get_bm25_u8_kernels().accumulate)
                                       : reinterpret_cast<void*>(get_bm25_kernels().accumulate);
        Dl_info dispatch{};
        if (dladdr(kernel, &dispatch) && dispatch.dli_sname)
            meta["kernel_symbol"] = dispatch.dli_sname;
        if (u4) {
            using namespace knowhere::sparse::inverted;
            uint32_t sections;
            std::memcpy(&sections, blob->data.get() + 32, 4);
            for (size_t i = 0; i < sections; ++i) {
                InvertedIndexSectionHeader h;
                std::memcpy(&h, blob->data.get() + 36 + i * sizeof(h), sizeof(h));
                if (h.type == InvertedIndexSectionType::POSTING_LISTS) {
                    std::memcpy(lut.decode.data(), blob->data.get() + h.offset + 24, 16);
                    std::memcpy(lut.ends.data(), blob->data.get() + h.offset + 40, 16);
                }
            }
            lut.k1 = k1;
            lut.validate_and_encode();
            meta["lut"] = {
                {"decode", lut.decode}, {"ends", lut.ends}, {"encode", lut.encode}, {"lut_id", lut.fingerprint()}};
            meta["posting_payload_bytes"] =
                u16_ids ? 2 * base.nnz + base.nnz / 2 + base.nnz % 2 : 2 * base.nnz + base.nnz % 2;
        }
        std::vector<float> lengths(base.rows.size());
        for (size_t d = 0; d < base.rows.size(); ++d)
            for (size_t j = 0; j < base.rows[d].size(); ++j) lengths[d] += base.rows[d][j].val;
        Json search{{"metric_type", "BM25"},
                    {"k", k},
                    {"bm25_k1", k1},
                    {"bm25_b", b},
                    {"bm25_avgdl", avgdl},
                    {"drop_ratio_search", 0},
                    {"dim_max_score_ratio", 1.05}};
        auto warm = index.Search(queries.Dataset(), search, nullptr);
        Check(warm.has_value(), "Warmup failed");
        std::vector<int64_t> reference_ids(warm.value()->GetIds(), warm.value()->GetIds() + queries.rows.size() * k);
        std::vector<float> reference_scores(warm.value()->GetDistance(),
                                            warm.value()->GetDistance() + queries.rows.size() * k);
        auto validate = [&](const auto& result) {
            Check(result.has_value(), "Search failed");
            size_t correct = 0;
            for (size_t q = 0; q < queries.rows.size(); ++q) {
                std::set<int64_t> seen;
                for (size_t j = 0; j < k; ++j) {
                    const auto id = result.value()->GetIds()[q * k + j];
                    Check(id == reference_ids[q * k + j], "Repeated IDs differ");
                    if (id < 0)
                        continue;
                    Check(uint64_t(id) < base.rows.size() && seen.insert(id).second, "Invalid/duplicate result ID");
                    const auto score = result.value()->GetDistance()[q * k + j];
                    Check(std::abs(score - reference_scores[q * k + j]) < 1e-5f, "Repeated scores differ");
                    const auto expected =
                        Score(queries.rows[q], base.rows[id], lengths[id], k1, b, avgdl, u4 ? &lut : nullptr);
                    Check(std::abs(score - expected) < 3e-5 * std::max(1., std::abs(expected)),
                          "Returned score disagrees with oracle");
                    for (size_t g = 0; g < k; ++g)
                        if (id == truth.Read<int64_t>(8 + 8 * (q * k + g))) {
                            ++correct;
                            break;
                        }
                }
            }
            return double(correct) / valid_truth;
        };
        meta["valid_truth_positions"] = valid_truth;
        meta["recall_denominator"] = "nonnegative ground-truth IDs; exclude sentinel padding";
        size_t outside_truth = 0, near_tie_hits = 0, below_original_threshold = 0;
        for (size_t q = 0; q < queries.rows.size(); ++q) {
            double threshold = 0;
            std::set<int64_t> truth_ids;
            for (size_t j = 0; j < k; ++j) {
                const auto id = truth.Read<int64_t>(8 + 8 * (q * k + j));
                if (id >= 0) {
                    truth_ids.insert(id);
                    threshold = truth.Read<float>(8 + 8 * truth_n + 4 * (q * k + j));
                }
            }
            for (size_t j = 0; j < k; ++j) {
                const auto id = reference_ids[q * k + j];
                if (id < 0 || truth_ids.count(id))
                    continue;
                ++outside_truth;
                const auto score = Score(queries.rows[q], base.rows[id], lengths[id], k1, b, avgdl, nullptr);
                if (score + 3e-5 * std::max(1., std::abs(threshold)) >= threshold)
                    ++near_tie_hits;
                else
                    ++below_original_threshold;
            }
        }
        meta["original_tf_diagnostics"] = {{"returned_hits_outside_exact_id_truth", outside_truth},
                                           {"within_or_above_threshold_tolerance", near_tie_hits},
                                           {"below_threshold_tolerance", below_original_threshold},
                                           {"relative_tolerance", 3e-5}};
        const auto recall = validate(warm);
        double timed = 0;
        for (int repetition = 0; repetition < requested || timed < 5; ++repetition) {
            t = Clock::now();
            auto result = index.Search(queries.Dataset(), search, nullptr);
            const double seconds = Seconds(t);
            timed += seconds;
            const auto repeated_recall = validate(result);
            Check(repeated_recall == recall, "Repeated recall differs");
            meta["runs"].push_back({{"seconds", seconds}, {"qps", queries.rows.size() / seconds}, {"recall", recall}});
        }
        auto load = build;
        load.erase("quant_type");
        load.erase("sindi_window_size");
        auto restored = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, 11).value();
        t = Clock::now();
        Check(restored.Deserialize(bytes, load) == knowhere::Status::success, "Reload failed");
        meta["load_seconds"] = Seconds(t);
        validate(restored.Search(queries.Dataset(), search, nullptr));
        const std::string save = get("--index-save", "");
        if (!save.empty()) {
            Check(!std::filesystem::exists(save), "Refuse to overwrite index");
            std::ofstream f(save, std::ios::binary);
            f.write(reinterpret_cast<char*>(blob->data.get()), blob->size);
            Check(f.good(), "Index save failed");
            f.close();
            auto mapped = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, 11).value();
            t = Clock::now();
            Check(mapped.DeserializeFromFile(save, load) == knowhere::Status::success, "File/mmap reload failed");
            meta["mmap_load_seconds"] = Seconds(t);
            validate(mapped.Search(queries.Dataset(), search, nullptr));
        }
        // Independent exhaustive scans of deterministic samples verify the
        // decoded score domain and quantify original-TF ties/losses separately.
        meta["sample_oracles"] = Json::array();
        for (size_t q = 0; q < std::min<size_t>(8, queries.rows.size()); ++q) {
            std::vector<std::pair<double, size_t>> top;
            for (size_t d = 0; d < base.rows.size(); ++d) {
                const auto score = Score(queries.rows[q], base.rows[d], lengths[d], k1, b, avgdl, u4 ? &lut : nullptr);
                if (score > 0) {
                    top.emplace_back(score, d);
                    std::push_heap(top.begin(), top.end(), std::greater<>());
                    if (top.size() > k) {
                        std::pop_heap(top.begin(), top.end(), std::greater<>());
                        top.pop_back();
                    }
                }
            }
            const double threshold = top.empty() ? 0 : top.front().first;
            size_t missing = 0;
            for (size_t j = 0; j < k; ++j) {
                auto id = reference_ids[q * k + j];
                if (id < 0)
                    continue;
                const auto score =
                    Score(queries.rows[q], base.rows[id], lengths[id], k1, b, avgdl, u4 ? &lut : nullptr);
                missing += score + 3e-5 * std::max(1., std::abs(threshold)) < threshold;
            }
            Check(missing == 0, "Full-corpus decoded-score oracle ranking mismatch");
            meta["sample_oracles"].push_back({{"query", q}, {"threshold", threshold}, {"below_threshold", missing}});
        }
        struct rusage resources {};
        getrusage(RUSAGE_SELF, &resources);
        meta["peak_rss_kib"] = resources.ru_maxrss;
        meta["recall"] = recall;
        meta["mean_qps"] = queries.rows.size() * meta["runs"].size() / timed;
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        sched_getaffinity(0, sizeof(affinity), &affinity);
        std::vector<int> cores;
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &affinity))
                cores.push_back(i);
        meta["affinity"] = cores;
#ifdef __aarch64__
        meta["sve_vl_bytes"] = prctl(PR_SVE_GET_VL) & PR_SVE_VL_LEN_MASK;
#endif
        std::ofstream f(output);
        f << meta.dump(2) << '\n';
        Check(f.good(), "Output write failed");
        std::cout << quant << " QPS=" << meta["mean_qps"] << " recall=" << recall << " bytes=" << index_bytes << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
