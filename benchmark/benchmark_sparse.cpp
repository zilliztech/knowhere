// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

#include <dlfcn.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
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

struct Options {
    std::string data, output = "sparse_results.csv", method = "all";
    uint64_t queries = 100, offset = 0, base_limit = 0;
    bool refine = false;
    float mass = 1, factor = 1, drop = 0;
    std::string sweep, split = "dev";
    int threads = std::max(2u, std::thread::hardware_concurrency()), repeats = 3, k = 10;
    std::optional<uint32_t> seed;
    int samples = 100;
    uint32_t pareto_seed = 42;
    float mass_min = .4f, mass_max = 1, factor_min = 1, factor_max = 20;
    std::string scheme = "mass";
};

Options
Parse(int argc, char** argv) {
    Options o;
    std::set<std::string> supplied;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") {
            std::cout
                << "benchmark_sparse --data-dir DIR [--queries 100] [--query-offset 0] [--seed N]\n"
                   "  [--threads N] [--repeats 3] [--k 10] [--output sparse_results.csv]\n"
                   "  [--refine 0|1] [--mass 1] [--factor 1] [--drop 0]\n"
                   "  [--sweep mass|drop|pareto] [--split dev|hidden]\n"
                   "  Pareto: [--scheme mass] [--samples 100] [--pareto-seed 42]\n"
                   "          [--mass-min 0.4] [--mass-max 1] [--factor-min 1] [--factor-max 20]\n"
                   "  Pareto writes .points.csv, .frontier.csv and .pareto.svg beside repeat CSV/JSON.\n"
                   "  [--method "
                   "all|TAAT_NAIVE|DAAT_WAND|DAAT_MAXSCORE|BLOCK_MAX_MAXSCORE|BLOCK_MAX_WAND|SINDI|SPARSE_WAND]\n"
                   "  [--base-limit N]  Reduced-base validation against brute force, not supplied ground truth.\n";
            std::exit(0);
        }
        Check(i + 1 < argc, "Missing value for " + key);
        const std::string value = argv[++i];
        supplied.insert(key);
        if (key == "--data-dir")
            o.data = value;
        else if (key == "--output")
            o.output = value;
        else if (key == "--refine")
            o.refine = std::stoi(value) != 0;
        else if (key == "--mass")
            o.mass = std::stof(value);
        else if (key == "--factor")
            o.factor = std::stof(value);
        else if (key == "--scheme")
            o.scheme = value;
        else if (key == "--mass-min")
            o.mass_min = std::stof(value);
        else if (key == "--mass-max")
            o.mass_max = std::stof(value);
        else if (key == "--factor-min")
            o.factor_min = std::stof(value);
        else if (key == "--factor-max")
            o.factor_max = std::stof(value);
        else if (key == "--drop")
            o.drop = std::stof(value);
        else if (key == "--sweep")
            o.sweep = value;
        else if (key == "--split")
            o.split = value;
        else if (key == "--method")
            o.method = value;
        else {
            Check(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                  "Expected unsigned integer for " + key);
            const uint64_t n = std::stoull(value);
            if (key == "--queries")
                o.queries = n;
            else if (key == "--query-offset")
                o.offset = n;
            else if (key == "--base-limit")
                o.base_limit = n;
            else if (key == "--seed") {
                Check(n <= UINT32_MAX, "Seed out of range");
                o.seed = n;
            } else if (key == "--pareto-seed") {
                Check(n <= UINT32_MAX, "Pareto seed out of range");
                o.pareto_seed = n;
            } else {
                Check(n > 0 && n <= INT32_MAX, "Option out of range: " + key);
                if (key == "--samples")
                    o.samples = n;
                else if (key == "--threads")
                    o.threads = n;
                else if (key == "--repeats")
                    o.repeats = n;
                else if (key == "--k")
                    o.k = n;
                else
                    throw std::runtime_error("Unknown option: " + key);
            }
        }
    }
    Check(o.sweep.empty() || o.sweep == "mass" || o.sweep == "drop" || o.sweep == "pareto",
          "Unsupported sweep: " + o.sweep + ". Use mass/pareto for mass refinement or drop for count pruning.");
    for (const auto* key :
         {"--scheme", "--samples", "--pareto-seed", "--mass-min", "--mass-max", "--factor-min", "--factor-max"}) {
        Check(o.sweep == "pareto" || !supplied.contains(key), std::string(key) + " requires --sweep pareto");
    }
    if (o.sweep == "pareto") {
        if (!supplied.contains("--method"))
            o.method = "SINDI";
        if (!supplied.contains("--refine"))
            o.refine = true;
        if (!supplied.contains("--queries"))
            o.queries = 3000;
        if (!supplied.contains("--threads"))
            o.threads = 8;
        Check(o.method == "SINDI" && o.refine && o.scheme == "mass" && o.drop == 0,
              "Pareto requires --method SINDI --refine 1 --scheme mass and --drop 0");
        Check(!supplied.contains("--mass") && !supplied.contains("--factor"),
              "Pareto uses --mass-min/--mass-max and --factor-min/--factor-max instead of scalar --mass/--factor");
        Check(std::isfinite(o.mass_min) && std::isfinite(o.mass_max) && o.mass_min > 0 && o.mass_min <= o.mass_max &&
                  o.mass_max <= 1,
              "Pareto bounds require 0 < mass-min <= mass-max <= 1");
        Check(std::isfinite(o.factor_min) && std::isfinite(o.factor_max) && o.factor_min >= 1 &&
                  o.factor_min <= o.factor_max,
              "Pareto bounds require finite 1 <= factor-min <= factor-max");
    }
    Check(!o.data.empty() && o.queries > 0 && o.threads >= 2, "Provide --data-dir, positive queries, and >=2 threads");
    return o;
}

struct ParetoPoint {
    size_t id;
    float factor, mass;
    size_t pool;
    double qps, recall;
};

// Both objectives are maximized. Exact objective ties remain nondominated.
std::vector<bool>
ParetoFrontier(const std::vector<ParetoPoint>& points) {
    std::vector<bool> frontier(points.size(), true);
    for (size_t i = 0; i < points.size(); ++i) {
        for (const auto& q : points) {
            const auto& p = points[i];
            if (q.qps >= p.qps && q.recall >= p.recall && (q.qps > p.qps || q.recall > p.recall)) {
                frontier[i] = false;
                break;
            }
        }
    }
    return frontier;
}

void
WritePareto(const std::string& output, const std::vector<ParetoPoint>& points) {
    const auto frontier = ParetoFrontier(points);
    std::vector<size_t> edge;
    for (size_t i = 0; i < points.size(); ++i)
        if (frontier[i])
            edge.push_back(i);
    std::sort(edge.begin(), edge.end(), [&](size_t a, size_t b) {
        if (points[a].recall != points[b].recall)
            return points[a].recall > points[b].recall;
        if (points[a].qps != points[b].qps)
            return points[a].qps > points[b].qps;
        return points[a].id < points[b].id;
    });
    std::ofstream all(output + ".points.csv"), boundary(output + ".frontier.csv"), svg(output + ".pareto.svg");
    for (auto* stream : {&all, &boundary})
        *stream << "sample_id,scheme,refine_k,refine_query_mass_percentage,candidate_pool,mean_qps,mean_recall,pareto\n"
                << std::setprecision(17);
    auto write = [&](std::ostream& stream, size_t i) {
        const auto& p = points[i];
        stream << p.id << ",mass," << p.factor << ',' << p.mass << ',' << p.pool << ',' << p.qps << ',' << p.recall
               << ',' << frontier[i] << '\n';
    };
    for (size_t i = 0; i < points.size(); ++i) write(all, i);  // evaluation order
    for (auto i : edge) write(boundary, i);
    double max_qps = 1;
    for (const auto& p : points) max_qps = std::max(max_qps, p.qps * 1.05);
    auto x = [](double recall) { return 80 + 800 * recall; };
    auto y = [&](double qps) { return 530 - 460 * qps / max_qps; };
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"960\" height=\"620\" viewBox=\"0 0 960 620\">"
           "<rect width=\"960\" height=\"620\" fill=\"white\"/>"
           "<g font-family=\"sans-serif\" font-size=\"14\" fill=\"#222\">"
           "<text x=\"80\" y=\"28\">SINDI mass QPS / recall Pareto frontier</text>"
           "<text x=\"80\" y=\"50\">Blue: all samples; red: nondominated samples</text>";
    for (int tick = 0; tick <= 5; ++tick) {
        const double recall = tick / 5.0, qps = max_qps * tick / 5;
        svg << "<path d=\"M " << x(recall) << " 70 V 530 M 80 " << y(qps) << " H 880\" stroke=\"#ddd\"/>"
            << "<text x=\"" << x(recall) << "\" y=\"555\" text-anchor=\"middle\">" << recall << "</text>"
            << "<text x=\"72\" y=\"" << y(qps) + 5 << "\" text-anchor=\"end\">" << qps << "</text>";
    }
    svg << "<text x=\"480\" y=\"590\" text-anchor=\"middle\">Recall@k</text>"
           "<text transform=\"translate(20 300) rotate(-90)\" text-anchor=\"middle\">QPS</text>"
           "<polyline fill=\"none\" stroke=\"#d33\" stroke-width=\"2\" points=\"";
    for (auto i : edge) svg << x(points[i].recall) << ',' << y(points[i].qps) << ' ';
    svg << "\"/>";
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        svg << "<circle cx=\"" << x(p.recall) << "\" cy=\"" << y(p.qps) << "\" r=\"4\" fill=\""
            << (frontier[i] ? "#d33" : "#4682b4") << "\"><title>sample=" << p.id << " refine_k=" << p.factor
            << " mass=" << p.mass << " pool=" << p.pool << " QPS=" << p.qps << " recall=" << p.recall
            << "</title></circle>";
    }
    svg << "</g></svg>\n";
    for (auto* stream : {&all, &boundary, &svg}) {
        stream->flush();
        Check(stream->good(), "Cannot write Pareto artifacts for " + output);
    }
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

Json
RecallMismatches(const knowhere::DataSetPtr& result, const Truth& truth, const std::vector<uint64_t>& selected, int k) {
    Json mismatches = Json::array();
    for (size_t q = 0; q < selected.size(); ++q) {
        const auto offset = q * k;
        std::set<int64_t> expected(truth.ids.begin() + offset, truth.ids.begin() + offset + k);
        std::set<int64_t> actual(result->GetIds() + offset, result->GetIds() + offset + k);
        if (expected != actual) {
            mismatches.push_back(
                {{"query_id", selected[q]},
                 {"expected_ids", std::vector<int64_t>(truth.ids.begin() + offset, truth.ids.begin() + offset + k)},
                 {"expected_scores",
                  std::vector<float>(truth.scores.begin() + offset, truth.scores.begin() + offset + k)},
                 {"actual_ids", std::vector<int64_t>(result->GetIds() + offset, result->GetIds() + offset + k)},
                 {"actual_scores",
                  std::vector<float>(result->GetDistance() + offset, result->GetDistance() + offset + k)}});
        }
    }
    return mismatches;
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        const auto o = Parse(argc, argv);
        const auto& ip_kernels = knowhere::sparse::inverted::sindi::get_ip_kernels();
        Dl_info accumulate_info{}, insert_info{};
        Check(dladdr(reinterpret_cast<void*>(ip_kernels.accumulate), &accumulate_info) != 0 &&
                  accumulate_info.dli_sname != nullptr,
              "Cannot identify IP kernel");
        Check(dladdr(reinterpret_cast<void*>(ip_kernels.batch_insert), &insert_info) != 0 &&
                  insert_info.dli_sname != nullptr,
              "Cannot identify selection kernel");
        Check(std::string(accumulate_info.dli_sname).find("sve") != std::string::npos &&
                  std::string(insert_info.dli_sname).find("sve") != std::string::npos,
              "Baseline requires SVE kernels");
        const int sve_vl = prctl(PR_SVE_GET_VL);
        Check(sve_vl > 0, "Cannot read SVE vector length");
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        Check(sched_getaffinity(0, sizeof(affinity), &affinity) == 0, "Cannot read CPU affinity");
        std::vector<int> cpus;
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &affinity))
                cpus.push_back(i);
        std::cerr << "IP kernel=" << accumulate_info.dli_sname << " SVE bytes=" << (sve_vl & PR_SVE_VL_LEN_MASK)
                  << std::endl;
        const std::vector<std::string> methods = {"TAAT_NAIVE",     "DAAT_WAND", "DAAT_MAXSCORE", "BLOCK_MAX_MAXSCORE",
                                                  "BLOCK_MAX_WAND", "SINDI",     "SPARSE_WAND"};
        Check(o.method == "all" || std::find(methods.begin(), methods.end(), o.method) != methods.end(),
              "Unknown method: " + o.method);
        knowhere::KnowhereConfig::SetBuildThreadPoolSize(o.threads);
        knowhere::KnowhereConfig::SetSearchThreadPoolSize(o.threads);
        const auto version = knowhere::Version::GetMaximumVersion().VersionNumber();
        Check(o.split == "dev" || o.split == "hidden", "Invalid split");
        const auto query_path = o.data + "/queries." + o.split + ".csr";
        uint64_t file_queries;
        {
            MappedFile query_file(query_path);
            file_queries = query_file.Read<uint64_t>(0);
            Check(file_queries <= query_file.size / 8, "Invalid query count");
        }
        Check(o.offset < file_queries && o.queries <= file_queries - o.offset, "Query selection outside dataset");
        std::vector<uint64_t> selected(file_queries - o.offset);
        std::iota(selected.begin(), selected.end(), o.offset);
        if (o.seed) {
            std::mt19937 rng(*o.seed);
            std::shuffle(selected.begin(), selected.end(), rng);
        }
        selected.resize(o.queries);
        auto queries = LoadCsr(query_path, &selected);
        auto load_start = Clock::now();
        std::cerr << "Loading base vectors..." << std::endl;
        auto base = LoadCsr(o.data + "/base_full.csr", nullptr, o.base_limit);
        Check(base.rows.size() >= static_cast<size_t>(o.k), "Base smaller than k");
        auto load_seconds = Seconds(load_start);
        // Allow a query file with a smaller declared vocabulary; use a shared dimension.
        base.dim = queries.dim = std::max(base.dim, queries.dim);
        std::set<uint32_t> indexed_dimensions;
        if (o.refine) {
            for (const auto& row : base.rows)
                for (size_t i = 0; i < row.size(); ++i)
                    if (std::abs(row[i].val) >= std::numeric_limits<float>::epsilon())
                        indexed_dimensions.insert(row[i].id);
        }
        auto base_ds = base.Dataset(), query_ds = queries.Dataset();
        Json search = {{"metric_type", "IP"},         {"k", o.k},           {"drop_ratio_search", 0.0},
                       {"dim_max_score_ratio", 1.05}, {"refine_factor", 1}, {"search_algo", "INHERIT"}};
        Truth truth;
        if (o.base_limit) {
            std::cerr << "Computing reduced-base FP32 brute-force ground truth..." << std::endl;
            auto gt = knowhere::BruteForce::SearchSparse(base_ds, query_ds, search, nullptr);
            Check(gt.has_value(), "Brute force failed: " + gt.what());
            truth.ids.assign(gt.value()->GetIds(), gt.value()->GetIds() + o.queries * o.k);
            truth.scores.assign(gt.value()->GetDistance(), gt.value()->GetDistance() + o.queries * o.k);
        } else {
            truth = LoadTruth(o.data + "/base_full." + o.split + ".gt", selected, file_queries, base.file_rows, o.k);
        }
        struct utsname host {};
        uname(&host);
        Json metadata = {
            {"data_dir", std::filesystem::absolute(o.data).string()},
            {"base_rows", base.rows.size()},
            {"base_nnz", base.nnz},
            {"dimension", base.dim},
            {"query_ids", selected},
            {"query_nnz", queries.nnz},
            {"threads", o.threads},
            {"repeats", o.repeats},
            {"index_version", version},
            {"quant_type", "fp16"},
            {"search_config", search},
            {"base_load_seconds", load_seconds},
            {"machine", host.machine},
            {"hostname", host.nodename},
            {"kernel", host.release},
            {"compiler", __VERSION__},
            {"source_revision", KNOWHERE_SPARSE_BENCHMARK_REVISION},
            {"ground_truth", o.base_limit ? "reduced-base FP32 brute force" : "base_full." + o.split + ".gt"},
            {"warmup_batches", 1},
            {"runs", Json::array()}};
        metadata["effective_ip_kernel"] = accumulate_info.dli_sname;
        metadata["effective_selection_kernel"] = insert_info.dli_sname;
        metadata["sve_vector_bytes"] = sve_vl & PR_SVE_VL_LEN_MASK;
        metadata["cpu_affinity"] = cpus;
        std::ifstream cpu("/proc/cpuinfo"), mem("/proc/meminfo");
        metadata["cpuinfo"] = std::string(std::istreambuf_iterator<char>(cpu), {});
        metadata["meminfo"] = std::string(std::istreambuf_iterator<char>(mem), {});
        std::ofstream csv(o.output);
        Check(csv.good(), "Cannot write " + o.output);
        csv << "method,index_type,index_version,quant_type,codec,base_rows,queries,k,threads,build_seconds,index_bytes,"
               "repeat,search_seconds,qps,recall,mass,refine_k,drop_ratio,refine,sample_id\n"
            << std::setprecision(10);
        for (const auto& method : methods) {
            if (o.method != "all" && o.method != method)
                continue;
            const bool alias = method == "SPARSE_WAND";
            const auto type =
                alias ? knowhere::IndexEnum::INDEX_SPARSE_WAND : knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX;
            const std::string algo = alias ? "DAAT_WAND" : method;
            Json build = {
                {"dim", base.dim}, {"metric_type", "IP"}, {"inverted_index_algo", algo}, {"quant_type", "fp16"}};
            const std::string codec = algo == "SINDI" ? "fixed_docid_windows" : "block_streamvbyte";
            if (algo == "SINDI") {
                build["sindi_window_size"] = 4096;
                build["refine"] = o.refine;
            }
            if (algo != "SINDI")
                build["inverted_index_codec"] = codec;
            std::cerr << "Building " << method << "..." << std::endl;
            auto created = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(type, version);
            Check(created.has_value(), "Create failed: " + created.what());
            auto index = std::move(created.value());
            auto start = Clock::now();
            auto status = index.Build(base_ds, build);
            Check(status == knowhere::Status::success, method + " Build failed, status=" + std::to_string(int(status)));
            const double build_seconds = Seconds(start);
            const auto bytes = index.Size();
            std::vector<Json> settings;
            auto add_setting = [&](float mass, float factor, float drop) {
                auto cfg = search;
                cfg["refine_query_mass_percentage"] = mass;
                cfg["refine_k"] = factor;
                cfg["drop_ratio_search"] = drop;
                settings.push_back(cfg);
            };
            if (o.sweep == "pareto") {
                std::mt19937 rng(o.pareto_seed);
                auto sample = [&](float low, float high) {
                    const double unit = double(rng()) / 4294967296.0;
                    return static_cast<float>(double(low) + (double(high) - low) * unit);
                };
                for (int i = 0; i < o.samples; ++i) {
                    const float factor = sample(o.factor_min, o.factor_max);
                    const float mass = sample(o.mass_min, o.mass_max);
                    add_setting(mass, factor, 0);
                }
                metadata["pareto"] = {{"scheme", o.scheme},
                                      {"samples", o.samples},
                                      {"seed", o.pareto_seed},
                                      {"mass_min", o.mass_min},
                                      {"mass_max", o.mass_max},
                                      {"factor_min", o.factor_min},
                                      {"factor_max", o.factor_max},
                                      {"settings", settings},
                                      {"sampling", "mt19937 uint32 / 2^32; factor then mass"}};
            } else if (o.sweep == "mass") {
                for (float mass : {1.0f, .9f, .8f, .7f, .6f, .5f})
                    for (float factor : {1.0f, 5.0f, 10.0f}) add_setting(mass, factor, 0);
            } else if (o.sweep == "drop") {
                for (float drop : {0.0f, .3f, .5f, .7f, .9f}) add_setting(1, 1, drop);
            } else
                add_setting(o.mass, o.factor, o.drop);
            std::vector<ParetoPoint> points;
            size_t setting_id = 0;
            for (const auto& setting : settings) {
                search = setting;
                const size_t sample_id = setting_id;
                const std::string hit_path = o.output + ".case" + std::to_string(setting_id++) + ".hits.csv";
                {
                    auto warmup = index.Search(query_ds, search, nullptr);
                    Check(warmup.has_value(), "Warmup failed: " + warmup.what());
                }
                Json run = {{"method", method},        {"index_type", type},
                            {"build_config", build},   {"effective_codec", codec},
                            {"search_config", search}, {"build_seconds", build_seconds},
                            {"index_bytes", bytes},    {"measurements", Json::array()}};
                run["sample_id"] = sample_id;
                double qps_sum = 0, recall_sum = 0;
                std::vector<int64_t> first_ids;
                std::vector<float> first_scores;
                for (int repeat = 0; repeat < o.repeats; ++repeat) {
                    start = Clock::now();
                    auto result = index.Search(query_ds, search, nullptr);
                    const double seconds = Seconds(start);
                    Check(result.has_value(), "Search failed: " + result.what());
                    const size_t result_count = o.queries * o.k;
                    const auto* ids = result.value()->GetIds();
                    const auto* scores = result.value()->GetDistance();
                    uint64_t result_hash = 14695981039346656037ULL;
                    for (size_t i = 0; i < result_count; ++i) {
                        result_hash = (result_hash ^ static_cast<uint64_t>(ids[i])) * 1099511628211ULL;
                        result_hash = (result_hash ^ std::bit_cast<uint32_t>(scores[i])) * 1099511628211ULL;
                    }
                    if (repeat == 0) {
                        first_ids.assign(ids, ids + result_count);
                        first_scores.assign(scores, scores + result_count);
                        std::ofstream hits(hit_path);
                        hits << "query_id,rank,document_id,score\n" << std::setprecision(9);
                        for (size_t i = 0; i < result_count; ++i)
                            hits << selected[i / o.k] << ',' << i % o.k + 1 << ',' << ids[i] << ',' << scores[i]
                                 << '\n';
                        hits.flush();
                        Check(hits.good(), "Cannot write hit file");
                    } else {
                        Check(std::memcmp(ids, first_ids.data(), result_count * sizeof(*ids)) == 0 &&
                                  std::memcmp(scores, first_scores.data(), result_count * sizeof(*scores)) == 0,
                              "Results changed across measured repetitions");
                    }
                    const auto recall = Recall(result.value(), truth, o.queries, o.k, base.rows.size());
                    if (repeat == 0) {
                        run["recall_mismatches"] = RecallMismatches(result.value(), truth, selected, o.k);
                    }
                    const double qps = o.queries / seconds;
                    Check(std::isfinite(qps) && qps > 0 && std::isfinite(recall) && recall >= 0 && recall <= 1,
                          "Invalid Pareto measurement");
                    qps_sum += qps;
                    recall_sum += recall;
                    csv << method << ',' << type << ',' << version << ",fp16," << codec << ',' << base.rows.size()
                        << ',' << o.queries << ',' << o.k << ',' << o.threads << ',' << build_seconds << ',' << bytes
                        << ',' << repeat + 1 << ',' << seconds << ',' << qps << ',' << recall << ','
                        << search["refine_query_mass_percentage"] << ',' << search["refine_k"] << ','
                        << search["drop_ratio_search"] << ',' << o.refine << ',' << sample_id << '\n';
                    csv.flush();
                    Check(csv.good(), "Failed writing results");
                    run["measurements"].push_back({{"repeat", repeat + 1},
                                                   {"search_seconds", seconds},
                                                   {"qps", qps},
                                                   {"recall", recall},
                                                   {"result_hash", result_hash}});
                    std::cout << method << " repeat=" << repeat + 1 << " seconds=" << seconds << " QPS=" << qps
                              << " recall@" << o.k << '=' << recall << std::endl;
                }
                // Separate, untimed coverage diagnostics. Materialize exactly the coarse query.
                SparseData selected_data;
                selected_data.dim = queries.dim;
                double mass_sum = 0;
                size_t retained_nnz = 0;
                for (const auto& q : queries.rows) {
                    Row selected;
                    Row eligible;
                    if (o.refine) {
                        eligible = knowhere::sparse::inverted::sindi::filter_query_dimensions(
                            q, [&](uint32_t dim) { return indexed_dimensions.contains(dim); });
                        selected = knowhere::sparse::inverted::sindi::retain_query_mass(
                            eligible, search["refine_query_mass_percentage"].get<float>());
                    } else {
                        std::vector<float> weights;
                        for (size_t j = 0; j < q.size(); ++j) weights.push_back(q[j].val);
                        std::sort(weights.begin(), weights.end());
                        const size_t count = size_t(search["drop_ratio_search"].get<float>() * weights.size());
                        float threshold = weights.empty() ? 0 : weights[std::min(count, weights.size() - 1)];
                        std::vector<std::pair<uint32_t, float>> terms;
                        for (size_t j = 0; j < q.size(); ++j)
                            if (q[j].val >= threshold)
                                terms.emplace_back(q[j].id, q[j].val);
                        selected = Row(terms);
                    }
                    double total = 0, retained = 0;
                    const auto& mass_query = o.refine ? eligible : q;
                    for (size_t j = 0; j < mass_query.size(); ++j) total += mass_query[j].val;
                    for (size_t j = 0; j < selected.size(); ++j) retained += selected[j].val;
                    mass_sum += total ? retained / total : 1;
                    retained_nnz += selected.size();
                    selected_data.rows.push_back(std::move(selected));
                }
                const size_t pool = knowhere::sparse::inverted::sindi::refinement_pool_size(
                    o.k, search["refine_k"].get<float>(), base.rows.size());
                auto diagnostic = search;
                diagnostic["k"] = pool;
                diagnostic["refine_query_mass_percentage"] = 1;
                diagnostic["refine_k"] = 1;
                diagnostic["drop_ratio_search"] = 0;
                auto coarse = index.Search(selected_data.Dataset(), diagnostic, nullptr);
                Check(coarse.has_value(), "Coarse coverage diagnostic failed");
                size_t covered = 0;
                for (size_t q = 0; q < o.queries; ++q)
                    for (int j = 0; j < o.k; ++j)
                        covered +=
                            std::find(coarse.value()->GetIds() + q * pool, coarse.value()->GetIds() + (q + 1) * pool,
                                      truth.ids[q * o.k + j]) != coarse.value()->GetIds() + (q + 1) * pool;
                run["mean_qps"] = qps_sum / o.repeats;
                run["mean_recall"] = recall_sum / o.repeats;
                if (o.sweep == "pareto") {
                    points.push_back({sample_id, search["refine_k"].get<float>(),
                                      search["refine_query_mass_percentage"].get<float>(), pool, qps_sum / o.repeats,
                                      recall_sum / o.repeats});
                    WritePareto(o.output, points);
                }
                run["candidate_pool"] = pool;
                run["candidate_coverage"] = double(covered) / (o.queries * o.k);
                run["retained_query_nnz_mean"] = double(retained_nnz) / o.queries;
                run["retained_mass_mean"] = mass_sum / o.queries;
                metadata["runs"].push_back(std::move(run));
                std::ofstream meta(o.output + ".json");
                meta << metadata.dump(2) << '\n';
                meta.flush();
                Check(meta.good(), "Cannot write metadata");
            }  // search settings
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "benchmark_sparse: " << e.what() << '\n';
        return 1;
    }
}
