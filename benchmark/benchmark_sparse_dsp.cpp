// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

//
// Sparse DSP benchmark: measures QPS, latency percentiles, recall, and result
// coverage for SPARSE_DSP_CC across a parameter sweep of (mode, mu, eta, gamma).
//
// Supports SPLADE/IP and MSMARCO/BM25 datasets in CSR binary format.
//
// Usage:
//   ./benchmark_sparse_dsp --data-dir ~/data/splade_full --metric IP
//       --gt ~/data/splade_full/base_small.dev.ip.gt
//   ./benchmark_sparse_dsp --data-dir ~/data/msmarco_full_bm25_v2 --metric BM25
//       --gt ~/data/msmarco_full_bm25_v2/base_small.dev.bm25.gt
//

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/operands.h"
#include "knowhere/sparse_utils.h"

// ============================================================================
// CSR binary file loader
// ============================================================================
struct CSRDataset {
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    std::vector<float> data;
    int64_t n_rows = 0;
    int64_t n_cols = 0;
    int64_t nnz = 0;

    bool
    load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            fprintf(stderr, "Error: Cannot open %s\n", path.c_str());
            return false;
        }
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(int64_t));
        printf("  Loading CSR: %ld rows, %ld cols, %ld nnz\n", n_rows, n_cols, nnz);

        indptr.resize(n_rows + 1);
        file.read(reinterpret_cast<char*>(indptr.data()), (n_rows + 1) * sizeof(int64_t));
        indices.resize(nnz);
        file.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int32_t));
        data.resize(nnz);
        file.read(reinterpret_cast<char*>(data.data()), nnz * sizeof(float));
        return file.good();
    }

    std::unique_ptr<knowhere::sparse::SparseRow<float>[]>
    to_sparse_rows() const {
        auto rows = std::make_unique<knowhere::sparse::SparseRow<float>[]>(n_rows);
        for (int64_t i = 0; i < n_rows; ++i) {
            int64_t start = indptr[i];
            int64_t end = indptr[i + 1];
            int64_t len = end - start;
            rows[i] = knowhere::sparse::SparseRow<float>(len);
            for (int64_t j = 0; j < len; ++j) {
                rows[i].set_at(j, indices[start + j], data[start + j]);
            }
        }
        return rows;
    }

    // Compute avgdl as sum of all values / n_rows.
    // For raw term-frequency CSR, this is the average document length.
    // For impact-weighted CSR, this is the average sum of impact scores.
    double compute_avgdl() const {
        double total = 0.0;
        for (int64_t i = 0; i < n_rows; ++i) {
            for (int64_t j = indptr[i]; j < indptr[i + 1]; ++j) {
                total += data[j];
            }
        }
        return total / n_rows;
    }

    void
    free_raw() {
        indptr.clear();
        indptr.shrink_to_fit();
        indices.clear();
        indices.shrink_to_fit();
        data.clear();
        data.shrink_to_fit();
    }
};

// ============================================================================
// Ground truth loader (binary: int32 nq, int32 k, then nq*k int32 IDs)
// ============================================================================
struct GroundTruth {
    std::vector<std::vector<int32_t>> ids;
    int64_t nq = 0;
    int64_t k = 0;

    bool
    load(const std::string& path, int64_t max_nq) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            fprintf(stderr, "Error: Cannot open GT file %s\n", path.c_str());
            return false;
        }
        int32_t nq32, k32;
        file.read(reinterpret_cast<char*>(&nq32), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&k32), sizeof(int32_t));
        nq = std::min(static_cast<int64_t>(nq32), max_nq);
        k = k32;
        printf("  Loading GT: %ld queries, k=%ld (file has %d queries)\n", nq, k, nq32);
        ids.resize(nq);
        for (int64_t i = 0; i < nq; ++i) {
            ids[i].resize(k);
            file.read(reinterpret_cast<char*>(ids[i].data()), k * sizeof(int32_t));
        }
        return true;
    }

    // Compute recall for a single query. Returns 0 if query has no results.
    float
    recall(const int64_t* result, int64_t qi, int64_t result_k) const {
        if (qi >= nq)
            return 0.0f;
        int64_t check_k = std::min(result_k, k);
        int matches = 0;
        for (int64_t i = 0; i < check_k; ++i) {
            if (result[i] == -1)
                continue;
            for (int64_t j = 0; j < check_k; ++j) {
                if (result[i] == ids[qi][j]) {
                    ++matches;
                    break;
                }
            }
        }
        return static_cast<float>(matches) / check_k;
    }
};

// ============================================================================
// Latency percentile helper
// ============================================================================
struct LatencyStats {
    double mean_ms;
    double p50_ms;
    double p95_ms;
    double p99_ms;
    double max_ms;

    static LatencyStats
    compute(std::vector<double>& latencies_ms) {
        std::sort(latencies_ms.begin(), latencies_ms.end());
        size_t n = latencies_ms.size();
        double sum = 0;
        for (double v : latencies_ms) sum += v;
        return {
            .mean_ms = sum / n,
            .p50_ms = latencies_ms[n / 2],
            .p95_ms = latencies_ms[static_cast<size_t>(n * 0.95)],
            .p99_ms = latencies_ms[static_cast<size_t>(n * 0.99)],
            .max_ms = latencies_ms[n - 1],
        };
    }
};

// ============================================================================
// Bitset generation for filtered benchmarks
// ============================================================================

// Generate a bitset where each bit is set with probability filter_rate.
// In knowhere, a set bit means the doc is FILTERED OUT (excluded from results).
std::vector<uint8_t>
generate_random_bitset(int64_t n_docs, float filter_rate, uint64_t seed) {
    size_t n_bytes = (n_docs + 7) / 8;
    std::vector<uint8_t> bitset(n_bytes, 0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int64_t n_set = 0;
    for (int64_t i = 0; i < n_docs; ++i) {
        if (dist(rng) < filter_rate) {
            bitset[i / 8] |= (1u << (i % 8));
            ++n_set;
        }
    }
    printf("  Generated random bitset: %ld/%ld docs filtered (%.1f%%)\n", n_set, n_docs, 100.0 * n_set / n_docs);
    return bitset;
}

// Generate a bitset that masks the docs with highest nnz (most non-zero dimensions).
// These dense docs appear in many posting lists and are likely to score well across
// diverse queries, making this an adversarial filter for pruning-based indexes.
std::vector<uint8_t>
generate_global_topk_bitset(const knowhere::sparse::SparseRow<float>* base_rows, int64_t n_docs, float filter_rate) {
    std::vector<int32_t> doc_nnz(n_docs, 0);
    for (int64_t i = 0; i < n_docs; ++i) {
        doc_nnz[i] = static_cast<int32_t>(base_rows[i].size());
    }
    // Sort indices by nnz descending — mask the densest docs first
    std::vector<int64_t> order(n_docs);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return doc_nnz[a] > doc_nnz[b]; });

    int64_t n_mask = static_cast<int64_t>(filter_rate * n_docs);
    size_t n_bytes = (n_docs + 7) / 8;
    std::vector<uint8_t> bitset(n_bytes, 0);
    for (int64_t i = 0; i < n_mask; ++i) {
        int64_t doc = order[i];
        bitset[doc / 8] |= (1u << (doc % 8));
    }
    printf("  Generated global-dense-nnz bitset: %ld/%ld docs filtered (%.1f%%)\n", n_mask, n_docs,
           100.0 * n_mask / n_docs);
    return bitset;
}

// Compute filtered ground truth using safe (exact) search.
GroundTruth
compute_filtered_gt(const knowhere::Index<knowhere::IndexNode>& brute_force_index,
                    const knowhere::sparse::SparseRow<float>* query_rows, int64_t nq, int64_t n_cols, int64_t topk,
                    const knowhere::Json& search_conf, const knowhere::BitsetView& bitset) {
    GroundTruth gt;
    gt.nq = nq;
    gt.k = topk;
    gt.ids.resize(nq);

    auto query_ds = knowhere::GenDataSet(1, n_cols, nullptr);
    query_ds->SetIsSparse(true);

    for (int64_t q = 0; q < nq; ++q) {
        gt.ids[q].resize(topk, -1);
        query_ds->SetTensor(&query_rows[q]);
        auto result = brute_force_index.Search(query_ds, search_conf, bitset);
        if (result.has_value()) {
            auto ids = result.value()->GetIds();
            for (int64_t i = 0; i < topk; ++i) {
                gt.ids[q][i] = static_cast<int32_t>(ids[i]);
            }
        }
    }
    return gt;
}

// ============================================================================
// Benchmark result with coverage metrics
// ============================================================================
struct BenchResult {
    std::vector<double> latencies_ms;
    std::vector<int64_t> result_ids;
    double total_ms;
    // Recall: averaged over ALL nq queries (failed queries contribute 0)
    float avg_recall;
    // Coverage: how many queries returned at least 1 result
    int64_t n_queries;
    int64_t n_failed;                     // queries where first result is -1
    float avg_filled;                     // average number of non-(-1) results per query out of topk
    std::vector<int64_t> failed_indices;  // query indices that returned zero results
};

BenchResult
run_search(const knowhere::Index<knowhere::IndexNode>& index, const knowhere::sparse::SparseRow<float>* query_rows,
           int64_t nq, int64_t n_cols, int64_t topk, const knowhere::Json& search_conf, const GroundTruth& gt,
           const knowhere::BitsetView& bitset = knowhere::BitsetView()) {
    BenchResult res;
    res.latencies_ms.resize(nq);
    res.result_ids.resize(nq * topk, -1);
    res.n_queries = nq;

    auto query_ds = knowhere::GenDataSet(1, n_cols, nullptr);
    query_ds->SetIsSparse(true);

    auto t_total_start = std::chrono::high_resolution_clock::now();
    for (int64_t q = 0; q < nq; ++q) {
        query_ds->SetTensor(&query_rows[q]);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = index.Search(query_ds, search_conf, bitset);
        auto t1 = std::chrono::high_resolution_clock::now();
        res.latencies_ms[q] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (result.has_value()) {
            memcpy(&res.result_ids[q * topk], result.value()->GetIds(), topk * sizeof(int64_t));
        }
    }
    auto t_total_end = std::chrono::high_resolution_clock::now();
    res.total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();

    // Compute recall over ALL queries (failed queries get recall=0).
    // Also compute coverage metrics.
    int64_t eval_nq = std::min(nq, gt.nq);
    float recall_sum = 0;
    int64_t n_failed = 0;
    int64_t total_filled = 0;

    for (int64_t q = 0; q < nq; ++q) {
        // Count filled (non -1) slots
        int64_t filled = 0;
        for (int64_t i = 0; i < topk; ++i) {
            if (res.result_ids[q * topk + i] != -1)
                ++filled;
        }
        total_filled += filled;

        if (filled == 0) {
            ++n_failed;
            res.failed_indices.push_back(q);
        }

        // Recall: every query in [0, eval_nq) contributes, even if failed (=0 recall)
        if (q < eval_nq) {
            recall_sum += gt.recall(&res.result_ids[q * topk], q, topk);
        }
    }

    res.avg_recall = (eval_nq > 0) ? recall_sum / eval_nq : 0.0f;
    res.n_failed = n_failed;
    res.avg_filled = static_cast<float>(total_filled) / nq;
    return res;
}

// ============================================================================
// Main
// ============================================================================
void
print_usage(const char* prog) {
    printf(
        "Usage: %s --data-dir <path> --metric <IP|BM25> --gt <path> [options]\n"
        "\n"
        "Required:\n"
        "  --data-dir <path>    Directory containing base and query CSR files\n"
        "  --metric <IP|BM25>   Metric type\n"
        "  --gt <path>          Ground truth file (binary: int32 nq, int32 k, nq*k int32 IDs)\n"
        "\n"
        "Options:\n"
        "  --topk <k>           Top-k results (default: 10)\n"
        "  --nq <n>             Number of queries, 0=all (default: 0)\n"
        "  --warmup <n>         Warmup runs before timed run (default: 1)\n"
        "  --base <file>        Base vectors file (default: base_small.csr)\n"
        "  --query <file>       Query vectors file (default: queries.dev.csr)\n"
        "  --bm25-k1 <f>        BM25 k1 (default: 1.2)\n"
        "  --bm25-b <f>         BM25 b (default: 0.75)\n"
        "  --avgdl <f>          Override avgdl (default: computed from base data)\n"
        "  --default-only       Only run default DSP config (no sweep)\n"
        "\n"
        "Filter options:\n"
        "  --filter-rate <f>    Corpus-level filter rate 0.0-1.0 (default: 0.0 = no filter)\n"
        "  --filter-mode <m>    Filter mode (default: random):\n"
        "                         random          uniform random docs\n"
        "                         global-topk     docs with highest nnz (most dimensions)\n"
        "  --filter-seed <n>    Random seed for filter generation (default: 42)\n"
        "\n",
        prog);
}

int
main(int argc, char** argv) {
    std::string data_dir;
    std::string metric;
    std::string gt_path;
    std::string base_file = "base_small.csr";
    std::string query_file = "queries.dev.csr";
    int64_t topk = 10;
    int64_t nq = 0;
    int warmup = 1;
    float bm25_k1 = 1.2f;
    float bm25_b = 0.75f;
    float avgdl_override = -1.0f;
    bool default_only = false;
    float filter_rate = 0.0f;
    std::string filter_mode = "random";
    uint64_t filter_seed = 42;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
            data_dir = argv[++i];
        else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc)
            metric = argv[++i];
        else if (strcmp(argv[i], "--gt") == 0 && i + 1 < argc)
            gt_path = argv[++i];
        else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc)
            topk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc)
            nq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--base") == 0 && i + 1 < argc)
            base_file = argv[++i];
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc)
            query_file = argv[++i];
        else if (strcmp(argv[i], "--bm25-k1") == 0 && i + 1 < argc)
            bm25_k1 = atof(argv[++i]);
        else if (strcmp(argv[i], "--bm25-b") == 0 && i + 1 < argc)
            bm25_b = atof(argv[++i]);
        else if (strcmp(argv[i], "--avgdl") == 0 && i + 1 < argc)
            avgdl_override = atof(argv[++i]);
        else if (strcmp(argv[i], "--default-only") == 0)
            default_only = true;
        else if (strcmp(argv[i], "--filter-rate") == 0 && i + 1 < argc)
            filter_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--filter-mode") == 0 && i + 1 < argc)
            filter_mode = argv[++i];
        else if (strcmp(argv[i], "--filter-seed") == 0 && i + 1 < argc)
            filter_seed = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_dir.empty() || metric.empty() || gt_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    bool is_bm25 = (metric == "BM25" || metric == "bm25");

    printf("==========================================================\n");
    printf("  Sparse DSP Benchmark\n");
    printf("==========================================================\n\n");

    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    // ---- Load data ----
    printf("[Loading Data]\n");
    CSRDataset base, queries;
    if (!base.load(data_dir + "/" + base_file))
        return 1;
    if (!queries.load(data_dir + "/" + query_file))
        return 1;

    if (nq == 0 || nq > queries.n_rows)
        nq = queries.n_rows;

    auto base_rows = base.to_sparse_rows();
    auto query_rows = queries.to_sparse_rows();

    double avgdl = 0.0;
    if (is_bm25) {
        if (avgdl_override > 0) {
            avgdl = avgdl_override;
            printf("  avgdl: %.2f (user-provided override)\n", avgdl);
        } else {
            avgdl = base.compute_avgdl();
            printf("  avgdl: %.2f (computed from base data)\n", avgdl);
        }
    }
    base.free_raw();

    // ---- Load ground truth ----
    printf("[Loading Ground Truth]\n");
    printf("  GT path: %s\n", gt_path.c_str());
    GroundTruth gt;
    if (!gt.load(gt_path, queries.n_rows)) {
        fprintf(stderr, "Error: failed to load ground truth from %s\n", gt_path.c_str());
        return 1;
    }

    printf("\n[Config]\n");
    printf("  base=%ld  nq=%ld  k=%ld  metric=%s  warmup=%d\n", base.n_rows, nq, topk, metric.c_str(), warmup);
    printf("  base_file=%s  query_file=%s\n", base_file.c_str(), query_file.c_str());
    printf("  gt=%s (nq=%ld, k=%ld)\n", gt_path.c_str(), gt.nq, gt.k);
    if (is_bm25)
        printf("  bm25: k1=%.2f b=%.2f avgdl=%.2f\n", bm25_k1, bm25_b, avgdl);
    if (filter_rate > 0)
        printf("  filter: rate=%.2f mode=%s seed=%lu\n", filter_rate, filter_mode.c_str(), filter_seed);
    printf("\n");

    // ---- Helper: populate BM25 params into JSON ----
    auto make_bm25_json = [&](knowhere::Json& json) {
        if (is_bm25) {
            json["bm25_k1"] = bm25_k1;
            json["bm25_b"] = bm25_b;
            json["bm25_avgdl"] = static_cast<float>(avgdl);
        }
    };

    // ---- Helper: print result row ----
    // Columns: config, params, recall, QPS, failed, avg_filled/k, latency percentiles
    auto print_header = [&]() {
        printf("  %-22s  %-26s  %-7s %-8s %-6s %-8s  %-50s\n", "Config", "Params", "Recall", "QPS", "Fail", "Fill/k",
               "Latency(ms): mean / p50 / p95 / p99 / max");
        printf("  %s\n", std::string(170, '-').c_str());
    };

    auto print_row = [&](const char* label, float mu, float eta, int gamma, const BenchResult& res,
                         const LatencyStats& lat) {
        char params_buf[64];
        snprintf(params_buf, sizeof(params_buf), "mu=%.2f eta=%.2f g=%-5d", mu, eta, gamma);
        printf(
            "  %-22s  %-26s  %.4f  %-8.1f %-6ld %.1f/%-3ld  "
            "%.2f / %.2f / %.2f / %.2f / %.2f\n",
            label, params_buf, res.avg_recall, res.n_queries * 1000.0 / res.total_ms, res.n_failed, res.avg_filled,
            topk, lat.mean_ms, lat.p50_ms, lat.p95_ms, lat.p99_ms, lat.max_ms);
    };

    // ---- Helper: diagnose failed queries ----
    auto print_failed_diag = [&](const char* label, const BenchResult& res) {
        if (res.failed_indices.empty())
            return;
        printf("\n  [%s] %ld failed queries (zero results):\n", label, res.n_failed);
        size_t show = std::min(res.failed_indices.size(), static_cast<size_t>(20));
        for (size_t i = 0; i < show; ++i) {
            int64_t qi = res.failed_indices[i];
            int64_t nnz = query_rows[qi].size();
            printf("    q[%ld]: nnz=%ld", qi, nnz);
            // Show first few dims for context
            if (nnz > 0) {
                printf("  dims=[");
                for (int64_t j = 0; j < std::min(nnz, static_cast<int64_t>(5)); ++j) {
                    if (j > 0)
                        printf(",");
                    printf("%d:%.2f", query_rows[qi][j].id, query_rows[qi][j].val);
                }
                if (nnz > 5)
                    printf(",...");
                printf("]");
            }
            printf("\n");
        }
        if (res.failed_indices.size() > show) {
            printf("    ... and %ld more\n", res.failed_indices.size() - show);
        }
        printf("\n");
    };

    // ---- Build base dataset ----
    auto ds = knowhere::GenDataSet(base.n_rows, base.n_cols, nullptr);
    ds->SetIsSparse(true);
    ds->SetTensor(base_rows.get());

    // DSP safe-search config for filtered ground truth
    // (mu=1, eta=1, mode=0, gamma=0, no kth-init gives exact results)
    knowhere::Json safe_search_json;
    if (filter_rate > 0) {
        safe_search_json["metric_type"] = metric;
        safe_search_json["topk"] = topk;
        safe_search_json["drop_ratio_search"] = 0.0f;
        safe_search_json["dsp_mode"] = 0;
        safe_search_json["dsp_mu"] = 1.0f;
        safe_search_json["dsp_eta"] = 1.0f;
        safe_search_json["dsp_gamma"] = 0;
        safe_search_json["dsp_kth_init"] = false;
        make_bm25_json(safe_search_json);
    }

    // ============================================================
    // Filter setup: generate bitset (filtered GT computed after DSP build)
    // ============================================================
    std::vector<uint8_t> filter_bitset_data;
    knowhere::BitsetView filter_bitset;

    if (filter_rate > 0) {
        printf("\n[Generating Filter]\n");
        if (filter_mode == "random") {
            filter_bitset_data = generate_random_bitset(base.n_rows, filter_rate, filter_seed);
        } else if (filter_mode == "global-topk") {
            filter_bitset_data = generate_global_topk_bitset(base_rows.get(), base.n_rows, filter_rate);
        } else {
            fprintf(stderr, "Error: unknown filter mode '%s' (supported: random, global-topk)\n", filter_mode.c_str());
            return 1;
        }
        filter_bitset = knowhere::BitsetView(filter_bitset_data.data(), base.n_rows);
    }

    // ============================================================
    // DSP: Build once, sweep params
    // ============================================================
    printf("[DSP Index]\n");
    auto dsp_or = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
        knowhere::IndexEnum::INDEX_SPARSE_DSP_CC, knowhere::Version::GetCurrentVersion().VersionNumber());
    if (!dsp_or.has_value()) {
        fprintf(stderr, "Error: failed to create DSP index\n");
        return 1;
    }
    auto dsp = dsp_or.value();

    {
        knowhere::Json build_json;
        build_json["metric_type"] = metric;
        make_bm25_json(build_json);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto st = dsp.Build(ds, build_json);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (st != knowhere::Status::success) {
            fprintf(stderr, "Error: DSP build failed\n");
            return 1;
        }
        printf("  Build: %.1f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    // Compute filtered GT using DSP safe search (exact results with mu=1, eta=1)
    if (filter_rate > 0) {
        printf("[Computing Filtered Ground Truth via DSP safe search]\n");
        gt = compute_filtered_gt(dsp, query_rows.get(), nq, queries.n_cols, topk, safe_search_json, filter_bitset);
        printf("  Filtered GT: %ld queries, k=%ld\n", gt.nq, gt.k);
    }

    // ---- Parameter sweep ----
    struct ParamSet {
        const char* label;
        int mode;  // 0=dsp, 1=lsp0, 2=lsp1, 3=lsp2
        float mu;
        float eta;
        int gamma;
        bool kth_init = true;
        float kth_alpha = 1.0f;
    };

    // clang-format off
    std::vector<ParamSet> params;
    if (default_only) {
        params = {
            {"dsp default",         0, 1.0f, 1.0f,    0},
            {"dsp a=0.50",          0, 1.0f, 1.0f,    0, true, 0.50f},
            {"dsp a=0.25",          0, 1.0f, 1.0f,    0, true, 0.25f},
            {"dsp no-kth",          0, 1.0f, 1.0f,    0, false},
        };
    } else {
        params = {
            // DSP mode (mode=0): dual-threshold (mu, eta) + optional top-gamma backstop
            {"dsp default",         0, 1.0f, 1.0f,    0},
            {"dsp mu=0.3",          0, 0.3f, 1.0f,    0},
            {"dsp mu=0.5 eta=1.0",  0, 0.5f, 1.0f,    0},
            {"dsp mu=0.5 eta=0.8",  0, 0.5f, 0.8f,    0},
            {"dsp mu=0.5 eta=0.5",  0, 0.5f, 0.5f,    0},
            {"dsp mu=0.3 g=100",    0, 0.3f, 1.0f,  100},

            // LSP/0 (mode=1): top-gamma from ub>=theta, no mu/asc gate
            {"lsp0 g=50",           1, 1.0f, 1.0f,   50},
            {"lsp0 g=100",          1, 1.0f, 1.0f,  100},
            {"lsp0 g=500",          1, 1.0f, 1.0f,  500},
            {"lsp0 g=1000",         1, 1.0f, 1.0f, 1000},

            // LSP/1 (mode=2): lsp0 safe set + mu gate
            {"lsp1 g=100",          2, 1.0f, 1.0f,  100},
            {"lsp1 g=100 mu=0.3",   2, 0.3f, 1.0f,  100},
            {"lsp1 g=100 mu=0.5",   2, 0.5f, 1.0f,  100},

            // LSP/2 (mode=3): lsp1 + asc gate
            {"lsp2 g=100",          3, 1.0f, 1.0f,  100},
            {"lsp2 g=100 mu=0.3",   3, 0.3f, 1.0f,  100},
            {"lsp2 g=100 mu=0.5",   3, 0.5f, 1.0f,  100},
            {"lsp2 g=100 mu=0.5 eta=0.8", 3, 0.5f, 0.8f, 100},

            // kth-init OFF: isolate hierarchy-only contribution
            {"dsp no-kth",          0, 1.0f, 1.0f,    0, false},
            {"lsp0 g=100 no-kth",   1, 1.0f, 1.0f,  100, false},
            {"lsp1 g=100 no-kth",   2, 1.0f, 1.0f,  100, false},

            // Alpha-clamped kth threshold (DSP mode)
            {"dsp a=0.25",          0, 1.0f, 1.0f,    0, true, 0.25f},
            {"dsp a=0.50",          0, 1.0f, 1.0f,    0, true, 0.50f},
            {"dsp a=0.75",          0, 1.0f, 1.0f,    0, true, 0.75f},
        };
    }
    // clang-format on

    // ============================================================
    // Parameter sweep
    // ============================================================
    {
        printf("\n");
        print_header();

        for (const auto& p : params) {
            knowhere::Json search_json;
            search_json["metric_type"] = metric;
            search_json["topk"] = topk;
            search_json["drop_ratio_search"] = 0.0f;
            search_json["dsp_mode"] = p.mode;
            search_json["dsp_mu"] = p.mu;
            search_json["dsp_eta"] = p.eta;
            search_json["dsp_gamma"] = p.gamma;
            search_json["dsp_kth_init"] = p.kth_init;
            search_json["dsp_kth_alpha"] = p.kth_alpha;
            make_bm25_json(search_json);

            for (int w = 0; w < warmup; ++w) {
                run_search(dsp, query_rows.get(), nq, queries.n_cols, topk, search_json, gt, filter_bitset);
            }

            auto res = run_search(dsp, query_rows.get(), nq, queries.n_cols, topk, search_json, gt, filter_bitset);
            auto lat = LatencyStats::compute(res.latencies_ms);
            print_row(p.label, p.mu, p.eta, p.gamma, res, lat);
            if (strstr(p.label, "default") != nullptr || strstr(p.label, "no-kth") != nullptr || res.n_failed > 0) {
                print_failed_diag(p.label, res);
            }
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}
