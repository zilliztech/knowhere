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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "knowhere/bitsetview.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/operands.h"
#include "knowhere/sparse_utils.h"

// CSR format loader for MSMARCO/SPLADE data from big-ann-benchmarks
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
            printf("Error: Cannot open file %s\n", path.c_str());
            return false;
        }

        // Read header: n_rows, n_cols, nnz (all int64_t)
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(int64_t));

        printf("  Loading CSR: %ld rows, %ld cols, %ld nnz\n", n_rows, n_cols, nnz);

        // Read indptr (n_rows + 1 int64_t values)
        indptr.resize(n_rows + 1);
        file.read(reinterpret_cast<char*>(indptr.data()), (n_rows + 1) * sizeof(int64_t));

        // Read indices (nnz int32_t values)
        indices.resize(nnz);
        file.read(reinterpret_cast<char*>(indices.data()), nnz * sizeof(int32_t));

        // Read data (nnz float values)
        data.resize(nnz);
        file.read(reinterpret_cast<char*>(data.data()), nnz * sizeof(float));

        if (!file) {
            printf("Error: Failed to read file %s\n", path.c_str());
            return false;
        }

        return true;
    }

    // Convert to knowhere SparseRow format
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
};

// Ground truth loader (binary format: nq, k header then nq x k int32_t)
struct GroundTruth {
    std::vector<std::vector<int32_t>> gt;
    int64_t nq = 0;
    int64_t k = 0;

    bool
    load(const std::string& path, int64_t expected_nq) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            printf("Error: Cannot open ground truth file %s\n", path.c_str());
            return false;
        }

        // Read header: nq and k (both int32_t)
        int32_t nq32, k32;
        file.read(reinterpret_cast<char*>(&nq32), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&k32), sizeof(int32_t));
        nq = nq32;
        k = k32;
        printf("  Loading GT: %ld queries, k=%ld\n", nq, k);

        if (nq != expected_nq) {
            printf("  Warning: GT has %ld queries but expected %ld, using min(%ld, %ld)\n", nq, expected_nq, nq,
                   expected_nq);
            nq = std::min(nq, expected_nq);
        }

        gt.resize(nq);
        for (int64_t i = 0; i < nq; ++i) {
            gt[i].resize(k);
            file.read(reinterpret_cast<char*>(gt[i].data()), k * sizeof(int32_t));
        }

        return file.good();
    }

    float
    compute_recall(const int64_t* results, int64_t query_idx, int64_t result_k) const {
        if (query_idx >= nq)
            return 0.0f;
        int64_t check_k = std::min(result_k, k);
        int matches = 0;
        for (int64_t i = 0; i < check_k; ++i) {
            for (int64_t j = 0; j < check_k; ++j) {
                if (results[i] == gt[query_idx][j]) {
                    matches++;
                    break;
                }
            }
        }
        return static_cast<float>(matches) / check_k;
    }
};

class Timer {
    std::chrono::high_resolution_clock::time_point start_;

 public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {
    }

    double
    elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    void
    reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

void
print_usage(const char* prog) {
    printf("Usage: %s --data-dir <path> --data-type <splade|bm25> [--topk <k>] [--nq <num_queries>]\n", prog);
    printf("\nRequired arguments:\n");
    printf("  --data-dir <path>    - Directory containing CSR data files\n");
    printf("  --data-type <type>   - Type of data: 'splade' (IP metric) or 'bm25' (BM25 metric)\n");
    printf("\nExpected files in data-dir:\n");
    printf("  base_small.csr       - Base vectors in CSR format\n");
    printf("  queries.dev.csr      - Query vectors in CSR format\n");
    printf("  base_small.dev.gt    - Ground truth (binary: nq, k header + nq*k int32 IDs)\n");
    printf("\nDownload from:\n");
    printf("  wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.csr.gz\n");
    printf("  wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/queries.dev.csr.gz\n");
    printf("  wget https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/base_small.dev.gt\n");
}

int
main(int argc, char** argv) {
    std::string data_dir;
    std::string data_type;
    int64_t topk = 10;
    int64_t nq = 0;  // 0 = use all queries

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--data-type") == 0 && i + 1 < argc) {
            data_type = argv[++i];
        } else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            topk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc) {
            nq = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_dir.empty() || data_type.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    if (data_type != "splade" && data_type != "bm25") {
        printf("Error: --data-type must be 'splade' or 'bm25'\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("==========================================================\n");
    printf("  Sparse Search Algorithm Benchmark (MaxScore vs MaxScore v2)\n");
    printf("==========================================================\n\n");

    // Initialize knowhere
    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    // Load datasets
    printf("[Loading Data]\n");
    CSRDataset base, queries;

    if (!base.load(data_dir + "/base_small.csr")) {
        return 1;
    }
    if (!queries.load(data_dir + "/queries.dev.csr")) {
        return 1;
    }

    if (nq == 0 || nq > queries.n_rows) {
        nq = queries.n_rows;
    }
    printf("  Using %ld queries\n", nq);

    // Load ground truth
    GroundTruth gt;
    if (!gt.load(data_dir + "/base_small.dev.gt", queries.n_rows)) {
        printf("Warning: Ground truth not loaded, recall will not be computed\n");
    }

    // Convert to sparse rows
    printf("\n[Converting to SparseRow format]\n");
    auto base_rows = base.to_sparse_rows();
    auto query_rows = queries.to_sparse_rows();
    printf("  Done\n");

    // Algorithms to benchmark
    std::vector<std::string> algos = {"DAAT_MAXSCORE", "DAAT_MAXSCORE_V2"};

    // Metrics to benchmark based on data type
    std::vector<std::string> metrics;
    if (data_type == "splade") {
        metrics = {"IP"};
    } else {
        metrics = {"BM25"};
    }

    // Benchmark parameters following DSP paper methodology:
    // 5 runs, drop first 2 (warmup), average last 3
    const int total_runs = 5;
    const int warmup_runs = 2;

    printf("\n[Benchmark Configuration]\n");
    printf("  Base vectors: %ld\n", base.n_rows);
    printf("  Queries: %ld\n", nq);
    printf("  Top-k: %ld\n", topk);
    printf("  Runs: %d (warmup: %d)\n", total_runs, warmup_runs);

    for (const auto& metric : metrics) {
        printf("\n==========================================================\n");
        printf("  Metric: %s\n", metric.c_str());
        printf("==========================================================\n");

        for (const auto& algo : algos) {
            printf("\n----------------------------------------------------------\n");
            printf("  Algorithm: %s (%s)\n", algo.c_str(), metric.c_str());
            printf("----------------------------------------------------------\n");

            // Build index
            printf("\n[Building Index]\n");
            Timer build_timer;

            auto index_result = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
                "SPARSE_INVERTED_INDEX", knowhere::Version::GetCurrentVersion().VersionNumber());
            if (!index_result.has_value()) {
                printf("Error: Failed to create index\n");
                continue;
            }
            auto index = index_result.value();

            knowhere::Json build_conf;
            build_conf["metric_type"] = metric;
            build_conf["inverted_index_algo"] = algo;
            if (metric == "BM25") {
                build_conf["bm25_k1"] = 1.2f;
                build_conf["bm25_b"] = 0.75f;
                build_conf["bm25_avgdl"] = 100.0f;
            }

            // Create dataset
            auto ds = knowhere::GenDataSet(base.n_rows, base.n_cols, nullptr);
            ds->SetIsSparse(true);
            ds->SetTensor(base_rows.get());

            auto status = index.Build(ds, build_conf);
            if (status != knowhere::Status::success) {
                printf("Error: Failed to build index: %s\n", knowhere::Status2String(status).c_str());
                continue;
            }

            double build_time = build_timer.elapsed_ms();
            printf("  Build time: %.2f ms\n", build_time);

            // Search configuration
            knowhere::Json search_conf;
            search_conf["metric_type"] = metric;
            search_conf["drop_ratio_search"] = 0.0f;
            search_conf["topk"] = topk;
            if (metric == "BM25") {
                search_conf["bm25_k1"] = 1.2f;
                search_conf["bm25_b"] = 0.75f;
                search_conf["bm25_avgdl"] = 100.0f;
            }

            // Pre-allocate reusable query dataset to avoid per-query allocation overhead
            auto query_ds = knowhere::GenDataSet(1, queries.n_cols, nullptr);
            query_ds->SetIsSparse(true);

            // Run benchmark
            printf("\n[Running Search Benchmark]\n");
            std::vector<double> run_times;
            std::vector<int64_t> all_results(nq * topk, -1);  // Initialize to -1 for invalid detection
            std::vector<bool> query_success(nq, false);
            int64_t failed_queries = 0;

            for (int run = 0; run < total_runs; ++run) {
                Timer search_timer;
                failed_queries = 0;

                for (int64_t q = 0; q < nq; ++q) {
                    query_ds->SetTensor(&query_rows[q]);

                    auto result = index.Search(query_ds, search_conf, knowhere::BitsetView());
                    if (!result.has_value()) {
                        failed_queries++;
                        query_success[q] = false;
                        // Clear stale results from previous runs
                        std::fill(&all_results[q * topk], &all_results[(q + 1) * topk], -1);
                        continue;
                    }

                    query_success[q] = true;
                    auto ids = result.value()->GetIds();
                    memcpy(&all_results[q * topk], ids, topk * sizeof(int64_t));
                }

                double elapsed = search_timer.elapsed_ms();
                run_times.push_back(elapsed);
                printf("  Run %d: %.2f ms (%.1f batch QPS)\n", run + 1, elapsed, nq * 1000.0 / elapsed);
            }

            if (failed_queries > 0) {
                printf("  Warning: %ld queries failed in last run\n", failed_queries);
            }

            // Calculate average of last (total_runs - warmup_runs) runs
            double avg_time = 0;
            for (int i = warmup_runs; i < total_runs; ++i) {
                avg_time += run_times[i];
            }
            avg_time /= (total_runs - warmup_runs);

            // Compute recall on last run's results (only for successful queries)
            float avg_recall = 0;
            int64_t valid_queries = 0;
            if (gt.nq > 0) {
                for (int64_t q = 0; q < nq; ++q) {
                    if (query_success[q]) {
                        avg_recall += gt.compute_recall(&all_results[q * topk], q, topk);
                        valid_queries++;
                    }
                }
                if (valid_queries > 0) {
                    avg_recall /= valid_queries;
                }
            }

            printf("\n[Results for %s (%s)]\n", algo.c_str(), metric.c_str());
            printf("  Avg search time: %.2f ms (over %d timed runs)\n", avg_time, total_runs - warmup_runs);
            printf("  Batch QPS: %.1f (nq=%ld)\n", nq * 1000.0 / avg_time, nq);
            printf("  Recall@%ld: %.4f (%.2f%%) [from last run, %ld/%ld queries]\n", topk, avg_recall, avg_recall * 100,
                   valid_queries, nq);
            // Debug: print first 3 successful queries' top-3 results to verify correctness
            printf("  Debug - First 3 successful queries top-3 IDs: ");
            int printed = 0;
            for (int64_t q = 0; q < nq && printed < 3; ++q) {
                if (query_success[q]) {
                    printf("[q%ld: %ld,%ld,%ld] ", q, all_results[q * topk], all_results[q * topk + 1],
                           all_results[q * topk + 2]);
                    printed++;
                }
            }
            printf("\n");
        }
    }

    printf("\n==========================================================\n");
    printf("  Benchmark Complete\n");
    printf("==========================================================\n");

    return 0;
}
