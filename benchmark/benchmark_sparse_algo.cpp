// Sparse Search Algorithm Benchmark using actual Knowhere library
// Compares: TAAT_NAIVE, DAAT_WAND, DAAT_WAND+BMW, DAAT_MAXSCORE
// Supports MSMARCO/SPLADE dataset from big-ann-benchmarks

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <vector>

#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/version.h"

// ============================================================================
// CSR Matrix I/O (big-ann-benchmarks format)
// Format: nrow(int64) ncol(int64) nnz(int64) indptr(int64[nrow+1]) indices(int32[nnz]) data(float32[nnz])
// ============================================================================

struct CSRMatrix {
    int64_t nrow;
    int64_t ncol;
    int64_t nnz;
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    std::vector<float> data;
};

bool
ReadCSRMatrix(const std::string& filename, CSRMatrix& mat) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        printf("Failed to open file: %s\n", filename.c_str());
        return false;
    }

    // Read header
    int64_t sizes[3];
    f.read(reinterpret_cast<char*>(sizes), sizeof(sizes));
    mat.nrow = sizes[0];
    mat.ncol = sizes[1];
    mat.nnz = sizes[2];

    printf("  Reading CSR matrix: %ld rows, %ld cols, %ld nnz\n", mat.nrow, mat.ncol, mat.nnz);

    // Read indptr
    mat.indptr.resize(mat.nrow + 1);
    f.read(reinterpret_cast<char*>(mat.indptr.data()), (mat.nrow + 1) * sizeof(int64_t));

    // Read indices
    mat.indices.resize(mat.nnz);
    f.read(reinterpret_cast<char*>(mat.indices.data()), mat.nnz * sizeof(int32_t));

    // Read data
    mat.data.resize(mat.nnz);
    f.read(reinterpret_cast<char*>(mat.data.data()), mat.nnz * sizeof(float));

    if (!f) {
        printf("Error reading file\n");
        return false;
    }

    // Verify
    if (mat.indptr[mat.nrow] != mat.nnz) {
        printf("Warning: indptr[nrow]=%ld != nnz=%ld\n", mat.indptr[mat.nrow], mat.nnz);
    }

    return true;
}

// Check if CSR matrix rows are already L2 normalized
bool
IsL2Normalized(const CSRMatrix& mat, int sample_rows = 100) {
    int check_count = std::min(static_cast<int64_t>(sample_rows), mat.nrow);
    int normalized_count = 0;

    for (int64_t i = 0; i < check_count; ++i) {
        int64_t start = mat.indptr[i];
        int64_t end = mat.indptr[i + 1];
        if (start == end)
            continue;

        float norm_sq = 0.0f;
        for (int64_t j = start; j < end; ++j) {
            norm_sq += mat.data[j] * mat.data[j];
        }
        float norm = std::sqrt(norm_sq);
        if (std::abs(norm - 1.0f) < 0.01f) {
            normalized_count++;
        }
    }

    return normalized_count > check_count * 0.9;  // 90% of sampled rows are normalized
}

knowhere::DataSetPtr
CSRToKnowhereDataSet(const CSRMatrix& mat, bool normalize_l2 = false) {
    auto tensor = std::make_unique<knowhere::sparse::SparseRow<float>[]>(mat.nrow);

    for (int64_t i = 0; i < mat.nrow; ++i) {
        int64_t start = mat.indptr[i];
        int64_t end = mat.indptr[i + 1];
        int64_t row_nnz = end - start;

        if (row_nnz > 0) {
            // Compute L2 norm if normalizing
            float norm = 1.0f;
            if (normalize_l2) {
                float norm_sq = 0.0f;
                for (int64_t j = start; j < end; ++j) {
                    norm_sq += mat.data[j] * mat.data[j];
                }
                norm = std::sqrt(norm_sq);
                if (norm < 1e-9f)
                    norm = 1.0f;  // Avoid division by zero
            }

            knowhere::sparse::SparseRow<float> row(row_nnz);
            for (int64_t j = 0; j < row_nnz; ++j) {
                float val = mat.data[start + j] / norm;
                row.set_at(j, mat.indices[start + j], val);
            }
            tensor[i] = std::move(row);
        } else {
            // Explicitly initialize empty rows to avoid UB
            tensor[i] = knowhere::sparse::SparseRow<float>();
        }
    }

    auto ds = knowhere::GenDataSet(mat.nrow, mat.ncol, tensor.release());
    ds->SetIsOwner(true);
    ds->SetIsSparse(true);
    return ds;
}

// Compute average document length from CSR matrix
float
ComputeAvgDocLen(const CSRMatrix& mat) {
    return static_cast<float>(mat.nnz) / static_cast<float>(mat.nrow);
}

// Read ground truth file (big-ann-benchmarks format)
// Format: nq(uint32) k(uint32) ids(int32[nq*k]) dists(float32[nq*k])
knowhere::DataSetPtr
ReadGroundTruth(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        printf("Failed to open ground truth file: %s\n", filename.c_str());
        return nullptr;
    }

    uint32_t nq, k;
    f.read(reinterpret_cast<char*>(&nq), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&k), sizeof(uint32_t));

    // Sanity check - if values are unreasonable, file is probably invalid
    if (nq > 10000000 || k > 10000 || nq == 0 || k == 0) {
        printf("  Ground truth file appears invalid (nq=%u, k=%u)\n", nq, k);
        return nullptr;
    }

    printf("  Ground truth: %u queries, k=%u\n", nq, k);

    auto ids = new int64_t[nq * k];
    auto dists = new float[nq * k];

    // Read int32 ids and convert to int64
    std::vector<int32_t> ids32(nq * k);
    f.read(reinterpret_cast<char*>(ids32.data()), nq * k * sizeof(int32_t));
    for (size_t i = 0; i < nq * k; ++i) {
        ids[i] = ids32[i];
    }

    f.read(reinterpret_cast<char*>(dists), nq * k * sizeof(float));

    auto ds = std::make_shared<knowhere::DataSet>();
    ds->SetRows(nq);
    ds->SetDim(k);
    ds->SetIds(ids);
    ds->SetDistance(dists);
    ds->SetIsOwner(true);
    return ds;
}

// ============================================================================
// Synthetic Data Generation (Zipfian distribution)
// ============================================================================

knowhere::DataSetPtr
GenZipfianDataSet(int32_t rows, int32_t cols, float avg_terms_per_doc, float zipf_exp, int seed = 42) {
    std::mt19937 rng(seed);
    auto uniform = std::uniform_real_distribution<float>(0, 1);

    std::vector<float> term_doc_freq(cols);
    float sum = 0;
    for (int32_t c = 0; c < cols; ++c) {
        term_doc_freq[c] = 1.0f / std::pow(c + 1, zipf_exp);
        sum += term_doc_freq[c];
    }
    float scale = avg_terms_per_doc * rows / (sum * rows);
    for (int32_t c = 0; c < cols; ++c) {
        term_doc_freq[c] = std::min(term_doc_freq[c] * scale, 1.0f);
    }

    std::vector<std::map<int32_t, float>> data(rows);

    for (int32_t col = 0; col < cols; ++col) {
        float prob = term_doc_freq[col];
        int32_t num_docs = static_cast<int32_t>(prob * rows);
        if (num_docs == 0 && uniform(rng) < prob * rows)
            num_docs = 1;

        std::vector<int32_t> doc_ids(rows);
        std::iota(doc_ids.begin(), doc_ids.end(), 0);
        std::shuffle(doc_ids.begin(), doc_ids.end(), rng);

        for (int32_t i = 0; i < num_docs && i < rows; ++i) {
            int32_t doc_id = doc_ids[i];
            float tf = -std::log(1.0f - uniform(rng) * 0.9999f);
            float max_tf = 10.0f / std::pow(col + 1, 0.3f);
            tf = std::min(tf, max_tf);
            data[doc_id][col] = tf;
        }
    }

    auto tensor = std::make_unique<knowhere::sparse::SparseRow<float>[]>(rows);
    for (int32_t i = 0; i < rows; ++i) {
        if (data[i].size() == 0) {
            continue;
        }
        knowhere::sparse::SparseRow<float> row(data[i].size());
        size_t j = 0;
        for (auto& [idx, val] : data[i]) {
            row.set_at(j++, idx, val);
        }
        tensor[i] = std::move(row);
    }

    auto ds = knowhere::GenDataSet(rows, cols, tensor.release());
    ds->SetIsOwner(true);
    ds->SetIsSparse(true);
    return ds;
}

knowhere::DataSetPtr
GenZipfianQuery(int32_t nq, int32_t cols, int32_t terms_per_query, float zipf_exp, int seed = 123) {
    std::mt19937 rng(seed);
    auto uniform = std::uniform_real_distribution<float>(0, 1);

    auto tensor = std::make_unique<knowhere::sparse::SparseRow<float>[]>(nq);

    for (int32_t q = 0; q < nq; ++q) {
        std::set<int32_t> selected_terms;
        while ((int32_t)selected_terms.size() < terms_per_query) {
            float u = uniform(rng);
            int32_t term = static_cast<int32_t>(cols * std::pow(u, zipf_exp));
            term = std::min(term, cols - 1);
            selected_terms.insert(term);
        }

        knowhere::sparse::SparseRow<float> row(selected_terms.size());
        size_t j = 0;
        for (int32_t term : selected_terms) {
            float weight = std::log(1.0f + cols / (term + 1.0f));
            row.set_at(j++, term, weight);
        }
        tensor[q] = std::move(row);
    }

    auto ds = knowhere::GenDataSet(nq, cols, tensor.release());
    ds->SetIsOwner(true);
    ds->SetIsSparse(true);
    return ds;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

struct BenchResult {
    double build_time_ms;
    double search_time_us;
    double qps;
    float recall;
};

// Run benchmark following DSP paper methodology:
// - 5 runs, drop first 2 warmup runs, report average of last 3
BenchResult
RunBenchmark(const std::string& algo, bool use_block_max, knowhere::DataSetPtr& train_ds,
             knowhere::DataSetPtr& query_ds, knowhere::DataSetPtr& gt, int topk, int runs,
             const std::string& metric = "IP", float avgdl = 0.0f) {
    knowhere::Json build_conf = {
        {knowhere::meta::DIM, train_ds->GetDim()},
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::indexparam::INVERTED_INDEX_ALGO, algo},
    };

    knowhere::Json search_conf = {
        {knowhere::meta::METRIC_TYPE, metric},
        {knowhere::meta::TOPK, topk},
        {knowhere::indexparam::DROP_RATIO_SEARCH, 0.0f},
        {knowhere::indexparam::USE_BLOCK_MAX, use_block_max && (algo == "DAAT_WAND")},
    };

    if (metric == "BM25") {
        // Use computed avgdl if provided, otherwise default to 100
        float bm25_avgdl = (avgdl > 0.0f) ? avgdl : 100.0f;
        build_conf[knowhere::meta::BM25_K1] = 1.2f;
        build_conf[knowhere::meta::BM25_B] = 0.75f;
        build_conf[knowhere::meta::BM25_AVGDL] = bm25_avgdl;
        search_conf[knowhere::meta::BM25_K1] = 1.2f;
        search_conf[knowhere::meta::BM25_B] = 0.75f;
        search_conf[knowhere::meta::BM25_AVGDL] = bm25_avgdl;
    }

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
        knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version);
    if (!idx.has_value()) {
        printf("Failed to create index\n");
        return {0, 0, 0, 0};
    }

    // Build
    auto build_start = std::chrono::high_resolution_clock::now();
    auto status = idx.value().Build(train_ds, build_conf);
    auto build_end = std::chrono::high_resolution_clock::now();
    double build_time_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();

    if (status != knowhere::Status::success) {
        printf("Build failed: %s\n", knowhere::Status2String(status).c_str());
        return {0, 0, 0, 0};
    }

    // DSP paper methodology: run 5 times, drop first 2, avg last 3
    // We use 'runs' parameter but ensure minimum of 5 for proper warmup
    int total_runs = std::max(runs, 5);
    int warmup_runs = 2;
    int measured_runs = total_runs - warmup_runs;

    auto nq = query_ds->GetRows();
    std::vector<double> run_times;
    knowhere::DataSetPtr last_result;

    for (int i = 0; i < total_runs; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto result = idx.value().Search(query_ds, search_conf, nullptr);
        auto t2 = std::chrono::high_resolution_clock::now();
        double run_time_us = std::chrono::duration<double, std::micro>(t2 - t1).count();

        if (i >= warmup_runs) {
            run_times.push_back(run_time_us);
        }
        if (result.has_value()) {
            last_result = result.value();
        }
    }

    // Average of measured runs (after warmup)
    double total_time_us = 0;
    for (double t : run_times) {
        total_time_us += t;
    }
    double search_time_us = total_time_us / measured_runs;
    double qps = nq / (search_time_us / 1e6);

    // Compute recall@k
    // Note: recall is computed as intersection(result, gt) / min(k, gt_k)
    // If gt has fewer than k items, we can only measure recall up to gt_k
    float recall = 0.0f;
    if (last_result && gt) {
        auto gt_k = static_cast<int64_t>(gt->GetDim());
        auto res_ids = last_result->GetIds();
        auto gt_ids = gt->GetIds();

        // We can only compute recall for min(topk, gt_k) items
        int64_t eval_k = std::min(static_cast<int64_t>(topk), gt_k);

        int matched = 0;
        for (int64_t i = 0; i < nq; ++i) {
            // Build set of ground truth top-eval_k results
            std::set<int64_t> gt_set(gt_ids + i * gt_k, gt_ids + i * gt_k + eval_k);
            // Check how many of our top-eval_k results are in ground truth
            for (int64_t j = 0; j < eval_k; ++j) {
                if (res_ids[i * topk + j] >= 0 && gt_set.count(res_ids[i * topk + j])) {
                    matched++;
                }
            }
        }
        recall = static_cast<float>(matched) / static_cast<float>(nq * eval_k);
    }

    return {build_time_ms, search_time_us, qps, recall};
}

void
PrintUsage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("Options:\n");
    printf("  --data-dir DIR     Directory containing MSMARCO/SPLADE data\n");
    printf("                     Expected files: base_full.csr, queries.dev.csr, base_full.gt\n");
    printf("  --synthetic        Run synthetic Zipfian benchmark (default if no data-dir)\n");
    printf("  --metric METRIC    IP or BM25 (default: IP)\n");
    printf("  --topk K           Top-k results (default: 10)\n");
    printf("  --runs N           Number of search runs (default: 10)\n");
    printf("  --help             Show this help\n");
}

int
main(int argc, char** argv) {
    std::string data_dir;
    std::string metric = "IP";
    int topk = 10;
    int runs = 10;
    bool use_synthetic = true;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
            use_synthetic = false;
        } else if (strcmp(argv[i], "--synthetic") == 0) {
            use_synthetic = true;
        } else if (strcmp(argv[i], "--metric") == 0 && i + 1 < argc) {
            metric = argv[++i];
        } else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            topk = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            PrintUsage(argv[0]);
            return 0;
        }
    }

    printf("Sparse Search Algorithm Benchmark (Knowhere Library)\n");
    printf("=====================================================\n\n");

    // ========================================================================
    // MSMARCO/SPLADE Dataset Benchmark
    // ========================================================================
    if (!data_dir.empty()) {
        printf("=== MSMARCO/SPLADE Dataset [%s metric] ===\n", metric.c_str());
        printf("Data directory: %s\n\n", data_dir.c_str());

        // Load base vectors
        printf("Loading base vectors...\n");
        CSRMatrix base_mat;
        std::string base_name = "base_full";
        if (!ReadCSRMatrix(data_dir + "/base_full.csr", base_mat)) {
            printf("  base_full.csr not found, trying base_small.csr...\n");
            base_name = "base_small";
            if (!ReadCSRMatrix(data_dir + "/base_small.csr", base_mat)) {
                printf("Failed to load base vectors.\n");
                return 1;
            }
        }

        // Check if data is L2 normalized
        bool base_normalized = IsL2Normalized(base_mat);
        printf("  Base vectors L2 normalized: %s\n", base_normalized ? "yes" : "no");

        // MSMARCO SPLADE GT was computed with raw IP (not normalized)
        // Do NOT normalize - use raw IP to match ground truth
        bool need_normalize = false;
        auto train_ds = CSRToKnowhereDataSet(base_mat, need_normalize);
        float avgdl = ComputeAvgDocLen(base_mat);
        printf("  Loaded %s: %ld vectors, dim=%ld, nnz=%ld, avgdl=%.2f\n", base_name.c_str(), base_mat.nrow,
               base_mat.ncol, base_mat.nnz, avgdl);

        // Load queries
        printf("Loading queries...\n");
        CSRMatrix query_mat;
        if (!ReadCSRMatrix(data_dir + "/queries.dev.csr", query_mat)) {
            printf("Failed to load queries.\n");
            return 1;
        }

        bool query_normalized = IsL2Normalized(query_mat);
        printf("  Query vectors L2 normalized: %s\n", query_normalized ? "yes" : "no");

        // MSMARCO SPLADE GT was computed with raw IP (not normalized)
        // Do NOT normalize - use raw IP to match ground truth
        bool need_normalize_query = false;
        auto query_ds = CSRToKnowhereDataSet(query_mat, need_normalize_query);
        printf("  Loaded %ld queries, dim=%ld, nnz=%ld\n", query_mat.nrow, query_mat.ncol, query_mat.nnz);

        // Load ground truth from .gt file
        // IMPORTANT: For MSMARCO, we MUST use the provided .gt file, not recompute
        // The .gt file contains ground truth computed with raw IP on unnormalized SPLADE vectors
        printf("Loading ground truth...\n");
        auto gt = ReadGroundTruth(data_dir + "/" + base_name + ".gt");
        if (!gt) {
            printf("ERROR: Ground truth file required for MSMARCO benchmark!\n");
            printf("  Expected: %s/%s.gt\n", data_dir.c_str(), base_name.c_str());
            printf("  Without ground truth, recall cannot be measured correctly.\n");
            printf("  Download from: https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/\n");
            return 1;
        }
        printf("  Loaded ground truth: %ld queries, k=%ld\n", gt->GetRows(), gt->GetDim());

        // Run benchmarks for multiple k values like DSP paper (k=10, k=1000)
        std::vector<int> k_values = {10, 1000};
        if (topk != 10 && topk != 1000) {
            k_values = {topk};  // Use user-specified k if different
        }

        printf("\nRunning benchmark (DSP paper methodology: 5 runs, drop first 2, avg last 3)...\n");
        printf("=====================================================================\n\n");

        for (int k : k_values) {
            printf("=== k=%d ===\n\n", k);

            // Use the .gt file for recall computation
            // If k > gt_k, recall will be computed for min(k, gt_k) items
            auto gt_k = static_cast<int64_t>(gt->GetDim());
            if (k > gt_k) {
                printf("Warning: k=%d > gt_k=%ld, recall will be computed for top-%ld only\n", k, gt_k, gt_k);
            }

            auto taat = RunBenchmark("TAAT_NAIVE", false, train_ds, query_ds, gt, k, runs, metric, avgdl);
            auto wand = RunBenchmark("DAAT_WAND", false, train_ds, query_ds, gt, k, runs, metric, avgdl);
            auto bmw = RunBenchmark("DAAT_WAND", true, train_ds, query_ds, gt, k, runs, metric, avgdl);
            auto maxscore = RunBenchmark("DAAT_MAXSCORE", false, train_ds, query_ds, gt, k, runs, metric, avgdl);

            // Sanity check: TAAT recall should be ~100% if using correct ground truth
            if (taat.recall < 0.95f) {
                printf("WARNING: TAAT recall is %.2f%% (expected ~100%%)\n", taat.recall * 100);
                printf("  This may indicate a mismatch between search metric and ground truth.\n");
                printf("  For MSMARCO SPLADE, use IP metric with L2-normalized vectors.\n\n");
            }

            // Print results in table format similar to DSP paper
            printf("%-12s %12s %15s %12s %10s\n", "Algorithm", "Build(ms)", "MRT(ms)", "QPS", "Recall");
            printf("%-12s %12.1f %15.3f %12.0f %9.2f%%\n", "TAAT", taat.build_time_ms, taat.search_time_us / 1000.0,
                   taat.qps, taat.recall * 100);
            printf("%-12s %12.1f %15.3f %12.0f %9.2f%%\n", "WAND", wand.build_time_ms, wand.search_time_us / 1000.0,
                   wand.qps, wand.recall * 100);
            printf("%-12s %12.1f %15.3f %12.0f %9.2f%%\n", "BMW", bmw.build_time_ms, bmw.search_time_us / 1000.0,
                   bmw.qps, bmw.recall * 100);
            printf("%-12s %12.1f %15.3f %12.0f %9.2f%%\n", "MaxScore", maxscore.build_time_ms,
                   maxscore.search_time_us / 1000.0, maxscore.qps, maxscore.recall * 100);

            printf("\nSpeedup vs TAAT (Mean Response Time):\n");
            printf("  WAND:     %.2fx\n", taat.search_time_us / wand.search_time_us);
            printf("  BMW:      %.2fx\n", taat.search_time_us / bmw.search_time_us);
            printf("  MaxScore: %.2fx\n", taat.search_time_us / maxscore.search_time_us);

            printf("\nSpeedup BMW vs WAND: %.2fx\n", wand.search_time_us / bmw.search_time_us);
            printf("\n");
        }
    }

    // ========================================================================
    // Synthetic Zipfian Benchmark
    // ========================================================================
    if (use_synthetic) {
        printf("\n=== Synthetic Zipfian Data [%s metric] ===\n", metric.c_str());
        printf("This simulates real text where:\n");
        printf("- Some terms are rare (high IDF) with concentrated high TF\n");
        printf("- Some terms are common (low IDF) appearing in many docs\n\n");

        struct ZipfConfig {
            const char* name;
            int32_t nb;
            int32_t nq;
            int32_t dim;
            float avg_terms_per_doc;
            int32_t query_terms;
            float zipf_exp;
        };

        std::vector<ZipfConfig> zipf_configs = {
            {"Small Zipf 1.5", 50000, 100, 10000, 50.0f, 10, 1.5f},
            {"Medium Zipf 1.5", 200000, 100, 20000, 100.0f, 15, 1.5f},
        };

        for (const auto& cfg : zipf_configs) {
            printf("Config: %s (nb=%d, nq=%d, dim=%d)\n", cfg.name, cfg.nb, cfg.nq, cfg.dim);

            auto train_ds = GenZipfianDataSet(cfg.nb, cfg.dim, cfg.avg_terms_per_doc, cfg.zipf_exp, 42);
            auto query_ds = GenZipfianQuery(cfg.nq, cfg.dim, cfg.query_terms, cfg.zipf_exp, 123);

            // Compute ground truth
            knowhere::Json gt_conf = {
                {knowhere::meta::METRIC_TYPE, metric},
                {knowhere::meta::TOPK, topk},
            };
            if (metric == "BM25") {
                gt_conf[knowhere::meta::BM25_K1] = 1.2f;
                gt_conf[knowhere::meta::BM25_B] = 0.75f;
                gt_conf[knowhere::meta::BM25_AVGDL] = 100.0f;
            }
            auto gt_result = knowhere::BruteForce::SearchSparse(train_ds, query_ds, gt_conf, nullptr);
            knowhere::DataSetPtr gt = gt_result.has_value() ? gt_result.value() : nullptr;

            auto taat = RunBenchmark("TAAT_NAIVE", false, train_ds, query_ds, gt, topk, runs, metric);
            auto wand = RunBenchmark("DAAT_WAND", false, train_ds, query_ds, gt, topk, runs, metric);
            auto bmw = RunBenchmark("DAAT_WAND", true, train_ds, query_ds, gt, topk, runs, metric);
            auto maxscore = RunBenchmark("DAAT_MAXSCORE", false, train_ds, query_ds, gt, topk, runs, metric);

            printf("  TAAT:      %8.0f us,          recall=%.2f%%\n", taat.search_time_us, taat.recall * 100);
            printf("  WAND:      %8.0f us (%.2fx),  recall=%.2f%%\n", wand.search_time_us,
                   taat.search_time_us / wand.search_time_us, wand.recall * 100);
            printf("  BMW:       %8.0f us (%.2fx),  recall=%.2f%%\n", bmw.search_time_us,
                   taat.search_time_us / bmw.search_time_us, bmw.recall * 100);
            printf("  MaxScore:  %8.0f us (%.2fx),  recall=%.2f%%\n", maxscore.search_time_us,
                   taat.search_time_us / maxscore.search_time_us, maxscore.recall * 100);
            printf("\n");
        }
    }

    return 0;
}
