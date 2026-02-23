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
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(int64_t));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(int64_t));
        printf("  CSR: %ld rows, %ld cols, %ld nnz\n", n_rows, n_cols, nnz);

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
};

struct GroundTruth {
    std::vector<std::vector<int32_t>> gt;
    int64_t nq = 0;
    int64_t k = 0;

    bool
    load(const std::string& path, int64_t expected_nq) {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            return false;
        int32_t nq32, k32;
        file.read(reinterpret_cast<char*>(&nq32), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&k32), sizeof(int32_t));
        nq = nq32;
        k = k32;
        printf("  GT: %ld queries, k=%ld\n", nq, k);
        nq = std::min(nq, expected_nq);
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

void
print_usage(const char* prog) {
    printf(
        "Usage: %s --data-dir <path> --data-type <splade|bm25> [--base <file>] [--query <file>] [--topk <k>] [--nq "
        "<n>] [--mu <f>] [--eta <f>]\n",
        prog);
}

int
main(int argc, char** argv) {
    std::string data_dir;
    std::string data_type;
    std::string base_file = "base_small.csr";
    std::string query_file = "queries.dev.csr";
    int64_t topk = 10;
    int64_t nq = 0;
    float dsp_mu = 1.0f;
    float dsp_eta = 1.0f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--data-type") == 0 && i + 1 < argc) {
            data_type = argv[++i];
        } else if (strcmp(argv[i], "--base") == 0 && i + 1 < argc) {
            base_file = argv[++i];
        } else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc) {
            query_file = argv[++i];
        } else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) {
            topk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nq") == 0 && i + 1 < argc) {
            nq = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mu") == 0 && i + 1 < argc) {
            dsp_mu = atof(argv[++i]);
        } else if (strcmp(argv[i], "--eta") == 0 && i + 1 < argc) {
            dsp_eta = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_dir.empty() || data_type.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::string metric = (data_type == "splade") ? "IP" : "BM25";

    printf("==========================================================\n");
    printf("  DSP Sparse Search Benchmark\n");
    printf("==========================================================\n\n");

    knowhere::KnowhereConfig::SetSimdType(knowhere::KnowhereConfig::SimdType::AUTO);

    // Load data
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

    // Compute avgdl for BM25
    double avgdl = 0.0;
    if (metric == "BM25") {
        double total_len = 0.0;
        for (int64_t i = 0; i < base.n_rows; ++i) {
            for (int64_t j = base.indptr[i]; j < base.indptr[i + 1]; ++j) {
                total_len += base.data[j];
            }
        }
        avgdl = total_len / base.n_rows;
        printf("  avgdl: %.2f\n", avgdl);
    }

    // Free CSR data
    base.indptr.clear();
    base.indptr.shrink_to_fit();
    base.indices.clear();
    base.indices.shrink_to_fit();
    base.data.clear();
    base.data.shrink_to_fit();

    // Load ground truth
    std::string metric_lower = metric;
    std::transform(metric_lower.begin(), metric_lower.end(), metric_lower.begin(), ::tolower);
    GroundTruth gt;
    std::string gt_path = data_dir + "/base_small.dev." + metric_lower + ".gt";
    if (!gt.load(gt_path, queries.n_rows)) {
        gt_path = data_dir + "/base_small.dev.gt";
        gt.load(gt_path, queries.n_rows);
    }

    printf("\n[Config] base=%ld, nq=%ld, k=%ld, metric=%s, mu=%.2f, eta=%.2f\n", base.n_rows, nq, topk, metric.c_str(),
           dsp_mu, dsp_eta);

#ifdef SEEK_INSTRUMENTATION
    knowhere::sparse::g_seek_stats.reset();
    knowhere::sparse::g_dsp_stats.reset();
#endif

    // Build DSP index
    printf("\n[Building DSP Index]\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    auto index_result = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
        "SPARSE_INVERTED_INDEX", knowhere::Version::GetCurrentVersion().VersionNumber());
    if (!index_result.has_value()) {
        printf("Error: Failed to create index\n");
        return 1;
    }
    auto index = index_result.value();

    knowhere::Json build_conf;
    build_conf["metric_type"] = metric;
    build_conf["inverted_index_algo"] = "DSP";
    if (metric == "BM25") {
        build_conf["bm25_k1"] = 1.2f;
        build_conf["bm25_b"] = 0.75f;
        build_conf["bm25_avgdl"] = static_cast<float>(avgdl);
    }

    auto ds = knowhere::GenDataSet(base.n_rows, base.n_cols, nullptr);
    ds->SetIsSparse(true);
    ds->SetTensor(base_rows.get());

    auto status = index.Build(ds, build_conf);
    if (status != knowhere::Status::success) {
        printf("Error: Failed to build index: %s\n", knowhere::Status2String(status).c_str());
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Build time: %.0f ms\n", build_ms);

    // Search
    knowhere::Json search_conf;
    search_conf["metric_type"] = metric;
    search_conf["drop_ratio_search"] = 0.0f;
    search_conf["topk"] = topk;
    search_conf["dsp_mu"] = dsp_mu;
    search_conf["dsp_eta"] = dsp_eta;
    if (metric == "BM25") {
        search_conf["bm25_k1"] = 1.2f;
        search_conf["bm25_b"] = 0.75f;
        search_conf["bm25_avgdl"] = static_cast<float>(avgdl);
    }

    auto query_ds = knowhere::GenDataSet(1, queries.n_cols, nullptr);
    query_ds->SetIsSparse(true);

    printf("\n[Searching]\n");
    std::vector<int64_t> all_results(nq * topk, -1);
    int64_t failed = 0;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int64_t q = 0; q < nq; ++q) {
        query_ds->SetTensor(&query_rows[q]);
        auto result = index.Search(query_ds, search_conf, knowhere::BitsetView());
        if (!result.has_value()) {
            failed++;
            continue;
        }
        memcpy(&all_results[q * topk], result.value()->GetIds(), topk * sizeof(int64_t));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double search_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Recall
    float avg_recall = 0;
    int64_t valid = 0;
    if (gt.nq > 0) {
        for (int64_t q = 0; q < std::min(nq, gt.nq); ++q) {
            if (all_results[q * topk] != -1) {
                avg_recall += gt.compute_recall(&all_results[q * topk], q, topk);
                valid++;
            }
        }
        if (valid > 0)
            avg_recall /= valid;
    }

    printf("\n[Results]\n");
    printf("  Search: %.0f ms (%.1f QPS)\n", search_ms, nq * 1000.0 / search_ms);
    printf("  Recall@%ld: %.2f%% (%ld/%ld queries)\n", topk, avg_recall * 100, valid, nq);
    if (failed > 0)
        printf("  Failed queries: %ld\n", failed);

#ifdef SEEK_INSTRUMENTATION
    knowhere::sparse::g_seek_stats.print("DSP");
    knowhere::sparse::g_dsp_stats.print("DSP");
#endif

    printf("\n=== Done ===\n");
    return 0;
}
