// Copyright (C) 2019-2024 Zilliz. All rights reserved.
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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "nlohmann/json.hpp"
#include "utils.h"

namespace {

// ============================================================================
// Configuration: Control test behavior
// ============================================================================
constexpr int32_t MAX_DOCS_TO_LOAD = 5000;    // Hard-coded limit on documents (matches random sampling default)
constexpr int32_t MAX_QUERIES_TO_LOAD = 100;  // Hard-coded limit on queries
constexpr bool SKIP_DIRECT_TEST = true;       // Set to true to skip Direct strategy (it's slow)
constexpr bool SKIP_MUVERA_TEST = true;       // Set to true to skip MUVERA strategy

// Helper to get env var with default value
inline std::string
GetEnvOr(const char* name, const std::string& default_val) {
    const char* val = std::getenv(name);
    return val ? std::string(val) : default_val;
}

// Model tag from environment (e.g., "colbertv2", "bgem3")
// Usage: MODEL_TAG=colbertv2 ./knowhere_tests "[scifact_emb_list]"
inline std::string
GetModelTag() {
    return GetEnvOr("MODEL_TAG", "gt");  // default "gt" for backward compatibility
}

// Build data file path: {dataset}_{model_tag}_{suffix}.jsonl
inline std::string
BuildDataPath(const std::string& dataset, const std::string& suffix) {
    std::string model_tag = GetModelTag();
    return dataset + "_" + model_tag + "_" + suffix + ".jsonl";
}

// MS MARCO data file paths (with official ground truth annotations)
// Can be overridden via environment: MSMARCO_DOCS_PATH, MSMARCO_QUERIES_PATH
// Or use MODEL_TAG to auto-generate: MODEL_TAG=bgem3 -> msmarco_bgem3_docs.jsonl
inline std::string
GetMsmarcoDocsPath() {
    return GetEnvOr("MSMARCO_DOCS_PATH", BuildDataPath("msmarco", "docs"));
}
inline std::string
GetMsmarcoQueriesPath() {
    return GetEnvOr("MSMARCO_QUERIES_PATH", BuildDataPath("msmarco", "queries"));
}

// LoTTE data file paths
inline std::string
GetLotteDocsPath() {
    return GetEnvOr("LOTTE_DOCS_PATH", BuildDataPath("lotte_lifestyle", "docs"));
}
inline std::string
GetLotteQueriesPath() {
    return GetEnvOr("LOTTE_QUERIES_PATH", BuildDataPath("lotte_lifestyle", "queries"));
}

// SciFact data file paths
inline std::string
GetScifactDocsPath() {
    return GetEnvOr("SCIFACT_DOCS_PATH", BuildDataPath("scifact", "docs"));
}
inline std::string
GetScifactQueriesPath() {
    return GetEnvOr("SCIFACT_QUERIES_PATH", BuildDataPath("scifact", "queries"));
}

// TREC-COVID data file paths (graded relevance: 0/1/2)
inline std::string
GetTrecCovidDocsPath() {
    return GetEnvOr("TREC_COVID_DOCS_PATH", BuildDataPath("trec_covid", "docs"));
}
inline std::string
GetTrecCovidQueriesPath() {
    return GetEnvOr("TREC_COVID_QUERIES_PATH", BuildDataPath("trec_covid", "queries"));
}

// DocVQA data file paths (multimodal: ColQwen2 multi-vector or Qwen3-VL-Embedding dense)
inline std::string
GetDocvqaDocsPath() {
    return GetEnvOr("DOCVQA_DOCS_PATH", BuildDataPath("docvqa", "docs"));
}
inline std::string
GetDocvqaQueriesPath() {
    return GetEnvOr("DOCVQA_QUERIES_PATH", BuildDataPath("docvqa", "queries"));
}

// ============================================================================
// EmbListData: Load and manage embedding list data
// ============================================================================

struct EmbListData {
    std::vector<float> vectors;
    std::vector<size_t> offsets;
    int32_t dim = 0;
    int64_t num_docs = 0;
    int64_t total_vectors = 0;

    bool
    LoadFromJsonl(const std::string& jsonl_path, int32_t max_docs) {
        std::ifstream file(jsonl_path);
        if (!file) {
            printf("Cannot open JSONL file: %s\n", jsonl_path.c_str());
            return false;
        }

        // Phase 1: Read all lines into memory
        printf("Reading JSONL file into memory...\n");
        std::vector<std::string> lines;
        lines.reserve(max_docs);
        std::string line;
        while (std::getline(file, line) && (int32_t)lines.size() < max_docs) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        file.close();
        printf("Read %zu lines, parsing with multiple threads...\n", lines.size());

        // Phase 2: Parse in parallel
        int num_threads = std::max(1, (int)std::thread::hardware_concurrency());
        int64_t total_lines = lines.size();
        int64_t chunk_size = (total_lines + num_threads - 1) / num_threads;

        struct ParsedDoc {
            std::vector<float> vecs;
            int32_t num_chunks = 0;
            int32_t dim = 0;
        };

        std::vector<std::vector<ParsedDoc>> thread_results(num_threads);

        auto parse_worker = [&](int tid) {
            int64_t start = tid * chunk_size;
            int64_t end = std::min(start + chunk_size, total_lines);
            if (start >= end)
                return;
            auto& results = thread_results[tid];
            results.reserve(end - start);

            for (int64_t i = start; i < end; ++i) {
                try {
                    auto json = nlohmann::json::parse(lines[i]);
                    const auto& chunks = json["chunks"];

                    ParsedDoc doc;
                    doc.num_chunks = chunks.size();

                    for (const auto& chunk : chunks) {
                        const auto& emb = chunk["emb"];
                        if (doc.dim == 0) {
                            doc.dim = static_cast<int32_t>(emb.size());
                        }
                        for (const auto& val : emb) {
                            doc.vecs.push_back(val.get<float>());
                        }
                    }
                    results.push_back(std::move(doc));
                } catch (const std::exception& e) {
                    printf("Error parsing line %ld: %s\n", i, e.what());
                }
            }
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(parse_worker, t);
        }
        for (auto& t : threads) {
            t.join();
        }

        // Free lines to reclaim memory before merge
        { std::vector<std::string>().swap(lines); }

        // Phase 3: Merge results in order
        offsets.clear();
        offsets.push_back(0);

        size_t total_floats = 0;
        size_t total_docs_parsed = 0;
        for (const auto& tr : thread_results) {
            for (const auto& doc : tr) {
                total_floats += doc.vecs.size();
                total_docs_parsed++;
            }
        }
        vectors.reserve(total_floats);

        for (auto& tr : thread_results) {
            for (auto& doc : tr) {
                if (dim == 0) {
                    dim = doc.dim;
                }
                vectors.insert(vectors.end(), doc.vecs.begin(), doc.vecs.end());
                offsets.push_back(offsets.back() + doc.num_chunks);
                std::vector<float>().swap(doc.vecs);
            }
            std::vector<ParsedDoc>().swap(tr);
        }

        num_docs = total_docs_parsed;
        total_vectors = offsets.back();

        printf("Loaded %ld docs, %ld vectors, dim=%d from JSONL (%d threads)\n", num_docs, total_vectors, dim,
               num_threads);
        return true;
    }

    knowhere::DataSetPtr
    ToDataSet() const {
        size_t* ofs = new size_t[offsets.size()];
        std::memcpy(ofs, offsets.data(), offsets.size() * sizeof(size_t));

        auto ds = knowhere::GenDataSet(total_vectors, dim, vectors.data());
        ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(ofs));
        ds->SetIsOwner(false);
        return ds;
    }

    void
    PrintStats() const {
        if (num_docs == 0)
            return;

        std::vector<int64_t> counts(num_docs);
        int64_t min_count = INT64_MAX, max_count = 0, sum_count = 0;

        for (int64_t i = 0; i < num_docs; ++i) {
            counts[i] = offsets[i + 1] - offsets[i];
            min_count = std::min(min_count, counts[i]);
            max_count = std::max(max_count, counts[i]);
            sum_count += counts[i];
        }

        std::sort(counts.begin(), counts.end());
        double avg_count = (double)sum_count / num_docs;
        int64_t median_count = counts[num_docs / 2];

        printf("Vectors per doc: min=%ld, max=%ld, avg=%.1f, median=%ld\n", min_count, max_count, avg_count,
               median_count);
    }
};

// Query data with ground truth annotations
struct QueryDataWithGT {
    std::vector<float> vectors;
    std::vector<size_t> offsets;
    std::vector<std::vector<int64_t>> gt_pids;              // Ground truth doc IDs per query
    std::vector<std::unordered_map<int64_t, int>> gt_rels;  // doc_id -> relevance grade (graded)
    bool has_graded_relevance = false;
    int32_t dim = 0;
    int64_t num_queries = 0;
    int64_t total_vectors = 0;

    bool
    LoadFromJsonl(const std::string& jsonl_path, int32_t max_queries) {
        std::ifstream file(jsonl_path);
        if (!file) {
            printf("Cannot open JSONL file: %s\n", jsonl_path.c_str());
            return false;
        }

        std::vector<std::string> lines;
        lines.reserve(max_queries);
        std::string line;
        while (std::getline(file, line) && (int32_t)lines.size() < max_queries) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        file.close();

        int num_threads = std::min((int)std::thread::hardware_concurrency(), (int)lines.size());
        num_threads = std::max(1, num_threads);
        int64_t total_lines = lines.size();
        int64_t chunk_size = (total_lines + num_threads - 1) / num_threads;

        struct ParsedQuery {
            std::vector<float> vecs;
            int32_t num_chunks = 0;
            int32_t dim = 0;
            std::vector<int64_t> gt;
            std::unordered_map<int64_t, int> rels;
        };

        std::vector<std::vector<ParsedQuery>> thread_results(num_threads);

        auto parse_worker = [&](int tid) {
            int64_t start = tid * chunk_size;
            int64_t end = std::min(start + chunk_size, total_lines);
            if (start >= end)
                return;
            auto& results = thread_results[tid];
            results.reserve(end - start);

            for (int64_t i = start; i < end; ++i) {
                try {
                    auto json = nlohmann::json::parse(lines[i]);
                    const auto& chunks = json["chunks"];

                    ParsedQuery q;
                    q.num_chunks = chunks.size();

                    for (const auto& chunk : chunks) {
                        const auto& emb = chunk["emb"];
                        if (q.dim == 0) {
                            q.dim = static_cast<int32_t>(emb.size());
                        }
                        for (const auto& val : emb) {
                            q.vecs.push_back(val.get<float>());
                        }
                    }

                    if (json.contains("gt_pids")) {
                        for (const auto& pid : json["gt_pids"]) {
                            q.gt.push_back(pid.get<int64_t>());
                        }
                    }
                    if (json.contains("gt_rels")) {
                        for (auto& [key, val] : json["gt_rels"].items()) {
                            q.rels[std::stoll(key)] = val.get<int>();
                        }
                    } else {
                        for (auto pid : q.gt) {
                            q.rels[pid] = 1;
                        }
                    }
                    results.push_back(std::move(q));
                } catch (const std::exception& e) {
                    printf("Error parsing query line %ld: %s\n", i, e.what());
                }
            }
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(parse_worker, t);
        }
        for (auto& t : threads) {
            t.join();
        }
        offsets.clear();
        offsets.push_back(0);
        gt_pids.clear();
        gt_rels.clear();

        bool found_graded = false;
        for (const auto& tr : thread_results) {
            for (const auto& q : tr) {
                if (dim == 0) {
                    dim = q.dim;
                }
                vectors.insert(vectors.end(), q.vecs.begin(), q.vecs.end());
                offsets.push_back(offsets.back() + q.num_chunks);
                gt_pids.push_back(q.gt);
                gt_rels.push_back(q.rels);
                for (const auto& [id, rel] : q.rels) {
                    if (rel > 1)
                        found_graded = true;
                }
            }
        }
        has_graded_relevance = found_graded;
        num_queries = gt_pids.size();
        total_vectors = offsets.back();

        printf("Loaded %ld queries, %ld vectors, dim=%d from JSONL\n", num_queries, total_vectors, dim);

        int64_t total_gt = 0, min_gt = INT64_MAX, max_gt = 0;
        for (const auto& gt : gt_pids) {
            total_gt += gt.size();
            min_gt = std::min(min_gt, (int64_t)gt.size());
            max_gt = std::max(max_gt, (int64_t)gt.size());
        }
        printf("GT per query: min=%ld, max=%ld, avg=%.1f\n", min_gt, max_gt, (double)total_gt / num_queries);
        if (has_graded_relevance) {
            printf("Graded relevance: YES (nDCG will use gain=2^rel-1)\n");
        }

        return true;
    }

    knowhere::DataSetPtr
    ToDataSet() const {
        size_t* ofs = new size_t[offsets.size()];
        std::memcpy(ofs, offsets.data(), offsets.size() * sizeof(size_t));

        auto ds = knowhere::GenDataSet(total_vectors, dim, vectors.data());
        ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(ofs));
        ds->SetIsOwner(false);
        return ds;
    }

    void
    PrintStats() const {
        if (num_queries == 0)
            return;

        std::vector<int64_t> counts(num_queries);
        int64_t min_count = INT64_MAX, max_count = 0, sum_count = 0;

        for (int64_t i = 0; i < num_queries; ++i) {
            counts[i] = offsets[i + 1] - offsets[i];
            min_count = std::min(min_count, counts[i]);
            max_count = std::max(max_count, counts[i]);
            sum_count += counts[i];
        }

        std::sort(counts.begin(), counts.end());
        double avg_count = (double)sum_count / num_queries;
        int64_t median_count = counts[num_queries / 2];

        printf("Vectors per query: min=%ld, max=%ld, avg=%.1f, median=%ld\n", min_count, max_count, avg_count,
               median_count);
    }
};

}  // namespace

TEST_CASE("MS MARCO ColBERT: Direct vs MUVERA", "[msmarco_emb_list]") {
    // Get data paths (can be overridden via MODEL_TAG or explicit env vars)
    const std::string docs_path = GetMsmarcoDocsPath();
    const std::string queries_path = GetMsmarcoQueriesPath();

    // Check if MS MARCO data files exist
    {
        std::ifstream docs_file(docs_path);
        std::ifstream queries_file(queries_path);

        if (!docs_file.good() || !queries_file.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("MS MARCO data files not found. Please prepare the data first.\n");
            printf("Expected files:\n");
            printf("  - %s\n", docs_path.c_str());
            printf("  - %s\n", queries_path.c_str());
            printf("\n");
            printf("Generate MS MARCO data with GT annotations:\n");
            printf("  python scripts/prepare_msmarco_with_gt.py --output-dir .\n");
            printf("Or specify MODEL_TAG: MODEL_TAG=bgem3 ./knowhere_tests\n");
            printf("=============================================================\n");
            SKIP("MS MARCO data files not found");
            return;
        }
    }

    // Load data
    printf("\n=== Loading MS MARCO Data (model_tag=%s) ===\n", GetModelTag().c_str());
    EmbListData doc_data;
    QueryDataWithGT query_data;

    REQUIRE(doc_data.LoadFromJsonl(docs_path, MAX_DOCS_TO_LOAD));
    doc_data.PrintStats();

    REQUIRE(query_data.LoadFromJsonl(queries_path, MAX_QUERIES_TO_LOAD));
    query_data.PrintStats();

    auto doc_ds = doc_data.ToDataSet();
    auto query_ds = query_data.ToDataSet();

    const int32_t dim = doc_data.dim;
    const int32_t num_docs = doc_data.num_docs;
    const int64_t total_vectors = doc_data.total_vectors;
    const int32_t num_queries = std::min((int32_t)query_data.num_queries, MAX_QUERIES_TO_LOAD);

    // Multiple topk values for evaluation
    const std::vector<int32_t> topk_values = {10, 20, 50};
    const int32_t max_topk = *std::max_element(topk_values.begin(), topk_values.end());

    printf("\n=== Test Configuration ===\n");
    printf("Documents: %d, Total vectors: %ld, Dim: %d\n", num_docs, total_vectors, dim);
    printf("Queries: %d, TopK values: ", num_queries);
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("%d%s", topk_values[i], i < topk_values.size() - 1 ? ", " : "\n");
    }
    fflush(stdout);

    // Base config - use max_topk for search, then evaluate at different k
    knowhere::Json base_conf;
    base_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    base_conf[knowhere::meta::DIM] = dim;
    base_conf[knowhere::meta::TOPK] = max_topk;
    base_conf[knowhere::indexparam::HNSW_M] = 16;
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;
    base_conf[knowhere::indexparam::EF] = std::max(128, max_topk * 2);
    base_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 2.0f;

    auto version = GenTestEmbListVersionList();

    // ========== Official Ground Truth ==========
    printf("\n[Ground Truth] Using official MS MARCO annotations (gt_pids)\n");
    fflush(stdout);

    // E2E Recall calculation based on official GT annotations (per-query average)
    // Recall@k = (number of GT documents found in top-k results) / (total GT documents)
    // result_ids has result_k items per query
    auto calc_recall_vs_gt = [&](const int64_t* result_ids, int32_t result_k, int32_t k) {
        float total_recall = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& gt = query_data.gt_pids[q];
            if (gt.empty())
                continue;

            std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
            int found = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                if (doc_id >= 0 && gt_set.count(doc_id) > 0) {
                    found++;
                }
            }
            total_recall += (float)found / gt.size();
            valid_queries++;
        }
        return valid_queries > 0 ? total_recall / valid_queries : 0.0f;
    };

    // nDCG@k calculation vs GT (per-query average, supports graded relevance)
    auto calc_ndcg_vs_gt = [&](const int64_t* result_ids, int32_t result_k, int32_t k) {
        float total_ndcg = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& rels = query_data.gt_rels[q];
            if (rels.empty())
                continue;
            double dcg = 0.0;
            for (int i = 0; i < k && i < result_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                auto it = rels.find(doc_id);
                if (doc_id >= 0 && it != rels.end()) {
                    dcg += (std::pow(2.0, it->second) - 1.0) / std::log2(i + 2.0);
                }
            }
            std::vector<int> sorted_rels;
            sorted_rels.reserve(rels.size());
            for (const auto& [id, rel] : rels) {
                sorted_rels.push_back(rel);
            }
            std::sort(sorted_rels.rbegin(), sorted_rels.rend());
            double idcg = 0.0;
            int ideal_count = std::min((int)sorted_rels.size(), k);
            for (int i = 0; i < ideal_count; ++i) {
                idcg += (std::pow(2.0, sorted_rels[i]) - 1.0) / std::log2(i + 2.0);
            }
            total_ndcg += idcg > 0 ? (float)(dcg / idcg) : 0.0f;
            valid_queries++;
        }
        return valid_queries > 0 ? total_ndcg / valid_queries : 0.0f;
    };

    // MRR@k calculation vs GT (per-query average)
    auto calc_mrr_vs_gt = [&](const int64_t* result_ids, int32_t result_k, int32_t k) {
        float total_rr = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& gt = query_data.gt_pids[q];
            if (gt.empty())
                continue;
            std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
            for (int i = 0; i < k && i < result_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                if (doc_id >= 0 && gt_set.count(doc_id) > 0) {
                    total_rr += 1.0f / (i + 1);
                    break;
                }
            }
            valid_queries++;
        }
        return valid_queries > 0 ? total_rr / valid_queries : 0.0f;
    };

    // Recall calculation vs BruteForce results (per-query average)
    // result_ids has result_k items per query, bf_ids has bf_k items per query
    auto calc_recall_vs_bf = [&](const int64_t* result_ids, int32_t result_k, const int64_t* bf_ids, int32_t bf_k,
                                 int32_t k) {
        if (bf_ids == nullptr)
            return 0.0f;
        float total_recall = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            std::unordered_set<int64_t> bf_set;
            int bf_count = 0;
            for (int i = 0; i < k && i < bf_k; ++i) {
                if (bf_ids[q * bf_k + i] >= 0) {
                    bf_set.insert(bf_ids[q * bf_k + i]);
                    bf_count++;
                }
            }
            if (bf_count == 0)
                continue;

            int overlap = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                if (result_ids[q * result_k + i] >= 0 && bf_set.count(result_ids[q * result_k + i]) > 0) {
                    overlap++;
                }
            }
            total_recall += (float)overlap / bf_count;
        }
        return total_recall / num_queries;
    };

    // ========== BruteForce MaxSim ==========
    double bf_time = 0;
    std::vector<float> bf_recalls_vs_gt(topk_values.size(), 0.0f);
    std::map<int32_t, knowhere::DataSetPtr> bf_results;
    std::map<int32_t, const int64_t*> bf_ids_map;

    if (!SKIP_DIRECT_TEST) {
        printf("\n[BruteForce] Computing MaxSim results for each k...\n");
        fflush(stdout);

        for (int32_t k : topk_values) {
            knowhere::Json bf_conf;
            bf_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
            bf_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_bf;
            auto bf_result = knowhere::BruteForce::Search<knowhere::fp32>(doc_ds, query_ds, bf_conf, nullptr);
            bf_time += sw_bf.elapsed();
            REQUIRE(bf_result.has_value());

            bf_results[k] = bf_result.value();
            bf_ids_map[k] = bf_results[k]->GetIds();
        }
        printf("[BruteForce] Total time: %.3f s\n", bf_time);

        // Calculate GT metrics at each topk
        printf("[BruteForce] Recall (vs GT): ");
        for (size_t i = 0; i < topk_values.size(); ++i) {
            int32_t k = topk_values[i];
            bf_recalls_vs_gt[i] = calc_recall_vs_gt(bf_ids_map[k], k, k);
            printf("@%d=%.1f%% ", k, bf_recalls_vs_gt[i] * 100);
        }
        printf("\n");
        // nDCG@10 and MRR@10 from BF k=10 results
        if (bf_ids_map.count(10) > 0) {
            printf("[BruteForce] nDCG@10=%.4f, MRR@10=%.4f\n", calc_ndcg_vs_gt(bf_ids_map[10], 10, 10),
                   calc_mrr_vs_gt(bf_ids_map[10], 10, 10));
        }
        fflush(stdout);
    }

    // ========== Direct Strategy ==========
    double direct_build_time = 0, direct_search_time = 0;
    std::vector<float> direct_recalls(topk_values.size(), 0.0f);

    if (SKIP_DIRECT_TEST) {
        printf("\n[Direct] SKIPPED (SKIP_DIRECT_TEST = true)\n");
        fflush(stdout);
    } else {
        printf("\n[Direct] Building HNSW index for %ld vectors...\n", total_vectors);
        fflush(stdout);

        knowhere::Json direct_conf = base_conf;
        direct_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
        direct_conf["emb_list_strategy"] = "tokenann";

        auto direct_index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
        REQUIRE(direct_index.has_value());

        StopWatch sw_direct_build;
        auto direct_build_status = direct_index.value().Build(doc_ds, direct_conf);
        direct_build_time = sw_direct_build.elapsed();
        REQUIRE(direct_build_status == knowhere::Status::success);
        printf("[Direct] Build time: %.3f s\n", direct_build_time);

        printf("[Direct] Searching with each k...\n");
        fflush(stdout);

        for (size_t i = 0; i < topk_values.size(); ++i) {
            int32_t k = topk_values[i];
            knowhere::Json search_conf = direct_conf;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_search;
            auto direct_result = direct_index.value().Search(query_ds, search_conf, nullptr);
            direct_search_time += sw_search.elapsed();
            REQUIRE(direct_result.has_value());

            auto direct_ids = direct_result.value()->GetIds();
            direct_recalls[i] = calc_recall_vs_bf(direct_ids, k, bf_ids_map[k], k, k);
        }

        printf("[Direct] Search time: %.3f ms, Recall: ", direct_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], direct_recalls[i] * 100);
        }
        printf("\n");
        fflush(stdout);
    }

    // ========== MUVERA Strategy with Multiple Parameter Combinations ==========
    // Define parameter combinations: (num_projections, num_repeats)
    std::vector<std::pair<int32_t, int32_t>> muvera_params = {
        {2, 3}, {2, 5}, {2, 7}, {3, 3}, {3, 5}, {3, 7}, {4, 3}, {4, 5},
        {4, 7}, {5, 3}, {5, 5}, {5, 7}, {6, 3}, {6, 5}, {6, 7},
    };

    // Store results for each combination
    struct MuveraResult {
        int32_t num_projections;
        int32_t num_repeats;
        double build_time;
        double search_time;
        std::vector<float> recalls;  // recall at each topk
    };
    std::vector<MuveraResult> muvera_results;

    printf("\n[MUVERA] Testing %zu parameter combinations...\n", muvera_params.size());
    fflush(stdout);

    for (const auto& params : muvera_params) {
        int32_t num_proj = params.first;
        int32_t num_rep = params.second;

        printf("\n[MUVERA-%d-%d] Building HNSW index...\n", num_proj, num_rep);
        fflush(stdout);

        knowhere::Json muvera_conf = base_conf;
        muvera_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
        muvera_conf["emb_list_strategy"] = "muvera";
        muvera_conf["muvera_num_projections"] = num_proj;
        muvera_conf["muvera_num_repeats"] = num_rep;
        muvera_conf["muvera_seed"] = 42;

        auto muvera_index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
        REQUIRE(muvera_index.has_value());

        StopWatch sw_muvera_build;
        auto muvera_build_status = muvera_index.value().Build(doc_ds, muvera_conf);
        double build_time = sw_muvera_build.elapsed();
        REQUIRE(muvera_build_status == knowhere::Status::success);
        printf("[MUVERA-%d-%d] Build time: %.3f s\n", num_proj, num_rep, build_time);

        printf("[MUVERA-%d-%d] Searching with each k...\n", num_proj, num_rep);
        fflush(stdout);

        std::vector<float> recalls;
        double search_time = 0;

        for (int32_t k : topk_values) {
            knowhere::Json search_conf = muvera_conf;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_muvera_search;
            auto muvera_result = muvera_index.value().Search(query_ds, search_conf, nullptr);
            search_time += sw_muvera_search.elapsed();
            REQUIRE(muvera_result.has_value());

            auto muvera_ids = muvera_result.value()->GetIds();
            float recall = bf_ids_map.count(k) > 0 ? calc_recall_vs_bf(muvera_ids, k, bf_ids_map[k], k, k) : 0.0f;
            recalls.push_back(recall);
        }

        printf("[MUVERA-%d-%d] Search time: %.3f ms, Recall: ", num_proj, num_rep, search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], recalls[i] * 100);
        }
        printf("\n");
        fflush(stdout);

        muvera_results.push_back({num_proj, num_rep, build_time, search_time, recalls});
    }

    // ========== Summary ==========
    printf("\n============================================================================================\n");
    printf("                              Summary (MS MARCO)                                            \n");
    printf("============================================================================================\n");

    // Header with topk columns
    printf("| Strategy      | Build Time | Search Time |");
    for (int32_t k : topk_values) {
        printf(" R@%-3d |", k);
    }
    printf("\n");

    printf("|---------------|------------|-------------|");
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("-------|");
    }
    printf("\n");

    if (!SKIP_DIRECT_TEST) {
        // BruteForce row
        printf("| BruteForce    | %10s | %9.2f s |", "-", bf_time);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", 100.0f);
        }
        printf("\n");

        // Direct row
        printf("| Direct        | %8.2f s | %9.2f ms |", direct_build_time, direct_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", direct_recalls[i] * 100);
        }
        printf("\n");
    }

    // MUVERA rows
    for (const auto& res : muvera_results) {
        char name[32];
        snprintf(name, sizeof(name), "MUVERA-%d-%d", res.num_projections, res.num_repeats);
        printf("| %-13s | %8.2f s | %9.2f ms |", name, res.build_time, res.search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", res.recalls[i] * 100);
        }
        printf("\n");
    }

    printf("============================================================================================\n");
    printf("Dataset: %d docs, %ld total vectors, avg %.1f vectors/doc\n", num_docs, total_vectors,
           (float)total_vectors / num_docs);
    if (!SKIP_DIRECT_TEST) {
        printf("BruteForce Recall (vs official GT): ");
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], bf_recalls_vs_gt[i] * 100);
        }
        printf("\n");
    }
    printf("\nNote: R@K = Recall at top-K, per-query averaged, compared to BruteForce MaxSim\n");
    fflush(stdout);

    // Basic sanity checks
    if (!SKIP_DIRECT_TEST) {
        for (float r : direct_recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
    for (const auto& res : muvera_results) {
        for (float r : res.recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
}

TEST_CASE("MS MARCO ColBERT: Direct vs LEMUR", "[msmarco_emb_list_lemur]") {
    // Get data paths (can be overridden via MODEL_TAG or explicit env vars)
    const std::string docs_path = GetMsmarcoDocsPath();
    const std::string queries_path = GetMsmarcoQueriesPath();

    // Check if MS MARCO data files exist
    {
        std::ifstream docs_file(docs_path);
        std::ifstream queries_file(queries_path);

        if (!docs_file.good() || !queries_file.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("MS MARCO data files not found. Please prepare the data first.\n");
            printf("Expected files:\n");
            printf("  - %s\n", docs_path.c_str());
            printf("  - %s\n", queries_path.c_str());
            printf("\n");
            printf("Generate MS MARCO data with GT annotations:\n");
            printf("  python scripts/prepare_msmarco_with_gt.py --output-dir .\n");
            printf("Or specify MODEL_TAG: MODEL_TAG=bgem3 ./knowhere_tests\n");
            printf("=============================================================\n");
            SKIP("MS MARCO data files not found");
            return;
        }
    }

    // Load data
    printf("\n=== Loading MS MARCO Data (model_tag=%s) ===\n", GetModelTag().c_str());
    EmbListData doc_data;
    QueryDataWithGT query_data;

    REQUIRE(doc_data.LoadFromJsonl(docs_path, MAX_DOCS_TO_LOAD));
    doc_data.PrintStats();

    REQUIRE(query_data.LoadFromJsonl(queries_path, MAX_QUERIES_TO_LOAD));
    query_data.PrintStats();

    auto doc_ds = doc_data.ToDataSet();
    auto query_ds = query_data.ToDataSet();

    const int32_t dim = doc_data.dim;
    const int32_t num_docs = doc_data.num_docs;
    const int64_t total_vectors = doc_data.total_vectors;
    const int32_t num_queries = std::min((int32_t)query_data.num_queries, MAX_QUERIES_TO_LOAD);

    // Multiple topk values for evaluation
    const std::vector<int32_t> topk_values = {10, 20, 50};
    const int32_t max_topk = *std::max_element(topk_values.begin(), topk_values.end());

    printf("\n=== Test Configuration ===\n");
    printf("Documents: %d, Total vectors: %ld, Dim: %d\n", num_docs, total_vectors, dim);
    printf("Queries: %d, TopK values: ", num_queries);
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("%d%s", topk_values[i], i < topk_values.size() - 1 ? ", " : "\n");
    }
    fflush(stdout);

    // Base config
    knowhere::Json base_conf;
    base_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    base_conf[knowhere::meta::DIM] = dim;
    base_conf[knowhere::meta::TOPK] = max_topk;
    base_conf[knowhere::indexparam::HNSW_M] = 16;
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;
    base_conf[knowhere::indexparam::EF] = std::max(128, max_topk * 2);
    base_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 2.0f;

    auto version = GenTestEmbListVersionList();

    // Recall calculation vs BruteForce results (per-query average)
    // result_ids has result_k items per query, bf_ids has bf_k items per query
    auto calc_recall_vs_bf = [&](const int64_t* result_ids, int32_t result_k, const int64_t* bf_ids, int32_t bf_k,
                                 int32_t k) {
        if (bf_ids == nullptr)
            return 0.0f;
        float total_recall = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            std::unordered_set<int64_t> bf_set;
            int bf_count = 0;
            for (int i = 0; i < k && i < bf_k; ++i) {
                if (bf_ids[q * bf_k + i] >= 0) {
                    bf_set.insert(bf_ids[q * bf_k + i]);
                    bf_count++;
                }
            }
            if (bf_count == 0)
                continue;

            int overlap = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                if (result_ids[q * result_k + i] >= 0 && bf_set.count(result_ids[q * result_k + i]) > 0) {
                    overlap++;
                }
            }
            total_recall += (float)overlap / bf_count;
        }
        return total_recall / num_queries;
    };

    // ========== BruteForce MaxSim ==========
    // Compute BruteForce results for each k value separately
    printf("\n[BruteForce] Computing MaxSim results for each k...\n");
    fflush(stdout);

    std::map<int32_t, knowhere::DataSetPtr> bf_results;
    std::map<int32_t, const int64_t*> bf_ids_map;
    double bf_time = 0;

    for (int32_t k : topk_values) {
        knowhere::Json bf_conf;
        bf_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
        bf_conf[knowhere::meta::TOPK] = k;

        StopWatch sw_bf;
        auto bf_result = knowhere::BruteForce::Search<knowhere::fp32>(doc_ds, query_ds, bf_conf, nullptr);
        bf_time += sw_bf.elapsed();
        REQUIRE(bf_result.has_value());

        bf_results[k] = bf_result.value();
        bf_ids_map[k] = bf_results[k]->GetIds();
    }
    printf("[BruteForce] Total time: %.3f s\n", bf_time);
    fflush(stdout);

    // ========== Direct Strategy ==========
    double direct_build_time = 0, direct_search_time = 0;
    std::vector<float> direct_recalls(topk_values.size(), 0.0f);

    if (SKIP_DIRECT_TEST) {
        printf("\n[Direct] SKIPPED (SKIP_DIRECT_TEST = true)\n");
    } else {
        printf("\n[Direct] Building HNSW index for %ld vectors...\n", total_vectors);
        fflush(stdout);

        knowhere::Json direct_conf = base_conf;
        direct_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
        direct_conf["emb_list_strategy"] = "tokenann";

        auto direct_index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
        REQUIRE(direct_index.has_value());

        StopWatch sw_direct_build;
        auto direct_build_status = direct_index.value().Build(doc_ds, direct_conf);
        direct_build_time = sw_direct_build.elapsed();
        REQUIRE(direct_build_status == knowhere::Status::success);
        printf("[Direct] Build time: %.3f s\n", direct_build_time);

        printf("[Direct] Searching with each k...\n");
        fflush(stdout);

        for (size_t i = 0; i < topk_values.size(); ++i) {
            int32_t k = topk_values[i];
            knowhere::Json search_conf = direct_conf;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_search;
            auto direct_result = direct_index.value().Search(query_ds, search_conf, nullptr);
            direct_search_time += sw_search.elapsed();
            REQUIRE(direct_result.has_value());

            auto direct_ids = direct_result.value()->GetIds();
            direct_recalls[i] = calc_recall_vs_bf(direct_ids, k, bf_ids_map[k], k, k);
        }

        printf("[Direct] Search time: %.3f ms, Recall: ", direct_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], direct_recalls[i] * 100);
        }
        printf("\n");
        fflush(stdout);
    }

    // ========== LEMUR Strategy ==========
    // Build parameters
    const int32_t hidden_dim = 512;
    const int32_t num_layers = 2;
    const int32_t num_epochs = 30;
    const int32_t num_train_samples = 5000;

    // Search parameters to test (RETRIEVAL_ANN_RATIO)
    std::vector<float> ann_ratios = {1.3f, 1.5f, 2.0f};

    // Store results for each ratio
    struct LemurResult {
        float ann_ratio;
        double search_time;
        std::vector<float> recalls;
    };
    std::vector<LemurResult> lemur_results;
    double lemur_build_time = 0;

    printf("\n[LEMUR] Building index (h%d-l%d-e%d-s%d)...\n", hidden_dim, num_layers, num_epochs, num_train_samples);
    fflush(stdout);

    knowhere::Json lemur_conf = base_conf;
    lemur_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    lemur_conf["emb_list_strategy"] = "lemur";
    lemur_conf["lemur_hidden_dim"] = hidden_dim;
    lemur_conf["lemur_num_layers"] = num_layers;
    lemur_conf["lemur_num_epochs"] = num_epochs;
    lemur_conf["lemur_num_train_samples"] = num_train_samples;
    lemur_conf["lemur_batch_size"] = 256;
    lemur_conf["lemur_learning_rate"] = 0.001f;
    lemur_conf["lemur_seed"] = 42;
    lemur_conf["emb_list_rerank"] = true;

    auto lemur_index =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
    REQUIRE(lemur_index.has_value());

    StopWatch sw_lemur_build;
    auto lemur_build_status = lemur_index.value().Build(doc_ds, lemur_conf);
    lemur_build_time = sw_lemur_build.elapsed();
    REQUIRE(lemur_build_status == knowhere::Status::success);
    printf("[LEMUR] Build time: %.3f s\n", lemur_build_time);

    // Test different ANN ratios with each k value
    printf("\n[LEMUR] Testing %zu ANN ratios x %zu k values...\n", ann_ratios.size(), topk_values.size());
    fflush(stdout);

    for (float ann_ratio : ann_ratios) {
        std::vector<float> recalls;
        double total_search_time = 0;

        for (int32_t k : topk_values) {
            knowhere::Json search_conf = lemur_conf;
            search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_lemur_search;
            auto lemur_result = lemur_index.value().Search(query_ds, search_conf, nullptr);
            double search_time = sw_lemur_search.elapsed();
            total_search_time += search_time;
            REQUIRE(lemur_result.has_value());

            auto lemur_ids = lemur_result.value()->GetIds();
            float recall = calc_recall_vs_bf(lemur_ids, k, bf_ids_map[k], k, k);
            recalls.push_back(recall);
        }

        printf("[LEMUR-ratio%.1f] Search time: %.3f ms, Recall: ", ann_ratio, total_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], recalls[i] * 100);
        }
        printf("\n");
        fflush(stdout);

        lemur_results.push_back({ann_ratio, total_search_time, recalls});
    }

    // ========== Summary ==========
    printf("\n=====================================================================================================\n");
    printf("                            Summary: Direct vs LEMUR (MS MARCO)                                      \n");
    printf("=====================================================================================================\n");

    // Header with topk columns
    printf("| Strategy                | Build Time | Search Time |");
    for (int32_t k : topk_values) {
        printf(" R@%-3d |", k);
    }
    printf("\n");

    printf("|-------------------------|------------|-------------|");
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("-------|");
    }
    printf("\n");

    // BruteForce row
    printf("| BruteForce              | %10s | %9.2f s |", "-", bf_time);
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf(" %4.1f%% |", 100.0f);
    }
    printf("\n");

    // Direct row
    if (!SKIP_DIRECT_TEST) {
        printf("| Direct                  | %8.2f s | %9.2f ms |", direct_build_time, direct_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", direct_recalls[i] * 100);
        }
        printf("\n");
    }

    // LEMUR rows (different ANN ratios)
    for (const auto& res : lemur_results) {
        char name[32];
        snprintf(name, sizeof(name), "LEMUR (ratio=%.1f)", res.ann_ratio);
        // First row shows build time, others show "-"
        if (&res == &lemur_results[0]) {
            printf("| %-23s | %8.2f s | %9.2f ms |", name, lemur_build_time, res.search_time * 1000);
        } else {
            printf("| %-23s | %10s | %9.2f ms |", name, "-", res.search_time * 1000);
        }
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", res.recalls[i] * 100);
        }
        printf("\n");
    }

    printf("=====================================================================================================\n");
    printf("Dataset: %d docs, %ld total vectors, dim=%d, avg %.1f vectors/doc\n", num_docs, total_vectors, dim,
           (float)total_vectors / num_docs);
    printf("\nLEMUR Build Config: hidden_dim=%d, num_layers=%d, epochs=%d, train_samples=%d\n", hidden_dim, num_layers,
           num_epochs, num_train_samples);
    printf("RETRIEVAL_ANN_RATIO: Controls how many ANN candidates to retrieve (ratio * topk)\n");
    printf("\nNote: R@K = Recall at top-K, per-query averaged, compared to BruteForce MaxSim\n");
    fflush(stdout);

    // Basic sanity checks
    if (!SKIP_DIRECT_TEST) {
        for (float r : direct_recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
    for (const auto& res : lemur_results) {
        for (float r : res.recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
}

// ============================================================================
// Unified test for MUVERA vs LEMUR comparison with different ANN ratios
// ============================================================================
static void
RunMuveraLemurComparison(const std::string& dataset_name, const std::string& docs_path, const std::string& queries_path,
                         int32_t max_docs, int32_t max_queries, const std::vector<int32_t>& custom_topk = {},
                         const std::vector<int32_t>& custom_e2e_topk = {}) {
    // Check if data files exist
    {
        std::ifstream docs_file(docs_path);
        std::ifstream queries_file(queries_path);

        if (!docs_file.good() || !queries_file.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("%s data files not found. Please prepare the data first.\n", dataset_name.c_str());
            printf("Expected files:\n");
            printf("  - %s\n", docs_path.c_str());
            printf("  - %s\n", queries_path.c_str());
            printf("=============================================================\n");
            SKIP(dataset_name + " data files not found");
            return;
        }
    }

    // Get model tag for display
    const std::string model_tag = GetModelTag();

    // Load data
    printf("\n=== Loading %s Data (model=%s) ===\n", dataset_name.c_str(), model_tag.c_str());
    EmbListData doc_data;
    QueryDataWithGT query_data;

    REQUIRE(doc_data.LoadFromJsonl(docs_path, max_docs));
    doc_data.PrintStats();

    REQUIRE(query_data.LoadFromJsonl(queries_path, max_queries));
    query_data.PrintStats();

    auto doc_ds = doc_data.ToDataSet();
    auto query_ds = query_data.ToDataSet();

    const int32_t dim = doc_data.dim;
    const int32_t num_docs = doc_data.num_docs;
    const int64_t total_vectors = doc_data.total_vectors;
    const int32_t num_queries = std::min((int32_t)query_data.num_queries, max_queries);

    // Multiple topk values for evaluation
    const std::vector<int32_t> topk_values = custom_topk.empty() ? std::vector<int32_t>{50, 100} : custom_topk;
    const std::vector<int32_t> e2e_topk_values = custom_e2e_topk.empty() ? std::vector<int32_t>{100} : custom_e2e_topk;
    const int32_t max_topk = *std::max_element(topk_values.begin(), topk_values.end());

    // ANN ratios to test
    const std::vector<float> ann_ratios = {3.0f, 5.0f};

    printf("\n=== Test Configuration ===\n");
    printf("Dataset: %s, Model: %s\n", dataset_name.c_str(), model_tag.c_str());
    printf("Documents: %d, Total vectors: %ld, Dim: %d\n", num_docs, total_vectors, dim);
    printf("Queries: %d, TopK values: ", num_queries);
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("%d%s", topk_values[i], i < topk_values.size() - 1 ? ", " : "\n");
    }
    printf("ANN Ratios to test: ");
    for (size_t i = 0; i < ann_ratios.size(); ++i) {
        printf("%.1f%s", ann_ratios[i], i < ann_ratios.size() - 1 ? ", " : "\n");
    }
    fflush(stdout);

    // Base config
    knowhere::Json base_conf;
    base_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    base_conf[knowhere::meta::DIM] = dim;
    base_conf[knowhere::meta::TOPK] = max_topk;
    base_conf[knowhere::indexparam::HNSW_M] = 16;
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;
    base_conf[knowhere::indexparam::EF] = std::max(128, max_topk * 2);

    auto version = GenTestEmbListVersionList();

    // Recall calculation vs BruteForce results
    // result_ids has result_k items per query (from searching with TOPK=result_k)
    // bf_ids has bf_k items per query (from searching with TOPK=bf_k)
    auto calc_recall_vs_bf = [&](const int64_t* result_ids, int32_t result_k, const int64_t* bf_ids, int32_t bf_k,
                                 int32_t k) {
        if (bf_ids == nullptr)
            return 0.0f;
        float total_recall = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            std::unordered_set<int64_t> bf_set;
            int bf_count = 0;
            for (int i = 0; i < k && i < bf_k; ++i) {
                if (bf_ids[q * bf_k + i] >= 0) {
                    bf_set.insert(bf_ids[q * bf_k + i]);
                    bf_count++;
                }
            }
            if (bf_count == 0)
                continue;

            int overlap = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                if (result_ids[q * result_k + i] >= 0 && bf_set.count(result_ids[q * result_k + i]) > 0) {
                    overlap++;
                }
            }
            total_recall += (float)overlap / bf_count;
        }
        return total_recall / num_queries;
    };

    // E2E Recall calculation vs Ground Truth
    // Recall@k = (number of GT documents found in top-k results) / (total GT documents)
    auto calc_recall_vs_gt = [&](const int64_t* result_ids, int32_t result_k, int32_t k) {
        float total_recall = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& gt = query_data.gt_pids[q];
            if (gt.empty())
                continue;

            std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
            int found = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                if (doc_id >= 0 && gt_set.count(doc_id) > 0) {
                    found++;
                }
            }
            total_recall += (float)found / gt.size();
            valid_queries++;
        }
        return valid_queries > 0 ? total_recall / valid_queries : 0.0f;
    };

    // nDCG@k calculation vs GT (per-query average, supports graded relevance)
    auto calc_ndcg_vs_gt = [&](const int64_t* result_ids, int32_t result_k, int32_t k) {
        float total_ndcg = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& rels = query_data.gt_rels[q];
            if (rels.empty())
                continue;
            double dcg = 0.0;
            for (int i = 0; i < k && i < result_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                auto it = rels.find(doc_id);
                if (doc_id >= 0 && it != rels.end()) {
                    dcg += (std::pow(2.0, it->second) - 1.0) / std::log2(i + 2.0);
                }
            }
            std::vector<int> sorted_rels;
            sorted_rels.reserve(rels.size());
            for (const auto& [id, rel] : rels) {
                sorted_rels.push_back(rel);
            }
            std::sort(sorted_rels.rbegin(), sorted_rels.rend());
            double idcg = 0.0;
            int ideal_count = std::min((int)sorted_rels.size(), k);
            for (int i = 0; i < ideal_count; ++i) {
                idcg += (std::pow(2.0, sorted_rels[i]) - 1.0) / std::log2(i + 2.0);
            }
            total_ndcg += idcg > 0 ? (float)(dcg / idcg) : 0.0f;
            valid_queries++;
        }
        return valid_queries > 0 ? total_ndcg / valid_queries : 0.0f;
    };

    // MRR@k calculation vs GT (per-query average)
    auto calc_mrr_vs_gt = [&](const int64_t* result_ids, int32_t result_k, int32_t k) {
        float total_rr = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& gt = query_data.gt_pids[q];
            if (gt.empty())
                continue;
            std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
            for (int i = 0; i < k && i < result_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                if (doc_id >= 0 && gt_set.count(doc_id) > 0) {
                    total_rr += 1.0f / (i + 1);
                    break;
                }
            }
            valid_queries++;
        }
        return valid_queries > 0 ? total_rr / valid_queries : 0.0f;
    };

    // ========== BruteForce MaxSim ==========
    // Compute BruteForce results for each k value separately
    printf("\n[BruteForce] Computing MaxSim results for each k...\n");
    fflush(stdout);

    std::map<int32_t, knowhere::DataSetPtr> bf_results;
    std::map<int32_t, const int64_t*> bf_ids_map;
    double bf_time = 0;

    // Compute for both topk_values, e2e_topk_values, and k=10 (for nDCG@10/MRR@10)
    std::set<int32_t> all_k_values(topk_values.begin(), topk_values.end());
    all_k_values.insert(e2e_topk_values.begin(), e2e_topk_values.end());
    all_k_values.insert(10);  // Always need k=10 for nDCG@10 and MRR@10

    for (int32_t k : all_k_values) {
        knowhere::Json bf_conf;
        bf_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
        bf_conf[knowhere::meta::TOPK] = k;

        StopWatch sw_bf;
        auto bf_result = knowhere::BruteForce::Search<knowhere::fp32>(doc_ds, query_ds, bf_conf, nullptr);
        bf_time += sw_bf.elapsed();
        REQUIRE(bf_result.has_value());

        bf_results[k] = bf_result.value();
        bf_ids_map[k] = bf_results[k]->GetIds();
    }
    printf("[BruteForce] Total time: %.3f s\n", bf_time);
    fflush(stdout);

    // Result structure
    struct SearchResult {
        std::string name;
        float ann_ratio;
        double build_time;
        double search_time;
        std::vector<float> recalls;          // vs BruteForce
        std::vector<double> math_latencies;  // avg latency per query for each Math Recall topk (ms)
        std::vector<float> e2e_recalls;      // vs Ground Truth
        float ndcg10 = 0.0f;                 // nDCG@10 vs Ground Truth
        float mrr10 = 0.0f;                  // MRR@10 vs Ground Truth
        std::vector<double> e2e_latencies;   // avg latency per query for each E2E topk (ms)
        float math_ndcg10 = 0.0f;            // nDCG@10 vs BruteForce (top-3=rel2, 4-10=rel1)
        float math_mrr10 = 0.0f;             // MRR@10 vs BruteForce (rank-1 doc)
    };
    std::vector<SearchResult> all_results;

    // BruteForce E2E metrics
    std::vector<float> bf_e2e_recalls;
    printf("[BruteForce] E2E Recall (vs GT): ");
    for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
        int32_t k = e2e_topk_values[i];
        float recall = calc_recall_vs_gt(bf_ids_map[k], k, k);
        bf_e2e_recalls.push_back(recall);
        printf("@%d=%.4f%s", k, recall, i < e2e_topk_values.size() - 1 ? ", " : "\n");
    }
    float bf_ndcg10 = calc_ndcg_vs_gt(bf_ids_map[10], 10, 10);
    float bf_mrr10 = calc_mrr_vs_gt(bf_ids_map[10], 10, 10);
    printf("[BruteForce] nDCG@10=%.4f, MRR@10=%.4f\n", bf_ndcg10, bf_mrr10);
    fflush(stdout);

    // Build BF-based relevance for Math nDCG@10 and Math MRR@10
    // BF top-3 → rel=2, BF rank 4-10 → rel=1
    std::vector<std::map<int64_t, int>> bf_math_rels(num_queries);
    std::vector<int64_t> bf_rank1(num_queries, -1);
    {
        const int64_t* bf_k10 = bf_ids_map[10];
        for (int q = 0; q < num_queries; ++q) {
            for (int i = 0; i < 10; ++i) {
                int64_t doc_id = bf_k10[q * 10 + i];
                if (doc_id >= 0) {
                    bf_math_rels[q][doc_id] = (i < 3) ? 2 : 1;
                }
            }
            bf_rank1[q] = bf_k10[q * 10];
        }
    }

    // Math nDCG@10 (approximate vs BruteForce, graded: top-3=rel2, rest=rel1)
    auto calc_ndcg_vs_bf = [&](const int64_t* result_ids, int32_t result_k) {
        float total_ndcg = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            const auto& rels = bf_math_rels[q];
            if (rels.empty())
                continue;
            double dcg = 0.0;
            int eval_k = std::min(10, result_k);
            for (int i = 0; i < eval_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                auto it = rels.find(doc_id);
                if (doc_id >= 0 && it != rels.end()) {
                    dcg += (std::pow(2.0, it->second) - 1.0) / std::log2(i + 2.0);
                }
            }
            std::vector<int> sorted_rels;
            for (const auto& [id, rel] : rels) {
                sorted_rels.push_back(rel);
            }
            std::sort(sorted_rels.rbegin(), sorted_rels.rend());
            double idcg = 0.0;
            int ideal_count = std::min((int)sorted_rels.size(), 10);
            for (int i = 0; i < ideal_count; ++i) {
                idcg += (std::pow(2.0, sorted_rels[i]) - 1.0) / std::log2(i + 2.0);
            }
            total_ndcg += idcg > 0 ? (float)(dcg / idcg) : 0.0f;
        }
        return num_queries > 0 ? total_ndcg / num_queries : 0.0f;
    };

    // Math MRR@10 (BF rank-1 doc position in approximate results)
    auto calc_mrr_vs_bf = [&](const int64_t* result_ids, int32_t result_k) {
        float total_rr = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            int64_t target = bf_rank1[q];
            if (target < 0)
                continue;
            int eval_k = std::min(10, result_k);
            for (int i = 0; i < eval_k; ++i) {
                if (result_ids[q * result_k + i] == target) {
                    total_rr += 1.0f / (i + 1);
                    break;
                }
            }
        }
        return num_queries > 0 ? total_rr / num_queries : 0.0f;
    };

    // ========== Direct Strategy ==========
    if (!SKIP_DIRECT_TEST) {
        printf("\n[Direct] Building HNSW index for %ld vectors...\n", total_vectors);
        fflush(stdout);

        knowhere::Json direct_conf = base_conf;
        direct_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
        direct_conf["emb_list_strategy"] = "tokenann";

        auto direct_index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
        REQUIRE(direct_index.has_value());

        StopWatch sw_direct_build;
        auto direct_build_status = direct_index.value().Build(doc_ds, direct_conf);
        double direct_build_time = sw_direct_build.elapsed();
        REQUIRE(direct_build_status == knowhere::Status::success);
        printf("[Direct] Build time: %.3f s\n", direct_build_time);

        // Test Direct with different ANN ratios and k values
        printf("\n[Direct] Testing %zu ANN ratios x %zu k values...\n", ann_ratios.size(), topk_values.size());
        fflush(stdout);

        for (size_t idx = 0; idx < ann_ratios.size(); ++idx) {
            float ann_ratio = ann_ratios[idx];

            std::vector<float> recalls;
            std::vector<double> math_latencies;
            std::vector<float> e2e_recalls;
            double total_search_time = 0;

            // Search for Math Recall (vs BF) with individual latency (avg per query)
            for (int32_t k : topk_values) {
                knowhere::Json search_conf = direct_conf;
                search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                search_conf[knowhere::meta::TOPK] = k;

                StopWatch sw_search;
                auto result = direct_index.value().Search(query_ds, search_conf, nullptr);
                double search_time = sw_search.elapsed();
                total_search_time += search_time;
                REQUIRE(result.has_value());

                auto result_ids = result.value()->GetIds();
                float recall = calc_recall_vs_bf(result_ids, k, bf_ids_map[k], k, k);
                recalls.push_back(recall);
                math_latencies.push_back(search_time * 1000 / num_queries);  // ms per query
            }

            // Dedicated k=10 search for Math nDCG@10/MRR@10
            float math_ndcg10 = 0.0f, math_mrr10 = 0.0f;
            {
                knowhere::Json search_conf = direct_conf;
                search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                search_conf[knowhere::meta::TOPK] = 10;
                auto result = direct_index.value().Search(query_ds, search_conf, nullptr);
                REQUIRE(result.has_value());
                auto result_ids = result.value()->GetIds();
                math_ndcg10 = calc_ndcg_vs_bf(result_ids, 10);
                math_mrr10 = calc_mrr_vs_bf(result_ids, 10);
            }

            // Search for E2E Recall (vs GT) with individual latency (avg per query)
            std::vector<double> e2e_latencies;
            for (int32_t k : e2e_topk_values) {
                knowhere::Json search_conf = direct_conf;
                search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                search_conf[knowhere::meta::TOPK] = k;

                StopWatch sw_e2e;
                auto result = direct_index.value().Search(query_ds, search_conf, nullptr);
                double e2e_latency = sw_e2e.elapsed() * 1000 / num_queries;  // ms per query
                REQUIRE(result.has_value());

                auto result_ids = result.value()->GetIds();
                e2e_recalls.push_back(calc_recall_vs_gt(result_ids, k, k));
                e2e_latencies.push_back(e2e_latency);
            }

            // Separate k=10 search for nDCG@10 and MRR@10
            float ndcg10 = 0.0f, mrr10 = 0.0f;
            {
                knowhere::Json search_conf = direct_conf;
                search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                search_conf[knowhere::meta::TOPK] = 10;
                auto result = direct_index.value().Search(query_ds, search_conf, nullptr);
                REQUIRE(result.has_value());
                auto result_ids = result.value()->GetIds();
                ndcg10 = calc_ndcg_vs_gt(result_ids, 10, 10);
                mrr10 = calc_mrr_vs_gt(result_ids, 10, 10);
            }

            printf("[Direct-ratio%.1f] Recall: ", ann_ratio);
            for (size_t i = 0; i < topk_values.size(); ++i) {
                printf("@%d=%.1f%% (%.2fms) ", topk_values[i], recalls[i] * 100, math_latencies[i]);
            }
            printf("  Math nDCG@10=%.4f MRR@10=%.4f\n", math_ndcg10, math_mrr10);
            printf("                  E2E Recall: ");
            for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
                printf("@%d=%.1f%% (%.2fms) ", e2e_topk_values[i], e2e_recalls[i] * 100, e2e_latencies[i]);
            }
            printf("  nDCG@10=%.4f MRR@10=%.4f\n", ndcg10, mrr10);
            fflush(stdout);

            char name[32];
            snprintf(name, sizeof(name), "Direct (ratio=%.1f)", ann_ratio);
            all_results.push_back({name, ann_ratio, idx == 0 ? direct_build_time : 0, total_search_time, recalls,
                                   math_latencies, e2e_recalls, ndcg10, mrr10, e2e_latencies, math_ndcg10, math_mrr10});
        }
    } else {
        printf("\n[Direct] SKIPPED (SKIP_DIRECT_TEST = true)\n");
        fflush(stdout);
    }

    // ========== MUVERA Strategy ==========
    if (!SKIP_MUVERA_TEST) {
        // Multiple MUVERA parameter combinations: (num_projections, num_repeats)
        std::vector<std::pair<int32_t, int32_t>> muvera_params = {{3, 7}, {4, 7}, {5, 7}};

        printf("\n[MUVERA] Testing %zu parameter combinations...\n", muvera_params.size());
        fflush(stdout);

        for (const auto& muvera_param : muvera_params) {
            int32_t num_proj = muvera_param.first;
            int32_t num_rep = muvera_param.second;

            printf("\n[MUVERA-%d-%d] Building index...\n", num_proj, num_rep);
            fflush(stdout);

            knowhere::Json muvera_conf = base_conf;
            muvera_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
            muvera_conf["emb_list_strategy"] = "muvera";
            muvera_conf["muvera_num_projections"] = num_proj;
            muvera_conf["muvera_num_repeats"] = num_rep;
            muvera_conf["muvera_seed"] = 42;

            auto muvera_index =
                knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
            REQUIRE(muvera_index.has_value());

            StopWatch sw_muvera_build;
            auto muvera_build_status = muvera_index.value().Build(doc_ds, muvera_conf);
            double muvera_build_time = sw_muvera_build.elapsed();
            REQUIRE(muvera_build_status == knowhere::Status::success);
            printf("[MUVERA-%d-%d] Build time: %.3f s\n", num_proj, num_rep, muvera_build_time);

            // Test with different ANN ratios and k values
            for (size_t idx = 0; idx < ann_ratios.size(); ++idx) {
                float ann_ratio = ann_ratios[idx];

                std::vector<float> recalls;
                std::vector<double> math_latencies;
                std::vector<float> e2e_recalls;
                double total_search_time = 0;

                // Search for Math Recall (vs BF) with individual latency (avg per query)
                for (int32_t k : topk_values) {
                    knowhere::Json search_conf = muvera_conf;
                    search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                    search_conf[knowhere::meta::TOPK] = k;

                    StopWatch sw_search;
                    auto result = muvera_index.value().Search(query_ds, search_conf, nullptr);
                    double search_time = sw_search.elapsed();
                    total_search_time += search_time;
                    REQUIRE(result.has_value());

                    auto result_ids = result.value()->GetIds();
                    float recall = calc_recall_vs_bf(result_ids, k, bf_ids_map[k], k, k);
                    recalls.push_back(recall);
                    math_latencies.push_back(search_time * 1000 / num_queries);  // ms per query
                }

                // Dedicated k=10 search for Math nDCG@10/MRR@10
                float math_ndcg10 = 0.0f, math_mrr10 = 0.0f;
                {
                    knowhere::Json search_conf = muvera_conf;
                    search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                    search_conf[knowhere::meta::TOPK] = 10;
                    auto result = muvera_index.value().Search(query_ds, search_conf, nullptr);
                    REQUIRE(result.has_value());
                    auto result_ids = result.value()->GetIds();
                    math_ndcg10 = calc_ndcg_vs_bf(result_ids, 10);
                    math_mrr10 = calc_mrr_vs_bf(result_ids, 10);
                }

                // Search for E2E Recall (vs GT) with individual latency (avg per query)
                std::vector<double> e2e_latencies;
                for (int32_t k : e2e_topk_values) {
                    knowhere::Json search_conf = muvera_conf;
                    search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                    search_conf[knowhere::meta::TOPK] = k;

                    StopWatch sw_e2e;
                    auto result = muvera_index.value().Search(query_ds, search_conf, nullptr);
                    double e2e_latency = sw_e2e.elapsed() * 1000 / num_queries;  // ms per query
                    REQUIRE(result.has_value());

                    auto result_ids = result.value()->GetIds();
                    e2e_recalls.push_back(calc_recall_vs_gt(result_ids, k, k));
                    e2e_latencies.push_back(e2e_latency);
                }

                // Separate k=10 search for nDCG@10 and MRR@10
                float ndcg10 = 0.0f, mrr10 = 0.0f;
                {
                    knowhere::Json search_conf = muvera_conf;
                    search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                    search_conf[knowhere::meta::TOPK] = 10;
                    auto result = muvera_index.value().Search(query_ds, search_conf, nullptr);
                    REQUIRE(result.has_value());
                    auto result_ids = result.value()->GetIds();
                    ndcg10 = calc_ndcg_vs_gt(result_ids, 10, 10);
                    mrr10 = calc_mrr_vs_gt(result_ids, 10, 10);
                }

                printf("[MUVERA-%d-%d-r%.1f] Recall: ", num_proj, num_rep, ann_ratio);
                for (size_t i = 0; i < topk_values.size(); ++i) {
                    printf("@%d=%.1f%% (%.2fms) ", topk_values[i], recalls[i] * 100, math_latencies[i]);
                }
                printf("  Math nDCG@10=%.4f MRR@10=%.4f\n", math_ndcg10, math_mrr10);
                printf("                    E2E Recall: ");
                for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
                    printf("@%d=%.1f%% (%.2fms) ", e2e_topk_values[i], e2e_recalls[i] * 100, e2e_latencies[i]);
                }
                printf("  nDCG@10=%.4f MRR@10=%.4f\n", ndcg10, mrr10);
                fflush(stdout);

                char name[48];
                snprintf(name, sizeof(name), "MUVERA-%d-%d (r=%.1f)", num_proj, num_rep, ann_ratio);
                all_results.push_back({name, ann_ratio, idx == 0 ? muvera_build_time : 0, total_search_time, recalls,
                                       math_latencies, e2e_recalls, ndcg10, mrr10, e2e_latencies, math_ndcg10,
                                       math_mrr10});
            }
        }
    } else {
        printf("\n[MUVERA] SKIPPED (SKIP_MUVERA_TEST = true)\n");
        fflush(stdout);
    }

    // ========== LEMUR Strategy ==========
    const int32_t hidden_dim = 512;
    const int32_t num_layers = 2;
    const int32_t num_epochs = 30;
    const int32_t num_train_samples = 10000;

    printf("\n[LEMUR] Building index (h%d-l%d-e%d-s%d)...\n", hidden_dim, num_layers, num_epochs, num_train_samples);
    fflush(stdout);

    knowhere::Json lemur_conf = base_conf;
    lemur_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    lemur_conf["emb_list_strategy"] = "lemur";
    lemur_conf["lemur_hidden_dim"] = hidden_dim;
    lemur_conf["lemur_num_layers"] = num_layers;
    lemur_conf["lemur_num_epochs"] = num_epochs;
    lemur_conf["lemur_num_train_samples"] = num_train_samples;
    lemur_conf["lemur_batch_size"] = 512;
    lemur_conf["lemur_learning_rate"] = 0.001f;
    lemur_conf["lemur_seed"] = 42;
    lemur_conf["emb_list_rerank"] = true;

    auto lemur_index =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
    REQUIRE(lemur_index.has_value());

    StopWatch sw_lemur_build;
    auto lemur_build_status = lemur_index.value().Build(doc_ds, lemur_conf);
    double lemur_build_time = sw_lemur_build.elapsed();
    REQUIRE(lemur_build_status == knowhere::Status::success);
    printf("[LEMUR] Build time: %.3f s\n", lemur_build_time);

    // Test LEMUR with different ANN ratios and k values
    printf("\n[LEMUR] Testing %zu ANN ratios x %zu k values...\n", ann_ratios.size(), topk_values.size());
    fflush(stdout);

    for (size_t idx = 0; idx < ann_ratios.size(); ++idx) {
        float ann_ratio = ann_ratios[idx];

        std::vector<float> recalls;
        std::vector<double> math_latencies;
        std::vector<float> e2e_recalls;
        double total_search_time = 0;

        // Search for Math Recall (vs BF) with individual latency (avg per query)
        for (int32_t k : topk_values) {
            knowhere::Json search_conf = lemur_conf;
            search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_search;
            auto result = lemur_index.value().Search(query_ds, search_conf, nullptr);
            double search_time = sw_search.elapsed();
            total_search_time += search_time;
            REQUIRE(result.has_value());

            auto result_ids = result.value()->GetIds();
            float recall = calc_recall_vs_bf(result_ids, k, bf_ids_map[k], k, k);
            recalls.push_back(recall);
            math_latencies.push_back(search_time * 1000 / num_queries);  // ms per query
        }

        // Dedicated k=10 search for Math nDCG@10/MRR@10
        float math_ndcg10 = 0.0f, math_mrr10 = 0.0f;
        {
            knowhere::Json search_conf = lemur_conf;
            search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
            search_conf[knowhere::meta::TOPK] = 10;
            auto result = lemur_index.value().Search(query_ds, search_conf, nullptr);
            REQUIRE(result.has_value());
            auto result_ids = result.value()->GetIds();
            math_ndcg10 = calc_ndcg_vs_bf(result_ids, 10);
            math_mrr10 = calc_mrr_vs_bf(result_ids, 10);
        }

        // Search for E2E Recall (vs GT) with individual latency (avg per query)
        std::vector<double> e2e_latencies;
        for (int32_t k : e2e_topk_values) {
            knowhere::Json search_conf = lemur_conf;
            search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_e2e;
            auto result = lemur_index.value().Search(query_ds, search_conf, nullptr);
            double e2e_latency = sw_e2e.elapsed() * 1000 / num_queries;  // ms per query
            REQUIRE(result.has_value());

            auto result_ids = result.value()->GetIds();
            e2e_recalls.push_back(calc_recall_vs_gt(result_ids, k, k));
            e2e_latencies.push_back(e2e_latency);
        }

        // Separate k=10 search for nDCG@10 and MRR@10
        float ndcg10 = 0.0f, mrr10 = 0.0f;
        {
            knowhere::Json search_conf = lemur_conf;
            search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
            search_conf[knowhere::meta::TOPK] = 10;
            auto result = lemur_index.value().Search(query_ds, search_conf, nullptr);
            REQUIRE(result.has_value());
            auto result_ids = result.value()->GetIds();
            ndcg10 = calc_ndcg_vs_gt(result_ids, 10, 10);
            mrr10 = calc_mrr_vs_gt(result_ids, 10, 10);
        }

        printf("[LEMUR-ratio%.1f] Recall: ", ann_ratio);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% (%.2fms) ", topk_values[i], recalls[i] * 100, math_latencies[i]);
        }
        printf("  Math nDCG@10=%.4f MRR@10=%.4f\n", math_ndcg10, math_mrr10);
        printf("                 E2E Recall: ");
        for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
            printf("@%d=%.1f%% (%.2fms) ", e2e_topk_values[i], e2e_recalls[i] * 100, e2e_latencies[i]);
        }
        printf("  nDCG@10=%.4f MRR@10=%.4f\n", ndcg10, mrr10);
        fflush(stdout);

        char name[32];
        snprintf(name, sizeof(name), "LEMUR (ratio=%.1f)", ann_ratio);
        all_results.push_back({name, ann_ratio, idx == 0 ? lemur_build_time : 0, total_search_time, recalls,
                               math_latencies, e2e_recalls, ndcg10, mrr10, e2e_latencies, math_ndcg10, math_mrr10});
    }

    // ========== Summary ==========
    printf("\n");
    // Calculate separator length: 25 (strategy) + 12 (build time) + (8 + 9) * topk_values.size()
    int math_separator_len = 25 + 12 + (8 + 9) * (int)topk_values.size();
    for (int i = 0; i < math_separator_len; ++i) printf("=");
    printf("\n");
    printf("                              %s [%s]: Math Recall (with Latency)                                      \n",
           dataset_name.c_str(), model_tag.c_str());
    for (int i = 0; i < math_separator_len; ++i) printf("=");
    printf("\n");

    // Header with R@K and Latency@K pairs + nDCG@10 + MRR@10
    printf("| Strategy                | Build Time |");
    for (int32_t k : topk_values) {
        printf(" R@%-3d | Lat@%-2d |", k, k);
    }
    printf(" nDCG@10 | MRR@10 |\n");

    printf("|-------------------------|------------|");
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("-------|--------|");
    }
    printf("---------|--------|\n");

    // BruteForce row (no individual latency for BF)
    printf("| BruteForce              | %10s |", "-");
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf(" %4.1f%% |      - |", 100.0f);
    }
    printf("  1.000  | 1.000  |\n");

    // All results with individual latencies
    for (const auto& res : all_results) {
        if (res.build_time > 0) {
            printf("| %-23s | %8.2f s |", res.name.c_str(), res.build_time);
        } else {
            printf("| %-23s | %10s |", res.name.c_str(), "-");
        }
        for (size_t i = 0; i < topk_values.size(); ++i) {
            if (i < res.math_latencies.size()) {
                printf(" %4.1f%% | %6.2fms |", res.recalls[i] * 100, res.math_latencies[i]);
            } else {
                printf(" %4.1f%% |      - |", res.recalls[i] * 100);
            }
        }
        printf("  %5.3f  | %5.3f  |\n", res.math_ndcg10, res.math_mrr10);
    }

    for (int i = 0; i < math_separator_len; ++i) printf("=");
    printf("\n");
    printf(
        "Note: R@K = Recall at top-K, per-query averaged, compared to BruteForce MaxSim, Lat@K = Avg latency per query "
        "(ms)\n");
    printf("      nDCG@10/MRR@10 = vs BruteForce (top-3=rel2, rank 4-10=rel1)\n\n");

    // ========== E2E Recall/nDCG@10/MRR@10 Summary ==========
    printf("\n");
    printf(
        "=============================================================================================================="
        "============================\n");
    printf(
        "                              %s [%s]: E2E Metrics                                                        \n",
        dataset_name.c_str(), model_tag.c_str());
    printf(
        "=============================================================================================================="
        "============================\n");

    // Header with R@K pairs + nDCG@10 + MRR@10
    printf("| Strategy                | Build Time |");
    for (int32_t k : e2e_topk_values) {
        printf(" R@%-3d | Lat@%-2d |", k, k);
    }
    printf(" nDCG@10 | MRR@10 |\n");

    printf("|-------------------------|------------|");
    for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
        printf("-------|--------|");
    }
    printf("---------|--------|\n");

    // BruteForce row
    printf("| BruteForce              | %10s |", "-");
    for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
        printf(" %4.1f%% |      - |", bf_e2e_recalls[i] * 100);
    }
    printf("  %5.3f  | %5.3f  |\n", bf_ndcg10, bf_mrr10);

    // All results
    for (const auto& res : all_results) {
        if (res.build_time > 0) {
            printf("| %-23s | %8.2f s |", res.name.c_str(), res.build_time);
        } else {
            printf("| %-23s | %10s |", res.name.c_str(), "-");
        }
        for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
            if (i < res.e2e_latencies.size()) {
                printf(" %4.1f%% | %6.2fms |", res.e2e_recalls[i] * 100, res.e2e_latencies[i]);
            } else {
                printf(" %4.1f%% |      - |", res.e2e_recalls[i] * 100);
            }
        }
        printf("  %5.3f  | %5.3f  |\n", res.ndcg10, res.mrr10);
    }

    printf(
        "=============================================================================================================="
        "============================\n");
    printf(
        "Note: R@K = E2E Recall (found GT / total GT), nDCG@10/MRR@10 = fixed at top-10, Lat@K = Avg latency/query "
        "(ms)\n\n");

    // ========== Dataset Info ==========
    printf("Dataset: %s, Model: %s\n", dataset_name.c_str(), model_tag.c_str());
    printf("  %d docs, %ld total vectors, dim=%d, avg %.1f vectors/doc\n", num_docs, total_vectors, dim,
           (float)total_vectors / num_docs);
    if (!SKIP_DIRECT_TEST) {
        printf("Direct Config: HNSW index on all token vectors\n");
    }
    if (!SKIP_MUVERA_TEST) {
        printf("MUVERA Config: tested (proj, rep) = (3,7), (4,7), (5,7)\n");
    }
    printf("LEMUR Config: hidden_dim=%d, num_layers=%d, epochs=%d, train_samples=%d\n", hidden_dim, num_layers,
           num_epochs, num_train_samples);
    fflush(stdout);

    // Basic sanity checks
    for (const auto& res : all_results) {
        for (float r : res.recalls) {
            REQUIRE(r >= 0.0f);
        }
        for (float r : res.e2e_recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
}

TEST_CASE("LoTTE: Direct vs MUVERA vs LEMUR", "[lotte_emb_list_all]") {
    RunMuveraLemurComparison("LoTTE", GetLotteDocsPath(), GetLotteQueriesPath(), MAX_DOCS_TO_LOAD, MAX_QUERIES_TO_LOAD);
}

TEST_CASE("MS MARCO: Direct vs MUVERA vs LEMUR", "[msmarco_emb_list_all]") {
    RunMuveraLemurComparison("MS MARCO", GetMsmarcoDocsPath(), GetMsmarcoQueriesPath(), MAX_DOCS_TO_LOAD,
                             MAX_QUERIES_TO_LOAD);
}

TEST_CASE("SciFact: Direct vs MUVERA vs LEMUR", "[scifact_emb_list_all]") {
    RunMuveraLemurComparison("SciFact", GetScifactDocsPath(), GetScifactQueriesPath(), MAX_DOCS_TO_LOAD,
                             MAX_QUERIES_TO_LOAD);
}

TEST_CASE("TREC-COVID: Direct vs MUVERA vs LEMUR", "[trec_covid_emb_list_all]") {
    RunMuveraLemurComparison("TREC-COVID", GetTrecCovidDocsPath(), GetTrecCovidQueriesPath(), 5000,
                             MAX_QUERIES_TO_LOAD);
}

TEST_CASE("DocVQA: Direct vs MUVERA vs LEMUR", "[docvqa_emb_list_all]") {
    // Small dataset (500 docs), only test recall@10 to avoid near-brute-force behavior
    RunMuveraLemurComparison("DocVQA", GetDocvqaDocsPath(), GetDocvqaQueriesPath(), 500, MAX_QUERIES_TO_LOAD, {10},
                             {10});
}
