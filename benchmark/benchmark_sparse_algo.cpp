// Sparse Search Algorithm Benchmark using actual Knowhere library
// Compares: TAAT_NAIVE, DAAT_WAND, DAAT_WAND+BMW, DAAT_MAXSCORE

#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/sparse_utils.h"
#include "knowhere/version.h"

// Generate realistic text-like sparse data with Zipfian term distribution
// This better simulates scenarios where BMW excels:
// 1. Some terms are rare (high IDF) and appear in few docs with high TF
// 2. Some terms are common (low IDF) and appear in many docs with low TF
// 3. High scores concentrate in specific documents per term
knowhere::DataSetPtr
GenZipfianDataSet(int32_t rows, int32_t cols, float avg_terms_per_doc, float zipf_exp, int seed = 42) {
    std::mt19937 rng(seed);
    auto uniform = std::uniform_real_distribution<float>(0, 1);

    // Generate Zipfian term frequencies (how many docs each term appears in)
    std::vector<float> term_doc_freq(cols);
    float sum = 0;
    for (int32_t c = 0; c < cols; ++c) {
        term_doc_freq[c] = 1.0f / std::pow(c + 1, zipf_exp);
        sum += term_doc_freq[c];
    }
    // Normalize so average doc has avg_terms_per_doc terms
    float scale = avg_terms_per_doc * rows / (sum * rows);
    for (int32_t c = 0; c < cols; ++c) {
        term_doc_freq[c] = std::min(term_doc_freq[c] * scale, 1.0f);
    }

    std::vector<std::map<int32_t, float>> data(rows);

    for (int32_t col = 0; col < cols; ++col) {
        float prob = term_doc_freq[col];
        // Determine how many docs contain this term
        int32_t num_docs = static_cast<int32_t>(prob * rows);
        if (num_docs == 0 && uniform(rng) < prob * rows)
            num_docs = 1;

        // Randomly select which docs contain this term
        std::vector<int32_t> doc_ids(rows);
        std::iota(doc_ids.begin(), doc_ids.end(), 0);
        std::shuffle(doc_ids.begin(), doc_ids.end(), rng);

        for (int32_t i = 0; i < num_docs && i < rows; ++i) {
            int32_t doc_id = doc_ids[i];
            // Generate TF with exponential distribution (few high, many low)
            float tf = -std::log(1.0f - uniform(rng) * 0.9999f);
            // Scale TF: rare terms (low col index) get higher max TF
            float max_tf = 10.0f / std::pow(col + 1, 0.3f);
            tf = std::min(tf, max_tf);
            data[doc_id][col] = tf;
        }
    }

    // Convert to sparse format
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

// Generate query that targets both common and rare terms (realistic query pattern)
knowhere::DataSetPtr
GenZipfianQuery(int32_t nq, int32_t cols, int32_t terms_per_query, float zipf_exp, int seed = 123) {
    std::mt19937 rng(seed);
    auto uniform = std::uniform_real_distribution<float>(0, 1);

    auto tensor = std::make_unique<knowhere::sparse::SparseRow<float>[]>(nq);

    for (int32_t q = 0; q < nq; ++q) {
        std::set<int32_t> selected_terms;
        while ((int32_t)selected_terms.size() < terms_per_query) {
            // Select term with Zipfian distribution (prefer common terms, but also include rare)
            float u = uniform(rng);
            int32_t term = static_cast<int32_t>(cols * std::pow(u, zipf_exp));
            term = std::min(term, cols - 1);
            selected_terms.insert(term);
        }

        knowhere::sparse::SparseRow<float> row(selected_terms.size());
        size_t j = 0;
        for (int32_t term : selected_terms) {
            // Query weight: rare terms (high index) get higher weight (like IDF)
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

struct BenchResult {
    double time_us;
    float recall;
};

BenchResult
RunBenchmark(const std::string& algo, bool use_block_max, knowhere::DataSetPtr& train_ds,
             knowhere::DataSetPtr& query_ds, knowhere::DataSetPtr& gt, int topk, int runs,
             const std::string& metric = "IP") {
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
        build_conf[knowhere::meta::BM25_K1] = 1.2f;
        build_conf[knowhere::meta::BM25_B] = 0.75f;
        build_conf[knowhere::meta::BM25_AVGDL] = 100.0f;
        search_conf[knowhere::meta::BM25_K1] = 1.2f;
        search_conf[knowhere::meta::BM25_B] = 0.75f;
        search_conf[knowhere::meta::BM25_AVGDL] = 100.0f;
    }

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    // Use INDEX_SPARSE_INVERTED_INDEX (use_wand=false) so the algorithm is determined by config
    // INDEX_SPARSE_WAND has use_wand=true which forces DAAT_WAND regardless of config
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::sparse_u32_f32>(
        knowhere::IndexEnum::INDEX_SPARSE_INVERTED_INDEX, version);
    if (!idx.has_value()) {
        printf("Failed to create index\n");
        return {0, 0};
    }

    auto status = idx.value().Build(train_ds, build_conf);
    if (status != knowhere::Status::success) {
        printf("Build failed: %s\n", knowhere::Status2String(status).c_str());
        return {0, 0};
    }

    // Warmup
    for (int i = 0; i < 3; ++i) {
        auto result = idx.value().Search(query_ds, search_conf, nullptr);
    }

    // Benchmark
    auto t1 = std::chrono::high_resolution_clock::now();
    knowhere::DataSetPtr last_result;
    for (int i = 0; i < runs; ++i) {
        auto result = idx.value().Search(query_ds, search_conf, nullptr);
        if (result.has_value()) {
            last_result = result.value();
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_us = std::chrono::duration<double, std::micro>(t2 - t1).count() / runs;

    // Compute recall
    float recall = 0.0f;
    if (last_result && gt) {
        auto nq = last_result->GetRows();
        auto k = last_result->GetDim();
        auto gt_k = gt->GetDim();
        auto res_ids = last_result->GetIds();
        auto gt_ids = gt->GetIds();

        int matched = 0;
        for (int64_t i = 0; i < nq; ++i) {
            std::set<int64_t> gt_set(gt_ids + i * gt_k, gt_ids + i * gt_k + k);
            for (int64_t j = 0; j < k; ++j) {
                if (res_ids[i * k + j] >= 0 && gt_set.count(res_ids[i * k + j])) {
                    matched++;
                }
            }
        }
        recall = (float)matched / (float)(nq * k);
    }

    return {time_us, recall};
}

int
main() {
    printf("Sparse Search Algorithm Benchmark (Knowhere Library)\n");
    printf("=====================================================\n\n");

    const int runs = 10;

    // ========== PART 1: Zipfian (realistic text-like) data ==========
    // Choose metric: "IP" or "BM25"
    const std::string metric = "BM25";  // Change to "IP" for IP metric

    printf("=== Zipfian Data (realistic text-like distribution) [%s metric] ===\n", metric.c_str());
    printf("This simulates real text where:\n");
    printf("- Some terms are rare (high IDF) with concentrated high TF\n");
    printf("- Some terms are common (low IDF) appearing in many docs\n");
    printf("- BMW should excel by skipping low-scoring common term blocks\n\n");

    struct ZipfConfig {
        const char* name;
        int32_t nb;
        int32_t nq;
        int32_t dim;
        float avg_terms_per_doc;
        int32_t query_terms;
        float zipf_exp;
        int topk;
    };

    std::vector<ZipfConfig> zipf_configs = {
        {"Zipf 1.0 (uniform)", 50000, 100, 10000, 50.0f, 10, 1.0f, 10},
        {"Zipf 1.5 (moderate)", 50000, 100, 10000, 50.0f, 10, 1.5f, 10},
        {"Zipf 2.0 (skewed)", 50000, 100, 10000, 50.0f, 10, 2.0f, 10},
        {"Large Zipf 1.5", 200000, 100, 20000, 100.0f, 15, 1.5f, 10},
    };

    for (const auto& cfg : zipf_configs) {
        printf("Config: %s (nb=%d, nq=%d, dim=%d, query_terms=%d, topk=%d)\n", cfg.name, cfg.nb, cfg.nq, cfg.dim,
               cfg.query_terms, cfg.topk);

        auto train_ds = GenZipfianDataSet(cfg.nb, cfg.dim, cfg.avg_terms_per_doc, cfg.zipf_exp, 42);
        auto query_ds = GenZipfianQuery(cfg.nq, cfg.dim, cfg.query_terms, cfg.zipf_exp, 123);

        // Compute ground truth
        knowhere::Json gt_conf = {
            {knowhere::meta::METRIC_TYPE, metric},
            {knowhere::meta::TOPK, cfg.topk},
        };
        if (metric == "BM25") {
            gt_conf[knowhere::meta::BM25_K1] = 1.2f;
            gt_conf[knowhere::meta::BM25_B] = 0.75f;
            gt_conf[knowhere::meta::BM25_AVGDL] = 100.0f;
        }
        auto gt_result = knowhere::BruteForce::SearchSparse(train_ds, query_ds, gt_conf, nullptr);
        knowhere::DataSetPtr gt = gt_result.has_value() ? gt_result.value() : nullptr;

        auto taat = RunBenchmark("TAAT_NAIVE", false, train_ds, query_ds, gt, cfg.topk, runs, metric);
        auto wand = RunBenchmark("DAAT_WAND", false, train_ds, query_ds, gt, cfg.topk, runs, metric);
        auto bmw = RunBenchmark("DAAT_WAND", true, train_ds, query_ds, gt, cfg.topk, runs, metric);
        auto maxscore = RunBenchmark("DAAT_MAXSCORE", false, train_ds, query_ds, gt, cfg.topk, runs, metric);

        printf("  TAAT:      %8.0f us,          recall=%.2f%%\n", taat.time_us, taat.recall * 100);
        printf("  WAND:      %8.0f us (%.2fx),  recall=%.2f%%\n", wand.time_us, taat.time_us / wand.time_us,
               wand.recall * 100);
        printf("  BMW:       %8.0f us (%.2fx),  recall=%.2f%%\n", bmw.time_us, taat.time_us / bmw.time_us,
               bmw.recall * 100);
        printf("  MaxScore:  %8.0f us (%.2fx),  recall=%.2f%%\n", maxscore.time_us, taat.time_us / maxscore.time_us,
               maxscore.recall * 100);
        printf("\n");
    }

    return 0;
}
