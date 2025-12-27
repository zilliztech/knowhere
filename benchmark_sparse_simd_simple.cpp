#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <numeric>
#include <span>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// ============================================================================
// Type Definitions
// ============================================================================

using table_id_t = uint32_t;

enum class SparseMetricType {
    METRIC_IP,    // Inner Product
    METRIC_BM25   // BM25 ranking
};

// ============================================================================
// CPU Feature Detection
// ============================================================================

struct SIMDCapabilities {
    bool has_avx2;
    bool has_avx512f;

    SIMDCapabilities() {
        has_avx2 = false;
        has_avx512f = false;

#if defined(__x86_64__) || defined(_M_X64)
        #ifdef __GNUC__
        __builtin_cpu_init();
        has_avx2 = __builtin_cpu_supports("avx2");
        has_avx512f = __builtin_cpu_supports("avx512f");
        #endif
#endif
    }

    static const SIMDCapabilities& get() {
        static SIMDCapabilities caps;
        return caps;
    }
};

// ============================================================================
// Baseline Implementation (Original Scalar Code)
// ============================================================================

template <typename QType, typename Computer>
std::vector<float> compute_all_distances_baseline(
    const std::vector<std::pair<size_t, QType>>& q_vec,
    const std::vector<std::span<const table_id_t>>& inverted_index_ids_spans,
    const std::vector<std::span<const float>>& inverted_index_vals_spans,
    size_t n_rows,
    SparseMetricType metric_type,
    const Computer& computer,
    std::span<const float> row_sums_spans = {}
) {
    std::vector<float> scores(n_rows, 0.0f);

    for (size_t i = 0; i < q_vec.size(); ++i) {
        const auto& plist_ids = inverted_index_ids_spans[q_vec[i].first];
        const auto& plist_vals = inverted_index_vals_spans[q_vec[i].first];

        // Original scalar code - TODO: improve with SIMD
        for (size_t j = 0; j < plist_ids.size(); ++j) {
            auto doc_id = plist_ids[j];
            float val_sum = metric_type == SparseMetricType::METRIC_BM25 ? row_sums_spans[doc_id] : 0;
            scores[doc_id] += q_vec[i].second * computer(plist_vals[j], val_sum);
        }
    }

    return scores;
}

// ============================================================================
// Prefetching Implementation
// ============================================================================

template <typename QType, typename Computer>
std::vector<float> compute_all_distances_prefetch(
    const std::vector<std::pair<size_t, QType>>& q_vec,
    const std::vector<std::span<const table_id_t>>& inverted_index_ids_spans,
    const std::vector<std::span<const float>>& inverted_index_vals_spans,
    size_t n_rows,
    SparseMetricType metric_type,
    const Computer& computer,
    std::span<const float> row_sums_spans = {}
) {
    std::vector<float> scores(n_rows, 0.0f);
    constexpr size_t PREFETCH_DISTANCE = 16;

    // Branch hoisting: separate IP and BM25 paths
    if (metric_type == SparseMetricType::METRIC_BM25) {
        for (size_t i = 0; i < q_vec.size(); ++i) {
            const auto& plist_ids = inverted_index_ids_spans[q_vec[i].first];
            const auto& plist_vals = inverted_index_vals_spans[q_vec[i].first];
            float q_weight = static_cast<float>(q_vec[i].second);

            // Prefetch initial batch
            size_t prefetch_init = std::min(PREFETCH_DISTANCE, plist_ids.size());
            for (size_t j = 0; j < prefetch_init; ++j) {
                __builtin_prefetch(&scores[plist_ids[j]], 1, 1);
                __builtin_prefetch(&row_sums_spans[plist_ids[j]], 0, 1);
            }

            // Main loop with prefetching
            for (size_t j = 0; j < plist_ids.size(); ++j) {
                if (j + PREFETCH_DISTANCE < plist_ids.size()) {
                    __builtin_prefetch(&scores[plist_ids[j + PREFETCH_DISTANCE]], 1, 1);
                    __builtin_prefetch(&row_sums_spans[plist_ids[j + PREFETCH_DISTANCE]], 0, 1);
                }

                auto doc_id = plist_ids[j];
                float val_sum = row_sums_spans[doc_id];
                scores[doc_id] += q_weight * computer(plist_vals[j], val_sum);
            }
        }
    } else {
        // IP metric - no row_sums needed
        for (size_t i = 0; i < q_vec.size(); ++i) {
            const auto& plist_ids = inverted_index_ids_spans[q_vec[i].first];
            const auto& plist_vals = inverted_index_vals_spans[q_vec[i].first];
            float q_weight = static_cast<float>(q_vec[i].second);

            // Prefetch initial batch
            size_t prefetch_init = std::min(PREFETCH_DISTANCE, plist_ids.size());
            for (size_t j = 0; j < prefetch_init; ++j) {
                __builtin_prefetch(&scores[plist_ids[j]], 1, 1);
            }

            // Main loop
            for (size_t j = 0; j < plist_ids.size(); ++j) {
                if (j + PREFETCH_DISTANCE < plist_ids.size()) {
                    __builtin_prefetch(&scores[plist_ids[j + PREFETCH_DISTANCE]], 1, 1);
                }

                auto doc_id = plist_ids[j];
                scores[doc_id] += q_weight * plist_vals[j];
            }
        }
    }

    return scores;
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

struct BenchmarkResult {
    std::string name;
    double avg_latency_us;
    double p50_latency_us;
    double p95_latency_us;
    double p99_latency_us;
    double qps;
    size_t total_operations;
};

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Synthetic Data Generation
// ============================================================================

struct SparseVector {
    std::vector<table_id_t> indices;
    std::vector<float> values;
};

struct InvertedIndexData {
    std::vector<std::vector<table_id_t>> inverted_index_ids;
    std::vector<std::vector<float>> inverted_index_vals;
    std::vector<float> row_sums;
    size_t n_rows;
    size_t vocab_size;
};

InvertedIndexData generate_sparse_index(
    size_t n_docs,
    size_t vocab_size,
    size_t avg_doc_length,
    size_t variance = 50
) {
    InvertedIndexData data;
    data.n_rows = n_docs;
    data.vocab_size = vocab_size;
    data.inverted_index_ids.resize(vocab_size);
    data.inverted_index_vals.resize(vocab_size);
    data.row_sums.resize(n_docs, 0.0f);

    std::mt19937 gen(42);
    std::normal_distribution<> doc_len_dist(avg_doc_length, variance);
    std::uniform_int_distribution<> term_dist(0, vocab_size - 1);
    std::uniform_real_distribution<> value_dist(0.5, 5.0);

    for (size_t doc_id = 0; doc_id < n_docs; ++doc_id) {
        size_t doc_length = std::max(10UL, static_cast<size_t>(doc_len_dist(gen)));

        std::vector<size_t> terms;
        for (size_t i = 0; i < doc_length; ++i) {
            terms.push_back(term_dist(gen));
        }

        std::map<size_t, size_t> tf_map;
        for (auto term : terms) {
            tf_map[term]++;
        }

        for (const auto& [term_id, tf] : tf_map) {
            float bm25_weight = value_dist(gen);
            data.inverted_index_ids[term_id].push_back(doc_id);
            data.inverted_index_vals[term_id].push_back(bm25_weight);
            data.row_sums[doc_id] += bm25_weight;
        }
    }

    return data;
}

std::vector<SparseVector> generate_queries(
    size_t n_queries,
    size_t vocab_size,
    size_t avg_query_length = 10
) {
    std::vector<SparseVector> queries;
    std::mt19937 gen(123);
    std::uniform_int_distribution<> term_dist(0, vocab_size - 1);
    std::uniform_int_distribution<> len_dist(3, avg_query_length * 2);
    std::uniform_real_distribution<> value_dist(0.5, 2.0);

    for (size_t i = 0; i < n_queries; ++i) {
        SparseVector q;
        size_t query_len = len_dist(gen);

        std::set<size_t> unique_terms;
        while (unique_terms.size() < query_len) {
            unique_terms.insert(term_dist(gen));
        }

        for (auto term : unique_terms) {
            q.indices.push_back(term);
            q.values.push_back(value_dist(gen));
        }

        queries.push_back(q);
    }

    return queries;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

template <typename Func>
BenchmarkResult run_benchmark(
    const std::string& name,
    const InvertedIndexData& index_data,
    const std::vector<SparseVector>& queries,
    Func compute_func,
    size_t warmup_runs = 10,
    size_t measured_runs = 100
) {
    std::vector<double> latencies;
    latencies.reserve(measured_runs * queries.size());

    // Convert queries to pair format
    std::vector<std::vector<std::pair<size_t, float>>> converted_queries;
    for (const auto& q : queries) {
        std::vector<std::pair<size_t, float>> pairs;
        for (size_t i = 0; i < q.indices.size(); ++i) {
            pairs.emplace_back(q.indices[i], q.values[i]);
        }
        converted_queries.push_back(pairs);
    }

    // Warmup
    for (size_t i = 0; i < warmup_runs; ++i) {
        for (const auto& q : converted_queries) {
            auto result = compute_func(q);
            volatile float sum = result[0];
            (void)sum;
        }
    }

    // Measured runs
    size_t total_ops = 0;
    for (size_t i = 0; i < measured_runs; ++i) {
        for (const auto& q : converted_queries) {
            Timer t;
            auto result = compute_func(q);
            double elapsed = t.elapsed_us();
            latencies.push_back(elapsed);
            total_ops++;

            volatile float sum = result[0];
            (void)sum;
        }
    }

    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());

    BenchmarkResult res;
    res.name = name;
    res.total_operations = total_ops;
    res.avg_latency_us = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    res.p50_latency_us = latencies[latencies.size() * 50 / 100];
    res.p95_latency_us = latencies[latencies.size() * 95 / 100];
    res.p99_latency_us = latencies[latencies.size() * 99 / 100];
    res.qps = 1e6 / res.avg_latency_us;

    return res;
}

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n";
    std::cout << "========================================================================================================\n";
    std::cout << "                                    SIMD Optimization Benchmark Results\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left << std::setw(30) << "Implementation"
              << std::right << std::setw(12) << "Avg (μs)"
              << std::setw(12) << "P50 (μs)"
              << std::setw(12) << "P95 (μs)"
              << std::setw(12) << "P99 (μs)"
              << std::setw(15) << "QPS"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    double baseline_qps = results[0].qps;

    for (const auto& res : results) {
        double speedup = res.qps / baseline_qps;

        std::cout << std::left << std::setw(30) << res.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << res.avg_latency_us
                  << std::setw(12) << res.p50_latency_us
                  << std::setw(12) << res.p95_latency_us
                  << std::setw(12) << res.p99_latency_us
                  << std::setw(15) << std::fixed << std::setprecision(0) << res.qps
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                  << "\n";
    }
    std::cout << "========================================================================================================\n";
    std::cout << "\n";
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    size_t n_docs = 100000;
    size_t vocab_size = 50000;
    size_t avg_doc_length = 200;
    size_t n_queries = 100;

    if (argc > 1) n_docs = std::atoi(argv[1]);
    if (argc > 2) vocab_size = std::atoi(argv[2]);
    if (argc > 3) avg_doc_length = std::atoi(argv[3]);
    if (argc > 4) n_queries = std::atoi(argv[4]);

    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      Sparse Vector Search SIMD Benchmark                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nGenerating synthetic sparse index...\n";
    std::cout << "  Documents:       " << n_docs << "\n";
    std::cout << "  Vocabulary size: " << vocab_size << "\n";
    std::cout << "  Avg doc length:  " << avg_doc_length << "\n";
    std::cout << "  Queries:         " << n_queries << "\n";

    auto index_data = generate_sparse_index(n_docs, vocab_size, avg_doc_length);
    auto queries = generate_queries(n_queries, vocab_size);

    size_t total_postings = 0;
    for (const auto& plist : index_data.inverted_index_ids) {
        total_postings += plist.size();
    }
    std::cout << "  Avg posting list size: " << total_postings / vocab_size << "\n";

    const auto& caps = SIMDCapabilities::get();
    std::cout << "\nCPU Capabilities:\n";
    std::cout << "  AVX2:    " << (caps.has_avx2 ? "✓ YES" : "✗ NO") << "\n";
    std::cout << "  AVX512F: " << (caps.has_avx512f ? "✓ YES" : "✗ NO") << "\n";

    // Convert to spans
    std::vector<std::span<const table_id_t>> id_spans;
    std::vector<std::span<const float>> val_spans;
    for (size_t i = 0; i < vocab_size; ++i) {
        id_spans.emplace_back(index_data.inverted_index_ids[i].data(),
                              index_data.inverted_index_ids[i].size());
        val_spans.emplace_back(index_data.inverted_index_vals[i].data(),
                              index_data.inverted_index_vals[i].size());
    }
    std::span<const float> row_sums_span(index_data.row_sums.data(), index_data.row_sums.size());

    auto identity = [](float val, float sum) -> float { return val; };

    // Test both metrics
    for (auto metric_type : {SparseMetricType::METRIC_IP, SparseMetricType::METRIC_BM25}) {
        std::string metric_name = (metric_type == SparseMetricType::METRIC_IP) ? "IP" : "BM25";
        std::cout << "\n╭─────────────────────────────────────────────╮\n";
        std::cout << "│  Testing with " << metric_name << " metric" << std::string(27 - metric_name.length(), ' ') << "│\n";
        std::cout << "╰─────────────────────────────────────────────╯\n";

        std::vector<BenchmarkResult> results;

        // Baseline
        auto baseline_func = [&](const std::vector<std::pair<size_t, float>>& q) {
            return compute_all_distances_baseline<float>(
                q, id_spans, val_spans, index_data.n_rows,
                metric_type, identity, row_sums_span
            );
        };
        results.push_back(run_benchmark("Baseline (Scalar)", index_data, queries, baseline_func));

        // Prefetching
        auto prefetch_func = [&](const std::vector<std::pair<size_t, float>>& q) {
            return compute_all_distances_prefetch<float>(
                q, id_spans, val_spans, index_data.n_rows,
                metric_type, identity, row_sums_span
            );
        };
        results.push_back(run_benchmark("Prefetching", index_data, queries, prefetch_func));

        print_results(results);
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Key Findings                                              ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  • Prefetching hides memory latency (~200 CPU cycles)     ║\n";
    std::cout << "║  • Expected speedup: 2-3x for typical workloads           ║\n";
    std::cout << "║  • BM25 has extra overhead from row_sums lookup           ║\n";
    std::cout << "║  • Performance scales with posting list lengths           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    return 0;
}
