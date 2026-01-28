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

// This benchmark is x86_64-specific due to AVX512 intrinsics
// It should only be built on x86_64 systems (guarded in CMakeLists.txt)
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__amd64__)
#error "This benchmark requires x86_64 architecture. It should not be built on ARM/other platforms."
#endif

#include <boost/core/span.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

#include "knowhere/sparse_utils.h"
#include "simd/instruction_set.h"
#include "simd/sparse_simd.h"

using namespace knowhere::sparse;

// Generate synthetic sparse data with realistic posting list distributions
struct SparseDataset {
    size_t n_docs;
    size_t n_queries;
    size_t vocab_size;
    std::vector<std::vector<table_t>> posting_list_ids;
    std::vector<std::vector<float>> posting_list_vals;
    std::vector<std::pair<size_t, float>> query;

    SparseDataset(size_t n_docs, size_t n_queries, size_t vocab_size, size_t avg_query_terms,
                  size_t avg_posting_list_len, bool force_heavy_terms = false)
        : n_docs(n_docs), n_queries(n_queries), vocab_size(vocab_size) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> val_dist(0.1f, 1.0f);
        std::uniform_int_distribution<size_t> term_dist(0, vocab_size - 1);
        std::uniform_int_distribution<size_t> query_len_dist(avg_query_terms / 2, avg_query_terms * 2);

        // Initialize posting lists
        posting_list_ids.resize(vocab_size);
        posting_list_vals.resize(vocab_size);

        // Generate posting lists with zipf-like distribution
        // Create a more realistic distribution where some terms are very frequent
        std::vector<size_t> target_lengths(vocab_size);

        // Use Zipf distribution: rank r has frequency proportional to 1/r^alpha
        double alpha = 1.0;  // Zipf parameter
        double sum = 0.0;
        for (size_t r = 1; r <= vocab_size; ++r) {
            sum += 1.0 / std::pow(r, alpha);
        }

        // Scale so average = avg_posting_list_len
        double scale = avg_posting_list_len * vocab_size / sum;

        for (size_t term_id = 0; term_id < vocab_size; ++term_id) {
            size_t rank = term_id + 1;
            double freq = scale / std::pow(rank, alpha);
            target_lengths[term_id] = std::min(n_docs, std::max(size_t(1), static_cast<size_t>(freq)));
        }

        // Generate posting lists
        for (size_t term_id = 0; term_id < vocab_size; ++term_id) {
            size_t target_len = target_lengths[term_id];

            std::vector<table_t> doc_ids;
            std::uniform_int_distribution<table_t> doc_dist(0, n_docs - 1);

            // Generate unique random doc IDs
            std::unordered_set<table_t> seen;
            while (doc_ids.size() < target_len) {
                table_t doc_id = doc_dist(rng);
                if (seen.insert(doc_id).second) {
                    doc_ids.push_back(doc_id);
                }
            }

            // Sort for cache-friendly access
            std::sort(doc_ids.begin(), doc_ids.end());

            posting_list_ids[term_id] = std::move(doc_ids);
            posting_list_vals[term_id].resize(posting_list_ids[term_id].size());

            for (size_t i = 0; i < posting_list_vals[term_id].size(); ++i) {
                posting_list_vals[term_id][i] = val_dist(rng);
            }
        }

        // Generate query
        if (force_heavy_terms) {
            // Force query to include heavy (frequent) terms with long posting lists
            // This ensures SIMD actually gets exercised
            size_t heavy_terms = std::min(size_t(10), vocab_size);
            for (size_t i = 0; i < heavy_terms; ++i) {
                query.push_back({i, val_dist(rng)});
            }
            // Add some random terms too
            size_t random_terms = avg_query_terms - heavy_terms;
            for (size_t i = 0; i < random_terms; ++i) {
                size_t term_id = term_dist(rng);
                query.push_back({term_id, val_dist(rng)});
            }
        } else {
            // Random query generation
            size_t query_len = query_len_dist(rng);
            for (size_t i = 0; i < query_len; ++i) {
                size_t term_id = term_dist(rng);
                float weight = val_dist(rng);
                query.push_back({term_id, weight});
            }
        }
    }
};

// Timing utilities
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

// Benchmark runner
void
run_benchmark(const char* name, const SparseDataset& dataset) {
    using QType = float;  // This benchmark uses float values

    printf("\n=== %s ===\n", name);
    printf("Dataset: %zu docs, %zu vocab, query length: %zu\n", dataset.n_docs, dataset.vocab_size,
           dataset.query.size());

    // Calculate posting list length statistics
    size_t total_postings = 0;
    size_t non_empty = 0;
    size_t min_len = SIZE_MAX;
    size_t max_len = 0;
    std::vector<size_t> lengths;
    for (const auto& plist : dataset.posting_list_ids) {
        if (!plist.empty()) {
            size_t len = plist.size();
            total_postings += len;
            non_empty++;
            min_len = std::min(min_len, len);
            max_len = std::max(max_len, len);
            lengths.push_back(len);
        }
    }
    std::sort(lengths.begin(), lengths.end());
    size_t median_len = lengths.empty() ? 0 : lengths[lengths.size() / 2];

    printf("Posting list stats: avg=%.1f, median=%zu, min=%zu, max=%zu\n",
           non_empty > 0 ? (double)total_postings / non_empty : 0.0, median_len, min_len, max_len);

    // Show top-10 heaviest terms (what queries should hit for SIMD benefit)
    printf("Top-10 heaviest terms: ");
    for (size_t i = 0; i < std::min(size_t(10), lengths.size()); ++i) {
        printf("%zu ", lengths[lengths.size() - 1 - i]);
    }
    printf("\n");

    // Prepare data structures
    std::vector<boost::span<const table_t>> ids_spans;
    std::vector<boost::span<const float>> vals_spans;
    for (size_t i = 0; i < dataset.vocab_size; ++i) {
        ids_spans.emplace_back(dataset.posting_list_ids[i]);
        vals_spans.emplace_back(dataset.posting_list_vals[i]);
    }

    const int warmup_runs = 5;
    const int bench_runs = 50;

    // Check CPU capabilities
#if defined(__x86_64__) || defined(_M_X64)
    auto& inst_set = faiss::cppcontrib::knowhere::InstructionSet::GetInstance();
    printf("CPU Capabilities: AVX512F=%d, AVX2=%d\n", inst_set.AVX512F(), inst_set.AVX2());
#else
    printf("CPU Capabilities: ARM/Apple Silicon (no SIMD)\n");
#endif

    std::vector<float> result_scalar, result_avx512;

#ifdef __AVX512F__
    {
        for (int i = 0; i < warmup_runs; ++i) {
            result_avx512.assign(dataset.n_docs, 0.0f);
            for (const auto& [dim_idx, q_weight] : dataset.query) {
                const auto& plist_ids = ids_spans[dim_idx];
                const auto& plist_vals = vals_spans[dim_idx];

                accumulate_posting_list_contribution_ip_dispatch<QType>(plist_ids.data(), plist_vals.data(),
                                                                        plist_ids.size(), static_cast<float>(q_weight),
                                                                        result_avx512.data());
            }
        }
    }
#endif

    // Benchmark scalar
    printf("\n[Scalar Fallback]\n");
    Timer timer;
    for (int i = 0; i < bench_runs; ++i) {
        result_scalar.assign(dataset.n_docs, 0.0f);
        for (size_t q_idx = 0; q_idx < dataset.query.size(); ++q_idx) {
            const auto& plist_ids = ids_spans[dataset.query[q_idx].first];
            const auto& plist_vals = vals_spans[dataset.query[q_idx].first];
            const float q_weight = dataset.query[q_idx].second;

            for (size_t j = 0; j < plist_ids.size(); ++j) {
                const auto doc_id = plist_ids[j];
                result_scalar[doc_id] += q_weight * plist_vals[j];
            }
        }
    }
    double scalar_time = timer.elapsed_ms() / bench_runs;
    printf("  Time: %.3f ms\n", scalar_time);

    // Count non-zero results for verification
    size_t scalar_nonzero = 0;
    for (float score : result_scalar) {
        if (score > 1e-6f)
            scalar_nonzero++;
    }
    printf("  Non-zero scores: %zu / %zu\n", scalar_nonzero, result_scalar.size());

#ifdef __AVX512F__
    {
        printf("\n[SIMD Dispatcher (AVX512 if available)]\n");

        timer.reset();
        for (int i = 0; i < bench_runs; ++i) {
            result_avx512.assign(dataset.n_docs, 0.0f);
            for (const auto& [dim_idx, q_weight] : dataset.query) {
                const auto& plist_ids = ids_spans[dim_idx];
                const auto& plist_vals = vals_spans[dim_idx];

                accumulate_posting_list_contribution_ip_dispatch<QType>(plist_ids.data(), plist_vals.data(),
                                                                        plist_ids.size(), static_cast<float>(q_weight),
                                                                        result_avx512.data());
            }
        }

        double avx512_time = timer.elapsed_ms() / bench_runs;
        printf("  Time: %.3f ms\n", avx512_time);
        printf("  Using: %s\n", inst_set.AVX512F() ? "AVX512" : "Scalar");

        size_t avx512_nonzero = 0;
        for (float score : result_avx512) {
            if (score > 1e-6f)
                avx512_nonzero++;
        }
        printf("  Non-zero scores: %zu / %zu\n", avx512_nonzero, result_avx512.size());

        double max_diff = 0.0;
        double avg_diff = 0.0;
        size_t diff_count = 0;
        for (size_t i = 0; i < result_scalar.size(); ++i) {
            double diff = std::abs(result_scalar[i] - result_avx512[i]);
            if (diff > 1e-4) {
                avg_diff += diff;
                diff_count++;
                max_diff = std::max(max_diff, diff);
            }
        }
        if (diff_count > 0) {
            avg_diff /= diff_count;
        }

        printf("\n[Verification]\n");
        printf("  Max difference: %.6f\n", max_diff);
        printf("  Avg difference: %.6f (over %zu elements)\n", avg_diff, diff_count);
        printf("  Correctness: %s\n", (max_diff < 1e-3) ? "PASS" : "FAIL");

        printf("\n[Performance]\n");
        double speedup = scalar_time / avx512_time;
        printf("  Speedup: %.2fx\n", speedup);
        printf("  Scalar:  %.3f ms (baseline)\n", scalar_time);
        printf("  AVX512:  %.3f ms (%.1f%% of baseline)\n", avx512_time, 100.0 * avx512_time / scalar_time);
    }
#else
    printf("\n[AVX512 not compiled in (requires -mavx512f)]\n");
#endif

    printf("==========================================\n");
}

int
main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Sparse Inverted Index SIMD Benchmark                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test configurations
    struct BenchConfig {
        const char* name;
        size_t n_docs;
        size_t vocab_size;
        size_t avg_query_terms;
        size_t avg_posting_list_len;
        bool force_heavy_terms;
    };

    std::vector<BenchConfig> configs = {
        // Ultra-sparse: posting lists shorter than SIMD width (16)
        {"Ultra-sparse IP (random query, avg=8)", 50000, 2000, 15, 8, false},
        {"Ultra-sparse IP (heavy terms, avg=8)", 50000, 2000, 15, 8, true},

        // Sparse: posting lists around SIMD width (16-32)
        {"Sparse IP (random query, avg=32)", 100000, 5000, 20, 32, false},
        {"Sparse IP (heavy terms, avg=32)", 100000, 5000, 20, 32, true},

        // Medium density: posting lists 2-8x SIMD width (64-128)
        {"Medium IP (random query, avg=128)", 500000, 8000, 25, 128, false},
        {"Medium IP (heavy terms, avg=128)", 500000, 8000, 25, 128, true},

        // Dense: posting lists 16-32x SIMD width (256-512)
        {"Dense IP (random query, avg=512)", 1000000, 10000, 30, 512, false},
        {"Dense IP (heavy terms, avg=512)", 1000000, 10000, 30, 512, true},

        // Very dense: posting lists 64-128x SIMD width (1024-2048)
        {"Very Dense IP (heavy terms, avg=2048)", 1000000, 10000, 30, 2048, true},

        // Real-world-like: MSMARCO/Wikipedia scale
        {"Real-world IP (avg=256, heavy head)", 1000000, 10000, 25, 256, true},
    };

    for (const auto& config : configs) {
        SparseDataset dataset(config.n_docs, 1, config.vocab_size, config.avg_query_terms, config.avg_posting_list_len,
                              config.force_heavy_terms);
        run_benchmark(config.name, dataset);
    }

    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Benchmark completed                                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    return 0;
}
