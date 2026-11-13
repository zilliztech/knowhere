// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

// Document reordering for sealed sparse inverted indexes.
//
// BP_kmeans is a recursive balanced bisection of the documents. Each bisection
// starts from the current halves and runs a few swap passes. A document's move
// gain is the sum of
//   - the approximate BP (log-gap) gain of Mackenzie et al., "Faster Recursive
//     Graph Bisection on Large Graphs" (SIGIR 2021), and
//   - a k-means gain: the change in squared L2 distance to the two halves'
//     centroids (Carlson et al., "Scalable K-Means Guided Partitioning for
//     Block-based Sparse Document Retrieval", SIGIR 2026).
// Left documents are sorted by gain descending, right documents ascending, and
// pairs are swapped while the combined gain is positive. This follows the
// BP_kmeans gain of pisa-engine/guidekp (Apache-2.0), commit abaa46cd.
//
// The result is only a permutation. Postings, values and document lengths are
// unchanged; the index stores the permutation and maps results back.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "index/sparse/parallel_build.h"
#include "knowhere/comp/task.h"
#include "knowhere/sparse_utils.h"

namespace knowhere::sparse::inverted::reorder {

// Defaults are the configuration measured in zilliztech/knowhere#1879.
struct BpKmeansOptions {
    // Swap passes per bisection.
    uint32_t iterations = 5;
    // Partitions with at most this many documents are not split further.
    uint32_t leaf_size = 16;
    uint32_t max_depth = 100;
    // Only terms with min_df <= df < max_df_ratio * rows are used as features.
    uint32_t min_df = 64;
    float max_df_ratio = 0.1f;
    // GuideKP BP_kmeans combines 0.5 * (100 * bp_gain) + 0.5 * kmeans_gain.
    float bp_weight = 50.0f;
    float kmeans_weight = 0.5f;
};

struct BpKmeansStats {
    size_t features = 0;
    size_t feature_postings = 0;
    size_t reordered_docs = 0;
    size_t swaps = 0;
};

namespace detail {

// Feature-only forward index: one row per document, compact feature ids.
struct ForwardIndex {
    std::vector<uint64_t> offsets;
    std::vector<uint32_t> terms;
    std::vector<float> weights;
    uint32_t nr_features = 0;
};

template <typename DType>
ForwardIndex
build_forward_index(const SparseRow<DType>* rows, size_t n, const BpKmeansOptions& opt) {
    // Invalid values are left to the index build's own validation; they never become features.
    const auto keep_value = [](DType value) {
        return std::isfinite(value) && std::abs(value) >= std::numeric_limits<DType>::epsilon();
    };

    std::unordered_map<table_t, uint32_t> df;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < rows[i].size(); ++j) {
            if (keep_value(rows[i][j].val)) {
                ++df[rows[i][j].id];
            }
        }
    }

    // Upper bound is exclusive and computed in float, as in GuideKP: df >= ceil(n * max_df_ratio) is dropped.
    const auto max_df = static_cast<uint64_t>(std::ceil(static_cast<float>(n) * opt.max_df_ratio));
    std::vector<table_t> selected;
    for (const auto& [dim, count] : df) {
        if (count >= opt.min_df && count < max_df) {
            selected.push_back(dim);
        }
    }
    // Feature ids must not depend on hash-map iteration order.
    std::sort(selected.begin(), selected.end());
    std::unordered_map<table_t, uint32_t> feature_of;
    feature_of.reserve(selected.size());
    for (size_t f = 0; f < selected.size(); ++f) {
        feature_of.emplace(selected[f], static_cast<uint32_t>(f));
    }

    ForwardIndex fwd;
    fwd.nr_features = static_cast<uint32_t>(selected.size());
    fwd.offsets.resize(n + 1, 0);
    if (selected.empty()) {
        return fwd;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < rows[i].size(); ++j) {
            const auto [dim, value] = rows[i][j];
            if (!keep_value(value)) {
                continue;
            }
            const auto it = feature_of.find(dim);
            if (it != feature_of.end()) {
                fwd.terms.push_back(it->second);
                fwd.weights.push_back(static_cast<float>(std::abs(value)));
            }
        }
        fwd.offsets[i + 1] = fwd.terms.size();
    }
    return fwd;
}

// log2(x) with log2(0) = 0, tabulated for small x like faster-graph-bisection.
inline float
fast_log2(int32_t x) {
    static const auto table = [] {
        std::array<float, 4096> t{};
        for (size_t i = 1; i < t.size(); ++i) {
            t[i] = std::log2(static_cast<float>(i));
        }
        return t;
    }();
    if (x <= 0) {
        return 0.0f;
    }
    return x < static_cast<int32_t>(table.size()) ? table[x] : std::log2(static_cast<float>(x));
}

class BpKmeans {
 public:
    BpKmeans(const ForwardIndex& fwd, const BpKmeansOptions& opt) : fwd_(fwd), opt_(opt) {
    }

    // Reorders docs[0, n) in place: one balanced bisection with opt_.iterations swap passes.
    // Returns the number of swaps.
    size_t
    bisect(uint32_t* docs, size_t n, bool parallel_gains) const {
        const size_t split = n / 2;
        if (split == 0 || split >= n) {
            return 0;
        }

        // Partition-local copy of the forward rows with dense local term ids, so the
        // per-term state is sized by the partition's vocabulary, not the corpus's.
        thread_local std::vector<int32_t> local_of;
        thread_local std::vector<uint32_t> touched;
        if (local_of.size() < fwd_.nr_features) {
            local_of.resize(fwd_.nr_features, -1);
        }
        touched.clear();

        std::vector<Item> items(n);
        std::vector<uint64_t> row_offsets(n + 1, 0);
        std::vector<uint32_t> row_terms;
        std::vector<float> row_weights;
        size_t postings = 0;
        for (size_t i = 0; i < n; ++i) {
            postings += fwd_.offsets[docs[i] + 1] - fwd_.offsets[docs[i]];
        }
        row_terms.reserve(postings);
        row_weights.reserve(postings);
        for (size_t i = 0; i < n; ++i) {
            const auto doc = docs[i];
            for (auto p = fwd_.offsets[doc]; p < fwd_.offsets[doc + 1]; ++p) {
                const auto f = fwd_.terms[p];
                if (local_of[f] < 0) {
                    local_of[f] = static_cast<int32_t>(touched.size());
                    touched.push_back(f);
                }
                row_terms.push_back(static_cast<uint32_t>(local_of[f]));
                row_weights.push_back(fwd_.weights[p]);
            }
            row_offsets[i + 1] = row_terms.size();
            items[i] = Item{0.0, doc, static_cast<uint32_t>(i)};
        }
        const size_t vocab = touched.size();
        for (const auto f : touched) {
            local_of[f] = -1;
        }

        std::vector<int32_t> left_deg(vocab, 0), right_deg(vocab, 0);
        std::vector<double> left_sum(vocab, 0.0), right_sum(vocab, 0.0);
        const auto add_row = [&](const Item& item, std::vector<int32_t>& deg, std::vector<double>& sum, int sign) {
            for (auto p = row_offsets[item.row]; p < row_offsets[item.row + 1]; ++p) {
                deg[row_terms[p]] += sign;
                sum[row_terms[p]] += sign * static_cast<double>(row_weights[p]);
            }
        };
        for (size_t i = 0; i < n; ++i) {
            if (i < split) {
                add_row(items[i], left_deg, left_sum, 1);
            } else {
                add_row(items[i], right_deg, right_sum, 1);
            }
        }

        const double n_left = static_cast<double>(split);
        const double n_right = static_cast<double>(n - split);
        std::vector<double> left_mu(vocab), right_mu(vocab);
        size_t total_swaps = 0;
        for (uint32_t iter = 0; iter < opt_.iterations; ++iter) {
            // Double precision keeps the k-means term finite for any finite float weights.
            double left_norm2 = 0.0, right_norm2 = 0.0;
            for (size_t t = 0; t < vocab; ++t) {
                left_mu[t] = left_sum[t] / n_left;
                right_mu[t] = right_sum[t] / n_right;
                left_norm2 += left_mu[t] * left_mu[t];
                right_norm2 += right_mu[t] * right_mu[t];
            }

            // Positive gain on the left means "move right"; negative gain on the right means "move left".
            const auto compute_gain = [&](size_t i) {
                auto& item = items[i];
                const bool on_left = i < split;
                const auto& from_deg = on_left ? left_deg : right_deg;
                const auto& to_deg = on_left ? right_deg : left_deg;
                const auto& from_mu = on_left ? left_mu : right_mu;
                const auto& to_mu = on_left ? right_mu : left_mu;
                float bp = 0.0f;
                double dot_from = 0.0, dot_to = 0.0;
                for (auto p = row_offsets[item.row]; p < row_offsets[item.row + 1]; ++p) {
                    const auto t = row_terms[p];
                    const double w = row_weights[p];
                    bp += fast_log2(to_deg[t] + 2) - fast_log2(from_deg[t]) -
                          kLog2E / (1.0f + static_cast<float>(to_deg[t]));
                    dot_from += w * from_mu[t];
                    dot_to += w * to_mu[t];
                }
                const double from_norm2 = on_left ? left_norm2 : right_norm2;
                const double to_norm2 = on_left ? right_norm2 : left_norm2;
                const double from_size = on_left ? n_left : n_right;
                // ||x - mu_from||^2 - ||x - mu_to||^2; the ||x||^2 terms cancel.
                double km = 0.0;
                if (from_size > 1.0) {
                    km = 2.0 * (dot_to - dot_from) + from_norm2 - to_norm2;
                }
                const double gain = opt_.bp_weight * bp + opt_.kmeans_weight * km;
                item.gain = on_left ? gain : -gain;
            };
            if (parallel_gains) {
                parallel_for(n, compute_gain);
            } else {
                for (size_t i = 0; i < n; ++i) {
                    compute_gain(i);
                }
            }

            // Ties are broken by document id so the result does not depend on the sort implementation.
            std::sort(items.begin(), items.begin() + split,
                      [](const Item& a, const Item& b) { return a.gain != b.gain ? a.gain > b.gain : a.doc < b.doc; });
            std::sort(items.begin() + split, items.end(),
                      [](const Item& a, const Item& b) { return a.gain != b.gain ? a.gain < b.gain : a.doc < b.doc; });

            size_t swaps = 0;
            for (size_t l = 0, r = split; l < split && r < n; ++l, ++r) {
                if (!(items[l].gain - items[r].gain > 0.0)) {
                    break;
                }
                add_row(items[l], left_deg, left_sum, -1);
                add_row(items[l], right_deg, right_sum, 1);
                add_row(items[r], right_deg, right_sum, -1);
                add_row(items[r], left_deg, left_sum, 1);
                std::swap(items[l], items[r]);
                ++swaps;
            }
            total_swaps += swaps;
            if (swaps == 0) {
                break;
            }
        }

        for (size_t i = 0; i < n; ++i) {
            docs[i] = items[i].doc;
        }
        return total_swaps;
    }

 private:
    struct Item {
        double gain;
        uint32_t doc;
        uint32_t row;
    };

    static constexpr float kLog2E = 1.44269504089f;

    const ForwardIndex& fwd_;
    const BpKmeansOptions& opt_;
};

}  // namespace detail

// Computes a document order for rows[0, n): order[new_id] = old_id.
// Documents without feature terms keep their relative order at the end.
// The result is deterministic and independent of the build thread count.
template <typename DType>
std::vector<uint32_t>
bp_kmeans_order(const SparseRow<DType>* rows, size_t n, const BpKmeansOptions& opt, BpKmeansStats* stats = nullptr) {
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    if (n <= std::max<size_t>(opt.leaf_size, 1) || opt.iterations == 0) {
        return order;
    }

    const auto fwd = detail::build_forward_index(rows, n, opt);
    if (stats != nullptr) {
        stats->features = fwd.nr_features;
        stats->feature_postings = fwd.terms.size();
    }
    if (fwd.nr_features == 0) {
        return order;
    }

    std::vector<uint32_t> docs;
    std::vector<uint32_t> empty_docs;
    docs.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        (fwd.offsets[i + 1] > fwd.offsets[i] ? docs : empty_docs).push_back(i);
    }
    if (stats != nullptr) {
        stats->reordered_docs = docs.size();
    }

    // Level-synchronous recursion: partitions of one level are independent. Few large
    // partitions parallelize their gain computation; many small ones run one per worker.
    const detail::BpKmeans bp(fwd, opt);
    const size_t workers = std::max<size_t>(1, GetBuildThreadPoolSize());
    const size_t leaf = std::max<size_t>(opt.leaf_size, 1);
    std::vector<std::pair<size_t, size_t>> ranges, next;
    if (docs.size() > leaf) {
        ranges.emplace_back(0, docs.size());
    }
    size_t swaps = 0;
    for (uint32_t depth = 1; depth <= opt.max_depth && !ranges.empty(); ++depth) {
        if (ranges.size() < workers) {
            for (const auto& [begin, end] : ranges) {
                swaps += bp.bisect(docs.data() + begin, end - begin, true);
            }
        } else {
            std::vector<size_t> range_swaps(ranges.size(), 0);
            parallel_for(ranges.size(), workers, [&](size_t i) {
                range_swaps[i] = bp.bisect(docs.data() + ranges[i].first, ranges[i].second - ranges[i].first, false);
            });
            swaps = std::accumulate(range_swaps.begin(), range_swaps.end(), swaps);
        }
        next.clear();
        for (const auto& [begin, end] : ranges) {
            const size_t mid = begin + (end - begin) / 2;
            if (mid - begin > leaf) {
                next.emplace_back(begin, mid);
            }
            if (end - mid > leaf) {
                next.emplace_back(mid, end);
            }
        }
        ranges.swap(next);
    }
    if (stats != nullptr) {
        stats->swaps = swaps;
    }

    std::copy(docs.begin(), docs.end(), order.begin());
    std::copy(empty_docs.begin(), empty_docs.end(), order.begin() + docs.size());
    return order;
}

}  // namespace knowhere::sparse::inverted::reorder
