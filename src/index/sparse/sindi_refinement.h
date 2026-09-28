// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "knowhere/sparse_utils.h"

namespace knowhere::sparse::inverted::sindi {

inline size_t
refinement_pool_size(size_t k, float factor, size_t count) {
    if (!std::isfinite(factor) || factor < 1) {
        throw std::invalid_argument("Invalid refine_k");
    }

    const long double requested = std::ceil(static_cast<long double>(k) * factor);
    return requested >= count ? count : static_cast<size_t>(requested);
}

template <typename T>
bool
valid_refinement_row(const SparseRow<T>& row) {
    for (size_t i = 0; i < row.size(); ++i) {
        const auto [dim, val] = row[i];
        if (!std::isfinite(val) || val < 0 || (i && row[i - 1].id >= dim)) {
            return false;
        }
    }

    return true;
}

// Select using original weights and external IDs, before dimension-map lookup.
// Query term selection does not depend on the physical posting value/ID codecs.
template <typename T>
SparseRow<T>
retain_query_mass(const SparseRow<T>& query, float mass) {
    if (!std::isfinite(mass) || mass <= 0 || mass > 1 || !valid_refinement_row(query)) {
        throw std::invalid_argument("Refinement requires finite nonnegative, sorted unique query coordinates");
    }

    std::vector<size_t> order(query.size());
    std::iota(order.begin(), order.end(), 0);

    double total = 0;
    for (size_t i = 0; i < query.size(); ++i) {
        total += query[i].val;
    }

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return (query[a].val != query[b].val) ? (query[a].val > query[b].val) : (query[a].id < query[b].id);
    });

    size_t keep = 0;
    double sum = 0;
    while (keep < order.size() && query[order[keep]].val > 0 && (mass == 1 || sum < mass * total)) {
        sum += query[order[keep++]].val;
    }

    order.resize(keep);
    std::sort(order.begin(), order.end());

    SparseRow<T> out(keep);
    for (size_t i = 0; i < keep; ++i) {
        out.set_at(i, query[order[i]].id, query[order[i]].val);
    }

    return out;
}

// Logical posting offsets, not byte offsets. Future packed backends may locate/decode
// blocks behind window_posting_range and score_candidate_ids without changing search.
struct RefinementSeek {
    bool sparse = false;
    std::vector<uint32_t> windows;
    std::vector<uint32_t> offsets;

    std::pair<uint32_t, uint32_t> inline range(uint32_t window) const {
        size_t pos = window;

        if (sparse) {
            auto it = std::lower_bound(windows.begin(), windows.end(), window);
            if (it == windows.end() || *it != window) {
                return {0, 0};
            }

            pos = it - windows.begin();
        }

        return {offsets.at(pos), offsets.at(pos + 1)};
    }
};

}  // namespace knowhere::sparse::inverted::sindi
