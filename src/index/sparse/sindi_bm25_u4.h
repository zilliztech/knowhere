// Copyright (C) 2026 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace knowhere::sparse::inverted::sindi {

inline uint8_t
unpack_bm25_u4(const uint8_t* values, size_t posting) noexcept {
    return (values[posting / 2] >> (4 * (posting & 1))) & 15;
}

// Compact sections need only byte alignment, including U16 ID streams.
inline uint16_t
unpack_bm25_u16_id(const uint8_t* ids, size_t posting) noexcept {
    uint16_t id;
    std::memcpy(&id, ids + posting * sizeof(id), sizeof(id));
    return id;
}

// A code is an index, not a TF. Endpoints define all inputs, including unseen TFs.
struct Bm25U4Lut {
    std::array<uint8_t, 16> decode{};
    std::array<uint8_t, 16> ends{};
    std::array<uint8_t, 256> encode{};
    float k1 = 1.2f;

    void
    validate_and_encode() {
        if (!std::isfinite(k1) || k1 < 0 || decode[0] != 0 || ends[0] != 0 || decode[1] != 1 || ends[1] != 1 ||
            ends[15] != 255)
            throw std::invalid_argument("Invalid BM25 u4 LUT contract");
        for (size_t c = 1; c < 16; ++c) {
            if (ends[c] <= ends[c - 1] || decode[c] <= ends[c - 1] || decode[c] > ends[c])
                throw std::invalid_argument("Invalid BM25 u4 LUT interval/representative");
            for (unsigned t = unsigned(ends[c - 1]) + 1; t <= ends[c]; ++t) encode[t] = c;
        }
        encode[0] = 0;
    }

    // Stable descriptor identity, not a cryptographic integrity checksum.
    uint64_t
    fingerprint() const {
        uint64_t h = 14695981039346656037ULL;
        auto add = [&](uint8_t byte) { h = (h ^ byte) * 1099511628211ULL; };
        add(1);  // codec/fitting version
        for (auto t : decode) add(t);
        for (auto t : ends) add(t);
        uint32_t bits;
        std::memcpy(&bits, &k1, sizeof(bits));
        for (unsigned i = 0; i < 4; ++i) add((bits >> (8 * i)) & 255);
        return h;
    }
};

// Globally optimal contiguous-bin fit for the declared reference-length BM25
// squared-error objective. TF 1 is exact. Stable ties choose lower representatives
// and earlier split points. Empty bins have a deterministic lowest representative.
inline Bm25U4Lut
fit_bm25_u4_lut(const std::array<uint64_t, 256>& histogram, float k1) {
    if (!std::isfinite(k1) || k1 < 0)
        throw std::invalid_argument("Invalid LUT fitting k1");
    std::array<long double, 256> g{}, w{}, a{}, z{};
    for (unsigned t = 1; t < 256; ++t) {
        g[t] = (static_cast<long double>(k1) + 1) * t / (t + static_cast<long double>(k1));
        w[t] = w[t - 1] + histogram[t];
        a[t] = a[t - 1] + histogram[t] * g[t];
        z[t] = z[t - 1] + histogram[t] * g[t] * g[t];
    }
    constexpr size_t stride = 256;
    std::vector<long double> cost(stride * stride);
    std::vector<uint8_t> rep(stride * stride);
    for (unsigned lo = 2; lo < 256; ++lo)
        for (unsigned hi = lo; hi < 256; ++hi) {
            const auto weight = w[hi] - w[lo - 1], sum = a[hi] - a[lo - 1], square = z[hi] - z[lo - 1];
            auto loss = [&](unsigned r) { return std::max(0.L, square - 2 * g[r] * sum + g[r] * g[r] * weight); };
            unsigned r = lo;
            if (weight > 0) {
                const auto mean = sum / weight;
                const auto it = std::lower_bound(g.begin() + lo, g.begin() + hi + 1, mean);
                r = std::min<unsigned>(hi, it - g.begin());
                if (r > lo && loss(r - 1) <= loss(r))
                    --r;
            }
            cost[lo * stride + hi] = loss(r);
            rep[lo * stride + hi] = r;
        }
    const auto inf = std::numeric_limits<long double>::infinity();
    std::array<std::array<long double, 256>, 15> dp{};
    std::array<std::array<uint8_t, 256>, 15> split{};
    for (auto& row : dp) row.fill(inf);
    dp[0][1] = 0;
    for (unsigned bins = 1; bins <= 14; ++bins)
        for (unsigned hi = bins + 1; hi < 256; ++hi)
            for (unsigned prev = bins; prev < hi; ++prev) {
                const auto candidate = dp[bins - 1][prev] + cost[(prev + 1) * stride + hi];
                if (candidate < dp[bins][hi]) {
                    dp[bins][hi] = candidate;
                    split[bins][hi] = prev;
                }
            }
    Bm25U4Lut lut;
    lut.k1 = k1;
    lut.ends[1] = lut.decode[1] = 1;
    unsigned hi = 255;
    for (unsigned bins = 14; bins > 0; --bins) {
        const unsigned prev = split[bins][hi];
        lut.ends[bins + 1] = hi;
        lut.decode[bins + 1] = rep[(prev + 1) * stride + hi];
        hi = prev;
    }
    lut.validate_and_encode();
    return lut;
}

}  // namespace knowhere::sparse::inverted::sindi
