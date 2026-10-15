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
    // reconstructed TF values for each of the 16 u4 codes.
    // It is implied that decode[0] == 0, decode[1] == 1 precisely.
    //   This is a design contract.
    std::array<uint8_t, 16> decode{};

    // inclusive upper TF boundary for each of the code's interval
    std::array<uint8_t, 16> ends{};

    // U4 code assigned to each of 8-bit TF
    std::array<uint8_t, 256> encode{};
    float k1 = 1.2f;

    void
    validate_and_encode() {
        if (!std::isfinite(k1) || k1 < 0 || decode[0] != 0 || ends[0] != 0 || decode[1] != 1 || ends[1] != 1 ||
            ends[15] != 255) {
            throw std::invalid_argument("Invalid BM25 u4 LUT contract");
        }

        for (size_t c = 1; c < 16; ++c) {
            if (ends[c] <= ends[c - 1] || decode[c] <= ends[c - 1] || decode[c] > ends[c]) {
                throw std::invalid_argument("Invalid BM25 u4 LUT interval/representative");
            }

            for (unsigned t = unsigned(ends[c - 1]) + 1; t <= ends[c]; ++t) {
                encode[t] = c;
            }
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

// Learn one corpus-wide U4 TF codebook from histogram[t], the number of postings
// with saturated TF t (0..255). Reserve codes 0 and 1 for exact TFs 0 and 1, then
// partition TFs 2..255 into 14 contiguous bins with integer TF representatives.
//
// Minimize sum_t histogram[t] * (g(t) - g(decode[encode[t]]))^2, where
// g(t) = (k1 + 1) * t / (t + k1) is the BM25 TF factor at dl == avgdl.
// At this reference length b cancels; actual search still uses original document
// lengths. The fit is globally optimal for this binning objective, not for recall.
// Stable ties choose lower representatives and earlier split points. Empty bins
// have a deterministic lowest representative. The returned LUT includes a complete
// encoder, so even TFs absent from the histogram have a defined code.
inline Bm25U4Lut
fit_bm25_u4_lut(const std::array<uint64_t, 256>& histogram, float k1) {
    if (!std::isfinite(k1) || k1 < 0) {
        throw std::invalid_argument("Invalid LUT fitting k1");
    }

    // Transform TFs into reference-length BM25 values. Prefix sums w, a and z
    // hold weights, weighted values and weighted squared values, respectively,
    // so any interval's statistics can be obtained by two prefix lookups.
    std::array<long double, 256> g{};
    std::array<long double, 256> w{};
    std::array<long double, 256> a{};
    std::array<long double, 256> z{};

    for (unsigned t = 1; t < 256; ++t) {
        g[t] = (static_cast<long double>(k1) + 1) * t / (t + static_cast<long double>(k1));
        w[t] = w[t - 1] + histogram[t];
        a[t] = a[t - 1] + histogram[t] * g[t];
        z[t] = z[t - 1] + histogram[t] * g[t] * g[t];
    }

    // Precompute the minimum loss and best integer representative for every
    // possible bin [lo, hi] within 2..255. Tables use lo * stride + hi indexing.
    constexpr size_t stride = 256;
    std::vector<long double> cost(stride * stride);
    std::vector<uint8_t> rep(stride * stride);

    for (unsigned lo = 2; lo < 256; ++lo) {
        for (unsigned hi = lo; hi < 256; ++hi) {
            const auto weight = w[hi] - w[lo - 1];
            const auto sum = a[hi] - a[lo - 1];
            const auto square = z[hi] - z[lo - 1];

            // Expand sum_t histogram[t] * (g(t) - g(r))^2 using the prefix
            // statistics. Clamp tiny negative results from roundoff to zero.
            auto loss = [&](unsigned r) { return std::max(0.L, square - 2 * g[r] * sum + g[r] * g[r] * weight); };

            // Squared error is minimized at the weighted mean in g-space.
            // Since g is nondecreasing, only the integer representatives on
            // either side of that mean need comparison. Prefer the lower one
            // on equal loss; a bin with no observations keeps r == lo.
            unsigned r = lo;
            if (weight > 0) {
                const auto mean = sum / weight;
                const auto it = std::lower_bound(g.begin() + lo, g.begin() + hi + 1, mean);
                r = std::min<unsigned>(hi, it - g.begin());
                if (r > lo && loss(r - 1) <= loss(r)) {
                    --r;
                }
            }

            cost[lo * stride + hi] = loss(r);
            rep[lo * stride + hi] = r;
        }
    }

    // dp[bins][hi] is the minimum loss for partitioning TFs 2..hi into exactly
    // 'bins' nonempty intervals. split records the previous interval's endpoint
    // so the winning partition can be reconstructed. Unreachable states are inf.
    const auto inf = std::numeric_limits<long double>::infinity();
    std::array<std::array<long double, 256>, 15> dp{};
    std::array<std::array<uint8_t, 256>, 15> split{};
    for (auto& row : dp) {
        row.fill(inf);
    }

    // The empty range before TF 2 needs zero bins and has zero loss. Try each
    // possible endpoint 'prev' before the final bin [prev + 1, hi]. Ascending
    // split order and strict improvement preserve earlier splits on equal loss.
    dp[0][1] = 0;
    for (unsigned bins = 1; bins <= 14; ++bins) {
        for (unsigned hi = bins + 1; hi < 256; ++hi) {
            for (unsigned prev = bins; prev < hi; ++prev) {
                const auto candidate = dp[bins - 1][prev] + cost[(prev + 1) * stride + hi];
                if (candidate < dp[bins][hi]) {
                    dp[bins][hi] = candidate;
                    split[bins][hi] = prev;
                }
            }
        }
    }

    // Codes 0 and 1 are fixed (zero initialization supplies code 0). Backtrack
    // from the full 14-bin solution through TF 255 to fill codes 2..15 with
    // inclusive upper boundaries and the precomputed TF representatives.
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

    // Check the interval/representative contract and expand the boundaries into
    // the encoder for all 256 input TFs, including TFs unseen during fitting.
    lut.validate_and_encode();

    return lut;
}

}  // namespace knowhere::sparse::inverted::sindi
