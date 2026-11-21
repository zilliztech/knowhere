/*
 * Copyright (c) Knowhere contributors.
 * Licensed under the MIT license in the root directory of this source tree.
 */

#include <faiss/EMeans.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>

namespace faiss {
namespace {

size_t round_power_of_two(double value) {
    const auto largest = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
    if (value >= static_cast<double>(largest)) {
        return largest;
    }
    return value <= 1
            ? 1
            : static_cast<size_t>(std::exp2(std::round(std::log2(value))));
}

void revive(
        std::vector<float>& positions,
        std::vector<double>& mass,
        size_t d,
        std::mt19937_64& rng) {
    if (d <= 2) {
        return;
    }
    const double threshold =
            0.1 * std::accumulate(mass.begin(), mass.end(), 0.0) / mass.size();
    std::vector<size_t> unused;
    std::vector<std::pair<double, size_t>> donors;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (size_t i = 0; i < mass.size(); ++i) {
        if (mass[i] < threshold) {
            unused.push_back(i);
        } else if (mass[i] > 0) {
            const double u =
                    std::max(uniform(rng), std::numeric_limits<double>::min());
            donors.emplace_back(std::log(mass[i]) - std::log(-std::log(u)), i);
        }
    }
    std::sort(donors.begin(), donors.end(), std::greater<>());
    for (size_t i = 0; i < std::min(unused.size(), donors.size()); ++i) {
        const auto donor = donors[i].second;
        const auto child = unused[i];
        mass[child] = mass[donor] *= 0.5;
        bool identical = true;
        for (size_t j = 0; j < d; ++j) {
            const double value = positions[donor * d + j];
            const double offset = j % 2 == 0 ? 1e-4 : -1e-4;
            const double limit = std::numeric_limits<float>::max();
            positions[donor * d + j] =
                    std::clamp(value * (1 - offset), -limit, limit);
            positions[child * d + j] =
                    std::clamp(value * (1 + offset), -limit, limit);
            identical &= positions[donor * d + j] == positions[child * d + j];
        }
        if (identical) {
            auto& value = positions[child * d];
            auto neighbor = std::nextafter(
                    value, std::numeric_limits<float>::infinity());
            value = std::isfinite(neighbor)
                    ? neighbor
                    : std::nextafter(
                              value, -std::numeric_limits<float>::infinity());
        }
    }
}

} // namespace

EMeans::EMeans(int d, int k, const ClusteringParameters& cp)
        : d_(d), k_(k), cp_(cp) {}

void EMeans::train(idx_t nx, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            d_ > 0 && k_ > 0 && nx >= k_ && x,
            "e-means requires d > 0, 0 < k <= n and non-null data");
    FAISS_THROW_IF_NOT_MSG(
            std::isfinite(cp_.e_means_epochs) && cp_.e_means_epochs > 0 &&
                    cp_.e_means_epochs <= std::numeric_limits<int>::max(),
            "e-means epochs must be finite and positive");
    FAISS_THROW_IF_NOT_MSG(
            cp_.nredo > 0 && cp_.max_points_per_centroid > 0,
            "e-means requires positive nredo and max_points_per_centroid");
    FAISS_THROW_IF_NOT_MSG(
            !cp_.frozen_centroids && !cp_.int_centroids && !cp_.update_index,
            "e-means does not support frozen_centroids, int_centroids or update_index");
    const size_t d = d_, k = k_;
    FAISS_THROW_IF_NOT_MSG(
            static_cast<uint64_t>(nx) <= std::numeric_limits<size_t>::max() / d,
            "e-means input shape overflows size_t");
    for (size_t i = 0; i < static_cast<size_t>(nx) * d; ++i) {
        FAISS_THROW_IF_NOT_MSG(
                std::isfinite(x[i]), "e-means requires finite input vectors");
    }
    const auto initial = centroids;
    FAISS_THROW_IF_NOT_MSG(
            initial.empty() || initial.size() == k * d,
            "e-means requires a full initial centroid table");
    for (float value : initial) {
        FAISS_THROW_IF_NOT_MSG(
                std::isfinite(value),
                "e-means initial centroids must be finite");
    }
    std::mt19937_64 rng(
            cp_.seed < 0 ? std::random_device{}()
                         : static_cast<uint64_t>(cp_.seed));
    const size_t cap = k * static_cast<size_t>(cp_.max_points_per_centroid);
    const size_t n = std::min(static_cast<size_t>(nx), cap);
    std::vector<size_t> sample(n);
    std::iota(sample.begin(), sample.end(), 0);
    // Reservoir sampling bounds row-id storage by the training cap even when
    // the source is much larger than max_points_per_centroid * k.
    for (size_t i = n; i < static_cast<size_t>(nx); ++i) {
        const auto slot = std::uniform_int_distribution<size_t>(0, i)(rng);
        if (slot < n) {
            sample[slot] = i;
        }
    }
    const double epochs = cp_.e_means_epochs;
    const auto whole = static_cast<size_t>(std::floor(epochs));
    // Match Python's ties-to-even rounding without depending on the caller's
    // floating-point rounding mode.
    const double fraction = (epochs - whole) * n;
    size_t remainder = static_cast<size_t>(std::floor(fraction));
    if (fraction - remainder > 0.5 ||
        (fraction - remainder == 0.5 && remainder % 2)) {
        ++remainder;
    }
    std::vector<size_t> epoch_rows(whole > 0 ? whole - 1 : 0, n);
    if (remainder) {
        epoch_rows.push_back(remainder);
    }
    if (whole) {
        epoch_rows.push_back(n);
    }
    FAISS_THROW_IF_NOT_MSG(
            !epoch_rows.empty(), "e-means epoch budget rounds to zero rows");
    const double coefficient = d <= 16 ? 0.00439175 : 0.035134;
    const double raw_batch = coefficient * std::pow(k, 0.134749) *
            std::pow(n, 0.876473) * std::pow(epochs, 0.813344);
    const size_t batch_cap = d <= 16 ? n / 2 : round_power_of_two(n / 2.0);
    const size_t batch_size = std::min(
            n,
            std::max(
                    size_t{1},
                    std::min(round_power_of_two(raw_batch), batch_cap)));
    const double alpha0 = std::clamp(
            0.8244 * std::pow(k, -0.4617) * std::pow(batch_size, 0.9003) *
                    std::pow(n, -0.4498) * std::pow(epochs, -0.1057),
            1e-4,
            1.0);
    size_t total_steps = 0;
    for (auto rows : epoch_rows) {
        const auto steps = rows / batch_size + (rows % batch_size != 0);
        FAISS_THROW_IF_NOT_MSG(
                steps <= std::numeric_limits<size_t>::max() - total_steps,
                "e-means update count overflow");
        total_steps += steps;
    }
    double best_loss = std::numeric_limits<double>::infinity();
    std::vector<float> best;
    for (int redo = 0; redo < cp_.nredo; ++redo) {
        std::vector<float> positions = initial;
        if (positions.empty()) {
            positions.resize(k * d);
            auto seeds = sample;
            std::shuffle(seeds.begin(), seeds.end(), rng);
            for (size_t i = 0; i < k; ++i) {
                std::copy_n(x + seeds[i] * d, d, positions.data() + i * d);
            }
        }
        std::vector<double> mass(k, std::ldexp(1.0, -16)), sums(k * d),
                closing_sums(k * d);
        std::vector<size_t> counts(k), closing_counts(k), order = sample;
        const size_t chunk_size = std::min(size_t{4096}, batch_size);
        std::vector<float> chunk(chunk_size * d), distances(chunk_size),
                readout;
        std::vector<idx_t> labels(chunk_size);
        IndexFlat assigner(d, cp_.spherical ? METRIC_INNER_PRODUCT : METRIC_L2);
        size_t step = 0;
        for (size_t epoch = 0; epoch < epoch_rows.size(); ++epoch) {
            std::shuffle(order.begin(), order.end(), rng);
            const auto rows = epoch_rows[epoch];
            const bool closing = epoch + 1 == epoch_rows.size();
            for (size_t start = 0; start < rows; start += batch_size) {
                InterruptCallback::check();
                std::fill(sums.begin(), sums.end(), 0.0);
                std::fill(counts.begin(), counts.end(), 0);
                readout = positions;
                if (cp_.spherical) {
                    fvec_renorm_L2(d, k, readout.data());
                }
                assigner.reset();
                assigner.add(k, readout.data());
                const auto end = std::min(rows, start + batch_size);
                for (size_t offset = start; offset < end;
                     offset += chunk_size) {
                    const auto size = std::min(chunk_size, end - offset);
                    for (size_t i = 0; i < size; ++i) {
                        std::copy_n(
                                x + order[offset + i] * d,
                                d,
                                chunk.data() + i * d);
                    }
                    assigner.search(
                            size,
                            chunk.data(),
                            1,
                            distances.data(),
                            labels.data());
                    for (size_t i = 0; i < size; ++i) {
                        FAISS_THROW_IF_NOT_MSG(
                                labels[i] >= 0 &&
                                        static_cast<size_t>(labels[i]) < k &&
                                        std::isfinite(distances[i]),
                                "e-means assignment failed: non-finite distance");
                        const auto id = static_cast<size_t>(labels[i]);
                        ++counts[id];
                        for (size_t j = 0; j < d; ++j) {
                            sums[id * d + j] += chunk[i * d + j];
                        }
                    }
                }
                const double alpha = alpha0 *
                        (0.1 +
                         0.45 *
                                 (1 +
                                  std::cos(
                                          std::acos(-1.0) * step /
                                          total_steps)));
                for (size_t i = 0; i < k; ++i) {
                    if (alpha != 1.0) {
                        mass[i] *= 1 - alpha;
                    }
                    if (!counts[i]) {
                        continue;
                    }
                    mass[i] =
                            (alpha == 1.0 ? 0.0 : mass[i]) + alpha * counts[i];
                    const double weight = alpha * counts[i] / mass[i];
                    for (size_t j = 0; j < d; ++j) {
                        const double mean = sums[i * d + j] / counts[i];
                        const double old = positions[i * d + j];
                        positions[i * d + j] = old + weight * (mean - old);
                        if (closing) {
                            closing_sums[i * d + j] += sums[i * d + j];
                        }
                    }
                    if (closing) {
                        closing_counts[i] += counts[i];
                    }
                }
                if (++step % 5 == 0) {
                    revive(positions, mass, d, rng);
                }
            }
        }
        for (size_t i = 0; i < k; ++i) {
            if (closing_counts[i]) {
                for (size_t j = 0; j < d; ++j) {
                    positions[i * d + j] =
                            closing_sums[i * d + j] / closing_counts[i];
                }
            }
        }
        if (cp_.spherical) {
            fvec_renorm_L2(d, k, positions.data());
        }
        assigner.reset();
        assigner.add(k, positions.data());
        double loss = 0;
        for (size_t offset = 0; offset < n; offset += chunk_size) {
            const auto size = std::min(chunk_size, n - offset);
            for (size_t i = 0; i < size; ++i) {
                std::copy_n(
                        x + sample[offset + i] * d, d, chunk.data() + i * d);
            }
            assigner.search(
                    size, chunk.data(), 1, distances.data(), labels.data());
            for (size_t i = 0; i < size; ++i) {
                loss += cp_.spherical ? -distances[i] : distances[i];
            }
        }
        FAISS_THROW_IF_NOT_MSG(
                std::isfinite(loss), "e-means final objective is not finite");
        if (loss < best_loss) {
            best_loss = loss;
            best = std::move(positions);
        }
    }
    centroids = std::move(best);
}

} // namespace faiss
