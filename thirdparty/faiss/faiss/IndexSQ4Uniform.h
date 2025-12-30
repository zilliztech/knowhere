// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

// 4-bit Uniform Scalar Quantization indices with COSINE and IP support

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/IndexCosine.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/DistanceComputer.h>

namespace faiss {

//////////////////////////////////////////////////////////////////////////////////
// Distance Computers for SQ4Uniform
//////////////////////////////////////////////////////////////////////////////////

/**
 * Distance computer for SQ4Uniform with COSINE metric.
 * Normalizes queries and assumes base vectors are normalized (||b||=1).
 * Converts L2^2 distance to cosine similarity: cosine = 1 - 0.5 * L2^2
 */
struct SQ4UniformCosineDistanceComputer : DistanceComputer {
    /// owned by this
    std::unique_ptr<DistanceComputer> basedis;

    /// cached dimensionality
    int d = 0;

    /// Storage for normalized query (needed to keep data alive)
    std::vector<float> query_storage;

    SQ4UniformCosineDistanceComputer(
            const int d_,
            std::unique_ptr<DistanceComputer>&& basedis_);

    /// Set query vector (normalizes internally)
    void set_query(const float* x) override;

    /// Compute cosine similarity to vector i
    float operator()(idx_t i) override;

    /// Compute distance batch
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override;

    /// Compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override;
};

/**
 * Distance computer for SQ4Uniform with IP metric.
 * Stores L2 norms squared of base vectors to convert L2^2 to IP.
 * Formula: IP(q,b) = 0.5 * (||q||^2 + ||b||^2 - L2^2(q,b))
 */
struct WithSQ4UniformNormIPDistanceComputer : DistanceComputer {
    /// owned by this
    std::unique_ptr<DistanceComputer> basedis;

    /// not owned by this - L2 norms squared of base vectors (||b||^2)
    const float* l2_norms_sqr = nullptr;

    /// computed internally - query norm squared ||q||^2
    float query_norm_sqr = 0;

    /// cached dimensionality
    int d = 0;

    WithSQ4UniformNormIPDistanceComputer(
            const float* l2_norms_sqr_,
            const int d_,
            std::unique_ptr<DistanceComputer>&& basedis_);

    /// Set query vector and compute its norm squared
    void set_query(const float* x) override;

    /// Compute IP distance to vector i
    float operator()(idx_t i) override;

    /// Compute distance batch
    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override;

    /// Compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override;
};

//////////////////////////////////////////////////////////////////////////////////
// Scalar Quantizer Storage Classes
//////////////////////////////////////////////////////////////////////////////////

/**
 * Scalar Quantizer specialized for 4-bit uniform quantization with COSINE
 * metric. Normalizes vectors internally during train() and add() to avoid
 * modifying caller's data. Query normalization handled by knowhere layer.
 * Implements HasInverseL2Norms.
 */
struct IndexScalarQuantizer4bitUniformCosine : IndexScalarQuantizer,
                                               HasInverseL2Norms {
    /// Storage for inverse L2 norms (all 1.0 for normalized vectors)
    mutable std::vector<float> inverse_l2_norms;

    IndexScalarQuantizer4bitUniformCosine(int d);
    IndexScalarQuantizer4bitUniformCosine();

    /// Train on data (normalizes internally without modifying input)
    void train(idx_t n, const float* x) override;

    /// Add vectors (normalizes internally without modifying input)
    void add(idx_t n, const float* x) override;

    /// Get distance computer for COSINE metric
    DistanceComputer* get_distance_computer() const override;

    /// Return inverse L2 norms (all 1.0 since vectors are normalized)
    const float* get_inverse_l2_norms() const override;

    /// Reset the index
    void reset() override;
};

/**
 * Scalar Quantizer specialized for 4-bit uniform quantization with IP metric.
 * Stores L2 norms squared of vectors to convert L2^2 distances to IP.
 */
struct IndexScalarQuantizer4bitUniformIP : IndexScalarQuantizer {
    /// Storage for L2 norms squared (||x||^2)
    std::vector<float> l2_norms_sqr;

    IndexScalarQuantizer4bitUniformIP(int d);
    IndexScalarQuantizer4bitUniformIP();

    /// Add vectors and compute their norms squared
    void add(idx_t n, const float* x) override;

    /// Reset the index
    void reset() override;

    /// Get distance computer for IP metric
    DistanceComputer* get_distance_computer() const override;

    /// Get L2 norms squared for distance computation
    const float* get_l2_norms_sqr() const;
};

//////////////////////////////////////////////////////////////////////////////////
// HNSW Wrapper Classes
//////////////////////////////////////////////////////////////////////////////////

struct IndexHNSWSQ4UniformCosine : IndexHNSW {
    IndexHNSWSQ4UniformCosine();

    IndexHNSWSQ4UniformCosine(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            int M);
};

struct IndexHNSWSQ4UniformIP : IndexHNSW {
    IndexHNSWSQ4UniformIP();

    IndexHNSWSQ4UniformIP(int d, ScalarQuantizer::QuantizerType qtype, int M);
};

} // namespace faiss
