/*
 * Copyright (c) Knowhere contributors.
 * Licensed under the MIT license in the root directory of this source tree.
 */

#pragma once

#include <faiss/Clustering.h>

namespace faiss {

/** Native CPU e-means (https://arxiv.org/abs/2611.09501, Algorithm 1).
 *
 * Uses the FAISS 1.16.0 reference's batch/step laws, cosine step decay,
 * mass-weighted donor revival, and last-pass closing means. FP32 coordinates
 * are accumulated in FP64. Random permutations are implementation-specific.
 * Supports L2 and spherical IP; input normalization is the caller's job.
 * max_points_per_centroid and nredo are honored. niter/early stopping do not
 * control e-means: e_means_epochs is the epoch budget, including fractions.
 */
struct EMeans {
    EMeans(int d, int k, const ClusteringParameters& cp);

    /// Optional full initial centroid table on input; trained table on output.
    std::vector<float> centroids;

    /// Train dense finite FP32 vectors. No encoded or weighted input path.
    void train(idx_t n, const float* x);

   private:
    int d_;
    int k_;
    ClusteringParameters cp_;
};

} // namespace faiss
