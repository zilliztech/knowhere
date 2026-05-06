/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/cppcontrib/knowhere/IndexCosine.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

using IndexIVFPQFastScan = ::faiss::IndexIVFPQFastScan;

struct IndexIVFPQFastScanCosine
        : ::faiss::IndexIVFPQFastScan,
          HasInverseL2Norms {
    L2NormsStorage inverse_norms_storage;

    IndexIVFPQFastScanCosine(
            Index* quantizer,
            size_t d,
            size_t nlist,
            size_t M,
            size_t nbits,
            MetricType metric = METRIC_L2,
            int bbs = 32);

    IndexIVFPQFastScanCosine();

    void train(idx_t n, const float* x) override;
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;
    const float* get_inverse_l2_norms() const override;
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
