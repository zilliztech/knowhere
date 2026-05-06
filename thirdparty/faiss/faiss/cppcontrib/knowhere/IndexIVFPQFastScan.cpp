/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/cppcontrib/knowhere/IndexIVFPQFastScan.h>

#include <cstring>
#include <memory>

#include <knowhere/utils.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

IndexIVFPQFastScanCosine::IndexIVFPQFastScanCosine(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits,
        MetricType metric,
        int bbs)
        : ::faiss::IndexIVFPQFastScan(
                  quantizer,
                  d,
                  nlist,
                  M,
                  nbits,
                  metric,
                  bbs) {}

IndexIVFPQFastScanCosine::IndexIVFPQFastScanCosine() = default;

void IndexIVFPQFastScanCosine::train(idx_t n, const float* x) {
    auto norm_data = std::make_unique<float[]>(n * d);
    std::memcpy(norm_data.get(), x, n * d * sizeof(float));
    ::knowhere::NormalizeVecs(norm_data.get(), n, d);
    ::faiss::IndexIVFPQFastScan::train(n, norm_data.get());
}

void IndexIVFPQFastScanCosine::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    auto norm_data = std::make_unique<float[]>(n * d);
    std::memcpy(norm_data.get(), x, n * d * sizeof(float));
    auto l2_norms = ::knowhere::NormalizeVecs(norm_data.get(), n, d);
    inverse_norms_storage.add_l2_norms(l2_norms.data(), n);
    ::faiss::IndexIVFPQFastScan::add_with_ids(n, norm_data.get(), xids);
}

const float* IndexIVFPQFastScanCosine::get_inverse_l2_norms() const {
    return inverse_norms_storage.inverse_l2_norms.data();
}

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
