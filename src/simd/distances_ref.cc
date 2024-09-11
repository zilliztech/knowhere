/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "distances_ref.h"

#include <cmath>
namespace faiss {

float
fvec_L2sqr_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

float
fvec_L1_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += std::fabs(tmp);
    }
    return res;
}

float
fvec_Linf_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        res = std::fmax(res, std::fabs(x[i] - y[i]));
    }
    return res;
}

float
fvec_inner_product_ref(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) res += x[i] * y[i];
    return res;
}

float
fvec_norm_L2sqr_ref(const float* x, size_t d) {
    size_t i;
    double res = 0;
    for (i = 0; i < d; i++) res += x[i] * x[i];
    return res;
}

void
fvec_L2sqr_ny_ref(float* dis, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        dis[i] = fvec_L2sqr_ref(x, y, d);
        y += d;
    }
}

void
fvec_inner_products_ny_ref(float* ip, const float* x, const float* y, size_t d, size_t ny) {
    for (size_t i = 0; i < ny; i++) {
        ip[i] = fvec_inner_product_ref(x, y, d);
        y += d;
    }
}

void
fvec_madd_ref(size_t n, const float* a, float bf, const float* b, float* c) {
    for (size_t i = 0; i < n; i++) c[i] = a[i] + bf * b[i];
}

int
fvec_madd_and_argmin_ref(size_t n, const float* a, float bf, const float* b, float* c) {
    float vmin = 1e20;
    int imin = -1;

    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + bf * b[i];
        if (c[i] < vmin) {
            vmin = c[i];
            imin = i;
        }
    }
    return imin;
}

}  // namespace faiss
