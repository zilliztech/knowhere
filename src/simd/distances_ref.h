#ifndef DISTANCES_REF_H
#define DISTANCES_REF_H

#include <cstdio>

namespace faiss {

/// Squared L2 distance between two vectors
float
fvec_L2sqr_ref(const float* x, const float* y, size_t d);

/// inner product
float
fvec_inner_product_ref(const float* x, const float* y, size_t d);

/// L1 distance
float
fvec_L1_ref(const float* x, const float* y, size_t d);

/// infinity distance
float
fvec_Linf_ref(const float* x, const float* y, size_t d);

/// squared norm of a vector
float
fvec_norm_L2sqr_ref(const float* x, size_t d);

/// compute ny square L2 distance between x and a set of contiguous y vectors
void
fvec_L2sqr_ny_ref(float* dis, const float* x, const float* y, size_t d, size_t ny);

/// compute the inner product between nx vectors x and one y
void
fvec_inner_products_ny_ref(float* ip, const float* x, const float* y, size_t d, size_t ny);

void
fvec_madd_ref(size_t n, const float* a, float bf, const float* b, float* c);

int
fvec_madd_and_argmin_ref(size_t n, const float* a, float bf, const float* b, float* c);

}  // namespace faiss

#endif /* DISTANCES_REF_H */
