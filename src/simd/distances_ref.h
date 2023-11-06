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

/// compute ny square L2 distance between x and a set of transposed contiguous
/// y vectors. squared lengths of y should be provided as well
void
fvec_L2sqr_ny_transposed_ref(float* dis, const float* x, const float* y, const float* y_sqlen, size_t d,
                             size_t d_offset, size_t ny);

/// compute ny square L2 distance between x and a set of contiguous y vectors
/// and return the index of the nearest vector.
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_ref(float* distances_tmp_buffer, const float* x, const float* y, size_t d, size_t ny);

/// compute ny square L2 distance between x and a set of transposed contiguous
/// y vectors and return the index of the nearest vector.
/// squared lengths of y should be provided as well
/// return 0 if ny == 0.
size_t
fvec_L2sqr_ny_nearest_y_transposed_ref(float* distances_tmp_buffer, const float* x, const float* y,
                                       const float* y_sqlen, size_t d, size_t d_offset, size_t ny);

void
fvec_madd_ref(size_t n, const float* a, float bf, const float* b, float* c);

int
fvec_madd_and_argmin_ref(size_t n, const float* a, float bf, const float* b, float* c);

/// Special version of inner product that computes 4 distances
/// between x and yi, which is performance oriented.
void
fvec_inner_product_batch_4_ref(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                               const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

/// Special version of L2sqr that computes 4 distances
/// between x and yi, which is performance oriented.
void
fvec_L2sqr_batch_4_ref(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                       const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

}  // namespace faiss

#endif /* DISTANCES_REF_H */
