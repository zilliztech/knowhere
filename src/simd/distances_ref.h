#ifndef DISTANCES_REF_H
#define DISTANCES_REF_H

#include <cstdint>
#include <cstdio>

#include "knowhere/operands.h"

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

///////////////////////////////////////////////////////////////////////////////
// for hnsw sq, obsolete
int32_t
ivec_inner_product_ref(const int8_t* x, const int8_t* y, size_t d);

int32_t
ivec_L2sqr_ref(const int8_t* x, const int8_t* y, size_t d);

///////////////////////////////////////////////////////////////////////////////
// fp16
float
fp16_vec_inner_product_ref(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_L2sqr_ref(const knowhere::fp16* x, const knowhere::fp16* y, size_t d);

float
fp16_vec_norm_L2sqr_ref(const knowhere::fp16* x, size_t d);

void
fp16_vec_inner_product_batch_4_ref(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                                   const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3);

void
fp16_vec_L2sqr_batch_4_ref(const knowhere::fp16* x, const knowhere::fp16* y0, const knowhere::fp16* y1,
                           const knowhere::fp16* y2, const knowhere::fp16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3);

///////////////////////////////////////////////////////////////////////////////
// bf16
float
bf16_vec_inner_product_ref(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_L2sqr_ref(const knowhere::bf16* x, const knowhere::bf16* y, size_t d);

float
bf16_vec_norm_L2sqr_ref(const knowhere::bf16* x, size_t d);

void
bf16_vec_inner_product_batch_4_ref(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                                   const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0,
                                   float& dis1, float& dis2, float& dis3);

void
bf16_vec_L2sqr_batch_4_ref(const knowhere::bf16* x, const knowhere::bf16* y0, const knowhere::bf16* y1,
                           const knowhere::bf16* y2, const knowhere::bf16* y3, const size_t d, float& dis0, float& dis1,
                           float& dis2, float& dis3);

///////////////////////////////////////////////////////////////////////////////
// for cardinal
float
fvec_L2sqr_bf16_patch_ref(const float* x, const float* y, size_t d);

float
fvec_inner_product_bf16_patch_ref(const float* x, const float* y, size_t d);

void
fvec_inner_product_batch_4_bf16_patch_ref(const float* x, const float* y0, const float* y1, const float* y2,
                                          const float* y3, const size_t d, float& dis0, float& dis1, float& dis2,
                                          float& dis3);

void
fvec_L2sqr_batch_4_bf16_patch_ref(const float* x, const float* y0, const float* y1, const float* y2, const float* y3,
                                  const size_t d, float& dis0, float& dis1, float& dis2, float& dis3);

}  // namespace faiss

#endif /* DISTANCES_REF_H */
