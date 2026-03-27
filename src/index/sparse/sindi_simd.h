#pragma once

#include <cstdint>

#include "knowhere/bitsetview.h"
#include "knowhere/heap.h"
#include "knowhere/operands.h"

namespace knowhere::sparse::inverted::sindi {

using ip_accumulate_fn_t = float (*)(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num,
                                     float* out);

using bm25_accumulate_fn_t = float (*)(float qval, const uint16_t* tf_vals, const uint16_t* ids, int32_t num,
                                       float* out, float k1, float b, float avgdl, const float* row_sums);

using batch_insert_fn_t = void (*)(const float* scores, size_t docid_start, size_t count,
                                   knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold,
                                   const BitsetView& bitset);

struct IPKernels {
    ip_accumulate_fn_t accumulate;
    batch_insert_fn_t batch_insert;
};

struct BM25Kernels {
    bm25_accumulate_fn_t accumulate;
    batch_insert_fn_t batch_insert;
};

const IPKernels&
get_ip_kernels();
const BM25Kernels&
get_bm25_kernels();

// Scalar implementations (always available)
float
ip_scatter_scalar_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out);
float
bm25_scatter_scalar_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                        float b, float avgdl, const float* row_sums);
void
batch_insert_scalar(const float* scores, size_t docid_start, size_t count,
                    knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset);

#if defined(__x86_64__)
// AVX2 implementations (compiled separately with -mavx2)
float
ip_scatter_avx2_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out);
float
bm25_scatter_avx2_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1, float b,
                      float avgdl, const float* row_sums);
void
batch_insert_avx2(const float* scores, size_t docid_start, size_t count,
                  knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset);

// AVX512 implementations (compiled separately with -mavx512f)
float
ip_scatter_avx512_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out);
float
bm25_scatter_avx512_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                        float b, float avgdl, const float* row_sums);
void
batch_insert_avx512(const float* scores, size_t docid_start, size_t count,
                    knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset);
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
// SVE implementations (compiled with SVE support)
float
ip_scatter_sve_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out);
float
bm25_scatter_sve_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1, float b,
                     float avgdl, const float* row_sums);
void
batch_insert_sve(const float* scores, size_t docid_start, size_t count,
                 knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset);
#endif

}  // namespace knowhere::sparse::inverted::sindi
