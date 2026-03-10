#pragma once

#include <algorithm>
#include <cstdint>

#include "knowhere/bitsetview.h"
#include "knowhere/heap.h"
#include "knowhere/operands.h"
#include "simd/hook.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#define SINDI_USE_SVE 1
#elif defined(__x86_64__)
#include <immintrin.h>
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__F16C__)
#define SINDI_USE_AVX512 1
#endif
#if defined(__AVX2__) && defined(__F16C__)
#define SINDI_USE_AVX2 1
#endif
#endif

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

static inline float
ip_scatter_scalar_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    float max_val = 0.0f;
    for (int32_t i = 0; i < num; ++i) {
        float new_val = (out[ids[i]] += qval * static_cast<float>(vals[i]));
        if (new_val > max_val) {
            max_val = new_val;
        }
    }
    return max_val;
}

#if SINDI_USE_AVX512
static inline float
ip_scatter_avx512_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    int32_t i = 0;
    const __m512 vq512 = _mm512_set1_ps(qval);
    const __m256 vq256 = _mm256_set1_ps(qval);
    __m512 v_max = _mm512_setzero_ps();
    __m256 v_max256 = _mm256_setzero_ps();
    for (; i + 16 <= num; i += 16) {
        const uint16_t* hptr = reinterpret_cast<const uint16_t*>(vals + i);
        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hptr));
        __m512 v_vals = _mm512_cvtph_ps(h);
        __m512 v_mul = _mm512_mul_ps(v_vals, vq512);

        __m256i idx16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids + i));
        __m512i v_idx = _mm512_cvtepu16_epi32(idx16);
        __m512 v_old = _mm512_i32gather_ps(v_idx, out, 4);
        __m512 v_sum = _mm512_add_ps(v_old, v_mul);
        _mm512_i32scatter_ps(out, v_idx, v_sum, 4);
        v_max = _mm512_max_ps(v_max, v_sum);
    }
    for (; i + 8 <= num; i += 8) {
        const uint16_t* hptr = reinterpret_cast<const uint16_t*>(vals + i);
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hptr));
        __m256 v_vals = _mm256_cvtph_ps(h);
        __m256 v_mul = _mm256_mul_ps(v_vals, vq256);
        __m128i idx16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i v_idx = _mm256_cvtepu16_epi32(idx16);
        __m256 v_old = _mm256_i32gather_ps(out, v_idx, 4);
        __m256 v_sum = _mm256_add_ps(v_old, v_mul);
        _mm256_i32scatter_ps(out, v_idx, v_sum, 4);
        v_max256 = _mm256_max_ps(v_max256, v_sum);
    }
    float max_val = _mm512_reduce_max_ps(v_max);
    // reduce v_max256 to scalar
    __m128 v_max128 = _mm_max_ps(_mm256_castps256_ps128(v_max256), _mm256_extractf128_ps(v_max256, 1));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(2, 3, 0, 1)));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(1, 0, 3, 2)));
    float max256_scalar = _mm_cvtss_f32(v_max128);
    if (max256_scalar > max_val) {
        max_val = max256_scalar;
    }
    for (; i < num; ++i) {
        float new_val = (out[ids[i]] += qval * static_cast<float>(vals[i]));
        if (new_val > max_val) {
            max_val = new_val;
        }
    }
    return max_val;
}
#endif

#if SINDI_USE_SVE
static inline float
ip_scatter_sve_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    const svfloat32_t vq32 = svdup_f32(qval);
    const uint32_t vl32 = svcntw();  // number of 32-bit lanes
    const svbool_t pg32 = svptrue_b32();
    const svbool_t pg16 = svptrue_b16();
    svfloat32_t v_max = svdup_f32(0.0f);

    int32_t i = 0;

    // Main loop: process 2*vl32 fp16 elements per iteration (vl32 in even pass + vl32 in odd pass)
    const int32_t step = static_cast<int32_t>(vl32 * 2);
    for (; i + step <= num; i += step) {
        const __fp16* hptr = reinterpret_cast<const __fp16*>(vals + i);

        // Load fp16 values as full vector
        svfloat16_t vh = svld1_f16(pg16, hptr);

        // Convert even lanes (0,2,4,...) to fp32 - svcvt extracts from even positions
        svfloat32_t vf_even = svcvt_f32_f16_x(pg32, vh);

        // Shift by 1 to get odd lanes (1,3,5,...) into even positions, then convert
        svfloat16_t vh_shift = svext_f16(vh, vh, 1);
        svfloat32_t vf_odd = svcvt_f32_f16_x(pg32, vh_shift);

        // Load indices and de-interleave
        svuint16_t id16 = svld1_u16(pg16, ids + i);
        svuint16_t id_even16 = svuzp1_u16(id16, id16);
        svuint16_t id_odd16 = svuzp2_u16(id16, id16);
        svuint32_t vidx_even = svunpklo_u32(id_even16);
        svuint32_t vidx_odd = svunpklo_u32(id_odd16);

        // Even lanes: gather, FMA, scatter
        svfloat32_t vold_even = svld1_gather_u32index_f32(pg32, out, vidx_even);
        svfloat32_t vsum_even = svmad_f32_x(pg32, vf_even, vq32, vold_even);
        svst1_scatter_u32index_f32(pg32, out, vidx_even, vsum_even);
        v_max = svmax_f32_x(pg32, v_max, vsum_even);

        // Odd lanes: gather, FMA, scatter
        svfloat32_t vold_odd = svld1_gather_u32index_f32(pg32, out, vidx_odd);
        svfloat32_t vsum_odd = svmad_f32_x(pg32, vf_odd, vq32, vold_odd);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd, vsum_odd);
        v_max = svmax_f32_x(pg32, v_max, vsum_odd);
    }

    // Handle remaining elements
    if (i < num) {
        int32_t remaining = num - i;
        svbool_t pg16_tail = svwhilelt_b16(static_cast<uint32_t>(0), static_cast<uint32_t>(remaining));

        const __fp16* hptr = reinterpret_cast<const __fp16*>(vals + i);
        svfloat16_t vh = svld1_f16(pg16_tail, hptr);
        svfloat16_t vh_shift = svext_f16(vh, vh, 1);

        uint32_t n_even = static_cast<uint32_t>((remaining + 1) >> 1);
        uint32_t n_odd = static_cast<uint32_t>(remaining >> 1);

        svbool_t pg32_even = svwhilelt_b32(static_cast<uint32_t>(0), n_even);
        svbool_t pg32_odd = svwhilelt_b32(static_cast<uint32_t>(0), n_odd);

        svfloat32_t vf_even = svcvt_f32_f16_x(pg32_even, vh);
        svfloat32_t vf_odd = svcvt_f32_f16_x(pg32_odd, vh_shift);

        svuint16_t id16 = svld1_u16(pg16_tail, ids + i);
        svuint16_t id_even16 = svuzp1_u16(id16, id16);
        svuint16_t id_odd16 = svuzp2_u16(id16, id16);
        svuint32_t vidx_even = svunpklo_u32(id_even16);
        svuint32_t vidx_odd = svunpklo_u32(id_odd16);

        if (n_even) {
            svfloat32_t vold_even = svld1_gather_u32index_f32(pg32_even, out, vidx_even);
            svfloat32_t vsum_even = svmad_f32_x(pg32_even, vf_even, vq32, vold_even);
            svst1_scatter_u32index_f32(pg32_even, out, vidx_even, vsum_even);
            v_max = svmax_f32_m(pg32_even, v_max, vsum_even);
        }

        if (n_odd) {
            svfloat32_t vold_odd = svld1_gather_u32index_f32(pg32_odd, out, vidx_odd);
            svfloat32_t vsum_odd = svmad_f32_x(pg32_odd, vf_odd, vq32, vold_odd);
            svst1_scatter_u32index_f32(pg32_odd, out, vidx_odd, vsum_odd);
            v_max = svmax_f32_m(pg32_odd, v_max, vsum_odd);
        }
    }

    return svmaxv_f32(svptrue_b32(), v_max);
}
#endif

#if SINDI_USE_AVX2
static inline float
ip_scatter_avx2_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    int32_t i = 0;
    const __m256 vq = _mm256_set1_ps(qval);
    __m256 v_max = _mm256_setzero_ps();
    for (; i + 8 <= num; i += 8) {
        // Load 8 half values and convert to float
        const uint16_t* hptr = reinterpret_cast<const uint16_t*>(vals + i);
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hptr));
        __m256 v_vals = _mm256_cvtph_ps(h);
        __m256 v_mul = _mm256_mul_ps(v_vals, vq);

        __m128i idx16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i v_idx = _mm256_cvtepu16_epi32(idx16);
        __m256 v_old = _mm256_i32gather_ps(out, v_idx, 4);
        __m256 v_sum = _mm256_add_ps(v_old, v_mul);

        // No AVX2 scatter; write back lane-by-lane
        alignas(32) uint32_t tmp_idx[8];
        alignas(32) float tmp_sum[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp_idx), v_idx);
        _mm256_store_ps(tmp_sum, v_sum);
        out[tmp_idx[0]] = tmp_sum[0];
        out[tmp_idx[1]] = tmp_sum[1];
        out[tmp_idx[2]] = tmp_sum[2];
        out[tmp_idx[3]] = tmp_sum[3];
        out[tmp_idx[4]] = tmp_sum[4];
        out[tmp_idx[5]] = tmp_sum[5];
        out[tmp_idx[6]] = tmp_sum[6];
        out[tmp_idx[7]] = tmp_sum[7];
        v_max = _mm256_max_ps(v_max, v_sum);
    }
    __m128 v_max128 = _mm_max_ps(_mm256_castps256_ps128(v_max), _mm256_extractf128_ps(v_max, 1));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(2, 3, 0, 1)));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(v_max128);
    for (; i < num; ++i) {
        float new_val = (out[ids[i]] += qval * static_cast<float>(vals[i]));
        if (new_val > max_val) {
            max_val = new_val;
        }
    }
    return max_val;
}
#endif

static inline float
bm25_scatter_scalar_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                        float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    float max_val = 0.0f;
    for (int32_t i = 0; i < num; ++i) {
        float tf = static_cast<float>(vals[i]);
        uint16_t docid = ids[i];
        float dl = row_sums[docid];
        // Full BM25: qval * (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl / avgdl))
        float bm25_score = qval * p1 * tf / (tf + p2 + p3 * dl);
        float new_val = (out[docid] += bm25_score);
        if (new_val > max_val) {
            max_val = new_val;
        }
    }
    return max_val;
}

#ifdef SINDI_USE_AVX512
static inline float
bm25_scatter_avx512_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                        float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    int32_t i = 0;
    const __m512 vqval = _mm512_set1_ps(qval);
    const __m512 vp1 = _mm512_set1_ps(p1);
    const __m512 vp2 = _mm512_set1_ps(p2);
    const __m512 vp3 = _mm512_set1_ps(p3);
    __m512 v_max = _mm512_setzero_ps();

    // Process 16 elements at a time
    for (; i + 16 <= num; i += 16) {
        const uint16_t* hptr = vals + i;
        _mm_prefetch(reinterpret_cast<const char*>(hptr + 32), _MM_HINT_NTA);

        // Load and convert 16 uint16_t term frequencies to float (single operation)
        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hptr));
        __m512i w = _mm512_cvtepu16_epi32(h);
        __m512 tf_vec = _mm512_cvtepi32_ps(w);

        // Load document IDs and gather document lengths
        __m256i idx16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids + i));
        __m512i v_idx = _mm512_cvtepu16_epi32(idx16);
        __m512 dl_vec = _mm512_i32gather_ps(v_idx, row_sums, 4);

        // BM25 calculation: qval * p1 * tf / (tf + p2 + p3 * dl)
        __m512 numerator = _mm512_mul_ps(tf_vec, vp1);
        numerator = _mm512_mul_ps(numerator, vqval);

        __m512 denominator = _mm512_fmadd_ps(dl_vec, vp3, vp2);
        denominator = _mm512_add_ps(tf_vec, denominator);

        __m512 bm25_vec = _mm512_div_ps(numerator, denominator);

        // Gather old values, add, and scatter
        __m512 v_old = _mm512_i32gather_ps(v_idx, out, 4);
        __m512 v_sum = _mm512_add_ps(v_old, bm25_vec);
        _mm512_i32scatter_ps(out, v_idx, v_sum, 4);
        v_max = _mm512_max_ps(v_max, v_sum);
    }

    float max_val = _mm512_reduce_max_ps(v_max);

    // Handle remaining elements with scalar code
    for (; i < num; ++i) {
        float tf = static_cast<float>(vals[i]);
        uint16_t docid = ids[i];
        float dl = row_sums[docid];
        float bm25_score = qval * p1 * tf / (tf + p2 + p3 * dl);
        float new_val = (out[docid] += bm25_score);
        if (new_val > max_val) {
            max_val = new_val;
        }
    }
    return max_val;
}
#endif

#if SINDI_USE_SVE
// Fast reciprocal with Newton-Raphson refinement for better accuracy
static inline svfloat32_t
sv_fast_recip_f32(svbool_t pg, svfloat32_t x) {
    // Initial approximation (~8-12 bits precision)
    svfloat32_t recip = svrecpe_f32(x);
    // Newton-Raphson step using dedicated instruction
    // svrecps_f32(a, b) computes (2 - a * b) optimally
    svfloat32_t step = svrecps_f32(x, recip);
    recip = svmul_f32_x(pg, recip, step);
    return recip;
}

static inline float
bm25_scatter_sve_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1, float b,
                     float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    const svfloat32_t vp2 = svdup_f32(p2);
    const svfloat32_t vp3 = svdup_f32(p3);
    const svfloat32_t vqp1 = svdup_f32(qval * p1);  // Pre-compute qval * p1
    svfloat32_t v_max = svdup_f32(0.0f);

    // Get the number of 32-bit lanes
    const uint32_t vl32 = svcntw();
    const svbool_t pg32 = svptrue_b32();
    const svbool_t pg16 = svptrue_b16();

    int32_t i = 0;

    // Main loop: process 2*vl32 u16 elements per iteration (vl32 in even pass + vl32 in odd pass)
    const int32_t step = static_cast<int32_t>(vl32 * 2);
    for (; i + step <= num; i += step) {
        // Load u16 term frequencies as full vector
        svuint16_t tf16 = svld1_u16(pg16, vals + i);

        // De-interleave even/odd lanes and widen to u32
        svuint16_t tf_even16 = svuzp1_u16(tf16, tf16);  // [tf0, tf2, tf4, ...]
        svuint16_t tf_odd16 = svuzp2_u16(tf16, tf16);   // [tf1, tf3, tf5, ...]
        svuint32_t tf_even_u32 = svunpklo_u32(tf_even16);
        svuint32_t tf_odd_u32 = svunpklo_u32(tf_odd16);

        // Convert to float
        svfloat32_t tf_even = svcvt_f32_u32_x(pg32, tf_even_u32);
        svfloat32_t tf_odd = svcvt_f32_u32_x(pg32, tf_odd_u32);

        // Load indices and de-interleave
        svuint16_t id16 = svld1_u16(pg16, ids + i);
        svuint16_t id_even16 = svuzp1_u16(id16, id16);
        svuint16_t id_odd16 = svuzp2_u16(id16, id16);
        svuint32_t vidx_even = svunpklo_u32(id_even16);
        svuint32_t vidx_odd = svunpklo_u32(id_odd16);

        // Gather document lengths for even/odd
        svfloat32_t dl_even = svld1_gather_u32index_f32(pg32, row_sums, vidx_even);
        svfloat32_t dl_odd = svld1_gather_u32index_f32(pg32, row_sums, vidx_odd);

        // BM25 calculation for even lanes: qval * p1 * tf / (tf + p2 + p3 * dl)
        svfloat32_t num_even = svmul_f32_x(pg32, tf_even, vqp1);
        svfloat32_t denom_even = svmad_f32_x(pg32, dl_even, vp3, vp2);
        denom_even = svadd_f32_x(pg32, tf_even, denom_even);
        svfloat32_t recip_even = sv_fast_recip_f32(pg32, denom_even);
        svfloat32_t bm25_even = svmul_f32_x(pg32, num_even, recip_even);

        // BM25 calculation for odd lanes
        svfloat32_t num_odd = svmul_f32_x(pg32, tf_odd, vqp1);
        svfloat32_t denom_odd = svmad_f32_x(pg32, dl_odd, vp3, vp2);
        denom_odd = svadd_f32_x(pg32, tf_odd, denom_odd);
        svfloat32_t recip_odd = sv_fast_recip_f32(pg32, denom_odd);
        svfloat32_t bm25_odd = svmul_f32_x(pg32, num_odd, recip_odd);

        // Even lanes: gather, add, scatter
        svfloat32_t vold_even = svld1_gather_u32index_f32(pg32, out, vidx_even);
        svfloat32_t vsum_even = svadd_f32_x(pg32, vold_even, bm25_even);
        svst1_scatter_u32index_f32(pg32, out, vidx_even, vsum_even);
        v_max = svmax_f32_x(pg32, v_max, vsum_even);

        // Odd lanes: gather, add, scatter
        svfloat32_t vold_odd = svld1_gather_u32index_f32(pg32, out, vidx_odd);
        svfloat32_t vsum_odd = svadd_f32_x(pg32, vold_odd, bm25_odd);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd, vsum_odd);
        v_max = svmax_f32_x(pg32, v_max, vsum_odd);
    }

    // Handle remaining elements
    if (i < num) {
        int32_t remaining = num - i;
        svbool_t pg16_tail = svwhilelt_b16(static_cast<uint32_t>(0), static_cast<uint32_t>(remaining));

        svuint16_t tf16 = svld1_u16(pg16_tail, vals + i);
        svuint16_t tf_even16 = svuzp1_u16(tf16, tf16);
        svuint16_t tf_odd16 = svuzp2_u16(tf16, tf16);

        uint32_t n_even = static_cast<uint32_t>((remaining + 1) >> 1);
        uint32_t n_odd = static_cast<uint32_t>(remaining >> 1);

        svbool_t pg32_even = svwhilelt_b32(static_cast<uint32_t>(0), n_even);
        svbool_t pg32_odd = svwhilelt_b32(static_cast<uint32_t>(0), n_odd);

        svuint32_t tf_even_u32 = svunpklo_u32(tf_even16);
        svuint32_t tf_odd_u32 = svunpklo_u32(tf_odd16);
        svfloat32_t tf_even = svcvt_f32_u32_x(pg32_even, tf_even_u32);
        svfloat32_t tf_odd = svcvt_f32_u32_x(pg32_odd, tf_odd_u32);

        svuint16_t id16 = svld1_u16(pg16_tail, ids + i);
        svuint16_t id_even16 = svuzp1_u16(id16, id16);
        svuint16_t id_odd16 = svuzp2_u16(id16, id16);
        svuint32_t vidx_even = svunpklo_u32(id_even16);
        svuint32_t vidx_odd = svunpklo_u32(id_odd16);

        if (n_even) {
            svfloat32_t dl_even = svld1_gather_u32index_f32(pg32_even, row_sums, vidx_even);
            svfloat32_t num_even = svmul_f32_x(pg32_even, tf_even, vqp1);
            svfloat32_t denom_even = svmad_f32_x(pg32_even, dl_even, vp3, vp2);
            denom_even = svadd_f32_x(pg32_even, tf_even, denom_even);
            svfloat32_t recip_even = sv_fast_recip_f32(pg32_even, denom_even);
            svfloat32_t bm25_even = svmul_f32_x(pg32_even, num_even, recip_even);

            svfloat32_t vold_even = svld1_gather_u32index_f32(pg32_even, out, vidx_even);
            svfloat32_t vsum_even = svadd_f32_x(pg32_even, vold_even, bm25_even);
            svst1_scatter_u32index_f32(pg32_even, out, vidx_even, vsum_even);
            v_max = svmax_f32_m(pg32_even, v_max, vsum_even);
        }

        if (n_odd) {
            svfloat32_t dl_odd = svld1_gather_u32index_f32(pg32_odd, row_sums, vidx_odd);
            svfloat32_t num_odd = svmul_f32_x(pg32_odd, tf_odd, vqp1);
            svfloat32_t denom_odd = svmad_f32_x(pg32_odd, dl_odd, vp3, vp2);
            denom_odd = svadd_f32_x(pg32_odd, tf_odd, denom_odd);
            svfloat32_t recip_odd = sv_fast_recip_f32(pg32_odd, denom_odd);
            svfloat32_t bm25_odd = svmul_f32_x(pg32_odd, num_odd, recip_odd);

            svfloat32_t vold_odd = svld1_gather_u32index_f32(pg32_odd, out, vidx_odd);
            svfloat32_t vsum_odd = svadd_f32_x(pg32_odd, vold_odd, bm25_odd);
            svst1_scatter_u32index_f32(pg32_odd, out, vidx_odd, vsum_odd);
            v_max = svmax_f32_m(pg32_odd, v_max, vsum_odd);
        }
    }

    // Reduce v_max to scalar
    return svmaxv_f32(svptrue_b32(), v_max);
}
#endif

#ifdef SINDI_USE_AVX2
static inline float
bm25_scatter_avx2_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1, float b,
                      float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    int32_t i = 0;
    const __m256 vqval = _mm256_set1_ps(qval);
    const __m256 vp1 = _mm256_set1_ps(p1);
    const __m256 vp2 = _mm256_set1_ps(p2);
    const __m256 vp3 = _mm256_set1_ps(p3);
    __m256 v_max = _mm256_setzero_ps();

    // Process 8 elements at a time
    for (; i + 8 <= num; i += 8) {
        // Load and convert 8 uint16_t values to float
        const uint16_t* hptr = vals + i;
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hptr));
        __m256i w = _mm256_cvtepu16_epi32(h);
        __m256 tf_vec = _mm256_cvtepi32_ps(w);

        // Load document IDs and gather document lengths
        __m128i idx16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i v_idx = _mm256_cvtepu16_epi32(idx16);
        __m256 dl_vec = _mm256_i32gather_ps(row_sums, v_idx, 4);

        // BM25 calculation: qval * p1 * tf / (tf + p2 + p3 * dl)
        __m256 numerator = _mm256_mul_ps(tf_vec, vp1);
        numerator = _mm256_mul_ps(numerator, vqval);

        __m256 denominator = _mm256_fmadd_ps(dl_vec, vp3, vp2);
        denominator = _mm256_add_ps(tf_vec, denominator);

        __m256 bm25_vec = _mm256_div_ps(numerator, denominator);

        // Gather old values and add
        __m256 v_old = _mm256_i32gather_ps(out, v_idx, 4);
        __m256 v_sum = _mm256_add_ps(v_old, bm25_vec);

        // No AVX2 scatter; write back lane-by-lane
        alignas(32) uint32_t tmp_idx[8];
        alignas(32) float tmp_sum[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(tmp_idx), v_idx);
        _mm256_store_ps(tmp_sum, v_sum);
        out[tmp_idx[0]] = tmp_sum[0];
        out[tmp_idx[1]] = tmp_sum[1];
        out[tmp_idx[2]] = tmp_sum[2];
        out[tmp_idx[3]] = tmp_sum[3];
        out[tmp_idx[4]] = tmp_sum[4];
        out[tmp_idx[5]] = tmp_sum[5];
        out[tmp_idx[6]] = tmp_sum[6];
        out[tmp_idx[7]] = tmp_sum[7];
        v_max = _mm256_max_ps(v_max, v_sum);
    }

    // Reduce v_max to scalar
    __m128 v_max128 = _mm_max_ps(_mm256_castps256_ps128(v_max), _mm256_extractf128_ps(v_max, 1));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(2, 3, 0, 1)));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(v_max128);

    // Handle remaining elements with scalar code
    for (; i < num; ++i) {
        float tf = static_cast<float>(vals[i]);
        uint16_t docid = ids[i];
        float dl = row_sums[docid];
        float bm25_score = qval * p1 * tf / (tf + p2 + p3 * dl);
        float new_val = (out[docid] += bm25_score);
        if (new_val > max_val) {
            max_val = new_val;
        }
    }
    return max_val;
}
#endif

// Batch insert helper with SIMD prefilter on threshold
static inline void
batch_insert_scalar(const float* scores, size_t docid_start, size_t count,
                    knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset) {
    for (size_t i = 0; i < count; ++i) {
        float s = scores[i];
        // Fast pre-check using current threshold
        if (s <= threshold) {
            continue;
        }
        // bitset filtering: skip masked ids
        if (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + i))) {
            continue;
        }
        if (topk_q.Push(s, static_cast<uint32_t>(docid_start + i))) {
            if (topk_q.Full()) {
                threshold = topk_q.Threshold();
            }
        }
    }
}

#if SINDI_USE_SVE
static inline void
batch_insert_sve(const float* scores, size_t docid_start, size_t count,
                 knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset) {
    const uint32_t vl = svcntw();  // Hoist VL query outside loop
    size_t i = 0;
    svfloat32_t vthr = svdup_f32(threshold);

    // Main SIMD loop
    for (; i + vl <= count; i += vl) {
        svbool_t pg = svptrue_b32();
        svfloat32_t v = svld1(pg, scores + i);
        svbool_t pg_sel = svcmpgt_f32(pg, v, vthr);

        // Check if any elements pass threshold
        if (!svptest_any(pg, pg_sel)) {
            continue;
        }

        // Extract mask bits and process only matching elements
        alignas(64) uint32_t tmp_mask[64];
        svuint32_t ones = svdup_u32(1);
        svuint32_t zeros = svdup_u32(0);
        svuint32_t vmask = svsel_u32(pg_sel, ones, zeros);
        svst1(pg, tmp_mask, vmask);

        for (uint32_t j = 0; j < vl; ++j) {
            if (tmp_mask[j]) {
                size_t idx = i + j;
                // bitset filtering: skip masked ids
                if (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + idx))) {
                    continue;
                }
                float s = scores[idx];  // Read directly from source
                if (topk_q.Push(s, static_cast<uint32_t>(docid_start + idx))) {
                    if (topk_q.Full()) {
                        threshold = topk_q.Threshold();
                        vthr = svdup_f32(threshold);  // Update threshold vector only when changed
                    }
                }
            }
        }
    }

    // Tail: handle remaining elements with predicate
    if (i < count) {
        const uint32_t step = static_cast<uint32_t>(count - i);
        svbool_t pg = svwhilelt_b32(static_cast<uint32_t>(0), step);
        svfloat32_t v = svld1(pg, scores + i);
        svbool_t pg_sel = svcmpgt_f32(pg, v, vthr);

        if (svptest_any(pg, pg_sel)) {
            alignas(64) uint32_t tmp_mask[64];
            svuint32_t ones = svdup_u32(1);
            svuint32_t zeros = svdup_u32(0);
            svuint32_t vmask = svsel_u32(pg_sel, ones, zeros);
            svst1(pg, tmp_mask, vmask);

            for (uint32_t j = 0; j < step; ++j) {
                if (tmp_mask[j]) {
                    size_t idx = i + j;
                    // bitset filtering: skip masked ids
                    if (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + idx))) {
                        continue;
                    }
                    float s = scores[idx];
                    if (topk_q.Push(s, static_cast<uint32_t>(docid_start + idx))) {
                        if (topk_q.Full()) {
                            threshold = topk_q.Threshold();
                            vthr = svdup_f32(threshold);
                        }
                    }
                }
            }
        }
    }
}
#endif

#if SINDI_USE_AVX512
static inline void
batch_insert_avx512(const float* scores, size_t docid_start, size_t count,
                    knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset) {
    size_t i = 0;
    __m512 vthr = _mm512_set1_ps(threshold);
    for (; i + 16 <= count; i += 16) {
        _mm_prefetch(reinterpret_cast<const char*>(scores + i + 64), _MM_HINT_T0);
        __m512 v = _mm512_loadu_ps(scores + i);
        __mmask16 m = _mm512_cmp_ps_mask(v, vthr, _CMP_GT_OQ);
        uint32_t mm = static_cast<uint32_t>(m);
        while (mm != 0u) {
            unsigned bit = __builtin_ctz(mm);
            mm &= (mm - 1);
            size_t idx = i + bit;
            // bitset filtering: skip masked ids
            if (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + idx))) {
                continue;
            }
            float s = scores[idx];
            if (topk_q.Push(s, static_cast<uint32_t>(docid_start + idx))) {
                if (topk_q.Full()) {
                    threshold = topk_q.Threshold();
                }
            }
        }
        vthr = _mm512_set1_ps(threshold);
    }
    // Tail
    for (; i < count; ++i) {
        float s = scores[i];
        if (s <= threshold) {
            continue;
        }
        // bitset filtering: skip masked ids
        if (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + i))) {
            continue;
        }
        if (topk_q.Push(s, static_cast<uint32_t>(docid_start + i))) {
            if (topk_q.Full()) {
                threshold = topk_q.Threshold();
            }
        }
    }
}
#endif

#if SINDI_USE_AVX2
static inline void
batch_insert_avx2(const float* scores, size_t docid_start, size_t count,
                  knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset) {
    size_t i = 0;
    __m256 vthr = _mm256_set1_ps(threshold);
    for (; i + 8 <= count; i += 8) {
        _mm_prefetch(reinterpret_cast<const char*>(scores + i + 32), _MM_HINT_T0);
        __m256 v = _mm256_loadu_ps(scores + i);
        __m256 cmp = _mm256_cmp_ps(v, vthr, _CMP_GT_OQ);
        int mm = _mm256_movemask_ps(cmp);
        while (mm != 0) {
            unsigned bit = __builtin_ctz(static_cast<unsigned>(mm));
            mm &= (mm - 1);
            size_t idx = i + bit;
            // bitset filtering: skip masked ids
            if (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + idx))) {
                continue;
            }
            float s = scores[idx];
            if (topk_q.Push(s, static_cast<uint32_t>(docid_start + idx))) {
                if (topk_q.Full()) {
                    threshold = topk_q.Threshold();
                    vthr = _mm256_set1_ps(threshold);
                }
            }
        }
    }
    // Tail
    for (; i < count; ++i) {
        float s = scores[i];
        if (s <= threshold) {
            continue;
        }
        // bitset filtering: skip masked ids
        if (!bitset.empty() && bitset.test(static_cast<int64_t>(docid_start + i))) {
            continue;
        }
        if (topk_q.Push(s, static_cast<uint32_t>(docid_start + i))) {
            if (topk_q.Full()) {
                threshold = topk_q.Threshold();
            }
        }
    }
}
#endif

inline const IPKernels&
get_ip_kernels() {
    static const IPKernels kernels = []() {
        IPKernels k{};
#if defined(__x86_64__)
#if defined(__AVX512F__)
        if (faiss::cppcontrib::knowhere::cpu_support_avx512()) {
            k.accumulate = ip_scatter_avx512_fp16;
            k.batch_insert = batch_insert_avx512;
            return k;
        }
#endif
#if defined(__AVX2__)
        if (faiss::cppcontrib::knowhere::cpu_support_avx2()) {
            k.accumulate = ip_scatter_avx2_fp16;
            k.batch_insert = batch_insert_avx2;
            return k;
        }
#endif
#elif defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
        if (faiss::cppcontrib::knowhere::supports_sve()) {
            k.scatter = ip_scatter_sve_fp16;
            k.batch_insert = batch_insert_sve;
            return k;
        }
#endif
        k.accumulate = ip_scatter_scalar_fp16;
        k.batch_insert = batch_insert_scalar;
        return k;
    }();
    return kernels;
}

inline const BM25Kernels&
get_bm25_kernels() {
    static const BM25Kernels kernels = []() {
        BM25Kernels k{};
#if defined(__x86_64__)
#if defined(__AVX512F__)
        if (faiss::cppcontrib::knowhere::cpu_support_avx512()) {
            k.accumulate = bm25_scatter_avx512_u16;
            k.batch_insert = batch_insert_avx512;
            return k;
        }
#endif
#if defined(__AVX2__)
        if (faiss::cppcontrib::knowhere::cpu_support_avx2()) {
            k.accumulate = bm25_scatter_avx2_u16;
            k.batch_insert = batch_insert_avx2;
            return k;
        }
#endif
#elif defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
        if (faiss::cppcontrib::knowhere::supports_sve()) {
            k.scatter = bm25_scatter_sve_u16;
            k.batch_insert = batch_insert_sve;
            return k;
        }
#endif
        k.accumulate = bm25_scatter_scalar_u16;
        k.batch_insert = batch_insert_scalar;
        return k;
    }();
    return kernels;
}

}  // namespace knowhere::sparse::inverted::sindi
