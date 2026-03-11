#include "index/sparse/sindi_simd.h"

#if defined(__x86_64__)
#include <immintrin.h>

namespace knowhere::sparse::inverted::sindi {

float
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

float
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

    for (; i + 16 <= num; i += 16) {
        const uint16_t* hptr = vals + i;
        _mm_prefetch(reinterpret_cast<const char*>(hptr + 32), _MM_HINT_NTA);

        __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hptr));
        __m512i w = _mm512_cvtepu16_epi32(h);
        __m512 tf_vec = _mm512_cvtepi32_ps(w);

        __m256i idx16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids + i));
        __m512i v_idx = _mm512_cvtepu16_epi32(idx16);
        __m512 dl_vec = _mm512_i32gather_ps(v_idx, row_sums, 4);

        __m512 numerator = _mm512_mul_ps(tf_vec, vp1);
        numerator = _mm512_mul_ps(numerator, vqval);

        __m512 denominator = _mm512_fmadd_ps(dl_vec, vp3, vp2);
        denominator = _mm512_add_ps(tf_vec, denominator);

        __m512 bm25_vec = _mm512_div_ps(numerator, denominator);

        __m512 v_old = _mm512_i32gather_ps(v_idx, out, 4);
        __m512 v_sum = _mm512_add_ps(v_old, bm25_vec);
        _mm512_i32scatter_ps(out, v_idx, v_sum, 4);
        v_max = _mm512_max_ps(v_max, v_sum);
    }

    float max_val = _mm512_reduce_max_ps(v_max);

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

void
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
    for (; i < count; ++i) {
        float s = scores[i];
        if (s <= threshold) {
            continue;
        }
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

}  // namespace knowhere::sparse::inverted::sindi

#endif  // __x86_64__
