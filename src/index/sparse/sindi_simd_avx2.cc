#include "index/sparse/sindi_simd.h"

#if defined(__x86_64__)
#include <immintrin.h>

namespace knowhere::sparse::inverted::sindi {

void
ip_scatter_avx2_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    int32_t i = 0;
    const __m256 vq = _mm256_set1_ps(qval);
    for (; i + 8 <= num; i += 8) {
        const uint16_t* hptr = reinterpret_cast<const uint16_t*>(vals + i);
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hptr));
        __m256 v_vals = _mm256_cvtph_ps(h);
        __m256 v_mul = _mm256_mul_ps(v_vals, vq);

        __m128i idx16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i v_idx = _mm256_cvtepu16_epi32(idx16);
        __m256 v_old = _mm256_i32gather_ps(out, v_idx, 4);
        __m256 v_sum = _mm256_add_ps(v_old, v_mul);

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
    }
    for (; i < num; ++i) {
        out[ids[i]] += qval * static_cast<float>(vals[i]);
    }
}

void
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

    for (; i + 8 <= num; i += 8) {
        const uint16_t* hptr = vals + i;
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hptr));
        __m256i w = _mm256_cvtepu16_epi32(h);
        __m256 tf_vec = _mm256_cvtepi32_ps(w);

        __m128i idx16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i v_idx = _mm256_cvtepu16_epi32(idx16);
        __m256 dl_vec = _mm256_i32gather_ps(row_sums, v_idx, 4);

        __m256 numerator = _mm256_mul_ps(tf_vec, vp1);
        numerator = _mm256_mul_ps(numerator, vqval);

        __m256 denominator = _mm256_fmadd_ps(dl_vec, vp3, vp2);
        denominator = _mm256_add_ps(tf_vec, denominator);

        __m256 bm25_vec = _mm256_div_ps(numerator, denominator);

        __m256 v_old = _mm256_i32gather_ps(out, v_idx, 4);
        __m256 v_sum = _mm256_add_ps(v_old, bm25_vec);

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
    }

    for (; i < num; ++i) {
        float tf = static_cast<float>(vals[i]);
        uint16_t docid = ids[i];
        float dl = row_sums[docid];
        float bm25_score = qval * p1 * tf / (tf + p2 + p3 * dl);
        out[docid] += bm25_score;
    }
}

void
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
