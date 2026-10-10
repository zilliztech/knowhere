#include "index/sparse/sindi_simd.h"

#if defined(__x86_64__)
#include <immintrin.h>

#include "index/sparse/sindi_packed12_x86.h"

namespace knowhere::sparse::inverted::sindi {

float
ip_accumulate_avx2_u12_e5m7(float q, const uint8_t* vals, const uint8_t* ids, size_t start, int32_t n, float* out) {
    if (n <= 0) {
        return 0;
    }

    float maximum = 0;
    if (start & 1) {
        maximum = ip_accumulate_scalar_u12_e5m7(q, vals, ids, start, 1, out);
        ++start;
        --n;
    }

    const auto vq = _mm256_set1_ps(q);
    auto vmax = _mm256_setzero_ps();

    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const auto* ip = ids + ((start + i) / 2) * 3;
        const auto* vp = vals + ((start + i) / 2) * 3;
        const auto id = _mm256_cvtepu16_epi32(unpack12_eight_x86(ip));
        const auto value = _mm256_cvtph_ps(_mm_slli_epi16(unpack12_eight_x86(vp), 3));
        const auto sum = _mm256_fmadd_ps(value, vq, _mm256_i32gather_ps(out, id, 4));

        // AVX2 has gather but no scatter.
        alignas(32) uint32_t indices[8];
        alignas(32) float scores[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(indices), id);
        _mm256_store_ps(scores, sum);
        for (int lane = 0; lane < 8; ++lane) {
            out[indices[lane]] = scores[lane];
        }

        vmax = _mm256_max_ps(vmax, sum);
    }

    alignas(32) float maxima[8];
    _mm256_store_ps(maxima, vmax);
    for (float value : maxima) {
        maximum = std::max(maximum, value);
    }

    return std::max(maximum, ip_accumulate_scalar_u12_e5m7(q, vals, ids, start + i, n - i, out));
}

float
bm25_accumulate_avx2_u12_u4_lut(float q, const uint8_t* vals, const uint8_t* ids, size_t start, int32_t n, float* out,
                                float k1, float b, float avgdl, const float* lengths, const uint8_t* table) {
    if (n <= 0) {
        return 0;
    }

    float maximum = 0;
    if (start & 1) {
        maximum = bm25_accumulate_scalar_u12_u4_lut(q, vals, ids, start, 1, out, k1, b, avgdl, lengths, table);
        ++start;
        --n;
    }
    const auto lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(table));
    const auto nibble_mask = _mm_set1_epi8(15);
    const auto vqp1 = _mm256_set1_ps(q * (k1 + 1.0f));
    const auto vp2 = _mm256_set1_ps(k1 * (1.0f - b)), vp3 = _mm256_set1_ps(k1 * b / avgdl);
    auto vmax = _mm256_setzero_ps();

    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const auto id = _mm256_cvtepu16_epi32(unpack12_eight_x86(ids + ((start + i) / 2) * 3));
        // Eight codes occupy exactly four bytes; the LUT load is exactly 16.
        uint32_t packed;
        std::memcpy(&packed, vals + (start + i) / 2, sizeof(packed));
        const auto bytes = _mm_cvtsi32_si128(packed);
        const auto codes = _mm_and_si128(_mm_unpacklo_epi8(bytes, _mm_srli_epi16(bytes, 4)), nibble_mask);
        const auto tf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi8(lut, codes)));
        const auto dl = _mm256_i32gather_ps(lengths, id, 4);
        const auto denominator = _mm256_add_ps(tf, _mm256_fmadd_ps(dl, vp3, vp2));
        const auto contribution = _mm256_div_ps(_mm256_mul_ps(tf, vqp1), denominator);
        const auto sum = _mm256_add_ps(_mm256_i32gather_ps(out, id, 4), contribution);

        // Match existing AVX2 BM25 writeback: gather is available, scatter is not.
        alignas(32) uint32_t indices[8];
        alignas(32) float scores[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(indices), id);
        _mm256_store_ps(scores, sum);
        for (int lane = 0; lane < 8; ++lane) {
            out[indices[lane]] = scores[lane];
        }
        vmax = _mm256_max_ps(vmax, sum);
    }
    auto tail_max = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    tail_max = _mm_max_ps(tail_max, _mm_shuffle_ps(tail_max, tail_max, _MM_SHUFFLE(2, 3, 0, 1)));
    tail_max = _mm_max_ps(tail_max, _mm_shuffle_ps(tail_max, tail_max, _MM_SHUFFLE(1, 0, 3, 2)));
    maximum = std::max(maximum, _mm_cvtss_f32(tail_max));
    return std::max(
        maximum, bm25_accumulate_scalar_u12_u4_lut(q, vals, ids, start + i, n - i, out, k1, b, avgdl, lengths, table));
}

float
bm25_accumulate_avx2_u16_u4_lut(float q, const uint8_t* vals, const uint8_t* ids, size_t start, int32_t n, float* out,
                                float k1, float b, float avgdl, const float* lengths, const uint8_t* table) {
    if (n <= 0) {
        return 0;
    }

    float maximum = 0;
    if (start & 1) {
        maximum = bm25_accumulate_scalar_u16_u4_lut(q, vals, ids, start, 1, out, k1, b, avgdl, lengths, table);
        ++start;
        --n;
    }
    const auto lut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(table));
    const auto nibble_mask = _mm_set1_epi8(15);
    const auto vqp1 = _mm256_set1_ps(q * (k1 + 1.0f));
    const auto vp2 = _mm256_set1_ps(k1 * (1.0f - b)), vp3 = _mm256_set1_ps(k1 * b / avgdl);
    auto vmax = _mm256_setzero_ps();

    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const auto id = _mm256_cvtepu16_epi32(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + (start + i) * sizeof(uint16_t))));
        // Eight codes occupy exactly four bytes; the LUT load is exactly 16.
        uint32_t packed;
        std::memcpy(&packed, vals + (start + i) / 2, sizeof(packed));
        const auto bytes = _mm_cvtsi32_si128(packed);
        const auto codes = _mm_and_si128(_mm_unpacklo_epi8(bytes, _mm_srli_epi16(bytes, 4)), nibble_mask);
        const auto tf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_shuffle_epi8(lut, codes)));
        const auto dl = _mm256_i32gather_ps(lengths, id, 4);
        const auto denominator = _mm256_add_ps(tf, _mm256_fmadd_ps(dl, vp3, vp2));
        const auto contribution = _mm256_div_ps(_mm256_mul_ps(tf, vqp1), denominator);
        const auto sum = _mm256_add_ps(_mm256_i32gather_ps(out, id, 4), contribution);

        // Match existing AVX2 BM25 writeback: gather is available, scatter is not.
        alignas(32) uint32_t indices[8];
        alignas(32) float scores[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(indices), id);
        _mm256_store_ps(scores, sum);
        for (int lane = 0; lane < 8; ++lane) {
            out[indices[lane]] = scores[lane];
        }
        vmax = _mm256_max_ps(vmax, sum);
    }
    auto tail_max = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
    tail_max = _mm_max_ps(tail_max, _mm_shuffle_ps(tail_max, tail_max, _MM_SHUFFLE(2, 3, 0, 1)));
    tail_max = _mm_max_ps(tail_max, _mm_shuffle_ps(tail_max, tail_max, _MM_SHUFFLE(1, 0, 3, 2)));
    maximum = std::max(maximum, _mm_cvtss_f32(tail_max));
    return std::max(
        maximum, bm25_accumulate_scalar_u16_u4_lut(q, vals, ids, start + i, n - i, out, k1, b, avgdl, lengths, table));
}

float
ip_accumulate_avx2_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    int32_t i = 0;
    const __m256 vq = _mm256_set1_ps(qval);
    __m256 v_max = _mm256_setzero_ps();
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

float
bm25_accumulate_avx2_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                         float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    int32_t i = 0;
    const __m256 vqval = _mm256_set1_ps(qval);
    const __m256 vp1 = _mm256_set1_ps(p1);
    const __m256 vp2 = _mm256_set1_ps(p2);
    const __m256 vp3 = _mm256_set1_ps(p3);
    __m256 v_max = _mm256_setzero_ps();

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
        v_max = _mm256_max_ps(v_max, v_sum);
    }

    __m128 v_max128 = _mm_max_ps(_mm256_castps256_ps128(v_max), _mm256_extractf128_ps(v_max, 1));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(2, 3, 0, 1)));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(v_max128);

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

float
bm25_accumulate_avx2_u8(float qval, const uint8_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                        float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    int32_t i = 0;
    const __m256 vqp1 = _mm256_set1_ps(qval * p1);
    const __m256 vp2 = _mm256_set1_ps(p2);
    const __m256 vp3 = _mm256_set1_ps(p3);
    __m256 v_max = _mm256_setzero_ps();

    for (; i + 8 <= num; i += 8) {
        __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(vals + i));
        __m256i words = _mm256_cvtepu8_epi32(bytes);
        __m256 tf_vec = _mm256_cvtepi32_ps(words);

        __m128i idx16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ids + i));
        __m256i v_idx = _mm256_cvtepu16_epi32(idx16);
        __m256 dl_vec = _mm256_i32gather_ps(row_sums, v_idx, 4);

        __m256 numerator = _mm256_mul_ps(tf_vec, vqp1);
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
        v_max = _mm256_max_ps(v_max, v_sum);
    }

    __m128 v_max128 = _mm_max_ps(_mm256_castps256_ps128(v_max), _mm256_extractf128_ps(v_max, 1));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(2, 3, 0, 1)));
    v_max128 = _mm_max_ps(v_max128, _mm_shuffle_ps(v_max128, v_max128, _MM_SHUFFLE(1, 0, 3, 2)));
    float max_val = _mm_cvtss_f32(v_max128);

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
