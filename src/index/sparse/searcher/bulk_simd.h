#pragma once

#include <cstdint>

#include "index/sparse/scorer.h"
#include "index/sparse/searcher/searcher.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__x86_64__) && defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace knowhere::sparse::inverted {

// Posting IDs within one list are unique, so each update is independent.
inline void
AccumulateIPBatch(const uint32_t* ids, const float* values, size_t count, float query_weight, uint32_t window_min,
                  float* window_scores, uint64_t* window_matches) {
    size_t i = 0;
#if (defined(__aarch64__) && defined(__ARM_NEON)) || (defined(__x86_64__) && defined(__SSE__))
    for (; i + 4 <= count; i += 4) {
        uint32_t offsets[4];
        float scores[4];
        for (size_t lane = 0; lane < 4; ++lane) {
            offsets[lane] = ids[i + lane] - window_min;
            scores[lane] = window_scores[offsets[lane]];
        }
#if defined(__aarch64__) && defined(__ARM_NEON)
        vst1q_f32(scores, vfmaq_n_f32(vld1q_f32(scores), vld1q_f32(values + i), query_weight));
#else
        const auto products = _mm_mul_ps(_mm_loadu_ps(values + i), _mm_set1_ps(query_weight));
        _mm_storeu_ps(scores, _mm_add_ps(_mm_loadu_ps(scores), products));
#endif
        for (size_t lane = 0; lane < 4; ++lane) {
            const uint32_t offset = offsets[lane];
            window_scores[offset] = scores[lane];
            window_matches[offset / 64] |= uint64_t{1} << (offset % 64);
        }
    }
#endif
    for (; i < count; ++i) {
        const uint32_t offset = ids[i] - window_min;
        window_scores[offset] += query_weight * values[i];
        window_matches[offset / 64] |= uint64_t{1} << (offset % 64);
    }
}

template <PostingFilterMode FilterMode, typename Cursor>
inline void
AccumulateIPBulk(Cursor& cursor, float query_weight, uint32_t window_min, uint32_t window_max, float* window_scores,
                 uint64_t* window_matches) {
    const auto accumulate = [&](uint32_t id, float value) {
        const uint32_t offset = id - window_min;
        window_scores[offset] += query_weight * value;
        window_matches[offset / 64] |= uint64_t{1} << (offset % 64);
        return true;
    };
    if constexpr (requires { cursor.template for_each_bulk<FilterMode>(window_max, accumulate); }) {
        if constexpr (FilterMode == PostingFilterMode::None) {
            cursor.template for_each_bulk<FilterMode>(window_max, accumulate);
        } else {
            constexpr size_t kBatchSize = 64;
            uint32_t ids[kBatchSize];
            float values[kBatchSize];
            while (cursor.vec_id() < window_max) {
                size_t count = 0;
                cursor.template for_each_bulk<FilterMode>(window_max, [&](uint32_t id, float value, bool accepted) {
                    ids[count] = id;
                    values[count] = value;
                    count += accepted;
                    return count < kBatchSize;
                });
                AccumulateIPBatch(ids, values, count, query_weight, window_min, window_scores, window_matches);
            }
        }
    } else {
        while (cursor.vec_id() < window_max) {
            accumulate(cursor.vec_id(), static_cast<float>(cursor.val()));
            NextPosting<FilterMode>(cursor);
        }
    }
}

inline void
ScoreBM25Bulk(const uint32_t* ids, float* values, size_t count, float query_weight, const BM25ScoringContext& scorer) {
    size_t i = 0;
#if (defined(__aarch64__) && defined(__ARM_NEON)) || (defined(__x86_64__) && defined(__SSE__))
    for (; i + 4 <= count; i += 4) {
        float norms[4];
        for (size_t lane = 0; lane < 4; ++lane) {
            norms[lane] = scorer.doc_norm(ids[i + lane]);
        }
#if defined(__aarch64__) && defined(__ARM_NEON)
        const auto norm = vld1q_f32(norms);
        const auto tf = vld1q_f32(values + i);
        const auto denominator = vaddq_f32(vaddq_f32(tf, vdupq_n_f32(scorer.denominator_bias())), norm);
        vst1q_f32(values + i, vdivq_f32(vmulq_f32(vdupq_n_f32(query_weight), tf), denominator));
#else
        const auto norm = _mm_loadu_ps(norms);
        const auto tf = _mm_loadu_ps(values + i);
        const auto denominator = _mm_add_ps(_mm_add_ps(tf, _mm_set1_ps(scorer.denominator_bias())), norm);
        _mm_storeu_ps(values + i, _mm_div_ps(_mm_mul_ps(_mm_set1_ps(query_weight), tf), denominator));
#endif
    }
#endif
    for (; i < count; ++i) {
        const float norm = scorer.doc_norm(ids[i]);
        values[i] = scorer.score(query_weight, values[i], norm);
    }
}

template <PostingFilterMode FilterMode, typename Cursor>
inline void
AccumulateBM25Bulk(Cursor& cursor, float query_weight, const BM25ScoringContext& scorer, uint32_t window_min,
                   uint32_t window_max, float* window_scores, uint64_t* window_matches) {
    const auto accumulate = [&](uint32_t id, float tf) {
        const uint32_t offset = id - window_min;
        const float doc_norm = scorer.doc_norm(id);
        window_scores[offset] += scorer.score(query_weight, tf, doc_norm);
        window_matches[offset / 64] |= uint64_t{1} << (offset % 64);
        return true;
    };
    if constexpr (requires { cursor.template for_each_bulk<FilterMode>(window_max, accumulate); }) {
        if constexpr (FilterMode == PostingFilterMode::None) {
            // Without an unpredictable filter branch, direct accumulation
            // avoids the extra buffering/scatter cost of the batch pipeline.
            cursor.template for_each_bulk<FilterMode>(window_max, accumulate);
        } else {
            // Separate posting/filter traversal from scoring, as in Lucene's
            // nextPostings -> filter -> bulk score pipeline. The small buffer
            // enables SIMD scoring of accepted postings.
            constexpr size_t kBatchSize = 64;
            uint32_t ids[kBatchSize];
            float values[kBatchSize];
            while (cursor.vec_id() < window_max) {
                size_t count = 0;
                cursor.template for_each_bulk<FilterMode>(window_max, [&](uint32_t id, float tf, bool accepted) {
                    ids[count] = id;
                    values[count] = tf;
                    count += accepted;
                    return count < kBatchSize;
                });
                ScoreBM25Bulk(ids, values, count, query_weight, scorer);
                for (size_t i = 0; i < count; ++i) {
                    const uint32_t offset = ids[i] - window_min;
                    window_scores[offset] += values[i];
                    window_matches[offset / 64] |= uint64_t{1} << (offset % 64);
                }
            }
        }
    } else {
        while (cursor.vec_id() < window_max) {
            accumulate(cursor.vec_id(), static_cast<float>(cursor.val()));
            NextPosting<FilterMode>(cursor);
        }
    }
}

// Screen a dense, 64-document score tile before candidate completion. Keep the
// addition in the same order as MaxScore's scalar upper-bound test: subtracting
// the bound from the threshold can change rounding and incorrectly prune ties.
// Rejected scores must be cleared as well, since the next window reuses them.
inline uint64_t
FilterBulkCandidates(float* scores, uint64_t matches, float remaining_bound, float threshold) {
    uint64_t competitive = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
    const auto bound = vdupq_n_f32(remaining_bound);
    const auto limit = vdupq_n_f32(threshold);
    const uint32x4_t weights = {1, 2, 4, 8};
    for (unsigned i = 0; i < 64; i += 4) {
        const auto values = vld1q_f32(scores + i);
        const auto keep = vcgtq_f32(vaddq_f32(values, bound), limit);
        competitive |= uint64_t{vaddvq_u32(vandq_u32(keep, weights))} << i;
        vst1q_f32(scores + i, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(values), keep)));
    }
#elif defined(__x86_64__) && defined(__SSE__)
    const auto bound = _mm_set1_ps(remaining_bound);
    const auto limit = _mm_set1_ps(threshold);
    for (unsigned i = 0; i < 64; i += 4) {
        const auto values = _mm_loadu_ps(scores + i);
        const auto keep = _mm_cmpgt_ps(_mm_add_ps(values, bound), limit);
        competitive |= uint64_t{static_cast<unsigned>(_mm_movemask_ps(keep))} << i;
        _mm_storeu_ps(scores + i, _mm_and_ps(values, keep));
    }
#else
    for (unsigned i = 0; i < 64; ++i) {
        if (scores[i] + remaining_bound > threshold) {
            competitive |= uint64_t{1} << i;
        } else {
            scores[i] = 0.0f;
        }
    }
#endif
    return matches & competitive;
}

}  // namespace knowhere::sparse::inverted
