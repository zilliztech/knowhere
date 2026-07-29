/** Port from https://github.com/fast-pack/simdcomp
 *
 * This code is released under a BSD License.
 *
 * Minimal SSE2-on-NEON shim for the generated kernels retained by Knowhere.
 */
#ifndef KNOWHERE_SIMDCOMP_NEON128_H_
#define KNOWHERE_SIMDCOMP_NEON128_H_

#include <arm_neon.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define KNOWHERE_SIMDCOMP_NEON_INLINE __inline__ __attribute__((always_inline))
#else
#define KNOWHERE_SIMDCOMP_NEON_INLINE inline
#endif

/* The packed stream can be byte-aligned even though every load moves 128 bits. */
#if defined(__GNUC__) || defined(__clang__)
typedef int32x4_t __m128i __attribute__((aligned(1)));
#else
typedef int32x4_t __m128i;
#endif

static KNOWHERE_SIMDCOMP_NEON_INLINE __m128i
_mm_loadu_si128(const __m128i* p) {
    return vreinterpretq_s32_u8(vld1q_u8((const uint8_t*)p));
}

static KNOWHERE_SIMDCOMP_NEON_INLINE void
_mm_storeu_si128(__m128i* p, __m128i value) {
    vst1q_u8((uint8_t*)p, vreinterpretq_u8_s32(value));
}

static KNOWHERE_SIMDCOMP_NEON_INLINE __m128i
_mm_set1_epi32(int value) {
    return vdupq_n_s32(value);
}

static KNOWHERE_SIMDCOMP_NEON_INLINE __m128i
_mm_and_si128(__m128i lhs, __m128i rhs) {
    return vandq_s32(lhs, rhs);
}

static KNOWHERE_SIMDCOMP_NEON_INLINE __m128i
_mm_or_si128(__m128i lhs, __m128i rhs) {
    return vorrq_s32(lhs, rhs);
}

static KNOWHERE_SIMDCOMP_NEON_INLINE __m128i
_mm_add_epi32(__m128i lhs, __m128i rhs) {
    return vaddq_s32(lhs, rhs);
}

/* NEON yields zero when the absolute shift count is at least the lane width. */
static KNOWHERE_SIMDCOMP_NEON_INLINE __m128i
_mm_slli_epi32(__m128i value, int count) {
    return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(value), vdupq_n_s32(count)));
}

static KNOWHERE_SIMDCOMP_NEON_INLINE __m128i
_mm_srli_epi32(__m128i value, int count) {
    return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(value), vdupq_n_s32(-count)));
}

static KNOWHERE_SIMDCOMP_NEON_INLINE int
_mm_cvtsi128_si32(__m128i value) {
    return vgetq_lane_s32(value, 0);
}

/* The immediate arguments remain compile-time constants in every retained call site. */
#define _mm_shuffle_epi32(value, imm)                                                                          \
    vsetq_lane_s32(                                                                                            \
        vgetq_lane_s32((value), ((imm) >> 6) & 3),                                                             \
        vsetq_lane_s32(vgetq_lane_s32((value), ((imm) >> 4) & 3),                                              \
                       vsetq_lane_s32(vgetq_lane_s32((value), ((imm) >> 2) & 3),                               \
                                      vsetq_lane_s32(vgetq_lane_s32((value), (imm)&3), vdupq_n_s32(0), 0), 1), \
                       2),                                                                                     \
        3)

#define _mm_slli_si128(value, imm) \
    vreinterpretq_s32_u8(vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s32(value), 16 - (imm)))

#undef KNOWHERE_SIMDCOMP_NEON_INLINE

#endif /* KNOWHERE_SIMDCOMP_NEON128_H_ */
