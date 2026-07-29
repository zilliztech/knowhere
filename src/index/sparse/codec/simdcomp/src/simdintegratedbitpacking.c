/** Port from https://github.com/fast-pack/simdcomp
 *
 * This code is released under a BSD License.
 */
#include "../include/portability.h"

/*
 * Knowhere keeps only simdunpackd1 and the generated unpack helpers used by the sparse codec. Its document-ID
 * payload stores delta - 1, so the omitted one is folded into every SIMD prefix sum below. The private decoder
 * therefore reconstructs absolute document IDs directly without a second output pass.
 */

#define PrefixSum(ret, curr, prev)                                             \
  do {                                                                         \
    const __m128i _adjusted = _mm_add_epi32(curr, _mm_set1_epi32(1));          \
    const __m128i _tmp1 =                                                       \
        _mm_add_epi32(_mm_slli_si128(_adjusted, 8), _adjusted);                \
    const __m128i _tmp2 = _mm_add_epi32(_mm_slli_si128(_tmp1, 4), _tmp1);      \
    ret = _mm_add_epi32(_tmp2, _mm_shuffle_epi32(prev, 0xff));                 \
  } while (0)

static __m128i iunpack1(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 1) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack2(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 2) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack3(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 3) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 3 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 3 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack4(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 4) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack5(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 5) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 5 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack6(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 6) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 6 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack7(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 7) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 7 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack8(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 8) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack9(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 9) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 9 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack10(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 10) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 10 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack11(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 11) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 11 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack12(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 12) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 12 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack13(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 13) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 13 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack14(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 14) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 14 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack15(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 15) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 15 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack16(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 16) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack17(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 17) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 17 - 15), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack18(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 18) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 18 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack19(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 19) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 17), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 15), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 19 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack20(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 20) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 20 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack21(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 21) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 19), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 17), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 15), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 21 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack22(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 22) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 22 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack23(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 23) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 19), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 15), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 21), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 17), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 23 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack24(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 24) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 24 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack25(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 25) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 15), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 19), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 23), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 17), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 21), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 25 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack26(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 26) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 26 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack27(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 27) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 17), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 19), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 26), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 21), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 23), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 25), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 15), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 27 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack28(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 28) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 28 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack29(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 29) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 26), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 23), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 17), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 28), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 25), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 19), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 27), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 21), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 15), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 29 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack30(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 30) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 28), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 26), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 28), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 26), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 30 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static __m128i iunpack31(__m128i initOffset, const __m128i *in, uint32_t *_out) {

  __m128i *out = (__m128i *)(_out);
  __m128i InReg = _mm_loadu_si128(in);
  __m128i OutReg;
  __m128i tmp;
  __m128i mask = _mm_set1_epi32((1U << 31) - 1);

  tmp = InReg;
  OutReg = _mm_and_si128(tmp, mask);
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 31);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 30), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 30);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 29), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 29);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 28), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 28);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 27), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 27);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 26), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 26);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 25), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 25);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 24), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 24);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 23), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 23);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 22), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 22);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 21), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 21);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 20), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 20);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 19), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 19);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 18), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 18);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 17), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 17);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 16), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 16);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 15), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 15);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 14), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 14);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 13), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 13);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 12), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 12);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 11), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 11);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 10), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 10);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 9), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 9);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 8), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 8);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 7), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 7);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 6), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 6);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 5), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 5);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 4), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 4);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 3), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 3);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 2), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 2);
  OutReg = tmp;
  ++in;
  InReg = _mm_loadu_si128(in);
  OutReg =
      _mm_or_si128(OutReg, _mm_and_si128(_mm_slli_epi32(InReg, 31 - 1), mask));

  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  tmp = _mm_srli_epi32(InReg, 1);
  OutReg = tmp;
  PrefixSum(OutReg, OutReg, initOffset);
  initOffset = OutReg;
  _mm_storeu_si128(out++, OutReg);

  return initOffset;
}

static void simdunpackd1(uint32_t initvalue, const __m128i *in, uint32_t *out,
                         const uint32_t bit) {
  __m128i initOffset = _mm_set1_epi32(initvalue);
  switch (bit) {
  case 1:
    iunpack1(initOffset, in, out);
    break;

  case 2:
    iunpack2(initOffset, in, out);
    break;

  case 3:
    iunpack3(initOffset, in, out);
    break;

  case 4:
    iunpack4(initOffset, in, out);
    break;

  case 5:
    iunpack5(initOffset, in, out);
    break;

  case 6:
    iunpack6(initOffset, in, out);
    break;

  case 7:
    iunpack7(initOffset, in, out);
    break;

  case 8:
    iunpack8(initOffset, in, out);
    break;

  case 9:
    iunpack9(initOffset, in, out);
    break;

  case 10:
    iunpack10(initOffset, in, out);
    break;

  case 11:
    iunpack11(initOffset, in, out);
    break;

  case 12:
    iunpack12(initOffset, in, out);
    break;

  case 13:
    iunpack13(initOffset, in, out);
    break;

  case 14:
    iunpack14(initOffset, in, out);
    break;

  case 15:
    iunpack15(initOffset, in, out);
    break;

  case 16:
    iunpack16(initOffset, in, out);
    break;

  case 17:
    iunpack17(initOffset, in, out);
    break;

  case 18:
    iunpack18(initOffset, in, out);
    break;

  case 19:
    iunpack19(initOffset, in, out);
    break;

  case 20:
    iunpack20(initOffset, in, out);
    break;

  case 21:
    iunpack21(initOffset, in, out);
    break;

  case 22:
    iunpack22(initOffset, in, out);
    break;

  case 23:
    iunpack23(initOffset, in, out);
    break;

  case 24:
    iunpack24(initOffset, in, out);
    break;

  case 25:
    iunpack25(initOffset, in, out);
    break;

  case 26:
    iunpack26(initOffset, in, out);
    break;

  case 27:
    iunpack27(initOffset, in, out);
    break;

  case 28:
    iunpack28(initOffset, in, out);
    break;

  case 29:
    iunpack29(initOffset, in, out);
    break;

  case 30:
    iunpack30(initOffset, in, out);
    break;

  case 31:
    iunpack31(initOffset, in, out);
    break;

  default:
    break;
  }
}

#undef PrefixSum
