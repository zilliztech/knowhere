/** Port from https://github.com/fast-pack/simdcomp
 *
 * This code is released under a BSD License.
 */
#ifndef KNOWHERE_SIMDCOMP_PORTABILITY_H_
#define KNOWHERE_SIMDCOMP_PORTABILITY_H_

#include <stdint.h>

#if defined(__aarch64__) || defined(__arm__) || defined(__ARM_NEON) || defined(_M_ARM64)
#include "neon128.h"
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <emmintrin.h>
#else
/* Keep the generated SSE2 kernels available on the other architectures supported by Knowhere. */
#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <simde/x86/sse2.h>
#endif

#endif /* KNOWHERE_SIMDCOMP_PORTABILITY_H_ */
