#include <faiss/utils/partitioning.h>

#include <cassert>
#include <cmath>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/ordered_key_value.h>
#include <faiss/utils/simdlib.h>

#include <faiss/impl/platform_macros.h>

namespace faiss {

/******************************************************************
 * Internal routines
 ******************************************************************/

namespace simd_partitioning_avx2 {

void find_minimax(
        const uint16_t* vals,
        size_t n,
        uint16_t& smin,
        uint16_t& smax) {
    simd16uint16 vmin(0xffff), vmax(0);
    for (size_t i = 0; i + 15 < n; i += 16) {
        simd16uint16 v(vals + i);
        vmin.accu_min(v);
        vmax.accu_max(v);
    }

    ALIGNED(32) uint16_t tab32[32];
    vmin.store(tab32);
    vmax.store(tab32 + 16);

    smin = tab32[0], smax = tab32[16];

    for (int i = 1; i < 16; i++) {
        smin = std::min(smin, tab32[i]);
        smax = std::max(smax, tab32[i + 16]);
    }

    // missing values
    for (size_t i = (n & ~15); i < n; i++) {
        smin = std::min(smin, vals[i]);
        smax = std::max(smax, vals[i]);
    }
}

// max func differentiates between CMin and CMax (keep lowest or largest)
template <class C>
simd16uint16 max_func(simd16uint16 v, simd16uint16 thr16) {
    constexpr bool is_max = C::is_max;
    if (is_max) {
        return max(v, thr16);
    } else {
        return min(v, thr16);
    }
}

template <class C>
void count_lt_and_eq(
        const uint16_t* vals,
        int n,
        uint16_t thresh,
        size_t& n_lt,
        size_t& n_eq) {
    n_lt = n_eq = 0;
    simd16uint16 thr16(thresh);

    size_t n1 = n / 16;

    for (size_t i = 0; i < n1; i++) {
        simd16uint16 v(vals);
        vals += 16;
        simd16uint16 eqmask = (v == thr16);
        simd16uint16 max2 = max_func<C>(v, thr16);
        simd16uint16 gemask = (v == max2);
        uint32_t bits = get_MSBs(uint16_to_uint8_saturate(eqmask, gemask));
        int i_eq = __builtin_popcount(bits & 0x00ff00ff);
        int i_ge = __builtin_popcount(bits) - i_eq;
        n_eq += i_eq;
        n_lt += 16 - i_ge;
    }

    for (size_t i = n1 * 16; i < (size_t)n; i++) {
        uint16_t v = *vals++;
        if (C::cmp(thresh, v)) {
            n_lt++;
        } else if (v == thresh) {
            n_eq++;
        }
    }
}

/* compress separated values and ids table, keeping all values < thresh and at
 * most n_eq equal values */
template <class C>
int simd_compress_array(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        uint16_t thresh,
        int n_eq) {
    simd16uint16 thr16(thresh);
    simd16uint16 mixmask(0xff00);

    int wp = 0;
    size_t i0;

    // loop while there are eqs to collect
    for (i0 = 0; i0 + 15 < n && n_eq > 0; i0 += 16) {
        simd16uint16 v(vals + i0);
        simd16uint16 max2 = max_func<C>(v, thr16);
        simd16uint16 gemask = (v == max2);
        simd16uint16 eqmask = (v == thr16);
        uint32_t bits = get_MSBs(
                blendv(simd32uint8(eqmask),
                       simd32uint8(gemask),
                       simd32uint8(mixmask)));
        bits ^= 0xAAAAAAAA;
        // bit 2*i     : eq
        // bit 2*i + 1 : lt

        while (bits) {
            int j = __builtin_ctz(bits) & (~1);
            bool is_eq = (bits >> j) & 1;
            bool is_lt = (bits >> j) & 2;
            bits &= ~(3 << j);
            j >>= 1;

            if (is_lt) {
                vals[wp] = vals[i0 + j];
                ids[wp] = ids[i0 + j];
                wp++;
            } else if (is_eq && n_eq > 0) {
                vals[wp] = vals[i0 + j];
                ids[wp] = ids[i0 + j];
                wp++;
                n_eq--;
            }
        }
    }

    // handle remaining, only striclty lt ones.
    for (; i0 + 15 < n; i0 += 16) {
        simd16uint16 v(vals + i0);
        simd16uint16 max2 = max_func<C>(v, thr16);
        simd16uint16 gemask = (v == max2);
        uint32_t bits = ~get_MSBs(simd32uint8(gemask));

        while (bits) {
            int j = __builtin_ctz(bits);
            bits &= ~(3 << j);
            j >>= 1;

            vals[wp] = vals[i0 + j];
            ids[wp] = ids[i0 + j];
            wp++;
        }
    }

    // end with scalar
    for (size_t i = (n & ~15); i < n; i++) {
        if (C::cmp(thresh, vals[i])) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
        } else if (vals[i] == thresh && n_eq > 0) {
            vals[wp] = vals[i];
            ids[wp] = ids[i];
            wp++;
            n_eq--;
        }
    }
    assert(n_eq == 0);
    return wp;
}

// #define MICRO_BENCHMARK

static uint64_t get_cy() {
#ifdef MICRO_BENCHMARK
    uint32_t high, low;
    asm volatile("rdtsc \n\t" : "=a"(low), "=d"(high));
    return ((uint64_t)high << 32) | (low);
#else
    return 0;
#endif
}

#define IFV if (false)

template <class C>
uint16_t simd_partition_fuzzy_with_bounds(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out,
        uint16_t s0i,
        uint16_t s1i) {
    if (q_min == 0) {
        if (q_out) {
            *q_out = 0;
        }
        return 0;
    }
    if (q_max >= n) {
        if (q_out) {
            *q_out = q_max;
        }
        return 0xffff;
    }
    if (s0i == s1i) {
        if (q_out) {
            *q_out = q_min;
        }
        return s0i;
    }
    uint64_t t0 = get_cy();

    // lower bound inclusive, upper exclusive
    size_t s0 = s0i, s1 = s1i + 1;

    IFV printf("bounds: %ld %ld\n", s0, s1 - 1);

    int thresh;
    size_t n_eq = 0, n_lt = 0;
    size_t q = 0;

    for (int it = 0; it < 200; it++) {
        // while(s0 + 1 < s1) {
        thresh = (s0 + s1) / 2;
        count_lt_and_eq<C>(vals, n, thresh, n_lt, n_eq);

        IFV printf(
                "   [%ld %ld] thresh=%d n_lt=%ld n_eq=%ld, q=%ld:%ld/%ld\n",
                s0,
                s1,
                thresh,
                n_lt,
                n_eq,
                q_min,
                q_max,
                n);
        if (n_lt <= q_min) {
            if (n_lt + n_eq >= q_min) {
                q = q_min;
                break;
            } else {
                if (C::is_max) {
                    s0 = thresh;
                } else {
                    s1 = thresh;
                }
            }
        } else if (n_lt <= q_max) {
            q = n_lt;
            break;
        } else {
            if (C::is_max) {
                s1 = thresh;
            } else {
                s0 = thresh;
            }
        }
    }

    uint64_t t1 = get_cy();

    // number of equal values to keep
    int64_t n_eq_1 = q - n_lt;

    IFV printf("shrink: thresh=%d q=%ld n_eq_1=%ld\n", thresh, q, n_eq_1);
    if (n_eq_1 < 0) { // happens when > q elements are at lower bound
        assert(s0 + 1 == s1);
        q = q_min;
        if (C::is_max) {
            thresh--;
        } else {
            thresh++;
        }
        n_eq_1 = q;
        IFV printf("  override: thresh=%d n_eq_1=%ld\n", thresh, n_eq_1);
    } else {
        assert((size_t)n_eq_1 <= n_eq);
    }

    size_t wp = simd_compress_array<C>(vals, ids, n, thresh, n_eq_1);

    IFV printf("wp=%ld\n", wp);
    assert(wp == q);
    if (q_out) {
        *q_out = q;
    }

    uint64_t t2 = get_cy();

    partition_stats.bissect_cycles += t1 - t0;
    partition_stats.compress_cycles += t2 - t1;

    return thresh;
}

template <class C>
uint16_t simd_partition_fuzzy(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
    assert(is_aligned_pointer(vals));

    uint16_t s0i, s1i;
    find_minimax(vals, n, s0i, s1i);
    // QSelect_stats.t0 += get_cy() - t0;

    return simd_partition_fuzzy_with_bounds<C>(
            vals, ids, n, q_min, q_max, q_out, s0i, s1i);
}

template <class C>
uint16_t simd_partition(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q) {
    assert(is_aligned_pointer(vals));

    if (q == 0) {
        return 0;
    }
    if (q >= n) {
        return 0xffff;
    }

    uint16_t s0i, s1i;
    find_minimax(vals, n, s0i, s1i);

    return simd_partition_fuzzy_with_bounds<C>(
            vals, ids, n, q, q, nullptr, s0i, s1i);
}

template <class C>
uint16_t simd_partition_with_bounds(
        uint16_t* vals,
        typename C::TI* ids,
        size_t n,
        size_t q,
        uint16_t s0i,
        uint16_t s1i) {
    return simd_partition_fuzzy_with_bounds<C>(
            vals, ids, n, q, q, nullptr, s0i, s1i);
}

} // namespace simd_partitioning


template <class C>
typename C::T partition_fuzzy_avx2(
        typename C::T* vals,
        typename C::TI* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out) {
    constexpr bool is_uint16 = std::is_same<typename C::T, uint16_t>::value;
    if (is_uint16 && is_aligned_pointer(vals)) {
        return simd_partitioning_avx2::simd_partition_fuzzy<C>(
                (uint16_t*)vals, ids, n, q_min, q_max, q_out);
    }
    return partition_fuzzy<C>(vals, ids, n, q_min, q_max, q_out);
}

// explicit template instanciations

template float partition_fuzzy_avx2<CMin<float, int64_t>>(
        float* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template float partition_fuzzy_avx2<CMax<float, int64_t>>(
        float* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template uint16_t partition_fuzzy_avx2<CMin<uint16_t, int64_t>>(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template uint16_t partition_fuzzy_avx2<CMax<uint16_t, int64_t>>(
        uint16_t* vals,
        int64_t* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template uint16_t partition_fuzzy_avx2<CMin<uint16_t, int>>(
        uint16_t* vals,
        int* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

template uint16_t partition_fuzzy_avx2<CMax<uint16_t, int>>(
        uint16_t* vals,
        int* ids,
        size_t n,
        size_t q_min,
        size_t q_max,
        size_t* q_out);

} // namespace faiss
