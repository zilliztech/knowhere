#include "index/sparse/sindi_simd.h"

#if defined(__aarch64__) && defined(KNOWHERE_USE_SVE) && defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

namespace knowhere::sparse::inverted::sindi {

// Direct U12-to-U32 expansion, adapted from the standalone packed12 SVE kernel.
// Tables are generated for the current VL and nibble phase. Byte predicates bound
// every load, including odd starts and final partial vectors.
static inline svuint32_t
load12(const std::uint8_t* base, std::uint32_t n, std::uint32_t phase, svuint32_t table, svuint32_t shift) {
    const auto bytes = (n * 12 + phase + 7) / 8;
    const auto raw = svld1_u8(svwhilelt_b8(0u, bytes), base);
    const auto pairs = svreinterpret_u32_u8(svtbl_u8(raw, svreinterpret_u8_u32(table)));
    return svand_n_u32_x(svptrue_b32(), svlsr_u32_x(svptrue_b32(), svand_n_u32_x(svptrue_b32(), pairs, 65535), shift),
                         4095);
}

// Decode two full U32 vectors from one packed-byte load. The second
// table advances by 3*vl/2 bytes; vl is always a multiple of four.
// Preserve the starting nibble phase and predicate the exact input byte count.
static inline svuint32x2_t
load_pair(const std::uint8_t* packed, svbool_t bytes, svuint8_t table0, svuint8_t table1, svuint32_t shift) {
    const auto pg = svptrue_b32();
    const auto raw = svld1_u8(bytes, packed);
    // Upper shuffled bytes are immaterial after the shift and U12 mask.
    const auto a = svand_n_u32_x(pg, svlsr_u32_x(pg, svreinterpret_u32_u8(svtbl_u8(raw, table0)), shift), 4095);
    const auto b = svand_n_u32_x(pg, svlsr_u32_x(pg, svreinterpret_u32_u8(svtbl_u8(raw, table1)), shift), 4095);
    return svcreate2_u32(a, b);
}

static inline svfloat32_t
expand_e5(svbool_t pg, svuint32_t code) {
    // FCVT consumes the low binary16 half of each U32 lane, including subnormals.
    return svcvt_f32_f16_x(pg, svreinterpret_f16_u32(svlsl_n_u32_x(pg, code, 3)));
}

float
ip_accumulate_sve_u12_e5m7(float q, const uint8_t* __restrict vals, const uint8_t* __restrict ids, size_t start,
                           int32_t count, float* __restrict out) {
    if (count <= 0) {
        return 0;
    }

    const auto vl = static_cast<std::uint32_t>(svcntw());
    const auto pg = svptrue_b32();
    const auto lane = svindex_u32(0, 1);
    const auto phase = static_cast<std::uint32_t>((start & 1) * 4);
    const auto byte = svlsr_n_u32_x(pg, svadd_n_u32_x(pg, svmul_n_u32_x(pg, lane, 12), phase), 3);
    const auto table = svorr_u32_x(pg, byte, svlsl_n_u32_x(pg, svadd_n_u32_x(pg, byte, 1), 8));
    const auto shift =
        svlsl_n_u32_x(pg, svand_n_u32_x(pg, sveor_n_u32_x(pg, lane, static_cast<std::uint32_t>(start & 1)), 1), 2);
    const auto table0 = svreinterpret_u8_u32(table);
    const auto table1 = svadd_n_u8_x(svptrue_b8(), table0, static_cast<std::uint8_t>(vl * 3 / 2));

    // All loop increments are even posting counts, so the phase and full-pair
    // byte predicate remain constant. Hoist these instead of recomputing them
    // for each of the four pairs in the unrolled loop.
    const auto pair_bytes = svwhilelt_b8(0u, 3 * vl + static_cast<uint32_t>(start & 1));
    const size_t byte_start = (start / 2) * 3 + (start & 1);
    vals += byte_start;
    ids += byte_start;

    const auto vq = svdup_f32(q);
    auto vmax = svdup_f32(0);

    uint32_t i = 0;
    uint32_t n = static_cast<std::uint32_t>(count);

    for (; i + 8 * vl <= n; i += 8 * vl) {
        const auto ids0 = load_pair(ids + (i / 2) * 3 + 0 * vl * 3, pair_bytes, table0, table1, shift);
        const auto words0 = load_pair(vals + (i / 2) * 3 + 0 * vl * 3, pair_bytes, table0, table1, shift);
        const auto id0 = svget2_u32(ids0, 0);
        const auto id1 = svget2_u32(ids0, 1);
        const auto v0 = expand_e5(pg, svget2_u32(words0, 0));
        const auto v1 = expand_e5(pg, svget2_u32(words0, 1));
        const auto ids2 = load_pair(ids + (i / 2) * 3 + 1 * vl * 3, pair_bytes, table0, table1, shift);
        const auto words2 = load_pair(vals + (i / 2) * 3 + 1 * vl * 3, pair_bytes, table0, table1, shift);
        const auto id2 = svget2_u32(ids2, 0);
        const auto id3 = svget2_u32(ids2, 1);
        const auto v2 = expand_e5(pg, svget2_u32(words2, 0));
        const auto v3 = expand_e5(pg, svget2_u32(words2, 1));
        const auto ids4 = load_pair(ids + (i / 2) * 3 + 2 * vl * 3, pair_bytes, table0, table1, shift);
        const auto words4 = load_pair(vals + (i / 2) * 3 + 2 * vl * 3, pair_bytes, table0, table1, shift);
        const auto id4 = svget2_u32(ids4, 0);
        const auto id5 = svget2_u32(ids4, 1);
        const auto v4 = expand_e5(pg, svget2_u32(words4, 0));
        const auto v5 = expand_e5(pg, svget2_u32(words4, 1));
        const auto ids6 = load_pair(ids + (i / 2) * 3 + 3 * vl * 3, pair_bytes, table0, table1, shift);
        const auto words6 = load_pair(vals + (i / 2) * 3 + 3 * vl * 3, pair_bytes, table0, table1, shift);
        const auto id6 = svget2_u32(ids6, 0);
        const auto id7 = svget2_u32(ids6, 1);
        const auto v6 = expand_e5(pg, svget2_u32(words6, 0));
        const auto v7 = expand_e5(pg, svget2_u32(words6, 1));
        const auto old0 = svld1_gather_u32index_f32(pg, out, id0);
        const auto old1 = svld1_gather_u32index_f32(pg, out, id1);
        const auto old2 = svld1_gather_u32index_f32(pg, out, id2);
        const auto old3 = svld1_gather_u32index_f32(pg, out, id3);
        const auto old4 = svld1_gather_u32index_f32(pg, out, id4);
        const auto old5 = svld1_gather_u32index_f32(pg, out, id5);
        const auto old6 = svld1_gather_u32index_f32(pg, out, id6);
        const auto old7 = svld1_gather_u32index_f32(pg, out, id7);
        const auto sum0 = svmad_f32_x(pg, v0, vq, old0);
        svst1_scatter_u32index_f32(pg, out, id0, sum0);
        vmax = svmax_f32_x(pg, vmax, sum0);
        const auto sum1 = svmad_f32_x(pg, v1, vq, old1);
        svst1_scatter_u32index_f32(pg, out, id1, sum1);
        vmax = svmax_f32_x(pg, vmax, sum1);
        const auto sum2 = svmad_f32_x(pg, v2, vq, old2);
        svst1_scatter_u32index_f32(pg, out, id2, sum2);
        vmax = svmax_f32_x(pg, vmax, sum2);
        const auto sum3 = svmad_f32_x(pg, v3, vq, old3);
        svst1_scatter_u32index_f32(pg, out, id3, sum3);
        vmax = svmax_f32_x(pg, vmax, sum3);
        const auto sum4 = svmad_f32_x(pg, v4, vq, old4);
        svst1_scatter_u32index_f32(pg, out, id4, sum4);
        vmax = svmax_f32_x(pg, vmax, sum4);
        const auto sum5 = svmad_f32_x(pg, v5, vq, old5);
        svst1_scatter_u32index_f32(pg, out, id5, sum5);
        vmax = svmax_f32_x(pg, vmax, sum5);
        const auto sum6 = svmad_f32_x(pg, v6, vq, old6);
        svst1_scatter_u32index_f32(pg, out, id6, sum6);
        vmax = svmax_f32_x(pg, vmax, sum6);
        const auto sum7 = svmad_f32_x(pg, v7, vq, old7);
        svst1_scatter_u32index_f32(pg, out, id7, sum7);
        vmax = svmax_f32_x(pg, vmax, sum7);
    }

    for (; i + 2 * vl <= n; i += 2 * vl) {
        const auto pair_ids = load_pair(ids + (i / 2) * 3, pair_bytes, table0, table1, shift);
        const auto words = load_pair(vals + (i / 2) * 3, pair_bytes, table0, table1, shift);
        const auto id0 = svget2_u32(pair_ids, 0), id1 = svget2_u32(pair_ids, 1);
        const auto v0 = expand_e5(pg, svget2_u32(words, 0));
        const auto v1 = expand_e5(pg, svget2_u32(words, 1));
        const auto sum0 = svmad_f32_x(pg, v0, vq, svld1_gather_u32index_f32(pg, out, id0));
        svst1_scatter_u32index_f32(pg, out, id0, sum0);
        vmax = svmax_f32_x(pg, vmax, sum0);
        const auto sum1 = svmad_f32_x(pg, v1, vq, svld1_gather_u32index_f32(pg, out, id1));
        svst1_scatter_u32index_f32(pg, out, id1, sum1);
        vmax = svmax_f32_x(pg, vmax, sum1);
    }

    for (; i < n; i += vl) {
        const auto active = std::min(vl, n - i);
        const auto tail = svwhilelt_b32(0u, active);
        const auto tail_ids = load12(ids + (i / 2) * 3, active, phase, table, shift);
        const auto v = expand_e5(tail, load12(vals + (i / 2) * 3, active, phase, table, shift));
        const auto old = svld1_gather_u32index_f32(tail, out, tail_ids);
        const auto sum = svmad_f32_x(tail, v, vq, old);
        svst1_scatter_u32index_f32(tail, out, tail_ids, sum);
        vmax = svmax_f32_m(tail, vmax, sum);
    }

    return svmaxv_f32(pg, vmax);
}

float
ip_accumulate_sve_fp16(float qval, const knowhere::fp16* __restrict vals, const uint16_t* __restrict ids, int32_t num,
                       float* __restrict out) {
    const svfloat32_t vq32 = svdup_f32(qval);
    const uint32_t vl32 = svcntw();
    const svbool_t pg32 = svptrue_b32();
    const svbool_t pg16 = svptrue_b16();
    svfloat32_t v_max = svdup_f32(0.0f);

    int32_t i = 0;
    const int32_t step = static_cast<int32_t>(vl32 * 2);

    for (; i + 4 * step <= num; i += 4 * step) {
        const __fp16* hptr_0 = reinterpret_cast<const __fp16*>(vals + i + 0 * step);
        const __fp16* hptr_1 = reinterpret_cast<const __fp16*>(vals + i + 1 * step);
        const __fp16* hptr_2 = reinterpret_cast<const __fp16*>(vals + i + 2 * step);
        const __fp16* hptr_3 = reinterpret_cast<const __fp16*>(vals + i + 3 * step);

        svfloat16_t vh_0 = svld1_f16(pg16, hptr_0);
        svfloat16_t vh_1 = svld1_f16(pg16, hptr_1);
        svfloat16_t vh_2 = svld1_f16(pg16, hptr_2);
        svfloat16_t vh_3 = svld1_f16(pg16, hptr_3);
        svfloat32_t vf_even_0 = svcvt_f32_f16_x(pg32, vh_0);
        svfloat32_t vf_even_1 = svcvt_f32_f16_x(pg32, vh_1);
        svfloat32_t vf_even_2 = svcvt_f32_f16_x(pg32, vh_2);
        svfloat32_t vf_even_3 = svcvt_f32_f16_x(pg32, vh_3);
        // // SVE2 replacement
        // svfloat32_t vf_odd_0 = svcvtlt_f32_f16_x(pg32, vh_0);
        // svfloat32_t vf_odd_1 = svcvtlt_f32_f16_x(pg32, vh_1);
        // svfloat32_t vf_odd_2 = svcvtlt_f32_f16_x(pg32, vh_2);
        // svfloat32_t vf_odd_3 = svcvtlt_f32_f16_x(pg32, vh_3);
        svfloat16_t vh_shift_0 = svext_f16(vh_0, vh_0, 1);
        svfloat16_t vh_shift_1 = svext_f16(vh_1, vh_1, 1);
        svfloat16_t vh_shift_2 = svext_f16(vh_2, vh_2, 1);
        svfloat16_t vh_shift_3 = svext_f16(vh_3, vh_3, 1);
        svfloat32_t vf_odd_0 = svcvt_f32_f16_x(pg32, vh_shift_0);
        svfloat32_t vf_odd_1 = svcvt_f32_f16_x(pg32, vh_shift_1);
        svfloat32_t vf_odd_2 = svcvt_f32_f16_x(pg32, vh_shift_2);
        svfloat32_t vf_odd_3 = svcvt_f32_f16_x(pg32, vh_shift_3);

        svuint16_t id16_0 = svld1_u16(pg16, ids + i + 0 * step);
        svuint16_t id16_1 = svld1_u16(pg16, ids + i + 1 * step);
        svuint16_t id16_2 = svld1_u16(pg16, ids + i + 2 * step);
        svuint16_t id16_3 = svld1_u16(pg16, ids + i + 3 * step);
        // Each 32-bit lane contains two adjacent uint16 ids. Splitting the
        // low/high halves avoids the unzip + unpack sequence for both lanes.
        svuint32_t id_pairs_0 = svreinterpret_u32_u16(id16_0);
        svuint32_t id_pairs_1 = svreinterpret_u32_u16(id16_1);
        svuint32_t id_pairs_2 = svreinterpret_u32_u16(id16_2);
        svuint32_t id_pairs_3 = svreinterpret_u32_u16(id16_3);
        svuint32_t vidx_even_0 = svand_n_u32_x(pg32, id_pairs_0, 0xffffu);
        svuint32_t vidx_even_1 = svand_n_u32_x(pg32, id_pairs_1, 0xffffu);
        svuint32_t vidx_even_2 = svand_n_u32_x(pg32, id_pairs_2, 0xffffu);
        svuint32_t vidx_even_3 = svand_n_u32_x(pg32, id_pairs_3, 0xffffu);
        svuint32_t vidx_odd_0 = svlsr_n_u32_x(pg32, id_pairs_0, 16);
        svuint32_t vidx_odd_1 = svlsr_n_u32_x(pg32, id_pairs_1, 16);
        svuint32_t vidx_odd_2 = svlsr_n_u32_x(pg32, id_pairs_2, 16);
        svuint32_t vidx_odd_3 = svlsr_n_u32_x(pg32, id_pairs_3, 16);

        // Issue both independent gathers before their arithmetic to expose
        // enough memory-level parallelism for the scatter-heavy loop.
        svfloat32_t vold_even_0 = svld1_gather_u32index_f32(pg32, out, vidx_even_0);
        svfloat32_t vold_even_1 = svld1_gather_u32index_f32(pg32, out, vidx_even_1);
        svfloat32_t vold_even_2 = svld1_gather_u32index_f32(pg32, out, vidx_even_2);
        svfloat32_t vold_even_3 = svld1_gather_u32index_f32(pg32, out, vidx_even_3);
        svfloat32_t vold_odd_0 = svld1_gather_u32index_f32(pg32, out, vidx_odd_0);
        svfloat32_t vold_odd_1 = svld1_gather_u32index_f32(pg32, out, vidx_odd_1);
        svfloat32_t vold_odd_2 = svld1_gather_u32index_f32(pg32, out, vidx_odd_2);
        svfloat32_t vold_odd_3 = svld1_gather_u32index_f32(pg32, out, vidx_odd_3);
        svfloat32_t vsum_even_0 = svmad_f32_x(pg32, vf_even_0, vq32, vold_even_0);
        svfloat32_t vsum_even_1 = svmad_f32_x(pg32, vf_even_1, vq32, vold_even_1);
        svfloat32_t vsum_even_2 = svmad_f32_x(pg32, vf_even_2, vq32, vold_even_2);
        svfloat32_t vsum_even_3 = svmad_f32_x(pg32, vf_even_3, vq32, vold_even_3);
        svfloat32_t vsum_odd_0 = svmad_f32_x(pg32, vf_odd_0, vq32, vold_odd_0);
        svfloat32_t vsum_odd_1 = svmad_f32_x(pg32, vf_odd_1, vq32, vold_odd_1);
        svfloat32_t vsum_odd_2 = svmad_f32_x(pg32, vf_odd_2, vq32, vold_odd_2);
        svfloat32_t vsum_odd_3 = svmad_f32_x(pg32, vf_odd_3, vq32, vold_odd_3);
        svst1_scatter_u32index_f32(pg32, out, vidx_even_0, vsum_even_0);
        svst1_scatter_u32index_f32(pg32, out, vidx_even_1, vsum_even_1);
        svst1_scatter_u32index_f32(pg32, out, vidx_even_2, vsum_even_2);
        svst1_scatter_u32index_f32(pg32, out, vidx_even_3, vsum_even_3);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd_0, vsum_odd_0);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd_1, vsum_odd_1);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd_2, vsum_odd_2);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd_3, vsum_odd_3);
        v_max = svmax_f32_x(pg32, v_max, vsum_even_0);
        v_max = svmax_f32_x(pg32, v_max, vsum_even_1);
        v_max = svmax_f32_x(pg32, v_max, vsum_even_2);
        v_max = svmax_f32_x(pg32, v_max, vsum_even_3);
        v_max = svmax_f32_x(pg32, v_max, vsum_odd_0);
        v_max = svmax_f32_x(pg32, v_max, vsum_odd_1);
        v_max = svmax_f32_x(pg32, v_max, vsum_odd_2);
        v_max = svmax_f32_x(pg32, v_max, vsum_odd_3);
    }

    for (; i + step <= num; i += step) {
        const __fp16* hptr = reinterpret_cast<const __fp16*>(vals + i);

        svfloat16_t vh = svld1_f16(pg16, hptr);
        svfloat32_t vf_even = svcvt_f32_f16_x(pg32, vh);
        // // SVE2 replacement
        // svfloat32_t vf_odd = svcvtlt_f32_f16_x(pg32, vh);
        svfloat16_t vh_shift = svext_f16(vh, vh, 1);
        svfloat32_t vf_odd = svcvt_f32_f16_x(pg32, vh_shift);

        svuint16_t id16 = svld1_u16(pg16, ids + i);
        // Each 32-bit lane contains two adjacent uint16 ids. Splitting the
        // low/high halves avoids the unzip + unpack sequence for both lanes.
        svuint32_t id_pairs = svreinterpret_u32_u16(id16);
        svuint32_t vidx_even = svand_n_u32_x(pg32, id_pairs, 0xffffu);
        svuint32_t vidx_odd = svlsr_n_u32_x(pg32, id_pairs, 16);

        // Issue both independent gathers before their arithmetic to expose
        // enough memory-level parallelism for the scatter-heavy loop.
        svfloat32_t vold_even = svld1_gather_u32index_f32(pg32, out, vidx_even);
        svfloat32_t vold_odd = svld1_gather_u32index_f32(pg32, out, vidx_odd);
        svfloat32_t vsum_even = svmad_f32_x(pg32, vf_even, vq32, vold_even);
        svfloat32_t vsum_odd = svmad_f32_x(pg32, vf_odd, vq32, vold_odd);
        svst1_scatter_u32index_f32(pg32, out, vidx_even, vsum_even);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd, vsum_odd);
        v_max = svmax_f32_x(pg32, v_max, vsum_even);
        v_max = svmax_f32_x(pg32, v_max, vsum_odd);
    }

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
        svuint32_t id_pairs = svreinterpret_u32_u16(id16);
        svuint32_t vidx_even = svand_n_u32_x(pg32_even, id_pairs, 0xffffu);
        svuint32_t vidx_odd = svlsr_n_u32_x(pg32_odd, id_pairs, 16);

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

float
bm25_accumulate_sve_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                        float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    const svfloat32_t vp2 = svdup_f32(p2);
    const svfloat32_t vp3 = svdup_f32(p3);
    const svfloat32_t vqp1 = svdup_f32(qval * p1);
    svfloat32_t v_max = svdup_f32(0.0f);

    const uint32_t vl32 = svcntw();
    const svbool_t pg32 = svptrue_b32();
    int32_t i = 0;
    const int32_t step = static_cast<int32_t>(vl32 * 2);
    for (; i + step <= num; i += step) {
        // SVE can load halfwords directly into 32-bit lanes. This avoids the
        // unzip + unpack sequence previously needed to widen tf and local ids.
        svuint32_t tf0_u32 = svld1uh_u32(pg32, vals + i);
        svuint32_t tf1_u32 = svld1uh_u32(pg32, vals + i + vl32);
        svfloat32_t tf0 = svcvt_f32_u32_x(pg32, tf0_u32);
        svfloat32_t tf1 = svcvt_f32_u32_x(pg32, tf1_u32);

        svuint32_t vidx0 = svld1uh_u32(pg32, ids + i);
        svuint32_t vidx1 = svld1uh_u32(pg32, ids + i + vl32);

        svfloat32_t dl0 = svld1_gather_u32index_f32(pg32, row_sums, vidx0);
        svfloat32_t dl1 = svld1_gather_u32index_f32(pg32, row_sums, vidx1);

        svfloat32_t numerator0 = svmul_f32_x(pg32, tf0, vqp1);
        svfloat32_t denominator0 = svmad_f32_x(pg32, dl0, vp3, vp2);
        denominator0 = svadd_f32_x(pg32, tf0, denominator0);
        svfloat32_t bm25_0 = svdiv_f32_x(pg32, numerator0, denominator0);

        svfloat32_t numerator1 = svmul_f32_x(pg32, tf1, vqp1);
        svfloat32_t denominator1 = svmad_f32_x(pg32, dl1, vp3, vp2);
        denominator1 = svadd_f32_x(pg32, tf1, denominator1);
        svfloat32_t bm25_1 = svdiv_f32_x(pg32, numerator1, denominator1);

        svfloat32_t old0 = svld1_gather_u32index_f32(pg32, out, vidx0);
        svfloat32_t sum0 = svadd_f32_x(pg32, old0, bm25_0);
        svst1_scatter_u32index_f32(pg32, out, vidx0, sum0);
        v_max = svmax_f32_x(pg32, v_max, sum0);

        svfloat32_t old1 = svld1_gather_u32index_f32(pg32, out, vidx1);
        svfloat32_t sum1 = svadd_f32_x(pg32, old1, bm25_1);
        svst1_scatter_u32index_f32(pg32, out, vidx1, sum1);
        v_max = svmax_f32_x(pg32, v_max, sum1);
    }

    for (; i < num; i += static_cast<int32_t>(vl32)) {
        const uint32_t remaining = static_cast<uint32_t>(num - i);
        svbool_t pg = svwhilelt_b32(static_cast<uint32_t>(0), remaining);

        svuint32_t tf_u32 = svld1uh_u32(pg, vals + i);
        svfloat32_t tf = svcvt_f32_u32_x(pg, tf_u32);
        svuint32_t vidx = svld1uh_u32(pg, ids + i);

        svfloat32_t dl = svld1_gather_u32index_f32(pg, row_sums, vidx);
        svfloat32_t numerator = svmul_f32_x(pg, tf, vqp1);
        svfloat32_t denominator = svmad_f32_x(pg, dl, vp3, vp2);
        denominator = svadd_f32_x(pg, tf, denominator);
        svfloat32_t bm25 = svdiv_f32_x(pg, numerator, denominator);

        svfloat32_t old = svld1_gather_u32index_f32(pg, out, vidx);
        svfloat32_t sum = svadd_f32_x(pg, old, bm25);
        svst1_scatter_u32index_f32(pg, out, vidx, sum);
        v_max = svmax_f32_m(pg, v_max, sum);
    }
    return svmaxv_f32(svptrue_b32(), v_max);
}

float
bm25_accumulate_sve_u8(float qval, const uint8_t* vals, const uint16_t* ids, int32_t num, float* out, float k1, float b,
                       float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    const svfloat32_t vqp1 = svdup_f32(qval * p1);
    const svfloat32_t vp2 = svdup_f32(p2);
    const svfloat32_t vp3 = svdup_f32(p3);
    svfloat32_t v_max = svdup_f32(0.0f);
    const uint32_t vl = svcntw();

    int32_t i = 0;
    while (i < num) {
        const svbool_t pg = svwhilelt_b32(static_cast<uint32_t>(i), static_cast<uint32_t>(num));
        const svuint32_t quantized = svld1ub_u32(pg, vals + i);
        const svfloat32_t tf = svcvt_f32_u32_x(pg, quantized);
        const svuint32_t indices = svld1uh_u32(pg, ids + i);
        const svfloat32_t dl = svld1_gather_u32index_f32(pg, row_sums, indices);

        const svfloat32_t numerator = svmul_f32_x(pg, tf, vqp1);
        svfloat32_t denominator = svmad_f32_x(pg, dl, vp3, vp2);
        denominator = svadd_f32_x(pg, tf, denominator);
        const svfloat32_t contribution = svdiv_f32_x(pg, numerator, denominator);

        const svfloat32_t old_scores = svld1_gather_u32index_f32(pg, out, indices);
        const svfloat32_t scores = svadd_f32_x(pg, old_scores, contribution);
        svst1_scatter_u32index_f32(pg, out, indices, scores);
        v_max = svmax_f32_m(pg, v_max, scores);
        i += static_cast<int32_t>(vl);
    }
    return svmaxv_f32(svptrue_b32(), v_max);
}

void
batch_insert_sve(const float* scores, size_t docid_start, size_t count,
                 knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset) {
    const uint32_t vl = svcntw();
    size_t i = 0;
    svfloat32_t vthr = svdup_f32(threshold);

    for (; i + vl <= count; i += vl) {
        svbool_t pg = svptrue_b32();
        svfloat32_t v = svld1(pg, scores + i);
        svbool_t pg_sel = svcmpgt_f32(pg, v, vthr);

        if (!svptest_any(pg, pg_sel)) {
            continue;
        }

        alignas(64) uint32_t tmp_mask[64];
        svuint32_t ones = svdup_u32(1);
        svuint32_t zeros = svdup_u32(0);
        svuint32_t vmask = svsel_u32(pg_sel, ones, zeros);
        svst1(pg, tmp_mask, vmask);

        for (uint32_t j = 0; j < vl; ++j) {
            if (tmp_mask[j]) {
                size_t idx = i + j;
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

}  // namespace knowhere::sparse::inverted::sindi

#endif
