#include "index/sparse/sindi_simd.h"

#if defined(__aarch64__) && defined(KNOWHERE_USE_SVE) && defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

namespace knowhere::sparse::inverted::sindi {

float
ip_accumulate_sve_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    const svfloat32_t vq32 = svdup_f32(qval);
    const uint32_t vl32 = svcntw();
    const svbool_t pg32 = svptrue_b32();
    const svbool_t pg16 = svptrue_b16();
    svfloat32_t v_max = svdup_f32(0.0f);

    int32_t i = 0;
    const int32_t step = static_cast<int32_t>(vl32 * 2);
    for (; i + step <= num; i += step) {
        const __fp16* hptr = reinterpret_cast<const __fp16*>(vals + i);

        svfloat16_t vh = svld1_f16(pg16, hptr);
        svfloat32_t vf_even = svcvt_f32_f16_x(pg32, vh);
        svfloat16_t vh_shift = svext_f16(vh, vh, 1);
        svfloat32_t vf_odd = svcvt_f32_f16_x(pg32, vh_shift);

        svuint16_t id16 = svld1_u16(pg16, ids + i);
        svuint16_t id_even16 = svuzp1_u16(id16, id16);
        svuint16_t id_odd16 = svuzp2_u16(id16, id16);
        svuint32_t vidx_even = svunpklo_u32(id_even16);
        svuint32_t vidx_odd = svunpklo_u32(id_odd16);

        svfloat32_t vold_even = svld1_gather_u32index_f32(pg32, out, vidx_even);
        svfloat32_t vsum_even = svmad_f32_x(pg32, vf_even, vq32, vold_even);
        svst1_scatter_u32index_f32(pg32, out, vidx_even, vsum_even);
        v_max = svmax_f32_x(pg32, v_max, vsum_even);

        svfloat32_t vold_odd = svld1_gather_u32index_f32(pg32, out, vidx_odd);
        svfloat32_t vsum_odd = svmad_f32_x(pg32, vf_odd, vq32, vold_odd);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd, vsum_odd);
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

namespace {

svfloat32_t
sv_fast_recip_f32(svbool_t pg, svfloat32_t x) {
    svfloat32_t recip = svrecpe_f32(x);
    svfloat32_t step = svrecps_f32(x, recip);
    return svmul_f32_x(pg, recip, step);
}

}  // namespace

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
    const svbool_t pg16 = svptrue_b16();

    int32_t i = 0;
    const int32_t step = static_cast<int32_t>(vl32 * 2);
    for (; i + step <= num; i += step) {
        svuint16_t tf16 = svld1_u16(pg16, vals + i);

        svuint16_t tf_even16 = svuzp1_u16(tf16, tf16);
        svuint16_t tf_odd16 = svuzp2_u16(tf16, tf16);
        svuint32_t tf_even_u32 = svunpklo_u32(tf_even16);
        svuint32_t tf_odd_u32 = svunpklo_u32(tf_odd16);

        svfloat32_t tf_even = svcvt_f32_u32_x(pg32, tf_even_u32);
        svfloat32_t tf_odd = svcvt_f32_u32_x(pg32, tf_odd_u32);

        svuint16_t id16 = svld1_u16(pg16, ids + i);
        svuint16_t id_even16 = svuzp1_u16(id16, id16);
        svuint16_t id_odd16 = svuzp2_u16(id16, id16);
        svuint32_t vidx_even = svunpklo_u32(id_even16);
        svuint32_t vidx_odd = svunpklo_u32(id_odd16);

        svfloat32_t dl_even = svld1_gather_u32index_f32(pg32, row_sums, vidx_even);
        svfloat32_t dl_odd = svld1_gather_u32index_f32(pg32, row_sums, vidx_odd);

        svfloat32_t num_even = svmul_f32_x(pg32, tf_even, vqp1);
        svfloat32_t denom_even = svmad_f32_x(pg32, dl_even, vp3, vp2);
        denom_even = svadd_f32_x(pg32, tf_even, denom_even);
        svfloat32_t bm25_even = svmul_f32_x(pg32, num_even, sv_fast_recip_f32(pg32, denom_even));

        svfloat32_t num_odd = svmul_f32_x(pg32, tf_odd, vqp1);
        svfloat32_t denom_odd = svmad_f32_x(pg32, dl_odd, vp3, vp2);
        denom_odd = svadd_f32_x(pg32, tf_odd, denom_odd);
        svfloat32_t bm25_odd = svmul_f32_x(pg32, num_odd, sv_fast_recip_f32(pg32, denom_odd));

        svfloat32_t vold_even = svld1_gather_u32index_f32(pg32, out, vidx_even);
        svfloat32_t vsum_even = svadd_f32_x(pg32, vold_even, bm25_even);
        svst1_scatter_u32index_f32(pg32, out, vidx_even, vsum_even);
        v_max = svmax_f32_x(pg32, v_max, vsum_even);

        svfloat32_t vold_odd = svld1_gather_u32index_f32(pg32, out, vidx_odd);
        svfloat32_t vsum_odd = svadd_f32_x(pg32, vold_odd, bm25_odd);
        svst1_scatter_u32index_f32(pg32, out, vidx_odd, vsum_odd);
        v_max = svmax_f32_x(pg32, v_max, vsum_odd);
    }

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
            svfloat32_t bm25_even = svmul_f32_x(pg32_even, num_even, sv_fast_recip_f32(pg32_even, denom_even));

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
            svfloat32_t bm25_odd = svmul_f32_x(pg32_odd, num_odd, sv_fast_recip_f32(pg32_odd, denom_odd));

            svfloat32_t vold_odd = svld1_gather_u32index_f32(pg32_odd, out, vidx_odd);
            svfloat32_t vsum_odd = svadd_f32_x(pg32_odd, vold_odd, bm25_odd);
            svst1_scatter_u32index_f32(pg32_odd, out, vidx_odd, vsum_odd);
            v_max = svmax_f32_m(pg32_odd, v_max, vsum_odd);
        }
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
