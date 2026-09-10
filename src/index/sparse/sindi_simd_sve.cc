#include "index/sparse/sindi_simd.h"

#if defined(__aarch64__) && defined(KNOWHERE_USE_SVE) && defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

namespace knowhere::sparse::inverted::sindi {

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
    for (; i + step <= num; i += step) {
        const __fp16* hptr = reinterpret_cast<const __fp16*>(vals + i);

        svfloat16_t vh = svld1_f16(pg16, hptr);
        svfloat32_t vf_even = svcvt_f32_f16_x(pg32, vh);
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
