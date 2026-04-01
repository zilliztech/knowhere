#include "index/sparse/sindi_simd.h"

#include "simd/hook.h"

namespace knowhere::sparse::inverted::sindi {

void
ip_scatter_scalar_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    for (int32_t i = 0; i < num; ++i) {
        out[ids[i]] += qval * static_cast<float>(vals[i]);
    }
}

void
bm25_scatter_scalar_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                        float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    for (int32_t i = 0; i < num; ++i) {
        float tf = static_cast<float>(vals[i]);
        uint16_t docid = ids[i];
        float dl = row_sums[docid];
        float bm25_score = qval * p1 * tf / (tf + p2 + p3 * dl);
        out[docid] += bm25_score;
    }
}

void
batch_insert_scalar(const float* scores, size_t docid_start, size_t count,
                    knowhere::ResultMinHeap<float, uint32_t>& topk_q, float& threshold, const BitsetView& bitset) {
    for (size_t i = 0; i < count; ++i) {
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

const IPKernels&
get_ip_kernels() {
    static const IPKernels kernels = []() {
        IPKernels k{};
#if defined(__x86_64__)
        if (faiss::cppcontrib::knowhere::cpu_support_avx512()) {
            k.accumulate = ip_scatter_avx512_fp16;
            k.batch_insert = batch_insert_avx512;
            return k;
        }
        if (faiss::cppcontrib::knowhere::cpu_support_avx2()) {
            k.accumulate = ip_scatter_avx2_fp16;
            k.batch_insert = batch_insert_avx2;
            return k;
        }
#elif defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
        if (faiss::cppcontrib::knowhere::supports_sve()) {
            k.accumulate = ip_scatter_sve_fp16;
            k.batch_insert = batch_insert_sve;
            return k;
        }
#endif
        k.accumulate = ip_scatter_scalar_fp16;
        k.batch_insert = batch_insert_scalar;
        return k;
    }();
    return kernels;
}

const BM25Kernels&
get_bm25_kernels() {
    static const BM25Kernels kernels = []() {
        BM25Kernels k{};
#if defined(__x86_64__)
        if (faiss::cppcontrib::knowhere::cpu_support_avx512()) {
            k.accumulate = bm25_scatter_avx512_u16;
            k.batch_insert = batch_insert_avx512;
            return k;
        }
        if (faiss::cppcontrib::knowhere::cpu_support_avx2()) {
            k.accumulate = bm25_scatter_avx2_u16;
            k.batch_insert = batch_insert_avx2;
            return k;
        }
#elif defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
        if (faiss::cppcontrib::knowhere::supports_sve()) {
            k.accumulate = bm25_scatter_sve_u16;
            k.batch_insert = batch_insert_sve;
            return k;
        }
#endif
        k.accumulate = bm25_scatter_scalar_u16;
        k.batch_insert = batch_insert_scalar;
        return k;
    }();
    return kernels;
}

}  // namespace knowhere::sparse::inverted::sindi
