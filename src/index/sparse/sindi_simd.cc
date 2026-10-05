#include "index/sparse/sindi_simd.h"

#include "index/sparse/sindi_bm25_u4.h"
#include "index/sparse/sindi_packed12.h"
#include "simd/hook.h"

namespace knowhere::sparse::inverted::sindi {

float
ip_accumulate_scalar_u12_e5m7(float q, const uint8_t* vals, const uint8_t* ids, size_t start, int32_t n, float* out) {
    float maximum = 0;
    for (int32_t i = 0; i < n; ++i) {
        auto id = unpack12(ids, start + i);
        out[id] = std::fma(q, decode_e5m7(unpack12(vals, start + i)), out[id]);
        maximum = std::max(maximum, out[id]);
    }

    return maximum;
}

packed_ip_accumulate_fn_t
get_packed_ip_kernel() {
#if defined(__x86_64__)
    namespace cpu = faiss::cppcontrib::knowhere;
    if (cpu::cpu_support_f16c() && __builtin_cpu_supports("fma")) {
        if (cpu::use_avx512 && cpu::cpu_support_avx512() && __builtin_cpu_supports("avx512vl") &&
            __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512f")) {
            return ip_accumulate_avx512_u12_e5m7;
        }
        if (cpu::use_avx2 && cpu::cpu_support_avx2() && __builtin_cpu_supports("avx2")) {
            return ip_accumulate_avx2_u12_e5m7;
        }
    }
#endif

#if defined(__aarch64__) && defined(KNOWHERE_USE_SVE)
    if (faiss::cppcontrib::knowhere::supports_sve()) {
        return ip_accumulate_sve_u12_e5m7;
    }
#endif

    return ip_accumulate_scalar_u12_e5m7;
}

float
ip_accumulate_scalar_fp16(float qval, const knowhere::fp16* vals, const uint16_t* ids, int32_t num, float* out) {
    float max_val = 0.0f;
    for (int32_t i = 0; i < num; ++i) {
        float new_val = (out[ids[i]] += qval * static_cast<float>(vals[i]));
        if (new_val > max_val) {
            max_val = new_val;
        }
    }
    return max_val;
}

float
bm25_accumulate_scalar_u16(float qval, const uint16_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                           float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    float max_val = 0.0f;
    for (int32_t i = 0; i < num; ++i) {
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
bm25_accumulate_scalar_u8(float qval, const uint8_t* vals, const uint16_t* ids, int32_t num, float* out, float k1,
                          float b, float avgdl, const float* row_sums) {
    const float p1 = k1 + 1.0f;
    const float p2 = k1 * (1.0f - b);
    const float p3 = k1 * b / avgdl;

    float max_val = 0.0f;
    for (int32_t i = 0; i < num; ++i) {
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
        const bool support_f16c = faiss::cppcontrib::knowhere::cpu_support_f16c();
        if (support_f16c && faiss::cppcontrib::knowhere::cpu_support_avx512()) {
            k.accumulate = ip_accumulate_avx512_fp16;
            k.batch_insert = batch_insert_avx512;
            return k;
        }
        if (support_f16c && faiss::cppcontrib::knowhere::cpu_support_avx2()) {
            k.accumulate = ip_accumulate_avx2_fp16;
            k.batch_insert = batch_insert_avx2;
            return k;
        }
#elif defined(__aarch64__) && defined(KNOWHERE_USE_SVE)
        if (faiss::cppcontrib::knowhere::supports_sve()) {
            k.accumulate = ip_accumulate_sve_fp16;
            k.batch_insert = batch_insert_sve;
            return k;
        }
#endif
        k.accumulate = ip_accumulate_scalar_fp16;
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
        const bool support_f16c = faiss::cppcontrib::knowhere::cpu_support_f16c();
        if (faiss::cppcontrib::knowhere::use_avx512 && support_f16c &&
            faiss::cppcontrib::knowhere::cpu_support_avx512()) {
            k.accumulate = bm25_accumulate_avx512_u16;
            k.batch_insert = batch_insert_avx512;
            return k;
        }
        if (faiss::cppcontrib::knowhere::use_avx2 && support_f16c && faiss::cppcontrib::knowhere::cpu_support_avx2()) {
            k.accumulate = bm25_accumulate_avx2_u16;
            k.batch_insert = batch_insert_avx2;
            return k;
        }
#elif defined(__aarch64__) && defined(KNOWHERE_USE_SVE)
        if (faiss::cppcontrib::knowhere::supports_sve()) {
            k.accumulate = bm25_accumulate_sve_u16;
            k.batch_insert = batch_insert_sve;
            return k;
        }
#endif
        k.accumulate = bm25_accumulate_scalar_u16;
        k.batch_insert = batch_insert_scalar;
        return k;
    }();
    return kernels;
}

const BM25U8Kernels&
get_bm25_u8_kernels() {
    static const BM25U8Kernels kernels = []() {
        BM25U8Kernels k{};
#if defined(__x86_64__)
        if (faiss::cppcontrib::knowhere::use_avx512 && faiss::cppcontrib::knowhere::cpu_support_avx512()) {
            k.accumulate = bm25_accumulate_avx512_u8;
            k.batch_insert = batch_insert_avx512;
            return k;
        }
        if (faiss::cppcontrib::knowhere::use_avx2 && faiss::cppcontrib::knowhere::cpu_support_avx2()) {
            k.accumulate = bm25_accumulate_avx2_u8;
            k.batch_insert = batch_insert_avx2;
            return k;
        }
#elif defined(__aarch64__) && defined(KNOWHERE_USE_SVE)
        if (faiss::cppcontrib::knowhere::supports_sve()) {
            k.accumulate = bm25_accumulate_sve_u8;
            k.batch_insert = batch_insert_sve;
            return k;
        }
#endif
        k.accumulate = bm25_accumulate_scalar_u8;
        k.batch_insert = batch_insert_scalar;
        return k;
    }();
    return kernels;
}

float
bm25_accumulate_scalar_u12_u4_lut(float q, const uint8_t* vals, const uint8_t* ids, size_t start, int32_t n, float* out,
                                  float k1, float b, float avgdl, const float* lengths, const uint8_t* lut) {
    const float p1 = q * (k1 + 1), p2 = k1 * (1 - b), p3 = k1 * b / avgdl;
    float maximum = 0;
    for (int32_t i = 0; i < n; ++i) {
        const size_t position = start + i;
        const auto id = unpack12(ids, position);
        const float tf = lut[(vals[position / 2] >> (4 * (position & 1))) & 15];
        const float contribution = p1 * tf / (tf + p2 + p3 * lengths[id]);
        out[id] += contribution;
        maximum = std::max(maximum, out[id]);
    }
    return maximum;
}

float
bm25_accumulate_scalar_u16_u4_lut(float q, const uint8_t* vals, const uint8_t* ids, size_t start, int32_t n, float* out,
                                  float k1, float b, float avgdl, const float* lengths, const uint8_t* lut) {
    const float p1 = q * (k1 + 1), p2 = k1 * (1 - b), p3 = k1 * b / avgdl;
    float maximum = 0;
    for (int32_t i = 0; i < n; ++i) {
        const size_t position = start + i;
        const auto id = unpack_bm25_u16_id(ids, position);
        const float tf = lut[(vals[position / 2] >> (4 * (position & 1))) & 15];
        const float contribution = p1 * tf / (tf + p2 + p3 * lengths[id]);
        out[id] += contribution;
        maximum = std::max(maximum, out[id]);
    }
    return maximum;
}

packed_bm25_accumulate_fn_t
get_packed_bm25_kernel(bool u16_ids) {
#if defined(__x86_64__)
    namespace cpu = faiss::cppcontrib::knowhere;
    if (cpu::use_avx512 && cpu::cpu_support_avx512() && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        cpu::cpu_support_f16c()) {
        return u16_ids ? bm25_accumulate_avx512_u16_u4_lut : bm25_accumulate_avx512_u12_u4_lut;
    }
    if (cpu::use_avx2 && cpu::cpu_support_avx2() && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        cpu::cpu_support_f16c()) {
        return u16_ids ? bm25_accumulate_avx2_u16_u4_lut : bm25_accumulate_avx2_u12_u4_lut;
    }
#endif
#if defined(__aarch64__) && defined(KNOWHERE_USE_SVE)
    if (faiss::cppcontrib::knowhere::supports_sve()) {
        return u16_ids ? bm25_accumulate_sve_u16_u4_lut : bm25_accumulate_sve_u12_u4_lut;
    }
#endif
    return u16_ids ? bm25_accumulate_scalar_u16_u4_lut : bm25_accumulate_scalar_u12_u4_lut;
}

}  // namespace knowhere::sparse::inverted::sindi
