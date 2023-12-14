#pragma once

#include "hnswlib.h"
#include "simd/hook.h"

namespace hnswlib {

static float
Cosine(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return faiss::fvec_inner_product((const float*)pVect1, (const float*)pVect2, *((size_t*)qty_ptr));
}

static float
CosineDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * Cosine(pVect1, pVect2, qty_ptr);
}

static float
CosineSQ8Distance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * faiss::ivec_inner_product((const int8_t*)pVect1, (const int8_t*)pVect2, *(size_t*)qty_ptr);
}

class CosineSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    DISTFUNC<float> fstdistfunc_sq_;
    size_t data_size_;
    size_t dim_;

 public:
    CosineSpace(size_t dim) {
        fstdistfunc_ = CosineDistance;
        fstdistfunc_sq_ = CosineSQ8Distance;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<float>
    get_dist_func() {
        return fstdistfunc_;
    }

    DISTFUNC<float>
    get_dist_func_sq() {
        return fstdistfunc_sq_;
    }

    void*
    get_dist_func_param() {
        return &dim_;
    }

    ~CosineSpace() {
    }
};

}  // namespace hnswlib
