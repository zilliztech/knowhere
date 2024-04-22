#pragma once

#include "hnswlib.h"
#include "simd/hook.h"

namespace hnswlib {

template <typename DataType, typename DistanceType>
static DistanceType
Cosine(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    if constexpr (!std::is_same<float, DataType>::value) {
        size_t qty = *((size_t*)qty_ptr);
        float res = 0;
        for (unsigned i = 0; i < qty; i++) {
            res += (DistanceType)((DataType*)pVect1)[i] * (DistanceType)((DataType*)pVect2)[i];
        }
        return res;
    } else {
        return faiss::fvec_inner_product((const float*)pVect1, (const float*)pVect2, *((size_t*)qty_ptr));
    }
}

template <typename DataType, typename DistanceType>
static DistanceType
CosineDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * Cosine<DataType, DistanceType>(pVect1, pVect2, qty_ptr);
}

static float
CosineSQ8Distance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * faiss::ivec_inner_product((const int8_t*)pVect1, (const int8_t*)pVect2, *(size_t*)qty_ptr);
}

template <typename DataType, typename DistanceType>
class CosineSpace : public SpaceInterface<DistanceType> {
    DISTFUNC<DistanceType> fstdistfunc_;
    DISTFUNC<float> fstdistfunc_sq_;
    size_t data_size_;
    size_t dim_;

 public:
    CosineSpace(size_t dim) {
        fstdistfunc_ = CosineDistance<DataType, DistanceType>;
        fstdistfunc_sq_ = CosineSQ8Distance;
        dim_ = dim;
        data_size_ = dim * sizeof(DataType);
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<DistanceType>
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
